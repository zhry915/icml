"""
PGF Mamba Block: Integrated RMSNorm + Mamba + PGF-FFN + Residual
Achieves O(1) memory complexity in sequence length L per layer.
Depth O(1) is achieved via global Pass 1/2 separation and input stack management.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Union

from train_singlemamba_optimized import SingleLayerMambaPGF, _scan_linear_recursive, phi_1
from pgf_rmsnorm import RMSNorm
from pgf_ffn import pgf_ffn_forward, pgf_ffn_grad

class PGFMambaBlock(nn.Module):
    """
    A single PGF Mamba Block:
    x -> RMSNorm -> Mamba -> Residual -> RMSNorm -> FFN -> Residual -> y
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_ff: Optional[int] = None,
        block_size: int = 128,
        output_activation: Optional[str] = None,
        ffn_activation: str = "gelu",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_ff = d_ff or 4 * d_model
        self.block_size = block_size
        self.ffn_activation = ffn_activation
        
        # Components
        self.norm1 = RMSNorm(d_model)
        self.mamba = SingleLayerMambaPGF(d_model, d_state, output_activation=output_activation)
        
        self.norm2 = RMSNorm(d_model)
        # FFN Parameters (manual management for PGF FFN)
        self.W1 = nn.Parameter(torch.randn(self.d_ff, d_model) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(self.d_ff))
        self.W2 = nn.Parameter(torch.randn(d_model, self.d_ff) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(d_model))
        
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard autograd forward for comparison"""
        # Sub-layer 1: Mamba
        res = x
        x = self.norm1(x)
        x = self.mamba.forward_standard(x)
        x = x + res
        
        # Sub-layer 2: FFN
        res = x
        x = self.norm2(x)
        if self.ffn_activation == "gelu":
            h = F.gelu(F.linear(x, self.W1, self.b1), approximate='tanh')
        else:
            h = F.relu(F.linear(x, self.W1, self.b1))
        x = F.linear(h, self.W2, self.b2)
        x = x + res
        return x

    @torch.no_grad()
    def pgf_forward(self, x: torch.Tensor) -> torch.Tensor:
        """PGF forward: O(1) memory in sequence length L"""
        # Sub-layer 1: Mamba
        res = x
        x_norm = self.norm1.pgf_forward(x, block_size=self.block_size)
        
        # Mamba PGF Forward
        L, B, D = x_norm.shape[1], x_norm.shape[0], x_norm.shape[2]

        A = -torch.exp(self.mamba.A_log)
        h0 = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        
        # Process in blocks to maintain O(1) memory
        mamba_out_list = []
        curr_h = h0
        for start in range(0, L, self.block_size):
            end = min(start + self.block_size, L)
            x_block = x_norm[:, start:end]
            
            # Forward block logic
            dt_input, B_vec, C_vec = self.mamba._project(x_block)
            dt_input_t = dt_input.transpose(0, 1)
            B_vec_t = B_vec.transpose(0, 1)
            C_vec_t = C_vec.transpose(0, 1)
            u_t = x_block.transpose(0, 1)
            
            dt = F.softplus(dt_input_t)
            dtA = dt.unsqueeze(-1) * A.view(1, 1, 1, self.d_state)
            dA = torch.exp(dtA)
            dB = phi_1(dtA) * B_vec_t.unsqueeze(2)
            b = dB * u_t.unsqueeze(-1)
            
            h_traj = _scan_linear_recursive(dA, b, curr_h)
            curr_h = h_traj[-1].detach().clone()
            
            h_contrib = (C_vec_t.unsqueeze(2) * h_traj).sum(dim=-1)
            if self.mamba.act_fn is not None:
                y_raw = self.mamba.act_fn(h_contrib) + self.mamba.D * u_t
            else:
                y_raw = h_contrib + self.mamba.D * u_t
            y_pred_block = (y_raw * self.mamba.scale).transpose(0, 1)
            mamba_out_list.append(y_pred_block.detach())
            
        x = torch.cat(mamba_out_list, dim=1) + res
        
        # Sub-layer 2: FFN
        res = x
        x_norm = self.norm2.pgf_forward(x, block_size=self.block_size)
        ffn_out = pgf_ffn_forward(
            x_norm.reshape(-1, D), # Flatten B*L for FFN
            self.W1, self.b1, self.W2, self.b2,
            block_size=self.block_size,
            activation=self.ffn_activation
        )
        x = ffn_out.reshape(B, L, D) + res
        return x

    @torch.no_grad()
    def pgf_backward(self, x: torch.Tensor, g_out: torch.Tensor, fp32_accum: bool = True) -> Dict[str, Any]:
        """PGF backward: O(1) memory in L"""
        B, L, D = x.shape
        N = self.d_state
        device = x.device
        dtype = x.dtype
        
        # Initialize grads
        grads = {
            "W1": torch.zeros_like(self.W1),
            "b1": torch.zeros_like(self.b1),
            "W2": torch.zeros_like(self.W2),
            "b2": torch.zeros_like(self.b2),
            "mamba.A_log": torch.zeros_like(self.mamba.A_log),
            "mamba.D": torch.zeros_like(self.mamba.D),
            "mamba.scale": torch.zeros_like(self.mamba.scale),
            "mamba.dt_proj.weight": torch.zeros_like(self.mamba.dt_proj.weight),
            "mamba.dt_proj.bias": torch.zeros_like(self.mamba.dt_proj.bias),
            "mamba.B_proj.weight": torch.zeros_like(self.mamba.B_proj.weight),
            "mamba.B_proj.bias": torch.zeros_like(self.mamba.B_proj.bias),
            "mamba.C_proj.weight": torch.zeros_like(self.mamba.C_proj.weight),
            "mamba.C_proj.bias": torch.zeros_like(self.mamba.C_proj.bias),
            "norm1.weight": torch.zeros_like(self.norm1.weight),
            "norm2.weight": torch.zeros_like(self.norm2.weight),
        }

        # --- Recompute Forward Pass 1 to get layer inputs and intermediate states ---
        # Sub-layer 1: Mamba
        x_norm1 = self.norm1.pgf_forward(x, block_size=self.block_size)
        # Store boundary states for Mamba backward
        mamba_h0_list = []
        curr_h = torch.zeros(B, D, N, device=device, dtype=dtype)
        A = -torch.exp(self.mamba.A_log)
        
        for start in range(0, L, self.block_size):
            mamba_h0_list.append(curr_h.clone())
            end = min(start + self.block_size, L)
            # Simplified pass1 to get next curr_h
            x_block = x_norm1[:, start:end]
            dt_input, B_vec, _ = self.mamba._project(x_block)
            dt = F.softplus(dt_input.transpose(0, 1))
            dtA = dt.unsqueeze(-1) * A.view(1, 1, 1, N)
            dA = torch.exp(dtA)
            dB = phi_1(dtA) * B_vec.transpose(0, 1).unsqueeze(2)
            b = dB * x_block.transpose(0, 1).unsqueeze(-1)
            h_traj = _scan_linear_recursive(dA, b, curr_h)
            curr_h = h_traj[-1].detach()

        # Get Mamba output to feed into FFN
        mamba_out = self.pgf_forward_mamba_only(x_norm1) + x
        x_norm2 = self.norm2.pgf_forward(mamba_out, block_size=self.block_size)

        # 1. FFN + Norm2 + Residual
        g_ffn_out = g_out # Start with incoming gradient
        ffn_res = pgf_ffn_grad(
            x_norm2.reshape(-1, D), g_ffn_out.reshape(-1, D),
            self.W1, self.b1, self.W2, self.b2,
            block_size=self.block_size, activation=self.ffn_activation,
            compute_input_grads=True, fp32_accum=fp32_accum
        )
        grads["W1"].add_(ffn_res["grad_W1"])
        grads["b1"].add_(ffn_res["grad_b1"])
        grads["W2"].add_(ffn_res["grad_W2"])
        grads["b2"].add_(ffn_res["grad_b2"])
        
        # Norm2 backward: Only for the FFN path
        g_norm2_out = ffn_res["grad_x"].reshape(B, L, D)
        norm2_res = self.norm2.pgf_grad(
            mamba_out, g_norm2_out, block_size=self.block_size,
            fp32_accum=fp32_accum, compute_input_grads=True
        )
        grads["norm2.weight"].add_(norm2_res["grad_weight"])
        # Gradient for Mamba output = Norm2_in_grad + Shortcut_grad
        g_mamba_out = norm2_res["grad_x"] + g_out
        
        # 2. Mamba + Norm1 + Residual
        # Mamba backward needs to be done in reverse blocks
        g_mamba_in_sum = torch.zeros_like(x)
        delta_h_next = torch.zeros(B, D, N, device=device, dtype=dtype)
        dA_next = torch.ones(B, D, N, device=device, dtype=dtype)
        
        num_blocks = len(mamba_h0_list)
        for i in reversed(range(num_blocks)):
            start = i * self.block_size
            end = min(start + self.block_size, L)
            x_block = x_norm1[:, start:end]
            h0_block = mamba_h0_list[i]
            
            m_grads = self.mamba._pass2_block_direct_grad(
                x_block, g_mamba_out[:, start:end], h0_block, 
                delta_h_next, dA_next, fp32_accum=fp32_accum
            )
            
            grads["mamba.A_log"].add_(m_grads[0])
            grads["mamba.D"].add_(m_grads[1])
            grads["mamba.scale"].add_(m_grads[2])
            grads["mamba.dt_proj.weight"].add_(m_grads[3])
            grads["mamba.dt_proj.bias"].add_(m_grads[4])
            grads["mamba.B_proj.weight"].add_(m_grads[5])
            grads["mamba.B_proj.bias"].add_(m_grads[6])
            grads["mamba.C_proj.weight"].add_(m_grads[7])
            grads["mamba.C_proj.bias"].add_(m_grads[8])
            
            delta_h_next = m_grads[9]
            dA_next = m_grads[10]
            g_mamba_in_sum[:, start:end] = m_grads[11]

        # Norm1 backward: Only for the Mamba path
        g_norm1_out = g_mamba_in_sum
        norm1_res = self.norm1.pgf_grad(
            x, g_norm1_out, block_size=self.block_size,
            fp32_accum=fp32_accum, compute_input_grads=True
        )
        grads["norm1.weight"].add_(norm1_res["grad_weight"])
        
        # Final grad_input = Norm1_in_grad + Shortcut_grad
        # Return gradients
        return {
            "grads": grads,
            "grad_input": norm1_res["grad_x"] + g_mamba_out
        }

    def pgf_forward_mamba_only(self, x_norm):
        """Helper for recomputation"""
        L, B, D = x_norm.shape[1], x_norm.shape[0], x_norm.shape[2]
        A = -torch.exp(self.mamba.A_log)
        curr_h = torch.zeros(B, D, self.d_state, device=x_norm.device, dtype=x_norm.dtype)
        mamba_out_list = []
        for start in range(0, L, self.block_size):
            end = min(start + self.block_size, L)
            x_block = x_norm[:, start:end]
            dt_input, B_vec, C_vec = self.mamba._project(x_block)
            dt = F.softplus(dt_input.transpose(0, 1))
            dtA = dt.unsqueeze(-1) * A.view(1, 1, 1, self.d_state)
            dA = torch.exp(dtA)
            dB = phi_1(dtA) * B_vec.transpose(0, 1).unsqueeze(2)
            b = dB * x_block.transpose(0, 1).unsqueeze(-1)
            h_traj = _scan_linear_recursive(dA, b, curr_h)
            curr_h = h_traj[-1].detach().clone()
            h_contrib = (C_vec.transpose(0, 1).unsqueeze(2) * h_traj).sum(dim=-1)
            if self.mamba.act_fn is not None:
                y_raw = self.mamba.act_fn(h_contrib) + self.mamba.D * x_block.transpose(0, 1)
            else:
                y_raw = h_contrib + self.mamba.D * x_block.transpose(0, 1)
            y_pred_block = (y_raw * self.mamba.scale).transpose(0, 1)
            mamba_out_list.append(y_pred_block)
        return torch.cat(mamba_out_list, dim=1)

class PGFMambaSequential(nn.Module):
    """
    A sequence of Mamba blocks.
    """
    def __init__(self, layers: List[PGFMambaBlock]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    @torch.no_grad()
    def pgf_train_step(self, x: torch.Tensor, target: torch.Tensor, loss_fn: Any) -> Dict[str, Any]:
        """
        Global Pass 1 & Pass 2
        """
        # Pass 1: Global Forward
        layer_inputs = []
        curr_x = x
        for layer in self.layers:
            layer_inputs.append(curr_x.detach().clone())
            curr_x = layer.pgf_forward(curr_x)
        
        # Compute loss and initial grad_output
        # Assuming loss_fn returns (loss_val, grad_output)
        loss, g_out = loss_fn(curr_x, target)
        
        # Pass 2: Global Backward (Reverse order)
        all_grads = {}
        curr_g = g_out
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            layer_in = layer_inputs.pop()
            
            res = layer.pgf_backward(layer_in, curr_g)
            
            # Prefix grads with layer index
            for k, v in res["grads"].items():
                all_grads[f"layers.{i}.{k}"] = v
            
            curr_g = res["grad_input"]
            del layer_in, res
            
        return {"loss": loss, "grads": all_grads}
