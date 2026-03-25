
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import pandas as pd
from pathlib import Path

# ====================  Math Helpers (JIT Optimized) ====================

@torch.jit.script
def phi_1(z):
    eps = 1e-12
    mask = torch.abs(z) < eps
    z_safe = torch.where(mask, torch.ones_like(z) * eps, z)
    res = torch.expm1(z_safe) / z_safe
    return torch.where(mask, torch.ones_like(res), res)

@torch.jit.script
def phi_1_prime(z):
    eps = 1e-9
    mask = torch.abs(z) < eps
    z_safe = torch.where(mask, torch.ones_like(z) * eps, z)
    res = ((z_safe - 1) * torch.exp(z_safe) + 1) / (z_safe ** 2)
    return torch.where(mask, torch.ones_like(res) * 0.5, res)

@torch.jit.script
def phi_1_double_prime(z):
    """Second derivative of phi_1: d²(phi_1)/dz²"""
    eps = 1e-9
    mask = torch.abs(z) < eps
    z_safe = torch.where(mask, torch.ones_like(z) * eps, z)
    # phi_1''(z) = [z²*exp(z) - 2z*exp(z) + 2*exp(z) - 2] / z⁴
    exp_z = torch.exp(z_safe)
    numerator = (z_safe ** 2) * exp_z - 2 * z_safe * exp_z + 2 * exp_z - 2
    res = numerator / (z_safe ** 4)
    return torch.where(mask, torch.ones_like(res) * (1.0 / 6.0), res)

@torch.jit.script
def _scan_linear_recursive(a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor):
    # a: (L, B, D, N)
    # b: (L, B, D, N)
    # h0: (B, D, N)
    L = a.shape[0]
    out = torch.empty_like(a)
    h = h0
    for t in range(L):
        h = a[t] * h + b[t]
        out[t] = h
    return out

@torch.jit.script
def _scan_linear_parallel(a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor):
    """
    Parallel Associative Scan (Sum-of-Products)
    h_t = P_t * h0 + P_t * cumsum(b_t / P_t)
    where P_t = cumprod(a_t)
    """
    # Use log space for numerical stability (a > 0 in Mamba)
    log_a = torch.log(a)
    log_P = torch.cumsum(log_a, dim=0)
    P = torch.exp(log_P)
    
    # Term 1: Initial state contribution
    term1 = P * h0.unsqueeze(0)
    
    # Term 2: Input contribution
    # We use a small epsilon to prevent division by zero, although exp is always > 0
    term2 = P * torch.cumsum(b / (P + 1e-12), dim=0)
    
    return term1 + term2

@torch.jit.script
def _scan_linear_reverse_recursive(a_fwd: torch.Tensor, b_fwd: torch.Tensor, h_next: torch.Tensor):
    L = a_fwd.shape[0]
    z = h_next
    out = torch.empty_like(a_fwd)
    for i in range(L):
        t = L - 1 - i
        z = a_fwd[t] * z + b_fwd[t]
        out[t] = z
    return out

# ====================  Mamba Layer (Batch Supported) ====================

class SingleLayerMambaPGF(nn.Module):
    def __init__(self, d_model, d_state=16, output_activation=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.output_activation = output_activation
        
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float() * 0.1))
        self.D = nn.Parameter(torch.ones(d_model))
        
        self.dt_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.scale = nn.Parameter(torch.ones(d_model))
        
        if output_activation == 'relu':
            self.act_fn = F.relu
            self.act_prime = lambda x: (x > 0).float()
            self.act_double_prime = lambda x: torch.zeros_like(x)
        elif output_activation == 'tanh':
            self.act_fn = torch.tanh
            self.act_prime = lambda x: 1.0 - torch.tanh(x)**2
            self.act_double_prime = lambda x: -2.0 * torch.tanh(x) * (1.0 - torch.tanh(x)**2)
        else:
            self.act_fn = None
            self.act_prime = lambda x: torch.ones_like(x)
            self.act_double_prime = lambda x: torch.zeros_like(x)

    def _project(self, u):
        # u: (B, L, D) -> project -> (B, L, total_dim)
        W = torch.cat([self.dt_proj.weight, self.B_proj.weight, self.C_proj.weight], dim=0)
        b = torch.cat([self.dt_proj.bias, self.B_proj.bias, self.C_proj.bias], dim=0)
        out = F.linear(u, W, b)
        dt_input = out[:, :, :self.d_model]
        B_vec = out[:, :, self.d_model:self.d_model + self.d_state]
        C_vec = out[:, :, self.d_model + self.d_state:]
        return dt_input, B_vec, C_vec

    def _pass1_block(self, u_block, target_block, h0, A):
        # u_block: (B, L, D)
        B_batch, L, D = u_block.shape
        N = self.d_state
        
        dt_input, B_vec, C_vec = self._project(u_block)
        
        # Transpose for Scan: (L, B, D)
        dt_input_t = dt_input.transpose(0, 1)
        B_vec_t = B_vec.transpose(0, 1)
        C_vec_t = C_vec.transpose(0, 1)
        u_t = u_block.transpose(0, 1)
        target_t = target_block.transpose(0, 1)
        
        dt = F.softplus(dt_input_t)
        dtA = dt.unsqueeze(-1) * A.view(1, 1, 1, N)
        dA = torch.exp(dtA)
        dB = phi_1(dtA) * B_vec_t.unsqueeze(2)
        b = dB * u_t.unsqueeze(-1)
        
        # JIT Scan
        # h_traj = buffers["h_traj"][:L] # Use buffer if available? 
        # For simplicity and JIT compat, let's create or use passed buffer
        # But buffers are sized by block_size. L might be smaller at end.
        
        # To avoid complex slicing logic with JIT, let's just run JIT.
        # It allocates output. This is cleaner.
        h_traj = _scan_linear_recursive(dA, b, h0)
        
        # Detach h_traj to cut graph
        h_traj = h_traj.detach()
        
        h_contrib = (C_vec_t.unsqueeze(2) * h_traj).sum(dim=-1)
        if self.act_fn is not None:
            y_raw = self.act_fn(h_contrib) + self.D * u_t
        else:
            y_raw = h_contrib + self.D * u_t
            
        y_pred = y_raw * self.scale
        diff = y_pred - target_t
        loss_sum = (diff * diff).sum()
        
        return h_traj[-1], loss_sum

    def _pass2_block(
        self,
        u_block,
        target_block,
        h0_block,
        A_local,
        delta_h_next,
        dA_next,
        scale_const,
        compute_state_hvp=False,
        v_h=None,
        xi0=None,
        zeta0=None,
    ):
        B_batch, L, D = u_block.shape
        N = self.d_state
        device = u_block.device
        dtype = u_block.dtype
        
        dt_input, B_vec, C_vec = self._project(u_block)
        dt_input = dt_input.detach()
        B_vec = B_vec.detach()
        C_vec = C_vec.detach()
        
        dt_input_t = dt_input.transpose(0, 1)
        B_vec_t = B_vec.transpose(0, 1)
        C_vec_t = C_vec.transpose(0, 1)
        u_t = u_block.transpose(0, 1)
        
        dt = F.softplus(dt_input_t)
        dtA = dt.unsqueeze(-1) * A_local.view(1, 1, 1, N)
        dA = torch.exp(dtA)
        dB = phi_1(dtA) * B_vec_t.unsqueeze(2)
        b = dB * u_t.unsqueeze(-1)
        
        # Recompute Forward (JIT)
        # Note: For state HVP computation, we need h_traj to have gradients
        # But for gradient computation, we detach it to avoid double backprop
        h_traj = _scan_linear_recursive(dA, b, h0_block)
        h_traj_for_output = h_traj.detach()  # Use detached version for output computation
        
        # Compute Output & Delta_in
        # Use detached version for output to avoid double backprop in gradient computation
        h_contrib = (C_vec_t.unsqueeze(2) * h_traj_for_output).sum(dim=-1)  # (L, B, D)
        if self.act_fn is not None:
            h_contrib_act = self.act_fn(h_contrib)  # Apply activation function
            y_raw = h_contrib_act + self.D * u_t
        else:
            y_raw = h_contrib + self.D * u_t
        y_pred = y_raw * self.scale
        target_t = target_block.transpose(0, 1)
        
        # delta_in calculation
        # delta_in_h is for path through h_contrib, delta_in_direct for path through D*u
        grad_y = scale_const * (y_pred - target_t)
        delta_in_direct = grad_y * self.scale
        if self.act_fn is not None:
            h_contrib = (C_vec_t.unsqueeze(2) * h_traj_for_output).sum(dim=-1)
            delta_in_h = delta_in_direct * self.act_prime(h_contrib)
        else:
            delta_in_h = delta_in_direct
            
        grad_scale = (grad_y * y_raw).sum()
        
        C_contrib = delta_in_h.unsqueeze(-1) * C_vec_t.unsqueeze(2)
        
        # Prepare Reverse Scan
        # We need a_fwd[t] = dA[t+1]
        a_fwd = torch.empty((L, B_batch, D, N), device=device, dtype=dtype)
        a_fwd[:-1] = dA[1:]
        a_fwd[-1] = dA_next
        
        # JIT Reverse Scan
        delta_h_block = _scan_linear_reverse_recursive(a_fwd, C_contrib, delta_h_next)
        
        state_hvp_result = None
        xi_next_out = None
        zeta_next_out = None

        if compute_state_hvp and v_h is not None:
            if xi0 is None:
                xi0 = torch.zeros_like(h0_block)
            
            b_zero = torch.zeros_like(b)
            xi_traj = _scan_linear_recursive(dA, b_zero, xi0)
            xi_next_out = xi_traj[-1].detach()
            
            l_double_prime = 2.0 * scale_const
            C_xi = (C_vec_t.unsqueeze(2) * xi_traj).sum(dim=-1)

            if self.act_fn is not None:
                sigma_prime = self.act_prime(h_contrib)
                sigma_double_prime = self.act_double_prime(h_contrib)
                hessian_factor = l_double_prime * (sigma_prime ** 2) + grad_y * sigma_double_prime
            else:
                hessian_factor = l_double_prime

            S_scalar = hessian_factor * C_xi
            S_t = S_scalar.unsqueeze(-1) * C_vec_t.unsqueeze(2)

            if zeta0 is None:
                zeta0 = torch.zeros_like(h0_block)
            
            zeta_block = _scan_linear_reverse_recursive(a_fwd, S_t, zeta0)
            zeta_next_out = zeta_block[0].detach()
            state_hvp_result = zeta_block

        # Compute Gradients (Pure PyTorch - Efficient Memory Management)
        # h_prev construction
        h_prev = torch.empty((L, B_batch, D, N), device=device, dtype=dtype)
        h_prev[0] = h0_block
        h_prev[1:] = h_traj[:-1]
        
        # J_Alog
        # term1 = dtA * dA * h_prev
        # term2 = phi_1_prime(dtA) * dtA * B * u
        # This part uses intermediate memory, but PyTorch Eager frees it quickly.
        
        # Helper terms
        B_exp = B_vec_t.unsqueeze(2)
        u_exp = u_t.unsqueeze(-1)
        term_common = phi_1_prime(dtA) * dtA * B_exp * u_exp
        
        J_Alog_local = dtA * dA * h_prev + term_common
        grad_A_log = (delta_h_block * J_Alog_local).sum(dim=(0, 1, 2))
        
        # J_dt
        A_view = A_local.view(1, 1, 1, N)
        J_dt_local = A_view * dA * h_prev + phi_1_prime(dtA) * A_view * B_exp * u_exp # wait, A_view * term_common / dtA?
        # Actually: d(dtA)/d(dt) = A.
        # d(dA)/d(dt) = A * dA.
        # d(dB)/d(dt) = phi_1_prime(dtA) * A * B.
        # So J_dt = delta_h * (A * dA * h + phi_1_prime * A * B * u)
        # Correct.
        
        g_dt = (delta_h_block * J_dt_local).sum(dim=-1)
        grad_dt_input_block = g_dt * torch.sigmoid(dt_input_t)
        
        # J_B
        J_B_local = phi_1(dtA) * u_exp
        grad_B_vec_block = (delta_h_block * J_B_local).sum(dim=2)
        
        # C grad
        C_vec_grad_block = (delta_in_h.unsqueeze(-1) * h_traj.detach()).sum(dim=2)
        grad_D = (delta_in_direct * u_t).sum(dim=(0, 1))
        
        # Linear Layer Gradients
        grad_stack = torch.cat([grad_dt_input_block, grad_B_vec_block, C_vec_grad_block], dim=2)
        # grad_stack: (L, B, Out)
        
        # Reshape for matmul
        u_flat = u_t.reshape(-1, D) # (L*B, D)
        grad_stack_flat = grad_stack.reshape(-1, D + 2*N) # (L*B, Out)
        
        grad_w_stack = u_flat.t() @ grad_stack_flat # (D, Out)
        grad_b_stack = grad_stack_flat.sum(dim=0)
        
        # TRANSPOSE WEIGHT GRADIENTS TO MATCH nn.Linear (Out, In)
        grad_w_stack_T = grad_w_stack.t() # (Out, In)
        
        grad_dt_w = grad_w_stack_T[:D, :]
        grad_B_w = grad_w_stack_T[D:D+N, :]
        grad_C_w = grad_w_stack_T[D+N:, :]
        
        grad_dt_b = grad_b_stack[:D]
        grad_B_b = grad_b_stack[D:D+N]
        grad_C_b = grad_b_stack[D+N:]
        
        delta_h_next_out = delta_h_block[0].detach()
        dA_next_out = dA[0].detach()
        
        return (
            grad_A_log, grad_D, grad_scale,
            grad_dt_w, grad_dt_b,
            grad_B_w, grad_B_b,
            grad_C_w, grad_C_b,
            delta_h_next_out, dA_next_out,
            state_hvp_result, xi_next_out, zeta_next_out
        )

    def _pass2_block_direct_grad(
        self,
        u_block,
        g_out_block,
        h0_block,
        delta_h_next,
        dA_next,
        fp32_accum=True
    ):
        """
        Similar to _pass2_block but takes incoming gradient g_out_block directly.
        Returns parameter grads and input gradient.
        """
        B_batch, L, D = u_block.shape
        N = self.d_state
        device = u_block.device
        dtype = u_block.dtype
        
        dt_input, B_vec, C_vec = self._project(u_block)
        dt_input = dt_input.detach()
        B_vec = B_vec.detach()
        C_vec = C_vec.detach()
        
        dt_input_t = dt_input.transpose(0, 1)
        B_vec_t = B_vec.transpose(0, 1)
        C_vec_t = C_vec.transpose(0, 1)
        u_t = u_block.transpose(0, 1)
        g_out_t = g_out_block.transpose(0, 1)
        
        dt = F.softplus(dt_input_t)
        dtA = dt.unsqueeze(-1) * (-torch.exp(self.A_log)).view(1, 1, 1, N)
        dA = torch.exp(dtA)
        dB = phi_1(dtA) * B_vec_t.unsqueeze(2)
        b = dB * u_t.unsqueeze(-1)
        
        h_traj = _scan_linear_recursive(dA, b, h0_block)
        h_traj_for_output = h_traj.detach()
        
        # delta_in calculation
        # delta_in_h is for the path through h_contrib (affected by activation)
        # delta_in_direct is for the path through D*u (not affected by activation)
        delta_in_direct = g_out_t * self.scale
        h_contrib = (C_vec_t.unsqueeze(2) * h_traj_for_output).sum(dim=-1)
        if self.act_fn is not None:
            delta_in_h = delta_in_direct * self.act_prime(h_contrib)
            y_raw = self.act_fn(h_contrib) + self.D * u_t
        else:
            delta_in_h = delta_in_direct
            y_raw = h_contrib + self.D * u_t
            
        grad_scale = (g_out_t * y_raw).sum()

        C_contrib = delta_in_h.unsqueeze(-1) * C_vec_t.unsqueeze(2)
        
        a_fwd = torch.empty((L, B_batch, D, N), device=device, dtype=dtype)
        a_fwd[:-1] = dA[1:]
        a_fwd[-1] = dA_next
        
        delta_h_block = _scan_linear_reverse_recursive(a_fwd, C_contrib, delta_h_next)
        
        h_prev = torch.empty((L, B_batch, D, N), device=device, dtype=dtype)
        h_prev[0] = h0_block
        h_prev[1:] = h_traj[:-1]
        
        B_exp = B_vec_t.unsqueeze(2)
        u_exp = u_t.unsqueeze(-1)
        term_common = phi_1_prime(dtA) * dtA * B_exp * u_exp
        
        J_Alog_local = dtA * dA * h_prev + term_common
        # A_log is negative in calculation: A = -exp(A_log)
        # dA/dA_log = dA/dA * dA/dA_log = dt * dA * (-exp(A_log)) = dtA * dA
        # Wait, the derivation in _pass2_block is:
        # A = -exp(A_log). dtA = dt * A.
        # d(dtA)/d(A_log) = dt * (-exp(A_log)) = dtA.
        # So J_Alog = delta_h * (dtA * dA * h_prev + phi_1_prime(dtA) * dtA * B * u)
        grad_A_log = (delta_h_block * J_Alog_local).sum(dim=(0, 1, 2))
        
        A_view = (-torch.exp(self.A_log)).view(1, 1, 1, N)
        J_dt_local = A_view * dA * h_prev + phi_1_prime(dtA) * A_view * B_exp * u_exp
        g_dt = (delta_h_block * J_dt_local).sum(dim=-1)
        grad_dt_input_block = g_dt * torch.sigmoid(dt_input_t)
        
        J_B_local = phi_1(dtA) * u_exp
        grad_B_vec_block = (delta_h_block * J_B_local).sum(dim=2)
        
        C_vec_grad_block = (delta_in_h.unsqueeze(-1) * h_traj_for_output).sum(dim=2)
        grad_D = (delta_in_direct * u_t).sum(dim=(0, 1))
        
        # Linear Layer Gradients
        grad_stack = torch.cat([grad_dt_input_block, grad_B_vec_block, C_vec_grad_block], dim=2)
        u_flat = u_t.reshape(-1, D)
        grad_stack_flat = grad_stack.reshape(-1, D + 2*N)
        grad_w_stack = u_flat.t() @ grad_stack_flat
        grad_b_stack = grad_stack_flat.sum(dim=0)
        grad_w_stack_T = grad_w_stack.t()
        
        grad_dt_w = grad_w_stack_T[:D, :]
        grad_B_w = grad_w_stack_T[D:D+N, :]
        grad_C_w = grad_w_stack_T[D+N:, :]
        grad_dt_b = grad_b_stack[:D]
        grad_B_b = grad_b_stack[D:D+N]
        grad_C_b = grad_b_stack[D+N:]
        
        # Input gradient: dy/du = D * scale + dy/dh * dh/du + dL/d(projections)
        dB_only = phi_1(dtA) * B_vec_t.unsqueeze(2)
        grad_u_recurrent = (delta_h_block * dB_only).sum(dim=-1)
        
        grad_u_direct = delta_in_direct * self.D
        
        grad_proj_combined = torch.cat([grad_dt_input_block, grad_B_vec_block, C_vec_grad_block], dim=-1) # (L, B, D+2N)
        W_concat = torch.cat([self.dt_proj.weight, self.B_proj.weight, self.C_proj.weight], dim=0) # (D+2N, D)
        grad_u_from_proj = torch.matmul(grad_proj_combined, W_concat) # (L, B, D)

        grad_u_block = (grad_u_direct + grad_u_recurrent + grad_u_from_proj).transpose(0, 1) # (B, L, D)

        return (
            grad_A_log, grad_D, grad_scale,
            grad_dt_w, grad_dt_b,
            grad_B_w, grad_B_b,
            grad_C_w, grad_C_b,
            delta_h_block[0].detach(), dA[0].detach(),
            grad_u_block
        )

    def pgf_train_step(self, u, target_y, optimizer, block_size):
        B_batch, L, D = u.shape
        N = self.d_state
        device = u.device
        A = -torch.exp(self.A_log)
        num_blocks = (L + block_size - 1) // block_size

        pass1_cache = []
        loss_sum = torch.tensor(0.0, device=device, dtype=u.dtype)
        scale_const = 2.0 / (L * D * B_batch)
        h0 = torch.zeros(B_batch, D, N, device=device)

        with torch.no_grad():
            for block_idx in range(num_blocks):
                start = block_idx * block_size
                end = min(start + block_size, L)
                u_block = u[:, start:end]
                target_block = target_y[:, start:end]

                h_last, loss_block = self._pass1_block(u_block, target_block, h0, A)
                loss_sum += loss_block

                pass1_cache.append({
                    "h0": h0.detach(),
                    "start": start,
                    "end": end,
                })
                h0 = h_last.detach().clone()

        loss = loss_sum / (L * D * B_batch)

        # Pass 2 Accumulation
        grad_accum = {
            "A_log": torch.zeros_like(self.A_log),
            "D": torch.zeros_like(self.D),
            "scale": torch.zeros_like(self.scale),
            "dt_w": torch.zeros_like(self.dt_proj.weight),
            "dt_b": torch.zeros_like(self.dt_proj.bias),
            "B_w": torch.zeros_like(self.B_proj.weight),
            "B_b": torch.zeros_like(self.B_proj.bias),
            "C_w": torch.zeros_like(self.C_proj.weight),
            "C_b": torch.zeros_like(self.C_proj.bias),
        }
        
        delta_h_next = torch.zeros(B_batch, D, N, device=device)
        dA_next = torch.ones(B_batch, D, N, device=device)

        for cache in reversed(pass1_cache):
            start = cache["start"]
            end = cache["end"]
            u_block = u[:, start:end]
            target_block = target_y[:, start:end]
            h0_block = cache["h0"]
            A_local = A.detach()
            
            grads = self._pass2_block(
                u_block, target_block, h0_block, A_local, delta_h_next, dA_next, scale_const,
                compute_state_hvp=False, v_h=None, xi0=None, zeta0=None
            )
            
            grad_accum["A_log"] += grads[0]
            grad_accum["D"] += grads[1]
            grad_accum["scale"] += grads[2]
            grad_accum["dt_w"] += grads[3]
            grad_accum["dt_b"] += grads[4]
            grad_accum["B_w"] += grads[5]
            grad_accum["B_b"] += grads[6]
            grad_accum["C_w"] += grads[7]
            grad_accum["C_b"] += grads[8]
            
            delta_h_next = grads[9]
            dA_next = grads[10]
            # grads[11], grads[12], grads[13] are state_hvp_result, xi_next_out, zeta_next_out (unused when compute_state_hvp=False)

        optimizer.zero_grad()
        self.A_log.grad = grad_accum["A_log"]
        self.D.grad = grad_accum["D"]
        self.scale.grad = grad_accum["scale"]
        self.dt_proj.weight.grad = grad_accum["dt_w"]
        self.dt_proj.bias.grad = grad_accum["dt_b"]
        self.B_proj.weight.grad = grad_accum["B_w"]
        self.B_proj.bias.grad = grad_accum["B_b"]
        self.C_proj.weight.grad = grad_accum["C_w"]
        self.C_proj.bias.grad = grad_accum["C_b"]
        
        optimizer.step()
        
        return loss.item()

    def compute_state_hvp(self, u, target_y, v_h, block_size):
        """
        Compute State HVP: ∂²L/∂h² · v_h
        
        Args:
            u: input sequence (B, L, D)
            target_y: target sequence (B, L, D)
            v_h: state direction vector (B, D, N) - perturbation on h0
            block_size: block size for streaming computation
            
        Returns:
            state_hvp_full: Full trajectory of HVP vectors (L, B, D, N)
        """
        B_batch, L, D = u.shape
        N = self.d_state
        device = u.device
        A = -torch.exp(self.A_log)
        num_blocks = (L + block_size - 1) // block_size
        
        # Pass 1: Forward Primal (h)
        pass1_cache = []
        h0 = torch.zeros(B_batch, D, N, device=device)
        
        with torch.no_grad():
            for block_idx in range(num_blocks):
                start = block_idx * block_size
                end = min(start + block_size, L)
                u_block = u[:, start:end]
                target_block = target_y[:, start:end]
                
                h_last, _ = self._pass1_block(u_block, target_block, h0, A)
                
                pass1_cache.append({
                    "h0": h0.detach(),
                    "start": start,
                    "end": end,
                })
                h0 = h_last.detach().clone()
        
        # Pass 1.5: Forward Tangent (xi) - Compute boundaries
        # xi_t = A_t * xi_{t-1} (Linear scan with b=0)
        # We need A_t for each block. A_t depends on u_block.
        # We can re-project u_block or cache A_t? Re-project is O(1) memory.
        tangent_cache = []
        xi0 = v_h.clone() if v_h is not None else torch.zeros(B_batch, D, N, device=device)
        
        with torch.no_grad():
            for block_idx in range(num_blocks):
                start = block_idx * block_size
                end = min(start + block_size, L)
                u_block = u[:, start:end]
                
                # Re-compute dA for this block
                dt_input, _, _ = self._project(u_block)
                dt_input_t = dt_input.transpose(0, 1)
                dt = F.softplus(dt_input_t)
                dtA = dt.unsqueeze(-1) * A.view(1, 1, 1, N)
                dA = torch.exp(dtA)
                
                # Scan xi
                # xi_traj = _scan_linear_recursive(dA, zeros, xi0)
                # We only need the last xi to pass to next block
                # Construct b=0
                b_zero = torch.zeros(end-start, B_batch, D, N, device=device)
                
                # Run scan
                xi_traj = _scan_linear_recursive(dA, b_zero, xi0)
                
                tangent_cache.append(xi0.detach()) # Save start xi for this block
                xi0 = xi_traj[-1].detach().clone() # Next start
        
        scale_const = 2.0 / (L * D * B_batch)
        
        # Pass 2: Backward Hessian (zeta)
        delta_h_next = torch.zeros(B_batch, D, N, device=device)
        dA_next = torch.ones(B_batch, D, N, device=device)
        # xi_next and zeta_next logic in _pass2_block
        # xi is recomputed from start. zeta is backward from end.
        zeta_next = torch.zeros(B_batch, D, N, device=device)
        
        # Accumulate state HVP results
        state_hvp_blocks = []
        
        # Iterate backwards
        # tangent_cache has xi0 for each block in order 0..N-1
        # pass1_cache has h0 for each block in order 0..N-1
        # We zip them and reverse
        
        for i in range(num_blocks - 1, -1, -1):
            cache = pass1_cache[i]
            xi0_block = tangent_cache[i]
            
            start = cache["start"]
            end = cache["end"]
            u_block = u[:, start:end]
            target_block = target_y[:, start:end]
            h0_block = cache["h0"]
            A_local = A.detach()
            
            # Call _pass2_block with xi0_block
            grads = self._pass2_block(
                u_block, target_block, h0_block, A_local, delta_h_next, dA_next, scale_const,
                compute_state_hvp=True, v_h=v_h, xi0=xi0_block, zeta0=zeta_next
            )
            
            delta_h_next = grads[9]
            dA_next = grads[10]
            state_hvp_result = grads[11]
            # xi_next_out = grads[12] # Not needed for backward
            zeta_next = grads[13] # For next (previous in time) block
            
            if state_hvp_result is not None:
                state_hvp_blocks.append(state_hvp_result)
        
        # Concatenate all blocks (in reverse order, since we processed backwards)
        if state_hvp_blocks:
            # Reverse to get correct temporal order
            state_hvp_blocks.reverse()
            state_hvp_full = torch.cat(state_hvp_blocks, dim=0)  # (L, B, D, N)
        else:
            state_hvp_full = None
        
        return state_hvp_full

    def _forward_block(self, x_block, h0_block):
        """Block-level forward pass (can be used for autograd inside block)"""
        dt_input, B_vec, C_vec = self._project(x_block)
        A = -torch.exp(self.A_log)
        dt = F.softplus(dt_input.transpose(0, 1))
        dtA = dt.unsqueeze(-1) * A.view(1, 1, 1, self.d_state)
        dA = torch.exp(dtA)
        dB = phi_1(dtA) * B_vec.transpose(0, 1).unsqueeze(2)
        b = dB * x_block.transpose(0, 1).unsqueeze(-1)
        
        if self.use_parallel_scan:
            h_traj = _scan_linear_parallel(dA, b, h0_block)
        else:
            h_traj = _scan_linear_recursive(dA, b, h0_block)
            
        h_contrib = (C_vec.transpose(0, 1).unsqueeze(2) * h_traj).sum(dim=-1)
        if self.act_fn is not None:
            y_raw = self.act_fn(h_contrib) + self.D * x_block.transpose(0, 1)
        else:
            y_raw = h_contrib + self.D * x_block.transpose(0, 1)
        return (y_raw * self.scale).transpose(0, 1), h_traj[-1]

    def pgf_backward_block_autograd(self, x_block, g_out_block, h0_block, g_h_next):
        """
        Calculates gradients for a block using PyTorch Autograd.
        Accumulates directly into self.parameters().grad.
        """
        # Ensure all inputs are detached but require grad
        x_block = x_block.detach().requires_grad_(True)
        h0_block = h0_block.detach().requires_grad_(True)
        
        with torch.enable_grad():
            y_block, h_last = self._forward_block(x_block, h0_block)
            
            # Backprop through y_block and h_last
            # This accumulates directly into parameters' .grad
            torch.autograd.backward(
                [y_block, h_last],
                [g_out_block, g_h_next]
            )
            
        # Return grad for h0 and grad for x_block (parameters are already updated)
        return h0_block.grad.detach(), x_block.grad.detach()

    def forward_standard(self, u):
        # Auto-grad version needs to handle batch dimension
        # u: (B, L, D)
        B, L, D = u.shape
        N = self.d_state
        
        dt_input, B_vec, C_vec = self._project(u)
        
        dt_input = dt_input.transpose(0, 1)
        B_vec = B_vec.transpose(0, 1)
        C_vec = C_vec.transpose(0, 1)
        u_t = u.transpose(0, 1)
        
        A = -torch.exp(self.A_log)
        dt = F.softplus(dt_input)
        
        dtA = dt.unsqueeze(-1) * A.view(1, 1, 1, N)
        dA = torch.exp(dtA)
        dB = phi_1(dtA) * B_vec.unsqueeze(2)
        
        b_term = dB * u_t.unsqueeze(-1)
        h0 = torch.zeros(B, D, N, device=u.device, dtype=u.dtype)
        
        # Ensure recursive scan is used for strictly identical comparisons
        h = _scan_linear_recursive(dA, b_term, h0)
        
        h_contrib = (C_vec.unsqueeze(2) * h).sum(dim=-1)
        if self.act_fn is not None:
            y_raw = self.act_fn(h_contrib) + self.D * u_t
        else:
            y_raw = h_contrib + self.D * u_t
            
        y = y_raw * self.scale
        
        return y.transpose(0, 1)

# ====================  End of Library ====================
