
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, List, Optional


@torch.jit.script
def _rmsnorm_forward_jit(
    x_block: torch.Tensor,  # (block_size, dim)
    weight: torch.Tensor,  # (dim,)
    eps: float = 1e-6,
) -> torch.Tensor:

    rms = torch.sqrt(torch.mean(x_block * x_block, dim=-1, keepdim=True) + eps)  # (block_size, 1)
    y_block = (x_block / rms) * weight.unsqueeze(0)  # (block_size, dim)
    return y_block


@torch.jit.script
def _process_single_block_grad_jit(
    x_block: torch.Tensor,  # (block_size, dim)
    g_out_block: torch.Tensor,  # (block_size, dim)
    weight: torch.Tensor,  # (dim,)
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_size, dim = x_block.shape
    
    x_sq = x_block * x_block  # (block_size, dim)
    rms = torch.sqrt(torch.mean(x_sq, dim=-1, keepdim=True) + eps)  # (block_size, 1)
    rms_inv = rms.reciprocal()  # (block_size, 1) -
    
    x_norm = x_block * rms_inv  # (block_size, dim) - 
    
    # grad_weight = sum(g_out * x_norm)
    grad_weight_block = (g_out_block * x_norm).sum(dim=0)  # (dim,)
    
    # grad_x 
    # grad_x = (g_out * weight) / rms - (x / (rms^3 * dim)) * sum(g_out * weight * x)
    g_weight = g_out_block * weight.unsqueeze(0)  # (block_size, dim)
    term1 = g_weight * rms_inv  # (block_size, dim) -
    
    #  dot_product = sum(g_out * weight * x) 
    dot_product = (g_weight * x_block).sum(dim=-1, keepdim=True)  # (block_size, 1)
    rms_inv_cubed = rms_inv * rms_inv * rms_inv  # (block_size, 1)
    dim_float = float(dim)
    term2 = (x_block * dot_product) * rms_inv_cubed / dim_float  # (block_size, dim)
    grad_x_block = term1 - term2  # (block_size, dim)
    
    return grad_weight_block, grad_x_block


@torch.jit.script
def _rmsnorm_jvp_jit(
    x: torch.Tensor, x_dot: torch.Tensor,
    weight: torch.Tensor, weight_dot: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """JIT optimized RMSNorm JVP"""
    dim = x.shape[-1]
    x_sq_mean = torch.mean(x * x, dim=-1, keepdim=True)
    rms = torch.sqrt(x_sq_mean + eps)
    rms_dot = torch.mean(x * x_dot, dim=-1, keepdim=True) / rms
    
    # y = (x / rms) * weight
    y = (x / rms) * weight
    
    # y_dot = ((x_dot * rms - x * rms_dot) / rms^2) * weight + (x / rms) * weight_dot
    y_dot = ((x_dot * rms - x * rms_dot) / (rms * rms)) * weight + (x / rms) * weight_dot
    
    return y, y_dot


class RMSNorm(nn.Module):
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS: sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        y = (x / rms) * self.weight
        return y
    
    @torch.no_grad()
    def pgf_forward(
        self,
        x: torch.Tensor,  # (L, dim) or (B, L, dim)
        block_size: int = 512,
    ) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() == 3:
            B, L, dim = x.shape
            x = x.reshape(-1, dim)
        else:
            L, dim = x.shape
            B = None

        total_L = x.shape[0]
        if total_L <= block_size:
            y = _rmsnorm_forward_jit(x, self.weight, self.eps)
        else:
            num_blocks = (total_L + block_size - 1) // block_size
            y_list = []
            for i in range(num_blocks):
                start = i * block_size
                end = min(start + block_size, total_L)
                x_block = x[start:end]
                y_block = _rmsnorm_forward_jit(x_block, self.weight, self.eps)
                y_list.append(y_block)
            y = torch.cat(y_list, dim=0)
            
        return y.reshape(orig_shape)

    @torch.no_grad()
    def pgf_jvp(
        self,
        x: torch.Tensor,
        x_dot: torch.Tensor,
        weight_dot: torch.Tensor,
        block_size: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        PGF JVP: Analytical tangent propagation.
        """
        orig_shape = x.shape
        if x.dim() == 3:
            B, L, dim = x.shape
            x = x.reshape(-1, dim)
            x_dot = x_dot.reshape(-1, dim)
        else:
            L, dim = x.shape
            B = None
            
        total_L = x.shape[0]
        if total_L <= block_size:
            y, y_dot = _rmsnorm_jvp_jit(x, x_dot, self.weight, weight_dot, self.eps)
        else:
            num_blocks = (total_L + block_size - 1) // block_size
            y_list = []
            y_dot_list = []
            for i in range(num_blocks):
                start = i * block_size
                end = min(start + block_size, total_L)
                y_blk, y_dot_blk = _rmsnorm_jvp_jit(x[start:end], x_dot[start:end], self.weight, weight_dot, self.eps)
                y_list.append(y_blk)
                y_dot_list.append(y_dot_blk)
            y = torch.cat(y_list, dim=0)
            y_dot = torch.cat(y_dot_list, dim=0)
            
        return y.reshape(orig_shape), y_dot.reshape(orig_shape)

    @torch.no_grad()
    def pgf_grad(
        self,
        x: torch.Tensor,  # (L, dim) or (B, L, dim)
        g_out: torch.Tensor,  # (L, dim) or (B, L, dim)
        block_size: int = 512,
        fp32_accum: bool = True,
        compute_input_grads: bool = False,
    ) -> dict:

        orig_shape = x.shape
        if x.dim() == 3:
            B, L, dim = x.shape
            x = x.reshape(-1, dim)
            g_out = g_out.reshape(-1, dim)
        else:
            L, dim = x.shape
            B = None
            
        x = x.detach()
        g_out = g_out.detach()
        
        total_L, dim = x.shape
        
        comp_dtype = torch.float32 if (fp32_accum and x.dtype in (torch.float16, torch.bfloat16)) else None
        work_dtype = comp_dtype if comp_dtype is not None else x.dtype
        
        grad_weight = torch.zeros_like(self.weight, dtype=work_dtype)
        if compute_input_grads:
            grad_x = torch.zeros_like(x, dtype=work_dtype)
        else:
            grad_x = None
        
        weight_comp = self.weight.to(comp_dtype) if comp_dtype is not None else self.weight
        
        num_blocks = (total_L + block_size - 1) // block_size
        for i in range(num_blocks):
            start = i * block_size
            end = min(start + block_size, total_L)
            x_block = x[start:end]
            g_out_block = g_out[start:end]
            
            grad_weight_block, grad_x_block = _process_single_block_grad_jit(
                x_block, g_out_block, weight_comp, self.eps
            )
            
            grad_weight.add_(grad_weight_block)
            if compute_input_grads:
                grad_x[start:end] = grad_x_block
        
        result = {
            "grad_weight": grad_weight.to(dtype=self.weight.dtype),
        }
        
        if compute_input_grads:
            result["grad_x"] = grad_x.reshape(orig_shape).to(dtype=x.dtype)
        
        return result


