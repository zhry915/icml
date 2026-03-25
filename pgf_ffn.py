
from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any, List, Optional


_USE_COMPILE = False
if _USE_COMPILE:

    _compile_fn = torch.compile(mode="reduce-overhead", fullgraph=False)
else:
    _compile_fn = lambda x: x


@torch.jit.script
def _gelu_jit(x: torch.Tensor) -> torch.Tensor:
    """GELU 激活函数（JIT 优化）"""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


@torch.jit.script
def _gelu_prime_jit(x: torch.Tensor) -> torch.Tensor:
    """GELU 激活函数的导数（JIT 优化）"""
    tanh_arg = 0.7978845608 * (x + 0.044715 * x * x * x)
    tanh_val = torch.tanh(tanh_arg)
    sech2 = 1.0 - tanh_val * tanh_val
    return 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * (0.7978845608 * (1.0 + 3.0 * 0.044715 * x * x))


@torch.jit.script
def _relu_prime_jit(x: torch.Tensor) -> torch.Tensor:
    """ReLU 激活函数的导数（JIT 优化）"""
    return (x > 0).to(x.dtype)


@torch.jit.script
def _ffn_block_forward_jit(
    x_block: torch.Tensor,  # (block_size, d_in)
    W1: torch.Tensor,  # (d_hidden, d_in)
    b1: torch.Tensor,  # (d_hidden,)
    W2: torch.Tensor,  # (d_out, d_hidden)
    b2: torch.Tensor,  # (d_out,)
    use_gelu: bool,
) -> torch.Tensor:
    """
    JIT 优化的 FFN block 前向传播
    注意：JIT script 不支持字符串比较，所以使用 bool 参数
    """
    W1_T = W1.t()  # (d_in, d_hidden) - transpose 
    z_block = torch.matmul(x_block, W1_T) + b1  # (block_size, d_hidden)
    
    if use_gelu:
        h_block = _gelu_jit(z_block)
    else:
        h_block = torch.relu(z_block)
    
    # y = h @ W2^T + b2
    W2_T = W2.t()  # (d_hidden, d_out) - transpose 
    y_block = torch.matmul(h_block, W2_T) + b2  # (block_size, d_out)
    
    return y_block


@_compile_fn
def _process_single_block_grad(
    x_block: torch.Tensor,  # (block_size, d_in)
    g_out_block: torch.Tensor,  # (block_size, d_out)
    W1: torch.Tensor,  # (d_hidden, d_in)
    b1: torch.Tensor,  # (d_hidden,)
    W2: torch.Tensor,  # (d_out, d_hidden)
    b2: torch.Tensor,  # (d_out,)
    activation: str = "gelu",
    comp_dtype: torch.dtype | None = None,
):
    
    if comp_dtype is not None:
        x_block = x_block.to(comp_dtype)
        g_out_block = g_out_block.to(comp_dtype)
        W1 = W1.to(comp_dtype)
        W2 = W2.to(comp_dtype)
        b1 = b1.to(comp_dtype)
    
    W1_T = W1.t()  # (d_in, d_hidden)
    
    z_block = torch.matmul(x_block, W1_T) + b1  # (block_size, d_hidden)
    
    if activation == "gelu":
        h_block = _gelu_jit(z_block)
        h_prime = _gelu_prime_jit(z_block)
    elif activation == "relu":
        h_block = F.relu(z_block)
        h_prime = _relu_prime_jit(z_block)
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    
    g_out_T = g_out_block.t()  # (d_out, block_size) - transpose 
    grad_W2_contrib = torch.matmul(g_out_T, h_block)  # (d_out, d_hidden)
    del g_out_T 
    
    # ∇_b2 = sum(g_out)
    grad_b2_contrib = g_out_block.sum(dim=0)  # (d_out,)
    
    # ∇_h = g_out @ W2
    grad_h = torch.matmul(g_out_block, W2)  # (block_size, d_hidden)
    
    # ∇_z = ∇_h * h'
    grad_z = grad_h * h_prime  # (block_size, d_hidden)
    del grad_h, h_prime  
    # ∇_W1 = sum(∇_z * x^T) = ∇_z^T @ x
    grad_z_T = grad_z.t()  # (d_hidden, block_size) - transpose 
    # (d_hidden, d_in)
    del grad_z_T  
    
    # ∇_b1 = sum(∇_z)
    grad_b1_contrib = grad_z.sum(dim=0)  # (d_hidden,)
    

    grad_x_block = torch.matmul(grad_z, W1)  # (block_size, d_in)
    

    del z_block, h_block, grad_z
    
    return grad_W1_contrib, grad_b1_contrib, grad_W2_contrib, grad_b2_contrib, grad_x_block


@torch.jit.script
def _ffn_block_jvp_jit(
    x: torch.Tensor, x_dot: torch.Tensor,
    W1: torch.Tensor, W1_dot: torch.Tensor,
    b1: torch.Tensor, b1_dot: torch.Tensor,
    W2: torch.Tensor, W2_dot: torch.Tensor,
    b2: torch.Tensor, b2_dot: torch.Tensor,
    use_gelu: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """JIT optimized FFN block JVP"""
    z1 = torch.matmul(x, W1.t()) + b1
    z1_dot = torch.matmul(x_dot, W1.t()) + torch.matmul(x, W1_dot.t()) + b1_dot
    
    if use_gelu:
        h = _gelu_jit(z1)
        h_dot = _gelu_prime_jit(z1) * z1_dot
    else:
        h = torch.relu(z1)
        h_dot = _relu_prime_jit(z1) * z1_dot
        
    y = torch.matmul(h, W2.t()) + b2
    y_dot = torch.matmul(h_dot, W2.t()) + torch.matmul(h, W2_dot.t()) + b2_dot
    
    return y, y_dot


@torch.no_grad()
def pgf_ffn_jvp(
    x: torch.Tensor, x_dot: torch.Tensor,
    W1: torch.Tensor, W1_dot: torch.Tensor,
    b1: torch.Tensor, b1_dot: torch.Tensor,
    W2: torch.Tensor, W2_dot: torch.Tensor,
    b2: torch.Tensor, b2_dot: torch.Tensor,
    block_size: int = 512,
    activation: str = "gelu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Analytical JVP for FFN.
    Computes y and y_dot in O(L) time and O(1) memory.
    """
    L, D = x.shape
    num_blocks = (L + block_size - 1) // block_size
    use_gelu = (activation == "gelu")
    
    y_list = []
    y_dot_list = []
    
    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, L)
        
        y_blk, y_dot_blk = _ffn_block_jvp_jit(
            x[start:end], x_dot[start:end],
            W1, W1_dot, b1, b1_dot,
            W2, W2_dot, b2, b2_dot,
            use_gelu
        )
        y_list.append(y_blk)
        y_dot_list.append(y_dot_blk)
        
    return torch.cat(y_list, dim=0), torch.cat(y_dot_list, dim=0)


@torch.no_grad()
def pgf_ffn_grad(
    x: torch.Tensor,  # (L, d_in)
    g_out: torch.Tensor,  # (L, d_out)
    W1: torch.Tensor,  # (d_hidden, d_in)
    b1: torch.Tensor,  # (d_hidden,)
    W2: torch.Tensor,  # (d_out, d_hidden)
    b2: torch.Tensor,  # (d_out,)
    block_size: int = 512,
    activation: str = "gelu",
    compute_input_grads: bool = False,
    fp32_accum: bool = True,
):

    x = x.detach()
    g_out = g_out.detach()
    W1 = W1.detach()
    b1 = b1.detach()
    W2 = W2.detach()
    b2 = b2.detach()
    
    L, d_in = x.shape
    d_out = g_out.shape[1]
    d_hidden = W1.shape[0]
    
    comp_dtype = torch.float32 if (fp32_accum and x.dtype in (torch.float16, torch.bfloat16)) else None
    work_dtype = comp_dtype if comp_dtype is not None else x.dtype
    
    grad_W1 = torch.zeros((d_hidden, d_in), device=x.device, dtype=work_dtype)
    grad_b1 = torch.zeros((d_hidden,), device=x.device, dtype=work_dtype)
    grad_W2 = torch.zeros((d_out, d_hidden), device=x.device, dtype=work_dtype)
    grad_b2 = torch.zeros((d_out,), device=x.device, dtype=work_dtype)
    
    if compute_input_grads:
        grad_x = torch.zeros_like(x, dtype=work_dtype)
    else:
        grad_x = None
    
    for start in range(0, L, block_size):
        end = min(start + block_size, L)
        x_block = x[start:end].detach()  # (block_size, d_in)
        g_out_block = g_out[start:end].detach()  # (block_size, d_out)
        
        grad_W1_contrib, grad_b1_contrib, grad_W2_contrib, grad_b2_contrib, grad_x_block = \
            _process_single_block_grad(
                x_block, g_out_block,
                W1, b1, W2, b2,
                activation=activation,
                comp_dtype=comp_dtype
            )
        
        grad_W1.add_(grad_W1_contrib)
        grad_b1.add_(grad_b1_contrib)
        grad_W2.add_(grad_W2_contrib)
        grad_b2.add_(grad_b2_contrib)
        
        del grad_W1_contrib, grad_b1_contrib, grad_W2_contrib, grad_b2_contrib
        del x_block, g_out_block  
        
        if compute_input_grads:         
            if grad_x_block.dtype != grad_x.dtype:
                grad_x_block = grad_x_block.to(dtype=grad_x.dtype)
            grad_x[start:end] = grad_x_block
            del grad_x_block 
    
    result = {
        "grad_W1": grad_W1.to(dtype=W1.dtype),
        "grad_b1": grad_b1.to(dtype=b1.dtype),
        "grad_W2": grad_W2.to(dtype=W2.dtype),
        "grad_b2": grad_b2.to(dtype=b2.dtype),
    }
    
    if compute_input_grads:
        result["grad_x"] = grad_x.to(dtype=x.dtype)
    
    return result


@torch.no_grad()
def pgf_ffn_forward(
    x: torch.Tensor,  # (L, d_in)
    W1: torch.Tensor,  # (d_hidden, d_in)
    b1: torch.Tensor,  # (d_hidden,)
    W2: torch.Tensor,  # (d_out, d_hidden)
    b2: torch.Tensor,  # (d_out,)
    block_size: int = 512,
    activation: str = "gelu",
    streaming: bool = False,
    callback=None,
):
    
    L, d_in = x.shape
    
    if not streaming:
        output_list = []
    
    num_blocks = (L + block_size - 1) // block_size
    
    use_gelu = (activation == "gelu")
    
    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, L)
        x_block = x[start:end]  # (block_size, d_in)
        
        y_block = _ffn_block_forward_jit(
            x_block,
            W1,
            b1,
            W2,
            b2,
            use_gelu,
        )
        
        y_block = y_block.detach()
        
        if streaming and callback:
            callback(y_block, start, end)
        elif not streaming:
            output_list.append(y_block)
    
    if streaming:
        return None
    else:
        return torch.cat(output_list, dim=0)


__all__ = [
    "pgf_ffn_grad",
    "pgf_ffn_forward",
]
