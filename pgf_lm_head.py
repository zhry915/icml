
from __future__ import annotations
import torch
import torch.nn.functional as F

_USE_COMPILE = hasattr(torch, 'compile')
if _USE_COMPILE:
    _compile_fn = torch.compile(mode="reduce-overhead", fullgraph=False)
else:
    _compile_fn = lambda x: x



@torch.jit.script
def _lm_head_block_forward_jit(
    hidden_block: torch.Tensor,  # (block_size, d_model)
    W_lm_head: torch.Tensor,  # (vocab_size, d_model)
) -> torch.Tensor:
    """
    JIT 优化的 LM Head block 前向传播
    """
    W_lm_head_T = W_lm_head.t()  # (d_model, vocab_size)
    logits_block = torch.matmul(hidden_block, W_lm_head_T)  # (block_size, vocab_size)
    return logits_block


@_compile_fn
def _process_single_block_grad(
    hidden_block: torch.Tensor,  # (block_size, d_model)
    g_logits_block: torch.Tensor,  # (block_size, vocab_size)
    W_lm_head: torch.Tensor,  # (vocab_size, d_model)
    comp_dtype: torch.dtype | None = None,
):
   
    if comp_dtype is not None:
        hidden_block = hidden_block.to(comp_dtype)
        g_logits_block = g_logits_block.to(comp_dtype)
        W_lm_head = W_lm_head.to(comp_dtype)
    
    grad_W_lm_head_contrib = torch.matmul(g_logits_block.t(), hidden_block)  # (vocab_size, d_model)
    
    grad_hidden_block = torch.matmul(g_logits_block, W_lm_head)  # (block_size, d_model)
    
    return grad_W_lm_head_contrib, grad_hidden_block


def pgf_lm_head_grad(
    hidden: torch.Tensor,  # (L, d_model)
    g_logits: torch.Tensor,  # (L, vocab_size)
    W_lm_head: torch.Tensor,  # (vocab_size, d_model)
    block_size: int = 512,
    compute_input_grads: bool = True,
    fp32_accum: bool = True,
) -> dict:
    
    L, d_model = hidden.shape
    vocab_size = W_lm_head.shape[0]

    comp_dtype = torch.float32 if (fp32_accum and hidden.dtype in (torch.float16, torch.bfloat16)) else None
    work_dtype = comp_dtype if comp_dtype is not None else hidden.dtype
    
    grad_W_lm_head = torch.zeros((vocab_size, d_model), device=hidden.device, dtype=work_dtype)
    
    if compute_input_grads:
        grad_hidden = torch.zeros_like(hidden, dtype=work_dtype)
    else:
        grad_hidden = None
    
    for start in range(0, L, block_size):
        end = min(start + block_size, L)
        hidden_block = hidden[start:end].detach()  # (block_size, d_model)
        g_logits_block = g_logits[start:end].detach()  # (block_size, vocab_size)
        
        grad_W_lm_head_contrib, grad_hidden_block = _process_single_block_grad(
            hidden_block, g_logits_block, W_lm_head, comp_dtype=comp_dtype
        )
        
        grad_W_lm_head.add_(grad_W_lm_head_contrib)
        
        del grad_W_lm_head_contrib, hidden_block, g_logits_block
        
        if compute_input_grads:
            if grad_hidden_block.dtype != grad_hidden.dtype:
                grad_hidden_block = grad_hidden_block.to(dtype=grad_hidden.dtype)
            grad_hidden[start:end] = grad_hidden_block
            del grad_hidden_block
    
    result = {
        "grad_W_lm_head": grad_W_lm_head.to(dtype=W_lm_head.dtype),
    }
    
    if compute_input_grads:
        result["grad_hidden"] = grad_hidden.to(dtype=hidden.dtype)
    
    return result


@torch.no_grad()
def pgf_lm_head_forward(
    hidden: torch.Tensor,  # (L, d_model)
    W_lm_head: torch.Tensor,  # (vocab_size, d_model)
    block_size: int = 512,
    streaming: bool = False,
    callback=None,
):
   
    L, d_model = hidden.shape
    num_blocks = (L + block_size - 1) // block_size
    
    if not streaming:
        output_list = []
    
    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, L)
        hidden_block = hidden[start:end]  # (block_size, d_model)
        
        logits_block = _lm_head_block_forward_jit(hidden_block, W_lm_head)  # (block_size, vocab_size)
        
        logits_block = logits_block.detach()
        
        del hidden_block
        
        if streaming and callback:
            callback(logits_block, start, end)
        elif not streaming:
            output_list.append(logits_block)
    
    if streaming:
        return None
    else:
        return torch.cat(output_list, dim=0)
