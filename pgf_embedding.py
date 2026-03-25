
from __future__ import annotations
import torch
import torch.nn.functional as F


def _embedding_block_forward_jit(
    input_ids_block: torch.Tensor,  # (block_size,)
    weight: torch.Tensor,  # (vocab_size, d_model)
) -> torch.Tensor:
    return F.embedding(input_ids_block, weight)  # (block_size, d_model)


@torch.no_grad()
def pgf_embedding_grad(
    input_ids: torch.Tensor,  
    g_out: torch.Tensor,  
    weight: torch.Tensor,  
    block_size: int = 512,
    compute_input_grads: bool = False,  
    fp32_accum: bool = True,
):

    input_ids = input_ids.detach()
    g_out = g_out.detach()
    weight = weight.detach()
    
 
    original_shape = input_ids.shape
    if len(original_shape) == 2:
        B, L = original_shape
        input_ids = input_ids.view(-1)  # (B*L,)
        g_out = g_out.view(-1, g_out.shape[-1])  # (B*L, d_model)
        total_length = B * L
    else:
        L = original_shape[0]
        total_length = L
    
    vocab_size, d_model = weight.shape
    
    comp_dtype = torch.float32 if (fp32_accum and g_out.dtype in (torch.float16, torch.bfloat16)) else None
    work_dtype = comp_dtype if comp_dtype is not None else weight.dtype
    
   
    grad_weight = torch.zeros((vocab_size, d_model), device=weight.device, dtype=work_dtype)
    
  
    for start in range(0, total_length, block_size):
        end = min(start + block_size, total_length)
        input_ids_block = input_ids[start:end]  # (block_size,)
        g_out_block = g_out[start:end]  # (block_size, d_model)
        
        if comp_dtype is not None:
            g_out_block = g_out_block.to(comp_dtype)
        
        
        grad_weight.index_add_(
            dim=0,  
            index=input_ids_block,  
            source=g_out_block,  
        )
        
        
        del input_ids_block, g_out_block
    
    
    result = {
        "grad_weight": grad_weight.to(dtype=weight.dtype),
    }
    
    return result


@torch.no_grad()
def pgf_embedding_forward(
    input_ids: torch.Tensor,  
    weight: torch.Tensor,  
    block_size: int = 512,
    streaming: bool = False,
    callback=None,
):
    
    original_shape = input_ids.shape
    if len(original_shape) == 2:
        B, L = original_shape
        input_ids = input_ids.view(-1)  # (B*L,)
        total_length = B * L
        has_batch = True
    else:
        L = original_shape[0]
        total_length = L
        has_batch = False
    
    if not streaming:
        output_list = []
    
   
    num_blocks = (total_length + block_size - 1) // block_size
    
    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, total_length)
        input_ids_block = input_ids[start:end]  # (block_size,)
        

        emb_block = _embedding_block_forward_jit(input_ids_block, weight)  # (block_size, d_model)
        

        emb_block = emb_block.detach()
        

        if streaming and callback:
            callback(emb_block, start, end)
        elif not streaming:
            output_list.append(emb_block)
    

    if streaming:
        return None
    else:
        embeddings = torch.cat(output_list, dim=0)  # (B*L, d_model) or (L, d_model)
        if has_batch:
            embeddings = embeddings.view(B, L, -1)  # (B, L, d_model)
        return embeddings


__all__ = [
    "pgf_embedding_grad",
    "pgf_embedding_forward",
]
