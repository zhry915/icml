import torch
import torch.nn.functional as F
import numpy as np
import csv

from pgf_embedding import pgf_embedding_forward, pgf_embedding_grad
from pgf_mamba_block import PGFMambaBlock, PGFMambaSequential
from pgf_lm_head import pgf_lm_head_forward, pgf_lm_head_grad

def verify_multi_layer_gradients():
    print("--- Running Multi-Layer Parameter Gradient Verification ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    seed = 1
    L = 512
    B = 2
    d_model = 64
    d_state = 16
    d_ff = 128
    block_size = 128
    vocab_size = 1000
    
    layer_configs = [2, 4, 8, 16]
    csv_file = "rebuttal_grads_multilayer_exactness.csv"
    results_data = []
    
    # Write to CSV header
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Layers", "Parameter", "Max_Diff", "Status"])
        
    for n_layers in layer_configs:
        print(f"\n======================================")
        print(f" Testing {n_layers} Layers Configuration")
        print(f"======================================")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        input_ids = torch.randint(0, vocab_size, (B, L), device=device)
        targets = torch.randint(0, vocab_size, (B, L), device=device)

        emb_weight_pgf = torch.randn(vocab_size, d_model, device=device, requires_grad=True)
        lm_head_weight_pgf = torch.randn(vocab_size, d_model, device=device, requires_grad=True)
        blocks_pgf = [PGFMambaBlock(d_model, d_state, d_ff, block_size).to(device) for _ in range(n_layers)]
        backbone_pgf = PGFMambaSequential(blocks_pgf).to(device)

        emb_weight_auto = emb_weight_pgf.detach().clone().requires_grad_(True)
        lm_head_weight_auto = lm_head_weight_pgf.detach().clone().requires_grad_(True)
        blocks_auto = [PGFMambaBlock(d_model, d_state, d_ff, block_size).to(device) for _ in range(n_layers)]
        backbone_auto = PGFMambaSequential(blocks_auto).to(device)
        
        backbone_auto.load_state_dict(backbone_pgf.state_dict())

        # ==========================================
        # Pass A: Standard Autograd
        # ==========================================
        # Forward
        hidden_auto = F.embedding(input_ids, emb_weight_auto)
        for layer in backbone_auto.layers:
            hidden_auto = layer.forward(hidden_auto)
        logits_auto = torch.matmul(hidden_auto, lm_head_weight_auto.t())
        
        loss_auto = F.cross_entropy(logits_auto.view(-1, vocab_size), targets.view(-1))
        # Backward
        loss_auto.backward()

        # ==========================================
        # Pass B: PGF O(1) Memory
        # ==========================================
        
        def ce_loss_and_grad(logits, targets_inner):
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = targets_inner.view(-1)
            
            loss_val = F.cross_entropy(logits_flat, targets_flat)
            
            probs = F.softmax(logits_flat, dim=-1)
            g_logits_flat = probs.clone()
            g_logits_flat[torch.arange(g_logits_flat.size(0)), targets_flat] -= 1.0
            g_logits_flat /= logits_flat.size(0) 
            
            g_logits_res = g_logits_flat.view(logits.shape)
            return loss_val, g_logits_res

        # Forward Pass 1
        hidden_pgf = pgf_embedding_forward(input_ids, emb_weight_pgf, block_size)
        layer_inputs = []
        curr_x = hidden_pgf
        for layer in backbone_pgf.layers:
            layer_inputs.append(curr_x.detach().clone())
            curr_x = layer.pgf_forward(curr_x)
        final_hidden_pgf = curr_x
        
        logits_pgf_flat = pgf_lm_head_forward(final_hidden_pgf.view(-1, d_model), lm_head_weight_pgf, block_size)
        logits_pgf = logits_pgf_flat.view(B, L, -1)
        
        loss_pgf, g_logits = ce_loss_and_grad(logits_pgf, targets)
        
        # Backward Pass 2
        lm_head_res = pgf_lm_head_grad(final_hidden_pgf.view(-1, d_model), g_logits.view(-1, vocab_size), lm_head_weight_pgf, block_size, compute_input_grads=True)
        g_hidden = lm_head_res["grad_hidden"].view(B, L, -1)
        
        # 2. Backbone
        curr_g = g_hidden
        for i in reversed(range(n_layers)):
            layer = backbone_pgf.layers[i]
            layer_in = layer_inputs.pop()
            res = layer.pgf_backward(layer_in, curr_g)
            for name, param in layer.named_parameters():
                if name in res["grads"]:
                    if param.grad is None:
                        param.grad = res["grads"][name].clone()
                    else:
                        param.grad += res["grads"][name]
            
            curr_g = res["grad_input"]
            del layer_in, res
            
        # 3. Embedding
        emb_res = pgf_embedding_grad(input_ids, curr_g, emb_weight_pgf, block_size)
        emb_weight_pgf.grad = emb_res["grad_weight"]
        # LM Head weight
        lm_head_weight_pgf.grad = lm_head_res["grad_W_lm_head"]

        # ==========================================
        # Output Results / Validation Comparison
        # ==========================================
        print(f"Autograd Loss: {loss_auto.item():.6f}")
        print(f"PGF Loss:      {loss_pgf.item():.6f}")
        
        diff_loss = abs(loss_auto.item() - loss_pgf.item())
        print(f"Loss Difference: {diff_loss:.6e}\n")

        print("Parameter Gradient Differences (L_inf norm):")
        all_match = True
        layer_results = []
        
        # Compare Embedding
        diff_emb = torch.max(torch.abs(emb_weight_auto.grad - emb_weight_pgf.grad)).item()
        status_emb = "PASS" if diff_emb < 1e-4 else "FAIL"
        layer_results.append([n_layers, "Embedding.weight", f"{diff_emb:.2e}", status_emb])
        if diff_emb >= 1e-4: all_match = False
        
        # Compare LM Head
        diff_lm = torch.max(torch.abs(lm_head_weight_auto.grad - lm_head_weight_pgf.grad)).item()
        status_lm = "PASS" if diff_lm < 1e-4 else "FAIL"
        layer_results.append([n_layers, "LM_Head.weight", f"{diff_lm:.2e}", status_lm])
        if diff_lm >= 1e-4: all_match = False

        # Compare parameters of all layers
        for i in range(n_layers):
            # Take some representative parameters for comparison
            diff_A = torch.max(torch.abs(backbone_auto.layers[i].mamba.A_log.grad - backbone_pgf.layers[i].mamba.A_log.grad)).item()
            status_A = "PASS" if diff_A < 1e-4 else "FAIL"
            layer_results.append([n_layers, f"Layer {i} mamba.A_log", f"{diff_A:.2e}", status_A])
            if diff_A >= 1e-4: all_match = False
            
            diff_W1 = torch.max(torch.abs(backbone_auto.layers[i].W1.grad - backbone_pgf.layers[i].W1.grad)).item()
            status_W1 = "PASS" if diff_W1 < 1e-4 else "FAIL"
            layer_results.append([n_layers, f"Layer {i} FFN W1", f"{diff_W1:.2e}", status_W1])
            if diff_W1 >= 1e-4: all_match = False
            
        for row in layer_results:
            if "Layer" not in row[1] or "Layer 0" in row[1] or f"Layer {n_layers-1}" in row[1]:
                print(f"{row[1]:<30}: Max Diff = {row[2]} | {row[3]}")
                
        if all_match:
            print(f"SUCCESS: {n_layers} Layers gradients match perfectly!")
        else:
            print(f"WARNING: Discrepancy detected at {n_layers} Layers!")
            
        results_data.extend(layer_results)
        
        del backbone_auto, backbone_pgf, emb_weight_auto, emb_weight_pgf, lm_head_weight_auto, lm_head_weight_pgf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(results_data)
        
    print(f"\nAll Multi-Layer tests finished. Data saved to {csv_file}")

if __name__ == "__main__":
    verify_multi_layer_gradients()
