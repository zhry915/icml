import torch
import torch.nn.functional as F
import numpy as np
import csv
import time
import matplotlib.pyplot as plt

# Import core components
from pgf_embedding import pgf_embedding_forward, pgf_embedding_grad
from pgf_mamba_block import PGFMambaBlock, PGFMambaSequential
from pgf_lm_head import pgf_lm_head_forward, pgf_lm_head_grad

def get_peak_memory():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2) 
    return 0

def run_memory_benchmark():
    print("--- Running Multi-Layer Memory Benchmark (O(1) vs O(L)) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("CUDA not available. Memory benchmark requires GPU.")
        return

    seed = 1
    B = 2
    d_model = 128 
    d_state = 16
    d_ff = 256
    block_size = 256
    vocab_size = 1000
    
    layer_configs = [2, 4, 8]
    L_configs = [512, 1024, 2048, 4096, 8192]
    
    csv_file = "rebuttal_multilayer_memory.csv"
    results_data = []
    
    # Append to CSV after each configuration to prevent data loss
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Layers", "Seq_Length", "Mode", "Peak_Memory_MB", "Time_Seconds"])

    # Store data for plotting
    plot_data = {
        "PGF": {2: [], 4: [], 8: []},
        "Auto": {2: [], 4: [], 8: []}
    }
    plot_time_data = {
        "PGF": {2: [], 4: [], 8: []},
        "Auto": {2: [], 4: [], 8: []}
    }
    
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

    for n_layers in layer_configs:
        for L in L_configs:
            print(f"\nTesting: Layers={n_layers}, L={L}")
            results_data = []
            
            torch.manual_seed(seed)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            input_ids = torch.randint(0, vocab_size, (B, L), device=device)
            targets = torch.randint(0, vocab_size, (B, L), device=device)
            
            # ==========================================
            # 1. PGF Memory Test (using identical instances for fair comparison)
            # ==========================================
            blocks = [PGFMambaBlock(d_model, d_state, d_ff, block_size).to(device) for _ in range(n_layers)]
            backbone = PGFMambaSequential(blocks).to(device)
            emb_weight = torch.randn(vocab_size, d_model, device=device, requires_grad=True)
            lm_head_weight = torch.randn(vocab_size, d_model, device=device, requires_grad=True)
            
            torch.cuda.reset_peak_memory_stats()
            
            # PGF Forward
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            
            hidden_pgf = pgf_embedding_forward(input_ids, emb_weight, block_size)
            layer_inputs = []
            curr_x = hidden_pgf
            for layer in backbone.layers:
                layer_inputs.append(curr_x.detach().clone())
                curr_x = layer.pgf_forward(curr_x)
            final_hidden_pgf = curr_x
            
            logits_pgf_flat = pgf_lm_head_forward(final_hidden_pgf.view(-1, d_model), lm_head_weight, block_size)
            logits_pgf = logits_pgf_flat.view(B, L, -1)
            loss_pgf, g_logits = ce_loss_and_grad(logits_pgf, targets)
            
            # PGF Backward
            lm_head_res = pgf_lm_head_grad(final_hidden_pgf.view(-1, d_model), g_logits.view(-1, vocab_size), lm_head_weight, block_size, compute_input_grads=True)
            g_hidden = lm_head_res["grad_hidden"].view(B, L, -1)
            
            curr_g = g_hidden
            for i in reversed(range(n_layers)):
                layer = backbone.layers[i]
                layer_in = layer_inputs.pop()
                res = layer.pgf_backward(layer_in, curr_g)
                curr_g = res["grad_input"]
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            pgf_time = time.time() - start_time
                
            pgf_mem = get_peak_memory()
            print(f"  PGF Peak Memory: {pgf_mem:.2f} MB, Time: {pgf_time:.4f} s")
            results_data.append([n_layers, L, "PGF", pgf_mem, pgf_time])
            plot_data["PGF"][n_layers].append(pgf_mem)
            plot_time_data["PGF"][n_layers].append(pgf_time)
            
            # Cleanup
            del blocks, backbone, emb_weight, lm_head_weight
            del hidden_pgf, curr_x, final_hidden_pgf, logits_pgf, loss_pgf, g_logits
            del lm_head_res, g_hidden, curr_g
            torch.cuda.empty_cache()
            
            # ==========================================
            # 2. Autograd Memory Test
            # ==========================================
            blocks = [PGFMambaBlock(d_model, d_state, d_ff, block_size).to(device) for _ in range(n_layers)]
            backbone = PGFMambaSequential(blocks).to(device)
            emb_weight = torch.randn(vocab_size, d_model, device=device, requires_grad=True)
            lm_head_weight = torch.randn(vocab_size, d_model, device=device, requires_grad=True)
            
            torch.cuda.reset_peak_memory_stats()
            
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.time()
                
                # Auto Forward
                hidden_auto = F.embedding(input_ids, emb_weight)
                for layer in backbone.layers:
                    hidden_auto = layer.forward(hidden_auto)
                logits_auto = torch.matmul(hidden_auto, lm_head_weight.t())
                loss_auto = F.cross_entropy(logits_auto.view(-1, vocab_size), targets.view(-1))
                
                # Auto Backward
                loss_auto.backward()
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                auto_time = time.time() - start_time
                
                auto_mem = get_peak_memory()
                print(f"  Auto Peak Memory: {auto_mem:.2f} MB, Time: {auto_time:.4f} s")
                results_data.append([n_layers, L, "Autograd", auto_mem, auto_time])
                plot_data["Auto"][n_layers].append(auto_mem)
                plot_time_data["Auto"][n_layers].append(auto_time)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  Auto Peak Memory: OOM (Out of Memory)")
                    results_data.append([n_layers, L, "Autograd", "OOM", "OOM"])
                    plot_data["Auto"][n_layers].append(None) # OOM
                    plot_time_data["Auto"][n_layers].append(None) # OOM
                else:
                    raise e
            finally:
                del blocks, backbone, emb_weight, lm_head_weight
                torch.cuda.empty_cache()

            # Append to CSV after each loop
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(results_data)

    print(f"\nMemory benchmark complete. Results saved to {csv_file}")
    
    # ==========================================
    # Plot 1: Peak Memory
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    colors_auto = ['#ff9999', '#ff4d4d', '#cc0000']
    colors_pgf = ['#99ccff', '#3399ff', '#0066cc']
    markers = ['o', 's', '^']
    
    for idx, n_layers in enumerate(layer_configs):
        # Plot Autograd curve (filter out OOM)
        valid_auto_L = []
        valid_auto_mem = []
        for i, mem in enumerate(plot_data["Auto"][n_layers]):
            if mem is not None:
                valid_auto_L.append(L_configs[i])
                valid_auto_mem.append(mem)
                
        if valid_auto_L:
            plt.plot(valid_auto_L, valid_auto_mem, color=colors_auto[idx], marker=markers[idx], 
                     linestyle='--', linewidth=2, label=f'Autograd ({n_layers} Layers)')
            
        # Plot PGF curve
        plt.plot(L_configs, plot_data["PGF"][n_layers], color=colors_pgf[idx], marker=markers[idx], 
                 linestyle='-', linewidth=2, label=f'PGF ({n_layers} Layers)')

    plt.xlabel('Sequence Length (L)', fontsize=12)
    plt.ylabel('Peak GPU Memory (MB)', fontsize=12)
    plt.title('Memory Scaling: Multi-layer O(1) PGF vs O(L) Autograd', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    
    pdf_file = "result/Fig_Rebuttal_Multilayer_Memory.pdf"
    import os
    if not os.path.exists("result"):
        os.makedirs("result")
    plt.savefig(pdf_file, dpi=300)
    print(f"Memory plot saved to {pdf_file}")
    plt.close()

    # ==========================================
    # Plot 2: Time Overhead
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    for idx, n_layers in enumerate(layer_configs):
        valid_auto_L = []
        valid_auto_time = []
        for i, t in enumerate(plot_time_data["Auto"][n_layers]):
            if t is not None:
                valid_auto_L.append(L_configs[i])
                valid_auto_time.append(t)
                
        if valid_auto_L:
            plt.plot(valid_auto_L, valid_auto_time, color=colors_auto[idx], marker=markers[idx], 
                     linestyle='--', linewidth=2, label=f'Autograd ({n_layers} Layers)')
            
        plt.plot(L_configs, plot_time_data["PGF"][n_layers], color=colors_pgf[idx], marker=markers[idx], 
                 linestyle='-', linewidth=2, label=f'PGF ({n_layers} Layers)')

    plt.xlabel('Sequence Length (L)', fontsize=12)
    plt.ylabel('Time (Seconds)', fontsize=12)
    plt.title('Time Scaling: Multi-layer PGF vs Autograd', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    
    pdf_time_file = "result/Fig_Rebuttal_Multilayer_Time.pdf"
    plt.savefig(pdf_time_file, dpi=300)
    print(f"Time plot saved to {pdf_time_file}")
    plt.close()

if __name__ == "__main__":
    run_memory_benchmark()