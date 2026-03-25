import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import csv
import pyarrow as pa
import matplotlib.pyplot as plt

# Import core components
from pgf_embedding import pgf_embedding_forward, pgf_embedding_grad
from pgf_mamba_block import PGFMambaBlock, PGFMambaSequential
from pgf_lm_head import pgf_lm_head_forward, pgf_lm_head_grad

# ==========================================
# 1. WikiText-2 Dataset Loader (Arrow Format)
# ==========================================
class WikiTextArrowDataset:
    def __init__(self, arrow_path, seq_len=256, max_samples=None):

        try:
            from datasets import Dataset
            ds = Dataset.from_file(arrow_path)
            texts = ds['text']
        except Exception as e:
            print(f"Failed to load with datasets: {e}, falling back to dummy data for benchmark")
            texts = ["Hello world, this is a test sequence. " * 100 for _ in range(max_samples if max_samples else 1000)]
            
        # Concatenate text
        full_text = " ".join([t for t in texts if t.strip()])
        # Minimal character-level Tokenizer (ASCII)
        self.vocab_size = 256
        data = torch.tensor([min(ord(c), 255) for c in full_text], dtype=torch.long)
        
        # Truncate to sequences of length L
        num_batches = len(data) // (seq_len + 1)
        if max_samples:
            num_batches = min(num_batches, max_samples)
            
        data = data[:num_batches * (seq_len + 1)]
        data = data.view(num_batches, seq_len + 1)
        
        self.inputs = data[:, :-1].clone()
        self.targets = data[:, 1:].clone()
        self.num_batches = num_batches
        self.seq_len = seq_len
        print(f"Loaded {num_batches} sequences of length {seq_len} from {arrow_path}")

    def get_batches(self, batch_size):
        for i in range(0, self.num_batches, batch_size):
            end_i = min(i + batch_size, self.num_batches)
            yield self.inputs[i:end_i], self.targets[i:end_i]

# ==========================================
# 2. External Assembly: PGF Language Model Wrapper
# ==========================================
class ExternalPGFLanguageModel:

    def __init__(self, vocab_size, d_model, d_state, d_ff, n_layers, block_size, device):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.block_size = block_size
        self.n_layers = n_layers
        self.device = device
        
        # Independent trainable parameters (manually managed, not Module params)
        self.emb_weight = nn.Parameter(torch.randn(vocab_size, d_model, device=device) * 0.02)
        self.lm_head_weight = nn.Parameter(torch.randn(vocab_size, d_model, device=device) * 0.02)
        
        # Backbone network
        blocks = [PGFMambaBlock(d_model, d_state, d_ff, block_size).to(device) for _ in range(n_layers)]
        self.backbone = PGFMambaSequential(blocks).to(device)
        
        # Collect all parameters to optimize
        self.params = [self.emb_weight, self.lm_head_weight] + list(self.backbone.parameters())
        self.optimizer = torch.optim.AdamW(self.params, lr=1e-3, weight_decay=0.01)

    def train_step_auto(self, input_ids, targets):
        """Standard Autograd Training Step"""
        self.optimizer.zero_grad()
        
        # Forward
        hidden = F.embedding(input_ids, self.emb_weight)
        for layer in self.backbone.layers:
            hidden = layer.forward(hidden)
        logits = torch.matmul(hidden, self.lm_head_weight.t())
        
        # Loss
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        
        # Backward & Step
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train_step_pgf(self, input_ids, targets):
        """PGF O(1) Training Step"""
        B, L = input_ids.shape
        self.optimizer.zero_grad()
        
        # Forward Pass 1
        hidden = pgf_embedding_forward(input_ids, self.emb_weight, self.block_size)
        layer_inputs = []
        curr_x = hidden
        for layer in self.backbone.layers:
            layer_inputs.append(curr_x.detach().clone())
            curr_x = layer.pgf_forward(curr_x)
        final_hidden = curr_x
        
        logits_flat = pgf_lm_head_forward(final_hidden.view(-1, self.d_model), self.lm_head_weight, self.block_size)
        logits = logits_flat.view(B, L, -1)
        
        # Compute Loss and Gradient
        logits_flat = logits.view(-1, self.vocab_size)
        targets_flat = targets.view(-1)
        loss = F.cross_entropy(logits_flat, targets_flat)
        
        with torch.no_grad():
            probs = F.softmax(logits_flat, dim=-1)
            g_logits_flat = probs.clone()
            g_logits_flat[torch.arange(g_logits_flat.size(0)), targets_flat] -= 1.0
            g_logits_flat /= logits_flat.size(0)
            g_logits = g_logits_flat.view(logits.shape)
        
        # Backward Pass 2
        lm_head_res = pgf_lm_head_grad(final_hidden.view(-1, self.d_model), g_logits.view(-1, self.vocab_size), self.lm_head_weight, self.block_size, compute_input_grads=True)
        g_hidden = lm_head_res["grad_hidden"].view(B, L, -1)
        
        curr_g = g_hidden
        for i in reversed(range(self.n_layers)):
            layer = self.backbone.layers[i]
            layer_in = layer_inputs.pop()
            res = layer.pgf_backward(layer_in, curr_g)
            for name, param in layer.named_parameters():
                if name in res["grads"]:
                    if param.grad is None:
                        param.grad = res["grads"][name].clone()
                    else:
                        param.grad += res["grads"][name]
            curr_g = res["grad_input"]
            
        emb_res = pgf_embedding_grad(input_ids, curr_g, self.emb_weight, self.block_size)
        self.emb_weight.grad = emb_res["grad_weight"]
        self.lm_head_weight.grad = lm_head_res["grad_W_lm_head"]
        
        self.optimizer.step()
        return loss.item()


# ==========================================
# 3. Training Main Loop
# ==========================================
def run_wikitext_training():
    print("--- WikiText-2 End-to-End Training (PGF vs Autograd) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Find dataset
    train_arrow = "wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-train.arrow"
    if not os.path.exists(train_arrow):
        print(f"Error: Dataset not found at {train_arrow}")
        return
        
  
    seq_len = 1024     
    batch_size = 2     
    d_model = 64       
    d_state = 16
    d_ff = 128         
    block_size = 64
    max_steps = 100    
    
   
    layer_configs = [2, 4, 8]  # Different layer counts to test
    
    # Load data
    dataset = WikiTextArrowDataset(train_arrow, seq_len=seq_len, max_samples=max_steps * batch_size)
    vocab_size = dataset.vocab_size
    
    plt.figure(figsize=(10, 6))
    colors = ['red', 'green', 'blue'] # Corresponds to 2, 4, 8 layers
    
    csv_path = "result/wikitext_convergence.csv"
    os.makedirs("result", exist_ok=True)
    
    # Initialize CSV and write header
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Layers", "Step", "Autograd_Loss", "PGF_Loss", "Auto_Time", "Auto_Mem_MB", "PGF_Time", "PGF_Mem_MB"])
        
    for idx, n_layers in enumerate(layer_configs):
        print(f"\n==========================================")
        print(f"Model config: Layers={n_layers}, d_model={d_model}, seq_len={seq_len}")
        print(f"==========================================")
        
        # Initialize two identical models
        torch.manual_seed(42)
        model_auto = ExternalPGFLanguageModel(vocab_size, d_model, d_state, d_ff, n_layers, block_size, device)
        
        torch.manual_seed(42)
        model_pgf = ExternalPGFLanguageModel(vocab_size, d_model, d_state, d_ff, n_layers, block_size, device)
        
        # Force weight synchronization
        model_pgf.emb_weight.data.copy_(model_auto.emb_weight.data)
        model_pgf.lm_head_weight.data.copy_(model_auto.lm_head_weight.data)
        model_pgf.backbone.load_state_dict(model_auto.backbone.state_dict())
        
        history_auto = []
        history_pgf = []
        
        # Training loop
        step = 0
        batches = list(dataset.get_batches(batch_size))
        
        for input_ids, targets in batches:
            if step >= max_steps: break
            
            input_ids, targets = input_ids.to(device), targets.to(device)
            
            # Train Auto
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
            start_auto = time.time()
            loss_auto = model_auto.train_step_auto(input_ids, targets)
            time_auto = time.time() - start_auto
            mem_auto = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            history_auto.append(loss_auto)
            
            # Train PGF
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
            start_pgf = time.time()
            loss_pgf = model_pgf.train_step_pgf(input_ids, targets)
            time_pgf = time.time() - start_pgf
            mem_pgf = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            history_pgf.append(loss_pgf)
            
            # Write to CSV in real-time to prevent data loss on OOM
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([n_layers, step, loss_auto, loss_pgf, time_auto, mem_auto, time_pgf, mem_pgf])
                
            if step % 20 == 0 or step == max_steps - 1:
                print(f"Layer {n_layers} | Step {step:03d} | Auto Loss: {loss_auto:.4f} | PGF Loss: {loss_pgf:.4f} | Auto Mem: {mem_auto:.1f}MB | PGF Mem: {mem_pgf:.1f}MB")
                
            step += 1
            
        # Plot curve for this layer
        plt.plot(history_auto, label=f'Autograd ({n_layers}L)', color=colors[idx], alpha=0.5, linewidth=2, linestyle='-')
        plt.plot(history_pgf, label=f'PGF ({n_layers}L)', color=colors[idx], alpha=0.9, linewidth=2, linestyle='--')
        
        # Free memory
        del model_auto, model_pgf, history_auto, history_pgf
        torch.cuda.empty_cache()

    # ==========================================
    # Save Plot
    # ==========================================
    plt.xlabel('Training Steps')
    plt.ylabel('Cross Entropy Loss')
    plt.title('WikiText-2 Convergence: Multi-Layer PGF vs Autograd')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    pdf_path = "result/Fig_Rebuttal_WikiText_Convergence.pdf"
    plt.savefig(pdf_path, dpi=300)
    print(f"\nTraining finished! Convergence plot saved to {pdf_path}")
    print(f"CSV data saved to {csv_path}")

if __name__ == "__main__":
    run_wikitext_training()