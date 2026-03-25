import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import csv
import pyarrow as pa
import matplotlib.pyplot as plt

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
def run_extreme_training():
    print("--- WikiText-2 Extreme Length PGF Training (L=10^5, 16 Layers) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Find dataset
    train_arrow = "rebuttal_code/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-train.arrow"
    if not os.path.exists(train_arrow):
        # Fallback to the other directory if needed
        train_arrow = "wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-train.arrow"
    
    if not os.path.exists(train_arrow):
        print(f"Error: Dataset not found at {train_arrow}")
        return
        
    seq_len = 100000   # Extreme length
    batch_size = 1     # Batch size 1 for extreme length
    d_model = 64       
    d_state = 16
    d_ff = 128         
    block_size = 64
    max_steps = 20     # Just run a few steps to prove convergence and no OOM
    n_layers = 16      # 16 layers
    
    # Load data
    dataset = WikiTextArrowDataset(train_arrow, seq_len=seq_len, max_samples=max_steps * batch_size)
    vocab_size = dataset.vocab_size
    
    csv_path = "results/extreme_length_convergence.csv"
    os.makedirs("results", exist_ok=True)
    
    # Initialize CSV and write header
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "PGF_Loss", "PGF_Time", "PGF_Mem_MB"])
        
    print(f"\n==========================================")
    print(f"EXTREME PGF: Layers={n_layers}, d_model={d_model}, seq_len={seq_len}")
    print(f"==========================================")
    
    # Initialize model
    torch.manual_seed(42)
    model_pgf = ExternalPGFLanguageModel(vocab_size, d_model, d_state, d_ff, n_layers, block_size, device)
    
    # Training loop
    step = 0
    batches = list(dataset.get_batches(batch_size))
    
    for input_ids, targets in batches:
        if step >= max_steps: break
        
        input_ids, targets = input_ids.to(device), targets.to(device)
        
        # We don't even try Autograd here because it will 100% OOM
        # Train PGF
        try:
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
            start_pgf = time.time()
            loss_pgf = model_pgf.train_step_pgf(input_ids, targets)
            time_pgf = time.time() - start_pgf
            mem_pgf = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            
            # Write to CSV in real-time
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step, loss_pgf, time_pgf, mem_pgf])
                
            log_str = f"Extreme PGF | Step {step:03d} | Loss: {loss_pgf:.4f} | Time: {time_pgf:.2f}s | Peak Mem: {mem_pgf:.1f}MB\n"
            print(log_str.strip())
            with open("results/extreme_length_convergence.txt", "a") as f:
                f.write(log_str)
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [STATUS] PGF: OOM on Extreme Length")
                break
            else:
                raise e
            
        step += 1
        torch.cuda.empty_cache()

    print(f"\nExtreme training finished! CSV data saved to {csv_path}")

if __name__ == "__main__":
    run_extreme_training()