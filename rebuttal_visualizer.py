import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set plotting style for academic publication
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (16, 10),
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": '--'
})

def plot_rebuttal_figures():
    results_dir = "results"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a 2x2 grid for the figures
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ---------------------------------------------------------
    # Panel 1: Memory Scaling (Log-Log)
    # ---------------------------------------------------------
    ax1 = axes[0, 0]
    mem_file = os.path.join(results_dir, "rebuttal_multilayer_memory.csv")
    if os.path.exists(mem_file):
        df_mem = pd.read_csv(mem_file)
        # Ensure we handle OOM cases if they are strings
        df_mem['Peak_Memory_MB'] = pd.to_numeric(df_mem['Peak_Memory_MB'], errors='coerce')
        
        for layers in df_mem['Layers'].unique():
            df_l = df_mem[df_mem['Layers'] == layers]
            
            # Autograd
            auto_data = df_l[df_l['Mode'] == 'Autograd'].dropna(subset=['Peak_Memory_MB'])
            ax1.plot(auto_data['Seq_Length'], auto_data['Peak_Memory_MB'], 
                    label=f'Autograd (L={layers})', marker='o', linestyle='--')
            
            # PGF
            pgf_data = df_l[df_l['Mode'] == 'PGF'].dropna(subset=['Peak_Memory_MB'])
            ax1.plot(pgf_data['Seq_Length'], pgf_data['Peak_Memory_MB'], 
                    label=f'PGF (L={layers})', marker='s', linestyle='-')
        
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log', base=10)
        ax1.set_xlabel('Sequence Length ($L$)')
        ax1.set_ylabel('Peak GPU Memory (MB)')
        ax1.set_title('Memory Scaling: $O(1)$ vs $O(L)$')
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, "Memory data not found", ha='center')

    # ---------------------------------------------------------
    # Panel 2: WikiText-2 Convergence (Loss vs Step)
    # ---------------------------------------------------------
    ax2 = axes[0, 1]
    conv_file = os.path.join(results_dir, "wikitext_convergence.csv")
    if os.path.exists(conv_file):
        df_conv = pd.read_csv(conv_file)
        for layers in df_conv['Layers'].unique():
            df_l = df_conv[df_conv['Layers'] == layers]
            # Smooth curves slightly for better visualization if needed
            ax2.plot(df_l['Step'], df_l['Autograd_Loss'], label=f'Autograd (L={layers})', alpha=0.5, linestyle='--')
            ax2.plot(df_l['Step'], df_l['PGF_Loss'], label=f'PGF (L={layers})', linestyle='-')
            
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Cross Entropy Loss')
        ax2.set_title('Convergence on WikiText-2')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "Convergence data not found", ha='center')

    # ---------------------------------------------------------
    # Panel 3: Gradient Exactness (Relative Error)
    # ---------------------------------------------------------
    ax3 = axes[1, 0]
    grad_file = os.path.join(results_dir, "rebuttal_grads_multilayer_exactness.csv")
    if os.path.exists(grad_file):
        df_grad = pd.read_csv(grad_file)
        # Group by layer count and show max difference
        # Filter only Mamba A_log and FFN weights for clarity if too many parameters
        # For simplicity, we'll plot the mean/max difference per layer configuration
        summary = df_grad.groupby('Layers')['Max_Diff'].max().reset_index()
        
        bars = ax3.bar(summary['Layers'].astype(str), summary['Max_Diff'], color='teal', alpha=0.7)
        ax3.set_yscale('log')
        ax3.set_xlabel('Number of Layers')
        ax3.set_ylabel('Max Absolute Gradient Difference')
        ax3.set_title('Numerical Stability (PGF vs Autograd)')
        
        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1e}', ha='center', va='bottom', fontsize=9)
    else:
        ax3.text(0.5, 0.5, "Gradient data not found", ha='center')

    # ---------------------------------------------------------
    # Panel 4: Throughput / Time Efficiency
    # ---------------------------------------------------------
    ax4 = axes[1, 1]
    if os.path.exists(mem_file):
        df_time = pd.read_csv(mem_file)
        # Ensure we handle OOM cases if they are strings
        df_time['Time_Seconds'] = pd.to_numeric(df_time['Time_Seconds'], errors='coerce')
        
        for layers in df_time['Layers'].unique():
            df_l = df_time[df_time['Layers'] == layers]
            # Plot only for one layer config to avoid clutter, or mean across layers
            if layers == 4: # Typical depth
                auto_data = df_l[df_l['Mode'] == 'Autograd'].dropna(subset=['Time_Seconds'])
                ax4.plot(auto_data['Seq_Length'], auto_data['Time_Seconds'], 
                        label='Autograd', marker='o', linestyle='--')
                
                pgf_data = df_l[df_l['Mode'] == 'PGF'].dropna(subset=['Time_Seconds'])
                ax4.plot(pgf_data['Seq_Length'], pgf_data['Time_Seconds'], 
                        label='PGF', marker='s', linestyle='-')
        
        ax4.set_xlabel('Sequence Length ($L$)')
        ax4.set_ylabel('Step Time (Seconds)')
        ax4.set_title('Time Efficiency (L=4)')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, "Time data not found", ha='center')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "Rebuttal_Consolidated_Analysis.png")
    plt.savefig(output_path)
    print(f"Success! Consolidated rebuttal figure saved to {output_path}")
    
    # Save a separate high-res PDF version
    plt.savefig(output_path.replace(".png", ".pdf"))
    plt.show()

if __name__ == "__main__":
    try:
        plot_rebuttal_figures()
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()
