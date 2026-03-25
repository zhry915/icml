# Rebuttal Summary

## Overview

In this rebuttal supplement, we provide comprehensive experimental evidence addressing concerns on **scalability, numerical precision, and convergence stability** of Phase Gradient Flow (PGF). Specifically, we include:

1. **Constant memory scaling** $O(1)$ analysis showing up to **17× memory reduction** ($L=4096$, 8 layers)
2. **Component-wise gradient exactness** with errors below **$10^{-5}$**
3. **End-to-end convergence** on WikiText-2 matching Autograd
4. **Extreme sequence length** (1M+) training with stable **1.1 GB** memory footprint

---

## 1. Efficiency & Scalability: Constant Memory $O(1)$

![Memory and Time Scaling (8 Layers)](results/Fig_Rebuttal_Multilayer_Efficiency.png)

**Figure: Memory and Time Scaling (8 Layers).** PGF maintains a near-constant memory footprint (**~285 MB**) across sequence lengths up to 8192, while Autograd memory grows linearly and triggers **OOM** at $L=8192$. At $L=4096$, PGF reduces peak memory from **4.9 GB** to **283 MB**.

---

## 2. Numerical Exactness & Convergence

We demonstrate that PGF is mathematically equivalent to standard backpropagation, with negligible numerical differences.

### 2.1 Gradient Heatmap

![Gradient Heatmap](results/Fig_Rebuttal_Gradient_Heatmap.png)

**Figure: Gradient Heatmap.** Max $L_\infty$ error per component. Numerical errors are consistently below **$10^{-5}$** across all model components (Embedding, LM Head, Mamba, FFN), ensuring long-term training stability.

### 2.2 WikiText-2 Convergence

![WikiText-2 Convergence](results/Fig_Rebuttal_WikiText_Convergence.png)

**Figure: WikiText-2 Convergence.** PGF and Autograd loss curves are indistinguishable, with both reaching a final loss of approximately **2.3** within 100 steps, confirming optimization integrity.

---

## 3. Extreme Length Capability (L=1M+)

![Extreme Sequence Length Analysis](results/Fig_Rebuttal_Extreme_Length.png)

**Figure: Extreme Sequence Length Analysis.** PGF successfully trains on ultra-long sequences (1M+ tokens) with a stable memory consumption of **~1.1 GB**. The loss rapidly decreases from **16.8** to **6.7**, demonstrating successful optimization in regimes entirely inaccessible to standard Autograd-based Mamba implementations.
