# Self-Pruning Neural Network

> **Tredence AI Engineering Internship Case Study**
> A feed-forward network that learns to remove its own redundant connections
> during training via learnable gate parameters and L1 sparsity regularisation.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Setup](#setup)
4. [Running the Code](#running-the-code)
5. [Architecture](#architecture)
6. [How Pruning Works](#how-pruning-works)
7. [Results](#results)
8. [Design Decisions](#design-decisions)

---

## Overview

Standard post-training pruning requires a separate pass after the model is
trained.  This project implements **dynamic self-pruning**: every weight in the
network is multiplied by a learnable **gate** ∈ (0, 1).  An L1 penalty on all
gate values during training drives most of them to zero, effectively removing
the corresponding connections from the network.

The result is a single training run that simultaneously optimises for:
- **Accuracy** — cross-entropy loss on CIFAR-10
- **Sparsity** — L1 norm of all gate activations

---

## Repository Structure

```
self_pruning_nn/
│
├── train.py          # Complete implementation + training loop + plots
├── tests.py          # Unit tests (PrunableLinear + SelfPruningNet)
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

After running, an `outputs/` directory is created containing:
- `gate_distribution.png`  — Histogram of gate values for each λ
- `training_curves.png`    — Accuracy / sparsity / loss vs epoch
- `REPORT.md`              — Auto-generated analysis report
- `results.json`           — Machine-readable numeric results
- `model_lam*.pt`          — Best checkpoint for each λ value

---

## Setup

```bash
# 1. Clone or download this repo
git clone <your-repo-url>
cd self_pruning_nn

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

Python ≥ 3.9 and PyTorch ≥ 2.0 are required.
CUDA is auto-detected; CPU training works but is slower.

---

## Running the Code

### Full experiment (three λ values, 40 epochs each)

```bash
python train.py
```

### Custom λ sweep

```bash
python train.py --lambdas 1e-7 1e-5 1e-3 --epochs 50
```

### All CLI options

```
--lambdas       LIST   Sparsity coefficients to sweep (default: 1e-6 1e-5 1e-4)
--epochs        INT    Epochs per lambda (default: 40)
--batch_size    INT    Mini-batch size   (default: 256)
--lr            FLOAT  Initial LR (Adam) (default: 1e-3)
--weight_decay  FLOAT  Adam weight decay (default: 1e-4)
--dropout       FLOAT  Dropout probability (default: 0.3)
--data_dir      PATH   CIFAR-10 download dir (default: ./data)
--out_dir       PATH   Output directory (default: ./outputs)
--num_workers   INT    DataLoader workers (default: 2)
--seed          INT    Random seed (default: 42)
```

### Unit tests

```bash
python tests.py -v
```

---

## Architecture

```
Input (3×32×32)
       │
    Flatten  →  3072
       │
  PrunableLinear(3072 → 1024)
  BatchNorm1d → ReLU → Dropout(0.3)
       │
  PrunableLinear(1024 → 512)
  BatchNorm1d → ReLU → Dropout(0.3)
       │
  PrunableLinear(512 → 256)
  BatchNorm1d → ReLU → Dropout(0.3)
       │
  PrunableLinear(256 → 128)
  BatchNorm1d → ReLU → Dropout(0.3)
       │
  PrunableLinear(128 → 10)
       │
   Logits (10)
```

Every `PrunableLinear` layer has a `gate_scores` parameter tensor of the
same shape as `weight`.  The gate activations are `sigmoid(gate_scores)`.

---

## How Pruning Works

### PrunableLinear forward pass

```python
gates         = sigmoid(gate_scores)     # ∈ (0, 1), same shape as weight
pruned_weight = weight * gates           # element-wise mask
output        = x @ pruned_weight.T + bias
```

Gradients flow through both `weight` and `gate_scores` via standard
autograd because all operations are differentiable.

### Total loss

```
Total Loss = CrossEntropy(logits, labels)
           + λ × Σ sigmoid(gate_score_i)
                  i
```

The second term is the **L1 norm** of all gate values.

### Why L1 drives sparsity

The L1 norm has a **constant gradient** with respect to each gate value
(gradient = +1 for positive values).  This creates an unrelenting push
toward zero that doesn't weaken as gates shrink — unlike L2, whose
gradient vanishes near zero, which is why L2 does *not* produce exact zeros.

This is the same principle as **LASSO regression**: L1 regularisation
naturally selects a sparse subset of features.

The optimiser sees two competing objectives:
- Classification loss → wants gates **open** (weight large, gate ≈ 1)
- L1 sparsity penalty → wants gates **closed** (gate ≈ 0)

At convergence, only gates for weights that genuinely reduce the
classification loss remain open.  All others collapse to zero.

---

## Results

> Typical results on CIFAR-10 with 40 epochs (values will vary by hardware/seed):

| Lambda | Test Accuracy | Sparsity Level (%) |
|--------|:------------:|:-----------------:|
| 1e-6   |   ~52–55 %   |      ~10–20 %     |
| 1e-5   |   ~48–53 %   |      ~40–60 %     |
| 1e-4   |   ~35–45 %   |      ~75–90 %     |

A bimodal gate distribution (large spike at 0, secondary cluster near 1)
confirms the network successfully separates important from unimportant weights.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Sigmoid for gates | Smooth, bounded (0,1); differentiable everywhere; gradient flows cleanly |
| L1 on gate values | Constant gradient → exact zeros (unlike L2); mathematically equivalent to LASSO |
| gate_scores init = 1.0 | sigmoid(1) ≈ 0.73 — gates start open so early training is unimpeded |
| BatchNorm after prunable layer | Normalises activations after weight masking; stabilises training |
| Cosine annealing LR | Smooth decay works well with the competing loss landscape |
| Gradient clipping (max=5) | Prevents exploding gradients when λ is large and many gates are being pushed |
