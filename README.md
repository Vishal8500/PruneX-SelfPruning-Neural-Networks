# Self-Pruning Neural Network

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-00A67E?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-20%20passing-34d399?style=flat-square)

**A feed-forward network that learns to remove its own redundant connections during training.**  
No post-hoc pruning pass. No separate compression step. Just one training run.

[Overview](#overview) · [Architecture](#architecture) · [How It Works](#how-it-works) · [Results](#results) · [Quickstart](#quickstart) · [Design](#design-decisions)

</div>

---

## Overview

Standard model compression pipelines look like this:

```
Train → Evaluate → Prune (post-hoc) → Fine-tune → Deploy
```

This project collapses that into one step:

```
Train (with self-pruning) → Deploy
```

Every weight in the network is paired with a learnable **gate parameter**. A sigmoid function maps each gate to (0, 1). An L1 penalty on all gate values during training drives the unnecessary ones to exactly zero — removing the corresponding connections without any explicit surgical step.

The result: a single Adam optimizer run that simultaneously optimises for accuracy **and** sparsity, with the balance controlled by a single hyperparameter λ.

---

## Architecture

```
Input (3 × 32 × 32)
        │
    Flatten ──────────────────────── 3072
        │
  ╔═══════════════╗
  ║ PrunableLinear ║  3072 → 1024   ← weight ⊙ sigmoid(gate_scores)
  ╚═══════════════╝
    BatchNorm1d → ReLU → Dropout(0.3)
        │
  ╔═══════════════╗
  ║ PrunableLinear ║  1024 → 512
  ╚═══════════════╝
    BatchNorm1d → ReLU → Dropout(0.3)
        │
  ╔═══════════════╗
  ║ PrunableLinear ║  512 → 256
  ╚═══════════════╝
    BatchNorm1d → ReLU → Dropout(0.3)
        │
  ╔═══════════════╗
  ║ PrunableLinear ║  256 → 128
  ╚═══════════════╝
    BatchNorm1d → ReLU → Dropout(0.3)
        │
  ╔═══════════════╗
  ║ PrunableLinear ║  128 → 10      ← logits (no activation)
  ╚═══════════════╝
```

All five FC layers are `PrunableLinear`. BatchNorm and Dropout are standard — they don't carry the semantic weight information we want to prune.

---

## How It Works

### The `PrunableLinear` layer

```python
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))
        # gate_scores is a registered parameter → updated by the optimizer
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.gate_scores, 1.0)   # sigmoid(1) ≈ 0.73 — starts open

    def forward(self, x):
        gates         = torch.sigmoid(self.gate_scores)   # ∈ (0, 1)
        pruned_weight = self.weight * gates               # element-wise mask
        return F.linear(x, pruned_weight, self.bias)      # standard affine
```

Gradients flow through both `weight` and `gate_scores` via autograd — `sigmoid` and element-wise multiply are both differentiable.

### The sparsity loss

```python
def sparsity_loss(self) -> torch.Tensor:
    total = torch.tensor(0.0, device=...)
    for layer in self.prunable_layers():
        total = total + torch.sigmoid(layer.gate_scores).sum()
    return total
```

```
Total Loss  =  CrossEntropy(logits, labels)  +  λ × Σ sigmoid(gate_score_i)
                                                        i
```

### Why L1 produces exact zeros (and L2 does not)

| Penalty | Gradient near 0 | Effect |
|---------|----------------|--------|
| **L1** | Constant `+1` | Pushes values to **exactly** zero — even the smallest gate keeps getting pushed |
| L2 | Proportional to value → `0` | Shrinks values but rarely reaches zero — gradient vanishes before the job is done |

The L1 norm has a constant subgradient of +1 for every positive gate value. This creates an unrelenting force independent of the gate's current magnitude — the mathematical mechanism behind **LASSO regression**. Because gates = sigmoid(gate_scores) > 0, the L1 norm is simply their sum.

The optimizer sees two competing objectives:
- **Classification loss** → rewards open gates (weight preserved → better predictions)
- **L1 sparsity penalty** → rewards closed gates (sum of gates reduced → lower penalty)

At convergence, only gates whose weights genuinely help the classification loss remain open. All others collapse to zero.

---

## Results

> Trained on CIFAR-10, 40 epochs, Adam (lr=1e-3), cosine-annealing schedule, batch size 256, dropout 0.3.

| Lambda (λ) | Regime | Test Accuracy | Sparsity Level |
|:----------:|:------:|:-------------:|:--------------:|
| `1 × 10⁻⁶` | Low    | **54.3%**     | 18.7%          |
| `1 × 10⁻⁵` | Medium | 51.8%         | 56.4%          |
| `1 × 10⁻⁴` | High   | 40.2%         | **83.1%**      |

*Threshold for "pruned": gate value < 0.01*

### Gate value distributions

A successful run shows a **bimodal histogram**: a large spike near 0 (pruned weights) and a secondary cluster near 1 (active weights). Higher λ pushes more mass toward zero.

```
λ = 1e-6 (low)         λ = 1e-5 (medium)      λ = 1e-4 (high)

Count                  Count                  Count
  │▐                     │▐▐                    │▐▐▐▐
  │▐                     │▐▐                    │▐▐▐▐▐
  │▐         ▌▐          │▐▐     ▐▌             │▐▐▐▐▐
  │▐         ▐▐▐         │▐▐   ▌▐▐▐             │▐▐▐▐▐     ▌
  └───────────────       └───────────────       └──────────────
  0          1           0          1           0          1
    Gate Value
```

The zero-spike grows dramatically with λ, confirming the L1 mechanism is driving sparsity.

### λ trade-off analysis

- **λ = 1e-6 (low)**: Classification dominates. Most gates remain open. Highest accuracy (54.3%), minimal compression (18.7% pruned).
- **λ = 1e-5 (medium)**: Balanced regime. 56.4% of weights pruned with only ~2.5% accuracy drop. Best accuracy-per-parameter trade-off.
- **λ = 1e-4 (high)**: Sparsity dominates. Aggressive compression (83.1% pruned) at the cost of ~14 percentage points of accuracy. Useful when memory budget is the hard constraint.

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/<your-username>/self-pruning-nn
cd self-pruning-nn

# 2. Install
pip install -r requirements.txt

# 3. Train with default λ sweep {1e-6, 1e-5, 1e-4}
python train.py

# 4. Custom sweep
python train.py --lambdas 1e-7 1e-5 1e-3 --epochs 60

# 5. Unit tests (20 tests, all green)
python tests.py -v
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--lambdas` | `1e-6 1e-5 1e-4` | Sparsity coefficients to sweep |
| `--epochs` | `40` | Training epochs per λ |
| `--batch_size` | `256` | Mini-batch size |
| `--lr` | `1e-3` | Initial Adam learning rate |
| `--weight_decay` | `1e-4` | Adam weight decay |
| `--dropout` | `0.3` | Dropout probability |
| `--data_dir` | `./data` | CIFAR-10 download path |
| `--out_dir` | `./outputs` | Results, plots, checkpoints |
| `--seed` | `42` | Random seed |

### Outputs (auto-generated in `./outputs/`)

```
outputs/
├── gate_distribution.png    ← histogram of gate values (3 panels, one per λ)
├── training_curves.png      ← accuracy / sparsity / loss vs epoch
├── REPORT.md                ← auto-generated analysis report
├── results.json             ← machine-readable numeric results
├── model_lam1e-06.pt        ← best checkpoint, λ = 1e-6
├── model_lam1e-05.pt        ← best checkpoint, λ = 1e-5
└── model_lam1e-04.pt        ← best checkpoint, λ = 1e-4
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| `sigmoid` for gates | Smooth, bounded (0,1), differentiable everywhere — clean gradient flow through both `weight` and `gate_scores` |
| L1 on gate values | Constant subgradient → exact zeros (LASSO). L2 tapers off near zero, rarely produces hard sparsity |
| `gate_scores` init = `1.0` | `sigmoid(1) ≈ 0.73` — gates start mostly open so early training is unimpeded |
| BatchNorm after prunable layer | Normalises activations after weight masking; stabilises training when many gates are in flux |
| Cosine annealing LR | Smooth decay avoids oscillation between "prune" and "restore" phases |
| Gradient clipping (`max_norm=5`) | Prevents exploding gradients on `gate_scores` when λ is large |
| Best-checkpoint restore | Reports accuracy on the best generalization checkpoint, not last epoch |

---

## File Structure

```
self_pruning_nn/
├── train.py          ← complete implementation (PrunableLinear + network + training + plots)
├── tests.py          ← 20 unit tests: shapes, gradient flow, sparsity, training steps
├── requirements.txt  ← torch, torchvision, numpy, matplotlib
└── README.md         ← this file
```

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
```

CUDA is auto-detected and used if available. CPU training works but is ~5–10× slower per epoch.

---

<div align="center">
  <sub>Tredence AI Engineering Internship · 2025 Cohort</sub>
</div>
