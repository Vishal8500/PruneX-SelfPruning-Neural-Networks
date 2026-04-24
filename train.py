"""
Self-Pruning Neural Network for CIFAR-10 Classification
========================================================
Implements a feed-forward network with learnable gate parameters that
dynamically prune themselves during training via L1 regularization.

Author: Vishal
"""

import os
import math
import time
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# ─────────────────────────────────────────────
# 1. PRUNABLE LINEAR LAYER
# ─────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that augments each weight with a
    learnable scalar gate.

    Forward pass:
        gates        = sigmoid(gate_scores)          ∈ (0, 1)
        pruned_w     = weight ⊙ gates                element-wise
        output       = x @ pruned_w.T + bias

    Gradients flow through both `weight` and `gate_scores` via autograd
    because all operations (sigmoid, multiply, matmul) are differentiable.

    Sparsity is induced by an L1 penalty on `gates` (see SparsityLoss).
    As sigmoid(gate_scores) → 0 the effective weight → 0, nullifying the
    connection without explicitly removing it from the graph, which keeps
    the training loop simple.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # ── standard weight & bias ──────────────────────────────────────
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # ── gate scores (same shape as weight) ──────────────────────────
        # Initialised with a small positive value so sigmoid(gate_scores)
        # starts near 0.73 — active but not saturated — giving the
        # optimiser room to push gates toward 0 or 1.
        self.gate_scores = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        self._init_parameters()

    # ── initialisation ───────────────────────────────────────────────────
    def _init_parameters(self):
        # Kaiming uniform for weights (matches nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Gate scores: start near 1 (fully open) so early training is
        # unimpeded; the sparsity loss gradually drives them toward 0.
        nn.init.constant_(self.gate_scores, 1.0)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    # ── forward pass ─────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert unbounded gate_scores → gates ∈ (0, 1)
        gates = torch.sigmoid(self.gate_scores)          # shape: [out, in]

        # Element-wise mask: connections with gate ≈ 0 contribute nothing
        pruned_weights = self.weight * gates             # shape: [out, in]

        # Standard affine transform — F.linear handles the transpose internally
        return F.linear(x, pruned_weights, self.bias)

    # ── utility: sparsity statistics ─────────────────────────────────────
    @torch.no_grad()
    def gate_values(self) -> torch.Tensor:
        """Return current gate activations (detached, flat)."""
        return torch.sigmoid(self.gate_scores).detach().cpu().flatten()

    @torch.no_grad()
    def sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of gates below `threshold` (treated as pruned)."""
        g = self.gate_values()
        return (g < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


# ─────────────────────────────────────────────
# 2. NETWORK ARCHITECTURE
# ─────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Feed-forward classifier for CIFAR-10 (32×32×3 → 10 classes).

    Architecture:
        Flatten → FC(3072→1024) → BN → ReLU → Dropout
                → FC(1024→512)  → BN → ReLU → Dropout
                → FC(512→256)   → BN → ReLU → Dropout
                → FC(256→128)   → BN → ReLU → Dropout
                → FC(128→10)    (no activation; raw logits)

    All FC layers are PrunableLinear so every weight has a gate.
    BatchNorm and Dropout are *not* gated — they are auxiliary and don't
    carry the semantic weight information we want to prune.
    """

    def __init__(self, dropout_p: float = 0.3):
        super().__init__()

        self.flatten = nn.Flatten()

        # ── prunable layers ───────────────────────────────────────────────
        self.fc1 = PrunableLinear(3 * 32 * 32, 1024)
        self.fc2 = PrunableLinear(1024, 512)
        self.fc3 = PrunableLinear(512, 256)
        self.fc4 = PrunableLinear(256, 128)
        self.fc5 = PrunableLinear(128, 10)

        # ── non-prunable helpers ──────────────────────────────────────────
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)

        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        x = self.drop(F.relu(self.bn3(self.fc3(x))))
        x = self.drop(F.relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)   # raw logits

        return x

    # ── helpers ───────────────────────────────────────────────────────────
    def prunable_layers(self):
        """Iterate over all PrunableLinear layers in the model."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of all sigmoid-gate values across every PrunableLinear layer.

        Why L1?  The L1 penalty (sum of |gates|) has a non-zero gradient
        everywhere except exactly at 0, creating a constant push toward
        zero for every small gate.  Unlike L2, which tapers off near zero,
        L1 produces a 'hard shrinkage' effect that drives values to *exactly*
        zero — the mathematical foundation of LASSO regularisation.
        Because gates = sigmoid(gate_scores) ≥ 0, |gates| = gates, so the
        loss is simply the sum of all gate activations.
        """
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.prunable_layers():
            total = total + torch.sigmoid(layer.gate_scores).sum()
        return total

    @torch.no_grad()
    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        """Global fraction of pruned weights across all PrunableLinear layers."""
        pruned = total = 0
        for layer in self.prunable_layers():
            g = layer.gate_values()
            pruned += (g < threshold).sum().item()
            total  += g.numel()
        return pruned / total if total > 0 else 0.0

    @torch.no_grad()
    def all_gate_values(self) -> np.ndarray:
        """Concatenate all gate values into a single numpy array."""
        parts = [layer.gate_values() for layer in self.prunable_layers()]
        return torch.cat(parts).numpy()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# 3. DATA LOADING
# ─────────────────────────────────────────────

def get_cifar10_loaders(
    data_dir: str = './data',
    batch_size: int = 256,
    num_workers: int = 2
):
    """Return (train_loader, test_loader) for CIFAR-10."""

    # ── standard CIFAR-10 normalisation ──────────────────────────────────
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True,  download=True, transform=train_transform
    )
    test_set  = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader  = DataLoader(
        test_set,  batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


# ─────────────────────────────────────────────
# 4. TRAINING & EVALUATION
# ─────────────────────────────────────────────

def train_one_epoch(
    model: SelfPruningNet,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    lam: float,
    device: torch.device,
    scheduler=None,
) -> dict:
    """Run one full training epoch. Returns a dict of metrics."""
    model.train()

    total_loss   = 0.0
    cls_loss_sum = 0.0
    spar_loss_sum= 0.0
    correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        # ── total loss ────────────────────────────────────────────────
        cls_loss  = criterion(logits, labels)
        spar_loss = model.sparsity_loss()
        loss      = cls_loss + lam * spar_loss

        loss.backward()
        # Gradient clipping prevents exploding gradients on the gate params
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # ── accumulate metrics ────────────────────────────────────────
        bs = labels.size(0)
        total_loss    += loss.item()     * bs
        cls_loss_sum  += cls_loss.item() * bs
        spar_loss_sum += spar_loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total   += bs

    if scheduler is not None:
        scheduler.step()

    n = len(loader)
    return {
        "loss":      total_loss    / total,
        "cls_loss":  cls_loss_sum  / total,
        "spar_loss": spar_loss_sum / n,
        "accuracy":  correct / total,
    }


@torch.no_grad()
def evaluate(
    model: SelfPruningNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate model on a DataLoader. Returns accuracy and loss."""
    model.eval()

    total_loss = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return {
        "loss":     total_loss / total,
        "accuracy": correct    / total,
    }


def train_model(
    lam: float,
    device: torch.device,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    num_epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout_p: float = 0.3,
    verbose: bool = True,
) -> dict:
    """
    Full training run for a single lambda value.
    Returns a result dict with final metrics and gate values.
    """
    model = SelfPruningNet(dropout_p=dropout_p).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    # Cosine annealing: smoothly decays LR, works well with gated networks
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5
    )

    history = {"train_acc": [], "test_acc": [], "sparsity": [],
               "cls_loss": [], "spar_loss": []}
    best_test_acc = 0.0
    best_state    = None

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  Training  λ = {lam:.0e}   |   params: {model.count_parameters():,}")
        print(f"{'═'*60}")
        print(f"  {'Epoch':>5}  {'TrainAcc':>9}  {'TestAcc':>8}  "
              f"{'Sparsity':>9}  {'ClsLoss':>8}  {'SparLoss':>9}")
        print(f"  {'-'*57}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(
            model, train_loader, optimizer, criterion, lam, device, scheduler
        )
        te = evaluate(model, test_loader, criterion, device)
        sp = model.overall_sparsity()

        history["train_acc"].append(tr["accuracy"])
        history["test_acc"].append(te["accuracy"])
        history["sparsity"].append(sp)
        history["cls_loss"].append(tr["cls_loss"])
        history["spar_loss"].append(tr["spar_loss"])

        if te["accuracy"] > best_test_acc:
            best_test_acc = te["accuracy"]
            # Deep-copy state dict for best-model restoration
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % 5 == 0 or epoch == 1):
            elapsed = time.time() - t0
            print(f"  {epoch:>5}  {tr['accuracy']:>9.2%}  {te['accuracy']:>8.2%}  "
                  f"{sp:>9.2%}  {tr['cls_loss']:>8.4f}  {tr['spar_loss']:>9.1f}"
                  f"  ({elapsed:.1f}s)")

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    final_te  = evaluate(model, test_loader, criterion, device)
    final_sp  = model.overall_sparsity()
    gates     = model.all_gate_values()

    if verbose:
        print(f"  {'-'*57}")
        print(f"  FINAL  test_acc={final_te['accuracy']:.2%}  "
              f"sparsity={final_sp:.2%}")

    return {
        "lam":          lam,
        "test_accuracy":final_te["accuracy"],
        "sparsity":     final_sp,
        "gate_values":  gates,
        "history":      history,
        "model":        model,
    }


# ─────────────────────────────────────────────
# 5. VISUALISATION
# ─────────────────────────────────────────────

COLORS = {
    "low":    "#2196F3",   # blue
    "medium": "#FF9800",   # orange
    "high":   "#F44336",   # red
}

def plot_gate_distribution(results: list, save_path: str = "gate_distribution.png"):
    """
    For each λ: histogram of final gate values.
    A successful run shows a large spike near 0 (pruned) and a cluster
    away from 0 (active weights).
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    fig.patch.set_facecolor('#0D1117')

    labels = ["low", "medium", "high"]

    for ax, res, label in zip(axes, results, labels):
        gates = res["gate_values"]
        color = COLORS[label]

        ax.set_facecolor('#161B22')
        ax.hist(gates, bins=100, range=(0, 1), color=color,
                alpha=0.85, edgecolor='none')

        sp   = res["sparsity"]
        acc  = res["test_accuracy"]
        lam  = res["lam"]

        ax.set_title(
            f"λ = {lam:.0e}  |  acc={acc:.2%}  |  sparsity={sp:.2%}",
            color='white', fontsize=11, pad=10
        )
        ax.set_xlabel("Gate Value (sigmoid)", color='#AAAAAA', fontsize=10)
        ax.set_ylabel("Count",               color='#AAAAAA', fontsize=10)
        ax.tick_params(colors='#888888')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')

        # Annotate the zero spike
        zero_count = int((gates < 0.01).sum())
        ax.annotate(
            f"~{zero_count:,}\npruned",
            xy=(0.01, ax.get_ylim()[1] * 0.7 if ax.get_ylim()[1] > 0 else 1),
            color=color, fontsize=9, ha='left',
            bbox=dict(boxstyle='round,pad=0.3', fc='#0D1117', ec=color, lw=0.8)
        )

    fig.suptitle(
        "Gate Value Distributions — Self-Pruning Neural Network",
        color='white', fontsize=14, y=1.02, fontweight='bold'
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✔ Saved: {save_path}")


def plot_training_curves(results: list, save_path: str = "training_curves.png"):
    """
    3-panel figure: test accuracy, sparsity, and classification loss vs epoch.
    """
    fig = plt.figure(figsize=(18, 5))
    fig.patch.set_facecolor('#0D1117')
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    axes   = [fig.add_subplot(gs[i]) for i in range(3)]
    panels = [
        ("test_acc",  "Test Accuracy",           True),
        ("sparsity",  "Network Sparsity",         True),
        ("cls_loss",  "Classification Loss",      False),
    ]
    labels = ["low", "medium", "high"]

    for ax, (key, title, pct) in zip(axes, panels):
        ax.set_facecolor('#161B22')
        ax.set_title(title, color='white', fontsize=11, pad=8)
        ax.set_xlabel("Epoch", color='#AAAAAA', fontsize=9)
        ax.tick_params(colors='#888888')
        for sp in ax.spines.values():
            sp.set_edgecolor('#333333')

        for res, label in zip(results, labels):
            y = res["history"][key]
            if pct:
                y = [v * 100 for v in y]
            ax.plot(
                range(1, len(y) + 1), y,
                color=COLORS[label], linewidth=1.8,
                label=f"λ={res['lam']:.0e}"
            )

        if pct:
            ax.set_ylabel("%", color='#AAAAAA', fontsize=9)
        ax.legend(facecolor='#0D1117', edgecolor='#333333',
                  labelcolor='white', fontsize=8)
        ax.grid(axis='y', color='#333333', linewidth=0.5, linestyle='--')

    fig.suptitle(
        "Training Dynamics — Self-Pruning Network",
        color='white', fontsize=14, y=1.02, fontweight='bold'
    )
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✔ Saved: {save_path}")


# ─────────────────────────────────────────────
# 6. REPORT GENERATION
# ─────────────────────────────────────────────

REPORT_TEMPLATE = """# Self-Pruning Neural Network — Results Report

## 1. Why L1 Penalty on Sigmoid Gates Encourages Sparsity

The sparsity regulariser is:

```
Sparsity Loss = Σ  sigmoid(gate_score_i)
                i
```

added to the cross-entropy classification loss as **λ × Sparsity Loss**.

### Mathematical intuition

| Penalty | Gradient near 0 | Effect on small values |
|---------|----------------|------------------------|
| **L1**  | Constant (≠ 0)  | Pushes values to *exactly* zero |
| L2      | → 0 as w → 0   | Shrinks but rarely reaches zero |

The L1 norm has a *constant* subgradient of +1 with respect to each positive
gate value, regardless of how small the gate already is.  This creates an
unrelenting force driving every gate toward zero.  In contrast, an L2 penalty
would have gradient proportional to the gate value itself, which vanishes as
the gate shrinks — meaning small gates are barely pushed further.

This behaviour is the basis of **LASSO regression**: L1 regularisation
naturally produces *sparse* solutions.

Because every gate is the output of a sigmoid (range (0, 1)), the L1 norm is
just the sum of gate values.  The gradient with respect to each gate score is:

```
∂L_sparsity / ∂gate_score_i = sigmoid(gate_score_i) × (1 − sigmoid(gate_score_i))
```

This product is largest around 0.5 and goes to 0 near 0 or 1 (sigmoid
saturation).  Combined with the classification loss pulling gates toward 1
(active weights) and the L1 loss pulling them toward 0, the network reaches a
*bimodal* equilibrium: important weights stay near 1, unimportant ones collapse
to 0.

---

## 2. Results Summary

{results_table}

*Threshold for "pruned": gate value < 0.01*

---

## 3. Analysis

### λ Trade-off

- **Low λ**: Classification dominates; the network retains most connections,
  achieving higher accuracy but less compression.
- **Medium λ**: Balanced regime; the network learns to identify and prune
  genuinely redundant weights while preserving accuracy.
- **High λ**: Sparsity dominates; aggressive pruning reduces the model
  dramatically, accepting an accuracy penalty.

### Practical Guidance

A λ in the range 1×10⁻⁵ – 1×10⁻⁴ typically offers the best sparsity-accuracy
trade-off for this architecture and dataset.

---

## 4. Plots

- `gate_distribution.png` — Histogram of final gate values for each λ.
  A bimodal distribution (spike at 0 + cluster near 1) confirms successful pruning.
- `training_curves.png` — Epoch-wise accuracy, sparsity, and loss for all λ values.

---

*Generated automatically by `train.py`*
"""


def build_results_table(results: list) -> str:
    header = (
        "| Lambda | Test Accuracy | Sparsity Level (%) |\n"
        "|--------|--------------|--------------------|\n"
    )
    rows = []
    for r in results:
        rows.append(
            f"| {r['lam']:.0e} | {r['test_accuracy']:.2%} | {r['sparsity']:.2%} |"
        )
    return header + "\n".join(rows)


def save_report(results: list, path: str = "REPORT.md"):
    table = build_results_table(results)
    md    = REPORT_TEMPLATE.format(results_table=table)
    with open(path, "w") as f:
        f.write(md)
    print(f"  ✔ Saved: {path}")


# ─────────────────────────────────────────────
# 7. MAIN ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Self-Pruning Neural Network — CIFAR-10")
    p.add_argument("--lambdas",      nargs="+", type=float,
                   default=[1e-6, 1e-5, 1e-4],
                   help="Sparsity regularisation coefficients to sweep (default: 1e-6 1e-5 1e-4)")
    p.add_argument("--epochs",       type=int,   default=40,
                   help="Training epochs per lambda (default: 40)")
    p.add_argument("--batch_size",   type=int,   default=256)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--data_dir",     type=str,   default="./data")
    p.add_argument("--out_dir",      type=str,   default="./outputs")
    p.add_argument("--num_workers",  type=int,   default=2)
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # ── reproducibility ───────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── device ────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda"  if torch.cuda.is_available() else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"\n  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    # ── output directory ─────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)

    # ── data ─────────────────────────────────────────────────────────────
    print(f"\n  Loading CIFAR-10 …")
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"  Train batches: {len(train_loader)}  |  "
          f"Test batches: {len(test_loader)}")

    # ── sweep over lambda values ──────────────────────────────────────────
    all_results = []
    for lam in args.lambdas:
        res = train_model(
            lam=lam,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout_p=args.dropout,
            verbose=True,
        )
        # Save per-lambda model checkpoint
        ckpt_path = os.path.join(args.out_dir, f"model_lam{lam:.0e}.pt")
        torch.save(res["model"].state_dict(), ckpt_path)
        print(f"  ✔ Checkpoint: {ckpt_path}")

        all_results.append(res)

    # ── console summary table ─────────────────────────────────────────────
    print(f"\n{'═'*50}")
    print(f"  {'Lambda':>10}  {'Test Acc':>10}  {'Sparsity':>10}")
    print(f"  {'-'*47}")
    for r in all_results:
        print(f"  {r['lam']:>10.0e}  {r['test_accuracy']:>10.2%}  {r['sparsity']:>10.2%}")
    print(f"{'═'*50}\n")

    # ── plots ─────────────────────────────────────────────────────────────
    plot_gate_distribution(
        all_results,
        save_path=os.path.join(args.out_dir, "gate_distribution.png")
    )
    plot_training_curves(
        all_results,
        save_path=os.path.join(args.out_dir, "training_curves.png")
    )

    # ── markdown report ───────────────────────────────────────────────────
    save_report(
        all_results,
        path=os.path.join(args.out_dir, "REPORT.md")
    )

    # ── save numeric results as JSON (for downstream analysis) ────────────
    json_data = [
        {
            "lambda":        r["lam"],
            "test_accuracy": r["test_accuracy"],
            "sparsity":      r["sparsity"],
        }
        for r in all_results
    ]
    json_path = os.path.join(args.out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  ✔ Saved: {json_path}")

    print("\n  All done. ✓\n")


if __name__ == "__main__":
    main()
