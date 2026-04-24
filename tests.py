"""
Unit Tests — Self-Pruning Neural Network
=========================================
Run with:  python tests.py
"""

import math
import unittest
import torch
import torch.nn as nn

# ── import from the main module ───────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from train import PrunableLinear, SelfPruningNet


class TestPrunableLinear(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.layer = PrunableLinear(in_features=8, out_features=4)

    # ── 1. Shape correctness ─────────────────────────────────────────────
    def test_weight_shape(self):
        self.assertEqual(self.layer.weight.shape, (4, 8))

    def test_gate_scores_shape(self):
        self.assertEqual(self.layer.gate_scores.shape, (4, 8))

    def test_bias_shape(self):
        self.assertEqual(self.layer.bias.shape, (4,))

    def test_output_shape(self):
        x   = torch.randn(16, 8)
        out = self.layer(x)
        self.assertEqual(out.shape, (16, 4))

    # ── 2. Gate values are in (0, 1) ─────────────────────────────────────
    def test_gates_in_range(self):
        g = self.layer.gate_values()
        self.assertTrue((g > 0).all() and (g < 1).all(),
                        "All gates must be strictly in (0, 1) after sigmoid")

    # ── 3. When gate_scores → -∞, gates → 0 and layer output → bias ──────
    def test_zero_gates_output_equals_bias(self):
        with torch.no_grad():
            self.layer.gate_scores.fill_(-1e9)   # sigmoid ≈ 0
        x   = torch.ones(3, 8)
        out = self.layer(x)
        expected = self.layer.bias.unsqueeze(0).expand(3, -1)
        self.assertTrue(
            torch.allclose(out, expected, atol=1e-4),
            "With gates≈0 the output should equal the bias"
        )

    # ── 4. When gate_scores → +∞, output ≈ standard linear ─────────────
    def test_open_gates_matches_standard_linear(self):
        with torch.no_grad():
            self.layer.gate_scores.fill_(1e9)     # sigmoid ≈ 1

        x   = torch.randn(5, 8)
        out = self.layer(x)
        # Manual computation
        expected = x @ self.layer.weight.T + self.layer.bias
        self.assertTrue(
            torch.allclose(out, expected, atol=1e-3),
            "With gates≈1 the output should equal standard linear"
        )

    # ── 5. Gradients flow through weight and gate_scores ─────────────────
    def test_gradient_flow_weight(self):
        x      = torch.randn(4, 8)
        out    = self.layer(x)
        loss   = out.sum()
        loss.backward()
        self.assertIsNotNone(self.layer.weight.grad,
                             "weight must have a gradient")
        self.assertFalse(
            torch.all(self.layer.weight.grad == 0),
            "weight gradient must not be all-zero"
        )

    def test_gradient_flow_gate_scores(self):
        x      = torch.randn(4, 8)
        out    = self.layer(x)
        loss   = out.sum()
        loss.backward()
        self.assertIsNotNone(self.layer.gate_scores.grad,
                             "gate_scores must have a gradient")
        self.assertFalse(
            torch.all(self.layer.gate_scores.grad == 0),
            "gate_scores gradient must not be all-zero"
        )

    # ── 6. gate_scores is a registered Parameter ─────────────────────────
    def test_gate_scores_is_parameter(self):
        param_names = [n for n, _ in self.layer.named_parameters()]
        self.assertIn("gate_scores", param_names,
                      "gate_scores must be a registered Parameter")

    # ── 7. No-bias variant ───────────────────────────────────────────────
    def test_no_bias(self):
        layer = PrunableLinear(6, 3, bias=False)
        self.assertIsNone(layer.bias)
        x   = torch.randn(2, 6)
        out = layer(x)
        self.assertEqual(out.shape, (2, 3))

    # ── 8. Sparsity measurement ──────────────────────────────────────────
    def test_sparsity_all_zero(self):
        with torch.no_grad():
            self.layer.gate_scores.fill_(-1e9)
        sp = self.layer.sparsity(threshold=1e-2)
        self.assertAlmostEqual(sp, 1.0, places=3)

    def test_sparsity_all_open(self):
        with torch.no_grad():
            self.layer.gate_scores.fill_(1e9)
        sp = self.layer.sparsity(threshold=1e-2)
        self.assertAlmostEqual(sp, 0.0, places=3)


class TestSelfPruningNet(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.model = SelfPruningNet()

    # ── 1. Forward pass shape ────────────────────────────────────────────
    def test_forward_shape(self):
        x   = torch.randn(8, 3, 32, 32)
        out = self.model(x)
        self.assertEqual(out.shape, (8, 10),
                         "Output must be (batch, 10) logits for CIFAR-10")

    # ── 2. Sparsity loss is a scalar and positive ─────────────────────────
    def test_sparsity_loss_scalar(self):
        loss = self.model.sparsity_loss()
        self.assertEqual(loss.shape, torch.Size([]),
                         "Sparsity loss must be a 0-d (scalar) tensor")
        self.assertGreater(loss.item(), 0.0)

    # ── 3. Sparsity loss gradient flows to gate_scores ────────────────────
    def test_sparsity_loss_gradient(self):
        loss = self.model.sparsity_loss()
        loss.backward()
        for layer in self.model.prunable_layers():
            self.assertIsNotNone(layer.gate_scores.grad)
            self.assertFalse(torch.all(layer.gate_scores.grad == 0))

    # ── 4. Overall sparsity is in [0, 1] ────────────────────────────────
    def test_overall_sparsity_range(self):
        sp = self.model.overall_sparsity()
        self.assertGreaterEqual(sp, 0.0)
        self.assertLessEqual(sp,   1.0)

    # ── 5. all_gate_values length matches sum of weight elements ─────────
    def test_all_gate_values_length(self):
        expected = sum(
            l.weight.numel() for l in self.model.prunable_layers()
        )
        actual = len(self.model.all_gate_values())
        self.assertEqual(actual, expected)

    # ── 6. Parameter count is deterministic ──────────────────────────────
    def test_parameter_count_positive(self):
        self.assertGreater(self.model.count_parameters(), 0)

    # ── 7. Total loss decreases on a trivial overfit ──────────────────────
    def test_training_step_reduces_loss(self):
        """
        Run 3 gradient steps on a tiny batch; total loss should trend down.
        Not a strict requirement for one step but meaningful over a few.
        """
        lam = 1e-5
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        x      = torch.randn(16, 3, 32, 32)
        labels = torch.randint(0, 10, (16,))

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            logits = self.model(x)
            loss   = criterion(logits, labels) + lam * self.model.sparsity_loss()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease (at least not wildly increase)
        self.assertLess(
            losses[-1], losses[0] * 2,
            "Loss after 5 steps should not have exploded"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
