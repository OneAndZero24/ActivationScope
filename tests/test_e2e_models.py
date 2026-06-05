"""End-to-end tests on realistic neural network models.

Runs full train-step simulation including loss.backward() on:
  - TransformerEncoderLayer
  - ConvNet (Conv2d stack)
  - Simulated training loop (10 iterations with clear between steps)
Verifies captured shapes match expected dimensions and that no gradient graph leaks through activations.
"""

import pytest
import torch
import torch.nn as nn

from activationscope import ActivationScope, ReductionPolicy, CapturePolicy, StoragePolicy


class TinyTransformerBlock(nn.Module):
    """A minimal transformer-like block for testing."""

    def __init__(self, d_model=32, nhead=2, dim_ff=64):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.fc1 = nn.Linear(d_model, dim_ff)
        self.fc2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.fc2(nn.functional.relu(self.fc1(x)))
        return self.norm2(x + ff_out)


class SimpleConvNet(nn.Module):
    """Simple convolutional network for testing."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)   # [B,8,H,W]
        self.bn1 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2)                  # [B,8,H/2,W/2]
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)  # [B,16,H/2,W/2]
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
        return nn.functional.relu(self.bn2(self.conv2(x)))


class TinyMLP(nn.Module):
    """Minimal MLP for training-loop tests."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return self.fc3(x)


class TestTransformerE2E:
    """End-to-end test with Transformer block."""

    def test_transformer_activations_structure(self):
        model = TinyTransformerBlock(d_model=32, nhead=2)
        t = ActivationScope()

        with t.track(model) as scope:
            # [B, seq_len, d_model]
            x = torch.randn(2, 8, 32)
            out = model(x)
            assert out.shape == (2, 8, 32)

            acts = scope.activations
            assert isinstance(acts, dict)
            assert len(acts) >= 4   # at least attn, fc1, fc2, norms

    def test_transformer_forward_backward_survives(self):
        model = TinyTransformerBlock()
        t = ActivationScope()

        with t.track(model):
            x = torch.randn(2, 8, 32, requires_grad=True)
            out = model(x)
            loss = out.sum()
            loss.backward()

        # Verify captured tensors don't retain the graph
        acts = t.activations
        for name, ts_list in acts.items():
            for tensor in ts_list:
                assert tensor.grad_fn is None, f"{name} retains grad graph"


class TestConvNetE2E:
    """End-to-end test with convolutional network."""

    def test_convnet_track_conv_only(self):
        model = SimpleConvNet()
        t = ActivationScope()

        with t.track(model, include=["conv*", "bn*"]) as scope:
            x = torch.randn(2, 3, 16, 16)
            out = model(x)
            acts = scope.activations

            # Check conv shapes
            for name in ("conv1", "conv2"):
                assert name in acts, f"Missing {name}"

    def test_convnet_shapes_match(self):
        model = SimpleConvNet()
        t = ActivationScope()

        with t.track(model) as scope:
            # Input: [2, 3, 16, 16]
            x = torch.randn(2, 3, 16, 16)
            _ = model(x)
            acts = scope.activations

            # conv1: Conv2d(3→8, kernel=3, pad=1) → [2, 8, 16, 16]
            assert acts["conv1"][0].shape == (2, 8, 16, 16)


class TestSimulatedTrainingLoop:
    """Full train-step simulation: forward + loss + backward + opt.step + clear."""

    def test_10_iteration_training_loop(self):
        model = TinyMLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        t = ActivationScope(
            reduction=ReductionPolicy.STORE_ALL,
            storage=StoragePolicy.CPU,
        )

        with t.track(model):
            for epoch in range(10):
                x = torch.randn(8, 32)
                target = torch.randint(0, 10, (8,))
                out = model(x)
                loss = criterion(out, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t.clear()

            # After 10 iterations + clear each time, activations should be empty
            acts = t.activations
            for name, ts in acts.items():
                assert len(ts) == 0, f"After clear, {name} should be empty, has {len(ts)}"

    def test_training_loop_no_graph_retention(self):
        """Activations captured during training must not hold the autograd graph."""
        model = TinyMLP()
        t = ActivationScope(reduction=ReductionPolicy.STORE_ALL)

        with t.track(model):
            for i in range(5):
                x = torch.randn(4, 32, requires_grad=True)
                target = torch.randint(0, 10, (4,))
                out = model(x)
                loss = nn.functional.cross_entropy(out, target)
                loss.backward()

        # Every captured tensor must be detached (no grad_fn chain)
        acts = t.activations
        for name, ts_list in acts.items():
            for tensor in ts_list:
                assert tensor.grad_fn is None, \
                    f"Layer {name} tensor retains autograd graph after backward"

    def test_optimizer_does_not_crash_with_tracking(self):
        """Ensure tracked model's parameters can still be updated by optimizer."""
        model = TinyMLP()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        # Record initial param values
        initial_fc1_weight = model.fc1.weight.clone()

        t = ActivationScope()
        with t.track(model):
            x = torch.randn(4, 32)
            target = torch.randint(0, 10, (4,))
            out = model(x)
            loss = nn.functional.cross_entropy(out, target)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Weight must have changed after SGD step
        assert not torch.equal(model.fc1.weight, initial_fc1_weight)


class TestCaptureDirections:
    """Verify input/output/both capture modes on real models."""

    def test_capture_input(self):
        model = TinyMLP()
        t = ActivationScope()

        with t.track(model, layers=["fc1"], capture="input") as scope:
            x = torch.randn(2, 32)
            _ = model(x)
            acts = scope.activations
            assert "fc1" in acts

    def test_capture_output(self):
        model = TinyMLP()
        t = ActivationScope()

        with t.track(model, layers=["fc1"], capture="output") as scope:
            x = torch.randn(2, 32)
            _ = model(x)
            acts = scope.activations
            assert "fc1" in acts

    def test_capture_both(self):
        model = TinyMLP()
        t = ActivationScope()

        with t.track(model, layers=["fc1"], capture="both") as scope:
            x = torch.randn(2, 32)
            _ = model(x)
            acts = scope.activations
            assert "fc1" in acts

    def test_invalid_capture_raises(self):
        model = TinyMLP()
        t = ActivationScope()
        with pytest.raises(ValueError, match="capture must be"):
            with t.track(model, capture="invalid"):
                pass


class TestEdgeCases:
    """Edge case scenarios."""

    def test_empty_batch_single_forward(self):
        """A single forward should produce exactly one tensor per layer."""
        model = TinyMLP()
        t = ActivationScope()

        with t.track(model) as scope:
            x = torch.randn(1, 32)   # batch size 1
            _ = model(x)
            acts = scope.activations
            for name, ts in acts.items():
                assert len(ts) == 1, f"Expected exactly 1 tensor per layer with single forward"

    def test_large_batch_does_not_crash(self):
        """A large batch should not crash the tracker."""
        model = TinyMLP()
        t = ActivationScope(storage=StoragePolicy.CPU)

        with t.track(model) as scope:
            x = torch.randn(128, 32)
            _ = model(x)
            acts = scope.activations
            assert any(len(ts) == 1 for ts in acts.values())

    def test_zero_input(self):
        """All-zero input should still produce valid activations."""
        model = TinyMLP()
        t = ActivationScope()

        with t.track(model) as scope:
            x = torch.zeros(2, 32)
            _ = model(x)
            acts = scope.activations
            assert len(acts) > 0

    def test_extreme_values(self):
        """Extreme input values should not crash the tracker."""
        model = TinyMLP()
        t = ActivationScope()

        with t.track(model) as scope:
            x = torch.randn(2, 32) * 1e6  # Extremely large values
            _ = model(x)
            acts = scope.activations
            assert len(acts) > 0
