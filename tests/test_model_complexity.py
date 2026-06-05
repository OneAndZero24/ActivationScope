"""Tests against complex real-world model architectures.

Covers:
  - Models with residual/skip connections
  - Models with tuple outputs (LSTM-style)
  - Nested ModuleDict + ModuleList combinations
  - Very deep models (20+ layers)
  - Mixed Sequential-in-ModuleList nesting
"""

import pytest
import torch
from copy import deepcopy

from activationscope import (
    ActivationScope,
    ReductionPolicy,
    CapturePolicy,
    StoragePolicy,
)


class TestTupleOutputModel:
    """Models that return tuples like LSTM."""

    def test_track_layers_in_tuple_model(self, lstm_tuple_model):
        """Track individual layers inside a wrapper with tuple output."""
        t = ActivationScope(reduction=ReductionPolicy.STORE_ALL)

        # Track internal linear layers only (embedding + proj layers)
        with t.track(lstm_tuple_model, include=["lstm_hidden_proj", "output_proj"]):
            x_indices = torch.randint(0, 50, (3, 5))  # [batch=3, seq=5]
            out, (h, c) = lstm_tuple_model(x_indices)

        acts = t.activations
        assert "lstm_hidden_proj" in acts or "output_proj" in acts, \
            f"Should have captured at least one layer, got keys: {acts.keys()}"

    def test_track_all_layers_tuple_model(self, lstm_tuple_model):
        """Track all layers including Embedding layer."""
        t = ActivationScope(reduction=ReductionPolicy.STORE_ALL)

        with t.track(lstm_tuple_model):
            x_indices = torch.randint(0, 50, (2, 4))
            _ = lstm_tuple_model(x_indices)

        acts = t.activations
        assert len(acts) >= 3, \
            f"Should capture embedding + proj layers, got only {len(acts)}: {acts.keys()}"

    def test_tuple_model_forward_backward(self, lstm_tuple_model):
        """forward+backward works through a tuple-output model."""
        t = ActivationScope()

        with t.track(lstm_tuple_model):
            x_indices = torch.randint(0, 50, (2, 4))
            out, _ = lstm_tuple_model(x_indices)
            loss = out.sum()
            loss.backward()

    def test_tuple_model_with_final_only(self, lstm_tuple_model):
        """FINAL_ONLY works on tuple-output models."""
        t = ActivationScope(reduction=ReductionPolicy.FINAL_ONLY)

        with t.track(lstm_tuple_model, include=["*proj"]):
            for _ in range(10):
                x_indices = torch.randint(0, 50, (2, 4))
                _ = lstm_tuple_model(x_indices)

        acts = t.activations
        for name, ts_list in acts.items():
            assert len(ts_list) <= 1


class TestResidualModel:
    """Models with skip connections (ResNet-style)."""

    def test_residual_forward_tracking(self, residual_model):
        """All layers in a residual network get tracked."""
        t = ActivationScope(reduction=ReductionPolicy.STORE_ALL)

        with t.track(residual_model):
            x = torch.randn(4, 16)
            _ = residual_model(x)

        acts = t.activations
        # Should capture fc1, act1, fc2, fc3, act2, fc_out
        assert len(acts) >= 5, \
            f"Residual model should track all non-container layers, got {acts.keys()}"

    def test_residual_backward_survives(self, residual_model):
        """Gradients flow properly through residual connections with tracking."""
        t = ActivationScope()

        with t.track(residual_model):
            x = torch.randn(2, 16, requires_grad=True)
            out = residual_model(x)
            loss = out.sum()
            loss.backward()

        # Check that model params have gradients
        for name, param in residual_model.named_parameters():
            assert param.grad is not None, \
                f"Parameter {name} missing gradient after backward"

    def test_residual_multi_batch(self, residual_model):
        """Multiple forward passes through a residual network."""
        t = ActivationScope(
            reduction=ReductionPolicy.STORE_ALL,
            capture=CapturePolicy.EVERY,
        )

        with t.track(residual_model, include=["fc*"]):
            for _ in range(8):
                _ = residual_model(torch.randn(4, 16))

        acts = t.activations
        for name, ts_list in acts.items():
            assert len(ts_list) == 8, \
                f"Residual layer {name} should have 8 tensors"

    def test_residual_with_streaming(self, residual_model):
        """STREAMING reduction on residual blocks."""
        t = ActivationScope(reduction=ReductionPolicy.STREAMING)
        t.register_reduction(ActivationScope.for_max(), layers=["fc*"])

        with t.track(residual_model):
            for _ in range(50):
                _ = residual_model(torch.randn(8, 16))


class TestDeepModel:
    """Very deep models (24+ layers) test stack/binding limits."""

    def test_deep_model_attaches_all_layers(self, deep_model):
        """All 24 linear + activation layers get hooks attached."""
        t = ActivationScope(reduction=ReductionPolicy.STORE_ALL)

        with t.track(deep_model):
            x = torch.randn(2, 32)
            _ = deep_model(x)

        acts = t.activations
        # Should have many layers (ModuleList children of layers + acts lists)
        assert len(acts) >= 10, \
            f"Deep model should track many layers, got {len(acts)}: {acts.keys()}"

    def test_deep_model_forward_backward(self, deep_model):
        """A very deep model survives forward+backward under tracking."""
        t = ActivationScope(storage=StoragePolicy.CPU)

        with t.track(deep_model):
            x = torch.randn(1, 32, requires_grad=True)
            out = deep_model(x)
            loss = out.sum()
            loss.backward()

    def test_deep_model_max_k(self, deep_model):
        """MAX_K on a very deep model caps correctly."""
        t = ActivationScope(
            reduction=ReductionPolicy.STORE_ALL,
            capture=CapturePolicy.MAX_K,
            max_batches=3,
        )

        with t.track(deep_model):
            for _ in range(20):
                _ = deep_model(torch.randn(2, 32))

        acts = t.activations
        for name, ts_list in acts.items():
            assert len(ts_list) <= 3, \
                f"Deep layer {name} should be capped at 3, got {len(ts_list)}"

    def test_deep_model_include_pattern(self, deep_model):
        """Glob pattern matching works across many layers."""
        t = ActivationScope()

        # Track only layers from one ModuleList
        with t.track(deep_model, include=["layers.*"]):
            _ = deep_model(torch.randn(2, 32))

        acts = t.activations
        for name in acts:
            assert "layers." in name, \
                f"Should only have layers. prefix, got {name}"

    def test_deep_model_exclude_pattern(self, deep_model):
        """Exclude pattern filters deep models correctly."""
        t = ActivationScope()

        with t.track(deep_model, exclude=["acts.*"]):
            _ = deep_model(torch.randn(2, 32))

        acts = t.activations
        for name in acts:
            assert "acts." not in name


class TestMixedContainerModel:
    """Nested ModuleDict containing ModuleLists."""

    def test_mixed_container_tracks_deep_children(self, mixed_container_model):
        """Layers inside ModuleDict → ModuleList get tracked."""
        t = ActivationScope(reduction=ReductionPolicy.STORE_ALL)

        with t.track(mixed_container_model):
            x = torch.randn(2, 10)
            _ = mixed_container_model(x)

        acts = t.activations
        # Should capture encoder and decoder layer children (not the containers)
        has_encoder = any("encoder" in k for k in acts)
        has_decoder = any("decoder" in k for k in acts)
        assert has_encoder and has_decoder, \
            f"Should have both encoder and decoder layers: {acts.keys()}"

    def test_mixed_container_include_encoder(self, mixed_container_model):
        """Selectively track only encoder block."""
        t = ActivationScope()

        with t.track(mixed_container_model, include=["blocks.encoder.*"]):
            _ = mixed_container_model(torch.randn(2, 10))

        acts = t.activations
        for name in acts:
            assert "encoder" in name, \
                f"Should only have encoder layers, got {name}"

    def test_mixed_container_forward_backward(self, mixed_container_model):
        """forward+backward survives nested container model."""
        t = ActivationScope()

        with t.track(mixed_container_model):
            x = torch.randn(2, 10, requires_grad=True)
            out = mixed_container_model(x)
            loss = out.sum()
            loss.backward()


class TestNestedSequentialModel:
    """ModuleList containing Sequential modules."""

    def test_nested_sequential_tracks_all(self, nested_sequential_model):
        """All leaf layers in nested structures get tracked."""
        t = ActivationScope(reduction=ReductionPolicy.STORE_ALL)

        with t.track(nested_sequential_model):
            x = torch.randn(2, 10)
            _ = nested_sequential_model(x)

        acts = t.activations
        # Should have stack.0.0 (Linear), stack.0.1 (ReLU), stack.1.0, stack.1.1, head
        assert len(acts) >= 5, \
            f"Should track nested layers, got only {len(acts)}: {acts.keys()}"

    def test_nested_sequential_include_specific(self, nested_sequential_model):
        """Pattern matching in deeply nested Sequential works."""
        t = ActivationScope()

        with t.track(nested_sequential_model, include=["stack.*.0"]):
            _ = nested_sequential_model(torch.randn(2, 10))

        acts = t.activations
        for name in acts:
            assert name.endswith(".0"), \
                f"Should only have .0 suffix (first Linear in each seq), got {name}"

    def test_nested_sequential_forward_backward(self, nested_sequential_model):
        """Gradients flow through deeply nested Sequential."""
        t = ActivationScope()

        with t.track(nested_sequential_model):
            x = torch.randn(2, 10, requires_grad=True)
            out = nested_sequential_model(x)
            loss = out.sum()
            loss.backward()


class TestComplexModelStress:
    """Stress tests: large batch + many layers + multiple forwards."""

    def test_deep_model_many_forwards_cpu(self, deep_model):
        """Many forwards through a deep model with CPU storage won't OOM."""
        t = ActivationScope(
            reduction=ReductionPolicy.FINAL_ONLY,  # Keep memory bounded
            storage=StoragePolicy.CPU,
        )

        with t.track(deep_model):
            for _ in range(30):
                x = torch.randn(8, 32)
                _ = deep_model(x)

    def test_resident_memory_stable_residual(self, residual_model):
        """Residual model with STORE_ALL + clear stays stable."""
        t = ActivationScope(reduction=ReductionPolicy.STORE_ALL)

        with t.track(residual_model):
            for _ in range(100):
                _ = residual_model(torch.randn(4, 16))
                t.clear()

    def test_deep_with_reduction(self, deep_model):
        """register_reduction works on many layers via glob."""
        t = ActivationScope(reduction=ReductionPolicy.STREAMING)
        t.register_reduction(ActivationScope.for_max(), layers=["layers.*"])
        t.register_reduction(ActivationScope.for_mean(), layers=["acts.*"])

        with t.track(deep_model):
            for _ in range(10):
                _ = deep_model(torch.randn(2, 32))


class TestActualLSTM:
    """Test with PyTorch's built-in nn.LSTM which returns tuple output."""

    def test_lstm_tracking(self):
        """Track layers inside nn.LSTM via parent module."""
        class LSTMWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(100, 32)
                self.lstm = torch.nn.LSTM(input_size=32, hidden_size=64, num_layers=2)
                self.fc = torch.nn.Linear(64, 10)

            def forward(self, x):
                emb = self.embedding(x)
                lstm_out, _ = self.lstm(emb)
                return self.fc(lstm_out[:, -1, :])   # last timestep

        model = LSTMWrapper()
        t = ActivationScope(reduction=ReductionPolicy.STORE_ALL)

        with t.track(model, include=["embedding", "fc"]):
            x = torch.randint(0, 100, (3, 5))
            out = model(x)
            assert out.shape == (3, 10)

        acts = t.activations
        assert "embedding" in acts or "fc" in acts

    def test_lstm_backward(self):
        """forward+backward through LSTM wrapper with tracking."""
        class LSTMWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(input_size=16, hidden_size=32)
                self.fc = torch.nn.Linear(32, 5)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        model = LSTMWrapper()
        t = ActivationScope()

        with t.track(model):
            x = torch.randn(2, 4, 16, requires_grad=True)
            out = model(x)
            loss = out.sum()
            loss.backward()


class TestAttentionModel:
    """Test against attention-heavy models."""

    def test_multihead_attention_tracking(self, transformer_block):
        """Track self-attention layers in a transformer block."""
        t = ActivationScope(reduction=ReductionPolicy.STORE_ALL)

        with t.track(transformer_block, include=["self_attn*", "linear*"]):
            x = torch.randn(2, 8, 64)
            _ = transformer_block(x)

        acts = t.activations
        assert len(acts) > 0

    def test_transformer_with_reductions(self, transformer_block):
        """Register different reductions for attention vs FF layers."""
        t = ActivationScope(reduction=ReductionPolicy.STREAMING)
        t.register_reduction(ActivationScope.for_max(), layers=["self_attn*"])
        t.register_reduction(ActivationScope.for_mean(), layers=["linear*"])

        with t.track(transformer_block):
            for _ in range(10):
                x = torch.randn(2, 8, 64)
                _ = transformer_block(x)
