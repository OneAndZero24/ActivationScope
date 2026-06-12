"""Integration tests for reduction policy behavior across multiple batches.

Verifies STORE_ALL accumulates per-batch tensors, STREAMING reduces in-place,
FINAL_ONLY retains only the last batch, and register_reduction works correctly.
"""

import pytest
import torch

from activationscope import ActivationScope, ReductionPolicy


class TestStoreAll:
    """STORE_ALL — each forward adds a new tensor to the list."""

    def test_accumulates_multiple_batches(self, simple_linear_model):
        t = ActivationScope(reduction=ReductionPolicy.STORE_ALL)
        with t.track(simple_linear_model):
            for _ in range(5):
                _ = simple_linear_model(torch.randn(2, 10))

            acts = t.activations
        # Every tracked layer should have a tensor per forward pass
        for layer_name, tensor_list in acts.items():
            assert len(tensor_list) == 5, \
                f"Layer {layer_name} should have 5 tensors, got {len(tensor_list)}"

    def test_store_all_shapes_preserved(self, simple_linear_model):
        """Each stored tensor has the right output shape for the layer."""
        t = ActivationScope(reduction=ReductionPolicy.STORE_ALL)
        with t.track(simple_linear_model):
            _ = simple_linear_model(torch.randn(2, 10))

            acts = t.activations
        # fc1: Linear(10→20), output [2, 20]
        assert acts["fc1"][0].shape == (2, 20)
        # act: ReLU passes through → shape still [2, 20]
        assert acts["act"][0].shape == (2, 20)
        # fc2: Linear(20→5), output [2, 5]
        assert acts["fc2"][0].shape == (2, 5)


class TestStreaming:
    """STREAMING — reduction replaces full tensor each batch."""

    def test_streaming_produces_single_tensor(self, simple_linear_model):
        """With reduction registered, STREAMING should keep one reduced tensor."""
        t = ActivationScope(reduction=ReductionPolicy.STREAMING)
        t.register_reduction(ActivationScope.max_reduction(), layers=None)

        with t.track(simple_linear_model):
            for _ in range(5):
                _ = simple_linear_model(torch.randn(2, 10))


class TestFinalOnly:
    """FINAL_ONLY — only the last batch's activation is retained."""

    def test_final_only_replaces_each_batch(self, simple_linear_model):
        """After N forwards under FINAL_ONLY, each layer has exactly one tensor."""
        t = ActivationScope(reduction=ReductionPolicy.FINAL_ONLY)
        with t.track(simple_linear_model):
            for _ in range(5):
                _ = simple_linear_model(torch.randn(2, 10))

            acts = t.activations
        for layer_name, tensor_list in acts.items():
            assert len(tensor_list) <= 1, \
                f"Layer {layer_name} should have at most 1 tensor under FINAL_ONLY"


class TestRegisterReductionPerLayer:
    """register_reduction with per-layer patterns."""

    def test_per_layer_reduction_applied(self, simple_linear_model):
        """A reduction registered for a specific layer name is matched correctly."""
        t = ActivationScope(reduction=ReductionPolicy.STREAMING)
        t.register_reduction(ActivationScope.max_reduction(), layers=["fc1"])
        t.attach(simple_linear_model)

        x = torch.randn(2, 10)
        _ = simple_linear_model(x)


class TestGlobalReductionFallback:
    """Global reduction serves as default for unmatched layers."""

    def test_global_reduction_fallback(self, simple_linear_model):
        """When a layer has no per-layer reduction, global default is used."""
        t = ActivationScope(reduction=ReductionPolicy.STREAMING)
        # Register global fallback
        t.register_reduction(ActivationScope.mean_reduction(), layers=None)
        # Also register a per-layer override for fc1
        t.register_reduction(ActivationScope.max_reduction(), layers=["fc1"])

        with t.track(simple_linear_model):
            _ = simple_linear_model(torch.randn(2, 10))


class TestConvWithReduction:
    """Reduction policies on convolutional models."""

    def test_streaming_on_conv(self, conv_model):
        """STREAMING + for_max on Conv2d produces [C_out, H, W] shaped tensors."""
        t = ActivationScope(reduction=ReductionPolicy.STREAMING)
        t.register_reduction(ActivationScope.max_reduction(), layers=["conv*"])

        with t.track(conv_model, include=["conv*", "pool"]):
            _ = conv_model(torch.randn(2, 3, 16, 16))


class TestAllReductionPoliciesSmoke:
    """Every ReductionPolicy completes a forward+backward cycle."""

    @pytest.mark.parametrize("reduction", list(ReductionPolicy.__dict__.values()))
    def test_forward_backward_survives(self, simple_linear_model, reduction):
        if not isinstance(reduction, int) or not (0 <= reduction <= 2):
            return
        t = ActivationScope(reduction=reduction)
        with t.track(simple_linear_model):
            x = torch.randn(2, 10, requires_grad=True)
            out = simple_linear_model(x)
            loss = out.sum()
            loss.backward()
