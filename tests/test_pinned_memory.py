"""Tests for use_pinned=True flag in ActivationScope.

Verifies that pinned memory construction works with AUTO and CPU storage,
and that tracking completes without error when pinned buffers are used.
Skips gracefully when CUDA is not available (pinned memory is most useful
with GPU targets).
"""

import pytest
import torch

from activationscope import (
    ActivationScope,
    StoragePolicy,
    ReductionPolicy,
    CapturePolicy,
)


@pytest.fixture
def cuda_available():
    """Helper to check CUDA availability for skip logic."""
    return torch.cuda.is_available()


class TestPinnedConstruction:
    """Pinned flag is accepted and tracker builds cleanly."""

    def test_pinned_with_cpu_storage(self):
        """use_pinned=True + StoragePolicy.CPU constructs without error."""
        t = ActivationScope(use_pinned=True, storage=StoragePolicy.CPU)
        assert t._session_id is not None
        t.remove()

    def test_pinned_with_auto_storage(self):
        """use_pinned=True + StoragePolicy.AUTO constructs without error."""
        t = ActivationScope(use_pinned=True, storage=StoragePolicy.AUTO)
        assert t._session_id is not None
        t.remove()

    def test_pinned_gpu_storage_constructs(self):
        """Even with GPU storage, use_pinned=True builds the tracker."""
        t = ActivationScope(use_pinned=True, storage=StoragePolicy.GPU)
        assert t._session_id is not None
        t.remove()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_pinned_with_cuda_model(self):
        """Pinned + GPU model forwards correctly (async DMA path)."""
        model = torch.nn.Linear(16, 8).cuda()
        t = ActivationScope(use_pinned=True, storage=StoragePolicy.CPU)

        with t.track(model):
            x = torch.randn(4, 16, device="cuda")
            _ = model(x)

        acts = t.activations
        # With CPU storage + pinned, tensors should still end up on CPU
        for name, ts_list in acts.items():
            for tensor in ts_list:
                assert tensor.device.type == "cpu", \
                    f"Layer {name} should be on CPU with pinned+CPU storage"

    def test_pinned_false_by_default(self):
        """Default construction has use_pinned=False."""
        t = ActivationScope()
        # We can't directly inspect the C++ flag, but we verify the tracker
        # works without pinned mode
        model = torch.nn.Linear(4, 5)
        with t.track(model):
            _ = model(torch.randn(2, 4))
        assert len(t.activations) > 0
        t.remove()


class TestPinnedTracking:
    """Full forward passes work with pinned memory buffers."""

    def test_pinned_single_forward(self, simple_linear_model):
        """A basic forward pass succeeds with pinned=True."""
        t = ActivationScope(use_pinned=True, storage=StoragePolicy.CPU)
        with t.track(simple_linear_model):
            _ = simple_linear_model(torch.randn(2, 10))

        acts = t.activations
        assert len(acts) > 0
        for name, ts_list in acts.items():
            assert len(ts_list) >= 1

    def test_pinned_multiple_forwards(self, simple_linear_model):
        """Many forwards with pinned=True + STORE_ALL."""
        t = ActivationScope(
            use_pinned=True,
            storage=StoragePolicy.CPU,
            reduction=ReductionPolicy.STORE_ALL,
        )
        with t.track(simple_linear_model):
            for _ in range(10):
                _ = simple_linear_model(torch.randn(4, 10))

        acts = t.activations
        for name, ts_list in acts.items():
            assert len(ts_list) == 10, \
                f"Layer {name} should have 10 tensors, got {len(ts_list)}"

    def test_pinned_with_auto_and_threshold(self):
        """Pinned + AUTO with custom threshold works."""
        model = torch.nn.Linear(8, 16)
        t = ActivationScope(
            use_pinned=True,
            storage=StoragePolicy.AUTO,
            auto_cpu_threshold_bytes=500,  # Very low threshold
        )
        with t.track(model):
            _ = model(torch.randn(2, 8))
        assert len(t.activations) > 0

    def test_pinned_clear_then_forward(self, simple_linear_model):
        """clear() works correctly when pinned buffers are used."""
        t = ActivationScope(use_pinned=True, storage=StoragePolicy.CPU)
        with t.track(simple_linear_model):
            _ = simple_linear_model(torch.randn(2, 10))
            t.clear()
            acts_after_clear = t.activations
            for name, ts_list in acts_after_clear.items():
                assert len(ts_list) == 0

            # Hooks still fire after clear
            _ = simple_linear_model(torch.randn(2, 10))
            acts_after_forward = t.activations
            assert any(len(v) > 0 for v in acts_after_forward.values())

    def test_pinned_final_only(self, simple_linear_model):
        """Pinned + FINAL_ONLY retains exactly one tensor per layer."""
        t = ActivationScope(
            use_pinned=True,
            storage=StoragePolicy.CPU,
            reduction=ReductionPolicy.FINAL_ONLY,
        )
        with t.track(simple_linear_model):
            for _ in range(20):
                _ = simple_linear_model(torch.randn(4, 10))

        acts = t.activations
        for name, ts_list in acts.items():
            assert len(ts_list) <= 1, \
                f"Layer {name} should have at most 1 tensor under FINAL_ONLY"


class TestPinnedForwardBackward:
    """Gradient flow is unaffected by pinned memory."""

    def test_backward_survives_with_pinned(self, simple_linear_model):
        """forward + backward works normally when pinned=True."""
        t = ActivationScope(use_pinned=True, storage=StoragePolicy.CPU)
        with t.track(simple_linear_model):
            x = torch.randn(2, 10, requires_grad=True)
            out = simple_linear_model(x)
            loss = out.sum()
            loss.backward()

        # Model params should have gradients populated
        for name, param in simple_linear_model.named_parameters():
            assert param.grad is not None, \
                f"Parameter {name} has no gradient after backward"

    def test_optimizer_step_with_pinned(self):
        """SGD step works fine with pinned tracking active."""
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 4),
        )
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        initial_w = model[0].weight.clone()

        t = ActivationScope(use_pinned=True, storage=StoragePolicy.CPU)
        with t.track(model):
            x = torch.randn(4, 8)
            out = model(x)
            loss = out.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

        assert not torch.equal(model[0].weight, initial_w), \
            "Model weights should have changed after optimizer step"


class TestPinnedEdgeCases:
    """Boundary cases with pinned memory."""

    def test_pinned_max_k(self, simple_linear_model):
        """Pinned + MAX_K capture policy caps properly."""
        t = ActivationScope(
            use_pinned=True,
            storage=StoragePolicy.CPU,
            reduction=ReductionPolicy.STORE_ALL,
            capture=CapturePolicy.MAX_K,
            max_batches=3,
        )
        with t.track(simple_linear_model):
            for _ in range(15):
                _ = simple_linear_model(torch.randn(2, 10))

        acts = t.activations
        for name, ts_list in acts.items():
            assert len(ts_list) <= 3, \
                f"Layer {name} should have at most 3 under MAX_K(3)"

    def test_pinned_sampling(self, simple_linear_model):
        """Pinned + SAMPLE_N skips correctly."""
        t = ActivationScope(
            use_pinned=True,
            storage=StoragePolicy.CPU,
            reduction=ReductionPolicy.STORE_ALL,
            capture=CapturePolicy.SAMPLE_N,
            sample_every=5,
        )
        with t.track(simple_linear_model):
            for _ in range(20):
                _ = simple_linear_model(torch.randn(2, 10))

        acts = t.activations
        for name, ts_list in acts.items():
            assert len(ts_list) <= 6, \
                f"Layer {name} should have few tensors under SAMPLE_N(5)"

    def test_pinned_remove_safe(self):
        """Removing a pinned tracker does not crash."""
        t = ActivationScope(use_pinned=True, storage=StoragePolicy.CPU)
        model = torch.nn.Linear(4, 5)
        with t.track(model):
            _ = model(torch.randn(2, 4))
        # Context manager auto-removes; no errors expected

    def test_pinned_reduction_registered(self, simple_linear_model):
        """register_reduction works when pinned=True."""
        t = ActivationScope(use_pinned=True, storage=StoragePolicy.CPU)
        t.register_reduction(ActivationScope.for_mean(), layers=["fc*"])

        with t.track(simple_linear_model):
            _ = simple_linear_model(torch.randn(2, 10))

    def test_pinned_zero_input(self):
        """All-zero input with pinned does not cause issues."""
        model = torch.nn.Linear(8, 4)
        t = ActivationScope(use_pinned=True, storage=StoragePolicy.CPU)

        with t.track(model):
            _ = model(torch.zeros(2, 8))


class TestPinnedVsUnpinnedParity:
    """Verify pinned and unpinned produce the same numerical results."""

    def test_same_activations_pinned_vs_unpinned(self, simple_linear_model):
        """Captured values with pinned=True match unpinned results."""
        torch.manual_seed(42)
        x = torch.randn(3, 10)

        # Unpinned run
        t_normal = ActivationScope(storage=StoragePolicy.CPU)
        with t_normal.track(simple_linear_model):
            _ = simple_linear_model(x)
        acts_normal = t_normal.activations

        # Pinned run (same input)
        torch.manual_seed(42)
        # Reset model weights to initial state by rebuilding
        from copy import deepcopy
        model_pinned = deepcopy(simple_linear_model)

        t_pinned = ActivationScope(use_pinned=True, storage=StoragePolicy.CPU)
        with t_pinned.track(model_pinned):
            _ = model_pinned(x)
        acts_pinned = t_pinned.activations

        # Compare keys and tensor values
        assert set(acts_normal.keys()) == set(acts_pinned.keys()), \
            "Pinned and unpinned should track the same layers"

        for name in acts_normal:
            for t_n, t_p in zip(acts_normal[name], acts_pinned[name]):
                assert torch.allclose(t_n, t_p, atol=1e-6), \
                    f"Mismatch in layer {name}: pinned vs unpinned values differ"
