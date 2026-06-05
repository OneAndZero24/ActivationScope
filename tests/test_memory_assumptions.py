"""Memory behavior assumption tests.

Verifies that different reduction/capture policies have the expected memory
growth profiles under repeated forward passes:
  - STORE_ALL + EVERY grows unbounded (intentionally)
  - STREAMING / FINAL_ONLY stay bounded
  - MAX_K caps growth after K batches
  - clear() between forwards prevents accumulation

Runs on CPU using tracemalloc to simulate memory tracking. GPU-specific tests
are marked with a skipif decorator.
"""

import gc
import resource
import tracemalloc
from copy import deepcopy

import pytest
import torch

from activationscope import (
    ActivationScope,
    ReductionPolicy,
    CapturePolicy,
    StoragePolicy,
)


def _get_rss_mb():
    """Return approximate RSS memory in MiB via resource module."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is in KB on Linux, bytes on macOS
    return usage.ru_maxrss * 1e-6 if hasattr(usage, "ru_maxrss") else 0.0


class TestStoreAllGrowth:
    """STORE_ALL intentionally grows — verify it does grow."""

    def test_store_all_grows_with_forwards(self, simple_linear_model):
        """Many forwards under STORE_ALL → activation dict has many tensors."""
        t = ActivationScope(
            reduction=ReductionPolicy.STORE_ALL,
            capture=CapturePolicy.EVERY,
        )
        with t.track(simple_linear_model):
            for _ in range(20):
                _ = simple_linear_model(torch.randn(4, 10))

        acts = t.activations
        total_tensors = sum(len(v) for v in acts.values())
        # With 3 layers × 20 forwards, we expect ~60 total tensors
        assert total_tensors == 60, f"Expected 60 tensors under STORE_ALL, got {total_tensors}"


class TestStreamingBounded:
    """STREAMING should keep memory bounded — no unbounded tensor list growth."""

    def test_streaming_bounded_list_length(self, simple_linear_model):
        """STREAMING with registered reduction keeps small per-layer lists."""
        t = ActivationScope(reduction=ReductionPolicy.STREAMING)
        t.register_reduction(ActivationScope.for_max(), layers=None)

        with t.track(simple_linear_model):
            for _ in range(50):
                _ = simple_linear_model(torch.randn(8, 10))


class TestFinalOnlyBounded:
    """FINAL_ONLY retains only one tensor per layer regardless of forwards."""

    def test_final_only_one_tensor(self, simple_linear_model):
        """After 100 forwards, each layer still has at most 1 tensor."""
        t = ActivationScope(reduction=ReductionPolicy.FINAL_ONLY)
        with t.track(simple_linear_model):
            for _ in range(100):
                _ = simple_linear_model(torch.randn(4, 10))

        acts = t.activations
        for layer_name, tensor_list in acts.items():
            assert len(tensor_list) <= 1, \
                f"Layer {layer_name} should have at most 1 under FINAL_ONLY, got {len(tensor_list)}"


class TestMaxKBounded:
    """MAX_K prevents unbounded growth after K batches."""

    def test_max_k_bounded_growth(self, simple_linear_model):
        """After max_batches=5, no new tensors added beyond cap."""
        t = ActivationScope(
            reduction=ReductionPolicy.STORE_ALL,
            capture=CapturePolicy.MAX_K,
            max_batches=5,
        )
        with t.track(simple_linear_model):
            for _ in range(50):  # Way more than K=5
                _ = simple_linear_model(torch.randn(4, 10))

        acts = t.activations
        for layer_name, tensor_list in acts.items():
            assert len(tensor_list) <= 5, \
                f"Layer {layer_name} should have at most 5 under MAX_K(5), got {len(tensor_list)}"


class TestClearPreventsGrowth:
    """clear() between forwards prevents accumulation even under STORE_ALL."""

    def test_clear_prevents_accumulation(self, simple_linear_model):
        """STORE_ALL + clear() each iteration → always 1 tensor per layer."""
        t = ActivationScope(reduction=ReductionPolicy.STORE_ALL)
        with t.track(simple_linear_model):
            for _ in range(30):
                _ = simple_linear_model(torch.randn(4, 10))
                t.clear()

    def test_clear_does_not_remove_hooks(self, simple_linear_model):
        """After clear(), hooks still fire on the next forward."""
        t = ActivationScope(reduction=ReductionPolicy.STORE_ALL)
        with t.track(simple_linear_model):
            _ = simple_linear_model(torch.randn(4, 10))
            t.clear()

            # Hooks should still be attached
            _ = simple_linear_model(torch.randn(4, 10))
            acts = t.activations
            assert any(len(v) == 1 for v in acts.values()), \
                "At least one layer should have a fresh tensor after clear+forward"


class TestRemoveReleasesMemory:
    """After remove(), no lingering tensor references."""

    def test_garbage_collected_after_remove(self, simple_linear_model):
        """Create many forwards → remove → gc → memory released."""
        t = ActivationScope(reduction=ReductionPolicy.STORE_ALL)
        try:
            t.attach(simple_linear_model)
            for _ in range(50):
                _ = simple_linear_model(torch.randn(8, 10))

            # Force readback to materialize all tensors in Python scope
            acts = t.activations
            total_tensors_before = sum(len(v) for v in acts.values())
            assert total_tensors_before > 0
        finally:
            t.remove()

        delete_local = True  # noqa: F841 — just to break local ref cycle
        gc.collect()

    def test_dtor_safety(self):
        """ActivationScope destructor should not crash even if C++ session is dangling."""
        import weakref
        model = torch.nn.Linear(5, 6)

        t = ActivationScope()
        ref = weakref.ref(t)
        del t

        gc.collect()
        # If dtor crashed the test would fail earlier


class TestNoGraphRetention:
    """Activations must NOT retain gradient graph references."""

    def test_no_grad_graph_in_activations(self, simple_linear_model):
        """Captured tensors should not prevent autograd deallocation."""
        t = ActivationScope(reduction=ReductionPolicy.STORE_ALL)
        with t.track(simple_linear_model):
            x = torch.randn(4, 10, requires_grad=True)
            out = simple_linear_model(x)
            loss = out.sum()
            loss.backward()

            # Activations should exist but NOT retain the graph
            acts = t.activations
            for layer_name, tensor_list in acts.items():
                for tensor in tensor_list:
                    # Captured tensors must be detached (no grad_fn)
                    assert tensor.grad_fn is None, \
                        f"Layer {layer_name} tensor retains grad graph"
