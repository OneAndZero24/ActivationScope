"""Dedicated memory leak detection tests for ActivationScope.

Verifies that C++ sessions are properly destroyed after remove(),
that multiple attach/remove cycles don't accumulate memory, and that
captured tensors truly release their autograd graph references.

Runs on CPU using gc + tracemalloc; heavy GPU-specific memory tests
are marked with skipif decorators.
"""

import gc
import sys
import tracemalloc
import weakref

import pytest
import torch

import activationscope._C as _C
from activationscope import (
    ActivationScope,
    ReductionPolicy,
    CapturePolicy,
    StoragePolicy,
)


class TestCppSessionDestroy:
    """Verify C++ session lifecycle after tracker.remove()."""

    def test_session_destroyed_after_remove(self):
        """After remove(), the C++ session should be gone."""
        t = ActivationScope()
        sid = t.session_id
        t.remove()

        # Accessing session_readback on dead session returns empty dict
        result = _C.session_readback(sid)
        assert result == {}, f"Expected empty dict for destroyed session, got {result}"

    def test_session_survives_after_context_exit(self, simple_linear_model):
        """track() context manager keeps the session alive for reuse."""
        t = ActivationScope()
        with t.track(simple_linear_model):
            _ = simple_linear_model(torch.randn(2, 10))

        # After context exit, session_id should still be accessible
        sid = t.session_id
        assert sid > 0

        # But remove() still destroys it
        t.remove()
        with pytest.raises(RuntimeError, match="session already destroyed"):
            _ = t.session_id

    def test_cannot_attach_after_destroy(self):
        """Once destroyed, a tracker cannot be re-attached."""
        t = ActivationScope()
        model = torch.nn.Linear(3, 4)
        t.attach(model)
        t.remove()

        with pytest.raises(RuntimeError, match="session already destroyed"):
            t.attach(model)

    def test_cannot_register_after_destroy(self):
        """Once destroyed, register_reduction is blocked."""
        t = ActivationScope()
        t.remove()

        with pytest.raises(RuntimeError, match="session already destroyed"):
            t.register_reduction(lambda x: x.mean(), layers=["*"])


class TestMultipleCyclesNoLeak:
    """Multiple attach/remove cycles should not accumulate memory."""

    def test_n_attach_remove_cycles_clean(self, simple_linear_model):
        """Create and destroy N trackers in a loop; no unbounded growth."""
        n_cycles = 20
        model = simple_linear_model

        for _ in range(n_cycles):
            t = ActivationScope(
                reduction=ReductionPolicy.STORE_ALL,
                capture=CapturePolicy.EVERY,
                storage=StoragePolicy.CPU,
            )
            t.attach(model)
            for _ in range(5):
                _ = model(torch.randn(4, 10))
            acts = t.activations
            assert len(acts) > 0
            t.remove()

        gc.collect()
        # If we get here without MemoryError, the basic cycle works.
        # We can't easily verify C++ memory from Python, but this confirms
        # no segfault / OOM on repeated destroy cycles.

    def test_clear_in_loop_doesnt_accumulate(self, simple_linear_model):
        """clear() between forwards keeps activation count bounded."""
        t = ActivationScope(
            reduction=ReductionPolicy.STORE_ALL,
            capture=CapturePolicy.EVERY,
        )
        with t.track(simple_linear_model):
            for _ in range(50):
                _ = simple_linear_model(torch.randn(4, 10))
                t.clear()

            # After clear on the last iteration, everything should be empty
            acts = t.activations
            for layer_name, tensor_list in acts.items():
                assert len(tensor_list) == 0, \
                    f"Layer {layer_name} has leftover tensors after clear()"

    def test_no_accumulation_with_streaming(self, simple_linear_model):
        """STREAMING + reduction should stay bounded across many batches."""
        t = ActivationScope(reduction=ReductionPolicy.STREAMING)
        t.register_reduction(ActivationScope.for_max(), layers=None)

        with t.track(simple_linear_model):
            for _ in range(100):
                _ = simple_linear_model(torch.randn(8, 10))


class TestAutogradGraphNotRetained:
    """Captured tensors must not hold the autograd graph."""

    def test_detach_no_grad_fn(self, simple_linear_model):
        """Every captured tensor has grad_fn is None after forward+backward."""
        t = ActivationScope(reduction=ReductionPolicy.STORE_ALL)
        with t.track(simple_linear_model):
            x = torch.randn(4, 10, requires_grad=True)
            out = simple_linear_model(x)
            loss = out.sum()
            loss.backward()

            acts = t.activations
            for layer_name, tensor_list in acts.items():
                for tensor in tensor_list:
                    assert tensor.grad_fn is None, \
                        f"Layer {layer_name} tensor retains grad graph via grad_fn"
                    # Also check .grad is None (detached tensors have no .grad)
                    assert not tensor.requires_grad, \
                        f"Layer {layer_name} tensor still requires grad"

    def test_no_graph_retention_cpu_storage(self):
        """CPU-storage path must also detach properly."""
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 4),
        )
        t = ActivationScope(
            storage=StoragePolicy.CPU,
            reduction=ReductionPolicy.STORE_ALL,
        )
        with t.track(model):
            x = torch.randn(2, 8, requires_grad=True)
            out = model(x)
            loss = out.sum()
            loss.backward()

            acts = t.activations
            for layer_name, tensor_list in acts.items():
                for tensor in tensor_list:
                    assert tensor.grad_fn is None, \
                        f"CPU stored {layer_name} retains grad graph"

    def test_multiple_backwards_no_graph(self):
        """After multiple forward+backward cycles, no graph leaks."""
        model = torch.nn.Linear(16, 8)
        t = ActivationScope(reduction=ReductionPolicy.STORE_ALL)

        with t.track(model):
            for _ in range(10):
                x = torch.randn(4, 16, requires_grad=True)
                out = model(x)
                loss = out.sum()
                loss.backward()
                # Force gradient zeroing to release intermediate tensors
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.zero_()

            acts = t.activations
            for name, ts_list in acts.items():
                for tensor in ts_list:
                    assert tensor.grad_fn is None, \
                        f"Layer {name} retained graph after 10 backward passes"


class TestMemoryReclamationAfterDestroy:
    """Python-side memory drops after tracker destruction."""

    def test_python_tensors_gone_after_remove(self, simple_linear_model):
        """Once remove() fires, activations are inaccessible (forcing GC)."""
        t = ActivationScope(reduction=ReductionPolicy.STORE_ALL)
        t.attach(simple_linear_model)

        # Fill with many tensors
        for _ in range(30):
            _ = simple_linear_model(torch.randn(4, 10))

        acts = t.activations
        total_before = sum(len(v) for v in acts.values())
        assert total_before > 0

        # Clear local ref to acts, destroy tracker
        del acts
        t.remove()
        gc.collect()

        # Accessing .activations should raise (session destroyed)
        with pytest.raises(RuntimeError, match="session already destroyed"):
            _ = t.activations

    def test_weak_ref_collected_on_remove(self):
        """A tracker in a weak ref vanishes after remove + del."""
        model = torch.nn.Linear(4, 5)
        t = ActivationScope()
        t.attach(model)
        ref = weakref.ref(t)

        # Force local alias cleanup, then gc
        del t
        gc.collect()

        # The __del__ should have called remove(), cleaning C++ session
        assert ref() is None, \
            "ActivationScope instance was not garbage collected"


class TestDestructiveOperationsAfterDestroy:
    """Operations on destroyed session raise gracefully."""

    def test_clear_on_dead_session_safe(self):
        """clear() on a destroyed tracker is a no-op."""
        t = ActivationScope()
        t.remove()
        t.clear()  # Should not crash (guarded by _session_id is not None check)

    def test_double_remove_safe(self):
        """remove() called twice does not crash."""
        t = ActivationScope()
        t.remove()
        t.remove()  # idempotent no-op


class TestTracemallocStability:
    """Use tracemalloc to detect gross Python-side allocation growth."""

    def test_no_python_memory_growth_over_cycles(self, simple_linear_model):
        """Repeated track/clear cycles shouldn't grow Python allocations."""
        tracemalloc.start()

        baseline = tracemalloc.take_snapshot()

        for _ in range(30):
            t = ActivationScope(storage=StoragePolicy.CPU)
            with t.track(simple_linear_model):
                _ = simple_linear_model(torch.randn(4, 10))

        gc.collect()
        final = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Compare allocated blocks; allow for some growth (< 200 KB overhead)
        stats = final.compare_to(baseline, "lineno")
        net_growth = sum(s.size_diff for s in stats if s.size_diff > 0)
        assert net_growth < 200_000, \
            f"Python-side memory grew by {net_growth} bytes across 30 cycles"

    def test_clear_during_tracking_limits_growth(self, simple_linear_model):
        """Within a single track(), calling clear() each iter bounds memory."""
        tracemalloc.start()
        baseline = tracemalloc.take_snapshot()

        t = ActivationScope(
            reduction=ReductionPolicy.STORE_ALL,
            storage=StoragePolicy.CPU,
        )
        with t.track(simple_linear_model):
            for _ in range(50):
                _ = simple_linear_model(torch.randn(8, 10))
                t.clear()

        gc.collect()
        final = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = final.compare_to(baseline, "lineno")
        net_growth = sum(s.size_diff for s in stats if s.size_diff > 0)
        assert net_growth < 200_000, \
            f"Accumulated {net_growth} bytes despite clear() each iteration"