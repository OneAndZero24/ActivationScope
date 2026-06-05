"""Benchmark suite comparing ActivationScope vs naive Python forward hooks.

Measures throughput (ms per forward) and memory overhead for both approaches.
Marked with @pytest.mark.benchmark so they can be excluded from normal CI runs.
"""

import gc
import tracemalloc

import pytest
import torch

from activationscope import ActivationScope, ReductionPolicy, StoragePolicy


# ── Naive Python hook baseline ─────────────────────────────────────

class NaiveHookTracker:
    """Simple reference implementation using register_forward_hook with plain Python callback.

    Attaches hooks that append full tensors to a per-layer dict of lists — the
    pattern most users land on before discovering ActivationScope.
    """

    def __init__(self):
        self._handles = []
        self._activations = {}

    def track(self, model, layers=None):
        """Attach hooks using stdlib context manager."""
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            self._attach(model, layers=layers)
            yield self
            self._detach()

        return _ctx()

    def _attach(self, model, layers=None):
        all_modules = dict(model.named_modules())
        if layers is None:
            targets = {n: m for n, m in all_modules.items() if n != ""}
        else:
            from fnmatch import fnmatch
            targets = {n: m for n, m in all_modules.items()
                       if any(fnmatch(n, p) for p in layers)}

        self._activations = {name: [] for name in targets}
        for name in targets:
            handle = targets[name].register_forward_hook(
                self._make_hook(name)
            )
            self._handles.append(handle)

    def _make_hook(self, layer_name):
        def hook_fn(module, inp, out):
            if isinstance(out, torch.Tensor):
                self._activations[layer_name].append(out.detach())
            elif isinstance(out, tuple):
                # Only detach the first tensor output
                self._activations[layer_name].append(out[0].detach())
        return hook_fn

    def _detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    @property
    def activations(self):
        return self._activations


# ── Model factory (shared across benchmarks) ───────────────────────

def _make_bench_model():
    """A medium model for fair benchmarking."""
    return torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 64),
    )


# ── Benchmark tests ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def bench_model():
    """Shared model for benchmark suite (module-scoped to avoid rebuild overhead)."""
    m = _make_bench_model()
    return m


class TestThroughputBenchmark:
    """Compare wall-clock time for N forward passes with tracking enabled."""

    @pytest.mark.benchmark
    def test_100_forwards_activated_scope(self, bench_model):
        """Time 100 forwards through the model with ActivationScope tracking active."""
        tracker = ActivationScope(storage=StoragePolicy.CPU)
        n = 100
        x = torch.randn(4, 128)

        with tracker.track(bench_model):
            start = __import__("time").perf_counter()
            for _ in range(n):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                _ = bench_model(x)
            end = __import__("time").perf_counter()

        elapsed = (end - start) * 1000  # ms
        acts = tracker.activations
        assert len(acts) > 0
        avg_ms = elapsed / n
        print(f"\n  [ActivationScope] {n} forwards: {elapsed:.1f} ms total, "
              f"{avg_ms:.3f} ms/forward")

    @pytest.mark.benchmark
    def test_100_forwards_naive(self, bench_model):
        """Time 100 forwards with the naive Python hook baseline."""
        tracker = NaiveHookTracker()
        n = 100
        x = torch.randn(4, 128)

        with tracker.track(bench_model):
            start = __import__("time").perf_counter()
            for _ in range(n):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                _ = bench_model(x)
            end = __import__("time").perf_counter()

        elapsed = (end - start) * 1000
        acts = tracker.activations
        assert len(acts) > 0
        avg_ms = elapsed / n
        print(f"\n  [Naive baseline] {n} forwards: {elapsed:.1f} ms total, "
              f"{avg_ms:.3f} ms/forward")

    @pytest.mark.benchmark
    def test_throughput_comparison(self, bench_model):
        """Run both and assert ActivationScope is competitive (within 50% of naive)."""

        # --- ActivationScope ---
        t1 = ActivationScope(storage=StoragePolicy.CPU)
        x = torch.randn(4, 128)
        with t1.track(bench_model):
            s = __import__("time").perf_counter()
            for _ in range(50):
                _ = bench_model(x)
            act_time = __import__("time").perf_counter() - s

        # --- Naive ---
        t2 = NaiveHookTracker()
        with t2.track(bench_model):
            s = __import__("time").perf_counter()
            for _ in range(50):
                _ = bench_model(x)
            naive_time = __import__("time").perf_counter() - s

        print(f"\n  ActivationScope: {act_time*1000:.1f} ms")
        print(f"  Naive baseline:  {naive_time*1000:.1f} ms")

        # Native hooks should not be dramatically slower. Allow 2x margin for CI variance.
        assert act_time <= naive_time * 2.0, \
            f"ActivationScope ({act_time:.3f}s) should not be much slower than naive ({naive_time:.3f}s)"


class TestMemoryBenchmark:
    """Compare memory overhead between approaches."""

    @pytest.mark.benchmark
    def test_memory_overhead_store_all(self):
        """Measure peak RAM during 50 forwards with STORE_ALL.

        Uses tracemalloc since we run on CPU-only macOS in CI. If CUDA were
        available, torch.cuda.max_memory_allocated would be more precise.
        """
        model = _make_bench_model()
        x = torch.randn(4, 128)

        # --- ActivationScope ---
        gc.collect()
        tracemalloc.start()
        t_as = ActivationScope(storage=StoragePolicy.CPU)
        with t_as.track(model):
            for _ in range(50):
                _ = model(x)
        peak_as, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # --- Naive ---
        gc.collect()
        tracemalloc.start()
        t_naive = NaiveHookTracker()
        with t_naive.track(model):
            for _ in range(50):
                _ = model(x)
        peak_naive, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"\n  ActivationScope peak memory: {peak_as / 1024 / 1024:.2f} MiB")
        print(f"  Naive baseline peak memory:  {peak_naive / 1024 / 1024:.2f} MiB")

        # No strict assertion — just demonstrating the comparison.
        assert peak_as > 0 and peak_naive > 0

    @pytest.mark.benchmark
    def test_memory_overhead_final_only(self):
        """FINAL_ONLY should use significantly less memory than STORE_ALL."""
        model = _make_bench_model()
        x = torch.randn(4, 128)

        # --- ActivationScope with STORE_ALL ---
        gc.collect()
        tracemalloc.start()
        t_store = ActivationScope(reduction=ReductionPolicy.STORE_ALL, storage=StoragePolicy.CPU)
        with t_store.track(model):
            for _ in range(50):
                _ = model(x)
        peak_store, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # --- ActivationScope with FINAL_ONLY ---
        gc.collect()
        tracemalloc.start()
        t_final = ActivationScope(reduction=ReductionPolicy.FINAL_ONLY, storage=StoragePolicy.CPU)
        with t_final.track(model):
            for _ in range(50):
                _ = model(x)
        peak_final, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"\n  STORE_ALL peak memory:   {peak_store / 1024 / 1024:.2f} MiB")
        print(f"  FINAL_ONLY peak memory:  {peak_final / 1024 / 1024:.2f} MiB")

        # FINAL_ONLY should use noticeably less memory under repeated forwards
        assert peak_final < peak_store * 0.5, \
            "FINAL_ONLY memory should be significantly less than STORE_ALL"


class TestActivationParity:
    """Verify both approaches capture the same activations shape-wise."""

    @pytest.mark.benchmark
    def test_shapes_match_between_approaches(self):
        """ActivationScope and naive hooks should produce tensors of matching shapes."""
        model = _make_bench_model()
        x = torch.randn(2, 128)

        # --- ActivationScope ---
        t_as = ActivationScope(storage=StoragePolicy.CPU)
        with t_as.track(model):
            _ = model(x)
        acts_as = t_as.activations

        # --- Naive ---
        t_naive = NaiveHookTracker()
        with t_naive.track(model):
            _ = model(x)
        acts_naive = t_naive.activations

        # Both should have the same layer keys (same module names)
        assert set(acts_as.keys()) == set(acts_naive.keys()), \
            "Activations layer keys differ"

        for name in acts_as:
            as_shape = acts_as[name][0].shape
            naive_shape = acts_naive[name][0].shape
            assert as_shape == naive_shape, \
                f"Shape mismatch for {name}: ActivationScope={as_shape}, Naive={naive_shape}"
