"""ActivationScope vs naive Python hooks — correctness & zero-copy parity tests.

Throughput benchmarks live in benchmark/runner.py.
These tests verify that both approaches produce identical results
and that ActivationScope's zero-copy readback actually works.
"""

import gc
import time
import tracemalloc

import pytest
import torch

from activationscope import ActivationScope, StoragePolicy
from activationscope._naive import NaiveHookTracker


# ══════════════════════════════════════════════════════════════════════


def _make_model():
    layers = []
    for i in range(12):
        layers.append(torch.nn.Linear(256, 256))
        if i < 11:
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


@pytest.fixture(scope="module")
def model():
    m = _make_model()
    m.eval()
    return m


# ══════════════════════════════════════════════════════════════════════
#  Parity tests — both approaches must produce identical results
# ══════════════════════════════════════════════════════════════════════


class TestParity:
    """Same input → same activations regardless of hook engine."""

    @pytest.mark.benchmark
    def test_same_layer_keys(self, model):
        x = torch.randn(4, 256)
        s = ActivationScope(storage=StoragePolicy.CPU)
        with s.track(model):
            _ = model(x)
        n = NaiveHookTracker()
        with n.track(model):
            _ = model(x)
        assert set(s.activations.keys()) == set(n.activations.keys())

    @pytest.mark.benchmark
    def test_same_shapes(self, model):
        x = torch.randn(4, 256)
        s = ActivationScope(storage=StoragePolicy.CPU)
        with s.track(model):
            _ = model(x)
        n = NaiveHookTracker()
        with n.track(model):
            _ = model(x)
        for name in s.activations:
            assert s.activations[name][0].shape == n.activations[name][0].shape, (
                f"Shape mismatch: {name}"
            )

    @pytest.mark.benchmark
    def test_same_values(self, model):
        """Deterministic eval mode → identical float values."""
        x = torch.randn(4, 256)
        s = ActivationScope(storage=StoragePolicy.CPU)
        with s.track(model):
            _ = model(x)
        n = NaiveHookTracker()
        with n.track(model):
            _ = model(x)
        for name in s.activations:
            assert torch.allclose(
                s.activations[name][0], n.activations[name][0], atol=1e-6
            ), f"Values differ: {name}"

    @pytest.mark.benchmark
    def test_multiple_forward_accumulation(self, model):
        """After N forwards, each layer should have exactly N tensors."""
        x = torch.randn(2, 256)
        N = 30
        s = ActivationScope(storage=StoragePolicy.CPU)
        with s.track(model):
            for _ in range(N):
                _ = model(x)
        acts = s.activations
        for name, tensors in acts.items():
            assert len(tensors) == N, f"{name}: expected {N}, got {len(tensors)}"
        total = sum(t.element_size() * t.numel() for ts in acts.values() for t in ts)
        assert total > 0


# ══════════════════════════════════════════════════════════════════════
#  Zero-copy tests — TensorImpl sharing, not data duplication
# ══════════════════════════════════════════════════════════════════════


class TestZeroCopy:
    """Verify readback doesn't copy tensor data."""

    @pytest.mark.benchmark
    def test_readback_is_near_instant(self, model):
        """Reading 70+ MiB of activations should take <10ms (no data copy)."""
        x = torch.randn(32, 256)
        s = ActivationScope(storage=StoragePolicy.CPU)
        with s.track(model):
            for _ in range(100):
                _ = model(x)
        t0 = time.perf_counter()
        acts = s.activations
        dt_ms = (time.perf_counter() - t0) * 1000.0
        total_mib = (
            sum(t.element_size() * t.numel() for ts in acts.values() for t in ts)
            / 1024
            / 1024
        )
        gbps = (total_mib / dt_ms * 1000) / 1024 if dt_ms > 0 else float("inf")
        print(
            f"\n    Readback: {total_mib:.1f} MiB in {dt_ms:.3f} ms = {gbps:.0f} GiB/s effective"
        )
        assert dt_ms < 10, f"Readback too slow: {dt_ms:.1f}ms — zero-copy violated?"

    @pytest.mark.benchmark
    def test_no_python_heap_growth_on_readback(self, model):
        """Accessing .activations should not allocate 70 MiB on Python heap."""
        x = torch.randn(32, 256)
        s = ActivationScope(storage=StoragePolicy.CPU)
        with s.track(model):
            for _ in range(50):
                _ = model(x)
        gc.collect()
        tracemalloc.start()
        acts = s.activations
        peak, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        total_mib = (
            sum(t.element_size() * t.numel() for ts in acts.values() for t in ts)
            / 1024
            / 1024
        )
        peak_mib = peak / 1024 / 1024
        print(f"\n    Data accessed: {total_mib:.1f} MiB")
        print(f"    Python heap peak during readback: {peak_mib:.2f} MiB")
        # If tensors were copied, peak heap ≈ data size.  Zero-copy: peak ≪ data.
        assert peak_mib * 10 < total_mib, (
            f"Python heap ({peak_mib:.1f} MiB) should be ≪ data ({total_mib:.1f} MiB)"
        )
