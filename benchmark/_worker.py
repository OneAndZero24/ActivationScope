"""Subprocess worker for the ActivationScope memory benchmark.

Each invocation measures *one* approach (none / naive / scope) in a clean
process, printing ``KEY=VALUE`` lines to stdout.  The runner parses these.

Memory is measured via ``psutil.Process().memory_info().vms`` (virtual
memory size) which accurately tracks all mmap'd tensor allocations.

Usage:
  PYTHONPATH=. python -m benchmark._worker --mode naive --n 200 --batch 32 --dim 256
  PYTHONPATH=. python -m benchmark._worker --mode scope --model resnet18 --n 20 --batch 8
"""

import argparse
import gc
import os
import platform
import resource
import time

import psutil
import torch


# ── peak memory measurement ────────────────────────────────────────────


def _vms_mib() -> float:
    return psutil.Process(os.getpid()).memory_info().vms / (1024 * 1024)


def _peak_rss_mib() -> float:
    """Peak resident set size (physical memory high-water mark).

    On macOS ``ru_maxrss`` is in bytes; on Linux it is in KiB.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if platform.system() == "Darwin":
        return usage.ru_maxrss / (1024 * 1024)
    else:
        return usage.ru_maxrss / 1024


# ── helpers ────────────────────────────────────────────────────────────


def _data_mib(activations: dict) -> float:
    """Exact activation tensor storage size in MiB (element_size * numel)."""
    total = 0
    for tensors in activations.values():
        for tensor in tensors:
            total += tensor.element_size() * tensor.numel()
    return total / (1024 * 1024)


def _make_toy_model(n_layers: int = 12, hidden: int = 256):
    layers = []
    for i in range(n_layers):
        layers.append(torch.nn.Linear(hidden, hidden))
        if i < n_layers - 1:
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


def _make_resnet_model():
    from torchvision.models import resnet18, ResNet18_Weights
    return resnet18(weights=ResNet18_Weights.DEFAULT)


def _build_model_and_input(model_type: str, batch: int,
                           n_layers: int = 12, dim: int = 256):
    if model_type == "toy":
        return _make_toy_model(n_layers, dim), torch.randn(batch, dim)
    elif model_type == "resnet18":
        return _make_resnet_model(), torch.randn(batch, 3, 224, 224)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _count_layers(model, n_layers_fallback: int) -> int:
    containers = (torch.nn.ModuleList, torch.nn.ModuleDict, torch.nn.Sequential)
    return sum(
        1 for n, m in model.named_modules()
        if n != "" and not isinstance(m, containers)
    )


# ── shared measurement protocol ───────────────────────────────────────

WARMUP_ITERS = 10
GC_PASSES = 2


def _warmup_and_measure(
    model: torch.nn.Module,
    x: torch.Tensor,
    n_forwards: int,
    mode: str,
    tracker_factory=None,
):
    device = x.device
    model = model.to(device).eval()

    # Pre-construct trackers / import extensions so their memory cost
    # lands in the baseline, not the delta.  Only hook registration
    # (tracker.track) and activation accumulation are measured.
    pre_tracker = None
    if mode == "scope":
        from activationscope import ActivationScope, StoragePolicy, ReductionPolicy  # noqa: E402
        pre_tracker = ActivationScope(
            storage=StoragePolicy.CPU, reduction=ReductionPolicy.STORE_ALL
        )
    elif mode == "naive" and tracker_factory is not None:
        pre_tracker = tracker_factory()

    # Phase 1: warmup (NOT measured)
    for _ in range(WARMUP_ITERS):
        _ = model(x)

    # Phase 2: GC + baseline  (now includes .so + session overhead)
    for _ in range(GC_PASSES):
        gc.collect()
    mem_baseline = _vms_mib()

    # Phase 3: measured forwards
    peak_mem = mem_baseline
    samples = []

    if mode == "none":
        t0 = time.perf_counter()
        for i in range(n_forwards):
            _ = model(x)
            if i % max(1, n_forwards // 10) == 0 or i == n_forwards - 1:
                m = _vms_mib()
                samples.append(m)
                if m > peak_mem:
                    peak_mem = m
        t1 = time.perf_counter()
        acts = {}
        read_ms = 0.0

    elif mode == "naive":
        tracker = pre_tracker
        t0 = time.perf_counter()
        with tracker.track(model):
            for i in range(n_forwards):
                _ = model(x)
                if i % max(1, n_forwards // 10) == 0 or i == n_forwards - 1:
                    m = _vms_mib()
                    samples.append(m)
                    if m > peak_mem:
                        peak_mem = m
        t1 = time.perf_counter()
        tr = time.perf_counter()
        acts = tracker.activations
        read_ms = (time.perf_counter() - tr) * 1000.0

    elif mode == "scope":
        tracker = pre_tracker
        t0 = time.perf_counter()
        with tracker.track(model):
            for i in range(n_forwards):
                _ = model(x)
                if i % max(1, n_forwards // 10) == 0 or i == n_forwards - 1:
                    m = _vms_mib()
                    samples.append(m)
                    if m > peak_mem:
                        peak_mem = m
        t1 = time.perf_counter()
        tr = time.perf_counter()
        acts = tracker.activations
        read_ms = (time.perf_counter() - tr) * 1000.0
    else:
        raise ValueError(f"Unknown mode: {mode}")

    total_ms = (t1 - t0) * 1000.0
    mem_after = _vms_mib()

    return dict(
        mem_baseline=mem_baseline,
        mem_peak=peak_mem,
        mem_after=mem_after,
        mem_delta=mem_after - mem_baseline,
        samples=samples,
        total_ms=total_ms,
        ms_per_fwd=total_ms / n_forwards,
        read_ms=read_ms,
        data_mib=_data_mib(acts),
        peak_rss_mib=_peak_rss_mib(),
    )


# ── naive tracker ─────────────────────────────────────────────────────

from activationscope._naive import NaiveHookTracker  # noqa: E402 (after activationscope import)


# ── entry ─────────────────────────────────────────────────────────────


def _run(mode: str, n_forwards: int, batch: int, dim: int,
         n_layers: int, model_type: str):
    model, sample_x = _build_model_and_input(model_type, batch, n_layers, dim)
    n_tracked = _count_layers(model, n_layers)
    tracker_factory = NaiveHookTracker if mode == "naive" else None

    r = _warmup_and_measure(model, sample_x, n_forwards, mode,
                            tracker_factory=tracker_factory)

    print(f"MODEL={model_type}")
    print(f"TRACKED_LAYERS={n_tracked}")
    print(f"MEM_BASELINE={r['mem_baseline']:.1f}")
    print(f"MEM_PEAK={r['mem_peak']:.1f}")
    print(f"MEM_AFTER={r['mem_after']:.1f}")
    print(f"MEM_DELTA={r['mem_delta']:.1f}")
    print(f"MEM_ALLOC={r['mem_peak'] - r['mem_baseline']:.1f}")
    print(f"TOTAL_MS={r['total_ms']:.1f}")
    print(f"MS_PER_FWD={r['ms_per_fwd']:.3f}")
    print(f"READBACK_MS={r['read_ms']:.2f}")
    print(f"DATA_MIB={r['data_mib']:.1f}")
    print(f"PEAK_RSS_MIB={r['peak_rss_mib']:.0f}")


# ── cli ────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("none", "naive", "scope"), required=True)
    ap.add_argument("--model", choices=("toy", "resnet18"), default="toy")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=12)
    args = ap.parse_args()

    _run(args.mode, args.n, args.batch, args.dim, args.layers, args.model)


if __name__ == "__main__":
    main()
