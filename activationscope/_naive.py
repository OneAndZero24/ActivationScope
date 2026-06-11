"""Shared naive PyTorch forward-hook tracker for benchmarking and testing.

Provides a simple reference implementation of activation accumulation using
Python ``register_forward_hook``.  Used by the benchmark runner to compare
memory/throughput against ActivationScope's C++ hooks, and by parity tests
to verify equivalent behavior.
"""

import contextlib
from fnmatch import fnmatch

import torch


class NaiveHookTracker:
    """Accumulate per-layer activations via Python forward hooks.

    Mirrors the behavior of ``ActivationScope`` with ``STORE_ALL`` reduction
    and ``EVERY`` capture policy, but uses pure-Python hooks.  Not intended
    for production use — exists solely as a reference for benchmarking and
    correctness testing.

    Usage::

        n = NaiveHookTracker()
        with n.track(model):
            for _ in range(n_batches):
                _ = model(x)
        acts = n.activations  # dict[str, list[Tensor]]
    """

    def __init__(self):
        self._handles: list = []
        self._activations: dict[str, list[torch.Tensor]] = {}

    def track(self, model, layers=None):
        """Context manager: attach → yield self → detach."""

        @contextlib.contextmanager
        def _ctx():
            self._attach(model, layers=layers)
            yield self
            self._detach()

        return _ctx()

    def _attach(self, model, layers=None):
        containers = (torch.nn.ModuleList, torch.nn.ModuleDict, torch.nn.Sequential)
        all_modules = {
            n: m
            for n, m in model.named_modules()
            if n != "" and not isinstance(m, containers)
        }
        targets = (
            all_modules
            if layers is None
            else {
                n: m
                for n, m in all_modules.items()
                if any(fnmatch(n, p) for p in layers)
            }
        )
        self._activations = {name: [] for name in targets}
        for name in targets:
            handle = targets[name].register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def _make_hook(self, layer_name):
        def hook_fn(_module, _inp, out):
            if isinstance(out, torch.Tensor):
                self._activations[layer_name].append(out.detach().cpu())
            elif isinstance(out, (tuple, list)) and len(out) > 0:
                self._activations[layer_name].append(out[0].detach().cpu())

        return hook_fn

    def _detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    @property
    def activations(self):
        return self._activations
