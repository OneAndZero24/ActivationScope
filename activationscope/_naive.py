"""Shared naive PyTorch forward-hook tracker for benchmarking and testing.

Provides a simple reference implementation of activation accumulation using
Python ``register_forward_hook``.  Used by the benchmark runner to compare
memory/throughput against ActivationScope's C++ hooks, and by parity tests
to verify equivalent behavior.
"""

import contextlib
from fnmatch import fnmatch

import torch

from activationscope.policies import CaptureMode


class NaiveHookTracker:
    """Accumulate per-layer activations via Python forward hooks.

    Mirrors the behavior of ``ActivationScope`` with ``STORE_ALL`` reduction
    and ``EVERY`` capture policy, but uses pure-Python hooks.  Not intended
    for production use — exists solely as a reference for benchmarking and
    correctness testing.

    Parameters
    ----------
    capture_mode : CaptureMode
        Controls copy behaviour of captured tensors:
        * ``CaptureMode.REFERENCE`` (default) — ``.detach().cpu()``.
          Tensors share storage with the autograd graph and may be
          invalidated on subsequent forward passes.
        * ``CaptureMode.SNAPSHOT`` — ``.detach().cpu().clone()``.
          Completely independent copies, safe to keep across forwards.

    Usage::

        n = NaiveHookTracker(capture_mode=CaptureMode.SNAPSHOT)
        with n.track(model):
            for _ in range(n_batches):
                _ = model(x)
        acts = n.activations  # dict[str, list[Tensor]]
    """

    def __init__(self, capture_mode: CaptureMode = CaptureMode.REFERENCE):
        self._handles: list = []
        self._activations: dict[str, list[torch.Tensor]] = {}
        self._capture_mode = capture_mode

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
        capture_mode = self._capture_mode

        def hook_fn(_module, _inp, out):
            if isinstance(out, torch.Tensor):
                t = out.detach().cpu()
                self._activations[layer_name].append(
                    t.clone() if capture_mode == CaptureMode.SNAPSHOT else t
                )
            elif isinstance(out, (tuple, list)) and len(out) > 0:
                t = out[0].detach().cpu()
                self._activations[layer_name].append(
                    t.clone() if capture_mode == CaptureMode.SNAPSHOT else t
                )

        return hook_fn

    def _detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    @property
    def activations(self):
        return self._activations
