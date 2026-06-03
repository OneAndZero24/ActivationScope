"""ActivationScope core tracker implementation."""

from contextlib import contextmanager
import torch
from typing import Dict, Optional, List, Generator, Any

import activationscope._C as _C


class ActivationScope:
    """High-performance activation tracker for PyTorch models.

    Mode A (`store`): Captures independent copies of activations via
        ``out.detach().clone()`` so the tracker never holds live references
        into PyTorch's autograd graph. Memory is released when the user
        calls ``.clear()`` or exits the ``track()`` context.

    Mode B (online stats): Computes running min / max / mean in C++ via ATen,
        reducing only over the batch dimension (dim 0) and per-element across
        forward passes. Introduces no Python dispatch overhead and never retains
        activation copies.
    """

    def __init__(self, mode: str = "store"):
        if mode not in ("store", "online_max", "online_min", "online_mean"):
            raise ValueError(f"Invalid mode '{mode}'. Choose from available modes.")

        self.mode = mode
        self._activations: Dict[str, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    @property
    def activations(self) -> Dict[str, torch.Tensor]:
        """Return captured activations (store mode)."""
        return self._activations

    def _make_store_hook(self, layer_name: str):
        """Create a forward hook that captures an independent copy of the output tensor."""

        def hook(_module, _inp, out: torch.Tensor):
            # Detach-on-Store: capture an independent copy so the tracker never holds
            # live references into PyTorch's autograd graph. Memory safe after backward().
            self._activations[layer_name] = out.detach().clone()

        return hook

    def _make_online_hook(self, layer_name: str):
        """Create a forward hook that delegates to C++ for online statistics."""
        cpp_func_map = {
            "online_max": _C.register_max_hook,
            "online_min": _C.register_min_hook,
            "online_mean": _C.register_mean_hook,
        }
        cpp_fn = cpp_func_map[self.mode]

        def hook(_module, _inp, out: torch.Tensor):
            cpp_fn(layer_name, out)

        return hook

    def attach(
        self,
        model: torch.nn.Module,
        layers: Optional[Dict[str, torch.nn.Module]] = None,
    ):
        """Attach hooks to specified layers or all named modules.

        Args:
            model: Target PyTorch model.
            layers: Optional mapping of {name -> module}. If None, attaches to
                    all non-container submodules of `model`.
        """
        if layers is None:
            # Auto-detect all trackable submodules (exclude container types)
            containers = (torch.nn.ModuleList, torch.nn.ModuleDict, torch.nn.Sequential)
            layers = {
                name: mod
                for name, mod in model.named_modules()
                if not isinstance(mod, containers) and name != ""
            }

        hook_fn = (
            self._make_store_hook if self.mode == "store" else self._make_online_hook
        )

        for layer_name, module in layers.items():
            handle = module.register_forward_hook(hook_fn(layer_name))
            self._handles.append(handle)

    @contextmanager
    def track(
        self,
        model: torch.nn.Module,
        layers: Optional[Dict[str, torch.nn.Module]] = None,
    ) -> Generator["ActivationScope", Any, None]:
        """Context manager that attaches hooks on enter and fully tears down on exit.

        Activations accumulate across all forward passes inside the block.
        Call ``.clear()`` explicitly when you want to reset mid-flight.
        On exit, hooks are removed AND activations are cleared regardless
        of prior state.

        Args:
            model: Target PyTorch model.
            layers: Optional mapping of {name -> module}. If None, attaches to
                    all non-container submodules of `model`.

        Yields:
            self (the tracker instance), not the activations dict. Access
            ``tracker.activations`` explicitly whenever you need them.

        Example:
            >>> with tracker.track(model) as t:
            ...     for batch in loader:
            ...         loss = criterion(model(batch[0]), batch[1])
            ...         loss.backward()
            ...         # read t.activations whenever you want — data accumulates
            ...     # auto-teardown: hooks removed, activations cleared
        """
        self.attach(model, layers)
        try:
            yield self  # yield the tracker instance, not the activations dict
        finally:
            self.remove()  # full teardown: detaches hooks + clears activations

    def clear(self):
        """Clear stored activations to release the computational graph.

        MUST be called after ``loss.backward()`` in store mode to prevent
        PyTorch from keeping the full graph alive. Hooks remain attached so
        the tracker keeps working on subsequent forward passes.
        """
        self._activations.clear()

    def remove(self):
        """Detach all hooks and clear stored activations (full teardown).

        After calling this, ``attach()`` must be invoked again before the
        tracker will capture anything.
        """
        self._activations.clear()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


# Convenience accessor functions for online statistics (C++ backend)
# Each value is a per-element tensor of shape [C, H, W] or [C, SeqLen].
def get_max_stats() -> Dict[str, torch.Tensor]:
    """Return current max activation statistics per layer (per-element tensors)."""
    return dict(_C.get_max_stats())


def get_min_stats() -> Dict[str, torch.Tensor]:
    """Return current min activation statistics per layer (per-element tensors)."""
    return dict(_C.get_min_stats())


def get_mean_stats() -> Dict[str, torch.Tensor]:
    """Return current mean activation statistics per layer (per-element tensors)."""
    return dict(_C.get_mean_stats())


def clear_online_stats():
    """Reset all online statistics computed in C++ backend."""
    _C.clear_stats()
