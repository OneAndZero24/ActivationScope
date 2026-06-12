"""ActivationScope core tracker implementation.

Session-scoped, zero-copy activation tracking via a native C++ backend.
TorchScript-compiled reductions run entirely in C++ — zero GIL, zero Python
on the forward hot path.
"""

from contextlib import contextmanager
from fnmatch import fnmatch
from typing import Callable, Dict, Generator, List, Optional, Any

import torch
import activationscope._C as _C
from activationscope.policies import StoragePolicy, ReductionPolicy, CapturePolicy, CaptureMode
from activationscope.utils import (
    parse_capture_dir,
    pattern_or_identity,
    select_layers,
    export_reduction,
    load_raw_tensor,
)


class ActivationScope:
    """High-performance activation tracker backed by native C++.

    * **Zero-copy readback** — ``.activations`` returns Python lists whose
      elements share the same TensorImpl as the C++ storage vectors.  No data
      is copied across the boundary at readback time.

    * **Native C++ hooks** — forward-pass callbacks run entirely in C++ with
      zero GIL acquisition.  Reductions are compiled via ``torch.jit.script``,
      serialised to a temporary .pt file, and loaded by the C++ backend as a
      TorchScript module that runs ``forward()`` natively.

    * **Four policy knobs** control memory/compute independently:

        * ``StoragePolicy``  (AUTO, CPU, GPU, DISK) — where data lives after detach.
        * ``ReductionPolicy`` (STORE_ALL, STREAMING, FINAL_ONLY) — what gets kept.
        * ``CapturePolicy``   (EVERY, SAMPLE_N, MAX_K) — capture cadence.
        * ``CaptureMode``     (REFERENCE, SNAPSHOT) — clone behaviour after detach.

    Parameters
    ----------
    storage : StoragePolicy
        Device placement policy after detach.  Default AUTO with 1 MiB heuristic.
    reduction : ReductionPolicy
        Retention strategy across forward passes.  Default STORE_ALL.
    capture : CapturePolicy
        Capture cadence.  Default EVERY.
    sample_every : int
        N for SAMPLE_N (every Nth forward).  Ignored when capture != SAMPLE_N.
    max_batches : int
        K for MAX_K (stop after K batches per layer).  0 = unlimited.
    auto_cpu_threshold_bytes : int
        Byte threshold for StoragePolicy.AUTO heuristic.  Default 1 MiB.
    use_pinned : bool
        When True, AUTO/CPU legs use pinned memory + non-blocking async DMA.
    session_dir : str, optional
        Directory for DISK-mode .dat file storage.
    capture_mode : CaptureMode
        Whether to clone captured tensors after detach.
    """

    def __init__(
        self,
        storage: StoragePolicy = StoragePolicy.AUTO,
        reduction: ReductionPolicy = ReductionPolicy.STORE_ALL,
        capture: CapturePolicy = CapturePolicy.EVERY,
        sample_every: int = 1,
        max_batches: int = 0,
        auto_cpu_threshold_bytes: int = 1_048_576,
        use_pinned: bool = False,
        session_dir: Optional[str] = None,
        capture_mode: CaptureMode = CaptureMode.REFERENCE,
    ):
        self._storage: int = int(storage)
        self._reduction: int = int(reduction)

        _sample_every = sample_every if capture == CapturePolicy.SAMPLE_N else 1
        _max_batches  = max_batches  if capture == CapturePolicy.MAX_K    else 0

        self._session_id: Optional[int] = _C.session_create(
            storage=self._storage,
            reduction=self._reduction,
            sample_every=_sample_every,
            max_batches=_max_batches,
            auto_cpu_threshold_bytes=auto_cpu_threshold_bytes,
            use_pinned=use_pinned,
            session_dir=session_dir or "",
            capture_mode=int(capture_mode),
        )

        # Per-layer reductions: list of (.pt path, layer_patterns)
        self._reductions: list = []
        self._global_reduction_path: Optional[str] = None

        # Temp files to clean up on teardown
        self._temp_files: list = []

    # ── Properties ───────────────────────────────────────────────

    @property
    def session_id(self) -> int:
        if self._session_id is None:
            raise RuntimeError("session already destroyed")
        return self._session_id

    @property
    def activations(self) -> Dict[str, List[torch.Tensor]]:
        """Zero-copy readback of accumulated activations.

        Returns a fresh ``dict[str, list[Tensor]]`` on each access.  Each
        tensor in the lists shares the underlying TensorImpl with the C++
        storage vector — no data duplication at boundary crossing.

        In DISK storage mode, tensors are loaded from .dat files on disk.
        """
        if self._session_id is None:
            raise RuntimeError(
                "session already destroyed — cannot read activations. "
                "Re-attach hooks via .attach() or .track()."
            )

        if self._storage == int(StoragePolicy.DISK):
            disk_map = _C.session_readback_disk(self._session_id)
            result: Dict[str, List[torch.Tensor]] = {}
            for layer_name, file_paths in disk_map.items():
                tensors = []
                for fpath in file_paths:
                    try:
                        tensors.append(load_raw_tensor(fpath))
                    except Exception:
                        pass
                if tensors:
                    result[layer_name] = tensors
            return result

        raw = _C.session_readback(self._session_id)
        result: Dict[str, List[torch.Tensor]] = {}
        for key, tensor_list in raw.items():
            if tensor_list:
                result[key] = list(tensor_list)
        return result

    # ── Reduction registration ───────────────────────────────────

    def register_reduction(
        self,
        fn: Callable[[Optional[torch.Tensor], torch.Tensor], torch.Tensor],
        layers: Optional[List[str]] = None,
        dummy_acc: Optional[torch.Tensor] = None,
        dummy_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        """Register a stateful reduction callable.

        The callable receives the current running accumulator (``None`` on the
        first call) and the latest forward-pass tensor, and returns the updated
        accumulator.  It is compiled via ``torch.jit.script``, serialised to a
        temporary .pt file, and loaded by the C++ backend for zero‑GIL execution.

        **State (e.g., count for running mean) must be embedded in the tensor**
        — the accumulator tensor shape is user‑defined and all metadata lives
        inside it.  For example, a running mean tensor can be
        ``[features..., count]`` where the last element tracks batch count.

        Parameters
        ----------
        fn : callable
            Reduction: ``(acc: Tensor | None, tensor: Tensor) -> Tensor``.
        layers : list[str] | None
            Layer-name ``fnmatch`` glob patterns.  ``None`` → global default.
        dummy_acc : Tensor, optional
            Example accumulator tensor for TorchScript warm-up.  Auto-generated if None.
        dummy_tensor : Tensor, optional
            Example activation tensor for TorchScript warm-up.  Auto-generated if None.
        """
        if self._session_id is None:
            raise RuntimeError("session already destroyed")

        if dummy_tensor is None:
            dummy_tensor = torch.randn(8, 64, dtype=torch.float32)
        if dummy_acc is None:
            dummy_acc = torch.randn(64, dtype=torch.float32)

        reduction_path = export_reduction(fn, dummy_acc, dummy_tensor)
        self._temp_files.append(reduction_path)

        if layers is None:
            self._global_reduction_path = reduction_path
        else:
            self._reductions.append((reduction_path, layers))

    # ── Built-in reduction factories ──────────────────────────────

    @classmethod
    def min_reduction(cls):
        """Stateful per‑element min."""
        from typing import Optional
        def _min(acc: Optional[torch.Tensor], new: torch.Tensor) -> torch.Tensor:
            reduced = torch.amin(new, dim=0)
            if acc is None:
                return reduced
            return torch.minimum(acc, reduced, out=acc)
        return _min

    @classmethod
    def max_reduction(cls):
        """Stateful per‑element max."""
        from typing import Optional
        def _max(acc: Optional[torch.Tensor], new: torch.Tensor) -> torch.Tensor:
            reduced = torch.amax(new, dim=0)
            if acc is None:
                return reduced
            return torch.maximum(acc, reduced, out=acc)
        return _max

    @classmethod
    def mean_reduction(cls):
        """Stateful per‑element running mean.
        Accumulator shape: [features + 1] where last element = batch count.
        """
        from typing import Optional
        def _mean(acc: Optional[torch.Tensor], new: torch.Tensor) -> torch.Tensor:
            batch_mean = torch.mean(new.float(), dim=0)
            if acc is None:
                # Append count as an extra row, matching batch_mean's ndim
                count_row = batch_mean[:1] * 0.0 + 1.0
                return torch.cat([batch_mean, count_row], dim=0)
            count = acc[-1]
            running_mean = acc[:-1]
            new_count = count + 1.0
            new_mean = (running_mean * count + batch_mean) / new_count
            # Reconstruct [mean..., count_row]
            return torch.cat([new_mean, new_count.unsqueeze(0)], dim=0)
        return _mean

    # ── Hook attach / detach ─────────────────────────────────────

    @contextmanager
    def track(
        self,
        model: torch.nn.Module,
        layers: Optional[List[str]] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        capture: str = "output",
    ) -> Generator["ActivationScope", Any, None]:
        """Context manager: attach → yield self → detach hooks."""
        self.clear()
        self.attach(model, layers=layers, include=include, exclude=exclude,
                    capture=capture)
        try:
            yield self
        finally:
            if self._session_id is not None:
                _C.session_detach_hooks(self._session_id)

    def attach(
        self,
        model: torch.nn.Module,
        layers: Optional[List[str]] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        capture: str = "output",
    ) -> None:
        """Attach hooks without auto-teardown.

        Glob matching happens once at attach time; the resulting layer set
        is locked and exclusive.

        Each layer's reduction is resolved at attach time: per‑layer patterns
        take priority, falling back to the global default.  The .pt file path
        is passed to the C++ backend which loads the TorchScript module and
        calls it directly on the forward hot path.
        """
        if self._session_id is None:
            raise RuntimeError("session already destroyed")

        selected = select_layers(model, layers=layers, include=include,
                                 exclude=exclude)
        capture_dir_int = parse_capture_dir(capture)

        id_self = self._session_id
        for layer_name, mod in selected.items():
            module_ptr = id(mod)

            # Resolve reduction for this layer
            reduction_path = self._global_reduction_path or ""
            for path, patterns in self._reductions:
                for pat in patterns:
                    if fnmatch(layer_name, pat):
                        reduction_path = path
                        break
                if reduction_path != (self._global_reduction_path or ""):
                    break

            _C.session_register_hooks(
                id_self, module_ptr, layer_name,
                capture_dir_int, reduction_path
            )

    # ── Teardown ─────────────────────────────────────────────────

    def clear(self) -> None:
        """Clear accumulated activations; reset batch counters."""
        if self._session_id is not None:
            _C.session_clear(self._session_id)

    def remove(self) -> None:
        """Full teardown: drop hooks, clear storage, release C++ session."""
        if self._session_id is not None:
            _C.session_destroy(self._session_id)
            self._session_id = None
        for path in self._temp_files:
            try:
                import os
                os.unlink(path)
            except OSError:
                pass
        self._temp_files.clear()

    def __del__(self) -> None:
        try:
            self.remove()
        except Exception:
            pass

    # ── Parameter snapshotting ────────────────────────────────────

    def capture_parameters(
        self,
        model: torch.nn.Module,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Snapshot baseline parameters for protection-loss workflows.

        Iterates ``model.named_parameters()``, applies optional glob filter,
        then returns detached CPU clones.  Fully independent of C++ session.
        """
        snapshot: Dict[str, torch.Tensor] = {}
        for pname, param in model.named_parameters():
            if layers is not None:
                base_name = pname.rsplit(".", 1)[0] if "." in pname else pname
                if not any(fnmatch(base_name, pat) for pat in layers):
                    continue
            snapshot[pname] = param.detach().cpu().clone()
        return snapshot
