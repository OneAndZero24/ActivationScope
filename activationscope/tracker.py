"""ActivationScope core tracker implementation.

Session-scoped, zero-copy activation tracking via a native C++ backend.
Three independently tunable policy knobs govern every aspect of memory
and compute behavior (see ``activationscope.policies``).
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
    try_compile,
    warmup,
    load_raw_tensor,
)


class ActivationScope:
    """High-performance activation tracker backed by native C++.

    * **Zero-copy readback** — ``.activations`` returns Python lists whose
      elements share the same TensorImpl as the C++ storage vectors.  No data
      is copied across the boundary at readback time.

    * **Native hooks** — C++ lambdas fire during forward pass; only a thin
      pybind11 dispatch frame traverses the boundary, eliminating per-forward
      GIL + frame creation overhead.

    * **Four policy knobs** control memory/compute independently:

        * ``StoragePolicy``  (AUTO, CPU, GPU, DISK) — where data lives after detach.
          Pinned-memory modifier enables async DMA for AUTO/CPU legs.
          DISK mode streams tensors directly to .dat files, bypassing RAM entirely.
        * ``ReductionPolicy`` (STORE_ALL, STREAMING, FINAL_ONLY) — what gets
          kept vs. reduced across batches.
        * ``CapturePolicy``   (EVERY, SAMPLE_N, MAX_K) — capture cadence
          and safety rail against unbounded growth.
        * ``CaptureMode``     (REFERENCE, SNAPSHOT) — whether to clone tensors
          for independent storage after detach.

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
        Whether to clone captured tensors after detach.  REFERENCE (default)
        shares storage with the autograd graph; SNAPSHOT creates an independent
        copy for safe post‑capture mutation.
    """

    def __init__(
        self,
        storage: StoragePolicy = StoragePolicy.AUTO,
        reduction: ReductionPolicy = ReductionPolicy.STORE_ALL,
        capture: CapturePolicy = CapturePolicy.EVERY,
        sample_every: int = 1,
        max_batches: int = 0,
        auto_cpu_threshold_bytes: int = 1_048_576,   # 1 MiB
        use_pinned: bool = False,
        session_dir: Optional[str] = None,  # DISK mode: where to store .dat files
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

        # Per-layer registered reductions: list of (compiled_fn_handle, layer_patterns)
        self._reductions: list = []
        self._global_reduction = None

    # ── Properties ───────────────────────────────────────────────

    @property
    def session_id(self) -> int:
        """Opaque C++ session identifier."""
        if self._session_id is None:
            raise RuntimeError("session already destroyed — cannot access session_id")
        return self._session_id

    @property
    def activations(self) -> Dict[str, List[torch.Tensor]]:
        """Zero-copy readback of accumulated activations.

        Returns a fresh ``dict[str, list[Tensor]]`` on each access.  Each
        tensor in the lists shares the underlying TensorImpl with the C++
        storage vector — no data duplication at boundary crossing.

        Returned tensors are **read-only views**.  To mutate safely, call
        ``tensor.clone()`` first.

        In DISK storage mode, tensors are loaded from .dat files on disk
        rather than read from in-memory C++ vectors.
        """
        if self._session_id is None:
            raise RuntimeError(
                "session already destroyed — cannot read activations. "
                "Re-attach hooks via .attach() or .track()."
            )

        # DISK mode: read tensors from raw binary .dat files on disk.
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

        # Standard in-memory readback path.
        raw = _C.session_readback(self._session_id)

        # Convert C++ map → Python dict.  TensorImpls are shared (zero-copy).
        result: Dict[str, List[torch.Tensor]] = {}
        for key, tensor_list in raw.items():
            if tensor_list:   # skip layers with empty vectors
                result[key] = list(tensor_list)
        return result

    # ── Compiled reduction registration ──────────────────────────

    def register_reduction(
        self,
        fn: Callable[[Optional[torch.Tensor], torch.Tensor], torch.Tensor],
        layers: Optional[List[str]] = None,
    ) -> None:
        """Register a stateful reduction callable.

        The callable receives the current running accumulator (``None`` on the
        first call) and the latest forward-pass tensor, and returns the updated
        accumulator.  It is compiled via ``torch.compile()`` and transferred to
        C++ for native execution.

        When *layers* is ``None``, fn becomes the **session-wide default** used
        as fallback for any layer without an explicit per-layer reduction.

        Parameters
        ----------
        fn : Callable[[Optional[Tensor], Tensor], Tensor]
            Reduction callable (accumulator, new_tensor) → updated_accumulator.
            Both arguments are views into C++‑owned storage — no copies are
            made at the boundary.  The reduction may either mutate the
            accumulator in‑place (return the same reference) or return a new
            tensor.  In‑place is recommended for allocation‑free hot paths.
        layers : list[str] | None
            Layer-name glob patterns (``fnmatch`` style).  When ``None``, fn is
            set as the global default reduction for unmatched layers.
        """
        if self._session_id is None:
            raise RuntimeError("session already destroyed")

        compiled_fn = try_compile(fn)
        warmup(compiled_fn)

        handle = _C.make_compiled_handle(compiled_fn)

        if layers is None:
            self._global_reduction = handle
            _C.set_global_reduction(self._session_id, handle)
        else:
            self._reductions.append((handle, layers))

    @classmethod
    def min_reduction(cls):
        """Return a stateful reduction: per-element min across all batches.

        Signature: ``(running_min, new_tensor) -> updated_running_min``.
        On first call (running_min is None), initialises from the first batch.
        Mutates the accumulator **in-place** — no allocation on the hot path.
        """
        def _min(acc, new):
            reduced = torch.amin(new, dim=0)
            if acc is None:
                return reduced
            return torch.minimum(acc, reduced, out=acc)
        return _min

    @classmethod
    def max_reduction(cls):
        """Return a stateful reduction: per-element max across all batches.

        Signature: ``(running_max, new_tensor) -> updated_running_max``.
        On first call (running_max is None), initialises from the first batch.
        Mutates the accumulator **in-place** — no allocation on the hot path.
        """
        def _max(acc, new):
            reduced = torch.amax(new, dim=0)
            if acc is None:
                return reduced
            return torch.maximum(acc, reduced, out=acc)
        return _max

    @classmethod
    def mean_reduction(cls):
        """Return a stateful reduction: per-element mean across all batches.

        Uses a weighted running average so memory stays O(features) regardless
        of batch count.  Mutates the accumulator **in-place** — no allocation.

        Signature: ``(running_mean, new_tensor) -> updated_running_mean``.

        Note
        ----
        The closure tracks a single global count shared across all layers this
        reduction is registered for.  For per‑layer counts, register per‑layer
        reductions with per‑layer closure dicts instead.
        """
        state = {"count": 0}

        def _mean(acc, new):
            batch_mean = torch.mean(new.float(), dim=0)
            state["count"] += 1
            if acc is None:
                return batch_mean
            # In-place weighted running average: acc = (1-w)*acc + w*batch
            w = 1.0 / state["count"]
            acc.mul_(1.0 - w).add_(batch_mean, alpha=w)
            return acc
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
        """Context manager: attach → yield self → detach hooks.

        Activations accumulate across all forward passes inside the block.
        On exit, hooks are removed but the session survives so ``track()``
        can be called again on the same tracker.

        Parameters
        ----------
        model : nn.Module
            Target PyTorch model.
        layers : list[str] | None
            Glob patterns matched against ``model.named_modules()``.  When both
            *layers* and *include* are ``None``, all non-container submodules
            are selected.
        include : list[str] | None
            Inclusion glob patterns (union).  Applied before *exclude*.
        exclude : list[str] | None
            Exclusion glob patterns (subtractive).  Matched names are removed.
        capture : str
            One of ``"output"`` (default), ``"input"``, or ``"both"``.
        """
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

        Glob matching happens exactly once at attach time; the resulting layer
        set is locked and exclusive for this session's lifetime.

        Parameters
        ----------
        model : nn.Module
            Target PyTorch model.
        layers : list[str] | None
            Explicit inclusion list (fnmatch patterns).
        include : list[str] | None
            Synonym for *layers*; applied before *exclude*.
        exclude : list[str] | None
            Subtractive fnmatch patterns.
        capture : str
            Capture direction: ``"output"`` | ``"input"`` | ``"both"``.
        """
        if self._session_id is None:
            raise RuntimeError("session already destroyed")

        selected = select_layers(model, layers=layers, include=include,
                                 exclude=exclude)
        capture_dir_int = parse_capture_dir(capture)

        id_self = self._session_id
        for layer_name, mod in selected.items():
            module_ptr = id(mod)
            _C.session_register_hooks(id_self, module_ptr, layer_name,
                                      capture_dir_int)

            for handle, layer_patterns in self._reductions:
                for lp in layer_patterns:
                    if fnmatch(layer_name, lp):
                        _C.set_layer_reduction(id_self, layer_name, handle)
                        break  # first matching pattern wins for this layer

    # ── Teardown ─────────────────────────────────────────────────

    def clear(self) -> None:
        """Clear accumulated activations; reset batch counters.

        Hooks remain attached so the tracker keeps working on subsequent
        forward passes.  Calling this repeatedly between backward passes in
        an iterative training loop prevents unbounded memory growth.
        """
        if self._session_id is not None:
            _C.session_clear(self._session_id)

    def remove(self) -> None:
        """Full teardown: drop hooks, clear all storage, release C++ session.

        After calling this, ``attach()`` must be invoked again before the
        tracker will capture anything.
        """
        if self._session_id is not None:
            _C.session_destroy(self._session_id)
            self._session_id = None

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

        Parameters
        ----------
        model : nn.Module
            Target PyTorch model.
        layers : list[str] | None
            Glob patterns for layer prefixes to include.  When ``None``, all
            parameters are captured.

        Returns
        -------
        dict[str, Tensor]
            Mapping from parameter name → detached, CPU-cloned tensor.
        """
        snapshot: Dict[str, torch.Tensor] = {}
        for pname, param in model.named_parameters():
            if layers is not None:
                base_name = pname.rsplit(".", 1)[0] if "." in pname else pname
                if not any(fnmatch(base_name, pat) for pat in layers):
                    continue
            snapshot[pname] = param.detach().cpu().clone()
        return snapshot
