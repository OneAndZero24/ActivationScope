"""ActivationScope core tracker implementation.

Session-scoped, zero-copy activation tracking via a native C++ backend.
Three independently tunable policy knobs govern every aspect of memory
and compute behavior (see StoragePolicy, ReductionPolicy, CapturePolicy).
"""

import sys
from contextlib import contextmanager
from fnmatch import fnmatch
from typing import Callable, Dict, Generator, List, Optional, Any

import torch
import activationscope._C as _C


# ──────────── Policy enums (mirror C++ enum values) ──────────────

class StoragePolicy(int):
    """Where tensor data lives after capture."""
    AUTO = 0    # Heuristic: < threshold → CPU, ≥ threshold → GPU
    CPU  = 1    # Blocking transfer to host memory
    GPU  = 2    # Stay on original device


class ReductionPolicy(int):
    """What gets kept vs. reduced across batches."""
    STORE_ALL  = 0  # Full tensor per batch appended
    STREAMING  = 1  # Per-batch reduction output replaces/accumulates in-place
    FINAL_ONLY = 2  # Last-batch activation overwrites previous


class CapturePolicy(int):
    """When and how often hooks fire."""
    EVERY    = 0  # Every forward fires hooks
    SAMPLE_N = 1  # Captures every Nth forward
    MAX_K    = 2  # Captures exactly K batches then stops

# ──────────── ActivationScope class ──────────────────────────────

class ActivationScope:
    """High-performance activation tracker backed by native C++.

    * **Zero-copy readback** — ``.activations`` returns Python lists whose
      elements share the same TensorImpl as the C++ storage vectors.  No data
      is copied across the boundary at readback time.

    * **Native hooks** — C++ lambdas fire during forward pass; only a thin
      pybind11 dispatch frame traverses the boundary, eliminating per-forward
      GIL + frame creation overhead.

    * **Three policy knobs** control memory/compute independently:

        * ``StoragePolicy``  (AUTO, CPU, GPU) — where data lives after detach.
          Pinned-memory modifier enables async DMA for AUTO/CPU legs.
        * ``ReductionPolicy`` (STORE_ALL, STREAMING, FINAL_ONLY) — what gets
          kept vs. reduced across batches.
        * ``CapturePolicy``   (EVERY, SAMPLE_N, MAX_K) — capture cadence
          and safety rail against unbounded growth.

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
    ):
        self._storage: int = int(storage)
        self._reduction: int = int(reduction)

        _capture_int  = int(capture)
        _sample_every = sample_every if capture == CapturePolicy.SAMPLE_N else 1
        _max_batches  = max_batches  if capture == CapturePolicy.MAX_K    else 0

        self._session_id: Optional[int] = _C.session_create(
            storage=self._storage,
            reduction=self._reduction,
            sample_every=_sample_every,
            max_batches=_max_batches,
            auto_cpu_threshold_bytes=auto_cpu_threshold_bytes,
            use_pinned=use_pinned,
        )

        # Per-layer registered reductions: pattern → (compiled_fn_handle, layer_names)
        self._reductions: Dict[str, tuple] = {}
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
        """
        if self._session_id is None:
            raise RuntimeError(
                "session already destroyed — cannot read activations. "
                "Re-attach hooks via .attach() or .track()."
            )
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
        fn: Callable[[torch.Tensor], torch.Tensor],
        layers: Optional[List[str]] = None,
    ) -> None:
        """Register a callable as a compiled reduction.

        The callable is compiled via ``torch.compile()`` for native execution
        speed, a warm-up forward with a synthetic tensor runs immediately so the
        first real batch never experiences cold-compile latency, and the compiled
        handle is transferred to C++ storage.

        When *layers* is ``None``, fn becomes the **session-wide default** used
        as fallback for any layer without an explicit per-layer reduction.

        Parameters
        ----------
        fn : Callable[[Tensor], Tensor]
            Reduction callable operating on a single batch activation tensor.
        layers : list[str] | None
            Layer-name glob patterns (``fnmatch`` style).  When ``None``, fn is
            set as the global default reduction for unmatched layers.
        """
        if self._session_id is None:
            raise RuntimeError("session already destroyed")

        # Compile fn via torch.compile() when available, fall back to raw callable
        compiled_fn = _try_compile(fn)

        # Warm-up forward with synthetic tensor so first real batch is cold-free
        _warmup(compiled_fn)

        # Transfer handle to C++
        handle = _C.make_compiled_handle(compiled_fn)

        if layers is None:
            # Global default fallback
            self._global_reduction = handle
            _C.set_global_reduction(self._session_id, handle)
        else:
            # Store patterns for lazy matching at attach time; also resolve any
            # already-attached modules immediately.
            self._reductions[pattern_or_identity(fn)] = (handle, layers)

    @classmethod
    def for_max(cls):
        """Return a callable that reduces to per-element max over dim 0."""
        return lambda x: torch.amax(x, dim=0)

    @classmethod
    def for_mean(cls):
        """Return a callable that reduces to per-element mean over dim 0."""
        return lambda x: torch.mean(x.float(), dim=0)

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
        """Context manager: attach → yield self → full teardown.

        Activations accumulate across all forward passes inside the block.
        On exit, hooks are removed AND activations are cleared regardless of
        prior state.

        Parameters
        ----------
        model : nn.Module
            Target PyTorch model.
        layers : list[str] | None
            Glob patterns matched against ``model.named_modules()``.  When both
            *layers* and *include* are ``None*, all non-container submodules are
            selected.
        include : list[str] | None
            Inclusion glob patterns (union).  Applied before *exclude*.
        exclude : list[str] | None
            Exclusion glob patterns (subtractive).  Matched names are removed.
        capture : str
            One of ``"output"`` (default), ``"input"``, or ``"both"``.
        """
        self.attach(model, layers=layers, include=include, exclude=exclude,
                    capture=capture)
        try:
            yield self
        finally:
            self.remove()

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

        # Glob-based layer selection — locked at attach time (DESIGN.md §6.1)
        selected = _select_layers(model, layers=layers, include=include,
                                  exclude=exclude)

        # Validate capture direction
        capture_dir_int = _parse_capture_dir(capture)

        # Attach native C++ hooks for each selected layer
        id_self = self._session_id
        for layer_name, mod in selected.items():
            module_ptr = id(mod)  # uintptr-like value for the module object
            _C.session_register_hooks(id_self, module_ptr, layer_name,
                                      capture_dir_int)

            # Resolve per-layer reductions that match this layer (DESIGN.md §2.3)
            for pattern in self._reductions:
                if fnmatch(layer_name, pattern):
                    handle = self._reductions[pattern][0]
                    _C.set_layer_reduction(id_self, layer_name, handle)

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
        """Destructor — ensures C++ session cleanup even if user forgets .remove()."""
        try:
            self.remove()
        except Exception:
            pass  # safe no-op during interpreter shutdown

    # ── Parameter snapshotting (DESIGN.md §7) ─────────────────────

    def capture_parameters(
        self,
        model: torch.nn.Module,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Snapshot baseline parameters for InTAct-style workflows.

        Iterates ``model.named_parameters()``, applies optional glob filter,
        then returns detached CPU clones.  Fully independent of C++ session —
        no cross-boundary tensor copies.

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
                # Match against layer-name prefixes (strip trailing .weight/.bias)
                base_name = pname.rsplit(".", 1)[0] if "." in pname else pname
                if not any(fnmatch(base_name, pat) for pat in layers):
                    continue
            snapshot[pname] = param.detach().cpu().clone()
        return snapshot


# ──────────── Helpers ─────────────────────────────────────────────

def _parse_capture_dir(capture: str) -> int:
    """Translate capture string to C++ enum int (CaptureDir)."""
    mapping = {"input": 0, "output": 1, "both": 2}
    cap = capture.lower()
    if cap not in mapping:
        raise ValueError(
            f"capture must be 'input', 'output', or 'both'; got '{capture}'"
        )
    return int(mapping[cap])


def _select_layers(
    model: torch.nn.Module,
    layers: Optional[List[str]] = None,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> Dict[str, torch.nn.Module]:
    """Apply glob filters to named_modules and return locked layer set.

    Steps (DESIGN.md §6.1):
        1. Enumerate model.named_modules()
        2. If *include* is None → use all non-container submodules as baseline
        3. Apply include patterns (fnmatch union)
        4. Subtract exclude patterns
    """
    # Baseline: exclude container types and the root module itself
    containers = (torch.nn.ModuleList, torch.nn.ModuleDict, torch.nn.Sequential)
    all_modules: Dict[str, torch.nn.Module] = {
        name: mod
        for name, mod in model.named_modules()
        if not isinstance(mod, containers) and name != ""
    }

    selected = all_modules  # start with everything

    patterns = layers if include is None else include
    if patterns:
        # Intersection: only keep modules that match at least one pattern
        selected = {
            name: mod for name, mod in selected.items()
            if any(fnmatch(name, pat) for pat in patterns)
        }

    if exclude:
        # Subtractive: remove matches
        selected = {
            name: mod for name, mod in selected.items()
            if not any(fnmatch(name, pat) for pat in exclude)
        }

    return selected


def _try_compile(
    fn: Callable[[torch.Tensor], torch.Tensor]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Try to compile a reduction callable via torch.compile()."""
    if hasattr(torch, "compile") and callable(torch.compile):
        try:
            return torch.compile(fn)       # type: ignore[return-value]
        except (RuntimeError, TypeError):
            pass                          # fall through to raw fn
    return fn


def _warmup(
    compiled_fn: Callable[[torch.Tensor], torch.Tensor]
) -> None:
    """Execute a synthetic warm-up forward so first real batch is not cold-start."""
    try:
        dummy = torch.randn(2, 64, dtype=torch.float32)
        _ = compiled_fn(dummy)
    except Exception:
        pass  # non-fatal; surface at registration if critical


def pattern_or_identity(fn: Callable[..., Any]) -> str:
    """Derive a simple matchable key for an arbitrary callable.

    Used as a rough pattern when the user passes fn but no explicit layer list —
    not ideal, but provides a fallback so register_reduction without *layers*
    can still be stored in ``self._reductions`` without colliding with the
    global default path.
    """
    return getattr(fn, "__name__", repr(fn))
