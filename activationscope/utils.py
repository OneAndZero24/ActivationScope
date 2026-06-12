"""ActivationScope helper utilities.

Layer selection, capture-direction parsing, compiled-reduction warm-up,
raw-tensor disk loading, and other functions shared across the package.
"""

from fnmatch import fnmatch
from typing import Callable, Dict, List, Optional, Any

import torch


# ── Layer selection ──────────────────────────────────────────────────

def parse_capture_dir(capture: str) -> int:
    """Translate capture direction string to C++ enum int (CaptureDir)."""
    mapping = {"input": 0, "output": 1, "both": 2}
    cap = capture.lower()
    if cap not in mapping:
        raise ValueError(
            f"capture must be 'input', 'output', or 'both'; got '{capture}'"
        )
    return int(mapping[cap])


def select_layers(
    model: torch.nn.Module,
    layers: Optional[List[str]] = None,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> Dict[str, torch.nn.Module]:
    """Apply glob filters to named_modules and return locked layer set.

    Steps:
        1. Enumerate model.named_modules()
        2. If *include* is None -> use all non-container submodules as baseline
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

    selected = all_modules

    patterns = layers if include is None else include
    if patterns:
        selected = {
            name: mod for name, mod in selected.items()
            if any(
                fnmatch(name, pat)
                or fnmatch(type(mod).__name__, pat)
                or fnmatch(f"{name}.{type(mod).__name__}", pat)
                for pat in patterns
            )
        }

    if exclude:
        selected = {
            name: mod for name, mod in selected.items()
            if not any(
                fnmatch(name, pat)
                or fnmatch(type(mod).__name__, pat)
                or fnmatch(f"{name}.{type(mod).__name__}", pat)
                for pat in exclude
            )
        }

    # Edge case: model is a leaf module with no non-container submodules
    if not selected and not isinstance(model, containers):
        if len(list(model.children())) == 0:
            selected = {"": model}

    return selected


# ── Compiled reduction helpers ────────────────────────────────────────

def try_compile(
    fn: Callable[..., torch.Tensor]
) -> Callable[..., torch.Tensor]:
    """Try to compile a reduction callable via torch.compile()."""
    if hasattr(torch, "compile") and callable(torch.compile):
        try:
            return torch.compile(fn)       # type: ignore[return-value]
        except (RuntimeError, TypeError):
            pass
    return fn


def warmup(
    compiled_fn: Callable[..., torch.Tensor]
) -> None:
    """Execute a synthetic warm-up forward so first real batch is not cold-start."""
    try:
        dummy = torch.randn(2, 64, dtype=torch.float32)
        # Call with None accumulator (simulates first batch).
        _ = compiled_fn(None, dummy)
    except Exception:
        pass


def pattern_or_identity(fn: Callable[..., Any]) -> str:
    """Derive a simple matchable key for an arbitrary callable."""
    return getattr(fn, "__name__", repr(fn))


# ── Raw tensor disk loading ───────────────────────────────────────────

def load_raw_tensor(filepath: str) -> torch.Tensor:
    """Load a tensor from the raw binary .dat format written by the C++ DISK path.

    File layout (all little-endian int64):
        [dtype: int64][ndim: int64][dim0..dimN: int64][raw data bytes]
    """
    import struct
    import numpy as np

    with open(filepath, "rb") as f:
        header = f.read(16)
        if len(header) < 16:
            raise ValueError(f"Truncated .dat file: {filepath}")
        dtype_code, ndim = struct.unpack("<qq", header)

        shape = []
        for _ in range(ndim):
            dim_bytes = f.read(8)
            if len(dim_bytes) < 8:
                raise ValueError(f"Truncated shape in .dat file: {filepath}")
            dim, = struct.unpack("<q", dim_bytes)
            shape.append(dim)

        data = f.read()

    torch_dtype = _ATEN_SCALAR_TO_TORCH.get(dtype_code, torch.float32)

    np_array = np.frombuffer(data, dtype=_torch_to_numpy_dtype(torch_dtype))
    tensor = torch.from_numpy(np_array.copy().reshape(shape))
    return tensor.contiguous()


# Map ATen ScalarType int codes to torch dtypes
_ATEN_SCALAR_TO_TORCH = {
    0:  torch.uint8,      1:  torch.int8,       2:  torch.int16,
    3:  torch.int32,      4:  torch.int64,       5:  torch.float16,
    6:  torch.float32,    7:  torch.float64,     8:  torch.complex64,
    9:  torch.complex64,   10: torch.complex128,  11: torch.bool,
    12: torch.qint8,       13: torch.quint8,      14: torch.qint32,
    15: torch.bfloat16,
}


def _torch_to_numpy_dtype(torch_dtype: torch.dtype):
    """Convert torch dtype to numpy dtype for buffer reading."""
    import numpy as np
    _t2n = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.float16: np.float16,
        torch.bfloat16: np.uint16,
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.uint8: np.uint8,
        torch.bool: np.bool_,
    }
    return _t2n.get(torch_dtype, np.float32)
