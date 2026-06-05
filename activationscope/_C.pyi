"""Type stubs for the compiled C++ extension ``activationscope._C``.

Generated from the PYBIND11_MODULE bindings in ``csrc/bindings.cpp``.
Mirrors the v2 session-scoped, zero-copy API.
"""

from typing import Any, Dict, List

import torch

# ── Session lifecycle ──────────────────────────────────────────────

def session_create(
    storage: int,
    reduction: int,
    sample_every: int,
    max_batches: int,
    auto_cpu_threshold_bytes: int,
    use_pinned: bool,
) -> int: ...

def session_destroy(session_id: int) -> None: ...

def session_readback(
    session_id: int,
) -> Dict[str, List[torch.Tensor]]: ...

def session_clear(session_id: int) -> None: ...

# ── Hook registration ───────────────────────────────────────

def session_register_hooks(
    session_id: int,
    module_ptr: Any,
    layer_name: str,
    capture_dir: int,
) -> None: ...

# ── Reduction registration ──────────────────────────────────

def make_compiled_handle(fn: Any) -> int: ...

def set_layer_reduction(
    session_id: int,
    layer_name: str,
    handle: int,
) -> None: ...

def set_global_reduction(
    session_id: int,
    handle: int,
) -> None: ...
