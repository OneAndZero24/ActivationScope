"""Type stubs for the compiled C++ extension ``activationscope._C``.

TorchScript reduction path — reductions are compiled via torch.jit.script,
serialised to .pt files, and loaded by the C++ backend.
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
    session_dir: str = "",
    capture_mode: int = 0,
) -> int: ...

def session_destroy(id: int) -> None: ...

def session_readback(
    id: int,
) -> Dict[str, List[torch.Tensor]]: ...

def session_readback_disk(
    id: int,
) -> Dict[str, List[str]]: ...

def session_clear(id: int) -> None: ...

def session_detach_hooks(id: int) -> None: ...

# ── Hook registration ──────────────────────────────────────────────

def session_register_hooks(
    id: int,
    module_ptr: Any,
    layer_key: str,
    capture_dir_int: int,
    reduction_path: str = "",
) -> None: ...
