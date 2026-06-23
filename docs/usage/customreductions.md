# Custom Reductions – Stateful User‑Provided Callables

ActivationScope allows you to register **arbitrary stateful reduction functions** that are compiled with `torch.jit.script` into TorchScript graphs and executed entirely in the C++ backend, eliminating Python overhead on the hot path.

## Stateful Reduction Contract

The reduction callable must follow a `(accumulator, new_tensor) -> updated_accumulator` signature:

- On the **first call** for a layer, `accumulator` is `None`. The reduction must initialise from `new_tensor` alone.
- On **subsequent calls**, `accumulator` holds the running result from previous batches. The reduction must merge `new_tensor` into it and return the updated accumulator.
- The accumulator lives in C++ storage; Python never materialises it until readback via `.activations`.
- Both `accumulator` and `new_tensor` are **views** into C++‑owned storage — no copies are made at the boundary. The reduction never owns or allocates memory for the raw activation data.

### In‑Place vs Allocating — Two Supported Patterns

Reductions are free to use **either** pattern:

| Pattern | Example | Memory behaviour |
|---------|---------|-----------------|
| **Allocating** (return new) | `return torch.minimum(acc, reduced)` | Old accumulator ref dropped, new tensor allocated. Safe but creates intermediate tensors. |
| **In‑place** (mutate + return same) | `return torch.minimum(acc, reduced, out=acc)` or `acc.add_(x); return acc` | Zero allocation — overwrites same storage. **Recommended for production hot paths.** |

The C++ backend handles both patterns correctly via `replace_last()` — when the reduction returns the same `TensorImpl` as the accumulator, the old reference is decremented only **after** the new reference takes its place, preventing use‑after‑free.

**All built‑in reductions** (`max_reduction`, `min_reduction`, `mean_reduction`) use in‑place operations — they do not allocate on the hot path after the first call.

## Registering a Reduction

```python
import activationscope, torch

tracker = activationscope.ActivationScope(
    reduction=activationscope.ReductionPolicy.STREAMING,
)

# In-place running-mean reduction: (acc, tensor) -> acc  (no allocation after init)
def running_mean(acc, new_tensor):
    reduced = torch.mean(new_tensor.float(), dim=0)
    if acc is None:
        return reduced
    acc.add_(reduced)          # in-place accumulation
    return acc                 # return same reference

tracker.register_reduction(running_mean)
```

The callable is compiled with `torch.jit.script`. Compilation happens eagerly at registration time, so the first real forward pass incurs no compilation latency.

## Global vs Per‑Layer Reductions

- **Global reduction** — `layers=None` applies to all tracked layers that lack a per‑layer override. This becomes the *session‑wide default* and is set via `_C.set_global_reduction`.
- **Per‑layer reduction** — supply a list of `fnmatch` patterns to target specific modules. Each matching layer gets its own cloned compiled handle (C++ calls `clone_compiled_handle()`), ensuring no double‑free when one pattern matches multiple layers.

```python
# Global running-max for every layer
tracker.register_reduction(activationscope.ActivationScope.max_reduction())

# Override a specific conv layer with running-min
tracker.register_reduction(
    activationscope.ActivationScope.min_reduction(), layers=["conv1"]
)
```

Resolution order at hook‑fire time: **per‑layer config → session‑wide default → identity** (store full tensor).

## Built‑in Stateful Reducers (Allocation‑Free)

The library ships ready‑made stateful reductions as classmethods.  **All three use in‑place operations** — after the first batch (which initialises), subsequent calls do not allocate.

| Method | Signature | Behaviour |
|--------|-----------|-----------|
| `ActivationScope.max_reduction()` | `(running_max, tensor) → max` | Per‑element max via `torch.maximum(..., out=acc)` |
| `ActivationScope.min_reduction()` | `(running_min, tensor) → min` | Per‑element min via `torch.minimum(..., out=acc)` |
| `ActivationScope.mean_reduction()` | `(running_mean, tensor) → mean` | Weighted running average via `acc.mul_(...).add_(...)` |

```python
tracker = activationscope.ActivationScope()
tracker.register_reduction(activationscope.ActivationScope.max_reduction())
tracker.register_reduction(
    activationscope.ActivationScope.mean_reduction(), layers=["fc1"]
)
```

Reference tests: `tests/test_integ_reduction_policies.py` (per‑layer, global fallback, and convolutional examples).

## Safety & Guarantees

- Reductions run **outside the GIL** in the compiled graph, so they do not block other Python threads.
- The reduction must accept and return tensors on the same device/dtype. Mismatches raise a runtime error inside the C++ hook.
- Because the reduction runs **before the tensor is detached**, gradients are not retained in the stored activation.
- Per‑layer handle cloning ensures that multiple layers matching one glob pattern each get their own independent compiled‑graph instance — preventing use‑after‑free double destruction.

## Interaction with Other Policies

Reductions are the **only Python‑level computation** that occurs inside the hook. After the reduction, the resulting tensor is passed through the `CaptureMode` check (clone on `SNAPSHOT`) and then to the storage policy (CPU/GPU/AUTO/DISK) for final placement.

## Advanced Example — Online SVD via Two‑Pass Stateful Reductions

The `test_svd_analysis.py` module demonstrates a streaming principal‑subspace extraction using two stateful reductions. **All state is embedded in the accumulator tensor itself** — no Python dicts or closures hold metadata.

### Pass 1: running sum with count in the tensor

```python
from typing import Optional
import torch

def _make_running_sum():
    """Accumulator shape: [features..., count] where last element = sample count."""
    def _sum_reduction(acc: Optional[torch.Tensor], new_tensor: torch.Tensor) -> torch.Tensor:
        reshaped = _reshape_for_svd(new_tensor.float())
        batch_count = torch.tensor(float(reshaped.size(0)))
        if acc is None:
            return torch.cat([reshaped.sum(dim=0), batch_count.unsqueeze(0)])
        running_sum = acc[:-1]
        count = acc[-1]
        return torch.cat([running_sum.add_(reshaped.sum(dim=0)),
                          (count + batch_count).unsqueeze(0)])
    return _sum_reduction

for name in layer_names:
    tracker.register_reduction(_make_running_sum(), layers=[name])
```

After forward pass 1, extract ``mu = accumulator[:-1] / accumulator[-1]`` from the readback tensor.

### Pass 2: covariance with mu-vector in the tensor

```python
def _make_cov_accum():
    """Accumulator shape: [cov_matrix_flattened..., mu_vector...] — mu is the last row,
    pre-seeded before the first forward via session_init_accumulator."""
    def _cov_reduction(acc: Optional[torch.Tensor], new_tensor: torch.Tensor) -> torch.Tensor:
        reshaped = _reshape_for_svd(new_tensor.float())
        if acc is None:
            return torch.zeros(1)               # placeholder, overwritten pre-seed
        mv = acc[-1]                            # mu vector
        cov = acc[:-1]                          # covariance matrix (view)
        for start in range(0, reshaped.size(0), 4096):
            xc = reshaped[start : start + 4096] - mv
            cov.add_(xc.T @ xc)                 # in-place blockwise update
        return torch.cat([cov, mv.unsqueeze(0)], dim=0)
    return _cov_reduction
```

This approach uses O(d²) memory per layer independent of the number of data rows N — ideal for very long data streams. See [SVD Analysis](svdanalysis.md) for the full two-pass pipeline with pre-seeding via ``session_init_accumulator``.

## Advanced Example — Running Mean with Weighted Count in Tensor

The built-in ``mean_reduction`` shows how to encode arbitrary bookkeeping as extra tensor dimensions, matching the ``mean_reduction`` pattern from ``ActivationScope.mean_reduction()``:

```python
from typing import Optional

def weighted_mean_reduction():
    """Accumulator shape: [features..., count] — last element tracks batch count.
    Computes welford-style running mean without external state."""
    def _reduce(acc: Optional[torch.Tensor], new_tensor: torch.Tensor) -> torch.Tensor:
        batch_mean = torch.mean(new_tensor.float(), dim=0)
        if acc is None:
            # First call: [mean..., 1.0] — count initialised to 1
            count_row = batch_mean[:1] * 0.0 + 1.0
            return torch.cat([batch_mean, count_row], dim=0)

        count = acc[-1]
        running_mean = acc[:-1]
        new_count = count + 1.0
        # Weighted update: (old_mean * old_count + new_batch_mean) / new_count
        new_mean = (running_mean * count + batch_mean) / new_count
        return torch.cat([new_mean, new_count.unsqueeze(0)], dim=0)
    return _reduce

tracker.register_reduction(weighted_mean_reduction(), layers=["fc1"])
```

After forward passes, the accumulator tensor holds ``[features..., count]``.  To read back:

```python
acts = tracker.activations["fc1"][0].float()
mu = acts[:-1] / acts[-1]    # final mean = stored_mean (count already applied)
```

This pattern — encoding all metadata inside the accumulator tensor shape — is required because reductions are compiled to TorchScript graphs that run in C++ outside the GIL. Dicts, closures, and mutable Python objects cannot be used at the reduction boundary.
