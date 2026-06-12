# ReductionPolicy – What Gets Kept vs Reduced

`ReductionPolicy` controls **how much data is retained** after each capture. The policy is evaluated inside the C++ hook after an optional user‑registered stateful reduction has been applied.

## Policies
| Policy | Memory Footprint | Behaviour |
|--------|------------------|-----------|
| `STORE_ALL` | O(batches × features) | Every captured tensor is stored unchanged. |
| `STREAMING` | O(features) | A user‑registered stateful reduction is executed each forward; only the reduced result is kept. |
| `FINAL_ONLY` | O(features) | Only the most recent activation per layer is retained.

## Registering Stateful Reductions

Reductions follow a **stateful** contract: `(accumulator, new_tensor) -> updated_accumulator`. Both arguments are views into C++‑owned tensor storage. On the first call, `accumulator` is `None`; the reduction initialises from `new_tensor`. On subsequent calls, it merges `new_tensor` into the accumulator and returns the updated result.

Reductions are compiled with `torch.jit.script` and loaded as TorchScript modules by the C++ backend for near‑native speed with zero GIL on the hot path.

```python
import activationscope, torch

tracker = activationscope.ActivationScope(
    reduction=activationscope.ReductionPolicy.STREAMING,
)

# In-place running-mean reduction (allocates only on first call)
def running_mean(acc, new_tensor):
    reduced = torch.mean(new_tensor.float(), dim=0)
    if acc is None:
        return reduced
    acc.add_(reduced)          # in-place — no allocation
    return acc                 # same reference

tracker.register_reduction(running_mean)
```

### In‑Place vs Allocating

Reductions may use **either** pattern:

- **In‑place** (recommended): mutate `acc` and return the same reference. Example: `acc.add_(x); return acc`. Zero allocation after the first call.
- **Allocating**: return a new tensor. Example: `return acc + x`. Safe, but creates intermediate tensors.

The C++ backend's `replace_last()` correctly handles both — when the reduction returns the same `TensorImpl`, the old reference is decremented only after the new one takes its place, preventing use‑after‑free.

**All built‑in reducers** (`max_reduction`, `min_reduction`, `mean_reduction`) use in‑place operations — they execute with zero allocation after initialisation.

### Stateful vs Stateless

The reduction **must** be stateful — it persists and updates a running accumulator across all batches. The `(accumulator, new_tensor) -> updated_accumulator` signature is required because the C++ backend reads the current accumulator from storage before each call and writes the return value back.

For convenience, common stateful reductions are available as classmethods:

```python
tracker.register_reduction(activationscope.ActivationScope.max_reduction())
tracker.register_reduction(activationscope.ActivationScope.min_reduction())
tracker.register_reduction(activationscope.ActivationScope.mean_reduction())
```

## Per‑Layer vs Global Reductions
- **Global reduction**: `tracker.register_reduction(fn, layers=None)` – applies to any layer without a per‑layer override. Set via `_C.set_global_reduction` in C++.
- **Per‑layer reduction**: `tracker.register_reduction(fn, layers=["conv1", "fc*"])` – matches the supplied fnmatch patterns. Each matched layer gets its own cloned compiled handle.
- If a layer has no explicit reduction, the global reduction (if any) is used; otherwise the identity (store full tensor) applies.

## Convenience Constructors
The library ships ready‑made reducers for common statistics:
```python
activationscope.ActivationScope.for_mean(layers=["linear1"])
activationscope.ActivationScope.for_max(layers=["conv1"])
activationscope.ActivationScope.for_min(layers=["conv1"])
```
These are thin wrappers around `ActivationScope(..., reduction=..., register_reduction=...)`.  The built‑in reducers use in‑place operations — after the first batch, they execute with **zero allocation**.

Reference tests: `tests/test_integ_reduction_policies.py` (including per‑layer and global reduction examples).

## Interaction with StoragePolicy & CapturePolicy & CaptureMode
`ReductionPolicy` works independently of where tensors live (`StoragePolicy`), how often they are captured (`CapturePolicy`), and whether they are cloned (`CaptureMode`). A typical low‑memory configuration for large models is:

```python
tracker = activationscope.ActivationScope(
    storage=activationscope.StoragePolicy.GPU,
    reduction=activationscope.ReductionPolicy.STREAMING,
    capture_policy=activationscope.CapturePolicy.MAX_K,
    max_batches=30,
    capture_mode=activationscope.CaptureMode.REFERENCE,
)
tracker.register_reduction(activationscope.ActivationScope.for_mean())
```

This keeps only a streaming mean on‑device and caps the number of captures per layer.