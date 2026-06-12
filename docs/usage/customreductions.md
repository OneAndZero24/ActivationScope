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

The `test_svd_analysis.py` module demonstrates a streaming principal‑subspace extraction using two stateful reductions:

```python
# Pass 1: per-layer running sum (for exact mean)
for name in layer_names:
    st = {"running_sum": None, "count": 0}
    def _make_running_sum(st_ref):
        def _sum_reduction(acc, new_tensor):
            reshaped = _reshape_for_svd(new_tensor.float())
            st_ref["count"] += reshaped.shape[0]
            if acc is None:
                return reshaped.sum(dim=0)
            return acc.add_(reshaped.sum(dim=0))   # in-place
        return _sum_reduction
    tracker.register_reduction(_make_running_sum(st), layers=[name])

# ... forward pass 1 ...
# mu = running_sum / count  (computed in Python after readback)

# Pass 2: per-layer blocked covariance accumulation (O(d²) memory)
def _make_cov_accum(d_dim, mu_vec):
    def _cov_accum(acc, new_tensor):
        reshaped = _reshape_for_svd(new_tensor.float())
        if acc is None:
            acc = torch.zeros((d_dim, d_dim))
        for start in range(0, reshaped.shape[0], chunk_size):
            xc = reshaped[start : start + chunk_size] - mu_vec
            acc.add_(xc.T @ xc)
        return acc
    return _cov_accum
tracker2.register_reduction(_make_cov_accum(d, mu), layers=[name])

# ... forward pass 2 ...
# Sigma = final accumulator
# U, S, Vh = torch.linalg.svd(Sigma)  → principal basis
```

This approach uses O(d²) memory per layer independent of the number of data rows N — ideal for very long data streams.

## Advanced Example — Per‑Layer State with Closure Dictionaries

Since closures capture the specific layer data when created, more sophisticated patterns use per‑layer state dictionaries:

```python
layer_states: dict[str, dict] = {}

for name in layer_names:
    st = {"running_sum": None, "count": 0}
    layer_states[name] = st

    def _make_stateful(st_ref: dict):
        def _reduce(acc, new_tensor):
            reduced = torch.mean(new_tensor.float(), dim=0)
            st_ref["count"] += 1
            if acc is None:
                return reduced
            acc.add_(reduced)        # in-place
            return acc               # same reference
        return _reduce

    tracker.register_reduction(_make_stateful(st), layers=[name])

# After forward: read per-layer external state
for name in layer_names:
    mu = layer_states[name]["running_sum"] / layer_states[name]["count"]
```

The closure dictionary pattern avoids the limitation of identity‑based `for_name` / `for_max` and allows arbitrary per‑layer metadata to be tracked alongside the reduction accumulator.
