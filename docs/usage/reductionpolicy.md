# ReductionPolicy – What Gets Kept vs Reduced

`ReductionPolicy` controls **how much data is retained** after each capture. The policy is evaluated inside the C++ hook after an optional user‑registered reduction has been applied.

## Policies
| Policy | Memory Footprint | Behaviour |
|--------|------------------|-----------|
| `STORE_ALL` | O(batches × features) | Every captured tensor is stored unchanged. |
| `STREAMING` | O(features) | A user‑registered reduction is executed each forward; only the reduced result is kept. |
| `FINAL_ONLY` | O(features) | Only the most recent activation per layer is retained.

## Registering Reductions
Reductions are callables that map a `torch.Tensor` → `torch.Tensor`. They are compiled with `torch.compile` (fallback to `torch.jit.script`) for near‑native speed.

```python
import activationscope, torch

tracker = activationscope.ActivationScope(
    reduction=activationscope.ReductionPolicy.STREAMING,
)
# Register a mean reduction for all layers
tracker.register_reduction(lambda t: torch.mean(t, dim=0))
```

## Per‑Layer vs Global Reductions
- **Global reduction**: `tracker.register_reduction(fn, layers=None)` – applies to any layer without a per‑layer override.
- **Per‑layer reduction**: `tracker.register_reduction(fn, layers=["conv1", "fc*"])` – matches the supplied fnmatch patterns.
- If a layer has no explicit reduction, the global reduction (if any) is used; otherwise the identity (store full tensor) applies.

## Convenience Constructors
The library ships ready‑made reducers for common statistics:
```python
activationscope.ActivationScope.for_mean(layers=["linear1"])
activationscope.ActivationScope.for_max(layers=["conv1"])
activationscope.ActivationScope.for_min(layers=["conv1"])
```
These are thin wrappers around `ActivationScope(..., reduction=..., register_reduction=...)`.

Reference tests: `tests/test_integ_reduction_policies.py` (including per‑layer and global reduction examples).
## Interaction with StoragePolicy & CapturePolicy
`ReductionPolicy` works independently of where tensors live (`StoragePolicy`) and how often they are captured (`CapturePolicy`). A typical low‑memory configuration for large models is:
```python
tracker = activationscope.ActivationScope(
    storage=activationscope.StoragePolicy.GPU,
    reduction=activationscope.ReductionPolicy.STREAMING,
    capture_policy=activationscope.CapturePolicy.MAX_K,
    max_batches=30,
)
tracker.register_reduction(activationscope.ActivationScope.for_mean())
```
This keeps only a streaming mean on‑device and caps the number of captures per layer.
