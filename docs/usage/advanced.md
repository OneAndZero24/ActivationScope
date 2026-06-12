# Advanced Usage Patterns

Beyond the core policies, ActivationScope supports several advanced features that enable sophisticated analysis pipelines.

## Capturing Inputs *and* Outputs
By default the tracker captures the **output** of each selected layer. You can also capture the **input** or **both** by setting the `capture` argument:
```python
tracker = activationscope.ActivationScope(capture="both")
with tracker.track(model, include=[".*conv.*"]):
    out = model(x)
# The activation dict now contains keys with ``.input`` and ``.output`` suffixes.
assert "conv1.input" in tracker.activations
assert "conv1.output" in tracker.activations
```
This is useful for analyzing layer‑wise Jacobians or for debugging shape mismatches.

## Parameter Snapshotting (Protection‑Loss Workflows)
ActivationScope can snapshot model parameters alongside activations, a pattern common in protection‑loss research (e.g., InTAct).
```python
params = tracker.capture_parameters(model, layers=["encoder.*", "decoder.*"])
# ``params`` is a plain ``dict[layer_name] -> Tensor`` (CPU‑detached).
```
The snapshot is completely independent of the activation storage and lives in regular Python tensors.

## Using `torch.compile` for Stateful Reductions

The library automatically compiles registered reductions with `torch.compile`. You can also pre‑compile a reduction manually and pass it in:

```python
import torch

def running_max(acc, new_tensor):
    """Stateful reduction: (accumulator, tensor) -> updated_accumulator.
    
    Uses in-place mutation — no allocation after the first call.
    Both ``acc`` and ``new_tensor`` are views into C++ storage.
    """
    reduced = torch.amax(new_tensor, dim=0)
    if acc is None:
        return reduced
    return torch.maximum(acc, reduced, out=acc)   # in-place

compiled_fn = torch.compile(running_max)
tracker = activationscope.ActivationScope()
tracker.register_reduction(compiled_fn)
```

Compiled functions enjoy the same zero‑overhead execution as built‑in reductions because they are stored as opaque handles and invoked directly from C++.

The reduction must accept **two** arguments: `(accumulator, new_tensor)`. On the first call, `accumulator` is `None` — the reduction must initialise from `new_tensor` alone. Both arguments are views into C++‑owned tensor storage — no copies are made at the boundary.

### In‑Place vs Allocating

Reductions can use **either** pattern:

- **In‑place** (recommended): mutate `acc` and return it — `acc.add_(x); return acc`. After the first call, zero allocation. This is the approach used by all built‑in reducers (`max_reduction`, `min_reduction`, `mean_reduction`).
- **Allocating**: return a new tensor — `return acc + x`. Safe, but creates intermediate tensors.

The C++ backend handles both patterns correctly via `replace_last()`.

### Per‑Layer State with Closure Dictionaries

For tracking per‑layer metadata alongside reductions (e.g., sample counts), use closure‑captured dictionaries:

```python
layer_states: dict[str, dict] = {}

for name in layer_names:
    st = {"running_sum": None, "count": 0}
    layer_states[name] = st

    def _make_stateful(st_ref):
        def _reduce(acc, new_tensor):
            reduced = new_tensor.float().sum(dim=0)
            st_ref["count"] += 1
            if acc is None:
                return reduced
            acc.add_(reduced)        # in-place
            return acc               # same reference
        return _reduce

    tracker.register_reduction(_make_stateful(st), layers=[name])
```

## CaptureMode — Reference vs Snapshot

`CaptureMode` controls whether captured tensors share storage with the autograd graph or are independently cloned:

```python
from activationscope import ActivationScope, CaptureMode

# REFERENCE (default): detach only — shares storage, fastest
tracker = ActivationScope(capture_mode=CaptureMode.REFERENCE)

# SNAPSHOT: detach + clone — independent copy, safe for mutation
tracker = ActivationScope(capture_mode=CaptureMode.SNAPSHOT)
```

| Mode | What happens | Use when |
|------|-------------|----------|
| `REFERENCE` | `.detach()` only — shares `TensorImpl` | Read‑only analysis; best performance |
| `SNAPSHOT` | `.detach()` + `.clone()` — independent tensor | Tensors may be mutated after capture; protection‑loss loops |

The native C++ backend implements `SNAPSHOT` as `result = result.clone()` in the hook callback — zero Python overhead. The pure‑Python tracker (`_naive.py`) also supports both modes.

## Zero‑Copy Guarantees & Read‑Only Views
All tensors returned via ``tracker.activations`` share the underlying `TensorImpl` with the C++ storage. They are **read‑only**; any in‑place operation raises a `RuntimeError`. To modify a tensor you must ``clone()`` it first:
```python
acts = tracker.activations
layer_tensor = acts["conv1"][0].clone()
layer_tensor.mul_(2)  # safe – does not affect C++ storage
```

Reference tests: `tests/test_memory_assumptions.py`, `tests/test_memory_leak_detection.py`, `tests/test_integ_capture_policies.py`, and `tests/test_e2e_models.py`.

## Interaction Summary
| Feature | Affects | Typical Combination |
|---------|----------|---------------------|
| Input/Output capture | `capture` argument | Use with any storage/reduction policy for full observability. |
| Parameter snapshotting | `capture_parameters` method | Often paired with `STREAMING` reduction to compare model drift over time. |
| Stateful reductions | `register_reduction` | Works with any `StoragePolicy`, `CapturePolicy`, and `CaptureMode`. |
| CaptureMode | `capture_mode` argument | `SNAPSHOT` when post‑processing mutates; `REFERENCE` otherwise. |
| Zero‑copy read‑back | All policies | Guarantees memory efficiency regardless of other settings.

These advanced patterns enable research‑grade activation analysis while preserving the library’s performance guarantees.
