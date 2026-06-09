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

## Using `torch.compile` for Custom Reductions
The library automatically compiles registered reductions with `torch.compile`. You can also pre‑compile a reduction manually and pass it in:
```python
import torch

compiled_fn = torch.compile(lambda t: torch.mean(t, dim=0))
tracker = activationscope.ActivationScope()
tracker.register_reduction(compiled_fn)
```
Compiled functions enjoy the same zero‑overhead execution as built‑in reductions because they are stored as opaque handles and invoked directly from C++.

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
| Compiled reductions | `register_reduction` | Works with any `StoragePolicy` and `CapturePolicy`. |
| Zero‑copy read‑back | All policies | Guarantees memory efficiency regardless of other settings.

These advanced patterns enable research‑grade activation analysis while preserving the library’s performance guarantees.
