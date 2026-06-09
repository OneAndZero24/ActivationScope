# Custom Reductions – User‑Provided Callables

ActivationScope allows you to **register arbitrary reduction functions** that are compiled and executed entirely in the C++ backend, eliminating Python overhead on the hot path.

## Registering a Reduction
```python
import activationscope, torch

tracker = activationscope.ActivationScope()
# Example: per‑layer standard deviation
tracker.register_reduction(
    lambda t: torch.std(t, dim=0),  # reduction logic
    layers=["encoder.layer.0"]
)
```
The callable is compiled with `torch.compile` (fallback to `torch.jit.script`). Compilation happens eagerly at registration time, so the first real forward pass incurs no compilation latency.

## Global vs Per‑Layer Reductions
- **Global reduction** – `layers=None` applies to all tracked layers that lack a per‑layer override.
- **Per‑layer reduction** – supply a list of fnmatch patterns to target specific modules.

```python
# Global mean reduction for every layer
tracker.register_reduction(activationscope.ActivationScope.for_mean())
# Override a specific conv layer with max reduction
tracker.register_reduction(activationscope.ActivationScope.for_max(), layers=["conv1"])
```

Reference tests: `tests/test_integ_reduction_policies.py` (per‑layer, global fallback, and convolutional examples).
## Safety & Guarantees
- Reductions run **outside the GIL** in the compiled graph, so they do not block other Python threads.
- The reduction must return a tensor of the same dtype/device (or a device‑compatible one). Mismatched devices raise a runtime error inside the C++ hook.
- Because the reduction runs **before the tensor is detached**, gradients are not retained in the stored activation.

## Interaction with Other Policies
Reductions are the **only Python‑level computation** that occurs inside the hook. After the reduction, the resulting tensor is passed to the storage policy (CPU/GPU/AUTO) for final placement.

## Advanced Example – Streaming Statistics
```python
import torch

tracker = activationscope.ActivationScope(
    reduction=activationscope.ReductionPolicy.STREAMING,
)
# Streaming mean and variance for each layer
tracker.register_reduction(lambda t: torch.mean(t, dim=0))   # mean
tracker.register_reduction(lambda t: torch.var(t, dim=0), layers=["*"], name="var")
```
The above registers two independent reductions; each layer will store both a streaming mean and variance, demonstrating that multiple reductions can be stacked (each uses its own internal state).
