# Convenience Constructors

ActivationScope provides a set of **class‑methods** that pre‑configure common reduction patterns. They are thin wrappers around the main constructor and automatically register the appropriate reduction.

## Available Constructors
| Constructor | What it does | Example |
|-------------|--------------|---------|
| `ActivationScope.for_mean(layers=None)` | Registers a global mean reduction (`torch.mean(t, dim=0)`). | `tracker = ActivationScope.for_mean(layers=["linear1"])` |
| `ActivationScope.for_max(layers=None)` | Registers a global max reduction (`torch.max(t, dim=0)`). | `tracker = ActivationScope.for_max(layers=["conv1"])` |
| `ActivationScope.for_min(layers=None)` | Registers a global min reduction (`torch.min(t, dim=0)`). | `tracker = ActivationScope.for_min(layers=["conv1"])` |

## When to Use
- **Quick prototyping** – you can create a tracker in a single line without manually calling `register_reduction`.
- **Standard analytics** – mean/max/min are common statistics for activation analysis, PCA, or attention rollout.
- **Consistency** – using the built‑in constructors ensures the reduction is compiled with the same settings as the rest of the library.

Reference tests: `tests/test_integ_reduction_policies.py` (convenience constructors).
## Interaction with Other Settings
These constructors respect the current `StoragePolicy` and `CapturePolicy` settings of the `ActivationScope` instance. For example:
```python
tracker = ActivationScope.for_mean(layers=["layer1"]).
tracker.storage = ActivationScope.StoragePolicy.GPU
tracker.capture_policy = ActivationScope.CapturePolicy.SAMPLE_N
tracker.sample_every = 10
```
All policies can be combined freely; the convenience constructor only adds the reduction step.
