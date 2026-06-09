# StoragePolicy – Where Tensor Data Lives

The `StoragePolicy` enum determines the *device* on which captured activations are stored. It is consulted inside the C++ hook after any registered reduction has been applied.

## Policies
| Policy | Behaviour | When to Use |
|--------|------------|-------------|
| `CPU`  | `.to(kCPU)` is performed inside the hook (blocking transfer). | Standard single‑GPU training where you want activations on host memory for downstream analysis. |
| `GPU`  | Activations stay on the original device; transfer occurs only on read‑back. | Large models where PCIe bandwidth would be a bottleneck, or when you plan to process activations on‑device. |
| `AUTO` | Heuristic based on tensor size (default threshold = 1 MiB). Small tensors go to CPU, large ones stay on GPU. | Mixed workloads where you want a sensible default without manual tuning. |

## Example
```python
import activationscope

# Force GPU storage for a diffusion‑style model, with a safety rail of 50 captures per layer.
tracker = activationscope.ActivationScope(
    storage=activationscope.StoragePolicy.GPU,
    capture_policy=activationscope.CapturePolicy.MAX_K,
    max_batches=50,
)
```

## Threshold Customisation
`AUTO` accepts `auto_cpu_threshold_bytes` (default 1 048 576). To force even modest tensors onto the CPU you can set a very low threshold:
```python
tracker = activationscope.ActivationScope(
    storage=activationscope.StoragePolicy.AUTO,
    auto_cpu_threshold_bytes=1,  # forces everything onto CPU
)
```

Reference tests: `tests/test_integ_storage_policies.py` and `tests/test_memory_assumptions.py`.
## Interaction with Other Policies
`StoragePolicy` works independently of `ReductionPolicy` and `CapturePolicy`. For example, you can combine `GPU` storage with `STREAMING` reduction to keep only reduced statistics on‑device, dramatically reducing memory pressure.
