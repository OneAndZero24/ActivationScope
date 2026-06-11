# StoragePolicy – Where Tensor Data Lives

The `StoragePolicy` enum determines the *device* on which captured activations are stored. It is consulted inside the C++ hook after any registered reduction has been applied.

## Policies
| Policy | Behaviour | When to Use |
|--------|------------|-------------|
| `CPU`  | `.to(kCPU)` is performed inside the hook (blocking transfer). | Standard single‑GPU training where you want activations on host memory for downstream analysis. |
| `GPU`  | Activations stay on the original device; transfer occurs only on read‑back. | Large models where PCIe bandwidth would be a bottleneck, or when you plan to process activations on‑device. |
| `AUTO` | Heuristic based on tensor size (default threshold = 1 MiB). Small tensors go to CPU, large ones stay on GPU. | Mixed workloads where you want a sensible default without manual tuning. |
| `DISK` | **Direct-to-disk streaming.** Activations are written from C++ directly to `.dat` files in a temporary directory, bypassing RAM entirely. | Very large models or long training runs where in-memory activation accumulation would exceed available RAM. Activations are loaded on demand during readback. |

## DISK Storage Mode

`StoragePolicy.DISK` streams every captured activation directly to disk from the C++ hook callback. No intermediate Python objects are created — the tensor is serialized in raw binary format and written to a temporary file before the hook returns.

### Key characteristics:
- **Zero RAM overhead**: activations never accumulate in memory. Each forward pass writes activations, and the tracker's `activations` property returns an empty dict while hooks are active.
- **On-demand readback**: after the tracking context exits (or hooks are detached), call `tracker.activations` to load all tensors from disk back into Python.
- **Automatic cleanup**: temporary files are deleted when the tracker is destroyed (via `__del__` or explicit `.remove()`).
- **File layout**: each layer gets a subdirectory under a session-specific temp dir (`/tmp/activationscope_<id>_<random>/<layer>/<batch_idx>.dat`).

### Format:
Each `.dat` file stores a single tensor in raw binary:
```
[int64 dtype_code] [int64 ndim] [int64 dim0]...[int64 dimN] [raw data bytes]
```
Dtype codes use the ATen `ScalarType` enum (0=Byte, ..., 6=Float, 7=Double, 15=BFloat16).

### Example:
```python
import activationscope

# Stream all activations directly to disk — zero RAM overhead per batch
tracker = activationscope.ActivationScope(
    storage=activationscope.StoragePolicy.DISK,
    capture=activationscope.CapturePolicy.MAX_K,
    max_batches=100,
)

with tracker.track(large_model) as t:
    for batch in dataloader:
        out = large_model(batch)
        loss.backward()

# Read activations from disk into Python memory on demand
acts = t.activations  # {layer_name: [Tensor, ...]}
```

### Performance trade-off:
DISK mode is slower than in-memory modes (1-3 ms/fwd vs 0.15-1.7 ms/fwd) due to disk I/O, but uses effectively zero RAM for stored activations. This makes it viable for models and batch counts that would otherwise OOM with `STORE_ALL`.

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
