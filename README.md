# ActivationScope

***Jan Miksa @ IDEAS Research Institute***

**High-performance PyTorch activation tracker with storage, reduction, and capture policies for efficient model analysis.**

Built on Python + C++ with native libtorch hooks and `torch.compile` reduction compilation.

---

## Development Installation

```bash
conda env create -f environment.yml -n activationscope
conda activate activationscope

pip install -e .
```

---

## Quick Start

Every tracked layer stores full activations by default — no registration needed:

```python
import activationscope

with activationscope.ActivationScope().track(model) as tracker:
    for x, y in dataloader:
        out = model(x)
        loss.backward()

acts = tracker.activations  # {layer_name: Tensor} across all batches
```

---

## Usage 

### StoragePolicy — Where Tensor Data Lives

| Policy      | Behavior                              | When to Use                  |
|-------------|---------------------------------------|------------------------------|
| `CPU`       | `.to(kCPU)` during hook (default)     | Standard single-GPU tracking |
| `GPU`       | Keep on device, transfer at readback  | Diffusion models, avoid PCIe saturation during forward |
| `AUTO`      | Small tensors → CPU, large → GPU      | Mixed workloads              |

```python
import activationscope

# Keep activations on GPU during forwards.
# Readback transfers only when you access .activations.
tracker = activationscope.ActivationScope(
    storage=activationscope.StoragePolicy.GPU,
    capture_policy=activationscope.CapturePolicy.MAX_K,
    max_batches=50,  # safety rail: stop after N denoising steps
)
```

### ReductionPolicy — What Gets Kept vs Reduced

| Policy       | Memory          | Use Case                          |
|-------------|-----------------|-----------------------------------|
| `STORE_ALL`  | O(batches × features) | Per-batch analysis, PCA, attention rollout |
| `STREAMING`  | O(features) | Online statistics only: running mean, max, variance across all batches |
| `FINAL_ONLY` | O(features), last forward | Debugging minimal overhead      |

```python
tracker = activationscope.ActivationScope(
    reduction_policy=activationscope.ReductionPolicy.STREAMING,
)
tracker.register_reduction(lambda t: torch.mean(t, dim=0))

with tracker.track(model) as t:
    for _ in range(1000):  # 1000 batches → still only O(features) memory
        model(x)

means = t.activations  # [features] per layer, running average
```

### CapturePolicy — How Often to Capture

| Policy      | Behavior                              | Use Case                            |
|-------------|---------------------------------------|-------------------------------------|
| `EVERY`     | Every forward pass (default)          | Standard use                        |
| `SAMPLE_N`  | Every Nth forward                     | Long training loops, periodic snapshots |
| `MAX_K`     | Stop after K batches per layer        | Safety rail against OOM             |

```python
# Hard cap: capture first 50 forwards per layer, then stop — prevents OOM.
tracker = activationscope.ActivationScope(
    capture_policy=activationscope.CapturePolicy.MAX_K,
    max_batches=50,
)

# Sample every 10th forward (captures 10% of passes).
# Note: can alias with periodic data (e.g., diffusion denoising steps).
tracker = activationscope.ActivationScope(
    capture_policy=activationscope.CapturePolicy.SAMPLE_N,
    sample_every=10,
)
```

### Layer Filtering

Use fnmatch patterns to track only specific submodules:

```python
with activationscope.ActivationScope().track(
    model,
    include=[".*attn.*", "*.attention.*"],   # track only attention layers
    exclude=[".*bias.*"],                     # skip bias tensors
) as t:
    out = model(x)
```

### Custom Reductions

Any callable is compiled at registration (`torch.compile` preferred, `torch.jit.script` fallback):

```python
tracker = activationscope.ActivationScope()
tracker.register_reduction(
    lambda t: torch.std(t, dim=0),           # per-element std dev
    layers=["encoder.layer.0"],
)

with tracker.track(model) as t:
    out = model(x)

stds = t.activations  # compiled reduction runs in C++ every forward
```

### Convenience Constructors

```python
activationscope.ActivationScope.for_mean(layers=["linear1"])
activationscope.ActivationScope.for_max(layers=["conv1"])
activationscope.ActivationScope.for_min(layers=["conv1"])
```
