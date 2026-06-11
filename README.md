# ActivationScope

***Jan Miksa @ IDEAS Research Institute***

**High-performance PyTorch activation tracker with online reduction functionality for efficient model analysis.**

Built on Python + C++ with native `libtorch` hooks and `torch.compile` reduction compilation.

**Key Benefits**
- Zero‑copy read‑back: activation tensors are shared between C++ and Python without extra copies.
- Native C++ hooks: no Python dispatch overhead per forward pass.
- Flexible policy knobs (`StoragePolicy`, `ReductionPolicy`, `CapturePolicy`) let you balance memory, compute, and I/O.
- Direct-to-disk streaming (`StoragePolicy.DISK`) — activations are written directly from C++ to disk. Ideal for long-running training loops with very large models. Activations are read back on demand from `.dat` files.
- Works with large models (e.g., diffusion) and supports streaming statistics for online use cases.

---

## Quick Start

Every tracked layer stores full activations by default — no registration needed:

```
import activationscope

with activationscope.ActivationScope().track(model) as tracker:
    for x, y in dataloader:
        out = model(x)
        loss.backward()

acts = tracker.activations  # {layer_name: [Tensor, ...]} across all batches
```

## Performance

### Toy model — 48 × Linear(256,256), batch=32, 200 forwards, CPU

| Approach | ms/forward | Overhead | Data captured |
|---|---|---|---|
| No tracking | 2.83 | — | — |
| Naive Python hooks | 3.05 | +8% | 594 MiB |
| **ActivationScope** | **2.72** | **−4%** | **594 MiB** |

- **Peak VMS identical** — Scope 402,658 vs Naive 402,825 MiB (0.04% diff, within ASLR noise)
- **12% faster** than naive Python hooks (3.05 → 2.72 ms/fwd)
- **Zero-copy readback**: 594 MiB in 3.2 ms

### ResNet-18 (pretrained), batch=8, 20 forwards, M1 CPU

| Approach | ms/forward | Overhead | Data captured |
|---|---|---|---|
| No tracking | 143.4 | — | — |
| Naive Python hooks | 162.3 | +13% | 5023 MiB |
| **ActivationScope** | **158.1** | **+10%** | **5023 MiB** |

- **Peak VMS identical** — Scope 405,496 vs Naive 405,463 MiB (0.01% diff, within ASLR noise)
- **3% faster** than naive Python hooks on a real pretrained model
- **60 layers tracked** across Conv2d, BatchNorm, ReLU, Linear, pooling layers
- **Zero-copy readback**: 5023 MiB in 0.2 ms

Run it yourself:
```bash
# Toy model (fast, GPU or CPU)
PYTHONPATH=. python -m benchmark.runner

# Pretrained ResNet-18 (requires torchvision)
PYTHONPATH=. python -m benchmark.runner --model resnet18
```

---

## Usage Guide

For detailed usage instructions, see the [Usage Documentation](docs/usage/README.md).

## Development Guide

Documentation and developer setup information is available in [Development Documentation](docs/development/README.md).

## Design Documentation

The design document outlining the architecture and implementation details can be found in [Design Documentation](docs/DESIGN.md).
