# ActivationScope

***Jan Miksa @ IDEAS Research Institute***

**High-performance PyTorch activation tracker with online reduction functionality for efficient model analysis.**

Built on Python + C++ with native `libtorch` hooks and **TorchScript** (`torch.jit.script`) reductions compiled to `.pt` files.

**Key Benefits**
- Zero‑copy read‑back: activation tensors are shared between C++ and Python without extra copies.
- Native C++ hooks: no Python compute overhead per forward pass.
- Flexible policy knobs (`StoragePolicy`, `ReductionPolicy`, `CapturePolicy`, `CaptureMode`) let you balance memory, compute, and I/O.
- Direct-to-disk streaming (`StoragePolicy.DISK`) — activations are written directly from C++ to disk. Ideal for long-running training loops with very large models. Activations are read back on demand from `.dat` files.
- Works with large models (e.g., diffusion) and supports streaming statistics for online use cases.

---

## Quick Start

Every tracked layer stores full activations by default — no registration needed:

```python
import activationscope

with activationscope.ActivationScope().track(model) as tracker:
    for x, y in dataloader:
        out = model(x)
        loss.backward()

acts = tracker.activations  # {layer_name: [Tensor, ...]} across all batches
```

## Performance

### Toy model — 48 × Linear(256,256), batch=32, 200 forwards, CPU

| Approach | ms/forward | Overhead vs baseline | Data captured |
|---|---|---|---|
| No tracking | 2.05 | — | — |
| Naive Python hooks | 3.13 | +52.7% | 594 MiB |
| **ActivationScope** | **2.65** | **+29.2%** | **594 MiB** |

- **Peak VMS identical** — Scope 402,506 vs Naive 402,630 MiB (~0.03% diff, within ASLR noise)
- **1.18× faster** than naive Python hooks (3.13 → 2.65 ms/fwd)
- **95 layers tracked** (inputs + outputs across 48 linear layers)
- **Zero-copy readback**: 594 MiB in 2.4 ms

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
