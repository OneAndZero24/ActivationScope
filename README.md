# ActivationScope

***Jan Miksa @ IDEAS Research Institute***

**High-performance PyTorch activation tracker with online reduction functionality for efficient model analysis.**

Built on Python + C++ with native `libtorch` hooks and `torch.compile` reduction compilation.

**Key Benefits**
- Zero‑copy read‑back: activation tensors are shared between C++ and Python without extra copies.
- Native C++ hooks: no Python dispatch overhead per forward pass.
- Flexible policy knobs (`StoragePolicy`, `ReductionPolicy`, `CapturePolicy`) let you balance memory, compute, and I/O.
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

acts = tracker.activations  # {layer_name: Tensor} across all batches
```

---

## Usage Guide

For detailed usage instructions, see the [Usage Documentation](docs/usage/README.md).

## Development Guide

Documentation and developer setup information is available in [Development Documentation](docs/development/README.md).

## Design Documentation

The design document outlining the architecture and implementation details can be found in [Design Documentation](docs/DESIGN.md).