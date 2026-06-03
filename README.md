# ActivationScope

High-performance PyTorch plugin for tracking, storing, and analyzing intermediate neural network activations during training and inference.

Built on `libtorch` C++ extensions to bypass dispatching overhead, delivering native-speed activation hooks.

---

## Overview

ActivationScope works at the C++ level by binding to PyTorch's internal forward hook infrastructure. In store mode it captures independent copies via ``out.detach().clone()`` so the tracker never holds live autograd references, and in online mode it reduces per-element statistics over the batch dimension — all without bloat or Python dispatch overhead.

### Operation Modes

| Mode | Description |
|------|-------------|
| **Store** | Captures detached clones of activations, safely decoupled from the autograd graph |
| **Online Max/Min/Mean** | Reduces per-element statistics over batch dim 0 and accumulates across forward passes |

---

## Installation

```bash
# Set up Conda environment
conda env create -f environment.yml
conda activate ActivationScope

# Install package (compiles C++ extension)
pip install -e .
```

---

## Usage

### Store Mode

Captures activations and releases them after the backward pass:

```python
import torch
from activationscope import ActivationScope

model = torch.nn.Sequential(
    torch.nn.Linear(10, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 5),
)

tracker = ActivationScope(mode="store")
with tracker.track() as activations:
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
# activations are automatically cleared on exit
```

### Online Mode

Computes running statistics in C++ with no memory overhead:

```python
from activationscope import ActivationScope, get_max_stats, clear_online_stats

model = torch.nn.Linear(10, 20)
tracker = ActivationScope(mode="online_max")
tracker.attach(model, {"fc": model})

output = model(x)
print(get_max_stats())  # {'fc': <max activation value>}

clear_online_stats()
```

---

## Architecture

- **`csrc/hooks.cpp`** — Native PyTorch hooks running under `torch::NoGradGuard`. Online stats reduce over the batch dimension (dim 0), preserving per-element shape `[C, H, W]` across forward passes. Running mean is tracked incrementally via a Welford-style update.
- **`activationscope/tracker.py`** — Python wrapper with context manager support for safe activation lifecycle management. Store mode captures ``out.detach().clone()`` so the tracker never holds live autograd references.
- **C++/Python boundary** — Detached clones in store mode; per-element tensor state mutation in online mode. Neither path retains live graph references.

---

## Development

### Local testing

```bash
# Quick test against your current environment
pytest tests/ -v --tb=short

# Full matrix test across all combos (CPU)
scripts/run_tests.sh

# Full matrix test with CUDA support
scripts/run_tests.sh --platform cu124
```

### Matrix-driven build system

The compatibility matrix lives in `matrix.yml` at the repo root. From it, two artifacts are auto-generated:

| Script | Output | Description |
|--------|--------|-------------|
| `python utils/pyproject.py` | `pyproject.toml` | Fill-in template with versions from matrix |
| `python utils/generate-compose.py` | `.docker/docker-compose.yml` | Docker Compose services for testing |

Both outputs are git-ignored and regenerated before each CI run. There are no hardcoded version floors or classifier logic — everything derives from `matrix.yml`.
