# ActivationScope – Overview

ActivationScope is a **high‑performance activation tracking library** for PyTorch models. It provides:

- **Zero‑copy read‑back**: Activations are stored in a native C++ backend and exposed to Python as read‑only views without extra copies.
- **Native libtorch hooks**: Hooks are registered directly in C++ avoiding the per‑forward Python dispatch overhead.
- **Three orthogonal policy knobs**
  - `StoragePolicy` – controls where tensor data lives (CPU, GPU, or `AUTO`).
  - `ReductionPolicy` – defines what is kept (full tensors, streaming reductions, or only the final tensor).
  - `CapturePolicy` – determines *when* activations are captured (every forward, every *N*th forward, or a hard cap of *K* captures).
- **Layer selection** via fnmatch patterns to track only the layers you care about.
- **Custom reduction registration** with `torch.compile`‑accelerated callables.
- **Convenient constructors** (`for_mean`, `for_max`, `for_min`) for common reductions.

All of these features are exercised in the test suite (`tests/`). See the respective sections in this documentation for detailed usage examples and links to the corresponding tests.
