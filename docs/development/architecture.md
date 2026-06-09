# Architecture Overview (for Developers)

ActivationScope’s architecture is intentionally split between a **high‑performance C++ backend** and a **thin Python wrapper**. The design goals are:

1. **Zero‑copy activation read‑back** – Tensor data lives exclusively in C++ (`std::vector<torch::Tensor>`). Python receives read‑only `torch.Tensor` views that share the underlying `TensorImpl` without copying.
2. **Native libtorch hooks** – Hooks are registered directly via libtorch’s `register_forward_hook` API in C++. The callback is a pure C++ lambda; no Python callable is invoked on the hot path, eliminating GIL contention.
3. **Policy‑driven memory management** – Three orthogonal knobs (`StoragePolicy`, `ReductionPolicy`, `CapturePolicy`) let users trade off memory, compute, and I/O:
   - `StoragePolicy` decides where tensors are stored (CPU, GPU, or heuristic `AUTO`).
   - `ReductionPolicy` determines what is retained (full tensors, streaming reductions, or final only).
   - `CapturePolicy` controls capture frequency (every forward, every Nth forward, or a hard cap `MAX_K`).
4. **Session‑scoped lifecycle** – Each `ActivationScope` instance creates a unique `SessionState` keyed by an atomic `uint64_t`. Multiple trackers can coexist without interference, supporting nested tracking and multi‑model experiments.
5. **Extensible reductions** – User‑provided callables are compiled with `torch.compile` (or `torch.jit.script` as fallback) and stored as opaque C++ handles (`CompiledFnHandle`). The C++ hook invokes the compiled graph directly, bypassing Python.
6. **Thread‑safe accumulation** – A single `std::mutex` protects the per‑layer `ActivationAccumulator` during concurrent forward passes. The lock scope is minimized to a map lookup + vector `push_back`, keeping contention low.

### Key Files
- `csrc/session.cpp/.hpp` – Session creation, destruction, and read‑back logic.
- `csrc/hook_register.cpp/.hpp` – Native hook registration and callback implementation.
- `csrc/core.cpp/.hpp` – Hot‑path logic: early‑exit checks, reduction dispatch, storage policy handling, and accumulation.
- `activationscope/tracker.py` – Python façade exposing `ActivationScope`, policy enums, and helper methods (`register_reduction`, `capture_parameters`).

### Further Reading
- **Design Document** – `docs/DESIGN.md` provides a deep dive into the zero‑copy guarantees, lifetime management, and the interplay of the three policy knobs.
- **Architecture‑Specific Tests** – `tests/test_memory_assumptions.py` and `tests/test_memory_leak_detection.py` verify that the design invariants hold under various configurations.

When extending the library, keep these invariants in mind. Any change that alters tensor ownership, introduces Python dispatch, or modifies the session lifecycle must be reflected in the design document and covered by new tests.
