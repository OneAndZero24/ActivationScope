# Architecture Overview (for Developers)

ActivationScope's architecture is intentionally split between a **high‑performance C++ backend** and a **thin Python wrapper**. The design goals are:

1. **Zero‑copy activation read‑back** — Tensor data lives exclusively in C++ (`std::vector<torch::Tensor>`). Python receives read‑only `torch.Tensor` views that share the underlying `TensorImpl` without copying.
2. **Native libtorch hooks** — Hooks are registered directly via libtorch's `register_forward_hook` API in C++. The callback is a pure C++ lambda; no Python callable is invoked on the hot path, eliminating GIL contention.
3. **Policy‑driven memory management** — Four orthogonal knobs (`StoragePolicy`, `ReductionPolicy`, `CapturePolicy`, `CaptureMode`) let users trade off memory, compute, and I/O:
   - `StoragePolicy` decides where tensors are stored (CPU, GPU, DISK, or heuristic `AUTO`).
   - `ReductionPolicy` determines what is retained (full tensors, streaming stateful reductions, or final only).
   - `CapturePolicy` controls capture frequency (every forward, every Nth forward, or a hard cap `MAX_K`).
   - `CaptureMode` controls tensor cloning: `REFERENCE` (detach only, shares storage) or `SNAPSHOT` (detach + clone, independent copy).
4. **Session‑scoped lifecycle** — Each `ActivationScope` instance creates a unique `SessionState` keyed by an atomic `uint64_t`. Multiple trackers can coexist without interference, supporting nested tracking and multi‑model experiments.
5. **Extensible stateful reductions** — User‑provided callables follow a `(Optional[Tensor], Tensor) -> Tensor` contract. Both arguments are views into C++‑owned tensor storage (no copies at the boundary). They are compiled with `torch.jit.script`, serialised to temporary `.pt` files, and loaded as `torch::jit::script::Module` objects by the C++ backend. The C++ hook invokes the TorchScript graph directly, bypassing Python — zero GIL, zero serialisation. The accumulator is stored via `replace_last()` which is safe for in‑place mutations that return the same `TensorImpl`.
6. **Thread‑safe accumulation** — A single `std::mutex` protects the per‑layer `ActivationAccumulator` during concurrent forward passes. The lock scope is minimized to a map lookup + accumulator update, keeping contention low.

### Key Files
- `csrc/session.cpp/.hpp` — Session creation (now accepts `CaptureMode`), destruction, and read‑back logic.
- `csrc/callback.cpp/.hpp` — Hot‑path logic: early‑exit checks, stateful reduction dispatch, capture mode (clone on `SNAPSHOT`), storage policy handling, and accumulation.
- `csrc/hook_register.cpp/.hpp` — Native hook registration and callback implementation.
- `csrc/reduction.hpp/.cpp` — TorchScript module wrapper loaded from `.pt` file, zero‑GIL `forward()`.
- `csrc/datastructures.hpp` — All shared enums (`StoragePolicy`, `ReductionPolicy`, `CapturePolicy`, `CaptureMode`) and structs.
- `activationscope/tracker.py` — Python façade exposing `ActivationScope`, policy enums, `register_reduction`, `capture_parameters`, and `capture_mode`.
- `activationscope/policies.py` — Python‑side enum definitions for all four policy knobs.

### Further Reading
- **Design Document** — `docs/DESIGN.md` provides a deep dive into the zero‑copy guarantees, lifetime management, and the interplay of the four policy knobs.
- **Architecture‑Specific Tests** — `tests/test_memory_assumptions.py` and `tests/test_memory_leak_detection.py` verify that the design invariants hold under various configurations.

When extending the library, keep these invariants in mind. Any change that alters tensor ownership, introduces Python dispatch, or modifies the session lifecycle must be reflected in the design document and covered by new tests.
