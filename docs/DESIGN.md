# ActivationScope — Design Document

A PyTorch activation tracking library built on a native C++ backend that
owns all tensor memory. Hooks are registered directly via libtorch module APIs;
no Python callable lives on the forward hot path. Four policy knobs (storage,
reduction, capture, capture-mode) govern every aspect of behaviour. Zero-copy
readback ensures Python views C++ storage directly.

---

## 1. Project Overview

ActivationScope intercepts forward-pass activations of named submodules in a
PyTorch model and makes them available after execution — without holding live
references into the autograd graph, without Python dispatch per hook fire, and
without unbounded memory growth.

The library is split across two layers:

- **C++ backend** (compiled via `torch.utils.cpp_extension`) owns all tensor
  storage in a per-session `SessionState` struct keyed by an atomic counter.
  Hooks fire as pure C++ lambdas; callbacks never touch Python on the hot path.

- **Python layer** (`activationscope.tracker.ActivationScope`) handles hook
  attachment, zero-copy readback, lifecycle management, and policy
  configuration.  Three policy enums live in `activationscope.policies`;
  shared helpers live in `activationscope.utils`.

---

## 2. File Structure

```
activationscope/
├── __init__.py           Public re-exports (ActivationScope, policies)
├── policies.py           StoragePolicy, ReductionPolicy, CapturePolicy enums
├── tracker.py            ActivationScope class — session lifecycle, reduction registration
├── utils.py              Layer selection, capture-dir parsing, warmup, disk I/O
└── _C.pyi                Type stubs for the compiled C++ extension

csrc/
├── bindings.cpp          PYBIND11_MODULE — thin wrappers, zero logic
├── session.cpp/.hpp      Session lifecycle, global registry, disk readback
├── callback.cpp/.hpp     HOT PATH — reduction, detach, storage, accumulate
├── hook_register.cpp/.hpp Hook registration on nn.Module via pybind11
├── capture_policy.cpp/.hpp Atomic capture-policy enforcement
├── compiled_fn.cpp/.hpp  Opaque PyObject* wrapper around torch.compile()d reductions
├── accumulator.hpp       ActivationAccumulator — vector<Tensor> with zero-copy readback
├── datastructures.hpp    All shared enums and structs (SessionState, LayerHookConfig, etc.)
├── utils.hpp             Shared utilities (sanitize_layer_name)
└── gil_utils.hpp         GIL RAII guard (GilStateGuard, ensure_gil_and_call)
```

---

## 3. Core Architectural Decisions

### 3.1 Zero-Copy Readback

Python does NOT copy data from C++. pybind11 transparently wraps `torch::Tensor`
so both sides share the same `TensorImpl`. The `.activations` property returns a
**fresh Python list** on each access — list topology is independent of the C++
`std::vector`, so concurrent hook appends cannot invalidate Python-side iterators.
Returned tensors are read-only views; users must `.clone()` to mutate.

### 3.2 Native Libtorch Hooks

Hooks are registered directly in C++ through libtorch's module hook API.
The callback is a pure C++ lambda — zero GIL contention on the forward hot path
and zero accidental autograd-graph retention through Python closures.

### 3.3 Stateful User-Provided Reductions

Reductions are user-supplied callables compiled via `torch.compile()`.
The signature is stateful: `fn(running_accumulator, new_tensor) -> updated_accumulator`.
On the first call the accumulator is `None`; the reduction initialises from the
first batch.  On subsequent calls, `new_tensor` is merged into `accumulator`
and the updated accumulator is returned.

Both `accumulator` and `new_tensor` are **views** into C++‑owned storage — no
copies are made at the Python/C++ boundary.  The reduction may either allocate
a new tensor (returning a fresh accumulator) or mutate the accumulator in-place
and return the same reference.

In C++, `replace_last()` stores the result atomically — the old accumulator
reference is decremented **after** the new one takes its place, so in-place
mutations that return the same `TensorImpl` cannot trigger use‑after‑free.

This enables running statistics (running max, running min, online variance,
online covariance) without materialising all batches.

The built‑in convenience reducers (`max_reduction`, `min_reduction`,
`mean_reduction`) use in‑place operations exclusively — after the first
batch they execute with **zero allocation**.

If no reduction is registered, the identity path stores the full tensor as-is.

Convenience methods (`max_reduction`, `min_reduction`, `mean_reduction`) are
pure Python classmethods — nothing hardcoded in C++.

```python
tracker.register_reduction(
    activationscope.ActivationScope.max_reduction(),
    layers=["transformer.layers.*"]
)
```

### 3.4 Per-Layer Reduction With Global Fallback

Resolution order: per-layer config → session-wide default → identity.
Dispatch resolves at hook-fire time in C++; no Python involved.

### 3.5 Session-Scoped State Isolation

Each `ActivationScope` instance owns a C++ session keyed by a unique `uint64_t`
(atomic counter) in a global `std::unordered_map`. Multiple concurrent trackers
coexist without collision. Session destruction atomically releases all memory,
drops all hooks, and frees all compiled reduction handles.

---

## 4. Lifecycle Diagram — Hook Callback Hot Path

```
HOOK FIRES (C++ callback in callback.cpp)
│
├── Tensor on GPU/CPU; owner: PyTorch autograd graph
│
├── EARLY EXIT  (capture_policy.cpp, atomic, zero overhead if skipped)
│   ├── EVERY    → proceed
│   ├── SAMPLE_N → batch_counter % N == 0
│   └── MAX_K    → stop after K batches; counter resets on clear()
│
├── LOAD ACCUMULATOR STATE (under mutex, brief)
│   Read ActivationAccumulator.last() → undefined on first call
│
├── REDUCTION  (stateful: acc, tensor → new_acc)
│   ├── Per-layer compiled fn → fn(acc, tensor) → new_acc
│   ├── Global fallback       → fn(acc, tensor) → new_acc
│   └── Neither               → identity: store full tensor
│
├── DETACH  (unconditional — sever autograd edges)
│   NoGradGuard wraps entire callback
│
├── CAPTURE MODE — copy-on-detach
│   ├── REFERENCE (default) → no clone (shares tensor storage, fastest)
│   └── SNAPSHOT              → .clone() after detach (independent copy)
│
├── STORAGE POLICY — device placement
│   ├── GPU                   → stays on original device
│   ├── CPU (unpinned)        → blocking .to(kCPU)
│   ├── CPU (pinned)          → pinned alloc + async copy
│   ├── AUTO                  → heuristic: < threshold → CPU, >= threshold → GPU
│   └── DISK                  → CPU + serialise
│
├── ACCUMULATE  (under mutex — minimal scope)
│   ├── Stateful reduction    → clear + append(new_acc)
│   ├── STORE_ALL             → append to vector
│   ├── STREAMING             → append to vector
│   └── FINAL_ONLY            → clear + append (overwrite)
│
├── DISK path  (separate branch when StoragePolicy::DISK)
│   Stream tensor to <session_dir>/<layer>/<batch_idx>.dat
│
└── No in-memory accumulation — tensor lives only on disk.

READBACK  (Python .activations property)
│  Under session mutex: materialise fresh Python list per layer.
│  Each tensor shares TensorImpl with C++ vector entry — zero-copy.
→ dict[str, List[Tensor]]

DESTROY  (Python .remove() or context manager exit)
│  Hook handles dropped → modules unhooked → vectors cleared →
│  SessionState freed → reduction handles destroyed.
```

---

## 5. Four Policy Knobs

### 5.1 StoragePolicy — Where Tensor Data Lives

| Policy | Behaviour |
|--------|-----------|
| **AUTO** (default) | Heuristic: < 1 MiB → CPU, >= 1 MiB → GPU. Configurable via `auto_cpu_threshold_bytes`. Pinned-memory modifier enables async DMA for CPU-bound transfers. |
| **CPU** | `.to(kCPU)` in hook callback. Blocks until complete (or async if pinned). |
| **GPU** | `.detach()` only; stays on original device. |
| **DISK** | Streams tensors directly to `.dat` files on disk. Bypasses RAM. |

### 5.2 ReductionPolicy — What Gets Kept vs Reduced

| Policy | Behaviour |
|--------|-----------|
| **STORE_ALL** (default) | Full tensor per batch appended to vector. |
| **STREAMING** | Per-batch reduction. Vector holds output of each batch's reduction. |
| **FINAL_ONLY** | At most one entry per layer — last batch overwrites previous. |

When a stateful reduction is registered, the accumulator holds exactly one
tensor regardless of policy — the reduction maintains its own running state.

### 5.3 CapturePolicy — When and How Often

| Policy | Behaviour |
|--------|-----------|
| **EVERY** (default) | Every forward fires hooks. |
| **SAMPLE_N** | Captures every Nth forward. Atomic check, zero overhead for skipped batches. |
| **MAX_K** | Captures K batches per layer then silently stops. Safety rail against OOM. |

### 5.4 CaptureMode — Reference vs Snapshot

`CaptureMode` controls whether captured tensors share storage with the autograd
graph or are independently cloned.  It applies **after** any registered reduction
and **before** the storage policy placement step.

| Mode | Behaviour | C++ Implementation |
|------|-----------|-------------------|
| **REFERENCE** (default) | `detach()` only — shares `TensorImpl` storage | Identity — fastest path |
| **SNAPSHOT** | `detach()` + `clone()` — independent copy | `result = result.clone()` after detach |

Use `SNAPSHOT` when captured tensors may be mutated after tracking (e.g., in
protection-loss workflows or long-running loops where autograd tensors might
be overwritten by subsequent operations).

```python
tracker = activationscope.ActivationScope(
    capture_mode=activationscope.CaptureMode.SNAPSHOT,
)
```

The native C++ backend executes the clone in the hook callback (in `callback.cpp`)
with zero Python overhead.  Both Python-side (`_naive.py`) and C++-native
(`tracker.py`) trackers support `CaptureMode`.

### 5.5 Recommended Combinations

| Scenario | Storage | Reduction | Capture | CaptureMode |
|----------|---------|-----------|---------|-------------|
| Standard training loop | AUTO | STORE_ALL | EVERY | REFERENCE |
| Long unattended run | AUTO + pinned | STORE_ALL | MAX_K(50) | REFERENCE |
| Running statistics (max) | GPU | STREAMING | EVERY | REFERENCE |
| Debug last-forward shapes | CPU | FINAL_ONLY | EVERY | REFERENCE |
| Large model, constrained memory | GPU | STREAMING | MAX_K(N) | REFERENCE |
| Post‑processing that mutates activations | CPU | STORE_ALL | EVERY | SNAPSHOT |

---

## 6. Hook Registration and Layer Selection

Glob-based layer selection via `fnmatch` patterns happens exactly once at attach
time. The resulting layer set is locked and exclusive for the session lifetime.

```python
tracker.track(model, layers=["transformer.layers.*.attn", "conv1"])
tracker.track(model, include=["*.attn*", "*.fc2"], exclude=["*.bias"])
```

Capture direction (`"input"`, `"output"`, `"both"`) is a first-class parameter.
When `"both"`, an input hook AND output hook are registered on the same module.
Keys are disambiguated with `.input` / `.output` suffixes.

---

## 7. Parameter Snapshotting

For protection-loss workflows, `capture_parameters()` snapshots model parameters
as detached CPU clones:

```python
baseline = tracker.capture_parameters(model, layers=["encoder.*", "decoder.*"])
# Returns dict[str, Tensor] — Python-side, independent of C++ session
```

---

## 8. Thread Safety

PyTorch's eager-mode dispatcher may fire hooks from multiple threads under
data-parallelism. A single `std::mutex` on `SessionState` guards `accum_data`.
Lock hold time is minimised: policy checks use lock-free atomics, reduction
dispatch runs outside the lock, and only the final accumulate step acquires it.

Readback materialises a fresh Python list topology under mutex in O(n) over
pointer-sized TensorImpl references — not O(data_size). Concurrent hook appends
cannot invalidate Python-side iterators.

---

## 9. Key Libraries and APIs

| Component | Library / API | Purpose |
|-----------|---------------|---------|
| Extension build | `torch.utils.cpp_extension` | Compile with libtorch linkage |
| Python↔C++ glue | pybind11 | TensorImpl sharing for zero-copy readback |
| Native hooks | libtorch hook APIs | Hook without Python callable; pure C++ lambda |
| Reductions | `torch.compile()` | Compile callables for native dispatch |
| Thread safety | `std::mutex` on SessionState | Guards accum_data during concurrent hooks |
| Atomic counters | `std::atomic<int64_t>` | Lock-free batch counting for capture policies |
| Tensor ops | ATen | `detach`, `to(device)`, `copy_` for pinned DMA |
