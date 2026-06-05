# ActivationScope — Design Document

A PyTorch activation tracking library built on a native C++ backend that
owns all tensor memory. Hooks are registered directly via libtorch module APIs; no Python
callable lives on the forward hot path. Three policy knobs (storage, reduction, capture)
govern every aspect of memory and compute behavior. A zero-copy readback contract ensures
Python views C++ storage directly — never duplicating activations at boundary crossing.

---

## 1. Project Overview

ActivationScope intercepts the forward-pass activations of named submodules in a PyTorch
model and makes them available after execution — without holding live references into the
autograd graph, without Python dispatch per hook fire, and without unbounded memory growth
the user has no control over. Beyond activation capture, it provides parameter-snapshotting
utilities for baseline tracking patterns common to protection-loss workflows (e.g., InTAct).

The library consists of:

- A **C++ backend** compiled via `torch.utils.cpp_extension` that owns all tensor storage,
  manages session-scoped state keyed by a unique identifier, and executes native libtorch
  hooks whose callbacks never touch Python. Memory is held in `std::vector<torch::Tensor>`
  structures inside a single source-of-truth struct per session.

- A **thin Python layer** (`activationscope.tracker.ActivationScope`) that handles hook
  attachment, zero-copy readback of accumulated activations, lifecycle management (context
  manager, explicit teardown), and policy configuration. Python never materializes a second
  copy of stored activations during readback.

- **Three policy knobs** set at construction time independently controlling *where* tensor
  data lives (`StoragePolicy`), *what* gets kept versus reduced (`ReductionPolicy`), and
  *when* capture fires (`CapturePolicy`). These combine multiplicatively, giving the user
  precise leverage over the memory/compute tradeoff.

The core insight:

> The dominant costs in activation tracking are GPU→CPU transfers, raw activation volume,
> synchronization points, and storing tensors you don't actually need — not map lookups,
> vector headers, or hook dispatch overhead. Optimizing storage layout before optimizing
> what gets stored misplaces effort. The three policy knobs address real costs directly,
> while zero-copy readback eliminates the historical "tensor cat memory spike" failure mode
> by making dual-copy an architectural impossibility.

---

## 2. Core Architectural Decisions

### 2.1 Zero-Copy Python↔C++ Boundary — A Foundational Guarantee

**Python does NOT copy data from C++. It views C++ tensors directly.** pybind11 provides
transparent type conversion for `torch::Tensor` — the Python `torch.Tensor` object shares
the underlying `TensorImpl` with the C++ `torch::Tensor`. No allocation crosses the
boundary during readback.

This is not an implementation detail; it reshapes the entire memory ownership model:

- `.activations` returns a `dict[str, List[Tensor]]` materialized as a **fresh Python list**
  on each access. Each list entry shares the underlying `TensorImpl` with a C++ vector entry,
  but the Python list topology itself is independent — it does NOT expose the mutable
  `std::vector<torch::Tensor>` directly. This prevents segfaults from concurrent `std::vector`
  reallocation while Python iterates (a pybind11 dynamic wrapper around a live C++ vector would
  invalidate iterators on `append()`-triggered realloc).

- **Read-only views:** Returned tensors carry `.set_requires_grad_(False)` and are documented
  as immutable. In-place mutation (`acts["layer"][0].mul_(2)`) silently corrupts the C++ session
  storage across all future readbacks. Users who need writable data must call `.clone()` on the
  returned tensor. A runtime warning is raised if an in-place op is attempted.

- **No `torch::cat()` materialization at readback time.** Under the prior design, readback
  would concatenate per-layer vectors, temporarily doubling memory (originals alongside
  concatenated result). Zero-copy readback makes this an architectural IMPOSSIBILITY: two
  copies of stored activations can never coexist within library code.

```python
# .activations returns zero-copy refs into C++ storage (fresh Python list each time)
acts = tracker.activations              # dict[str, List[Tensor]] — shares TensorImpl w/ C++
batch_3 = acts["transformer.layers.0.attn"][3]  # zero overhead index access

# Returned tensors are READ-ONLY views. To mutate safely:
writable = batch_3.clone()
writable.mul_(2)                       # safe — does not touch C++ session storage

# If concatenated form needed, user creates and owns on-demand:
concatenated = torch.cat(acts["layer_name"], dim=0)  # exists briefly, user-managed lifetime
del concatenated                                  # free it; C++ originals intact + unaffected
```

Three consequences: (1) peak memory during readback equals stored activations, not doubled;
(2) readback latency stays O(n) per layer (list materialization) but the list is thread-safe;
(3) the library can never accidentally retain a second copy that outlives user intent.

### 2.2 Native Libtorch Hooks Over Python Wrappers

Hooks are registered directly in C++ through libtorch's module hook API:

```
Python attach() → pybind11 wrapper passes {module_ptr, layer_name} to C++
                → HookRegistration calls libtorch module method
                 → Hook callback is pure C++ lambda (core.cpp)
```

No `pybind11::cpp_function` functor traverses the boundary during the forward pass. The
hook lambda lives entirely in C++. Eliminates per-forward Python dispatch overhead (~3-8 μs
GIL + frame creation). For models with hundreds to thousands of hooked layers, this sums
to measurable stalls. Native hooks also eliminate the risk of accidental autograd-graph
retention through Python closures.

### 2.3 User-Provided Reductions, No Hardcoded Statistics in C++

C++ contains zero logic for mean, max, min, or any other aggregate statistic. The only
"default" reduction is identity: store the full tensor as-is.

When a user calls `register_reduction(fn)`:
1. Callable compiled via `torch.compile()` (preferred).
2. A warm-up forward with a synthetic dummy tensor of inferred/declared shape+dtype executes
   immediately during registration, so the first real batch does not experience cold-compile latency.
3. Compiled graph handle transferred to C++ and stored in `LayerHookConfig`.
4. Hook callbacks dispatch into the compiled graph via `execute_compiled()` — pure libtorch,
   zero Python overhead.

Convenience helpers (`for_max`, `for_mean`) are pure Python classmethods — not special-cased
in C++. This keeps reductions open to any `Callable[[Tensor], Tensor]` and eliminates enum-
driven switch statements in the hot path.

### 2.4 Per-Layer Hook Registration With Global Fallback

Users register reductions per-layer using glob patterns; unmatched layers fall back to a
session-level default:

```python
tracker.register_reduction(lambda x: torch.amax(x, dim=0), layers=["layers.0.*"])
tracker.register_reduction(lambda x: x.sum(-1), layers=None)  # global fallback
```

`LayerHookConfig` stores per-layer compiled handles; the session maintains a global default.
Dispatch resolves: check per-layer config first, then session default, then identity. This
allows fine-grained tuning without requiring every layer to be explicitly configured.

### 2.5 Session-Scoped State Isolation

Each `ActivationScope` instance owns a C++ session keyed by a unique `uint64_t` ID (atomic
counter). The global registry is an `std::unordered_map<uint64_t, SessionState*>`. Multiple
concurrent trackers coexist without collision via closure-captured pointers. Enables nested
tracking and multi-model experiments. Session destruction atomically releases all memory,
drops all hooks, and frees all compiled reduction handles.

---

## 3. File / Class Structure

```
csrc/
├── bindings.cpp          PYBIND11_MODULE ONLY — thin wrappers passing through to session/
│                          hook/core APIs. Zero logic; parameter translation only.
├── session.cpp + .hpp    Session lifecycle: create(), destroy(), readback(), clear().
│                          Manages global registry. ActivationScopeSession class with statics.
├── hook_register.cpp     Native libtorch hook registration on nn::Module objects. Creates
│   + .hpp               C++ lambda callbacks bound to session state via captured pointer.
├── core.cpp              THE HOT PATH: early-exit policy check, reduction dispatch, detach,
│   + .hpp               storage policy transfer, mutex-protected accumulation. Perf-critical.
├── capture_policy.cpp    CapturePolicy enforcement: sample_every stride, max_batches cap,
│   + .hpp               batch counters. Thread-safe atomic incrementors. should_capture().
├── accumulators.hpp      ActivationAccumulator, streaming reducer types. Templates and
│                         inline-only — no out-of-line definitions.
├── datastructures.hpp    All shared structs: SessionState, LayerHookConfig, CaptureDir enum,
│                         policy enums. No implementation bodies — pure declarations/aliases.
└── compiled_fn.hpp       CompiledFnHandle type, execute_compiled() dispatcher. Header-only.
```

Python side:

```
activationscope/
├── __init__.py           Public API re-exports (ActivationScope, StoragePolicy, etc.)
├── tracker.py            ActivationScope class: track()/attach()/remove()/clear(),
│                          .activations property (zero-copy), register_reduction() with per-layer
│                          patterns, capture_parameters(). Glob pattern matching here at attach time.
└── _C.pyi                Type stubs for compiled extension module.
```

### 3.1 Key C++ Types (Conceptual)

**`SessionState`** — Single source-of-truth per tracker instance:
- `unordered_map<string, LayerHookConfig>` — per-layer policy overrides, batch counters,
  reduction handles, capture direction.
- `unordered_map<string, ActivationAccumulator>` — accumulated tensors keyed by layer name
  (with `.input` / `.output` suffix when capturing both).
- Three policy enum copies from Python construction.
- Single `std::mutex` guarding accum_data map access. Per-layer hook handles for safe teardown.
- Destruction = full teardown: hooks dropped, vectors cleared, reduction handles destroyed.

**`LayerHookConfig`** — Per-layer config stored in SessionState:
- Capture direction (`CaptureDir` enum; immutable after attach). Enables dual-hook on one module.
- Storage policy override (may differ from session default per layer).
- Atomic batch counter for SAMPLE_N / MAX_K enforcement.
- Optional compiled reduction handle (`CompiledFnHandle*`). Null → identity or global fallback.

**`ActivationAccumulator`** — Thin wrapper around `std::vector<torch::Tensor>`:
- `append(Tensor)` under session mutex; `clear()` releases all storage; readback materializes
  a fresh Python list (thread-safe snapshot) from the vector — pybind11 wraps each `torch::Tensor`
  so Python shares TensorImpl refs without exposing the mutable C++ vector directly.

**`CompiledFnHandle`** — Opaque wrapper around compiled graph callable:
- `execute(Tensor) → Tensor` via `execute_compiled()`. Destruction frees cached compilation state.

---

## 4. Memory Data Flow and Ownership — Zero-Copy + Thread-Safe Snapshot Model

Complete lifecycle of a captured activation tensor, ownership marked at every step. The critical
departure from prior design: **readback returns read-only, zero-copy views into C++ storage.**
pybind11 transparently wraps `torch::Tensor` objects — both Python and C++ reference the same
`TensorImpl`. A fresh Python `list` object is materialized under mutex on each access, so the
list topology cannot be invalidated by concurrent `std::vector` reallocation.

### 4.1 Lifecycle Diagram

```
HOOK FIRES (C++ callback in core.cpp)
│
├── Tensor on GPU/CPU; owner: PyTorch autograd graph, connected to computation graph
│
┌── EARLY EXIT CHECK  (capture_policy.cpp, atomic, zero overhead if skipped)
│  ├── EVERY    → unconditionally proceed
│  ├── SAMPLE_N → batch_counter % N == 0 (atomic, outside mutex)
│  └── MAX_K    → stop after K batches; counter resets on clear()
│      Skipped: return immediately. No allocations, no locks, no Python.
│
├── REDUCTION STEP  (core.cpp — optional, per-layer → global fallback → identity)
│  ├── Per-layer compiled handle exists → execute_compiled(config.reduce_fn, tensor)
│  ├── No per-layer reduction           → session-level global default (if registered)
│  └── Neither                          → identity: store full tensor as-is
│
├── DETACH  (unconditional — first mutation after reduction)
│   .detach() severs autograd edges. Owner → C++ session (detached storage).
│   NoGradGuard wraps entire callback for belt-and-suspenders safety.
│
├── STORAGE POLICY TRANSFER  (core.cpp — device placement, final for this tensor)
│  ├── CPU (pinned = false) → .to(kCPU) blocks here, GPU→CPU via PCIe
│  ├── CPU (pinned = true) → .pin_memory().to(kCPU, non_blocking=True) async DMA launch
│  ├── GPU                   → stays on original device; transfer deferred to readback
│  ├── AUTO (pinned = false) → < threshold → CPU (blocking), ≥ threshold → GPU
│  └── AUTO (pinned = true)  → < threshold → pinned CPU (async), ≥ threshold → GPU
│
├── ACCUMULATE  (under mutex — minimal scope: map lookup + vector push_back)
│   ActivationAccumulator.append(std::move(stored_tensor)) in accum_data[layer_key]
│   Owner: C++ std::vector<torch::Tensor> inside SessionState


READBACK (Python .activations property — ZERO COPY + THREAD-SAFE SNAPSHOT)
│  Under session mutex: for each layer, materialize a fresh Python list whose elements are
  zero-copy TensorImpl shares of the C++ vector entries. The LIST TOPOLOGY is independent of
  the C++ vector (no live pybind11 dynamic wrapper), so concurrent append + std::vector realloc
  cannot invalidate Python iterators. Tensors themselves remain read-only views into C++ storage.
→ dict[str, List[Tensor]] — fresh list topology, shared TensorImpl data, read-only tensors.


DESTROY (Python .remove() or context manager exit)
│
Hook handles dropped → modules unhooked → vectors cleared → SessionState freed from
registry (unique_ptr) → reduction handles destroyed. All memory accounted for.
```

### 4.2 Ownership Table

| Stage | Memory Owner | Location | Notes |
|-------|-------------|----------|-------|
| Hook fires | PyTorch autograd graph | GPU or CPU | Raw activation, connected to computation graph |
| After `.detach()` | C++ session | Same device | Graph edge severed; no `backward()` through captured tensor |
| After storage policy transfer | C++ session | CPU or GPU per policy | Owned by `vector<Tensor>` inside `SessionState` |
| **Readback returns dict/lists** | **C++ (SHARED)** | Same as above | **Fresh Python list per access (thread-safe, independent topology). Tensor data remains zero-copy — pybind11 wraps `torch::Tensor` sharing the same `TensorImpl`. No cat() of tensor data.** |
| User calls explicit `torch.cat()` | Python caller | Caller chooses | User owns allocation and lifetime. Library never holds result alongside originals. |
| Session destroyed | — | freed | All storage released; unique_ptr to SessionState deallocated |

> Readback returns views into C++ storage, NOT copies. pybind11 transparently wraps
> `torch::Tensor` objects held by the C++ vector. Two copies of activations can never be
> materialized by library code at readback. The "tensor cat memory spike" risk is
> **eliminated by design**, not merely mitigated.

### 4.3 Memory Footprint

Dominant cost = size of captured activations (can reach tens of GB for large transformers).
Three policies interact multiplicatively: `StoragePolicy.CPU` trades VRAM for RAM + PCIe;
`ReductionPolicy.STREAMING` reduces memory from O(batches × features) to O(features);
`CapturePolicy.MAX_K` caps total captures. No hard limit imposed by the library — user
controls resources through policy selection and per-layer overrides.

---

## 5. Three Policy Knobs

### 5.1 StoragePolicy — Where Tensor Data Lives

| Policy | Behavior | Tradeoffs |
|--------|----------|-----------|
| **AUTO** (default) | Heuristic: <1 MB → CPU, ≥1 MB → GPU. Threshold is configurable via `auto_cpu_threshold_bytes` (default 1 MiB). Balances VRAM vs PCIe saturation. The CPU leg blocks by default; set session-level `use_pinned=True` to switch it to non-blocking async DMA. | Balanced out-of-the-box. Tune threshold for your hardware: lower values on NVLink GPUs where tiny transfers are cheap, higher on desktop PCIe Gen 3. Benchmark before relying on heuristic. |
| **CPU** | `.to(kCPU)` in hook callback. Blocks until transfer completes. Prevents GPU VRAM pressure. Session-level `use_pinned=True` enables non-blocking pinned transfer here too. | Every batch stalls forward by ≥1 PCIe transfer when unpinned. Cumulative blocking measurable for large layer counts. Use when off-device capture is needed immediately. |
| **GPU** | `.detach()` only; stays on original device. Transfer deferred to readback. Keeps forward pass free of PCIe sync. Essential when activation volume exceeds PCIe bandwidth. | GPU VRAM pressure grows with captures. Combine with MAX_K or STREAMING. Ideal for iterative: forward → read subset → clear → repeat. |

> **Pinned-memory modifier:** Both `AUTO` and `CPU` accept a session-level pinned (page-locked)
> host flag. When enabled, any GPU→CPU transfer uses `.pin_memory().to(kCPU, non_blocking=True)`
> instead of blocking `.to(kCPU)`, launching an async DMA and decoupling the forward pass from
> transfer completion. For `AUTO` this applies to every activation the heuristic routes to CPU.
> Tradeoff: page-locked RAM is scarce (capped by OS limits on many systems) and cannot be
> swapped; oversubscribing triggers OOM harder than standard CPU memory.

### 5.2 ReductionPolicy — What Gets Kept vs Reduced

| Policy | Memory | Behavior |
|--------|--------|----------|
| **STORE_ALL** (default) | O(batches × features) | Full tensor per batch appended to C++ vector. Readback returns fresh snapshot list (thread-safe, independent topology) — zero-copy TensorImpl data, no `cat()`. Complete fidelity for PCA, attention inspection, distance metrics. |
| **STREAMING** | O(features) per layer | Per-batch reduction output replaces/accumulates in place. After 10K forwards: single reduced tensor per layer. Requires `register_reduction()` — no default streaming exists. |
| **FINAL_ONLY** | O(features) | Last-batch activation overwrites previous. Vector holds exactly one element after first forward. Minimal overhead debugging. |

### 5.3 CapturePolicy — When and How Often

| Policy | Behavior | Config |
|--------|----------|--------|
| **EVERY** (default) | Every forward fires hooks for all tracked layers | None |
| **SAMPLE_N** | Captures on every Nth forward per layer. Atomic check, no mutex; skipped returns immediately, zero overhead. | `sample_every=N` (default 1 = EVERY). Long runs with periodic snapshots. Warning: can alias with data patterns. |
| **MAX_K** | Captures exactly K batches per layer, then silently bails at early exit while hooks remain registered. Counter resets on clear(). | `max_batches=K`. Safety rail against OOM in unattended jobs. |

### 5.4 Recommended Policy Combinations

| Scenario | Storage | Reduction | Capture |
|----------|---------|-----------|---------|
| Standard training loop tracking | AUTO (default) | STORE_ALL | EVERY |
| Long unattended run | AUTO + pinned | STORE_ALL | MAX_K(50) |
| Streaming online statistics | GPU or AUTO + pinned | STREAMING | EVERY |
| Debug last-forward shapes | CPU | FINAL_ONLY | EVERY |
| Memory-constrained large model | GPU | STREAMING | MAX_K(N) |

---

## 6. Hook Registration and Layer Selection

### 6.1 Glob-Based Layer Pattern Matching (Exclusive, Locked at Attach)

Users select modules via fnmatch-style patterns. **Matching happens exactly once at attach
time and the resulting layer set is then locked and exclusive.** The tracker collects activations
ONLY from these explicitly selected layers — no layer gets added dynamically after creation,
and no unselected layer contributes data even if its output flows through a tracked module.

Python enumerates `model.named_modules()`, applies filters, passes selected `(name, module_ptr)`
pairs to C++. No re-evaluation occurs during forward pass or at any later point.

```python
tracker.track(model, layers=["transformer.layers.*.attn", "conv1"])
# Or via include/exclude:
tracker.track(model, include=["*.attn*", "*.fc2"], exclude=["*.bias"])
```

Flow: (1) `named_modules()` enumeration; (2) include filter via `fnmatch` — module included if name matches ≥1 pattern; (3) exclude filter removes matches; (4) final candidate set is **locked** and forwarded to C++ for native hook registration. After this step, the tracked layer roster is immutable for the session lifetime. Adding or removing layers requires creating a new tracker session.

### 6.2 Per-Layer Reduction With Global Fallback

```python
tracker.register_reduction(lambda x: torch.amax(x, dim=0), layers=["layers.0.*"])
tracker.register_reduction(lambda x: torch.mean(x), layers=["layers.1.*", "layers.2.*"])
tracker.register_reduction(lambda x: x.sum(-1), layers=None)  # global default for unmatched
```

Resolution order: per-layer config → session global default → identity (store full tensor).

### 6.3 Input / Output / Both Capture Direction

First-class feature via `capture` parameter on `track()`/`attach()`:

```python
tracker.track(model, layers=["*.attn"], capture="output")    # only module outputs
tracker.track(model, layers=["*.attn"], capture="both")      # both inputs and outputs
```

When `"both"`, an input hook AND output hook register on the same module. Keys disambiguate
with `.input` / `.output` suffixes (`"transformer.layers.0.attn.input"`). Immutable after
attach — requires deregistration to change. This invariant simplifies C++ lifecycle: hook
handles created once, live for full session duration.

---

## 7. Parameter Snapshotting Utility

For InTAct-style protection-loss workflows, capture baseline parameters alongside activations:

```python
baseline_params = tracker.capture_parameters(model, layers=["encoder.*", "decoder.*"])
# Returns dict[str, Tensor] — Python-side, detached, CPU by default
```

Implemented primarily in Python via `model.named_parameters()` iteration. Each matched
parameter gets `.detach().cpu().clone()`. Returned as plain dict — fully independent of C++
session. Optional co-located mode stores snapshots within the C++ session alongside activation
data for unified lifecycle: same creation and destruction boundary.

Usage pattern: snapshot baseline params → track activations during modified execution → compute
deviation metrics between current and baseline parameters for drift-based protection loss.

---

## 8. Key Libraries and APIs Used

| Component | Library / API | Purpose |
|-----------|---------------|---------|
| Extension build | `setuptools` + `torch.utils.cpp_extension` | Compile with CUDA visibility, libtorch linkage |
| Python↔C++ glue | pybind11 | **TensorImpl sharing for zero-copy readback**; session IDs, policy enums, compiled handles |
| Native hooks | libtorch hook APIs (`register_forward_pre_hook`, `register_forward_hook`) | Hook without Python callable; callback is pure C++ lambda in core.cpp |
| Reduction compilation | `torch.compile()` | Compile callables into executable graph handles for native dispatch via `execute_compiled()` |
| Thread safety | `std::mutex` on SessionState | Protects `accum_data` during concurrent hooks from data-parallelism |
| Atomic counters | `std::atomic<int64_t>` | Lock-free batch counting in SAMPLE_N / MAX_K early-exit paths |
| Tensor ops | ATen | `detach`, `to(device)`. No `cat` at readback time. |

### 8.1 Hook Callback Hot Path — Conceptual Outline

```cpp
void hook_callback(SessionState& state, std::string layer_key, torch::Tensor tensor) {
  NoGradGuard _;
  if (!config.should_capture()) return;                          // early exit, zero cost

  Tensor result = config.reduce_fn
      ? execute_compiled(*config.reduce_fn, tensor)
      : (state.global_reduce_fn
           ? execute_compiled(*state.global_reduce_fn, tensor)
           : tensor);                                           // identity fallback

  Tensor stored = apply_storage_policy(result.detach(), config.storage);
  {
    lock_guard<mutex> lock(state.mutex);                         // minimal scope
    state.accum_data[layer_key].append(std::move(stored));
  }
}
```

- `NoGradGuard` wraps entire callback — no gradient leakage.
- Early exit: zero allocation, zero locks for skipped batches.
- Reduction before detach — compiled graph operates on connected tensor.
- Mutex scope limited to map lookup + vector push_back only.

### 8.2 Zero-Copy Readback — Conceptual Outline

```python
acts = tracker.activations           # dict[str, List[Tensor]] — zero-copy TensorImpl refs

single_batch = acts["layer.0"][5]    # read-only view
for batch_tensor in acts["encoder.block.2.mlp"]:  # iterate without copy
    do_analysis(batch_tensor.clone()) if needs_write else do_readonly(batch_tensor)

# In-place ops on returned tensors raise RuntimeError — use .clone() for writable copies:
writable = single_batch.clone()
writable.mul_(2)  # safe, does not mutate C++ session storage

# Concatenation is user-owned, on-demand, transient:
concat = torch.cat(acts["large_layer"], dim=0)  # user allocates
del concat                                      # user frees; C++ originals unaffected
```

---

## 9. Thread Safety and Concurrency Model

PyTorch's eager-mode dispatcher may fire hooks from multiple threads under data-parallelism.
Accumulation map is a shared resource accessed concurrently.

A single `std::mutex` on `SessionState` guards `accum_data`. Lock hold time minimized:
(1) policy checks use atomics — early-exit path executes outside lock; (2) reduction dispatch
runs outside lock on independent tensor storage; (3) only the final append acquires the mutex,
holding for microseconds during vector mutation.

Readback acquires the session mutex briefly to materialize a fresh Python list per layer from
the C++ vector snapshot. The lock hold time is O(n) over pointer-sized TensorImpl references
(microseconds), not O(data_size). Once the Python list is returned, it carries an independent
topology — concurrent hook appends that trigger `std::vector` reallocation cannot invalidate
Python-side iterators or pointers. "Blocking hooks during expensive concatenation" concern
disappears — there is no library-side concatenation. Data remains shared (zero-copy TensorImpl),
but list topology is thread-safe. Users who need a point-in-time data snapshot still clone
their slice explicitly: `[t.clone() for t in acts[layer_name]]`.

---

## Design Principles Summary

```
1  Zero-copy readback: Python views C++ tensor storage directly via TensorImpl sharing.
    A fresh Python list is materialized under mutex (thread-safe snapshot, independent of
    std::vector reallocation). Returned tensors are READ-ONLY — in-place ops raise RuntimeError;
    users call .clone() for writable copies. No intermediate cat(), no double memory.
    "Tensor cat spike" eliminated by architecture.

2  C++ owns memory: All activation storage, reduction handles, and accumulation vectors
    live in C++. Python sees read-only views of tensor data, never copies (unless user opts
    into explicit clone or concat).

3  Native hooks: libtorch's forward hook API called directly from C++. Callback entirely
     in core.cpp. No pybind11 round-trips per forward. Zero GIL contention on hot path.

4  Store full tensors by default: Identity reduction is the baseline. All statistics
    (mean, max, min) are user-provided callables — nothing hardcoded in C++.

5  Per-layer config with global floor: Fine-tune which reductions apply to which layers
    via glob patterns. Unmatched layers inherit a session-level default.

6  Dual capture on same layer: A module captures both inputs and outputs when "both".
    Immutable after attach. Keys disambiguate with .input / .output suffixes.

7  Compiled reduction backend: register_reduction(fn) accepts any callable, compiles for
    native C++ execution, then runs a warm-up forward immediately so the first real batch
    never pays cold-compile latency. Compilation errors surface at registration time.

8  Storage policy governs transfer timing: AUTO (default) balances via size heuristic with
     a configurable threshold (auto_cpu_threshold_bytes, default 1 MiB); CPU for safety-critical
     off-device capture; GPU avoids forward stalls. Pinned-memory modifier on AUTO/CPU enables
     non-blocking async DMA at the cost of page-locked RAM. No migration after initial placement.

9  Session scoping: Each tracker owns a keyed C++ session. Multiple concurrent instances
    don't collide. Deterministic destruction via context manager, explicit remove(), finalizer.

10 Parameter snapshotting for protection workflows: capture_parameters() enables baseline
    tracking alongside activation capture for drift-based loss computation patterns.
```
