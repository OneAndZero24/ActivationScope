/*
 * ActivationScope - Session-scoped state management declarations.
 *
 * Each Python ActivationScope instance owns exactly one C++ SessionState, keyed
 * by an atomic uint64_t counter.  The global registry is a std::unordered_map
 * that maps session IDs to unique_ptr<SessionState>.  All public entry points
 * (create, destroy, readback, clear, hook registration) operate through this
 * header.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <torch/extension.h>
#include <unordered_map>
#include <vector>

#include "accumulator.hpp"
#include "capture_policy.hpp"
#include "compiled_fn.hpp"
#include "datastructures.hpp"

namespace torch {
namespace nn {
class Module;
}
} // namespace torch

namespace activationscope {

/* ── Per-layer hook configuration ───────────────────────────────────── */

struct LayerHookConfig {
  /// Capture direction (immutable after attach).
  CaptureDir capture_dir = CaptureDir::OUTPUT;

  /// Per-layer storage override; StoragePolicy::AUTO means "use session
  /// default".
  StoragePolicy storage_override = StoragePolicy::AUTO;

  /// Effective storage policy after merge with session-level defaults.
  StoragePolicy effective_storage() const;

  /// Atomic batch counter for SAMPLE_N / MAX_K enforcement.
  CaptureCounter counter;

  /// Per-layer compiled reduction handle (null → try global fallback).
  std::unique_ptr<CompiledFnHandle> reduce_fn = nullptr;

  /// Monotonically-increasing batch index for DISK storage mode.
  /// Incremented atomically inside hook_callback before each disk write.
  std::atomic<int64_t> disk_batch_idx{0};
};

/* ── Session state — single source of truth per tracker instance ─────── */

struct SessionState {
  // -- Policy knobs (set at creation, immutable) --------------------
  StoragePolicy default_storage = StoragePolicy::AUTO;
  ReductionPolicy reduction = ReductionPolicy::STORE_ALL;
  int64_t sample_every = 1;
  int64_t max_batches = 0; // 0 == unlimited

  // -- AUTO heuristic threshold (bytes) ----------------------------
  int64_t auto_cpu_threshold_bytes = 1048576; // 1 MiB default

  // -- Pinned-memory modifier for GPU→CPU transfers ----------------
  bool use_pinned = false;

  // -- Capture mode — detach-only vs detach+clone ------------------
  CaptureMode capture_mode = CaptureMode::REFERENCE;

  // -- Disk storage mode --------------------------------------------
  /// Root directory for per-layer .pt files when storage=DISK.
  /// Created in session_create, cleaned up in release().
  std::string session_dir;

  // -- Per-layer configuration -------------------------------------
  std::unordered_map<std::string, LayerHookConfig> layer_configs;

  // -- Accumulated tensor storage ----------------------------------
  std::unordered_map<std::string, ActivationAccumulator> accum_data;

  // -- Global default reduction handle (fallback for unmatched layers)
  std::unique_ptr<CompiledFnHandle> global_reduce_fn = nullptr;

  // -- Thread safety -----------------------------------------------
  std::mutex mutex; ///< Guards accum_data map access only.

  // -- Hook handles for teardown -----------------------------------
  // Each entry stores a raw pointer to the torch::nn::ModuleHook object
  // returned by libtorch's register_forward_hook / register_forward_pre_hook.
  // We store as std::shared_ptr because libtorch uses intrusive_ptr internally
  // and we need the hook to live as long as the session.
  using HookHandlePtr = void *; ///< Opaque pybind11-managed hook handle
  std::vector<std::pair<std::string, HookHandlePtr>> m_hook_handles;

  // -- Public factory / teardown ------------------------------------

  /// Look up session by ID (internal helper).
  static SessionState *get(uint64_t id);

  /// Release all storage, drop hooks, destroy reduction handles.
  void release();
};

/* ── Global registry API (exposed via bindings.cpp) ──────────────────── */

/// Create a new session and return its unique ID.
uint64_t session_create(StoragePolicy storage, ReductionPolicy reduction,
                        int64_t sample_every, int64_t max_batches,
                        int64_t auto_cpu_threshold_bytes, bool use_pinned,
                        const std::string &session_dir = "",
                        CaptureMode capture_mode = CaptureMode::REFERENCE);

/// Destroy the session (drops hooks, clears vectors).  No-op if ID invalid.
void session_destroy(uint64_t id);

/// Zero-copy readback: for each layer return a fresh vector<Tensor> list.
std::unordered_map<std::string, std::vector<torch::Tensor>>
session_readback(uint64_t id);

/// Clear all accumulated activations (hook stays active, counters reset).
void session_clear(uint64_t id);

/// Detach hooks from modules but *keep* the session alive for reuse.
/// Used by track() context manager exit (not by remove()).
void session_detach_hooks(uint64_t id);

/// Register native hooks on a module for the given layer key + capture
/// direction.
void session_register_hooks(uint64_t id, uintptr_t module_ptr,
                            const std::string &layer_key,
                            int32_t capture_dir_int);

/// Attach a per-layer compiled reduction handle (layer may use fnmatch
/// pattern).
void session_set_layer_reduction(uint64_t id, const std::string &layer_name,
                                 void *compiled_handle);

/// Set the global default compiled reduction for unmatched layers.
void session_set_global_reduction(uint64_t id, void *compiled_handle);

/// Read activations back from disk (storage=DISK mode only).
/// Returns a dict mapping each layer directory to a list of .pt file paths on
/// disk.
std::unordered_map<std::string, std::vector<std::string>>
session_readback_disk(uint64_t id);

} // namespace activationscope
