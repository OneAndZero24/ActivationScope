/*
 * ActivationScope - Shared data structures for session-scoped activation tracking.
 *
 * Pure declarations and type aliases. No implementation bodies.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <torch/extension.h>
#include <unordered_map>

namespace activationscope {

/* ── Enumerations ────────────────────────────────────────────────────── */

/// Where tensor data lives after capture.
enum class StoragePolicy : int32_t {
    AUTO = 0,   ///< Heuristic: < threshold → CPU (pinned if use_pinned), ≥ threshold → GPU
    CPU  = 1,   ///< Blocking transfer to host memory (or async if use_pinned)
    GPU  = 2,   ///< Stay on original device; defer transfer to readback
    DISK = 3,   ///< Stream tensors directly to disk; bypass in-memory accumulation
};

/// What gets kept vs reduced across batches.
enum class ReductionPolicy : int32_t {
    STORE_ALL   = 0, ///< Full tensor per batch appended to C++ vector
    STREAMING   = 1, ///< Per-batch reduction output replaces/accumulates in-place
    FINAL_ONLY  = 2, ///< Last-batch activation overwrites previous
};

/// When and how often hooks fire.
enum class CapturePolicy : int32_t {
    EVERY    = 0, ///< Every forward fires hooks
    SAMPLE_N = 1, ///< Captures on every Nth forward
    MAX_K    = 2, ///< Captures exactly K batches then silently bails
};

/// Which tensor to capture: module inputs, outputs, or both.
enum class CaptureDir : int32_t {
    INPUT  = 0,
    OUTPUT = 1,
    BOTH   = 2,
};

/// Whether to clone captured tensors (independent copy vs shared storage).
enum class CaptureMode : int32_t {
    REFERENCE = 0,  ///< detach() only — shares storage with autograd graph
    SNAPSHOT  = 1,  ///< detach() + clone() — completely independent copy
};

} // namespace activationscope
