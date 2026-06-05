/*
 * ActivationScope - CapturePolicy enforcement declarations.
 *
 * Manages SAMPLE_N stride and MAX_K cap with lock-free atomics so that
 * early-exit decisions happen outside any mutex.
 */

#pragma once

#include <atomic>
#include "datastructures.hpp"

namespace activationscope {

/* ── Per-layer capture counter / config ─────────────────────────────── */

/**
 * Holds per-layer batch counting state and policy parameters used to decide
 * whether the next hook fire should produce an activation capture.
 *
 * All members are atomic so the early-exit path is lock-free.
 */
struct CaptureCounter {
    CapturePolicy policy;
    int64_t sample_every;   ///< N for SAMPLE_N (default 1)
    int64_t max_batches;    ///< K for MAX_K (default 0 = unlimited)

    /// Monotonically increasing batch counter; resets on clear().
    std::atomic<int64_t> batch_count{0};

    /// Increment internal counter and return whether this batch should be captured.
    bool should_capture();

    /// Reset counter — called by SessionState::clear().
    void reset() noexcept { batch_count.store(0, std::memory_order_relaxed); }
};

} // namespace activationscope
