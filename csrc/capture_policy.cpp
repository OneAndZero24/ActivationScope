/*
 * ActivationScope - CapturePolicy enforcement: should_capture() logic.
 *
 * Lock-free early-exit decisions using std::atomic counters.  No mutex, no
 * heap allocation on the hot path — skipped batches return immediately with
 * zero overhead.
 */

#include "capture_policy.hpp"

namespace activationscope {

/* ------------------------------------------------------------------ */

bool CaptureCounter::should_capture() {
    switch (policy) {
        case CapturePolicy::EVERY:
            // Unconditionally allow every batch.  Cheap, no counter tick needed.
            return true;

        case CapturePolicy::SAMPLE_N: {
            // Atomically increment; capture only when new_count % sample_every == 0.
            int64_t count = batch_count.fetch_add(1, std::memory_order_acq_rel) + 1;
            return (count % sample_every) == 0;
        }

        case CapturePolicy::MAX_K: {
            // Atomically increment; reject once we've already captured K batches.
            int64_t count = batch_count.fetch_add(1, std::memory_order_acq_rel) + 1;
            return (count <= max_batches);
        }
    }
    return false;   /* unreachable */
}

/* ------------------------------------------------------------------ */

} // namespace activationscope
