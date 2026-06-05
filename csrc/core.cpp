/*
 * ActivationScope - Hook callback hot path.
 *
 * Called by native libtorch hooks for every matched module on every forward pass.
 * Zero Python overhead: everything runs in C++.
 *
 * Execution order inside the callback:
 *   1. NoGradGuard — belt-and-suspenders autograd isolation
 *   2. Early-exit capture policy (lock-free atomic) — return immediately if skipped
 *   3. Reduction dispatch  (per-layer → global fallback → identity)
 *   4. .detach() — sever autograd edges (owner → C++ session)
 *   5. Storage policy — device placement (GPU / CPU / AUTO heuristic, pinned option)
 *   6. Mutex-guarded accumulation into ActivationAccumulator vector
 */

#include "core.hpp"
#include "session.hpp"

namespace activationscope {

/* ── Storage policy: device placement logic ─────────────────────────── */

/// Apply the storage policy to decide where the tensor should live.
static torch::Tensor apply_storage_policy(torch::Tensor tensor,
                                          StoragePolicy policy,
                                          int64_t auto_threshold_bytes,
                                          bool use_pinned) {
    switch (policy) {
        case StoragePolicy::GPU:
            // Stay on original device — nothing to do after detach.
            return tensor;

        case StoragePolicy::CPU: {
            if (tensor.is_cuda()) {
                if (use_pinned) {
                    // Async DMA via pinned memory — non-blocking.
                    tensor = tensor.pin_memory();
                    tensor = tensor.to(torch::kCPU, /*non_blocking=*/true);
                } else {
                    // Blocking transfer to host memory.
                    tensor = tensor.to(torch::kCPU);
                }
            }
            return tensor;
        }

        case StoragePolicy::AUTO: {
            // Heuristic: small tensors → CPU, large tensors → GPU.
            int64_t numel_bytes = static_cast<int64_t>(tensor.numel()) *
                                 static_cast<int64_t>(tensor.element_size());
            if (numel_bytes < auto_threshold_bytes && tensor.is_cuda()) {
                // Small enough — move to CPU now and store on host memory.
                if (use_pinned) {
                    tensor = tensor.pin_memory();
                    tensor = tensor.to(torch::kCPU, /*non_blocking=*/true);
                } else {
                    tensor = tensor.to(torch::kCPU);
                }
            }
            // Large or already-CPU tensor — stays where it is.
            return tensor;
        }
    }
    return tensor;   // unreachable fallback
}


/* ------------------------------------------------------------------ */

/// HOT PATH: invoked by every native libtorch hook callback.
void hook_callback(SessionState* state, const std::string& layer_key,
                   torch::Tensor tensor) {
    torch::NoGradGuard no_grad;

    if (!state) return;   // Safety — session already destroyed.

    // ── (1) Look up per-layer config ────────────────────────────────
    auto cfg_it = state->layer_configs.find(layer_key);
    if (cfg_it == state->layer_configs.end()) {
        // Layer was removed after hook registered — bail silently.
        return;
    }

    LayerHookConfig& cfg = cfg_it->second;

    // ── (2) Early-exit: capture policy check (lock-free atomic) ─────
    if (!cfg.counter.should_capture()) {
        return;   // Skipped — zero allocations, zero locks.
    }

    // ── (3) Reduction dispatch (outside mutex for parallelism) ───────
    torch::Tensor result = tensor;  // identity default

    if (cfg.reduce_fn && *cfg.reduce_fn) {
        result = cfg.reduce_fn->execute(tensor);          // per-layer compiled fn
    } else if (state->global_reduce_fn && *state->global_reduce_fn) {
        result = state->global_reduce_fn->execute(tensor); // global fallback
    }
    // else: identity — store the full tensor as-is.

    // ── (4) Detach — sever autograd edges (first mutation) ──────────
    result = result.detach();

    // ── (5) Storage policy — device placement ───────────────────────
    StoragePolicy effective = cfg.effective_storage();
    torch::Tensor stored = apply_storage_policy(
        std::move(result),
        effective,
        state->auto_cpu_threshold_bytes,
        state->use_pinned
    );

    // ── (6) Accumulate under mutex — minimal scope ──────────────────
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        auto& accum = state->accum_data[layer_key];   // default-construct if absent

        switch (state->reduction) {
            case ReductionPolicy::STORE_ALL:
                accum.append(std::move(stored));
                break;

            case ReductionPolicy::STREAMING:
                /* For STREAMING we keep appends — the user's reduction fn already
                 * collapsed the tensor shape, so each append is a single-element tensor.
                 * The vector grows by one slim tensor per batch rather than a full activation. */
                accum.append(std::move(stored));
                break;

            case ReductionPolicy::FINAL_ONLY:
                /* Overwrite previous — always keep exactly one entry per layer key. */
                if (accum.size() > 0) {
                    accum.clear();   // hold on mutex, so this is safe
                }
                accum.append(std::move(stored));
                break;
        }
    }
}

} // namespace activationscope
