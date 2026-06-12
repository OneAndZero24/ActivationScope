/*
 * ActivationScope — Hook callback hot path.
 *
 * Called by registration thunks.  Avoids overhead:
 *   - NoGradGuard isolation
 *   - Lock-free capture policy early-exit
 *   - TorchScript reduction via Reduction::run()
 *   - .detach() + optional .clone()
 *   - Storage policy device placement
 *   - Accumulate under mutex
 *
 * All state (config, reduction, accumulator) is captured in the
 * hook closure — no dict lookups, no string keys on the hot path.
 */
#include "callback.hpp"
#include "session.hpp"
#include "utils.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>

namespace activationscope {

/* ── Storage policy: device placement ──────────────────────── */
static torch::Tensor apply_storage_policy(torch::Tensor tensor,
                                          StoragePolicy policy,
                                          int64_t auto_threshold_bytes,
                                          bool use_pinned) {
    switch (policy) {
        case StoragePolicy::GPU:
            return tensor;
        case StoragePolicy::CPU:
        case StoragePolicy::DISK:
            if (tensor.is_cuda()) {
                if (use_pinned) {
                    auto pinned = torch::empty(
                        tensor.sizes(),
                        tensor.options().device(torch::kCPU).pinned_memory(true));
                    pinned.copy_(tensor, true);
                    tensor = pinned;
                } else {
                    tensor = tensor.to(torch::kCPU);
                }
            }
            return tensor;
        case StoragePolicy::AUTO:
            if (tensor.is_cuda()) {
                int64_t bytes = static_cast<int64_t>(tensor.numel()) *
                                static_cast<int64_t>(tensor.element_size());
                if (bytes < auto_threshold_bytes) {
                    if (use_pinned) {
                        auto pinned = torch::empty(
                            tensor.sizes(),
                            tensor.options().device(torch::kCPU).pinned_memory(true));
                        pinned.copy_(tensor, true);
                        tensor = pinned;
                    } else {
                        tensor = tensor.to(torch::kCPU);
                    }
                }
            }
            return tensor;
    }
    return tensor;
}

/* ── HOT PATH ──────────────────────────────────────────────── */
void hook_callback(SessionState*              state,
                   LayerHookConfig*           cfg,
                   std::shared_ptr<LayerAccumulator> accum,
                   const std::string&         layer_key,
                   torch::Tensor              tensor) {
    torch::NoGradGuard no_grad;
    if (!state || !cfg) return;

    // 1) Early-exit: capture policy (lock-free atomic)
    if (!cfg->counter.should_capture()) return;

    // 2) Load running accumulator (brief mutex)
    torch::Tensor acc;
    {
        std::lock_guard<std::mutex> lock(accum->mtx);
        const torch::Tensor* last = accum->data.last();
        if (last) acc = *last;
    }

    // 3) Reduction — TorchScript via C++
    torch::Tensor result;
    if (cfg->reduction) {
        result = cfg->reduction->run(acc, tensor);
    } else {
        result = tensor; // identity
    }

    // 4) Detach + optional clone
    result = result.detach();
    if (state->capture_mode == CaptureMode::SNAPSHOT)
        result = result.clone();

    // 5) Storage policy
    StoragePolicy effective = cfg->effective_storage();
    if (effective == StoragePolicy::AUTO)
        effective = state->default_storage;
    torch::Tensor stored = apply_storage_policy(
        std::move(result), effective,
        state->auto_cpu_threshold_bytes, state->use_pinned);

    // 6) Accumulate — DISK path or in-memory
    if (effective == StoragePolicy::DISK && !state->session_dir.empty()) {
        torch::Tensor cpu_tensor = stored.to(torch::kCPU).contiguous();
        std::string safe_name = sanitize_layer_name(layer_key);
        std::string layer_dir = state->session_dir + "/" + safe_name;
        ensure_dir(layer_dir);

        int64_t batch_idx = cfg->disk_batch_idx.fetch_add(1, std::memory_order_relaxed);
        std::ostringstream fname;
        fname << layer_dir << "/" << std::setw(8) << std::setfill('0')
              << batch_idx << ".dat";

        std::ofstream ofs(fname.str(), std::ios::binary | std::ios::trunc);
        if (ofs.is_open()) {
            int64_t dtype = static_cast<int64_t>(cpu_tensor.scalar_type());
            int64_t ndim  = cpu_tensor.dim();
            ofs.write(reinterpret_cast<const char*>(&dtype), sizeof(int64_t));
            ofs.write(reinterpret_cast<const char*>(&ndim), sizeof(int64_t));
            for (int64_t i = 0; i < ndim; ++i) {
                int64_t dim = cpu_tensor.size(i);
                ofs.write(reinterpret_cast<const char*>(&dim), sizeof(int64_t));
            }
            ofs.write(reinterpret_cast<const char*>(cpu_tensor.data_ptr()),
                      cpu_tensor.numel() * cpu_tensor.element_size());
        }
        return;
    }

    // In-memory path
    {
        std::lock_guard<std::mutex> lock(accum->mtx);
        if (cfg->reduction) {
            // Stateful: single entry in accumulator (updated in-place)
            if (accum->data.size() > 0)
                accum->data.replace_last(std::move(stored));
            else
                accum->data.append(std::move(stored));
        } else {
            switch (state->reduction_policy) {
                case ReductionPolicy::STORE_ALL:
                case ReductionPolicy::STREAMING:
                    accum->data.append(std::move(stored));
                    break;
                case ReductionPolicy::FINAL_ONLY:
                    if (accum->data.size() > 0) accum->data.clear();
                    accum->data.append(std::move(stored));
                    break;
            }
        }
    }
}

} // namespace activationscope
