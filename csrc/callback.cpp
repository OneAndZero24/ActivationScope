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
 *   6. For DISK: stream tensor directly to .pt file on disk, bypass RAM
 *      For others: mutex-guarded accumulation into ActivationAccumulator vector
 */

#include "callback.hpp"
#include "session.hpp"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>

namespace activationscope {

/* ── Helpers ─────────────────────────────────────────────────────── */

/// Sanitize a layer name into a filesystem-safe directory name.
/// Must match the version in session.cpp for consistent reading.
static std::string sanitize_layer_name(const std::string& raw) {
    std::string out;
    out.reserve(raw.size());
    for (char c : raw) {
        if (c == '/' || c == '\\' || c == ':' || c == '?' || c == '*') {
            out += '_';
        } else if (c == '.') {
            out += '_';
        } else {
            out += c;
        }
    }
    return out;
}

/// Create directory (and parents) if it doesn't exist.
/// Returns true on success or if already exists.
static bool ensure_dir(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
        return true;
    return (mkdir(path.c_str(), 0700) == 0);
}

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

        case StoragePolicy::CPU:
        case StoragePolicy::DISK: {
            // DISK needs tensors on CPU before serialization.
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
    if (effective == StoragePolicy::AUTO) {
        effective = state->default_storage;   // merge with session-level default
    }
    torch::Tensor stored = apply_storage_policy(
        std::move(result),
        effective,
        state->auto_cpu_threshold_bytes,
        state->use_pinned
    );

    // ── (6) Accumulate — DISK path or in-memory path ─────────────────
    if (effective == StoragePolicy::DISK && !state->session_dir.empty()) {
        // ── DISK mode: write tensor directly to .dat file, bypass RAM ──

        // Ensure tensor is on CPU and contiguous for serialization.
        torch::Tensor cpu_tensor = stored.to(torch::kCPU).contiguous();

        // Sanitize layer name for filesystem path.
        std::string safe_name = sanitize_layer_name(layer_key);

        // Build output path: <session_dir>/<safe_layer_name>/NNNNNN.dat
        std::string layer_dir = state->session_dir + "/" + safe_name;
        ensure_dir(layer_dir);

        // Get a monotonically-increasing batch index for this layer.
        int64_t batch_idx = cfg.disk_batch_idx.fetch_add(1, std::memory_order_relaxed);

        std::ostringstream fname;
        fname << layer_dir << "/" << std::setw(8) << std::setfill('0') << batch_idx << ".dat";
        std::string filepath = fname.str();

        // Raw binary format: [dtype: int64][ndim: int64][dim0..dimN: int64][raw data bytes]
        // Fast, GIL-free, no TorchScript overhead.
        std::ofstream ofs(filepath, std::ios::binary | std::ios::trunc);
        if (ofs.is_open()) {
            int64_t dtype = static_cast<int64_t>(cpu_tensor.scalar_type());
            int64_t ndim  = cpu_tensor.dim();
            ofs.write(reinterpret_cast<const char*>(&dtype), sizeof(int64_t));
            ofs.write(reinterpret_cast<const char*>(&ndim), sizeof(int64_t));
            for (int64_t i = 0; i < ndim; i++) {
                int64_t dim = cpu_tensor.size(i);
                ofs.write(reinterpret_cast<const char*>(&dim), sizeof(int64_t));
            }
            ofs.write(reinterpret_cast<const char*>(cpu_tensor.data_ptr()),
                      cpu_tensor.numel() * cpu_tensor.element_size());
            ofs.close();
        }

        // No in-memory accumulation — tensor data lives only on disk.
        return;
    }

    // ── In-memory path: accumulate under mutex — minimal scope ───────
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
