/*
 * ActivationScope — Session-scoped state management.
 *
 * Each Python ActivationScope instance owns one C++ SessionState, keyed
 * by an atomic uint64_t counter in a global registry.
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
#include "datastructures.hpp"
#include "reduction.hpp"

namespace activationscope {

/* ── Per-layer hook configuration ─────────────────────────── */
struct LayerHookConfig {
    CaptureDir   capture_dir      = CaptureDir::OUTPUT;
    StoragePolicy storage_override = StoragePolicy::AUTO;
    CaptureCounter counter;

    /// Reduction loaded from TorchScript .pt file (nullptr → identity).
    std::shared_ptr<Reduction> reduction;

    /// Monotonically-increasing batch index for DISK storage mode.
    std::atomic<int64_t> disk_batch_idx{0};

    StoragePolicy effective_storage() const {
        return storage_override != StoragePolicy::AUTO
                   ? storage_override
                   : StoragePolicy::AUTO;
    }
};

/* ── Session state — single source of truth per tracker ────── */
struct SessionState {
    StoragePolicy   default_storage         = StoragePolicy::AUTO;
    ReductionPolicy reduction_policy        = ReductionPolicy::STORE_ALL;
    int64_t         sample_every            = 1;
    int64_t         max_batches             = 0;
    int64_t         auto_cpu_threshold_bytes = 1048576; // 1 MiB
    bool            use_pinned              = false;
    CaptureMode     capture_mode            = CaptureMode::REFERENCE;
    std::string     session_dir;

    std::unordered_map<std::string, std::shared_ptr<LayerHookConfig>> layer_configs;
    std::unordered_map<std::string, std::shared_ptr<LayerAccumulator>> accum_data;
    std::unordered_map<std::string, std::vector<std::string>>      disk_paths;

    std::mutex mutex; // guards accum_data, layer_configs, disk_paths

    using HookHandlePtr = void*;
    std::vector<std::pair<std::string, HookHandlePtr>> m_hook_handles;

    static SessionState* get(uint64_t id);
    void release();
};

/* ── Global registry API ─────────────────────────────────── */

uint64_t session_create(StoragePolicy storage, ReductionPolicy reduction,
                        int64_t sample_every, int64_t max_batches,
                        int64_t auto_cpu_threshold_bytes, bool use_pinned,
                        const std::string& session_dir = "",
                        CaptureMode capture_mode = CaptureMode::REFERENCE);

void session_destroy(uint64_t id);

std::unordered_map<std::string, std::vector<torch::Tensor>>
session_readback(uint64_t id);

std::unordered_map<std::string, std::vector<std::string>>
session_readback_disk(uint64_t id);

void session_clear(uint64_t id);
void session_detach_hooks(uint64_t id);

/// Pre-initialise an accumulator for stateful reductions (e.g. SVD second pass).
/// Writes *tensor* as the sole entry in the per-layer accumulator so that the
/// reduction's first call sees an existing state instead of None.
void session_init_accumulator(uint64_t id, const std::string& layer_key,
                              torch::Tensor tensor);

/// Register native hooks on a module.  reduction_path is a .pt file
/// compiled via torch.jit.script; empty string → identity reduction.
void session_register_hooks(uint64_t id, uintptr_t module_ptr,
                            const std::string& layer_key,
                            int32_t capture_dir_int,
                            const std::string& reduction_path = "");

} // namespace activationscope
