/*
 * ActivationScope - Session lifecycle implementations.
 *
 * Manages the global registry of sessions, each keyed by an atomic uint64_t counter.
 * All public entry points (create, destroy, readback, clear, hook registration)
 * are implemented here.
 */

#include "session.hpp"
#include "hook_register.hpp"
#include <Python.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace activationscope {

/* ── Global registry storage ──────────────────────────────────────── */

static std::unordered_map<uint64_t, std::unique_ptr<SessionState>> global_registry;
static std::atomic<uint64_t> g_session_counter{0};
static std::mutex registry_mutex;

/* ── Compile-time helper: decide capture policy from session params ── */

static CapturePolicy infer_capture_policy(int64_t sample_every, int64_t max_batches) {
    if (max_batches > 0) return CapturePolicy::MAX_K;
    if (sample_every > 1) return CapturePolicy::SAMPLE_N;
    return CapturePolicy::EVERY;
}

/* ── SessionState members ─────────────────────────────────────────── */

SessionState* SessionState::get(uint64_t id) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto it = global_registry.find(id);
    if (it == global_registry.end()) return nullptr;
    return it->second.get();
}

void SessionState::release() {
    // 1) Drop all hook handles — decrement PyObject ref counts, then remove().
    {
        std::lock_guard<std::mutex> lock(registry_mutex);   // safety: hooks may access registry
        for (auto& [key, handle_ptr] : m_hook_handles) {
            if (!handle_ptr) continue;
            PyObject* handle = reinterpret_cast<PyObject*>(handle_ptr);
            // Call handle.remove() to detach the hook from the Python module.
            PyObject* method = PyObject_GetAttrString(handle, "remove");
            if (method && PyCallable_Check(method)) {
                PyObject_CallObject(method, nullptr);
                Py_DECREF(method);
            }
            Py_XDECREF(handle);
        }
    }

    // 2) Clear all accumulated tensor data.
    for (auto& [key, accum] : accum_data) {
        accum.clear();
    }

    // 3) layer_configs cleared — unique_ptrs auto-destroy CompiledFnHandle instances.
    //    unique_ptr<CompiledFnHandle> in each LayerHookConfig destructs properly.

    // 4) Global reduction handle reset (auto via unique_ptr).
    global_reduce_fn.reset();

    // 5) Clear handles vector.
    m_hook_handles.clear();

    // Final cleanup — maps themselves are destroyed when SessionState is deleted.
}

/* ── Effective storage policy merge ───────────────────────────────── */

StoragePolicy LayerHookConfig::effective_storage() const {
    if (storage_override != StoragePolicy::AUTO) return storage_override;
    // Falls through — caller should merge with session default elsewhere.
    return StoragePolicy::AUTO;
}

/* ── Public entry points ──────────────────────────────────────────── */

uint64_t session_create(StoragePolicy storage, ReductionPolicy reduction,
                       int64_t sample_every, int64_t max_batches,
                       int64_t auto_cpu_threshold_bytes, bool use_pinned) {
    uint64_t id = g_session_counter.fetch_add(1, std::memory_order_relaxed) + 1;

    auto state = std::make_unique<SessionState>();
    state->default_storage   = storage;
    state->reduction         = reduction;
    state->sample_every      = sample_every;
    state->max_batches       = max_batches;
    state->auto_cpu_threshold_bytes = auto_cpu_threshold_bytes;
    state->use_pinned        = use_pinned;

    // Infer capture policy from session knobs.
    CapturePolicy cap_policy = infer_capture_policy(sample_every, max_batches);

    {
        std::lock_guard<std::mutex> lock(registry_mutex);
        global_registry[id] = std::move(state);
    }
    return id;
}

void session_destroy(uint64_t id) {
    // Release session resources before erasing from registry.
    SessionState* state = SessionState::get(id);
    if (state) state->release();

    std::lock_guard<std::mutex> lock(registry_mutex);
    global_registry.erase(id);
}

std::unordered_map<std::string, std::vector<torch::Tensor>> session_readback(uint64_t id) {
    SessionState* state = SessionState::get(id);
    if (!state) return {};

    std::unordered_map<std::string, std::vector<torch::Tensor>> result;
    std::lock_guard<std::mutex> lock(state->mutex);
    for (const auto& [key, accum] : state->accum_data) {
        // readback() returns vector<Tensor> by value — shallow copies of TensorImpl refs.
        std::vector<torch::Tensor> tensors = accum.readback();
        if (!tensors.empty()) {
            result[key] = std::move(tensors);
        }
    }
    return result;
}

void session_clear(uint64_t id) {
    SessionState* state = SessionState::get(id);
    if (!state) return;

    std::lock_guard<std::mutex> lock(state->mutex);
    for (auto& [key, accum] : state->accum_data) {
        accum.clear();
    }
    for (auto& [key, cfg] : state->layer_configs) {
        cfg.counter.reset();
    }
}

void session_register_hooks(uint64_t id, uintptr_t module_ptr,
                           const std::string& layer_key, int32_t capture_dir_int) {
    SessionState* state = SessionState::get(id);
    if (!state) return;

    // Create or update layer config before registering hooks.
    {
        auto& cfg = state->layer_configs[layer_key];
        cfg.capture_dir = static_cast<CaptureDir>(capture_dir_int);
        if (cfg.counter.policy == CapturePolicy::EVERY && state->sample_every > 1) {
            cfg.counter.policy = CapturePolicy::SAMPLE_N;
            cfg.counter.sample_every = state->sample_every;
        }
    }

    // Cast module_ptr back to PyObject* and register hooks via pybind11.
    void* module_py_obj = reinterpret_cast<void*>(module_ptr);

    /* Ensure GIL is held before calling into pybind11 (hooks fire w/o GIL). */
    PyGILState_STATE gstate = PyGILState_Ensure();
    register_hooks_on_module(module_py_obj, state, layer_key, capture_dir_int);
    PyGILState_Release(gstate);
}

void session_set_layer_reduction(uint64_t id, const std::string& layer_name,
                                void* compiled_handle) {
    SessionState* state = SessionState::get(id);
    if (!state || !compiled_handle) return;

    auto& cfg = state->layer_configs[layer_name];
    cfg.reduce_fn = std::make_unique<CompiledFnHandle>(compiled_handle);
}

void session_set_global_reduction(uint64_t id, void* compiled_handle) {
    SessionState* state = SessionState::get(id);
    if (!state || !compiled_handle) return;

    state->global_reduce_fn = std::make_unique<CompiledFnHandle>(compiled_handle);
}

} // namespace activationscope
