/*
 * ActivationScope — Session lifecycle implementations.
 */
#include "session.hpp"
#include "hook_register.hpp"
#include "utils.hpp"
#include <Python.h>
#include <pybind11/pybind11.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <random>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

namespace py = pybind11;
namespace activationscope {

/* ── Global registry ──────────────────────────────────────── */
static std::unordered_map<uint64_t, std::unique_ptr<SessionState>> global_registry;
static std::atomic<uint64_t> g_session_counter{0};
static std::mutex registry_mutex;

/* ── SessionState members ─────────────────────────────────── */
SessionState* SessionState::get(uint64_t id) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto it = global_registry.find(id);
    return it != global_registry.end() ? it->second.get() : nullptr;
}

void SessionState::release() {
    // 1) Drop hook handles
    for (auto& [key, handle_ptr] : m_hook_handles) {
        if (!handle_ptr) continue;
        PyObject* handle = reinterpret_cast<PyObject*>(handle_ptr);
        PyObject* method = PyObject_GetAttrString(handle, "remove");
        if (method && PyCallable_Check(method)) {
            PyObject_CallObject(method, nullptr);
            Py_DECREF(method);
        }
        Py_XDECREF(handle);
    }

    // 2) Clear accumulators
    for (auto& [key, accum] : accum_data)
        accum->data.clear();

    // 3) Clean session dir
    if (!session_dir.empty()) {
        DIR* dir = opendir(session_dir.c_str());
        if (dir) {
            struct dirent* entry;
            while ((entry = readdir(dir)) != nullptr) {
                if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
                    continue;
                std::string full = session_dir + "/" + entry->d_name;
                struct stat st;
                if (stat(full.c_str(), &st) == 0) {
                    if (S_ISDIR(st.st_mode)) {
                        DIR* sub = opendir(full.c_str());
                        if (sub) {
                            struct dirent* se;
                            while ((se = readdir(sub)) != nullptr) {
                                if (strcmp(se->d_name, ".") == 0 || strcmp(se->d_name, "..") == 0)
                                    continue;
                                unlink((full + "/" + se->d_name).c_str());
                            }
                            closedir(sub);
                        }
                        rmdir(full.c_str());
                    } else {
                        unlink(full.c_str());
                    }
                }
            }
            closedir(dir);
        }
        rmdir(session_dir.c_str());
        session_dir.clear();
    }

    m_hook_handles.clear();
}

/* ── Public entry points ──────────────────────────────────── */
uint64_t session_create(StoragePolicy storage, ReductionPolicy reduction,
                        int64_t sample_every, int64_t max_batches,
                        int64_t auto_cpu_threshold_bytes, bool use_pinned,
                        const std::string& session_dir,
                        CaptureMode capture_mode) {
    uint64_t id = g_session_counter.fetch_add(1, std::memory_order_relaxed) + 1;
    auto state = std::make_unique<SessionState>();
    state->default_storage          = storage;
    state->reduction_policy         = reduction;
    state->sample_every             = sample_every;
    state->max_batches              = max_batches;
    state->auto_cpu_threshold_bytes = auto_cpu_threshold_bytes;
    state->use_pinned               = use_pinned;
    state->capture_mode             = capture_mode;

    if (storage == StoragePolicy::DISK) {
        if (!session_dir.empty()) {
            mkdir(session_dir.c_str(), 0700);
            state->session_dir = session_dir;
        } else {
            state->session_dir = make_session_temp_dir(id);
        }
    }
    {
        std::lock_guard<std::mutex> lock(registry_mutex);
        global_registry[id] = std::move(state);
    }
    return id;
}

void session_destroy(uint64_t id) {
    SessionState* state = SessionState::get(id);
    if (state) state->release();
    std::lock_guard<std::mutex> lock(registry_mutex);
    global_registry.erase(id);
}

std::unordered_map<std::string, std::vector<torch::Tensor>>
session_readback(uint64_t id) {
    SessionState* state = SessionState::get(id);
    if (!state) return {};

    std::unordered_map<std::string, std::vector<torch::Tensor>> result;
    std::lock_guard<std::mutex> lock(state->mutex);
    for (const auto& [key, accum] : state->accum_data) {
        auto tensors = accum->data.readback();
        if (!tensors.empty())
            result[key] = std::move(tensors);
    }
    return result;
}

void session_clear(uint64_t id) {
    SessionState* state = SessionState::get(id);
    if (!state) return;
    std::lock_guard<std::mutex> lock(state->mutex);
    for (auto& [key, accum] : state->accum_data)
        accum->data.clear();
    for (auto& [key, cfg] : state->layer_configs)
        cfg.counter.reset();
}

void session_detach_hooks(uint64_t id) {
    SessionState* state = SessionState::get(id);
    if (!state) return;
    for (auto& [key, handle_ptr] : state->m_hook_handles) {
        if (!handle_ptr) continue;
        PyObject* handle = reinterpret_cast<PyObject*>(handle_ptr);
        PyObject* method = PyObject_GetAttrString(handle, "remove");
        if (method && PyCallable_Check(method)) {
            PyObject_CallObject(method, nullptr);
            Py_DECREF(method);
        }
        Py_XDECREF(handle);
    }
    state->m_hook_handles.clear();
}

/* ── Pre-initialise accumulator for stateful reductions ── */
void session_init_accumulator(uint64_t id, const std::string& layer_key,
                              torch::Tensor tensor) {
    SessionState* state = SessionState::get(id);
    if (!state) return;
    std::lock_guard<std::mutex> lock(state->mutex);
    auto accum = state->accum_data[layer_key];
    if (!accum) {
        accum = std::make_shared<LayerAccumulator>();
        state->accum_data[layer_key] = accum;
    }
    accum->data.clear();
    accum->data.append(std::move(tensor));
}

/* ── Hook registration (modified — accepts reduction path) ── */
void session_register_hooks(uint64_t id, uintptr_t module_ptr,
                            const std::string& layer_key, int32_t capture_dir_int,
                            const std::string& reduction_path) {
    SessionState* state = SessionState::get(id);
    if (!state) return;

    // 1) Create per-layer config
    auto& cfg = state->layer_configs[layer_key];
    cfg.capture_dir = static_cast<CaptureDir>(capture_dir_int);

    CapturePolicy cap = CapturePolicy::EVERY;
    if (state->max_batches > 0)      cap = CapturePolicy::MAX_K;
    else if (state->sample_every > 1) cap = CapturePolicy::SAMPLE_N;
    cfg.counter.policy        = cap;
    cfg.counter.sample_every  = state->sample_every;
    cfg.counter.max_batches   = state->max_batches;

    // 2) Load reduction from .pt file if non-empty
    if (!reduction_path.empty()) {
        cfg.reduction = std::make_shared<Reduction>(reduction_path);
    }

    // 3) Create or reuse shared accumulator (pre-seeded by session_init_accumulator)
    std::shared_ptr<LayerAccumulator> accum;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        auto it = state->accum_data.find(layer_key);
        if (it != state->accum_data.end()) {
            accum = it->second;  // reuse pre-seeded accumulator
        } else {
            accum = std::make_shared<LayerAccumulator>();
            state->accum_data[layer_key] = accum;
        }
    }

    // 4) Register hooks on module (GIL required for pybind11 call)
    {
        py::gil_scoped_acquire gil;
        register_hooks_on_module(
            reinterpret_cast<void*>(module_ptr), state, layer_key,
            capture_dir_int, accum);
    }
}

/* ── Disk readback ───────────────────────────────────────── */
std::unordered_map<std::string, std::vector<std::string>>
session_readback_disk(uint64_t id) {
    std::unordered_map<std::string, std::vector<std::string>> result;
    SessionState* state = SessionState::get(id);
    if (!state || state->session_dir.empty()) return result;

    DIR* root = opendir(state->session_dir.c_str());
    if (!root) return result;

    struct dirent* entry;
    while ((entry = readdir(root)) != nullptr) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
        std::string layer_dir = state->session_dir + "/" + entry->d_name;
        DIR* sub = opendir(layer_dir.c_str());
        if (!sub) continue;
        std::vector<std::string> files;
        struct dirent* se;
        while ((se = readdir(sub)) != nullptr) {
            if (strcmp(se->d_name, ".") == 0 || strcmp(se->d_name, "..") == 0) continue;
            files.push_back(layer_dir + "/" + se->d_name);
        }
        closedir(sub);
        if (!files.empty()) {
            std::sort(files.begin(), files.end());
            result[std::string(entry->d_name)] = std::move(files);
        }
    }
    closedir(root);
    return result;
}

} // namespace activationscope
