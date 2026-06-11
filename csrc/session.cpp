/*
 * ActivationScope - Session lifecycle implementations.
 *
 * Manages the global registry of sessions, each keyed by an atomic uint64_t
 * counter. All public entry points (create, destroy, readback, clear, hook
 * registration) are implemented here.
 */

#include "session.hpp"
#include "gil_utils.hpp"
#include "hook_register.hpp"
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

/* ── Global registry storage ──────────────────────────────────────── */

static std::unordered_map<uint64_t, std::unique_ptr<SessionState>>
    global_registry;
static std::atomic<uint64_t> g_session_counter{0};
static std::mutex registry_mutex;

/* ── Helpers ─────────────────────────────────────────────────────── */

/// Sanitize a layer name into a filesystem-safe directory/filename.
static std::string sanitize_layer_name(const std::string &raw) {
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

/// Generate a unique temporary directory path for this session.
static std::string make_session_temp_dir(uint64_t id) {
  // Use a random suffix to avoid collisions with concurrent sessions.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(100000, 999999);
  int suffix = dis(gen);

  std::ostringstream oss;
  oss << "/tmp/activationscope_" << id << "_" << suffix;
  std::string dir = oss.str();

  if (mkdir(dir.c_str(), 0700) != 0) {
    // Fallback: try with a different suffix if directory already exists.
    suffix = dis(gen) + 1;
    oss.str("");
    oss << "/tmp/activationscope_" << id << "_" << suffix;
    dir = oss.str();
    mkdir(dir.c_str(), 0700); // best-effort
  }
  return dir;
}

/* ── Compile-time helper: decide capture policy from session params ── */

static CapturePolicy infer_capture_policy(int64_t sample_every,
                                          int64_t max_batches) {
  if (max_batches > 0)
    return CapturePolicy::MAX_K;
  if (sample_every > 1)
    return CapturePolicy::SAMPLE_N;
  return CapturePolicy::EVERY;
}

/* ── SessionState members ─────────────────────────────────────────── */

SessionState *SessionState::get(uint64_t id) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  auto it = global_registry.find(id);
  if (it == global_registry.end())
    return nullptr;
  return it->second.get();
}

void SessionState::release() {
  // 1) Drop all hook handles — decrement PyObject ref counts, then remove().
  {
    std::lock_guard<std::mutex> lock(
        registry_mutex); // safety: hooks may access registry
    for (auto &[key, handle_ptr] : m_hook_handles) {
      if (!handle_ptr)
        continue;
      PyObject *handle = reinterpret_cast<PyObject *>(handle_ptr);
      // Call handle.remove() to detach the hook from the Python module.
      PyObject *method = PyObject_GetAttrString(handle, "remove");
      if (method && PyCallable_Check(method)) {
        PyObject_CallObject(method, nullptr);
        Py_DECREF(method);
      }
      Py_XDECREF(handle);
    }
  }

  // 2) Clear all accumulated tensor data.
  for (auto &[key, accum] : accum_data) {
    accum.clear();
  }

  // 3) Clean up session directory (DISK mode only).
  if (!session_dir.empty()) {
    // Recursively remove session_dir using POSIX calls.
    // Walk directory, unlink files, rmdir subdirs.
    DIR *dir = opendir(session_dir.c_str());
    if (dir) {
      struct dirent *entry;
      while ((entry = readdir(dir)) != nullptr) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
          continue;
        std::string full = session_dir + "/" + entry->d_name;
        // Remove files and empty directories.
        struct stat st;
        if (stat(full.c_str(), &st) == 0) {
          if (S_ISDIR(st.st_mode)) {
            // Remove per-layer subdirectory: unlink all .dat files, then rmdir.
            DIR *sub = opendir(full.c_str());
            if (sub) {
              struct dirent *sub_entry;
              while ((sub_entry = readdir(sub)) != nullptr) {
                if (strcmp(sub_entry->d_name, ".") == 0 ||
                    strcmp(sub_entry->d_name, "..") == 0)
                  continue;
                std::string fpath = full + "/" + sub_entry->d_name;
                unlink(fpath.c_str());
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

  // 4) layer_configs cleared — unique_ptrs auto-destroy CompiledFnHandle
  // instances.
  //    unique_ptr<CompiledFnHandle> in each LayerHookConfig destructs properly.

  // 5) Global reduction handle reset (auto via unique_ptr).
  global_reduce_fn.reset();

  // 6) Clear handles vector.
  m_hook_handles.clear();

  // Final cleanup — maps themselves are destroyed when SessionState is deleted.
}

/* ── Effective storage policy merge ───────────────────────────────── */

StoragePolicy LayerHookConfig::effective_storage() const {
  if (storage_override != StoragePolicy::AUTO)
    return storage_override;
  // Falls through — caller should merge with session default elsewhere.
  return StoragePolicy::AUTO;
}

/* ── Public entry points ──────────────────────────────────────────── */

uint64_t session_create(StoragePolicy storage, ReductionPolicy reduction,
                        int64_t sample_every, int64_t max_batches,
                        int64_t auto_cpu_threshold_bytes, bool use_pinned,
                        const std::string &session_dir) {
  uint64_t id = g_session_counter.fetch_add(1, std::memory_order_relaxed) + 1;

  auto state = std::make_unique<SessionState>();
  state->default_storage = storage;
  state->reduction = reduction;
  state->sample_every = sample_every;
  state->max_batches = max_batches;
  state->auto_cpu_threshold_bytes = auto_cpu_threshold_bytes;
  state->use_pinned = use_pinned;

  // If DISK storage is selected, set up the session directory.
  if (storage == StoragePolicy::DISK) {
    if (!session_dir.empty()) {
      // User-specified directory: create it if needed.
      mkdir(session_dir.c_str(), 0700);
      state->session_dir = session_dir;
    } else {
      // Auto-generate a temp directory.
      state->session_dir = make_session_temp_dir(id);
    }
  }

  // Infer capture policy from session knobs.
  (void)infer_capture_policy(sample_every, max_batches);

  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    global_registry[id] = std::move(state);
  }
  return id;
}

void session_destroy(uint64_t id) {
  // Release session resources before erasing from registry.
  SessionState *state = SessionState::get(id);
  if (state)
    state->release();

  std::lock_guard<std::mutex> lock(registry_mutex);
  global_registry.erase(id);
}

std::unordered_map<std::string, std::vector<torch::Tensor>>
session_readback(uint64_t id) {
  SessionState *state = SessionState::get(id);
  if (!state)
    return {};

  std::unordered_map<std::string, std::vector<torch::Tensor>> result;
  std::lock_guard<std::mutex> lock(state->mutex);
  for (const auto &[key, accum] : state->accum_data) {
    // readback() returns vector<Tensor> by value — shallow copies of TensorImpl
    // refs.
    std::vector<torch::Tensor> tensors = accum.readback();
    if (!tensors.empty()) {
      result[key] = std::move(tensors);
    }
  }
  return result;
}

void session_clear(uint64_t id) {
  SessionState *state = SessionState::get(id);
  if (!state)
    return;

  std::lock_guard<std::mutex> lock(state->mutex);
  for (auto &[key, accum] : state->accum_data) {
    accum.clear();
  }
  for (auto &[key, cfg] : state->layer_configs) {
    cfg.counter.reset();
  }
}

void session_detach_hooks(uint64_t id) {
  SessionState *state = SessionState::get(id);
  if (!state)
    return;

  // Detach hooks: call handle.remove() on each hook object.
  for (auto &[key, handle_ptr] : state->m_hook_handles) {
    if (!handle_ptr)
      continue;
    PyObject *handle = reinterpret_cast<PyObject *>(handle_ptr);
    PyObject *method = PyObject_GetAttrString(handle, "remove");
    if (method && PyCallable_Check(method)) {
      PyObject_CallObject(method, nullptr);
      Py_DECREF(method);
    }
    Py_XDECREF(handle);
  }
  state->m_hook_handles.clear();
}

void session_register_hooks(uint64_t id, uintptr_t module_ptr,
                            const std::string &layer_key,
                            int32_t capture_dir_int) {
  SessionState *state = SessionState::get(id);
  if (!state)
    return;

  // Create or update layer config before registering hooks.
  {
    auto &cfg = state->layer_configs[layer_key];
    cfg.capture_dir = static_cast<CaptureDir>(capture_dir_int);
    // Propagate session-level capture policy to per-layer counter.
    CapturePolicy cap_policy = CapturePolicy::EVERY;
    if (state->max_batches > 0)
      cap_policy = CapturePolicy::MAX_K;
    else if (state->sample_every > 1)
      cap_policy = CapturePolicy::SAMPLE_N;
    cfg.counter.policy = cap_policy;
    cfg.counter.sample_every = state->sample_every;
    cfg.counter.max_batches = state->max_batches;
  }

  // Cast module_ptr back to PyObject* and register hooks via pybind11.
  void *module_py_obj = reinterpret_cast<void *>(module_ptr);

  /* Ensure GIL is held before calling into pybind11 (hooks fire w/o GIL). */
  activationscope::ensure_gil_and_call([module_py_obj, state,
                                        layer_key = std::string(layer_key),
                                        capture_dir_int]() {
    register_hooks_on_module(module_py_obj, state, layer_key, capture_dir_int);
  });
}

void session_set_layer_reduction(uint64_t id, const std::string &layer_name,
                                 void *compiled_handle) {
  SessionState *state = SessionState::get(id);
  if (!state || !compiled_handle)
    return;

  auto &cfg = state->layer_configs[layer_name];
  cfg.reduce_fn = std::make_unique<CompiledFnHandle>(compiled_handle);
}

void session_set_global_reduction(uint64_t id, void *compiled_handle) {
  SessionState *state = SessionState::get(id);
  if (!state || !compiled_handle)
    return;

  state->global_reduce_fn = std::make_unique<CompiledFnHandle>(compiled_handle);
}

/* ── Disk readback ────────────────────────────────────────────────── */

std::unordered_map<std::string, std::vector<std::string>>
session_readback_disk(uint64_t id) {
  std::unordered_map<std::string, std::vector<std::string>> result;
  SessionState *state = SessionState::get(id);
  if (!state || state->session_dir.empty())
    return result;

  DIR *root = opendir(state->session_dir.c_str());
  if (!root)
    return result;

  struct dirent *entry;
  while ((entry = readdir(root)) != nullptr) {
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
      continue;

    std::string layer_dir = state->session_dir + "/" + entry->d_name;
    DIR *sub = opendir(layer_dir.c_str());
    if (!sub)
      continue;

    std::vector<std::string> files;
    struct dirent *sub_entry;
    while ((sub_entry = readdir(sub)) != nullptr) {
      if (strcmp(sub_entry->d_name, ".") == 0 ||
          strcmp(sub_entry->d_name, "..") == 0)
        continue;
      files.push_back(layer_dir + "/" + sub_entry->d_name);
    }
    closedir(sub);

    if (!files.empty()) {
      // Sort by filename for stable batch ordering.
      std::sort(files.begin(), files.end());
      result[std::string(entry->d_name)] = std::move(files);
    }
  }
  closedir(root);
  return result;
}

} // namespace activationscope
