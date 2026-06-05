/*
 * ActivationScope - PyBind11 Python bindings for the C++ hook backend.
 *
 * Thin wrappers only — zero logic; parameter translation to session/
 * hook/core APIs.  Policy enums and compiled-handle factory exposed so
 * Python can construct sessions, attach hooks, and register reductions.
 */

#include "session.hpp"
#include "compiled_fn.hpp"
#include <torch/extension.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "ActivationScope C++ backend for session-scoped activation "
            "tracking with zero-copy readback and native libtorch hooks";

  /* ── Session lifecycle ─────────────────────────────────────────── */

  m.def("session_create",
        [](int32_t storage, int32_t reduction, int64_t sample_every, int64_t max_batches,
           int64_t auto_cpu_threshold_bytes, bool use_pinned) -> uint64_t {
            return activationscope::session_create(
                static_cast<activationscope::StoragePolicy>(storage),
                static_cast<activationscope::ReductionPolicy>(reduction),
                sample_every, max_batches, auto_cpu_threshold_bytes, use_pinned);
        },
        py::arg("storage"),    py::arg("reduction"),
        py::arg("sample_every"),      py::arg("max_batches"),
        py::arg("auto_cpu_threshold_bytes"), py::arg("use_pinned"),
        "Create new session, return uint64_t ID.");

  m.def("session_destroy", &activationscope::session_destroy,
        py::arg("id"),
        "Destroy the session by ID (drops hooks + clears vectors). No-op if invalid.");

  m.def("session_readback", &activationscope::session_readback,
        py::arg("id"),
        "Zero-copy readback: dict[str, List[torch.Tensor]] sharing TensorImpl with C++.");

  m.def("session_clear", &activationscope::session_clear,
        py::arg("id"),
        "Clear activations + reset batch counters (hooks stay active).");

  /* ── Hook registration ─────────────────────────────────────────── */

  m.def("session_register_hooks", &activationscope::session_register_hooks,
        py::arg("id"),      py::arg("module_ptr"),
        py::arg("layer_key"),   py::arg("capture_dir_int"),
        "Register native hooks on a submodule. module_ptr = id(module) uintptr.");

  /* ── Compiled reduction handles ─────────────────────────────────── */

  m.def("make_compiled_handle",
        [](py::object fn) -> void* {
            return activationscope::make_compiled_handle(fn.ptr());
        },
        py::arg("fn"),
        "Wrap a Python callable (compiled reduction) into an opaque C++ handle.");

  m.def("set_layer_reduction", &activationscope::session_set_layer_reduction,
        py::arg("id"),      py::arg("layer_name"),
        py::arg("compiled_handle"),
        "Attach compiled reduction to a specific layer pattern.");

  m.def("set_global_reduction", &activationscope::session_set_global_reduction,
        py::arg("id"),           py::arg("compiled_handle"),
        "Set session-wide default compiled reduction for unmatched layers.");

  /* ── Parameter snapshotting (C++-stored mode) ───────────────────── */

  m.def("capture_parameters",
        [](std::unordered_map<std::string, torch::Tensor> params) {
            // Already built on Python side; this is a no-op stub that validates
            // the map layout.  Actual snapshotting lives in tracker.py.
            return params.size();
        },
        py::arg("params"),
        "Validate parameter snapshot dict (returns key count).");
}
