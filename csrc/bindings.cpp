/*
 * ActivationScope - PyBind11 Python bindings for the C++ hook backend.
 *
 * Thin wrappers only — zero logic; parameter translation to session/
 * hook/core APIs.  Policy enums and compiled-handle factory exposed so
 * Python can construct sessions, attach hooks, and register reductions.
 */

#include "compiled_fn.hpp"
#include "session.hpp"
#include <torch/extension.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "ActivationScope C++ backend for session-scoped activation "
            "tracking with zero-copy readback and native libtorch hooks";

  /* ── Session lifecycle ─────────────────────────────────────────── */

  m.def(
      "session_create",
      [](int32_t storage, int32_t reduction, int64_t sample_every,
         int64_t max_batches, int64_t auto_cpu_threshold_bytes, bool use_pinned,
         const std::string &session_dir, int32_t capture_mode) -> uint64_t {
        return activationscope::session_create(
            static_cast<activationscope::StoragePolicy>(storage),
            static_cast<activationscope::ReductionPolicy>(reduction),
            sample_every, max_batches, auto_cpu_threshold_bytes, use_pinned,
            session_dir,
            static_cast<activationscope::CaptureMode>(capture_mode));
      },
      py::arg("storage"), py::arg("reduction"), py::arg("sample_every"),
      py::arg("max_batches"), py::arg("auto_cpu_threshold_bytes"),
      py::arg("use_pinned"), py::arg("session_dir") = std::string(""),
      py::arg("capture_mode") = 0,
      "Create new session, return uint64_t ID.");

  m.def("session_destroy", &activationscope::session_destroy, py::arg("id"),
        "Destroy the session by ID (drops hooks + clears vectors). No-op if "
        "invalid.");

  m.def("session_readback", &activationscope::session_readback, py::arg("id"),
        "Zero-copy readback: dict[str, List[torch.Tensor]] sharing TensorImpl "
        "with C++.");

  m.def("session_readback_disk", &activationscope::session_readback_disk,
        py::arg("id"),
        "Read back DISK-mode activations: dict[str, List[str]] mapping layer "
        "names "
        "to sorted .pt file paths on disk.");

  m.def("session_clear", &activationscope::session_clear, py::arg("id"),
        "Clear activations + reset batch counters (hooks stay active).");

  m.def("session_detach_hooks", &activationscope::session_detach_hooks,
        py::arg("id"),
        "Detach all hooks from modules; keep session alive for reuse.");

  /* ── Hook registration ─────────────────────────────────────────── */

  m.def(
      "session_register_hooks", &activationscope::session_register_hooks,
      py::arg("id"), py::arg("module_ptr"), py::arg("layer_key"),
      py::arg("capture_dir_int"),
      "Register native hooks on a submodule. module_ptr = id(module) uintptr.");

  /* ── Compiled reduction handles ─────────────────────────────────── */

  m.def(
      "make_compiled_handle",
      [](py::object fn) -> void * {
        return activationscope::make_compiled_handle(fn.ptr());
      },
      py::arg("fn"),
      "Wrap a Python callable (compiled reduction) into an opaque C++ handle.");

  m.def("set_layer_reduction", &activationscope::session_set_layer_reduction,
        py::arg("id"), py::arg("layer_name"), py::arg("compiled_handle"),
        "Attach compiled reduction to a specific layer pattern.");

  m.def("set_global_reduction", &activationscope::session_set_global_reduction,
        py::arg("id"), py::arg("compiled_handle"),
        "Set session-wide default compiled reduction for unmatched layers.");

}

