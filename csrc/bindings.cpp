/*
 * ActivationScope — PyBind11 Python bindings for the C++ hook backend.
 */
#include "session.hpp"
#include <torch/extension.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ActivationScope C++ backend — session-scoped activation "
              "tracking with TorchScript reductions (zero GIL)";

    /* ── Session lifecycle ──────────────────────────── */
    m.def(
        "session_create",
        [](int32_t storage, int32_t reduction, int64_t sample_every,
           int64_t max_batches, int64_t auto_cpu_threshold_bytes,
           bool use_pinned, const std::string& session_dir,
           int32_t capture_mode) -> uint64_t {
            return activationscope::session_create(
                static_cast<activationscope::StoragePolicy>(storage),
                static_cast<activationscope::ReductionPolicy>(reduction),
                sample_every, max_batches, auto_cpu_threshold_bytes,
                use_pinned, session_dir,
                static_cast<activationscope::CaptureMode>(capture_mode));
        },
        py::arg("storage"), py::arg("reduction"), py::arg("sample_every"),
        py::arg("max_batches"), py::arg("auto_cpu_threshold_bytes"),
        py::arg("use_pinned"), py::arg("session_dir") = std::string(""),
        py::arg("capture_mode") = 0);

    m.def("session_destroy", &activationscope::session_destroy, py::arg("id"));
    m.def("session_readback", &activationscope::session_readback, py::arg("id"));
    m.def("session_readback_disk", &activationscope::session_readback_disk,
          py::arg("id"));
    m.def("session_clear", &activationscope::session_clear, py::arg("id"));
    m.def("session_detach_hooks", &activationscope::session_detach_hooks,
          py::arg("id"));

    m.def("session_init_accumulator",
          &activationscope::session_init_accumulator,
          py::arg("id"), py::arg("layer_key"), py::arg("tensor"),
          "Pre-seed accumulator so stateful reductions see existing state.");

    /* ── Hook registration ────────────────────────────
     * reduction_path: path to a torch.jit.script .pt file (empty = identity) */
    m.def(
        "session_register_hooks",
        &activationscope::session_register_hooks,
        py::arg("id"), py::arg("module_ptr"), py::arg("layer_key"),
        py::arg("capture_dir_int"),
        py::arg("reduction_path") = std::string(""));
}
