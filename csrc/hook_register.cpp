/*
 * ActivationScope — Hook registration implementation.
 *
 * Hooks are registered via Python's nn.Module.register_forward_hook() /
 * register_forward_pre_hook().  The callback is a pybind11 cpp_function
 * thunk: it extracts the tensor under GIL, then releases the GIL and
 * delegates to hook_callback().
 *
 * The closure captures LayerHookConfig* (for the reduction), a
 * shared_ptr<LayerAccumulator>, and SessionState* — all pre-resolved
 * so the hot path has zero dict lookups and zero string comparisons.
 */
#include "hook_register.hpp"
#include "callback.hpp"
#include "session.hpp"
#include <Python.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;
namespace activationscope {

static void store_hook_handle(SessionState* state, const std::string& key,
                              void* handle) {
    if (handle) Py_INCREF(reinterpret_cast<PyObject*>(handle));
    state->m_hook_handles.emplace_back(key, handle);
}

void register_hooks_on_module(void* module_py_ptr,
                              SessionState* state,
                              const std::string& layer_key,
                              int32_t direction_int,
                              std::shared_ptr<LayerAccumulator> accum) {
    CaptureDir direction = static_cast<CaptureDir>(direction_int);

    /* Build a pybind11 handle to the Python module */
    py::object module_py = py::reinterpret_borrow<py::object>(
        py::handle(reinterpret_cast<PyObject*>(module_py_ptr)));

    /* Resolve LayerHookConfig* once — never looked up on the hot path */
    LayerHookConfig* cfg = &state->layer_configs[layer_key];

    /* ── Output hook (register_forward_hook) ────────────────────
     * GIL discipline: tensor extraction (pybind11 casts) requires GIL.
     * We release the GIL only for the pure-C++ hot path where the
     * TorchScript reduction runs, then re-acquire on scope exit. */
    auto make_output_hook = [&]() -> void* {
        auto thunk = py::cpp_function(
            [state, cfg, layer_key, accum](
                py::object /*module*/, const py::object& /*inputs*/,
                const py::object& output_obj) {
                // 1) Extract tensor — GIL held (required for pybind11 casts)
                torch::Tensor tensor;
                PyObject* out_ptr = output_obj.ptr();
                if (PyTuple_Check(out_ptr)) {
                    py::tuple tup = output_obj.cast<py::tuple>();
                    if (tup.size() > 0 && py::isinstance<torch::Tensor>(tup[0]))
                        tensor = tup[0].cast<torch::Tensor>();
                    else return;
                } else {
                    tensor = output_obj.cast<torch::Tensor>();
                }
                // 2) Release GIL for the C++/TorchScript hot path
                {
                    py::gil_scoped_release release;
                    hook_callback(state, cfg, accum, layer_key, tensor);
                }
                // GIL re-acquired here (scope exit)
            });

        py::object hook_obj = module_py.attr("register_forward_hook")(thunk);
        return hook_obj.release().ptr();
    };

    /* ── Input hook (register_forward_pre_hook) ────────────────── */
    auto make_input_hook = [&]() -> void* {
        auto thunk = py::cpp_function(
            [state, cfg, layer_key, accum](
                py::object /*module*/, const py::object& inputs) {
                if (inputs.is_none()) return;
                // 1) Extract tensor — GIL held (required for pybind11 casts)
                torch::Tensor tensor;
                PyObject* ptr = inputs.ptr();
                if (PyTuple_Check(ptr)) {
                    py::tuple tup = inputs.cast<py::tuple>();
                    if (tup.size() > 0)
                        tensor = tup[0].cast<torch::Tensor>();
                    else return;
                } else {
                    tensor = inputs.cast<torch::Tensor>();
                }
                // 2) Release GIL for the C++/TorchScript hot path
                {
                    py::gil_scoped_release release;
                    hook_callback(state, cfg, accum, layer_key, tensor);
                }
                // GIL re-acquired here (scope exit)
            });

        py::object hook_obj = module_py.attr("register_forward_pre_hook")(thunk);
        return hook_obj.release().ptr();
    };

    if (direction == CaptureDir::OUTPUT || direction == CaptureDir::BOTH) {
        void* handle = make_output_hook();
        store_hook_handle(state, layer_key, handle);
    }
    if (direction == CaptureDir::INPUT || direction == CaptureDir::BOTH) {
        std::string input_key = layer_key + ".input";
        void* handle = make_input_hook();
        store_hook_handle(state, input_key, handle);
    }
}

} // namespace activationscope
