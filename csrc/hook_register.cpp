/*
 * ActivationScope - Hook registration implementation.
 *
 * Hooks are registered via Python's nn.Module.register_forward_hook() API
 * through pybind11, but the callback is a pybind11 cpp_function that calls
 * into pure C++ (core.cpp hook_callback).  The forward pass never touches
 * Python code beyond the initial pybind11 dispatch frame.
 *
 * Hook handles are stored as PyObject* so they can be released during session
 * teardown via handle.remove().
 */

#include "hook_register.hpp"
#include "session.hpp"
#include "core.hpp"
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <Python.h>

namespace py = pybind11;

namespace activationscope {

/* ------------------------------------------------------------------ */

/// Store a hook handle (PyObject*) keyed by layer name for teardown.
void store_hook_handle(SessionState* state, const std::string& key,
                       void* handle) {
    // Py_INCREF to keep the object alive while session holds reference.
    if (handle) Py_INCREF(reinterpret_cast<PyObject*>(handle));
    state->m_hook_handles.emplace_back(key, handle);
}

/* ------------------------------------------------------------------ */

void register_hooks_on_module(void* module_py_ptr, SessionState* state,
                              const std::string& layer_key, int32_t direction_int) {
    CaptureDir direction = static_cast<CaptureDir>(direction_int);

    // Recreate py::object from raw PyObject* (borrowed reference).
    // Cast void* to PyObject*, then wrap in handle + reinterpret_borrow.
    py::object module_py = py::reinterpret_borrow<py::object>(
        py::handle(reinterpret_cast<PyObject*>(module_py_ptr))
    );

    auto register_forward_output = [&]() -> void* {
        // Forward hook signature: hook(module, args_tuple_or_tensor, output_tensor)
        // We want the OUTPUT tensor (3rd positional arg).
        auto thunk = py::cpp_function([state, layer_key](
                py::object /*module*/,           // arg 0 — module reference
                const py::object& /*inputs*/,    // arg 1 — (input_tuple,) or single tensor
                const torch::Tensor& output)     // arg 2 — the forward pass output tensor
            {
            activationscope::hook_callback(state, layer_key, output);
        }, py::call_guard<py::gil_scoped_release>());

        py::object hook_obj = module_py.attr("register_forward_hook")(thunk);
        return hook_obj.release().ptr();
    };

    auto register_forward_pre_input = [&]() -> void* {
        // Pre-hook signature: hook(module, args_tuple_or_tensor)
        // We want the INPUT tensor (2nd positional arg).
        // Inputs is either a tuple or a single tensor.
        auto thunk = py::cpp_function([state, layer_key](
                py::object /*module*/,                    // arg 0 — module reference
                const py::object& inputs)                 // arg 1 — input tuple or tensor
            {
                if (inputs.is_none()) {
                    return;  // No arguments passed (rare but possible)
                }

                torch::Tensor tensor;
                PyObject* ptr = inputs.ptr();
                if (PyTuple_Check(ptr)) {
                    // Unpack tuple — take the first element (the actual input tensor).
                    py::tuple tup = inputs.cast<py::tuple>();
                    if (tup.size() > 0) {
                        tensor = tup[0].cast<torch::Tensor>();
                    } else {
                        return;  // empty tuple, nothing to capture
                    }
                } else {
                    // Single tensor argument.
                    tensor = inputs.cast<torch::Tensor>();
                }
                activationscope::hook_callback(state, layer_key, tensor);
            }, py::call_guard<py::gil_scoped_release>());

        py::object hook_obj = module_py.attr("register_forward_pre_hook")(thunk);
        return hook_obj.release().ptr();
    };

    if (direction == CaptureDir::OUTPUT || direction == CaptureDir::BOTH) {
        void* handle = register_forward_output();
        store_hook_handle(state, layer_key, handle);
    }

    if (direction == CaptureDir::INPUT || direction == CaptureDir::BOTH) {
        std::string input_key = layer_key + ".input";
        void* handle = register_forward_pre_input();
        store_hook_handle(state, input_key, handle);
    }
}

/* ------------------------------------------------------------------ */

} // namespace activationscope
