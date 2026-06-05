/*
 * ActivationScope - CompiledFnHandle implementation.
 *
 * The handle wraps a Python-side torch.compile()d callable via an internal
 * struct that stores a PyObject* reference.  execute() performs a lightweight
 * round-trip through the Python/C API to invoke the callable, passing the
 * tensor in and receiving the reduced tensor back.
 */

#include "compiled_fn.hpp"

#include <torch/extension.h>  // provides pybind11 + torch tensor bridge

namespace activationscope {

/* ------------------------------------------------------------------ */

/// Opaque internal storage for Python-side compiled callables.
struct CompiledCallableStorage {
    PyObject* fn = nullptr;   ///< Compiled callable (refcount managed here)
};

/* ------------------------------------------------------------------ */

torch::Tensor CompiledFnHandle::execute(torch::Tensor tensor) const {
    if (!m_handle) return tensor;   // No-op fallback — identity

    CompiledCallableStorage* storage = static_cast<CompiledCallableStorage*>(m_handle);
    if (!storage || !storage->fn) return tensor;

    // Acquire GIL — hooks may fire without it in eager mode dispatch.
    PyGILState_STATE gstate = PyGILState_Ensure();

    torch::Tensor result = tensor;  // identity fallback on any error
    try {
        pybind11::gil_scoped_acquire gil;
        pybind11::handle handle(storage->fn);
        pybind11::object fn_obj = pybind11::reinterpret_borrow<pybind11::object>(handle);
        pybind11::object result_obj = fn_obj(tensor);
        result = result_obj.cast<torch::Tensor>();
    } catch (...) {
        /* Swallow — return identity tensor.  Errors surface at registration time. */
    }

    PyGILState_Release(gstate);
    return result;
}

void CompiledFnHandle::reset() {
    if (!m_handle) return;
    CompiledCallableStorage* storage = static_cast<CompiledCallableStorage*>(m_handle);
    if (storage) {
        Py_XDECREF(storage->fn);
        delete storage;
    }
    m_handle = nullptr;
}

/* ------------------------------------------------------------------ */

/// Allocate a new CompiledCallableStorage backed by the given PyObject* fn.
void* make_compiled_handle(PyObject* fn) {
    auto* s = new CompiledCallableStorage{nullptr};
    Py_XINCREF(fn);
    s->fn = fn;
    return static_cast<void*>(s);
}

} // namespace activationscope
