/*
 * ActivationScope - CompiledFnHandle implementation.
 *
 * The handle wraps a Python-side torch.compile()d callable via an internal
 * struct that stores a PyObject* reference.  execute() performs a lightweight
 * round-trip through the Python/C API to invoke the callable, passing the
 * running accumulator tensor and the new forward-pass tensor, receiving the
 * updated accumulator back.
 */

#include "compiled_fn.hpp"
#include "gil_utils.hpp"

#include <torch/extension.h>  // provides pybind11 + torch tensor bridge

namespace activationscope {

/* ------------------------------------------------------------------ */

/// Opaque internal storage for Python-side compiled callables.
struct CompiledCallableStorage {
    PyObject* fn = nullptr;   ///< Compiled callable (refcount managed here)
};

/* ------------------------------------------------------------------ */

torch::Tensor CompiledFnHandle::execute(torch::Tensor acc,
                                        torch::Tensor tensor) const {
    if (!m_handle) return tensor;   // No-op fallback — identity

    CompiledCallableStorage* storage = static_cast<CompiledCallableStorage*>(m_handle);
    if (!storage || !storage->fn) return tensor;

    // Acquire GIL via RAII — hooks may fire without it in eager mode dispatch.
    GilStateGuard gil_guard;

    torch::Tensor result = tensor;  // identity fallback on any error
    try {
        pybind11::gil_scoped_acquire gil;
        pybind11::handle handle(storage->fn);
        pybind11::object fn_obj = pybind11::reinterpret_borrow<pybind11::object>(handle);

        // Call fn(accumulator, new_tensor).
        // Pass None as first arg when accumulator is undefined (first call).
        pybind11::object result_obj;
        if (!acc.defined()) {
            result_obj = fn_obj(pybind11::none(), tensor);
        } else {
            result_obj = fn_obj(acc, tensor);
        }
        result = result_obj.cast<torch::Tensor>();
    } catch (const std::exception& ex) {
        TORCH_WARN("Compiled reduction failed for tensor "
                  "(device={}, shape={}). Falling back to identity. Reason: {}",
                  tensor.device(), tensor.sizes(), ex.what());
        PyErr_Clear();
    } catch (...) {
        TORCH_WARN("Compiled reduction threw unknown exception for tensor "
                  "(device={}, shape={}). Falling back to identity.",
                  tensor.device(), tensor.sizes());
        PyErr_Clear();
    }

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

/// Clone a compiled handle — each clone gets its own PyObject* reference.
/// This prevents double-free when the same handle is assigned to multiple
/// LayerHookConfig entries (e.g. a glob pattern matching several layers).
void* clone_compiled_handle(void* handle) {
    if (!handle) return nullptr;
    CompiledCallableStorage* src = static_cast<CompiledCallableStorage*>(handle);
    if (!src || !src->fn) return nullptr;
    auto* dst = new CompiledCallableStorage{nullptr};
    Py_XINCREF(src->fn);
    dst->fn = src->fn;
    return static_cast<void*>(dst);
}

} // namespace activationscope
