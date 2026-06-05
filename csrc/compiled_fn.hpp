/*
 * ActivationScope - CompiledFnHandle: opaque wrapper around a torch.compile()d
 * graph callable.  Header-only.
 *
 * In practice the compiled handle is stored as an intrusive_ptr to an ATen
 * FuncTorchCompiledAutogradFunction (or similar) but we only need to be able to
 * execute it via at::functionalize / torch::jit dispatch.  To keep the boundary
 * simple, Python materialises the compiled callable and transfers its underlying
 * `torch::jit::ScriptFunction` (or equivalent) through pybind11 as a raw pointer.
 *
 * Since torch.compile() returns a Callable with no stable ABI for native C++ use,
 * we take a pragmatic approach: the compiled function is executed by calling back
 * through a thin C-ABI thunk that holds an `PyObject*` reference to the compiled
 * callable.  This avoids needing to decode FX graphs at compile time while still
 * keeping hot-path overhead minimal (one PyCFunction call, no GIL release needed
 * because torch.compile output runs under eager mode).
 */

#pragma once

#include <Python.h>
#include <memory>
#include <torch/extension.h>

namespace activationscope {

/// Factory: create an opaque PyObject*-backed compiled handle from a raw pointer.
void* make_compiled_handle(PyObject* fn);

/**
 * Opaque handle wrapping a compiled Python callable.
 *
 * `m_handle` is an opaque pointer to a helper struct that stores both the
 * PyObject* reference (for round-trip dispatch) and whether the handle
 * has been released.  The struct lives in compiled_fn.cpp — this header
 * only declares the execute_compiled() dispatcher signature.
 */
class CompiledFnHandle {
public:
    /// Construct from a raw handle pointer transferred via pybind11.
    explicit CompiledFnHandle(void* handle) noexcept
        : m_handle(handle) {}

    /// Non-copyable — handles are uniquely owned per LayerHookConfig session.
    CompiledFnHandle(const CompiledFnHandle&) = delete;
    CompiledFnHandle& operator=(const CompiledFnHandle&) = delete;

    /// Movable (transferred when swapping configs).
    CompiledFnHandle(CompiledFnHandle&& other) noexcept
        : m_handle(std::exchange(other.m_handle, nullptr)) {}

    CompiledFnHandle& operator=(CompiledFnHandle&& other) noexcept {
        reset();
        m_handle = std::exchange(other.m_handle, nullptr);
        return *this;
    }

    ~CompiledFnHandle() { reset(); }

    /// Whether this handle points at a live compiled function.
    explicit operator bool() const noexcept { return m_handle != nullptr; }

    /// Execute the compiled reduction on the input tensor.  Runs under
    /// torch::NoGradGuard (called by hook_callback which already wraps).
    torch::Tensor execute(torch::Tensor tensor) const;

private:
    void reset();

    void* m_handle = nullptr;   // Opaque pointer to Python-side compiled callable storage
};

} // namespace activationscope
