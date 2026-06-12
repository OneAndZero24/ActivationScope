/*
 * ActivationScope - CompiledFnHandle: opaque wrapper around a torch.compile()d
 * graph callable.  Header-only.
 *
 * Architecture rationale — why we call back through Python (PyObject* round-trip)
 * instead of keeping the compiled graph entirely in C++:
 *
 *   1. torch.compile() returns an _orchestration_ object (TorchFunctions, FX graph,
 *      aot_autern dispatch tree) that has no stable C++ ABI across PyTorch versions.
 *      Attempting to decode the compiled graph at build time would require parsing
 *      internal ATen/FuncTorch structures and break on every torch minor release.
 *
 *   2. The PyObject* round-trip is acceptable because:
 *        - GIL is acquired inside execute(), which lives outside the critical hot
 *          path (the reduction dispatch in callback.cpp already runs under a NoGradGuard
 *          with lock-free early-exit checks before calling execute()).
 *        - compile()d callables run eagerly (no graph re-building), so each invoke
 *          is roughly equivalent to calling a pre-fused kernel wrapper.
 *        - The fallback-on-error path (log warning + identity) ensures forward passes
 *          never abort due to corrupted or mismatched compiled handles.
 *
 * In practice the compiled handle stores an `PyObject*` to the torch.compile()d
 * callable reference-counted in CompiledCallableStorage.  execute() performs a
 * lightweight PyGILState_Ensure / pybind11 round-trip, invokes the callable with
 * the tensor argument, and casts the result back to torch::Tensor.
 */

#pragma once

#include <Python.h>
#include <memory>
#include <torch/extension.h>

namespace activationscope {

/// Factory: create an opaque PyObject*-backed compiled handle from a raw pointer.
void* make_compiled_handle(PyObject* fn);

/// Clone a compiled handle — each clone owns its own PyObject* reference.
/// Needed when the same handle must be assigned to multiple LayerHookConfig
/// entries (e.g. a glob pattern matching several layers).
void* clone_compiled_handle(void* handle);

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

    /// Execute the compiled reduction with the running accumulator and new tensor.
    /// Runs under torch::NoGradGuard (called by hook_callback which already wraps).
    /// @param acc   Current accumulated tensor (empty/undefined on first call).
    /// @param tensor  New forward-pass activation tensor.
    /// @returns  Updated accumulator tensor.
    torch::Tensor execute(torch::Tensor acc, torch::Tensor tensor) const;

private:
    void reset();

    void* m_handle = nullptr;   // Opaque pointer to Python-side compiled callable storage
};

} // namespace activationscope
