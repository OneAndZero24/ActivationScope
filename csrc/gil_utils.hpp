/*
 * ActivationScope - GIL (Global Interpreter Lock) RAII utilities.
 *
 * Centralized, reusable abstractions for safe PyGILState_Ensure / Release.
 *
 * Design rules enforced by this module:
 *   1. GIL acquisition is allowed at registration / compilation time
 *      (session_create, session_register_hooks, make_compiled_handle, etc.).
 *   2. Hot-path hooks (callback.cpp) MUST remain GIL-free — those callbacks are
 *      registered via py::gil_scoped_release and run entirely in native C++.
 *   3. Scoped ensures/leases must never straddle async dispatch frames: mixing
 *      PyGILState_Ensure in one thread with a raw PyEval_SaveThread / release in
 *      a child callback can segfault on interpreter teardown.  Always use the
 *      RAII guard so destruction is deterministic (stack-unwinding safe).
 *   4. When calling into Python from compiled reduction handles, always clear
 *      pending exceptions (PyErr_Clear) before releasing GIL to avoid poisoning
 *      interpreter state for subsequent eager-mode hooks.
 */

#pragma once

#include <functional>
#include <memory>
#include <Python.h>
#include <utility>

namespace activationscope {

/**
 * @brief RAII guard that ensures GIL on construction and releases it on destruction.
 *
 * Non-copyable, movable.  Destruction is exception-safe: even if the body throws
 * inside a guarded scope, the destructor runs during stack unwinding and the GIL
 * state is released back to its previous ownership level.
 */
class GilStateGuard {
public:
    /* Construction — ensure GIL for current thread. */
    GilStateGuard() noexcept
        : m_state(PyGILState_Ensure()),
          _owns_gil(true) {}

    /* Non-copyable. */
    GilStateGuard(const GilStateGuard&)            = delete;
    GilStateGuard& operator=(const GilStateGuard&) = delete;

    /* Movable — source becomes empty (no-op destructor). */
    GilStateGuard(GilStateGuard&& other) noexcept
        : m_state(std::exchange(other.m_state, {})),
          _owns_gil(std::exchange(other._owns_gil, false)) {}

    GilStateGuard& operator=(GilStateGuard&& other) noexcept {
        if (this != &other) {
            reset();
            m_state       = std::exchange(other.m_state, {});
            _owns_gil     = std::exchange(other._owns_gil, false);
        }
        return *this;
    }

    /* Destruction — release GIL back. */
    ~GilStateGuard() { reset(); }

private:
    void reset() noexcept {
        if (_owns_gil) {
            PyGILState_Release(m_state);
            _owns_gil = false;
        }
    }

    PyGILState_STATE m_state{};  /* Opaque struct — aggregate zero-initialize */
    bool             _owns_gil;  /* Tracks whether we still own the GIL state */
};

/**
 * @brief Convenience: acquire GIL, invoke callable `func`, release GIL.
 *
 * Deduces return type from `std::invoke_result`.  Returns whatever the
 * callable returns so callers write:
 *     auto result = ensure_gil_and_call([]{ return py_fn(tensor); });
 */
template <typename F>
auto ensure_gil_and_call(F&& func) -> decltype(auto) {
    GilStateGuard guard;
    return std::invoke(std::forward<F>(func));
}

} // namespace activationscope
