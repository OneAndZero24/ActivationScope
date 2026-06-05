/*
 * ActivationScope - Hook registration declarations.
 *
 * Hooks are registered via Python's module.register_forward_hook() /
 * register_forward_pre_hook() but the callback is a pybind11 cpp_function
 * thunk that immediately calls into pure C++ (core.cpp hook_callback).
 */

#pragma once

#include <string>
#include "datastructures.hpp"

namespace pybind11 { class object; }

namespace activationscope {

struct SessionState;

/// Register hooks on a Python-side nn.Module via pybind11.
/// @param module_py  The Python module object (py::object reference).
/// @param state      Pointer to owning session (never-null, managed by RAII).
/// @param layer_key  Dot-separated module name (+ ".input"/".output" suffix for BOTH).
/// @param direction   CaptureDir enum value: INPUT, OUTPUT, or BOTH.
void register_hooks_on_module(void* module_py_ptr, SessionState* state,
                              const std::string& layer_key, int32_t direction);

} // namespace activationscope
