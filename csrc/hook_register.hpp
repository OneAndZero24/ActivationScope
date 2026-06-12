/*
 * ActivationScope — Hook registration declarations.
 *
 * Registers native libtorch hooks via Python module API (pybind11).
 * The hook closure captures all state directly — no dict lookups
 * on the forward hot path.
 */
#pragma once

#include <memory>
#include <string>
#include "datastructures.hpp"

namespace activationscope {

struct SessionState;
struct LayerAccumulator;

/// Register hooks on a Python-side nn.Module.
/// @param module_py_ptr  Raw PyObject* of the module.
/// @param state          Owning session pointer.
/// @param layer_key      Dot-separated module name.
/// @param direction       CaptureDir enum int value.
/// @param accum          Shared accumulator for this layer (pre-created).
void register_hooks_on_module(void* module_py_ptr,
                              SessionState* state,
                              const std::string& layer_key,
                              int32_t direction,
                              std::shared_ptr<LayerAccumulator> accum);

} // namespace activationscope
