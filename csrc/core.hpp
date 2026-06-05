/*
 * ActivationScope - Hook callback hot path declaration.
 */

#pragma once

#include <string>
#include <torch/extension.h>

namespace activationscope {

/// Opaque forward declarations (avoid cyclic dependency on full SessionState).
struct SessionState;

/**
 * The HOT PATH — called by native libtorch hooks for every matched module.
 *
 * Runs entirely in C++; no Python callable traverses this function.
 *
 * @param state    Pointer to the session owning this hook (never-null, managed by RAII).
 * @param layer_key Full dot-separated module name (+ ".input"/".output" suffix when both).
 * @param tensor   Activation tensor captured from the module's forward pass.
 */
void hook_callback(SessionState* state, const std::string& layer_key,
                   torch::Tensor tensor);

} // namespace activationscope
