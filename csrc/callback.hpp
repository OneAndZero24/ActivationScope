/*
 * ActivationScope — Hot path callback declaration.
 * Pure C++ — no Python, no GIL.
 */
#pragma once

#include <memory>
#include <string>
#include <torch/extension.h>

namespace activationscope {

struct SessionState;
struct LayerHookConfig;
struct LayerAccumulator;

/// Hot path: called by native libtorch hook lambdas.
/// All state is captured in the closure — no dict lookups.
/// cfg->reduction->run() is called without GIL (TorchScript module).
void hook_callback(SessionState*              state,
                   LayerHookConfig*           cfg,
                   std::shared_ptr<LayerAccumulator> accum,
                   const std::string&         layer_key,
                   torch::Tensor              tensor);

} // namespace activationscope
