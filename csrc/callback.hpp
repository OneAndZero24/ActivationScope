/*
 * ActivationScope — Hot path callback declaration.
 */
#pragma once

#include <memory>
#include <string>
#include <torch/extension.h>

namespace activationscope {

struct SessionState;
struct LayerHookConfig;
struct LayerAccumulator;

/// Hot path: called by hook lambdas.  All state is captured in the closure
/// — no dict lookups.  The reduction runs via TorchScript, so the caller
/// should release the GIL before invoking hook_callback.
void hook_callback(SessionState*              state,
                   LayerHookConfig*           cfg,
                   std::shared_ptr<LayerAccumulator> accum,
                   const std::string&         layer_key,
                   torch::Tensor              tensor);

} // namespace activationscope
