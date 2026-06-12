/*
 * ActivationScope — TorchScript reduction wrapper.
 *
 * Wraps a torch::jit::script::Module loaded from a .pt file.
 * run() calls forward() directly — no serialisation.
 */
#pragma once

#include <memory>
#include <string>
#include <torch/script.h>

namespace activationscope {

/// Thin wrapper owning a torch::jit::script::Module loaded from file.
/// The module must have a forward(acc: Tensor | None, tensor: Tensor) -> Tensor
/// signature.  State (count, etc.) is embedded in the tensor by the
/// TorchScript function; AccumulatorState holds extra C++‑side metadata.
class Reduction {
public:
    explicit Reduction(const std::string& path);
    ~Reduction() = default;

    Reduction(const Reduction&) = delete;
    Reduction& operator=(const Reduction&) = delete;

    /// Run forward(acc, tensor).  acc may be undefined (first call).
    /// Returns updated accumulator tensor.
    torch::Tensor run(const torch::Tensor& acc, const torch::Tensor& tensor) const;

private:
    torch::jit::script::Module module_;
};

} // namespace activationscope
