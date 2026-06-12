/*
 * ActivationScope — Reduction implementation: load + run + no GIL.
 */
#include "reduction.hpp"

namespace activationscope {

Reduction::Reduction(const std::string& path) {
    module_ = torch::jit::load(path);
    TORCH_CHECK(module_.find_method("forward").has_value(),
                "TorchScript reduction missing forward() method");
}

torch::Tensor Reduction::run(const torch::Tensor& acc,
                             const torch::Tensor& tensor) const {
    std::vector<torch::jit::IValue> args;
    if (acc.defined())
        args.emplace_back(acc);
    else
        args.emplace_back();
    args.emplace_back(tensor);
    // forward() is non-const in torch::jit::Module — need mutable access
    auto& m = const_cast<torch::jit::script::Module&>(module_);
    return m.forward(args).toTensor();
}

} // namespace activationscope
