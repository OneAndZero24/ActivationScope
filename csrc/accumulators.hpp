/*
 * ActivationScope - ActivationAccumulator: thin wrapper around vector<Tensor>.
 * Header-only, inline templates. No out-of-line definitions.
 */

#pragma once

#include <torch/extension.h>
#include <vector>

namespace activationscope {

/// Thin wrapper around std::vector<torch::Tensor> with append/clear/readback.
class ActivationAccumulator {
public:
    void append(torch::Tensor tensor) {
        m_tensors.push_back(std::move(tensor));
    }

    /// Release all stored tensor storage. Caller holds no references after this.
    void clear() {
        m_tensors.clear();
        // Ensure underlying storage is freed (vector<Tensor> may hold references).
        std::vector<torch::Tensor>().swap(m_tensors);
    }

    /// Materialise a fresh Python-accessible list of tensors.  Each entry shares
    /// the same TensorImpl as the corresponding C++ vector element — zero-copy.
    std::vector<torch::Tensor> readback() const {
        // Returning by-value copies TensorImpl references (shallow tensor copy),
        // not the underlying data buffer.
        return m_tensors;
    }

    /// Number of captured batches for this layer.
    size_t size() const noexcept {
        return m_tensors.size();
    }

private:
    std::vector<torch::Tensor> m_tensors;
};

} // namespace activationscope
