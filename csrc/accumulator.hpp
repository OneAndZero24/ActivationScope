/*
 * ActivationScope - ActivationAccumulator: thin wrapper around vector<Tensor>.
 * Header-only, inline templates. No out-of-line definitions.
 */

#pragma once

#include <mutex>
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

    /// Returns the last stored tensor (the running state for reductions).
    /// Returns nullptr if the accumulator is empty.
    const torch::Tensor* last() const noexcept {
        return m_tensors.empty() ? nullptr : &m_tensors.back();
    }

    /// Replace the last stored tensor in-place — safe even when the new
    /// tensor shares the same TensorImpl (in-place reductions).  Does NOT
    /// pre-destroy the old entry.
    void replace_last(torch::Tensor tensor) {
        m_tensors.back() = std::move(tensor);
    }

private:
    std::vector<torch::Tensor> m_tensors;
};

/// ActivationAccumulator guarded by a per-layer mutex.
/// Shared across the hook closure and session.
struct LayerAccumulator {
    ActivationAccumulator data;
    std::mutex             mtx;
};

} // namespace activationscope
