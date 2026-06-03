/*
 * ActivationScope - High-performance activation tracking with libtorch.
 * Native PyTorch hooks operating at the C++ level to bypass Python GIL
 * and dispatching overhead for online stats computation.
 *
 * Reduces only over the batch dimension (dim 0), preserving per-element shape
 * [C, H, W] (or [C, SeqLen], etc.) across forward passes. Online max/min/mean
 * accumulate element-wise so that spatial/channel structure is never lost.
 */

#include "hooks.hpp"
#include <mutex>

// Global state for online metrics. Forward hooks may fire on arbitrary
// dispatch threads (DataLoader workers, DataParallel), so all access to
// these maps is serialized through stats_mutex.
static std::mutex stats_mutex;

// Per-element running tensors — shape mirrors the reduced tensor after
// dropping batch dim 0: [C, H, W] for conv features, [C, SeqLen] for LLMs.
static std::unordered_map<std::string, torch::Tensor> online_max_stats;
static std::unordered_map<std::string, torch::Tensor> online_min_stats;
static std::unordered_map<std::string, torch::Tensor> online_mean_stats;

// Number of forward passes per layer (for Welford mean update).
static std::unordered_map<std::string, int64_t> online_forward_count;

namespace activationscope {

/**
 * Update running max statistic for a given layer.
 * Reduces over batch dim 0 only, then element-wise max with the running value.
 */
void register_max_hook(const std::string &layer_name,
                       const torch::Tensor &output) {
  torch::NoGradGuard no_grad;

  // Reduce over batch dimension → shape [C, H, W] (or [C, SeqLen], etc.)
  auto current_max = output.amax(/*dims=*/0);

  std::lock_guard<std::mutex> lock(stats_mutex);
  if (online_max_stats.find(layer_name) == online_max_stats.end()) {
    // First forward pass — store directly on CPU to keep autograd out of it.
    online_max_stats[layer_name] =
        current_max.detach().to(/*device=*/torch::kCPU);
  } else {
    // Element-wise max: running_max = maximum(running_max, current_max)
    online_max_stats[layer_name] = torch::maximum(online_max_stats[layer_name],
                                                  current_max.to(torch::kCPU));
  }
}

/**
 * Update running min statistic for a given layer.
 * Reduces over batch dim 0 only, then element-wise min with the running value.
 */
void register_min_hook(const std::string &layer_name,
                       const torch::Tensor &output) {
  torch::NoGradGuard no_grad;

  // Reduce over batch dimension → shape [C, H, W] (or [C, SeqLen], etc.)
  auto current_min = output.amin(/*dims=*/0);

  std::lock_guard<std::mutex> lock(stats_mutex);
  if (online_min_stats.find(layer_name) == online_min_stats.end()) {
    online_min_stats[layer_name] =
        current_min.detach().to(/*device=*/torch::kCPU);
  } else {
    // Element-wise min: running_min = minimum(running_min, current_min)
    online_min_stats[layer_name] = torch::minimum(online_min_stats[layer_name],
                                                  current_min.to(torch::kCPU));
  }
}

/**
 * Update running mean statistic via a batched Welford-style update.
 * Reduces over batch dim 0 only, then incrementally blends the per-element
 * means across forward passes (not element counts).
 */
void register_mean_hook(const std::string &layer_name,
                        const torch::Tensor &output) {
  torch::NoGradGuard no_grad;

  // Reduce mean over batch dimension, promote to Float64 for precision.
  auto current_mean = output.to(torch::kFloat64).mean(/*dims=*/0);

  std::lock_guard<std::mutex> lock(stats_mutex);
  int64_t &count = online_forward_count[layer_name];
  if (count == 0) {
    // First forward pass — store the reduced mean tensor directly.
    online_mean_stats[layer_name] =
        current_mean.detach().to(/*device=*/torch::kCPU);
    count = 1;
  } else {
    // Welford-style batched update (per-element):
    //   new_mean = old_mean + (current_mean - old_mean) / N
    // where N is the cumulative number of forward passes.
    count += 1;
    auto delta = current_mean.to(torch::kCPU) - online_mean_stats[layer_name];
    online_mean_stats[layer_name] =
        online_mean_stats[layer_name] + delta / static_cast<double>(count);
  }
}

/**
 * Get all max statistics.
 */
std::unordered_map<std::string, torch::Tensor> get_max_stats() {
  torch::NoGradGuard no_grad;
  std::lock_guard<std::mutex> lock(stats_mutex);
  return online_max_stats;
}

/**
 * Get all min statistics.
 */
std::unordered_map<std::string, torch::Tensor> get_min_stats() {
  torch::NoGradGuard no_grad;
  std::lock_guard<std::mutex> lock(stats_mutex);
  return online_min_stats;
}

/**
 * Get all mean statistics.
 */
std::unordered_map<std::string, torch::Tensor> get_mean_stats() {
  torch::NoGradGuard no_grad;
  std::lock_guard<std::mutex> lock(stats_mutex);
  return online_mean_stats;
}

/**
 * Clear all online statistics.
 */
void clear_stats() {
  torch::NoGradGuard no_grad;
  std::lock_guard<std::mutex> lock(stats_mutex);
  online_max_stats.clear();
  online_min_stats.clear();
  online_mean_stats.clear();
  online_forward_count.clear();
}

} // namespace activationscope
