#pragma once

#include <string>
#include <torch/extension.h>
#include <unordered_map>

namespace activationscope {

/// Update running max statistic for a given layer.
void register_max_hook(const std::string &layer_name,
                       const torch::Tensor &output);

/// Update running min statistic for a given layer.
void register_min_hook(const std::string &layer_name,
                       const torch::Tensor &output);

/// Update running mean statistic (count-weighted, numerically stable).
void register_mean_hook(const std::string &layer_name,
                        const torch::Tensor &output);

/// Get all max activation statistics (per-element tensors).
std::unordered_map<std::string, torch::Tensor> get_max_stats();

/// Get all min activation statistics (per-element tensors).
std::unordered_map<std::string, torch::Tensor> get_min_stats();

/// Get all mean activation statistics (per-element tensors).
std::unordered_map<std::string, torch::Tensor> get_mean_stats();

/// Clear all online statistics.
void clear_stats();

} // namespace activationscope
