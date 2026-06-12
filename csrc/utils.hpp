/*
 * ActivationScope — shared utility functions.
 *
 * Small, reusable helpers used across multiple translation units.
 * Header-only to avoid a separate .cpp for trivial helpers.
 */

#pragma once

#include <string>
#include <cstdint>
#include <torch/extension.h>
#include "datastructures.hpp"

namespace activationscope {

/// Sanitize a layer name into a filesystem-safe directory / filename.
std::string sanitize_layer_name(const std::string& raw);

/// Create directory (and parents) if it doesn't exist.
bool ensure_dir(const std::string& path);


/// Generate a unique temporary directory path for this session.
std::string make_session_temp_dir(uint64_t id);

/// Compile-time helper: decide capture policy from session params.
CapturePolicy infer_capture_policy(int64_t sample_every, int64_t max_batches);

} // namespace activationscope
