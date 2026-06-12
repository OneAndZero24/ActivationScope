/*
 * ActivationScope — shared utility functions.
 *
 * Small, reusable helpers used across multiple translation units.
 * Header-only to avoid a separate .cpp for trivial helpers.
 */

#pragma once

#include <string>

namespace activationscope {

/// Sanitize a layer name into a filesystem-safe directory / filename.
/// Replaces path separators, colons, wildcards, and dots with underscores.
inline std::string sanitize_layer_name(const std::string& raw) {
    std::string out;
    out.reserve(raw.size());
    for (char c : raw) {
        if (c == '/' || c == '\\' || c == ':' || c == '?' || c == '*') {
            out += '_';
        } else if (c == '.') {
            out += '_';
        } else {
            out += c;
        }
    }
    return out;
}

} // namespace activationscope
