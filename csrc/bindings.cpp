/*
 * ActivationScope - PyBind11 Python bindings for the C++ hook backend.
 */

#include "hooks.hpp"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "ActivationScope C++ native hooks for high-performance activation "
            "tracking";

  // Online statistics hooks
  m.def("register_max_hook", &activationscope::register_max_hook,
        "Register a hook to track max activation per layer");
  m.def("register_min_hook", &activationscope::register_min_hook,
        "Register a hook to track min activation per layer");
  m.def("register_mean_hook", &activationscope::register_mean_hook,
        "Register a hook to track mean activation per layer");

  // Stat retrieval
  m.def("get_max_stats", &activationscope::get_max_stats,
        "Get all max statistics");
  m.def("get_min_stats", &activationscope::get_min_stats,
        "Get all min statistics");
  m.def("get_mean_stats", &activationscope::get_mean_stats,
        "Get all mean statistics");

  // Utility
  m.def("clear_stats", &activationscope::clear_stats,
        "Clear all online statistics");
}
