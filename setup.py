"""ActivationScope build configuration using PyTorch C++ extensions."""

import os
import platform
from setuptools import setup, Extension

from torch.utils.cpp_extension import BuildExtension, CppExtension

here = os.path.abspath(os.path.dirname(__file__))
sources = [
    os.path.join(here, "csrc", "bindings.cpp"),       # PYBIND11_MODULE entry point
    os.path.join(here, "csrc", "compiled_fn.cpp"),     # CompiledFnHandle execute / reset
    os.path.join(here, "csrc", "core.cpp"),            # hook_callback hot path (C++ only)
    os.path.join(here, "csrc", "hook_register.cpp"),   # register via Python module API + thunk
    os.path.join(here, "csrc", "session.cpp"),         # session lifecycle + global registry
    os.path.join(here, "csrc", "capture_policy.cpp"),  # capture cadence policy
]

extra_compile_args = {"cxx": ["-O2", "-std=c++17"]}
if platform.system() == "Darwin":
    # Use libc++ on macOS to match the toolchain PyTorch is built against.
    extra_compile_args["cxx"].append("-stdlib=libc++")
    # Suppress false-positive 'is_arithmetic' specialization warnings from
    # PyTorch headers under recent Clang/libc++ (macOS 14+ SDK).
    extra_compile_args["cxx"].append("-Wno-invalid-specialization")

cpp_extension = CppExtension(
    name="activationscope._C",
    sources=sources,
    include_dirs=[os.path.join(here, "csrc")],
    extra_compile_args=extra_compile_args,
)

setup(
    name="activationscope",
    version="0.1.0",
    packages=["activationscope"],
    ext_modules=[cpp_extension],
    cmdclass={"build_ext": BuildExtension},
)
