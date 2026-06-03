"""ActivationScope build configuration using PyTorch C++ extensions."""

import os
import platform
from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CppExtension

here = os.path.abspath(os.path.dirname(__file__))
sources = [
    os.path.join(here, "csrc", "hooks.cpp"),
    os.path.join(here, "csrc", "bindings.cpp"),
]

extra_compile_args = ["-O2", "-std=c++17"]
if platform.system() == "Darwin":
    # Use libc++ on macOS to match the toolchain PyTorch is built against.
    extra_compile_args.append("-stdlib=libc++")
    # Suppress spurious -Winvalid-specialization from PyTorch's own strong_type.h
    # under newer Xcode SDKs (macOS 15+ ARM64).
    extra_compile_args.append("-Wno-invalid-specialization")

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
