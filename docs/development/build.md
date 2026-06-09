# Building the Native Extension

ActivationScope’s performance-critical code lives in a **C++/ATen** extension compiled with `torch.utils.cpp_extension`. The extension is built automatically during `pip install -e .`.

## Normal Build (during editable install)
```bash
pip install -e .[dev]
```
If the required C++17 compiler and libtorch headers are available, the build will succeed and you can start using the library immediately.

## Clean Re‑Build
After editing C++ source files you may want to purge stale artifacts:
```bash
# Remove old build directories and compiled shared objects
rm -rf build/ activationscope/*.egg-info activationscope/*_C*.so

# Re‑install (triggers a fresh compilation)
pip install -e .[dev]
```
The build process logs the detected PyTorch version and the include directories used. Any compilation errors will surface here.

## Compiler Requirements
- A C++17‑compatible compiler (`gcc >= 7`, `clang >= 6`, or MSVC 2017+ on Windows).
- The same CUDA toolkit version used to compile your installed PyTorch (if you are building with GPU support).
- `torch` must be importable in the environment so that `torch.utils.cpp_extension` can locate the libtorch headers.

For more details on the build pipeline, see the CI configuration in `.github/workflows/ci-release.yml`.
