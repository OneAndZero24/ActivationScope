# Building the Native Extension

ActivationScope’s performance-critical code lives in a **C++/ATen** extension compiled with `torch.utils.cpp_extension`. The extension is built automatically during `pip install -e .`.

## Normal Build (during editable install)
By default, PEP 517 build isolation may download a newer version of PyTorch from PyPI into a temporary build environment, leading to ABI/linking mismatches with your runtime environment. 

To build using the dependencies in your active environment, run:
```bash
python -m pip install -e .[dev] --no-build-isolation
```
If the required C++17 compiler and libtorch headers are available, the build will succeed and you can start using the library immediately.

## Clean Re‑Build
After editing C++ source files, you should purge stale build artifacts and reinstall:
```bash
# 1. Remove old build directories and compiled shared objects
rm -rf build/ activationscope/*.egg-info activationscope/*_C*.so

# 2. Re‑install (triggers a fresh compilation using active environment packages)
python -m pip install -e .[dev] --no-build-isolation
```
The build process logs the detected PyTorch version and the include directories used. Any compilation errors will surface here.

## Troubleshooting Linking & Environment Conflicts

### Symbol Mismatch (`Symbol not found`)
If you see runtime import errors like:
`ImportError: Symbol not found: __ZNK3c1010TensorImpl15decref_pyobjectEv`
This indicates that the C++ extension was compiled against a different PyTorch version than the one loaded at runtime.
- **Cause**: A standard `pip install` without `--no-build-isolation` downloaded a newer version of PyTorch to compile the extension.
- **Solution**: Run the clean re-build commands above, ensuring the `--no-build-isolation` flag is appended.

### PATH Mismatch (e.g. Conda vs. pyenv)
If you are using Conda but your shell's `PATH` resolves `python` or `pip` to `pyenv` shims or a system Python installation, the compilation will compile against the wrong SDK.
- **Solution**: Invoke the compilation using the absolute path to your environment's Python executable:
  ```bash
  /path/to/conda/envs/activationscope/bin/python -m pip install -e .[dev] --no-build-isolation
  ```

## Compiler Requirements
- A C++17‑compatible compiler (`gcc >= 7`, `clang >= 6`, or MSVC 2017+ on Windows).
- The same CUDA toolkit version used to compile your installed PyTorch (if you are building with GPU support).
- `torch` must be importable in the environment so that `torch.utils.cpp_extension` can locate the libtorch headers.

For more details on the build pipeline, see the CI configuration in `.github/workflows/ci-release.yml`.
