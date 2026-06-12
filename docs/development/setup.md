# Development Setup – Getting Started

## 1. Clone the Repository
```bash
git clone https://github.com/OneAndZero24/ActivationScope.git
cd ActivationScope
```

## 2. Create a Conda Environment
The project ships an `environment.yml` that pins the required Python version and core dependencies.
```bash
conda env create -f environment.yml -n activationscope
conda activate activationscope
```

## 3. Install in Editable Mode
This compiles the native C++ extension and installs the package in “editable” mode so changes are reflected without reinstalling.
```bash
python -m pip install -e .[dev] --no-build-isolation
```
The ``[dev]`` extra pulls in testing, linting, and documentation dependencies (`pytest`, `ruff`, `mkdocs`, etc.).

> [!IMPORTANT]
> To prevent Python/PyTorch ABI version conflicts and runtime link/import errors (such as missing symbols like `decref_pyobject`), **always** build with the `--no-build-isolation` flag. If your system PATH is overridden by system Python or pyenv shims, run the installation directly with the environment's python binary: `/path/to/conda/envs/activationscope/bin/python -m pip install -e .[dev] --no-build-isolation`. For more troubleshooting, see the [build guide](build.md).

## 4. Verify the Installation
```bash
python -c "import activationscope; print(activationscope.__file__)"
```
If the path points inside your cloned repository, the installation succeeded.
