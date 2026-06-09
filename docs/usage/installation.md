# Installation

## PyPI (recommended)

The latest stable release is uploaded to PyPI by the CI pipeline. Install with:

```bash
pip install activationscope
```

## From source (development)

If you need to work on the library itself, follow these steps:

1. Clone the repo and create a Conda environment:
```bash
git clone https://github.com/OneAndZero24/ActivationScope.git
cd ActivationScope
conda env create -f environment.yml -n activationscope
conda activate activationscope
```
2. Install the package in editable mode, which compiles the native C++ extension:
```bash
pip install -e .[dev]
```
3. Verify the installation:
```bash
python -c "import activationscope; print(activationscope.__file__)"
```

> The `[dev]` extra pulls in testing and linting dependencies. The CI workflow (`.github/workflows/ci-release.yml`) builds and uploads wheels to TestPyPI before publishing to PyPI; you can inspect that file for the exact upload steps.
