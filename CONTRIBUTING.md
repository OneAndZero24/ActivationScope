# Contributing to ActivationScope

Thank you for your interest in contributing! This project uses PyTorch C++ extensions via `libtorch` bindings, so there are some specifics around native compilation and matrix compatibility. Read on.

---

## Table of Contents

- [Development Setup](#development-setup)
- [Building the C++ Extension](#building-the-c-extension)
- [Running Tests](#running-tests)
  - [Quick Local Tests](#quick-local-tests)
  - [Full Matrix Tests via Docker Compose](#full-matrix-tests-via-docker-compose)
  - [Single Service Test](#single-service-test)
  - [CUDA Testing](#cuda-testing)
- [Version Compatibility Matrix](#version-compatibility-matrix)
- [Auto-Generated Artifacts](#auto-generated-artifacts)
- [Code Style and Conventions](#code-style-and-conventions)
- [Pull Request Guidelines](#pull-request-guidelines)
- [CI/CD Overview](#cicd-overview)
- [Release Process](#release-process)

---

## Development Setup

### Step-by-step

```bash
# 1. Clone the repo
git clone https://github.com/OneAndZero24/ActivationScope.git
cd ActivationScope

# 2. Create and activate the development environment (Python 3.9, CPU PyTorch 2.4)
conda env create -f environment.yml -n activationscope 
conda activate activationscope

# 3. Install in editable mode — this compiles the C++ extension for you
pip install -e .[dev]
```

The `[dev]` extras install `pytest` and `build`. If everything succeeds, you can verify:

```bash
python -c "import activationscope; print(activationscope.__file__)"
```

### Upgrading your local environment

If you want to test against a newer PyTorch:

```bash
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e .[dev]  # recompile extension against new libtorch headers
```

---

## Building the C++ Extension

The C++ sources live in `csrc/` and are compiled via `torch.utils.cpp_extension`. A normal editable install handles this automatically. If you ever need a clean rebuild after touching C++ files:

```bash
# Remove old build artifacts
rm -rf build/ activationscope/_C*.so activationscope/*.egg-info

# Re-install (rebuilds from scratch)
pip install -e .[dev]
```

You must have C++17 support available.

---

## Running Tests

### Quick Local Tests

Run the full test suite against your current environment:

```bash
pytest tests/ -v --tb=short
```

This is the fastest feedback loop during development. It only covers the single Python version + PyTorch version combination currently active in your conda environment.

### Full Matrix Tests via Docker Compose

To validate across all supported Python and PyTorch versions, use the scripts at the repo root:

```bash
scripts/run_tests.sh          # generates pyproject.toml + compose, runs all combos (CPU)
```

This script automatically regenerates both `pyproject.toml` and `.docker/docker-compose.yml` from `matrix.yml`, then builds and runs every Docker service via `docker compose`.

### Single Service Test

Run a specific matrix entry by its Docker service name:

```bash
scripts/run_tests.sh           # (regenerates artifacts)
cd .docker && docker compose up <service_name>        # e.g. py310-torch251-cpu
```

Service labels follow the pattern `py{PYVERSION}-torch{TORCHVERSION}-{platform}`. See `.docker/docker-compose.yml` for the full list of services.

### CUDA Testing

To run the full matrix with CUDA support instead of CPU:

```bash
scripts/run_tests.sh --platform cu124
```

This generates a `docker-compose.yml` that installs PyTorch from the `cu124` wheel index.

---

## Version Compatibility Matrix

The supported combinations live in `matrix.yml` at the repo root. That file is the **single source of truth** — no other file should contain hardcoded Python/PyTorch version pairs.

---

## Auto-Generated Artifacts

Two files are produced from templates and scripts — they are git-ignored and regenerated before each CI run:

| Script | Output | Description |
|--------|--------|-------------|
| `python utils/pyproject.py` | `pyproject.toml` | Fills the template (versions, classifiers) from `matrix.yml` |
| `python utils/generate-compose.py` | `.docker/docker-compose.yml` | Builds one Compose service per matrix entry |

Both scripts accept an optional `--matrix /path/to/matrix.yml` argument. If omitted, they default to the repo root `matrix.yml`.

**Do not edit `pyproject.toml` or `.docker/docker-compose.yml` directly** — they will be overwritten on the next generation run. To change dependencies or metadata, edit `pyproject.toml.template`.

---

## Code Style and Conventions

### Python (`activationscope/`)
- Follow standard PEP 8 conventions.
- Write docstrings using the reStructuredText / Google style for parameters and return types.
- All public API functions and classes must have a docstring explaining purpose, accepted arguments, and any memory-safety notes (e.g., detach-on-store guarantees).

### C++ (`csrc/`)

Architectural rules that **no agent or contributor** may bypass:

1. **Detach-on-Store Rule:** Forward hooks operating in `"store"` mode must capture an independent copy of the output via ``out.detach().clone()``. This prevents the tracker from holding live references into PyTorch's autograd graph while still preserving tensor data for later inspection. Online-stats mode is unaffected — it only reads per-element scalars/tensors and never retains activation copies.
2. **No-Grad Safety:** All inline C++ statistics modifications *must* run inside a `torch::NoGradGuard no_grad;` block so the tracker never mutates or bloats the training graph.
3. **Memory Cleanup:** The Python context management interface must explicitly call `.clear()` on storage dictionaries after a `loss.backward()` step to eliminate lingering references.
4. **Per-Element Reductions:** Online min/max/mean statistics reduce only over the batch dimension (dim 0). The resulting shape `[C, H, W]` (or `[C, SeqLen]`, etc.) is preserved across forward passes and accumulates element-wise running stats per layer component.

---

## Pull Request Guidelines

1. **Fork and branch** from the latest `main`.
2. Write a descriptive title and include:
   - What changed and why.
   - Any impact on the compatibility matrix.
3. Ensure all tests pass in your environment before pushing.
4. For C++ changes, note which PyTorch headers were relied upon — this affects compatibility with older versions.
5. Squash related commits and rebase on `main` before requesting review.

---

## CI/CD Overview

Two GitHub Actions workflows automate the lifecycle:

### `ci-test.yml` — Test Matrix
Runs on push to `main`, pull requests targeting `main`, or manual dispatch. Each run:
1. Installs `pyyaml` at workflow runtime.
2. Regenerates `pyproject.toml` and `.docker/docker-compose.yml` (CPU platform) from `matrix.yml`.
3. Executes `docker compose up` to build and test every matrix combination sequentially.

### `ci-release.yml` — Build & Publish
Triggered by `v*` tags or manual dispatch. For each Python + PyTorch combination in the matrix:
1. Parses `matrix.yml` into a GitHub Actions strategy matrix using `utils/matrix.py`.
2. Builds a Docker image via plain `docker build` (not compose).
3. Runs `python -m build` inside the container to produce a wheel.
4. Uploads all wheels as artifacts and publishes them to PyPI (or TestPyPI for dry-runs).

---

## Release Process

Releases are tag-driven. To cut a new release:

```bash
# 1. Update version in pyproject.toml.template (or wherever version lives)
git add pyproject.toml.template
git commit -m "Bump version to X.Y.Z"

# 2. Tag the commit
git tag vX.Y.Z
git push origin vX.Y.Z
```

The `ci-release.yml` workflow will compile wheels across the matrix and publish them automatically. TestPyPI receives artifacts first for verification; production PyPI publishes only when a `v*` tag is present.
