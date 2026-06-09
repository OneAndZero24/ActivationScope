# CI & Matrix Testing

ActivationScope uses **GitHub Actions** to run a rigorous CI matrix across multiple Python and PyTorch versions.

## Matrix Definition (`matrix.yml`)
`matrix.yml` lives at the repository root and enumerates supported Python versions (e.g., 3.10, 3.11) and PyTorch releases (e.g., 2.5.1, 2.4.0). Each entry specifies:
- The Docker base image (CPU or CUDA).
- The Conda/YAML environment to install build dependencies.
- The command to build the C++ extension and run the full pytest suite.

The CI workflow (`.github/workflows/ci-test.yml`) reads this file, spawns a Docker service for each matrix cell, and executes `scripts/run_tests.sh` inside the container.

## Release Workflow (`ci-release.yml`)
When a version tag (`vX.Y.Z`) is pushed:
1. The CI builds wheels for every matrix entry.
2. Wheels are first uploaded to **TestPyPI** for verification.
3. If the test‑upload succeeds, the wheels are promoted to the official PyPI index.
4. Release artifacts are also stored as GitHub Action artifacts for audit.

## Reproducing the CI Matrix Locally
The repository provides a convenience script:
```bash
scripts/run_tests.sh
```
It performs the following steps:
1. Parses `matrix.yml` to generate Docker Compose files.
2. Builds each Docker image (including the appropriate CUDA toolkit if needed).
3. Starts containers, installs the package, and runs `pytest -v`.
4. Collects and aggregates the results.

You can limit the run to a single configuration:
```bash
scripts/run_tests.sh --only py310-torch251-cpu
```
Or run GPU tests by adding the platform flag:
```bash
scripts/run_tests.sh --platform cu124
```

## Adding a New Matrix Entry
1. Extend `matrix.yml` with the new Python/PyTorch combination and the appropriate Docker image tag.
2. Ensure the Dockerfile for that image exists under `.docker/` (or create one following the existing pattern).
3. The CI will automatically pick up the new entry on the next push.

## Troubleshooting CI Failures
- **Compilation errors** – Verify that the Docker image contains the matching CUDA toolkit version for the targeted PyTorch build.
- **Test flakes** – Run the failing test locally inside the corresponding Docker container (`docker compose run <service> bash`) to reproduce the environment.
- **Dependency conflicts** – Update `environment.yml` or the `requirements` section of `setup.py`/`pyproject.toml` to reflect the new versions.

For a deeper dive into the architecture that makes the zero‑copy guarantees possible, see the **Design Document** (`docs/DESIGN.md`).
