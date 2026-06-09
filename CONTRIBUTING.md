# Contributing to ActivationScope

Thank you for considering a contribution! This file provides a concise overview for contributors. For detailed, technical guidance see the **Development Guide** at `docs/development/README.md`.

## Quick start

1. **Fork the repository** and clone your fork.
2. **Set up the development environment** – see the step‑by‑step instructions in `docs/development/setup.md`.
3. **Install the package in editable mode** (this also builds the native C++ extension):
   ```bash
   pip install -e .[dev]
   ```

## Building the extension

The C++ backend is compiled automatically during the editable install. For a clean rebuild after source changes, follow the instructions in `docs/development/build.md`.

## Running tests

Run the full test suite locally with:
```bash
pytest -v
```
For the exhaustive CI matrix (Docker‑based testing across many Python/PyTorch versions), refer to `docs/development/testing.md`.

## Continuous integration (CI)

CI is powered by GitHub Actions and validates every supported Python + PyTorch combination defined in `matrix.yml`. Details about the CI workflow and how to reproduce the matrix locally are in `docs/development/ci.md`.

## Style & conventions

We enforce:
- **Conventional Commits** for commit messages.
- **PEP 8** compliance for Python code (formatted with Ruff).
- Existing C++ formatting conventions (clang‑format) for the native extension.
- Architectural guidelines and memory‑safety guarantees as described in `docs/development/conventions.md`.

## Pull request workflow

1. Create a feature branch off the latest `main`.
2. Write clear, concise commit messages that follow the Conventional Commits spec.
3. Ensure the full test suite passes locally (`pytest -v`).
4. Push your branch and open a PR on GitHub.
5. The CI will run automatically; address any failures before merging.

## Releasing

Releases are tag‑driven. When a new version tag (`vX.Y.Z`) is pushed, the CI builds wheels for every matrix entry and publishes them to PyPI. The release process is documented in the Development Guide.

Happy hacking!
