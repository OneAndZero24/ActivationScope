# Contributing to ActivationScope

Thank you for your interest in contributing! This document is your go-to guide for setting up a development environment, understanding the codebase structure, running tests, and submitting changes.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Project Layout](#project-layout)
- [Development Workflow](#development-workflow)
  - [Building the Extension](#building-the-extension)
  - [Running Tests Locally](#running-tests-locally)
  - [Full Matrix Testing](#full-matrix-testing)
- [Conventions](#conventions)
  - [Commit Messages](#commit-messages)
  - [Code Style and Architecture](#code-style-and-architecture)
- [Pull Requests](#pull-requests)
- [Releasing](#releasing)

---

## Getting Started

The fastest way to go from a fresh clone to a working development environment:

```bash
# 1. Clone the repository
git clone https://github.com/OneAndZero24/ActivationScope.git
cd ActivationScope

# 2. Create and activate your development environment
conda env create -f environment.yml -n activationscope
conda activate activationscope

# 3. Install the package in editable mode (compiles native extension automatically)
pip install -e .[dev]

# 4. Smoke test — verify the import works
python -c "import activationscope; print(activationscope.__file__)"
```

The `[dev]` extras pull in `pytest` and other development dependencies. If the smoke-test
printout shows a valid path inside your workspace, everything is wired up correctly.

### Changing PyTorch versions locally

If you want to test or develop against a different PyTorch release:

```bash
pip install torch==<version> --index-url https://download.pytorch.org/whl/cpu
pip install -e .[dev]   # recompiles the extension against new headers
```

Whenever PyTorch (or any dependency providing native headers) is upgraded, you must reinstall
the editable package so the C++ extension rebuilds.

---

## Project Layout

A high-level map of the repository:

| Path | Purpose |
|---|---|
| `activationscope/` | Python package — public API, context managers, hook management |
| `csrc/` | Compiled C++ extension — native hooks and performance-critical path |
| `tests/` | Unit and integration test suite (pytest) |
| `scripts/` | Test runners, utility scripts for local development |
| `utils/` | Tooling that generated config files from the compatibility matrix |
| `.docker/` | Docker Compose services for multi-version CI testing |
| `.github/workflows/` | GitHub Actions pipelines for CI and publishing |
| `matrix.yml` | **Single source of truth** for supported Python + PyTorch versions |
| `environment.yml` | Base Conda environment definition for local development |
| `pyproject.toml.template` | Template used to generate the active `pyproject.toml` |

### Auto-generated files

`pyproject.toml` and `.docker/docker-compose.yml` are generated from templates using the
compatibility definitions in `matrix.yml`. Do not edit them by hand — they will be overwritten
on the next generation run. To change build metadata or container configuration, edit the
template/source instead.

---

## Development Workflow

### Building the Extension

The native extension is compiled during `pip install -e .`. Most of the time you do not need to
think about it — editable installs rebuild when their trigger files change.

If you have made changes to C++ sources and want a clean rebuild:

```bash
# Remove stale build artifacts
rm -rf build/ activationscope/*.egg-info activationscope/*_C*.so

# Re-install (rebuilds the extension from scratch)
pip install -e .[dev]
```

A working C++17-capable compiler is required. The extension depends on PyTorch's libtorch
headers, which are discovered automatically from whichever `torch` package is active in your
environment.

### Running Tests Locally

The quickest feedback loop during active development:

```bash
pytest tests/ -v --tb=short
```

This runs the entire suite against whatever Python and PyTorch versions you currently have
installed. It covers correctness and basic regression but does not span the full compatibility
matrix.

### Full Matrix Testing

The CI validates every supported Python + PyTorch combination using Docker Compose. You can
reproduce this locally:

```bash
# Regenerates config artifacts from matrix.yml, builds all services, runs tests (CPU)
scripts/run_tests.sh
```

To test a single service by name (e.g., `py310-torch251-cpu`):

```bash
cd .docker && docker compose up <service_name>
```

Docker service names follow the pattern `py{python}-torch{version}-{platform}` so you can
identify each matrix entry at a glance. For GPU testing, pass a CUDA platform flag:

```bash
scripts/run_tests.sh --platform cu124
```

---

## Conventions

### Commit Messages

ActivationScope follows [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Example | When to use |
|---|---|---|
| `feat:` | `feat: add per-layer variance tracking` | New visible functionality |
| `fix:` | `fix: prevent graph retention in store mode` | Bug fixes with no API changes |
| `refactor:` | `refactor: simplify hook registration flow` | Internal restructuring |
| `perf:` | `perf: avoid redundant tensor copies on forward` | Performance improvements |
| `test:` | `test: add coverage for reduction overflow` | Test-only additions or changes |
| `docs:` | `docs: clarify context manager lifecycle` | Documentation updates |
| `chore:` | `chore: update matrix.yml Python version` | Maintenance, config bumps, deps |

#### Commit message anatomy

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Keep the subject line under ~72 characters. The body is your space for context: what changed,
why it matters, and any trade-offs you considered. Use footers to reference related issues.

#### Skipping CI builds

For documentation-only commits or quick typo fixes that should not trigger a full matrix build,
append a footer:

```
docs: fix typo in contributing guide

[skip ci]
```

Both `[skip ci]` and `[ci skip]` are recognized by the GitHub Actions configuration. Use them
sparingly — only when you are certain no automated checks need to run.

### Code Style and Architecture

- **Python code:** Follow [PEP 8](https://peps.python.org/pep-0008/). Write docstrings in Google
  or NumPy/reStructuredText style for public-facing functions, classes, and modules. Every
  public API element should explain its purpose, parameters, return types, and any memory-safety
  guarantees.

- **C++ code:** Follow the existing formatting conventions of the native source directory. The
  project has architectural constraints around graph retention, memory lifecycle, and execution
  safety at the Python/C++ boundary. These rules are documented in [`AGENTS.md`](AGENTS.md) —
  read that file before modifying native code.

- **General approach:** Match the surrounding code rather than introducing new conventions. When
  refactoring, keep changes scoped to a single concern. Prefer small, focused commits over large
  monolithic diffs so reviewers can evaluate them efficiently.

If you are unsure about a style question that is not covered here, look at nearby files in the
same directory for precedent and ask in your PR description rather than guessing.

---

## Pull Requests

1. **Fork the repository** and create a feature branch off the latest `main`.
2. **Write a clear title** summarizing what the PR changes. In the body, include:
   - What changed and why.
   - Any impact on the compatibility matrix or native extension ABI.
   - Links to related issues or discussions.
3. **Run tests locally** before pushing — both `pytest tests/` and, for C++ changes, the full
   matrix via `scripts/run_tests.sh`.
4. **For native code changes**, note which PyTorch APIs or headers you depend on so reviewers
   can assess backward compatibility impact.
5. **Squash related commits** into logical units and rebase on `main` before requesting review.

PRs are expected to pass CI in all matrix combinations before merge. If a PR intentionally
skips certain platform tests, document the reasoning in the description.

---

## Releasing

Releases are tag-driven: creating and pushing a versioned tag triggers the automated build and
publish pipeline.

```bash
# 1. Update the version string in pyproject.toml.template
git add pyproject.toml.template
git commit -m "chore: bump version to X.Y.Z"

# 2. Create an annotated tag and push it
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

The release workflow:

1. Parses `matrix.yml` to determine every Python + PyTorch combination.
2. Builds a wheel inside an isolated container for each matrix entry.
3. Publishes artifacts — TestPyPI receives wheels first for verification; production PyPI
   publishes only on confirmed `v*` tags.
4. Uploads all built wheels as GitHub Actions artifacts for download and audit.

Dry runs can be triggered manually via the workflow dispatch without tagging, useful for
validating build health before an official release.
