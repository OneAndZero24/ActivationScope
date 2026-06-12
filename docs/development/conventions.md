# Conventions – Git, Code Style, and Architecture

## Git Workflow & Commit Messages
ActivationScope follows the **Conventional Commits** specification. Each commit should start with a type (e.g., `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`) optionally followed by a scope, a colon, and a concise description.

```text
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```
Example:
```text
feat(tracker): add per‑layer variance reduction
```
The repository enforces this style via a pre‑commit hook (`ruff format`) and CI checks in `.github/workflows/ci-test.yml`.

## Code Style
- **Python** – Follow **PEP 8** and format with **Ruff** (`ruff format`). Docstrings should use **Google** or **NumPy** style and document parameters, return types, and any memory‑safety guarantees.
- **C++** – Use the existing formatting conventions in `csrc/` (clang‑format). Keep header‑only utilities minimal; implementation files (`.cpp`) should contain the hot‑path logic. Do not introduce additional Python‑level dispatch in the C++ hot path.

## Architectural Guidance
The library’s core design is documented in **`docs/DESIGN.md`**. Key architectural constraints include:
- **Zero‑copy read‑back** – activations are never duplicated at the Python/C++ boundary. See the *Zero‑Copy Readback* section in the design doc.
- **Native libtorch hooks** – C++ callbacks are registered; tensor extraction under GIL, reduction in C++.
- **Memory safety** – The session‑scoped `SessionState` owns all tensors; proper teardown is required (`tracker.remove()` or exiting the context manager).

When modifying the C++ backend, always:
1. Verify that the new code does not introduce hidden Python references that could prevent memory release.
2. Add a test in `tests/` that exercises the new path and checks for leaks using `tests/test_memory_leak_detection.py`.
3. Update the design document if the change alters the public API or the memory model.

## Pull Request Checklist
- [ ] Follow the commit‑message format.
- [ ] Run `ruff format` and `ruff check` locally.
- [ ] Ensure all new/changed tests pass locally (`pytest -q`).
- [ ] Verify that `flake8` or other linting tools report no issues.
- [ ] Update documentation in the appropriate `docs/usage/` or `docs/development/` file.
- [ ] If the change impacts the architecture, add a brief paragraph to `docs/DESIGN.md`.

Adhering to these conventions helps keep the codebase clean, ensures the CI can enforce standards automatically, and maintains the memory‑safety guarantees that are central to ActivationScope.
