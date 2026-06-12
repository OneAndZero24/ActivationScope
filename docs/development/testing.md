# Testing the Library

ActivationScope includes a **pytest** suite that validates functionality, memory safety, and policy interactions.

## Running the Full Test Suite Locally
To execute the tests within your active Conda environment, run:
```bash
python -m pytest -v
```
> [!TIP]
> Running `python -m pytest` instead of just `pytest` ensures that the tests run using the exact python interpreter and package dependencies of your active Conda environment, preventing path conflicts (e.g. running a pyenv or globally-installed `pytest` version).

This will run:
- Unit tests (`test_unit_*.py`)
- Integration tests (`test_integ_*.py`)
- End‑to‑end model tests (`test_e2e_models.py`)
- Memory‑leak and assumption checks (`test_memory_*.py`)

You can also run a subset for quicker feedback:
```bash
# Only smoke tests (fast import verification)
pytest tests/test_smoke.py -q

# Only storage‑policy integration tests
pytest tests/test_integ_storage_policies.py -q
```

## Full Matrix Testing (CI)
The CI matrix tests every supported Python + PyTorch combination inside Docker containers. To reproduce the full matrix locally:
```bash
scripts/run_tests.sh
```
The script reads `matrix.yml`, builds Docker images for each configuration, and runs the full test suite inside each container. For a single configuration, use:
```bash
cd .docker && docker compose up py310-torch251-cpu
```
Add `--platform cu124` to test GPU-enabled images.

## Adding New Tests
When extending the library:
1. Add a new file under `tests/` following the naming convention (`test_<area>.py`).
2. Use `pytest` fixtures from `conftest.py` (e.g., `simple_linear_model`, `conv_model`).
3. If the change affects a new Python or PyTorch version, update `matrix.yml` accordingly so CI will test the new matrix entry.
4. Run the full suite locally before pushing to verify that all existing tests still pass.

## Test Coverage Reference
- **`tests/test_smoke.py`** – Basic import and minimal forward pass.
- **`tests/test_integ_storage_policies.py`** – Storage policy behavior and heuristics.
- **`tests/test_integ_reduction_policies.py`** – Reduction policies, custom reductions, and per‑layer overrides.
- **`tests/test_integ_capture_policies.py`** – Capture frequency policies.
- **`tests/test_unit_layer_selection.py`** – Layer filtering logic.
- **`tests/test_memory_assumptions.py`** – Memory usage expectations per policy.
- **`tests/test_memory_leak_detection.py`** – Checks that the C++ session is fully released.
- **`tests/test_e2e_models.py`** – End‑to‑end runs combining multiple policies.
- **`tests/test_model_complexity.py`** – Stress tests on deep models.
