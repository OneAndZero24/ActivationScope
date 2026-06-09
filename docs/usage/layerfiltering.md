# Layer Filtering – Include / Exclude Patterns

`ActivationScope.track` accepts two optional arguments, `include` and `exclude`, each a list of **fnmatch** patterns. The pattern matching is performed once at attachment time, producing an immutable set of leaf modules that will be hooked.

## Pattern Syntax
- `*` matches any number of characters (e.g., `"*.conv*"`).
- `?` matches a single character.
- `[seq]` matches any character in *seq*.
- `{a,b}` matches either *a* or *b* (supported via the standard `fnmatch` package).

## Typical Use‑Cases
- **Track only attention layers**:
  ```python
  tracker = activationscope.ActivationScope()
  with tracker.track(model, include=[".*attn.*", "*.attention.*"]):
      out = model(x)
  ```
- **Exclude bias tensors** (which are often not needed for activation analysis):
  ```python
  with tracker.track(model, exclude=[".*bias.*"]):
      ...
  ```
- **Combine include and exclude** – inclusion is applied first, then exclusion removes any matches:
  ```python
  with tracker.track(model, include=["*"], exclude=["*.bn*", "*.dropout*"]):
      ...
  ```

Reference tests: `tests/test_unit_layer_selection.py`, `tests/test_e2e_models.py`, and `tests/test_model_complexity.py`.
## Interaction with CapturePolicy & ReductionPolicy
Layer filtering is entirely orthogonal: once a set of layers is selected, each layer follows the global `CapturePolicy` and `ReductionPolicy` unless you register per‑layer overrides (e.g., a different reduction for a subset of layers).

## Advanced Tips
- Use **root‑module exclusion** (`exclude=[""]`) to explicitly skip tracking the top‑level container. The root module is already excluded internally by design, so this is only needed if you want to make the exclusion intent explicit in your code.
- Patterns are **case‑sensitive**; match the exact names returned by `model.named_modules()`.
- For very large models, limiting the tracked layer set dramatically reduces both memory and runtime overhead.
