# CapturePolicy – When and How Often to Capture

`CapturePolicy` determines **the frequency** of activation capture inside the native hook. It is implemented as a lightweight atomic check, so skipped captures incur virtually **zero overhead**.

## Policies
| Policy | Behaviour | Typical Use‑Case |
|--------|------------|-----------------|
| `EVERY` | Capture on **every** forward pass (default). | Standard training loops where you need a full activation history. |
| `SAMPLE_N` | Capture **every Nth** forward (`sample_every=N`). | Long training runs where you only need periodic snapshots (e.g., every 10th step). |
| `MAX_K` | Capture up to **K** batches per layer (`max_batches=K`). After reaching the limit the hook returns early without allocation. | Safety rail to prevent OOM in uncontrolled loops (e.g., diffusion denoising). |

## Example Usage
```python
# Sample every 5th forward
tracker = activationscope.ActivationScope(
    capture=activationscope.CapturePolicy.SAMPLE_N,
    sample_every=5,
)

# Hard‑cap at 20 captures per layer
tracker = activationscope.ActivationScope(
    capture=activationscope.CapturePolicy.MAX_K,
    max_batches=20,
)
```

Reference tests: `tests/test_integ_capture_policies.py` (plus related end‑to‑end tests).

## Distinction from CaptureMode

`CapturePolicy` controls **when** to capture (frequency).  `CaptureMode` controls **how** the captured tensor is stored:

- **`CaptureMode.REFERENCE`** (default): `detach()` only — shares storage with the autograd graph.  Fastest path.
- **`CaptureMode.SNAPSHOT`**: `detach()` + `clone()` — independent copy safe for post‑capture mutation.

The two are orthogonal and can be combined freely:

```python
tracker = activationscope.ActivationScope(
    capture=activationscope.CapturePolicy.SAMPLE_N,
    capture_mode=activationscope.CaptureMode.SNAPSHOT,
    sample_every=5,
)
```

## Interaction with Other Policies
Capture frequency is orthogonal to both `StoragePolicy` and `ReductionPolicy`. A common pattern for memory‑constrained workloads is:
```python
tracker = activationscope.ActivationScope(
    storage=activationscope.StoragePolicy.GPU,
    reduction=activationscope.ReductionPolicy.STREAMING,
    capture=activationscope.CapturePolicy.MAX_K,
    max_batches=30,
    capture_mode=activationscope.CaptureMode.REFERENCE,
)
tracker.register_reduction(activationscope.ActivationScope.for_mean())
```
This captures at most 30 activations per layer, keeps only the streamed mean on‑device, and never moves tensors to CPU unless explicitly requested.
