# SVD Activation Analysis — Online & Offline

ActivationScope can be the data‑collection engine for Singular Value Decomposition
(SVD) on layer activations — a powerful tool for understanding which input
directions dominate a model's internal representations.  This chapter presents two
complementary SVD strategies, both built on top of the tracker, and explains when
to choose each.

---

## Motivation

Every layer in a neural network projects its input into a feature space.  Often a
small subset of directions carries most of the signal; the rest is noise or
redundant.  SVD lets you **recover those principal directions** from a set of
recorded activations.

Typical use-cases include:

- **Dimensionality reduction** — compress a layer's input to its top‑*k* SVD
  components before feeding it downstream.
- **Protection / unlearning** — constrain weight updates to the orthogonal
  complement of a principal subspace (InTAct‑style workflows).
- **Interpretability** — inspect the top singular vectors to see which input
  patterns the layer "cares about".
- **Anomaly detection** — monitor the projection residual (‖x − μ − U Uᵀ (x−μ)‖)
  to flag out‑of‑distribution inputs.

---

## Two Strategies

| Strategy      | Data passes | Memory      | Best for                           |
|---------------|-------------|-------------|------------------------------------|
| **Online**    | 2           | O(d²)       | Very long streams, streaming data  |
| **Offline**   | 1           | O(N·d)      | Fixed datasets, exact comparison   |

*d = feature dimension of the layer input, N = total number of activation rows.*

Both produce the same output structure — a `SVDBasis` dataclass — and give
*equivalent* principal subspaces when run on the same data.

---

## The Result Structure

```python
from activationscope.tests.test_svd_analysis import SVDBasis
# (dataclass also available directly from the test module)
```

```python
@dataclass
class SVDBasis:
    mu: torch.Tensor          # [d]      per-feature mean
    U: torch.Tensor           # [k, d]   principal basis (rows orthonormal)
    U_residual: torch.Tensor  # [d-k, d] orthogonal complement of U
    S_residual: torch.Tensor  # [d-k]    singular values for residual directions
    k: int                    # number of retained components
```

- **`mu`** is the per‑feature mean activation — center data before projection.
- **`U`** is an orthonormal matrix whose rows span the top‑*k* principal
  directions.  `U @ U.T ≈ I_k`.
- **`U_residual`** and **`S_residual`** capture everything *outside* the
  retained subspace.  Useful for measuring reconstruction quality or for
  protection‑loss constraints.

---

## Online SVD (Streaming Covariance)

Online SVD streams the data in two passes **without ever storing the full
activation matrix**. It uses TorchScript‑compiled stateful reductions that
embed state (count, mu vector) directly in the accumulator tensor shape:

```
Pass 1: register running‑sum reduction → accumulator = [∑xᵢ | count]
        After pass 1: μ = sum / count

Pass 2: register covariance reduction, pre‑seed accumulator with [zeros | μ]
        via session_init_accumulator.  Reduction accumulates (x−μ)ᵀ(x−μ).
        After pass 2: Σ = accumulator

Final:  svd(Σ) → Vh  →  U = Vh[:k]
```

### Key Implementation Details

**TorchScript‑compatible reductions required.** Every reduction registered
with ActivationScope must be expressible in pure `torch.jit.script` — the
accumulator parameter must be typed `Optional[torch.Tensor]`, and all
operations must be supported by TorchScript.

**State embedded in tensor.** The pass‑1 running‑sum reduction stores count
as the last element of the accumulator tensor: `[features..., count]`. The
pass‑2 covariance reduction stores the mu vector as an extra row:
`[cov_matrix | mu_row]`. No Python closures or side‑channel dicts hold state.

**`session_init_accumulator` for pass‑2 pre‑seeding.** The covariance
reduction needs `μ` available on the first call. The accumulator is
pre‑seeded *after* `track()` (which clears accumulators) but *before* the
first forward, using the C‑extension API:

```python
import activationscope._C as _C

with tracker.track(model, ...):
    _C.session_init_accumulator(tracker.session_id, name, seed_tensor)
    # ... forward passes now see pre‑seeded accumulator
```

### Example

```python
from torch import nn, randn
from activationscope.tests.test_svd_analysis import online_svd, SVDBasis

# A simple model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 32)
    def forward(self, x):
        x = self.fc1(x).relu()
        return self.fc2(x)

model = MLP()

# 1000 batches streaming through — O(d²) memory regardless of count
batches = [(randn(8, 20),) for _ in range(1000)]

basis = online_svd(
    model,
    layer_names=["fc1", "fc2"],
    dataloader=batches,
    reduced_dim=16,
)

# Access the principal subspace of the first layer
print(basis["fc1"].U.shape)   # torch.Size([16, 20]) — 16 directions of dim 20
print(basis["fc1"].k)         # 16
```

Because Σ is d × d, the peak memory depends only on the *feature dimension*, not
on how many batches you process.  The residual singular values are recovered as
`sqrt(clamp(σ, 0))` to match the scale of the original centered data.

### When to Use

- Data is too large to fit in memory at once (e.g., a full training set).
- You are capturing activations from an ongoing training loop and don't want to
  materialise everything.
- You need *exact* principal components (covariance‑based SVD is equivalent to
  centered‑data SVD for the same data).

---

## Offline SVD (Materialised Data)

Offline SVD collects all activation rows into a single `[N, d]` matrix and then
runs SVD directly on the centered data:

```
Pass 1: collect all activations → stack into [N, d]
Final:  svd(X − μ) → Vh       → U = Vh[:k]
```

This is simpler and avoids the square‑root correction on singular values, but
requires `O(N·d)` memory. No TorchScript reductions or pre‑seeding needed —
just default `STORE_ALL` collection.

### Example

```python
from activationscope.tests.test_svd_analysis import offline_svd

basis = offline_svd(
    model,
    layer_names=["fc1", "fc2"],
    dataloader=batches,
    reduced_dim=16,
)

# Same output structure as online SVD
assert basis["fc1"].U.shape == (16, 20)
assert torch.allclose(basis["fc1"].U @ basis["fc1"].U.T,
                      torch.eye(16), atol=1e-5)
```

### When to Use

- Your dataset fits comfortably in RAM.
- You want the simplest possible pipeline with no intermediate statistics.
- You need the raw data singular values (σ) without the sqrt recovery step.

---

## Equivalence of Online and Offline

When run on the same data, the two methods produce the **same mean** μ and
**span the same subspace** with U:

```python
online  = online_svd(model, ["fc1"], batches, reduced_dim=8)
offline = offline_svd(model, ["fc1"], batches, reduced_dim=8)

# Means are numerically identical
assert torch.allclose(online["fc1"].mu, offline["fc1"].mu, atol=1e-5)

# Subspaces match: U_on @ U_off.T has Frobenius norm ≈ sqrt(k)
cross = online["fc1"].U @ offline["fc1"].U.T
expected = cross.shape[0] ** 0.5
assert torch.norm(cross, p="fro") / expected > 0.98
```

Individual principal *vectors* may differ by a sign flip (since SVD sign is
arbitrary), but the subspace they span is identical.

---

## Handling Different Layer Types

Both functions automatically reshape activations for SVD:

| Layer type    | Raw shape           | Reshaped for SVD    |
|---------------|---------------------|---------------------|
| `nn.Linear`   | `[N, in_features]`  | `[N, in_features]`  |
| `nn.Conv2d`   | `[N, C, H, W]`      | `[N·H·W, C]`        |
| Transformer   | `[N, T, d_model]`   | `[N·T, d_model]`    |

Call the reshape helper directly when you need it:

```python
from activationscope.tests.test_svd_analysis import _reshape_for_svd

conv_acts = torch.randn(2, 3, 8, 8)      # [B, C, H, W]
svd_ready = _reshape_for_svd(conv_acts)   # [128, 3] — each pixel is a row
```

---

## Integration with ActivationScope Tracker

Both functions use `ActivationScope` under the hood for zero‑copy activation
collection.  The tracker sessions are created and destroyed inside each call, so
there is no persistent state leak:

```python
# Safe to call repeatedly — each call creates fresh tracker sessions
online_svd(model, ["fc1"], batches_1, reduced_dim=8)
offline_svd(model, ["fc1"], batches_2, reduced_dim=16)
```

Model weights are never modified — only hooks are attached temporarily and
removed after the data pass completes.

---

## Advanced: Building Your Own SVD Pipeline

The building blocks are fully reusable.  To construct a custom SVD workflow
with stateful TorchScript reductions and accumulator pre‑seeding:

```python
from typing import Optional
import torch
import activationscope._C as _C
from activationscope import ActivationScope, StoragePolicy
from activationscope.tests.test_svd_analysis import _reshape_for_svd

# ── Pass 1: running sum reduction (state in tensor: [sum | count]) ──
def running_sum_reduce(
    acc: Optional[torch.Tensor],
    new: torch.Tensor,
) -> torch.Tensor:
    reshaped = _reshape_for_svd(new.float())
    batch_count = torch.tensor(float(reshaped.size(0)))
    if acc is None:
        return torch.cat([reshaped.sum(dim=0), batch_count.unsqueeze(0)])
    running_sum = acc[:-1]
    count = acc[-1]
    return torch.cat([running_sum.add_(reshaped.sum(dim=0)),
                      (count + batch_count).unsqueeze(0)])

tracker1 = ActivationScope(storage=StoragePolicy.CPU)
tracker1.register_reduction(running_sum_reduce, layers=["fc1"])

with tracker1.track(model, layers=["fc1"], capture="input"):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            _ = model(batch[0])

# Extract μ from pass‑1 accumulator
t = tracker1.activations["fc1"][0].float()
mu = t[:-1] / t[-1]  # sum / count

# ── Pass 2: covariance reduction with pre‑seeded μ ──
def cov_reduce(
    acc: Optional[torch.Tensor],
    new: torch.Tensor,
) -> torch.Tensor:
    reshaped = _reshape_for_svd(new.float())
    if acc is None:
        return torch.zeros(1)
    mv = acc[-1]
    cov = acc[:-1]
    for start in range(0, reshaped.size(0), 4096):
        xc = reshaped[start:start+4096] - mv
        cov.add_(xc.T @ xc)
    return torch.cat([cov, mv.unsqueeze(0)], dim=0)

tracker2 = ActivationScope(storage=StoragePolicy.CPU)
tracker2.register_reduction(cov_reduce, layers=["fc1"])

with tracker2.track(model, layers=["fc1"], capture="input"):
    # Pre‑seed accumulator with [zeros | mu_row]
    d = mu.size(0)
    seed = torch.cat([torch.zeros(d, d), mu.unsqueeze(0)], dim=0)
    _C.session_init_accumulator(tracker2.session_id, "fc1", seed)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            _ = model(batch[0])

# SVD on covariance → principal basis
cov = tracker2.activations["fc1"][0][:-1]  # strip mu row
_, S, Vh = torch.linalg.svd(cov, full_matrices=False)
U = Vh[:reduced_dim]  # top‑k basis
```

**Key constraints for custom reductions:**
- The accumulator parameter **must** be `Optional[torch.Tensor]`.
- All operations must be TorchScript‑compatible (no dicts, closures, or external state).
- State (count, μ, etc.) must be embedded in the tensor shape.
- `session_init_accumulator` must be called **after** `track()` (which clears) but **before** the first forward.

---

## Reference

- **Test suite**: `tests/test_svd_analysis.py` — exercises both methods
  (online, offline, equivalence, conv handling, tracker integration).
- **Original design**: derived from the InTAct protection‑loss workflow.
- **Related docs**: [ReductionPolicy](reductionpolicy.md) for streaming
  statistics, [CapturePolicy](capturepolicy.md) for controlling capture cadence,
  [Custom Reductions](customreductions.md) for TorchScript requirements.
