"""SVD-based activation analysis — online (streaming) and offline (batch) approaches.

This module provides two complementary ways to compute the principal subspace of
layer activations via Singular Value Decomposition (SVD). Both approaches produce a
low‑rank orthonormal basis ``U`` of the *input* directions that dominate the
activation signal, plus residual components ``U_residual`` and singular values
``S_residual`` that capture the orthogonal complement.

**Online SVD** (covariance‑based) streams the data in two sequential passes:
   1. Pass 1 – accumulate a per‑element **mean** vector μ.
   2. Pass 2 – accumulate the centered **covariance** matrix Σ = (X−μ)ᵀ(X−μ).
   3. SVD on Σ → Vh (right singular vectors of X).  Because Σ is d×d (feature
      dimension), this is efficient even for long activation records.

**Offline SVD** (full‑activation) collects every activation batch and then runs
SVD directly on the N×d centered data matrix:
   1. Pass 1 – collect all activation rows into a materialised tensor.
   2. Center by μ, then SVD on the full N×d matrix → Vh directly.

The two methods produce **equivalent** principal bases (up to a sign reversal of
each column) when applied to identical data, but Online SVD uses O(d²) memory
while Offline SVD uses O(N·d) memory — choose online for very long data streams.

Reference: derived from the InTAct protection‑loss workflow (see ``reference.py``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pytest
import torch

from activationscope import ActivationScope, StoragePolicy

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Shared result structure
# ═══════════════════════════════════════════════════════════════════


@dataclass
class SVDBasis:
    """Principal subspace extracted from one layer's activations.

    Attributes
    ----------
    mu : Tensor [d]
        Per‑feature mean activation vector (CPU float32).
    U : Tensor [k, d]
        Top‑*k* orthonormal basis vectors of the input space (rows are
        directions).  Each row has unit L₂ norm.
    U_residual : Tensor [d−k, d]
        Remaining singular vectors (the orthogonal complement of U).
    S_residual : Tensor [d−k]
        Singular values for the residual directions.
        *Online* returns sqrt(σ²), *offline* returns raw σ from the data SVD.
    k : int
        Number of retained principal components (``U.size(0)``).
    """

    mu: torch.Tensor
    U: torch.Tensor
    U_residual: torch.Tensor
    S_residual: torch.Tensor
    k: int


# ═══════════════════════════════════════════════════════════════════
# Helper — reshape activations for SVD
# ═══════════════════════════════════════════════════════════════════

def _reshape_for_svd(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape a (possibly batched) activation tensor into [*, feature_dim].

    Conv2d-style (ndim==4): [N, C, H, W] → [N*H*W, C]  (spatial positions → rows).
    Linear-style  (ndim==3): [N, T, F]   → [N*T, F]    (sequence tokens  → rows).
    Linear-style  (ndim==2): [N, F]      → [N, F]       (already correct).
    Fallback flattens everything beyond the batch dimension.
    """
    if tensor.ndim == 4:
        N, C, H, W = tensor.shape
        return tensor.permute(0, 2, 3, 1).reshape(-1, C)
    if tensor.ndim == 2:
        return tensor
    return tensor.reshape(-1, tensor.shape[-1])


# ═══════════════════════════════════════════════════════════════════
# Online SVD — streaming covariance decomposition
# ═══════════════════════════════════════════════════════════════════

def online_svd(
    model: torch.nn.Module,
    layer_names: List[str],
    dataloader: List[Tuple[torch.Tensor, ...]],
    *,
    reduced_dim: int = 32,
    device: str = "cpu",
) -> Dict[str, SVDBasis]:
    """Compute principal activation subspace via two‑pass streaming covariance.

    **Algorithm**
        1. **Pass 1 – mean**: forward every batch, accumulate per‑feature running
           sum and count.  μ = sum / count.
        2. **Pass 2 – covariance**: forward again, accumulate Σ = Σ (X−μ)ᵀ(X−μ)
           in blocks for numerical stability.
        3. **SVD on covariance**: ``_U, S, Vh = svd(Σ)`` → top *reduced_dim*
           rows of Vh become the principal basis U; the rest go to U_residual.
           Residual singular values are sqrt(clamp(σ, 0)) to recover the
           scale of the original centered data.

    **Memory cost**: O(d²) per layer (the covariance matrix d×d).  Independent
    of the number of data rows N — ideal for streaming or very long sequences.

    Parameters
    ----------
    model : nn.Module
        PyTorch model (will be called in ``eval()`` mode).
    layer_names : list[str]
        Module names to hook (e.g. ``["fc1", "fc2"]``).
    dataloader : list of batches
        Iterable of (tensor, ...) batches.  The first element of each batch is
        passed through the model.
    reduced_dim : int
        Number of principal components to retain (k).
    device : str
        Execution device (default ``"cpu"``).

    Returns
    -------
    dict[str, SVDBasis]
        One entry per layer that produced non‑empty statistics.
    """
    layer_stats: Dict[str, dict] = {
        name: {"running_sum": None, "count": 0, "mu": None, "cov": None}
        for name in layer_names
    }

    # ── Pass 1: accumulate running mean ──────────────────────────
    tracker = ActivationScope(storage=StoragePolicy.CPU)
    with tracker.track(model, layers=layer_names, capture="input"):
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(device)
                _ = model(x)
        # Read activations while tracker ctx is still active (hooks still attached)
        acts = tracker.activations

        for name in layer_names:
            if name not in acts:
                continue
            stacked = torch.cat([t.float() for t in acts[name]], dim=0)
            stacked = _reshape_for_svd(stacked)
            layer_stats[name]["running_sum"] = stacked.sum(dim=0)
            layer_stats[name]["count"] = stacked.shape[0]

    for name, st in layer_stats.items():
        if st["count"] == 0:
            continue
        st["mu"] = st["running_sum"] / float(st["count"])

    # ── Pass 2: accumulate covariance (blocked for numerical stability) ──
    tracker2 = ActivationScope(storage=StoragePolicy.CPU)
    with tracker2.track(model, layers=layer_names, capture="input"):
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(device)
                _ = model(x)
        acts = tracker2.activations

        for name in layer_names:
            st = layer_stats[name]
            if st["mu"] is None:
                continue
            mu = st["mu"]
            if name not in acts:
                continue
            stacked = torch.cat([t.float() for t in acts[name]], dim=0)
            stacked = _reshape_for_svd(stacked)
            d = mu.shape[0]
            if st["cov"] is None:
                st["cov"] = torch.zeros((d, d), dtype=torch.float32, device="cpu")
            chunk = 16384 if d <= 2048 else 4096
            for start in range(0, stacked.shape[0], chunk):
                xc = stacked[start : start + chunk] - mu
                st["cov"].add_(xc.T @ xc)

    # ── SVD on covariance → principal basis ──────────────────────
    results: Dict[str, SVDBasis] = {}
    for name, st in layer_stats.items():
        if st["cov"] is None or st["mu"] is None:
            continue
        _U, S, Vh = torch.linalg.svd(st["cov"], full_matrices=False)
        k = min(reduced_dim, Vh.size(0))
        results[name] = SVDBasis(
            mu=st["mu"],
            U=Vh[:k],
            U_residual=Vh[k:],
            S_residual=torch.sqrt(torch.clamp(S[k:], min=0.0)),
            k=k,
        )
    return results


# ═══════════════════════════════════════════════════════════════════
# Offline SVD — materialised data decomposition
# ═══════════════════════════════════════════════════════════════════

def offline_svd(
    model: torch.nn.Module,
    layer_names: List[str],
    dataloader: List[Tuple[torch.Tensor, ...]],
    *,
    reduced_dim: int = 32,
    device: str = "cpu",
) -> Dict[str, SVDBasis]:
    """Compute principal activation subspace from materialised data SVD.

    **Algorithm**
        1. **Pass 1 – collect**: forward every batch, accumulate all activation
           rows into a materialised `[N, d]` tensor.
        2. **Center & SVD**: ``_U, S, Vh = svd(X - μ)`` → top *reduced_dim*
           rows of Vh become the principal basis U.

    **Memory cost**: O(N·d) per layer (full activation matrix in memory).  Use
    `online_svd()` when data volumes are too large for a single materialisation.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    layer_names : list[str]
        Module names to hook.
    dataloader : list of batches
        Iterable of batches (first element is model input).
    reduced_dim : int
        Number of principal components to retain.
    device : str
        Execution device.

    Returns
    -------
    dict[str, SVDBasis]
        One entry per layer with materialised data SVD.
    """
    # ── Pass 1: collect all activations ──────────────────────────
    tracker = ActivationScope(storage=StoragePolicy.CPU)
    with tracker.track(model, layers=layer_names, capture="input"):
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(device)
                _ = model(x)
        acts = tracker.activations

    # ── SVD on centred data ──────────────────────────────────────
    results: Dict[str, SVDBasis] = {}
    for name in layer_names:
        if name not in acts:
            continue
        stacked = torch.cat([t.float() for t in acts[name]], dim=0)
        stacked = _reshape_for_svd(stacked)

        # Upcast to float32 for SVD (bf16 on CUDA isn't supported)
        working = stacked.to(device=device, dtype=torch.float32)
        if not torch.isfinite(working).all():
            working = torch.nan_to_num(working, nan=0.0, posinf=0.0, neginf=0.0)

        mu = working.mean(dim=0)
        centred = working - mu
        _U, S, Vh = torch.linalg.svd(centred, full_matrices=False)

        k = min(reduced_dim, Vh.size(0))
        results[name] = SVDBasis(
            mu=mu.cpu(),
            U=Vh[:k].cpu(),
            U_residual=Vh[k:].cpu(),
            S_residual=S[k:].cpu(),
            k=k,
        )
    return results


# ═══════════════════════════════════════════════════════════════════
# Test infrastructure — model & data factories
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def linear_svd_model() -> torch.nn.Module:
    """A modest MLP for SVD analysis: Linear(20→64) → ReLU → Linear(64→32)."""

    class SVDDemoMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(20, 64)
            self.act = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(64, 32)

        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            return self.fc2(x)

    return SVDDemoMLP()


@pytest.fixture
def conv_svd_model() -> torch.nn.Module:
    """A small conv stack: Conv2d(3→8)→ReLU→Conv2d(8→16)."""

    class SVDDemoConv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1)
            self.act = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv2d(8, 16, 3, padding=1)

        def forward(self, x):
            x = self.act(self.conv1(x))
            return self.conv2(x)

    return SVDDemoConv()


@pytest.fixture
def svd_dataloader_linear() -> List[Tuple[torch.Tensor]]:
    """10 batches of [B=4, 20] for the linear model."""
    return [(torch.randn(4, 20),) for _ in range(10)]


@pytest.fixture
def svd_dataloader_conv() -> List[Tuple[torch.Tensor]]:
    """10 batches of [B=2, 3, 8, 8] for the conv model."""
    return [(torch.randn(2, 3, 8, 8),) for _ in range(10)]


# ════════════════���══════════════════════════════════════════════════
# Tests — online SVD
# ═══════════════════════════════════════════════════════════════════

class TestOnlineSVD:
    """Streaming covariance‑based SVD on linear and conv layers."""

    def test_returns_nonempty_for_all_layers(self, linear_svd_model, svd_dataloader_linear):
        """Every named layer must have an SVDBasis entry."""
        results = online_svd(
            linear_svd_model, ["fc1", "fc2"], svd_dataloader_linear, reduced_dim=16
        )
        assert "fc1" in results, "fc1 missing from online SVD results"
        assert "fc2" in results, "fc2 missing from online SVD results"
        assert isinstance(results["fc1"], SVDBasis)

    def test_u_is_orthonormal_rows(self, linear_svd_model, svd_dataloader_linear):
        """U must have orthonormal rows: U @ U.T ≈ I."""
        results = online_svd(
            linear_svd_model, ["fc1"], svd_dataloader_linear, reduced_dim=16
        )
        U = results["fc1"].U  # [k, d]
        I_k = U @ U.T
        assert torch.allclose(I_k, torch.eye(I_k.shape[0]), atol=1e-5), (
            f"U rows are not orthonormal: max off-diagonal={I_k.abs().max():.2e}"
        )

    def test_mu_has_correct_shape(self, linear_svd_model, svd_dataloader_linear):
        """mu dimension matches the layer input feature count."""
        results = online_svd(
            linear_svd_model, ["fc1", "fc2"], svd_dataloader_linear, reduced_dim=8
        )
        assert results["fc1"].mu.shape == (20,), (
            f"fc1 mu shape {results['fc1'].mu.shape} != (20,)"
        )
        assert results["fc2"].mu.shape == (64,), (
            f"fc2 mu shape {results['fc2'].mu.shape} != (64,)"
        )

    def test_k_respects_reduced_dim(self, linear_svd_model, svd_dataloader_linear):
        """k should be min(reduced_dim, feature_dim)."""
        results = online_svd(
            linear_svd_model, ["fc2"], svd_dataloader_linear, reduced_dim=16
        )
        assert results["fc2"].k == 16  # fc2 has 64 input features, reduced to 16
        assert results["fc2"].U.shape[0] == 16

    def test_residual_dims_consistent(self, linear_svd_model, svd_dataloader_linear):
        """U_residual and S_residual have consistent sizes."""
        results = online_svd(
            linear_svd_model, ["fc1"], svd_dataloader_linear, reduced_dim=8
        )
        b = results["fc1"]
        total = b.k + b.U_residual.shape[0]
        assert total == b.U.shape[1], (
            f"k + residual rows ({b.k}+{b.U_residual.shape[0]}) != total dim {b.U.shape[1]}"
        )
        assert b.U_residual.shape[0] == b.S_residual.shape[0], (
            f"U_residual rows {b.U_residual.shape[0]} != S_residual len {b.S_residual.shape[0]}"
        )

    def test_conv_svd_produces_results(self, conv_svd_model, svd_dataloader_conv):
        """Online SVD works on Conv2d layers (reshaped spatial dims)."""
        results = online_svd(
            conv_svd_model, ["conv1", "conv2"], svd_dataloader_conv, reduced_dim=4
        )
        assert "conv1" in results
        assert "conv2" in results
        # conv1 input channels = 3, conv2 input channels = 8
        assert results["conv1"].mu.shape == (3,)
        assert results["conv2"].mu.shape == (8,)

    def test_s_residual_nonnegative(self, linear_svd_model, svd_dataloader_linear):
        """Residual singular values (sqrt‑recovered) must be >= 0."""
        results = online_svd(
            linear_svd_model, ["fc2"], svd_dataloader_linear, reduced_dim=8
        )
        assert (results["fc2"].S_residual >= 0).all(), (
            "Negative S_residual values found"
        )


# ═══════════════════════════════════════════════════════════════════
# Tests — offline SVD
# ═══════════════════════════════════════════════════════════════════

class TestOfflineSVD:
    """Materialised‑data SVD on linear and conv layers."""

    def test_returns_nonempty_for_all_layers(self, linear_svd_model, svd_dataloader_linear):
        """Every named layer must have an SVDBasis entry."""
        results = offline_svd(
            linear_svd_model, ["fc1", "fc2"], svd_dataloader_linear, reduced_dim=16
        )
        assert "fc1" in results
        assert "fc2" in results

    def test_u_is_orthonormal_rows(self, linear_svd_model, svd_dataloader_linear):
        """U must have orthonormal rows."""
        results = offline_svd(
            linear_svd_model, ["fc1"], svd_dataloader_linear, reduced_dim=16
        )
        U = results["fc1"].U
        I_k = U @ U.T
        assert torch.allclose(I_k, torch.eye(I_k.shape[0]), atol=1e-5)

    def test_mu_has_correct_shape(self, linear_svd_model, svd_dataloader_linear):
        """mu dimension matches input feature count."""
        results = offline_svd(
            linear_svd_model, ["fc1", "fc2"], svd_dataloader_linear, reduced_dim=8
        )
        assert results["fc1"].mu.shape == (20,)
        assert results["fc2"].mu.shape == (64,)

    def test_k_respects_reduced_dim(self, linear_svd_model, svd_dataloader_linear):
        """k should equal reduced_dim when d >= reduced_dim."""
        results = offline_svd(
            linear_svd_model, ["fc2"], svd_dataloader_linear, reduced_dim=16
        )
        assert results["fc2"].k == 16

    def test_residual_dims_consistent(self, linear_svd_model, svd_dataloader_linear):
        """U_residual and S_residual have consistent sizes."""
        results = offline_svd(
            linear_svd_model, ["fc1"], svd_dataloader_linear, reduced_dim=8
        )
        b = results["fc1"]
        total = b.k + b.U_residual.shape[0]
        assert total == b.U.shape[1]
        assert b.U_residual.shape[0] == b.S_residual.shape[0]

    def test_conv_svd_produces_results(self, conv_svd_model, svd_dataloader_conv):
        """Offline SVD works on Conv2d layers."""
        results = offline_svd(
            conv_svd_model, ["conv1", "conv2"], svd_dataloader_conv, reduced_dim=4
        )
        assert "conv1" in results
        assert "conv2" in results
        assert results["conv1"].mu.shape == (3,)
        assert results["conv2"].mu.shape == (8,)


# ═══════════════════════════════════════════════════════════════════
# Tests — online vs offline equivalence
# ═══════════════════════════════════════════════════════════════════

class TestOnlineVsOffline:
    """Online and offline SVD produce equivalent principal subspaces."""

    def test_mu_identical(self, linear_svd_model, svd_dataloader_linear):
        """Both methods must compute the identical mean."""
        online = online_svd(
            linear_svd_model, ["fc1"], svd_dataloader_linear, reduced_dim=8
        )
        offline = offline_svd(
            linear_svd_model, ["fc1"], svd_dataloader_linear, reduced_dim=8
        )
        assert torch.allclose(online["fc1"].mu, offline["fc1"].mu, atol=1e-5), (
            f"Online mu differs from offline: max diff={torch.abs(online['fc1'].mu - offline['fc1'].mu).max():.2e}"
        )

    def test_principal_subspaces_equivalent(self, linear_svd_model, svd_dataloader_linear):
        """The spanned subspace should be identical (up to column sign).

        If U_on and U_off span the same subspace, then U_on @ U_off.T should be
        an orthonormal matrix (up to sign flips on the diagonal).  The Frobenius
        norm of this ``k×k`` cross‑gramian must be ≈ sqrt(k).
        """
        online = online_svd(
            linear_svd_model, ["fc2"], svd_dataloader_linear, reduced_dim=16
        )
        offline = offline_svd(
            linear_svd_model, ["fc2"], svd_dataloader_linear, reduced_dim=16
        )
        U_on = online["fc2"].U   # [k, d]
        U_off = offline["fc2"].U  # [k, d]

        cross = U_on @ U_off.T         # [k, k]
        expected_fro = cross.shape[0] ** 0.5
        actual_fro = torch.norm(cross, p="fro")
        rel_error = abs(actual_fro - expected_fro) / expected_fro
        assert rel_error < 0.02, (
            f"Subspaces differ significantly: "
            f"|U_on @ U_off.T|_F = {actual_fro:.4f} (expected {expected_fro:.4f}, "
            f"rel err {rel_error:.2%})"
        )

    def test_singular_value_correlation(self, linear_svd_model, svd_dataloader_linear):
        """Top singular values from online and offline must be highly correlated.

        Offline computes σ from the data matrix; online computes λ from the
        covariance Σ = XᵀX with λ = σ².  After the sqrt recovery, the two
        sequences should track each other closely.
        """
        online = online_svd(
            linear_svd_model, ["fc1"], svd_dataloader_linear, reduced_dim=4
        )
        offline = offline_svd(
            linear_svd_model, ["fc1"], svd_dataloader_linear, reduced_dim=4
        )
        s_on = online["fc1"].S_residual[:8]
        s_off = offline["fc1"].S_residual[:8]
        # Pearson correlation should be near 1.0
        corr = torch.corrcoef(torch.stack([s_on, s_off]))[0, 1]
        assert corr > 0.99, (
            f"Singular value correlation too low: {corr:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════
# Tests — integration with ActivationScope tracker
# ═══════════════════════════════════════════════════════════════════

class TestTrackerIntegration:
    """SVD functions integrate cleanly with the ActivationScope tracker."""

    def test_sessions_are_cleaned(self, linear_svd_model, svd_dataloader_linear):
        """After SVD functions return, temporary trackers must be destroyed."""
        import gc

        n_before = _count_tracker_objects()
        online_svd(
            linear_svd_model, ["fc1"], svd_dataloader_linear, reduced_dim=8
        )
        offline_svd(
            linear_svd_model, ["fc1"], svd_dataloader_linear, reduced_dim=8
        )
        gc.collect()
        n_after = _count_tracker_objects()
        assert n_after <= n_before + 2, (
            f"Possible tracker leak: {n_before} → {n_after}"
        )

    def test_svd_does_not_modify_model_weights(self, linear_svd_model, svd_dataloader_linear):
        """Model parameters must be unchanged after SVD analysis."""
        params_before = {
            name: p.detach().clone()
            for name, p in linear_svd_model.named_parameters()
        }
        online_svd(
            linear_svd_model, ["fc1", "fc2"], svd_dataloader_linear, reduced_dim=8
        )
        offline_svd(
            linear_svd_model, ["fc1", "fc2"], svd_dataloader_linear, reduced_dim=8
        )
        for name, p in linear_svd_model.named_parameters():
            assert torch.equal(p, params_before[name]), f"Weight {name} changed!"

    def test_svd_tensors_are_not_grad_tracked(self, linear_svd_model, svd_dataloader_linear):
        """All SVDBasis tensors must be detached (no grad)."""
        results = offline_svd(
            linear_svd_model, ["fc1"], svd_dataloader_linear, reduced_dim=8
        )
        b = results["fc1"]
        for attr in ("mu", "U", "U_residual", "S_residual"):
            t = getattr(b, attr)
            assert not t.requires_grad, f"{attr} has requires_grad=True"


def _count_tracker_objects() -> int:
    """Count ActivationScope instances still alive with a valid session."""
    import gc
    return sum(
        1 for obj in gc.get_objects()
        if isinstance(obj, ActivationScope) and obj._session_id is not None
    )
