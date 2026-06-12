"""Smoke tests — import, instantiation, minimal forward pass.

These run first to confirm the package is loadable before any deeper
exercise of individual modules begins.
"""

import pytest
import torch
from activationscope import (
    ActivationScope,
    StoragePolicy,
    ReductionPolicy,
    CapturePolicy,
)


class TestImportSmoke:
    """Basic import checks."""

    def test_import_package(self):
        """The top-level package must be importable without error."""
        import activationscope  # noqa: F811
        assert hasattr(activationscope, "ActivationScope")

    def test_all_exports_exist(self):
        """Every symbol in __all__ is actually defined."""
        import activationscope
        for name in activationscope.__all__:
            assert hasattr(activationscope, name), f"Missing export: {name}"

    def test_module_has_C_backend(self):
        """The native _C submodule must be importable."""
        import activationscope._C as _C  # noqa: N811
        assert callable(getattr(_C, "session_create"))

    def test_tracker_importable(self):
        """Internal modules expose expected symbols."""
        from activationscope.utils import (
            parse_capture_dir,
            select_layers,
            pattern_or_identity,
        )  # noqa: F401
        pass  # If imports succeed, the test passes


class TestInstantiationSmoke:
    """Create tracker objects with various parameter combos."""

    def test_default_construction(self):
        """ActivationScope builds with all defaults without error."""
        t = ActivationScope()
        assert isinstance(t.session_id, int)
        assert t.session_id > 0
        t.remove()

    def test_explicit_policies(self):
        """ActivationScope accepts every enum combo explicitly."""
        t = ActivationScope(
            storage=StoragePolicy.CPU,
            reduction=ReductionPolicy.STORE_ALL,
            capture=CapturePolicy.EVERY,
            sample_every=1,
            max_batches=0,
            auto_cpu_threshold_bytes=2_000_000,
            use_pinned=False,
        )
        assert isinstance(t.session_id, int)
        t.remove()

    def test_all_storage_combos(self):
        """Every StoragePolicy produces a valid tracker."""
        for sp in (StoragePolicy.AUTO, StoragePolicy.CPU, StoragePolicy.GPU):
            t = ActivationScope(storage=sp)
            assert isinstance(t.session_id, int)
            t.remove()

    def test_all_reduction_combos(self):
        """Every ReductionPolicy produces a valid tracker."""
        for rp in (ReductionPolicy.STORE_ALL, ReductionPolicy.STREAMING, ReductionPolicy.FINAL_ONLY):
            t = ActivationScope(reduction=rp)
            assert isinstance(t.session_id, int)
            t.remove()

    def test_all_capture_combos(self):
        """Every CapturePolicy produces a valid tracker."""
        for cp in (CapturePolicy.EVERY, CapturePolicy.SAMPLE_N, CapturePolicy.MAX_K):
            t = ActivationScope(capture=cp, sample_every=2, max_batches=5)
            assert isinstance(t.session_id, int)
            t.remove()

    def test_use_pinned(self):
        """use_pinned=True is accepted cleanly."""
        t = ActivationScope(use_pinned=True)
        assert t._session_id is not None
        t.remove()


class TestMinimalForwardPass:
    """Smallest end-to-end flow through track()."""

    def test_simple_forward_through_track(self, simple_linear_model):
        """A tiny model inside a track context manager yields non-empty activations."""
        tracker = ActivationScope(storage=StoragePolicy.CPU)
        with tracker.track(simple_linear_model):
            x = torch.randn(2, 10)
            _ = simple_linear_model(x)

            acts = tracker.activations
            assert isinstance(acts, dict)
            assert len(acts) >= 3  # fc1, act, fc2 at minimum

    def test_activations_before_forward_is_empty(self):
        """Before any forward pass inside track(), activations dict is empty."""
        model = torch.nn.Linear(3, 4)
        tracker = ActivationScope()
        with tracker.track(model):
            acts = tracker.activations
            assert acts == {}, f"Expected empty dict before forward, got {acts}"

    def test_activations_populated_after_forward(self, simple_linear_model):
        """After one forward pass, every tracked layer has an entry."""
        model = simple_linear_model
        tracker = ActivationScope()
        with tracker.track(model) as t:
            x = torch.randn(2, 10)
            _ = model(x)

            acts = t.activations
            assert "fc1" in acts
            assert "act" in acts
            assert "fc2" in acts
            # Each value is a list of tensors (STORE_ALL default)
            for layer_name, tensor_list in acts.items():
                assert isinstance(tensor_list, list)
                assert len(tensor_list) == 1  # one forward pass so far

    def test_session_id_positive_int(self):
        """session_id property returns int > 0."""
        tracker = ActivationScope()
        sid = tracker.session_id
        assert isinstance(sid, int)
        assert sid > 0
        tracker.remove()
