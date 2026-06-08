"""Integration tests for attach/detach lifecycle, clear(), and context manager."""

import pytest
import torch

from activationscope import ActivationScope


class TestAttachRemoveLifecycle:
    """attach() → forward → remove() full cycle."""

    def test_attach_forward_remove(self, simple_linear_model):
        t = ActivationScope()
        t.attach(simple_linear_model)

        x = torch.randn(2, 10)
        _ = simple_linear_model(x)

        acts = t.activations
        assert len(acts) > 0
        assert "fc1" in acts

        # Remove destroys the session
        t.remove()
        with pytest.raises(RuntimeError, match="session already destroyed"):
            _ = t.session_id
        with pytest.raises(RuntimeError, match="session already destroyed"):
            _ = t.activations

    def test_remove_is_idempotent(self):
        """Calling remove() multiple times does not crash."""
        t = ActivationScope()
        model = torch.nn.Linear(3, 4)
        t.attach(model)
        t.remove()
        t.remove()  # should be safe no-op

    def test_activations_populated_after_attach(self, simple_linear_model):
        t = ActivationScope()
        t.attach(simple_linear_model)
        x = torch.randn(2, 10)
        _ = simple_linear_model(x)
        acts = t.activations
        assert any(len(tensors) == 1 for tensors in acts.values())
        t.remove()


class TestClearBehavior:
    """clear() resets activations but keeps hooks attached."""

    def test_clear_resets_activations(self, simple_linear_model):
        t = ActivationScope()
        t.attach(simple_linear_model)

        x = torch.randn(2, 10)
        _ = simple_linear_model(x)
        acts_before = t.activations
        assert len(acts_before) > 0

        # Clear should reset activations
        t.clear()
        acts_after = t.activations
        assert all(len(v) == 0 for v in acts_after.values()), \
            "Activations should be empty after clear()"

    def test_clear_keeps_hooks_attached(self, simple_linear_model):
        """After clear(), a new forward pass still captures activations."""
        t = ActivationScope()
        t.attach(simple_linear_model)
        _ = simple_linear_model(torch.randn(2, 10))
        t.clear()

        # Forward again — hooks should still fire
        _ = simple_linear_model(torch.randn(2, 10))
        acts = t.activations
        assert any(len(v) > 0 for v in acts.values())
        t.remove()


class TestTrackContextManager:
    """track() context manager auto-teardown behavior."""

    def test_track_auto_removes(self, simple_linear_model):
        """After track() exits, hooks are removed and session destroyed."""
        t = ActivationScope()
        with t.track(simple_linear_model):
            x = torch.randn(2, 10)
            _ = simple_linear_model(x)
            acts = t.activations
            assert len(acts) > 0

    def test_track_teardown_on_exception(self, simple_linear_model):
        """Even if an exception is raised inside track(), teardown happens."""
        t = ActivationScope()
        with pytest.raises(ValueError):
            with t.track(simple_linear_model):
                _ = simple_linear_model(torch.randn(2, 10))
                raise ValueError("simulated error during tracking")

    def test_track_yields_self(self, simple_linear_model):
        """The context manager yields the tracker instance."""
        t = ActivationScope()
        with t.track(simple_linear_model) as yield_t:
            assert yield_t is t


class TestCaptureParameters:
    """capture_parameters returns detached CPU clones."""

    def test_returns_dict_of_tensors(self, simple_linear_model):
        t = ActivationScope()
        snapshot = t.capture_parameters(simple_linear_model)
        assert isinstance(snapshot, dict)
        for v in snapshot.values():
            assert isinstance(v, torch.Tensor)

    def test_all_params_detached(self, simple_linear_model):
        """All returned tensors must be detached (no grad)."""
        t = ActivationScope()
        model_with_grad = simple_linear_model
        for p in model_with_grad.parameters():
            p.requires_grad = True

        snapshot = t.capture_parameters(model_with_grad)
        for v in snapshot.values():
            assert not v.requires_grad, "Snapshot tensor should be detached"

    def test_all_params_on_cpu(self):
        """All returned tensors must live on CPU."""
        t = ActivationScope()
        model = torch.nn.Linear(4, 8)
        snapshot = t.capture_parameters(model)
        for v in snapshot.values():
            assert v.device.type == "cpu"

    def test_filter_by_layers(self):
        class LabeledModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(8, 16)
                self.decoder = torch.nn.Linear(16, 4)

            def forward(self, x):
                return self.decoder(self.encoder(x))

        model = LabeledModel()
        t = ActivationScope()
        snapshot = t.capture_parameters(model, layers=["encoder*"])
        for key in snapshot:
            assert "encoder" in key, f"Should only have encoder params, got {key}"


class TestMultipleAttachCycles:
    """Multiple attach → remove cycles work correctly."""

    def test_reuse_after_remove(self, simple_linear_model):
        """After remove(), .attach() can be called again on same tracker."""
        for _ in range(3):
            t = ActivationScope()
            t.attach(simple_linear_model)
            x = torch.randn(2, 10)
            _ = simple_linear_model(x)
            acts = t.activations
            assert len(acts) > 0
            t.remove()

    def test_attach_after_destroy_raises(self):
        t = ActivationScope()
        t.remove()
        model = torch.nn.Linear(3, 4)
        with pytest.raises(RuntimeError, match="session already destroyed"):
            t.attach(model)


class TestRegisterReduction:
    """register_reduction with per-layer and global patterns."""

    def test_global_reduction(self, simple_linear_model):
        """Global reduction (layers=None) is settable without error."""
        t = ActivationScope()
        t.register_reduction(ActivationScope.for_max(), layers=None)
        t.remove()

    def test_per_layer_reduction(self, simple_linear_model):
        """Per-layer reduction with explicit layer list works."""
        t = ActivationScope()
        t.register_reduction(ActivationScope.for_mean(), layers=["fc1"])
        t.attach(simple_linear_model)
        t.remove()

    def test_for_max_returns_callable(self):
        fn = ActivationScope.for_max()
        x = torch.randn(4, 8)
        result = fn(x)
        assert result.shape == (8,)

    def test_for_mean_returns_callable(self):
        fn = ActivationScope.for_mean()
        x = torch.randn(4, 8)
        result = fn(x)
        assert result.shape == (8,)


class TestAttachAfterDestroy:
    """Accessing properties or calling methods on a destroyed session raises."""

    def test_attach_raises_on_destroyed(self):
        t = ActivationScope()
        t.remove()
        with pytest.raises(RuntimeError, match="session already destroyed"):
            t.attach(torch.nn.Linear(2, 3))

    def test_register_reduction_raises_on_destroyed(self):
        t = ActivationScope()
        t.remove()
        with pytest.raises(RuntimeError, match="session already destroyed"):
            t.register_reduction(lambda x: x.mean(), layers=["fc"])
