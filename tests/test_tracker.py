"""Tests for ActivationScope core module."""

import pytest
import torch

from activationscope.tracker import (
    ActivationScope,
    get_max_stats,
    get_min_stats,
    get_mean_stats,
    clear_online_stats,
)


@pytest.fixture(autouse=True)
def _reset_online_stats():
    """Reset C++ global online stats before and after every test for isolation."""
    clear_online_stats()
    yield
    clear_online_stats()


class TestTrackerStoreMode:
    """Test store mode functionality without C++ extension."""

    def test_store_mode_captures_activations(self):
        from activationscope.tracker import ActivationScope

        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5),
        )
        tracker = ActivationScope(mode="store")
        target_layers = {"layer0": model[0], "layer1": model[1], "layer2": model[2]}
        tracker.attach(model, target_layers)

        x = torch.randn(2, 10)
        _ = model(x)

        assert len(tracker.activations) == 3
        assert "layer0" in tracker.activations
        assert tracker.activations["layer0"].shape == (2, 20)
        tracker.clear()

    def test_clear_releases_memory(self):
        from activationscope.tracker import ActivationScope

        model = torch.nn.Linear(10, 20)
        tracker = ActivationScope(mode="store")
        tracker.attach(model, {"fc": model})

        x = torch.randn(2, 10, requires_grad=True)
        _ = model(x)

        assert len(tracker.activations) > 0
        tracker.clear()
        assert len(tracker.activations) == 0

    def test_context_manager_full_teardown(self):
        from activationscope.tracker import ActivationScope

        model = torch.nn.Linear(10, 20)
        tracker = ActivationScope(mode="store")

        with tracker.track(model, {"fc": model}) as t:
            x = torch.randn(2, 10)
            _ = model(x)
            assert "fc" in t.activations
            assert t.activations["fc"].shape == (2, 20)

        # After context exit, hooks are removed AND activations are cleared (full teardown)
        assert len(tracker.activations) == 0
        assert len(tracker._handles) == 0

    def test_context_manager_accumulates_across_batches(self):
        """Activations accumulate across multiple forward passes inside a single track() call."""
        from activationscope.tracker import ActivationScope

        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5),
        )
        tracker = ActivationScope(mode="store")
        target_layers = {"layer0": model[0], "layer2": model[2]}

        with tracker.track(model, target_layers) as t:
            # First batch
            x1 = torch.ones(2, 10) * 3.0
            _ = model(x1)
            assert "layer0" in t.activations
            assert "layer2" in t.activations

            # Second batch (activations from last forward pass are kept)
            x2 = torch.zeros(2, 10)
            _ = model(x2)

            # layer0 activations should now reflect the SECOND forward pass
            # (all zeros passed through Linear → output is just bias, not the first batch's data)
            assert t.activations["layer0"].shape == (2, 20)

            # Verify activations did NOT auto-clear between batches
            assert len(t.activations) == 2

        # After exit: everything is cleaned up
        assert len(tracker._activations) == 0
        assert len(tracker._handles) == 0


class TestTrackerAutoDetection:
    """Test automatic layer detection."""

    def test_auto_detect_layers(self):
        from activationscope.tracker import ActivationScope

        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5),
        )
        tracker = ActivationScope(mode="store")
        tracker.attach(model)  # No layers dict passed

        x = torch.randn(2, 10)
        _ = model(x)

        assert len(tracker.activations) == 3
        tracker.clear()


class TestInvalidMode:
    """Test that invalid mode raises an error."""

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            ActivationScope(mode="invalid_mode")


class TestTrackerOnlineMode:
    """Test online statistics modes (max / min / mean) against C++ backend."""

    def test_online_max_tracks_stats(self):
        """Max stats accumulate element-wise across forward passes; shape is dim-0 reduced."""
        model = torch.nn.Linear(8, 4)
        tracker = ActivationScope(mode="online_max")
        tracker.attach(model, {"fc": model})

        # First forward pass
        x1 = torch.tensor(
            [[-1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0]] * 2
        )  # [B=2, D=8]
        _ = model(x1)

        stats_after_1 = get_max_stats()
        assert "fc" in stats_after_1
        first_shape = stats_after_1["fc"].shape

        # Second forward pass with a different input that has larger activations
        x2 = torch.tensor([[[-99.0] * 8], [[-98.0] * 8]])  # [B=2, D=8]
        _ = model(x2)

        stats_after_2 = get_max_stats()
        assert "fc" in stats_after_2

        # Shape should be [D_out] (batch dim reduced)
        assert len(first_shape) == 1
        assert first_shape[0] == 4  # D_out of Linear(8, 4)

        # Max values should only ever grow or stay the same (element-wise max)
        assert torch.all(stats_after_2["fc"] >= stats_after_1["fc"])

        tracker.remove()

    def test_online_min_tracks_stats(self):
        """Min stats accumulate element-wise across forward passes; shape is dim-0 reduced."""
        model = torch.nn.Linear(8, 4)
        tracker = ActivationScope(mode="online_min")
        tracker.attach(model, {"fc": model})

        # First forward pass
        x1 = torch.randn(2, 8)
        _ = model(x1)

        stats_after_1 = get_min_stats()
        assert "fc" in stats_after_1
        assert len(stats_after_1["fc"].shape) == 1 and stats_after_1["fc"].shape[0] == 4

        # Second forward pass with more extreme negative values
        x2 = torch.randn(2, 8) * -5.0
        _ = model(x2)

        stats_after_2 = get_min_stats()
        assert "fc" in stats_after_2

        # Min values should only ever shrink or stay the same (element-wise min)
        assert torch.all(stats_after_2["fc"] <= stats_after_1["fc"])
        tracker.remove()

    def test_online_mean_tracks_float64(self):
        """Mean uses float64 for precision and converges toward true batch mean."""
        model = torch.nn.Linear(8, 4)
        tracker = ActivationScope(mode="online_mean")
        tracker.attach(model, {"fc": model})

        # Use a constant input so we can predict the exact output
        with torch.no_grad():
            init_weight = model.weight.clone()
            init_bias = model.bias.clone()
            model.weight.copy_(torch.ones(4, 8))
            model.bias.copy_(torch.zeros(4))

        # Feed all-ones input: each unit sums to 8.0 per sample → mean should be 8.0
        x = torch.ones(2, 8)
        _ = model(x)

        stats = get_mean_stats()
        assert "fc" in stats
        assert stats["fc"].dtype == torch.float64
        assert len(stats["fc"].shape) == 1 and stats["fc"].shape[0] == 4
        assert torch.allclose(
            stats["fc"],
            torch.tensor([8.0, 8.0, 8.0, 8.0], dtype=torch.float64),
        )

        # Restore weights (cleanup)
        with torch.no_grad():
            model.weight.copy_(init_weight)
            model.bias.copy_(init_bias)
        tracker.remove()

    def test_online_max_conv2d_shape(self):
        """Max stats for Conv2d reduce over batch dim → shape [C_out, H, W]."""
        model = torch.nn.Conv2d(
            3, 4, kernel_size=3, padding=1
        )  # preserves spatial size
        tracker = ActivationScope(mode="online_max")
        tracker.attach(model, {"conv": model})

        x = torch.randn(2, 3, 8, 8)
        _ = model(x)

        stats = get_max_stats()
        assert "conv" in stats
        expected_shape = (4, 8, 8)  # [C_out, H, W] — batch dim reduced
        assert stats["conv"].shape == expected_shape
        tracker.remove()

    def test_clear_online_stats_resets(self):
        """clear_online_stats wipes all accumulated max / min / mean data."""
        model = torch.nn.Linear(4, 2)
        tracker_max = ActivationScope(mode="online_max")
        tracker_min = ActivationScope(mode="online_min")
        tracker_mean = ActivationScope(mode="online_mean")
        for t in (tracker_max, tracker_min, tracker_mean):
            t.attach(model, {"fc": model})

        # Populate all three stat maps
        x = torch.randn(2, 4)
        _ = model(x)

        assert len(get_max_stats()) > 0
        assert len(get_min_stats()) > 0
        assert len(get_mean_stats()) > 0

        clear_online_stats()

        assert get_max_stats() == {}
        assert get_min_stats() == {}
        assert get_mean_stats() == {}

        for t in (tracker_max, tracker_min, tracker_mean):
            t.remove()
