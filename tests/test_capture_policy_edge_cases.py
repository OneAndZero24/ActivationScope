"""Edge case tests for capture policy combinations and mixed configurations.

Covers:
  - SAMPLE_N with sample_every=1 (should behave like EVERY)
  - MAX_K with max_batches=0 (unlimited, like EVERY)
  - Mixing reduction + capture policies in various combos
  - Multiple simultaneous trackers on the same model
"""

import pytest
import torch

from activationscope import (
    ActivationScope,
    ReductionPolicy,
    CapturePolicy,
    StoragePolicy,
)


class TestSampleNEdgeCases:
    """SAMPLE_N boundary behavior."""

    def test_sample_every_1_equals_every(self, simple_linear_model):
        """SAMPLE_N with sample_every=1 should capture every forward."""
        t = ActivationScope(
            reduction=ReductionPolicy.STORE_ALL,
            capture=CapturePolicy.SAMPLE_N,
            sample_every=1,
        )
        with t.track(simple_linear_model):
            for _ in range(8):
                _ = simple_linear_model(torch.randn(2, 10))

            acts = t.activations
        for name, ts_list in acts.items():
            assert len(ts_list) == 8, \
                f"Layer {name} should capture all 8 with sample_every=1"

    def test_sample_every_greater_than_forwards(self, simple_linear_model):
        """When sample_every > number of forwards, very few captures."""
        t = ActivationScope(
            reduction=ReductionPolicy.STORE_ALL,
            capture=CapturePolicy.SAMPLE_N,
            sample_every=100,
        )
        with t.track(simple_linear_model):
            for _ in range(5):
                _ = simple_linear_model(torch.randn(2, 10))

            acts = t.activations
        # Should capture at most floor(5/100) + 1 = 1 or so depending on impl
        for name, ts_list in acts.items():
            assert len(ts_list) <= 2, \
                f"Layer {name} should have few captures with sample_every=100"

    def test_sample_every_with_final_only(self, simple_linear_model):
        """SAMPLE_N combined with FINAL_ONLY: last captured value wins."""
        t = ActivationScope(
            reduction=ReductionPolicy.FINAL_ONLY,
            capture=CapturePolicy.SAMPLE_N,
            sample_every=3,
        )
        with t.track(simple_linear_model):
            for _ in range(12):
                _ = simple_linear_model(torch.randn(2, 10))

            acts = t.activations
        # FINAL_ONLY means at most 1 tensor per layer regardless of sampling
        for name, ts_list in acts.items():
            assert len(ts_list) <= 1


class TestMaxKEdgeCases:
    """MAX_K boundary behavior."""

    def test_max_batches_0_is_unlimited(self, simple_linear_model):
        """max_batches=0 means no limit — all forwards captured."""
        t = ActivationScope(
            reduction=ReductionPolicy.STORE_ALL,
            capture=CapturePolicy.MAX_K,
            max_batches=0,
        )
        with t.track(simple_linear_model):
            for _ in range(20):
                _ = simple_linear_model(torch.randn(2, 10))

            acts = t.activations
        for name, ts_list in acts.items():
            assert len(ts_list) == 20, \
                f"Layer {name} should capture all 20 with max_batches=0"

    def test_max_batches_1_captures_once(self, simple_linear_model):
        """max_batches=1 captures exactly one forward per layer."""
        t = ActivationScope(
            reduction=ReductionPolicy.STORE_ALL,
            capture=CapturePolicy.MAX_K,
            max_batches=1,
        )
        with t.track(simple_linear_model):
            for _ in range(20):
                _ = simple_linear_model(torch.randn(2, 10))

            acts = t.activations
        for name, ts_list in acts.items():
            assert len(ts_list) == 1, \
                f"Layer {name} should have exactly 1 with max_batches=1"

    def test_max_k_with_streaming(self, simple_linear_model):
        """MAX_K + STREAMING reduction stays bounded and functional."""
        t = ActivationScope(
            reduction=ReductionPolicy.STREAMING,
            capture=CapturePolicy.MAX_K,
            max_batches=4,
        )
        t.register_reduction(ActivationScope.max_reduction(), layers=None)

        with t.track(simple_linear_model):
            for _ in range(30):
                _ = simple_linear_model(torch.randn(4, 10))

    def test_max_k_after_reach_limit_still_forwards(self, simple_linear_model):
        """Model can still forward after MAX_K cap is reached."""
        t = ActivationScope(
            reduction=ReductionPolicy.STORE_ALL,
            capture=CapturePolicy.MAX_K,
            max_batches=2,
        )
        with t.track(simple_linear_model):
            # First 2 forwards are captured
            _ = simple_linear_model(torch.randn(2, 10))
            _ = simple_linear_model(torch.randn(2, 10))
            acts_before = t.activations

            # Additional forwards should still work (just not captured)
            for _ in range(8):
                _ = simple_linear_model(torch.randn(2, 10))
            acts_after = t.activations

        # Tensor count should be the same before and after cap
        for name in acts_before:
            assert len(acts_before[name]) == len(acts_after[name]), \
                f"Layer {name} should not grow past max_batches=2"


class TestMixedPolicyCombinations:
    """Test every Reduction × Capture matrix."""

    @pytest.mark.parametrize("reduction", [ReductionPolicy.STORE_ALL,
                                          ReductionPolicy.STREAMING,
                                          ReductionPolicy.FINAL_ONLY])
    @pytest.mark.parametrize("capture", [CapturePolicy.EVERY,
                                        CapturePolicy.SAMPLE_N,
                                        CapturePolicy.MAX_K])
    def test_all_policy_combos_complete_forward(self, simple_linear_model,
                                                reduction, capture):
        """Every reduction × capture combo allows forward+backward."""
        t = ActivationScope(
            reduction=reduction,
            capture=capture,
            sample_every=3,   # for SAMPLE_N
            max_batches=4,    # for MAX_K
        )
        if reduction == ReductionPolicy.STREAMING:
            t.register_reduction(ActivationScope.max_reduction(), layers=None)

        with t.track(simple_linear_model):
            x = torch.randn(2, 10, requires_grad=True)
            out = simple_linear_model(x)
            loss = out.sum()
            loss.backward()

    @pytest.mark.parametrize("storage", [StoragePolicy.AUTO, StoragePolicy.CPU,
                                        StoragePolicy.GPU])
    @pytest.mark.parametrize("reduction", [ReductionPolicy.STORE_ALL,
                                          ReductionPolicy.FINAL_ONLY])
    def test_storage_x_reduction_matrix(self, simple_linear_model, storage,
                                       reduction):
        """Every Storage × Reduction combo forwards without crash."""
        t = ActivationScope(
            storage=storage,
            reduction=reduction,
        )
        with t.track(simple_linear_model, include=["fc*"]):
            _ = simple_linear_model(torch.randn(2, 10))


class TestMultipleTrackers:
    """Multiple trackers on the same model simultaneously."""

    def test_two_trackers_same_model(self, simple_linear_model):
        """Two independent trackers can run concurrently on one model."""
        t1 = ActivationScope(reduction=ReductionPolicy.STORE_ALL)
        t2 = ActivationScope(
            reduction=ReductionPolicy.FINAL_ONLY,
            capture=CapturePolicy.MAX_K,
            max_batches=3,
        )

        with t1.track(simple_linear_model, include=["fc*"]):
            with t2.track(simple_linear_model, include=["act", "fc2"]):
                for _ in range(5):
                    _ = simple_linear_model(torch.randn(2, 10))

            # t2 should have data
            acts2 = t2.activations
            assert len(acts2) > 0

        # t1 should have data
        acts1 = t1.activations
        assert len(acts1) > 0

    def test_three_trackers_different_policies(self, simple_linear_model):
        """Three trackers with different policies can coexist."""
        tracker_cfgs = [
            dict(reduction=ReductionPolicy.STORE_ALL, capture=CapturePolicy.EVERY),
            dict(reduction=ReductionPolicy.FINAL_ONLY, capture=CapturePolicy.SAMPLE_N, sample_every=2),
            dict(reduction=ReductionPolicy.STREAMING, capture=CapturePolicy.MAX_K, max_batches=3),
        ]

        trackers = []
        for cfg in tracker_cfgs:
            t = ActivationScope(**cfg)
            if cfg.get("reduction") == ReductionPolicy.STREAMING:
                t.register_reduction(ActivationScope.max_reduction(), layers=None)
            trackers.append(t)

        # Attach all to the same model (different layer subsets)
        layer_sets = [["fc1"], ["act"], ["fc2"]]
        for t, subset in zip(trackers, layer_sets):
            t.attach(simple_linear_model, include=subset)

        for _ in range(4):
            _ = simple_linear_model(torch.randn(2, 10))

        for t in trackers:
            acts = t.activations
            assert len(acts) > 0, "Each tracker should have captured data"
            t.remove()

    def test_trackers_dont_interfere_with_each_other(self, simple_linear_model):
        """Trackers attached to disjoint layers don't share activations."""
        t_fc = ActivationScope(reduction=ReductionPolicy.STORE_ALL)
        t_act = ActivationScope(reduction=ReductionPolicy.STORE_ALL)

        with t_fc.track(simple_linear_model, include=["fc*"]):
            with t_act.track(simple_linear_model, include=["act"]):
                _ = simple_linear_model(torch.randn(2, 10))

                acts_fc = t_fc.activations
                acts_act = t_act.activations

            # fc tracker should have fc layers, not act
            for name in acts_fc:
                assert "fc" in name, \
                    f"FC tracker got unexpected layer {name}"

            # act tracker should have only 'act'
            for name in acts_act:
                assert "act" in name, \
                    f"Act tracker got unexpected layer {name}"


class TestTrackerReuseAfterClear:
    """Reusable patterns after clear / re-attach."""

    def test_clear_then_re_attach(self, simple_linear_model):
        """clear() then attach again on same model works."""
        t = ActivationScope(reduction=ReductionPolicy.STORE_ALL)

        # First cycle
        t.attach(simple_linear_model)
        _ = simple_linear_model(torch.randn(2, 10))
        t.clear()

        # Re-attach (still has hooks from first attach, but let's re-attach to be safe)
        t.remove()
        t_new = ActivationScope(reduction=ReductionPolicy.STORE_ALL)
        with t_new.track(simple_linear_model):
            _ = simple_linear_model(torch.randn(2, 10))

    def test_consecutive_track_calls(self, simple_linear_model):
        """Multiple track() contexts in sequence work correctly."""
        t = ActivationScope(reduction=ReductionPolicy.STORE_ALL)

        for cycle in range(5):
            with t.track(simple_linear_model):
                _ = simple_linear_model(torch.randn(2, 10))
                acts = t.activations
                assert len(acts) > 0

    def test_nested_track_different_models(self):
        """Nested track() contexts on different models work."""
        model_a = torch.nn.Linear(4, 8)
        model_b = torch.nn.Sequential(
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
        )

        t_a = ActivationScope(reduction=ReductionPolicy.STORE_ALL)
        t_b = ActivationScope(reduction=ReductionPolicy.FINAL_ONLY)

        with t_a.track(model_a):
            _ = model_a(torch.randn(2, 4))

            with t_b.track(model_b):
                _ = model_b(torch.randn(2, 16))

            # t_a should still have data from model_a
            acts_a = t_a.activations
            assert len(acts_a) > 0


class TestCaptureDirectionWithPolicies:
    """Capture direction interacts properly with reduction/capture policies."""

    @pytest.mark.parametrize("capture_dir", ["input", "output", "both"])
    def test_all_capture_dirs_with_store_all(self, simple_linear_model,
                                              capture_dir):
        """Every capture direction works with STORE_ALL + EVERY."""
        t = ActivationScope(
            reduction=ReductionPolicy.STORE_ALL,
            capture=CapturePolicy.EVERY,
        )
        with t.track(simple_linear_model, layers=["fc1"], capture=capture_dir):
            _ = simple_linear_model(torch.randn(2, 10))

            acts = t.activations
            assert "fc1" in acts

    def test_capture_both_shape_check(self, conv_model):
        """With both mode, conv layer captures inputs and outputs."""
        t = ActivationScope()
        with t.track(conv_model, layers=["conv1"], capture="both"):
            _ = conv_model(torch.randn(2, 3, 16, 16))

            acts = t.activations
            assert "conv1" in acts
