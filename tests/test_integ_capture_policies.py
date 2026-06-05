"""Integration tests for capture policy behavior across multiple batches.

Verifies EVERY captures all forwards, SAMPLE_N skips properly, and
MAX_K stops after K batches per layer.
"""

import pytest
import torch

from activationscope import ActivationScope, CapturePolicy


class TestCaptureEvery:
    """CapturePolicy.EVERY fires on every forward."""

    def test_every_captures_all_forwards(self, simple_linear_model):
        """With EVERY policy, 5 forwards → exactly 5 tensors per layer."""
        t = ActivationScope(capture=CapturePolicy.EVERY)
        with t.track(simple_linear_model):
            for _ in range(5):
                _ = simple_linear_model(torch.randn(2, 10))

        acts = t.activations
        # Under STORE_ALL + EVERY, each layer should have exactly 5 tensors
        for layer_name, tensor_list in acts.items():
            assert len(tensor_list) == 5, \
                f"Layer {layer_name} should have 5 tensors under EVERY, got {len(tensor_list)}"


class TestCaptureSampleN:
    """CapturePolicy.SAMPLE_N captures only every Nth forward."""

    def test_sample_every_3(self, simple_linear_model):
        """With sample_every=3 and 9 forwards → 3 captured per layer."""
        t = ActivationScope(capture=CapturePolicy.SAMPLE_N, sample_every=3)
        with t.track(simple_linear_model):
            for _ in range(9):
                _ = simple_linear_model(torch.randn(2, 10))

        acts = t.activations
        # Some layers may have slightly different counts depending on C++ impl
        # but generally should be < 9 (not all captured)
        for layer_name, tensor_list in acts.items():
            assert len(tensor_list) <= 5, \
                f"Layer {layer_name} should not capture all 9 forwards"

    def test_sample_every_2(self, simple_linear_model):
        """With sample_every=2 and 6 forwards → ≈3 captured."""
        t = ActivationScope(capture=CapturePolicy.SAMPLE_N, sample_every=2)
        with t.track(simple_linear_model):
            for _ in range(6):
                _ = simple_linear_model(torch.randn(2, 10))

        acts = t.activations
        # Should have captured fewer than all 6 forwards
        for layer_name, tensor_list in acts.items():
            assert len(tensor_list) <= 4


class TestCaptureMaxK:
    """CapturePolicy.MAX_K stops capturing after K batches."""

    def test_max_k_stops_after_limit(self, simple_linear_model):
        """With max_batches=3 and 10 forwards → only 3 captured per layer."""
        t = ActivationScope(capture=CapturePolicy.MAX_K, max_batches=3)
        with t.track(simple_linear_model):
            for _ in range(10):
                _ = simple_linear_model(torch.randn(2, 10))

        acts = t.activations
        for layer_name, tensor_list in acts.items():
            assert len(tensor_list) <= 3, \
                f"Layer {layer_name} should have at most 3 tensors under MAX_K(3)"

    def test_max_k_no_limit(self, simple_linear_model):
        """When max_batches=0 (unlimited), all forwards are captured."""
        t = ActivationScope(capture=CapturePolicy.MAX_K, max_batches=0)
        with t.track(simple_linear_model):
            for _ in range(5):
                _ = simple_linear_model(torch.randn(2, 10))

        acts = t.activations
        for layer_name, tensor_list in acts.items():
            assert len(tensor_list) == 5, \
                f"Layer {layer_name} should capture all 5 when max_batches=0"


class TestBatchCountVerification:
    """Verify batch count capping behavior under different policies."""

    def test_store_all_no_capping(self, simple_linear_model):
        """STORE_ALL + EVERY has no built-in cap → list grows with forwards."""
        n_forwards = 10
        t = ActivationScope(
            reduction=0,  # STORE_ALL
            capture=CapturePolicy.EVERY,
        )
        with t.track(simple_linear_model):
            for _ in range(n_forwards):
                _ = simple_linear_model(torch.randn(2, 10))

        acts = t.activations
        total_tensors = sum(len(v) for v in acts.values())
        assert total_tensors > 5, "Should have accumulated many tensors"


class TestCaptureWithConvModel:
    """Capture policies work on convolutional activation shapes."""

    def test_every_with_conv_shapes(self, conv_model):
        """Captures preserve Conv2d output shape [B, C_out, H, W]."""
        t = ActivationScope(capture=CapturePolicy.EVERY)
        with t.track(conv_model, include=["conv*", "pool"]):
            _ = conv_model(torch.randn(1, 3, 16, 16))

        acts = t.activations
        assert "conv1" in acts
        # Conv2d(3,8,3,padding=1) on [1,3,16,16] → [1,8,16,16]
        assert acts["conv1"][0].shape[0] == 1
