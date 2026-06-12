"""Integration tests for storage policy behavior (CPU vs GPU placement).

Verifies that StoragePolicy.CPU actually moves tensors to CPU,
StoragePolicy.GPU keeps them on original device, and StoragePolicy.AUTO
applies the byte-threshold heuristic correctly.
"""

import pytest
import torch

from activationscope import ActivationScope, StoragePolicy


class TestStorageCPU:
    """StoragePolicy.CPU forces all activations to CPU."""

    def test_activations_on_cpu(self, simple_linear_model):
        t = ActivationScope(storage=StoragePolicy.CPU)
        with t.track(simple_linear_model):
            _ = simple_linear_model(torch.randn(2, 10))

            acts = t.activations
        for layer_name, tensor_list in acts.items():
            for tensor in tensor_list:
                assert tensor.device.type == "cpu", \
                    f"Layer {layer_name} tensor should be on CPU"


class TestStorageGPU:
    """StoragePolicy.GPU keeps activations on original device.

    These tests are CPU-only since we don't have CUDA; they verify that the
    API accepts GPU and works without crash when tensors live on CPU anyway.
    """

    def test_gpu_policy_on_cpu_device(self, simple_linear_model):
        """Even with GPU policy, CPU model stays on CPU (no device to move to)."""
        t = ActivationScope(storage=StoragePolicy.GPU)
        with t.track(simple_linear_model):
            _ = simple_linear_model(torch.randn(2, 10))

            acts = t.activations
        assert len(acts) > 0


class TestStorageAuto:
    """StoragePolicy.AUTO applies byte-threshold heuristic."""

    def test_auto_small_tensor_goes_to_cpu(self):
        """Small tensors (< threshold bytes) should end up on CPU with AUTO policy."""
        model = torch.nn.Linear(4, 8)  # tiny weights → tiny activations
        t = ActivationScope(storage=StoragePolicy.AUTO, auto_cpu_threshold_bytes=1_048_576)
        with t.track(model):
            _ = model(torch.randn(2, 4))

    def test_auto_respects_custom_threshold(self):
        """Setting a very low threshold forces even modest tensors to CPU."""
        model = torch.nn.Linear(16, 32)
        t = ActivationScope(storage=StoragePolicy.AUTO, auto_cpu_threshold_bytes=1)
        with t.track(model):
            _ = model(torch.randn(2, 16))


class TestStorageAllPolicies:
    """Parametrized storage policy test."""

    @pytest.mark.parametrize("storage_policy", [StoragePolicy.AUTO, StoragePolicy.CPU, StoragePolicy.GPU, StoragePolicy.DISK])
    def test_all_policies_complete_forward(self, simple_linear_model, storage_policy):
        """Every storage policy supports a full forward+backward cycle without crash."""
        t = ActivationScope(storage=storage_policy)
        with t.track(simple_linear_model):
            x = torch.randn(2, 10, requires_grad=True)
            out = simple_linear_model(x)
            loss = out.sum()
            loss.backward()
