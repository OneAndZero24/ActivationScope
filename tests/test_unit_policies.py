"""Unit tests for policy enums, helper functions, and internal utilities.

Every enum value, helper method, and private-but-testable function gets
at least one assertion that it behaves correctly in isolation.
"""

import pytest
import torch

from activationscope import (
    StoragePolicy,
    ReductionPolicy,
    CapturePolicy,
)
from activationscope.tracker import (
    _parse_capture_dir,
    pattern_or_identity,
)


class TestStoragePolicyEnum:
    """StoragePolicy enum values and basic behaviour."""

    def test_auto_value(self):
        assert StoragePolicy.AUTO == 0

    def test_cpu_value(self):
        assert StoragePolicy.CPU == 1

    def test_gpu_value(self):
        assert StoragePolicy.GPU == 2

    def test_all_values_present(self):
        """All three members have correct int values."""
        assert {StoragePolicy.AUTO, StoragePolicy.CPU, StoragePolicy.GPU} == {0, 1, 2}

    def test_inherits_from_int(self):
        """StoragePolicy values are ints at runtime."""
        for member in StoragePolicy:
            assert isinstance(member, int)


class TestReductionPolicyEnum:
    """ReductionPolicy enum values and basic behaviour."""

    def test_store_all_value(self):
        assert ReductionPolicy.STORE_ALL == 0

    def test_streaming_value(self):
        assert ReductionPolicy.STREAMING == 1

    def test_final_only_value(self):
        assert ReductionPolicy.FINAL_ONLY == 2

    def test_all_values_present(self):
        assert {ReductionPolicy.STORE_ALL, ReductionPolicy.STREAMING, ReductionPolicy.FINAL_ONLY} == {0, 1, 2}


class TestCapturePolicyEnum:
    """CapturePolicy enum values and basic behaviour."""

    def test_every_value(self):
        assert CapturePolicy.EVERY == 0

    def test_sample_n_value(self):
        assert CapturePolicy.SAMPLE_N == 1

    def test_max_k_value(self):
        assert CapturePolicy.MAX_K == 2

    def test_iterable(self):
        members = list(CapturePolicy)
        assert len(members) == 3
        values = {int(m) for m in members}
        assert values == {0, 1, 2}


class TestParseCaptureDir:
    """_parse_capture_dir(capture_str) → int mapping and error handling."""

    @pytest.mark.parametrize(
        ("input_str", "expected"),
        [
            ("input", 0),
            ("output", 1),
            ("both", 2),
            ("INPUT", 0),   # case-insensitive
            ("Output", 1),
            ("BoTh", 2),
        ],
    )
    def test_valid_inputs(self, input_str, expected):
        assert _parse_capture_dir(input_str) == expected

    @pytest.mark.parametrize(
        "bad_input",
        [
            "",
            "middle",
            "none",
            "input,output",
            "random_string",
            None,  # will fail on .lower() call
        ],
    )
    def test_invalid_inputs_raise(self, bad_input):
        with pytest.raises((ValueError, AttributeError)):
            _parse_capture_dir(bad_input)


class TestPatternOrIdentity:
    """pattern_or_identity(fn) derives string key from callable."""

    def test_named_function_returns_name(self):
        def my_reduction(x):
            return x.abs().mean()

        key = pattern_or_identity(my_reduction)
        assert "__name__" in repr(key) or key.startswith("my_reduction")

    def test_lambda_returns_repr(self):
        lam = lambda x: torch.amax(x, dim=0)  # noqa: E731
        key = pattern_or_identity(lam)
        assert "<lambda>" in key

    def test_class_callable_returns_repr_of_instance(self):
        class MyCallable:
            def __call__(self, x):
                return x.sum()

        obj = MyCallable()
        key = pattern_or_identity(obj)
        assert isinstance(key, str) and len(key) > 0


class TestTorchBackendImports:
    """The native _C module exposes all expected bindings."""

    def test_session_create_exists(self):
        import activationscope._C as _C  # noqa: N811
        assert callable(_C.session_create)

    def test_session_destroy_exists(self):
        import activationscope._C as _C  # noqa: N811
        assert callable(_C.session_destroy)

    def test_session_readback_exists(self):
        import activationscope._C as _C  # noqa: N811
        assert callable(_C.session_readback)

    def test_session_clear_exists(self):
        import activationscope._C as _C  # noqa: N811
        assert callable(_C.session_clear)

    def test_session_register_hooks_exists(self):
        import activationscope._C as _C  # noqa: N811
        assert callable(_C.session_register_hooks)

    def test_make_compiled_handle_exists(self):
        import activationscope._C as _C  # noqa: N811
        assert callable(_C.make_compiled_handle)

    def test_set_layer_reduction_exists(self):
        import activationscope._C as _C  # noqa: N811
        assert callable(_C.set_layer_reduction)

    def test_set_global_reduction_exists(self):
        import activationscope._C as _C  # noqa: N811
        assert callable(_C.set_global_reduction)


class TestActivationScopeProperties:
    """session_id and activations behave correctly."""

    def test_session_id_is_int(self):
        from activationscope import ActivationScope

        t = ActivationScope()
        sid = t.session_id
        assert isinstance(sid, int)
        assert sid > 0
        t.remove()

    def test_session_id_raised_after_destroy(self):
        from activationscope import ActivationScope

        t = ActivationScope()
        _ = t.session_id  # fine
        t.remove()
        with pytest.raises(RuntimeError, match="session already destroyed"):
            _ = t.session_id

    def test_activations_returns_dict(self):
        from activationscope import ActivationScope

        t = ActivationScope()
        model = torch.nn.Linear(3, 4)
        with t.track(model):
            acts = t.activations
            assert isinstance(acts, dict)
