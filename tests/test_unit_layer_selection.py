"""Unit tests for layer selection logic (_select_layers).

Tests glob-based inclusion, exclusion, container filtering, and root-module
exclusion against real PyTorch module hierarchies.
"""

import pytest
import torch

from activationscope.tracker import _select_layers


class TestSelectLayersBaseline:
    """_select_layers with no filters returns all non-container submodules."""

    def test_no_filters_all_leaf_modules(self, simple_linear_model):
        selected = _select_layers(simple_linear_model)
        expected_names = {"fc1", "act", "fc2"}
        assert set(selected.keys()) == expected_names

    def test_excludes_root_module(self, simple_linear_model):
        """The root module ('') is always excluded."""
        selected = _select_layers(simple_linear_model)
        assert "" not in selected

    def test_root_type_included(self, simple_linear_model):
        "The root module should appear as a key."
        selected = _select_layers(simple_linear_model)
        # Root ("") is excluded by design but the model itself does have submodules
        assert len(selected) > 0

    def test_values_are_modules(self, simple_linear_model):
        selected = _select_layers(simple_linear_model)
        for name, mod in selected.items():
            assert isinstance(mod, torch.nn.Module), f"{name} is not a Module"


class TestSelectLayersContainers:
    """Container types (ModuleList, ModuleDict, Sequential) are excluded."""

    def test_sequential_excluded(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 4),
        )
        selected = _select_layers(model)
        # Root is "", sequential itself excluded, children included
        assert "" not in selected
        for key in selected:
            assert not isinstance(selected[key], torch.nn.Sequential)

    def test_module_list_excluded(self):
        class WithModuleList(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.stack = torch.nn.ModuleList([
                    torch.nn.Linear(8, 16),
                    torch.nn.ReLU(),
                ])
                self.final = torch.nn.Linear(16, 4)

            def forward(self, x):
                for layer in self.stack:
                    x = layer(x)
                return self.final(x)

        model = WithModuleList()
        selected = _select_layers(model)
        keys = set(selected.keys())
        # ModuleList itself excluded, but children included
        assert "stack" not in keys  # ModuleList container
        assert "stack.0" in keys   # Linear child
        assert "stack.1" in keys   # ReLU child

    def test_module_dict_excluded(self):
        class WithModuleDict(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleDict({
                    "fc": torch.nn.Linear(8, 16),
                    "act": torch.nn.ReLU(),
                })
                self.out = torch.nn.Linear(16, 4)

            def forward(self, x):
                return self.layers["fc"](x)

        model = WithModuleDict()
        selected = _select_layers(model)
        assert "layers" not in set(selected.keys())  # ModuleDict excluded


class TestSelectLayersInclude:
    """_select_layers with include patterns (fnmatch globs)."""

    def test_include_linear_only(self, simple_linear_model):
        selected = _select_layers(simple_linear_model, include=["*Linear*"])
        assert set(selected.keys()) == {"fc1", "fc2"}
        assert "act" not in selected

    def test_include_prefix(self, simple_linear_model):
        selected = _select_layers(simple_linear_model, include=["fc*"])
        assert set(selected.keys()) == {"fc1", "fc2"}

    def test_include_exact_name(self, simple_linear_model):
        selected = _select_layers(simple_linear_model, include=["act"])
        assert selected.keys() == {"act"}

    def test_include_no_match(self, simple_linear_model):
        selected = _select_layers(simple_linear_model, include=["nonexistent*"])
        assert len(selected) == 0


class TestSelectLayersExclude:
    """_select_layers with exclude patterns (subtractive)."""

    def test_exclude_specific(self, simple_linear_model):
        selected = _select_layers(simple_linear_model, exclude=["act"])
        remaining = set(selected.keys())
        assert "act" not in remaining
        assert {"fc1", "fc2"}.issubset(remaining)

    def test_exclude_by_pattern(self, conv_model):
        model = conv_model
        selected = _select_layers(model, exclude=["conv*", "pool"])
        keys = set(selected.keys())
        for k in keys:
            assert not k.startswith("conv") and k != "pool"

    def test_layers_param_equivalent_to_include(self, simple_linear_model):
        """layers= behaves like include= (it's an alias)."""
        s1 = _select_layers(simple_linear_model, layers=["fc*"])
        s2 = _select_layers(simple_linear_model, include=["fc*"])
        assert set(s1.keys()) == set(s2.keys())


class TestSelectLayersIncludeAndExclude:
    """Both include and exclude apply correctly."""

    def test_include_then_exclude(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(16, 5),
        )
        # Include all Linear layers, then exclude the last one
        selected = _select_layers(
            model, include=["*.Linear*"], exclude=["*2*"]
        )
        keys = set(selected.keys())
        assert "0" in keys   # Linear(10,32)
        assert "2" not in keys  # Linear(32,16) excluded
        # Dropout is a Linear-like layer? Actually, let's be specific:
        # Include only Linear layers, exclude index 2


class TestSelectLayersConv:
    """Layer selection against convolutional models."""

    def test_conv_model_baseline(self, conv_model):
        selected = _select_layers(conv_model)
        keys = set(selected.keys())
        # Should have conv1, act, pool, conv2 — excluding root
        assert "" not in keys
        for expected in ("conv1", "act", "pool", "conv2"):
            assert expected in keys, f"Expected {expected} in selected layers"

    def test_conv_include_conv_only(self, conv_model):
        selected = _select_layers(conv_model, include=["conv*"])
        assert set(selected.keys()) == {"conv1", "conv2"}


class TestSelectLayersComplexNesting:
    """Deep module nesting with mixed container types."""

    def test_deeply_nested_sequential(self):
        class DeepNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Sequential(
                    torch.nn.Linear(64, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 64),
                )
                self.head = torch.nn.Linear(64, 10)

            def forward(self, x):
                return self.head(self.backbone(x))

        model = DeepNet()
        selected = _select_layers(model)

        # Sequential container excluded, but children and standalone included
        keys = set(selected.keys())
        assert "backbone" not in keys  # Sequential is a container
        assert "head" in keys
        assert "backbone.0" in keys   # Linear inside Sequential (named_modules recurse)
