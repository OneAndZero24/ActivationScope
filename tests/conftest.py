"""Shared pytest fixtures for ActivationScope v2 test suite.

Fixtures cover:
  - Model factories (linear, conv, transformer, lstm-style, residual, deep)
  - Tracker factories by reduction / capture policy
  - Input data for each model type
"""

import pytest
import torch
from activationscope import (
    ActivationScope,
    StoragePolicy,
    ReductionPolicy,
    CapturePolicy,
)


# ─── Model factories ──────────────────────────────────────────────

@pytest.fixture
def simple_linear_model():
    """A 3-layer linear model: Linear(10→20) → ReLU → Linear(20→5)."""

    class SimpleLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 20)
            self.act = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(20, 5)

        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            return self.fc2(x)

    return SimpleLinear()


@pytest.fixture
def conv_model():
    """A small convolutional model: Conv2d → ReLU → Conv2d."""

    class SmallConvNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.act = torch.nn.ReLU()
            self.pool = torch.nn.MaxPool2d(2)
            self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, padding=1)

        def forward(self, x):
            x = self.act(self.conv1(x))
            x = self.pool(x)
            return self.conv2(x)

    return SmallConvNet()


@pytest.fixture
def transformer_block():
    """A minimal TransformerEncoderLayer-like block."""

    class TinyTransformerBlock(torch.nn.Module):
        def __init__(self, d_model=64, nhead=2, dim_feedforward=128, dropout=0.0):
            super().__init__()
            self.d_model = d_model
            self.self_attn = torch.nn.MultiheadAttention(
                d_model=d_model, num_heads=nhead, batch_first=True
            )
            self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
            self.linear2 = torch.nn.Linear(dim_feedforward, d_model)
            self.norm1 = torch.nn.LayerNorm(d_model)
            self.norm2 = torch.nn.LayerNorm(d_model)
            self.dropout = torch.nn.Dropout(dropout)

        def forward(self, x):
            # Self-attention
            attn_out, _ = self.self_attn(x, x, x)
            x = self.norm1(x + self.dropout(attn_out))
            # Feed-forward
            ff_out = self.linear2(self.dropout(torch.relu(self.linear1(x))))
            return self.norm2(x + self.dropout(ff_out))

    return TinyTransformerBlock()


# ─── Tracker factories (reduction policies) ──────────────────────

@pytest.fixture
def tracker_store():
    """Tracker with STORE_ALL reduction policy."""
    t = ActivationScope(reduction=ReductionPolicy.STORE_ALL)
    yield t
    if t._session_id is not None:
        t.remove()


@pytest.fixture
def tracker_streaming():
    """Tracker with STREAMING reduction policy."""
    t = ActivationScope(reduction=ReductionPolicy.STREAMING)
    yield t
    if t._session_id is not None:
        t.remove()


@pytest.fixture
def tracker_final_only():
    """Tracker with FINAL_ONLY reduction policy."""
    t = ActivationScope(reduction=ReductionPolicy.FINAL_ONLY)
    yield t
    if t._session_id is not None:
        t.remove()


# ─── Tracker factories (capture policies) ────────────────────────

@pytest.fixture
def tracker_every():
    """Tracker with EVERY capture policy."""
    t = ActivationScope(capture=CapturePolicy.EVERY)
    yield t
    if t._session_id is not None:
        t.remove()


@pytest.fixture
def tracker_sample_n():
    """Tracker with SAMPLE_N capture policy (every 3rd forward)."""
    t = ActivationScope(capture=CapturePolicy.SAMPLE_N, sample_every=3)
    yield t
    if t._session_id is not None:
        t.remove()


@pytest.fixture
def tracker_max_k():
    """Tracker with MAX_K capture policy (max 5 batches)."""
    t = ActivationScope(capture=CapturePolicy.MAX_K, max_batches=5)
    yield t
    if t._session_id is not None:
        t.remove()


# ─── Input data fixtures ─────────────────────────────────────────

@pytest.fixture
def linear_input():
    """Random input for simple_linear_model (batch 2)."""
    return torch.randn(2, 10)


@pytest.fixture
def conv_input():
    """Random input for conv_model."""
    return torch.randn(2, 3, 16, 16)


@pytest.fixture
def transformer_input():
    """Random input for transformer_block: [batch=2, seq_len=8, d_model=64]."""
    return torch.randn(2, 8, 64)


# ─── Additional model factories ──────────────────────────────────

@pytest.fixture
def lstm_tuple_model():
    """Model that returns tuple outputs like nn.LSTM.

    Returns (output, (hidden, cell)) mimicking LSTM behavior so we can
    test how ActivationScope handles modules whose hooks see tuple data.
    We track the internal layers, not the wrapper itself.
    """

    class TupleOutputWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(50, 16)
            self.lstm_hidden_proj = torch.nn.Linear(16, 32)
            self.lstm_cell_proj = torch.nn.Linear(16, 32)
            self.output_proj = torch.nn.Linear(32, 10)

        def forward(self, x_indices):
            emb = self.embedding(x_indices)          # [B, T, 16]
            h = torch.tanh(self.lstm_hidden_proj(emb))  # [B, T, 32]
            c = torch.tanh(self.lstm_cell_proj(emb))    # [B, T, 32]
            out = self.output_proj(h)                # [B, T, 10]
            return out, (h, c)                       # Like LSTM

    return TupleOutputWrapper()


@pytest.fixture
def residual_model():
    """Model with skip/residual connections.

    fc1 → ReLU → fc2  +---+ (skip)
                ↓     |   |
              fc3 ←───┘   ↓
                ↓         ↓
              ReLU      fc_out
                └─────→ ↓
    """

    class ResidualBlock(torch.nn.Module):
        def __init__(self, dim=16):
            super().__init__()
            self.fc1 = torch.nn.Linear(dim, dim)
            self.act1 = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(dim, dim * 2)
            self.fc3 = torch.nn.Linear(dim * 2, dim)
            self.act2 = torch.nn.ReLU()
            self.fc_out = torch.nn.Linear(dim, 5)

        def forward(self, x):
            identity = x
            x = self.act1(self.fc1(x))
            x = self.fc3(self.fc2(x)) + identity   # residual skip
            x = self.act2(x)
            return self.fc_out(x)

    return ResidualBlock()


@pytest.fixture
def deep_model():
    """Very deep model (24 Linear layers) to test stack/binding limits."""

    class DeepNet(torch.nn.Module):
        def __init__(self, n_layers=24):
            super().__init__()
            self.layers = torch.nn.ModuleList()
            for i in range(n_layers):
                ch_in = 32 if i == 0 else 16
                ch_out = 16 if i < n_layers - 1 else 5
                self.layers.append(torch.nn.Linear(ch_in, ch_out))
            self.acts = torch.nn.ModuleList([
                torch.nn.ReLU() for _ in range(n_layers)
            ])

        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.acts):
                    x = self.acts[i](x)
            return x

    return DeepNet()


@pytest.fixture
def mixed_container_model():
    """Model with nested ModuleDict + ModuleList combinations."""

    class MixedContainer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # ModuleDict of ModuleLists
            self.blocks = torch.nn.ModuleDict({
                "encoder": torch.nn.ModuleList([
                    torch.nn.Linear(10, 20),
                    torch.nn.ReLU(),
                    torch.nn.Linear(20, 30),
                ]),
                "decoder": torch.nn.ModuleList([
                    torch.nn.Linear(30, 20),
                    torch.nn.ReLU(),
                    torch.nn.Linear(20, 5),
                ]),
            })

        def forward(self, x):
            for layer in self.blocks["encoder"]:
                x = layer(x)
            for layer in self.blocks["decoder"]:
                x = layer(x)
            return x

    return MixedContainer()


@pytest.fixture
def nested_sequential_model():
    """Model with Sequential inside another ModuleList."""

    class NestedSequential(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stack = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(10, 20),
                    torch.nn.ReLU(),
                ),
                torch.nn.Sequential(
                    torch.nn.Linear(20, 30),
                    torch.nn.ReLU(),
                ),
            ])
            self.head = torch.nn.Linear(30, 5)

        def forward(self, x):
            for seq in self.stack:
                x = seq(x)
            return self.head(x)

    return NestedSequential()


# ─── Tracker factory with pinned memory ──────────────────────────

@pytest.fixture
def tracker_pinned():
    """Tracker with use_pinned=True."""
    t = ActivationScope(use_pinned=True, storage=StoragePolicy.CPU)
    yield t
    if t._session_id is not None:
        t.remove()
