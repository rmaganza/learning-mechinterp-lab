"""Pytest fixtures for mechinterp-lab tests."""

import os
import unittest.mock as mock

import pytest
import torch

from mechinterp_lab.models import ModelConfig, TransformerModel


def _skip_model_tests() -> bool:
    """Skip tests that require model download (e.g. in CI without cache)."""
    return os.environ.get("MECHINTERP_SKIP_MODEL_TESTS", "").lower() in ("1", "true", "yes")


@pytest.fixture
def small_model():
    """Load a real model for integration tests. Uses gpt2 which is ~124M params."""
    if _skip_model_tests():
        pytest.skip("MECHINTERP_SKIP_MODEL_TESTS is set; skipping model download")
    from mechinterp_lab.models import load_model

    return load_model("gpt2", device="cpu")


# --- Mock/fake fixtures for unit tests (no model download) ---

# Default fake dimensions (GPT-2 small-like)
N_LAYERS = 12
N_HEADS = 12
D_MODEL = 768
D_VOCAB = 50257
BATCH = 1
SEQ_LEN = 5


def _make_fake_cache(n_layers: int = N_LAYERS, batch: int = BATCH, seq_len: int = SEQ_LEN) -> dict:
    """Build a fake activation cache dict with correct shapes."""
    cache = {}
    for layer in range(n_layers):
        cache[f"blocks.{layer}.hook_resid_pre"] = torch.randn(batch, seq_len, D_MODEL)
        cache[f"blocks.{layer}.hook_resid_post"] = torch.randn(batch, seq_len, D_MODEL)
        cache[f"blocks.{layer}.hook_attn_out"] = torch.randn(batch, seq_len, D_MODEL)
        cache[f"blocks.{layer}.hook_mlp_out"] = torch.randn(batch, seq_len, D_MODEL * 4)
    cache["hook_embed"] = torch.randn(batch, seq_len, D_MODEL)
    cache["hook_pos_embed"] = torch.randn(batch, seq_len, D_MODEL)
    cache["ln_final.hook_normalized"] = torch.randn(batch, seq_len, D_MODEL)
    return cache


@pytest.fixture
def fake_model():
    """Fake HookedTransformer-like model for unit tests. No network calls."""
    cache_dict = _make_fake_cache()

    def run_with_cache(input_ids: torch.Tensor, **kwargs):
        batch, seq_len = input_ids.shape
        logits = torch.randn(batch, seq_len, D_VOCAB)
        # Return plain dict - probes/hooks use cache[key] or getattr(cache, "cache_dict", cache)
        return logits, cache_dict

    hooked = mock.MagicMock()
    hooked.cfg = mock.MagicMock()
    hooked.cfg.n_layers = N_LAYERS
    hooked.cfg.n_heads = N_HEADS
    hooked.cfg.d_model = D_MODEL
    hooked.cfg.d_vocab = D_VOCAB
    hooked.W_U = torch.randn(D_MODEL, D_VOCAB)
    hooked.ln_final = None  # Skip LN for simplicity

    def run_with_hooks(input_ids: torch.Tensor, fwd_hooks=None, **kwargs):
        batch, seq_len = input_ids.shape
        return torch.randn(batch, seq_len, D_VOCAB)

    hooked.run_with_cache = mock.MagicMock(side_effect=run_with_cache)
    hooked.run_with_hooks = mock.MagicMock(side_effect=run_with_hooks)
    hooked.to_tokens = mock.MagicMock(return_value=torch.randint(0, D_VOCAB, (1, SEQ_LEN)))

    # When model(tokens) is called, return real tensor (needed for causal_trace metric_fn)
    def _forward(input_ids):
        return torch.randn(1, input_ids.shape[1], D_VOCAB)

    hooked.return_value = torch.randn(1, SEQ_LEN, D_VOCAB)
    hooked.side_effect = _forward
    return hooked


@pytest.fixture
def fake_wrapped_model(fake_model):
    """TransformerModel wrapper around fake_model for unit tests."""
    config = ModelConfig(
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_model=D_MODEL,
        d_head=D_MODEL // N_HEADS,
        d_mlp=D_MODEL * 4,
        d_vocab=D_VOCAB,
        n_ctx=1024,
    )
    return TransformerModel(model=fake_model, name="gpt2", config=config)


@pytest.fixture
def mock_model():
    """Minimal mock for tests that only need basic attributes (e.g. ModelName checks)."""
    model = mock.MagicMock()
    model.to_tokens = mock.MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
    model.to_single_token = mock.MagicMock(return_value=42)
    model.hook_dict = {
        "blocks.0.attn.hook_z": None,
        "blocks.0.hook_resid_pre": None,
    }
    return model
