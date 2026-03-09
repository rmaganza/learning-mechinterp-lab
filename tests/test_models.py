"""Tests for model loading."""

import unittest.mock as mock

from mechinterp_lab.models import ModelConfig, ModelName, TransformerModel, load_model


def test_model_name_enum() -> None:
    """Test ModelName enum (no model, no network)."""
    assert ModelName.GPT2_SMALL.value == "gpt2"
    assert ModelName.GPT2_MEDIUM.value == "gpt2-medium"


def test_load_model_returns_wrapper() -> None:
    """Test load_model returns TransformerModel with correct structure (mocked)."""
    mock_hooked = mock.MagicMock()
    mock_hooked.cfg.n_layers = 12
    mock_hooked.cfg.n_heads = 12
    mock_hooked.cfg.d_model = 768
    mock_hooked.cfg.d_head = 64
    mock_hooked.cfg.d_mlp = 3072
    mock_hooked.cfg.d_vocab = 50257
    mock_hooked.cfg.n_ctx = 1024

    with mock.patch(
        "mechinterp_lab.models.loader.HookedTransformer.from_pretrained",
        return_value=mock_hooked,
    ):
        model = load_model("gpt2", device="cpu")

    assert isinstance(model, TransformerModel)
    assert model.name == "gpt2"
    assert model.config.n_layers == 12
    assert model.config.n_heads == 12
    assert model.config.d_model == 768


def test_model_config_from_hooked() -> None:
    """Test ModelConfig.from_hooked_transformer (no network)."""
    mock_hooked = mock.MagicMock()
    mock_hooked.cfg.n_layers = 6
    mock_hooked.cfg.n_heads = 6
    mock_hooked.cfg.d_model = 384
    mock_hooked.cfg.d_head = 64
    mock_hooked.cfg.d_mlp = 1536
    mock_hooked.cfg.d_vocab = 50257
    mock_hooked.cfg.n_ctx = 1024

    config = ModelConfig.from_hooked_transformer(mock_hooked)
    assert config.n_layers == 6
    assert config.d_model == 384


def test_load_gpt2_small_integration(small_model) -> None:
    """Integration test: load real GPT-2 small (skipped if MECHINTERP_SKIP_MODEL_TESTS)."""
    assert small_model.config.n_layers == 12
    assert small_model.config.n_heads == 12
    assert small_model.config.d_model == 768


def test_model_forward_integration(small_model) -> None:
    """Integration test: forward pass (skipped if MECHINTERP_SKIP_MODEL_TESTS)."""
    tokens = small_model.model.to_tokens("Hello")
    out = small_model.model(tokens)
    assert out.shape[2] == small_model.config.d_vocab
