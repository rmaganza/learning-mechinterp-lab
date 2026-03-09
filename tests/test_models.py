"""Tests for model loading."""


from mechinterp_lab.models import ModelName


def test_load_gpt2_small(small_model) -> None:
    """Test loading GPT-2 small via fixture."""
    assert small_model.config.n_layers == 12
    assert small_model.config.n_heads == 12
    assert small_model.config.d_model == 768


def test_load_model_from_string(small_model) -> None:
    """Test model has expected attributes."""
    assert small_model.name == "gpt2"
    assert small_model.config.d_vocab > 0


def test_model_forward(small_model) -> None:
    """Model can run forward pass."""
    tokens = small_model.model.to_tokens("Hello")
    out = small_model.model(tokens)
    assert out.shape[2] == small_model.config.d_vocab


def test_get_available_models_mock(mock_model) -> None:
    """Test that ModelName enum provides expected models (no network)."""
    assert ModelName.GPT2_SMALL.value == "gpt2"
    assert ModelName.GPT2_MEDIUM.value == "gpt2-medium"
