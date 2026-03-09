"""Tests for activation patching logic."""

import unittest.mock as mock

import torch


def _make_fake_cache(n_layers=12, batch=1, seq_len=8, d_model=768):
    """Build fake activation cache for patching tests."""
    cache = {}
    for layer in range(n_layers):
        cache[f"blocks.{layer}.hook_resid_pre"] = torch.randn(batch, seq_len, d_model)
    return cache


def test_activation_patch_resid_pre(fake_wrapped_model):
    """Activation patch (resid_pre) returns tensor of metric per layer (mocked)."""
    from mechinterp_lab.patching import activation_patch

    corrupt_tokens = torch.randint(0, 50257, (1, 8))
    clean_cache = _make_fake_cache(seq_len=8)

    def metric_fn(logits: torch.Tensor) -> torch.Tensor:
        return logits.max()

    with mock.patch(
        "mechinterp_lab.patching.activation_patching.get_act_patch_resid_pre"
    ) as mock_patch:
        mock_patch.return_value = torch.randn(12, 8)  # [n_layers, pos]
        result = activation_patch(
            fake_wrapped_model,
            corrupted_tokens=corrupt_tokens,
            clean_cache=clean_cache,
            metric_fn=metric_fn,
            activation_type="resid_pre",
        )
        assert mock_patch.called
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == fake_wrapped_model.config.n_layers


def test_patch_residual_stream(fake_wrapped_model):
    """patch_residual_stream returns [n_layers] tensor (mocked)."""
    from mechinterp_lab.patching import patch_residual_stream

    corrupt_tokens = torch.randint(0, 50257, (1, 6))
    clean_cache = _make_fake_cache(seq_len=6)

    def metric_fn(logits: torch.Tensor) -> torch.Tensor:
        return logits.max()

    with mock.patch(
        "mechinterp_lab.patching.activation_patching.get_act_patch_resid_pre"
    ) as mock_patch:
        mock_patch.return_value = torch.randn(12, 6)  # [n_layers, pos]
        result = patch_residual_stream(
            fake_wrapped_model, corrupt_tokens, clean_cache, metric_fn
        )
        assert result.shape == (fake_wrapped_model.config.n_layers,)


def test_causal_trace(fake_wrapped_model):
    """Causal trace returns tensor of metric per layer (mocked)."""
    from mechinterp_lab.patching import causal_trace

    clean_tokens = torch.randint(0, 50257, (1, 8))
    corrupt_tokens = torch.randint(0, 50257, (1, 8))

    def metric_fn(logits: torch.Tensor) -> torch.Tensor:
        return logits.max()

    result = causal_trace(
        fake_wrapped_model, clean_tokens, corrupt_tokens, metric_fn
    )
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == fake_wrapped_model.config.n_layers + 1


def test_activation_patch_integration(small_model):
    """Integration test with real model (skipped if MECHINTERP_SKIP_MODEL_TESTS)."""
    from mechinterp_lab.patching import activation_patch

    clean = "The capital of France is Paris"
    corrupt = "The capital of France is London"
    clean_tokens = small_model.model.to_tokens(clean)
    corrupt_tokens = small_model.model.to_tokens(corrupt)
    min_len = min(clean_tokens.shape[1], corrupt_tokens.shape[1])
    clean_tokens = clean_tokens[:, :min_len]
    corrupt_tokens = corrupt_tokens[:, :min_len]

    _, clean_cache = small_model.model.run_with_cache(clean_tokens)

    def metric_fn(logits: torch.Tensor) -> torch.Tensor:
        last_pos = logits.shape[1] - 1
        correct_tok = clean_tokens[0, last_pos].item()
        wrong_tok = corrupt_tokens[0, last_pos].item()
        return logits[0, last_pos, correct_tok] - logits[0, last_pos, wrong_tok]

    result = activation_patch(
        small_model,
        corrupted_tokens=corrupt_tokens,
        clean_cache=clean_cache,
        metric_fn=metric_fn,
        activation_type="resid_pre",
    )
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == small_model.config.n_layers
