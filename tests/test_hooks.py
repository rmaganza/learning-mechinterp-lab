"""Tests for activation hooks."""

import torch

from mechinterp_lab.hooks import ActivationSpec, HookTarget, capture_activations


def test_capture_activations_basic(fake_wrapped_model) -> None:
    """Test basic activation capture (mocked model, no download)."""
    tokens = torch.randint(0, 50257, (1, 5))
    logits, cache = capture_activations(fake_wrapped_model, tokens)

    assert logits.shape[0] == 1
    assert logits.shape[1] == tokens.shape[1]
    assert len(cache.cache) > 0


def test_capture_with_spec(fake_wrapped_model) -> None:
    """Test capture with ActivationSpec (mocked model, no download)."""
    tokens = torch.randint(0, 50257, (1, 4))

    spec = ActivationSpec(
        targets=[HookTarget.RESIDUAL_PRE, HookTarget.MLP_OUTPUT],
        layers=[0, 1],
    )
    logits, cache = capture_activations(fake_wrapped_model, tokens, spec=spec)

    assert "blocks.0.hook_resid_pre" in cache.cache or len(cache.cache) > 0


def test_capture_activations_integration(small_model) -> None:
    """Integration test with real model (skipped if MECHINTERP_SKIP_MODEL_TESTS)."""
    tokens = small_model.model.to_tokens("Hello world", prepend_bos=False)
    tokens = tokens[:, :5]

    logits, cache = capture_activations(small_model.model, tokens)

    assert logits.shape[0] == 1
    assert logits.shape[1] == tokens.shape[1]
    assert len(cache.cache) > 0
