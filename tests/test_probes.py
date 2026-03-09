"""Tests for probes."""

import torch

from mechinterp_lab.probes import TunedLensProbe, logit_lens, probe_layer_logits, tuned_lens_probe


def test_logit_lens(fake_wrapped_model) -> None:
    """Test logit lens (mocked model, no download)."""
    tokens = torch.randint(0, 50257, (1, 5))
    logits_per_layer, cache = logit_lens(
        fake_wrapped_model, tokens, layers=[0, 6, 11]
    )

    assert logits_per_layer.shape[0] == 3
    assert logits_per_layer.shape[1] == 1  # batch
    assert logits_per_layer.shape[2] == fake_wrapped_model.config.d_vocab


def test_probe_layer_logits(fake_wrapped_model) -> None:
    """Test single layer probe (mocked model, no download)."""
    tokens = torch.randint(0, 50257, (1, 4))
    logits = probe_layer_logits(fake_wrapped_model, tokens, layer=0)
    assert logits.shape == (1, fake_wrapped_model.config.d_vocab)


def test_tuned_lens_probe(fake_wrapped_model) -> None:
    """Test tuned lens probe (mocked model, no download)."""
    probe = TunedLensProbe(
        fake_wrapped_model.config.d_model,
        fake_wrapped_model.config.d_vocab,
    )
    tokens = torch.randint(0, 50257, (1, 4))
    logits = tuned_lens_probe(fake_wrapped_model, tokens, probe, layer=0)
    assert logits.shape == (1, fake_wrapped_model.config.d_vocab)


def test_logit_lens_integration(small_model) -> None:
    """Integration test with real model (skipped if MECHINTERP_SKIP_MODEL_TESTS)."""
    tokens = small_model.model.to_tokens("The capital of France is", prepend_bos=False)
    logits_per_layer, _ = logit_lens(small_model.model, tokens, layers=[0, 6, 11])
    assert logits_per_layer.shape[0] == 3
    assert logits_per_layer.shape[2] == small_model.config.d_vocab
