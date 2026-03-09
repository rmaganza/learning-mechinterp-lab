"""Tests for probes."""



from mechinterp_lab.probes import TunedLensProbe, logit_lens, probe_layer_logits, tuned_lens_probe


def test_logit_lens(small_model) -> None:
    """Test logit lens on GPT-2."""
    tokens = small_model.model.to_tokens("The capital of France is", prepend_bos=False)

    logits_per_layer, cache = logit_lens(small_model.model, tokens, layers=[0, 6, 11])

    assert logits_per_layer.shape[0] == 3
    assert logits_per_layer.shape[1] == 1  # batch
    assert logits_per_layer.shape[2] == small_model.config.d_vocab


def test_probe_layer_logits(small_model) -> None:
    """Test single layer probe."""
    tokens = small_model.model.to_tokens("Hello", prepend_bos=False)

    logits = probe_layer_logits(small_model.model, tokens, layer=0)
    assert logits.shape == (1, small_model.config.d_vocab)


def test_tuned_lens_probe(small_model) -> None:
    """Test tuned lens probe."""
    probe = TunedLensProbe(
        small_model.config.d_model,
        small_model.config.d_vocab,
    )
    tokens = small_model.model.to_tokens("Test", prepend_bos=False)
    logits = tuned_lens_probe(small_model.model, tokens, probe, layer=0)
    assert logits.shape == (1, small_model.config.d_vocab)
