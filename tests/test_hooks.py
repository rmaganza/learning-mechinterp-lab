"""Tests for activation hooks."""



from mechinterp_lab.hooks import ActivationSpec, HookTarget, capture_activations


def test_capture_activations_basic(small_model) -> None:
    """Test basic activation capture."""
    tokens = small_model.model.to_tokens("Hello world", prepend_bos=False)
    tokens = tokens[:, :5]  # Shorten for speed

    logits, cache = capture_activations(small_model.model, tokens)

    assert logits.shape[0] == 1
    assert logits.shape[1] == tokens.shape[1]
    assert len(cache.cache) > 0


def test_capture_with_spec(small_model) -> None:
    """Test capture with ActivationSpec."""
    tokens = small_model.model.to_tokens("Test", prepend_bos=False)

    spec = ActivationSpec(
        targets=[HookTarget.RESIDUAL_PRE, HookTarget.MLP_OUTPUT],
        layers=[0, 1],
    )
    logits, cache = capture_activations(small_model.model, tokens, spec=spec)

    assert "blocks.0.hook_resid_pre" in cache.cache or len(cache.cache) > 0
