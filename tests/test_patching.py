"""Tests for activation patching logic."""

import torch


def test_activation_patch_resid_pre(small_model):
    """Activation patch (resid_pre) returns tensor of metric per layer."""
    from mechinterp_lab.patching import activation_patch

    clean = "The capital of France is Paris"
    corrupt = "The capital of France is London"
    clean_tokens = small_model.model.to_tokens(clean)
    corrupt_tokens = small_model.model.to_tokens(corrupt)

    # Align lengths
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


def test_patch_residual_stream(small_model):
    """patch_residual_stream returns [n_layers] tensor."""
    from mechinterp_lab.patching import patch_residual_stream

    clean_tokens = small_model.model.to_tokens("The answer is 42")
    corrupt_tokens = small_model.model.to_tokens("The answer is 99")
    _, clean_cache = small_model.model.run_with_cache(clean_tokens)

    def metric_fn(logits: torch.Tensor) -> torch.Tensor:
        return logits.max()

    result = patch_residual_stream(
        small_model, corrupt_tokens, clean_cache, metric_fn
    )
    assert result.shape == (small_model.config.n_layers,)


def test_causal_trace(small_model):
    """Causal trace returns tensor of metric per layer."""
    from mechinterp_lab.patching import causal_trace

    clean_tokens = small_model.model.to_tokens("The capital of France is Paris")
    corrupt_tokens = small_model.model.to_tokens("The capital of France is London")
    min_len = min(clean_tokens.shape[1], corrupt_tokens.shape[1])
    clean_tokens = clean_tokens[:, :min_len]
    corrupt_tokens = corrupt_tokens[:, :min_len]

    def metric_fn(logits: torch.Tensor) -> torch.Tensor:
        return logits[0, -1, clean_tokens[0, -1].item()]

    result = causal_trace(
        small_model, clean_tokens, corrupt_tokens, metric_fn
    )
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == small_model.config.n_layers + 1
