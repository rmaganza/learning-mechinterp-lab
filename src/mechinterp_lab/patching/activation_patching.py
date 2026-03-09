"""Activation patching and causal tracing.

Patch activations from a clean (source) run into a corrupted run to measure
causal effect of specific components.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache as TLActivationCache
from transformer_lens.patching import (
    get_act_patch_attn_head_out_all_pos,
    get_act_patch_mlp_out,
    get_act_patch_resid_pre,
)


def _get_model(model: Any) -> HookedTransformer:
    """Extract HookedTransformer from wrapper or return as-is."""
    return model.model if hasattr(model, "model") else model


def _get_tl_cache(cache: Any, model: HookedTransformer | None = None) -> TLActivationCache:
    """Convert to TransformerLens ActivationCache if needed."""
    if isinstance(cache, TLActivationCache):
        return cache
    if hasattr(cache, "to_tl_cache"):
        return cache.to_tl_cache(model)
    if isinstance(cache, dict):
        return TLActivationCache(cache, model=model)
    raise TypeError(f"Expected ActivationCache or dict, got {type(cache)}")


def activation_patch(
    model: HookedTransformer | Any,
    corrupted_tokens: torch.Tensor,
    clean_cache: TLActivationCache | dict | Any,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    activation_type: str = "resid_pre",
    **kwargs: Any,
) -> torch.Tensor:
    """Run activation patching: patch activations from clean run into corrupted run.

    Uses TransformerLens patching utilities. The metric_fn receives logits
    [batch, pos, d_vocab] and returns a scalar (e.g. correct token logit, loss).

    Args:
        model: HookedTransformer or TransformerModel.
        corrupted_tokens: Tokens for the corrupted (incorrect) run [batch, pos].
        clean_cache: Cached activations from the clean (correct) run.
        metric_fn: Function mapping logits -> scalar metric.
        activation_type: One of 'resid_pre', 'attn_out', 'mlp_out', etc.
        **kwargs: Passed to TransformerLens patching functions.

    Returns:
        Tensor of metric values per (layer,) or (layer, head) etc.
    """
    hooked = _get_model(model)
    tl_cache = _get_tl_cache(clean_cache, hooked)

    if activation_type == "resid_pre":
        return get_act_patch_resid_pre(
            hooked,
            corrupted_tokens,
            tl_cache,
            patching_metric=metric_fn,
            **kwargs,
        )
    if activation_type == "attn_out":
        return get_act_patch_attn_head_out_all_pos(
            hooked,
            corrupted_tokens,
            tl_cache,
            patching_metric=metric_fn,
            **kwargs,
        )
    if activation_type == "mlp_out":
        return get_act_patch_mlp_out(
            hooked,
            corrupted_tokens,
            tl_cache,
            patching_metric=metric_fn,
            **kwargs,
        )
    raise ValueError(
        f"activation_type must be one of resid_pre, attn_out, mlp_out; got {activation_type}"
    )


def patch_residual_stream(
    model: HookedTransformer | Any,
    corrupted_tokens: torch.Tensor,
    clean_cache: TLActivationCache | dict | Any,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Patch residual stream (resid_pre) at each layer. Returns [n_layers]."""
    result = activation_patch(
        model, corrupted_tokens, clean_cache, metric_fn, activation_type="resid_pre"
    )
    # TL returns [n_layers, pos]; average over positions for [n_layers]
    if result.dim() > 1:
        result = result.mean(dim=-1)
    return result


def patch_attention_head(
    model: HookedTransformer | Any,
    corrupted_tokens: torch.Tensor,
    clean_cache: TLActivationCache | dict | Any,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Patch attention head outputs at each layer. Returns [n_layers, n_heads]."""
    return activation_patch(
        model, corrupted_tokens, clean_cache, metric_fn, activation_type="attn_out"
    )


def causal_trace(
    model: HookedTransformer | Any,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    positions: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Causal tracing: ablate residual stream at each layer and measure effect.

    Runs corrupted input, then patches in clean resid_pre at each layer one at a time.
    Measures how much the metric recovers when we restore each layer's clean input.

    Args:
        model: HookedTransformer or TransformerModel.
        clean_tokens: Tokens for clean (correct) run.
        corrupted_tokens: Tokens for corrupted run.
        metric_fn: Function mapping logits -> scalar.
        positions: Optional positions to evaluate metric at (default: last position).

    Returns:
        Tensor of shape [n_layers + 1] with metric when patching resid_pre at each layer.
    """
    hooked = _get_model(model)
    _, clean_cache = hooked.run_with_cache(clean_tokens)

    n_layers = hooked.cfg.n_layers
    results = torch.zeros(n_layers + 1, device=corrupted_tokens.device)

    # Baseline: fully corrupted
    logits_corrupted = hooked(corrupted_tokens)
    results[0] = metric_fn(logits_corrupted)

    # Patch resid_pre at each layer
    for layer in range(n_layers):
        clean_resid = clean_cache[f"blocks.{layer}.hook_resid_pre"]

        def patch_hook(
            act: torch.Tensor, hook: Any, resid: torch.Tensor = clean_resid
        ) -> torch.Tensor:
            return resid

        logits = hooked.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", patch_hook)],
        )
        results[layer + 1] = metric_fn(logits)

    return results
