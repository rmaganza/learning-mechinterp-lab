"""Logit lens and tuned-lens style probing.

Probe intermediate layer activations to predict next token.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformer_lens import HookedTransformer


def _get_model(model: Any) -> HookedTransformer:
    """Extract HookedTransformer from wrapper or return as-is."""
    return model.model if hasattr(model, "model") else model


def logit_lens(
    model: HookedTransformer | Any,
    input_ids: torch.Tensor,
    layers: list[int] | None = None,
    position: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Logit lens: project residual stream at each layer through unembedding.

    The residual stream at each layer is multiplied by W_U to get "virtual logits"
    as if we stopped the forward pass at that layer. No learned probe - uses
    the model's own unembedding.

    Args:
        model: HookedTransformer or TransformerModel.
        input_ids: Token ids [batch, pos].
        layers: Which layers to probe. Default: all layers.
        position: Position to read logits from (-1 = last).

    Returns:
        (logits_per_layer, cache) where logits_per_layer is [n_layers, batch, d_vocab].
    """
    hooked = _get_model(model)
    if layers is None:
        layers = list(range(hooked.cfg.n_layers))

    logits_list: list[torch.Tensor] = []
    _, cache = hooked.run_with_cache(input_ids)

    W_U = hooked.W_U  # [d_model, d_vocab]
    ln_final = hooked.ln_final
    cache_dict = getattr(cache, "cache_dict", cache)

    for layer in layers:
        key = f"blocks.{layer}.hook_resid_post"
        if key not in cache_dict:
            key = f"blocks.{layer}.hook_resid_pre"
        resid = cache[key]
        if position is not None:
            pos = resid.shape[1] + position if position < 0 else position
            resid = resid[:, pos : pos + 1, :]  # [batch, 1, d_model]
        # Apply final LN if present
        if ln_final is not None:
            resid = ln_final(resid)
        logits = resid @ W_U  # [batch, 1, d_vocab] or [batch, d_vocab]
        logits = logits.squeeze(1)
        logits_list.append(logits)

    return torch.stack(logits_list, dim=0), cache


def probe_layer_logits(
    model: HookedTransformer | Any,
    input_ids: torch.Tensor,
    layer: int,
    position: int = -1,
) -> torch.Tensor:
    """Get logits from projecting a single layer's residual through W_U.

    Args:
        model: HookedTransformer or TransformerModel.
        input_ids: Token ids [batch, pos].
        layer: Layer index.
        position: Position to read (-1 = last).

    Returns:
        Logits [batch, d_vocab].
    """
    hooked = _get_model(model)
    _, cache = hooked.run_with_cache(input_ids)
    resid = cache[f"blocks.{layer}.hook_resid_post"]
    if position < 0:
        position = resid.shape[1] + position
    resid = resid[:, position, :]  # [batch, d_model]
    if hooked.ln_final is not None:
        resid = hooked.ln_final(resid.unsqueeze(0)).squeeze(0)
    return resid @ hooked.W_U


class TunedLensProbe(nn.Module):
    """Learned linear probe from residual stream to vocabulary (tuned-lens style).

    Maps d_model -> d_vocab with a learned bias. Can be trained to align
    intermediate layer activations with final token predictions.
    """

    def __init__(self, d_model: int, d_vocab: int, device: torch.device | None = None) -> None:
        super().__init__()
        self.probe = nn.Linear(d_model, d_vocab, bias=True)
        if device:
            self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project residual stream to logits. x: [batch, pos, d_model]."""
        return self.probe(x)


def tuned_lens_probe(
    model: HookedTransformer | Any,
    input_ids: torch.Tensor,
    probe: TunedLensProbe,
    layer: int,
    position: int = -1,
) -> torch.Tensor:
    """Apply a tuned learned probe to a layer's residual stream.

    Args:
        model: HookedTransformer or TransformerModel.
        input_ids: Token ids [batch, pos].
        probe: TunedLensProbe instance.
        layer: Layer index.
        position: Position to read (-1 = last).

    Returns:
        Logits [batch, d_vocab].
    """
    hooked = _get_model(model)
    _, cache = hooked.run_with_cache(input_ids)
    resid = cache[f"blocks.{layer}.hook_resid_post"]
    if position < 0:
        position = resid.shape[1] + position
    resid = resid[:, position, :]  # [batch, d_model]
    return probe(resid)
