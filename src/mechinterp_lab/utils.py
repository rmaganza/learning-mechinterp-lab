"""Shared utilities for model access, cache helpers, and common patterns."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformer_lens import HookedTransformer


def get_model(model: HookedTransformer | Any) -> HookedTransformer:
    """Extract HookedTransformer from TransformerModel wrapper or return as-is."""
    return model.model if hasattr(model, "model") else model


def resolve_position(resid: torch.Tensor, position: int) -> int:
    """Resolve negative position index relative to sequence length."""
    if position < 0:
        return resid.shape[1] + position
    return position


def get_attention_pattern(cache: dict[str, torch.Tensor] | Any, layer: int) -> torch.Tensor | None:
    """Get attention pattern for a layer, handling TransformerLens key variants."""
    store = getattr(cache, "cache_dict", cache) if not isinstance(cache, dict) else cache
    for key in (f"blocks.{layer}.attn.hook_pattern", f"blocks.{layer}.attn.hook_attn"):
        if key in store:
            return store[key]
    return None


def ensure_output_dir(path: str | Path) -> Path:
    """Create output directory and return Path."""
    output_dir = Path(path) if isinstance(path, str) else path
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
