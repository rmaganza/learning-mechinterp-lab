"""Activation capture hooks for transformer layers, attention, and MLP.

Returns structured activation tensors keyed by layer and component.
"""

from __future__ import annotations

from collections.abc import ItemsView, KeysView, ValuesView
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache as TLActivationCache

from mechinterp_lab.utils import get_model


class HookTarget(str, Enum):
    """Target locations for activation capture."""

    RESIDUAL_PRE = "resid_pre"
    RESIDUAL_POST = "resid_post"
    ATTN_INPUT = "attn_in"
    ATTN_OUTPUT = "attn_out"
    ATTN_PATTERN = "pattern"
    ATTN_Q = "q"
    ATTN_K = "k"
    ATTN_V = "v"
    MLP_INPUT = "mlp_in"
    MLP_OUTPUT = "mlp_out"
    MLP_PRE = "pre"
    MLP_POST = "post"


@dataclass
class ActivationSpec:
    """Specification for which activations to capture."""

    targets: list[HookTarget]
    layers: list[int] | None = None  # None = all layers
    include_embed: bool = True
    include_ln_final: bool = True


@dataclass
class ActivationCache:
    """Structured cache of captured activations.

    Keys follow TransformerLens naming: e.g. 'blocks.0.attn_out', 'blocks.0.mlp_out'.
    Also provides layer-indexed access for common patterns.
    """

    cache: dict[str, torch.Tensor] = field(default_factory=dict)
    _layer_activations: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict, repr=False)

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.cache[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.cache.get(key, default)

    def keys(self) -> KeysView[str]:
        return self.cache.keys()

    def values(self) -> ValuesView[torch.Tensor]:
        return self.cache.values()

    def items(self) -> ItemsView[str, torch.Tensor]:
        return self.cache.items()

    def residual_pre(self, layer: int) -> torch.Tensor | None:
        """Residual stream before layer (0-indexed)."""
        return self.get(f"blocks.{layer}.resid_pre")

    def residual_post(self, layer: int) -> torch.Tensor | None:
        """Residual stream after layer (0-indexed)."""
        return self.get(f"blocks.{layer}.resid_post")

    def attn_out(self, layer: int) -> torch.Tensor | None:
        """Attention output for layer."""
        return self.get(f"blocks.{layer}.attn_out")

    def mlp_out(self, layer: int) -> torch.Tensor | None:
        """MLP output for layer."""
        return self.get(f"blocks.{layer}.mlp_out")

    def pattern(self, layer: int) -> torch.Tensor | None:
        """Attention pattern for layer [batch, head, dest, src]."""
        return self.get(f"blocks.{layer}.attn.hook_pattern")

    def to_tl_cache(self, model: HookedTransformer | None = None) -> TLActivationCache:
        """Convert to TransformerLens ActivationCache for patching etc."""
        return TLActivationCache(self.cache, model=model)


def _build_hook_names(
    model: HookedTransformer,
    spec: ActivationSpec,
) -> list[str]:
    """Build list of TransformerLens hook names to cache."""
    layers = spec.layers
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    names: list[str] = []
    for layer in layers:
        base = f"blocks.{layer}."
        for target in spec.targets:
            if target == HookTarget.RESIDUAL_PRE:
                names.append(f"{base}hook_resid_pre")
            elif target == HookTarget.RESIDUAL_POST:
                names.append(f"{base}hook_resid_post")
            elif target == HookTarget.ATTN_OUTPUT:
                names.append(f"{base}hook_attn_out")
            elif target == HookTarget.ATTN_PATTERN:
                names.append(f"{base}attn.hook_pattern")
            elif target == HookTarget.ATTN_Q:
                names.append(f"{base}attn.hook_q")
            elif target == HookTarget.ATTN_K:
                names.append(f"{base}attn.hook_k")
            elif target == HookTarget.ATTN_V:
                names.append(f"{base}attn.hook_v")
            elif target in (HookTarget.MLP_INPUT, HookTarget.MLP_PRE):
                names.append(f"{base}mlp.hook_pre")
            elif target == HookTarget.MLP_OUTPUT:
                names.append(f"{base}hook_mlp_out")
            elif target == HookTarget.MLP_POST:
                names.append(f"{base}mlp.hook_post")

    if spec.include_embed:
        names.extend(["hook_embed", "hook_pos_embed"])
    if spec.include_ln_final:
        names.append("ln_final.hook_normalized")

    return list(dict.fromkeys(names))


def capture_activations(
    model: HookedTransformer | Any,
    input_ids: torch.Tensor,
    spec: ActivationSpec | None = None,
    names: list[str] | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, ActivationCache]:
    """Capture activations during a forward pass.

    Uses TransformerLens run_with_cache when possible. Returns logits and
    a structured ActivationCache.

    Args:
        model: HookedTransformer or TransformerModel (uses .model).
        input_ids: Token ids [batch, pos].
        spec: Which activations to capture. Defaults to resid, attn_out, mlp_out.
        names: Explicit hook names (overrides spec if provided).
        **kwargs: Passed to run_with_cache.

    Returns:
        (logits, ActivationCache)
    """
    hooked = get_model(model)

    if spec is None:
        spec = ActivationSpec(
            targets=[
                HookTarget.RESIDUAL_PRE,
                HookTarget.RESIDUAL_POST,
                HookTarget.ATTN_OUTPUT,
                HookTarget.MLP_OUTPUT,
            ],
        )

    if names is None:
        names = _build_hook_names(hooked, spec)

    # TransformerLens run_with_cache: names_filter can be list or callable
    names_set = set(names) if names else None
    logits, cache_obj = hooked.run_with_cache(
        input_ids,
        names_filter=(lambda n: n in names_set) if names_set else None,
        **kwargs,
    )

    # Extract cache dict from TransformerLens ActivationCache
    cache_dict = getattr(cache_obj, "cache_dict", cache_obj)
    if not isinstance(cache_dict, dict):
        cache_dict = dict(cache_obj)

    return logits, ActivationCache(cache=cache_dict)
