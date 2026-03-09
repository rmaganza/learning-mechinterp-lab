"""Model loading with support for GPT-2, Pythia, and TinyStories-scale models via TransformerLens."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

import torch
from transformer_lens import HookedTransformer


class ModelName(str, Enum):
    """Supported model identifiers for loading."""

    GPT2_SMALL = "gpt2"
    GPT2_MEDIUM = "gpt2-medium"
    GPT2_LARGE = "gpt2-large"
    GPT2_XL = "gpt2-xl"
    PYTHIA_14M = "EleutherAI/pythia-14m"
    PYTHIA_70M = "EleutherAI/pythia-70m"
    PYTHIA_160M = "EleutherAI/pythia-160m"
    PYTHIA_410M = "EleutherAI/pythia-410m"
    TINYSTORIES_1M = "tiny-stories-1M"
    TINYSTORIES_3M = "tiny-stories-3M"
    TINYSTORIES_33M = "tiny-stories-33M"
    # HuggingFace fallback for TinyStories variants not in TransformerLens
    TINYSTORIES_HF_10M = "vijaymohan/gpt2-tinystories-from-scratch-10m"


class ModelConfig:
    """Configuration for a loaded transformer model."""

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        d_model: int,
        d_head: int,
        d_mlp: int,
        d_vocab: int,
        n_ctx: int,
    ) -> None:
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_head
        self.d_mlp = d_mlp
        self.d_vocab = d_vocab
        self.n_ctx = n_ctx

    @classmethod
    def from_hooked_transformer(cls, model: HookedTransformer) -> ModelConfig:
        """Extract config from a HookedTransformer instance."""
        cfg = model.cfg
        return cls(
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            d_model=cfg.d_model,
            d_head=cfg.d_head,
            d_mlp=cfg.d_mlp,
            d_vocab=cfg.d_vocab,
            n_ctx=cfg.n_ctx,
        )


class TransformerModel:
    """Typed wrapper around HookedTransformer for mechanistic interpretability."""

    def __init__(
        self,
        model: HookedTransformer,
        name: str,
        config: ModelConfig,
    ) -> None:
        self._model = model
        self.name = name
        self.config = config

    @property
    def model(self) -> HookedTransformer:
        """Access the underlying HookedTransformer."""
        return self._model

    def to(self, device: str | torch.device) -> TransformerModel:
        """Move model to device."""
        self._model.to(device)
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        return_type: Literal["logits", "loss", "both"] = "logits",
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass. Input can be tokens [batch, pos] or string."""
        return self._model(input_ids, return_type=return_type, **kwargs)

    def run_with_cache(
        self,
        input_ids: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Run forward pass and return logits plus activation cache."""
        return self._model.run_with_cache(input_ids, **kwargs)

    def __repr__(self) -> str:
        return f"TransformerModel(name={self.name!r}, config={self.config})"


def load_model(
    model_name: ModelName | str = ModelName.GPT2_SMALL,
    device: str | torch.device | None = None,
    fold_ln: bool = True,
    center_writing_weights: bool = True,
    center_unembed: bool = True,
    use_tl_if_available: bool = True,
    **kwargs: Any,
) -> TransformerModel:
    """Load a transformer model for mechanistic interpretability.

    Supports GPT-2, Pythia, and TinyStories-scale models. Uses TransformerLens
    when the model is in its official list; falls back to HuggingFace for
    other models (e.g. custom TinyStories variants).

    Args:
        model_name: Model identifier (ModelName enum or string).
        device: Device to load model onto. Defaults to CUDA if available.
        fold_ln: Fold LayerNorm into subsequent linear layers.
        center_writing_weights: Center weights that write to residual stream.
        center_unembed: Center unembedding matrix.
        use_tl_if_available: Prefer TransformerLens when model is supported.
        **kwargs: Additional arguments passed to from_pretrained.

    Returns:
        TransformerModel wrapper with typed interface.

    Example:
        >>> model = load_model(ModelName.GPT2_SMALL)
        >>> logits, cache = model.run_with_cache(tokens)
    """
    name_str = model_name.value if isinstance(model_name, ModelName) else model_name

    # Models that TransformerLens supports natively
    tl_models = {
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "tiny-stories-1M",
        "tiny-stories-3M",
        "tiny-stories-33M",
    }
    # Pythia models - TransformerLens uses short names
    pythia_map = {
        "EleutherAI/pythia-14m": "pythia-14m",
        "EleutherAI/pythia-70m": "pythia-70m",
        "EleutherAI/pythia-160m": "pythia-160m",
        "EleutherAI/pythia-410m": "pythia-410m",
    }

    tl_name: str | None = None
    if name_str in tl_models:
        tl_name = name_str
    elif name_str in pythia_map:
        tl_name = pythia_map[name_str]

    if use_tl_if_available and tl_name is not None:
        model = HookedTransformer.from_pretrained(
            tl_name,
            fold_ln=fold_ln,
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
            device=device,
            **kwargs,
        )
    else:
        # Fallback: load via HuggingFace (for custom TinyStories etc.)
        model = HookedTransformer.from_pretrained(
            name_str,
            fold_ln=fold_ln,
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
            device=device,
            **kwargs,
        )

    config = ModelConfig.from_hooked_transformer(model)
    return TransformerModel(model=model, name=name_str, config=config)
