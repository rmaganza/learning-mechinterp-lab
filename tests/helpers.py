"""Shared test helpers."""

import torch

# Default fake dimensions (GPT-2 small-like)
N_LAYERS = 12
N_HEADS = 12
D_MODEL = 768
D_VOCAB = 50257
BATCH = 1
SEQ_LEN = 5


def make_fake_cache(
    n_layers: int = N_LAYERS,
    batch: int = BATCH,
    seq_len: int = SEQ_LEN,
    d_model: int = D_MODEL,
    keys: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Build a fake activation cache dict with correct shapes.

    If keys is None, builds full cache. If keys is provided (e.g. ["hook_resid_pre"]),
    only builds those layer hook names for each layer.
    """
    cache: dict[str, torch.Tensor] = {}
    layer_hooks = ["hook_resid_pre", "hook_resid_post", "hook_attn_out", "hook_mlp_out"]
    build_layer = layer_hooks if keys is None else [k for k in layer_hooks if k in keys]

    for layer in range(n_layers):
        for k in build_layer:
            full_key = f"blocks.{layer}.{k}"
            if k == "hook_mlp_out":
                cache[full_key] = torch.randn(batch, seq_len, d_model * 4)
            else:
                cache[full_key] = torch.randn(batch, seq_len, d_model)

    if keys is None:
        cache["hook_embed"] = torch.randn(batch, seq_len, d_model)
        cache["hook_pos_embed"] = torch.randn(batch, seq_len, d_model)
        cache["ln_final.hook_normalized"] = torch.randn(batch, seq_len, d_model)

    return cache
