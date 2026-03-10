"""Logit lens analysis across layers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformer_lens import HookedTransformer

from mechinterp_lab.utils import ensure_output_dir


def run_logit_lens_experiment(
    model: HookedTransformer,
    prompt: str,
    output_dir: Path | str,
    layer_indices: list[int] | None = None,
    top_k: int = 5,
    target_token: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run logit lens: project residual stream at each layer through unembedding
    to see when the model "decides" on its prediction.

    Args:
        target_token: If provided, track this token (e.g. " Paris") instead of
            the model's top prediction. Useful for factual recall analysis.
    Returns:
        Dict with per-layer top predictions, when correct answer appears, etc.
    """
    output_dir = ensure_output_dir(output_dir)

    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)

    n_layers = model.cfg.n_layers
    if layer_indices is None:
        layer_indices = list(range(n_layers + 1))

    W_U = model.W_U
    ln_f = model.ln_final if hasattr(model, "ln_final") else None

    if target_token is not None:
        target_token_id = model.to_single_token(target_token)
        target_str = target_token
    else:
        target_token_id = logits[0, -1].argmax(dim=-1).item()
        target_str = model.to_string(target_token_id)

    layer_predictions = []

    for layer in layer_indices:
        if layer == 0:
            if "hook_embed" in cache:
                resid = cache["hook_embed"].squeeze(0)
            elif "blocks.0.hook_resid_pre" in cache:
                resid = cache["blocks.0.hook_resid_pre"].squeeze(0)
            else:
                continue
        elif layer <= n_layers:
            hook_name = f"blocks.{layer - 1}.hook_resid_post"
            if hook_name not in cache:
                hook_name = f"blocks.{layer - 1}.hook_resid_final"
            if hook_name not in cache:
                continue
            resid = cache[hook_name].squeeze(0)
        else:
            continue

        if ln_f is not None:
            resid = ln_f(resid)
        resid = resid.detach()

        last_pos = resid.shape[0] - 1
        hidden = resid[last_pos : last_pos + 1]
        lens_logits = (hidden @ W_U).squeeze(0)

        probs = torch.softmax(lens_logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, top_k)
        pred_strs = [model.to_string(i.item()) for i in top_ids]

        target_rank = (lens_logits > lens_logits[target_token_id]).sum().item() + 1
        target_prob = probs[target_token_id].item()

        layer_predictions.append(
            {
                "layer": layer,
                "top_tokens": pred_strs,
                "top_probs": top_probs.detach().cpu().numpy().tolist(),
                "target_token": target_str,
                "target_rank": int(target_rank),
                "target_prob": float(target_prob),
            }
        )

    first_correct_layer = None
    for lp in layer_predictions:
        if lp["top_tokens"][0] == target_str or lp["target_rank"] == 1:
            first_correct_layer = lp["layer"]
            break

    results = {
        "prompt": prompt,
        "target_token": target_str,
        "layer_predictions": layer_predictions,
        "first_correct_layer": first_correct_layer,
        "config": config or {},
    }

    with open(output_dir / "logit_lens_results.json", "w") as f:
        json.dump(
            {
                "prompt": prompt,
                "target_token": target_str,
                "first_correct_layer": first_correct_layer,
                "layer_predictions": [
                    {
                        "layer": lp["layer"],
                        "top_tokens": lp["top_tokens"],
                        "target_prob": lp["target_prob"],
                    }
                    for lp in layer_predictions
                ],
            },
            f,
            indent=2,
        )

    return {
        "layer_predictions": layer_predictions,
        "first_correct_layer": first_correct_layer,
        "prompt": prompt,
        "target_token": target_str,
        "output_dir": output_dir,
        "results": results,
    }
