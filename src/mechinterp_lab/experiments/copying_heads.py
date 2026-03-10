"""Identify attention heads involved in copying previous tokens."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from transformer_lens import HookedTransformer

from mechinterp_lab.utils import ensure_output_dir, get_attention_pattern


def run_copying_heads_experiment(
    model: HookedTransformer,
    prompt: str,
    output_dir: Path | str,
    layer_indices: list[int] | None = None,
    max_prev_token_offset: int = 10,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Identify attention heads that copy from previous token positions.

    Copying heads attend strongly to the previous token (position -1) when
    predicting the next token. We measure this via attention pattern analysis.

    Returns:
        Dict with copying scores per head, top copying heads, and paths to saved outputs.
    """
    output_dir = ensure_output_dir(output_dir)

    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[-1]

    if layer_indices is None:
        layer_indices = list(range(n_layers))

    copying_scores = np.zeros((n_layers, n_heads))
    attention_to_prev = {}

    for layer in layer_indices:
        pattern_tensor = get_attention_pattern(cache, layer)
        if pattern_tensor is None:
            continue

        pattern = pattern_tensor.squeeze(0)
        for head in range(n_heads):
            head_pattern = pattern[head].detach().cpu().numpy()
            if head_pattern.ndim == 3:
                head_pattern = head_pattern[0]
            n_q, n_k = head_pattern.shape
            score = 0.0
            count = 0
            for q_pos in range(1, min(n_q, seq_len)):
                for offset in range(1, min(max_prev_token_offset + 1, q_pos + 1)):
                    k_pos = q_pos - offset
                    if k_pos >= 0 and k_pos < n_k:
                        score += head_pattern[q_pos, k_pos]
                        count += 1
            if count > 0:
                copying_scores[layer, head] = score / count
            attention_to_prev[f"L{layer}H{head}"] = head_pattern

    top_heads = []
    for layer in range(n_layers):
        for head in range(n_heads):
            top_heads.append((layer, head, float(copying_scores[layer, head])))
    top_heads.sort(key=lambda x: x[2], reverse=True)
    top_heads = top_heads[:20]

    token_strs = model.to_str_tokens(prompt, prepend_bos=False)

    results = {
        "copying_scores": copying_scores.tolist(),
        "top_copying_heads": [(lyr, hd, sc) for lyr, hd, sc in top_heads],
        "prompt": prompt,
        "config": config or {},
    }

    np.save(output_dir / "copying_scores.npy", copying_scores)
    with open(output_dir / "copying_results.json", "w") as f:
        json.dump(
            {
                "top_copying_heads": [(lyr, hd, round(sc, 4)) for lyr, hd, sc in top_heads],
                "prompt": prompt,
            },
            f,
            indent=2,
        )

    return {
        "copying_scores": copying_scores,
        "top_copying_heads": top_heads,
        "attention_to_prev": attention_to_prev,
        "tokens": token_strs,
        "results": results,
        "output_dir": output_dir,
    }
