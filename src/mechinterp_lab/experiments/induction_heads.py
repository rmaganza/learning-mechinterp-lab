"""Induction heads experiment (Olsson et al. 2022).

Identifies heads that implement the [A][B]...[A] -> [B] pattern: at the second occurrence
of token A, attend to the token B that followed the first A, to predict B.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from transformer_lens import HookedTransformer


def run_induction_heads_experiment(
    model: HookedTransformer,
    prompt: str,
    output_dir: Path | str,
    layer_indices: list[int] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Identify induction heads: heads that attend from the last token (A) to the
    position of the token (B) that followed the previous occurrence of A.

    Uses prompts with repeated patterns, e.g. "The cat sat on the mat. The cat sat on the"
    where the model should predict "mat" by attending from "the" to the first "mat".

    Returns:
        Dict with induction scores per head, top induction heads, attention patterns.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokens = model.to_tokens(prompt)
    token_ids = tokens[0].tolist()
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(token_ids)

    # Find induction target: last token A, previous occurrence of A, token B that followed
    if seq_len < 3:
        raise ValueError("Prompt too short for induction pattern")
    last_token_id = token_ids[-1]
    prev_occurrence = None
    for i in range(seq_len - 2, -1, -1):
        if token_ids[i] == last_token_id:
            prev_occurrence = i
            break
    if prev_occurrence is None:
        raise ValueError("No previous occurrence of last token for induction pattern")
    induction_target_pos = prev_occurrence + 1  # B follows A
    if induction_target_pos >= seq_len - 1:
        raise ValueError("Induction target position invalid")

    if layer_indices is None:
        layer_indices = list(range(n_layers))

    induction_scores = np.zeros((n_layers, n_heads))
    attention_to_target = {}

    for layer in layer_indices:
        attn_key = f"blocks.{layer}.attn.hook_pattern"
        if attn_key not in cache:
            attn_key = f"blocks.{layer}.attn.hook_attn"
        if attn_key not in cache:
            continue

        pattern = cache[attn_key].squeeze(0)
        for head in range(n_heads):
            head_pattern = pattern[head].detach().cpu().numpy()
            if head_pattern.ndim == 3:
                head_pattern = head_pattern[0]
            # Attention from last query position to induction target (B)
            score = float(head_pattern[seq_len - 1, induction_target_pos])
            induction_scores[layer, head] = score
            attention_to_target[f"L{layer}H{head}"] = head_pattern

    top_heads = []
    for layer in range(n_layers):
        for head in range(n_heads):
            top_heads.append((layer, head, float(induction_scores[layer, head])))
    top_heads.sort(key=lambda x: x[2], reverse=True)
    top_heads = top_heads[:20]

    token_strs = model.to_str_tokens(prompt, prepend_bos=False)
    target_token_str = model.to_string(token_ids[induction_target_pos])

    results = {
        "induction_scores": induction_scores.tolist(),
        "top_induction_heads": [(lyr, hd, round(sc, 4)) for lyr, hd, sc in top_heads],
        "prompt": prompt,
        "last_token": model.to_string(last_token_id),
        "induction_target_pos": induction_target_pos,
        "induction_target_token": target_token_str,
        "config": config or {},
    }

    np.save(output_dir / "induction_scores.npy", induction_scores)
    with open(output_dir / "induction_results.json", "w") as f:
        json.dump(
            {
                "top_induction_heads": results["top_induction_heads"],
                "prompt": prompt,
                "induction_target_token": target_token_str,
            },
            f,
            indent=2,
        )

    return {
        "induction_scores": induction_scores,
        "top_induction_heads": top_heads,
        "attention_to_target": attention_to_target,
        "tokens": token_strs,
        "induction_target_token": target_token_str,
        "results": results,
        "output_dir": output_dir,
    }
