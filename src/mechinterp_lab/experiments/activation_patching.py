"""Activation patching for factual recall or induction behavior."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from transformer_lens import HookedTransformer
from transformer_lens.patching import (
    get_act_patch_attn_head_out_all_pos,
    get_act_patch_attn_head_out_by_pos,
)

from mechinterp_lab.patching import activation_patch, patch_residual_stream
from mechinterp_lab.utils import ensure_output_dir


def _find_subject_positions(
    model: HookedTransformer,
    prompt: str,
    subject: str,
) -> list[int]:
    """Find token positions for subject in prompt (Meng et al. style: last token of subject)."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    token_strs = model.to_str_tokens(prompt, prepend_bos=True)
    subject_clean = subject.strip().lower()
    positions = []
    for i, s in enumerate(token_strs):
        s_clean = s.strip().lower().replace(" ", "")
        if subject_clean in s_clean or s_clean == subject_clean.replace(" ", ""):
            positions.append(i)
    if not positions:
        # Fallback: match by token IDs
        subject_tokens = model.to_tokens(
            " " + subject if not subject.startswith(" ") else subject, prepend_bos=False
        )
        sub_ids = subject_tokens[0].tolist()
        tok_ids = tokens[0].tolist()
        for start in range(len(tok_ids) - len(sub_ids) + 1):
            if tok_ids[start : start + len(sub_ids)] == sub_ids:
                positions = list(range(start, start + len(sub_ids)))
                break
    return positions[-1:] if positions else []  # Last token of subject (ROME convention)


def _patch_residual_stream_noise_corruption(
    model: HookedTransformer,
    tokens: torch.Tensor,
    clean_cache: Any,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    subject_positions: list[int],
    noise_std: float,
) -> torch.Tensor:
    """Patch resid_pre with Meng et al. noise corruption (Gaussian noise on subject embeddings)."""
    n_layers = model.cfg.n_layers
    device = tokens.device
    results = torch.zeros(n_layers, device=device)

    cdict = getattr(clean_cache, "cache_dict", clean_cache)
    corruption_hook_name = (
        "blocks.0.hook_resid_pre" if "blocks.0.hook_resid_pre" in cdict else "hook_embed"
    )

    def make_corruption_hook(positions: list[int], std: float):
        def hook(act: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            out = act.clone()
            for pos in positions:
                if pos < out.shape[1]:
                    out[:, pos, :] = out[:, pos, :] + std * torch.randn_like(
                        out[:, pos, :], device=act.device
                    )
            return out

        return hook

    corruption_hook = make_corruption_hook(subject_positions, noise_std)

    for layer in range(n_layers):
        patch_name = f"blocks.{layer}.hook_resid_pre"
        clean_resid = clean_cache[patch_name]

        def make_patch_hook(resid: torch.Tensor):
            def patch_hook(act: torch.Tensor, **kwargs: Any) -> torch.Tensor:
                return resid.to(act.device)

            return patch_hook

        hooks = [
            (corruption_hook_name, corruption_hook),
            (patch_name, make_patch_hook(clean_resid)),
        ]
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        results[layer] = metric_fn(logits)

    return results


def _save_patching_results(
    output_dir: Path,
    results_np: np.ndarray,
    clean_prompt: str,
    corrupted_prompt: str,
    extra: dict,
) -> None:
    """Save patching results to npy and json."""
    np.save(output_dir / "patching_results.npy", results_np)
    with open(output_dir / "patching_results.json", "w") as f:
        json.dump(
            {
                "clean_prompt": clean_prompt,
                "corrupted_prompt": corrupted_prompt,
                **extra,
            },
            f,
            indent=2,
        )


def run_activation_patching_experiment(
    model: HookedTransformer,
    clean_prompt: str,
    corrupted_prompt: str,
    output_dir: Path | str,
    metric_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    target_token_id: int | None = None,
    target_token: str | None = None,
    layer_indices: list[int] | None = None,
    by_position: bool = False,
    activation_type: Literal["resid_pre", "attn_out", "mlp_out"] = "attn_out",
    corruption_method: Literal["prompt_swap", "noise"] = "prompt_swap",
    subject: str | None = None,
    noise_std: float | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run activation patching to identify components important for factual recall or induction.

    Clean prompt produces correct answer; corrupted produces wrong answer.
    Patching clean activations into corrupted run reveals which components matter.

    Args:
        model: HookedTransformer model
        clean_prompt: Prompt that yields correct output
        corrupted_prompt: Prompt that yields incorrect output (same length as clean for prompt_swap)
        output_dir: Where to save results
        metric_fn: Custom metric(logits) -> scalar; default is logit diff for target
        target_token_id: Token ID to measure (e.g. correct answer)
        target_token: Token string to measure (e.g. " Paris"); overrides target_token_id if set
        layer_indices: Layers to patch; None = all
        by_position: If True, patch by position (slower, attn_out only)
        activation_type: "resid_pre" (layer), "attn_out" (head), "mlp_out" (Meng et al. MLP)
        corruption_method: "prompt_swap" (default) or "noise" (Meng et al.: Gaussian noise on subject)
        subject: For noise corruption, the subject string (e.g. "France") to corrupt
        noise_std: For noise corruption, std of Gaussian; default 3 * embed_std
        config: Optional config dict

    Returns:
        Dict with patching results, top_heads (attn_out) or top_layers (resid_pre)
    """
    output_dir = ensure_output_dir(output_dir)

    use_noise_corruption = corruption_method == "noise" and activation_type != "mlp_out"
    clean_tokens = model.to_tokens(clean_prompt)
    if use_noise_corruption:
        if not subject:
            raise ValueError("subject is required when corruption_method is 'noise'")
        clean_prompt = corrupted_prompt  # Same prompt; corruption is via noise
        clean_tokens = model.to_tokens(clean_prompt)
        subject_positions = _find_subject_positions(model, clean_prompt, subject)
        if not subject_positions:
            raise ValueError(f"Could not find subject '{subject}' in prompt '{clean_prompt}'")
    else:
        subject_positions = []

    corrupted_tokens = (
        model.to_tokens(corrupted_prompt) if not use_noise_corruption else clean_tokens
    )

    if not use_noise_corruption and clean_tokens.shape != corrupted_tokens.shape:
        raise ValueError(
            "Clean and corrupted prompts must have same token length. "
            f"Got {clean_tokens.shape[-1]} vs {corrupted_tokens.shape[-1]}"
        )

    # Must use remove_batch_dim=False for attn_out: TransformerLens head patching expects
    # [batch, pos, head, d_head]; with remove_batch_dim the cache has [pos, head, d_head]
    # and layer_head_vector_patch_setter's [:, :, head_index] indexes the wrong axis.
    _, clean_cache = model.run_with_cache(clean_tokens, remove_batch_dim=False)

    if use_noise_corruption and noise_std is None:
        # Meng et al.: ~3 * std of embeddings; noise_scale in config for stronger corruption
        cdict = getattr(clean_cache, "cache_dict", clean_cache)
        emb = cdict.get("hook_embed", cdict.get("blocks.0.hook_resid_pre"))
        patching_cfg = (config or {}).get("patching", {})
        noise_scale = patching_cfg.get("noise_scale", 3.0)
        noise_std = noise_scale * float(emb.std().item())

    if metric_fn is None:
        if target_token is not None:
            target_token_id = model.to_single_token(target_token)
        elif target_token_id is None:
            target_token_id = clean_tokens[0, -1].item()
        correct_token = target_token_id

        def default_metric(logits: torch.Tensor) -> torch.Tensor:
            last_pos = logits.shape[1] - 1
            return logits[0, last_pos, correct_token] - logits[0, last_pos, :].logsumexp(dim=-1)

        metric_fn = default_metric

    n_layers = model.cfg.n_layers
    if layer_indices is None:
        layer_indices = list(range(n_layers))

    if activation_type in ("resid_pre", "mlp_out"):
        if activation_type == "resid_pre" and use_noise_corruption:
            layer_effects = _patch_residual_stream_noise_corruption(
                model,
                corrupted_tokens,
                clean_cache,
                metric_fn,
                subject_positions,
                noise_std,
            )
        else:
            if activation_type == "mlp_out":
                layer_effects = activation_patch(
                    model, corrupted_tokens, clean_cache, metric_fn, activation_type="mlp_out"
                )
            else:
                layer_effects = patch_residual_stream(
                    model, corrupted_tokens, clean_cache, metric_fn
                )
        # Average over positions if needed
        if layer_effects.dim() > 1:
            layer_effects = layer_effects.mean(dim=-1)
        results_np = layer_effects.detach().cpu().numpy()
        top_layers = sorted(
            enumerate(results_np.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        results = {
            "layer_effects": results_np.tolist(),
            "top_layers": [(int(lyr), float(val)) for lyr, val in top_layers],
            "clean_prompt": clean_prompt,
            "corrupted_prompt": corrupted_prompt,
            "activation_type": activation_type,
            "corruption_method": corruption_method,
            "subject": subject if use_noise_corruption else None,
            "config": config or {},
        }
        _save_patching_results(
            output_dir,
            results_np,
            clean_prompt,
            corrupted_prompt,
            {
                "top_layers": [[int(lyr), float(val)] for lyr, val in top_layers[:5]],
                "layer_effects": results_np.tolist(),
            },
        )
        return {
            "layer_effects": layer_effects,
            "top_layers": top_layers,
            "clean_prompt": clean_prompt,
            "corrupted_prompt": corrupted_prompt,
            "output_dir": output_dir,
            "results": results,
        }

    if by_position:
        patching_results = get_act_patch_attn_head_out_by_pos(
            model,
            corrupted_tokens,
            clean_cache,
            metric_fn,
            index_axis_names=("layer", "pos", "head"),
        )
        results_np = patching_results.detach().cpu().numpy()
    else:
        patching_results = get_act_patch_attn_head_out_all_pos(
            model,
            corrupted_tokens,
            clean_cache,
            metric_fn,
        )
        results_np = patching_results.detach().cpu().numpy()

    head_importance = results_np
    if by_position and results_np.ndim >= 3:
        head_importance = results_np.mean(axis=1)

    flat_importance = head_importance.flatten()
    top_indices = np.argsort(flat_importance)[::-1][:20]
    top_heads = []
    for idx in top_indices:
        if by_position and results_np.ndim == 3:
            layer, pos, head = np.unravel_index(idx, results_np.shape)
            top_heads.append((int(layer), int(pos), int(head), float(flat_importance[idx])))
        else:
            layer, head = np.unravel_index(idx, head_importance.shape)
            top_heads.append((int(layer), int(head), float(flat_importance[idx])))

    results = {
        "patching_results": results_np.tolist(),
        "top_heads": top_heads,
        "clean_prompt": clean_prompt,
        "corrupted_prompt": corrupted_prompt,
        "config": config or {},
    }
    _save_patching_results(
        output_dir,
        results_np,
        clean_prompt,
        corrupted_prompt,
        {"top_heads": [[int(x) for x in t[:3]] + [float(t[-1])] for t in top_heads[:10]]},
    )
    return {
        "patching_results": patching_results,
        "head_importance": head_importance,
        "top_heads": top_heads,
        "clean_prompt": clean_prompt,
        "corrupted_prompt": corrupted_prompt,
        "output_dir": output_dir,
        "results": results,
    }


def run_multi_fact_patching_experiment(
    model: HookedTransformer,
    fact_tuples: list[tuple[str, str, str]],
    output_dir: Path | str,
    activation_type: Literal["resid_pre", "attn_out"] = "resid_pre",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run activation patching across multiple (clean, corrupted, target_token) facts.
    Aggregates layer/head importance across facts for robust factual recall analysis.

    Returns:
        Dict with aggregated top_layers or top_heads, per-fact results.
    """
    output_dir = ensure_output_dir(output_dir)

    all_layer_effects: list[np.ndarray] = []
    all_head_importance: list[np.ndarray] = []
    per_fact_results = []

    for idx, (clean, corrupted, target_token) in enumerate(fact_tuples):
        try:
            result = run_activation_patching_experiment(
                model,
                clean,
                corrupted,
                output_dir / f"fact_{idx}",
                target_token=target_token,
                activation_type=activation_type,
                config=config,
            )
        except ValueError:
            continue
        if "top_layers" in result:
            layer_effects = result["layer_effects"].detach().cpu().numpy()
            all_layer_effects.append(layer_effects)
        else:
            head_imp = result.get("head_importance")
            if head_imp is not None:
                imp_np = head_imp if isinstance(head_imp, np.ndarray) else np.array(head_imp)
                all_head_importance.append(imp_np)
        per_fact_results.append({"clean": clean, "corrupted": corrupted, "target": target_token})

    if activation_type == "resid_pre" and all_layer_effects:
        agg = np.mean(all_layer_effects, axis=0)
        top_layers = sorted(
            enumerate(agg.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        results = {
            "aggregated_layer_effects": agg.tolist(),
            "top_layers": [(int(layer), float(v)) for layer, v in top_layers],
            "n_facts": len(all_layer_effects),
            "per_fact": per_fact_results,
        }
        np.save(output_dir / "aggregated_patching.npy", agg)
        with open(output_dir / "multi_fact_results.json", "w") as f:
            json.dump(results, f, indent=2)
        return {
            "layer_effects": agg,
            "top_layers": top_layers,
            "n_facts": len(all_layer_effects),
            "output_dir": output_dir,
        }
    elif activation_type == "attn_out" and all_head_importance:
        agg = np.mean(all_head_importance, axis=0)
        flat = agg.flatten()
        top_indices = np.argsort(flat)[::-1][:20]
        top_heads = [
            (
                int(np.unravel_index(i, agg.shape)[0]),
                int(np.unravel_index(i, agg.shape)[1]),
                float(flat[i]),
            )
            for i in top_indices
        ]
        results = {"top_heads": top_heads, "n_facts": len(all_head_importance)}
        with open(output_dir / "multi_fact_results.json", "w") as f:
            json.dump(results, f, indent=2)
        return {
            "top_heads": top_heads,
            "n_facts": len(all_head_importance),
            "output_dir": output_dir,
        }
    return {"output_dir": output_dir, "n_facts": 0}
