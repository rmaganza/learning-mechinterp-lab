"""Attention head analysis, MLP neuron analysis, and feature direction analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformer_lens import HookedTransformer


def _get_model(model: Any) -> HookedTransformer:
    """Extract HookedTransformer from wrapper or return as-is."""
    return model.model if hasattr(model, "model") else model


@dataclass
class HeadAnalysisResult:
    """Per-head attention analysis for a layer."""

    layer: int
    head: int
    pattern: torch.Tensor  # [batch, head, dest, src]
    output: torch.Tensor  # [batch, pos, d_model] for this head
    qk_circuit: torch.Tensor  # [d_model, d_model] QK circuit
    ov_circuit: torch.Tensor  # [d_model, d_model] OV circuit


def attention_head_analysis(
    model: HookedTransformer | Any,
    input_ids: torch.Tensor,
    layer: int,
    head: int | None = None,
) -> HeadAnalysisResult | list[HeadAnalysisResult]:
    """Analyze attention heads: patterns, outputs, and QK/OV circuits.

    Args:
        model: HookedTransformer or TransformerModel.
        input_ids: Token ids [batch, pos].
        layer: Layer index.
        head: Specific head index, or None for all heads in layer.

    Returns:
        HeadAnalysisResult or list of HeadAnalysisResult per head.
    """
    hooked = _get_model(model)
    _, cache = hooked.run_with_cache(input_ids)

    n_heads = hooked.cfg.n_heads
    heads_to_analyze = [head] if head is not None else list(range(n_heads))

    results: list[HeadAnalysisResult] = []
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
    resid_pre = cache[f"blocks.{layer}.hook_resid_pre"]

    W_Q = hooked.W_Q[layer]  # [n_heads, d_model, d_head]
    W_K = hooked.W_K[layer]
    W_V = hooked.W_V[layer]
    W_O = hooked.W_O[layer]

    for h in heads_to_analyze:
        # Per-head output: pattern @ V @ W_O
        # Pattern: [batch, head, dest, src], V: [batch, src, head, d_head]
        # For single head we need to index
        p = pattern[:, h, :, :]  # [batch, dest, src]
        v = resid_pre @ W_V[h]  # [batch, pos, d_head]
        head_out = torch.einsum("bds,bsd->bd", p, v) @ W_O[h]  # [batch, pos, d_model]

        # QK circuit: W_Q @ W_K^T
        qk = W_Q[h].T @ W_K[h]  # [d_model, d_model]
        ov = W_V[h] @ W_O[h]  # [d_model, d_model]

        results.append(
            HeadAnalysisResult(
                layer=layer,
                head=h,
                pattern=pattern[:, h : h + 1, :, :],
                output=head_out,
                qk_circuit=qk,
                ov_circuit=ov,
            )
        )

    return results[0] if head is not None else results


@dataclass
class MLPNeuronAnalysis:
    """Per-neuron MLP activation analysis."""

    layer: int
    neuron_idx: int
    activations: torch.Tensor  # [batch, pos]
    pre_activations: torch.Tensor  # [batch, pos] pre-ReLU
    out_direction: torch.Tensor  # [d_model] output direction


def mlp_neuron_analysis(
    model: HookedTransformer | Any,
    input_ids: torch.Tensor,
    layer: int,
    neuron_indices: list[int] | None = None,
) -> MLPNeuronAnalysis | list[MLPNeuronAnalysis]:
    """Analyze MLP neuron activations and output directions.

    Args:
        model: HookedTransformer or TransformerModel.
        input_ids: Token ids [batch, pos].
        layer: Layer index.
        neuron_indices: Which neurons to analyze, or None for top-k by activation.

    Returns:
        MLPNeuronAnalysis or list of analyses.
    """
    hooked = _get_model(model)
    _, cache = hooked.run_with_cache(input_ids)

    pre = cache[f"blocks.{layer}.mlp.hook_pre"]  # [batch, pos, d_mlp]
    post = cache[f"blocks.{layer}.mlp.hook_post"]  # [batch, pos, d_mlp] (after ReLU)

    W_out = hooked.W_out[layer]  # [d_mlp, d_model]

    d_mlp = pre.shape[-1]
    if neuron_indices is None:
        # Default: top 10 neurons by mean activation
        mean_act = post.mean(dim=(0, 1))
        _, top_indices = torch.topk(mean_act, min(10, d_mlp))
        neuron_indices = top_indices.tolist()

    results: list[MLPNeuronAnalysis] = []
    for idx in neuron_indices:
        activations = post[..., idx]  # [batch, pos]
        pre_act = pre[..., idx]
        out_direction = W_out[idx, :]  # [d_model]
        results.append(
            MLPNeuronAnalysis(
                layer=layer,
                neuron_idx=idx,
                activations=activations,
                pre_activations=pre_act,
                out_direction=out_direction,
            )
        )

    return results[0] if len(results) == 1 else results


@dataclass
class FeatureDirectionResult:
    """Result of projecting activations onto a feature direction."""

    projections: torch.Tensor  # [batch, pos]
    direction: torch.Tensor  # [d_model]
    cosine_with_unembed: torch.Tensor  # [d_vocab] cosine with each token dir


def feature_direction_analysis(
    model: HookedTransformer | Any,
    input_ids: torch.Tensor,
    direction: torch.Tensor,
    layer: int,
    position: int = -1,
) -> FeatureDirectionResult:
    """Project residual stream onto a feature direction and analyze.

    Args:
        model: HookedTransformer or TransformerModel.
        input_ids: Token ids [batch, pos].
        direction: Feature direction [d_model], will be normalized.
        layer: Layer index.
        position: Position to analyze (-1 = last).

    Returns:
        FeatureDirectionResult with projections and unembed similarities.
    """
    hooked = _get_model(model)
    _, cache = hooked.run_with_cache(input_ids)

    resid = cache[f"blocks.{layer}.hook_resid_post"]
    if position < 0:
        position = resid.shape[1] + position
    resid = resid[:, position, :]  # [batch, d_model]

    direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
    projections = (resid * direction).sum(dim=-1)  # [batch]

    # Cosine similarity with each unembedding direction
    W_U = hooked.W_U  # [d_model, d_vocab]
    W_U_norm = W_U / (W_U.norm(dim=0, keepdim=True) + 1e-8)
    cosine_with_unembed = (direction.unsqueeze(0) @ W_U_norm).squeeze(0)

    return FeatureDirectionResult(
        projections=projections,
        direction=direction,
        cosine_with_unembed=cosine_with_unembed,
    )
