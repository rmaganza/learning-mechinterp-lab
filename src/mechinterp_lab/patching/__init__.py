"""Activation patching and causal tracing for mechanistic interpretability."""

from mechinterp_lab.patching.activation_patching import (
    activation_patch,
    causal_trace,
    patch_attention_head,
    patch_residual_stream,
)

__all__ = [
    "activation_patch",
    "causal_trace",
    "patch_residual_stream",
    "patch_attention_head",
]
