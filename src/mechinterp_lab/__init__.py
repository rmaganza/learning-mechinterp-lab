"""Mechanistic interpretability lab for research-quality experiments on small transformer models."""

__version__ = "0.1.0"

from mechinterp_lab.analysis import (
    attention_head_analysis,
    feature_direction_analysis,
    mlp_neuron_analysis,
)
from mechinterp_lab.hooks import ActivationCache, capture_activations
from mechinterp_lab.models import ModelName, load_model
from mechinterp_lab.patching import activation_patch, causal_trace
from mechinterp_lab.probes import logit_lens, tuned_lens_probe

__all__ = [
    "__version__",
    "load_model",
    "ModelName",
    "capture_activations",
    "ActivationCache",
    "activation_patch",
    "causal_trace",
    "logit_lens",
    "tuned_lens_probe",
    "attention_head_analysis",
    "mlp_neuron_analysis",
    "feature_direction_analysis",
]
