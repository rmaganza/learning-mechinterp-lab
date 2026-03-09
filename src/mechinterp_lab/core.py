"""Core utilities for model loading and config handling."""

from pathlib import Path
from typing import Any

import torch
import yaml
from transformer_lens import HookedTransformer


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config from file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_config_or_empty(config_path: Path | None) -> dict[str, Any]:
    """Load config from path if it exists, else return empty dict."""
    if config_path and config_path.exists():
        return load_config(config_path)
    return {}


# Map config-friendly names to TransformerLens model identifiers
_MODEL_NAME_MAP = {
    "gpt2-small": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "gpt2-xl": "gpt2-xl",
    "pythia-14m": "pythia-14m",
    "pythia-70m": "pythia-70m",
    "pythia-160m": "pythia-160m",
    "pythia-410m": "pythia-410m",
}


def load_model(
    model_name: str,
    device: str | None = None,
) -> HookedTransformer:
    """Load a HookedTransformer model by name."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tl_name = _MODEL_NAME_MAP.get(model_name, model_name)
    model = HookedTransformer.from_pretrained(tl_name, device=device)
    return model


def ensure_output_dirs(base_dir: Path) -> dict[str, Path]:
    """Create and return output directory paths."""
    dirs = {
        "plots": base_dir / "plots",
        "experiments": base_dir / "experiments",
        "attention": base_dir / "plots" / "attention",
        "neurons": base_dir / "plots" / "neurons",
        "contributions": base_dir / "plots" / "contributions",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
