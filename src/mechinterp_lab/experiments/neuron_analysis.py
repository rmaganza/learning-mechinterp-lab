"""Neuron activation analysis for specific token classes."""

from pathlib import Path
from typing import Any

import numpy as np
from transformer_lens import HookedTransformer


def run_neuron_analysis_experiment(
    model: HookedTransformer,
    prompts: list[str],
    output_dir: Path | str,
    layer_indices: list[int] | None = None,
    token_classes: dict[str, list[int]] | None = None,
    max_neurons: int = 32,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Analyze MLP neuron activations for specific token classes.

    Token classes can be e.g. {"punctuation": [1,2,3], "numbers": [4,5,6]}
    (token type IDs or position indices). If None, analyzes all positions.

    Returns:
        Dict with activations by layer, token class stats, and output paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_layers = model.cfg.n_layers
    if layer_indices is None:
        layer_indices = list(range(n_layers))

    all_activations = {}
    activations_by_class = {}

    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

        for layer in layer_indices:
            hook_name = f"blocks.{layer}.mlp.hook_post"
            if hook_name not in cache:
                hook_name = f"blocks.{layer}.mlp.hook_pre"
            if hook_name not in cache:
                continue

            act = cache[hook_name].squeeze(0).detach().cpu().numpy()
            key = f"L{layer}"
            if key not in all_activations:
                all_activations[key] = []
            all_activations[key].append(act)

    for key in all_activations:
        all_activations[key] = np.concatenate(all_activations[key], axis=0)

    if token_classes:
        for class_name, positions in token_classes.items():
            activations_by_class[class_name] = {}
            for key, acts in all_activations.items():
                if positions:
                    mask = np.array([i in positions for i in range(acts.shape[0])])
                    if len(mask) > acts.shape[0]:
                        mask = mask[: acts.shape[0]]
                    subset = acts[mask] if mask.any() else acts
                else:
                    subset = acts
                activations_by_class[class_name][key] = subset

    results = {
        "layer_indices": layer_indices,
        "n_prompts": len(prompts),
        "token_classes": token_classes,
        "config": config or {},
    }

    np.savez_compressed(
        output_dir / "neuron_activations.npz",
        **dict(all_activations),
    )

    with open(output_dir / "neuron_analysis_results.json", "w") as f:
        import json

        json.dump(results, f, indent=2)

    return {
        "activations": all_activations,
        "activations_by_class": activations_by_class,
        "output_dir": output_dir,
        "results": results,
    }
