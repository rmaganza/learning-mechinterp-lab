"""Neuron activation distribution visualization."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def plot_neuron_activation_distribution(
    activations: np.ndarray | torch.Tensor,
    layer: int,
    neuron_indices: list[int] | None = None,
    token_labels: list[str] | None = None,
    output_path: Path | str | None = None,
    backend: str = "matplotlib",
    figsize: tuple[int, int] = (12, 6),
) -> Path | None:
    """
    Plot distribution of neuron activations across positions or samples.

    Args:
        activations: Shape (batch, pos, d_mlp) or (pos, d_mlp)
        layer: Layer index for labeling
        neuron_indices: Specific neurons to plot; if None, plot first 8
        token_labels: Labels for x-axis (positions)
        output_path: Where to save
        backend: 'matplotlib', 'seaborn', or 'plotly'
        figsize: Figure size

    Returns:
        Path to saved file if output_path given, else None
    """
    if isinstance(activations, torch.Tensor):
        act = activations.detach().cpu().numpy()
    else:
        act = np.asarray(activations)

    if act.ndim == 2:
        act = act[np.newaxis, ...]

    n_neurons = act.shape[-1]
    if neuron_indices is None:
        neuron_indices = list(range(min(8, n_neurons)))
    else:
        neuron_indices = [i for i in neuron_indices if i < n_neurons]

    output_path = Path(output_path) if output_path else None

    if backend == "plotly":
        return _plot_neuron_plotly(act, layer, neuron_indices, token_labels, output_path)
    elif backend == "seaborn":
        return _plot_neuron_seaborn(act, layer, neuron_indices, token_labels, output_path, figsize)
    else:
        return _plot_neuron_matplotlib(
            act, layer, neuron_indices, token_labels, output_path, figsize
        )


def _plot_neuron_matplotlib(
    act: np.ndarray,
    layer: int,
    neuron_indices: list[int],
    token_labels: list[str] | None,
    output_path: Path | None,
    figsize: tuple[int, int],
) -> Path | None:
    n_neurons = len(neuron_indices)
    cols = min(n_neurons, 4)
    rows = (n_neurons + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if n_neurons == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for idx, ni in enumerate(neuron_indices):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]
        neuron_act = act[..., ni].flatten()
        ax.hist(neuron_act, bins=50, edgecolor="black", alpha=0.7)
        ax.set_title(f"Layer {layer}, Neuron {ni}")
        ax.set_xlabel("Activation")
        ax.set_ylabel("Count")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path
    plt.close()
    return None


def _plot_neuron_seaborn(
    act: np.ndarray,
    layer: int,
    neuron_indices: list[int],
    token_labels: list[str] | None,
    output_path: Path | None,
    figsize: tuple[int, int],
) -> Path | None:
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    for ni in neuron_indices[:4]:
        neuron_act = act[..., ni].flatten()
        sns.histplot(neuron_act, ax=axes[0], label=f"Neuron {ni}", kde=True, alpha=0.5)
    axes[0].set_title(f"Layer {layer} - Neuron Activation Distributions")
    axes[0].legend()

    if act.ndim >= 2 and token_labels:
        mean_act = act.mean(axis=0)
        for ni in neuron_indices[:4]:
            axes[1].plot(mean_act[:, ni], label=f"Neuron {ni}")
        axes[1].set_xticks(range(len(token_labels)))
        axes[1].set_xticklabels(token_labels, rotation=45, ha="right")
        axes[1].set_title("Mean Activation by Position")
        axes[1].legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path
    plt.close()
    return None


def _plot_neuron_plotly(
    act: np.ndarray,
    layer: int,
    neuron_indices: list[int],
    token_labels: list[str] | None,
    output_path: Path | None,
) -> Path | None:
    import plotly.graph_objects as go

    fig = go.Figure()
    for ni in neuron_indices[:8]:
        neuron_act = act[..., ni].flatten()
        fig.add_trace(go.Histogram(x=neuron_act, name=f"Neuron {ni}", opacity=0.6))
    fig.update_layout(
        title=f"Layer {layer} - Neuron Activation Distributions",
        barmode="overlay",
        xaxis_title="Activation",
        yaxis_title="Count",
    )
    if output_path:
        out = output_path.with_suffix(".html") if output_path.suffix != ".html" else output_path
        fig.write_html(str(out))
        return out
    return None
