"""Token-level contribution change visualization."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def plot_token_contribution_changes(
    contributions: np.ndarray | torch.Tensor,
    tokens: list[str],
    layer_labels: list[str] | None = None,
    output_path: Path | str | None = None,
    backend: str = "matplotlib",
    figsize: tuple[int, int] = (12, 8),
) -> Path | None:
    """
    Plot token-level contribution changes across layers or interventions.

    Args:
        contributions: Shape (n_layers_or_conditions, n_tokens) or (n_tokens,)
        tokens: Token strings for x-axis
        layer_labels: Labels for each row/condition
        output_path: Where to save
        backend: 'matplotlib', 'seaborn', or 'plotly'
        figsize: Figure size

    Returns:
        Path to saved file if output_path given, else None
    """
    if isinstance(contributions, torch.Tensor):
        contrib = contributions.detach().cpu().numpy()
    else:
        contrib = np.asarray(contributions)

    if contrib.ndim == 1:
        contrib = contrib[np.newaxis, :]

    output_path = Path(output_path) if output_path else None

    if backend == "plotly":
        return _plot_contrib_plotly(contrib, tokens, layer_labels, output_path)
    elif backend == "seaborn":
        return _plot_contrib_seaborn(contrib, tokens, layer_labels, output_path, figsize)
    else:
        return _plot_contrib_matplotlib(contrib, tokens, layer_labels, output_path, figsize)


def _plot_contrib_matplotlib(
    contrib: np.ndarray,
    tokens: list[str],
    layer_labels: list[str] | None,
    output_path: Path | None,
    figsize: tuple[int, int],
) -> Path | None:
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(tokens))
    width = 0.8 / contrib.shape[0]
    for i in range(contrib.shape[0]):
        offset = (i - contrib.shape[0] / 2 + 0.5) * width
        label = layer_labels[i] if layer_labels and i < len(layer_labels) else f"Layer {i}"
        ax.bar(x + offset, contrib[i], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_ylabel("Contribution")
    ax.set_title("Token-Level Contribution Changes")
    ax.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path
    plt.close()
    return None


def _plot_contrib_seaborn(
    contrib: np.ndarray,
    tokens: list[str],
    layer_labels: list[str] | None,
    output_path: Path | None,
    figsize: tuple[int, int],
) -> Path | None:
    n_conditions, n_tokens = contrib.shape
    data = []
    for i in range(n_conditions):
        for j in range(n_tokens):
            label = layer_labels[i] if layer_labels and i < len(layer_labels) else f"L{i}"
            data.append({"Token": tokens[j], "Condition": label, "Contribution": contrib[i, j]})

    import pandas as pd

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=df, x="Token", y="Contribution", hue="Condition", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Token-Level Contribution Changes")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path
    plt.close()
    return None


def _plot_contrib_plotly(
    contrib: np.ndarray,
    tokens: list[str],
    layer_labels: list[str] | None,
    output_path: Path | None,
) -> Path | None:
    import plotly.graph_objects as go

    fig = go.Figure()
    for i in range(contrib.shape[0]):
        label = layer_labels[i] if layer_labels and i < len(layer_labels) else f"Layer {i}"
        fig.add_trace(go.Bar(name=label, x=tokens, y=contrib[i]))
    fig.update_layout(
        title="Token-Level Contribution Changes",
        barmode="group",
        xaxis_tickangle=-45,
    )
    if output_path:
        out = output_path.with_suffix(".html") if output_path.suffix != ".html" else output_path
        fig.write_html(str(out))
        return out
    return None
