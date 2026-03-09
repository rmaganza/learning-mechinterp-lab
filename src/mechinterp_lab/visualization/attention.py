"""Attention pattern visualization."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def plot_attention_heatmap(
    attention_pattern: np.ndarray | torch.Tensor,
    layer: int,
    head: int,
    tokens: list[str] | None = None,
    output_path: Path | str | None = None,
    backend: str = "matplotlib",
    figsize: tuple[int, int] = (10, 8),
) -> Path | None:
    """
    Plot attention pattern as a heatmap for a specific layer/head.

    Args:
        attention_pattern: Shape (seq_len, seq_len) or (n_heads, seq_len, seq_len)
        layer: Layer index for labeling
        head: Head index for labeling
        tokens: Token strings for axis labels
        output_path: Where to save the plot
        backend: 'matplotlib', 'seaborn', or 'plotly'
        figsize: Figure size for matplotlib/seaborn

    Returns:
        Path to saved file if output_path given, else None
    """
    if isinstance(attention_pattern, torch.Tensor):
        attn = attention_pattern.detach().cpu().numpy()
    else:
        attn = np.asarray(attention_pattern)

    if attn.ndim == 3:
        attn = attn[head] if head < attn.shape[0] else attn[0]

    output_path = Path(output_path) if output_path else None

    if backend == "plotly":
        return _plot_attention_plotly(attn, layer, head, tokens, output_path)
    elif backend == "seaborn":
        return _plot_attention_seaborn(attn, layer, head, tokens, output_path, figsize)
    else:
        return _plot_attention_matplotlib(attn, layer, head, tokens, output_path, figsize)


def _plot_attention_matplotlib(
    attn: np.ndarray,
    layer: int,
    head: int,
    tokens: list[str] | None,
    output_path: Path | None,
    figsize: tuple[int, int],
) -> Path | None:
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(attn, cmap="viridis", aspect="auto", interpolation="nearest")
    ax.set_title(f"Attention Pattern - Layer {layer}, Head {head}")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    if tokens:
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path
    plt.close()
    return None


def _plot_attention_seaborn(
    attn: np.ndarray,
    layer: int,
    head: int,
    tokens: list[str] | None,
    output_path: Path | None,
    figsize: tuple[int, int],
) -> Path | None:
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        attn,
        ax=ax,
        cmap="viridis",
        xticklabels=tokens if tokens else False,
        yticklabels=tokens if tokens else False,
        cbar_kws={"label": "Attention"},
    )
    ax.set_title(f"Attention Pattern - Layer {layer}, Head {head}")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    if tokens:
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path
    plt.close()
    return None


def _plot_attention_plotly(
    attn: np.ndarray,
    layer: int,
    head: int,
    tokens: list[str] | None,
    output_path: Path | None,
) -> Path | None:
    import plotly.express as px

    fig = px.imshow(
        attn,
        labels={"x": "Key", "y": "Query", "color": "Attention"},
        x=tokens,
        y=tokens,
        title=f"Attention Pattern - Layer {layer}, Head {head}",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(xaxis_tickangle=-45)
    if output_path:
        fig.write_html(str(output_path.with_suffix(".html")))
        return output_path.with_suffix(".html")
    return None


def plot_attention_by_layer_head(
    attention_patterns: dict[str, np.ndarray | torch.Tensor],
    tokens: list[str] | None = None,
    output_path: Path | str | None = None,
    max_heads: int = 4,
) -> Path | None:
    """
    Plot attention patterns for multiple layer/head combinations.

    attention_patterns: Dict mapping "L{layer}H{head}" -> pattern array
    """
    output_path = Path(output_path) if output_path else None
    keys = list(attention_patterns.keys())[:max_heads]
    n = len(keys)
    if n == 0:
        return None

    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for idx, key in enumerate(keys):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]
        attn = attention_patterns[key]
        if isinstance(attn, torch.Tensor):
            attn = attn.detach().cpu().numpy()
        ax.imshow(attn, cmap="viridis", aspect="auto")
        ax.set_title(key)
        if tokens:
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha="right")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path
    plt.close()
    return None
