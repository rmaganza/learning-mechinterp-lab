"""Typer-based CLI for mechanistic interpretability analyses."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from mechinterp_lab.core import ensure_output_dirs, load_config_or_empty, load_model
from mechinterp_lab.experiments import (
    run_activation_patching_experiment,
    run_copying_heads_experiment,
    run_induction_heads_experiment,
    run_logit_lens_experiment,
    run_multi_fact_patching_experiment,
    run_neuron_analysis_experiment,
)
from mechinterp_lab.visualization import (
    plot_attention_heatmap,
    plot_neuron_activation_distribution,
)

app = typer.Typer(
    name="mechinterp",
    help="Mechanistic interpretability lab - run analyses on transformer models",
)
console = Console()


def _cli_context(config_path: Path | None, model_name: str | None, output_dir: Path | None):
    """Load config and resolve model/output. Returns (config, model_name, output_base, dirs)."""
    config = load_config_or_empty(config_path)
    model_name = model_name or config.get("model", {}).get("name", "gpt2-small")
    output_base = output_dir or Path("output")
    dirs = ensure_output_dirs(output_base)
    return config, model_name, output_base, dirs


def _activation_patching_params(config: dict, clean: str, corrupted: str):
    """Extract activation patching params from config. Config overrides defaults."""
    prompts = config.get("prompts", {}).get("activation_patching", {})
    if isinstance(prompts, dict):
        clean = prompts.get("clean", clean)
        corrupted = prompts.get("corrupted", corrupted)
        target_token = prompts.get("target_token")
    else:
        target_token = None
    patching_cfg = config.get("patching", {})
    activation_type = patching_cfg.get("activation_type", "attn_out")
    corruption_method = patching_cfg.get("corruption_method", "prompt_swap")
    subject = patching_cfg.get("subject")
    noise_std = patching_cfg.get("noise_std")
    return clean, corrupted, target_token, activation_type, corruption_method, subject, noise_std


@app.command()
def run_experiment(
    experiment: str = typer.Argument(
        ...,
        help="Experiment: induction-heads, copying-heads, activation-patching, multi-fact-patching, neuron-analysis, logit-lens",
    ),
    config_path: Path = typer.Option(
        None, "--config", "-c", path_type=Path, help="Path to YAML config"
    ),
    output_dir: Path = typer.Option(
        None, "--output", "-o", path_type=Path, help="Output directory"
    ),
    model_name: str = typer.Option(None, "--model", "-m", help="Override model name"),
    prompt: str = typer.Option(
        None, "--prompt", "-p", help="Override prompt (for single-prompt experiments)"
    ),
) -> None:
    """Run a reproducible experiment from config or overrides."""
    config, model_name, output_base, dirs = _cli_context(config_path, model_name, output_dir)
    if config_path and config_path.exists():
        console.print(f"[green]Loaded config from {config_path}[/green]")
    exp_dir = dirs["experiments"] / experiment.replace("-", "_")

    console.print(f"[bold]Loading model: {model_name}[/bold]")
    model = load_model(model_name)

    if experiment == "induction-heads":
        prompt = prompt or config.get("prompts", {}).get(
            "induction_heads", "The cat sat on the mat. The cat sat on the"
        )
        result = run_induction_heads_experiment(
            model,
            prompt,
            exp_dir,
            layer_indices=config.get("model", {}).get("layer_indices"),
            config=config,
        )
        console.print(f"[green]Top induction heads:[/green] {result['top_induction_heads'][:5]}")
        console.print(f"[green]Target token: {result['induction_target_token']}[/green]")
        if result.get("attention_to_target") and result.get("tokens"):
            plot_dir = dirs["attention"]
            for key, pattern in list(result["attention_to_target"].items())[:4]:
                layer_str, head_str = key.replace("L", "").split("H")
                out_path = plot_dir / f"induction_{key}.png"
                plot_attention_heatmap(
                    pattern,
                    int(layer_str),
                    int(head_str),
                    tokens=result["tokens"],
                    output_path=out_path,
                )
            console.print(f"[green]Attention plots saved to {plot_dir}[/green]")

    elif experiment == "copying-heads":
        prompt = prompt or config.get("prompts", {}).get(
            "copying", "The quick brown fox jumps over the lazy dog."
        )
        result = run_copying_heads_experiment(
            model,
            prompt,
            exp_dir,
            layer_indices=config.get("model", {}).get("layer_indices"),
            config=config,
        )
        console.print(f"[green]Top copying heads:[/green] {result['top_copying_heads'][:5]}")
        if result.get("attention_to_prev") and result.get("tokens"):
            plot_dir = dirs["attention"]
            for key, pattern in list(result["attention_to_prev"].items())[:4]:
                layer_str, head_str = key.replace("L", "").split("H")
                out_path = plot_dir / f"copying_{key}.png"
                plot_attention_heatmap(
                    pattern,
                    int(layer_str),
                    int(head_str),
                    tokens=result["tokens"],
                    output_path=out_path,
                )
            console.print(f"[green]Attention plots saved to {plot_dir}[/green]")

    elif experiment == "activation-patching":
        clean, corrupted, target_token, activation_type, corruption_method, subject, noise_std = (
            _activation_patching_params(
                config, "The capital of France is", "The capital of Germany is"
            )
        )
        result = run_activation_patching_experiment(
            model,
            clean,
            corrupted,
            exp_dir,
            target_token=target_token,
            activation_type=activation_type,
            corruption_method=corruption_method,
            subject=subject,
            noise_std=noise_std,
            config=config,
        )
        if "top_layers" in result:
            act_type = result.get("results", {}).get("activation_type", "resid_pre")
            console.print(
                f"[green]Top patched layers ({act_type}):[/green] {result['top_layers'][:5]}"
            )
        else:
            console.print(f"[green]Top patched heads:[/green] {result['top_heads'][:5]}")

    elif experiment == "multi-fact-patching":
        fact_tuples = config.get("prompts", {}).get("multi_fact_patching", [])
        if not fact_tuples:
            fact_tuples = [
                ("The capital of France is", "The capital of Germany is", " Paris"),
                ("The capital of England is", "The capital of France is", " London"),
            ]
        activation_type = config.get("patching", {}).get("activation_type", "resid_pre")
        result = run_multi_fact_patching_experiment(
            model,
            fact_tuples,
            exp_dir,
            activation_type=activation_type,
            config=config,
        )
        if "top_layers" in result:
            console.print(
                f"[green]Aggregated top layers ({result['n_facts']} facts):[/green] {result['top_layers'][:5]}"
            )
        elif "top_heads" in result:
            console.print(
                f"[green]Aggregated top heads ({result['n_facts']} facts):[/green] {result['top_heads'][:5]}"
            )

    elif experiment == "neuron-analysis":
        prompts = config.get("prompts", {}).get("neuron_analysis", ["Hello world.", "The cat sat."])
        if prompt:
            prompts = [prompt]
        result = run_neuron_analysis_experiment(
            model,
            prompts,
            exp_dir,
            layer_indices=config.get("model", {}).get("layer_indices"),
            config=config,
        )
        console.print(f"[green]Neuron analysis saved to {result['output_dir']}[/green]")

    elif experiment == "logit-lens":
        prompt = prompt or config.get("prompts", {}).get("logit_lens", "The capital of France is")
        target_token = config.get("prompts", {}).get("logit_lens_target")
        result = run_logit_lens_experiment(
            model,
            prompt,
            exp_dir,
            layer_indices=config.get("model", {}).get("layer_indices"),
            target_token=target_token,
            config=config,
        )
        console.print(f"[green]First correct at layer: {result['first_correct_layer']}[/green]")
        if target_token:
            console.print(f"[green]Tracking token: {target_token}[/green]")

    else:
        console.print(f"[red]Unknown experiment: {experiment}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Results saved to {exp_dir}[/green]")


@app.command()
def analyze_attention(
    config_path: Path = typer.Option(
        None, "--config", "-c", path_type=Path, help="Path to YAML config"
    ),
    output_dir: Path = typer.Option(
        None, "--output", "-o", path_type=Path, help="Output directory"
    ),
    model_name: str = typer.Option(None, "--model", "-m", help="Model name"),
    prompt: str = typer.Option("The quick brown fox", "--prompt", "-p", help="Prompt to analyze"),
    layer: int = typer.Option(0, "--layer", "-l", help="Layer index"),
    head: int = typer.Option(0, "--head", "-H", help="Head index"),
    backend: str = typer.Option(
        "matplotlib", "--backend", "-b", help="Plot backend: matplotlib, seaborn, plotly"
    ),
) -> None:
    """Analyze and plot attention patterns for a layer/head."""
    config, model_name, _, dirs = _cli_context(config_path, model_name, output_dir)
    attn_dir = dirs["attention"]

    model = load_model(model_name)
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

    hook_key = f"blocks.{layer}.attn.hook_pattern"
    if hook_key not in cache:
        hook_key = f"blocks.{layer}.attn.hook_attn"
    if hook_key not in cache:
        console.print(f"[red]Attention not found for layer {layer}[/red]")
        raise typer.Exit(1)

    pattern = cache[hook_key].squeeze(0)
    token_strs = model.to_str_tokens(prompt, prepend_bos=False)

    out_path = attn_dir / f"attention_L{layer}_H{head}.png"
    if backend == "plotly":
        out_path = attn_dir / f"attention_L{layer}_H{head}.html"
    plot_attention_heatmap(
        pattern,
        layer,
        head,
        tokens=token_strs,
        output_path=out_path,
        backend=backend,
    )
    console.print(f"[green]Saved attention plot to {out_path}[/green]")


@app.command()
def patch_activations(
    config_path: Path = typer.Option(
        None, "--config", "-c", path_type=Path, help="Path to YAML config"
    ),
    output_dir: Path = typer.Option(
        None, "--output", "-o", path_type=Path, help="Output directory"
    ),
    model_name: str = typer.Option(None, "--model", "-m", help="Model name"),
    clean_prompt: str = typer.Option("The capital of France is", "--clean", help="Clean prompt"),
    corrupted_prompt: str = typer.Option(
        "The capital of Germany is", "--corrupted", help="Corrupted prompt"
    ),
) -> None:
    """Run activation patching between clean and corrupted prompts."""
    config, model_name, _, dirs = _cli_context(config_path, model_name, output_dir)
    exp_dir = dirs["experiments"] / "activation_patching"
    (
        clean_prompt,
        corrupted_prompt,
        target_token,
        activation_type,
        corruption_method,
        subject,
        noise_std,
    ) = _activation_patching_params(config, clean_prompt, corrupted_prompt)

    model = load_model(model_name)
    result = run_activation_patching_experiment(
        model,
        clean_prompt,
        corrupted_prompt,
        exp_dir,
        target_token=target_token,
        activation_type=activation_type,
        corruption_method=corruption_method,
        subject=subject,
        noise_std=noise_std,
        config=config,
    )

    if "top_layers" in result:
        table = Table(title="Top Patching Layers (resid_pre)")
        table.add_column("Layer", style="cyan")
        table.add_column("Effect", style="green")
        for layer, effect in result["top_layers"][:10]:
            table.add_row(str(layer), f"{effect:.4f}")
    else:
        table = Table(title="Top Patching Heads")
        table.add_column("Layer", style="cyan")
        table.add_column("Head", style="cyan")
        table.add_column("Score", style="green")
        for t in result["top_heads"][:10]:
            if len(t) == 4:
                table.add_row(str(t[0]), str(t[2]), f"{t[3]:.4f}")
            else:
                table.add_row(str(t[0]), str(t[1]), f"{t[2]:.4f}")
    console.print(table)
    console.print(f"[green]Results saved to {exp_dir}[/green]")


@app.command()
def logit_lens(
    config_path: Path = typer.Option(
        None, "--config", "-c", path_type=Path, help="Path to YAML config"
    ),
    output_dir: Path = typer.Option(
        None, "--output", "-o", path_type=Path, help="Output directory"
    ),
    model_name: str = typer.Option(None, "--model", "-m", help="Model name"),
    prompt: str = typer.Option("The capital of France is", "--prompt", "-p", help="Prompt"),
) -> None:
    """Run logit lens analysis across layers."""
    config, model_name, _, dirs = _cli_context(config_path, model_name, output_dir)
    prompt = prompt or config.get("prompts", {}).get("logit_lens", prompt)
    exp_dir = dirs["experiments"] / "logit_lens"

    model = load_model(model_name)
    result = run_logit_lens_experiment(
        model,
        prompt,
        exp_dir,
        config=config,
    )

    table = Table(title="Logit Lens - Layer Predictions")
    table.add_column("Layer", style="cyan")
    table.add_column("Top Token", style="green")
    table.add_column("Target Prob", style="yellow")
    for lp in result["layer_predictions"][:10]:
        table.add_row(
            str(lp["layer"]),
            lp["top_tokens"][0] if lp["top_tokens"] else "-",
            f"{lp['target_prob']:.4f}",
        )
    console.print(table)
    console.print(f"[green]First correct at layer: {result['first_correct_layer']}[/green]")
    console.print(f"[green]Results saved to {exp_dir}[/green]")


@app.command()
def neuron_analysis(
    config_path: Path = typer.Option(
        None, "--config", "-c", path_type=Path, help="Path to YAML config"
    ),
    output_dir: Path = typer.Option(
        None, "--output", "-o", path_type=Path, help="Output directory"
    ),
    model_name: str = typer.Option(None, "--model", "-m", help="Model name"),
    prompt: str = typer.Option(
        "Hello world.", "--prompt", "-p", help="Prompt(s) - comma-separated for multiple"
    ),
) -> None:
    """Analyze neuron activations and plot distributions."""
    config, model_name, _, dirs = _cli_context(config_path, model_name, output_dir)
    prompts = [p.strip() for p in prompt.split(",")] if "," in prompt else [prompt]
    exp_dir = dirs["experiments"] / "neuron_analysis"
    neuron_plot_dir = dirs["neurons"]

    model = load_model(model_name)
    result = run_neuron_analysis_experiment(
        model,
        prompts,
        exp_dir,
        config=config,
    )

    for layer_key, acts in list(result["activations"].items())[:3]:
        layer_num = int(layer_key.replace("L", ""))
        out_path = neuron_plot_dir / f"neuron_dist_L{layer_num}.png"
        plot_neuron_activation_distribution(
            acts,
            layer_num,
            output_path=out_path,
        )
        console.print(f"[green]Saved neuron plot to {out_path}[/green]")

    console.print(f"[green]Results saved to {exp_dir}[/green]")


if __name__ == "__main__":
    app()
