# mechinterp-lab

**A learning resource for mechanistic interpretability** — hands-on experiments on small transformers for personal study and exploration. Built on [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) for clean access to model internals.

> This is a **personal learning project**, not production software or published research. Use it to understand how interpretability techniques work and to build intuition on models like GPT-2.

## What's Inside

Utilities and experiments for learning mechanistic interpretability:

- **Model loading** — GPT-2 style models via TransformerLens
- **Hook capture** — Intercept and record activations at any layer
- **Activation patching** — Causal intervention to identify important components
- **Probe utilities** — Logit lens, linear probes, attribution analysis

See [docs/METHODS.md](docs/METHODS.md) for step-by-step explanations of each technique and what conclusions are valid vs invalid.

## Installation

Uses [uv](https://docs.astral.sh/uv/) for fast, reproducible dependency management:

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install
uv sync

# With dev dependencies (pytest, etc.)
uv sync --extra dev
```

## Quick Start

```python
from mechinterp_lab.models import load_model
from mechinterp_lab.hooks import capture_activations
from mechinterp_lab.patching import patch_residual_stream

# Load a small model (e.g., GPT-2)
model = load_model("gpt2")

# Run with hooks to capture activations
tokens = model.model.to_tokens("The capital of France is")
logits, cache = capture_activations(model, tokens)

# Activation patching: identify which components matter
clean = "The capital of France is Paris"
corrupt = "The capital of France is London"
clean_tokens = model.model.to_tokens(clean)
corrupt_tokens = model.model.to_tokens(corrupt)
_, clean_cache = model.model.run_with_cache(clean_tokens)

def metric_fn(logits):
    return logits[0, -1, clean_tokens[0, -1].item()] - logits[0, -1, corrupt_tokens[0, -1].item()]

patch_effects = patch_residual_stream(model, corrupt_tokens, clean_cache, metric_fn)
```

See [notebooks/workflow_demo.ipynb](notebooks/workflow_demo.ipynb) for a complete walkthrough with explanations.

## Project Structure

```
mechinterp-lab/
├── src/mechinterp_lab/    # Core library
│   ├── models/            # Model loading (GPT-2, Pythia, TinyStories)
│   ├── hooks/             # Hook capture utilities
│   ├── patching/          # Activation patching, causal trace
│   ├── probes/            # Logit lens, tuned lens
│   ├── analysis/          # Attention, MLP, feature analysis
│   └── experiments/       # Experiment runners
├── docs/                  # Documentation
│   ├── METHODS.md         # How each method works (learning reference)
│   ├── PITFALLS.md        # Common pitfalls
│   ├── EXPERIMENT_RESULTS.md # Results from running experiments
│   └── REPRODUCIBILITY.md # Config usage, experiment reproduction
├── notebooks/             # Example notebooks
└── tests/                 # Pytest test suite
```

## Documentation

- **[METHODS.md](docs/METHODS.md)** — How each method works, valid vs invalid conclusions
- **[PITFALLS.md](docs/PITFALLS.md)** — Common pitfalls in mechanistic interpretability
- **[EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md)** — Results from running experiments on GPT-2 small
- **[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)** — Config usage, how to reproduce experiments

## Running Tests

```bash
uv run pytest tests/ -v
```

To skip model-dependent tests (e.g. in CI):

```bash
MECHINTERP_SKIP_MODEL_TESTS=1 uv run pytest tests/ -v
```

## CLI

```bash
# Run experiments
uv run mechinterp run-experiment logit-lens -c configs/gpt2_small.yaml -o output
uv run mechinterp run-experiment copying-heads -c configs/default.yaml
uv run mechinterp run-experiment activation-patching -c configs/gpt2_small.yaml

# Individual analyses
uv run mechinterp analyze-attention -p "The quick brown fox" -l 0 -H 0
uv run mechinterp logit-lens -p "The capital of France is"
uv run mechinterp neuron-analysis -p "Hello world."
```
