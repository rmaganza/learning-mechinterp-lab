# Reproducibility Guide

How to reproduce experiments in this lab and use configuration for consistent runs.

## Environment

Use `uv` for reproducible installs:

```bash
uv sync
# or with dev deps
uv sync --extra dev
```

Lock file (`uv.lock`) pins exact versions. Commit it for reproducibility.

## Configuration

Experiments can use YAML config files for parameters. Example `configs/activation_patch.yaml`:

```yaml
model:
  name: gpt2-small
  device: null  # auto-detect

experiment:
  clean_prompt: "The capital of France is Paris"
  corrupt_prompt: "The capital of France is London"
  hook_names:
    - blocks.0.attn.hook_z
    - blocks.1.attn.hook_z
  metric: logit_diff

seed: 42
```

### Loading Config

```python
import yaml

with open("configs/activation_patch.yaml") as f:
    config = yaml.safe_load(f)

model = load_model(config["model"]["name"], device=config["model"].get("device"))
```

## Reproducing Each Experiment Type

### 1. Model Loading

```bash
# Same model, same device
uv run python -c "
from mechinterp_lab.models import load_model
model = load_model('gpt2-small')
print(model.cfg.model_name)
"
```

**Reproducibility:** Model name and `transformers`/`transformer_lens` versions determine weights. Use fixed versions in `uv.lock`.

---

### 2. Hook Capture

```bash
uv run python -c "
from mechinterp_lab.models import load_model
from mechinterp_lab.hooks import capture_activations

model = load_model('gpt2-small')
acts = capture_activations(model, 'Hello world', layers=[0, 1])
for k, v in acts.items():
    print(k, v.shape)
"
```

**Reproducibility:** Same model, same prompt, same `layers`/`names` → same activations. No randomness.

---

### 3. Activation Patching

```bash
# Via CLI (uses config)
uv run mechinterp run-experiment activation-patching -c configs/gpt2_small.yaml -o output

# Or patch-activations for head-level or resid_pre (set patching.activation_type in config)
uv run mechinterp patch-activations -c configs/gpt2_small.yaml
```

Config: `patching.activation_type` = `resid_pre` (layer-level) or `attn_out` (head-level). For factual recall, set `prompts.activation_patching.target_token` (e.g. `" Paris"`).

**Reproducibility:** Deterministic for same model and prompts. Document exact prompt strings and activation type.

---

### 4. Probe / Logit Lens

```bash
uv run python -c "
from mechinterp_lab.models import load_model
from mechinterp_lab.hooks import capture_activations
from mechinterp_lab.probes import LogitProbe

model = load_model('gpt2-small')
acts = capture_activations(model, 'The capital of France is', names=['blocks.0.hook_resid_post'])
probe = LogitProbe(model)
for name, res in acts.items():
    top = probe.top_k_tokens(res, k=3)
    print(name, top)
"
```

**Reproducibility:** Deterministic. No training in basic logit lens.

---

## Seeding

For any stochastic parts (e.g., dropout, sampling), set seeds:

```python
import torch
import numpy as np
import random

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
```

---

## Reporting Checklist

When documenting an experiment, include:

1. **Environment:** Python version, `uv.lock` or `pip freeze`
2. **Model:** Exact name (e.g., `gpt2-small`), source
3. **Prompts:** Full clean/corrupt strings, tokenization details if relevant
4. **Hooks:** Exact hook names
5. **Metric:** Definition (e.g., logit diff between which tokens)
6. **Hardware:** CPU vs GPU, any precision settings
7. **Seeds:** If applicable

---

## Config Schema (Reference)

| Key           | Description                    | Example                    |
|---------------|--------------------------------|----------------------------|
| `model.name`  | TransformerLens model name    | `gpt2-small`               |
| `model.device`| Device string or null          | `cuda`, `cpu`, `null`      |
| `experiment.*`| Experiment-specific params    | prompts, hooks, metric     |
| `seed`        | Random seed                    | `42`                       |

Extend the schema as you add new experiment types.
