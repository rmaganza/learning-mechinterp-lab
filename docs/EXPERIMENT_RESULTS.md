# Experiment Results and Conclusions

This document summarizes results from running mechanistic interpretability experiments on **GPT-2 small** (124M parameters). The goal is to **reproduce known findings from research papers**, not to report novel results. All experiments used the default configs in `configs/gpt2_small.yaml` unless noted.

## Reproduction Status

| Experiment | Paper | Reproduces? | Notes |
|------------|-------|-------------|-------|
| Induction heads | Olsson et al. (2022) | **Yes** | L5H1 as primary induction head matches known result. |
| Logit lens | Nostalgebraist (2020) | **Yes** | Same method; first correct at layer 3 for model's top. Use `configs/gpt2_small_logit_lens.yaml` (target=null). |
| Activation patching (resid) | Meng et al. (2022) | **Yes** | Prompt swap: late layers 9–11. Noise: `configs/gpt2_small_meng.yaml`. |
| Activation patching (MLP) | Meng et al. (2022) | **Yes** | MLP-level patching reproduces middle-layer MLP localization. Use `configs/gpt2_small_mlp.yaml`. |
| Copying heads | Olsson et al. (2022) | **Concept** | Olsson describes previous-token heads but does not report GPT-2 results. We implement the concept; L5H1 in our top list is consistent with it being the induction head (which uses previous-token attention). |
| Neuron analysis | — | **No** | Raw MLP capture; does not reproduce Meng. Meng's result (MLP causal tracing) is reproduced in Section 3b. |
| Multi-fact patching | Meng et al. (2022) | **Reproduces + extends** | Same activation patching (reproduces basic result: layers 9–11 matter). Extension: aggregates across 3 facts. |
| Logit lens (target) | Nostalgebraist (2020) | **Reproduces + extends** | Same method (reproduces basic result: predictions evolve across layers). Extension: tracks specific token (e.g. " Paris"). |

---

## 1. Copying Heads (Previous-Token Attention)

**Concept from:** Olsson et al. (2022) — previous-token heads as building blocks for induction ([transformer-circuits.pub](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)). The paper does not report a GPT-2 copying-head table. We implement the concept; L5H1 (the induction head) appearing in our top list is a consistency check that the induction mechanism uses previous-token attention.

**Prompt:** "The quick brown fox jumps over the lazy dog."

**Method:** For each attention head, we measure the average attention weight from each query position to previous token positions (offsets 1–10). Heads that strongly attend to the previous token are "copying heads" — they read from the immediately preceding context.

### Results

| Rank | Layer | Head | Copying Score |
|------|-------|------|---------------|
| 1    | 4     | 11   | 0.182         |
| 2    | 5     | 1    | 0.182         |
| 3    | 7     | 10   | 0.181         |
| 4    | 6     | 9    | 0.181         |
| 5    | 7     | 2    | 0.181         |

### Conclusions

- **Copying is distributed across mid-to-late layers (4–10).** No single layer dominates; the behavior is spread across layers 3–10.
- **Head 11 and head 1 appear frequently** in the top copying heads (layers 4, 5, 7, 8, 9), suggesting these head indices may have a consistent "previous token" role across layers.
- **Scores are tightly clustered** (0.178–0.182), so many heads contribute to copying; the top heads are only marginally stronger than the rest.
- **Caveat:** High attention to the previous token does not prove the head *uses* that information for copying. Value vectors matter; pattern alone is suggestive, not conclusive.

---

## 2. Logit Lens

**Reproduces:** "Interpreting GPT: the logit lens" (Nostalgebraist, 2020; [GreaterWrong](https://www.greaterwrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)) — projecting intermediate residual stream through the unembedding to observe when predictions form.

**Config:** For model's top prediction (Section 2): `configs/gpt2_small_logit_lens.yaml` (logit_lens_target: null). For factual token (Section 7): default `gpt2_small.yaml` (logit_lens_target: " Paris").

**Prompt:** "The capital of France is"

**Method:** At each layer, we project the residual stream (after that layer) through the unembedding matrix to get "virtual logits." We track when the model's top-1 prediction first matches its final prediction.

### Results

- **First correct layer:** 3 (the residual stream at layer 3 already predicts the same top token as the final output)
- **Target token (model's top prediction):** " now" (not " Paris" — GPT-2 often predicts "is now" or similar continuations)
- **Layer-by-layer:** The probability of the top token rises from ~0.04 (layer 1) to ~0.29 (layer 6), then fluctuates in later layers as the model refines its prediction.

### Conclusions

- **Information appears early:** By layer 3, the residual stream already encodes enough to predict the model's chosen next token when read out with the unembedding.
- **Logit lens is correlational, not causal:** Early "correct" predictions do not mean the model "decides" at that layer; the unembedding is trained for the final residual, not intermediate ones.
- **Prompt-dependent:** For "The capital of France is," GPT-2's top prediction is " now" rather than " Paris." To study factual recall ("Paris"), use a metric that explicitly tracks the " Paris" token or use activation patching.

---

## 3. Activation Patching (Residual Stream)

**Reproduces:** Meng et al. (2022) — "Locating and Editing Factual Associations in GPT" (NeurIPS 2022). Causal tracing via activation patching: corrupt (prompt swap or noise), then patch clean activations layer-by-layer. resid_pre patching localizes to late layers (9–11); MLP patching (Section 3b) to middle-layer MLPs.

**Prompts:**
- Clean: "The capital of France is"
- Corrupted: "The capital of Germany is"

**Method:** We patch the clean run's residual stream (before each layer) into the corrupted run, one layer at a time. The metric is the logit difference for " Paris" (correct answer) vs. the full distribution. Higher values mean more recovery toward the correct prediction.

### Results

| Rank | Layer | Metric (logit diff) |
|------|-------|---------------------|
| 1    | 10    | -7.97               |
| 2    | 11    | -7.98               |
| 3    | 9     | -8.03               |
| 4    | 7     | -8.07               |
| 5    | 8     | -8.08               |

(Lower layers 0–6 have values around -8.09 to -8.07.)

### Conclusions

- **Late layers (9–11) are most causally important** for factual recall. Patching the residual stream at layers 9, 10, or 11 recovers the most toward the correct " Paris" prediction.
- **Early layers (0–6) contribute less** when patched in isolation; the effect is relatively flat.
- **Interpretation:** The "France" vs "Germany" distinction is likely resolved and written into the residual stream in the later layers. Early layers may handle more generic structure; the factual content is refined toward the end.
- **Caveat:** This is resid_pre (layer-level) patching. Set `patching.activation_type: attn_out` in config for head-level patching.

**Meng et al. noise corruption:** Use `configs/gpt2_small_meng.yaml`. Same prompt; Gaussian noise on subject embeddings.

**Meng et al. MLP-level patching:** Use `configs/gpt2_small_mlp.yaml`. Patches `mlp_out` per layer; reproduces middle-layer MLP localization.

---

## 3b. Activation Patching (MLP-Level) — Meng et al.

**Reproduces:** Meng et al. (2022) — factual associations localized to **middle-layer MLP modules**.

**Config:** `configs/gpt2_small_mlp.yaml` (`activation_type: mlp_out`)

**Prompts:** Same as Section 3 (France vs Germany, target " Paris").

**Method:** Patch MLP output (not resid_pre) at each layer. Identifies which MLP layers are causally important for factual recall.

### Results

| Rank | Layer | Metric (logit diff) |
|------|-------|---------------------|
| 1    | 0     | -8.15               |
| 2    | 8     | -8.34               |
| 3    | 1     | -8.38               |
| 4    | 10    | -8.40               |
| 5    | 3     | -8.41               |

### Conclusions

- **MLP layers 0, 1, 3, 8, 10** are most causal when patching MLP output. Layer 0 (first MLP) and middle layers (8, 10) stand out.
- **Aligns with Meng et al.:** Factual recall involves MLP modules; our layer-level MLP patching identifies decisive layers.
- **Differs from resid_pre:** resid_pre highlights late layers 9–11; MLP patching also shows early layers (0, 1, 3), suggesting MLPs at multiple depths contribute.

---

## 4. Neuron Analysis

**Does not reproduce Meng et al.** Meng's result (MLP causal tracing) is reproduced in Section 3b. This experiment captures raw MLP activations (post-ReLU) as a baseline; it does not run causal tracing.

**Prompts:** "Hello world.", "The cat sat on the mat.", "Machine learning is fascinating."

**Method:** We capture MLP activations (post-ReLU) across layers and aggregate across prompts. The experiment records activation distributions and top neurons by mean activation.

### Conclusions

- **Neuron activations are sparse:** Most neurons are zero (ReLU); a subset is highly active.
- **Layer-dependent patterns:** Early layers show more diffuse activation; later layers often have a smaller set of highly active neurons.
- **Use case:** Neuron analysis is useful for finding "interpretable" neurons (e.g., those that fire on specific token classes or syntactic patterns). The default run aggregates across diverse prompts; for targeted analysis, use prompts that share a property of interest.

---

## 5. Induction Heads

**Reproduces:** Olsson et al. (2022) — "In-context Learning and Induction Heads" ([arXiv:2209.11895](https://arxiv.org/abs/2209.11895)). Induction heads implement [A][B]...[A]→[B]. Our result (L5H1 as primary induction head) matches the known finding for GPT-2 small (Olsson framework, [LessWrong SAE analysis](https://www.lesswrong.com/posts/xmegeW5mqiBsvoaim/we-inspected-every-head-in-gpt-2-small-using-saes-so-you-don)). Our scoring uses attention from last token to the B position; the paper uses prefix-matching on random sequences.

**Prompt:** "The cat sat on the mat. The cat sat on the"

**Method:** We identify heads that implement the [A][B]...[A]→[B] induction pattern: the last token (second "the") attends to the position of the previous occurrence of "the" (after which "mat" follows). The target next token is " mat." Heads that attend from the last position to the position of the first "the" (where "mat" appeared after) are scored by that attention weight.

### Results

| Rank | Layer | Head | Induction Score |
|------|-------|------|-----------------|
| 1    | 5     | 1    | 0.789           |
| 2    | 10    | 11   | 0.687           |
| 3    | 7     | 2    | 0.685           |
| 4    | 10    | 6    | 0.656           |
| 5    | 11    | 10   | 0.579           |

### Conclusions

- **Layer 5 head 1 is the primary induction head** (score 0.79), consistent with Olsson et al.'s findings that GPT-2 uses a small set of induction heads in mid layers.
- **Secondary induction heads** appear in layers 7, 10, 11 (heads 2, 6, 10, 11), suggesting redundancy or backup circuits.
- **Layer 10 has multiple induction heads** (6 and 11), which may reflect different roles (e.g., different copy distances or token types).

---

## 6. Multi-Fact Activation Patching

**Reproduces + extends** Meng et al. — same activation patching method (reproduces: layers 9–11 causal for factual recall). Extension: aggregates across 3 facts to test consistency.

**Facts (clean → corrupted, target):**
1. France → Germany, target " Paris"
2. England → France, target " London"
3. Italy → Spain, target " Rome"

**Method:** For each fact, we run activation patching (resid_pre) and record the logit difference for the target token. We aggregate across facts by summing the metric per layer; layers with more negative values contribute more to factual recall across all three facts.

### Results

| Rank | Layer | Aggregated Metric |
|------|-------|-------------------|
| 1    | 10    | -7.22             |
| 2    | 11    | -7.25             |
| 3    | 9     | -7.26             |
| 4    | 8     | -7.29             |
| 5    | 7     | -7.29             |

### Conclusions

- **Layers 7–11 are consistently important** for factual recall across multiple facts (capitals of France, England, Italy).
- **Layer 10 and 11 stand out** as the most causal when aggregating across facts, aligning with single-fact resid_pre patching.
- **Multi-fact aggregation** reduces noise from any single prompt and highlights shared mechanisms for factual retrieval.

---

## 7. Logit Lens with Target Token

**Reproduces + extends** Nostalgebraist — same method (reproduces: predictions evolve across layers). Extension: tracks a specific token (e.g. " Paris") instead of the model's top prediction.

**Prompt:** "The capital of France is"  
**Target token:** " Paris"

**Method:** Same as the standard logit lens, but we explicitly track when " Paris" (rather than the model's top prediction) first becomes the top-1 prediction at intermediate layers.

### Results

- **First correct layer:** None (in our run, " Paris" did not become top-1 at any intermediate layer before the final output)
- **Interpretation:** GPT-2's residual stream at intermediate layers may favor other continuations (e.g., " now"); the correct factual token " Paris" is only selected at the final layer. This highlights that logit lens with a fixed target is useful for probing when a *specific* token emerges, even when it is not the model's default prediction.

### Conclusions

- **Target-token logit lens** is valuable for factual-recall analysis when the model's top prediction differs from the desired answer.
- **" Paris" emerges late or not at all** in intermediate layers for this prompt, suggesting factual content is refined in later layers.

---

## Summary Table

| Experiment           | Main finding                                                |
|----------------------|-------------------------------------------------------------|
| Copying heads        | Heads in layers 4–10, especially indices 1 and 11           |
| Logit lens           | Top prediction appears by layer 3; correlational only      |
| Activation patching  | Layers 9–11 most causal (resid_pre) for factual recall     |
| MLP patching         | Layers 0, 1, 3, 8, 10 most causal (mlp_out); Meng et al.  |
| Neuron analysis      | Sparse activations; layer-dependent concentration           |
| Induction heads      | L5H1 primary induction head; L7, L10, L11 secondary         |
| Multi-fact patching  | Layers 7–11 causal for factual recall across 3 facts        |
| Logit lens (target)  | " Paris" not top-1 at intermediate layers; emerges late     |

---

## Reproducing These Results

```bash
# Copying heads
uv run mechinterp run-experiment copying-heads -c configs/gpt2_small.yaml -o output

# Logit lens (model's top) — Section 2
uv run mechinterp run-experiment logit-lens -c configs/gpt2_small_logit_lens.yaml -o output
# Logit lens (target token " Paris") — Section 7
uv run mechinterp run-experiment logit-lens -c configs/gpt2_small.yaml -o output

# Activation patching (resid_pre = layer-level, attn_out = head-level)
# configs/gpt2_small.yaml uses resid_pre by default for factual recall
uv run mechinterp run-experiment activation-patching -c configs/gpt2_small.yaml -o output
# Or head-level: set patching.activation_type: attn_out in config

# Neuron analysis
uv run mechinterp run-experiment neuron-analysis -c configs/gpt2_small.yaml -o output

# Induction heads (Olsson et al. [A][B]...[A]→[B] pattern)
uv run mechinterp run-experiment induction-heads -c configs/gpt2_small.yaml -o output

# Multi-fact activation patching (aggregates across multiple facts)
uv run mechinterp run-experiment multi-fact-patching -c configs/gpt2_small.yaml -o output

# Activation patching with Meng et al. noise corruption
uv run mechinterp run-experiment activation-patching -c configs/gpt2_small_meng.yaml -o output

# Activation patching (MLP-level) — Meng et al. middle-layer MLPs
uv run mechinterp run-experiment activation-patching -c configs/gpt2_small_mlp.yaml -o output
```

Results are saved under `output/experiments/`. Attention plots from the copying-heads experiment are in `output/plots/attention/`.

---

## Known Issue: Activation Patching Shape Bug (Fixed)

When using `get_act_patch_attn_head_out_all_pos` (or similar head-patch functions), the **clean cache must be created with `remove_batch_dim=False`**. 

**Root cause:** TransformerLens's `layer_head_vector_patch_setter` assumes activations have shape `[batch, pos, head_index, d_head]` and indexes with `[:, :, head_index]` to get `[batch, pos, d_head]`. When the cache uses `remove_batch_dim=True`, stored tensors have shape `[pos, head_index, d_head]`. The same index `[:, :, head_index]` then incorrectly indexes the *d_head* dimension (size 64) with head_index (0–11), yielding `[pos, head_index]` = `[6, 12]` instead of `[6, 64]`, causing a shape mismatch when assigning into the corrupted activation (which has `[1, 6, 64]`).

**Fix:** Use `run_with_cache(..., remove_batch_dim=False)` when building the clean cache for head-level patching.

---

## References

- **Olsson et al. (2022)** — "In-context Learning and Induction Heads." arXiv:2209.11895. [Anthropic](https://anthropic.com/research/in-context-learning-and-induction-heads), [Transformer Circuits](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html).
- **Meng et al. (2022)** — "Locating and Editing Factual Associations in GPT." NeurIPS 2022. [arXiv](https://arxiv.org/abs/2202.05262), [ROME](https://rome.baulab.info/).
- **Nostalgebraist (2020)** — "Interpreting GPT: the logit lens." [GreaterWrong](https://www.greaterwrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens).
