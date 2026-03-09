# Methods in Mechanistic Interpretability

This document explains **how each method works** (step-by-step), what it measures, and what conclusions are **valid** vs **invalid**. Intended as a learning reference.

Methods 1–9 are implemented in this lab. Methods 10–15 are **advanced/recent** (2023–2026) and included for learning—not implemented here.

---

## 1. Activation Patching (Causal Tracing)

**What it does:** Identifies which model components are *causally* important for a behavior by running on corrupted input, then "patching in" activations from a clean run one at a time.

### How it works

1. **Two runs:** Run the model on (a) **clean** input (e.g. "The capital of France is") and (b) **corrupted** input (e.g. "The capital of Germany is"). Cache all activations from the clean run.
2. **Corruption:** The corrupted run produces the wrong answer (e.g. predicts " Berlin" instead of " Paris"). Corruption can be:
  - **Prompt swap:** Different tokens (France vs Germany); clean and corrupted have same length.
  - **Noise (Meng et al.):** Same prompt; corrupt subject token embedding: $\mathbf{x}_s \gets \mathbf{x}_s + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$, so the model gets confused.
3. **Patching loop:** For each component (layer, head, or MLP):
  - Run the model on the **corrupted** input.
  - At the target component, **replace** the corrupted activation with the corresponding activation from the clean cache.
  - Let the rest of the model run normally.
  - Measure the output (e.g. logit of correct token).
4. **Interpretation:** If patching at layer $L$ significantly improves the metric (e.g. logit of " Paris" goes up), that layer is causally important. Compare across layers to localize where the computation happens.

### Activation types

- **resid_pre:** Patch the residual stream *before* each block. Identifies which layers matter (coarse).
- **attn_out:** Patch each attention head's output. Identifies which heads matter (finer).
- **mlp_out:** Patch each MLP's output. Identifies which MLP layers matter (Meng et al. style).

### Valid conclusions

- The patched component is **causally involved** (patching changes the output).
- Relative importance of components (compare patch effects).
- Localization of computation (which layers/heads/MLPs matter).

### Invalid conclusions

- That the component *alone* causes the behavior (others may also matter).
- That the component *represents* the information (it may merely transmit it).
- Generalization to different prompts without further evidence.

**Caveats:** Clean and corrupted should differ minimally. Same token length for prompt swap. Metric choice matters (e.g. logit diff for target token).

---

## 2. Logit Lens

**What it does:** At each layer, projects the residual stream through the unembedding to get "virtual logits"—what the model would predict if it stopped at that layer.

### How it works

1. **Forward pass:** Run the model on a prompt and cache activations.
2. **Per-layer projection:** For each layer $L$ (0 to $n_{\text{layers}}$):
  - Take the residual stream at the **last position** after layer $L$ (or after embedding for layer 0).
  - Apply final layer norm if the model has it.
  - Multiply by the unembedding matrix: $\mathbf{l}_L = \mathbf{r}_L \mathbf{W}_U$, where $\mathbf{r}_L \in \mathbb{R}^{d_{\text{model}}}$ is the residual at layer $L$ and $\mathbf{W}_U \in \mathbb{R}^{d_{\text{model}} \times \lvert\mathcal{V}\rvert}$.
  - This gives a vector of size $\lvert\mathcal{V}\rvert$ (vocab)—"logits" as if we stopped there.
3. **Readout:** Convert to probabilities via $\text{softmax}(\mathbf{l}_L)$, get top-$k$ predictions. Track when the target token (model's top or a specific token like " Paris") first becomes the top prediction.
4. **Interpretation:** "First correct layer" = earliest layer where the residual, when read out with $\mathbf{W}_U$, predicts the same token as the final output. Shows when information *appears* in the residual stream.

### Why it's correlational

The unembedding $\mathbf{W}_U$ is trained to read the **final** residual. Intermediate residuals were never optimized for this readout. So early "correct" predictions may be coincidental—the model doesn't "decide" there; we're just probing with a mismatched readout.

### Valid conclusions

- When information *first appears* in the residual (exploratory).
- Relative "readiness" of the representation at each layer.
- Signal for where to look next (e.g. for activation patching).

### Invalid conclusions

- That the model "actually" predicts at that layer.
- Causal importance (logit lens is correlational, not interventional).

**Caveats:** Use tuned lens for more faithful early readouts. Target-token variant tracks a specific token (e.g. " Paris") instead of the model's top—useful when the model's top differs from the desired answer.

---

## 3. Copying Heads (Previous-Token Attention)

**What it does:** Identifies attention heads that strongly attend to the *previous* token—a building block for induction and in-context learning (Olsson et al.).

### How it works

1. **Forward pass:** Run the model on a prompt, cache attention patterns for each layer/head.
2. **Attention pattern:** For each head, we have a matrix $\text{pattern}(q, k)$ = attention from query position $q$ to key position $k$.
3. **Copying score:** For each query position $q$ (except 0), sum attention to *previous* positions: $\text{score} = \frac{1}{T-1} \sum_{q=1}^{T-1} \sum_{k \leq q-1} \text{pattern}(q, k)$ (or similar). Heads with high scores attend strongly to the token immediately before the current one.
4. **Interpretation:** "Copying heads" or "previous-token heads" read from the immediately preceding context. They are components of induction heads (which need to know "what came before" to complete $A\ B\ \ldots\ A \to B$).

### Valid conclusions

- Which heads attend strongly to the previous token.
- Hypotheses about head roles (e.g. L5H1 in top list is consistent with it being the induction head, which uses previous-token info).

### Invalid conclusions

- That high attention = the head *uses* that info for copying (value vectors matter; pattern is suggestive only).
- That these heads alone implement copying (need output analysis).

**Caveats:** Attention pattern shows *where* the head looks, not *what* it copies. Value vectors carry the content.

---

## 4. Induction Heads

**What it does:** Identifies attention heads that, when the model sees a token for the *second* time in context, look back to what followed that token the *first* time and use that to predict the next token. Example: in "The cat sat on the mat. The cat sat on the", the last "the" should predict "mat"—an induction head attends from that "the" back to the earlier "mat" to complete the pattern. Formally: $A\ B\ \ldots\ A \to B$.

### How it works

1. **Prompt structure:** Use a prompt with a repeated pattern, e.g. "The cat sat on the mat. The cat sat on the". The last token is "the" ($A$); the model should predict "mat" ($B$). The first "mat" appeared after the first "the".
2. **Induction target:** Find the position of $B$ (the token that followed the previous $A$). In the example, "mat" is at position $p_B$.
3. **Score each head:** For each layer/head, look at the attention pattern. The **induction score** = $\text{pattern}(T-1, p_B)$, i.e. attention from the last query position (where we're predicting) to the induction target position (where $B$ is). High score = the head attends from "the" to "mat", i.e. it's implementing the induction pattern.
4. **Interpretation:** Heads with high scores are induction heads—they look back to find "what followed the previous occurrence of the current token" and promote that as the next prediction.

### Valid conclusions

- Which heads implement the $A\ B\ \ldots\ A \to B$ pattern.
- L5H1 in GPT-2 small as primary induction head (known result).

### Invalid conclusions

- That induction heads *alone* explain in-context learning (Olsson presents correlational evidence for large models).
- That the score captures the full mechanism (value vectors and composition matter).

**Caveats:** Our scoring uses attention to the B position; Olsson uses prefix-matching on random sequences. Both target the same mechanism.

---

## 5. Neuron Analysis (MLP Activations)

**What it does:** Captures raw MLP (feed-forward) activations post-ReLU across layers and prompts. Baseline for understanding MLP activity; does not run causal tracing.

### How it works

1. **Forward pass:** Run the model on one or more prompts, cache activations.
2. **MLP hook:** For each layer $L$, read `blocks.{L}.mlp.hook_post` (activations after ReLU) or `hook_pre` (before ReLU). Shape: $\mathbb{R}^{T \times d_{\text{mlp}}}$ where $T$ = sequence length.
3. **Aggregation:** Concatenate across prompts and positions. Record activation distributions, sparsity, top neurons by mean activation.
4. **Interpretation:** MLP activations are sparse ($\max(0, \cdot)$ zeros out many neurons). A subset is highly active. Layer-dependent patterns (early vs late) can suggest where computation concentrates. For causal MLP analysis, use activation patching with `activation_type: mlp_out` (Section 1).

### Valid conclusions

- Activation sparsity and layer-dependent patterns.
- Which neurons are most active (exploratory).

### Invalid conclusions

- Causal importance (this is descriptive, not interventional).
- That active neurons are "important" (need patching to test).

**Caveats:** Raw capture only. For Meng et al.–style MLP causal tracing, use activation patching with mlp_out.

---

## 6. Multi-Fact Activation Patching

**What it does:** Runs activation patching on multiple (clean, corrupted, target) fact tuples and aggregates layer importance across facts.

### How it works

1. **Fact tuples:** Each fact is $(\text{clean}, \text{corrupted}, \text{target})$, e.g. ("The capital of France is", "The capital of Germany is", " Paris").
2. **Per-fact patching:** For each fact $f$, run activation patching (resid_pre) as in Section 1. Get layer effects $\Delta_f(L)$ (metric per layer).
3. **Aggregation:** Average the layer effects: $\bar{\Delta}(L) = \frac{1}{N} \sum_f \Delta_f(L)$. Layers with consistently high $\bar{\Delta}(L)$ are robustly important for factual recall.
4. **Interpretation:** Reduces noise from any single prompt. Highlights shared mechanisms (e.g. layers 9–11 for capital facts).

### Valid conclusions

- Layers that are consistently causal across multiple facts.
- More robust localization than single-fact patching.

**Caveats:** Same as activation patching. Facts should be comparable (same structure, same metric).

---

## 7. Attention Analysis (General)

**What it does:** Inspects attention patterns (which positions attend to which) and sometimes head outputs. Includes visualization, head norms, QK analysis.

### How it works

- **Attention pattern:** $\text{pattern}(q, k) = \text{softmax}\left(\frac{\mathbf{Q}_q \mathbf{K}_k^\top}{\sqrt{d_k}}\right)$, i.e. how much position $q$ attends to position $k$ (scaled dot-product attention).
- **Visualization:** Heatmap of pattern for a layer/head.
- **Analysis:** Identify heads that attend to specific positions (e.g. previous token, [CLS], etc.).

### Valid conclusions

- Which positions a head reads from.
- Hypotheses about head roles.

### Invalid conclusions

- That high attention = high causal importance (value content matters).
- That pattern alone determines head purpose.

**Caveats:** Attention shows *where*; value vectors show *what*.

---

## 8. Direct Logit Attribution (DLA)

**What it does:** Decomposes final logits into contributions from each component using the unembedding as a linear readout.

### How it works

- For each component $\mathbf{c}$ (e.g. each layer's residual), compute its contribution to the logit of token $t$: $\text{contrib}_t = \mathbf{c}^\top \mathbf{W}_U[:, t]$.
- Sum of contributions (with baseline) equals final logit: $\ell_t = \sum_i \text{contrib}_i + \text{baseline}$.
- Correlational: large contribution $\neq$ causal importance.

### Valid conclusions

- Which components contribute to the final logit (decomposition).
- Relative magnitude of contributions.

### Invalid conclusions

- Causal importance (contribution $\neq$ causation).
- That negative contribution means "against" (interpretation is subtle).

**Caveats:** Use activation patching for causal claims.

---

## 9. Linear Probes

**What it does:** Trains a linear classifier on activations to predict a property. Evaluates on held-out data.

### How it works

- Extract activations $\mathbf{h} \in \mathbb{R}^d$ at various layers for a dataset.
- Train linear probe: $\hat{y} = \mathbf{W}\mathbf{h} + \mathbf{b}$ to predict property (e.g. part of speech).
- Measure accuracy. High accuracy = information is linearly accessible in $\mathbf{h}$.

### Valid conclusions

- That the information exists in the activations.
- Where it is most linearly accessible.

### Invalid conclusions

- That the model *uses* that information.
- Causal role.

**Caveats:** Pair with causal interventions. Good probe performance is necessary but not sufficient.

---

## 10. Tuned Lens *(not implemented)*

**What it does:** Improves on the logit lens by training *per-layer affine probes* (linear + bias) to decode hidden states into vocabulary distributions. Produces more faithful "latent predictions" at each layer than the raw unembedding.

### How it works

1. **Probe training:** For each layer $L$, train an affine map $\mathbf{W}_L \mathbf{h}_L + \mathbf{b}_L$ to predict the *final* model output (next-token distribution). The probes are trained on a corpus while the base model is frozen.
2. **Readout:** At inference, apply each probe to the cached residual at that layer. Get a distribution over the vocabulary—"what would the model predict if it stopped here, *as decoded by a probe trained for that*."
3. **Interpretation:** The trajectory of latent predictions across layers shows how the model iteratively refines its answer. More reliable than logit lens because intermediate representations may be rotated/shifted; the tuned lens accounts for that.

### Valid conclusions

- How predictions evolve layer-by-layer (more faithful than logit lens).
- When the model "commits" to an answer.
- Detection of anomalous inputs (trajectory can differ for adversarial examples).

### Invalid conclusions

- Still correlational—probes are trained to match final output, not to prove causation.
- Probe quality depends on training data and objective.

**Caveats:** Requires training probes per model. See [Tuned Lens](https://github.com/AlignmentResearch/tuned-lens) (EleutherAI/FAR, 2023).

---

## 11. Sparse Autoencoders (SAEs) *(not implemented)*

**What it does:** Decomposes activations into a sparse, overcomplete set of "features"—directions in activation space that activate on interpretable concepts. Addresses *polysemanticity* (one neuron, many meanings) by learning monosemantic feature directions.

### How it works

1. **Architecture:** An encoder maps activations $\mathbf{h} \in \mathbb{R}^d$ to a *larger* latent space $\mathbf{f} \in \mathbb{R}^{d'}$ ($d' \gg d$, e.g. 4$\times$), then a decoder reconstructs: $\hat{\mathbf{h}} = \mathbf{W}_{\text{dec}} \mathbf{f}$. The decoder columns are the "feature directions."
2. **Loss:** Reconstruction (MSE) + sparsity (L1 on $\mathbf{f}$). Sparsity encourages each activation to be explained by few features.
3. **Interpretation:** Each decoder column is a feature. Inspect which inputs activate it; use automated tools (e.g. AutoInterp) to generate text descriptions. Features often correspond to concepts (e.g. "Python code", "female pronoun").

### Valid conclusions

- Which sparse directions in activation space correspond to interpretable concepts.
- That the model *represents* certain information (features exist); not that it *uses* it (need causal interventions).
- Decomposition of polysemantic neurons into monosemantic features.

### Invalid conclusions

- That features are "ground truth" (SAE is a learned approximation).
- Causal importance (feature presence $\neq$ causal role).
- That all important structure is captured (SAEs can miss distributed or nonlinear structure).

**Caveats:** Training is expensive; feature quality varies. Superposition hypothesis underlies the approach. See [Anthropic](https://www.anthropic.com/research/sparse-autoencoders), [EleutherAI AutoInterp](https://blog.eleuther.ai/autointerp/) (2023–2024).

---

## 12. Distributed Alignment Search (DAS) *(not implemented)*

**What it does:** Answers: *"Where in the neural net does the model encode variable X?"* You specify a high-level causal model (e.g. subject, relation, object for "Paris is the capital of France"). DAS finds a *rotation* of the activation space such that different *subspaces* (groups of dimensions) correspond to those variables—and interventions in the high-level model match interventions in the neural net.

### The problem DAS solves

**Causal abstraction** asks: is there a mapping from neural activations to a simple causal model (e.g. a graph with variables and arrows) such that when we *intervene* on the causal model, the neural net behaves as if we had intervened on the corresponding part of the net?

**Old approaches** assumed each variable = a *disjoint* set of neurons (e.g. neurons 1–50 = subject). They used brute-force search over all such mappings—intractable for large models. They also missed *distributed* encodings: one variable spread across many neurons, one neuron contributing to many variables.

### How it works

1. **High-level causal model:** Define variables (e.g. subject, relation, object) and how they cause the output. Example: subject="Paris", relation="capital-of", object="France" → "Paris is the capital of France."

2. **Learn an orthogonal rotation $\mathbf{R}$:** DAS learns a rotation of the activation space. After rotation, we *partition* the $d$ dimensions into subspaces—e.g. dimensions 1–100 = "subject", 101–200 = "relation", etc. In this rotated basis, each subspace is hypothesized to encode one causal variable.

3. **Interchange intervention (the test):** Take two inputs—e.g. "Paris is the capital of France" (base) and "London is the capital of France" (source). Run both, get activations $\mathbf{h}_{\text{base}}$ and $\mathbf{h}_{\text{source}}$. Rotate: $\mathbf{r} = \mathbf{R} \mathbf{h}$. In the rotated space, *swap* the "subject" subspace of the base with the subject subspace of the source. Rotate back and feed through the rest of the model. The output should change as if we had swapped the subject in the causal model (Paris → London). If it does, the alignment is *faithful*.

4. **Optimization:** DAS uses *gradient descent* to find $\mathbf{R}$ (and subspace boundaries) that maximize faithfulness—i.e. that make interchange interventions in the neural net match interventions in the causal model. No brute-force search.

### Why "distributed"?

The rotation $\mathbf{R}$ puts activations in a *non-standard basis*. In the standard basis (raw neurons), "subject" might be spread across hundreds of neurons. In the learned basis, "subject" lives in a clean subspace. One neuron can contribute to multiple variables (it has components in multiple subspaces). This is *distributed* representation.

### Valid conclusions

- Which subspaces (in the learned basis) encode which causal variables.
- That the model's behavior is *consistent* with the hypothesized causal structure.
- That representations are distributed (not one-neuron-per-variable).

### Invalid conclusions

- That the alignment is unique (many rotations may work).
- That the causal model is "true" (we test consistency, not ground truth).
- That the model "uses" these variables (alignment $\neq$ causal necessity).

**Caveats:** Requires specifying a causal model. Subspace sizes can be learned (Boundless DAS) or fixed. Scaling to large models (e.g. Alpaca-7B) uses Boundless DAS. See [Geiger et al.](https://proceedings.mlr.press/v236/geiger24a.html) (2024).

---

## 13. Attribution Patching *(not implemented)*

**What it does:** Estimates the causal importance of each *edge* (connection) in the computational graph using a *linear approximation* to activation patching. Much more efficient than running full activation patching over all components.

### How it works

1. **Idea:** Activation patching measures $\Delta y$ when patching component $c$. Attribution patching approximates this with $\nabla_c \ell \cdot \Delta c$ (first-order Taylor), where $\Delta c$ is the difference between clean and corrupt activations at $c$.
2. **Computation:** One forward pass (clean), one forward pass (corrupt), one backward pass. Per-edge attributions come from the gradient of the metric w.r.t. activations, combined with the clean–corrupt difference.
3. **Circuit discovery:** Rank edges by attribution; prune low-attribution edges. The remaining subgraph is the "circuit."

### Valid conclusions

- Which edges/components contribute most to the behavior (approximate).
- Efficient circuit localization for large models (O(1) in number of components vs O(n) for activation patching).

### Invalid conclusions

- Exact causal effect (linear approximation can be wrong; residual error exists).
- That pruned edges are irrelevant (approximation may miss nonlinear effects).

**Caveats:** Linear approximation fails when effects are nonlinear. AtP* (DeepMind, 2024) improves robustness and provides error bounds. See [Attribution Patching](https://arxiv.org/abs/2310.10348) (2023), [AtP*](https://deepmind.google/research/publications/68553/) (2024).

---

## 14. Causal Abstraction *(theoretical framework)*

**What it does:** Provides a *theoretical framework* that unifies mechanistic interpretability methods. A high-level (e.g. symbolic) model is a *causal abstraction* of a low-level (neural) model if interventions on the high-level model correspond to interventions on the low-level model.

### How it works

1. **Interchange intervention:** The core operation—replace low-level activations with values they would have under a different input. This is exactly *activation patching*.
2. **Abstraction:** A mapping from low-level states to high-level states is faithful if, when we intervene on the high-level model and translate to low-level, the low-level model behaves as if we had done the corresponding low-level intervention.
3. **Unified view:** Activation patching, path patching, causal mediation analysis, SAEs, DAS, and others can be cast as testing or instantiating causal abstractions.

### Valid conclusions

- A common language for comparing interpretability methods.
- Formal criteria for when a high-level explanation is "faithful" to the model.
- That methods like activation patching are instances of interchange interventions.

### Invalid conclusions

- A single "correct" abstraction (many may exist).
- That the framework itself does interpretability (it organizes and formalizes; you still need to run experiments).

**Caveats:** Theoretical; implementation is method-specific. See [Geiger et al.](https://www.jmlr.org/papers/v26/23-0058.html) (2023).

---

## 15. Representation Engineering / Steering Vectors *(not implemented)*

**What it does:** Identifies *directions* in activation space that correspond to concepts (e.g. "honesty", "sycophancy") and *adds* them to activations during inference to steer model behavior—without changing weights or prompts.

### How it works

1. **Concept vectors:** Collect activations on prompts that exhibit a concept vs. prompts that don't. The difference (or PCA of differences) gives a "concept vector" $\mathbf{v}$.
2. **Steering:** At inference, add $\alpha \mathbf{v}$ to activations at chosen layers/positions. Positive $\alpha$ amplifies the concept; negative suppresses it.
3. **Interpretability:** Concept vectors can be decomposed (e.g. with SAEs) to see which features contribute. Used for control (reduce sycophancy, detect deception) and for understanding what the model represents.

### Valid conclusions

- That a concept is *represented* in a linear (or approximately linear) direction.
- Ability to *steer* behavior by intervening on activations.
- Where in the model the concept is most strongly represented (which layers).

### Invalid conclusions

- That steering is "safe" or robust (can have unintended side effects).
- That the concept vector is the "true" representation (it's a learned approximation).
- Causal necessity (steering changes behavior but doesn't prove the model "uses" that direction normally).

**Caveats:** Steering can affect unrelated behaviors. SAE-based steering (SAE-TS, 2024) aims to reduce side effects. See [Representation Engineering](https://www.alignmentforum.org/posts/3ghj8EuKzwD3MQR5G/an-introduction-to-representation-engineering-an-activation) (2023–2024).

---

## Summary Table

### Implemented (1–9)

| Method              | Causal? | What it measures                    | Best for                        |
| ------------------- | ------- | ----------------------------------- | ------------------------------- |
| Activation patching | Yes     | Causal importance of components     | Localizing circuits             |
| Logit lens          | No      | Intermediate "predictions"          | Exploratory, when info appears  |
| Copying heads       | No      | Attention to previous token         | Finding previous-token heads    |
| Induction heads     | No      | Attention to B in [A][B]...[A]→[B]  | Finding induction heads         |
| Neuron analysis     | No      | Raw MLP activations                 | Baseline, sparsity patterns     |
| Multi-fact patching | Yes     | Causal importance across facts      | Robust localization             |
| Attention analysis  | No      | Where heads read from               | Understanding head behavior     |
| DLA                 | No      | Decomposition of logits             | Attribution, not causation      |
| Linear probes       | No      | Linear accessibility of information | Checking if info exists in acts |

### Advanced / Recent (10–15, not implemented)

| Method              | Causal? | What it measures                    | Best for                        |
| ------------------- | ------- | ----------------------------------- | ------------------------------- |
| Tuned lens          | No      | Faithful latent predictions per layer | Better than logit lens       |
| Sparse autoencoders | No      | Sparse interpretable features       | Decomposing polysemantic neurons |
| DAS                 | No      | Aligned causal subspaces            | Finding distributed concepts    |
| Attribution patching| Yes*    | Approximate edge importance         | Efficient circuit discovery     |
| Causal abstraction  | —       | Theoretical framework               | Unifying interpretability       |
| Steering vectors   | No      | Concept directions, steerable     | Control, concept localization   |


