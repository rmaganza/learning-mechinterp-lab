# Common Pitfalls in Mechanistic Interpretability

Avoid these mistakes to produce more reliable and interpretable research.

## 1. Confusing Correlation with Causation

**Pitfall:** A component has high logit attribution or probe accuracy, so you conclude it *causes* the behavior.

**Why it's wrong:** The model may have redundant pathways. A component can contribute to the logits without being necessary—if you ablate it, another pathway may compensate.

**Fix:** Use activation patching or other causal interventions. If patching a component doesn't change the output, it's not causally necessary for that run.

---

## 2. Single-Example Generalization

**Pitfall:** You find a circuit on one prompt and conclude it generalizes to the whole task.

**Why it's wrong:** Models can use different strategies for different inputs. One example may be an outlier.

**Fix:** Run experiments on multiple prompts. Report variance. Consider stratified evaluation (e.g., by length, difficulty, or template).

---

## 3. Position Alignment in Patching

**Pitfall:** Clean and corrupted prompts have different token lengths. You patch by position, so you're comparing misaligned positions.

**Why it's wrong:** Token "Paris" at position 5 in clean may correspond to "London" at position 5 in corrupt, but if lengths differ, position 5 may mean different things (e.g., different words).

**Fix:** Use prompts with the same number of tokens when possible. If not, consider patching by *token index* or using a position mapping.

---

## 4. Ignoring the Residual Stream

**Pitfall:** You focus only on attention heads and ignore residual stream contributions.

**Why it's wrong:** The residual stream carries information between layers. MLPs write to it. Many "attention" effects are actually mediated by the residual. Skipping layers (e.g., only looking at attn) can miss important computation.

**Fix:** Include residual stream and MLP components in your analysis. Use logit lens or DLA on the residual.

---

## 5. Overinterpreting Attention Weights

**Pitfall:** A head attends strongly to position X, so you conclude it "reads" or "uses" that information.

**Why it's wrong:** Attention weights determine *mixing*, but the *content* comes from value vectors. A head can attend to X but get low-norm values, so the actual contribution is small.

**Fix:** Analyze attention *output* (values weighted by attention), not just patterns. Use activation patching on head outputs.

---

## 6. Cherry-Picking Visualizations

**Pitfall:** You show one attention pattern or one logit lens curve that looks clean and supports your story.

**Why it's wrong:** It's easy to find one example that fits. The reader can't tell if it's representative.

**Fix:** Show multiple examples. Report aggregate statistics. Include counterexamples or failure cases.

---

## 7. Baseline Choice in Attribution

**Pitfall:** You use a zero baseline for DLA or similar, without considering whether that's meaningful.

**Why it's wrong:** Attribution is relative to a baseline. Different baselines (zero, mean, corrupted run) give different results. Zero may not correspond to "no information."

**Fix:** Justify your baseline. Consider ablating to a corrupted or neutral input. Report sensitivity to baseline choice.

---

## 8. Probe Overinterpretation

**Pitfall:** A linear probe gets 95% accuracy, so you conclude the model "represents" that feature.

**Why it's wrong:** Probes find the best linear readout. The model might use a nonlinear combination, or the information might exist but be unused. High probe accuracy ≠ model uses it.

**Fix:** Treat probes as *necessary* (info exists) not *sufficient* (model uses it). Follow up with causal interventions.

---

## 9. Small Model ≠ Big Model

**Pitfall:** You find a circuit in GPT-2 small and assume the same structure exists in GPT-4.

**Why it's wrong:** Scaling changes internal structure. Larger models may use different algorithms, more redundancy, or different factorization.

**Fix:** Be explicit that results are for the model you tested. Avoid overgeneralizing to other scales or architectures.

---

## 10. Ignoring Training Dynamics

**Pitfall:** You analyze a trained model and assume the circuit you find was learned for the task you're studying.

**Why it's wrong:** The model may have learned the circuit for a different objective (e.g., next-token prediction) that correlates with your task. Or the circuit may be a byproduct of other learning.

**Fix:** Consider what the model was trained on. Be cautious about "why" the model does X—you're often inferring from static analysis.

---

## Quick Checklist

Before publishing or presenting:

- Did I use causal interventions (e.g., patching) for causal claims?
- Did I test on multiple examples?
- Are my clean/corrupt prompts aligned (same length, minimal difference)?
- Did I consider residual stream and MLPs, not just attention?
- Did I avoid overinterpreting attention weights alone?
- Did I show variance and counterexamples?
- Is my baseline choice justified?
- Did I distinguish "information exists" (probes) from "model uses it" (causal)?
- Did I scope my claims to the model I actually tested?

