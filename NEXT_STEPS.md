# Next Steps: Temporal Awareness Steering

## Issues (Prioritized)

### 1. URGENT

#### HF Token Exposed in Notebook

**Cell 5 of `notebooks/06_probing_and_steering.ipynb`** contains a hardcoded Hugging Face token:
```python
login(token="hf_YOUR_TOKEN_HERE")
```

**Fix:**
1. Revoke the token immediately at https://huggingface.co/settings/tokens
2. Generate a new token
3. Replace the hardcoded value with:
   ```python
   import os
   login(token=os.environ["HF_TOKEN"])
   ```
4. Add `HF_TOKEN` to your environment (e.g., `.env` file, Colab secrets, or shell profile)
5. Confirm `.env` is in `.gitignore`

---

### 2. HIGH PRIORITY

#### Domain-Matched CAA Sign is Inverted at L24

The DM-CAA vector at L24 has cosine **−0.0801** with the B-A unembedding direction — *negative*, meaning the "LT" steering label actually pushes toward short-term and vice versa. Once this sign flip is accounted for, DM-CAA produces **4-7x larger effects** than implicit CAA (up to ~15-37% ΔP per question vs ~3-7%). This transforms the DM-CAA from the apparent worst performer to the best.

**Immediate fix:** Negate the DM-CAA vector (swap `+1.0` / `-1.0` in `_make_dm_hooks` calls) and re-run the binary choice sweep to confirm corrected effects. Alternatively, steer at L13 where cosine is +0.0130 (correct sign).

**Implication:** The weak effect sizes reported for implicit CAA are a directional alignment problem, not a fundamental limitation. The temporal feature is strongly steerable.

#### Implicit CAA Small Effect Sizes (~3-7% ΔP)

The implicit CAA produces monotonic but weak effects: +3.6% max LT, −7.3% max ST. This is because the implicit CAA vector has cosine ~0.01 with the logit decision direction — it disperses probability rather than redirecting it along the A/B axis.

**Context (now less urgent):** The DM-CAA sign-flip finding shows the model is highly steerable with a better-aligned vector. The implicit CAA issue is secondary — the priority is to exploit the DM-CAA result and validate with B-A direction steering on open-ended text.

**Fix approaches (see Next Steps below):** Use sign-corrected DM-CAA, probe weights, or B-A direction steering for stronger effects.

#### Layer 31 CAA Contamination

Both explicit and implicit datasets show peak CAA magnitude at Layer 31, but the explicit dataset's peak is disproportionately large. Late layers in transformers handle surface-level token prediction — the L31 signal likely reflects token-level artifacts (e.g., words like "future", "now") rather than semantic temporal reasoning. The switch to the implicit dataset for probing was correct and validated by the 83.3% cross-dataset generalization.

---

### 3. MEDIUM PRIORITY

#### Only 6 Binary Choice Questions

The current evaluation set of 6 questions limits statistical power. Individual questions show high variance (Q4 baseline P(LT) = 0.783 vs Q1 = 0.186), and we cannot estimate confidence intervals or effect heterogeneity with so few items.

**Fix:** Expand to 30-50 questions spanning diverse domains (finance, health, career, education, policy, environment). Balance so baseline P(LT) clusters around 0.3-0.7 for maximum sensitivity.

#### No Statistical Significance Tests

All reported ΔP values are point estimates with no uncertainty quantification. We cannot distinguish a +3.6% effect from noise without confidence intervals.

**Fix:** Add bootstrap CIs (resample questions with replacement, 1000 iterations), permutation tests (shuffle steering labels), and report p-values for the monotonicity claim.

#### No Probe Regularization Sweep

A single `C=0.1` was used for all probes. The regularization strength affects which layers appear "best" and how much the probe overfits.

**Fix:** Sweep `C` ∈ {0.001, 0.01, 0.1, 1.0, 10.0} with nested cross-validation. Compare logistic regression to a 1-hidden-layer MLP to check for nonlinear signal.

---

### 4. LOW PRIORITY

#### Hardcoded Colab Paths

Several cells reference `/content/temporal-awareness` which is Colab-specific and not portable. Output paths should use relative paths or the existing `project_paths.py` utilities.

#### Template Variables Unfilled

Cell 42 (markdown summary) contains placeholder text: `{best_layer}`, `X.XXX`. These should be filled in or replaced with dynamically generated output.

#### Lambda Hook Style

Minor code quality issue: hook lambdas include an unused `hook` parameter (e.g., `lambda act, hook=None, v=vec: act + v`). While functionally harmless, it adds noise. TransformerLens requires this signature, so this is cosmetic only.

---

## Next Steps (Building Off Current Work)

### 1. B-A Direction Steering on Open-Ended Text Generation

**Rationale:** The B-A unembedding direction achieves P(LT) = 1.000 on binary choice at L24 — it is the strongest confirmed steering vector. But binary choice only measures token-level preference between ' A' and ' B'. The critical question is: **does steering with the B-A direction also shift the semantic content of free-form text toward longer or shorter time horizons?** If open-ended generations become measurably more future-oriented (or present-oriented) under B-A steering, this validates that the logit-level temporal signal is connected to genuine semantic temporal reasoning, not just surface token manipulation.

**Implementation:**
1. Use the existing text generation pipeline (Part 3, Section 10 in the notebook)
2. Replace the CAA vector with the B-A unembedding direction: `b_minus_a_unit = (W_U[:, id_B] - W_U[:, id_A]) / norm`
3. Steer at L24 (where binary choice effect is strongest) and L13/L18 (for comparison)
4. Use moderate strengths first (4.0, 8.0) — B-A is extremely potent at 16-20, which may cause incoherence in open-ended generation
5. Generate on the same neutral prompts used in Part 3 (decision-making stems with no temporal cues)
6. Evaluate generated text for temporal horizon bias:
   - **Qualitative:** Does +B-A text discuss long-term planning, future consequences, investment? Does −B-A text discuss immediate action, quick fixes, present concerns?
   - **Quantitative:** Run a temporal keyword classifier or use an LLM judge to score each generation on a short-term ↔ long-term scale

**Expected outcome:**
- If text shifts semantically: this is strong evidence that the A/B logit direction encodes genuine temporal preference, not just token bias. It would mean the binary choice format is a valid probe of deeper temporal reasoning.
- If text does *not* shift (or becomes incoherent): the B-A direction only controls surface-level A/B token selection and does not touch the semantic temporal circuits. This would motivate using the sign-corrected DM-CAA or probe weights instead, since those vectors were derived from *semantic* activation differences.

**Connection:** This is the most important experiment to run next because it bridges the gap between "we can flip a binary choice token" and "we can steer the model's temporal reasoning." It directly tests the research question of whether intertemporal preference is a steerable feature.

### 2. Validate Sign-Corrected DM-CAA

**Rationale:** The DM-CAA vector produces 4-7x larger effects than implicit CAA, but with an inverted sign at L24. Before building on this, we need to confirm the sign-corrected version works as expected and characterize its behavior on both binary choice and open-ended generation.

**Implementation:**
1. Negate the DM-CAA vector: swap `+1.0` to `-1.0` for LT steering (and vice versa) in `_make_dm_hooks`
2. Re-run the binary choice sweep to confirm corrected ΔP values
3. Run open-ended text generation with the corrected DM-CAA at L24
4. Compare text quality and temporal bias to B-A direction steering and implicit CAA steering

**Expected outcome:** Corrected DM-CAA should show ΔP of +15-37% in the right direction on binary choice. On open-ended text, it may produce more coherent temporal shifts than B-A steering since it was derived from semantic activation differences rather than raw token unembeddings.

**Connection:** If DM-CAA produces both strong binary choice effects *and* coherent semantic shifts in open-ended text, it is the best all-around steering vector — stronger than implicit CAA and more semantically grounded than B-A.

### 3. Use Probe Weights as Steering Vectors

**Rationale:** The L14 logistic regression probe achieves 92.5% accuracy, meaning its weight vector `probe.coef_[0]` defines a hyperplane that *optimally separates* temporal horizons. This is by definition a better temporal direction than the mean-difference CAA vector.

**Implementation:**
1. Load the trained probe from `out/experiments/week3_steering/probe_layer_14.pkl`
2. Extract `probe.coef_[0]` (shape `[4096]`), normalize to unit length
3. Use as a drop-in replacement for `caa_unit[14]` in the steering pipeline
4. Run the same binary choice evaluation with identical strengths

**Expected outcome:** Higher cosine similarity with the B-A direction and larger ΔP, since the probe weight is trained to maximize class separation on the same task the steering targets.

**Connection:** Directly addresses the core issue (CAA orthogonal to logit direction) without requiring new data or infrastructure.

### 4. Distributed Alignment Search (DAS)

**Rationale:** DAS trains a rotation matrix to find the 1D subspace that maximally separates two conditions, handling cases where the relevant direction is not axis-aligned or not capturable by a simple mean difference.

**Implementation:**
1. Install `pyvene` (already a supported backend in the codebase)
2. Train a DAS rotation on the implicit dataset activations at layers 12-16
3. Extract the learned rotation's principal direction as the steering vector
4. Compare DAS vectors to CAA and probe-weight vectors via cosine similarity

**Expected outcome:** A more principled steering direction that may capture nonlinear aspects of the temporal encoding. The DAS direction should have higher alignment with both the probe decision boundary and the logit direction.

**Connection:** This is the standard mechanistic interpretability approach when mean-difference CAA underperforms.

### 5. Multi-Layer Steering

**Rationale:** The current single-layer (L24) steering leaves temporal signal on the table. Probes are strong at L11-L15, and the grid search found L18 as the best single layer (ΔP = +0.1717). Steering multiple layers simultaneously may compound the effect.

**Implementation:**
1. Set `W3_STEERING_MODE = "multi"` in the notebook
2. Steer layers 12, 14, 16, 18 simultaneously at moderate per-layer strength (e.g., 2.0-4.0)
3. Compare total ΔP to best single-layer result
4. Monitor for coherence degradation via text generation

**Expected outcome:** Additive steering effects from multiple layers, potentially reaching ΔP > 0.20 without coherence loss.

**Connection:** The grid search showed 26 monotone-improvement (layer, strength) combinations — multi-layer steering exploits this breadth.

### 6. Expand Binary Choice Evaluation

**Rationale:** 6 questions is insufficient for statistical claims. Effect heterogeneity across questions (Q1 ΔP = +0.042, Q4 ΔP = +0.019) suggests domain-specific variation that we cannot characterize with the current set.

**Implementation:**
1. Generate 30-50 binary choice questions using the existing `scripts/intertemporal/generate_prompt_dataset.py`
2. Span domains: finance, health, career, education, policy, environment, technology
3. Filter for baseline P(LT) in [0.25, 0.75] to maximize sensitivity
4. Add bootstrap CIs (1000 resamples) and permutation tests

**Expected outcome:** Statistically robust effect estimates with per-domain breakdown, enabling claims like "steering is strongest for financial planning questions."

### 7. Cross-Model Generalization

**Rationale:** If the temporal direction transfers across architectures, this is a much stronger finding than model-specific results. It would suggest temporal horizon is a universal feature of instruction-tuned LLMs.

**Implementation:**
1. Run the extraction + probing pipeline on Mistral-7B-Instruct and Gemma-2B-IT (both supported by the existing `ModelRunner` backend system)
2. Compare: (a) probe accuracy per layer, (b) CAA vector cosine similarity with Llama-3.1-8B, (c) steering ΔP
3. Optionally test Llama-3-70B to check whether larger scale strengthens the signal

**Expected outcome:** Similar probe accuracy profiles (peak in middle layers) with varying CAA alignment. Cross-model steering transfer would be the strongest possible evidence for a universal temporal feature.

### 8. Nonlinear Probes

**Rationale:** Logistic regression captures only linear separability. If there's additional nonlinear temporal signal, an MLP probe will find it — and its representations may yield better steering vectors.

**Implementation:**
1. Train a 1-hidden-layer MLP (e.g., 4096 → 256 → 1) with the same train/test split
2. Compare accuracy to logistic regression at each layer
3. If MLP significantly outperforms LR (>3% gap), extract the learned representation and use it for steering

**Expected outcome:** If MLP accuracy ≈ LR accuracy, the temporal feature is fully linear (good news — simpler to steer). If MLP >> LR, there's exploitable nonlinear structure.

### 9. Causal Validation via Activation Patching

**Rationale:** Probing shows correlation (temporal information is *present* at layer 14), but not causation (layer 14 activations *determine* temporal choice). Activation patching tests causality by replacing activations and measuring downstream effects.

**Implementation:**
1. Use the existing `src/activation_patching/` module
2. For each binary choice question: patch layer-L activations from a "long-term prompt" into a "short-term prompt" run
3. Measure whether the patched model switches its binary choice
4. Sweep layers to find the causally important layers

**Expected outcome:** A causal importance profile across layers. If L14 is both probing-best and causally-important, that strongly validates it as the steering target.

### 10. Per-Category Analysis

**Rationale:** Not all temporal reasoning may be encoded equally. Financial planning questions may activate different circuits than health-related temporal tradeoffs.

**Implementation:**
1. Tag existing dataset pairs by category (planning, resource allocation, health, education, etc.)
2. Train per-category probes and compare accuracy
3. Compute per-category CAA vectors and check inter-category cosine similarity

**Expected outcome:** Identifies which domains have the strongest/weakest temporal encoding, guiding targeted improvements to the dataset and steering approach.

---

## Priority Order for Maximum Research Impact

1. **B-A direction on open-ended text** (1-2 hours) — tests whether logit-level steering translates to semantic temporal shifts; answers the core research question
2. **Validate sign-corrected DM-CAA** (1 hour) — confirms the 4-7x stronger effects work in the correct direction; quick win
3. **Probe weights as steering vectors** (1-2 hours) — high expected ROI, may combine strengths of semantic grounding (like DM-CAA) with correct alignment (like B-A)
4. **Expand binary choice evaluation** (2-3 hours) — required for any statistical claims
5. **Multi-layer steering** (1 hour) — quick experiment, already supported by notebook config
6. **Causal validation** (3-4 hours) — infrastructure exists, validates the entire approach
7. **DAS** (4-6 hours) — principled but requires more setup
8. **Cross-model generalization** (6-8 hours) — strongest evidence but most compute
9. **Nonlinear probes** (2-3 hours) — diagnostic, may not change steering approach
10. **Per-category analysis** (2-3 hours) — useful for paper framing but not blocking
