# LTDM — Phase R Memo: Go/No-Go on D1 + D3

| Field | Value |
|---|---|
| Status | v1.0 — Decision memo |
| Date | 2026-06-09 |
| Owner | Manuel Grande Écija |
| Reviewer | Claude (architect-advisor role) |
| Inputs | Full-text reads: arXiv 2412.06060 (Intel), 2503.02851 (Sparks/HCL), 2604.04743 (Basins), 2507.16488 (ICR Probe). Adversarial check on D3: 2410.20290 (Speculative Rejection, NeurIPS 2024) + descendants. Skim-level: 2504.09389 (Novelty Frontier). New neighbors surfaced during reads: 2510.04933 (LSD), 2604.15400 (Trajectory Commitment), 2605.05953 (PCNet). |
| Verdict | **D1: GO (conditions C1–C3). D3: GO with restated claim (conditions C4–C5).** Phase R closed. |

---

## 1. D1 validation — the latent measurement gap

Question asked: what exact mathematical signal does each neighbor compute over internal states, and is it viable online (ms-scale, during inference) or post-hoc?

### 1.1 Intel — Steering LLMs to Evaluate and Amplify Creativity (2412.06060)

**Signal (exact):** a single static *creativity direction* at one empirically chosen layer (layer 8 of Llama3-8B):

    a = normalize( mean_{x_c ∈ X_c} LLM_l(x_c) − mean_{x_u ∈ X_u} LLM_l(x_u) )

built from 500 creative vs GPT-4o-negated ("uncreative") prompt pairs.

**Scoring (exact):** mean per-token cosine similarity to that direction over the generation:

    score = (1/(T+1)) · Σ_{t=0..T} cos( LLM_l(x_t), a )

**Steering (exact):** LLM_l'(x) = LLM_l(x) + λ·a, with λ = 3 found manually.

**Online viability:** trivially yes — one dot product per token at one layer (microseconds). This is the cheapest signal in the field.

**What it is NOT:** it is a *static projection on one direction at one layer*, not a trajectory measure — no cross-layer dynamics, no per-token evolution, no divergence magnitude. **No validity/hallucination axis at all** (creativity only, judged pairwise by Llama3-70B/humans). Domain: creative writing only; model: Llama3-8B only. Evaluation is a 2-page workshop-scale study.

### 1.2 Sparks — HCL framework (2503.02851)

**Signal (exact): there is no latent metric.** HCL decodes *full text responses* from intermediate layers via Layer-Skip early-exit (50 samples per question per layer), then scores the **text**:

    S_H^i = N_err^(i) / D^(i)                      (error rate at layer i)
    S_C^i = N_clusters^(i)                          (# semantic clusters among CORRECT answers;
                                                     all-MiniLM-L6-v2 embeddings, cosine τ = 0.8)
    S_HCB^i = 0.5 · S_C_norm^i + 0.5 · (1 − S_H^i)  (the joint trade-off score)

**Online viability: no.** 50 generations per question per layer, ~1066 GPU-hours per model. This is a post-hoc statistical framework, not a monitor.

**What it proves for us:** the joint creativity↔hallucination trade-off is real and measurable per layer, with a stable optimal *depth* (e.g., layer 8 in LLaMA2-7B) — i.e., a fertile band exists **over the depth axis, at output level, by brute force**. Their own limitations: closed-ended QA only; creativity = diversity-of-correct (no novelty/originality); small model set.

### 1.3 Hallucination Basins (2604.04743)

**Signals (exact):** per-layer geometry against a reference centroid μ^(ℓ) built from ~1000 *uninformative contexts*:

    d_basin^(ℓ)(h) = ‖ h^(ℓ) − μ^(ℓ) ‖₂                         (radial distance to basin center)
    κ_local^(ℓ)(h) = ‖h^(ℓ+1) − μ^(ℓ+1)‖₂ / (‖h^(ℓ) − μ^(ℓ)‖₂ + ε)   (local contraction ratio; κ<1 = collapsing into basin)
    Φ(x) = [ min_ℓ d^(ℓ) , mean_ℓ κ^(ℓ) ]                        (2-D risk signature)
    h_steered^(ℓ) = h^(ℓ) + λ(Φ(x)) · (μ_fact^(ℓ) − μ_hall^(ℓ))     (adaptive steering; λ(Φ) = trained logistic map)

Plus offline separation diagnostics: variance ratio ρ_var, Fisher ratio, Mahalanobis AUROC.

**Online viability: yes for the controller.** Algorithm 2 = one forward pass with hidden states + O(L) norms + a logistic regression → milliseconds. Centroids and steering vectors require **offline construction with labeled factual/hallucinated sets per task**.

**What it is NOT:** the entire apparatus points toward **suppression** (steer toward μ_fact, away from hallucination basins). No creativity/novelty axis. **Critical caveat for us:** their detection collapses to AUROC ≈ 0.5 on generation (summarization) and misconception tasks — basin geometry is task-dependent and weakest exactly where outputs are open-ended. Models mostly 1B–3B.

### 1.4 ICR Probe (2507.16488)

**Signal (exact):** per token i, per layer ℓ, with update Δx_i^ℓ = x_i^ℓ − x_i^{ℓ−1}:

    p_{i,j}^ℓ = (Δx_i^ℓ)ᵀ · x_j^ℓ / ‖x_j^ℓ‖        (projection of the update onto each context token state)
    Proj_i^ℓ  = softmax_j( p_{i,j}^ℓ )
    Attn_i^ℓ  = softmax( mean over H heads of QᵀK/√d )
    ICR_i^ℓ   = JSD( Proj_i^ℓ[S] ‖ Attn_i^ℓ[S] ),   S = top-k=20 attention indices

Probe: token-average the N×L matrix → 1×L vector → MLP (L,128,64,32,1), <16K params. AUROC 0.74–0.84 on Llama-3-8B-It / Qwen2.5-7B-It / Gemma-2-9B-it.

**Online viability: yes** — authors claim real-time detection from a single generation; needs hooks on hidden states of consecutive layers **and attention logits** (heavier capture than Intel/Basins, still single-pass). Training-based (needs labeled data; cross-dataset drop ~8.6%).

**What it is NOT:** detection only; their own stated future work is residual-stream *intervention to reduce* hallucination. No creativity axis.

### 1.5 D1 verdict

**GO.** The precise gap statement that survives full-text reads:

> Every neighbor measures EITHER creativity from internals (Intel — static, single direction, single layer, no validity axis) OR hallucination from trajectory dynamics (Basins, ICR, LSD, Trajectory Commitment — no creativity axis, suppression-oriented). The one paper that measures the JOINT trade-off (Sparks) does it at output level, by brute-force sampling, over the depth axis, offline. **No published work measures latent trajectory divergence against the joint novelty×validity outcome, and none tests a fertile band over divergence *magnitude* under steering.**

Bonus de-risking: we do **not** need to invent divergence metrics. The candidate family D(t) is pre-built and citable: {Intel cosine-to-direction, d_basin, κ_local, ICR-JSD, LSD velocity/acceleration}. Our contribution is the *axis rotation* (suppression → harvest) and the joint outcome.

**Conditions:**
- **C1 (baselines):** temperature sweep, min-p, Intel static projection, and a Sparks-style HCB computed on our outputs must all be included. If latent D(t) does not beat the static projection and temperature as a predictor of novel∧valid, D1 is a clean negative.
- **C2 (task design):** Basins' AUROC collapse on open-ended tasks is a direct threat. Tasks must have a *verifiable validity component* (constrained problem solving / micro code or math domain), not open summarization.
- **C3 (pre-registration):** fix the metric family, layers, aggregation, and judge protocol before running. Report all metrics, not the best one.

---

## 2. D3 validation — the early-reject router

Question asked: confirm FunSearch-class systems are output-level brute force, and that no early-exit routing from intermediate signals exists.

### 2.1 FunSearch / AlphaEvolve: CONFIRMED output-level

FunSearch (Nature 625, 2024): frozen LLM proposes programs ("creative solutions"); a systematic evaluator **executes and scores every sampled program**, explicitly to "guard against confabulations." AlphaEvolve (2506.13131): LLM ensemble generates diffs; **evaluator pools execute and score**; program database + MAP-Elites archive drives selection. In neither system — nor in the descendant family (EoH, HeurAgenix, ThetaEvolve, concept-tree search) — is any internal state of the generator read. Every candidate pays full generation cost + full evaluation cost. **Confirmed.**

### 2.2 The strong-form gap claim is FALSE — correction

**Speculative Rejection (Sun et al., NeurIPS 2024, arXiv 2410.20290)** already does early-exit routing that aborts doomed branches mid-generation: in Best-of-N, a **reward model scores partial utterances** at a decision token and halts the lowest-ranked quantile, exploiting the positive correlation between partial and final rewards. Reported savings ≈ 85% of tokens; Best-of-N needs 16–32 GPUs to match its single-GPU reward. Descendants: self-certainty variants (logit-level confidence), Early-Stop Self-Consistency (answer-convergence windows), optimal-stopping/Pandora's-box framings (2510.01394). Additionally, **PCNet (2605.05953)** gates a decoding intervention using a density estimate (exact NLL via probabilistic circuit) over a projection of the **final hidden state** — an internal-signal router, single state, factuality-correction objective.

Phase-0's sentence "the literature lacks an early-exit routing system that aborts doomed branches" is therefore **retired**. Do not write it anywhere.

### 2.3 The gap that survives (restated D3)

What no surfaced system does — and what becomes the defensible claim:

1. **Signal source:** abort/keep decisions driven by the generator's **latent trajectory** (hooks already running, near-zero marginal cost) instead of a reward model repeatedly scoring partial *text* (SpecRej pays RM forward passes per checkpoint) or a verifier executing outputs (FunSearch).
2. **Objective:** maximize **discovery yield per verifier cost** — novel∧valid candidates surviving per unit of evaluation compute — in FunSearch-style harvest loops; not alignment reward (SpecRej), not factuality correction (PCNet, Basins).
3. **Selection semantics:** keep a **band** (the fertile zone), not just kill the bottom of a reward ranking. Testable side-hypothesis with real teeth: *partial-reward models are biased against novel-but-valid candidates* (novelty looks low-reward early), so SpecRej-style pruning systematically destroys the fertile band that LTDM preserves. If true, LTDM wins on yield, not just on cost.

### 2.4 D3 verdict

**GO, reframed.** Conditions:
- **C4 (mandatory baseline):** Speculative Rejection (partial-reward pruning) becomes the primary baseline, alongside full Best-of-N-and-verify (FunSearch-style). LTDM must beat SpecRej on verifier-cost-per-discovery and/or on novelty preservation.
- **C5 (cost accounting):** the experiment domain must have a real, meterable verifier (unit tests for code / symbolic check for math) so "yield per verifier cost" is a measured number, not an estimate.

---

## 3. Positioning statement v2 (supersedes Phase 0 §6)

> Internal-state monitoring of LLMs is mature for *suppressing* hallucination — static creativity directions exist (Intel), and trajectory-dynamics detectors exist (ICR, Basins, LSD) — while discovery systems harvest "hallucination" at the *output* level by verifying every sample (FunSearch, AlphaEvolve) or pruning by partial reward (Speculative Rejection). **LTDM tests whether latent trajectory divergence, measured with the field's own metrics, predicts the joint novelty×validity outcome; whether a fertile divergence band exists under steering; and whether routing on that free internal signal yields more novel-and-valid discoveries per unit of verification compute than reward-based pruning.**

---

## 4. Phase H requirements seed (prototype: monitor, hooks, tensor extraction)

**H-1 Instrumentation (the harness, RED first):**
- Model: open-weights 7–8B (Qwen2.5-7B or Llama-3.1-8B), fp16, fixed seeds.
- Hooks: `output_hidden_states=True` (all layers, per generated token) + attention logits capture (required only for ICR-JSD; make it a toggle, it doubles memory).
- Tensor budget: per generation of T tokens — hidden states T×L×d (fp16). For T=256, L=32, d=4096 ≈ 64 MB/gen. Store reduced features online (the four metrics), raw tensors only for a debug subsample.
- Reference trajectories: (a) greedy decode of same prompt; (b) Basins-style uninformative-context centroid μ^(ℓ); compute D(t) against both, pre-registered.

**H-2 Metric module (D(t) family, all computed per token, per layer):**
cos-to-creativity-direction (Intel; build contrast set for our task domain), d_basin, κ_local, ICR-JSD (toggle), trajectory velocity/acceleration (LSD-style ‖Δh‖ across layers).

**H-3 Generation conditions:** {temperature ∈ sweep} × {steering strength λ ∈ sweep along Intel-style direction} × {base vs instruct arm (mode-collapse confound)}.

**H-4 Outcome labels:** validity = verifier (unit tests / symbolic check); novelty = CreativityPrism-validated judge + n-gram originality (2504.09389 protocol) + small human-rated subsample.

**H-5 Analyses (pre-registered):** (1) inverted-U test of P(novel∧valid) vs D-band; (2) predictor comparison D(t) vs baselines (AUROC / Spearman); (3) router simulation: yield-per-verifier-cost for LTDM band-keep vs SpecRej partial-reward pruning vs verify-everything.

**H-6 Exit criteria:** any of the three analyses positive at pre-registered thresholds → write-up (workshop paper). All flat/negative → clean negative preprint. Both outcomes close the loop.

---

## 5. Risks carried forward

- **R1:** Basins shows latent geometry weakens on open-ended tasks → C2 mitigates but does not eliminate; the fertile band may be task-narrow.
- **R2:** Intel's static projection may already capture most of the harvestable signal; if D(t) only matches it, the contribution shrinks to the router economics (D3 alone).
- **R3:** This sweep + reads still aren't a citation-graph pass; one Connected-Papers/Semantic-Scholar sweep over 2412.06060, 2503.02851, 2604.04743 remains pending before any public preprint claim (carried from Phase 0 §7.3).
- **R4:** New neighbors are appearing monthly (3 surfaced *during* this Phase R); re-run the sweep immediately before submission.
