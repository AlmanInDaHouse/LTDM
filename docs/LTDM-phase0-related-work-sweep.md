# LTDM — Latent Trajectory Divergence Monitoring
## Phase 0: Related-Work Sweep & Delta Analysis

| Field | Value |
|---|---|
| Status | Draft v0.1 — for review |
| Date | 2026-06-09 |
| Owner | Manuel Grande Écija |
| Reviewer | Claude (architect-advisor role) |
| Scope | Targeted literature sweep (not a PRISMA systematic review) |

---

## 0. Executive summary

**Original idea (restated as a hypothesis).** Instead of suppressing hallucination, induce controlled divergence in a generator LLM (G) within a steered region of latent space, while a monitor (M) reads G's hidden-state trajectory — not its output — and routes each generation into *discard / verify / promote* based on a divergence signal. Hypothesis: there exists a "fertile band" of latent trajectory divergence where the density of novel-AND-valid outputs is maximal.

**Verdict of this sweep.** Every individual component is published, several within the last 6–18 months, and three papers are *very close neighbors*. The defensible delta has shrunk relative to the initial assessment, but it has also sharpened: four delta candidates survive (Section 5). None of them can be claimed publicly until the five nearest-neighbor papers (Section 7.3) are read in full text — this sweep worked at abstract/snippet level.

**Recommended positioning.** Not a framework manifesto. A workshop-paper-sized empirical study: *"Latent trajectory divergence as a control and harvesting signal for valuable novelty."*

---

## 1. Method and limitations of this sweep

- **Method:** targeted web/arXiv search, June 2026, six thematic blocks, ~30 primary sources triaged from snippets and abstracts.
- **Limitation 1 — depth:** abstract-level reading. Claims about what the nearest neighbors *did not* do are provisional until full-text reads.
- **Limitation 2 — coverage:** single-pass keyword search. No citation-graph chasing (Semantic Scholar / Connected Papers), no non-arXiv venue sweep. New preprints appear weekly in this area (multiple 2026 hits below).
- **Consequence:** any novelty claim in a public artifact (preprint, LinkedIn post, talk) is gated on completing the reading list in 7.3 and one citation-graph pass over the three closest neighbors.

---

## 2. Taxonomy: six research lines intersecting LTDM

### A. The hallucination–creativity trade-off (the "let it hallucinate" premise)

The premise that hallucination and creativity share machinery is now an active, named research line — not a fringe intuition.

- **Does Less Hallucination Mean Less Creativity?** (arXiv 2512.11509, Dec 2025). Applies three hallucination-reduction techniques (CoVe, DoLa, RAG) across LLaMA/Qwen/Mistral, 1B–70B, and measures convergent + divergent creativity before/after. Finds that aggressive hallucination control yields models that are precise but constrained in imagination, and frames this as inherent to LLMs, not an artifact of scale.
- **Hallucinating LLM Could Be Creative** (OpenReview, ICLR 2025 submission). Proposes metrics for "good hallucinations" — correctness, consistency, reasoning diversity — and reports that LLMs can produce creative hallucinations with minimal factual damage.
- **Heaven-Sent or Hell-Bent?** (arXiv 2512.21635, Dec 2025). A benchmark explicitly calibrated for *hallucination-driven creativity*, extending TTCT/Guilford dimensions (originality, elaboration, fluency, flexibility) to hallucinated content.
- **Shakespearean Sparks** (arXiv 2503.02851). Studies the "dance" of hallucination and creativity at the level of *decoding layers*; observes that stronger models are simultaneously more creative and more hallucination-prone. **Nearest-neighbor alert:** layer-level analysis of exactly the creativity↔hallucination link. Full-text read mandatory.
- **Advancing the Scientific Method with LLMs** (arXiv 2505.16477). Review that explicitly asks whether hallucination is bug or feature, and sketches the architecture of "hallucination as a stream of novel conjectures + a second LLM as filter" — i.e., the two-instance WhatsApp idea, at concept level.
- **FiSTECH** (arXiv 2408.05365). Distinguishes a *divergent-phase* hallucination (creativity-inducing, controllable) from a *convergent-phase* hallucination (factual failure, RAG-controllable).

**Takeaway:** "hallucination can be harvested for creativity" is taken as a thesis. What remains open is *which signal* identifies harvestable divergence, and *where* (latent vs output).

### B. Internal-state hallucination detection: from static probes to trajectory dynamics (the "monitor" premise)

- **Semantic Entropy Probes** (arXiv 2406.15927, Kossen et al. 2024). Linear probes on hidden states capture semantic uncertainty before any token is generated; builds on the semantic-entropy line (Farquhar et al., Nature 2024). Establishes that hidden states encode hallucination-relevant signal cheaply.
- **LLM-Check** (NeurIPS 2024). Eigen-analysis of internal representations and attention shows consistent latent-space signatures separating hallucinated from grounded responses.
- **ICR Probe** (arXiv 2507.16488, ACL 2025). Shifts from static representations to the *update process across layers* (Information Contribution to Residual stream), aggregating cross-layer dynamics to detect hallucination with few parameters. **This is trajectory-dynamics monitoring, named and published.**
- **MultiHaluDet** (arXiv 2605.24919, May 2026). Probes *full hidden-state trajectories* of frozen LLMs, multi-scale, multilingual.
- **Hallucination Basins** (arXiv 2604.04743, Apr 2026). Links hallucination probability to *geometric properties of hidden trajectories* — distances, volumes, curvature — modeling how the model "runs away" into a hallucination basin. Notes that prior probe methods flag ex post but don't connect to trajectory geometry. **Nearest-neighbor alert:** this is the geometry of latent trajectories applied to hallucination. Full-text read mandatory.
- **HSAD / MHAD family** (2024–2025). Temporal/frequency-domain analysis of the generation trajectory for real-time detection; plus ACL 2025 work on token-by-token monitoring that *alters the decoding trajectory mid-generation* when imminent hallucination is detected (preventative decoding).
- **Grey literature signal:** a March 2026 open project ("latent trajectory collapse" detection, <1 ms on a 4 GB GPU) shows practitioner-level commoditization of trajectory monitoring.

**Takeaway:** "monitor the latent trajectory of a model" is fully taken — *for suppression and detection*. The sign-inversion (monitor to harvest, not to suppress) is the live question.

### C. Activation steering / representation engineering (the "marked vector field")

- **Foundations:** ActAdd (Turner et al. 2023, arXiv 2308.10248), ITI (Li et al. 2023, arXiv 2306.03341), CAA (Rimsky et al., arXiv 2312.06681), RepE (Zou et al. 2023, arXiv 2310.01405), mean-centring improvements (arXiv 2312.03813). Concepts are (approximately) linear directions in activation space; adding scaled vectors at inference steers behavior without retraining.
- **Steering LLMs to Evaluate and Amplify Creativity** (Intel Labs, arXiv 2412.06060). Extracts the internal-state difference between "respond boringly" and "respond creatively," uses it (a) as a creativity *measure* that correlates strongly with human judgment and (b) as a steering vector to *amplify* creativity at inference. **Nearest-neighbor alert: this is the single closest paper to LTDM's steering+scoring loop.** Full-text read mandatory. Key open question for the delta: their measure is a *static projection on a direction*; LTDM proposes *trajectory dynamics* (divergence over layers/tokens) as the signal, jointly predicting novelty AND validity.
- **BILLY** (arXiv 2510.10157). Blends persona vectors in activation space to boost creative generation in a single model, replacing multi-LLM collaboration — steering-for-creativity is an active subfield.
- **Activation Steering with a Feedback Controller** (arXiv 2510.04309). Closed-loop control of steering strength at inference. Neighbor of LTDM's "divergence budget controller" — but aimed at alignment/quality control, not novelty harvesting.

**Takeaway:** "marked vector field" exists (steering), and even "steer + score creativity from internals" exists (Intel). The trajectory-dynamics framing and the joint novelty×validity objective are where daylight may remain.

### D. Generate-and-verify discovery systems (the "two instances" loop, output level)

- **FunSearch** (Romera-Paredes et al., *Nature* 625, 2024). Explicitly pairs a frozen LLM "whose goal is to provide creative solutions" with a systematic evaluator that "guards against confabulations," producing genuinely new mathematics (cap set problem). This is the canonical proof that *delirium + verifier ⇒ discovery*.
- **AlphaEvolve** (Novikov et al., arXiv 2506.13131, 2025). Generalizes to evolving entire codebases with LLM ensembles + evaluator pools + MAP-Elites-style archives; SOTA results across math and infrastructure. Descendants: EoH, HeurAgenix, ThetaEvolve, contrastive concept-tree search (arXiv 2602.03132, 2026).
- **AI co-scientist** (Google, 2025). Multi-agent hypothesis generator / critic / reviewer roles for scientific ideation.
- **Hypothesis-generation line:** SCIMON (literature-driven novelty optimization), LiveIdeaBench (scientific divergent thinking from minimal context), knowledge-graph idea generation (Gu & Krenn), plus the older conceptual anchor: Swanson's literature-based discovery (1986) — undiscovered links between concept X and concept Y.

**Takeaway:** harvesting "hallucination" for discovery is industrially proven — *at the output level, by brute-force verification of every sample*. None of these systems read the generator's internals. This is the gap D3 targets.

### E. Decoding-level divergence control (the cheap baseline LTDM must beat)

- **Measuring LLM Novelty as the Frontier of Original and High-Quality Output** (arXiv 2504.09389). **Already demonstrates an inverted-U: raising temperature first raises novelty (n-gram originality up), then collapses it as quality deteriorates; the optimum varies by task.** The fertile-band intuition is therefore *confirmed at the sampling level*. LTDM cannot claim the inverted-U as such — only (potentially) its latent-trajectory version and superiority as a predictor/controller.
- **Is Temperature the Creativity Parameter?** (arXiv 2405.00492). Pushes back: temperature↔novelty correlation is weak; coherence trade-off moderate. Good epistemic counterweight.
- **min-p sampling** (arXiv 2407.01082) and **Selective Sampling** (arXiv 2510.01218). The latter trains a lightweight *risk classifier* that switches between greedy and high-temperature sampling per token position — i.e., an adaptive divergence budget, implemented at output/logit level. Direct neighbor of LTDM's router, one level up the stack.
- **DoLa** (Chuang et al., arXiv 2309.03883). Contrasts early vs late layers at decoding for factuality — uses cross-layer signal, in the suppression direction.

**Takeaway:** any LTDM experiment must include temperature/min-p/selective-sampling baselines. If latent divergence does not predict or control valuable novelty *better than these cheap knobs*, the result is negative (which is still a publishable result if cleanly executed).

### F. Creativity measurement (the metrics layer for any experiment)

- **CreativityPrism** (arXiv 2510.20091). Decomposes creativity into quality / novelty / diversity, nine tasks, validated automatic judges. Strong candidate as primary evaluation harness.
- **MUTATE** (arXiv 2605.28465, 2026). Interactive, agentic divergent thinking (path-level and action-level).
- **TTCT/Guilford adaptations** (arXiv 2509.09702 and others): fluency, flexibility, originality, elaboration; Alternate Uses Test; Forward Flow (semantic divergence of a thought chain from a fixed start — conceptually a *semantic-level* trajectory-divergence metric, useful as another baseline).
- **Creative homogeneity / mode collapse** (arXiv 2501.19361): LLM outputs cluster; relevant confound when measuring novelty.
- **Nature Comms 2026** (s41467-026-70245-1): divergent thinking for scientific idea generation with minimal context; surveys SCIMON, knowledge-graph ideation; useful for task selection.

### G. Oversight architectures: one model monitoring another

- **CoT Monitorability** (Korbak et al., arXiv 2507.11473, 2025; 40+ cross-lab authors). A monitor reads another model's reasoning trace and flags/blocks/escalates. Two facts matter for LTDM: (1) monitor-routes-generator is an established *architecture pattern*; (2) the community explicitly flags **white-box (activation-level) monitoring as an underdeveloped, called-for direction** — current monitors read text, and latent-reasoning architectures threaten that channel.
- **Behavior-labeled CoT + RepE monitoring** (arXiv 2510.18154). Motivates activation-space monitoring precisely because textual traces can be unfaithful.

**Takeaway:** M-monitors-G is taken as a safety pattern at text level. Reading G's *activations* for an *economic/creative* objective (harvest routing) rather than a safety objective is an inversion of purpose with little visible occupancy — but verify against 2026 safety-tooling literature before claiming.

---

## 3. Key papers table (triage: 22 entries)

| # | Paper | Year | Line | One-line contribution | Relation to LTDM |
|---|---|---|---|---|---|
| 1 | Does Less Hallucination Mean Less Creativity? (2512.11509) | 2025 | A | Hallucination-reduction methods suppress creativity across families/scales | Motivates premise; cite in intro |
| 2 | Hallucinating LLM Could Be Creative (OpenReview) | 2024 | A | Metrics for "good hallucinations" | Prior art on harvest framing |
| 3 | Heaven-Sent or Hell-Bent? (2512.21635) | 2025 | A/F | Benchmark for hallucination-driven creativity | Candidate eval harness |
| 4 | Shakespearean Sparks (2503.02851) | 2025 | A/B | Creativity↔hallucination analyzed at decoding layers | **Nearest neighbor — read full** |
| 5 | Advancing the Scientific Method with LLMs (2505.16477) | 2025 | A/D | Hallucination as conjecture stream + LLM filter (concept) | Concept-level prior art for two-instance loop |
| 6 | FiSTECH (2408.05365) | 2024 | A | Divergent vs convergent phase hallucination | Terminology anchor |
| 7 | Semantic Entropy Probes (2406.15927) | 2024 | B | Hidden states encode semantic uncertainty pre-generation | Probe baseline |
| 8 | LLM-Check (NeurIPS 2024) | 2024 | B | Eigen-signatures of hallucination in internals | Probe baseline |
| 9 | ICR Probe (2507.16488) | 2025 | B | Cross-layer hidden-state *dynamics* detect hallucination | **Nearest neighbor — read full** |
| 10 | Hallucination Basins (2604.04743) | 2026 | B | Trajectory geometry (curvature, distance) bounds hallucination risk | **Nearest neighbor — read full** |
| 11 | MultiHaluDet (2605.24919) | 2026 | B | Full hidden-state trajectory probing, multilingual | Confirms trajectory framing is occupied (suppression) |
| 12 | RepE (2310.01405) / ActAdd (2308.10248) / CAA (2312.06681) / ITI (2306.03341) | 2023 | C | Linear concept directions; inference-time steering | Tooling for "marked vector field" |
| 13 | Steering LLMs to Evaluate and Amplify Creativity (2412.06060, Intel) | 2024 | C | Creativity direction in activations: scores AND amplifies creativity | **Closest single neighbor — read full** |
| 14 | BILLY (2510.10157) | 2025 | C | Persona-vector blending for creative generation | Steering-for-creativity is active |
| 15 | Activation Steering w/ Feedback Controller (2510.04309) | 2025 | C | Closed-loop steering strength control | Neighbor of divergence-budget controller |
| 16 | FunSearch (Nature 625) | 2024 | D | Creative LLM + evaluator ⇒ new mathematics | Output-level harvest, brute-force verify |
| 17 | AlphaEvolve (2506.13131) | 2025 | D | Evolutionary LLM discovery at codebase scale | Output-level harvest, industrial proof |
| 18 | Novelty as Frontier of Original+Quality (2504.09389) | 2025 | E | **Inverted-U of novelty vs temperature already shown (output level)** | LTDM must beat this baseline & reframe claim |
| 19 | Selective Sampling (2510.01218) | 2025 | E | Per-token risk classifier switches sampling regime | Output-level adaptive divergence budget |
| 20 | min-p (2407.01082) / DoLa (2309.03883) / Is Temperature the Creativity Parameter? (2405.00492) | 2024 | E | Decoding knobs and their limits | Mandatory baselines |
| 21 | CreativityPrism (2510.20091) / MUTATE (2605.28465) | 2025–26 | F | Validated multi-dimensional creativity evaluation | Primary metrics harness |
| 22 | CoT Monitorability (2507.11473) | 2025 | G | Monitor-reads-generator pattern; calls for white-box monitors | Architecture precedent; gap statement quotable |

---

## 4. Claims already taken (do not claim these anywhere)

1. "Hidden-state / latent trajectories can detect hallucination" — ICR Probe, MultiHaluDet, Hallucination Basins, HSAD, SEP, LLM-Check.
2. "Hallucination can be creative; good hallucinations are measurable" — OpenReview ICLR'25 sub; Heaven-Sent or Hell-Bent.
3. "Suppressing hallucination suppresses creativity" — 2512.11509.
4. "A creativity direction exists in activation space and can both score and amplify creativity" — Intel 2412.06060.
5. "Creativity and hallucination are linked at the layer level during decoding" — Shakespearean Sparks (pending full-text confirmation of exact claims).
6. "Novelty vs sampling divergence follows an inverted-U (output level)" — 2504.09389.
7. "An adaptive controller can modulate divergence during generation" — Selective Sampling (logit level), Feedback-Controller steering (activation level, alignment objective).
8. "Letting an LLM 'hallucinate' and filtering with a verifier produces genuine discoveries" — FunSearch, AlphaEvolve.
9. "One model can monitor another's reasoning and route outcomes" — CoT monitorability program (text level).

---

## 5. Delta candidates, ranked

**D1 — The fertile band at the *latent trajectory* level (primary).**
Claim: latent trajectory divergence D(t) of a steered generator predicts the *joint* outcome novelty×validity, exhibits an inverted-U, and outperforms (a) temperature, (b) output-level diversity metrics, (c) static creativity-direction projection (Intel) as a predictor.
Falsification: if AUROC/correlation of D(t) ≤ best baseline, D1 is dead.
Novelty risk: **medium-high** — hinges entirely on what papers #4, #10, #13 already measured. Cannot be settled from abstracts.
Feasibility: high (open-weights 7–8B, PyTorch hooks, existing benchmarks).

**D2 — Harvest router with divergence budget (engineering contribution).**
Claim: a closed loop where M classifies generations into discard/verify/promote bands and adaptively modulates steering strength mid-generation increases yield of novel-valid outputs per compute unit.
Differentiation required from: Selective Sampling (logit-level, error-avoidance objective) and Feedback-Controller steering (alignment objective). LTDM's objective is *novelty harvest*, i.e., controlling **toward** a target divergence band, not away from risk.
Novelty risk: medium. Feasibility: medium (control loop on top of D1's instrumentation).

**D3 — Latent monitoring as verifier economics (practical angle).**
Claim: trajectory monitoring rejects doomed delirium *early* (mid-generation), achieving FunSearch-style discovery yield at a fraction of verifier/evaluation cost.
This attacks the explicit weakness of the D-line systems (verify everything, expensive) with the B-line's tooling (cheap internal probes). No source in this sweep combines them.
Novelty risk: **lowest of the four** at sweep level. Feasibility: medium (needs a task with a real, costed verifier — code or math micro-domain).

**D4 — Cross-model white-box monitor for a creative objective (exploratory).**
Claim: M ≠ G, M reads G's activations, purpose is harvesting rather than safety. Inverts the purpose of the white-box monitoring direction that the CoT-monitorability agenda calls for.
Novelty risk: medium (safety tooling in 2026 may already cover mechanics). Feasibility: lower (open-weights only; probe transfer across models is a known pain).

**Ranking (novelty risk ↓ × feasibility ↑ × publishability):** D3 ≥ D1 > D2 > D4.
Recommended scope for a first artifact: **D1 + D3 combined** — "latent trajectory divergence as an early, cheap predictor of valuable novelty" — with D2 as follow-up and D4 parked.

---

## 6. Recommended positioning statement (draft)

> Prior work monitors latent trajectories to *suppress* hallucination, and harvests hallucination for discovery at the *output* level by verifying every sample. We test whether latent trajectory divergence is a usable *control and harvesting* signal: a fertile band of steered divergence, identifiable mid-generation, where outputs are disproportionately novel **and** valid — at a fraction of the verification cost of sample-and-filter pipelines.

The name LTDM can stand as the method label. Avoid: "new field," "super AI," "first to propose hallucination as creativity" (all falsified by Section 4).

---

## 7. Threats, open questions, and gating actions before SPEC

### 7.1 Methodological threats
- **Metric choice is a degree of freedom:** divergence can be cosine distance to a reference trajectory, curvature (Hallucination Basins style), ICR-style update contribution, or projection dynamics on the Intel creativity direction. Pre-register the metric set before running; report all.
- **Judge validity:** LLM-as-judge novelty scoring is biased; use CreativityPrism's validated judges + a small human-rated subset.
- **Construct validity:** "useful novelty" needs a task with a verifiable component. Candidates: MacGyver-style constrained problem solving, LiveIdeaBench keyword ideation with feasibility scoring, or a micro code/math domain (enables D3's cost accounting).
- **Mode collapse confound** (2501.19361): instruct-tuned models homogenize; include a base model arm.

### 7.2 Reproducibility requirements
Open-weights model 7–8B (e.g., Llama-3.1-8B or Qwen2.5-7B), fp16, fixed seeds, hooks on residual stream at all layers, N ≥ 100 prompts × ≥ 20 samples per condition, conditions = {temperature sweep} × {steering strength sweep}. Public repo from day one (this is also the job-search visibility asset).

### 7.3 Mandatory full-text reading list (gate for any novelty claim)
1. Intel — Steering LLMs to Evaluate and Amplify Creativity (2412.06060)
2. Shakespearean Sparks (2503.02851)
3. Hallucination Basins (2604.04743)
4. ICR Probe (2507.16488)
5. Novelty as the Frontier of Original and High-Quality Output (2504.09389)
Plus one citation-graph pass (Semantic Scholar / Connected Papers) over #1–#3.

### 7.4 Open questions for the prototype SPEC
- Reference trajectory definition: greedy decode of same prompt? Mean trajectory over low-temperature samples?
- Token-level vs sequence-level divergence aggregation?
- Where does M live: same-model probe (cheap) first, cross-model later (D4)?
- Verifier for D3's cost claim: unit tests (code) vs symbolic check (math) vs LLM-judge (weakest)?

---

## 8. Next step

**SPEC-LTDM-001 (proposal):** harness-first, RED→GREEN.
1. **Phase R (read):** full-text pass on 7.3 list → one-page delta confirmation memo (go/no-go per delta candidate).
2. **Phase H (harness):** instrumentation + metrics pipeline with a deliberately trivial model assertion failing (RED): hooks capture trajectories, divergence metrics computed, CreativityPrism-style judging wired, baselines (temperature, min-p) running.
3. **Phase E (experiment):** D1+D3 run; pre-registered analysis; inverted-U test + predictor comparison + early-rejection cost accounting.
4. **Phase W (write):** preprint regardless of sign of result. A clean negative ("latent divergence adds nothing over temperature") is publishable and cheap; a positive is a workshop paper.

Decision required from owner: confirm D1+D3 as target scope, or reorder after Phase R.

---

## Appendix A — Source list (as surfaced in this sweep)

- arXiv 2512.11509 — Does Less Hallucination Mean Less Creativity? An Empirical Investigation in LLMs
- OpenReview W48CPXEpXR — Hallucinating LLM Could Be Creative (ICLR 2025 submission)
- arXiv 2512.21635 — Heaven-Sent or Hell-Bent? Benchmarking the Intelligence and Defectiveness of LLM Hallucinations
- arXiv 2503.02851 — Shakespearean Sparks: The Dance of Hallucination and Creativity in LLMs' Decoding Layers
- arXiv 2505.16477 — Advancing the Scientific Method with Large Language Models: From Hypothesis to Discovery
- arXiv 2408.05365 — FiSTECH: Financial Style Transfer to Enhance Creativity without Hallucinations in LLMs
- arXiv 2406.15927 — Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs
- OpenReview LYx4w3CAgy — LLM-Check: Investigating Detection of Hallucinations in Large Language Models (NeurIPS 2024)
- arXiv 2507.16488 — ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs
- arXiv 2604.04743 — Hallucination Basins: A Dynamic Framework for Understanding and Controlling LLM Hallucinations
- arXiv 2605.24919 — MultiHaluDet: Multilingual Hallucination Detection via LLM Hidden State Probing
- arXiv 2310.01405 — Representation Engineering (Zou et al.)
- arXiv 2308.10248 — Activation Addition (Turner et al.)
- arXiv 2312.06681 — Contrastive Activation Addition (Rimsky et al.)
- arXiv 2306.03341 — Inference-Time Intervention (Li et al.)
- arXiv 2312.03813 — Improving Activation Steering with Mean-Centring
- arXiv 2412.06060 — Steering Large Language Models to Evaluate and Amplify Creativity (Intel Labs)
- arXiv 2510.10157 — BILLY: Steering LLMs via Merging Persona Vectors for Creative Generation
- arXiv 2510.04309 — Activation Steering with a Feedback Controller
- Nature 625 (2024) — Mathematical discoveries from program search with large language models (FunSearch, Romera-Paredes et al.)
- arXiv 2506.13131 — AlphaEvolve: A coding agent for scientific and algorithmic discovery (Novikov et al.)
- arXiv 2602.03132 — Contrastive Concept-Tree Search for LLM-Assisted Algorithm Discovery
- arXiv 2511.02864 — Mathematical exploration and discovery at scale
- arXiv 2504.09389 — Measuring LLM Novelty as the Frontier of Original and High-Quality Output
- arXiv 2405.00492 — Is Temperature the Creativity Parameter of Large Language Models?
- arXiv 2407.01082 — Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs
- arXiv 2510.01218 — Control the Temperature: Selective Sampling for Diverse and High-Quality LLM Outputs
- arXiv 2309.03883 — DoLa: Decoding by Contrasting Layers Improves Factuality in LLMs
- arXiv 2510.20091 — CreativityPrism: A Holistic Benchmark for Large Language Model Creativity
- arXiv 2605.28465 — Beyond One Path: Evaluating and Enhancing Divergent Thinking in Interactive LLM Agents (MUTATE)
- arXiv 2509.09702 — Creativity Benchmark (TTCT-derived, marketing domain)
- arXiv 2501.19361 — We're Different, We're the Same: Creative Homogeneity Across LLMs
- Nature Communications (2026) s41467-026-70245-1 — Evaluating LLMs' divergent thinking for scientific idea generation
- arXiv 2507.11473 — Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety (Korbak et al.)
- arXiv 2510.18154 — Annotating the Chain-of-Thought: A Behavior-Labeled Dataset for AI Safety
- Grey literature: HN item 47412822 (Mar 2026) — sub-millisecond "latent trajectory collapse" detector

**Note on IDs:** all arXiv IDs were captured from search-result URLs in this sweep except RepE/ActAdd/CAA/ITI/DoLa (standard, well-known IDs from prior knowledge). Verify IDs during Phase R full-text reads before citing in any public artifact.
