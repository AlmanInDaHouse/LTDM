# SPEC-LTDM-001 — Trajectory Extraction Harness

| Field | Value |
|---|---|
| Status | Proposed (owner gate pending: G-0 hardware) |
| Date | 2026-06-09 |
| Owner | Manuel Grande Écija |
| Depends on | LTDM-phaseR-go-nogo-memo v1.0 (D1+D3 GO, conditions C1–C5) |
| Scope | Phase H-1/H-2 only: intercept latent trajectories during generation, compute the D(t) metric family online, persist reproducibly. |
| Non-goals | NO router, NO steering, NO experiment runs, NO ICR-JSD (deferred toggle), NO batch>1, NO Mahalanobis. The harness must be boringly correct before anything decides anything. |

---

## 1. Objective

Given an open-weights decoder-only LLM generating T tokens, produce — at ≤10% throughput overhead and <2 MB resident monitor VRAM — a per-token, per-layer metrics tensor:

    M ∈ R^{T × L × 3}   with channels  [ d_basin , κ_local , cos_dir ]

plus optional raw trajectory capture (debug subsample only), persisted with full run metadata.

Convention (fixed now, never renegotiated mid-experiment):
- L = `config.num_hidden_layers` decoder layers. Embedding output is **excluded**.
- Hook on layer ℓ ∈ {0..L−1} captures the residual stream **after** that layer ↔ HF `output_hidden_states` index ℓ+1.
- Step t indexes **generated** tokens: flush t corresponds to the hidden state that predicted generated token t (prefill flush = t=0, predicting the first new token). For N generated tokens there are exactly N flushes.

---

## 2. Architecture

```
ltdm/
├── references.py   ReferenceBank      — builds & holds μ [L,d] and a_dir [L,d] (fp32, GPU-resident)
├── monitor.py      TrajectoryMonitor  — forward hooks, step buffer, online metric flush, export
├── metrics.py      pure tensor fns    — d_basin / kappa / cos (unit-testable on CPU)
├── runner.py       run_traced_generation — seeds, generate(), monitor lifecycle, persistence
└── io.py           parquet/npz writer + run-metadata schema
```

Data flow per decode step: layers fire sequentially → each hook writes its layer's last-position vector into a preallocated step buffer `[L, d]` (fp16) → the **final layer's hook** triggers `_flush_step()` → metrics computed on-GPU in fp32 → written into preallocated `M` → buffer reused. One single GPU→CPU transfer at `export()`.

### 2.1 Reference implementation — TrajectoryMonitor

```python
# ltdm/monitor.py
import torch
from contextlib import contextmanager

class TrajectoryMonitor:
    """Intercepts per-layer residual-stream states during generate() and
    computes the D(t) metric family online. batch=1 only (v0.1)."""

    def __init__(self, model, refs, eps: float = 1e-6, debug_raw: bool = False):
        base = getattr(model, "model", model)           # LlamaForCausalLM/Qwen2ForCausalLM -> .model
        self.layers = base.layers
        self.L = model.config.num_hidden_layers
        self.d = model.config.hidden_size
        assert len(self.layers) == self.L
        self.device = next(model.parameters()).device
        self.refs = refs                                # ReferenceBank (validated dtype/device/shape)
        self.eps = eps
        self.debug_raw = debug_raw
        self._handles = []

    # ---- lifecycle -------------------------------------------------------
    @contextmanager
    def attached(self, max_steps: int):
        self._alloc(max_steps)
        try:
            for idx, layer in enumerate(self.layers):
                self._handles.append(layer.register_forward_hook(self._make_hook(idx)))
            yield self
        finally:
            for h in self._handles:
                h.remove()
            self._handles.clear()

    def _alloc(self, max_steps: int):
        dev = self.device
        self._t = 0
        self._step_buf = torch.empty(self.L, self.d, dtype=torch.float16, device=dev)
        self.M = torch.full((max_steps, self.L, 3), float("nan"), dtype=torch.float32, device=dev)
        self._raw = (torch.empty(max_steps, self.L, self.d, dtype=torch.float16, device=dev)
                     if self.debug_raw else None)

    # ---- hooks -----------------------------------------------------------
    def _make_hook(self, idx: int):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output   # [B, S, d]
            self._step_buf[idx] = h[0, -1, :]                        # last position only
            if idx == self.L - 1:
                self._flush_step()
        return hook

    def _flush_step(self):
        t = self._t
        if t >= self.M.shape[0]:
            return                                                   # safety: ignore overflow steps
        h32 = self._step_buf.float()                                 # [L, d] fp32 (cheap upcast, ~0.5 MB)
        diff = h32 - self.refs.mu                                    # [L, d]
        d_basin = torch.linalg.vector_norm(diff, dim=-1)             # [L]
        kappa = d_basin[1:] / (d_basin[:-1] + self.eps)              # [L-1]
        hn = torch.nn.functional.normalize(h32, dim=-1)              # [L, d]
        cos = torch.einsum("ld,ld->l", hn, self.refs.a_dir)          # [L]  (a_dir pre-normalized)
        self.M[t, :, 0] = d_basin
        self.M[t, :-1, 1] = kappa                                    # last layer: NaN by design
        self.M[t, :, 2] = cos
        if self._raw is not None:
            self._raw[t] = self._step_buf
        self._t += 1

    # ---- export ----------------------------------------------------------
    def export(self):
        out = {"M": self.M[: self._t].cpu(), "steps": self._t}       # the ONLY device sync
        if self._raw is not None:
            out["raw"] = self._raw[: self._t].cpu()
        return out
```

### 2.2 ReferenceBank

```python
# ltdm/references.py
import torch

class ReferenceBank:
    """mu:    [L, d] fp32 — centroid of last-position states over ~1000 uninformative
              contexts (Basins-style: 'The', 'Hello', single common tokens, empty-ish prompts).
       a_dir: [L, d] fp32, row-normalized — Intel-style creative-vs-negated contrast direction.
              v0.1 may ship a_dir = zeros (cos channel inert) until the contrast set lands (S-H3)."""

    def __init__(self, mu: torch.Tensor, a_dir: torch.Tensor, device):
        self.mu = mu.to(device=device, dtype=torch.float32)
        self.a_dir = torch.nn.functional.normalize(
            a_dir.to(device=device, dtype=torch.float32), dim=-1)

    @staticmethod
    @torch.inference_mode()
    def build_mu(model, tok, contexts: list[str]) -> torch.Tensor:
        """Offline. output_hidden_states is acceptable here (not in sweeps)."""
        L = model.config.num_hidden_layers
        acc = torch.zeros(L, model.config.hidden_size, dtype=torch.float64, device=model.device)
        for c in contexts:
            ids = tok(c, return_tensors="pt").to(model.device)
            hs = model(**ids, output_hidden_states=True).hidden_states   # tuple len L+1
            for l in range(L):
                acc[l] += hs[l + 1][0, -1, :].double()
        return (acc / len(contexts)).float()
```

### 2.3 Runner

```python
# ltdm/runner.py
import torch

@torch.inference_mode()
def run_traced_generation(model, tok, prompt: str, mon, *, max_new_tokens=256,
                          temperature=1.0, greedy=False, seed=0):
    torch.manual_seed(seed)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with mon.attached(max_steps=max_new_tokens):
        out = model.generate(**ids, max_new_tokens=max_new_tokens,
                             do_sample=not greedy,
                             temperature=temperature, top_p=1.0, top_k=0,
                             pad_token_id=tok.eos_token_id)
    gen_len = out.shape[1] - ids["input_ids"].shape[1]
    return out, gen_len, mon.export()
```

---

## 3. d_basin as tensor operations — performance budget

**The math, literally:** `d_basin = torch.linalg.vector_norm(h_step.float() - mu, dim=-1)`
where `h_step` is `[L, d]` (this step's trajectory across layers) and `mu` is `[L, d]` fp32 resident on GPU. Broadcast subtract + reduction over d. For a future batch arm: `h [B, L, d] − mu [L, d] → norm(dim=-1) → [B, L]`. No `torch.cdist`, no loops.

**Cost accounting (L=32, d=4096, per generated token):**
- Metric math: subtract + square-reduce + normalize + einsum ≈ L·d·6 ≈ **0.8 MFLOPs**.
- A 7B forward pass ≈ 2·7e9 ≈ **14 GFLOPs**. Metric overhead ≈ **0.006%** of compute. Arithmetic is never the bottleneck.
- Monitor resident VRAM: step_buf 256 KB + M (T=256: 96 KB) + μ/a_dir 1 MB ≈ **<2 MB**. Debug raw: 64 MB/gen — debug subsample only.
- The real overhead is **Python hook dispatch** (L callable invocations/token, typically 1–3% wall-clock) and any accidental device sync. Hence AC-3's 10% ceiling is comfortable *if and only if* the anti-patterns below are respected.

**Anti-patterns (each one individually can multiply latency):**
1. **No `.item()`, `.cpu()`, `print`, or logging inside hooks** — forces a CUDA sync per layer per token.
2. **No `torch.cat`/list-append of tensors in the hot path** — preallocate `M`, write in place.
3. **μ and a_dir live on-device in fp32 from init** — never `.to()` per step.
4. **Upcast only the `[L, d]` step slice** to fp32 for accumulation precision; raw storage stays fp16.
5. **Never use `output_hidden_states=True` inside `generate()` for sweeps** — it materializes L+1 full tensors per step in the returned struct. It is reserved for offline μ-building and the AC-2 equivalence test (single forward).
6. **One sync point**: `export()` at end of generation.
7. `torch.inference_mode()` everywhere; `attn_implementation="sdpa"` (default, Windows-safe); flash-attn-2 optional on Linux only.

---

## 4. Acceptance Criteria — extraction must be GREEN before any router code exists

Each AC is a pytest. S-H1 commits them **failing** (RED). CPU-runnable ACs run in CI against `hf-internal-testing/tiny-random-LlamaForCausalLM`; GPU-only ACs are marked `@pytest.mark.gpu`.

**AC-1 — Shape & coverage** *(CI)*
For a traced generation of N tokens: `export()["M"].shape == (N, L, 3)`; no NaN/Inf anywhere except channel κ at layer L−1 (NaN by design); flush count == N (prefill counted as t=0).
`tests/test_extraction.py::test_shape_and_coverage`

**AC-2 — Ground-truth equivalence** *(CI)*
Single forward (no generate) on a fixed prompt with hooks attached AND `output_hidden_states=True`: for every ℓ, the hook-captured vector equals `hidden_states[ℓ+1][0, -1, :]` under `torch.equal` (same tensor object semantics; zero tolerance). This proves we intercept exactly the residual stream the model reports.
`::test_hook_equals_output_hidden_states`

**AC-3 — Throughput & leak budget** *(GPU)*
Greedy, T=256, warmup + median of 5 runs, `torch.cuda.synchronize()` around timing: `tok/s(monitor ON) ≥ 0.90 × tok/s(monitor OFF)`. Across 100 consecutive traced generations: steady-state `torch.cuda.memory_allocated` delta < 50 MB (no leak from handles/buffers).
`::test_overhead_budget`, `::test_no_vram_leak`

**AC-4 — Signal sanity (the metric is not dead)** *(GPU)*
(a) Mean d_basin over informative prompts (TriviaQA-style, n=50) differs from uninformative prompts (n=50) with Cohen's d ≥ 0.8 at the best layer. (b) Per-prompt trajectory dispersion (std of d_basin across 20 samples) at temperature 1.2 strictly exceeds that at 0.2 for ≥70% of prompts. If AC-4 fails, the harness is fine but D1 is in trouble — that is a finding, logged, owner-STOP.
`::test_signal_separates_conditions`

**AC-5 — Persistence round-trip** *(CI)*
Run → parquet (schema §5) → reload → tensor-equality with exported M; metadata (model_id, revision SHA, seed, condition, prompt_id, code git SHA) present and non-null.
`::test_persistence_roundtrip`

**AC-6 — Reference stability** *(GPU)*
Split the 1000 uninformative contexts into halves A/B; per-layer cosine(μ_A[ℓ], μ_B[ℓ]) ≥ 0.99 for all ℓ. Guards against a noisy reference poisoning every downstream number.
`::test_centroid_split_half_stability`

**Definition of Done (house rules):** AC-1/2/5 green in CI, AC-3/4/6 green locally with evidence (timings + plots committed to `results/ac/`), CI ALL GREEN, Known debt zero, `docs/preregistration.md` committed **before** any pilot analysis runs.

---

## 5. Storage schema

Long-format parquet, one row per (run, t, layer):
`run_id, code_sha, model_id, model_revision, seed, prompt_id, condition_temperature, condition_lambda (null v0.1), t, layer, d_basin, kappa, cos_dir`
plus `runs.parquet` (run-level: gen_len, wall_ms, tok_s, monitor_overhead_pct, timestamps). Raw debug tensors: `results/raw/{run_id}.npz`, never in git.

---

## 6. G-0 Hardware gate (owner-STOP — blocks model choice)

| VRAM available | Model | Rationale |
|---|---|---|
| ≥ 16 GB | Qwen2.5-7B(-Instruct) bf16 | Matches ICR's evaluated family; headroom for KV + debug raw |
| 10–12 GB | **Llama-3.2-3B or Qwen2.5-3B fp16** | Basins got clean basins at 1B–3B; smaller-clean beats bigger-quantized |
| ≤ 8 GB | Llama-3.2-1B fp16 | Same rationale; pilot still valid (Basins' best AUROCs were at 1B) |

**Quantized 7B (4-bit) is rejected for v0.1:** quantization perturbs the very hidden states we are measuring; a confound at the instrument level. Revisit only as a robustness arm in Phase E.

## 7. Deferred (explicitly out of v0.1)

ICR-JSD channel (needs attention-logit capture — memory toggle, S-H4+), batch>1, diagonal-Mahalanobis variant of d_basin, steering injection (Phase E), second reference trajectory (greedy-decode reference; pre-registered as analysis variant, computed offline from persisted runs — no harness change needed).

---

## 8. Amendments (ratified at the S-H2 gate, 2026-06-10)

Binding. Where noted, they supersede the conflicting reference snippets in §2.

- **A1 — Pre-tokenized runner.** `run_traced_generation(model, inputs, mon, ...)` takes a pre-tokenized dict (`input_ids`, optional `attention_mask`); `trace_prompt(model, tok, prompt, mon, ...)` is the tokenizing wrapper. Rationale: hermetic CI — the tiny test model is built from `LlamaConfig` (no hub, no tokenizer, no network). `pad_token_id` is left to `model.generation_config`.

- **A2 — Per-backend dtype.** `rocm/cuda → float16`, `cpu → float32` (README install matrix). §6's blanket fp16 holds for the accelerator path only.

- **A3 — Final slot is post-norm (instrument convention).** HF `output_hidden_states[L]` is the state **after** `model.norm` (final RMSNorm), not the raw last decoder-layer output. The §2.1 reference snippet, which hooks only `base.layers`, would fail AC-2 at layer L−1 (verified empirically: `torch.equal` False, `maxabsdiff = 2.255` on the tiny Llama). Resolution: slots 0..L−2 are sourced from the decoder layers, slot L−1 from `base.norm`. This is the only choice that keeps the instrument identical to the reference — `build_mu` also constructs μ[L−1] from `hidden_states[L]` (post-norm), so live measurement and reference live in the same space.

  **Analysis note (BINDING — carries to `preregistration.md` at S-H4):** slot L−1 lives in normalized space while slots 0..L−2 are raw residual stream, so κ[L−2] = d[L−1]/d[L−2] crosses two geometries of different scale. Not a defect (same instrument convention as the Basins/HF literature), but Phase E's primary analysis must treat the final layer and κ[L−2] as a **separate regime** — or exclude them with a sensitivity analysis. Pre-registered, not discovered post-hoc in the pilot.

- **A4 — Step-buffer dtype follows model params.** `_step_buf`/`_raw` use `next(model.parameters()).dtype` (fp16 on GPU → SPEC's <2 MB budget preserved; fp32 on the hermetic CI model → AC-2 `torch.equal` holds). The fp32 upcast inside `_flush_step` remains the numerical source of truth.

### §5 clarification (BINDING) — null is a value, not an absence

Every row carries every `REQUIRED_META` column. `condition_lambda` is `null` in v0.1, but the caller passes `condition_lambda=None` **explicitly**; `write_run` raises on a missing key (`meta[k]`) rather than materializing a phantom null. A key omitted at write-time must fail loudly, not surface three weeks later as a null that silently contaminates analysis.
