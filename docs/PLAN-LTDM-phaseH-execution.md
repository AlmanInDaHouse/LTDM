# PLAN-LTDM-H — Phase H Execution Plan (Real Environment)

| Field | Value |
|---|---|
| Status | v1.0 — ready to execute pending G-0 |
| Date | 2026-06-09 |
| Owner | Manuel Grande Écija |
| Implements | SPEC-LTDM-001 |
| Methodology | Spec-Driven, harness-first RED→GREEN, two-AI workflow (Claude = architect-advisor, Claude Code = executor, owner = gates + GPU runs) |

---

## 0. Gates before the first commit (owner-STOPs)

- **G-0 Hardware:** report GPU model + VRAM → fixes the model row in SPEC §6. Blocks S-H2.
- **G-1 Repo visibility:** repository is **public** from day one. Nothing in LTDM touches client or confidential material; this artifact is simultaneously the research vehicle and the GitHub-visibility asset flagged in the job-search workstream. License: Apache-2.0.
- **G-2 Pre-registration discipline:** `docs/preregistration.md` (metrics, layers, aggregations, thresholds, analysis plan) is committed and SHA-pinned **before** the first pilot analysis. Deviations require an amendment commit, never a silent edit.

## 1. Repository bootstrap

Name: `ltdm` (org: AlmanInDaHouse). Structure:

```
ltdm/
├── src/ltdm/{__init__,monitor,references,metrics,runner,io}.py
├── tests/{test_metrics.py, test_extraction.py, conftest.py}     # conftest: tiny-random model fixture
├── configs/{model.yaml, sweep_pilot.yaml}
├── data/prompts/{uninformative.txt, informative_pilot.jsonl, contrast_pairs_v0.jsonl}
├── docs/{LTDM-phase0-related-work-sweep.md, LTDM-phaseR-go-nogo-memo.md,
│         SPEC-LTDM-001.md, preregistration.md, adr/}
├── results/{ac/, pilot/}        # plots + evidence; raw/ is .gitignored
├── .github/workflows/ci.yml
└── pyproject.toml               # uv-managed
```

The two memos and the SPEC move into `docs/` in the bootstrap commit — repo becomes canonical source of truth from minute one, per house convention.

## 2. Environment

- **OS:** if the local box is the Windows/ETW machine → **WSL2 Ubuntu 24.04** recommended (CUDA-on-WSL is mature; enables flash-attn-2 later; CI parity). Native Windows is acceptable for v0.1 (`attn_implementation="sdpa"`); decide once, record as ADR-LTDM-0001.
- **Stack:** Python 3.12, `uv`; `torch` (CUDA build matching driver), `transformers>=4.46`, `accelerate`, `pyarrow`, `pandas`, `pytest`, `ruff`. Pin in `uv.lock`.
- **Determinism policy:** greedy decoding for all AC tests; sampling runs persist `seed` per run; document that bitwise reproducibility across GPU archs is NOT claimed (only seed-level within-machine).

## 3. CI design (GitHub Actions, CPU-only)

- Fixture model: `hf-internal-testing/tiny-random-LlamaForCausalLM` (L=2, d≈16) — the monitor is architecture-shape-agnostic, so AC-1, AC-2, AC-5 and all `metrics.py` unit tests run on CPU in <2 min.
- GPU-marked tests (AC-3/4/6) are `pytest -m gpu`, excluded in CI, executed locally by owner; their evidence artifacts (timings JSON, plots) are committed to `results/ac/` and linked from the session close note.
- House rule applies: **CI ALL GREEN + Known debt zero** at every session close.

## 4. Session plan (two-AI workflow)

**S-H1 — RED (1 session).** Claude Code: scaffold repo, pyproject, CI, conftest with tiny-random fixture, and **all six AC tests written and failing for the right reason** (NotImplementedError / missing module — not collection errors). Gate: owner verifies red-for-the-right-reason locally + CI runs.

**S-H2 — GREEN core (1–2 sessions).** Implement `metrics.py`, `TrajectoryMonitor`, `ReferenceBank.build_mu`, `runner`, `io`. Exit: AC-1, AC-2, AC-5 green in CI and locally on the G-0 model. Anti-pattern checklist from SPEC §3 reviewed line-by-line in the PR.

**S-H3 — References & instrument validation (1 session + ~1 GPU-h).** Build μ from 1000 uninformative contexts (curate `uninformative.txt`: Basins-style single common tokens / generic stubs; filter tokenizer specials). Build `contrast_pairs_v0.jsonl` (≥100 creative-vs-negated pairs, Intel protocol, adapted to the pilot task domain) → a_dir. Run AC-3, AC-6, then AC-4. Gate: all six AC green, evidence committed. **If AC-4 fails: owner-STOP** — instrument works, signal is weak; decide pivot (layer range, reference variant, model size) before spending pilot compute.

**S-H4 — Mini-pilot (1 session + 2–4 GPU-h).** 20 prompts × temperature {0.2, 0.7, 1.2} × 20 samples, traced; plus ONE toy code task with a real unit-test verifier to smoke-test the yield-per-verifier-cost plumbing end to end (D3's metering, no router yet). Outputs: dispersion plots (d_basin and κ distributions per condition per layer), `pilot-memo.md` with go/no-go for Phase E. Pre-registration committed **before** this session's analysis.

**Gate → SPEC-LTDM-002 (Phase E):** fertile-band experiment + router + SpecRej baseline. Out of scope here by design.

## 5. Effort & risk

- Effort: S-H1→H3 ≈ 3–5 evening sessions; pilot adds 2–4 GPU-h. Calendar: ~2 weeks at current cadence.
- **R-H1:** VRAM below expectation → SPEC §6 row applies; never quantize to compensate.
- **R-H2:** uninformative-context set contaminated by tokenizer artifacts (BOS-only, byte tokens) → AC-6 is the tripwire; curate, re-run.
- **R-H3:** EOS-early generations shrink T unevenly across conditions → persist `gen_len`, analyze per-token-position, never truncate-pad silently.
- **R-H4:** scope creep toward the router before AC green — SPEC §0 non-goals are binding; router code in v0.1 is a session-close blocker.
- **R-H5 (carried):** new neighbor papers monthly → re-run the Phase-0 sweep before any public claim (Phase R memo R3/R4).
