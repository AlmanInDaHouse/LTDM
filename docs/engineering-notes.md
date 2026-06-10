# Engineering notes — institutional lessons

Append-only. One lesson per entry: what bit us, why, and the rule that prevents
the next bite.

## 2026-06-10 — HF `hidden_states[L]` is post-final-norm (S-H2, amendment A3)

HuggingFace `output_hidden_states` applies `model.norm` (the final RMSNorm) to
the last tuple entry. So `hidden_states[L]` is **post-norm**, while a forward hook
on the last decoder layer sees the **pre-norm** residual stream (verified
empirically: `torch.equal` False, `maxabsdiff = 2.255` on the tiny Llama). AC-2
caught it because it asserts `torch.equal(hook, hidden_states[ℓ+1])` for every
layer, L−1 included.

**Rule:** any hidden-states instrument must declare, **per slot**, whether it
captures pre- or post-norm. Never assume "layer output == reported hidden state"
for the last layer.

The close of A3 on the *reference* side — μ[L−1] built from `hs[L]` post-norm,
identical to the slot the monitor captures via `base.norm` — is what turns the
finding from a patch into a coherent convention: an instrument with both ends in
the same space.

## 2026-06-10 — References are backend-bound (S-H3, SPEC amendment A5)

`mu` built in CPU-fp32 and measured against ROCm-fp16 mixes numerical regimes — the same
class of bug as measuring a pre-norm state against a post-norm reference (A3), only subtler.

**Rule:** `mu` and `a_dir` are rebuilt per backend (cheap — minutes); their artifacts carry
`backend+dtype` in the filename and embedded metadata, and `ReferenceBank.load` validates that
metadata against the caller's expected `(model_id, revision, backend, dtype)` and raises on any
mismatch. The naming is the label; the load signature is the lock. Both ends of the instrument
in the same space — now numerically too.

## 2026-06-10 — Honest CPU evidence accounting (S-H3)

On CPU `accel_mem_allocated()` returns 0, so AC-3's no-VRAM-leak assertion passes **vacuously** —
it proves nothing. AC-3 overhead on CPU also under-stresses the 10% budget (a slow forward makes
hook dispatch weigh less in relative terms). Verified before the first datum existed:
`check_device.py` on CPU reported `mem_allocated=0.0 MiB`.

**Rule:** a metric that is structurally inert on a backend is logged as inert in `results/ac/`,
never as green-with-meaning; preliminary CPU numbers are re-run on ROCm if Track A passes.

## 2026-06-10 — An unverified data fixture is broken by construction (S-H3)

S-H1's `uninformative.txt` shipped with literal `\n` instead of newlines: a chain `&&` died at
the `tomllib` check just before the heredoc, so the file was never rewritten, while the two jsonl
that followed a plain newline did run. No CI test read it, so nothing caught it.

**Rule:** a data fixture no CI test reads is unverified by construction — generate it by script
AND guard it with a test, ideally both. **Corollary (structural):** a file does not self-generate
by reading itself — the seed list (`uninformative.seed.txt`, versioned input) is separate from the
generated output (`uninformative.txt`). The additive `test_data_integrity.py` caught this very bug
red *before its own first commit* — the lesson working in real time.

## 2026-06-10 — A per-item validator misses distributional confounds (S-H3, lesson #5)

The S-H3 contrast set passed a green per-pair validator (parity, negator cap) yet carried three
dataset-level confounds invisible to per-item checks: one routine word ("ordinary") saturated ~67%
of the negated side; the ideation arm opened every creative with Imagine/Design and every negated
with Describe; the descriptive arm silently swapped the subject. Each would have leaked into
`a_dir = mean(creative) − mean(negated)`. A per-pair pass cannot see any of them — by construction.

**Rule:** any dataset that defines a *direction* needs distribution checks on top of per-item ones —
surface saturation (no single token in >25% of one side), structural parity across arms (matched
opening-verb distribution on both sides), subject preservation. Encode them in the validator:
what the validator does not check is not checked (the recurring law of this project).
