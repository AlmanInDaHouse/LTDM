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
