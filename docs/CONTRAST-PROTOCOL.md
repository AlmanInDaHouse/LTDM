# CONTRAST-PROTOCOL — building the creativity direction a_dir (LTDM)

Binding protocol for `data/prompts/contrast_pairs_v1.jsonl`. The contrast set defines
`a_dir` (SPEC §2.2): per layer, a direction separating *creative* from *negated* states.
If the negation is sloppy, `a_dir` captures a confound — length, lexical negativity — instead
of creativity, and contaminates the entire `cos_dir` channel downstream. Adapted from the
Intel creativity-direction protocol (arXiv 2412.06060), narrowed to the pilot's task domain.

## Rules (each is enforceable; `scripts/validate_contrast.py` checks the structural ones)

1. **Minimal-pair.** Same scenario and entities in both members. The *only* thing removed is
   the creative tension: the negated member **affirms routine** ("delivers without incident,
   completely uneventful"), it does NOT add boring adjectives or a different scenario.
2. **Length parity, ±25%.** Per pair, |creative| vs |negated| within ±25%. A systematic length
   gap turns `a_dir` into a length detector. **Operational metric: parity is measured in WORDS
   (whitespace split), not tokens** — the validator is CI-hermetic and must not depend on the
   gated pilot tokenizer. Word parity is the binding gate; a tokenizer-based re-validation against
   the pilot tokenizer is an optional Track-B check (word count correlates with token count, so a
   word-parity pass is not expected to flip under tokenization).
3. **Negator cap.** At most 2 explicit negators (`not / without / no / never / none / nor`)
   per negated member. If every negated stacks negative lexicon, the direction learns
   "presence of NO", not creativity. Prefer positive routine markers: "proceeds normally",
   "an ordinary day", "as expected", "routine", "uneventful".
4. **Domain mix.** ~50% narrative (Intel-faithful core), ~25% ideation / what-if,
   ~25% descriptive-explanatory — aligned with the pilot's task set. **Open question, recorded
   here:** the validity of the creativity direction *outside* creative writing is unestablished
   (Intel itself scopes its claim to creative writing); the non-narrative arms are exploratory.
5. **No duplicate topics.** Distinct scenario/entity per pair.
6. **IDs continue v0.** `c001`–`c003` are absorbed verbatim from v0; new pairs run `c004`→.

## Schema

JSONL, one object per line: `{"id": "cNNN", "creative": "...", "negated": "..."}`.

## Review gate (before commit)

- `scripts/validate_contrast.py` run, report attached.
- 15 pairs sampled with a fixed seed → architect reviews against this protocol.
- Owner (Manuel) runs the domain gate on the artifact.

## Provenance into a_dir metadata

When `a_dir` is built from this set, the ReferenceBank artifact records
`a_dir_source = "contrast_v1@<sha>"` (today it is `"inert_e0"`). Both ends of the instrument
carry their provenance.
