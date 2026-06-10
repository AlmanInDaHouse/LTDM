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
   *Exception (descriptive arm, c079–c104):* metaphor prompts have no routine counterpart that
   keeps the same subject, so they use **frame-preserved, subject-analog pairs** (an abstract /
   emotional subject vs a concrete object in the same syntactic frame). This arm measures something
   looser and is a diagnostic only — never part of the primary direction (see rule 7).
2. **Length parity, ±25%.** Per pair, |creative| vs |negated| within ±25%. A systematic length
   gap turns `a_dir` into a length detector. **Operational metric: parity is measured in WORDS
   (whitespace split), not tokens** — the validator is CI-hermetic and must not depend on the
   gated pilot tokenizer. Word parity is the binding gate. A tokenizer-based re-validation against
   the **official** pilot tokenizer is **mandatory once, before `a_dir` is built**, with evidence
   committed to `results/` — "word count correlates with token count" is exactly the kind of claim
   that gets measured here, not assumed.
3. **Negator cap.** At most 2 explicit negators (`not / without / no / never / none / nor`)
   per negated member. If every negated stacks negative lexicon, the direction learns
   "presence of NO", not creativity. Prefer positive routine markers: "proceeds normally",
   "an ordinary day", "as expected", "routine", "uneventful".
3b. **Surface-saturation cap (enforced).** No single routine marker may appear in >25% of the
   negated members. Rule 3 caps *negators*; this caps individual *routine surfaces*: a token in
   two-thirds of one side and absent from the other injects a consistent component into
   `a_dir = mean(creative) − mean(negated)`, so the direction would partly encode that word (H1,
   S-H3 gate). `validate_contrast.py` checks a routine-marker lexicon against the 25% threshold and
   prints the distribution. Spread markers across the wide palette (routine, usual, standard, plain,
   conventional, typical, unremarkable, everyday, customary, predictable, scheduled, quiet,
   commonplace, regular, uneventful, …).
4. **Domain mix.** ~50% narrative (Intel-faithful core), ~25% ideation / what-if,
   ~25% descriptive-explanatory — aligned with the pilot's task set. **Open question, recorded
   here:** the validity of the creativity direction *outside* creative writing is unestablished
   (Intel itself scopes its claim to creative writing); the non-narrative arms are exploratory.
5. **No duplicate topics.** Distinct scenario/entity per pair.
6. **IDs continue v0.** `c001`–`c003` are absorbed from v0 and brought to this protocol (c003
   parity-fixed; all three de-saturated under rule 3b); new pairs run `c004`→. `contrast_pairs_v0.jsonl`
   is deleted — single source of truth.
7. **Primary direction = narrative arm only.** `a_dir` is built from the narrative arm alone
   (`c001`–`c053`, the true Intel-faithful minimal pairs). The full set and each arm are also built
   as **comparative diagnostics** — that comparison is the cheap experiment that answers rule 4's
   open question: does the direction generalize beyond creative writing? Decide on evidence, not assumption.

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
