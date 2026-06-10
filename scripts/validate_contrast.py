#!/usr/bin/env python3
"""Structural validator for the contrast set (docs/CONTRAST-PROTOCOL.md).

Enforces the auto-checkable rules: schema, unique ids, >=100 pairs, length parity
(+-25%, whitespace word-count proxy), explicit-negator cap (<=2 per negated), and
exact-duplicate detection. Topic-distinctness and minimal-pair faithfulness stay human.

Usage:
    python scripts/validate_contrast.py [--path data/prompts/contrast_pairs_v1.jsonl]
                                        [--sample 15] [--seed 0]
Exit 0 if all structural checks pass; 1 otherwise.
"""
import argparse
import json
import re
from random import Random

NEGATORS = {"not", "without", "no", "never", "none", "nor"}
# Routine-affirming surface lexicon. Rule 3b: no single marker may saturate the negated
# side, or a_dir partly encodes that word instead of creative-vs-routine (H1, S-H3 gate).
ROUTINE_MARKERS = {
    "ordinary", "routine", "usual", "standard", "plain", "conventional", "typical",
    "unremarkable", "everyday", "customary", "predictable", "scheduled", "quiet",
    "commonplace", "regular", "uneventful", "normal", "normally", "fixed",
}
PARITY = 1.25          # longer / shorter must be <= this
MIN_PAIRS = 100
MAX_NEGATORS = 2
SATURATION = 0.25      # rule 3b: no single routine marker in > this fraction of negated


def words(s: str) -> list[str]:
    return s.split()


def negator_count(s: str) -> int:
    toks = re.findall(r"[a-z']+", s.lower())
    return sum(t in NEGATORS for t in toks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default="data/prompts/contrast_pairs_v1.jsonl")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.path, encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()]

    pairs, errors = [], []
    seen_ids, seen_creative, seen_negated = set(), set(), set()
    marker_counts = {m: 0 for m in ROUTINE_MARKERS}
    covered = 0
    for i, ln in enumerate(lines, 1):
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError as e:
            errors.append(f"line {i}: invalid JSON ({e})")
            continue
        if not {"id", "creative", "negated"} <= obj.keys():
            errors.append(f"line {i}: missing id/creative/negated")
            continue
        pid, cre, neg = obj["id"], obj["creative"], obj["negated"]
        if pid in seen_ids:
            errors.append(f"{pid}: duplicate id")
        if cre in seen_creative:
            errors.append(f"{pid}: duplicate creative string")
        if neg in seen_negated:
            errors.append(f"{pid}: duplicate negated string")
        seen_ids.add(pid)
        seen_creative.add(cre)
        seen_negated.add(neg)

        wc, wn = len(words(cre)), len(words(neg))
        ratio = max(wc, wn) / max(1, min(wc, wn))
        if ratio > PARITY:
            errors.append(f"{pid}: length parity {wc}/{wn} -> ratio {ratio:.2f} > {PARITY}")
        nc = negator_count(neg)
        if nc > MAX_NEGATORS:
            errors.append(f"{pid}: {nc} explicit negators in negated > {MAX_NEGATORS}")
        present = {t for t in re.findall(r"[a-z']+", neg.lower())} & ROUTINE_MARKERS
        for m in present:
            marker_counts[m] += 1
        covered += bool(present)
        pairs.append((pid, cre, neg, wc, wn, nc))

    n = len(pairs)
    for m, cnt in marker_counts.items():
        if n and cnt / n > SATURATION:
            errors.append(f"marker '{m}' in {cnt}/{n} negated ({cnt/n:.0%}) > {SATURATION:.0%} (rule 3b)")

    if len(pairs) < MIN_PAIRS:
        errors.append(f"only {len(pairs)} pairs < required {MIN_PAIRS}")

    ratios = [max(wc, wn) / max(1, min(wc, wn)) for _, _, _, wc, wn, _ in pairs]
    print(f"== contrast validator: {args.path} ==")
    print(f"pairs            : {len(pairs)}")
    print(f"unique ids       : {len(seen_ids)}")
    if ratios:
        print(f"length parity    : max ratio {max(ratios):.2f} (limit {PARITY})")
    print(f"max negators/neg : {max((p[5] for p in pairs), default=0)} (limit {MAX_NEGATORS})")
    if n:
        top = sorted(((c, m) for m, c in marker_counts.items() if c), reverse=True)
        peak = f"{top[0][1]} {top[0][0]}/{n} ({top[0][0]/n:.0%})" if top else "none"
        print(f"routine coverage : {covered}/{n} negated ({covered/n:.0%}) carry a marker")
        print(f"marker peak      : {peak} (rule 3b limit {SATURATION:.0%})")
        print("marker dist      : " + ", ".join(f"{m}={c}" for c, m in top))
    print(f"errors           : {len(errors)}")
    for e in errors:
        print(f"  - {e}")

    if args.sample and pairs:
        idx = list(range(len(pairs)))
        Random(args.seed).shuffle(idx)
        print(f"\n== sample of {min(args.sample, len(pairs))} (seed={args.seed}) ==")
        for j in idx[: args.sample]:
            pid, cre, neg, wc, wn, nc = pairs[j]
            print(f"[{pid}] ({wc}w/{wn}w, neg={nc})")
            print(f"  creative: {cre}")
            print(f"  negated : {neg}")

    print("\nVALIDATE:", "PASS" if not errors else "FAIL")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
