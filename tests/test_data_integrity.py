"""Additive data-integrity gate (S-H3). Touches no frozen AC; CI scope (not gpu).

Closes the S-H1 class of bug (engineering-notes lesson #3): a data fixture that no CI
test reads is unverified by construction. These guard the curated/generated prompt files
so a silently-broken fixture (e.g. literal '\\n' instead of real newlines) fails loudly.
"""
import json
import pathlib

PROMPTS = pathlib.Path("data/prompts")


def _real_lines(path):
    with open(path, encoding="utf-8") as fh:
        return [ln.rstrip("\n") for ln in fh if ln.strip() and not ln.startswith("#")]


def test_uninformative_curated():
    lines = _real_lines(PROMPTS / "uninformative.txt")
    assert all("\\n" not in ln for ln in lines), "literal '\\n' found (placeholder not regenerated)"
    assert len(lines) >= 900, f"only {len(lines)} contexts < 900"
    assert len(set(lines)) == len(lines), "duplicate contexts"


def test_informative_pilot_schema():
    for ln in _real_lines(PROMPTS / "informative_pilot.jsonl"):
        obj = json.loads(ln)
        assert "id" in obj and obj.get("prompt"), "missing id/prompt"


def test_contrast_v1_schema():
    pairs = [json.loads(ln) for ln in _real_lines(PROMPTS / "contrast_pairs_v1.jsonl")]
    assert len(pairs) >= 100, f"only {len(pairs)} pairs < 100"
    ids = [p["id"] for p in pairs]
    assert len(set(ids)) == len(ids), "duplicate ids"
    for p in pairs:
        assert p.get("creative") and p.get("negated"), f"{p.get('id')}: empty member"
