"""Hermetic mechanism tests for scripts/build_refs.py + ReferenceBank.build_a_dir (S-H3).

Additive, CI scope (not gpu). The a_dir math runs on the tiny config model (no tokenizer, no
network); the rule-2 token-parity gate runs against a stub tokenizer -- the real-tokenizer
re-validation is Track B, gated by construction (CONTRAST-PROTOCOL rule 2).
"""
import importlib.util
import pathlib

import torch

from ltdm.references import ReferenceBank

_spec = importlib.util.spec_from_file_location(
    "build_refs", pathlib.Path("scripts/build_refs.py"))
build_refs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_refs)


def test_build_a_dir_shape_norms_determinism(tiny_model):
    torch.manual_seed(7)
    vocab = tiny_model.config.vocab_size
    L, d = tiny_model.config.num_hidden_layers, tiny_model.config.hidden_size

    def enc():
        return {"input_ids": torch.randint(0, vocab, (1, 6))}

    pairs = [(enc(), enc()) for _ in range(4)]               # mini-fixture: 4 contrast pairs
    a = ReferenceBank.build_a_dir_from_encodings(tiny_model, pairs)
    assert a.shape == (L, d)
    assert a.dtype == torch.float32
    norms = torch.linalg.vector_norm(a, dim=-1)
    assert torch.allclose(norms, torch.ones(L), atol=1e-5)   # per-layer unit direction
    a2 = ReferenceBank.build_a_dir_from_encodings(tiny_model, pairs)
    assert torch.equal(a, a2)                                # deterministic: no sampling in build


class _StubTok:
    """Token count == whitespace word count -- enough to exercise the parity ratio, hermetic."""

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": list(range(len(text.split())))}


def test_token_parity_flags_only_offenders():
    tok = _StubTok()
    pairs = [
        {"id": "c001", "creative": "a b c d", "negated": "a b c d"},       # 4/4 = 1.00  ok
        {"id": "c002", "creative": "a b c d e", "negated": "a b c d"},     # 5/4 = 1.25  ok (==cap)
        {"id": "c003", "creative": "a b c d e f", "negated": "a b"},       # 6/2 = 3.00  offender
    ]
    rows, offenders = build_refs.token_parity(pairs, tok)
    assert len(rows) == 3
    assert {o[0] for o in offenders} == {"c003"}


def test_arm_pairs_ranges():
    pairs = [{"id": f"c{n:03d}"} for n in range(1, 105)]
    assert len(build_refs.arm_pairs(pairs, "narrative")) == 53      # c001-c053
    assert len(build_refs.arm_pairs(pairs, "ideation")) == 25       # c054-c078
    assert len(build_refs.arm_pairs(pairs, "descriptive")) == 26    # c079-c104
    assert len(build_refs.arm_pairs(pairs, "full")) == 104
