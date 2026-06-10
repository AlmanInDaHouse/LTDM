"""ReferenceBank.save/load persistence (SPEC-LTDM-001 A5). Additive, CI scope (not gpu).

Pure-tensor banks (no model, no tokenizer, no network). Covers: exact mu/a_dir round-trip; the
4-field load lock raising per field on mismatch (owner gate S-H3-run: A5's literal four --
model_id, model_revision, backend, dtype); allow_pickle=False on load; provenance
(a_dir_source, n_contexts, seed) preserved (presence, not locked).
"""
from unittest import mock

import numpy as np
import pytest
import torch

from ltdm.references import ReferenceBank


def _bank(device="cpu"):
    torch.manual_seed(0)
    mu = torch.randn(3, 4)
    a_dir = torch.randn(3, 4)
    meta = dict(backend="cpu", dtype="float32", model_id="meta-llama/Llama-3.2-1B",
                model_revision="abc123", n_contexts=1000, seed=0,
                a_dir_source="contrast_v1@deadbeef#narrative")
    return ReferenceBank(mu, a_dir, device, meta=meta), meta


def _save(tmp_path):
    bank, meta = _bank()
    path = tmp_path / "refs_cpu_float32.npz"
    bank.save(path)
    return bank, meta, path


def _ok_kwargs(meta):
    return dict(model_id=meta["model_id"], model_revision=meta["model_revision"],
                backend=meta["backend"], dtype=meta["dtype"])


def test_roundtrip_mu_a_dir_exact(tmp_path):
    bank, meta, path = _save(tmp_path)
    loaded = ReferenceBank.load(path, "cpu", **_ok_kwargs(meta))
    assert torch.equal(loaded.mu, bank.mu)
    assert torch.equal(loaded.a_dir, bank.a_dir)


def test_provenance_fields_preserved(tmp_path):
    _, meta, path = _save(tmp_path)
    loaded = ReferenceBank.load(path, "cpu", **_ok_kwargs(meta))
    assert loaded.meta["a_dir_source"] == "contrast_v1@deadbeef#narrative"
    assert loaded.meta["n_contexts"] == 1000
    assert loaded.meta["seed"] == 0


@pytest.mark.parametrize("field", ["model_id", "model_revision", "backend", "dtype"])
def test_load_lock_raises_on_mismatch(tmp_path, field):
    _, meta, path = _save(tmp_path)
    kwargs = _ok_kwargs(meta)
    kwargs[field] = "WRONG-VALUE"
    with pytest.raises(ValueError, match=field):
        ReferenceBank.load(path, "cpu", **kwargs)


def test_load_uses_allow_pickle_false(tmp_path):
    _, meta, path = _save(tmp_path)
    with mock.patch("numpy.load", wraps=np.load) as spy:
        ReferenceBank.load(path, "cpu", **_ok_kwargs(meta))
    assert spy.call_args.kwargs.get("allow_pickle") is False


def test_save_requires_full_meta(tmp_path):
    bank, _ = _bank()
    bank.meta = {"backend": "cpu"}                            # incomplete -> must fail loudly
    with pytest.raises(ValueError, match="missing"):
        bank.save(tmp_path / "x.npz")
