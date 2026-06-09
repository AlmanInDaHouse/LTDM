"""Acceptance Criteria AC-1..AC-6 (SPEC-LTDM-001 §4). Final form - GREEN edits src/ only.

CI scope: AC-1, AC-2, AC-5 (hermetic). Owner scope (-m gpu): AC-3, AC-4, AC-6.
"""
import math
import os
import statistics
import time

import pytest
import torch

from ltdm import io as ltdm_io
from ltdm.device import accel_mem_allocated, accel_sync
from ltdm.monitor import TrajectoryMonitor
from ltdm.references import ReferenceBank
from ltdm.runner import run_traced_generation

gpu = pytest.mark.gpu


# ---------------------------------------------------------------- AC-1 (CI)
def test_AC1_shape_and_coverage(tiny_model, dummy_ids, synthetic_ref_tensors):
    mu, a = synthetic_ref_tensors
    refs = ReferenceBank(mu, a, device="cpu")
    mon = TrajectoryMonitor(tiny_model, refs)
    _, gen_len, exp = run_traced_generation(
        tiny_model, {"input_ids": dummy_ids}, mon, max_new_tokens=5, greedy=True)
    M, L = exp["M"], tiny_model.config.num_hidden_layers
    assert exp["steps"] == gen_len and M.shape == (gen_len, L, 3)
    assert not torch.isnan(M[:, :, 0]).any() and not torch.isinf(M).any()
    assert torch.isnan(M[:, L - 1, 1]).all()          # kappa undefined at last layer, by design
    assert not torch.isnan(M[:, : L - 1, 1]).any()
    assert not torch.isnan(M[:, :, 2]).any()


# ---------------------------------------------------------------- AC-2 (CI)
def test_AC2_hook_equals_output_hidden_states(tiny_model, dummy_ids, synthetic_ref_tensors):
    mu, a = synthetic_ref_tensors
    refs = ReferenceBank(mu, a, device="cpu")
    mon = TrajectoryMonitor(tiny_model, refs, debug_raw=True)
    with mon.attached(max_steps=1):
        out = tiny_model(input_ids=dummy_ids, output_hidden_states=True)
    raw = mon.export()["raw"][0]                       # [L, d] captured at the single forward
    for layer in range(tiny_model.config.num_hidden_layers):
        assert torch.equal(raw[layer], out.hidden_states[layer + 1][0, -1, :]), (
            f"hook capture != output_hidden_states at layer {layer}")


# ---------------------------------------------------------------- AC-3 (owner / accelerator)
@gpu
def test_AC3_overhead_budget(real_model_bundle):
    model, tok, refs = real_model_bundle
    ids = tok("The quick brown fox", return_tensors="pt").to(model.device)

    def timed(with_monitor: bool) -> float:
        runs = []
        for _ in range(5):
            accel_sync(); t0 = time.perf_counter()
            if with_monitor:
                mon = TrajectoryMonitor(model, refs)
                run_traced_generation(model, dict(ids), mon, max_new_tokens=256, greedy=True)
            else:
                model.generate(**ids, max_new_tokens=256, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            accel_sync(); runs.append(time.perf_counter() - t0)
        return statistics.median(runs)

    timed(False)                                       # warmup
    base, mon_t = timed(False), timed(True)
    assert mon_t <= base / 0.90, f"overhead {(mon_t/base-1)*100:.1f}% > 10% budget"


@gpu
def test_AC3_no_vram_leak(real_model_bundle):
    model, tok, refs = real_model_bundle
    ids = tok("Hello", return_tensors="pt").to(model.device)
    mon = TrajectoryMonitor(model, refs)
    with mon.attached(max_steps=32):
        run_traced_generation(model, dict(ids), mon, max_new_tokens=32, greedy=True)
    accel_sync(); m0 = accel_mem_allocated()
    for _ in range(100):
        mon = TrajectoryMonitor(model, refs)
        run_traced_generation(model, dict(ids), mon, max_new_tokens=32, greedy=True)
    accel_sync()
    assert accel_mem_allocated() - m0 < 50 * 2**20, "steady-state VRAM grew > 50 MB"


# ---------------------------------------------------------------- AC-4 (owner)
@gpu
def test_AC4_signal_separates_conditions(real_model_bundle, prompt_sets):
    model, tok, refs = real_model_bundle
    informative, uninformative = prompt_sets

    def mean_d(prompts, temperature, n_samples):
        vals = []
        for i, p in enumerate(prompts):
            ids = tok(p, return_tensors="pt").to(model.device)
            for s in range(n_samples):
                mon = TrajectoryMonitor(model, refs)
                _, _, exp = run_traced_generation(
                    model, dict(ids), mon, max_new_tokens=64,
                    temperature=temperature, greedy=False, seed=1000 * i + s)
                vals.append(exp["M"][:, :, 0].mean().item())
        return vals

    inf, uni = mean_d(informative, 0.7, 1), mean_d(uninformative, 0.7, 1)
    pooled = math.sqrt((statistics.pstdev(inf) ** 2 + statistics.pstdev(uni) ** 2) / 2)
    cohens_d = abs(statistics.mean(inf) - statistics.mean(uni)) / max(pooled, 1e-9)
    assert cohens_d >= 0.8, f"AC-4a: Cohen's d {cohens_d:.2f} < 0.8 -> owner-STOP (SPEC §4)"

    wins = 0
    for i, p in enumerate(informative[:10]):
        lo = statistics.pstdev(mean_d([p], 0.2, 8))
        hi = statistics.pstdev(mean_d([p], 1.2, 8))
        wins += hi > lo
    assert wins >= 7, f"AC-4b: dispersion(T=1.2) > dispersion(T=0.2) on only {wins}/10 prompts"


# ---------------------------------------------------------------- AC-5 (CI)
def test_AC5_persistence_roundtrip(tmp_path):
    torch.manual_seed(3)
    export = {"M": torch.randn(4, 2, 3), "steps": 4}
    meta = {k: ("x" if "id" in k or "sha" in k else 0) for k in ltdm_io.REQUIRED_META}
    path = tmp_path / "run.parquet"
    ltdm_io.write_run(path, export, meta)
    M2, meta2 = ltdm_io.read_run(path)
    assert torch.allclose(export["M"], M2)
    assert all(k in meta2 and meta2[k] is not None for k in ltdm_io.REQUIRED_META)


# ---------------------------------------------------------------- AC-6 (owner)
@gpu
def test_AC6_centroid_split_half_stability(real_model_bundle, uninformative_contexts):
    model, tok, _ = real_model_bundle
    half = len(uninformative_contexts) // 2
    mu_a = ReferenceBank.build_mu(model, tok, uninformative_contexts[:half])
    mu_b = ReferenceBank.build_mu(model, tok, uninformative_contexts[half:])
    cos = torch.nn.functional.cosine_similarity(mu_a, mu_b, dim=-1)
    assert (cos >= 0.99).all(), f"min layer cosine {cos.min():.4f} < 0.99 (SPEC AC-6)"


# ------------------------------------------------- owner-machine fixtures (gpu scope)
@pytest.fixture(scope="session")
def real_model_bundle():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ltdm.device import backend, get_device
    model_id = os.environ.get("LTDM_MODEL_ID", "meta-llama/Llama-3.2-1B")
    dtype = torch.float16 if backend() in ("rocm", "cuda") else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation="sdpa").to(get_device()).eval()
    ctxs = _read_lines("data/prompts/uninformative.txt")[:200]
    mu = ReferenceBank.build_mu(model, tok, ctxs)
    a = torch.zeros_like(mu); a[:, 0] = 1.0            # inert direction until S-H3 contrast set
    return model, tok, ReferenceBank(mu, a, get_device())


@pytest.fixture(scope="session")
def uninformative_contexts():
    return _read_lines("data/prompts/uninformative.txt")


@pytest.fixture(scope="session")
def prompt_sets():
    import json
    inf = [json.loads(x)["prompt"] for x in _read_lines("data/prompts/informative_pilot.jsonl")]
    return inf * 10, _read_lines("data/prompts/uninformative.txt")[:50]


def _read_lines(path):
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
