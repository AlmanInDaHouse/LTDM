#!/usr/bin/env python3
"""Build backend-bound ReferenceBank artifacts (SPEC-LTDM-001 A5 + CONTRAST-PROTOCOL.md).

mu    : centroid over data/prompts/uninformative.txt (ReferenceBank.build_mu).
a_dir : per-layer creative-minus-negated contrast direction (ReferenceBank.build_a_dir), built
        once per arm. Canonical = narrative arm ALONE (rule 7) -> refs_{backend}_{dtype}.npz;
        full / ideation / descriptive are diagnostics -> ..._arm-<x>.npz. mu is identical in all
        four (it is independent of the contrast set).

RULE 2 (BLOCKING): before any a_dir is built, parity is re-validated IN TOKENS against the real
tokenizer (<=1.25 per pair, CONTENT tokens, add_special_tokens=False -- BOS is a constant on both
sides). A single violation refuses the build, lists the offenders, and is a session STOP: escalate,
do not edit pairs unilaterally (pattern A3). The report lands in results/data/ either way.

A5: artifacts are backend-bound and must be built in the regime they will be measured in
(Track B = cpu/fp32). dtype follows the backend (A2): rocm/cuda -> float16, cpu -> float32.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, "src")

import torch  # noqa: E402

from ltdm import device as dv  # noqa: E402
from ltdm.references import ReferenceBank  # noqa: E402

# Arm -> inclusive id range (CONTRAST-PROTOCOL rule 7 + arm map). "full" = every pair.
ARMS = {
    "narrative": ("c001", "c053"),
    "ideation": ("c054", "c078"),
    "descriptive": ("c079", "c104"),
}
BUILD_ORDER = ("narrative", "full", "ideation", "descriptive")   # narrative first = canonical
PARITY = 1.25                                                     # rule 2 cap (longer / shorter)


def read_pairs(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def read_contexts(path):
    with open(path, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]


def arm_pairs(pairs, arm):
    if arm == "full":
        return list(pairs)
    lo, hi = ARMS[arm]
    return [p for p in pairs if lo <= p["id"] <= hi]


def token_parity(pairs, tok):
    """Returns (rows, offenders). rows: (id, n_cre, n_neg, ratio) on CONTENT tokens. Ratio is the
    word-parity gate's metric (max/min) re-measured in tokens -- the protocol's mandatory once-off
    re-validation against the official tokenizer before a_dir is built (rule 2)."""
    rows = []
    for p in pairs:
        nc = len(tok(p["creative"], add_special_tokens=False)["input_ids"])
        nn = len(tok(p["negated"], add_special_tokens=False)["input_ids"])
        ratio = max(nc, nn) / max(1, min(nc, nn))
        rows.append((p["id"], nc, nn, ratio))
    offenders = [r for r in rows if r[3] > PARITY]
    return rows, offenders


def blob_sha(path):
    return subprocess.check_output(["git", "hash-object", path]).decode().strip()


def write_parity_report(path, model_id, revision, rows, offenders):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mx = max(rows, key=lambda r: r[3]) if rows else ("-", 0, 0, 0.0)
    lines = [
        "# contrast_v1 token-parity re-validation (CONTRAST-PROTOCOL rule 2)",
        f"# tokenizer={model_id}@{revision} add_special_tokens=False cap={PARITY}",
        f"# pairs={len(rows)} max_ratio={mx[3]:.3f}({mx[0]}) offenders={len(offenders)}",
        "# id\tn_creative\tn_negated\tratio",
    ]
    for pid, nc, nn, ratio in rows:
        flag = "\t<-- VIOLATION" if ratio > PARITY else ""
        lines.append(f"{pid}\t{nc}\t{nn}\t{ratio:.3f}{flag}")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contrast", default="data/prompts/contrast_pairs_v1.jsonl")
    ap.add_argument("--uninformative", default="data/prompts/uninformative.txt")
    ap.add_argument("--model-id", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--model-revision", default="main")
    ap.add_argument("--n-contexts", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/refs")
    ap.add_argument("--parity-report", default="results/data/contrast_v1_token_parity.txt")
    args = ap.parse_args()

    backend = dv.backend()
    device = dv.get_device()
    dtype_s = "float16" if backend in ("rocm", "cuda") else "float32"
    dtype_t = torch.float16 if backend in ("rocm", "cuda") else torch.float32

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[build_refs] backend={backend} dtype={dtype_s} model={args.model_id}")
    tok = AutoTokenizer.from_pretrained(args.model_id, revision=args.model_revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, revision=args.model_revision,
        torch_dtype=dtype_t, attn_implementation="sdpa").to(device).eval()
    revision = getattr(model.config, "_commit_hash", None) or args.model_revision

    pairs = read_pairs(args.contrast)

    # ---- RULE 2 (BLOCKING) -- token parity over EVERY pair used across arms --------------
    rows, offenders = token_parity(pairs, tok)
    write_parity_report(args.parity_report, args.model_id, revision, rows, offenders)
    mx = max(rows, key=lambda r: r[3])
    print(f"[rule 2] token parity: {len(pairs)} pairs, max ratio {mx[3]:.3f} ({mx[0]}), "
          f"{len(offenders)} offenders -> {args.parity_report}")
    if offenders:
        print("[rule 2] BLOCKING: token parity violated -> refusing to build a_dir. STOP and "
              "escalate (do not edit pairs unilaterally, pattern A3). Offenders:")
        for pid, nc, nn, ratio in offenders:
            print(f"  {pid}: {nc}/{nn} tokens -> ratio {ratio:.3f} > {PARITY}")
        return 2

    # ---- mu (shared across arms) ---------------------------------------------------------
    ctxs = read_contexts(args.uninformative)[: args.n_contexts]
    print(f"[mu] building over {len(ctxs)} uninformative contexts ...")
    mu = ReferenceBank.build_mu(model, tok, ctxs)

    sha = blob_sha(args.contrast)
    base_meta = dict(backend=backend, dtype=dtype_s, model_id=args.model_id,
                     model_revision=revision, n_contexts=len(ctxs), seed=args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- a_dir per arm -------------------------------------------------------------------
    for arm in BUILD_ORDER:
        ap_pairs = arm_pairs(pairs, arm)
        print(f"[a_dir] arm={arm} ({len(ap_pairs)} pairs) ...")
        a_dir = ReferenceBank.build_a_dir(
            model, tok, [(p["creative"], p["negated"]) for p in ap_pairs])
        meta = {**base_meta, "a_dir_source": f"contrast_v1@{sha}#{arm}"}
        bank = ReferenceBank(mu, a_dir, device, meta=meta)
        name = (f"refs_{backend}_{dtype_s}.npz" if arm == "narrative"
                else f"refs_{backend}_{dtype_s}_arm-{arm}.npz")
        out = os.path.join(args.out_dir, name)
        bank.save(out)
        print(f"  saved {out}")
        print(f"  meta: {json.dumps(meta, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
