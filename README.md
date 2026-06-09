# ltdm - Latent Trajectory Divergence Monitoring

Extraction harness for the LTDM research program. Implements **SPEC-LTDM-001**.
Status: **S-H1 RED** - all Acceptance Criteria committed failing for the right reason.

## Known CI debt (house rule: declared here, deleted in the closing SHA)

| Debt | Declared | Cleared by |
|---|---|---|
| Full AC suite RED by design (stubs raise NotImplementedError) | S-H1 (this commit) | S-H2 GREEN commit |

## Install (backend matrix - G-0: AMD RX 6600, 8 GB)

torch wheels are backend-specific and intentionally NOT pinned to an index in pyproject:

| Target | Command |
|---|---|
| **ROCm, native Linux (recommended for RX 6600)** | `pip install torch --index-url https://download.pytorch.org/whl/rocm6.4` then `export HSA_OVERRIDE_GFX_VERSION=10.3.0` (gfx1032 -> gfx1030 mapping, community-standard for RDNA2) |
| CPU fallback / CI | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| WSL2 + ROCm | **Officially unsupported for RDNA2** (AMD WSL matrix lists RDNA3/RDNA4 only). If attempted, time-box to 90 min with `python scripts/check_device.py --require rocm` as the judge. |

Then: `pip install -e ".[dev]"`

dtype policy (SPEC amendment A2): `rocm -> float16`, `cpu -> float32`. Never quantize (SPEC §6).

## Verify the device stack (gate G-0b, blocks S-H3)

    python scripts/check_device.py --require rocm   # or: --require any

## Run the suite

    pytest -m "not gpu"        # CI scope (hermetic tiny-Llama, no network, no accelerator)
    pytest -m gpu              # owner machine: real model + accelerator ACs (AC-3/4/6)

PyTorch ROCm note: the ROCm backend implements the `torch.cuda` namespace (HIP masquerades
as CUDA; `torch.version.hip` is set). All harness code goes through `ltdm.device` helpers anyway.

## SPEC amendments (pending owner ack at S-H2 gate)

- **A1**: `run_traced_generation(model, inputs, mon, ...)` takes pre-tokenized inputs
  (dict with `input_ids`); `trace_prompt(model, tok, prompt, mon, ...)` is the convenience
  wrapper. Rationale: hermetic CI - the tiny test model is built from `LlamaConfig`
  (no HF hub, no tokenizer, no network), so CI never flakes on downloads.
- **A2**: per-backend dtype map (see above); SPEC §6 said fp16 - holds for ROCm only.

## Layout

    src/ltdm/        device, metrics, references, monitor, runner, io
    tests/           conftest (hermetic fixtures) + AC suite (test_extraction) + unit tests
    configs/         model.yaml
    data/prompts/    uninformative.txt, informative_pilot.jsonl, contrast_pairs_v0.jsonl
    scripts/         check_device.py (G-0b)
    docs/            SPEC-LTDM-001, PLAN-LTDM-H, Phase 0 sweep, Phase R memo

License: Apache-2.0.
