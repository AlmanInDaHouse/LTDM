"""Persistence (SPEC-LTDM-001 §5): long-format parquet, one row per (run, t, layer)."""
import torch  # noqa: F401

REQUIRED_META = (
    "run_id", "code_sha", "model_id", "model_revision", "seed",
    "prompt_id", "condition_temperature", "condition_lambda",
)


def write_run(path, export: dict, meta: dict) -> None:
    """Flatten export["M"] [T, L, 3] to long rows + attach REQUIRED_META columns."""
    raise NotImplementedError("S-H2: implement per SPEC-LTDM-001 §5")


def read_run(path):
    """Returns (M [T, L, 3] fp32, meta: dict). Round-trip must be tensor-exact (AC-5)."""
    raise NotImplementedError("S-H2: implement per SPEC-LTDM-001 §5")
