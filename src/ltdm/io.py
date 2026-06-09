"""Persistence (SPEC-LTDM-001 §5): long-format parquet, one row per (run, t, layer)."""
import pandas as pd
import torch

REQUIRED_META = (
    "run_id", "code_sha", "model_id", "model_revision", "seed",
    "prompt_id", "condition_temperature", "condition_lambda",
)


def write_run(path, export: dict, meta: dict) -> None:
    """Flatten export["M"] [T, L, 3] to long rows + attach REQUIRED_META columns."""
    M = export["M"]
    if isinstance(M, torch.Tensor):
        M = M.detach().to(device="cpu", dtype=torch.float32)
    T, L, _ = M.shape
    rows = M.reshape(T * L, 3).tolist()
    t_col = [t for t in range(T) for _ in range(L)]
    layer_col = [layer for _ in range(T) for layer in range(L)]
    data = {k: [meta[k]] * (T * L) for k in REQUIRED_META}
    data["t"] = t_col
    data["layer"] = layer_col
    data["d_basin"] = [r[0] for r in rows]
    data["kappa"] = [r[1] for r in rows]
    data["cos_dir"] = [r[2] for r in rows]
    pd.DataFrame(data).to_parquet(path)


def read_run(path):
    """Returns (M [T, L, 3] fp32, meta: dict). Round-trip must be tensor-exact (AC-5)."""
    df = pd.read_parquet(path).sort_values(["t", "layer"]).reset_index(drop=True)
    T = int(df["t"].max()) + 1
    L = int(df["layer"].max()) + 1
    arr = df[["d_basin", "kappa", "cos_dir"]].to_numpy()
    M = torch.tensor(arr, dtype=torch.float32).reshape(T, L, 3)
    first = df.iloc[0]
    meta = {k: (first[k].item() if hasattr(first[k], "item") else first[k]) for k in REQUIRED_META}
    return M, meta
