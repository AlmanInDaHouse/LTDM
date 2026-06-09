"""Accelerator abstraction (implemented infra - not part of the AC RED surface).

PyTorch's ROCm backend implements the torch.cuda namespace (HIP masquerades as
CUDA): torch.cuda.is_available(), torch.cuda.memory_allocated() and
torch.cuda.synchronize() all work on ROCm builds; torch.version.hip is set
instead of torch.version.cuda. Harness code MUST route through these helpers -
never branch on vendor strings.
"""
import torch


def backend() -> str:
    if torch.cuda.is_available():
        return "rocm" if getattr(torch.version, "hip", None) else "cuda"
    return "cpu"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accel_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def accel_mem_allocated() -> int:
    return torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
