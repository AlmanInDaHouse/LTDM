"""Pure tensor metric functions (SPEC-LTDM-001 §3). CPU-unit-testable, no model needed."""
import torch
import torch.nn.functional as F


def d_basin(h_step: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """h_step [L, d] (any float dtype), mu [L, d] fp32 -> [L] fp32.

    SPEC §3: torch.linalg.vector_norm(h_step.float() - mu, dim=-1).
    fp32 accumulation; mu must already be fp32 and device-resident.
    """
    return torch.linalg.vector_norm(h_step.float() - mu, dim=-1)


def kappa_local(d: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """d [L] -> [L-1]; kappa[l] = d[l+1] / (d[l] + eps). SPEC §2.1 _flush_step.

    kappa < 1 means the trajectory is contracting toward the reference (Basins,
    arXiv 2604.04743 Def. C.1).
    """
    return d[1:] / (d[:-1] + eps)


def cos_dir(h_step: torch.Tensor, a_dir: torch.Tensor) -> torch.Tensor:
    """h_step [L, d], a_dir [L, d] row-normalized fp32 -> [L] cosine per layer.

    Intel-style projection on the creativity direction (arXiv 2412.06060 §2),
    generalized to one direction per layer. Only h_step is normalized here:
    a_dir is pre-normalized by ReferenceBank (SPEC §2.1 hot-path convention).
    """
    hn = F.normalize(h_step.float(), dim=-1)
    return torch.einsum("ld,ld->l", hn, a_dir)
