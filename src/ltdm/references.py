"""ReferenceBank (SPEC-LTDM-001 §2.2): mu [L, d] and a_dir [L, d], fp32, GPU-resident."""
import torch


class ReferenceBank:
    """mu:    centroid of last-position states over ~1000 uninformative contexts
              (Basins-style reference state, arXiv 2604.04743 §5.1).
       a_dir: Intel-style creative-vs-negated contrast direction, row-normalized
              (arXiv 2412.06060 §2). May be zeros until the contrast set lands (S-H3).
    """

    def __init__(self, mu: torch.Tensor, a_dir: torch.Tensor, device):
        raise NotImplementedError("S-H2: implement per SPEC-LTDM-001 §2.2")

    @staticmethod
    @torch.inference_mode()
    def build_mu(model, tok, contexts: list[str]) -> torch.Tensor:
        """Offline mu construction. output_hidden_states is acceptable HERE
        (never in sweeps - SPEC §3 anti-pattern 5). fp64 accumulation -> fp32."""
        raise NotImplementedError("S-H2: implement per SPEC-LTDM-001 §2.2")
