"""ReferenceBank (SPEC-LTDM-001 §2.2): mu [L, d] and a_dir [L, d], fp32, GPU-resident."""
import torch
import torch.nn.functional as F


class ReferenceBank:
    """mu:    centroid of last-position states over ~1000 uninformative contexts
              (Basins-style reference state, arXiv 2604.04743 §5.1).
       a_dir: Intel-style creative-vs-negated contrast direction, row-normalized
              (arXiv 2412.06060 §2). May be zeros until the contrast set lands (S-H3).

    Both tensors are forced to fp32 and onto `device` at construction so the hot
    path never pays a per-step .to() (SPEC §3 anti-pattern 3).
    """

    def __init__(self, mu: torch.Tensor, a_dir: torch.Tensor, device):
        self.mu = mu.to(device=device, dtype=torch.float32)
        self.a_dir = F.normalize(a_dir.to(device=device, dtype=torch.float32), dim=-1)
        self.device = self.mu.device

    @staticmethod
    @torch.inference_mode()
    def build_mu(model, tok, contexts: list[str]) -> torch.Tensor:
        """Offline mu construction. output_hidden_states is acceptable HERE
        (never in sweeps - SPEC §3 anti-pattern 5). fp64 accumulation -> fp32.

        Uses hidden_states[l+1] for l in 0..L-1, i.e. the exact convention the
        monitor reproduces online (the final layer is the post-norm state HF
        reports). This keeps mu and the live measurement on the same manifold.
        """
        L = model.config.num_hidden_layers
        acc = torch.zeros(L, model.config.hidden_size, dtype=torch.float64, device=model.device)
        for c in contexts:
            ids = tok(c, return_tensors="pt").to(model.device)
            hs = model(**ids, output_hidden_states=True).hidden_states   # tuple len L+1
            for layer in range(L):
                acc[layer] += hs[layer + 1][0, -1, :].double()
        return (acc / len(contexts)).float()
