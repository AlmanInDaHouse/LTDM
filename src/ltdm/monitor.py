"""TrajectoryMonitor (SPEC-LTDM-001 §2.1).

Forward hooks on every decoder layer; per-step buffer [L, d] fp16; the FINAL
layer's hook triggers _flush_step() computing [d_basin, kappa, cos_dir] into a
preallocated M [T, L, 3] fp32; exactly ONE GPU->CPU sync, in export().

Step convention (SPEC §1): t indexes GENERATED tokens; the prefill forward is
flush t=0 (the state that predicted the first new token). Embedding excluded;
hook on layer l <-> HF output_hidden_states index l+1.

Anti-patterns enforced here (SPEC §3): no .item()/.cpu()/print in hooks; no
torch.cat in the hot path; refs resident fp32; upcast only the [L, d] slice.
batch=1 only in v0.1.
"""
from contextlib import contextmanager

import torch  # noqa: F401  (used by the S-H2 implementation)


class TrajectoryMonitor:
    def __init__(self, model, refs, eps: float = 1e-6, debug_raw: bool = False):
        raise NotImplementedError("S-H2: implement per SPEC-LTDM-001 §2.1")

    @contextmanager
    def attached(self, max_steps: int):
        raise NotImplementedError("S-H2: implement per SPEC-LTDM-001 §2.1")
        yield self  # pragma: no cover

    def export(self) -> dict:
        """Returns {"M": Tensor[steps, L, 3] (cpu), "steps": int, ["raw": ...]}."""
        raise NotImplementedError("S-H2: implement per SPEC-LTDM-001 §2.1")
