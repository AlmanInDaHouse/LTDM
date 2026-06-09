"""TrajectoryMonitor (SPEC-LTDM-001 §2.1).

Forward hooks on every decoder layer; per-step buffer [L, d]; the FINAL slot's
hook triggers _flush_step() computing [d_basin, kappa, cos_dir] into a
preallocated M [T, L, 3] fp32; exactly ONE GPU->CPU sync, in export().

Step convention (SPEC §1): t indexes GENERATED tokens; the prefill forward is
flush t=0 (the state that predicted the first new token). Embedding excluded;
hook on layer l <-> HF output_hidden_states index l+1.

Final-layer sourcing (deviation from the §2.1 reference snippet, see PR note):
HF's `output_hidden_states[L]` is the POST-`model.norm` state, not the raw last
decoder-layer output. AC-2 demands torch.equal against output_hidden_states, and
build_mu builds mu[L-1] from that same post-norm state. So slots 0..L-2 come from
the decoder layers and slot L-1 comes from `base.norm`; both reproduce HF's tuple
exactly. The plain layer hook in the §2.1 snippet would mismatch the last layer.

Anti-patterns enforced here (SPEC §3): no .item()/.cpu()/print in hooks; no
torch.cat in the hot path; refs resident fp32; upcast only the [L, d] slice.
batch=1 only in v0.1.
"""
from contextlib import contextmanager

import torch

from ltdm import metrics


class TrajectoryMonitor:
    def __init__(self, model, refs, eps: float = 1e-6, debug_raw: bool = False):
        base = getattr(model, "model", model)            # LlamaForCausalLM/Qwen2ForCausalLM -> .model
        self.layers = base.layers
        self.final_norm = getattr(base, "norm", None)
        self.L = model.config.num_hidden_layers
        self.d = model.config.hidden_size
        assert len(self.layers) == self.L
        assert self.final_norm is not None, "expected base.norm (Llama/Qwen2 final RMSNorm)"
        param = next(model.parameters())
        self.device = param.device
        self._dtype = param.dtype                        # capture in model dtype: fp16 on GPU,
        self.refs = refs                                 # fp32 on the hermetic CI model (AC-2 torch.equal)
        self.eps = eps
        self.debug_raw = debug_raw
        self._handles = []

    # ---- lifecycle -------------------------------------------------------
    @contextmanager
    def attached(self, max_steps: int):
        self._alloc(max_steps)
        try:
            for idx in range(self.L - 1):
                self._handles.append(
                    self.layers[idx].register_forward_hook(self._make_layer_hook(idx)))
            self._handles.append(
                self.final_norm.register_forward_hook(self._make_final_hook()))
            yield self
        finally:
            for h in self._handles:
                h.remove()
            self._handles.clear()

    def _alloc(self, max_steps: int):
        dev = self.device
        self._t = 0
        self._step_buf = torch.empty(self.L, self.d, dtype=self._dtype, device=dev)
        self.M = torch.full((max_steps, self.L, 3), float("nan"), dtype=torch.float32, device=dev)
        self._raw = (torch.empty(max_steps, self.L, self.d, dtype=self._dtype, device=dev)
                     if self.debug_raw else None)

    # ---- hooks -----------------------------------------------------------
    def _make_layer_hook(self, idx: int):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output   # [B, S, d]
            self._step_buf[idx] = h[0, -1, :]                        # last position only
        return hook

    def _make_final_hook(self):
        last = self.L - 1
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output   # post-norm [B, S, d]
            self._step_buf[last] = h[0, -1, :]
            self._flush_step()
        return hook

    def _flush_step(self):
        t = self._t
        if t >= self.M.shape[0]:
            return                                                   # safety: ignore overflow steps
        h32 = self._step_buf.float()                                 # [L, d] fp32 (cheap upcast)
        d = metrics.d_basin(h32, self.refs.mu)                       # [L]
        self.M[t, :, 0] = d
        self.M[t, :-1, 1] = metrics.kappa_local(d, self.eps)         # last layer: NaN by design
        self.M[t, :, 2] = metrics.cos_dir(h32, self.refs.a_dir)      # [L]  (a_dir pre-normalized)
        if self._raw is not None:
            self._raw[t] = self._step_buf
        self._t += 1

    # ---- export ----------------------------------------------------------
    def export(self) -> dict:
        """Returns {"M": Tensor[steps, L, 3] (cpu), "steps": int, ["raw": ...]}."""
        out = {"M": self.M[: self._t].cpu(), "steps": self._t}       # the ONLY device sync
        if self._raw is not None:
            out["raw"] = self._raw[: self._t].cpu()
        return out
