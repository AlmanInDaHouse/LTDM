"""ReferenceBank (SPEC-LTDM-001 §2.2, amendment A5).

mu [L, d] and a_dir [L, d], fp32, accelerator-resident. A5: references are numerical
instruments, built and measured in one backend+dtype regime and never mixed; artifacts are
`refs_{backend}_{dtype}.npz` carrying {backend, dtype, model_id, model_revision, n_contexts,
seed, a_dir_source} embedded. The naming is the label; `load`'s 4-field lock is the lock.
"""
import json

import numpy as np
import torch
import torch.nn.functional as F

# A5 (SPEC §8): every artifact embeds these seven fields.
REQUIRED_REF_META = (
    "backend", "dtype", "model_id", "model_revision",
    "n_contexts", "seed", "a_dir_source",
)
# The load lock (owner gate S-H3-run: A5's literal four). These pin the numerical regime +
# model identity -- the CPU-fp32 vs ROCm-fp16 confound A5 targets. n_contexts/seed/a_dir_source
# ride along as provenance (presence, not lock): a mu over 800 vs 1000 contexts lives in the
# SAME space; AC-6 guards its quality separately.
LOCK_FIELDS = ("model_id", "model_revision", "backend", "dtype")


class ReferenceBank:
    """mu:    centroid of last-position states over ~1000 uninformative contexts
              (Basins-style reference state, arXiv 2604.04743 §5.1).
       a_dir: Intel-style creative-vs-negated contrast direction, row-normalized
              (arXiv 2412.06060 §2). Built from the contrast set per CONTRAST-PROTOCOL.md.

    Both tensors are forced to fp32 and onto `device` at construction so the hot path never
    pays a per-step .to() (SPEC §3 anti-pattern 3). `meta` (optional) carries A5 provenance and
    is required only to `save()`.
    """

    def __init__(self, mu: torch.Tensor, a_dir: torch.Tensor, device, meta: dict | None = None):
        self.mu = mu.to(device=device, dtype=torch.float32)
        self.a_dir = F.normalize(a_dir.to(device=device, dtype=torch.float32), dim=-1)
        self.device = self.mu.device
        self.meta = meta

    # ---- offline builders ------------------------------------------------
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

    @staticmethod
    @torch.inference_mode()
    def build_a_dir(model, tok, pairs: list[tuple[str, str]]) -> torch.Tensor:
        """Tokenizing wrapper (A1 pre-tokenized-core pattern). `pairs`: (creative, negated)."""
        enc_pairs = [(tok(cre, return_tensors="pt").to(model.device),
                      tok(neg, return_tensors="pt").to(model.device)) for cre, neg in pairs]
        return ReferenceBank.build_a_dir_from_encodings(model, enc_pairs)

    @staticmethod
    @torch.inference_mode()
    def build_a_dir_from_encodings(model, enc_pairs) -> torch.Tensor:
        """Pre-tokenized core (hermetic-CI path, A1). `enc_pairs`: list of (creative, negated)
        where each member is a mapping with input_ids [1, P].

        Intel protocol adapted (CONTRAST-PROTOCOL.md): capture the last-position state per layer
        for both members with build_mu's convention (hidden_states[l+1]; final slot post-norm,
        A3-coherent), mean each side over the pairs, difference, row-normalize -> a_dir [L, d].
        """
        if not enc_pairs:
            raise ValueError("build_a_dir requires at least one contrast pair")
        L = model.config.num_hidden_layers
        d = model.config.hidden_size
        acc_cre = torch.zeros(L, d, dtype=torch.float64, device=model.device)
        acc_neg = torch.zeros(L, d, dtype=torch.float64, device=model.device)
        for cre, neg in enc_pairs:
            for inputs, acc in ((cre, acc_cre), (neg, acc_neg)):
                hs = model(**inputs, output_hidden_states=True).hidden_states
                for layer in range(L):
                    acc[layer] += hs[layer + 1][0, -1, :].double()
        diff = (acc_cre - acc_neg) / len(enc_pairs)          # mean(creative) - mean(negated)
        return F.normalize(diff.float(), dim=-1)             # per-layer (row) unit direction

    # ---- persistence (A5) ------------------------------------------------
    def save(self, path) -> None:
        """Write `refs_{backend}_{dtype}.npz`: mu, a_dir, and metadata as a JSON string in a
        numpy U-array under key `meta_json` (no pickle surface, no new dependency). Requires the
        full REQUIRED_REF_META on `self.meta` -- a key omitted at save-time fails loudly here,
        not three weeks later as a silent null (the §5 doctrine, applied to references)."""
        if self.meta is None:
            raise ValueError("ReferenceBank.save needs meta (A5 embedded provenance)")
        missing = [k for k in REQUIRED_REF_META if k not in self.meta]
        if missing:
            raise ValueError(
                f"ReferenceBank.save meta missing {missing} (A5 requires {list(REQUIRED_REF_META)})")
        meta_json = json.dumps(self.meta, sort_keys=True)
        np.savez(
            str(path),
            mu=self.mu.detach().cpu().numpy(),
            a_dir=self.a_dir.detach().cpu().numpy(),
            meta_json=np.array(meta_json),
        )

    @classmethod
    def load(cls, path, device, *, model_id, model_revision, backend, dtype):
        """Restore a bank, validating the A5 lock (LOCK_FIELDS) against the embedded metadata.
        Strict, no defaults: a missing locked field is a KeyError, a differing one a ValueError.
        Loads with allow_pickle=False (the npz carries only float arrays + a U-string)."""
        data = np.load(str(path), allow_pickle=False)
        meta = json.loads(data["meta_json"].item())
        expected = {"model_id": model_id, "model_revision": model_revision,
                    "backend": backend, "dtype": dtype}
        for k, want in expected.items():
            got = meta[k]                                    # strict (A5: no null-is-value)
            if got != want:
                raise ValueError(
                    f"ReferenceBank.load mismatch on {k!r}: artifact={got!r} != "
                    f"expected={want!r} (A5 lock; references are backend-bound)")
        return cls._restore(torch.from_numpy(data["mu"]),
                            torch.from_numpy(data["a_dir"]), device, meta)

    @classmethod
    def _restore(cls, mu, a_dir, device, meta):
        """Faithful restore: install the already-final tensors WITHOUT re-normalizing a_dir
        (the stored direction is the instrument; re-running normalize would drift it by ~1e-7).
        This is what makes the round-trip torch.equal-exact, not merely allclose."""
        self = cls.__new__(cls)
        self.mu = mu.to(device=device, dtype=torch.float32)
        self.a_dir = a_dir.to(device=device, dtype=torch.float32)
        self.device = self.mu.device
        self.meta = meta
        return self
