"""Traced generation entry points (SPEC-LTDM-001 §2.3, amendment A1)."""
import torch  # noqa: F401


@torch.inference_mode()
def run_traced_generation(model, inputs: dict, mon, *, max_new_tokens: int = 256,
                          temperature: float = 1.0, greedy: bool = False, seed: int = 0):
    """inputs: pre-tokenized dict with input_ids [1, P] (hermetic-CI path, A1).

    Returns (output_ids, gen_len, mon.export()). Seeds torch; greedy => do_sample=False.
    """
    raise NotImplementedError("S-H2: implement per SPEC-LTDM-001 §2.3 + A1")


def trace_prompt(model, tok, prompt: str, mon, **kw):
    """Convenience wrapper: tokenize then delegate to run_traced_generation."""
    raise NotImplementedError("S-H2: implement per SPEC-LTDM-001 §2.3 + A1")
