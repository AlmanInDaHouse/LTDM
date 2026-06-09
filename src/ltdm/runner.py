"""Traced generation entry points (SPEC-LTDM-001 §2.3, amendment A1)."""
import torch


@torch.inference_mode()
def run_traced_generation(model, inputs: dict, mon, *, max_new_tokens: int = 256,
                          temperature: float = 1.0, greedy: bool = False, seed: int = 0):
    """inputs: pre-tokenized dict with input_ids [1, P] (hermetic-CI path, A1).

    Returns (output_ids, gen_len, mon.export()). Seeds torch; greedy => do_sample=False.
    pad_token_id is left to model.generation_config (A1: no tokenizer in this path).
    """
    torch.manual_seed(seed)
    input_ids = inputs["input_ids"]
    gen_kwargs = {"max_new_tokens": max_new_tokens}
    if greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
    if inputs.get("attention_mask") is not None:
        gen_kwargs["attention_mask"] = inputs["attention_mask"]
    with mon.attached(max_steps=max_new_tokens):
        out = model.generate(input_ids=input_ids, **gen_kwargs)
    gen_len = out.shape[1] - input_ids.shape[1]
    return out, gen_len, mon.export()


def trace_prompt(model, tok, prompt: str, mon, **kw):
    """Convenience wrapper: tokenize then delegate to run_traced_generation."""
    enc = tok(prompt, return_tensors="pt").to(model.device)
    return run_traced_generation(model, dict(enc), mon, **kw)
