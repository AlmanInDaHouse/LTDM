"""Hermetic fixtures: tiny Llama built FROM CONFIG - no HF hub, no tokenizer, no network.

Deviation from PLAN-LTDM-H §3 (hub tiny-random model) recorded as SPEC amendment A1:
offline-deterministic CI. The monitor is shape-agnostic, so L=2, d=16 exercises the
exact same code paths as the real 1B.
"""
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

TINY = dict(
    hidden_size=16, intermediate_size=32, num_hidden_layers=2,
    num_attention_heads=2, num_key_value_heads=2, vocab_size=128,
    max_position_embeddings=64,
)


@pytest.fixture(scope="session")
def tiny_model():
    torch.manual_seed(0)
    cfg = LlamaConfig(**TINY)
    model = LlamaForCausalLM(cfg).eval()
    model.generation_config.pad_token_id = 0
    return model


@pytest.fixture()
def dummy_ids(tiny_model):
    torch.manual_seed(1)
    return torch.randint(0, tiny_model.config.vocab_size, (1, 8))


@pytest.fixture()
def synthetic_ref_tensors(tiny_model):
    """(mu zeros [L, d] fp32, a_dir one-hot rows [L, d] fp32) - pure tensors,
    enough for AC-1/AC-2 without ReferenceBank.build_mu."""
    L, d = tiny_model.config.num_hidden_layers, tiny_model.config.hidden_size
    mu = torch.zeros(L, d, dtype=torch.float32)
    a = torch.zeros(L, d, dtype=torch.float32)
    a[:, 0] = 1.0
    return mu, a
