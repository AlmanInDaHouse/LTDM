"""Unit tests for the pure metric functions (SPEC §3). Final form - S-H2 must not edit."""
import torch

from ltdm import metrics


def test_d_basin_matches_hand_computed():
    h = torch.tensor([[3.0, 4.0], [6.0, 8.0]])
    mu = torch.zeros(2, 2)
    out = metrics.d_basin(h, mu)
    assert out.dtype == torch.float32
    assert torch.allclose(out, torch.tensor([5.0, 10.0]))


def test_d_basin_accepts_fp16_input():
    h = torch.tensor([[3.0, 4.0]], dtype=torch.float16)
    out = metrics.d_basin(h, torch.zeros(1, 2))
    assert out.dtype == torch.float32 and torch.allclose(out, torch.tensor([5.0]), atol=1e-2)


def test_kappa_local_values_and_shape():
    d = torch.tensor([2.0, 4.0, 8.0])
    k = metrics.kappa_local(d, eps=0.0)
    assert k.shape == (2,) and torch.allclose(k, torch.tensor([2.0, 2.0]))


def test_cos_dir_unit_vectors():
    h = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    c = metrics.cos_dir(h, a)
    assert torch.allclose(c, torch.tensor([1.0, 0.0]), atol=1e-6)
