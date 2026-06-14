from __future__ import annotations

import torch

from eocrc_dat.underwriters.sparsemax import sparsemax


def test_rows_sum_to_one() -> None:
    z = torch.randn(16, 11)
    out = sparsemax(z)
    assert torch.allclose(out.sum(dim=-1), torch.ones(16), atol=1e-5)


def test_produces_exact_zeros() -> None:
    z = torch.tensor([[3.0, 0.1, -2.0, -5.0]])
    out = sparsemax(z)
    assert (out == 0.0).any()
    assert (out >= 0.0).all()


def test_peaked_input_is_onehot() -> None:
    z = torch.tensor([[20.0, 0.0, 0.0]])
    out = sparsemax(z)
    assert torch.argmax(out).item() == 0
    assert out[0, 0].item() == 1.0
