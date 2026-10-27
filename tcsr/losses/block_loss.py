"""Normalized block-reconstruction loss (Stage 1, teacher-free).

The common L_blk used by every variant measures the normalized squared
reconstruction error of a quantized block against the frozen full-precision
student block output:

    L_blk = || out_Q - out_P ||^2 / (|| out_P ||^2 + eps )

Block partitioning is provided by the model adapters
(``DetectorAdapter.blocks()``); see tcsr/models/base.py.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch

_EPS = 1e-8


def normalized_block_loss(out_q: torch.Tensor,
                          out_p: torch.Tensor) -> torch.Tensor:
    """Normalized squared error between a quantized and a reference block
    output (any shape)."""
    denom = out_p.detach().pow(2).sum() + _EPS
    return (out_q - out_p).pow(2).sum() / denom


def block_losses(outputs_q: Iterable[torch.Tensor],
                 outputs_p: Iterable[torch.Tensor]) -> Tuple[torch.Tensor, int]:
    """Sum of normalized losses over aligned block outputs."""
    total = torch.zeros(())
    n = 0
    for q, p in zip(outputs_q, outputs_p):
        total = total + normalized_block_loss(q, p)
        n += 1
    return total, n
