"""Calibration observers.

Weights are quantized per output channel and activations per tensor (paper
Sec. "Datasets, Models, and Quantization"). Two initializers are provided:
min-max and a small grid search minimizing the dequantization MSE over
clipped ranges.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


class MinMaxObserver:
    """Plain min/max range estimation."""

    def __init__(self, per_channel: bool = False, symmetric: bool = False,
                 store_cache: bool = False):
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.store_cache = store_cache
        self.min_val: torch.Tensor | None = None
        self.max_val: torch.Tensor | None = None
        self.cache: list[torch.Tensor] = []
        self._seen = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if self.per_channel:  # x: (out_channels, ...) -> reduce trailing dims
            dims = tuple(range(1, x.dim()))
            mn = x.amin(dim=dims) if dims else x.clone()
            mx = x.amax(dim=dims) if dims else x.clone()
        else:
            mn, mx = x.min(), x.max()
        if self.min_val is None:
            self.min_val, self.max_val = mn.detach().clone(), mx.detach().clone()
        else:
            self.min_val = torch.minimum(self.min_val, mn.detach())
            self.max_val = torch.maximum(self.max_val, mx.detach())
        if self.store_cache and self._seen < 4:
            if self.per_channel:  # (out_channels, ...) -> (C, n) rows
                self.cache.append(
                    x.detach().float().reshape(x.shape[0], -1).cpu())
            else:
                flat = x.detach().float().flatten()
                self.cache.append(flat[: 8192].cpu())
        self._seen += 1

    def range(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.min_val is not None, "observer saw no data"
        if self.symmetric:
            bound = torch.maximum(self.min_val.abs(), self.max_val.abs())
            return -bound.clone(), bound.clone()
        return self.min_val.clone(), self.max_val.clone()


class MSEObserver(MinMaxObserver):
    """Grid-search clipped min/max minimizing squared reconstruction error.

    Candidates shrink the observed min/max range symmetrically by a factor
    alpha in [0.5, 1.0]; the pair with the lowest 8-bit round-to-nearest
    dequantization MSE is kept (bit widths rescale the range afterwards).
    """

    GRID = np.linspace(0.50, 1.0, 11)

    def __init__(self, per_channel: bool = False, symmetric: bool = False):
        super().__init__(per_channel=per_channel, symmetric=symmetric,
                         store_cache=True)

    @torch.no_grad()
    def range(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.min_val is not None, "observer saw no data"
        lo, hi = super().range()
        if not self.cache:
            return lo, hi
        if self.per_channel:  # cache entries are (C, n) rows -> (C, total)
            n_ch = int(self.min_val.numel())
            x = torch.cat([t.reshape(n_ch, -1) for t in self.cache], dim=1)
        else:
            x = torch.cat(self.cache)
        best_lo, best_hi, best_err = lo, hi, float("inf")
        for alpha in self.GRID:
            c_lo = lo + (hi - lo) * (1.0 - alpha) / 2.0
            c_hi = hi - (hi - lo) * (1.0 - alpha) / 2.0
            err = self._err(x, c_lo, c_hi)
            if err < best_err:
                best_err, best_lo, best_hi = err, c_lo, c_hi
        return best_lo, best_hi

    @staticmethod
    def _err(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> float:
        qmin, qmax = 0, 255
        if lo.dim() > 0:  # per-channel: (C,) -> (C, 1) vs (C, n) data
            lo, hi = lo.unsqueeze(-1), hi.unsqueeze(-1)
        scale = (hi - lo).clamp_min(1e-12) / (qmax - qmin)
        zero = torch.clamp(torch.round(-lo / scale), qmin, qmax)
        q = torch.clamp(torch.round(x / scale) + zero, qmin, qmax)
        deq = (q - zero) * scale
        return float((deq - x).pow(2).mean())
