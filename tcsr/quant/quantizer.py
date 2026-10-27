"""Uniform dequantizer with learnable scale/clipping and optional AdaRound
soft rounding (paper Eq. 2 and Sec. "Quantized Set Prediction").

    v_hat = s * [ clip( round(v/s) + z, q_min, q_max ) - z ],
    q_max - q_min + 1 = 2^b

The trainable quantizer state theta contains only quantizer scales, clipping
parameters, and optional soft-rounding variables. Weights use per-output-
channel scales; activations use per-tensor scales. During calibration the
rounding of weights is relaxed (AdaRound) and hardened at export; activations
always use straight-through rounding.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def round_to_nearest(v: torch.Tensor) -> torch.Tensor:
    """Round half to even, matching hardware rounding."""
    return torch.round(v)


def _target_to_log_scale(target: torch.Tensor) -> torch.Tensor:
    """Invert softplus: x with softplus(x) == target (elementwise)."""
    target = target.clamp_min(1e-8)
    return torch.log(torch.expm1(target.clamp_max(80.0)) + 1e-12).clamp_(-60.0, 60.0)


class UniformQuantizer(nn.Module):
    """Differentiable uniform fake-quantizer with broadcastable theta.

    Parameters
    ----------
    bits : quantization bit width b (q_max - q_min + 1 == 2^b).
    symmetric : if True use a signed symmetric grid with z = 0.
    learnable : register scale/clipping as parameters (theta) or buffers.
    shape : trailing shape for per-channel quantization, e.g. (C, 1, 1) for a
        conv weight; a scalar shape gives per-tensor quantization.
    """

    def __init__(self, bits: int = 8, symmetric: bool = False,
                 learnable: bool = True, shape: tuple = ()):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        if symmetric:
            self.qmin = -(2 ** (bits - 1))
            self.qmax = 2 ** (bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** bits - 1
        # theta: scale (log-space), clipping bounds. Shape is broadcastable
        # to the quantized tensor.
        self.log_scale = nn.Parameter(torch.zeros(shape), requires_grad=learnable)
        self.clip_lo = nn.Parameter(torch.zeros(shape), requires_grad=learnable)
        self.clip_hi = nn.Parameter(torch.zeros(shape), requires_grad=learnable)
        self.register_buffer("zero_point", torch.zeros(shape, dtype=torch.long))
        self.learnable = learnable
        self.hardened = False

    @torch.no_grad()
    def configure(self, min_val: torch.Tensor, max_val: torch.Tensor) -> None:
        """Initialize theta from an observed [min, max] range.

        ``min_val``/``max_val`` may be tensors whose shape broadcasts to the
        target tensor (per-output-channel for weights) or scalars.
        """
        lo = torch.as_tensor(min_val, dtype=torch.float32)
        hi = torch.as_tensor(max_val, dtype=torch.float32)
        shape = self.log_scale.shape
        if lo.numel() == 1 and math.prod(shape) > 1:
            # Scalar range broadcast to the per-channel layout.
            lo = lo.reshape(()).expand(shape).clone()
            hi = hi.reshape(()).expand(shape).clone()
        else:
            lo = lo.reshape(shape).clone()
            hi = hi.reshape(shape).clone()
        if self.symmetric:
            bound = torch.maximum(lo.abs(), hi.abs()).clamp_min(1e-8)
            lo, hi = -bound, bound
        hi = torch.maximum(hi, lo + 1e-8)
        scale = (hi - lo) / max(self.qmax - self.qmin, 1)
        scale = scale.clamp_min(1e-10).to(self.log_scale.device)
        self.log_scale.data.copy_(_target_to_log_scale(scale))
        self.clip_lo.data.copy_(lo.to(self.clip_lo.device))
        self.clip_hi.data.copy_(hi.to(self.clip_hi.device))
        self._refresh_zero_point()

    @property
    def scale(self) -> torch.Tensor:
        return F.softplus(self.log_scale) + 1e-12

    @torch.no_grad()
    def _refresh_zero_point(self) -> None:
        s = self.scale.detach()
        if self.symmetric:
            z = torch.zeros_like(s)
        else:
            z = torch.round(self.qmin - self.clip_lo.detach() / s)
            z = torch.clamp(z, self.qmin, self.qmax)
        self.zero_point.copy_(z.to(torch.long))

    def _round(self, v_c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # Straight-through estimator around the rounding operator only: the
        # value is round(v_c / s) but the gradient passes through v_c / s,
        # hence to the scale, the clipping bounds, and the input (LSQ-style).
        q = v_c / s
        return q + (round_to_nearest(q) - q).detach()

    def quantize(self, v: torch.Tensor) -> torch.Tensor:
        """Fake-quantized value; gradients flow to scale and clip bounds."""
        s = self.scale
        z = self.zero_point.to(v.dtype)
        v_c = torch.clamp(v, self.clip_lo, self.clip_hi)
        q = self._round(v_c, s) + z
        q = torch.clamp(q, self.qmin, self.qmax)
        return s * (q - z)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # ``_round`` already carries the straight-through estimator, so the
        # output is differentiable w.r.t. theta (scale, clipping bounds and,
        # for AdaRound, the soft-rounding variables).
        return self.quantize(v)


class AdaRoundQuantizer(UniformQuantizer):
    """Weight quantizer with soft rounding during calibration.

    round(v/s) is relaxed as floor(v/s) + h with
    h = clamp( sigmoid(alpha * (beta - 0.5)), 0, 1 ),  beta = v/s - floor(v/s),
    the rectified-sigmoid AdaRound relaxation. ``harden()`` freezes h to its
    rounding for export. The soft-rounding regularizer is
    R_round = sum( (1 - |2h - 1|)^rho ).
    """

    def __init__(self, bits: int = 8, symmetric: bool = False,
                 rho: float = 20.0, init_alpha: float = 2.0, shape: tuple = (),
                 weight_shape: Optional[tuple] = None):
        super().__init__(bits=bits, symmetric=symmetric, learnable=True,
                         shape=shape)
        self.rho = rho
        # One soft-rounding variable per weight (standard AdaRound): h is a
        # rectified sigmoid of alpha * (beta - 0.5) evaluated elementwise.
        if weight_shape is None:
            self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        else:
            self.alpha = nn.Parameter(
                torch.full(tuple(weight_shape), float(init_alpha)))
        self.soft = True

    def _h(self, v_c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        beta = (v_c / s) - torch.floor(v_c / s)
        h = torch.sigmoid(self.alpha * (beta - 0.5))
        return torch.clamp(h, 0.0, 1.0)

    def _round(self, v_c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        base = torch.floor(v_c / s)
        h = self._h(v_c, s)
        if not (self.soft and self.training):
            h = round_to_nearest(h.detach())
        return base + h

    def rounding_regularizer(self, v: torch.Tensor) -> torch.Tensor:
        """R_round on the current soft rounding variables."""
        s = self.scale.detach()
        v_c = torch.clamp(v, self.clip_lo.detach(), self.clip_hi.detach())
        h = self._h(v_c, s)
        return (1.0 - (2.0 * h - 1.0).abs().pow(self.rho)).sum()

    @torch.no_grad()
    def harden(self) -> None:
        self.soft = False
        self.hardened = True
