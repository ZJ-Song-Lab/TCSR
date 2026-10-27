r"""Residual-set regularizer: Gaussian-kernel maximum mean discrepancy
(paper Sec. 4.4, Eq. 12).

Residual queries R_P = [N_P] \ A_P and R_Q = [N_Q] \ rho(A_P) have no
index-wise semantic correspondence, so their smoothed class-probability sets
are compared with an MMD:

    MMD^2 = 1/n^2 sum k(v,v') + 1/m^2 sum k(w,w') - 2/(nm) sum k(v,w),
    k(v,v') = exp(-||v-v'||^2 / (2 sigma^2)),
    L_res = 0.5 * MMD^2.

The bandwidth is fixed from cached development residuals by a median-distance
rule with lower bound sigma_min = 0.020 (frozen values: 0.084 / 0.079 /
0.112 / 0.108 for HRSID, SSDD, COCO/RT-DETR, COCO/Deformable DETR).
"""

from __future__ import annotations

import torch

from ..config import MethodConfig


def _sq_dists(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distances between rows of a (n,d) and b (m,d)."""
    a2 = (a * a).sum(-1, keepdim=True)              # (n, 1)
    b2 = (b * b).sum(-1).unsqueeze(0)               # (1, m)
    d = a2 + b2 - 2.0 * a @ b.t()
    return d.clamp_min(0.0)


def residual_mmd(v: torch.Tensor, w: torch.Tensor, sigma: float) -> torch.Tensor:
    """L_res = 0.5 * MMD_k^2 between residual probability sets.

    v: (n, C) student residual probabilities; w: (m, C) quantized residual
    probabilities. Differentiable in w.
    """
    if v.numel() == 0 or w.numel() == 0:
        return torch.zeros((), device=v.device if v.numel() else w.device)
    gamma = 1.0 / (2.0 * sigma * sigma)
    k_pp = torch.exp(-gamma * _sq_dists(v, v)).mean()
    k_qq = torch.exp(-gamma * _sq_dists(w, w)).mean()
    k_pq = torch.exp(-gamma * _sq_dists(v, w)).mean()
    mmd2 = k_pp + k_qq - 2.0 * k_pq
    return 0.5 * mmd2.clamp_min(0.0)


@torch.no_grad()
def median_bandwidth(v: torch.Tensor, sigma_min: float = 0.020,
                     max_points: int = 2048) -> float:
    """Median-distance bandwidth rule with the frozen lower bound."""
    if v.numel() == 0:
        return sigma_min
    v = v.detach().float().reshape(-1, v.shape[-1])
    if v.shape[0] > max_points:
        idx = torch.randint(0, v.shape[0], (max_points,), device=v.device)
        v = v[idx]
    d2 = _sq_dists(v, v)
    iu = torch.triu_indices(d2.shape[0], d2.shape[0], offset=1)
    vals = d2[iu[0], iu[1]]
    if vals.numel() == 0:
        return sigma_min
    med = torch.median(vals).sqrt().item()
    return max(med, sigma_min)


class ResidualMMD(torch.nn.Module):
    """Convenience module holding the frozen per-setting bandwidth."""

    def __init__(self, bandwidth: float):
        super().__init__()
        self.bandwidth = float(bandwidth)

    def forward(self, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return residual_mmd(v, w, self.bandwidth)

    @staticmethod
    def calibrate(dev_residuals: torch.Tensor,
                  cfg: MethodConfig) -> "ResidualMMD":
        bw = median_bandwidth(dev_residuals, sigma_min=cfg.sigma_min)
        return ResidualMMD(bw)
