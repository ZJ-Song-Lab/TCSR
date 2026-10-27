"""Bounded class-box discrepancy (paper Sec. 4.1, Eqs. 3-6).

For an independent-sigmoid head the class term uses the label-smoothed
temperature-scaled confidence

    a_c = eps + (1 - 2 eps) * sigmoid(z_c / tau_cal)

and a foreground/background-weighted Bernoulli Jensen-Shannon divergence. For
a softmax head it is the log2-normalized Jensen-Shannon divergence between
the full class-plus-background probability vectors; anchor confidence
excludes the background component. Box terms are D_l1 = ||b - b'||_1 / 4 and
D_giou = (1 - GIoU) / 2, and

    d = eta_cls D_cls + eta_l1 D_l1 + eta_giou D_giou,  with d in [0, 1].

Boxes are normalized cxcywh in [0, 1]^4. All functions are differentiable in
both predictions; see diagnostics/direction_conflict.py for the raw
pre-activation variant used by the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from ..config import MethodConfig

_EPS_LOG = 1e-8


@dataclass
class Prediction:
    """A set prediction: class logits and normalized cxcywh boxes."""

    logits: torch.Tensor   # (N, C_logits)
    boxes: torch.Tensor    # (N, 4) in [0, 1]
    # Raw pre-activation box parameters t (before sigmoid), kept by model
    # adapters for the direction-conflict diagnostic; optional.
    raw_boxes: Optional[torch.Tensor] = None


def class_probs(logits: torch.Tensor, cfg: MethodConfig,
                head_type: str) -> torch.Tensor:
    """Smoothed class probabilities used by the class term.

    sigmoid head: a_c = eps + (1-2eps) sigmoid(z_c / tau_cal)  -> (N, C)
    softmax head: full class-plus-background softmax           -> (N, C+1)
    """
    if head_type == "sigmoid":
        a = torch.sigmoid(logits / cfg.tau_cal)
        return cfg.epsilon + (1.0 - 2.0 * cfg.epsilon) * a
    return F.softmax(logits, dim=-1)


def anchor_confidence(logits: torch.Tensor, cfg: MethodConfig,
                      head_type: str) -> torch.Tensor:
    """s_r: maximum foreground probability (background excluded for softmax).

    Returns (s, l): confidence and its class index in the foreground space.
    """
    if head_type == "sigmoid":
        a = torch.sigmoid(logits)
        s, l = a.max(dim=-1)
    else:
        p = F.softmax(logits, dim=-1)
        fg = p[..., :-1]  # background is the last node
        s, l = fg.max(dim=-1)
    return s, l


def bernoulli_js(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Normalized Bernoulli Jensen-Shannon divergence J(a, b) in [0, 1].

    k_B(a,b) = a log(a/b) + (1-a) log((1-a)/(1-b)),
    J(a,b)   = [k_B(a,m) + k_B(b,m)] / (2 log 2),  m = (a+b)/2.
    """
    a = a.clamp(_EPS_LOG, 1.0 - _EPS_LOG)
    b = b.clamp(_EPS_LOG, 1.0 - _EPS_LOG)
    m = 0.5 * (a + b)

    def kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))

    return (kl(a, m) + kl(b, m)) / (2.0 * torch.log(torch.tensor(2.0)))


def d_cls_sigmoid_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-pair class term for aligned rows: (N, C) x (N, C) -> (N,).

    For each row pair, w+ = (a_c + b_c)/2, w- = 1 - w+, and
    D_cls = 0.5 * sum_c(w+ J)/sum_c(w+) + 0.5 * sum_c(w- J)/sum_c(w-).
    """
    j = bernoulli_js(a, b)                      # (N, C)
    w_pos = 0.5 * (a + b)                       # (N, C)
    w_neg = 1.0 - w_pos
    pos = (w_pos * j).sum(-1) / (w_pos.sum(-1) + _EPS_LOG)
    neg = (w_neg * j).sum(-1) / (w_neg.sum(-1) + _EPS_LOG)
    return 0.5 * pos + 0.5 * neg


def d_cls_sigmoid(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Set-level class term: mean of the per-pair D_cls over aligned rows."""
    return d_cls_sigmoid_rows(a, b).mean()


def d_cls_softmax_rows(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Per-pair log2-normalized JS between full class-plus-background
    probability vectors: (N, C+1) x (N, C+1) -> (N,)."""
    p = p.clamp_min(_EPS_LOG)
    q = q.clamp_min(_EPS_LOG)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(-1)
    kl_qm = (q * (q / m).log()).sum(-1)
    return 0.5 * (kl_pm + kl_qm) / math.log(2.0)


def d_cls_softmax(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Set-level class term: mean of the per-pair JS over aligned rows."""
    return d_cls_softmax_rows(p, q).mean()


def d_l1(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """D_l1 = ||b - b'||_1 / 4, averaged over the set."""
    return (b1 - b2).abs().sum(dim=-1).mean() / 4.0


def box_cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def giou(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """Generalized IoU between normalized cxcywh boxes, elementwise."""
    a = box_cxcywh_to_xyxy(b1)
    c = box_cxcywh_to_xyxy(b2)
    lt = torch.maximum(a[..., :2], c[..., :2])
    rb = torch.minimum(a[..., 2:], c[..., 2:])
    wh = (rb - lt).clamp_min(0.0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[..., 2] - a[..., 0]).clamp_min(0) * (a[..., 3] - a[..., 1]).clamp_min(0)
    area_c = (c[..., 2] - c[..., 0]).clamp_min(0) * (c[..., 3] - c[..., 1]).clamp_min(0)
    union = area_a + area_c - inter
    iou = inter / (union + _EPS_LOG)
    # smallest enclosing box
    elt = torch.minimum(a[..., :2], c[..., :2])
    erb = torch.maximum(a[..., 2:], c[..., 2:])
    ewh = (erb - elt).clamp_min(0.0)
    enc = ewh[..., 0] * ewh[..., 1]
    return iou - (enc - union) / (enc + _EPS_LOG)


def d_giou(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """D_giou = (1 - GIoU) / 2, averaged over the set."""
    return ((1.0 - giou(b1, b2)) / 2.0).mean()


def discrepancy_components(y1: Prediction, y2: Prediction, cfg: MethodConfig,
                           head_type: str) -> Dict[str, torch.Tensor]:
    """Per-component discrepancies {cls, l1, giou, total} between two sets.

    The two sets must have equal length and be index-aligned (caller is
    responsible for matching).
    """
    if head_type == "sigmoid":
        a = class_probs(y1.logits, cfg, "sigmoid")
        b = class_probs(y2.logits, cfg, "sigmoid")
        cls = d_cls_sigmoid(a, b)
    else:
        p = class_probs(y1.logits, cfg, "softmax")
        q = class_probs(y2.logits, cfg, "softmax")
        cls = d_cls_softmax(p, q)
    l1 = d_l1(y1.boxes, y2.boxes)
    g = d_giou(y1.boxes, y2.boxes)
    total = cfg.eta_cls * cls + cfg.eta_l1 * l1 + cfg.eta_giou * g
    return {"cls": cls, "l1": l1, "giou": g, "total": total}


def discrepancy(y1: Prediction, y2: Prediction, cfg: MethodConfig,
                head_type: str) -> torch.Tensor:
    """Bounded class-box discrepancy d(y1, y2) in [0, 1]."""
    return discrepancy_components(y1, y2, cfg, head_type)["total"]


def pairwise_discrepancy(y1: Prediction, y2: Prediction, cfg: MethodConfig,
                         head_type: str) -> torch.Tensor:
    """Dense pairwise d between all pairs of two sets -> (N1, N2).

    Each component is computed pairwise (no set reduction), then combined.
    """
    n1, n2 = y1.logits.shape[0], y2.logits.shape[0]
    if head_type == "sigmoid":
        a = class_probs(y1.logits, cfg, "sigmoid")          # (N1, C)
        b = class_probs(y2.logits, cfg, "sigmoid")          # (N2, C)
        j = bernoulli_js(a[:, None, :].expand(n1, n2, a.shape[-1]),
                         b[None, :, :].expand(n1, n2, b.shape[-1]))
        w_pos = 0.5 * (a[:, None, :] + b[None, :, :])
        w_neg = 1.0 - w_pos
        pos = (w_pos * j).sum(-1) / (w_pos.sum(-1) + _EPS_LOG)
        neg = (w_neg * j).sum(-1) / (w_neg.sum(-1) + _EPS_LOG)
        cls = 0.5 * pos + 0.5 * neg                          # (N1, N2)
    else:
        p = class_probs(y1.logits, cfg, "softmax")
        q = class_probs(y2.logits, cfg, "softmax")
        p = p.clamp_min(_EPS_LOG)
        q = q.clamp_min(_EPS_LOG)
        m = 0.5 * (p[:, None, :] + q[None, :, :])
        kl_pm = (p[:, None, :] * (p[:, None, :] / m).log()).sum(-1)
        kl_qm = (q[None, :, :] * (q[None, :, :] / m).log()).sum(-1)
        cls = (0.5 * (kl_pm + kl_qm) / torch.log(torch.tensor(2.0)))  # (N1, N2)
    l1 = (y1.boxes[:, None, :] - y2.boxes[None, :, :]).abs().sum(-1) / 4.0
    g = (1.0 - giou(y1.boxes[:, None, :].expand(n1, n2, 4),
                    y2.boxes[None, :, :].expand(n1, n2, 4))) / 2.0
    return cfg.eta_cls * cls + cfg.eta_l1 * l1 + cfg.eta_giou * g


def aligned_discrepancy(y1: Prediction, y2: Prediction, cfg: MethodConfig,
                        head_type: str) -> torch.Tensor:
    """Per-pair d(y1_r, y2_r) for index-aligned prediction sets -> (N,).

    This is the per-anchor quantity whose mean (1/max(1, N)) enters L_fg and
    L_cor (Eqs. fg and corloss) and whose values form the two-discrepancy
    diagnostics g_D, g_Q, g_TQ.
    """
    if head_type == "sigmoid":
        a = class_probs(y1.logits, cfg, "sigmoid")
        b = class_probs(y2.logits, cfg, "sigmoid")
        cls = d_cls_sigmoid_rows(a, b)
    else:
        p = class_probs(y1.logits, cfg, "softmax")
        q = class_probs(y2.logits, cfg, "softmax")
        cls = d_cls_softmax_rows(p, q)
    l1 = (y1.boxes - y2.boxes).abs().sum(-1) / 4.0
    g = (1.0 - giou(y1.boxes, y2.boxes)) / 2.0
    return cfg.eta_cls * cls + cfg.eta_l1 * l1 + cfg.eta_giou * g
