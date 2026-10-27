"""Direction conflict kappa (paper Sec. 5, direction-conflict diagnostic).

For every anchor j with a feasible teacher partner iota(j) and a valid
Q_0 match rho_0(j), let u = (z, t) collect the raw pre-activation
parameters of the matched Q_0 query rho_0(j): the full class-plus-background
logit vector z (softmax head) or the C class logits (sigmoid head), and the
pre-sigmoid box parameters t.  The reconstructed prediction is
R(u) = (z, sigma(t)); the class activation (softmax, or the tau_cal-scaled
smoothed sigmoid inside D_cls) is applied inside the class term exactly as
in losses.discrepancy.  The per-component gradients

    g^P_{j,r} = grad_u d_r(y^P_j, R(u))   at u = u^{Q0}_{rho_0(j)}
    g^T_{j,r} = grad_u d_r(y^T_{iota(j)}, R(u))

are taken with the target side held fixed (detached constants), and the
direction conflict is the cosine

    kappa_{j,r} = <g^P, g^T> / (||g^P|| ||g^T|| + eps_g),   r in
                  {cls, l1, giou, all}.

The l1 gradient therefore carries the sigmoid derivative b_k (1 - b_k), and
the softmax-head class gradient spans the full class-plus-background logit
vector.  Pairs where either gradient norm is <= eps_g (zero-norm pairs) are
excluded from the statistics and counted separately.

Because every anchor row of the distance only depends on the same row of u,
the per-anchor gradients are the row blocks of a single vector-Jacobian
product: all anchors of one image are differentiated in one backward pass
per component.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import torch

from ..config import MethodConfig
from ..losses import (Prediction, class_probs, d_cls_sigmoid_rows,
                      d_cls_softmax_rows, giou)
from ..teacher import ImageRecord

_CLAMP_MIN = 1e-6
_CLAMP_MAX = 1.0 - 1e-6

COMPONENTS = ("cls", "l1", "giou", "all")


def _gather(pred: Prediction, idx: torch.Tensor) -> Prediction:
    raw = pred.raw_boxes
    return Prediction(logits=pred.logits[idx], boxes=pred.boxes[idx],
                      raw_boxes=None if raw is None else raw[idx])


def _raw_box_params(pred: Prediction, idx: torch.Tensor) -> torch.Tensor:
    """Raw pre-activation box parameters t for the selected rows.

    Uses pred.raw_boxes when the model adapter kept it; otherwise inverts
    the sigmoid exactly, t = log(b / (1 - b)), with saturation clamping
    (DETR box heads output b = sigma(t), so the inversion is exact away
    from saturation).
    """
    if pred.raw_boxes is not None:
        return pred.raw_boxes[idx]
    b = pred.boxes[idx].clamp(_CLAMP_MIN, _CLAMP_MAX)
    return torch.log(b) - torch.log1p(-b)


def _row_distances(u: torch.Tensor, target_logits: torch.Tensor,
                   target_boxes: torch.Tensor, cfg: MethodConfig,
                   head_type: str) -> Dict[str, torch.Tensor]:
    """Row-level component distances between R(u) and the target rows.

    u: (K, C + 4) leaf with z = u[:, :C] and t = u[:, C:]; the targets are
    detached constants.  Returns {cls, l1, giou, total}, each (K,).
    """
    n_feat = u.shape[1]
    c = n_feat - 4
    z, t = u[:, :c], u[:, c:]
    boxes = torch.sigmoid(t)
    if head_type == "sigmoid":
        a = class_probs(z, cfg, "sigmoid")
        b = class_probs(target_logits, cfg, "sigmoid")
        cls = d_cls_sigmoid_rows(a, b)
    else:
        p = class_probs(z, cfg, "softmax")
        q = class_probs(target_logits, cfg, "softmax")
        cls = d_cls_softmax_rows(p, q)
    l1 = (boxes - target_boxes).abs().sum(-1) / 4.0
    g = (1.0 - giou(boxes, target_boxes)) / 2.0
    total = cfg.eta_cls * cls + cfg.eta_l1 * l1 + cfg.eta_giou * g
    return {"cls": cls, "l1": l1, "giou": g, "total": total}


def _component_gradients(u: torch.Tensor, target_logits: torch.Tensor,
                         target_boxes: torch.Tensor, cfg: MethodConfig,
                         head_type: str) -> Dict[str, torch.Tensor]:
    """grad_u d_r(target, R(u)) per component, as (K, C + 4) row blocks."""
    rows = _row_distances(u, target_logits, target_boxes, cfg, head_type)
    grads: Dict[str, torch.Tensor] = {}
    for name, r in rows.items():
        (g,) = torch.autograd.grad(r.sum(), u, retain_graph=True)
        grads[name] = g
    return grads


def _kappa_image(rec: ImageRecord, q0: Prediction, cfg: MethodConfig,
                 head_type: str, eps_g: float
                 ) -> Dict[str, torch.Tensor]:
    """Per-anchor kappa for one image; NaN marks zero-norm pairs.

    Returns {cls, l1, giou, all} tensors of shape (n_valid,), or empty
    tensors on the record's device when the image has no valid anchors.
    """
    dev = rec.student_pred.logits.device
    out = {k: torch.zeros(0, device=dev) for k in COMPONENTS}
    if rec.rho0 is None or rec.partner is None or rec.teacher_pred is None:
        return out
    valid = (rec.partner >= 0) & (rec.rho0 >= 0)
    if not bool(valid.any()):
        return out
    # Leaf u = (z, t) of the matched Q_0 queries.
    z = q0.logits[rec.rho0[valid]].detach()
    t = _raw_box_params(q0, rec.rho0[valid]).detach()
    u = torch.cat([z, t], dim=1).requires_grad_(True)
    # Frozen target values: student anchors and teacher partners.
    p_logit = rec.student_pred.logits[rec.anchors[valid]].detach()
    p_box = rec.student_pred.boxes[rec.anchors[valid]].detach()
    t_logit = rec.teacher_pred.logits[rec.partner[valid]].detach()
    t_box = rec.teacher_pred.boxes[rec.partner[valid]].detach()
    g_p = _component_gradients(u, p_logit, p_box, cfg, head_type)
    g_t = _component_gradients(u, t_logit, t_box, cfg, head_type)
    for name in COMPONENTS:
        key = "total" if name == "all" else name
        gp, gt = g_p[key], g_t[key]
        n_p = gp.norm(dim=1)
        n_t = gt.norm(dim=1)
        cos = (gp * gt).sum(dim=1) / (n_p * n_t + eps_g)
        zero = (n_p <= eps_g) | (n_t <= eps_g)
        cos = torch.where(zero, torch.full_like(cos, float("nan")), cos)
        out[name] = cos
    return out


def kappa_components(records: Sequence[ImageRecord], q0_predictions,
                     cfg: MethodConfig, head_type: str
                     ) -> Dict[str, torch.Tensor]:
    """Per-anchor direction cosines kappa_{j,r} concatenated over images.

    q0_predictions is either a Mapping image_id -> Prediction (the Stage-1
    outputs Q_0) or a sequence aligned with records.  Zero-norm pairs are
    NaN and excluded by direction_conflict(); records must already carry
    rho_0 (run tcsr.priority.compute_priorities first).
    """
    is_map = isinstance(q0_predictions, Mapping)
    seq = None if is_map else list(q0_predictions)
    chunks: Dict[str, list] = {k: [] for k in COMPONENTS}
    for i, rec in enumerate(records):
        q0: Prediction = q0_predictions[rec.image_id] if is_map else seq[i]
        per_image = _kappa_image(rec, q0, cfg, head_type, cfg.eps_g)
        for k in COMPONENTS:
            chunks[k].append(per_image[k])
    out: Dict[str, torch.Tensor] = {}
    for k, v in chunks.items():
        out[k] = torch.cat(v) if v else torch.zeros(0)
    return out


def direction_conflict(records: Sequence[ImageRecord], q0_predictions,
                       cfg: MethodConfig, head_type: str
                       ) -> Dict[str, Dict[str, float]]:
    """Summary statistics of the direction conflict (paper Sec. 5).

    Per component r in {cls, l1, giou, all}: the number of valid pairs n,
    the fraction of negative cosines pr_negative, the median kappa, and the
    number of zero-norm pairs excluded.  Plus the anchor bookkeeping
    (anchors_total, anchors_valid).
    """
    kappa = kappa_components(records, q0_predictions, cfg, head_type)
    n_anchors = sum(int(r.anchors.numel()) for r in records)
    n_valid = 0
    for rec in records:
        if (rec.partner is not None and rec.rho0 is not None
                and rec.teacher_pred is not None):
            n_valid += int(((rec.partner >= 0) & (rec.rho0 >= 0)).sum())
    stats: Dict[str, Dict[str, float]] = {
        "anchors_total": float(n_anchors),
        "anchors_valid": float(n_valid),
    }
    for name in COMPONENTS:
        k = kappa[name]
        finite = torch.isfinite(k)
        n = int(finite.sum())
        kv = k[finite]
        stats[name] = {
            "n": float(n),
            "zero_norm": float(int((~finite).sum())),
            "pr_negative": float((kv < 0).to(torch.float64).mean().item())
            if n else 0.0,
            "median": float(kv.median().item()) if n else float("nan"),
        }
    return stats
