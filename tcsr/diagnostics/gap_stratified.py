"""Gap-stratified analysis (paper Sec. 5, stratified table).

The valid anchors are stratified by the two-discrepancy gaps g_D
(distillation-inherited) and g_Q (quantization-introduced) into tertiles.
The tertile edges are frozen on the dev split and applied unchanged to the
held-out split; each of the 3 x 3 held-out cells reports the anchor count,
the mean corroboration c_j, the mean un-corroborated priority u^0_j, and
the mean reconstruction improvement

    -Delta_r(j) = g_Q(j) - d(y^P_j, y^{Qf}_{rho_f(j)}),

where rho_f is the anchor-to-Q_f assignment re-solved on the reconstructed
quantized student Q_f (a fresh LSAP, not the frozen rho_0).  Positive
-Delta_r means Stage-2 shrank the quantization-introduced gap.  When
ground truth and letterbox parameters are supplied, the cells additionally
report anchor correctness (best IoU >= 0.5 against the image's GT boxes,
in original-image absolute coordinates) and the mean best IoU.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from ..config import MethodConfig
from ..evaluation import bbox_iou_matrix
from ..losses import Prediction, aligned_discrepancy, pairwise_discrepancy
from ..matching import solve_anchor_assignment
from ..priority import compute_priorities
from ..teacher import ImageRecord
from .gap_statistics import collect_gaps


def _gather(pred: Prediction, idx: torch.Tensor) -> Prediction:
    raw = pred.raw_boxes
    return Prediction(logits=pred.logits[idx], boxes=pred.boxes[idx],
                      raw_boxes=None if raw is None else raw[idx])


def _tertile_edges(x: np.ndarray) -> Tuple[float, float]:
    q = np.quantile(x, [1.0 / 3.0, 2.0 / 3.0])
    return float(q[0]), float(q[1])


def _denormalize_anchor_boxes(norm_cxcywh: np.ndarray, scale: float,
                              pad_x: float, pad_y: float, target_size: float
                              ) -> np.ndarray:
    """Normalized cxcywh (letterboxed input frame) -> absolute xywh on the
    original image: (v * T - pad) / scale."""
    cx = (norm_cxcywh[:, 0] * target_size - pad_x) / scale
    cy = (norm_cxcywh[:, 1] * target_size - pad_y) / scale
    w = norm_cxcywh[:, 2] * target_size / scale
    h = norm_cxcywh[:, 3] * target_size / scale
    return np.stack([cx - w / 2.0, cy - h / 2.0, w, h], axis=1)


def _cell_stats(neg_dr: np.ndarray, c: np.ndarray, u0: np.ndarray,
                best_iou: Optional[np.ndarray]) -> Dict[str, float]:
    out = {"n": float(neg_dr.size),
           "mean_c": float(np.mean(c)) if c.size else float("nan"),
           "mean_u0": float(np.mean(u0)) if u0.size else float("nan")}
    finite = np.isfinite(neg_dr)
    out["n_delta"] = float(int(finite.sum()))
    out["mean_neg_delta_r"] = (float(np.mean(neg_dr[finite]))
                               if finite.any() else float("nan"))
    if best_iou is not None and best_iou.size:
        out["mean_correct"] = float(np.mean(best_iou >= 0.5))
        out["mean_best_iou"] = float(np.mean(best_iou))
    else:
        out["mean_correct"] = float("nan")
        out["mean_best_iou"] = float("nan")
    return out


@torch.no_grad()
def gap_stratified(records_dev: Sequence[ImageRecord], q0_predictions_dev,
                   records_held: Sequence[ImageRecord], q0_predictions_held,
                   qf_predictions_held, cfg: MethodConfig, head_type: str,
                   gt_by_image: Optional[Mapping[str, Tuple[np.ndarray,
                                                            Tuple[int, int]]]] = None,
                   letterbox_by_image: Optional[Mapping[str, Tuple[float,
                                                                   float,
                                                                   float]]] = None,
                   input_size: int = 640) -> Dict:
    """3 x 3 (g_D tertile x g_Q tertile) stratification of held-out anchors.

    q0/qf predictions are each either a Mapping image_id -> Prediction or a
    sequence aligned with the respective records.  gt_by_image maps
    image_id -> (gt_boxes_abs_xywh, (w, h)); letterbox_by_image maps
    image_id -> (scale, pad_x, pad_y) as produced by
    tcsr.datasets.letterbox_params.  Returns the frozen edges, the 3 x 3
    cell table, and the overall row.
    """
    if any(r.rho0 is None for r in records_dev):
        compute_priorities(records_dev, q0_predictions_dev, cfg, head_type,
                           in_place=True)
    if any(r.rho0 is None for r in records_held):
        compute_priorities(records_held, q0_predictions_held, cfg, head_type,
                           in_place=True)
    dev_gaps = collect_gaps(records_dev, q0_predictions_dev, cfg, head_type)
    if dev_gaps["n_valid"] == 0:
        return {"edges_g_D": (float("nan"),) * 2,
                "edges_g_Q": (float("nan"),) * 2,
                "cells": [[_cell_stats(np.zeros(0), np.zeros(0),
                                       np.zeros(0), None) for _ in range(3)]
                          for _ in range(3)],
                "overall": _cell_stats(np.zeros(0), np.zeros(0), np.zeros(0),
                                       None),
                "n_no_gt": 0.0}
    edges_d = _tertile_edges(dev_gaps["g_D"])
    edges_q = _tertile_edges(dev_gaps["g_Q"])

    is_map_q0 = isinstance(q0_predictions_held, Mapping)
    seq_q0 = None if is_map_q0 else list(q0_predictions_held)
    is_map_qf = isinstance(qf_predictions_held, Mapping)
    seq_qf = None if is_map_qf else list(qf_predictions_held)

    flat: Dict[str, List[np.ndarray]] = {k: [] for k in
                                         ("g_D", "g_Q", "c", "u0", "ndr",
                                          "iou")}
    n_no_gt = 0
    for i, rec in enumerate(records_held):
        q0: Prediction = (q0_predictions_held[rec.image_id] if is_map_q0
                          else seq_q0[i])
        qf: Prediction = (qf_predictions_held[rec.image_id] if is_map_qf
                          else seq_qf[i])
        if (rec.partner is None or rec.rho0 is None
                or rec.teacher_pred is None):
            continue
        valid = (rec.partner >= 0) & (rec.rho0 >= 0)
        if not bool(valid.any()):
            continue
        vm = torch.nonzero(valid, as_tuple=False).flatten()
        ap = _gather(rec.anchor_predictions(), vm)
        tp = _gather(rec.teacher_pred, rec.partner[valid])
        qp = _gather(q0, rec.rho0[valid])
        g_d = aligned_discrepancy(tp, ap, cfg, head_type)
        g_q = aligned_discrepancy(ap, qp, cfg, head_type)
        # Re-solve the anchor-to-Q_f assignment on the reconstructed student.
        a_pred = rec.anchor_predictions()
        d_f = pairwise_discrepancy(a_pred, qf, cfg, head_type)
        rho_f = solve_anchor_assignment(d_f)
        ndr = torch.full_like(g_q, float("nan"))
        ok = rho_f[valid] >= 0
        if bool(ok.any()):
            rows = vm[ok]
            ndr[ok] = g_q[ok] - d_f[rows, rho_f[valid][ok]]
        flat["g_D"].append(g_d.cpu().numpy())
        flat["g_Q"].append(g_q.cpu().numpy())
        flat["c"].append(rec.c[valid].cpu().numpy())
        flat["u0"].append(rec.u0[valid].cpu().numpy())
        flat["ndr"].append(ndr.cpu().numpy())
        # Optional ground-truth correctness of the anchor boxes.
        best_iou = None
        if gt_by_image is not None and letterbox_by_image is not None:
            gt = gt_by_image.get(rec.image_id)
            lb = letterbox_by_image.get(rec.image_id)
            if gt is not None and lb is not None:
                gt_boxes = np.asarray(gt[0], dtype=np.float64)
                norm = rec.student_pred.boxes[rec.anchors[valid]]
                anchor_abs = _denormalize_anchor_boxes(
                    norm.cpu().numpy().astype(np.float64),
                    float(lb[0]), float(lb[1]), float(lb[2]), input_size)
                if gt_boxes.size:
                    ious = bbox_iou_matrix(anchor_abs, gt_boxes,
                                           np.zeros(gt_boxes.shape[0],
                                                    dtype=bool))
                    best_iou = ious.max(axis=1)
                else:
                    best_iou = np.zeros(anchor_abs.shape[0])
            else:
                n_no_gt += int(valid.sum())
        flat["iou"].append(best_iou if best_iou is not None
                           else np.full(int(valid.sum()), np.nan))
    g_d = (np.concatenate(flat["g_D"]) if flat["g_D"]
           else np.zeros(0, dtype=np.float32))
    g_q = (np.concatenate(flat["g_Q"]) if flat["g_Q"]
           else np.zeros(0, dtype=np.float32))
    c = (np.concatenate(flat["c"]) if flat["c"]
         else np.zeros(0, dtype=np.float32))
    u0 = (np.concatenate(flat["u0"]) if flat["u0"]
          else np.zeros(0, dtype=np.float32))
    ndr = (np.concatenate(flat["ndr"]) if flat["ndr"]
           else np.zeros(0, dtype=np.float32))
    iou = np.concatenate(flat["iou"]) if flat["iou"] else np.zeros(0)
    has_gt = np.isfinite(iou)

    bin_d = np.digitize(g_d, np.asarray(edges_d))
    bin_q = np.digitize(g_q, np.asarray(edges_q))
    cells = []
    for bi in range(3):
        row = []
        for bj in range(3):
            m = (bin_d == bi) & (bin_q == bj)
            iou_cell = iou[m & has_gt] if has_gt.any() else np.zeros(0)
            cell = _cell_stats(ndr[m], c[m], u0[m],
                               iou_cell if iou_cell.size else None)
            cell["n_gt"] = float(int((m & has_gt).sum()))
            row.append(cell)
        cells.append(row)
    iou_all = iou[has_gt] if has_gt.any() else None
    overall = _cell_stats(ndr, c, u0, iou_all)
    overall["n_gt"] = float(int(has_gt.sum()))
    return {"edges_g_D": edges_d, "edges_g_Q": edges_q, "cells": cells,
            "overall": overall, "n_no_gt": float(n_no_gt)}
