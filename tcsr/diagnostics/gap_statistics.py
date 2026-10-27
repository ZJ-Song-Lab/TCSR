"""Two-discrepancy gap statistics (paper Sec. 5, double-gap diagnostic).

For anchors j with a feasible teacher partner iota(j) and a valid Q_0
match rho_0(j), the two-discrepancy triple is

    g_D(j)  = d(y^T_{iota(j)}, y^P_j)          distillation-inherited gap
    g_Q(j)  = d(y^P_j, y^{Q0}_{rho_0(j)})      quantization-introduced gap
    g_TQ(j) = d(y^T_{iota(j)}, y^{Q0}_{rho_0(j)})

and the un-corroborated priority is the consistency check
u^0_j = [g_TQ(j) - g_D(j)]_+ (Eq. priority), which must agree with the
u^0 recorded by tcsr.priority.compute_priorities.  The summary reports
the coverage of valid anchors, per-column medians with quartiles, the
Spearman rank correlation between g_D and g_Q, and the top-20% overlap
between the two gap rankings.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
from scipy import stats as sps
import torch

from ..config import MethodConfig
from ..losses import Prediction, aligned_discrepancy
from ..priority import compute_priorities
from ..teacher import ImageRecord

GAP_COLUMNS = ("g_D", "g_Q", "g_TQ")


def _gather(pred: Prediction, idx: torch.Tensor) -> Prediction:
    raw = pred.raw_boxes
    return Prediction(logits=pred.logits[idx], boxes=pred.boxes[idx],
                      raw_boxes=None if raw is None else raw[idx])


@torch.no_grad()
def collect_gaps(records: Sequence[ImageRecord], q0_predictions,
                 cfg: MethodConfig, head_type: str) -> Dict[str, np.ndarray]:
    """Collect the two-discrepancy triple per anchor, flattened over images.

    q0_predictions is either a Mapping image_id -> Prediction (the Stage-1
    outputs Q_0) or a sequence aligned with records.  Records missing rho_0
    are filled by running compute_priorities first (in place).  Returns a
    dict of flat 1-D numpy arrays over the *valid* anchors
    (partner >= 0 and rho_0 >= 0): g_D, g_Q, g_TQ, c, u0 (as recorded), and
    the derived u0_check = [g_TQ - g_D]_+ used for the consistency audit,
    plus the integer counts n_anchors and n_valid.
    """
    if any(r.rho0 is None for r in records):
        compute_priorities(records, q0_predictions, cfg, head_type,
                           in_place=True)
    is_map = isinstance(q0_predictions, Mapping)
    seq = None if is_map else list(q0_predictions)
    cols: Dict[str, list] = {k: [] for k in
                             ("g_D", "g_Q", "g_TQ", "c", "u0", "u0_check")}
    n_anchors = 0
    n_valid = 0
    for i, rec in enumerate(records):
        q0: Prediction = q0_predictions[rec.image_id] if is_map else seq[i]
        n_anchors += int(rec.anchors.numel())
        if (rec.partner is None or rec.rho0 is None
                or rec.teacher_pred is None):
            continue
        valid = (rec.partner >= 0) & (rec.rho0 >= 0)
        n_valid += int(valid.sum())
        if not bool(valid.any()):
            continue
        ap = _gather(rec.anchor_predictions(),
                     torch.nonzero(valid, as_tuple=False).flatten())
        tp = _gather(rec.teacher_pred, rec.partner[valid])
        qp = _gather(q0, rec.rho0[valid])
        g_d = aligned_discrepancy(tp, ap, cfg, head_type)
        g_q = aligned_discrepancy(ap, qp, cfg, head_type)
        g_tq = aligned_discrepancy(tp, qp, cfg, head_type)
        cols["g_D"].append(g_d.cpu())
        cols["g_Q"].append(g_q.cpu())
        cols["g_TQ"].append(g_tq.cpu())
        cols["c"].append(rec.c[valid].cpu())
        cols["u0"].append(rec.u0[valid].cpu())
        cols["u0_check"].append((g_tq - g_d).clamp_min(0.0).cpu())
    out: Dict[str, np.ndarray] = {}
    for k, v in cols.items():
        out[k] = (torch.cat(v).numpy() if v else
                  np.zeros(0, dtype=np.float32))
    out["n_anchors"] = np.int64(n_anchors)
    out["n_valid"] = np.int64(n_valid)
    return out


def _column_stats(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"n": 0.0, "mean": float("nan"), "median": float("nan"),
                "q1": float("nan"), "q3": float("nan")}
    q1, med, q3 = np.percentile(x, [25.0, 50.0, 75.0])
    return {"n": float(x.size), "mean": float(np.mean(x)),
            "median": float(med), "q1": float(q1), "q3": float(q3)}


def _top_overlap(a: np.ndarray, b: np.ndarray, frac: float = 0.2) -> float:
    """Overlap fraction between the top-`frac` rankings of two arrays."""
    n = a.size
    if n == 0:
        return float("nan")
    n20 = max(1, int(round(frac * n)))
    top_a = set(np.argsort(-a, kind="mergesort")[:n20].tolist())
    top_b = set(np.argsort(-b, kind="mergesort")[:n20].tolist())
    return len(top_a & top_b) / n20


def gap_statistics(gaps: Mapping[str, np.ndarray]) -> Dict[str, float]:
    """Summary statistics over the collected two-discrepancy gaps.

    Expects the dict returned by collect_gaps.  Reports the valid-anchor
    coverage, per-column medians with quartiles, the Spearman rank
    correlation rho(g_D, g_Q) with its p-value, the top-20% ranking
    overlap, and the u^0 consistency check
    max |u^0 - [g_TQ - g_D]_+|.
    """
    g_d = np.asarray(gaps["g_D"])
    g_q = np.asarray(gaps["g_Q"])
    n_anchors = int(gaps.get("n_anchors", 0))
    n_valid = int(gaps.get("n_valid", g_d.size))
    out: Dict[str, float] = {
        "n_anchors": float(n_anchors),
        "n_valid": float(n_valid),
        "coverage": float(n_valid / n_anchors) if n_anchors else 0.0,
    }
    for name in ("g_D", "g_Q", "g_TQ", "c", "u0"):
        for k, v in _column_stats(np.asarray(gaps[name])).items():
            out[f"{name}_{k}"] = v
    if g_d.size >= 2:
        rho, p = sps.spearmanr(g_d, g_q)
        out["spearman_gD_gQ"] = float(rho)
        out["spearman_p"] = float(p)
    else:
        out["spearman_gD_gQ"] = float("nan")
        out["spearman_p"] = float("nan")
    out["top20_overlap"] = _top_overlap(g_d, g_q)
    u0 = np.asarray(gaps["u0"])
    u0c = np.asarray(gaps["u0_check"])
    out["u0_consistency_max_abs"] = (float(np.max(np.abs(u0 - u0c)))
                                     if u0.size else float("nan"))
    return out
