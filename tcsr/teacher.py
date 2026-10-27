"""Teacher corroboration: cached per-image state and the detached bounded
corroboration coefficients (paper Sec. 4.2, Eqs. anchors, E_x, tp_match,
corroboration).

The frozen teacher T only ever enters the optimization through the detached,
bounded coefficients c_j in [0, 1]; all differentiable target values remain
student-valued.  Unmatched anchors receive c_j = 0.  The per-image cache
follows Algorithm 1, lines 1-4: cache the student predictions P, build A_P,
then (if a teacher exists) cache T, solve the feasible-edge matching, and
cache c; otherwise c = 0 for every anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from .config import MethodConfig
from .losses import Prediction, pairwise_discrepancy
from .matching import (feasible_edges, max_cardinality_min_cost_matching,
                       select_anchors, teacher_candidates)


@dataclass
class ImageRecord:
    """Cached per-image state (Algorithm 1, lines 1-4).

    student_pred / teacher_pred hold the *full* frozen prediction sets of
    the full-precision student P and the frozen teacher T (None when no
    teacher exists).  The trainable quantized student Q is never stored
    here; Stage 2 recomputes it every minibatch.
    """

    image_id: str
    student_pred: Prediction                       # full student set (N_P)
    teacher_pred: Optional[Prediction] = None      # full teacher set (N_T)
    anchors: Optional[torch.Tensor] = None         # (K,) long, into student_pred
    anchor_scores: Optional[torch.Tensor] = None   # (K,) s_j^P
    anchor_labels: Optional[torch.Tensor] = None   # (K,) l_j^P
    c: Optional[torch.Tensor] = None               # (K,) detached, in [0, 1]
    partner: Optional[torch.Tensor] = None         # (K,) long, into teacher_pred
    # Stage-1 derived quantities (frozen across all Stage-2 variants):
    rho0: Optional[torch.Tensor] = None            # (K,) long, Q_0 assignment
    u0: Optional[torch.Tensor] = None              # (K,) un-corroborated priority
    g_q0: Optional[torch.Tensor] = None            # (K,) Stage-1 difficulty g_Q
    # Active-variant coefficients (set by variants.assign_variant_weights):
    w: Optional[torch.Tensor] = None               # (K,) frozen variant weights

    def anchor_predictions(self) -> Prediction:
        """Student predictions restricted to A_P (frozen target values)."""
        i = self.anchors
        raw = self.student_pred.raw_boxes
        return Prediction(logits=self.student_pred.logits[i],
                          boxes=self.student_pred.boxes[i],
                          raw_boxes=None if raw is None else raw[i])

    def corroborated(self) -> torch.Tensor:
        """Bool mask of V_x = {j in A_P : c_j > 0} over the anchor axis."""
        return self.c > 0

    def partner_predictions(self) -> Tuple[torch.Tensor, Prediction]:
        """(mask, partner predictions) for the matched anchors.

        mask is the Bool mask over A_P with partner >= 0; the Prediction
        contains the matched teacher candidates, ordered by anchor index.
        """
        mask = self.partner >= 0
        idx = self.partner[mask]
        tp = self.teacher_pred
        raw = tp.raw_boxes
        pred = Prediction(logits=tp.logits[idx], boxes=tp.boxes[idx],
                          raw_boxes=None if raw is None else raw[idx])
        return mask, pred


@torch.no_grad()
def solve_corroboration(student_pred: Prediction, teacher_pred: Prediction,
                        cfg: MethodConfig, head_type: str
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                   torch.Tensor, torch.Tensor]:
    """Solve the teacher-student corroboration for one image.

    Returns (anchor indices, anchor scores, anchor labels, c, partner).
    """
    a_idx, s_p, l_p = select_anchors(student_pred, cfg, head_type)
    k = a_idx.numel()
    dev = a_idx.device
    c = torch.zeros(k, device=dev)
    partner = torch.full((k,), -1, dtype=torch.long, device=dev)
    if k == 0:
        return a_idx, s_p, l_p, c, partner
    t_idx, s_t, l_t = teacher_candidates(teacher_pred, cfg, head_type)
    if t_idx.numel() == 0:
        return a_idx, s_p, l_p, c, partner
    # Dense pairwise discrepancy between teacher candidates and anchors.
    y_p = Prediction(logits=student_pred.logits[a_idx],
                     boxes=student_pred.boxes[a_idx])
    y_t = Prediction(logits=teacher_pred.logits[t_idx],
                     boxes=teacher_pred.boxes[t_idx])
    d = pairwise_discrepancy(y_t, y_p, cfg, head_type)      # (|F_T|, |A_P|)
    feas = feasible_edges(l_t, l_p, d, cfg.tau_d)
    rows, cols = max_cardinality_min_cost_matching(d, feas)
    if len(rows):
        r = torch.as_tensor(rows, dtype=torch.long, device=dev)
        j = torch.as_tensor(cols, dtype=torch.long, device=dev)
        # Eq. corroboration: c_j = sg(sqrt(s_iT * s_jP) * (1 - d(y_iT, y_jP)))
        c[j] = torch.sqrt(s_t[r] * s_p[j]) * (1.0 - d[r, j])
        partner[j] = t_idx[r]
    return a_idx, s_p, l_p, c, partner


def make_record(image_id: str, student_pred: Prediction,
                teacher_pred: Optional[Prediction], cfg: MethodConfig,
                head_type: str) -> ImageRecord:
    """Build the per-image cache entry (Algorithm 1, lines 1-4).

    With no teacher, anchors are still selected from P and c = 0.
    """
    if teacher_pred is None:
        a_idx, s_p, l_p = select_anchors(student_pred, cfg, head_type)
        dev = a_idx.device
        c = torch.zeros(a_idx.numel(), device=dev)
        partner = torch.full((a_idx.numel(),), -1, dtype=torch.long, device=dev)
    else:
        a_idx, s_p, l_p, c, partner = solve_corroboration(
            student_pred, teacher_pred, cfg, head_type)
    return ImageRecord(image_id=image_id, student_pred=student_pred,
                       teacher_pred=teacher_pred, anchors=a_idx,
                       anchor_scores=s_p, anchor_labels=l_p,
                       c=c, partner=partner)


def _pred_to(p: Prediction, device: torch.device) -> Prediction:
    return Prediction(logits=p.logits.to(device), boxes=p.boxes.to(device),
                      raw_boxes=None if p.raw_boxes is None
                      else p.raw_boxes.to(device))


class TeacherCache:
    """Per-image cache of frozen student/teacher predictions and derived
    state (Algorithm 1).  Indexable by position or by image_id."""

    def __init__(self, records: List[ImageRecord]):
        self._records = list(records)
        self._by_id: Dict[str, ImageRecord] = {r.image_id: r
                                               for r in self._records}

    @classmethod
    def build(cls, items: Iterable[Tuple[str, Prediction, Optional[Prediction]]],
              cfg: MethodConfig, head_type: str) -> "TeacherCache":
        """items: iterable of (image_id, student_pred, teacher_pred | None)."""
        return cls([make_record(i, sp, tp, cfg, head_type)
                    for i, sp, tp in items])

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._by_id[key]
        return self._records[key]

    def get(self, image_id: str, default=None) -> Optional[ImageRecord]:
        return self._by_id.get(image_id, default)

    def has_teacher(self, image_id: str) -> bool:
        rec = self._by_id.get(image_id)
        return rec is not None and rec.teacher_pred is not None

    def to(self, device) -> "TeacherCache":
        """Move all cached tensors to device (in place)."""
        for rec in self._records:
            rec.student_pred = _pred_to(rec.student_pred, device)
            if rec.teacher_pred is not None:
                rec.teacher_pred = _pred_to(rec.teacher_pred, device)
            for name in ("anchors", "partner", "rho0"):
                t = getattr(rec, name)
                if t is not None:
                    setattr(rec, name, t.to(device))
            for name in ("anchor_scores", "anchor_labels", "c", "u0",
                         "g_q0", "w"):
                t = getattr(rec, name)
                if t is not None:
                    setattr(rec, name, t.to(device))
        return self

    def stats(self) -> Dict[str, float]:
        """Corroboration statistics over the cache."""
        n_img = len(self._records)
        n_anchors = sum(int(r.anchors.numel()) for r in self._records)
        n_matched = sum(int((r.partner >= 0).sum()) for r in self._records)
        n_corr = sum(int((r.c > 0).sum()) for r in self._records)
        c_vals = torch.cat([r.c for r in self._records]) if n_anchors else \
            torch.zeros(0)
        pos = c_vals[c_vals > 0]
        return {
            "images": float(n_img),
            "anchors": float(n_anchors),
            "corroborated_anchors": float(n_corr),
            "matched_anchors": float(n_matched),
            "matched_coverage": float(n_corr / n_anchors) if n_anchors else 0.0,
            "mean_c_on_corroborated": float(pos.mean()) if pos.numel() else 0.0,
        }
