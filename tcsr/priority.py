"""Un-corroborated-anchor priorities u^0 from the Stage-1 assignment
(paper Sec. 4.2, Eq. priority).

    u_j^0 = sg([d(y_{iota(j)}^T, y_{rho0*(j)}^{Q0}) - d(y_{iota(j)}^T, y_j^P)]_+)

for anchors with a matched teacher partner iota(j) and a valid rho_0
assignment; u_j^0 = 0 otherwise (un-corroborated).  The priorities are
computed once from the same Q_0 for every Stage-2 variant (B-I share Q_0,
A_P, and the P/Q matching) and then frozen.  The same pass records the
Stage-1 difficulty g_Q(j) = d(y_j^P, y_{rho0*(j)}^{Q0}) used by the C_Q
variant and the two-discrepancy diagnostics.
"""

from __future__ import annotations

from typing import List, Mapping, Sequence, Tuple

import torch

from .config import MethodConfig
from .losses import Prediction, aligned_discrepancy, pairwise_discrepancy
from .matching import solve_anchor_assignment
from .teacher import ImageRecord


def _gather(pred: Prediction, idx: torch.Tensor) -> Prediction:
    raw = pred.raw_boxes
    return Prediction(logits=pred.logits[idx], boxes=pred.boxes[idx],
                      raw_boxes=None if raw is None else raw[idx])


@torch.no_grad()
def compute_priorities(records: Sequence[ImageRecord], q0_predictions,
                       cfg: MethodConfig, head_type: str, in_place: bool = True
                       ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Solve rho_0*, record g_Q and u^0 for each record.

    q0_predictions is either a Mapping image_id -> Prediction (the Stage-1
    outputs Q_0) or a sequence aligned with records.  Returns a list of
    (u0, g_q0, rho0); with in_place=True the values are also stored on the
    records (rec.u0, rec.g_q0, rec.rho0).
    """
    is_map = isinstance(q0_predictions, Mapping)
    seq = None if is_map else list(q0_predictions)
    out: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for i, rec in enumerate(records):
        q0: Prediction = q0_predictions[rec.image_id] if is_map else seq[i]
        a_pred = rec.anchor_predictions()                    # (K,) targets
        D = pairwise_discrepancy(a_pred, q0, cfg, head_type)  # (K, N_Q)
        rho0 = solve_anchor_assignment(D)                    # (K,) long
        k = a_pred.logits.shape[0]
        dev = D.device
        g_q = torch.zeros(k, device=dev)
        valid = rho0 >= 0
        if k:
            ar = torch.arange(k, device=dev)
            g_q[valid] = D[ar[valid], rho0[valid]]
        # u^0 only on anchors with a matched teacher partner and valid rho_0.
        u0 = torch.zeros(k, device=dev)
        vm = valid & (rec.partner >= 0)
        if k and bool(vm.any()):
            tp = _gather(rec.teacher_pred, rec.partner[vm])
            ap = _gather(a_pred, torch.nonzero(vm, as_tuple=False).flatten())
            d_t_a = aligned_discrepancy(tp, ap, cfg, head_type)
            d_t_q = aligned_discrepancy(tp, _gather(q0, rho0[vm]),
                                        cfg, head_type)
            u0[vm] = (d_t_q - d_t_a).clamp_min(0.0)
        if in_place:
            rec.rho0 = rho0
            rec.u0 = u0
            rec.g_q0 = g_q
        out.append((u0, g_q, rho0))
    return out
