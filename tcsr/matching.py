"""Anchor selection, feasible edges, and LSAP-based matchings.

Implements the selection and matching rules of Sec. 4.2 (Eqs. anchors, E_x,
tp_match, pq0):

- A_P / F_T: the threshold is applied *first*, then the top-K budget; ties
  in confidence are broken by ascending query index (stable descending sort).
- Feasible edges require the same predicted class and d <= tau_d.
- The teacher-partner matching is one-to-one and partial, maximizing
  cardinality first and then minimizing total cost.
- The anchor assignment rho: A_P -> [N_Q] is a deterministic LSAP solved on
  the dense pairwise discrepancy matrix; it is re-solved at every Stage-2
  forward pass (Eq. pq) and treated as a constant in backward.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .config import MethodConfig
from .losses import Prediction, anchor_confidence


def threshold_topk(scores: torch.Tensor, k: int,
                   threshold: float) -> torch.Tensor:
    """Indices of the top-k entries with score >= threshold (Eq. anchors).

    The threshold is applied before the top-K cap; ties in score are broken
    by ascending query index.  Returns a LongTensor of indices.
    """
    idx = torch.nonzero(scores >= threshold, as_tuple=False).flatten()
    if idx.numel() == 0:
        return idx
    order = torch.sort(scores[idx], descending=True, stable=True)[1]
    idx = idx[order]
    if k is not None and idx.numel() > k:
        idx = idx[:k]
    return idx


def select_anchors(pred: Prediction, cfg: MethodConfig, head_type: str
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Student anchor set A_P: top-K_P student queries with s_j^P >= tau_P.

    Returns (indices into pred, confidences s_j^P, foreground labels l_j^P).
    """
    s, l = anchor_confidence(pred.logits, cfg, head_type)
    idx = threshold_topk(s, cfg.k_p, cfg.tau_p)
    return idx, s[idx], l[idx]


def teacher_candidates(pred: Prediction, cfg: MethodConfig, head_type: str
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Teacher candidate set F_T: top-K_T teacher queries with s_i^T >= tau_T.

    Returns (indices into pred, confidences s_i^T, foreground labels l_i^T).
    """
    s, l = anchor_confidence(pred.logits, cfg, head_type)
    idx = threshold_topk(s, cfg.k_t, cfg.tau_t)
    return idx, s[idx], l[idx]


def feasible_edges(labels_t: torch.Tensor, labels_p: torch.Tensor,
                   pairwise_d: torch.Tensor, tau_d: float) -> torch.Tensor:
    """Feasible-edge indicator (Eq. E_x).

    An edge (i, j) between a teacher candidate i and a student anchor j is
    feasible iff they predict the same class (l_i^T = l_j^P) and
    d(y_i^T, y_j^P) <= tau_d.  Returns a BoolTensor of shape (|F_T|, |A_P|).
    """
    same = labels_t[:, None] == labels_p[None, :]
    close = pairwise_d <= tau_d
    return same & close


def max_cardinality_min_cost_matching(cost: torch.Tensor, feasible: torch.Tensor
                                      ) -> Tuple[np.ndarray, np.ndarray]:
    """One-to-one partial teacher-partner matching (Eq. tp_match).

    Maximizes the number of matched pairs first, then minimizes the total
    cost, over feasible edges only (filtering first prevents infeasible
    teacher candidates from occupying assignment endpoints).

    The lexicographic objective is realized by a single LSAP with an
    infeasibility penalty M = n + m + 2: every feasible total cost is at
    most min(n, m) < M, so removing one infeasible edge strictly decreases
    the objective; the LSAP therefore first maximizes feasibility
    (= cardinality), then minimizes the feasible cost.  Infeasible pairs in
    the returned solution are dropped.

    Returns (rows, cols): numpy int64 arrays of teacher-candidate and anchor
    indices (into the given sub-arrays), possibly empty.
    """
    empty = np.empty(0, dtype=np.int64)
    n, m = cost.shape
    if n == 0 or m == 0 or not bool(feasible.any()):
        return empty, empty.copy()
    big = float(n + m) + 2.0
    cost_np = cost.detach().to("cpu").numpy()
    feas_np = feasible.detach().to("cpu").numpy()
    cost_m = np.where(feas_np, cost_np, big)
    rows, cols = linear_sum_assignment(cost_m)
    keep = feas_np[rows, cols]
    return rows[keep].astype(np.int64), cols[keep].astype(np.int64)


@torch.no_grad()
def solve_anchor_assignment(pairwise: torch.Tensor) -> torch.Tensor:
    """Deterministic LSAP rho: A_P -> [N_Q] (Eqs. pq0 and pq).

    Minimizes sum_j d(y_j^P, y_{rho(j)}^Q) over injective maps.  If
    |A_P| > N_Q, the anchors left unassigned by the LSAP receive rho = -1;
    callers must guard with rho >= 0.  The assignment is treated as a
    constant in backward (re-solved under no_grad at every use).
    """
    k, n_q = pairwise.shape
    rho = torch.full((k,), -1, dtype=torch.long, device=pairwise.device)
    if k == 0 or n_q == 0:
        return rho
    cost = pairwise.detach().to("cpu").numpy()
    rows, cols = linear_sum_assignment(cost)
    r = torch.as_tensor(rows, dtype=torch.long, device=pairwise.device)
    c = torch.as_tensor(cols, dtype=torch.long, device=pairwise.device)
    rho[r] = c
    return rho
