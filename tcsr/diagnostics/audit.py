"""Structural audit of the per-image cache and assignment stability.

structure_audit summarizes the anchor bookkeeping of a TeacherCache /
record list: anchor counts, the double-mapped valid set (feasible teacher
partner AND valid rho_0), the corroboration mass, and the effective sample
size of the frozen variant weights,

    ESS = (sum_j w_j)^2 / sum_j w_j^2,

reported both absolutely and as a ratio to the number of weighted
anchors.  assignment_change measures how often the anchor-to-Q assignment
moves between the frozen Stage-1 rho_0 and the assignments re-solved on
the reconstructed students across seed runs.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

from ..teacher import ImageRecord


def structure_audit(records: Sequence[ImageRecord]) -> Dict[str, float]:
    """Anchor / corroboration / weight summary over the cached records.

    matched: anchors with a feasible teacher partner (c > 0 by Eq.
    corroboration); assigned: anchors with a valid rho_0; valid: the
    double-mapped intersection used by the diagnostics; ess_ratio is the
    ESS of the variant weights divided by the number of weighted anchors
    (NaN when no variant weights have been assigned yet).
    """
    n_img = len(records)
    n_anchors = 0
    n_matched = 0
    n_assigned = 0
    n_valid = 0
    sum_c = 0.0
    sum_u0 = 0.0
    w_all: List[torch.Tensor] = []
    for rec in records:
        if rec.anchors is None:
            continue
        k = int(rec.anchors.numel())
        n_anchors += k
        matched = rec.partner >= 0 if rec.partner is not None else \
            torch.zeros(k, dtype=torch.bool)
        n_matched += int(matched.sum())
        assigned = rec.rho0 >= 0 if rec.rho0 is not None else \
            torch.zeros(k, dtype=torch.bool)
        n_assigned += int(assigned.sum())
        n_valid += int((matched & assigned).sum())
        if rec.c is not None:
            sum_c += float(rec.c.sum())
        if rec.u0 is not None:
            sum_u0 += float(rec.u0.sum())
        if rec.w is not None:
            w_all.append(rec.w.detach().cpu())
    sum_w = ess = ess_ratio = float("nan")
    if w_all:
        w = torch.cat(w_all).to(torch.float64)
        sum_w = float(w.sum())
        ess = float(w.sum() ** 2 / w.pow(2).sum())
        ess_ratio = float(ess / w.numel()) if w.numel() else float("nan")
    return {
        "images": float(n_img),
        "anchors": float(n_anchors),
        "matched": float(n_matched),
        "unmatched": float(n_anchors - n_matched),
        "assigned": float(n_assigned),
        "valid": float(n_valid),
        "matched_coverage": float(n_matched / n_anchors) if n_anchors else 0.0,
        "valid_coverage": float(n_valid / n_anchors) if n_anchors else 0.0,
        "sum_c": sum_c,
        "mean_c": float(sum_c / n_matched) if n_matched else 0.0,
        "sum_u0": sum_u0,
        "sum_w": sum_w,
        "ess": ess,
        "ess_ratio": ess_ratio,
    }


def assignment_change(rho0_list: Sequence[torch.Tensor],
                      rho_runs: Sequence[Sequence[torch.Tensor]]
                      ) -> Dict[str, float]:
    """Anchor-assignment stability between the frozen rho_0 and the
    re-solved assignments of the reconstructed students.

    rho0_list: per-record baseline (from compute_priorities).  rho_runs:
    one entry per seed run, each a per-record list of assignment tensors
    aligned with rho0_list.  per_transition_mean is the mean fraction of
    reassigned anchors over all (record, run) transitions;
    at_least_once is the fraction of baseline-assigned anchors that move
    in at least one run; n_mismatched counts (record, run) pairs skipped
    because of shape disagreement.
    """
    n_runs = len(rho_runs)
    n_mismatched = 0
    total_valid = 0
    total_changed = 0
    any_changed = 0
    if n_runs == 0:
        return {"n_runs": 0.0, "per_transition_mean": float("nan"),
                "at_least_once": float("nan"), "n_mismatched": 0.0}
    per_run_rates: List[float] = []
    for rec_i, rho0 in enumerate(rho0_list):
        if rho0 is None:
            continue
        base_valid = rho0 >= 0
        n_valid = int(base_valid.sum())
        if n_valid == 0:
            continue
        total_valid += n_valid
        moved_any = torch.zeros_like(base_valid)
        for run in rho_runs:
            if rec_i >= len(run):
                n_mismatched += 1
                continue
            rho = run[rec_i]
            if rho is None or rho.shape != rho0.shape:
                n_mismatched += 1
                continue
            changed = base_valid & (rho != rho0)
            n_changed = int(changed.sum())
            total_changed += n_changed
            per_run_rates.append(n_changed / n_valid)
            moved_any |= changed
        any_changed += int(moved_any.sum())
    return {
        "n_runs": float(n_runs),
        "per_transition_mean": (float(np.mean(per_run_rates))
                                if per_run_rates else float("nan")),
        "at_least_once": float(any_changed / total_valid) if total_valid
        else float("nan"),
        "total_valid_anchors": float(total_valid),
        "n_mismatched": float(n_mismatched),
    }
