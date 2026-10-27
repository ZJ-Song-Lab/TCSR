"""Diagnostics tests: structure audit, gap statistics, direction conflict."""

import numpy as np
import pytest
import torch

from tcsr.config import MethodConfig
from tcsr.diagnostics import (COMPONENTS, assignment_change, collect_gaps,
                              direction_conflict, gap_statistics,
                              structure_audit)
from tcsr.losses import Prediction
from tcsr.priority import compute_priorities
from tcsr.teacher import TeacherCache, make_record


def _pred_pair(seed=0, n=6, c=1):
    """Confident, well-separated student/teacher predictions."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.full((n, c + 1), -6.0)
    for i in range(n):
        logits[i, i % c] = 6.0
    sp = Prediction(logits=logits.clone(),
                    boxes=torch.rand(n, 4, generator=g) * 0.3 + 0.35)
    tp = Prediction(logits=logits.clone(),
                    boxes=torch.rand(n, 4, generator=g) * 0.3 + 0.35)
    return sp, tp


def _cache(n_images=2):
    cfg = MethodConfig()
    items = []
    for i in range(n_images):
        sp, tp = _pred_pair(seed=i)
        items.append((f"img_{i}", sp, tp))
    return TeacherCache.build(items, cfg, "softmax")


def _q0_predictions(cache):
    g = torch.Generator().manual_seed(7)
    out = {}
    for rec in cache:
        n = rec.student_pred.logits.shape[0]
        logits = rec.student_pred.logits + torch.randn(n, 2, generator=g) * 0.3
        boxes = (rec.student_pred.boxes
                 + torch.randn(n, 4, generator=g) * 0.02).clamp(0.05, 0.95)
        out[rec.image_id] = Prediction(logits=logits, boxes=boxes)
    return out


class TestStructureAudit:
    def test_keys_and_counts(self):
        cache = _cache()
        records = list(cache)
        s = structure_audit(records)
        for key in ("images", "anchors", "matched", "unmatched", "assigned",
                    "valid", "matched_coverage", "valid_coverage", "sum_c",
                    "mean_c", "sum_u0", "sum_w", "ess", "ess_ratio"):
            assert key in s
        assert s["images"] == 2.0
        assert s["anchors"] > 0
        assert s["matched"] == s["anchors"] - s["unmatched"]

    def test_no_weights_nan(self):
        s = structure_audit(list(_cache()))
        assert np.isnan(s["sum_w"]) and np.isnan(s["ess"])

    def test_ess_uniform(self):
        records = list(_cache())
        for rec in records:
            rec.w = torch.ones(rec.anchors.numel())
        s = structure_audit(records)
        # uniform weights: ESS == count, ratio == 1
        assert s["ess"] == pytest.approx(s["anchors"])
        assert s["ess_ratio"] == pytest.approx(1.0)


class TestAssignmentChange:
    def test_identical_runs(self):
        cache = _cache()
        q0 = _q0_predictions(cache)
        records = list(cache)
        out = compute_priorities(records, q0, MethodConfig(), "softmax")
        rho0_list = [t[2] for t in out]
        same = [list(rho0_list)]
        s = assignment_change(rho0_list, same)
        assert s["n_runs"] == 1.0
        assert s["per_transition_mean"] == pytest.approx(0.0)
        assert s["at_least_once"] == pytest.approx(0.0)
        assert s["n_mismatched"] == 0.0

    def test_changed_runs(self):
        cache = _cache()
        q0 = _q0_predictions(cache)
        records = list(cache)
        out = compute_priorities(records, q0, MethodConfig(), "softmax")
        rho0_list = [t[2] for t in out]
        # permute every assignment by shifting valid entries
        changed = []
        for rho in rho0_list:
            r = rho.clone()
            valid = r >= 0
            vals = r[valid].roll(1)
            r[valid] = vals
            changed.append(r)
        s = assignment_change(rho0_list, [changed])
        assert s["per_transition_mean"] > 0.0

    def test_no_runs(self):
        s = assignment_change([], [])
        assert s["n_runs"] == 0.0


class TestGapStatistics:
    def test_collect_and_summarize(self):
        cache = _cache()
        q0 = _q0_predictions(cache)
        records = list(cache)
        # collect_gaps fills missing rho0 via compute_priorities
        gaps = collect_gaps(records, q0, MethodConfig(), "softmax")
        for key in ("g_D", "g_Q", "g_TQ", "c", "u0", "u0_check",
                    "n_anchors", "n_valid"):
            assert key in gaps
        assert int(gaps["n_valid"]) > 0
        s = gap_statistics(gaps)
        assert 0.0 < s["coverage"] <= 1.0
        assert s["n_valid"] == int(gaps["n_valid"])
        # u0 consistency: recorded u0 == [g_TQ - g_D]_+
        assert s["u0_consistency_max_abs"] == pytest.approx(0.0, abs=1e-5)
        # triangle-inequality consistency of the triple
        g_d = gaps["g_D"]
        g_q = gaps["g_Q"]
        g_tq = gaps["g_TQ"]
        assert np.all(g_d >= -1e-6) and np.all(g_q >= -1e-6)
        assert np.all(g_tq >= -1e-6)

    def test_identical_student_teacher_zero_gd(self):
        cfg = MethodConfig()
        sp, _ = _pred_pair(seed=3)
        items = [("img", sp, sp)]
        cache = TeacherCache.build(items, cfg, "softmax")
        q0 = _q0_predictions(cache)
        gaps = collect_gaps(list(cache), q0, cfg, "softmax")
        # teacher == student -> g_D == 0 exactly
        assert np.allclose(gaps["g_D"], 0.0, atol=1e-6)


class TestDirectionConflict:
    def test_structure(self):
        cache = _cache()
        q0 = _q0_predictions(cache)
        records = list(cache)
        compute_priorities(records, q0, MethodConfig(), "softmax")
        stats = direction_conflict(records, q0, MethodConfig(), "softmax")
        assert stats["anchors_total"] > 0
        assert stats["anchors_valid"] > 0
        assert set(COMPONENTS) == {"cls", "l1", "giou", "all"}
        for comp in COMPONENTS:
            entry = stats[comp]
            assert set(entry) == {"n", "zero_norm", "pr_negative", "median"}
            assert entry["n"] + entry["zero_norm"] == stats["anchors_valid"]
            assert 0.0 <= entry["pr_negative"] <= 1.0

    def test_identical_targets_no_conflict(self):
        # teacher == student -> g^P and g^T are identical -> kappa == 1
        cfg = MethodConfig()
        # Unsaturated logits (|z| = 1): softmax(1) = 0.88 > tau_p keeps all
        # anchors selected while keeping cls gradients (~1e-2) far above the
        # eps_g noise floor.  Saturated logits (|z| = 6) would drive the cls
        # gradient norm (~1e-6) down to the eps_g scale, so the cosine's
        # (n_p * n_t + eps_g) denominator would dominate and mask kappa == 1.
        g = torch.Generator().manual_seed(5)
        logits = torch.full((6, 2), -1.0)
        logits[:, 0] = 1.0
        sp = Prediction(logits=logits,
                        boxes=torch.rand(6, 4, generator=g) * 0.3 + 0.35)
        cache = TeacherCache.build([("img", sp, sp)], cfg, "softmax")
        q0 = _q0_predictions(cache)
        records = list(cache)
        compute_priorities(records, q0, cfg, "softmax")
        stats = direction_conflict(records, q0, cfg, "softmax")
        for comp in COMPONENTS:
            assert stats[comp]["median"] == pytest.approx(1.0, abs=1e-5)
            assert stats[comp]["pr_negative"] == pytest.approx(0.0, abs=1e-6)
