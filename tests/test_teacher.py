"""Teacher corroboration and per-image cache tests."""

import pytest
import torch

from tcsr.config import MethodConfig
from tcsr.losses import Prediction
from tcsr.teacher import (ImageRecord, TeacherCache, make_record,
                          solve_corroboration)


def _pred(logits, boxes, seed=None):
    return Prediction(logits=logits, boxes=boxes)


def _toy_pair(n=6, c=2, seed=0):
    """A student/teacher prediction pair with clearly confident queries."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.full((n, c + 1), -5.0)
    fg = torch.randint(0, c, (n,), generator=g)
    for i in range(n):
        logits[i, fg[i]] = 5.0
    logits[:, -1] = -5.0                       # background suppressed
    boxes = torch.rand(n, 4, generator=g) * 0.3 + 0.3
    return _pred(logits, boxes)


class TestSolveCorroboration:
    def test_identical_predictions_full_c(self):
        cfg = MethodConfig()
        p = _toy_pair(seed=0)
        a_idx, s_p, l_p, c, partner = solve_corroboration(p, p, cfg, "softmax")
        assert a_idx.numel() > 0
        # Identical teacher/student: d = 0 -> c = sqrt(s*s) = s.
        s_t = torch.softmax(p.logits, -1)[:, :-1].max(-1).values[a_idx]
        assert torch.allclose(c, s_t, atol=1e-5)
        assert bool((partner >= 0).all())

    def test_empty_anchor_set(self):
        cfg = MethodConfig(tau_p=0.99)
        p = _pred(torch.tensor([[-5.0, -5.0, 5.0]]), torch.rand(1, 4))
        t = _toy_pair()
        a_idx, s_p, l_p, c, partner = solve_corroboration(p, t, cfg, "softmax")
        assert a_idx.numel() == 0
        assert c.numel() == 0 and partner.numel() == 0

    def test_no_teacher_candidates(self):
        cfg = MethodConfig()
        p = _toy_pair()
        t = _pred(torch.tensor([[-5.0, -5.0, 5.0]]), torch.rand(1, 4))
        a_idx, s_p, l_p, c, partner = solve_corroboration(p, t, cfg, "softmax")
        assert a_idx.numel() > 0
        assert bool((c == 0).all()) and bool((partner == -1).all())


class TestMakeRecord:
    def test_no_teacher_c_zero(self):
        cfg = MethodConfig()
        sp = _toy_pair()
        rec = make_record("img", sp, None, cfg, "softmax")
        assert rec.teacher_pred is None
        assert bool((rec.c == 0).all())
        assert bool((rec.partner == -1).all())
        assert rec.anchors.numel() > 0

    def test_with_teacher(self):
        cfg = MethodConfig()
        sp = _toy_pair(seed=0)
        tp = _toy_pair(seed=1)
        rec = make_record("img", sp, tp, cfg, "softmax")
        assert rec.teacher_pred is not None
        assert bool((rec.c >= 0).all()) and bool((rec.c <= 1).all())

    def test_anchor_predictions_subset(self):
        cfg = MethodConfig()
        sp = _toy_pair()
        rec = make_record("img", sp, None, cfg, "softmax")
        ap = rec.anchor_predictions()
        assert ap.logits.shape[0] == rec.anchors.numel()
        assert torch.equal(ap.logits, sp.logits[rec.anchors])

    def test_corroborated_mask(self):
        cfg = MethodConfig()
        sp = _toy_pair()
        rec = make_record("img", sp, None, cfg, "softmax")
        assert not bool(rec.corroborated().any())


class TestTeacherCache:
    def test_build_and_indexing(self):
        cfg = MethodConfig()
        items = [(f"img_{i}", _toy_pair(seed=i), _toy_pair(seed=i + 100))
                 for i in range(3)]
        cache = TeacherCache.build(items, cfg, "softmax")
        assert len(cache) == 3
        assert cache[0] is cache["img_0"]
        assert cache.get("img_1") is cache["img_1"]
        assert cache.get("missing") is None
        assert cache.has_teacher("img_0") is True
        assert [r.image_id for r in cache] == ["img_0", "img_1", "img_2"]

    def test_stats_keys(self):
        cfg = MethodConfig()
        items = [(f"img_{i}", _toy_pair(seed=i), _toy_pair(seed=i + 100))
                 for i in range(2)]
        cache = TeacherCache.build(items, cfg, "softmax")
        s = cache.stats()
        assert set(s) == {"images", "anchors", "corroborated_anchors",
                          "matched_anchors", "matched_coverage",
                          "mean_c_on_corroborated"}
        assert s["images"] == 2.0
        assert s["anchors"] > 0

    def test_stats_no_teacher(self):
        cfg = MethodConfig()
        items = [(f"img_{i}", _toy_pair(seed=i), None) for i in range(2)]
        cache = TeacherCache.build(items, cfg, "softmax")
        s = cache.stats()
        assert s["corroborated_anchors"] == 0.0
        assert s["matched_anchors"] == 0.0
        assert s["matched_coverage"] == 0.0
        assert s["mean_c_on_corroborated"] == 0.0

    def test_to_device(self):
        cfg = MethodConfig()
        items = [("img", _toy_pair(), None)]
        cache = TeacherCache.build(items, cfg, "softmax")
        out = cache.to("cpu")
        assert out is cache
