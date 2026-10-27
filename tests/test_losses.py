"""Bounded class-box discrepancy, residual MMD, and block-loss tests."""

import math

import pytest
import torch

from tcsr.config import MethodConfig
from tcsr.losses import (Prediction, aligned_discrepancy,
                         bernoulli_js, block_losses, class_probs,
                         d_giou, d_l1, discrepancy, discrepancy_components,
                         giou, median_bandwidth, normalized_block_loss,
                         pairwise_discrepancy, residual_mmd, ResidualMMD)


def _pred(n, c, seed=0, head="softmax"):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, c, generator=g)
    boxes = torch.rand(n, 4, generator=g)
    return Prediction(logits=logits, boxes=boxes)


class TestClassProbs:
    def test_sigmoid_smoothing(self):
        cfg = MethodConfig()
        z = torch.zeros(2, 3)
        p = class_probs(z, cfg, "sigmoid")
        a = cfg.epsilon + (1 - 2 * cfg.epsilon) * 0.5
        assert torch.allclose(p, torch.full((2, 3), a))

    def test_softmax_full(self):
        cfg = MethodConfig()
        z = torch.tensor([[1.0, 2.0, 0.5]])
        p = class_probs(z, cfg, "softmax")
        e = torch.exp(z - z.max())
        assert torch.allclose(p, e / e.sum())


class TestBernoulliJS:
    def test_bounds_and_zeros(self):
        a = torch.tensor([0.3, 0.5, 0.9])
        b = torch.tensor([0.3, 0.5, 0.9])
        assert torch.allclose(bernoulli_js(a, b), torch.zeros(3), atol=1e-8)
        c = torch.tensor([0.1, 0.9, 0.1])
        j = bernoulli_js(a, c)
        assert bool((j >= 0).all()) and bool((j <= 1.0).all())

    def test_symmetry(self):
        a = torch.tensor([0.2, 0.7])
        b = torch.tensor([0.6, 0.3])
        assert torch.allclose(bernoulli_js(a, b), bernoulli_js(b, a))


class TestDiscrepancy:
    def test_zero_for_identical(self):
        cfg = MethodConfig()
        y = _pred(5, 2)
        d = discrepancy(y, y, cfg, "softmax")
        assert d.item() == pytest.approx(0.0, abs=1e-6)

    def test_bounded(self):
        cfg = MethodConfig()
        y1 = _pred(4, 2, seed=1)
        y2 = _pred(4, 2, seed=2)
        d = discrepancy(y1, y2, cfg, "softmax")
        assert 0.0 <= d.item() <= 1.0

    def test_sigmoid_head(self):
        cfg = MethodConfig()
        y1 = _pred(4, 3, seed=1)
        y2 = _pred(4, 3, seed=2)
        d = discrepancy(y1, y2, cfg, "sigmoid")
        assert 0.0 <= d.item() <= 1.0

    def test_components_keys(self):
        cfg = MethodConfig()
        y1 = _pred(3, 2)
        y2 = _pred(3, 2, seed=9)
        comps = discrepancy_components(y1, y2, cfg, "softmax")
        assert set(comps) == {"cls", "l1", "giou", "total"}
        total = (cfg.eta_cls * comps["cls"] + cfg.eta_l1 * comps["l1"]
                 + cfg.eta_giou * comps["giou"])
        assert comps["total"].item() == pytest.approx(total.item(), rel=1e-6)

    def test_pairwise_shape_and_diag(self):
        cfg = MethodConfig()
        y1 = _pred(4, 2, seed=3)
        y2 = _pred(6, 2, seed=4)
        D = pairwise_discrepancy(y1, y2, cfg, "softmax")
        assert D.shape == (4, 6)
        assert bool((D >= 0).all()) and bool((D <= 1.0).all())

    def test_aligned_equals_mean_of_pairwise_diag(self):
        cfg = MethodConfig()
        y1 = _pred(5, 2, seed=5)
        y2 = _pred(5, 2, seed=6)
        a = aligned_discrepancy(y1, y2, cfg, "softmax")
        D = pairwise_discrepancy(y1, y2, cfg, "softmax")
        assert torch.allclose(a, D.diagonal(), atol=1e-6)


class TestBoxTerms:
    def test_d_l1(self):
        b1 = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        b2 = torch.tensor([[0.5, 0.5, 0.3, 0.2]])
        # |dw| / 4 = 0.1 / 4
        assert d_l1(b1, b2).item() == pytest.approx(0.025)

    def test_giou_identical(self):
        b = torch.rand(4)
        assert giou(b, b).item() == pytest.approx(1.0, abs=1e-6)

    def test_giou_disjoint_small_boxes(self):
        b1 = torch.tensor([0.2, 0.2, 0.1, 0.1])
        b2 = torch.tensor([0.8, 0.8, 0.1, 0.1])
        # Two disjoint 0.1 x 0.1 boxes in a 0.7 x 0.7 enclosing box:
        # iou = 0, union = 0.02, enc = 0.49 -> giou = -0.47/0.49
        assert giou(b1, b2).item() == pytest.approx(-0.47 / 0.49, abs=1e-4)

    def test_d_giou_range(self):
        b1 = torch.rand(5, 4)
        b2 = torch.rand(5, 4)
        d = d_giou(b1, b2)
        assert 0.0 <= d.item() <= 1.0


class TestResidualMMD:
    def test_same_distribution_zero(self):
        v = torch.rand(16, 3)
        w = v.clone()
        assert residual_mmd(v, w, sigma=0.1).item() == pytest.approx(0.0, abs=1e-8)

    def test_different_distributions_positive(self):
        v = torch.zeros(8, 3)
        w = torch.ones(8, 3) * 0.5
        assert residual_mmd(v, w, sigma=0.1).item() > 0.1

    def test_empty_set(self):
        v = torch.rand(4, 3)
        e = torch.zeros(0, 3)
        assert residual_mmd(v, e, 0.1).item() == 0.0
        assert residual_mmd(e, v, 0.1).item() == 0.0

    def test_module(self):
        m = ResidualMMD(0.084)
        v = torch.rand(8, 3)
        w = torch.rand(8, 3)
        assert m(v, w).item() == residual_mmd(v, w, 0.084).item()

    def test_median_bandwidth_single_point(self):
        v = torch.zeros(1, 3)
        assert median_bandwidth(v) == 0.020

    def test_median_bandwidth_lower_bound(self):
        v = torch.tensor([[0.0, 0.0], [1e-6, 0.0]])
        assert median_bandwidth(v) == 0.020

    def test_median_bandwidth_scale(self):
        v = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
        # distance 5 -> median 5 -> max(5, 0.02) = 5
        assert median_bandwidth(v) == pytest.approx(5.0)


class TestBlockLoss:
    def test_normalized_block_loss(self):
        p = torch.ones(4)
        q = torch.zeros(4)
        # ||q - p||^2 / ||p||^2 = 4 / 4
        assert normalized_block_loss(q, p).item() == pytest.approx(1.0)

    def test_block_losses_sum(self):
        p1 = torch.ones(2)
        q1 = torch.zeros(2)
        p2 = torch.tensor([2.0, 2.0])
        q2 = torch.tensor([1.0, 1.0])
        total, n = block_losses([q1, q2], [p1, p2])
        assert n == 2
        # ||q1 - p1||^2 / ||p1||^2 = 4 / 4 = 1.0
        # ||q2 - p2||^2 / ||p2||^2 = 2 / 8 = 0.25
        assert total.item() == pytest.approx(1.0 + 0.25)


class TestAnchorConfidence:
    def test_softmax_excludes_background(self):
        from tcsr.losses import anchor_confidence
        cfg = MethodConfig()
        logits = torch.tensor([[2.0, -5.0, -5.0],   # fg0 dominant
                               [-5.0, -5.0, 3.0]])  # background dominant
        s, l = anchor_confidence(logits, cfg, "softmax")
        assert s[0].item() > 0.9
        assert l.tolist() == [0, 0]

    def test_sigmoid_max(self):
        from tcsr.losses import anchor_confidence
        cfg = MethodConfig()
        logits = torch.tensor([[3.0, -1.0], [-1.0, 3.0]])
        s, l = anchor_confidence(logits, cfg, "sigmoid")
        assert l.tolist() == [0, 1]
        assert s[0].item() == pytest.approx(torch.sigmoid(torch.tensor(3.0)).item())
