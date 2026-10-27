"""Anchor selection, feasible edges, and LSAP matching tests."""

import numpy as np
import pytest
import torch

from tcsr.config import MethodConfig
from tcsr.losses import Prediction
from tcsr.matching import (feasible_edges, max_cardinality_min_cost_matching,
                           select_anchors, solve_anchor_assignment,
                           teacher_candidates, threshold_topk)


class TestThresholdTopk:
    def test_threshold_first_then_topk(self):
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        idx = threshold_topk(scores, k=3, threshold=0.45)
        assert idx.tolist() == [0, 1, 2]

    def test_tie_ascending_index(self):
        scores = torch.tensor([0.5, 0.9, 0.5, 0.9])
        idx = threshold_topk(scores, k=3, threshold=0.4)
        assert idx.tolist() == [1, 3, 0]

    def test_empty(self):
        idx = threshold_topk(torch.tensor([0.1, 0.2]), k=3, threshold=0.5)
        assert idx.numel() == 0


def _pred_from(logits, boxes):
    return Prediction(logits=logits, boxes=boxes)


class TestSelectAnchors:
    def test_select(self):
        cfg = MethodConfig(k_p=3, tau_p=0.5)
        # Softmax head: the background is the LAST channel, so the
        # foreground confidence of row r is softmax(logits)[r, 0].
        logits = torch.tensor([[-5.0, 2.0], [2.0, -5.0], [2.0, -5.0],
                               [2.0, -5.0]])
        p = _pred_from(logits, torch.rand(4, 4))
        idx, s, l = select_anchors(p, cfg, "softmax")
        # rows 1-3 pass the threshold, budget 3 keeps them all
        assert idx.tolist() == [1, 2, 3]
        assert s.tolist() == [float(x) for x in torch.softmax(logits, -1)[1:, 0]]

    def test_teacher_candidates(self):
        cfg = MethodConfig(k_t=2, tau_t=0.5)
        # background LAST channel: rows 1-2 carry the foreground mass
        logits = torch.tensor([[-5.0, 3.0], [3.0, -5.0], [3.0, -5.0]])
        p = _pred_from(logits, torch.rand(3, 4))
        idx, s, l = teacher_candidates(p, cfg, "softmax")
        assert idx.tolist() == [1, 2]


class TestFeasibleEdges:
    def test_same_class_and_close(self):
        lt = torch.tensor([0, 1])
        lp = torch.tensor([0, 1, 0])
        d = torch.tensor([[0.1, 0.4, 0.2],
                          [0.9, 0.2, 0.3]])
        feas = feasible_edges(lt, lp, d, tau_d=0.35)
        # (0,0): same class + 0.1 <= 0.35 -> True
        # (0,1): class mismatch -> False
        # (0,2): same class + 0.2 <= 0.35 -> True
        # (1,*): class mismatch with all but (1,1); d=0.2 -> True
        assert feas.tolist() == [[True, False, True],
                                 [False, True, False]]


class TestMaxCardinalityMatching:
    def test_prefers_cardinality_over_cost(self):
        # 2 teacher candidates, 2 anchors: rows/cols of the feasible graph.
        cost = torch.tensor([[0.1, 0.2],
                             [0.0, 5.0]])
        feas = torch.tensor([[True, True],
                             [True, False]])
        # Row 1 can only use column 0 (its (1,1) edge is infeasible), so
        # the unique cardinality-2 matching is (0,1) + (1,0), total
        # 0.2 + 0.0.  A min-cost-only solver would prefer the
        # cardinality-1 match (1,0) with cost 0.0 -> cardinality wins.
        rows, cols = max_cardinality_min_cost_matching(cost, feas)
        assert sorted(zip(rows.tolist(), cols.tolist())) == [(0, 1), (1, 0)]

    def test_drops_infeasible(self):
        cost = torch.tensor([[0.0, 1.0]])
        feas = torch.tensor([[True, False]])
        rows, cols = max_cardinality_min_cost_matching(cost, feas)
        assert rows.tolist() == [0] and cols.tolist() == [0]

    def test_empty(self):
        cost = torch.zeros(0, 3)
        feas = torch.zeros(0, 3, dtype=torch.bool)
        rows, cols = max_cardinality_min_cost_matching(cost, feas)
        assert rows.size == 0 and cols.size == 0

    def test_no_feasible(self):
        cost = torch.ones(2, 2)
        feas = torch.zeros(2, 2, dtype=torch.bool)
        rows, cols = max_cardinality_min_cost_matching(cost, feas)
        assert rows.size == 0 and cols.size == 0


class TestSolveAnchorAssignment:
    def test_optimal_assignment(self):
        d = torch.tensor([[0.1, 0.9],
                          [0.8, 0.2]])
        rho = solve_anchor_assignment(d)
        assert rho.tolist() == [0, 1]

    def test_reversed(self):
        d = torch.tensor([[0.9, 0.1],
                          [0.2, 0.8]])
        rho = solve_anchor_assignment(d)
        assert rho.tolist() == [1, 0]

    def test_more_anchors_than_queries(self):
        d = torch.tensor([[0.1, 0.2],
                          [0.3, 0.4],
                          [0.1, 0.1]])
        rho = solve_anchor_assignment(d)
        # LSAP over a 3x2 matrix assigns 2 anchors; the third is -1.
        assert (rho >= 0).sum().item() == 2
        assert (rho == -1).sum().item() == 1
        # the two assigned anchors get distinct queries
        assigned = rho[rho >= 0].tolist()
        assert sorted(assigned) == [0, 1]

    def test_empty(self):
        rho = solve_anchor_assignment(torch.zeros(0, 5))
        assert rho.shape == (0,)

    def test_returns_long(self):
        rho = solve_anchor_assignment(torch.rand(3, 4))
        assert rho.dtype == torch.long
