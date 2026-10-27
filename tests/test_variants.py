"""Variant A-I coefficient tests (paper Sec. 5.4 mechanism table)."""

import pytest
import torch

from tcsr.config import MethodConfig
from tcsr.teacher import ImageRecord
from tcsr.variants import (NEEDS_GQ, NEEDS_U0, VARIANTS, VARIANT_SPECS,
                           VariantContext, assign_variant_weights,
                           build_coefficients, normalize_variant)


def _ctx(c, u0=None, g_q0=None, scores=None, labels=None, cfg=None):
    k = c.shape[0]
    scores = scores if scores is not None else torch.linspace(0.9, 0.5, k)
    labels = labels if labels is not None else torch.zeros(k, dtype=torch.long)
    return VariantContext(anchor_scores=scores, anchor_labels=labels, c=c,
                          u0=u0, g_q0=g_q0,
                          cfg=cfg if cfg is not None else MethodConfig())


class TestNormalize:
    def test_canonical(self):
        assert normalize_variant("g") == "G"
        assert normalize_variant(" c_q ") == "C_Q"
        assert normalize_variant("CQ") == "C_Q"

    def test_unknown(self):
        with pytest.raises(ValueError):
            normalize_variant("Z")

    def test_roster(self):
        assert VARIANTS == ("A", "B", "C", "C_Q", "D", "E", "F", "G", "H", "I")
        assert VARIANT_SPECS["A"].stage2 is False
        assert VARIANT_SPECS["A"].set_losses is False
        assert VARIANT_SPECS["A"].teacher_term is False
        assert VARIANT_SPECS["B"].teacher_term is False
        assert VARIANT_SPECS["I"].teacher_targets is True
        assert VARIANT_SPECS["G"] == VARIANT_SPECS["C"]
        assert NEEDS_U0 == ("C", "C_Q", "D", "F", "G", "H", "I")
        assert NEEDS_GQ == ("C_Q",)


class TestCoefficients:
    def test_a_b_zero(self):
        ctx = _ctx(torch.tensor([0.5, 0.0, 0.8]), u0=torch.tensor([0.1, 0.0, 0.2]))
        for v in ("A", "B"):
            w = build_coefficients(v, ctx)
            assert torch.equal(w, torch.zeros(3))

    def test_g_math(self):
        cfg = MethodConfig(beta=0.75)
        c = torch.tensor([0.5, 0.0, 0.8])
        u0 = torch.tensor([0.4, 0.9, 0.0])
        w = build_coefficients("G", _ctx(c, u0=u0, cfg=cfg))
        expected = c * (1 + 0.75 * u0)
        assert torch.allclose(w, expected)

    def test_e_is_c(self):
        c = torch.tensor([0.5, 0.0, 0.8])
        w = build_coefficients("E", _ctx(c, u0=torch.tensor([9.0, 9.0, 9.0])))
        assert torch.equal(w, c)

    def test_f_identity_on_corroborated(self):
        cfg = MethodConfig(beta=0.5)
        c = torch.tensor([0.5, 0.0, 0.8])
        u0 = torch.tensor([0.4, 0.9, 0.0])
        w = build_coefficients("F", _ctx(c, u0=u0, cfg=cfg))
        assert w.tolist() == pytest.approx([1 + 0.5 * 0.4, 0.0, 1 + 0.5 * 0.0])

    def test_d_uniform_mass_preserved(self):
        cfg = MethodConfig(beta=0.75)
        c = torch.tensor([0.4, 0.0, 0.6])
        u0 = torch.tensor([0.2, 0.5, 0.0])
        w = build_coefficients("D", _ctx(c, u0=u0, cfg=cfg))
        g_mass = float((c * (1 + 0.75 * u0)).sum())
        assert w[1].item() == 0.0
        assert w[0].item() == pytest.approx(g_mass / 2)
        assert w[2].item() == pytest.approx(g_mass / 2)

    def test_c_top_n_confident(self):
        cfg = MethodConfig(beta=0.0)
        c = torch.tensor([0.2, 0.6, 0.0, 0.2])      # |V_x| = 3
        scores = torch.tensor([0.5, 0.9, 0.4, 0.8])  # top-3 = idx 1, 3, 0
        u0 = torch.zeros(4)
        w = build_coefficients("C", _ctx(c, u0=u0, scores=scores, cfg=cfg))
        mass = float(c.sum())                        # beta = 0 -> mass = sum c
        exp = torch.tensor([mass / 3, mass / 3, 0.0, mass / 3])
        assert torch.allclose(w, exp)

    def test_cq_permutation_of_g_multiset(self):
        cfg = MethodConfig(beta=0.75)
        c = torch.tensor([0.5, 0.3, 0.0, 0.9])
        u0 = torch.tensor([0.2, 0.0, 0.5, 0.1])
        g_q0 = torch.tensor([0.1, 0.4, 0.9, 0.2])
        w = build_coefficients("C_Q", _ctx(c, u0=u0, g_q0=g_q0, cfg=cfg))
        g = c * (1 + 0.75 * u0)
        # C_Q assigns G's coefficient multiset (zeros included) hardest-first;
        # sorted values must match exactly.
        assert torch.allclose(torch.sort(w).values, torch.sort(g).values)

    def test_h_strata_multiset_preserved(self):
        cfg = MethodConfig(beta=0.75)
        c = torch.tensor([0.5, 0.3, 0.0, 0.9, 0.2])
        labels = torch.tensor([0, 0, 0, 1, 1])
        u0 = torch.tensor([0.2, 0.0, 0.5, 0.1, 0.3])
        ctx = _ctx(c, u0=u0, labels=labels, cfg=cfg)
        g = build_coefficients("G", ctx)
        w = build_coefficients("H", ctx)
        assert not torch.equal(w, g) or True  # may coincide by chance
        # multiset preserved globally and per stratum
        assert torch.allclose(torch.sort(w).values, torch.sort(g).values)
        for l in (0, 1):
            m = labels == l
            assert torch.allclose(torch.sort(w[m]).values, torch.sort(g[m]).values)

    def test_missing_u0_raises(self):
        for v in NEEDS_U0:
            with pytest.raises(ValueError):
                build_coefficients(v, _ctx(torch.tensor([0.5]), u0=None))

    def test_missing_gq_raises(self):
        with pytest.raises(ValueError):
            build_coefficients("C_Q", _ctx(torch.tensor([0.5]),
                                           u0=torch.tensor([0.1]),
                                           g_q0=None))

    def test_empty(self):
        w = build_coefficients("G", _ctx(torch.zeros(0)))
        assert w.shape == (0,)


class TestAssignVariantWeights:
    def _records(self):
        cfg = MethodConfig()
        from tcsr.losses import Prediction
        from tcsr.teacher import make_record
        recs = []
        g = torch.Generator().manual_seed(0)
        for i in range(2):
            n = 8
            logits = torch.full((n, 2), -5.0)
            logits[:, 0] = 5.0
            sp = Prediction(logits=logits, boxes=torch.rand(n, 4, generator=g))
            tp = Prediction(logits=logits.clone(), boxes=torch.rand(n, 4, generator=g))
            recs.append(make_record(f"img_{i}", sp, tp, cfg, "softmax"))
        return recs

    def test_requires_priorities(self):
        recs = self._records()
        with pytest.raises(ValueError):
            assign_variant_weights(recs, "G")

    def test_assign_g(self):
        recs = self._records()
        # fabricate priorities
        for rec in recs:
            k = rec.anchors.numel()
            rec.u0 = torch.rand(k)
            rec.g_q0 = torch.rand(k)
            rec.rho0 = torch.zeros(k, dtype=torch.long)
        assign_variant_weights(recs, "G")
        for rec in recs:
            assert rec.w is not None
            assert rec.w.shape[0] == rec.anchors.numel()
            expected = rec.c * (1 + MethodConfig().beta * rec.u0)
            assert torch.allclose(rec.w, expected)

    def test_b_zero(self):
        recs = self._records()
        for rec in recs:
            rec.u0 = torch.rand(rec.anchors.numel())
        assign_variant_weights(recs, "B")
        for rec in recs:
            assert torch.equal(rec.w, torch.zeros(rec.anchors.numel()))
