"""Paired statistics tests (bootstrap CI, randomization test, Holm)."""

import numpy as np
import pytest

from tcsr.stats import (DEFAULT_CONTRASTS, cohens_dz, holm_adjust,
                        method_comparison, paired_difference,
                        paired_randomization_test, percentile_bootstrap_ci,
                        summarize_metric)


def test_summarize_metric():
    s = summarize_metric([1.0, 2.0, 3.0])
    assert s["n"] == 3.0
    assert s["mean"] == pytest.approx(2.0)
    assert s["sd"] == pytest.approx(1.0)


def test_paired_difference_shape():
    with pytest.raises(ValueError):
        paired_difference([1.0, 2.0], [1.0])
    d = paired_difference([3.0, 5.0], [1.0, 2.0])
    assert d.tolist() == [2.0, 3.0]


def test_cohens_dz():
    assert np.isnan(cohens_dz([1.0]))
    assert np.isnan(cohens_dz([2.0, 2.0, 2.0]))
    d = cohens_dz([1.0, 2.0, 3.0])   # mean 2, sd 1
    assert d == pytest.approx(2.0)


def test_holm_adjust():
    out = holm_adjust([0.01, 0.04, 0.03])
    # order: 0.01 (x3), 0.03 (x2), 0.04 (x1) -> [0.03, 0.06, 0.06]
    assert out.tolist() == pytest.approx([0.03, 0.06, 0.06])


def test_holm_adjust_cap():
    out = holm_adjust([0.9, 0.8])
    assert (out <= 1.0).all()


def test_randomization_exact_all_positive():
    d = list(map(float, np.arange(1, 11)))    # n = 10 <= 20 -> exact
    res = paired_randomization_test(d, seed=0, n_mc=200)
    assert res["exact"] is True
    assert res["n"] == 10.0
    # Only the two all-same-sign flips match |T_obs| exactly; ties count one
    # half each: p = (0 + 0.5 * 2) / 1024 = 1/1024.
    assert res["p"] == pytest.approx(1.0 / 1024.0)
    assert res["n_null"] == 1024.0


def test_randomization_mc_mode():
    rng = np.random.default_rng(0)
    d = rng.normal(0.5, 0.1, size=30)         # n = 30 > 20 -> Monte-Carlo
    res = paired_randomization_test(d, seed=0, n_mc=500)
    assert res["exact"] is False
    assert res["n_null"] == 500.0
    assert 0.0 <= res["p"] <= 1.0


def test_randomization_empty():
    res = paired_randomization_test([])
    assert np.isnan(res["p"])


def test_bootstrap_ci_contains_mean():
    d = list(map(float, np.arange(1, 11)))
    lo, hi = percentile_bootstrap_ci(d, n_resamples=200, seed=0)
    m = np.mean(d)
    assert lo < m < hi
    assert lo < hi


def test_bootstrap_ci_stratified():
    d = np.arange(6, dtype=float)
    strata = np.array([0, 0, 0, 1, 1, 1])
    lo, hi = percentile_bootstrap_ci(d, n_resamples=200, strata=strata, seed=0)
    assert lo < hi
    with pytest.raises(ValueError):
        percentile_bootstrap_ci(d, n_resamples=10, strata=np.array([0, 1]))


def test_method_comparison_structure():
    rng = np.random.default_rng(0)
    n = 12
    metrics = {
        "G": 60 + rng.normal(0, 0.5, n),
        "B": 59 + rng.normal(0, 0.5, n),
    }
    out = method_comparison(metrics, contrasts=(("G", "B"),),
                            n_resamples=200, n_mc=200, seed=0)
    assert "G-B" in out
    entry = out["G-B"]
    for key in ("n", "mean_diff", "sd_diff", "cohens_dz", "ci_low",
                "ci_high", "p_randomization", "exact", "p_holm"):
        assert key in entry
    assert entry["n"] == float(n)
    assert entry["mean_diff"] == pytest.approx(
        float(np.mean(metrics["G"] - metrics["B"])))
    assert entry["ci_low"] <= entry["mean_diff"] <= entry["ci_high"]
    assert out["_holm_order"] == ["G-B"]


def test_default_contrasts():
    assert DEFAULT_CONTRASTS == (("G", "B"), ("G", "C_Q"), ("G", "H"),
                                 ("G", "I"))
