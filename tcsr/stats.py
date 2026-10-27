"""Paired statistics for the method comparison (paper Sec. 5, protocol).

Each experiment setting reports paired draws of the same metric (AP x 100)
across shared random draws: HRSID uses 3 checkpoint pairs x 20 paired
draws (n = 60), the other settings a representative pair with 20 draws
(two additional pairs replicate the B/G contrast pair-by-pair, never
pooled).  For every contrast the module reports

  - the paired difference mean with SD (ddof = 1),
  - Cohen's d_z = mean(d) / sd(d),
  - a paired 95% percentile-bootstrap CI from 50,000 paired resamples,
    resampled within checkpoint strata when strata are supplied (HRSID),
  - a two-sided paired randomization (sign-flip) p-value, exact by full
    enumeration of the 2^n flip combinations for n <= 20 and by 1,000,000
    Monte-Carlo flips otherwise; zero differences are kept and ties
    contribute 1/2 to the tail,
  - Holm step-down adjustment across the four contrasts
    (G-B, G-C_Q, G-H, G-I) of each setting.

Draw seeds are documented in manifests/seeds.csv (HR-01: 12001-12020,
HR-02: 12101-12120, HR-03: 12201-12220, SS-01: 22001-22020, CR-01:
32001-32020, CD-01: 42001-42020).
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

N_BOOTSTRAP = 50_000
N_MC = 1_000_000
EXACT_MAX_N = 20
DEFAULT_CONTRASTS = (("G", "B"), ("G", "C_Q"), ("G", "H"), ("G", "I"))


def summarize_metric(values: Sequence[float]) -> Dict[str, float]:
    """Mean and sample SD (ddof = 1) of a metric's paired draws."""
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    return {"n": float(n),
            "mean": float(x.mean()) if n else float("nan"),
            "sd": float(x.std(ddof=1)) if n >= 2 else float("nan")}


def paired_difference(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """Elementwise paired difference d = a - b (draw-aligned)."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"paired draws must align, got {x.shape} vs "
                         f"{y.shape}")
    return x - y


def cohens_dz(d: Sequence[float]) -> float:
    """Paired effect size d_z = mean(d) / sd(d, ddof = 1)."""
    x = np.asarray(d, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else float("nan")


def _stratified_indices(rng: np.random.Generator, strata: np.ndarray,
                        n_resamples: int) -> np.ndarray:
    """Paired resampling indices, independently within each stratum."""
    n = strata.size
    idx = np.empty((n_resamples, n), dtype=np.int64)
    for s in np.unique(strata):
        pos = np.nonzero(strata == s)[0]
        draw = pos[rng.integers(0, pos.size, size=(n_resamples, pos.size))]
        idx[:, pos] = draw
    return idx


def percentile_bootstrap_ci(d: Sequence[float],
                            confidence: float = 0.95,
                            n_resamples: int = N_BOOTSTRAP,
                            strata: Optional[Sequence[int]] = None,
                            seed: int = 0) -> Tuple[float, float]:
    """Paired percentile-bootstrap CI of the mean difference.

    Zero differences are kept.  With strata (e.g. the three HRSID
    checkpoint pairs), each resample draws within stratum and concatenates,
    preserving the stratified pairing structure.
    """
    x = np.asarray(d, dtype=np.float64)
    n = x.size
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    if strata is None:
        idx = rng.integers(0, n, size=(n_resamples, n))
    else:
        st = np.asarray(strata)
        if st.shape[0] != n:
            raise ValueError("strata must align with the paired draws")
        idx = _stratified_indices(rng, st, n_resamples)
    means = x[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.percentile(means, [100.0 * alpha,
                                   100.0 * (1.0 - alpha)])
    return float(lo), float(hi)


def paired_randomization_test(d: Sequence[float], seed: int = 0,
                              n_mc: int = N_MC) -> Dict[str, float]:
    """Two-sided paired randomization (sign-flip) test on mean(d).

    Exact by full enumeration of all 2^n sign patterns for n <= 20
    (1,048,576 combinations at n = 20); otherwise n_mc Monte-Carlo flips
    in chunks.  Zero differences are kept (no pruning); ties in |T| count
    one half toward the two-sided tail:

        p = (#|T0| > |T_obs| + 0.5 #|T0| == |T_obs|) / #null.
    """
    x = np.asarray(d, dtype=np.float64)
    n = x.size
    if n == 0:
        return {"n": 0.0, "t_obs": float("nan"), "p": float("nan"),
                "exact": False, "n_null": 0.0}
    t_obs = float(x.mean())
    abs_obs = abs(t_obs)
    if n <= EXACT_MAX_N:
        flips = ((np.arange(2 ** n, dtype=np.int64)[:, None]
                  >> np.arange(n)) & 1).astype(np.int8)
        signs = (1 - 2 * flips).astype(np.float64)
        t0 = signs @ x / n
        gt = int((np.abs(t0) > abs_obs).sum())
        eq = int((np.abs(t0) == abs_obs).sum())
        n_null = 2 ** n
        return {"n": float(n), "t_obs": t_obs,
                "p": float((gt + 0.5 * eq) / n_null),
                "exact": True, "n_null": float(n_null)}
    rng = np.random.default_rng(seed)
    gt = 0
    eq = 0
    done = 0
    chunk = 100_000
    while done < n_mc:
        b = min(chunk, n_mc - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(b, n))
        t0 = signs @ x / n
        abs_t = np.abs(t0)
        gt += int((abs_t > abs_obs).sum())
        eq += int((abs_t == abs_obs).sum())
        done += b
    return {"n": float(n), "t_obs": t_obs,
            "p": float((gt + 0.5 * eq) / n_mc),
            "exact": False, "n_null": float(n_mc)}


def holm_adjust(pvals: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    adjusted = np.empty(m, dtype=np.float64)
    order = np.argsort(p, kind="mergesort")
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def method_comparison(metrics: Mapping[str, Sequence[float]],
                      contrasts: Sequence[Tuple[str, str]] = DEFAULT_CONTRASTS,
                      strata: Optional[Sequence[int]] = None,
                      confidence: float = 0.95,
                      n_resamples: int = N_BOOTSTRAP,
                      n_mc: int = N_MC,
                      seed: int = 0) -> Dict[str, Dict]:
    """Full paired comparison of the paper's four contrasts for one setting.

    metrics maps method name -> aligned paired draws of the metric
    (AP x 100); contrasts are (method, baseline) pairs evaluated as
    method - baseline.  strata aligns with the draws (checkpoint pairs for
    HRSID's n = 60).  Returns per-contrast summary plus the Holm-adjusted
    p-values across the contrasts.
    """
    out: Dict[str, Dict] = {}
    raw_p: Dict[str, float] = {}
    for k, (a, b) in enumerate(contrasts):
        if a not in metrics or b not in metrics:
            continue
        d = paired_difference(metrics[a], metrics[b])
        s = summarize_metric(d)
        lo, hi = percentile_bootstrap_ci(d, confidence=confidence,
                                         n_resamples=n_resamples,
                                         strata=strata, seed=seed + 1000 * k)
        rand = paired_randomization_test(d, seed=seed + 1000 * k, n_mc=n_mc)
        name = f"{a}-{b}"
        out[name] = {
            "n": s["n"], "mean_diff": s["mean"], "sd_diff": s["sd"],
            "cohens_dz": cohens_dz(d),
            "ci_low": lo, "ci_high": hi,
            "p_randomization": rand["p"], "exact": rand["exact"],
        }
        raw_p[name] = rand["p"]
    names = list(raw_p)
    adjusted = holm_adjust([raw_p[nm] for nm in names]) if names else \
        np.zeros(0)
    for nm, p_adj in zip(names, adjusted):
        out[nm]["p_holm"] = float(p_adj)
    out["_holm_order"] = names
    return out
