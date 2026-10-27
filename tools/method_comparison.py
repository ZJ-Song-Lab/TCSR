"""Paired method comparison over evaluate.py records (paper Sec. 5).

Windows note: the shell does not expand glob patterns, so this tool
expands them itself::

    python tools/method_comparison.py \
        --eval-json "outputs/hrsid_rtdetr/HR-*/seed*/variant_*/eval_test.json" \
        --eval-json "outputs/hrsid_rtdetr/HR-*/seed*/eval_test_fp_student.json" \
        --output outputs/hrsid_rtdetr/comparison.json

Records are grouped by experiment config (``--group-by pair`` switches to
the checkpoint pair); within a group the draws are aligned on the seeds
covered by *all* methods (incomplete seeds are dropped with a warning)
and fed to :func:`tcsr.stats.method_comparison` as paired differences
(metric x 100).  By default the stratified bootstrap uses the checkpoint
pair as stratum (the paper's HRSID n = 60 = 3 pairs x 20 seeds);
``--no-strata`` disables stratification.

``--contrast A-B`` compares method A against baseline B (repeatable;
default: G-B, G-C_Q, G-H, G-I).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import save_json  # noqa: E402

from tcsr.stats import (DEFAULT_CONTRASTS, N_BOOTSTRAP, N_MC,  # noqa: E402
                        method_comparison)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--eval-json", action="append", required=True,
                   help="glob pattern of evaluate.py output JSON files "
                        "(repeatable)")
    p.add_argument("--metric", default="AP",
                   help="metric key of the records' 'metrics' dict "
                        "(default: AP)")
    p.add_argument("--contrast", action="append", default=None,
                   help="contrast 'A-B' evaluated as A - B (repeatable; "
                        "default: G-B, G-C_Q, G-H, G-I)")
    p.add_argument("--group-by", choices=("config", "pair"),
                   default="config",
                   help="grouping key of the records (default: config)")
    p.add_argument("--merge", action="store_true",
                   help="merge all groups into one comparison "
                        "(stratified by group)")
    p.add_argument("--no-strata", action="store_true",
                   help="disable the checkpoint-pair strata")
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--n-resamples", type=int, default=None,
                   help="bootstrap resamples (default: 50000)")
    p.add_argument("--n-mc", type=int, default=None,
                   help="randomization-test Monte-Carlo draws "
                        "(default: 10000)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None, help="output JSON path")
    p.add_argument("--csv", default=None,
                   help="optional long-table CSV of the aligned draws")
    p.add_argument("--table-out", default=None,
                   help="optional CSV of the contrast table")
    return p.parse_args(argv)


def collect_records(patterns: List[str]) -> List[dict]:
    """Expand the glob patterns and load every matched record."""
    files: List[str] = []
    for pat in patterns:
        matched = sorted(glob.glob(pat))
        if not matched:
            print(f"[warn] no files match {pat!r}", file=sys.stderr)
        files.extend(matched)
    records = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        rec["_file"] = path
        records.append(rec)
    return records


def build_cells(records: List[dict], group_by: str, metric: str
                ) -> Tuple[Dict[Tuple[str, str, int], Tuple[float, str]],
                           Dict[Tuple[str, int], str]]:
    """(group, method, seed) -> (value x 100, pair) plus (group, seed) -> pair.

    Records whose metrics dict lacks the requested metric are skipped.
    Duplicate cells (same group/method/seed) keep the first occurrence and
    warn -- e.g. two quant profiles collapsed onto one pair group.
    """
    cells: Dict[Tuple[str, str, int], Tuple[float, str]] = {}
    pair_of: Dict[Tuple[str, int], str] = {}
    for rec in records:
        group = rec.get(group_by) or "unknown"
        method = rec.get("method")
        seed = int(rec.get("seed"))
        pair = rec.get("pair") or "unknown"
        value = (rec.get("metrics") or {}).get(metric)
        if method is None or value is None:
            continue
        key = (group, method, seed)
        if key in cells:
            print(f"[warn] duplicate cell {key} (kept "
                  f"{cells[key][0]:.4f} from {pair_of.get((group, seed))}, "
                  f"dropped {float(value) * 100.0:.4f} from {pair})",
                  file=sys.stderr)
            continue
        cells[key] = (float(value) * 100.0, pair)
        pair_of.setdefault((group, seed), pair)
    return cells, pair_of


def group_draws(cells, pair_of, group: str
                ) -> Tuple[Dict[str, List[float]], List[int], List[str]]:
    """Aligned draws of one group over the seeds common to all methods.

    Returns (draws, strata, seeds) where draws maps method -> values in
    sorted-seed order, strata is the per-draw checkpoint-pair index, and
    seeds are the retained seed values.
    """
    methods = sorted({m for g, m, _ in cells if g == group})
    seeds_by_method = {
        m: {s for g, mm, s in cells if g == group and mm == m}
        for m in methods}
    common: set = set.intersection(*seeds_by_method.values()) \
        if seeds_by_method else set()
    for m in methods:
        dropped = sorted(seeds_by_method[m] - common)
        if dropped:
            print(f"[warn] group {group!r} method {m!r}: dropping seeds "
                  f"not covered by all methods: {dropped}", file=sys.stderr)
    seeds = sorted(common)
    draws = {m: [cells[(group, m, s)][0] for s in seeds] for m in methods}
    pairs = [pair_of[(group, s)] for s in seeds]
    uniq = sorted(set(pairs))
    strata = [uniq.index(p) for p in pairs]
    return draws, strata, seeds


def parse_contrasts(texts):
    if not texts:
        return list(DEFAULT_CONTRASTS)
    out = []
    for t in texts:
        parts = t.split("-")
        if len(parts) != 2 or not all(parts):
            raise SystemExit(f"--contrast {t!r}: expected 'A-B'")
        out.append((parts[0], parts[1]))
    return out


def main(argv=None):
    args = parse_args(argv)
    contrasts = parse_contrasts(args.contrast)
    records = collect_records(args.eval_json)
    if not records:
        raise SystemExit("no evaluate.py records matched --eval-json")
    cells, pair_of = build_cells(records, args.group_by, args.metric)
    if not cells:
        raise SystemExit(f"no records carry metric {args.metric!r}")

    groups = sorted({g for g, _, _ in cells})
    prepared = {g: group_draws(cells, pair_of, g) for g in groups}

    n_resamples = args.n_resamples if args.n_resamples is not None \
        else N_BOOTSTRAP
    n_mc = args.n_mc if args.n_mc is not None else N_MC

    results: Dict[str, dict] = {}
    if args.merge:
        # Concatenate the groups' draws; strata = group index.
        method_sets = [set(prepared[g][0]) for g in groups]
        common_methods = sorted(set.intersection(*method_sets)) \
            if method_sets else []
        all_methods = sorted(set.union(*method_sets)) if method_sets else []
        for m in sorted(set(all_methods) - set(common_methods)):
            print(f"[warn] --merge: dropping method {m!r} "
                  f"missing in some groups", file=sys.stderr)
        if not common_methods:
            raise SystemExit("--merge: no method present in every group")
        merged = {m: [] for m in common_methods}
        strata: List[int] = []
        for gi, g in enumerate(groups):
            draws_g = prepared[g][0]
            n_g = len(draws_g[common_methods[0]])
            for m in common_methods:
                merged[m].extend(draws_g[m])
            strata.extend([gi] * n_g)
        results["merged"] = method_comparison(
            merged, contrasts=contrasts,
            strata=None if args.no_strata else (strata or None),
            confidence=args.confidence, n_resamples=n_resamples,
            n_mc=n_mc, seed=args.seed)
    else:
        for g in groups:
            draws, strata, _seeds = prepared[g]
            results[g] = method_comparison(
                draws, contrasts=contrasts,
                strata=None if args.no_strata else (strata or None),
                confidence=args.confidence, n_resamples=n_resamples,
                n_mc=n_mc, seed=args.seed)

    # ------------------------------------------------------------------
    # Report.
    # ------------------------------------------------------------------
    for g, res in results.items():
        order = res.get("_holm_order", [])
        n_draws = int(res[order[0]]["n"]) if order else 0
        print(f"\n=== group: {g} | metric {args.metric} x 100 "
              f"| {n_draws} paired draws ===")
        if not order:
            print("  (no requested contrast has both methods)")
            continue
        for name in order:
            r = res[name]
            print(f"  {name:>14}: n={int(r['n']):3d} "
                  f"mean_diff={r['mean_diff']:+7.2f} "
                  f"CI[{r['ci_low']:+7.2f},{r['ci_high']:+7.2f}] "
                  f"d_z={r['cohens_dz']:+5.2f} "
                  f"p_holm={r['p_holm']:.4f}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["group", "method", "seed", "pair", args.metric])
            for g in groups:
                draws, _strata, seeds = prepared[g]
                for m, vals in draws.items():
                    for s, v in zip(seeds, vals):
                        w.writerow([g, m, s, pair_of[(g, s)], f"{v:.4f}"])
        print(f"[method_comparison] wrote draws to {args.csv}")

    if args.table_out:
        with open(args.table_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["group", "contrast", "n", "mean_diff", "sd_diff",
                        "cohens_dz", "ci_low", "ci_high",
                        "p_randomization", "p_holm"])
            for g, res in results.items():
                for name in res.get("_holm_order", []):
                    r = res[name]
                    w.writerow([g, name, int(r["n"]),
                                f"{r['mean_diff']:.4f}",
                                f"{r['sd_diff']:.4f}",
                                f"{r['cohens_dz']:.4f}",
                                f"{r['ci_low']:.4f}",
                                f"{r['ci_high']:.4f}",
                                f"{r['p_randomization']:.6f}",
                                f"{r['p_holm']:.6f}"])
        print(f"[method_comparison] wrote table to {args.table_out}")

    if args.output:
        save_json(args.output, {
            "metric": args.metric,
            "scale": 100.0,
            "group_by": args.group_by,
            "merged": bool(args.merge),
            "strata": ("none" if args.no_strata else
                       ("group" if args.merge else "pair")),
            "contrasts": [f"{a}-{b}" for a, b in contrasts],
            "confidence": args.confidence,
            "n_resamples": n_resamples,
            "n_mc": n_mc,
            "n_records": len(records),
            "groups": results,
        })
        print(f"[method_comparison] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
