"""Drive the paper's repeated-calibration protocol for one checkpoint pair.

For every draw seed ``d`` the script materializes the draw's calibration
manifest, calibrates the requested variant table with the paired
optimization seed ``d + 500000``, and evaluates every model on the
held-out split; finally it runs the paired method comparison over the
produced records.  The seed convention is fixed in
``manifests/seeds.json``::

    draw seed            -> tools/calibration_manifest.py  (image list)
    optimization seed    -> tools/calibrate.py / evaluate.py
                             (= draw seed + 500000)

Full protocol for the representative HRSID pair (20 draws, complete
variant table, paper defaults 6000 + 8000 updates)::

    python scripts/reproduce.py --config configs/hrsid_rtdetr.json

Practical variations::

    # a different checkpoint pair of the same setting (the config is
    # derived via the ``base`` inheritance mechanism):
    python scripts/reproduce.py --config configs/hrsid_rtdetr.json --pair HR-01

    # two draws, two variants, 50 + 50 updates, on a GPU:
    python scripts/reproduce.py --config configs/hrsid_rtdetr.json \
        --first-n 2 --variants B,G --stage1-updates 50 --stage2-updates 50 \
        --device cuda

    # resume an interrupted run (finished seeds/variants are skipped):
    python scripts/reproduce.py --config configs/hrsid_rtdetr.json

    # print the command sequence without running anything:
    python scripts/reproduce.py --config configs/hrsid_rtdetr.json --dry-run

Per-draw artifacts land in ``outputs/{config}/{pair}/seed{opt}/`` (see
tools/calibrate.py); the FP reference baselines (fp:student, fp:teacher)
are evaluated once per pair on the first draw's optimization seed; the
paired comparison (contrasts G-B, G-C_Q, G-H, G-I over the aligned
draws) is written to ``outputs/{config}/{pair}/comparison.json``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

from common import DEFAULT_EVAL_SPLIT  # noqa: E402
from tcsr.config import load_config  # noqa: E402
from tcsr.stats import DEFAULT_CONTRASTS  # noqa: E402
from tcsr.variants import normalize_variant  # noqa: E402

SEEDS_FILE = os.path.join(REPO_ROOT, "manifests", "seeds.json")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="experiment config JSON")
    p.add_argument("--pair", default=None,
                   help="checkpoint pair (default: the config's pair); a "
                        "derived config is written via base inheritance")
    p.add_argument("--seeds", default=None,
                   help="comma-separated draw seeds (default: the pair's "
                        "namespace in manifests/seeds.json)")
    p.add_argument("--first-n", type=int, default=None,
                   help="use only the first N draws of the namespace")
    p.add_argument("--variants", default=None,
                   help="comma-separated variants (default: config table)")
    p.add_argument("--stage1-updates", type=int, default=None)
    p.add_argument("--stage2-updates", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default=None,
                   help="output root (default: config output_dir)")
    p.add_argument("--calib-budget", type=int, default=None,
                   help="calibration images per draw (default: config)")
    # checkpoint overrides (passed through to the tools)
    p.add_argument("--teacher", default=None)
    p.add_argument("--student", default=None)
    p.add_argument("--repo-path", default=None)
    p.add_argument("--manifest", default=None,
                   help="checkpoint manifest (default: manifests/"
                        "checkpoints.json)")
    # flow control
    p.add_argument("--skip-manifests", action="store_true",
                   help="do not generate the per-draw calibration manifests")
    p.add_argument("--skip-calibrate", action="store_true")
    p.add_argument("--skip-evaluate", action="store_true")
    p.add_argument("--no-comparison", action="store_true")
    p.add_argument("--with-diagnostics", action="store_true",
                   help="also run tools/diagnostics.py on the first draw")
    p.add_argument("--dry-run", action="store_true",
                   help="print the commands without running them")
    return p.parse_args(argv)


def derive_pair_config(config_path: str, pair: str) -> str:
    """Write a derived config for ``pair`` via the base mechanism."""
    config_path = os.path.abspath(config_path)
    out_dir = os.path.join(REPO_ROOT, "outputs", "derived")
    stem = os.path.splitext(os.path.basename(config_path))[0]
    out = os.path.join(out_dir, f"{stem}_{pair}.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(config_path, encoding="utf-8") as f:
        if json.load(f).get("checkpoint_pair") == pair:
            return config_path  # nothing to derive
    rel = os.path.relpath(config_path, out_dir).replace("\\", "/")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"base": rel, "checkpoint_pair": pair}, f, indent=2)
        f.write("\n")
    print(f"[reproduce] derived config for pair {pair}: "
          f"{os.path.relpath(out, REPO_ROOT)}")
    return out


def resolve_seeds(pair: str, setting: str, args) -> list:
    """Draw seeds: explicit list or the pair's namespace from seeds.json."""
    if args.seeds:
        seeds = sorted({int(s) for s in args.seeds.split(",") if s.strip()})
    else:
        entry = None
        if os.path.exists(SEEDS_FILE):
            with open(SEEDS_FILE, encoding="utf-8") as f:
                doc = json.load(f)
            entry = (doc.get("pairs") or {}).get(pair)
        if entry is None:
            raise SystemExit(
                f"pair {pair!r} is not listed in manifests/seeds.json; "
                "pass --seeds explicitly")
        if entry.get("setting") not in (None, setting):
            print(f"[reproduce] WARNING: seeds.json lists pair {pair!r} "
                  f"under setting {entry.get('setting')!r}, "
                  f"but the config is {setting!r}", file=sys.stderr)
        seeds = list(range(int(entry["seeds_start"]),
                           int(entry["seeds_end"]) + 1))
    if args.first_n is not None:
        seeds = seeds[:args.first_n]
    if not seeds:
        raise SystemExit("empty draw-seed list")
    return seeds


def opt_seed(draw_seed: int) -> int:
    """Optimization seed = draw seed + offset (manifests/seeds.json)."""
    offset = 500000
    if os.path.exists(SEEDS_FILE):
        with open(SEEDS_FILE, encoding="utf-8") as f:
            offset = int((json.load(f).get("convention") or {})
                         .get("optimization_seed_offset", offset))
    return draw_seed + offset


class Runner:
    """Subprocess wrapper with dry-run support."""

    def __init__(self, dry_run: bool):
        self.dry_run = dry_run

    def __call__(self, step: str, cmd: list) -> None:
        print(f"\n--- {step} ---\n  " + " ".join(cmd), flush=True)
        if self.dry_run:
            return
        proc = subprocess.run(cmd, cwd=REPO_ROOT)
        if proc.returncode != 0:
            raise SystemExit(f"[reproduce] step {step!r} failed with code "
                             f"{proc.returncode}")


def main(argv=None):
    args = parse_args(argv)
    run = Runner(args.dry_run)

    config_path = args.config
    if args.pair:
        config_path = derive_pair_config(config_path, args.pair)
    exp = load_config(config_path)
    pair = exp.checkpoint_pair
    out_root = args.output_dir or exp.output_dir
    pair_dir = os.path.join(out_root, exp.name, pair)
    split = DEFAULT_EVAL_SPLIT[exp.dataset]
    variants = ([normalize_variant(v) for v in args.variants.split(",")
                 if v.strip()] if args.variants else list(exp.variants))
    ckpt_flags = (["--teacher", args.teacher] if args.teacher else []) + \
                 (["--student", args.student] if args.student else []) + \
                 (["--repo-path", args.repo_path] if args.repo_path else []) + \
                 (["--manifest", args.manifest] if args.manifest else [])
    seeds = resolve_seeds(pair, exp.name, args)

    print(f"[reproduce] setting={exp.name} pair={pair} dataset={exp.dataset} "
          f"quant_profile={exp.quant_profile}")
    print(f"[reproduce] draws: {len(seeds)} "
          f"(draw seeds {seeds[0]}..{seeds[-1]})")
    print(f"[reproduce] variants: {','.join(variants)}")
    print(f"[reproduce] eval split: {split}; outputs under {pair_dir}/")

    py = [sys.executable, "-u"]
    base = ["--config", config_path, "--device", args.device] + ckpt_flags

    for i, d in enumerate(seeds):
        opt = opt_seed(d)
        sdir = os.path.join(pair_dir, f"seed{opt:05d}")
        manifest = os.path.join(REPO_ROOT, "manifests", "calibration",
                                f"{pair}_{d}.json")
        print(f"\n{'=' * 72}\n[reproduce] draw {i + 1}/{len(seeds)}: "
              f"draw seed {d} -> optimization seed {opt}\n{'=' * 72}")

        # 1. per-draw calibration manifest (fixed image list)
        if not args.skip_manifests and not os.path.exists(manifest):
            cmd = py + ["tools/calibration_manifest.py", "--config",
                        config_path, "--seed", str(d)]
            if args.calib_budget:
                cmd += ["--budget", str(args.calib_budget)]
            run(f"calibration manifest ({d})", cmd)
        elif not args.skip_manifests:
            print(f"[reproduce] calibration manifest exists: "
                  f"{os.path.relpath(manifest, REPO_ROOT)}")

        # 2. calibrate the variant table
        if not args.skip_calibrate:
            qmodels = {v: os.path.join(sdir, f"variant_{v}", "qmodel.pt")
                       for v in variants}
            missing = [v for v, p in qmodels.items() if not os.path.exists(p)]
            if not missing:
                print(f"[reproduce] all {len(variants)} variants already "
                      f"calibrated under {os.path.relpath(sdir, REPO_ROOT)}/")
            else:
                cmd = py + ["tools/calibrate.py"] + base + \
                    ["--seed", str(opt), "--variants", ",".join(missing),
                     "--calib-manifest",
                     os.path.relpath(manifest, REPO_ROOT)]
                if args.output_dir:
                    cmd += ["--output-dir", args.output_dir]
                if args.stage1_updates:
                    cmd += ["--stage1-updates", str(args.stage1_updates)]
                if args.stage2_updates:
                    cmd += ["--stage2-updates", str(args.stage2_updates)]
                run(f"calibrate ({opt}, variants {','.join(missing)})", cmd)

        # 3. evaluate every model on the held-out split
        if not args.skip_evaluate:
            models = []
            for v in variants:
                qm = os.path.join(sdir, f"variant_{v}", "qmodel.pt")
                if run.dry_run or os.path.exists(qm):
                    expected = os.path.join(os.path.dirname(qm),
                                           f"eval_{split}.json")
                    if not os.path.exists(expected):
                        models += ["--model", qm]
            if i == 0:  # FP reference baselines: once per pair
                for which in ("student", "teacher"):
                    expected = os.path.join(
                        sdir, f"eval_{split}_fp_{which}.json")
                    if not os.path.exists(expected):
                        models += ["--model", f"fp:{which}"]
            if models:
                cmd = py + ["tools/evaluate.py"] + base + \
                    ["--seed", str(opt)] + models
                if args.output_dir:
                    cmd += ["--output-dir", args.output_dir]
                run(f"evaluate ({opt}, {len(models) // 2} models)", cmd)
            else:
                print("[reproduce] all eval records already present "
                      "for this draw")

    # 4. optional diagnostics on the first draw
    if args.with_diagnostics:
        opt = opt_seed(seeds[0])
        cmd = py + ["tools/diagnostics.py"] + base + ["--seed", str(opt),
                                                      "--variant", "G"]
        if args.output_dir:
            cmd += ["--output-dir", args.output_dir]
        if args.stage1_updates:
            cmd += ["--stage1-updates", str(args.stage1_updates)]
        if args.stage2_updates:
            cmd += ["--stage2-updates", str(args.stage2_updates)]
        run(f"diagnostics ({opt}, variant G)", cmd)

    # 5. paired method comparison over the produced records
    pattern = os.path.join(pair_dir, "seed*", "variant_*",
                           f"eval_{split}.json").replace("\\", "/")
    if not args.no_comparison:
        n_records = len(glob.glob(pattern))
        contrasts = [f"{a}-{b}" for a, b in DEFAULT_CONTRASTS
                     if a in variants and b in variants]
        if n_records == 0 and not run.dry_run:
            print("[reproduce] no eval records found; skipping comparison")
        else:
            cmd = py + ["tools/method_comparison.py"]
            for c in contrasts:
                cmd += ["--contrast", c]
            cmd += ["--group-by", "config",
                    "--eval-json", pattern,
                    "--output", os.path.join(pair_dir, "comparison.json"),
                    "--csv", os.path.join(pair_dir, "comparison_draws.csv"),
                    "--table-out",
                    os.path.join(pair_dir, "comparison_table.csv")]
            run(f"method comparison ({n_records} records)", cmd)

    # ------------------------------------------------------------------
    # Summary: mean +- sd AP per method over the evaluated draws.
    # ------------------------------------------------------------------
    if not args.dry_run and not args.skip_evaluate:
        print(f"\n{'=' * 72}\n[reproduce] summary: pair {pair} "
              f"({split}, AP x 100)\n{'=' * 72}")
        fp_pattern = os.path.join(pair_dir, "seed*",
                                  f"eval_{split}_fp_*.json").replace("\\", "/")
        records = []
        for pat in (pattern, fp_pattern):
            for path in sorted(glob.glob(pat)):
                with open(path, encoding="utf-8") as f:
                    records.append(json.load(f))
        by_method = {}
        for rec in records:
            ap = 100.0 * (rec.get("metrics") or {}).get("AP", float("nan"))
            by_method.setdefault(rec.get("method"), []).append(ap)
        for method in sorted(by_method):
            vals = by_method[method]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
            print(f"  {method:>10}: {mean:6.2f} +- {var ** 0.5:4.2f} "
                  f"(n={len(vals)} draws)")
        if not records:
            print("  (no eval records found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
