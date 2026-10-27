"""End-to-end smoke test on synthetic toy detectors (no downloads).

Drives the full pipeline -- calibrate, evaluate, method comparison,
diagnostics -- in toy mode (``--toy``), which swaps the real detectors
for deterministic synthetic ToyDetector modules (see
``tcsr/models/toy.py``) so the whole repository runs on a CPU in about a
minute without checkpoints or datasets::

    python scripts/smoke.py

Equivalent to the manual commands::

    python tools/calibrate.py --config configs/hrsid_rtdetr.json --toy \
        --seed 12001 --variants B,G --stage1-updates 5 --stage2-updates 5
    python tools/evaluate.py --config configs/hrsid_rtdetr.json --toy \
        --seed 12001 --model fp:student \
        --model outputs/hrsid_rtdetr/toy/seed12001/variant_B/qmodel.pt \
        --model outputs/hrsid_rtdetr/toy/seed12001/variant_G/qmodel.pt
    python tools/method_comparison.py \
        --eval-json "outputs/hrsid_rtdetr/toy/seed12001/variant_*/eval_toy_eval.json" \
        --eval-json "outputs/hrsid_rtdetr/toy/seed12001/eval_toy_eval_fp_student.json" \
        --contrast G-B --contrast G-fp_student --group-by config
    python tools/diagnostics.py --config configs/hrsid_rtdetr.json --toy \
        --seed 12001 --stage1-updates 5 --stage2-updates 5

The toy numbers are meaningless by construction (synthetic weights, 5
updates instead of 6000/8000); the script only verifies that every
stage runs, writes its artifacts, and the artifacts agree with the
readers downstream of them.  Add ``--full`` for the complete variant
table A..I.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO_ROOT, "tools")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=os.path.join("configs",
                                                    "hrsid_rtdetr.json"))
    p.add_argument("--seed", type=int, default=12001,
                   help="toy seed (default: 12001)")
    p.add_argument("--variants", default="B,G",
                   help="comma-separated variants (default: B,G)")
    p.add_argument("--full", action="store_true",
                   help="run the complete variant table A..I")
    p.add_argument("--stage1-updates", type=int, default=5)
    p.add_argument("--stage2-updates", type=int, default=5)
    p.add_argument("--toy-images", type=int, default=16)
    p.add_argument("--eval-images", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-diagnostics", action="store_true")
    return p.parse_args(argv)


def run(step: str, cmd: list) -> None:
    print(f"\n{'=' * 72}\n[smoke] step: {step}\n{'=' * 72}", flush=True)
    print("  " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"[smoke] step {step!r} failed with code "
                         f"{proc.returncode}")


def main(argv=None):
    args = parse_args(argv)
    if args.full:
        args.variants = "A,B,C,C_Q,D,E,F,G,H,I"

    py = [sys.executable, "-u"]
    common = ["--config", args.config, "--toy", "--device", args.device]
    updates = ["--stage1-updates", str(args.stage1_updates),
               "--stage2-updates", str(args.stage2_updates)]

    # 1. Calibrate the variant table (Stage 1 + Stage 2).
    run("calibrate", py + ["tools/calibrate.py"] + common
        + ["--seed", str(args.seed), "--variants", args.variants,
           "--toy-images", str(args.toy_images)] + updates)

    # 2. Evaluate the FP student and every calibrated variant.
    # The output directory depends on the config's name field.
    with open(os.path.join(REPO_ROOT, args.config), encoding="utf-8") as f:
        exp_name = json.load(f)["name"]
    seed_dir = os.path.join("outputs", exp_name, "toy",
                            f"seed{args.seed:05d}")
    models = ["--model", "fp:student"]
    for v in [x.strip() for x in args.variants.split(",") if x.strip()]:
        models += ["--model",
                   os.path.join(seed_dir, f"variant_{v}", "qmodel.pt")]
    run("evaluate", py + ["tools/evaluate.py"] + common
        + ["--seed", str(args.seed), "--toy-images",
           str(args.eval_images)] + models)

    # 3. Paired method comparison over the toy records.
    eval_split = "toy_eval"
    run("method-comparison", py + ["tools/method_comparison.py"]
        + ["--eval-json",
           os.path.join(seed_dir, "variant_*", f"eval_{eval_split}.json"),
           "--eval-json",
           os.path.join(seed_dir, f"eval_{eval_split}_fp_student.json"),
           "--contrast", "G-B", "--contrast", "G-fp_student",
           "--group-by", "config",
           "--output", os.path.join(seed_dir, "comparison.json")])

    # 4. Two-discrepancy diagnostics (variant G).
    if not args.skip_diagnostics:
        run("diagnostics", py + ["tools/diagnostics.py"] + common
            + ["--seed", str(args.seed), "--variant", "G",
               "--toy-images", str(args.toy_images)] + updates)

    # ------------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------------
    print(f"\n{'=' * 72}\n[smoke] summary\n{'=' * 72}")
    records = sorted(glob.glob(os.path.join(
        REPO_ROOT, seed_dir, "**", f"eval_{eval_split}*.json"),
        recursive=True))
    for path in records:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        ap = rec.get("metrics", {}).get("AP", float("nan"))
        print(f"  {rec.get('method', '?'):>10}: AP={100 * ap:6.2f} "
              f"(n={rec.get('n_images', 0)})")
    artifacts = ["run.json", "diagnostics.json", "comparison.json"]
    for name in artifacts:
        path = os.path.join(REPO_ROOT, seed_dir, name)
        status = "OK" if os.path.exists(path) else "--"
        print(f"  {name:>20}: {status}")
    print(f"\n[smoke] PASSED -- toy pipeline end-to-end "
          f"({len(records)} eval records under {seed_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
