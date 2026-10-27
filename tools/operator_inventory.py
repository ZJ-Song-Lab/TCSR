"""Dump the per-operator quantization inventory of a detector.

Real setting (the student detector of a checkpoint pair)::

    python tools/operator_inventory.py --config configs/hrsid_rtdetr.json \
        --model fp:student --output results/tab_operatorinventory.csv

``--model`` accepts ``fp:student`` / ``fp:teacher``, a saved
``qmodel.pt``, or ``toy`` / ``toy:teacher`` (synthetic detector, no
checkpoints needed).  ``--profile`` defaults to the config's profile
(``mixed`` for the published results; ``w8a8`` / ``w4a4`` for the
bit-sweep).  Each supported Linear/Conv operator is mapped through
``tcsr.quant.policy`` and reported with

- the fake-quant bits used by the framework evaluation (W4A8 / W8A8 / W4A4),
- the engine format of the exported TensorRT engine: the mixed profile's
  decoder runs weight-only INT4 with FP16 compute (W4A16) while every
  other quantized operator runs W8A8; detection heads, the input stem,
  embeddings and positional/reference/query projections stay FP16
  (they are excluded from the quantized inventory).

With ``--output`` the table is written as CSV (the paper's "operator
inventories" release item); without it the table is printed.  The
archived W4A16/W8A8/FP16 MACs shares of the mixed profile are reported
from ``tcsr.quant.policy.macs_shares`` (HRSID/HR-02 engine inventory).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, Iterator, List, Optional

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (build_models, build_toy_pair,  # noqa: E402
                    load_checkpoint_manifest, model_label,
                    resolve_checkpoints, run_metadata, save_json)

from tcsr.config import load_config  # noqa: E402
from tcsr.quant.policy import PROFILES, get_bit_assignment, macs_shares  # noqa: E402

#: engine format of each (profile, fake-quant) combination
_ENGINE_FORMATS = {
    ("mixed", "W4A8"): "W4A16 (weight-only INT4, FP16 compute)",
    ("mixed", "W8A8"): "W8A8 (Q/DQ)",
    ("w8a8", "W8A8"): "W8A8 (Q/DQ)",
    ("w4a4", "W4A4"): "framework bit-sweep (not deployed)",
}


def iter_quant_ops(model: torch.nn.Module) -> Iterator[Dict[str, str]]:
    """Yield one row per supported Linear/Conv operator of ``model``."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            op_type, shape = "Linear", (f"{module.out_features}"
                                        f"x{module.in_features}")
        elif isinstance(module, torch.nn.Conv2d):
            ks = module.kernel_size
            k = (str(ks[0]) if len(set(ks)) == 1
                 else "x".join(str(d) for d in ks))
            op_type, shape = "Conv2d", (f"{module.out_channels}"
                                        f"x{module.in_channels}x{k}")
        else:
            continue
        yield {"op_name": name, "op_type": op_type, "shape": shape}


def inventory_rows(model: torch.nn.Module, setting: str, label: str,
                   profile: str) -> List[Dict[str, str]]:
    """Full operator-inventory table of one model under one profile."""
    rows: List[Dict[str, str]] = []
    for op in iter_quant_ops(model):
        bits = get_bit_assignment(op["op_name"], profile)
        if bits is None:
            fake_quant, engine = "FP16 (excluded)", "FP16"
        else:
            fake_quant = f"W{bits.w_bits}A{bits.a_bits}"
            engine = _ENGINE_FORMATS.get((profile, fake_quant), fake_quant)
        rows.append({"setting": setting, "model": label, "profile": profile,
                     "op_name": op["op_name"], "op_type": op["op_type"],
                     "shape": op["shape"], "fake_quant_bits": fake_quant,
                     "engine_format": engine})
    return rows


def write_csv(rows: List[Dict[str, str]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fields = ["setting", "model", "profile", "op_name", "op_type", "shape",
              "fake_quant_bits", "engine_format"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def print_rows(rows: List[Dict[str, str]]) -> None:
    if not rows:
        print("(no Linear/Conv operators found)")
        return
    fields = ["op_name", "op_type", "shape", "fake_quant_bits",
              "engine_format"]
    widths = {f: max(len(f), *(len(r[f]) for r in rows)) for f in fields}
    print("  ".join(f.rjust(widths[f]) for f in fields))
    for r in rows:
        print("  ".join(r[f].rjust(widths[f]) for f in fields))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="experiment config JSON")
    p.add_argument("--model", required=True,
                   help="fp:student | fp:teacher | qmodel.pt | toy | "
                        "toy:teacher")
    p.add_argument("--seed", type=int, default=None,
                   help="toy pair seed (required for toy sources)")
    p.add_argument("--profile", default=None,
                   help=f"bit profile (default: the config's profile; "
                        f"one of {PROFILES})")
    p.add_argument("--output", default=None,
                   help="CSV output path (default: print to stdout)")
    p.add_argument("--device", default="cpu")
    # real-mode checkpoint resolution (same as tools/evaluate.py)
    p.add_argument("--teacher", default=None, help="teacher checkpoint")
    p.add_argument("--student", default=None, help="student checkpoint")
    p.add_argument("--manifest", default=None,
                   help="checkpoint manifest (default: manifests/checkpoints.json)")
    p.add_argument("--repo-path", default=None,
                   help="official Deformable DETR repo (defdetr only)")
    return p.parse_args(argv)


def _resolve_model(exp, args) -> torch.nn.Module:
    """Load the requested model (mirrors tools/export_onnx.py)."""
    spec = args.model
    if spec.startswith("toy"):
        if args.seed is None:
            raise SystemExit("--seed is required for toy sources")
        teacher, student = build_toy_pair(exp, args.seed)
        which = "teacher" if spec.startswith("toy:teacher") else "student"
        model = student if which == "student" else teacher
        return model.to(args.device), f"toy_{which}"
    if spec.startswith("fp:"):
        which = spec.split(":", 1)[1]
        if which not in ("student", "teacher"):
            raise SystemExit(f"--model {spec}: expected fp:student or "
                             "fp:teacher")
        manifest = load_checkpoint_manifest(args.manifest)
        t_ckpt, s_ckpt, repo = resolve_checkpoints(
            exp, manifest, teacher=args.teacher, student=args.student,
            repo_path=args.repo_path)
        teacher, student = build_models(exp, t_ckpt, s_ckpt, repo,
                                        args.device)
        model = student if which == "student" else teacher
        return model, f"fp_{which}"
    model = torch.load(spec, map_location="cpu", weights_only=False)
    model.to(args.device)
    model.eval()
    return model, model_label(spec)


def main(argv=None):
    args = parse_args(argv)
    exp = load_config(args.config)
    profile = args.profile or exp.quant_profile
    if profile not in PROFILES:
        raise SystemExit(f"--profile: expected one of {PROFILES}")
    model, label = _resolve_model(exp, args)
    rows = inventory_rows(model, exp.setting, label, profile)
    if args.output:
        write_csv(rows, args.output)
        print(f"[operator_inventory] wrote {args.output} "
              f"({len(rows)} operators)")
    else:
        print_rows(rows)
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["fake_quant_bits"]] = counts.get(r["fake_quant_bits"], 0) + 1
    print(f"[operator_inventory] profile={profile} model={label}: "
          + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if profile == "mixed":
        w4, w8, fp = macs_shares("mixed")
        print(f"[operator_inventory] archived HRSID/HR-02 engine MACs "
              f"shares: W4A16 {w4 * 100:.1f}% / W8A8 {w8 * 100:.1f}% / "
              f"FP16 {fp * 100:.1f}%")
    if args.output:
        save_json(args.output + ".inventory.json",
                  run_metadata(sys.argv if argv is None else argv,
                               mode="operator_inventory",
                               extra={"config": args.config,
                                      "model": args.model,
                                      "profile": profile,
                                      "n_operators": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
