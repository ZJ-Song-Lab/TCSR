"""Export a detector to ONNX for deployment (paper deployment protocol).

Real setting::

    python tools/export_onnx.py --config configs/hrsid_rtdetr.json \
        --model fp:student \
        --model outputs/hrsid_rtdetr/HR-02/seed512101/variant_G/qmodel.pt \
        --check

``--model`` accepts ``fp:student`` / ``fp:teacher`` (full-precision
checkpoints), a saved ``qmodel.pt`` (hardened fake-quant model, the exact
deployed arithmetic), or ``toy`` / ``toy:teacher`` (synthetic detectors;
no checkpoints or datasets needed) and may be given multiple times.

Outputs (under ``--output-dir``, default ``exports``)::

    exports/{label}.onnx               the traced graph
    exports/{label}.onnx.export.json   provenance + verification report

``--check`` verifies the graph against the framework model with
onnxruntime when installed (the source--engine closure check of the
paper).  The exported graph carries the exact fake-quant arithmetic of
the calibrated model; native W4A16/W8A8 TensorRT engines additionally
require Q/DQ-annotated quantization of the same graph (see
tools/build_engine.py and README "Deployment").
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (TOY_INPUT_SIZE, build_models, build_toy_pair,  # noqa: E402
                    load_checkpoint_manifest, model_label, resolve_checkpoints,
                    run_metadata, save_json)

from tcsr.config import load_config  # noqa: E402
from tcsr.deployment import (DEPLOYMENT_PROTOCOL, export_onnx,  # noqa: E402
                             verify_onnx)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="experiment config JSON")
    p.add_argument("--model", action="append", required=True,
                   help="fp:student | fp:teacher | qmodel.pt | toy | "
                        "toy:teacher (repeatable)")
    p.add_argument("--seed", type=int, default=None,
                   help="toy pair seed (required for toy sources)")
    p.add_argument("--output-dir", default="exports",
                   help="directory for the exported graphs")
    p.add_argument("--input-size", type=int, default=None,
                   help="export input size (default: config input_size; "
                        "toy sources always use the toy size)")
    p.add_argument("--opset", type=int,
                   default=DEPLOYMENT_PROTOCOL["opset"],
                   help=f"ONNX opset (default: {DEPLOYMENT_PROTOCOL['opset']})")
    p.add_argument("--device", default="cpu")
    p.add_argument("--check", action="store_true",
                   help="verify engine outputs against the framework model "
                        "(requires onnxruntime)")
    p.add_argument("--atol", type=float, default=1e-3,
                   help="closure tolerance of --check (default: 1e-3)")
    # real-mode checkpoint resolution (same as tools/evaluate.py)
    p.add_argument("--teacher", default=None, help="teacher checkpoint")
    p.add_argument("--student", default=None, help="student checkpoint")
    p.add_argument("--manifest", default=None,
                   help="checkpoint manifest (default: manifests/checkpoints.json)")
    p.add_argument("--repo-path", default=None,
                   help="official Deformable DETR repo (defdetr only)")
    return p.parse_args(argv)


def resolve_models(exp, args):
    """Yield (label, model, kind) triples for the requested sources."""
    fp_models = {}
    for spec in args.model:
        if spec.startswith("toy"):
            if args.seed is None:
                raise SystemExit("--seed is required for toy sources")
            if "student" not in fp_models:
                teacher, student = build_toy_pair(exp, args.seed)
                teacher.to(args.device)
                student.to(args.device)
                fp_models = {"teacher": teacher, "student": student}
            which = "teacher" if spec.startswith("toy:teacher") else "student"
            label = f"toy_{which}"
            yield label, fp_models[which], "toy"
        elif spec.startswith("fp:"):
            which = spec.split(":", 1)[1]
            if which not in ("student", "teacher"):
                raise SystemExit(f"--model {spec}: expected fp:student or "
                                 "fp:teacher")
            if "student" not in fp_models:
                manifest = load_checkpoint_manifest(args.manifest)
                t_ckpt, s_ckpt, repo = resolve_checkpoints(
                    exp, manifest, teacher=args.teacher,
                    student=args.student, repo_path=args.repo_path)
                teacher, student = build_models(exp, t_ckpt, s_ckpt, repo,
                                                args.device)
                fp_models = {"teacher": teacher, "student": student}
            yield f"fp_{which}", fp_models[which], exp.model
        else:
            label = model_label(spec)
            try:
                model = torch.load(spec, map_location="cpu",
                                   weights_only=False)
            except TypeError:  # older torch without weights_only
                model = torch.load(spec, map_location="cpu")
            model.to(args.device)
            model.eval()
            yield label, model, exp.model


def main(argv=None):
    args = parse_args(argv)
    exp = load_config(args.config)
    provenance = []
    for label, model, kind in resolve_models(exp, args):
        input_size = TOY_INPUT_SIZE if kind == "toy" else (
            args.input_size or exp.input_size)
        out_path = os.path.join(args.output_dir, f"{label}.onnx")
        export_onnx(model, out_path, input_size, kind, opset=args.opset,
                    device=args.device)
        verify = verify_onnx(out_path, model, input_size, kind,
                             device=args.device, atol=args.atol) \
            if args.check else "skipped (--check not given)"
        if args.check and verify is None:
            verify = "skipped (onnxruntime not installed)"
        record = {
            "label": label,
            "source_kind": kind,
            "input_size": input_size,
            "opset": args.opset,
            "verify": verify,
        }
        save_json(f"{out_path}.export.json", record)
        provenance.append(record)
        print(f"[export_onnx] {label}: {out_path} "
              f"(kind={kind}, input={input_size}, opset={args.opset}, "
              f"verify={record['verify']})")
    meta = run_metadata(sys.argv if argv is None else argv, mode="export_onnx")
    meta["exports"] = provenance
    save_json(os.path.join(args.output_dir, "export_onnx.json"), meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
