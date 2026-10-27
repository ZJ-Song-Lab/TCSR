"""Build a TensorRT engine from an exported ONNX graph.

::

    python tools/build_engine.py --onnx exports/variant_G.onnx \
        --out exports/variant_G.engine --fp16

This is the engine-building step of the deployment protocol (TensorRT
10.3).  ``--fp16`` produces the full-precision FP16 engines of the paper
(the exported graph already carries the exact fake-quant arithmetic of
the calibrated model, so the FP16 engine reproduces the framework
evaluation).  The paper's native W4A16 (weight-only INT4) / W8A8 mixed
engines start from a Q/DQ-annotated version of the same graph, produced
with TensorRT ModelOptimizer or Polygraphy outside this repository, and
are then built the same way.  Requires CUDA and the ``tensorrt``
python package.
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import run_metadata, save_json  # noqa: E402

from tcsr.deployment import build_trt_engine  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--onnx", required=True, help="input ONNX graph")
    p.add_argument("--out", default=None,
                   help="output engine path (default: input path + .engine)")
    p.add_argument("--fp16", action="store_true", default=True,
                   help="enable the FP16 build flag (default: on)")
    p.add_argument("--no-fp16", dest="fp16", action="store_false",
                   help="build an FP32 engine")
    p.add_argument("--workspace-gib", type=float, default=None,
                   help="builder workspace limit in GiB")
    p.add_argument("--verbose", action="store_true",
                   help="verbose TensorRT logger")
    p.add_argument("--report", default=None,
                   help="write a JSON provenance report next to the engine")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = args.out or os.path.splitext(args.onnx)[0] + ".engine"
    build_trt_engine(args.onnx, out, fp16=args.fp16,
                     workspace_gib=args.workspace_gib, verbose=args.verbose)
    print(f"[build_engine] {out} (fp16={args.fp16})")
    if args.report:
        save_json(args.report, run_metadata(
            sys.argv if argv is None else argv, mode="build_engine",
            extra={"onnx": args.onnx, "engine": out, "fp16": args.fp16,
                   "workspace_gib": args.workspace_gib}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
