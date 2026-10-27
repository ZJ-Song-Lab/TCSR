"""Cross-split duplicate audit (reproducibility record).

Reports per-split image counts, pairwise overlaps (leakage), and the
union size for one experiment's splits.  The paper requires parent
scenes and crops to remain within one split; every overlap_* field must
be zero.

Example::

    python tools/audit_splits.py --config configs/hrsid_rtdetr.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import DATASET_SPLITS, split_paths  # noqa: E402

from tcsr.config import load_config  # noqa: E402
from tcsr.datasets import audit_splits  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--output", default=None,
                   help="write the audit dict to this JSON path")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    exp = load_config(args.config)
    files = {}
    for split in DATASET_SPLITS[exp.dataset]:
        ann, _ = split_paths(exp, split)
        if os.path.exists(ann):
            files[split] = ann
        else:
            print(f"[audit] WARNING: missing annotation file: {ann}",
                  file=sys.stderr)
    if not files:
        raise SystemExit("no annotation files found; place the dataset "
                         "as described in manifests/datasets.json")

    report = audit_splits(files)
    for key in sorted(report):
        print(f"{key}: {report[key]:.0f}")
    overlaps = [k for k in report if k.startswith("overlap_")]
    leaked = {k: report[k] for k in overlaps if report[k] > 0}
    print("duplicate audit: " + ("FAILED " + json.dumps(leaked)
                                 if leaked else "OK (no cross-split overlap)"))

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)),
                    exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"config": args.config, "audit": report}, f,
                      indent=2)
            f.write("\n")
        print(f"[audit] wrote {args.output}")


if __name__ == "__main__":
    main()
