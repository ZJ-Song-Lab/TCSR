"""Generate one per-draw calibration manifest (fixed image list).

The paper fixes a calibration budget of 512 SAR / 1024 COCO images and
one image list per (checkpoint pair, draw seed); the same draw manifest
is shared by all methods within a checkpoint pair.  This tool samples
the draw's image list from the calibration split with the draw's
calibration seed and writes a JSON image-id list consumable by
``tools/calibrate.py --calib-manifest``.

Example::

    python tools/calibration_manifest.py \
        --config configs/hrsid_rtdetr.json --seed 12001

writes manifests/calibration/HR-01_12001.json (512 ids from the train
split).  Pass ``--output`` to override the destination.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import DEFAULT_CALIB_SPLIT, split_paths  # noqa: E402

from tcsr.config import load_config  # noqa: E402
from tcsr.datasets import load_coco_json  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True,
                   help="draw seed (the calibration seed)")
    p.add_argument("--budget", type=int, default=None,
                   help="images per manifest (default: the config's "
                        "calibration budget)")
    p.add_argument("--split", default=None,
                   help="source split (default: train / train2017)")
    p.add_argument("--output", default=None,
                   help="output path (default: manifests/calibration/"
                        "{pair}_{seed}.json)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    exp = load_config(args.config)
    budget = args.budget if args.budget is not None else exp.calib_images
    split = args.split or DEFAULT_CALIB_SPLIT[exp.dataset]

    ann, _ = split_paths(exp, split)
    if not os.path.exists(ann):
        raise FileNotFoundError(
            f"annotation file not found: {ann}; place the dataset under "
            f"{exp.data_root}/ as described in manifests/datasets.json")
    data = load_coco_json(ann)
    ids = [int(img["id"]) for img in data.get("images", [])]
    if not ids:
        raise ValueError(f"no images listed in {ann}")
    if budget > len(ids):
        raise ValueError(
            f"budget {budget} exceeds the {split} split size {len(ids)}")

    rng = np.random.default_rng(args.seed)
    chosen = sorted(int(i) for i in
                    rng.choice(np.asarray(ids), size=budget, replace=False))

    out = args.output or os.path.join(
        REPO_ROOT, "manifests", "calibration",
        f"{exp.checkpoint_pair}_{args.seed}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(chosen, f)
        f.write("\n")
    print(f"[calibration-manifest] {exp.checkpoint_pair} seed={args.seed} "
          f"split={split}: {len(chosen)}/{len(ids)} images -> {out}")


if __name__ == "__main__":
    main()
