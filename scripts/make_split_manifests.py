"""Generate the split manifests under ``manifests/splits/``.

The split manifests (JSON image-id lists, see ``manifests/datasets.json``)
are part of the reproducibility record but depend on the dataset files,
so the repository ships a generator instead of hard-coding ids.  Run this
once after placing the datasets under ``data/`` as described in
``manifests/datasets.json``::

    python scripts/make_split_manifests.py --dataset hrsid
    python scripts/make_split_manifests.py --dataset ssdd
    python scripts/make_split_manifests.py --dataset coco
    python scripts/make_split_manifests.py --dataset all

What is written (JSON arrays of sorted image ids):

    manifests/splits/hrsid_train.json   3000 ids   (calibration source)
    manifests/splits/hrsid_val.json      642 ids   (dev)
    manifests/splits/hrsid_test.json    1962 ids   (held-out evaluation)
    manifests/splits/ssdd_train.json     800 ids   (calibration source)
    manifests/splits/ssdd_val.json       128 ids   (dev)
    manifests/splits/ssdd_test.json      232 ids   (held-out evaluation)
    manifests/splits/coco_dev.json      5000 ids   (dev subset of train2017)
    manifests/splits/coco_val2017.json  5000 ids   (held-out evaluation)

HRSID / SSDD: the split ids are read verbatim from each split's
``annotation.json`` (official split; scenes and crops stay within one
split) and verified against the paper's counts.

COCO: ``coco_val2017.json`` lists all val2017 ids; ``coco_dev.json`` is a
fixed 5000-image subset of train2017 sampled with the documented release
seed ``COCO_DEV_SEED`` below (any fixed seed makes the subset -- and
hence every dev-split diagnostic -- reproducible; the held-out
evaluation set is the full official val2017 either way).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

#: release convention for the COCO dev subset (documented, fixed)
COCO_DEV_SEED = 1701
#: paper split sizes (manifests/datasets.json)
EXPECTED = {
    "hrsid": {"train": 3000, "val": 642, "test": 1962},
    "ssdd": {"train": 800, "val": 128, "test": 232},
}
COCO_DEV_SIZE = 5000
COCO_VAL2017_SIZE = 5000

SPLITS = {
    "hrsid": ("train", "val", "test"),
    "ssdd": ("train", "val", "test"),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True,
                   choices=("hrsid", "ssdd", "coco", "all"))
    p.add_argument("--data-root", default="data",
                   help="dataset root (default: data/)")
    p.add_argument("--output-dir", default=None,
                   help="manifest directory (default: manifests/splits)")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the expected-count checks")
    return p.parse_args(argv)


def load_image_ids(annotation_file: str) -> list:
    from tcsr.datasets import load_coco_json
    data = load_coco_json(annotation_file)
    ids = sorted({int(img["id"]) for img in data.get("images", [])})
    if not ids:
        raise ValueError(f"no images listed in {annotation_file}")
    return ids


def write_manifest(path: str, ids: list) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ids, f)
        f.write("\n")
    print(f"[splits] wrote {len(ids):5d} ids -> {path}")


def make_sar(dataset: str, data_root: str, out_dir: str,
             verify: bool) -> None:
    for split in SPLITS[dataset]:
        ann = os.path.join(data_root, dataset, split, "annotation.json")
        if not os.path.exists(ann):
            raise FileNotFoundError(
                f"annotation file not found: {ann}; place {dataset.upper()} "
                "under the data root as described in manifests/datasets.json")
        ids = load_image_ids(ann)
        if verify:
            want = EXPECTED[dataset][split]
            if len(ids) != want:
                raise SystemExit(
                    f"{ann}: expected {want} images for split {split!r}, "
                    f"found {len(ids)} (--no-verify skips this check)")
        write_manifest(os.path.join(out_dir, f"{dataset}_{split}.json"), ids)
    # Cross-split leakage must be zero (paper: scenes/crops stay within
    # one split); report it here, tools/audit_splits.py is the formal audit.
    sets = {}
    for split in SPLITS[dataset]:
        with open(os.path.join(out_dir, f"{dataset}_{split}.json"),
                  encoding="utf-8") as f:
            sets[split] = set(json.load(f))
    names = sorted(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = sets[names[i]] & sets[names[j]]
            if overlap:
                raise SystemExit(
                    f"cross-split overlap {names[i]} & {names[j]}: "
                    f"{len(overlap)} leaked ids (duplicate audit FAILED)")
    print(f"[splits] {dataset}: duplicate audit OK "
          "(no cross-split overlap)")


def make_coco(data_root: str, out_dir: str, verify: bool) -> None:
    val_ann = os.path.join(data_root, "coco", "val2017", "annotation.json")
    train_ann = os.path.join(data_root, "coco", "train2017", "annotation.json")
    for path in (val_ann, train_ann):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"annotation file not found: {path}; place COCO under the "
                "data root as described in manifests/datasets.json")

    val_ids = load_image_ids(val_ann)
    if verify and len(val_ids) != COCO_VAL2017_SIZE:
        raise SystemExit(f"{val_ann}: expected {COCO_VAL2017_SIZE} val2017 "
                         f"images, found {len(val_ids)}")
    write_manifest(os.path.join(out_dir, "coco_val2017.json"), val_ids)

    train_ids = load_image_ids(train_ann)
    if len(train_ids) < COCO_DEV_SIZE:
        raise SystemExit(f"{train_ann}: need at least {COCO_DEV_SIZE} "
                         f"train2017 images, found {len(train_ids)}")
    rng = np.random.default_rng(COCO_DEV_SEED)
    dev = sorted(int(i) for i in rng.choice(np.asarray(train_ids),
                                            size=COCO_DEV_SIZE,
                                            replace=False))
    overlap = set(dev) & set(val_ids)
    if overlap:
        raise SystemExit(f"coco dev subset overlaps val2017: {len(overlap)} "
                         "ids (check the annotation files)")
    write_manifest(os.path.join(out_dir, "coco_dev.json"), dev)
    print(f"[splits] coco dev subset: seed={COCO_DEV_SEED} "
          f"{COCO_DEV_SIZE} of {len(train_ids)} train2017 images")


def main(argv=None):
    args = parse_args(argv)
    data_root = os.path.join(REPO_ROOT, args.data_root) \
        if not os.path.isabs(args.data_root) else args.data_root
    out_dir = args.output_dir or os.path.join(REPO_ROOT, "manifests", "splits")

    datasets = (("hrsid", "ssdd", "coco") if args.dataset == "all"
               else (args.dataset,))
    for dataset in datasets:
        print(f"\n=== split manifests: {dataset} (root: {data_root}) ===")
        if dataset in ("hrsid", "ssdd"):
            make_sar(dataset, data_root, out_dir, verify=not args.no_verify)
        else:
            make_coco(data_root, out_dir, verify=not args.no_verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
