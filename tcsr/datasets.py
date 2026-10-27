"""COCO-format detection datasets and letterbox preprocessing.

The paper evaluates HRSID, SSDD (scene-grouped split), and COCO, all in
COCO json format.  Images are letterboxed to a square input (default
640 x 640, right-bottom padding) and ImageNet-normalized; the letterbox
parameters (scale, pad_left, pad_top) are returned alongside each batch so
that normalized cxcywh predictions can be mapped back to original-image
absolute xywh coordinates (see tcsr.evaluation.predictions_to_detections
and tcsr.diagnostics.gap_stratified).

Dataset archives and expected sizes are listed in manifests/; the code
never downloads them itself.

SPLIT_SIZES documents the split sizes used in the paper:

    HRSID  train 3000 / val 642 / test 1962  (5604 images)
    SSDD   train 800 / val 128 / test 232    (1160, scene-grouped split)
    COCO   dev = 5000-image train2017 subset, held-out = val2017 (5000)
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

try:  # Pillow >= 9.1
    _RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:  # pragma: no cover - old Pillow
    _RESAMPLE = Image.BILINEAR

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SPLIT_SIZES: Dict[str, Dict[str, int]] = {
    "hrsid": {"train": 3000, "val": 642, "test": 1962, "total": 5604},
    "ssdd": {"train": 800, "val": 128, "test": 232, "total": 1160},
    "coco": {"dev": 5000, "heldout": 5000},
}


def letterbox_params(width: int, height: int,
                     size: int = 640) -> Tuple[float, float, float]:
    """Letterbox parameters for a right-bottom-padded square canvas.

    Returns (scale, pad_left, pad_top) with scale = size / max(w, h) side
    and zero padding on the left/top; the resized image occupies the
    top-left corner and the remaining rows/columns are zero-padded.
    """
    scale = min(size / float(width), size / float(height))
    return scale, 0.0, 0.0


def load_coco_json(annotation_file: str) -> dict:
    """Load a COCO-format annotation file."""
    with open(annotation_file, "r", encoding="utf-8") as f:
        return json.load(f)


class COCODetectionDataset(torch.utils.data.Dataset):
    """COCO-format detection dataset with letterboxed inputs.

    Yields dicts {"images": (3, size, size) float tensor, "image_ids": id,
    "letterbox": (scale, pad_left, pad_top), "width", "height"}; the
    collate_fn stacks images and keeps ids / letterbox tuples as lists.
    Annotations are not attached here - ground truth is consumed from the
    json directly via load_ground_truth / ground_truth_by_image.
    """

    def __init__(self, image_dir: str, annotation_file: str,
                 input_size: int = 640,
                 image_ids: Optional[Sequence[int]] = None):
        self.image_dir = image_dir
        self.input_size = input_size
        data = load_coco_json(annotation_file)
        self.coco = data
        self.images = list(data.get("images", []))
        if image_ids is not None:
            keep = {int(i) for i in image_ids}
            self.images = [img for img in self.images
                           if int(img["id"]) in keep]
        self._index = {int(img["id"]): i for i, img in enumerate(self.images)}

    def __len__(self) -> int:
        return len(self.images)

    def image_info(self, idx: int) -> dict:
        return self.images[idx]

    def __getitem__(self, idx: int) -> dict:
        info = self.images[idx]
        path = f"{self.image_dir}/{info['file_name']}"
        with Image.open(path) as im:
            rgb = im.convert("RGB")
        w = int(info.get("width") or rgb.width)
        h = int(info.get("height") or rgb.height)
        scale, pad_left, pad_top = letterbox_params(w, h, self.input_size)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        canvas = Image.new("RGB", (self.input_size, self.input_size))
        canvas.paste(rgb.resize((new_w, new_h), _RESAMPLE),
                     (int(pad_left), int(pad_top)))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        for c in range(3):
            arr[..., c] = (arr[..., c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        images = torch.from_numpy(arr.transpose(2, 0, 1).copy())
        return {"images": images, "image_ids": int(info["id"]),
                "letterbox": (scale, pad_left, pad_top),
                "width": w, "height": h}


def collate(batch: Sequence[dict]) -> dict:
    """Collate COCODetectionDataset items into a batch dict."""
    return {
        "images": torch.stack([b["images"] for b in batch]),
        "image_ids": [b["image_ids"] for b in batch],
        "letterbox": [b.get("letterbox") for b in batch],
    }


def make_loader(dataset: COCODetectionDataset, batch_size: int = 8,
                shuffle: bool = False, num_workers: int = 0,
                seed: Optional[int] = None) -> torch.utils.data.DataLoader:
    """DataLoader with the dataset's collate_fn (optional seeded shuffle)."""
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collate, generator=generator)


def load_ground_truth(annotation_file: str) -> List[dict]:
    """COCO annotations as evaluator entries.

    Returns a list of {image_id, category_id, bbox (abs xywh), area,
    iscrowd} dicts in json order, one per annotation with a valid bbox
    (the format consumed by tcsr.evaluation.COCOStyleEvaluator
    .add_ground_truth; 'ignore' flags, when present in the json, are
    passed through).
    """
    data = load_coco_json(annotation_file)
    out: List[dict] = []
    for ann in data.get("annotations", []):
        bbox = ann.get("bbox")
        if not bbox or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        entry = {"image_id": int(ann["image_id"]),
                 "category_id": int(ann["category_id"]),
                 "bbox": [float(v) for v in bbox[:4]],
                 "area": float(ann.get("area", bbox[2] * bbox[3])),
                 "iscrowd": bool(ann.get("iscrowd", 0))}
        if "ignore" in ann:
            entry["ignore"] = bool(ann["ignore"])
        out.append(entry)
    return out


def ground_truth_by_image(annotation_file: str) -> Dict[int, Tuple[np.ndarray,
                                                                   Tuple[int,
                                                                         int]]]:
    """image_id -> (gt boxes abs xywh (M, 4), (width, height)).

    The per-image image size comes from the json 'images' table; used by
    the gap-stratified ground-truth correctness diagnostic.
    """
    data = load_coco_json(annotation_file)
    sizes = {int(img["id"]): (int(img.get("width", 0)),
                              int(img.get("height", 0)))
             for img in data.get("images", [])}
    by_img: Dict[int, List[List[float]]] = {}
    for ann in data.get("annotations", []):
        bbox = ann.get("bbox")
        if not bbox or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        by_img.setdefault(int(ann["image_id"]), []).append(
            [float(v) for v in bbox[:4]])
    return {i: (np.asarray(b, dtype=np.float64).reshape(-1, 4), sizes.get(i))
            for i, b in by_img.items()}


def audit_splits(annotation_files: Dict[str, str]) -> Dict[str, float]:
    """Cross-split image overlap audit.

    annotation_files maps a split name to its json path.  Reports the
    per-split image counts, the pairwise intersection sizes (leakage), and
    the union size.
    """
    ids: Dict[str, set] = {}
    for name, path in annotation_files.items():
        data = load_coco_json(path)
        ids[name] = {int(img["id"]) for img in data.get("images", [])}
    out: Dict[str, float] = {}
    names = sorted(ids)
    for name in names:
        out[f"n_{name}"] = float(len(ids[name]))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[f"overlap_{a}_{b}"] = float(len(ids[a] & ids[b]))
    out["union"] = float(len(set().union(*ids.values())) if ids else 0)
    return out
