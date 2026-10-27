"""Shared helpers for the command-line tools.

All tools are run from the repository root::

    python tools/calibrate.py --config configs/hrsid_rtdetr.json --toy

Data layout convention (see manifests/ for download pointers):

    {data_root}/{dataset}/{split}/annotation.json   COCO-format annotations
    {data_root}/{dataset}/{split}/{file_name}       images

with splits ``train/val/test`` for the SAR datasets (HRSID, SSDD) and
``train2017/val2017`` for COCO (the train2017 folder holds the paper's
5000-image dev subset; val2017 is the held-out evaluation set).
"""

from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from tcsr.config import ExperimentConfig, load_config  # noqa: E402

# ----------------------------------------------------------------------
# Reproducibility.
# ----------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    """Seed python / numpy / torch (CPU and all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# Dataset splits.
# ----------------------------------------------------------------------

#: splits of each dataset under {data_root}/{dataset}/
DATASET_SPLITS: Dict[str, Tuple[str, ...]] = {
    "hrsid": ("train", "val", "test"),
    "ssdd": ("train", "val", "test"),
    "coco": ("train2017", "val2017"),
}

#: where calibration images are drawn from (first N images, no shuffle)
DEFAULT_CALIB_SPLIT: Dict[str, str] = {
    "hrsid": "train",
    "ssdd": "train",
    "coco": "train2017",
}

#: where AP is evaluated
DEFAULT_EVAL_SPLIT: Dict[str, str] = {
    "hrsid": "test",
    "ssdd": "test",
    "coco": "val2017",
}


def split_paths(exp: ExperimentConfig, split: str) -> Tuple[str, str]:
    """(annotation_file, image_dir) of one split under ``exp.data_root``."""
    image_dir = os.path.join(exp.data_root, exp.dataset, split)
    return os.path.join(image_dir, "annotation.json"), image_dir


def open_split(exp: ExperimentConfig, split: str, shuffle: bool = False,
               seed: Optional[int] = None,
               batch_size: Optional[int] = None,
               image_ids: Optional[set] = None):
    """DataLoader over one split (deterministic order by default).

    ``image_ids`` restricts the split to a calibration manifest or a
    subset split (see load_image_id_manifest).
    """
    from tcsr.datasets import COCODetectionDataset, make_loader
    ann, img_dir = split_paths(exp, split)
    if not os.path.exists(ann):
        raise FileNotFoundError(
            f"annotation file not found: {ann}; place the dataset under "
            f"{exp.data_root}/ as described in manifests/")
    bs = batch_size if batch_size is not None else exp.method.batch_size
    dataset = COCODetectionDataset(img_dir, ann, input_size=exp.input_size,
                                   image_ids=image_ids)
    return make_loader(dataset, batch_size=bs, shuffle=shuffle,
                       num_workers=0, seed=seed)


def load_image_id_manifest(path: str) -> Optional[set]:
    """Optional JSON file holding a list of image ids.

    Used to materialize subset splits (e.g. the paper's 5000-image COCO
    dev subset of train2017) from a full annotation file.  Returns None
    when the file does not exist.
    """
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        ids = json.load(f)
    return {int(i) for i in ids}


# ----------------------------------------------------------------------
# Checkpoints.
# ----------------------------------------------------------------------


def load_checkpoint_manifest(path: Optional[str] = None) -> Dict:
    """manifests/checkpoints.json: {pairs: {id: {teacher, student, ...}}}."""
    path = path or os.path.join(REPO_ROOT, "manifests", "checkpoints.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_checkpoints(exp: ExperimentConfig, manifest: Dict,
                        teacher: Optional[str] = None,
                        student: Optional[str] = None,
                        repo_path: Optional[str] = None
                        ) -> Tuple[str, str, Optional[str]]:
    """Resolve the (teacher, student, repo_path) checkpoint triple.

    Priority: explicit CLI arguments > config fields > the manifest entry
    of ``exp.checkpoint_pair``.
    """
    entry = (manifest.get("pairs") or {}).get(exp.checkpoint_pair, {})
    t = teacher or exp.teacher_ckpt or entry.get("teacher") or ""
    s = student or exp.student_ckpt or entry.get("student") or ""
    rp = repo_path or entry.get("repo_path")
    if not t or not s:
        raise FileNotFoundError(
            f"no teacher/student checkpoints for pair {exp.checkpoint_pair!r}; "
            "pass --teacher/--student or fill manifests/checkpoints.json "
            "(see manifests/ for the published weight links)")
    return t, s, rp


def build_models(exp: ExperimentConfig, teacher_ckpt: str, student_ckpt: str,
                 repo_path: Optional[str], device: str
                 ) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Load the frozen (teacher, student) pair for a real setting."""
    if exp.model == "rtdetr":
        from tcsr.models.rtdetr import load_rtdetr
        teacher = load_rtdetr(teacher_ckpt, num_classes=exp.num_classes,
                              device=device, head_type=exp.head_type)
        student = load_rtdetr(student_ckpt, num_classes=exp.num_classes,
                              device=device, head_type=exp.head_type)
    elif exp.model == "defdetr":
        from tcsr.models.defdetr import load_defdetr
        teacher = load_defdetr(teacher_ckpt, repo_path=repo_path,
                               num_classes=exp.num_classes, device=device)
        student = load_defdetr(student_ckpt, repo_path=repo_path,
                               num_classes=exp.num_classes, device=device)
    else:
        raise ValueError(f"unknown model {exp.model!r}")
    teacher.eval()
    student.eval()
    return teacher, student


def adapter_factory_for(exp: ExperimentConfig, toy: bool = False):
    """``adapter_factory`` callable for the experiment's model field.

    ``toy=True`` selects the toy-detector factory regardless of the
    config's model field (the --toy smoke runs pair ToyDetector modules
    with the experiment's head type / class count).
    """
    from tcsr.models import get_adapter_factory
    model = "toy" if toy else exp.model
    return get_adapter_factory(model, head_type=exp.head_type,
                               num_classes=exp.num_classes)


# ----------------------------------------------------------------------
# Toy (smoke-test) mode.
# ----------------------------------------------------------------------

#: toy detectors consume (B, 3, 8, 8) inputs
TOY_INPUT_SIZE = 8
#: virtual original-image size for toy detections (absolute coordinates)
TOY_EVAL_SIZE = 64


def build_toy_pair(exp: ExperimentConfig, seed: int
                   ) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Deterministic (teacher, student) toy pair for --toy smoke runs.

    The student uses ``seed`` and the teacher ``seed + 10000`` so that
    every tool rebuilds the identical pair from the same seed.
    """
    from tcsr.models.toy import make_toy_detector
    student = make_toy_detector(seed=seed, head_type=exp.head_type,
                                num_classes=exp.num_classes)
    teacher = make_toy_detector(seed=seed + 10000, head_type=exp.head_type,
                                num_classes=exp.num_classes)
    teacher.eval()
    student.eval()
    return teacher, student


def synthetic_batches(n_images: int, input_size: int, batch_size: int,
                      seed: int, in_ch: int = 3) -> List[Tuple]:
    """Deterministic synthetic (images, ids) batches.

    ids are ``img_{i:06d}`` strings; the loader protocol consumed by
    ``tcsr.batching.collect_batches`` accepts the dict form produced by
    :func:`batches_to_loader`.
    """
    g = torch.Generator().manual_seed(int(seed))
    images = torch.rand(int(n_images), in_ch, int(input_size), int(input_size),
                        generator=g)
    batches: List[Tuple] = []
    for i in range(0, images.shape[0], int(batch_size)):
        chunk = images[i:i + int(batch_size)]
        ids = [f"img_{j:06d}" for j in range(i, i + chunk.shape[0])]
        batches.append((chunk, ids))
    return batches


def batches_to_loader(batches: Sequence[Tuple]) -> List[Dict]:
    """Adapt (images, ids) tuples to the loader protocol (batch dicts)."""
    return [{"images": images, "image_ids": ids} for images, ids in batches]


def toy_loader(exp: ExperimentConfig, n_images: int, seed: int):
    """Synthetic calibration-style loader for --toy runs."""
    batches = synthetic_batches(n_images, TOY_INPUT_SIZE,
                                exp.method.batch_size, seed)
    return batches_to_loader(batches)


# ----------------------------------------------------------------------
# Prediction / evaluation plumbing.
# ----------------------------------------------------------------------


@torch.no_grad()
def predict_loader(adapter, loader, device: str
                   ) -> Tuple[List, List, Dict]:
    """Run the adapter's model over a raw DataLoader.

    Unlike ``tcsr.batching.collect_batches`` this keeps the letterbox
    parameters (needed to map normalized boxes back to the original
    image).  Returns (image_ids, predictions, letterboxes) in loader
    order, where letterboxes maps image_id -> (scale, pad_left, pad_top).
    """
    model = adapter.model
    model.eval()
    ids: List = []
    preds: List = []
    boxes: Dict = {}
    for batch in loader:
        images = batch["images"].to(device)
        out = adapter.predict(images)
        for image_id, p, lb in zip(batch["image_ids"], out,
                                    batch["letterbox"]):
            ids.append(image_id)
            preds.append(p)
            boxes[image_id] = lb
    return ids, preds, boxes


def category_ids_from_annotation(annotation_file: str, num_classes: int
                                 ) -> Optional[List[int]]:
    """Channel -> COCO category id mapping implied by the annotation json.

    Detections index the model's logit channels (channel c -> category
    c + 1, the default of ``predictions_to_detections``) whenever the json
    categories are exactly ``1..num_classes``; otherwise the sorted
    category ids are returned and must be passed as ``category_ids``.
    """
    from tcsr.datasets import load_coco_json
    data = load_coco_json(annotation_file)
    ids = sorted({int(c["id"]) for c in data.get("categories", [])})
    if len(ids) == num_classes and ids == list(range(1, num_classes + 1)):
        return None
    return ids


# ----------------------------------------------------------------------
# Output plumbing.
# ----------------------------------------------------------------------


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)!r}")


def save_json(path: str, obj) -> None:
    """Dump ``obj`` as pretty JSON (numpy scalars are cast)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def run_metadata(argv: Sequence[str], mode: str,
                 seed: Optional[int] = None, extra: Optional[Dict] = None
                 ) -> Dict:
    """Provenance block written next to every tool run."""
    meta = {
        "mode": mode,
        "argv": [str(a) for a in argv],
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    if extra:
        meta.update(extra)
    return meta


def seed_dir(output_root: str, seed: int) -> str:
    """Per-seed directory ``{output_root}/seed{seed:05d}``."""
    return os.path.join(output_root, f"seed{int(seed):05d}")


def model_label(path: str) -> str:
    """Method label implied by the model's directory (variant_X -> X)."""
    d = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return d[len("variant_"):] if d.startswith("variant_") else d


def resolve_output_dir(exp: ExperimentConfig, pair_label: str,
                       out_root: Optional[str] = None) -> str:
    """``{out_root|exp.output_dir}/{exp.name}/{pair_label}``."""
    root = out_root or exp.output_dir
    return os.path.join(root, exp.name, pair_label)


def parse_variant_list(text: Optional[str]) -> Optional[List[str]]:
    """Parse a comma-separated variant list ('G,B,CQ' -> ['G','B','C_Q'])."""
    if text is None:
        return None
    from tcsr.variants import normalize_variant
    return [normalize_variant(v) for v in text.split(",") if v.strip()]
