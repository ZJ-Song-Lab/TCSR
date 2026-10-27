"""Configuration handling.

Every default value below is frozen from the paper (Table
"Frozen method and optimization settings") and is shared by all published
variants unless a config JSON explicitly overrides it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class MethodConfig:
    """Method hyper-parameters (paper Table 3)."""

    # Student / teacher candidate selection (threshold precedes top-K; ties use
    # ascending query index).
    k_p: int = 30                # K_P: student anchor budget
    tau_p: float = 0.25          # tau_P: student confidence threshold
    k_t: int = 50                # K_T: teacher candidate budget
    tau_t: float = 0.30          # tau_T: teacher confidence threshold
    tau_d: float = 0.35          # feasibility threshold on d(y_T, y_P)

    # Bounded class-box discrepancy.
    epsilon: float = 1e-4        # label smoothing floor for sigmoid heads
    tau_cal: float = 2.0         # calibration temperature for sigmoid heads
    eta_cls: float = 0.50        # eta = (eta_cls, eta_l1, eta_giou), sum = 1
    eta_l1: float = 0.25
    eta_giou: float = 0.25

    # Teacher weighting.
    beta: float = 0.75           # priority gain in (1 + beta * u_j^0)
    lambda_cor: float = 0.60     # weight of the corroboration term

    # Student objective weights (shared by teacher and non-teacher variants).
    lambda_blk: float = 1.0
    lambda_fg: float = 1.0
    lambda_res: float = 0.20
    lambda_round: float = 0.01   # AdaRound soft-rounding regularizer weight

    # Optimization.
    stage1_lr: float = 2e-4      # Adam, per-block updates
    stage1_updates_per_block: int = 6000
    stage2_lr: float = 5e-5      # AdamW
    stage2_updates: int = 8000
    batch_size: int = 8

    # Residual MMD.
    sigma_min: float = 0.020     # lower bound of the median-distance bandwidth

    # Direction-conflict diagnostic.
    eps_g: float = 1e-12


@dataclass
class ExperimentConfig:
    """A complete experiment configuration (one dataset/model/quant setting)."""

    name: str = "hrsid_rtdetr"
    setting: str = "HRSID / RT-DETR"
    dataset: str = "hrsid"
    model: str = "rtdetr"             # rtdetr | defdetr | toy
    head_type: str = "softmax"        # softmax (explicit background node) | sigmoid
    num_classes: int = 1              # foreground classes
    quant_profile: str = "mixed"      # w8a8 | mixed (decoder W4A8, rest W8A8) | w4a4
    calib_images: int = 512           # 512 SAR / 1024 COCO
    input_size: int = 640
    mmd_bandwidth: float = 0.084      # frozen per setting: 0.084/0.079/0.112/0.108
    checkpoint_pair: str = "HR-02"    # representative pair id
    teacher_ckpt: str = ""            # filled from manifests/checkpoints.json
    student_ckpt: str = ""
    data_root: str = "data"
    output_dir: str = "outputs"
    method: MethodConfig = field(default_factory=MethodConfig)
    variants: List[str] = field(default_factory=lambda: [
        "A", "B", "C", "C_Q", "D", "E", "F", "G", "H", "I"])

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def _load_raw(path: str) -> dict:
    """Load a config JSON, resolving the optional ``base`` inheritance.

    ``base`` names a parent config relative to the referring file's
    directory; the child overrides the parent shallowly, except the
    ``method`` block, which merges per key.  Chains of configs are
    resolved recursively (child over parent over grandparent).
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    base_ref = raw.pop("base", None)
    if base_ref:
        base_path = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(path)), base_ref))
        parent = _load_raw(base_path)
        parent_method = parent.pop("method", {})
        raw = {**parent, **raw,
               "method": {**parent_method, **raw.get("method", {})}}
    return raw


def load_config(path: str) -> ExperimentConfig:
    """Load an experiment config JSON, filling MethodConfig defaults."""
    raw = _load_raw(os.path.abspath(path))
    method = MethodConfig(**raw.pop("method", {}))
    known = {f for f in ExperimentConfig.__dataclass_fields__}
    raw = {k: v for k, v in raw.items() if k in known}
    cfg = ExperimentConfig(**raw)
    cfg.method = method
    return cfg


# Frozen per-setting MMD bandwidths (paper Sec. "Complete Student Set
# Reconstruction"): HRSID / SSDD / COCO-RTDETR / COCO-DefDETR.
FROZEN_MMD_BANDWIDTH: Dict[str, float] = {
    "hrsid_rtdetr": 0.084,
    "ssdd_rtdetr": 0.079,
    "coco_rtdetr": 0.112,
    "coco_defdetr": 0.108,
}


def default_configs() -> Dict[str, ExperimentConfig]:
    """The four published settings."""
    return {
        "hrsid_rtdetr": ExperimentConfig(
            name="hrsid_rtdetr", setting="HRSID / RT-DETR", dataset="hrsid",
            model="rtdetr", head_type="softmax", num_classes=1,
            quant_profile="mixed", calib_images=512, mmd_bandwidth=0.084,
            checkpoint_pair="HR-02"),
        "ssdd_rtdetr": ExperimentConfig(
            name="ssdd_rtdetr", setting="SSDD / RT-DETR", dataset="ssdd",
            model="rtdetr", head_type="softmax", num_classes=1,
            quant_profile="mixed", calib_images=512, mmd_bandwidth=0.079,
            checkpoint_pair="SS-01"),
        "coco_rtdetr": ExperimentConfig(
            name="coco_rtdetr", setting="COCO / RT-DETR", dataset="coco",
            model="rtdetr", head_type="sigmoid", num_classes=80,
            quant_profile="mixed", calib_images=1024, mmd_bandwidth=0.112,
            checkpoint_pair="CR-01"),
        "coco_defdetr": ExperimentConfig(
            name="coco_defdetr", setting="COCO / Deformable DETR", dataset="coco",
            model="defdetr", head_type="sigmoid", num_classes=80,
            quant_profile="mixed", calib_images=1024, mmd_bandwidth=0.108,
            checkpoint_pair="CD-01"),
    }
