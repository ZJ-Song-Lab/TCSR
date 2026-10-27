"""Deformable DETR adapter over the official ``fundamentalvision/Deformable-DETR``
codebase (paper setting: COCO / Deformable DETR, R101 teacher / R50 student).

The official repository ships a custom CUDA operator (multi-scale deformable
attention), so it is not a pip dependency: point ``load_defdetr`` at a local
clone via ``repo_path`` (or put the clone on ``PYTHONPATH``).  All imports of
the official package are lazy so that this module can be imported on machines
without the CUDA extension.

Block partition (BRECQ-style, execution order, every block executed exactly
once per forward):

====================  ====================================================
``backbone``           ``Joiner`` (ResNet ``IntermediateLayerGetter`` +
                       sine position embedding), called once.
``input_proj.{i}``     one ``nn.Conv2d`` per feature level (the official
                       ``input_proj`` is an ``nn.ModuleList`` that is never
                       called as a whole, hence per-child blocks).
``transformer.encoder.{i}``  one deformable encoder layer per block.
``transformer.decoder.{i}``  one deformable decoder layer per block.
``class_head``         ``Linear(d_model, num_classes)`` -- sigmoid head,
                       no background node (paper: COCO uses sigmoid heads).
``bbox_head``          ``MLP(d_model, d_model, 4, 3)``.
====================  ====================================================

Operator names under ``tcsr.quant.policy``: ``transformer.encoder.*`` /
``transformer.decoder.*`` carry the encoder/decoder fragments (mixed profile:
decoder W4A8, others W8A8), ``class_head`` / ``bbox_head`` carry the ``head``
fragment and stay full precision, and ``query`` / ``reference`` modules
(embeddings, reference-point projection) are skipped by the policy.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..losses import Prediction
from .base import DetectorAdapter, check_logits_channels

# Defaults of the official r50 / r101 Deformable DETR checkpoints (four
# feature levels, 300 queries, no box refinement, no two-stage).
DEFAULT_ARGS = dict(
    num_classes=80, num_queries=300, num_feature_levels=4,
    hidden_dim=256, backbone="resnet50", dilation=False, position_embedding="sine",
    position_embedding_scale=2 * 3.141592653589793,
    enc_layers=6, dec_layers=6, dim_feedforward=1024, dropout=0.0, nheads=8,
    dec_n_points=4, enc_n_points=4,
    aux_loss=True, with_box_refine=False, two_stage=False, masks=False,
)


def _ensure_repo_path(repo_path: Optional[str]) -> None:
    if repo_path is None:
        return
    p = str(Path(repo_path).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def defdetr_args(**overrides) -> Namespace:
    """Argparse-style namespace for the official ``build_model``.

    Defaults mirror the released single-scale Deformable DETR checkpoints;
    any keyword argument overrides a default (e.g. ``num_queries=``).
    """
    unknown = set(overrides) - set(DEFAULT_ARGS)
    if unknown:
        raise TypeError(f"unknown Deformable DETR args: {sorted(unknown)}")
    return Namespace(**{**DEFAULT_ARGS, **overrides})


def load_defdetr(checkpoint: str, repo_path: Optional[str] = None,
                 num_classes: int = 80, device: str = "cpu",
                 **overrides) -> nn.Module:
    """Build the official Deformable DETR and load a checkpoint.

    ``checkpoint`` is a ``.pth`` file in the official format
    (``{"model": state_dict}``; a bare state dict is also accepted).
    """
    _ensure_repo_path(repo_path)
    try:
        # Official repository modules: ``models`` (detector) and ``util``.
        from models.deformable_detr import build_model  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - needs the official repo
        raise ImportError(
            "the official Deformable-DETR repository could not be imported; "
            "pass repo_path=<path to a clone of "
            "github.com/fundamentalvision/Deformable-DETR> (it provides the "
            "`models` package and its multi-scale deformable-attention CUDA "
            "op)") from exc

    args = defdetr_args(num_classes=num_classes, **overrides)
    model = build_model(args)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model.to(device)


class DefDetrAdapter(DetectorAdapter):
    """Adapter for the official ``DeformableDETR`` module.

    ``head_type``/``num_classes`` come from the experiment config; defaults
    follow the checkpoint (paper: COCO uses sigmoid heads, so
    ``num_classes = class_head.out_features``; a softmax head would carry
    one extra background node).
    """

    def __init__(self, model: nn.Module,
                 head_type: Optional[str] = None,
                 num_classes: Optional[int] = None):
        head_type = head_type or "sigmoid"
        if num_classes is None:
            out = int(model.class_head.out_features)
            num_classes = out if head_type == "sigmoid" else out - 1
        super().__init__(model, head_type, num_classes)

    def blocks(self) -> List[Tuple[str, nn.Module]]:
        m = self.model
        parts: List[Tuple[str, nn.Module]] = [("backbone", m.backbone)]
        for i, proj in enumerate(m.input_proj):
            parts.append((f"input_proj.{i}", proj))
        for i, layer in enumerate(m.transformer.encoder.layers):
            parts.append((f"transformer.encoder.{i}", layer))
        for i, layer in enumerate(m.transformer.decoder.layers):
            parts.append((f"transformer.decoder.{i}", layer))
        parts.append(("class_head", m.class_head))
        parts.append(("bbox_head", m.bbox_head))
        return parts

    def predict(self, images: torch.Tensor) -> List[Prediction]:
        # The official forward converts a batched tensor into a NestedTensor
        # (equal-size 640x640 batches need no padding, mask stays all-valid).
        outputs = self.model(images)
        logits = outputs["pred_logits"]           # (B, Q, num_classes)
        boxes = outputs["pred_boxes"]             # (B, Q, 4) cxcywh in [0, 1]
        check_logits_channels(self.head_type, self.num_classes,
                              int(logits.shape[-1]), "DeformableDETR")
        return [Prediction(logits=logits[b], boxes=boxes[b])
                for b in range(logits.shape[0])]


def defdetr_adapter_factory(head_type: Optional[str] = None,
                            num_classes: Optional[int] = None):
    """``adapter_factory`` callable for :func:`tcsr.calibration`."""
    return lambda model: DefDetrAdapter(model, head_type, num_classes)
