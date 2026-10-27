"""RT-DETR adapter over HuggingFace ``transformers``
(``RTDetrForObjectDetection``; paper settings: HRSID, SSDD and COCO).

``transformers`` is an optional dependency (see ``requirements.txt``): it is
imported lazily so the rest of the package works without it.

Two HuggingFace layouts are supported, discovered from the module tree (the
layout drifted across major ``transformers`` versions):

**Current layout (transformers >= 5.x)** -- ``RTDetrModel`` owns
``backbone``, ``encoder_input_proj`` (one ``Conv2d+BN`` entry per backbone
output, each called once), the hybrid ``encoder`` (AIFI + FPN/PAN, called as
one block), ``decoder_input_proj`` (one entry per feature level, each called
once), the query-selection modules ``enc_output``/``enc_score_head``/
``enc_bbox_head`` and the ``decoder``; the heads live in
``decoder.class_embed``/``decoder.bbox_embed`` (ModuleLists with one entry
per decoder layer, executed interleaved with the layers).  Block partition,
in execution order: ``backbone``, ``encoder_input_proj.{i}``, ``encoder``,
``decoder_input_proj.{i}``, ``enc_output``, ``enc_score_head``,
``enc_bbox_head``, then per decoder layer ``decoder.layers.{i}``,
``decoder.bbox_embed.{i}``, ``decoder.class_embed.{i}``.  Modules breaking
the exactly-once contract are excluded from the partition:
``decoder.query_pos_head`` (re-entered by every decoder layer) and
``denoising_class_embed`` (training only, never executed in eval).  Blocks
without quantized operators (the heads) are automatically skipped by Stage 1.

**Legacy layout (transformers 4.x)** -- ``RTDetrModel`` owns ``backbone``,
``input_proj``, ``encoder``, ``decoder``; the shared ``class_head``/
``bbox_head`` sit on the wrapper and are re-entered once per decoder layer,
so the partition stops at the decoder layers (they stay full precision
anyway, the policy skips ``head`` fragments).

Every block executes exactly once per forward pass (the adapter protocol);
:func:`tcsr.models.base.validate_block_execution` checks this at runtime
for the concrete checkpoint in use.

Head conventions (paper): HRSID/SSDD checkpoints keep an explicit background
node (``head_type="softmax"``, background = last logit channel, matching the
RT-DETR class-head layout); the COCO setting uses sigmoid heads (the
transformers 5.x RT-DETR head is natively sigmoid: ``logits`` carry exactly
``num_labels`` channels, no background node).  ``logits`` returned by
:meth:`RTDetrAdapter.predict` always keep the full head output (background
channel included when present); ``boxes`` are normalized cxcywh.

Operator names under ``tcsr.quant.policy``: paths through ``decoder``
(``model.decoder_input_proj.*`` and ``model.decoder.layers.*``) get W4A8
under the mixed profile and everything else W8A8;
``class_embed``/``bbox_embed`` carry the ``embed`` fragment and
``enc_score_head``/``enc_bbox_head`` the ``head`` fragment, so every head
stays full precision; the backbone embedding stem is skipped as well.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..losses import Prediction
from .base import DetectorAdapter, check_logits_channels


def _core_model(net: nn.Module) -> nn.Module:
    """The ``RTDetrModel`` carrying backbone/projections/encoder/decoder."""
    inner = getattr(net, "model", None)
    if isinstance(inner, nn.Module) and inner is not net:
        return inner
    return net


def _children(net: nn.Module) -> Dict[str, nn.Module]:
    return dict(net.named_children())


def _dedupe_check(parts: List[Tuple[str, nn.Module]]
                  ) -> List[Tuple[str, nn.Module]]:
    ids = [id(m) for _, m in parts]
    if len(set(ids)) != len(ids):
        raise AttributeError("RT-DETR block discovery found duplicated modules")
    return parts


def _expand_module_list(name: str, module: nn.ModuleList
                        ) -> List[Tuple[str, nn.Module]]:
    """One block per entry of a ModuleList that is never called as a whole."""
    if len(module) == 0:
        raise AttributeError(f"RT-DETR stage {name!r} is empty")
    return [(f"{name}.{i}", child) for i, child in enumerate(module)]


def _rtdetr_blocks_current(core: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Block partition for the transformers 5.x RT-DETR layout."""
    children = _children(core)
    required = ("backbone", "encoder_input_proj", "encoder",
                "decoder_input_proj", "enc_output", "enc_score_head",
                "enc_bbox_head", "decoder")
    missing = [n for n in required if n not in children]
    if missing:
        raise AttributeError(
            "could not locate RT-DETR stage(s) "
            f"{missing} on {type(core).__name__}; the HuggingFace layout "
            "may have changed")
    decoder = children["decoder"]
    layers = getattr(decoder, "layers", None)
    class_embed = getattr(decoder, "class_embed", None)
    bbox_embed = getattr(decoder, "bbox_embed", None)
    if not isinstance(layers, nn.ModuleList) or len(layers) == 0:
        raise AttributeError(
            "unexpected RT-DETR decoder layout: no non-empty .layers "
            "ModuleList (the HuggingFace layout may have changed)")
    if (not isinstance(class_embed, nn.ModuleList)
            or not isinstance(bbox_embed, nn.ModuleList)
            or len(class_embed) != len(layers)
            or len(bbox_embed) != len(layers)):
        raise AttributeError(
            "unexpected RT-DETR decoder head layout: class_embed/bbox_embed "
            "must be ModuleLists with one entry per decoder layer")

    # Execution order of RTDetrModel.forward (eval path): backbone ->
    # encoder_input_proj (per backbone output) -> encoder ->
    # decoder_input_proj (per feature level) -> query selection
    # (enc_output -> enc_score_head -> enc_bbox_head) -> decoder, where each
    # decoder layer runs query_pos_head (re-entered, excluded) -> layer ->
    # bbox_embed[i] -> class_embed[i].
    parts: List[Tuple[str, nn.Module]] = [("backbone", children["backbone"])]
    parts.extend(_expand_module_list("encoder_input_proj",
                                     children["encoder_input_proj"]))
    parts.append(("encoder", children["encoder"]))
    parts.extend(_expand_module_list("decoder_input_proj",
                                     children["decoder_input_proj"]))
    parts.append(("enc_output", children["enc_output"]))
    parts.append(("enc_score_head", children["enc_score_head"]))
    parts.append(("enc_bbox_head", children["enc_bbox_head"]))
    for i, layer in enumerate(layers):
        parts.append((f"decoder.layers.{i}", layer))
        parts.append((f"decoder.bbox_embed.{i}", bbox_embed[i]))
        parts.append((f"decoder.class_embed.{i}", class_embed[i]))
    return _dedupe_check(parts)


def _rtdetr_blocks_legacy(core: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Block partition for the transformers 4.x RT-DETR layout."""
    children = _children(core)
    required = ("backbone", "input_proj", "encoder", "decoder")
    missing = [n for n in required if n not in children]
    if missing:
        raise AttributeError(
            "could not locate RT-DETR stage(s) "
            f"{missing} on {type(core).__name__}; the HuggingFace layout "
            "may have changed")
    parts: List[Tuple[str, nn.Module]] = [("backbone", children["backbone"])]
    parts.extend(_expand_module_list("input_proj", children["input_proj"]))
    encoder = children["encoder"]
    enc_layers = getattr(encoder, "layers", None)
    if isinstance(enc_layers, nn.ModuleList) and len(enc_layers) > 0:
        parts.extend((f"encoder.layers.{i}", layer)
                     for i, layer in enumerate(enc_layers))
    else:
        parts.append(("encoder", encoder))
    dec_layers = getattr(children["decoder"], "layers", None)
    if not isinstance(dec_layers, nn.ModuleList) or len(dec_layers) == 0:
        raise AttributeError(
            "unexpected RT-DETR decoder layout: no non-empty .layers "
            "ModuleList (the HuggingFace layout may have changed)")
    parts.extend((f"decoder.layers.{i}", layer)
                 for i, layer in enumerate(dec_layers))
    # The wrapper-level class_head/bbox_head are re-entered once per decoder
    # layer in this layout, so they cannot join the exactly-once partition;
    # they stay full precision anyway (the policy skips "head" fragments).
    return _dedupe_check(parts)


def rtdetr_blocks(net: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Ordered block partition of an RT-DETR model (execution order)."""
    core = _core_model(net)
    children = _children(core)
    if "encoder_input_proj" in children and "decoder_input_proj" in children:
        return _rtdetr_blocks_current(core)
    if "input_proj" in children:
        return _rtdetr_blocks_legacy(core)
    raise AttributeError(
        f"could not recognize an RT-DETR layout on {type(net).__name__}; "
        "expected a transformers RTDetrForObjectDetection (current or "
        "legacy module layout) -- the HuggingFace layout may have changed")


class RTDetrAdapter(DetectorAdapter):
    """Adapter for ``transformers.RTDetrForObjectDetection``.

    ``head_type``/``num_classes`` come from the experiment config (the paper
    freezes them per setting).  Defaults follow the checkpoint:
    ``head_type="sigmoid"`` (the transformers RT-DETR head is natively
    sigmoid) and ``num_classes`` derived from ``config.num_labels`` (a
    softmax head carries one extra background node, so
    ``num_classes = num_labels - 1``).
    """

    def __init__(self, model: nn.Module,
                 head_type: Optional[str] = None,
                 num_classes: Optional[int] = None):
        config = getattr(model, "config", None)
        if head_type is None:
            head_type = "sigmoid"
        if num_classes is None:
            if config is None or not hasattr(config, "num_labels"):
                raise ValueError(
                    "num_classes must be given for models without a "
                    "num_labels config")
            num_labels = int(config.num_labels)
            num_classes = (num_labels if head_type == "sigmoid"
                           else num_labels - 1)
        super().__init__(model, head_type, num_classes)

    def blocks(self) -> List[Tuple[str, nn.Module]]:
        return rtdetr_blocks(self.model)

    def predict(self, images: torch.Tensor) -> List[Prediction]:
        outputs = self.model(pixel_values=images)
        logits = outputs.logits        # (B, Q, C) or (B, Q, C+1) softmax
        boxes = outputs.pred_boxes     # (B, Q, 4) normalized cxcywh
        check_logits_channels(self.head_type, self.num_classes,
                              int(logits.shape[-1]),
                              "RTDetrForObjectDetection")
        return [Prediction(logits=logits[b], boxes=boxes[b])
                for b in range(logits.shape[0])]


def rtdetr_adapter_factory(head_type: Optional[str] = None,
                           num_classes: Optional[int] = None):
    """``adapter_factory`` callable for :func:`tcsr.calibration`."""
    return lambda model: RTDetrAdapter(model, head_type, num_classes)


def load_rtdetr(checkpoint: str, num_classes: Optional[int] = None,
                device: str = "cpu", head_type: str = "sigmoid",
                **from_pretrained_kwargs) -> nn.Module:
    """Load an RT-DETR checkpoint via HuggingFace ``from_pretrained``.

    ``checkpoint`` is a HF hub id or a local directory containing the
    distilled teacher/student weights (see ``manifests/checkpoints.json``).
    ``num_classes`` refers to foreground classes: a softmax head checkpoint
    stores ``num_labels = num_classes + 1`` (background last), a sigmoid
    head checkpoint stores ``num_labels = num_classes``.
    """
    try:
        from transformers import RTDetrForObjectDetection  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "transformers is required for RT-DETR checkpoints: "
            "pip install transformers") from exc
    model = RTDetrForObjectDetection.from_pretrained(checkpoint,
                                                     **from_pretrained_kwargs)
    if num_classes is not None:
        expected_labels = (num_classes if head_type == "sigmoid"
                           else num_classes + 1)
        if int(model.config.num_labels) != expected_labels:
            raise ValueError(
                f"checkpoint {checkpoint!r} has num_labels="
                f"{model.config.num_labels}, expected {expected_labels} for "
                f"head_type={head_type!r} with num_classes={num_classes}")
    model.eval()
    return model.to(device)


def make_tiny_rtdetr(num_labels: int = 1, d_model: int = 32,
                     input_size: int = 64, seed: int = 0) -> nn.Module:
    """Tiny randomly-initialized RT-DETR for CPU self-tests.

    Not a trained detector: it only mirrors the module layout so the
    adapter protocol, quantization policy and optimization paths can be
    exercised end to end without checkpoint downloads.
    """
    try:
        from transformers import (ResNetConfig, RTDetrConfig,  # noqa: PLC0415
                                  RTDetrForObjectDetection)
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "transformers is required for the tiny RT-DETR test model: "
            "pip install transformers") from exc
    backbone = ResNetConfig(
        num_channels=3,
        embeddings_size=8,
        hidden_sizes=[16, 32, 64],
        depths=[1, 1, 1],
        out_features=["stage1", "stage2", "stage3"],
    )
    config = RTDetrConfig(
        backbone_config=backbone,
        num_labels=num_labels,
        d_model=d_model,
        encoder_layers=1,
        decoder_layers=1,
        num_queries=8,
    )
    torch.manual_seed(seed)
    model = RTDetrForObjectDetection(config)
    model.eval()
    model._tcsr_input_size = input_size  # used by tests only
    return model
