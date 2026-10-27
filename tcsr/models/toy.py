"""A small self-contained detector used by the repository's self-tests.

The toy detector mirrors the structural contract of the distilled DETR
 detectors used in the paper (projection -> encoder blocks -> detection
head) while staying small enough for CPU-only smoke tests:

- ``ToyProjection``: per-pixel Conv stem producing ``H*W`` query tokens.
- ``ToyEncoderLayer``: residual MLP token mixer (two Linear layers).
- ``ToyHeadBlock``: parallel class / box projections concatenated into a
  single output tensor (the only block output the adapter protocol needs).

Under ``tcsr.quant.policy`` the operator names ``proj.conv`` and
``enc*.fc*`` are quantized while ``head.cls`` / ``head.box`` contain the
"head" fragment and stay full precision, matching the paper's rule that
detection heads are excluded from the quantized inventory.

``head_type`` follows the paper: ``softmax`` uses an explicit background
node (last logit), ``sigmoid`` uses independent per-class sigmoids; boxes
are normalized cxcywh obtained by a sigmoid on the raw box parameters.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses import Prediction
from .base import DetectorAdapter


class ToyProjection(nn.Module):
    """Conv stem: (B, C_in, H, W) -> (B, H*W, D) query tokens."""

    def __init__(self, in_ch: int = 3, dim: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)                       # (B, D, H, W)
        b, d, h, w = z.shape
        return z.reshape(b, d, h * w).transpose(1, 2)   # (B, N, D)


class ToyEncoderLayer(nn.Module):
    """Residual MLP token mixer: x + fc2(gelu(fc1(gelu(x))))."""

    def __init__(self, dim: int = 32, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.gelu(self.fc1(F.gelu(x))))


class ToyHeadBlock(nn.Module):
    """Detection head: cat([class logits, raw box parameters], -1)."""

    def __init__(self, dim: int, num_classes: int, head_type: str):
        super().__init__()
        if head_type not in ("softmax", "sigmoid"):
            raise ValueError(f"unknown head_type {head_type!r}")
        self.head_type = head_type
        self.num_classes = num_classes
        cls_out = num_classes + 1 if head_type == "softmax" else num_classes
        self.cls = nn.Linear(dim, cls_out)
        self.box = nn.Linear(dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.cls(x), self.box(x)], dim=-1)


class ToyDetector(nn.Module):
    """proj -> enc1 -> enc2 -> head, single-tensor output."""

    def __init__(self, head_type: str = "softmax", num_classes: int = 1,
                 in_ch: int = 3, size: int = 8, dim: int = 32,
                 hidden: int = 64):
        super().__init__()
        self.head_type = head_type
        self.num_classes = num_classes
        self.size = size
        self.num_queries = size * size
        self.proj = ToyProjection(in_ch, dim)
        self.enc1 = ToyEncoderLayer(dim, hidden)
        self.enc2 = ToyEncoderLayer(dim, hidden)
        self.head = ToyHeadBlock(dim, num_classes, head_type)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B, C_in, H, W) -> (B, N, C_logits + 4)."""
        z = self.proj(images)
        z = self.enc1(z)
        z = self.enc2(z)
        return self.head(z)

    def split(self, out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split the head output into (logits, raw box parameters)."""
        n_cls = self.head.cls.out_features
        return out[..., :n_cls], out[..., n_cls:]

    def predict(self, images: torch.Tensor) -> List[Prediction]:
        """Per-image Prediction sets (boxes = sigmoid(raw))."""
        out = self.forward(images)
        logits, raw = self.split(out)
        boxes = torch.sigmoid(raw)
        return [Prediction(logits=logits[b], boxes=boxes[b], raw_boxes=raw[b])
                for b in range(logits.shape[0])]


def make_toy_detector(seed: int = 0, head_type: str = "softmax",
                      num_classes: int = 1, in_ch: int = 3, size: int = 8,
                      dim: int = 32, hidden: int = 64) -> ToyDetector:
    """Deterministic toy detector (fixed-seed initialization)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ToyDetector(head_type=head_type, num_classes=num_classes,
                           in_ch=in_ch, size=size, dim=dim, hidden=hidden)


class ToyAdapter(DetectorAdapter):
    """Adapter for :class:`ToyDetector` (head_type taken from the model)."""

    def __init__(self, model: ToyDetector,
                 head_type: Optional[str] = None,
                 num_classes: Optional[int] = None):
        super().__init__(model,
                         head_type or model.head_type,
                         num_classes if num_classes is not None
                         else model.num_classes)

    def blocks(self) -> List[Tuple[str, nn.Module]]:
        m = self.model
        return [("proj", m.proj), ("enc1", m.enc1),
                ("enc2", m.enc2), ("head", m.head)]

    def predict(self, images: torch.Tensor) -> List[Prediction]:
        return self.model.predict(images)


def toy_adapter_factory(head_type: Optional[str] = None,
                        num_classes: Optional[int] = None):
    """``adapter_factory`` callable for :func:`tcsr.calibration`."""
    return lambda model: ToyAdapter(model, head_type, num_classes)
