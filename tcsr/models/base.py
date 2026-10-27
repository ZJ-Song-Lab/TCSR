"""Detector adapter protocol shared by Stage-1 block reconstruction and
Stage-2 set reconstruction.

An adapter binds a concrete detector (full-precision student P, frozen
teacher T, or the quantized student Q_theta) to the two operations the
calibration pipeline needs:

- ``blocks()``: the ordered block partition used by Stage 1.  Blocks must be
  listed in execution order and every block must be executed exactly once
  per forward pass (shared or re-entered modules are not supported).
- ``predict(images)``: the full forward returning one ``Prediction`` per
  image (class logits, normalized cxcywh boxes, optional raw box
  parameters).

``capture_blocks`` (no-grad, records block inputs and outputs) and
``forward_with_blocks`` (gradient-enabled, records block outputs alongside
the predictions) are implemented once here on top of forward hooks; concrete
adapters only implement ``blocks`` and ``predict``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..losses import Prediction


def _first_tensor(x: Any) -> Optional[torch.Tensor]:
    """First tensor found in a (possibly nested) module output."""
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)):
        for v in x:
            t = _first_tensor(v)
            if t is not None:
                return t
    if isinstance(x, dict):
        for v in x.values():
            t = _first_tensor(v)
            if t is not None:
                return t
    return None


@dataclass
class BlockIO:
    """Inputs and first-tensor output of one block execution.

    ``args``/``kwargs`` are the exact call arguments of the block; both the
    cached inputs and the cached output are detached (no autograd graph).
    """

    args: tuple = ()
    kwargs: dict = None  # type: ignore[assignment]
    output: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


class DetectorAdapter:
    """Binds one detector instance to the block/set-loss protocol.

    ``model`` is used as-is (never copied): adapters over P/T run under
    ``torch.no_grad`` and adapters over Q_theta run with gradients on the
    quantizer state.
    """

    def __init__(self, model: nn.Module, head_type: str, num_classes: int):
        self.model = model
        self.head_type = head_type
        self.num_classes = num_classes

    # ------------------------------------------------------------------
    # To be implemented by concrete adapters.
    # ------------------------------------------------------------------
    def blocks(self) -> List[Tuple[str, nn.Module]]:
        """Ordered block partition (execution order, single execution)."""
        raise NotImplementedError

    def predict(self, images: torch.Tensor) -> List[Prediction]:
        """Full forward: one Prediction per image in the batch."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Default machinery.
    # ------------------------------------------------------------------
    def run_block(self, index: int, args: tuple, kwargs: dict) -> Any:
        """Re-execute a single block with the given call arguments."""
        return self.blocks()[index][1](*args, **(kwargs or {}))

    def _register_output_hooks(self, wanted: Optional[int], detach: bool,
                               outputs: List[Optional[torch.Tensor]]):
        """Forward hooks recording the first tensor of each block output."""
        handles = []

        def make_hook(slot: int):
            def hook(module, args, kwargs, output):
                t = _first_tensor(output)
                if t is not None and detach:
                    t = t.detach()
                outputs[slot] = t
            return hook

        blocks = self.blocks()
        for slot, (_, module) in enumerate(blocks):
            if wanted is not None and slot != wanted:
                continue
            handles.append(module.register_forward_hook(make_hook(slot),
                                                        with_kwargs=True))
        return handles

    @torch.no_grad()
    def capture_blocks(self, images: torch.Tensor,
                       index: Optional[int] = None) -> List[BlockIO]:
        """Run one full forward under no_grad, capturing block I/O.

        Returns one BlockIO per block (or only for ``index``); the recorded
        inputs and outputs are detached.  ``index`` only restricts which
        hooks are installed, saving memory for single-block captures.
        """
        n = len(self.blocks())
        ios: List[BlockIO] = [BlockIO() for _ in range(n)]
        handles = []
        blocks = self.blocks()

        def make_hook(slot: int):
            def hook(module, args, kwargs, output):
                ios[slot].args = tuple(args)
                ios[slot].kwargs = dict(kwargs) if kwargs else {}
                t = _first_tensor(output)
                ios[slot].output = None if t is None else t.detach()
            return hook

        for slot, (_, module) in enumerate(blocks):
            if index is not None and slot != index:
                continue
            handles.append(module.register_forward_hook(make_hook(slot),
                                                        with_kwargs=True))
        self.predict(images)
        for h in handles:
            h.remove()
        if index is not None:
            return [ios[index]]
        return ios

    def forward_with_blocks(
            self, images: torch.Tensor
    ) -> Tuple[List[Prediction], List[torch.Tensor]]:
        """Gradient-enabled forward returning predictions and block outputs.

        The block outputs keep their autograd graph; used by Stage 2 to
        retain the block-reconstruction term while the set losses flow
        through the same forward.
        """
        n = len(self.blocks())
        outputs: List[Optional[torch.Tensor]] = [None] * n
        handles = self._register_output_hooks(None, detach=False,
                                              outputs=outputs)
        try:
            preds = self.predict(images)
        finally:
            for h in handles:
                h.remove()
        for i, t in enumerate(outputs):
            if t is None:
                raise RuntimeError(
                    f"block {i} ({self.blocks()[i][0]}) produced no tensor "
                    "output; the adapter protocol requires tensor outputs")
        return preds, outputs  # type: ignore[return-value]

    def block_output_tensors(self, images: torch.Tensor
                             ) -> List[torch.Tensor]:
        """Block outputs only (no_grad), used as Stage-2 targets."""
        return [io.output for io in self.capture_blocks(images)]


@torch.no_grad()
def validate_block_execution(adapter: DetectorAdapter,
                             images: torch.Tensor) -> None:
    """Assert every block executes exactly once during one forward pass.

    The block-reconstruction protocol relies on this invariant (shared or
    re-entered modules are unsupported); concrete checkpoint layouts may
    drift across upstream versions, so adapters can be checked at runtime
    with a representative input batch.
    """
    blocks = adapter.blocks()
    counts = [0] * len(blocks)
    handles = []

    def make_hook(slot: int):
        def hook(module, args, kwargs, output):
            counts[slot] += 1
        return hook

    for slot, (_, module) in enumerate(blocks):
        handles.append(module.register_forward_hook(make_hook(slot),
                                                    with_kwargs=True))
    try:
        adapter.predict(images)
    finally:
        for h in handles:
            h.remove()
    bad = [(blocks[i][0], c) for i, c in enumerate(counts) if c != 1]
    if bad:
        raise RuntimeError(
            "block execution contract violated (name, forward calls): "
            f"{bad}; the block partition must list each module exactly once "
            "per forward pass")


def check_logits_channels(head_type: str, num_classes: int, channels: int,
                          model_name: str = "model") -> None:
    """Validate the logit-channel count against the head-type contract.

    The head convention used across the paper's settings: a softmax head
    emits ``num_classes + 1`` channels with the background node last; a
    sigmoid head emits exactly ``num_classes`` channels.
    """
    if head_type not in ("softmax", "sigmoid"):
        raise ValueError(
            f"unknown head_type {head_type!r}; expected 'softmax' or "
            "'sigmoid'")
    expected = num_classes + 1 if head_type == "softmax" else num_classes
    if int(channels) != expected:
        raise ValueError(
            f"{model_name} produced {int(channels)} logit channels, expected "
            f"{expected} for head_type={head_type!r} with "
            f"num_classes={num_classes}")
