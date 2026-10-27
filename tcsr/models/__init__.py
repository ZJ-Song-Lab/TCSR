"""Model adapters: the block/set-loss protocol, a CPU self-test model, and
the paper's detectors (RT-DETR via HuggingFace transformers, Deformable DETR
via the official repository).

``transformers`` and the official Deformable-DETR repository are optional
runtime dependencies -- both adapter modules import lazily, so importing
this package never requires them.
"""

from .base import (BlockIO, DetectorAdapter, _first_tensor,
                   check_logits_channels, validate_block_execution)
from .defdetr import (DefDetrAdapter, defdetr_adapter_factory, defdetr_args,
                      load_defdetr)
from .rtdetr import (RTDetrAdapter, load_rtdetr, make_tiny_rtdetr,
                     rtdetr_adapter_factory, rtdetr_blocks)
from .toy import (ToyAdapter, ToyDetector, ToyHeadBlock, ToyProjection,
                  make_toy_detector, toy_adapter_factory)

# Experiment-config ``model`` field -> adapter factory constructor.
ADAPTER_FACTORIES = {
    "toy": toy_adapter_factory,
    "rtdetr": rtdetr_adapter_factory,
    "defdetr": defdetr_adapter_factory,
}


def get_adapter_factory(model: str, head_type: str = None,
                        num_classes: int = None):
    """``adapter_factory`` callable for the experiment config's model field."""
    try:
        make = ADAPTER_FACTORIES[model]
    except KeyError:
        raise ValueError(
            f"unknown model {model!r}; expected one of "
            f"{sorted(ADAPTER_FACTORIES)}") from None
    return make(head_type=head_type, num_classes=num_classes)


__all__ = [
    "BlockIO", "DetectorAdapter", "_first_tensor", "check_logits_channels",
    "validate_block_execution",
    "ToyAdapter", "ToyDetector", "ToyHeadBlock", "ToyProjection",
    "make_toy_detector", "toy_adapter_factory",
    "RTDetrAdapter", "load_rtdetr", "make_tiny_rtdetr",
    "rtdetr_adapter_factory", "rtdetr_blocks",
    "DefDetrAdapter", "defdetr_adapter_factory", "defdetr_args",
    "load_defdetr",
    "ADAPTER_FACTORIES", "get_adapter_factory",
]
