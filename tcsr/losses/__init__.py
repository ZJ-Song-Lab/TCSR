"""Losses: bounded class-box discrepancy, residual MMD, block reconstruction."""

from .discrepancy import (
    Prediction,
    bernoulli_js,
    class_probs,
    d_cls_sigmoid_rows,
    d_cls_sigmoid,
    d_cls_softmax_rows,
    d_cls_softmax,
    d_l1,
    d_giou,
    giou,
    discrepancy_components,
    discrepancy,
    pairwise_discrepancy,
    aligned_discrepancy,
    anchor_confidence,
)
from .residual_mmd import residual_mmd, median_bandwidth, ResidualMMD
from .block_loss import normalized_block_loss, block_losses

__all__ = [
    "Prediction", "bernoulli_js", "class_probs",
    "d_cls_sigmoid_rows", "d_cls_sigmoid", "d_cls_softmax_rows", "d_cls_softmax",
    "d_l1", "d_giou", "giou",
    "discrepancy_components", "discrepancy", "pairwise_discrepancy",
    "aligned_discrepancy", "anchor_confidence",
    "residual_mmd", "median_bandwidth", "ResidualMMD",
    "normalized_block_loss", "block_losses",
]
