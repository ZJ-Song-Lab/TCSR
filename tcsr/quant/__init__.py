"""Uniform dequantizer (paper Eq. 2), observers, fake-quant wrappers and
operator-level bit-width policies."""

from .observer import MinMaxObserver, MSEObserver
from .quantizer import UniformQuantizer, AdaRoundQuantizer, round_to_nearest
from .fake_quant import (
    QLinear,
    QConv2d,
    quantize_model,
    calibrate_quantizers,
    trainable_quantizer_parameters,
    rounding_regularizer,
    harden_all_quantizers,
    iter_qmodules,
)
from .policy import OpBits, get_bit_assignment, PROFILES

__all__ = [
    "MinMaxObserver", "MSEObserver",
    "UniformQuantizer", "AdaRoundQuantizer", "round_to_nearest",
    "QLinear", "QConv2d", "quantize_model", "calibrate_quantizers",
    "trainable_quantizer_parameters", "rounding_regularizer",
    "harden_all_quantizers", "iter_qmodules",
    "OpBits", "get_bit_assignment", "PROFILES",
]
