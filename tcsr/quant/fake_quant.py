"""Fake-quant module wrappers and whole-model quantization.

``quantize_model`` copies a frozen full-precision student P into a quantized
student Q_theta whose supported Linear/Conv operators route through
``AdaRoundQuantizer`` (weights, per output channel) and ``UniformQuantizer``
(activations, per tensor). All detector weights are frozen; only the
quantizer state theta (scales, clipping parameters, soft-rounding variables)
is trainable.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, Iterator, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .observer import MinMaxObserver, MSEObserver
from .quantizer import AdaRoundQuantizer, UniformQuantizer


def _per_channel_shape(weight: torch.Tensor) -> tuple:
    return (weight.shape[0],) + (1,) * (weight.dim() - 1)


class _QModule(nn.Module):
    """Base wrapper: frozen operator + trainable quantizer state."""

    def __init__(self, src: nn.Module, w_bits: int, a_bits: int,
                 weight_symmetric: bool = False):
        super().__init__()
        self.weight = src.weight.detach().clone()
        self.weight.requires_grad_(False)
        self.bias = None if src.bias is None else src.bias.detach().clone()
        self.w_quant = AdaRoundQuantizer(
            bits=w_bits, symmetric=weight_symmetric,
            shape=_per_channel_shape(self.weight),
            weight_shape=tuple(self.weight.shape))
        self.a_quant = UniformQuantizer(bits=a_bits, symmetric=False,
                                        learnable=True)
        self.src_class = type(src)

    def quantized_weight(self) -> torch.Tensor:
        return self.w_quant(self.weight)

    def rounding_regularizer(self) -> torch.Tensor:
        return self.w_quant.rounding_regularizer(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class QLinear(_QModule):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(self.a_quant(x), self.quantized_weight(), self.bias)


class QConv2d(_QModule):
    def __init__(self, src: nn.Conv2d, w_bits: int, a_bits: int,
                 weight_symmetric: bool = False):
        super().__init__(src, w_bits, a_bits, weight_symmetric)
        self.stride = src.stride
        self.padding = src.padding
        self.dilation = src.dilation
        self.groups = src.groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(self.a_quant(x), self.quantized_weight(), self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def quantize_model(model: nn.Module, profile: str,
                   bit_assignment: Optional[Callable[[str, str], object]] = None,
                   weight_symmetric: bool = False) -> nn.Module:
    """Return a deep-copied model whose supported operators are quantized.

    ``bit_assignment(name, profile)`` maps an operator name to an ``OpBits``
    or None; the default uses :func:`tcsr.quant.policy.get_bit_assignment`.
    The returned model is in float precision until
    :func:`calibrate_quantizers` initializes theta.
    """
    from .policy import get_bit_assignment
    if bit_assignment is None:
        bit_assignment = get_bit_assignment

    qmodel = copy.deepcopy(model)
    for p in qmodel.parameters():
        p.requires_grad_(False)
    _replace_recursively(qmodel, "", profile, bit_assignment, weight_symmetric)
    return qmodel


def _replace_recursively(module: nn.Module, prefix: str, profile: str,
                         bit_assignment, weight_symmetric: bool) -> None:
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        bits = bit_assignment(full, profile)
        if bits is not None and isinstance(child, nn.Linear):
            setattr(module, name, QLinear(child, bits.w_bits, bits.a_bits,
                                          weight_symmetric))
        elif bits is not None and isinstance(child, nn.Conv2d):
            setattr(module, name, QConv2d(child, bits.w_bits, bits.a_bits,
                                          weight_symmetric))
        else:
            _replace_recursively(child, full, profile, bit_assignment,
                                 weight_symmetric)


def iter_qmodules(model: nn.Module) -> Iterator[_QModule]:
    for m in model.modules():
        if isinstance(m, _QModule):
            yield m


@torch.no_grad()
def calibrate_quantizers(qmodel: nn.Module, loader, device: str = "cpu",
                         weight_observer: str = "minmax",
                         max_batches: int = 8) -> nn.Module:
    """Initialize all quantizer ranges.

    Weight ranges come from per-output-channel statistics of the frozen
    weights; activation ranges from a few calibration batches collected with
    forward-pre hooks. ``loader`` yields either dicts with an ``images``
    tensor or plain (images, ...) tuples.
    """
    # 1) Weight ranges (no data pass required).
    for m in iter_qmodules(qmodel):
        dims = tuple(range(1, m.weight.dim()))
        if dims:
            lo, hi = m.weight.amin(dim=dims), m.weight.amax(dim=dims)
        else:
            lo, hi = m.weight.min(), m.weight.max()
        if weight_observer == "mse":
            w_obs = MSEObserver(per_channel=True)
            w_obs.min_val, w_obs.max_val = lo, hi
            w_obs.cache = [m.weight.detach().float()
                           .reshape(m.weight.shape[0], -1).cpu()]
            lo, hi = w_obs.range()
        m.w_quant.configure(lo, hi)

    # 2) Activation ranges via hooks.
    qmodules = list(iter_qmodules(qmodel))
    act_obs: List[MinMaxObserver] = [MinMaxObserver(per_channel=False)
                                     for _ in qmodules]
    handles = []

    def make_hook(obs: MinMaxObserver):
        def hook(module, inputs):
            x = inputs[0]
            if isinstance(x, torch.Tensor) and x.dtype.is_floating_point:
                obs.update(x.detach())
        return hook

    for m, o in zip(qmodules, act_obs):
        handles.append(m.register_forward_pre_hook(make_hook(o)))

    was_training = qmodel.training
    qmodel.eval()
    qmodel.to(device)
    n = 0
    with torch.no_grad():
        for batch in loader:
            if n >= max_batches:
                break
            if isinstance(batch, dict):
                images = batch["images"]
            else:
                images = batch[0]
            if isinstance(images, (list, tuple)):
                images = images[0]
            if isinstance(images, torch.Tensor):
                qmodel(images.to(device))
                n += 1
    for h in handles:
        h.remove()
    for m, o in zip(qmodules, act_obs):
        if o.min_val is not None:
            m.a_quant.configure(o.min_val, o.max_val)
    if was_training:
        qmodel.train()
    return qmodel


def trainable_quantizer_parameters(model: nn.Module) -> List[nn.Parameter]:
    """theta: every trainable parameter belongs to quantizer modules."""
    params: List[nn.Parameter] = []
    for m in iter_qmodules(model):
        for p in m.parameters():
            if p.requires_grad:
                params.append(p)
    return params


def rounding_regularizer(model: nn.Module) -> torch.Tensor:
    """R_round summed over all AdaRound weight quantizers."""
    total: Optional[torch.Tensor] = None
    for m in iter_qmodules(model):
        r = m.rounding_regularizer()
        total = r if total is None else total + r
    if total is None:
        return torch.zeros(())
    return total


def set_soft_rounding(model: nn.Module, soft: bool) -> nn.Module:
    """Select soft (``soft=True``) or hard (``soft=False``) weight rounding.

    Only the AdaRound weight quantizers switch train/eval mode; every other
    module flag is untouched.  Optimization passes run with soft rounding
    (quantizers in train mode); deployment / Q_0 prediction passes run with
    hard rounding (eval mode rounds to nearest even without following
    ``harden``, which is permanent and reserved for export).
    """
    for m in iter_qmodules(model):
        m.w_quant.train(soft)
    return model


def mean_rounding_regularizer(model: nn.Module) -> torch.Tensor:
    """R_round averaged per quantized weight.

    ``rounding_regularizer`` sums over all weights, which would make the
    regularizer scale with model size; the per-weight mean keeps
    lambda_round comparable across blocks and bit profiles.  Each term
    (1 - |2h - 1|^rho) lies in [0, 1], so the mean is in [0, 1].
    """
    total: Optional[torch.Tensor] = None
    n = 0
    for m in iter_qmodules(model):
        r = m.rounding_regularizer()
        total = r if total is None else total + r
        n += m.weight.numel()
    if total is None or n == 0:
        return torch.zeros(())
    return total / float(n)


@torch.no_grad()
def harden_all_quantizers(model: nn.Module) -> nn.Module:
    """Harden soft rounding for export; evaluate exactly as deployed."""
    for m in iter_qmodules(model):
        m.w_quant.harden()
    return model
