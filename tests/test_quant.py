"""Quantizer, observer, fake-quant wrapper, and bit-policy tests."""

import pytest
import torch
import torch.nn as nn

from tcsr.models.toy import make_toy_detector
from tcsr.quant import (AdaRoundQuantizer, MinMaxObserver, MSEObserver,
                        OpBits, PROFILES, QConv2d, QLinear, UniformQuantizer,
                        calibrate_quantizers, get_bit_assignment,
                        harden_all_quantizers, iter_qmodules,
                        quantize_model, round_to_nearest,
                        trainable_quantizer_parameters)
from tcsr.quant.fake_quant import (calibrate_quantizers as _cq,
                                   mean_rounding_regularizer,
                                   set_soft_rounding)


# ----------------------------------------------------------------------
# round_to_nearest / UniformQuantizer
# ----------------------------------------------------------------------

def test_round_to_nearest_bankers():
    v = torch.tensor([0.5, 1.5, -0.5, -1.5, 2.4])
    out = round_to_nearest(v)
    assert out.tolist() == [0.0, 2.0, -0.0, -2.0, 2.0]


def test_uniform_quantizer_defaults():
    q = UniformQuantizer(bits=8)
    assert q.qmin == 0 and q.qmax == 255
    assert q.symmetric is False and q.learnable is True
    assert isinstance(q.log_scale, nn.Parameter)
    assert isinstance(q.zero_point, torch.Tensor) and q.zero_point.dtype == torch.long


def test_symmetric_grid():
    q = UniformQuantizer(bits=8, symmetric=True)
    assert q.qmin == -128 and q.qmax == 127


def test_configure_recovers_scale():
    q = UniformQuantizer(bits=8)
    q.configure(0.0, 1.0)
    # _target_to_log_scale is the exact softplus inverse: scale == 1/255 to
    # float32 precision.
    assert q.scale.item() == pytest.approx(1.0 / 255.0, rel=5e-3)


def test_grid_points_recovered_exactly():
    q = UniformQuantizer(bits=8)
    q.configure(0.0, 1.0)
    s = q.scale.detach()
    for k in (0, 1, 63, 100, 254, 255):
        v = s * k
        assert q.quantize(v).item() == pytest.approx(v.item(), abs=1e-9)


def test_grid_points_symmetric():
    q = UniformQuantizer(bits=8, symmetric=True)
    q.configure(-0.5, 1.0)
    s = q.scale.detach()
    for k in (-100, -64, -1, 0, 1, 63, 100):
        v = s * k
        assert q.quantize(v).item() == pytest.approx(v.item(), abs=1e-9)


def test_asymmetric_zero_point():
    q = UniformQuantizer(bits=8)
    q.configure(-0.5, 1.5)
    s = q.scale.detach()
    z = q.zero_point
    assert z.item() == 64
    # Representable values are v = s * k with the integer grid index
    # k in [qmin - z, qmax - z] = [-64, 191]; all stay inside the clip
    # range [-0.5, 1.5] and are recovered exactly.
    for k in (-63, -50, 0, 100, 190):
        v = s * k
        assert q.quantize(v).item() == pytest.approx(v.item(), abs=1e-9)


def test_ste_gradient_ones():
    q = UniformQuantizer(bits=8)
    q.configure(-1.0, 1.0)
    v = torch.tensor([0.37, -0.61], requires_grad=True)
    out = q(v)
    out.sum().backward()
    assert v.grad.abs().tolist() == pytest.approx([1.0, 1.0], abs=1e-6)


def test_clipping_out_of_range():
    q = UniformQuantizer(bits=8)
    q.configure(0.0, 1.0)
    out = q(torch.tensor([5.0, -5.0]))
    hi = (q.scale * (q.qmax - q.zero_point.to(q.scale.dtype))).item()
    lo = (q.scale * (q.qmin - q.zero_point.to(q.scale.dtype))).item()
    assert out[0].item() == pytest.approx(hi, abs=1e-8)
    assert out[1].item() == pytest.approx(lo, abs=1e-8)


def test_learnable_false_parameters_frozen():
    q = UniformQuantizer(bits=8, learnable=False)
    names = [n for n, _ in q.named_parameters()]
    assert set(names) == {"log_scale", "clip_lo", "clip_hi"}
    assert all(not p.requires_grad for _, p in q.named_parameters())


# ----------------------------------------------------------------------
# AdaRoundQuantizer
# ----------------------------------------------------------------------

def test_adaround_soft_vs_hard():
    q = AdaRoundQuantizer(bits=8, rho=20.0)
    q.configure(0.0, 1.0)
    s = q.scale.detach().item()
    v = torch.tensor([s * 10.25])
    q.train()
    soft = q.quantize(v).item()
    q.eval()  # eval rounds hard even before harden()
    hard = q.quantize(v).item()
    assert s * 10.0 <= soft <= s * 10.5
    assert hard == pytest.approx(s * 10.0, abs=1e-9)
    q.train()
    q.harden()
    assert q.soft is False and q.hardened is True
    assert q.quantize(v).item() == pytest.approx(s * 10.0, abs=1e-9)


def test_adaround_regularizer():
    q = AdaRoundQuantizer(bits=8, rho=20.0, weight_shape=(3, 4))
    q.configure(0.0, 1.0)
    w = torch.zeros(3, 4)
    # beta = 0 -> h = sigmoid(-1) ~ 0.269 -> each weight contributes
    # 1 - |2h - 1|^20 ~ 1 - 1.9e-7; the sum over the 12 weights is ~12.
    r = q.rounding_regularizer(w)
    h0 = float(torch.sigmoid(torch.tensor(-1.0)))
    expected = 12.0 * (1.0 - abs(2.0 * h0 - 1.0) ** 20.0)
    assert r.item() == pytest.approx(expected, rel=1e-3)
    with torch.no_grad():
        q.alpha.fill_(100.0)
    r2 = q.rounding_regularizer(w)
    assert r2.item() == pytest.approx(0.0, abs=1e-4)


# ----------------------------------------------------------------------
# Observers
# ----------------------------------------------------------------------

def test_minmax_observer_merge():
    o = MinMaxObserver()
    o.update(torch.tensor([1.0, -2.0]))
    o.update(torch.tensor([0.5, 3.0]))
    lo, hi = o.range()
    assert float(lo) == -2.0 and float(hi) == 3.0


def test_minmax_observer_symmetric():
    o = MinMaxObserver(symmetric=True)
    o.update(torch.tensor([-1.0, 4.0]))
    lo, hi = o.range()
    assert float(lo) == -4.0 and float(hi) == 4.0


def test_minmax_observer_per_channel():
    o = MinMaxObserver(per_channel=True)
    o.update(torch.tensor([[1.0, -1.0], [5.0, 2.0]]))
    lo, hi = o.range()
    assert lo.tolist() == [-1.0, 2.0]
    assert hi.tolist() == [1.0, 5.0]


def test_minmax_observer_no_data():
    o = MinMaxObserver()
    with pytest.raises(AssertionError):
        o.range()


def test_mse_observer_shrinks_outliers():
    o = MSEObserver()
    torch.manual_seed(0)
    x = torch.randn(8192)
    x[0] = 8.0
    x[-1] = -8.0
    o.update(x)                      # 1 batch with outliers
    for _ in range(3):               # 3 clean batches
        o.update(torch.randn(8192))
    lo, hi = o.range()
    assert 4.0 < float(hi) < 8.0


def test_mse_observer_dense_uniform_keeps_range():
    o = MSEObserver()
    o.update(torch.linspace(-1.0, 1.0, 4096))
    lo, hi = o.range()
    assert float(hi) > 0.95
    assert float(lo) < -0.95


# ----------------------------------------------------------------------
# fake_quant: quantize_model / calibrate_quantizers
# ----------------------------------------------------------------------

def test_quantize_model_toy_inventory():
    model = make_toy_detector(seed=0, head_type="softmax", num_classes=1,
                              size=8, dim=16, hidden=32)
    qmodel = quantize_model(model, "w8a8")
    qnames = {n for n, m in qmodel.named_modules() if isinstance(m, (QLinear, QConv2d))}
    assert qnames == {"proj.conv", "enc1.fc1", "enc1.fc2", "enc2.fc1", "enc2.fc2"}
    assert isinstance(qmodel.head.cls, nn.Linear)


def test_quantize_model_mixed_decoder_w4a8():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Module()
            self.decoder.fc = nn.Linear(8, 8)
            self.encoder = nn.Module()
            self.encoder.fc = nn.Linear(8, 8)

        def forward(self, x):
            return self.decoder.fc(self.encoder.fc(x))

    qm = quantize_model(M(), "mixed")
    assert qm.decoder.fc.w_quant.bits == 4
    assert qm.decoder.fc.a_quant.bits == 8
    assert qm.encoder.fc.w_quant.bits == 8
    assert qm.encoder.fc.a_quant.bits == 8


def test_quantize_model_weight_copied_and_frozen():
    model = make_toy_detector(seed=0, size=8, dim=16, hidden=32)
    qmodel = quantize_model(model, "w8a8")
    assert torch.equal(qmodel.enc1.fc1.weight, model.enc1.fc1.weight)
    assert not qmodel.enc1.fc1.weight.requires_grad


def test_uncalibrated_quantized_weight_degenerate():
    model = make_toy_detector(seed=0, size=8, dim=16, hidden=32)
    qmodel = quantize_model(model, "w8a8")
    # eval mode rounds hard: h = round(sigmoid(-1)) = 0 -> w_q == 0 exactly.
    qmodel.eval()
    w = qmodel.enc1.fc1.quantized_weight()
    assert torch.all(w == 0)


def test_calibrate_quantizers_finite_output(toy_detector, toy_batches):
    qmodel = quantize_model(toy_detector, "w8a8")
    calibrate_quantizers(qmodel, toy_batches, device="cpu")
    w = qmodel.enc1.fc1.quantized_weight()
    assert torch.isfinite(w).all() and bool((w != 0).any())
    out = qmodel(toy_batches[0][0])
    assert torch.isfinite(out).all()


def test_calibrate_quantizers_mse_path(toy_detector, toy_batches):
    qmodel = quantize_model(toy_detector, "w8a8")
    calibrate_quantizers(qmodel, toy_batches, device="cpu",
                         weight_observer="mse")
    w = qmodel.enc1.fc1.quantized_weight()
    assert torch.isfinite(w).all() and bool((w != 0).any())


def test_trainable_quantizer_parameters(toy_detector):
    qmodel = quantize_model(toy_detector, "w8a8")
    params = trainable_quantizer_parameters(qmodel)
    assert len(params) > 0
    assert all(p.requires_grad for p in params)


def test_soft_rounding_and_harden(toy_detector, toy_batches):
    qmodel = quantize_model(toy_detector, "w8a8")
    calibrate_quantizers(qmodel, toy_batches, device="cpu")
    set_soft_rounding(qmodel, True)
    assert all(m.w_quant.training for m in iter_qmodules(qmodel))
    set_soft_rounding(qmodel, False)
    assert all(not m.w_quant.training for m in iter_qmodules(qmodel))
    r = mean_rounding_regularizer(qmodel)
    assert 0.0 <= float(r) <= 1.0
    harden_all_quantizers(qmodel)
    assert all(m.w_quant.hardened for m in iter_qmodules(qmodel))


# ----------------------------------------------------------------------
# policy
# ----------------------------------------------------------------------

def test_bit_policy():
    assert get_bit_assignment("backbone.layer1", "w8a8") == OpBits(8, 8)
    assert get_bit_assignment("decoder.layers.fc", "mixed") == OpBits(4, 8)
    assert get_bit_assignment("backbone.conv", "mixed") == OpBits(8, 8)
    assert get_bit_assignment("backbone.conv", "w4a4") == OpBits(4, 4)
    assert get_bit_assignment("head.cls", "w8a8") is None
    assert get_bit_assignment("model.stem.conv", "mixed") is None
    with pytest.raises(ValueError):
        get_bit_assignment("backbone.conv", "int4")


def test_macs_shares_sum_to_one():
    from tcsr.quant.policy import macs_shares
    a, b, c = macs_shares()
    assert a + b + c == pytest.approx(1.0)


def test_profiles():
    assert set(PROFILES) == {"w8a8", "mixed", "w4a4"}
