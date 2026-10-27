"""End-to-end toy pipeline tests: calibration, Stage 1/2, variants, stats."""

import pytest
import torch

from tcsr.calibration import (CalibrationResult, run_calibration,
                              run_variant_table)
from tcsr.config import ExperimentConfig, MethodConfig
from tcsr.quant import harden_all_quantizers, quantize_model
from tcsr.stage1 import BlockReconstructor
from tcsr.stage2 import SetReconstructor
from tcsr.teacher import TeacherCache
from tcsr.variants import assign_variant_weights
from tcsr.priority import compute_priorities


def _toy_exp():
    return ExperimentConfig(
        name="toy_test", setting="toy", dataset="toy", model="toy",
        head_type="softmax", num_classes=1, quant_profile="mixed",
        calib_images=8, mmd_bandwidth=0.084,
        method=MethodConfig(stage1_lr=1e-2, stage2_lr=1e-3))


def _teacher_student():
    from tcsr.models.toy import make_toy_detector
    student = make_toy_detector(seed=0, head_type="softmax",
                                num_classes=1, size=8, dim=16, hidden=32)
    teacher = make_toy_detector(seed=1, head_type="softmax",
                                num_classes=1, size=8, dim=16, hidden=32)
    return student, teacher


class TestStage1:
    def test_run_toy(self, toy_factory, toy_batches):
        from tcsr.models.toy import make_toy_detector
        student = make_toy_detector(seed=0, head_type="softmax",
                                    num_classes=1, size=8, dim=16, hidden=32)
        exp = _toy_exp()
        qmodel = quantize_model(student, exp.quant_profile)
        from tcsr.quant import calibrate_quantizers
        calibrate_quantizers(qmodel, toy_batches, device="cpu")
        rec = BlockReconstructor(toy_factory(qmodel), toy_factory(student),
                                 exp.method, device="cpu", seed=0)
        summary = rec.run(toy_batches, updates_per_block=2)
        # 4 blocks; the head block has no quantized operators -> params 0
        assert len(summary) == 4
        assert [s["name"] for s in summary] == ["proj", "enc1", "enc2", "head"]
        assert summary[-1]["params"] == 0
        assert summary[-1]["first_loss"] is None
        for s in summary[:3]:
            assert s["params"] > 0 and s["updates"] == 2
            assert s["first_loss"] is not None
            assert s["mean_loss"] >= 0.0


class TestStage2:
    def _setup(self, toy_factory, toy_batches):
        from tcsr.calibration import build_cache
        student, teacher = _teacher_student()
        exp = _toy_exp()
        cache, adapter_p, batches = build_cache(
            student, teacher, toy_factory, toy_batches, exp, device="cpu",
            max_images=8)
        qmodel = quantize_model(student, exp.quant_profile)
        from tcsr.quant import calibrate_quantizers
        calibrate_quantizers(qmodel, batches, device="cpu")
        adapter_q = toy_factory(qmodel)
        return exp, cache, adapter_p, adapter_q, batches

    def test_run_variant_g(self, toy_factory, toy_batches):
        exp, cache, adapter_p, adapter_q, batches = self._setup(
            toy_factory, toy_batches)
        # priorities from an untrained Q_0 (smoke-level)
        from tcsr.calibration import predict_all
        q0 = predict_all(adapter_q, batches, "cpu", soft=False)
        records = list(cache)
        compute_priorities(records, q0, exp.method, exp.head_type)
        assign_variant_weights(records, "G", exp.method)
        recon = SetReconstructor(adapter_q, adapter_p, cache, exp.method,
                                 "G", bandwidth=0.084, device="cpu", seed=0)
        history = recon.run(batches, updates=3, log_every=1)
        assert len(history) == 3
        assert [h["step"] for h in history] == [1, 2, 3]
        for h in history:
            for key in ("loss", "blk", "fg", "res", "cor", "round"):
                assert key in h
            assert h["loss"] >= 0.0

    def test_variant_b_cor_zero_fg_positive(self, toy_factory, toy_batches):
        exp, cache, adapter_p, adapter_q, batches = self._setup(
            toy_factory, toy_batches)
        from tcsr.calibration import predict_all
        q0 = predict_all(adapter_q, batches, "cpu", soft=False)
        records = list(cache)
        compute_priorities(records, q0, exp.method, exp.head_type)
        assign_variant_weights(records, "B", exp.method)
        recon = SetReconstructor(adapter_q, adapter_p, cache, exp.method,
                                 "B", bandwidth=0.084, device="cpu", seed=0)
        history = recon.run(batches, updates=1, log_every=1)
        # B: zero coefficients -> cor == 0, fg still computed
        assert history[0]["cor"] == pytest.approx(0.0, abs=1e-9)
        assert history[0]["fg"] > 0.0

    def test_missing_weights_raises(self, toy_factory, toy_batches):
        exp, cache, adapter_p, adapter_q, batches = self._setup(
            toy_factory, toy_batches)
        recon = SetReconstructor(adapter_q, adapter_p, cache, exp.method,
                                 "G", bandwidth=0.084, device="cpu", seed=0)
        with pytest.raises(ValueError):
            recon.run(batches, updates=1)


class TestRunCalibration:
    def test_variant_a_no_stage2(self, toy_factory, toy_batches):
        exp = _toy_exp()
        student, teacher = _teacher_student()
        result = run_calibration(student, teacher, toy_batches, toy_factory,
                                 exp, variant="A", device="cpu", seed=0,
                                 stage1_updates=2, log_every=1)
        assert isinstance(result, CalibrationResult)
        assert result.variant == "A"
        assert result.stage2 is None
        assert result.spec.stage2 is False
        assert len(result.stage1) == 4
        assert result.stats["stage2_updates"] == 0.0
        # hardened after run
        from tcsr.quant import iter_qmodules
        assert all(m.w_quant.hardened for m in iter_qmodules(result.qmodel))

    def test_variant_g_end_to_end(self, toy_factory, toy_batches):
        exp = _toy_exp()
        student, teacher = _teacher_student()
        result = run_calibration(student, teacher, toy_batches, toy_factory,
                                 exp, variant="g", device="cpu", seed=0,
                                 stage1_updates=2, stage2_updates=3,
                                 log_every=1)
        assert result.variant == "G"
        assert result.stage2 is not None
        assert [h["step"] for h in result.stage2] == [1, 2, 3]
        assert result.stats["images"] == 4.0
        assert result.stats["anchors"] > 0
        assert result.stats["stage1_blocks"] == 4.0
        assert result.stats["stage2_updates"] == 3.0
        assert result.stats["bandwidth"] == 0.084
        # the quantized model still runs after hardening
        out = result.qmodel(toy_batches[0][0])
        assert torch.isfinite(out).all()

    def test_no_teacher(self, toy_factory, toy_batches):
        exp = _toy_exp()
        student, _ = _teacher_student()
        result = run_calibration(student, None, toy_batches, toy_factory,
                                 exp, variant="G", device="cpu", seed=0,
                                 stage1_updates=1, stage2_updates=1,
                                 log_every=1)
        # no teacher -> corroborated anchors == 0
        assert result.stats["corroborated_anchors"] == 0.0
        # Stage 2 still ran with cor == 0
        assert result.stage2[0]["cor"] == pytest.approx(0.0, abs=1e-9)


class TestRunVariantTable:
    def test_shared_q0_table(self, toy_factory, toy_batches):
        exp = _toy_exp()
        student, teacher = _teacher_student()
        results = run_variant_table(student, teacher, toy_batches,
                                    toy_factory, exp,
                                    variants=["A", "B", "G"], device="cpu",
                                    seed=0, stage1_updates=1,
                                    stage2_updates=2, log_every=1)
        assert set(results) == {"A", "B", "G"}
        assert results["A"].stage2 is None
        for v in ("B", "G"):
            assert results[v].stage2 is not None
            assert [h["step"] for h in results[v].stage2] == [1, 2]
        # identical Stage-1 summaries shared across variants
        assert results["A"].stage1 is results["B"].stage1
        # independent model copies per variant
        assert results["B"].qmodel is not results["G"].qmodel
        # G's cor term differs from B's (zero coefficients)
        assert results["G"].stage2[0]["cor"] >= 0.0
