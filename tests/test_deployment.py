"""Deployment protocol tests: statistics, export wrapper, tools CLIs."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tcsr.deployment import (DEPLOYMENT_CSV_COLUMNS, DEPLOYMENT_PROTOCOL,
                             MODEL_KINDS, ExportWrapper, PowerSampler,
                             aggregate_sessions, build_trt_engine,
                             export_onnx, latency_stats, peak_memory_gib,
                             percentile, reset_peak_memory, session_metrics,
                             verify_onnx)
from tcsr.models.toy import make_toy_detector

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


# ----------------------------------------------------------------------
# Frozen protocol and the published table format.
# ----------------------------------------------------------------------


def test_protocol_frozen():
    assert DEPLOYMENT_PROTOCOL == {
        "input_size": 640, "batch_size": 1, "sessions": 5,
        "warmup_iterations": 200, "timed_iterations": 1000,
        "power_interval_ms": 100, "opset": 21,
    }


def test_csv_columns_match_paper():
    with open(RESULTS / "tab_deployment.csv", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == list(DEPLOYMENT_CSV_COLUMNS)


def test_published_rows_follow_paper_aggregation():
    """throughput = 1000 / p50 and energy = power / throughput."""
    with open(RESULTS / "tab_deployment.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    for r in rows:
        p50 = float(r["latency_p50_ms"])
        assert float(r["throughput_img_s"]) == pytest.approx(1000.0 / p50,
                                                             rel=1e-3)
        if r["power_w"]:
            power = float(r["power_w"])
            energy = float(r["energy_j_per_image"])
            # rel 5e-3: the paper table stores rounded values, so
            # recomputing from the printed power/throughput carries the
            # accumulated rounding of both operands
            assert energy == pytest.approx(
                power / float(r["throughput_img_s"]), rel=5e-3)


# ----------------------------------------------------------------------
# Statistics primitives.
# ----------------------------------------------------------------------


def test_percentile():
    assert percentile([2.0], 50) == pytest.approx(2.0)
    assert percentile([2.0, 4.0], 50) == pytest.approx(3.0)
    assert percentile([2.0, 4.0], 95) == pytest.approx(3.9)
    with pytest.raises(ValueError):
        percentile([], 50)


def test_latency_stats():
    stats = latency_stats([3.0, 1.0, 5.0, 2.0, 4.0])
    assert stats["latency_p50_ms"] == pytest.approx(3.0)
    assert stats["latency_p95_ms"] == pytest.approx(4.8)  # linear interp


def test_session_metrics_paper_relations():
    row = session_metrics([2.0, 4.0], power_w=72.42, peak_memory_gib=1.5)
    assert row["latency_p50_ms"] == pytest.approx(3.0)
    assert row["latency_p95_ms"] == pytest.approx(3.9)
    assert row["throughput_img_s"] == pytest.approx(1000.0 / 3.0)
    assert row["power_w"] == pytest.approx(72.42)
    assert row["energy_j_per_image"] == pytest.approx(
        72.42 / (1000.0 / 3.0))
    assert row["peak_memory_gib"] == pytest.approx(1.5)
    # without power / memory the corresponding keys stay absent
    lean = session_metrics([2.0, 4.0])
    assert "power_w" not in lean and "energy_j_per_image" not in lean
    assert "peak_memory_gib" not in lean


def test_aggregate_sessions_mean_and_sd():
    rows = [
        {"latency_p50_ms": 9.84, "latency_p95_ms": 12.0,
         "throughput_img_s": 101.6, "energy_j_per_image": 0.713,
         "power_w": 72.42},
        {"latency_p50_ms": 10.16, "latency_p95_ms": 12.4,
         "throughput_img_s": 98.4, "energy_j_per_image": 0.72,
         "power_w": 70.85},
    ]
    agg = aggregate_sessions(rows)
    assert agg["latency_p50_ms"] == pytest.approx(10.0)
    assert agg["power_w"] == pytest.approx((72.42 + 70.85) / 2)
    # sample standard deviation (ddof=1) of the SD_KEYS metrics
    for key in ("latency_p50_ms", "latency_p95_ms", "throughput_img_s",
                "energy_j_per_image"):
        expected = np.std([r[key] for r in rows], ddof=1)
        assert agg[f"sd_{key}"] == pytest.approx(expected)
    assert "sd_power_w" not in agg  # sd only for the paper's SD_KEYS


def test_aggregate_sessions_single_row_has_no_sd():
    agg = aggregate_sessions([{"latency_p50_ms": 9.84,
                               "latency_p95_ms": 12.0,
                               "throughput_img_s": 101.6}])
    assert agg["latency_p50_ms"] == pytest.approx(9.84)
    assert not [k for k in agg if k.startswith("sd_")]
    with pytest.raises(ValueError):
        aggregate_sessions([])


# ----------------------------------------------------------------------
# ExportWrapper.
# ----------------------------------------------------------------------


def test_export_wrapper_toy_matches_split():
    model = make_toy_detector(seed=0, head_type="softmax", num_classes=1,
                              size=8, dim=16, hidden=32)
    model.eval()
    wrapper = ExportWrapper(model, "toy")
    x = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        logits, boxes = wrapper(x)
        out = model(x)
        ref_logits, ref_raw = model.split(out)
    q, c = ref_logits.shape[1], ref_logits.shape[2]
    assert logits.shape == (1, q, c) and boxes.shape == (1, q, 4)
    assert torch.equal(logits, ref_logits)
    assert torch.equal(boxes, torch.sigmoid(ref_raw))
    assert 0.0 <= float(boxes.min()) and float(boxes.max()) <= 1.0


def test_export_wrapper_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown model kind"):
        ExportWrapper(torch.nn.Identity(), "yolo")
    assert "toy" in MODEL_KINDS and "rtdetr" in MODEL_KINDS \
        and "defdetr" in MODEL_KINDS


# ----------------------------------------------------------------------
# ONNX round-trip (skipped without onnxruntime).
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_onnx(tmp_path_factory):
    ort = pytest.importorskip("onnxruntime")
    assert ort is not None
    model = make_toy_detector(seed=0, head_type="softmax", num_classes=1,
                              size=8, dim=16, hidden=32)
    model.eval()
    path = str(tmp_path_factory.mktemp("onnx") / "toy_student.onnx")
    export_onnx(model, path, 8, "toy", opset=DEPLOYMENT_PROTOCOL["opset"])
    return path, model


def test_toy_onnx_roundtrip_closure(toy_onnx):
    path, model = toy_onnx
    diffs = verify_onnx(path, model, 8, "toy", atol=1e-3)
    assert diffs is not None
    assert set(diffs) == {"logits", "boxes"}
    assert max(diffs.values()) < 1e-3


def test_toy_onnx_opset(toy_onnx):
    onnx = pytest.importorskip("onnx")
    path, _ = toy_onnx
    model = onnx.load(path)
    versions = {o.version for o in model.opset_import if o.domain in ("", "ai.onnx")}
    assert versions == {DEPLOYMENT_PROTOCOL["opset"]}


# ----------------------------------------------------------------------
# Graceful degradation without GPU-only dependencies.
# ----------------------------------------------------------------------


def test_power_sampler_degrades_without_nvml():
    sampler = PowerSampler(interval_ms=100)
    if sampler.available:
        pytest.skip("NVML is available on this machine")
    with sampler:  # context manager must be usable without NVML
        pass
    assert sampler.stop() == []


def test_build_trt_engine_requires_tensorrt():
    try:
        import tensorrt  # noqa: F401,PLC0415
        pytest.skip("tensorrt is installed")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="tensorrt"):
        build_trt_engine("does_not_exist.onnx", "out.engine")


def test_cuda_memory_helpers_without_cuda():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this machine")
    assert reset_peak_memory("cuda") is False
    assert peak_memory_gib("cuda") is None


# ----------------------------------------------------------------------
# tools CLI end-to-end on the toy detector.
# ----------------------------------------------------------------------


def _run_cli(args):
    return subprocess.run(
        [sys.executable] + args, cwd=str(ROOT), capture_output=True,
        text=True, timeout=600)


def test_export_onnx_cli_toy(tmp_path):
    out = str(tmp_path / "exports")
    proc = _run_cli(["tools/export_onnx.py",
                     "--config", "configs/hrsid_rtdetr.json",
                     "--model", "toy", "--seed", "12001",
                     "--output-dir", out, "--check"])
    assert proc.returncode == 0, proc.stderr
    assert os.path.exists(os.path.join(out, "toy_student.onnx"))
    assert os.path.exists(os.path.join(out, "toy_student.onnx.export.json"))
    summary = json.load(open(os.path.join(out, "export_onnx.json"),
                             encoding="utf-8"))
    assert [e["label"] for e in summary["exports"]] == ["toy_student"]
    assert summary["exports"][0]["input_size"] == 8


def test_benchmark_cli_toy(tmp_path):
    pytest.importorskip("onnxruntime")
    exports = tmp_path / "exports"
    proc = _run_cli(["tools/export_onnx.py",
                     "--config", "configs/hrsid_rtdetr.json",
                     "--model", "toy", "--seed", "12001",
                     "--output-dir", str(exports)])
    assert proc.returncode == 0, proc.stderr
    onnx_path = exports / "toy_student.onnx"
    csv_path = tmp_path / "bench.csv"
    proc = _run_cli(["tools/benchmark.py", "--backend", "ort",
                     "--model", str(onnx_path),
                     "--input-size", "8", "--head-type", "softmax",
                     "--sessions", "1", "--warmup", "1", "--iterations", "2",
                     "--csv", str(csv_path)])
    assert proc.returncode == 0, proc.stderr
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["measurement_boundary"] for r in rows] == \
        ["model_only", "end_to_end"]
    assert all(float(r["latency_p50_ms"]) > 0 for r in rows)
    assert all(float(r["throughput_img_s"]) > 0 for r in rows)
