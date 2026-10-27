"""Deployment support: ONNX export and deployment-efficiency statistics.

Two groups of helpers back the deployment study of the paper (Sec.
"Calibration Cost and Deployment", Table "Deployment efficiency"):

**ONNX export** -- :class:`ExportWrapper` normalizes every detector family
(toy / RT-DETR / Deformable DETR, full-precision or hardened fake-quant
``qmodel.pt``) to a traced graph ``images -> (logits, boxes)``;
:func:`export_onnx` writes the ONNX file and :func:`verify_onnx` compares
engine outputs against the framework model when ``onnxruntime`` is
installed (the accuracy-closure check of Table "Framework and TensorRT
accuracy").

**Benchmark statistics** -- pure-python/numpy primitives implementing the
reported aggregation: latency p50/p95 over the timed iterations, batch-1
throughput ``1000 / p50``, energy per image ``average power / throughput``
(NVML power sampled in the background), peak memory, and the
between-session mean and standard deviation of :func:`aggregate_sessions`.
The frozen measurement protocol (NVIDIA L4 24 GiB, TensorRT 10.3.0, ONNX
1.16.1 opset 21, 640x640 input, batch size 1, five sessions of 200 warm-up
+ 1000 timed iterations, NVML sampled every 100 ms) lives in
:data:`DEPLOYMENT_PROTOCOL` and is the default of ``tools/benchmark.py``.

Native low-bit engines: the exported graph carries the exact fake-quant
arithmetic of the calibrated model (weights dequantized to their quantized
values, activations clamped to the calibrated grids), so an FP16 engine
built from it reproduces the framework evaluation of the quantized model.
The paper's native W4A16/W8A8 TensorRT engines additionally require
Q/DQ-annotated quantization (TensorRT ModelOptimizer / Polygraphy) applied
to the same graph; :func:`build_trt_engine` covers engine building itself.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

#: Frozen measurement protocol of the paper (deployment paragraph).
DEPLOYMENT_PROTOCOL: Dict[str, Any] = {
    "input_size": 640,
    "batch_size": 1,
    "sessions": 5,
    "warmup_iterations": 200,
    "timed_iterations": 1000,
    "power_interval_ms": 100,
    "opset": 21,
}

#: Column order of ``results/tab_deployment.csv`` (paper Table deployment).
DEPLOYMENT_CSV_COLUMNS: Tuple[str, ...] = (
    "measurement_boundary", "model",
    "latency_p50_ms", "latency_p95_ms", "throughput_img_s",
    "power_w", "energy_j_per_image", "peak_memory_gib",
    "sd_latency_p50_ms", "sd_latency_p95_ms",
    "sd_throughput_img_s", "sd_energy_j_per_image",
)

#: metrics whose between-session standard deviation is reported (sd_* keys)
SD_KEYS: Tuple[str, ...] = ("latency_p50_ms", "latency_p95_ms",
                            "throughput_img_s", "energy_j_per_image")

#: supported detector families for :class:`ExportWrapper`
MODEL_KINDS: Tuple[str, ...] = ("toy", "rtdetr", "defdetr")


# ----------------------------------------------------------------------
# ONNX export.
# ----------------------------------------------------------------------


class ExportWrapper(nn.Module):
    """Normalize every detector to ``forward(images) -> (logits, boxes)``.

    ``kind`` selects the calling convention of the wrapped model:

    - ``"toy"``: ``model(images)`` returns one ``(B, Q, C + 4)`` tensor of
      concatenated class logits and raw box parameters; the wrapper splits
      it and applies the sigmoid of the normalized cxcywh boxes.
    - ``"rtdetr"``: ``model(pixel_values=images)`` returns an object with
      ``.logits`` / ``.pred_boxes`` (HuggingFace RTDetrForObjectDetection).
    - ``"defdetr"``: ``model(images)`` returns a dict with
      ``"pred_logits"`` / ``"pred_boxes"`` (official DeformableDETR).

    A hardened ``qmodel.pt`` from ``tools/calibrate.py`` is a deep copy of
    the student with quantized operators, so it is exported with the kind
    of its backbone family; the traced graph keeps the exact fake-quant
    arithmetic (deployment-faithful numerics).
    """

    def __init__(self, model: nn.Module, kind: str):
        super().__init__()
        if kind not in MODEL_KINDS:
            raise ValueError(f"unknown model kind {kind!r}; "
                             f"expected one of {MODEL_KINDS}")
        self.model = model
        self.kind = kind

    def forward(self, images: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.kind == "toy":
            out = self.model(images)
            logits, raw = self.model.split(out)
            return logits, torch.sigmoid(raw)
        if self.kind == "rtdetr":
            out = self.model(pixel_values=images)
            return out.logits, out.pred_boxes
        out = self.model(images)
        return out["pred_logits"], out["pred_boxes"]


def _export_call(wrapper: nn.Module, dummy: torch.Tensor, path: str,
                 opset: int) -> None:
    """torch.onnx.export with a version-robust keyword set.

    Recent torch warns that the TorchScript exporter only supports
    opset <= 20 but still emits the requested version (the paper's
    opset 21); the result runs fine under onnxruntime.
    """
    kwargs: Dict[str, Any] = dict(
        opset_version=opset, input_names=["images"],
        output_names=["logits", "boxes"], do_constant_folding=True)
    try:
        torch.onnx.export(wrapper, (dummy,), path, dynamo=False, **kwargs)
    except TypeError:
        # older torch without the dynamo keyword
        torch.onnx.export(wrapper, (dummy,), path, **kwargs)


@torch.no_grad()
def export_onnx(model: nn.Module, path: str, input_size: int, kind: str,
                opset: int = 21, device: str = "cpu") -> str:
    """Trace ``model`` under :class:`ExportWrapper` and write ``path``.

    The exported graph has one fixed input ``images (1, 3, S, S)`` and two
    outputs ``logits`` / ``boxes`` (normalized cxcywh), matching the
    deployment protocol (batch size 1, paper input size).
    """
    wrapper = ExportWrapper(model, kind).to(device)
    wrapper.eval()
    dummy = torch.randn(1, 3, int(input_size), int(input_size),
                        device=device)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    _export_call(wrapper, dummy, path, int(opset))
    return path


@torch.no_grad()
def verify_onnx(path: str, model: nn.Module, input_size: int, kind: str,
                device: str = "cpu", atol: float = 1e-3
                ) -> Optional[Dict[str, float]]:
    """Run ``path`` under onnxruntime and compare with the framework model.

    Returns ``{output_name: max_abs_diff}``, or None when onnxruntime is
    not installed.  Raises AssertionError when a difference exceeds
    ``atol`` (the source--engine closure check of the paper).
    """
    try:
        import onnxruntime as ort  # noqa: PLC0415
    except ImportError:
        return None
    wrapper = ExportWrapper(model, kind).to(device)
    wrapper.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 3, int(input_size), int(input_size), device=device)
    expected = [t.detach().cpu().numpy() for t in wrapper(x)]
    providers = (["CPUExecutionProvider"] if device == "cpu" else None)
    sess = ort.InferenceSession(path, providers=providers)
    outputs = sess.run(None, {"images": x.detach().cpu().numpy()})
    diffs: Dict[str, float] = {}
    for name, exp, got in zip(("logits", "boxes"), expected, outputs):
        diff = float(np.abs(np.asarray(exp) - np.asarray(got)).max())
        diffs[name] = diff
        assert diff <= atol, f"onnx output {name!r} differs by {diff:.3e} " \
                             f"(atol={atol:.1e})"
    return diffs


# ----------------------------------------------------------------------
# Benchmark statistics (pure python / numpy).
# ----------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """``q``-th percentile (linear interpolation, numpy default)."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("percentile of an empty sequence")
    return float(np.percentile(arr, q))


def latency_stats(latencies_ms: Sequence[float]) -> Dict[str, float]:
    """p50 / p95 latency over the timed iterations of one session."""
    return {"latency_p50_ms": percentile(latencies_ms, 50),
            "latency_p95_ms": percentile(latencies_ms, 95)}


def session_metrics(latencies_ms: Sequence[float],
                    power_w: Optional[float] = None,
                    peak_memory_gib: Optional[float] = None
                    ) -> Dict[str, float]:
    """Metrics of one measurement session (paper aggregation).

    Batch-1 throughput is ``1000 / p50``; energy per image is average
    power divided by throughput.  Power / memory keys are only present
    when the corresponding measurement is available.
    """
    row = latency_stats(latencies_ms)
    row["throughput_img_s"] = 1000.0 / row["latency_p50_ms"]
    if power_w is not None:
        row["power_w"] = float(power_w)
        row["energy_j_per_image"] = row["power_w"] / row["throughput_img_s"]
    if peak_memory_gib is not None:
        row["peak_memory_gib"] = float(peak_memory_gib)
    return row


def aggregate_sessions(sessions: Sequence[Dict[str, Any]]
                       ) -> Dict[str, float]:
    """Between-session mean and standard deviation (paper: five sessions).

    The mean is taken over every numeric key present in all session rows;
    for the metrics in :data:`SD_KEYS` the sample standard deviation
    (ddof=1) is added under the ``sd_`` prefix reported in the paper.
    """
    rows = list(sessions)
    if not rows:
        raise ValueError("aggregate_sessions needs at least one session")
    keys = set(k for k, v in rows[0].items()
               if isinstance(v, (int, float)) and not isinstance(v, bool))
    for r in rows[1:]:
        keys &= {k for k, v in r.items()
                 if isinstance(v, (int, float)) and not isinstance(v, bool)}
    out: Dict[str, float] = {}
    for k in sorted(keys):
        vals = np.asarray([float(r[k]) for r in rows], dtype=np.float64)
        out[k] = float(vals.mean())
        if k in SD_KEYS and vals.size > 1:
            out[f"sd_{k}"] = float(vals.std(ddof=1))
    return out


# ----------------------------------------------------------------------
# NVML power sampling and CUDA peak memory.
# ----------------------------------------------------------------------


class PowerSampler:
    """Background NVML power sampling (paper: every 100 ms).

    Degrades to a no-op when ``pynvml`` or an NVIDIA driver is missing
    (CPU-only machines); :attr:`available` reports this, and :meth:`stop`
    then returns an empty sample list so callers pass ``power_w=None``.
    """

    def __init__(self, interval_ms: int = 100, gpu_index: int = 0):
        self.interval_ms = int(interval_ms)
        self.gpu_index = int(gpu_index)
        self.samples: List[float] = []
        self.available = False
        self._handle = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        try:
            import pynvml  # noqa: PLC0415
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self.available = True
        except Exception:
            self._pynvml = None

    def start(self) -> "PowerSampler":
        if self.available:
            self._stop.clear()
            self.samples = []
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def _loop(self) -> None:
        interval = max(self.interval_ms, 1) / 1000.0
        while not self._stop.wait(interval):
            try:
                self.samples.append(
                    float(self._pynvml.nvmlDeviceGetPowerWatts(self._handle)))
            except Exception:
                return

    def stop(self) -> List[float]:
        """Join the sampler thread and return the collected samples (W)."""
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None
        return list(self.samples)

    def __enter__(self) -> "PowerSampler":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def reset_peak_memory(device: str = "cuda") -> bool:
    """Reset torch's CUDA peak-memory counter; False without CUDA."""
    if not torch.cuda.is_available():
        return False
    torch.cuda.set_device(torch.device(device))
    torch.cuda.reset_peak_memory_stats()
    return True


def peak_memory_gib(device: str = "cuda") -> Optional[float]:
    """Peak allocated CUDA memory in GiB since the last reset, if any."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated(
        torch.device(device)) / 2 ** 30


# ----------------------------------------------------------------------
# TensorRT engine building (optional dependency).
# ----------------------------------------------------------------------


def build_trt_engine(onnx_path: str, engine_path: str, fp16: bool = True,
                     workspace_gib: Optional[float] = None,
                     verbose: bool = False) -> str:
    """Build a TensorRT engine from an ONNX graph.

    Used for the paper's engines (TensorRT 10.3, FP16 flag for the
    full-precision P engine; the mixed W4A16/W8A8 engines start from a
    Q/DQ-annotated export of the same graph).  Requires the ``tensorrt``
    python package.
    """
    try:
        import tensorrt as trt  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "tensorrt is required to build engines: "
            "pip install tensorrt (see NVIDIA docs for the matching "
            "CUDA/driver stack)") from exc
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flag = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    network = builder.create_network(1 << int(flag) if flag is not None else 0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        errors = "\n".join(str(parser.get_error(i))
                           for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT failed to parse {onnx_path}:\n{errors}")
    config = builder.create_builder_config()
    if workspace_gib is not None:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                     int(float(workspace_gib) * 2 ** 30))
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    os.makedirs(os.path.dirname(os.path.abspath(engine_path)) or ".",
                exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized.numpy() if hasattr(serialized, "numpy")
                else bytes(serialized))
    return engine_path


__all__ = [
    "DEPLOYMENT_CSV_COLUMNS", "DEPLOYMENT_PROTOCOL", "MODEL_KINDS",
    "SD_KEYS", "ExportWrapper", "PowerSampler", "aggregate_sessions",
    "build_trt_engine", "export_onnx", "latency_stats", "peak_memory_gib",
    "percentile", "reset_peak_memory", "session_metrics", "verify_onnx",
]
