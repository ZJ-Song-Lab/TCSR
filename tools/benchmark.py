"""Benchmark a deployed engine under the paper's measurement protocol.

Real setting (onnxruntime)::

    python tools/benchmark.py --backend ort --model exports/variant_G.onnx \
        --label tensorrt_fp16_G --csv results/tab_deployment.csv

TensorRT engines (requires CUDA + the ``tensorrt`` package)::

    python tools/benchmark.py --backend trt --model exports/variant_G.engine

The defaults reproduce the frozen protocol of the paper: 640x640 input,
batch size 1, five sessions of 200 warm-up + 1000 timed iterations, NVML
power sampled every 100 ms.  ``--boundary`` selects the measurement
boundary: ``model_only`` times the engine alone with inputs resident on
the device, ``end_to_end`` additionally times input transfer, output
transfer and the deployment decode (class probabilities + top-k) of
every iteration; both boundaries are reported by default, exactly as in
Table "Deployment efficiency".

Outputs: per-session metrics on stdout plus, when given, ``--output``
(JSON, rows are appended) and ``--csv`` (``results/tab_deployment.csv``
format, appended or created with the paper's header).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from time import perf_counter

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import run_metadata, save_json  # noqa: E402

from tcsr.config import load_config  # noqa: E402
from tcsr.deployment import (DEPLOYMENT_CSV_COLUMNS, DEPLOYMENT_PROTOCOL,  # noqa: E402
                             PowerSampler, aggregate_sessions,
                             peak_memory_gib, reset_peak_memory,
                             session_metrics)

_BOUNDARIES = ("model_only", "end_to_end")

#: trt.DataType -> torch dtype (TensorRT 10 tensor I/O buffers)
_TRT_DTYPES = {"float32": torch.float32, "float16": torch.float16,
               "int32": torch.int32, "int64": torch.int64,
               "int8": torch.int8, "bool": torch.bool}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend", required=True, choices=("ort", "trt"),
                   help="onnxruntime (.onnx) or TensorRT (.engine)")
    p.add_argument("--model", required=True,
                   help="engine/optimizer file (.onnx for ort, .engine for trt)")
    p.add_argument("--label", default=None,
                   help="model label of the result rows (default: file stem)")
    p.add_argument("--config", default=None,
                   help="experiment config JSON (input_size / head_type; "
                        "otherwise --input-size / --head-type)")
    p.add_argument("--input-size", type=int, default=None,
                   help="benchmark input size (default: config or 640)")
    p.add_argument("--head-type", default=None,
                   help="sigmoid | softmax (default: config or softmax)")
    p.add_argument("--sessions", type=int,
                   default=DEPLOYMENT_PROTOCOL["sessions"])
    p.add_argument("--warmup", type=int,
                   default=DEPLOYMENT_PROTOCOL["warmup_iterations"])
    p.add_argument("--iterations", type=int,
                   default=DEPLOYMENT_PROTOCOL["timed_iterations"])
    p.add_argument("--power-interval-ms", type=int,
                   default=DEPLOYMENT_PROTOCOL["power_interval_ms"])
    p.add_argument("--boundary", action="append", choices=_BOUNDARIES,
                   help="measurement boundary (repeatable; default: both)")
    p.add_argument("--device", default=None,
                   help="default: cuda if available else cpu")
    p.add_argument("--max-det", type=int, default=100,
                   help="top-k truncation of the end_to_end decode")
    p.add_argument("--seed", type=int, default=0,
                   help="seed of the fixed benchmark input")
    p.add_argument("--output", default=None, help="JSON results (appended)")
    p.add_argument("--csv", default=None,
                   help="CSV results in tab_deployment format (appended)")
    return p.parse_args(argv)


# ----------------------------------------------------------------------
# Backends: probe / run_bound / run_e2e.
# ----------------------------------------------------------------------


class OrtBackend:
    """onnxruntime session; CUDA EP uses io_binding with resident tensors."""

    def __init__(self, path: str, device: str):
        try:
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError:
            raise SystemExit("onnxruntime is required for --backend ort: "
                             "pip install onnxruntime (or onnxruntime-gpu)")
        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda") and torch.cuda.is_available():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.device = device
        self._binding = None
        self._x_dev = None
        self._out_dev = []

    def _probe(self, x_np: np.ndarray):
        """Run once on CPU to learn the output shapes."""
        return self.sess.run(None, {self.input_name: x_np})

    def _bind(self, x_np: np.ndarray):
        """Move the input to the device and allocate output buffers."""
        self._x_dev = torch.from_numpy(x_np).to(self.device).contiguous()
        dev_id = self._x_dev.device.index or 0
        self._out_dev = [torch.empty(tuple(o.shape), device=self.device,
                                     dtype=o.dtype)
                         for o in self._probe_outputs]
        self._binding = self.sess.io_binding()
        self._binding.bind_input(
            name=self.input_name, device_type="cuda", device_id=dev_id,
            element_type=np.float32, shape=tuple(self._x_dev.shape),
            buffer_ptr=self._x_dev.data_ptr())
        for name, buf in zip(self.output_names, self._out_dev):
            self._binding.bind_output(
                name=name, device_type="cuda", device_id=dev_id,
                element_type=np.float32, shape=tuple(buf.shape),
                buffer_ptr=buf.data_ptr())

    def probe(self, x_np: np.ndarray):
        """Learn output shapes and, on CUDA, allocate resident buffers."""
        outs = self._probe(x_np)
        self._probe_outputs = [torch.from_numpy(np.ascontiguousarray(o))
                               for o in outs]
        self._x_np = np.ascontiguousarray(x_np)
        if self.device.startswith("cuda") and torch.cuda.is_available():
            self._bind(x_np)
        return outs

    def run_bound(self):
        """One model_only step: input resident on the device, timed."""
        if self._binding is not None:
            self.sess.run_with_iobinding(self._binding)
            return [o for o in self._out_dev]
        return self.sess.run(None, {self.input_name: self._x_np})

    def run_e2e(self, x_np: np.ndarray):
        """One end_to_end step: numpy in -> H2D -> run -> D2H -> numpy."""
        if self._binding is None:  # CPU path
            return self.sess.run(None, {self.input_name: x_np})
        x = torch.from_numpy(x_np).to(self.device)
        binding = self.sess.io_binding()
        dev_id = x.device.index or 0
        binding.bind_input(name=self.input_name, device_type="cuda",
                           device_id=dev_id, element_type=np.float32,
                           shape=tuple(x.shape), buffer_ptr=x.data_ptr())
        out = [torch.empty_like(o) for o in self._probe_outputs]
        for name, buf in zip(self.output_names, out):
            binding.bind_output(name=name, device_type="cuda",
                                device_id=dev_id, element_type=np.float32,
                                shape=tuple(buf.shape),
                                buffer_ptr=buf.data_ptr())
        self.sess.run_with_iobinding(binding)
        return [o.cpu().numpy() for o in out]


class TrtBackend:
    """TensorRT 10 CUDA engine (tensor I/O, execute_async_v3)."""

    def __init__(self, path: str, device: str):
        if not (device.startswith("cuda") and torch.cuda.is_available()):
            raise SystemExit("--backend trt requires a CUDA device")
        try:
            import tensorrt as trt  # noqa: PLC0415
        except ImportError:
            raise SystemExit("tensorrt is required for --backend trt: "
                             "pip install tensorrt")
        self._trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine is None:
            raise SystemExit(f"failed to deserialize engine {path}")
        self.engine = engine
        self.context = engine.create_execution_context()
        self.device = device
        self.input_name = None
        self.output_names = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)
        self._x_dev = None
        self._out_dev = []

    def _alloc(self, name: str, shape) -> torch.Tensor:
        trt_dtype = str(self.engine.get_tensor_dtype(name)).split(".")[-1]
        if trt_dtype not in _TRT_DTYPES:
            raise SystemExit(f"unsupported tensor dtype {trt_dtype!r} "
                             f"for {name!r}")
        return torch.empty(tuple(int(d) for d in shape), device="cuda",
                           dtype=_TRT_DTYPES[trt_dtype])

    def probe(self, x_np: np.ndarray):
        """Set shapes, allocate device buffers and bind their addresses."""
        self._x_dev = torch.from_numpy(x_np).to(self.device).contiguous()
        in_shape = tuple(int(d) for d in self._x_dev.shape)
        if self.engine.get_tensor_shape(self.input_name)[-1] == -1:
            self.context.set_input_shape(self.input_name, in_shape)
        self.context.set_tensor_address(self.input_name,
                                        self._x_dev.data_ptr())
        self._out_dev = [
            self._alloc(n, self.context.get_tensor_shape(n))
            for n in self.output_names]
        for name, buf in zip(self.output_names, self._out_dev):
            self.context.set_tensor_address(name, buf.data_ptr())
        self._sync()
        return [o.cpu().numpy() for o in self._out_dev]

    def _sync(self) -> None:
        torch.cuda.synchronize(torch.device(self.device))

    def run_bound(self):
        """One model_only step (input resident, async v3 + synchronize)."""
        stream = torch.cuda.current_stream(torch.device(self.device))
        ok = self.context.execute_async_v3(
            stream_handle=stream.cuda_stream)
        if not ok:
            raise RuntimeError("execute_async_v3 failed")
        self._sync()
        return list(self._out_dev)

    def run_e2e(self, x_np: np.ndarray):
        """One end_to_end step: numpy in -> H2D -> run -> D2H -> numpy."""
        self._x_dev.copy_(torch.from_numpy(x_np))
        self.run_bound()
        return [o.cpu().numpy() for o in self._out_dev]


# ----------------------------------------------------------------------
# Deployment decode (end_to_end boundary).
# ----------------------------------------------------------------------


def deployment_decode(logits: np.ndarray, boxes: np.ndarray,
                      head_type: str, max_det: int):
    """Class probabilities + top-k of one image (deployment decode).

    Mirrors the score rule of ``tcsr.evaluation`` (softmax head: the
    background is the last channel; sigmoid head: all channels); the
    result ordering is irrelevant, only the compute is timed.
    """
    if head_type == "softmax":
        z = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(z)
        probs = (e / e.sum(axis=-1, keepdims=True))[..., :-1]
    elif head_type == "sigmoid":
        probs = 1.0 / (1.0 + np.exp(-logits))
    else:
        raise ValueError(f"unknown head_type {head_type!r}")
    scores = probs.max(axis=-1).reshape(-1)
    k = min(int(max_det), scores.size)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    return np.argpartition(-scores, k - 1)[:k]


# ----------------------------------------------------------------------
# Session loop.
# ----------------------------------------------------------------------


def run_boundary(backend, x_np: np.ndarray, boundary: str, head_type: str,
                 max_det: int, args) -> list:
    """All sessions of one measurement boundary; returns session rows."""
    session_rows = []
    n_sessions = int(args.sessions)
    for s in range(n_sessions):
        with PowerSampler(interval_ms=args.power_interval_ms) as sampler:
            reset_peak_memory(args.device)
            if boundary == "model_only":
                for _ in range(args.warmup):
                    backend.run_bound()
                latencies = []
                for _ in range(args.iterations):
                    t0 = perf_counter()
                    backend.run_bound()
                    latencies.append((perf_counter() - t0) * 1000.0)
            else:
                for _ in range(args.warmup):
                    outs = backend.run_e2e(x_np)
                    deployment_decode(outs[0], outs[1], head_type, max_det)
                latencies = []
                for _ in range(args.iterations):
                    t0 = perf_counter()
                    outs = backend.run_e2e(x_np)
                    deployment_decode(outs[0], outs[1], head_type, max_det)
                    latencies.append((perf_counter() - t0) * 1000.0)
            samples = sampler.stop()
        power_w = float(np.mean(samples)) if samples else None
        peak = peak_memory_gib(args.device)
        row = session_metrics(latencies, power_w, peak)
        session_rows.append(row)
        print(f"[benchmark] {boundary} session {s + 1}/{n_sessions}: "
              f"p50={row['latency_p50_ms']:.3f} ms "
              f"p95={row['latency_p95_ms']:.3f} ms "
              f"thr={row['throughput_img_s']:.1f} img/s"
              + (f" power={power_w:.2f} W" if power_w is not None else ""))
    return session_rows


def append_csv_row(path: str, row: dict) -> None:
    """Append one result row in the tab_deployment.csv format."""
    exists = os.path.exists(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DEPLOYMENT_CSV_COLUMNS,
                                extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: ("" if row.get(k) is None else row.get(k))
                         for k in DEPLOYMENT_CSV_COLUMNS})


def append_json_row(path: str, row: dict) -> None:
    """Append one result row to a JSON ``{"rows": [...]}`` file."""
    data = {"rows": []}
    if os.path.exists(path):
        import json
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            loaded.setdefault("rows", [])
            data = loaded
        elif isinstance(loaded, list):
            data = {"rows": loaded}
    data["rows"].append(row)
    save_json(path, data)


def main(argv=None):
    args = parse_args(argv)
    boundaries = args.boundary or list(_BOUNDARIES)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    if args.config:
        exp = load_config(args.config)
        input_size = args.input_size or exp.input_size
        head_type = args.head_type or exp.head_type
    else:
        input_size = args.input_size or DEPLOYMENT_PROTOCOL["input_size"]
        head_type = args.head_type or "softmax"

    label = args.label or os.path.splitext(os.path.basename(args.model))[0]
    backend_cls = OrtBackend if args.backend == "ort" else TrtBackend
    backend = backend_cls(args.model, device)

    rng = np.random.default_rng(args.seed)
    x_np = rng.standard_normal(
        (1, 3, int(input_size), int(input_size))).astype(np.float32)
    backend.probe(x_np)  # learn shapes / allocate resident buffers

    for boundary in boundaries:
        rows = run_boundary(backend, x_np, boundary, head_type,
                            args.max_det, args)
        agg = aggregate_sessions(rows)
        record = {"measurement_boundary": boundary, "model": label,
                  "backend": args.backend, "engine": args.model,
                  "input_size": int(input_size), "head_type": head_type,
                  "sessions": args.sessions, "warmup": args.warmup,
                  "iterations": args.iterations, "max_det": args.max_det}
        record.update(agg)
        if args.output:
            record["run"] = run_metadata(
                sys.argv if argv is None else argv, mode="benchmark")
            append_json_row(args.output, record)
        if args.csv:
            append_csv_row(args.csv, record)
        print(f"[benchmark] {label} ({boundary}) over {args.sessions} "
              f"sessions: p50={agg['latency_p50_ms']:.3f} ms, "
              f"thr={agg['throughput_img_s']:.1f} img/s"
              + (f", energy={agg['energy_j_per_image']:.3f} J/img"
                 if "energy_j_per_image" in agg else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
