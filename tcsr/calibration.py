"""Calibration orchestration (Algorithm 1) and the shared-Q_0 variant table.

``run_calibration`` reproduces Algorithm 1 end to end for a single variant:

1. Cache the full-precision student predictions P and build the anchor set
   A_P; if a teacher exists, cache T, solve the feasible-edge matching and
   cache the detached coefficients c (``tcsr.teacher.TeacherCache``).
2. Copy the frozen student into the quantized model Q_theta and initialize
   the quantizer ranges.
3. Stage 1: teacher-free sequential block reconstruction producing the
   common starting point Q_0 (``tcsr.stage1.BlockReconstructor``).
4. Solve the Q_0 assignment rho_0*, record the frozen priorities u^0 and
   the Stage-1 difficulty g_Q (``tcsr.priority.compute_priorities``), and
   freeze the variant coefficients w (``tcsr.variants``).
5. Stage 2: set reconstruction with the per-minibatch re-solved assignment
   (``tcsr.stage2.SetReconstructor``), updating only theta.
6. Harden the rounding and return the quantized model; the teacher is used
   nowhere after step 1.

``run_variant_table`` runs the paper's mechanism table: Stage 1 is executed
once, the resulting Q_0 state is snapshotted (a ``copy.deepcopy`` -- the
soft/hard rounding flags are plain attributes and must survive the
snapshot), the priorities are computed once from the same Q_0, and every
variant B-I restarts Stage 2 from that shared snapshot with the same
optimizer, update count, anchor sets, matching, and residual regularizer
(paper Sec. 5.4).  Variant A is the Stage-1-only baseline.

Memory note: the variant table holds one hardened model per variant; for
the published detectors this is roughly one model copy per variant on the
calibration device.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .batching import collect_batches
from .config import ExperimentConfig
from .models import DetectorAdapter
from .priority import compute_priorities
from .quant.fake_quant import (calibrate_quantizers, harden_all_quantizers,
                               quantize_model, set_soft_rounding)
from .stage1 import BlockReconstructor
from .stage2 import SetReconstructor
from .teacher import TeacherCache
from .variants import (VARIANT_SPECS, VariantSpec, assign_variant_weights,
                       normalize_variant)


@dataclass
class CalibrationResult:
    """Outcome of one calibration run."""

    variant: str
    qmodel: nn.Module                # hardened quantized model, no teacher refs
    spec: VariantSpec
    stage1: List[Dict]               # per-block Stage-1 summaries
    stage2: Optional[List[Dict]]     # Stage-2 loss history (None for variant A)
    cache: TeacherCache              # frozen per-image state (diagnostics)
    bandwidth: float                 # frozen residual-MMD bandwidth
    stats: Dict[str, float] = field(default_factory=dict)


def _no_log(msg: str) -> None:
    pass


def _build_cache(student, teacher, adapter_factory, batches, exp, device,
                 log) -> Tuple[TeacherCache, DetectorAdapter]:
    """Algorithm 1, lines 1-4: cache P, A_P, and (if present) T and c.

    Returns the cache plus the full-precision student adapter (reused as
    the reconstruction-target adapter by both stages).
    """
    adapter_p = adapter_factory(student)
    adapter_p.model.eval()
    adapter_t = None
    if teacher is not None:
        adapter_t = adapter_factory(teacher)
        adapter_t.model.eval()
    items: List[Tuple[str, object, object]] = []
    with torch.no_grad():
        for images, ids in batches:
            images = images.to(device)
            sp = adapter_p.predict(images)
            tp = (adapter_t.predict(images) if adapter_t is not None
                  else [None] * len(ids))
            items.extend(zip(ids, sp, tp))
    cache = TeacherCache.build(items, exp.method, exp.head_type)
    cache.to(device)
    log(f"[calib] cache built: {cache.stats()}")
    return cache, adapter_p


def _predict_all(adapter: DetectorAdapter, batches, device, soft: bool
                 ) -> Dict[str, object]:
    """Per-image predictions of the bound model (soft toggles rounding)."""
    model = adapter.model
    model.eval()
    set_soft_rounding(model, soft)
    out: Dict[str, object] = {}
    with torch.no_grad():
        for images, ids in batches:
            preds = adapter.predict(images.to(device))
            for image_id, p in zip(ids, preds):
                out[image_id] = p
    set_soft_rounding(model, True)
    return out


# ----------------------------------------------------------------------
# Public building blocks (used by the command-line tools; the private
# helpers above are internal pipeline details).
# ----------------------------------------------------------------------

def prepare_calibration(student: nn.Module, teacher: Optional[nn.Module],
                        loader, adapter_factory, exp: ExperimentConfig,
                        device: str = "cpu", seed: int = 0,
                        stage1_updates: Optional[int] = None,
                        log: Optional[Callable[[str], None]] = None
                        ) -> Tuple[List, TeacherCache, DetectorAdapter,
                                   nn.Module, List[Dict]]:
    """Collect batches, build the cache, quantize, and run Stage 1.

    Returns (batches, cache, adapter_p, qmodel, stage1_summary); ``qmodel``
    is the Stage-1 output Q_0 with soft rounding still active (call
    :func:`predict_all` with ``soft=False`` for deployment-consistent Q_0
    predictions, then continue with Stage 2 or harden).
    """
    return _shared_setup(student, teacher, loader, adapter_factory, exp,
                         torch.device(device), seed, stage1_updates,
                         log if log is not None else _no_log)


def build_cache(student: nn.Module, teacher: Optional[nn.Module],
                adapter_factory, loader, exp: ExperimentConfig,
                device: str = "cpu", max_images: Optional[int] = None,
                log: Optional[Callable[[str], None]] = None
                ) -> Tuple[TeacherCache, DetectorAdapter, List]:
    """Build the frozen per-image cache (P, T, A_P, c) over a loader.

    ``max_images`` defaults to ``exp.calib_images``.  Returns
    (cache, adapter_p, batches) where batches are the materialized
    (images, ids) pairs the cache was built from.
    """
    log = log if log is not None else _no_log
    batches = collect_batches(
        loader, max_images if max_images is not None else exp.calib_images)
    cache, adapter_p = _build_cache(student, teacher, adapter_factory,
                                    batches, exp, torch.device(device), log)
    return cache, adapter_p, batches


def predict_all(adapter: DetectorAdapter, batches, device,
                soft: bool = False) -> Dict[str, object]:
    """Per-image predictions of the adapter's bound model.

    ``soft`` toggles the rounding mode during the forward pass (False =
    deployment / hard rounding); soft rounding is restored afterwards.
    Returns a mapping image_id -> Prediction.
    """
    return _predict_all(adapter, batches, torch.device(device), soft)


def _shared_setup(student, teacher, loader, adapter_factory, exp, device,
                  seed, stage1_updates, log):
    """Common prefix of both entry points: cache, quantize, Stage 1."""
    method = exp.method
    batches = collect_batches(loader, exp.calib_images)
    if not batches:
        raise ValueError("the calibration loader produced no batches")
    cache, adapter_p = _build_cache(student, teacher, adapter_factory,
                                    batches, exp, device, log)
    qmodel = quantize_model(student, exp.quant_profile)
    calibrate_quantizers(qmodel, batches, device=device)
    adapter_q = adapter_factory(qmodel)
    stage1 = BlockReconstructor(adapter_q, adapter_p, method, device=device,
                                seed=seed, log=log)
    s1_summary = stage1.run(batches, updates_per_block=stage1_updates)
    return batches, cache, adapter_p, qmodel, s1_summary


def run_calibration(student: nn.Module, teacher: Optional[nn.Module],
                    loader, adapter_factory, exp: ExperimentConfig,
                    variant: str = "G", device: str = "cpu", seed: int = 0,
                    stage1_updates: Optional[int] = None,
                    stage2_updates: Optional[int] = None,
                    bandwidth: Optional[float] = None,
                    log_every: int = 100,
                    log: Optional[Callable[[str], None]] = None
                    ) -> CalibrationResult:
    """Run Algorithm 1 for one variant and return the hardened Q_theta."""
    log = log if log is not None else _no_log
    v = normalize_variant(variant)
    spec = VARIANT_SPECS[v]
    method = exp.method
    dev = torch.device(device)

    batches, cache, adapter_p, qmodel, s1_summary = _shared_setup(
        student, teacher, loader, adapter_factory, exp, dev, seed,
        stage1_updates, log)

    # Q_0 predictions with deployment (hard) rounding; frozen u^0 and g_Q.
    q0 = _predict_all(adapter_q := adapter_factory(qmodel), batches, dev,
                      soft=False)
    records = list(cache)
    compute_priorities(records, q0, method, exp.head_type, in_place=True)
    assign_variant_weights(records, v, method, perm_seed=seed)

    sigma = float(bandwidth if bandwidth is not None else exp.mmd_bandwidth)
    stage2_hist: Optional[List[Dict]] = None
    if spec.stage2:
        recon = SetReconstructor(adapter_q, adapter_p, cache, method, v,
                                 bandwidth=sigma, device=device, seed=seed,
                                 log=log)
        stage2_hist = recon.run(batches, updates=stage2_updates,
                                log_every=log_every)

    harden_all_quantizers(qmodel)
    qmodel.eval()
    stats = dict(cache.stats())
    stats["bandwidth"] = sigma
    stats["stage1_blocks"] = float(len(s1_summary))
    stats["stage2_updates"] = float(
        stage2_hist[-1]["step"] if stage2_hist else 0)
    return CalibrationResult(variant=v, qmodel=qmodel, spec=spec,
                             stage1=s1_summary, stage2=stage2_hist,
                             cache=cache, bandwidth=sigma, stats=stats)


def run_variant_table(student: nn.Module, teacher: Optional[nn.Module],
                      loader, adapter_factory, exp: ExperimentConfig,
                      variants: Optional[Sequence[str]] = None,
                      device: str = "cpu", seed: int = 0,
                      stage1_updates: Optional[int] = None,
                      stage2_updates: Optional[int] = None,
                      bandwidth: Optional[float] = None,
                      log_every: int = 100,
                      log: Optional[Callable[[str], None]] = None
                      ) -> Dict[str, CalibrationResult]:
    """Run the mechanism table: one Stage 1, shared Q_0, per-variant Stage 2.

    Variants B-I restart from the same snapshotted Q_0 with the same
    optimizer settings and update count; variant A stops after Stage 1.
    Returns one CalibrationResult per variant (hardened models included).
    """
    log = log if log is not None else _no_log
    table = [normalize_variant(v) for v in
             (variants if variants is not None else exp.variants)]
    method = exp.method
    dev = torch.device(device)

    batches, cache, adapter_p, qmodel, s1_summary = _shared_setup(
        student, teacher, loader, adapter_factory, exp, dev, seed,
        stage1_updates, log)

    # Shared snapshot: deepcopy preserves the soft-rounding flags (they are
    # plain attributes, not state_dict entries).
    base = copy.deepcopy(qmodel)
    q0 = _predict_all(adapter_factory(base), batches, dev, soft=False)
    records = list(cache)
    compute_priorities(records, q0, method, exp.head_type, in_place=True)

    sigma = float(bandwidth if bandwidth is not None else exp.mmd_bandwidth)
    results: Dict[str, CalibrationResult] = {}
    for v in table:
        spec = VARIANT_SPECS[v]
        q_v = copy.deepcopy(base)
        adapter_v = adapter_factory(q_v)
        records_v = copy.deepcopy(records)
        assign_variant_weights(records_v, v, method, perm_seed=seed)
        cache_v = TeacherCache(records_v)
        stage2_hist: Optional[List[Dict]] = None
        if spec.stage2:
            recon = SetReconstructor(adapter_v, adapter_p, cache_v, method,
                                     v, bandwidth=sigma, device=device,
                                     seed=seed, log=log)
            stage2_hist = recon.run(batches, updates=stage2_updates,
                                    log_every=log_every)
        harden_all_quantizers(q_v)
        q_v.eval()
        stats = dict(cache_v.stats())
        stats["bandwidth"] = sigma
        stats["stage1_blocks"] = float(len(s1_summary))
        stats["stage2_updates"] = float(
            stage2_hist[-1]["step"] if stage2_hist else 0)
        results[v] = CalibrationResult(
            variant=v, qmodel=q_v, spec=spec, stage1=s1_summary,
            stage2=stage2_hist, cache=cache_v, bandwidth=sigma, stats=stats)
        log(f"[calib] variant {v} done")
    return results
