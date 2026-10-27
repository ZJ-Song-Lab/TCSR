"""Diagnostics for one (checkpoint pair, seed): two-discrepancy gaps,
direction conflict, structure audit, assignment stability, and the
gap-stratified table (paper Sec. 5).

Real setting::

    python tools/diagnostics.py --config configs/hrsid_rtdetr.json \
        --seed 12101 --variant G

Smoke test (synthetic toy detectors)::

    python tools/diagnostics.py --config configs/hrsid_rtdetr.json --toy \
        --seed 12001 --stage1-updates 5 --stage2-updates 5

The tool runs Stage 1 + Stage 2 (variant G by default) once and writes

    outputs/{name}/{pair}/seed{seed:05d}/diagnostics.json

containing the calibration-split diagnostics (direction conflict, gap
statistics, structure audit), the anchor-assignment stability between the
frozen rho_0 and the assignment re-solved on the reconstructed Q_f, and
the 3 x 3 gap-stratified table with tertile edges frozen on the dev split
(val / train2017) and applied to the held-out split (test / val2017).
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (DEFAULT_EVAL_SPLIT, adapter_factory_for,  # noqa: E402
                    batches_to_loader, build_models, build_toy_pair,
                    load_checkpoint_manifest, open_split,
                    parse_variant_list, resolve_checkpoints,
                    resolve_output_dir, run_metadata, save_json, seed_dir,
                    set_global_seed, split_paths, synthetic_batches,
                    toy_loader)

from tcsr.calibration import (build_cache, predict_all,  # noqa: E402
                              prepare_calibration)
from tcsr.config import load_config  # noqa: E402
from tcsr.diagnostics import (assignment_change, collect_gaps,  # noqa: E402
                              direction_conflict, gap_statistics,
                              gap_stratified, structure_audit)
from tcsr.losses import pairwise_discrepancy  # noqa: E402
from tcsr.matching import solve_anchor_assignment  # noqa: E402
from tcsr.priority import compute_priorities  # noqa: E402
from tcsr.quant.fake_quant import harden_all_quantizers  # noqa: E402
from tcsr.stage2 import SetReconstructor  # noqa: E402
from tcsr.teacher import TeacherCache  # noqa: E402
from tcsr.variants import assign_variant_weights  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="experiment config JSON")
    p.add_argument("--seed", type=int, required=True,
                   help="run seed (must match tools/calibrate.py)")
    p.add_argument("--variant", default="G",
                   help="Stage-2 variant to reconstruct (default: G)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default=None,
                   help="output root (default: config output_dir)")
    p.add_argument("--stage1-updates", type=int, default=None,
                   help="override Stage-1 updates per block (smoke tests)")
    p.add_argument("--stage2-updates", type=int, default=None,
                   help="override Stage-2 updates (smoke tests)")
    p.add_argument("--log-every", type=int, default=100)
    # splits (defaults follow the paper: dev = val / train2017)
    p.add_argument("--dev-split", default=None,
                   help="dev split for the frozen tertile edges "
                        "(default: val for SAR, train2017 for COCO)")
    p.add_argument("--held-split", default=None,
                   help="held-out split (default: test / val2017)")
    p.add_argument("--dev-images", type=int, default=None,
                   help="dev images (default: config calib_images)")
    p.add_argument("--held-images", type=int, default=None,
                   help="held-out images (default: config calib_images)")
    p.add_argument("--batch-size", type=int, default=None)
    # real mode
    p.add_argument("--calib-split", default=None,
                   help="calibration split (default: train / train2017)")
    p.add_argument("--teacher", default=None, help="teacher checkpoint")
    p.add_argument("--student", default=None, help="student checkpoint")
    p.add_argument("--repo-path", default=None,
                   help="official Deformable DETR repo (defdetr only)")
    p.add_argument("--manifest", default=None,
                   help="checkpoint manifest (default: manifests/checkpoints.json)")
    # toy mode
    p.add_argument("--toy", action="store_true",
                   help="synthetic toy detectors (no checkpoints/datasets)")
    p.add_argument("--toy-images", type=int, default=16,
                   help="toy calibration images (default: 16)")
    return p.parse_args(argv)


def default_dev_split(dataset: str) -> str:
    """Paper protocol: dev = val for the SAR datasets, train2017 for COCO."""
    return "val" if dataset in ("hrsid", "ssdd") else "train2017"


def main(argv=None):
    args = parse_args(argv)
    exp = load_config(args.config)
    method = exp.method
    set_global_seed(args.seed)
    adapter_factory = adapter_factory_for(exp, toy=args.toy)
    batch_size = args.batch_size or method.batch_size
    variants = parse_variant_list(args.variant)
    variant = variants[0] if variants else "G"

    # ------------------------------------------------------------------
    # Models + the three data splits (calibration / dev / held-out).
    # ------------------------------------------------------------------
    if args.toy:
        mode = "toy"
        pair_label = "toy"
        teacher, student = build_toy_pair(exp, args.seed)
        teacher.to(args.device)
        student.to(args.device)
        calib_loader = toy_loader(exp, args.toy_images, args.seed)
        dev_loader = batches_to_loader(synthetic_batches(
            args.dev_images or exp.calib_images, 8, batch_size,
            args.seed + 2))
        held_loader = batches_to_loader(synthetic_batches(
            args.held_images or exp.calib_images, 8, batch_size,
            args.seed + 3))
        dev_split = held_split = None
        calib_split = "toy"
        gt_by_image = letterbox_by_image = None
        ckpts = {"teacher": "toy", "student": "toy", "repo_path": None}
    else:
        mode = "real"
        pair_label = exp.checkpoint_pair
        manifest = load_checkpoint_manifest(args.manifest)
        t_ckpt, s_ckpt, repo = resolve_checkpoints(
            exp, manifest, teacher=args.teacher, student=args.student,
            repo_path=args.repo_path)
        ckpts = {"teacher": t_ckpt, "student": s_ckpt, "repo_path": repo}
        teacher, student = build_models(exp, t_ckpt, s_ckpt, repo,
                                        args.device)
        calib_split = args.calib_split or (
            "train" if exp.dataset in ("hrsid", "ssdd") else "train2017")
        calib_loader = open_split(exp, calib_split, batch_size=batch_size)
        dev_split = args.dev_split or default_dev_split(exp.dataset)
        held_split = args.held_split or DEFAULT_EVAL_SPLIT[exp.dataset]
        dev_loader = open_split(exp, dev_split, batch_size=batch_size)
        held_loader = open_split(exp, held_split, batch_size=batch_size)

        from tcsr.datasets import ground_truth_by_image
        held_ann, _img = split_paths(exp, held_split)
        gt_by_image = {str(k): v for k, v in
                       ground_truth_by_image(held_ann).items()}
        # Letterbox parameters are dropped by collect_batches; re-collect
        # them from the raw held-out loader (str keys match record ids).
        letterbox_by_image = {}
        for batch in held_loader:
            for image_id, lb in zip(batch["image_ids"], batch["letterbox"]):
                letterbox_by_image[str(image_id)] = lb

    # ------------------------------------------------------------------
    # Stage 1 + the frozen Q_0 / cache / priorities on the calib split.
    # ------------------------------------------------------------------
    batches, cache, adapter_p, qmodel, s1_summary = prepare_calibration(
        student, teacher, calib_loader, adapter_factory, exp,
        device=args.device, seed=args.seed,
        stage1_updates=args.stage1_updates, log=print)
    adapter_q0 = adapter_factory(qmodel)
    q0_calib = predict_all(adapter_q0, batches, args.device, soft=False)
    records = list(cache)
    compute_priorities(records, q0_calib, method, exp.head_type,
                       in_place=True)

    calib_diag = {
        "cache": cache.stats(),
        "stage1_blocks": len(s1_summary),
        "direction_conflict": direction_conflict(
            records, q0_calib, method, exp.head_type),
        "gaps": gap_statistics(collect_gaps(records, q0_calib, method,
                                            exp.head_type)),
        "structure_audit": structure_audit(records),
    }

    # ------------------------------------------------------------------
    # Dev / held-out caches and their Q_0 predictions.
    # ------------------------------------------------------------------
    cache_dev, _a, batches_dev = build_cache(
        student, teacher, adapter_factory, dev_loader, exp,
        device=args.device, max_images=args.dev_images)
    cache_held, _a, batches_held = build_cache(
        student, teacher, adapter_factory, held_loader, exp,
        device=args.device, max_images=args.held_images)
    records_dev = list(cache_dev)
    records_held = list(cache_held)
    q0_dev = predict_all(adapter_q0, batches_dev, args.device, soft=False)
    q0_held = predict_all(adapter_q0, batches_held, args.device,
                          soft=False)
    compute_priorities(records_dev, q0_dev, method, exp.head_type,
                       in_place=True)
    compute_priorities(records_held, q0_held, method, exp.head_type,
                       in_place=True)

    # ------------------------------------------------------------------
    # Stage 2 (variant) from a deep copy of the shared Stage-1 state.
    # ------------------------------------------------------------------
    q_f = copy.deepcopy(qmodel)
    records_f = copy.deepcopy(records)
    assign_variant_weights(records_f, variant, method, perm_seed=args.seed)
    cache_f = TeacherCache(records_f)
    adapter_f = adapter_factory(q_f)
    recon = SetReconstructor(adapter_f, adapter_p, cache_f, method, variant,
                             bandwidth=exp.mmd_bandwidth,
                             device=args.device, seed=args.seed, log=print)
    history = recon.run(batches, updates=args.stage2_updates,
                        log_every=args.log_every)
    harden_all_quantizers(q_f)
    q_f.eval()
    qf_calib = predict_all(adapter_f, batches, args.device, soft=False)
    qf_held = predict_all(adapter_f, batches_held, args.device,
                          soft=False)

    # ------------------------------------------------------------------
    # Assignment stability: re-solve rho on the reconstructed Q_f.
    # ------------------------------------------------------------------
    rho_run = []
    for rec in records:
        d_f = pairwise_discrepancy(rec.anchor_predictions(),
                                   qf_calib[rec.image_id], method,
                                   exp.head_type)
        rho_run.append(solve_anchor_assignment(d_f))
    stability = assignment_change([rec.rho0 for rec in records], [rho_run])

    # ------------------------------------------------------------------
    # Gap-stratified table (edges frozen on dev, applied to held-out).
    # ------------------------------------------------------------------
    stratified = gap_stratified(
        records_dev, q0_dev, records_held, q0_held, qf_held,
        method, exp.head_type,
        gt_by_image=gt_by_image, letterbox_by_image=letterbox_by_image,
        input_size=exp.input_size)

    out = {
        "config": exp.name,
        "pair": pair_label,
        "seed": args.seed,
        "mode": mode,
        "variant": variant,
        "splits": {"calibration": calib_split,
                   "dev": dev_split, "held": held_split},
        "calibration": calib_diag,
        "dev": {"cache": cache_dev.stats()},
        "held": {"cache": cache_held.stats()},
        "stage2": {"updates": (args.stage2_updates
                               if args.stage2_updates is not None
                               else method.stage2_updates),
                   "final_loss": (history[-1].get("loss")
                                  if history else None)},
        "assignment_change": stability,
        "stratified": stratified,
        "provenance": run_metadata(
            sys.argv, mode=mode, seed=args.seed,
            extra={"config": args.config, "checkpoint_pair": pair_label,
                   "checkpoints": ckpts,
                   "stage1_updates": (args.stage1_updates
                                      if args.stage1_updates is not None
                                      else method.stage1_updates_per_block),
                   "stage2_updates": (args.stage2_updates
                                      if args.stage2_updates is not None
                                      else method.stage2_updates)}),
    }
    root = resolve_output_dir(exp, pair_label, args.output_dir)
    out_path = os.path.join(seed_dir(root, args.seed), "diagnostics.json")
    save_json(out_path, out)

    ov = stratified.get("overall", {})
    print(f"[diagnostics] pair={pair_label} seed={args.seed} "
          f"variant={variant} -> {out_path}")
    print(f"  valid anchors (calib): {int(calib_diag['structure_audit']['valid'])}"
          f"  assignment change: "
          f"{stability.get('per_transition_mean', float('nan')):.3f}")
    print(f"  held-out overall: n={int(ov.get('n', 0))} "
          f"mean_c={ov.get('mean_c', float('nan')):.3f} "
          f"mean_neg_delta_r={ov.get('mean_neg_delta_r', float('nan')):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
