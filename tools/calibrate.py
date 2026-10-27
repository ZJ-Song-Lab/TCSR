"""Calibrate one (checkpoint pair, seed): Stage 1 + the variant table.

Real setting (weights from manifests/checkpoints.json or CLI overrides)::

    python tools/calibrate.py --config configs/hrsid_rtdetr.json --seed 12101

Smoke test (synthetic toy detectors, no downloads)::

    python tools/calibrate.py --config configs/hrsid_rtdetr.json --toy \
        --seed 12001 --stage1-updates 5 --stage2-updates 5

Outputs (per seed):

    outputs/{name}/{pair}/seed{seed:05d}/config.json
    outputs/{name}/{pair}/seed{seed:05d}/run.json
    outputs/{name}/{pair}/seed{seed:05d}/variant_{V}/qmodel.pt          hardened Q (whole module)
    outputs/{name}/{pair}/seed{seed:05d}/variant_{V}/stats.json
    outputs/{name}/{pair}/seed{seed:05d}/variant_{V}/stage1.json
    outputs/{name}/{pair}/seed{seed:05d}/variant_{V}/stage2_history.json  (Stage-2 variants)

The paper's protocol runs the full variant table per seed; pass
``--variants G`` to calibrate a single variant.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (DEFAULT_CALIB_SPLIT, adapter_factory_for,  # noqa: E402
                    build_models, build_toy_pair, load_checkpoint_manifest,
                    parse_variant_list, resolve_checkpoints, resolve_output_dir,
                    save_json, seed_dir, set_global_seed, toy_loader)

from tcsr.calibration import run_variant_table  # noqa: E402
from tcsr.config import load_config  # noqa: E402
from tcsr.datasets import make_loader  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="experiment config JSON")
    p.add_argument("--seed", type=int, required=True,
                   help="run seed (paper: 12101-12120 for HR-02, ...)")
    p.add_argument("--variants", default=None,
                   help="comma-separated variants (default: config table)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default=None,
                   help="output root (default: config output_dir)")
    p.add_argument("--stage1-updates", type=int, default=None,
                   help="override Stage-1 updates per block (smoke tests)")
    p.add_argument("--stage2-updates", type=int, default=None,
                   help="override Stage-2 updates (smoke tests)")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=None)
    # real mode
    p.add_argument("--calib-split", default=None,
                   help="calibration split (default: train / train2017)")
    p.add_argument("--calib-manifest", default=None,
                   help="per-draw calibration manifest (JSON image-id "
                        "list, see tools/calibration_manifest.py)")
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


def main(argv=None):
    args = parse_args(argv)
    exp = load_config(args.config)
    set_global_seed(args.seed)

    variants = parse_variant_list(args.variants)
    batch_size = args.batch_size or exp.method.batch_size

    if args.toy:
        pair_label = "toy"
        teacher, student = build_toy_pair(exp, args.seed)
        teacher.to(args.device)
        student.to(args.device)
        loader = toy_loader(exp, args.toy_images, args.seed)
        calib_split = "toy"
        ckpts = {"teacher": "toy", "student": "toy", "repo_path": None}
    else:
        pair_label = exp.checkpoint_pair
        manifest = load_checkpoint_manifest(args.manifest)
        t_ckpt, s_ckpt, repo = resolve_checkpoints(
            exp, manifest, teacher=args.teacher, student=args.student,
            repo_path=args.repo_path)
        teacher, student = build_models(exp, t_ckpt, s_ckpt, repo,
                                        args.device)
        calib_split = args.calib_split or DEFAULT_CALIB_SPLIT[exp.dataset]
        from common import load_image_id_manifest, open_split
        calib_ids = load_image_id_manifest(args.calib_manifest)
        if args.calib_manifest and calib_ids is None:
            raise FileNotFoundError(
                f"calibration manifest not found: {args.calib_manifest}")
        loader = open_split(exp, calib_split, batch_size=batch_size,
                            image_ids=calib_ids)
        ckpts = {"teacher": t_ckpt, "student": s_ckpt, "repo_path": repo}

    adapter_factory = adapter_factory_for(exp, toy=args.toy)

    t0 = time.time()
    results = run_variant_table(
        student, teacher, loader, adapter_factory, exp,
        variants=variants, device=args.device, seed=args.seed,
        stage1_updates=args.stage1_updates,
        stage2_updates=args.stage2_updates,
        log_every=args.log_every, log=print)
    elapsed = time.time() - t0

    out_root = resolve_output_dir(exp, pair_label, args.output_dir)
    sdir = seed_dir(out_root, args.seed)
    os.makedirs(sdir, exist_ok=True)
    exp.save(os.path.join(sdir, "config.json"))

    from common import run_metadata
    save_json(os.path.join(sdir, "run.json"), run_metadata(
        sys.argv, mode="toy" if args.toy else "real", seed=args.seed,
        extra={
            "config": args.config,
            "checkpoint_pair": pair_label,
            "calib_split": calib_split,
            "calib_manifest": (None if args.toy else args.calib_manifest),
            "calib_images": exp.calib_images,
            "batch_size": batch_size,
            "variants": sorted(results),
            "stage1_updates": (args.stage1_updates
                               if args.stage1_updates is not None
                               else exp.method.stage1_updates_per_block),
            "stage2_updates": (args.stage2_updates
                               if args.stage2_updates is not None
                               else exp.method.stage2_updates),
            "quant_profile": exp.quant_profile,
            "checkpoints": ckpts,
            "elapsed_seconds": elapsed,
        }))

    for v, res in results.items():
        vdir = os.path.join(sdir, f"variant_{v}")
        os.makedirs(vdir, exist_ok=True)
        # Whole hardened module: the hard/soft rounding flags are plain
        # attributes and survive torch.save / torch.load.
        torch.save(res.qmodel, os.path.join(vdir, "qmodel.pt"))
        save_json(os.path.join(vdir, "stats.json"), {
            "variant": v,
            "stats": res.stats,
            "bandwidth": res.bandwidth,
            "spec": {"stage2": res.spec.stage2,
                     "set_losses": res.spec.set_losses,
                     "teacher_term": res.spec.teacher_term,
                     "teacher_targets": res.spec.teacher_targets},
        })
        save_json(os.path.join(vdir, "stage1.json"), res.stage1)
        if res.stage2 is not None:
            save_json(os.path.join(vdir, "stage2_history.json"), res.stage2)

    print(f"[calibrate] wrote {len(results)} variants to {sdir} "
          f"({elapsed:.1f}s)")
    for v in sorted(results):
        s = results[v].stats
        print(f"  variant {v:>3}: anchors={int(s.get('anchors', 0))} "
              f"corroborated={s.get('matched_coverage', 0.0):.3f} "
              f"stage2_updates={int(s.get('stage2_updates', 0))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
