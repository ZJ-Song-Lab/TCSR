"""Evaluate AP of calibrated / full-precision models on a held-out split.

Real setting::

    python tools/evaluate.py --config configs/hrsid_rtdetr.json \
        --seed 12101 --model fp:student \
        --model outputs/hrsid_rtdetr/HR-02/seed12101/variant_G/qmodel.pt

``--model`` accepts a saved ``qmodel.pt`` (from tools/calibrate.py) or
``fp:student`` / ``fp:teacher`` for the full-precision baselines and may
be given multiple times.

Smoke test (synthetic toy detectors; the ground truth is derived from
the full-precision student's own high-confidence predictions)::

    python tools/evaluate.py --config configs/hrsid_rtdetr.json --toy \
        --seed 12001 --model fp:student \
        --model outputs/hrsid_rtdetr/toy/seed12001/variant_G/qmodel.pt

Outputs (next to each model; consumed by tools/method_comparison.py):

    .../variant_G/eval_{split}.json          quantized model
    .../seed{seed}/eval_{split}_fp_student.json   FP baselines
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (DEFAULT_EVAL_SPLIT, TOY_EVAL_SIZE, TOY_INPUT_SIZE,  # noqa: E402
                    adapter_factory_for, build_models, build_toy_pair,
                    category_ids_from_annotation, load_checkpoint_manifest,
                    model_label, predict_loader, resolve_checkpoints,
                    resolve_output_dir, save_json, seed_dir, set_global_seed,
                    split_paths, synthetic_batches)

from tcsr.config import load_config  # noqa: E402
from tcsr.datasets import load_ground_truth  # noqa: E402
from tcsr.evaluation import (evaluate_detections,  # noqa: E402
                             predictions_to_detections)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="experiment config JSON")
    p.add_argument("--seed", type=int, required=True,
                   help="run seed (must match tools/calibrate.py)")
    p.add_argument("--model", action="append", required=True,
                   help="qmodel.pt path, or fp:student / fp:teacher "
                        "(repeatable)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default=None,
                   help="output root for the FP baselines "
                        "(default: config output_dir)")
    p.add_argument("--eval-split", default=None,
                   help="evaluation split (default: test / val2017)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-images", type=int, default=None,
                   help="cap the number of evaluated images (smoke tests)")
    p.add_argument("--max-det", type=int, default=100,
                   help="max detections per image (COCO default 100)")
    # real mode
    p.add_argument("--teacher", default=None, help="teacher checkpoint")
    p.add_argument("--student", default=None, help="student checkpoint")
    p.add_argument("--repo-path", default=None,
                   help="official Deformable DETR repo (defdetr only)")
    p.add_argument("--manifest", default=None,
                   help="checkpoint manifest (default: manifests/checkpoints.json)")
    # toy mode
    p.add_argument("--toy", action="store_true",
                   help="synthetic toy detectors (no checkpoints/datasets)")
    p.add_argument("--toy-images", type=int, default=32,
                   help="toy evaluation images (default: 32)")
    return p.parse_args(argv)


def load_qmodel(path: str, device: str) -> torch.nn.Module:
    """Load a hardened qmodel.pt saved by tools/calibrate.py."""
    try:
        model = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch without the weights_only kwarg
        model = torch.load(path, map_location="cpu")
    model.to(device)
    model.eval()
    return model


def evaluate_real(exp, adapter, gt, cat_ids, eval_split, args):
    """AP over one real split (letterbox-aware absolute coordinates)."""
    from common import open_split
    loader = open_split(exp, eval_split, batch_size=args.batch_size)
    ids, preds, letterboxes = predict_loader(adapter, loader, args.device)
    if args.max_images is not None:
        ids = ids[:args.max_images]
        preds = preds[:args.max_images]
        keep = set(ids)
        gt = [g for g in gt if g["image_id"] in keep]
    detections = predictions_to_detections(
        preds, ids, category_ids=cat_ids, head_type=exp.head_type,
        score_threshold=0.0, max_det_per_image=args.max_det,
        letterbox=[letterboxes[i] for i in ids], target_size=exp.input_size)
    return evaluate_detections(detections, gt), len(ids)


def evaluate_toy(exp, adapter, student_adapter, args):
    """AP over synthetic images against the FP student's own predictions.

    The evaluation batches use ``seed + 1`` (distinct from the calibration
    batches of tools/calibrate.py) and are mapped onto a virtual
    ``TOY_EVAL_SIZE`` square so that the absolute-coordinate AP has
    non-degenerate object sizes.
    """
    bs = args.batch_size or exp.method.batch_size
    batches = synthetic_batches(args.toy_images, TOY_INPUT_SIZE, bs,
                                args.seed + 1)
    ids, preds = [], []
    fp_ids, fp_preds = [], []
    model = adapter.model
    model.eval()
    with torch.no_grad():
        for images, idl in batches:
            images = images.to(args.device)
            preds.extend(adapter.predict(images))
            ids.extend(idl)
            fp_preds.extend(student_adapter.predict(images))
            fp_ids.extend(idl)
    image_sizes = [(TOY_EVAL_SIZE, TOY_EVAL_SIZE)] * len(ids)
    gt_dets = predictions_to_detections(
        fp_preds, fp_ids, head_type=exp.head_type,
        score_threshold=0.5, max_det_per_image=10, image_sizes=image_sizes)
    gt = [{"image_id": d["image_id"], "category_id": d["category_id"],
           "bbox": d["bbox"], "area": d["bbox"][2] * d["bbox"][3],
           "iscrowd": 0} for d in gt_dets]
    detections = predictions_to_detections(
        preds, ids, head_type=exp.head_type, score_threshold=0.0,
        max_det_per_image=args.max_det, image_sizes=image_sizes)
    return evaluate_detections(detections, gt), len(ids)


def main(argv=None):
    args = parse_args(argv)
    exp = load_config(args.config)
    set_global_seed(args.seed)
    adapter_factory = adapter_factory_for(exp, toy=args.toy)

    fp_models = {}
    if args.toy:
        mode = "toy"
        pair_label = "toy"
        teacher, student = build_toy_pair(exp, args.seed)
        teacher.to(args.device)
        student.to(args.device)
        fp_models = {"teacher": teacher, "student": student}
        eval_split = "toy_eval"
        gt = cat_ids = ckpts = None
    else:
        mode = "real"
        pair_label = exp.checkpoint_pair
        manifest = load_checkpoint_manifest(args.manifest)
        t_ckpt, s_ckpt, repo = resolve_checkpoints(
            exp, manifest, teacher=args.teacher, student=args.student,
            repo_path=args.repo_path)
        ckpts = {"teacher": t_ckpt, "student": s_ckpt, "repo_path": repo}
        if any(m.startswith("fp:") for m in args.model):
            teacher, student = build_models(exp, t_ckpt, s_ckpt, repo,
                                            args.device)
            fp_models = {"teacher": teacher, "student": student}
        eval_split = args.eval_split or DEFAULT_EVAL_SPLIT[exp.dataset]
        ann, _img = split_paths(exp, eval_split)
        gt = load_ground_truth(ann)
        cat_ids = category_ids_from_annotation(ann, exp.num_classes)

    for spec in args.model:
        if spec.startswith("fp:"):
            which = spec.split(":", 1)[1]
            if which not in ("student", "teacher"):
                raise SystemExit(
                    f"--model fp:{which}: expected fp:student or fp:teacher")
            model = fp_models[which]
            method = f"fp_{which}"
            root = resolve_output_dir(exp, pair_label, args.output_dir)
            out_path = os.path.join(
                seed_dir(root, args.seed), f"eval_{eval_split}_{method}.json")
        else:
            model = load_qmodel(spec, args.device)
            method = model_label(spec)
            out_path = os.path.join(os.path.dirname(os.path.abspath(spec)),
                                    f"eval_{eval_split}.json")
        model.eval()
        adapter = adapter_factory(model)

        if args.toy:
            student_adapter = adapter_factory(fp_models["student"])
            metrics, n_eval = evaluate_toy(exp, adapter, student_adapter,
                                           args)
        else:
            metrics, n_eval = evaluate_real(exp, adapter, gt, cat_ids,
                                            eval_split, args)

        record = {
            "config": exp.name,
            "pair": pair_label,
            "seed": args.seed,
            "mode": mode,
            "method": method,
            "model": spec,
            "quant_profile": ("fp" if method.startswith("fp_")
                              else exp.quant_profile),
            "split": eval_split,
            "n_images": n_eval,
            "input_size": exp.input_size,
            "head_type": exp.head_type,
            "metrics": metrics,
        }
        save_json(out_path, record)

        ap = metrics.get("AP", float("nan"))
        ap50 = metrics.get("AP50", float("nan"))
        print(f"[evaluate] pair={pair_label} seed={args.seed} "
              f"method={method} AP={100 * ap:.2f} AP50={100 * ap50:.2f} "
              f"(n={n_eval}) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
