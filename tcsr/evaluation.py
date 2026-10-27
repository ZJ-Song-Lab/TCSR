"""Pure-NumPy COCO-style bounding-box evaluation (AP / AR).

A dependency-free reimplementation of the ``pycocotools`` COCOeval
semantics used by the paper's reported endpoints (AP@[.50:.95], AP50,
AP75, AP_S, AR@100, all at ``maxDets = 100``).  The algorithm follows
``pycocotools`` exactly:

- ``bbox_iou_matrix`` mirrors ``bbIou``: absolute xywh coordinates, no
  ``+1`` pixel correction, ``w <= 0 | h <= 0 -> IoU 0`` and crowd ground
  truths use ``union = detection area``.
- ``evaluate_image`` mirrors ``evaluateImg``: a ground truth is ignored
  when it is a crowd, its (JSON ``area``) is strictly below or above the
  area range; ground truths are stable-sorted with ignored ones last;
  detections are stable-sorted by descending score and truncated at
  ``max_det``; greedy matching walks detections in score order, matches
  the best feasible ground truth at ``IoU >= t`` (ties resolved to the
  later ground-truth index), allows crowd ground truths to be matched by
  several detections, stops considering ignored ground truths once a
  regular (non-ignored) match is held, and finally flags unmatched
  detections whose area falls outside the range as ignored.
- ``COCOStyleEvaluator.accumulate`` mirrors ``accumulate``: pooled
  detection scores are stable-sorted descending per (category, area
  range) and the match / ignore columns are reordered with them;
  ``npig = 0`` cells stay at ``-1``; precision is the monotone
  right-to-left envelope of ``tp / (fp + tp + spacing(1))``; recall
  thresholds are located with ``searchsorted(..., side='left')`` and
  recall points beyond the last achieved recall keep precision ``0``
  (pycocotools initializes the row to zeros and lets the index error
  pass); ``recall = rc[-1]`` (``0`` when there are no detections).
- ``summarize`` averages every entry ``> -1``.

Because every endpoint reported in the paper uses ``maxDets = 100`` this
implementation exposes a single ``max_det`` (mathematically equivalent
to evaluating ``pycocotools`` at ``maxDets[-1] = 100``).

Area-range boundaries follow COCO: a box with area exactly 1024 counts
as both small and medium (the ignore test is strict ``<`` / ``>``).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "IOU_THRS", "REC_THRS", "AREA_RANGES", "DEFAULT_MAX_DET",
    "bbox_iou_matrix", "evaluate_image", "COCOStyleEvaluator",
    "predictions_to_detections", "evaluate_detections",
]

#: COCO IoU thresholds; AP50 -> index 0, AP75 -> index 5.
IOU_THRS: np.ndarray = np.linspace(0.5, 0.95, 10)
#: COCO recall thresholds (101 points).
REC_THRS: np.ndarray = np.linspace(0.0, 1.0, 101)
#: (label, (min_area, max_area)) pairs; boundaries belong to both bins.
AREA_RANGES: List[Tuple[str, Tuple[float, float]]] = [
    ("all", (0.0, 1e10)),
    ("small", (0.0, 1024.0)),
    ("medium", (1024.0, 9216.0)),
    ("large", (9216.0, 1e10)),
]
DEFAULT_MAX_DET = 100

_AREA_INDEX = {name: i for i, (name, _) in enumerate(AREA_RANGES)}


def bbox_iou_matrix(dt_boxes: np.ndarray, gt_boxes: np.ndarray,
                    gt_iscrowd: np.ndarray) -> np.ndarray:
    """IoU between detections and ground truths -> (D, G).

    All boxes are absolute xywh.  Follows ``bbIou``: no ``+1`` correction,
    non-positive overlap width/height yields 0, and crowd ground truths
    replace the union with the detection area.
    """
    dt_boxes = np.asarray(dt_boxes, dtype=np.float64).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    gt_iscrowd = np.asarray(gt_iscrowd, dtype=bool).reshape(-1)
    d = dt_boxes.shape[0]
    g = gt_boxes.shape[0]
    if d == 0 or g == 0:
        return np.zeros((d, g), dtype=np.float64)
    da = dt_boxes[:, 2] * dt_boxes[:, 3]                       # (D,)
    ga = gt_boxes[:, 2] * gt_boxes[:, 3]                       # (G,)
    dx1 = dt_boxes[:, 0]
    dy1 = dt_boxes[:, 1]
    dx2 = dx1 + dt_boxes[:, 2]
    dy2 = dy1 + dt_boxes[:, 3]
    gx1 = gt_boxes[:, 0]
    gy1 = gt_boxes[:, 1]
    gx2 = gx1 + gt_boxes[:, 2]
    gy2 = gy1 + gt_boxes[:, 3]
    w = np.minimum(dx2[:, None], gx2[None, :]) - np.maximum(dx1[:, None], gx1[None, :])
    h = np.minimum(dy2[:, None], gy2[None, :]) - np.maximum(dy1[:, None], gy1[None, :])
    inter = np.where((w > 0) & (h > 0), w * h, 0.0)
    union = da[:, None] + ga[None, :] - inter
    union = np.where(gt_iscrowd[None, :], da[:, None], union)
    return inter / np.maximum(union, np.spacing(1))


def evaluate_image(gt_boxes: np.ndarray, gt_areas: np.ndarray,
                   gt_iscrowd: np.ndarray, dt_boxes: np.ndarray,
                   dt_areas: np.ndarray, dt_scores: np.ndarray,
                   area_range: Tuple[float, float],
                   iou_thrs: np.ndarray = IOU_THRS,
                   max_det: int = DEFAULT_MAX_DET
                   ) -> Optional[Dict[str, np.ndarray]]:
    """Single (image, category, area-range) cell (``evaluateImg``).

    Returns ``None`` when there is no ground truth and no detection;
    otherwise a dict with ``dt_scores`` (D,), ``dt_matches`` (T, D) bool,
    ``dt_ignore`` (T, D) bool, ``gt_ignore`` (G,) bool and ``npig`` int.
    """
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    gt_areas = np.asarray(gt_areas, dtype=np.float64).reshape(-1)
    gt_iscrowd = np.asarray(gt_iscrowd, dtype=bool).reshape(-1)
    dt_boxes = np.asarray(dt_boxes, dtype=np.float64).reshape(-1, 4)
    dt_areas = np.asarray(dt_areas, dtype=np.float64).reshape(-1)
    dt_scores = np.asarray(dt_scores, dtype=np.float64).reshape(-1)
    g = gt_boxes.shape[0]
    d = dt_boxes.shape[0]
    if g == 0 and d == 0:
        return None

    a_lo, a_hi = area_range
    gt_ignore = gt_iscrowd | (gt_areas < a_lo) | (gt_areas > a_hi)

    # Stable sort: regular (non-ignored) ground truths first.
    if g:
        order = np.argsort(gt_ignore.astype(np.int8), kind="mergesort")
        gt_boxes = gt_boxes[order]
        gt_iscrowd = gt_iscrowd[order]
        gt_ignore = gt_ignore[order]
        n_reg = int((~gt_ignore).sum())
    else:
        n_reg = 0

    # Stable sort detections by descending score, truncate at max_det.
    if d:
        dord = np.argsort(-dt_scores, kind="mergesort")[:max_det]
        dt_boxes = dt_boxes[dord]
        dt_areas = dt_areas[dord]
        dt_scores = dt_scores[dord]
        d = dt_scores.shape[0]
        ious = bbox_iou_matrix(dt_boxes, gt_boxes, gt_iscrowd) if g else \
            np.zeros((d, 0))
    else:
        ious = np.zeros((0, g))

    t_n = len(iou_thrs)
    dt_matches = np.zeros((t_n, d), dtype=bool)
    dt_ignore = np.zeros((t_n, d), dtype=bool)

    for ti in range(t_n):
        thr = min(float(iou_thrs[ti]), 1.0 - 1e-10)
        reg_avail = np.ones(n_reg, dtype=bool)  # unmatched regular GTs
        for di in range(d):
            row = ious[di]
            m = -1
            if n_reg:
                cand = np.nonzero(reg_avail & (row[:n_reg] >= thr))[0]
                if cand.size:
                    m = int(cand[-1])          # ties -> later GT index
            if m < 0 and g > n_reg:
                # Ignored / crowd segment: only reachable when no regular
                # candidate was held (evaluateImg's break rule).
                cand = np.nonzero(row[n_reg:] >= thr)[0]
                if cand.size:
                    m = n_reg + int(cand[-1])
            if m < 0:
                continue
            dt_matches[ti, di] = True
            dt_ignore[ti, di] = bool(gt_ignore[m])
            if m < n_reg:
                reg_avail[m] = False           # one-to-one on regular GTs

    # Unmatched detections outside the area range are ignored.
    if d:
        out_of_range = (dt_areas < a_lo) | (dt_areas > a_hi)
        dt_ignore = dt_ignore | (~dt_matches & out_of_range[None, :])

    return {
        "dt_scores": dt_scores,
        "dt_matches": dt_matches,
        "dt_ignore": dt_ignore,
        "gt_ignore": gt_ignore,
        "npig": int((~gt_ignore).sum()) if g else 0,
    }


class COCOStyleEvaluator:
    """Accumulate COCO-style AP / AR metrics for bounding boxes.

    Ground truths are dicts with keys ``image_id``, ``category_id``,
    ``bbox`` (absolute xywh), ``area`` and optional ``iscrowd`` /
    ``ignore`` flags; detections are dicts with ``image_id``,
    ``category_id``, ``bbox`` and ``score``.  Category ids are the sorted
    union of ground-truth and detection categories.
    """

    def __init__(self, iou_thrs: np.ndarray = IOU_THRS,
                 rec_thrs: np.ndarray = REC_THRS,
                 area_ranges: Sequence[Tuple[str, Tuple[float, float]]] = AREA_RANGES,
                 max_det: int = DEFAULT_MAX_DET):
        self.iou_thrs = np.asarray(iou_thrs, dtype=np.float64)
        self.rec_thrs = np.asarray(rec_thrs, dtype=np.float64)
        self.area_ranges = list(area_ranges)
        self.max_det = int(max_det)
        self._gts: List[dict] = []
        self._dts: List[dict] = []
        self._evals: Dict[Tuple, Optional[dict]] = {}
        self._images: List = []
        self._cats: List = []
        self._precision: Optional[np.ndarray] = None
        self._recall: Optional[np.ndarray] = None

    # -- population ---------------------------------------------------

    def add_ground_truth(self, entries: Iterable[dict]) -> None:
        for e in entries:
            self._gts.append({
                "image_id": e["image_id"],
                "category_id": e["category_id"],
                "bbox": [float(v) for v in e["bbox"]],
                "area": float(e.get("area", e["bbox"][2] * e["bbox"][3])),
                "iscrowd": int(e.get("iscrowd", 0)),
                "ignore": int(e.get("ignore", 0)),
            })

    def add_detections(self, entries: Iterable[dict]) -> None:
        for e in entries:
            self._dts.append({
                "image_id": e["image_id"],
                "category_id": e["category_id"],
                "bbox": [float(v) for v in e["bbox"]],
                "score": float(e["score"]),
            })

    @property
    def categories(self) -> List:
        cats = {e["category_id"] for e in self._gts} | \
               {e["category_id"] for e in self._dts}
        return sorted(cats)

    @property
    def images(self) -> List:
        imgs = {e["image_id"] for e in self._gts} | \
               {e["image_id"] for e in self._dts}
        return sorted(imgs)

    # -- evaluation ---------------------------------------------------

    def evaluate(self) -> Dict[Tuple, Optional[dict]]:
        """Run ``evaluate_image`` on every (image, category) cell."""
        gts: Dict[Tuple, List[dict]] = {}
        for e in self._gts:
            gts.setdefault((e["image_id"], e["category_id"]), []).append(e)
        dts: Dict[Tuple, List[dict]] = {}
        for e in self._dts:
            dts.setdefault((e["image_id"], e["category_id"]), []).append(e)
        self._images = self.images
        self._cats = self.categories
        # The ignore masks depend on the area range, so evaluate_image is
        # re-run per area range; results are stored keyed by range name
        # and self._evals aliases the first range for backwards compat.
        per_range: Dict[Tuple, Dict[Tuple, Optional[dict]]] = {}
        for name, rng in self.area_ranges:
            table: Dict[Tuple, Optional[dict]] = {}
            for img in self._images:
                for cat in self._cats:
                    key = (img, cat)
                    g = gts.get(key, [])
                    dd = dts.get(key, [])
                    if not g and not dd:
                        table[key] = None
                        continue
                    table[key] = evaluate_image(
                        np.array([e["bbox"] for e in g], dtype=np.float64).reshape(-1, 4),
                        np.array([e["area"] for e in g], dtype=np.float64),
                        np.array([bool(e["iscrowd"] or e["ignore"]) for e in g]),
                        np.array([e["bbox"] for e in dd], dtype=np.float64).reshape(-1, 4),
                        np.array([e["bbox"][2] * e["bbox"][3] for e in dd], dtype=np.float64),
                        np.array([e["score"] for e in dd], dtype=np.float64),
                        rng, self.iou_thrs, self.max_det)
            per_range[name] = table
        self._evals = per_range[self.area_ranges[0][0]]  # backwards compat
        self._per_range = per_range
        return per_range

    def accumulate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute precision (T, R, K, A) and recall (T, K, A)."""
        if not hasattr(self, "_per_range"):
            self.evaluate()
        t_n = len(self.iou_thrs)
        r_n = len(self.rec_thrs)
        k_n = len(self._cats)
        a_n = len(self.area_ranges)
        precision = -np.ones((t_n, r_n, k_n, a_n), dtype=np.float64)
        recall = -np.ones((t_n, k_n, a_n), dtype=np.float64)
        for ai, (name, _) in enumerate(self.area_ranges):
            table = self._per_range[name]
            for ki, cat in enumerate(self._cats):
                e_list = [table[(img, cat)] for img in self._images
                          if table.get((img, cat)) is not None]
                if not e_list:
                    continue
                dt_scores = np.concatenate([e["dt_scores"] for e in e_list])
                # Stable mergesort: consistent with the Matlab implementation.
                inds = np.argsort(-dt_scores, kind="mergesort")
                dtm = np.concatenate(
                    [e["dt_matches"] for e in e_list], axis=1)[:, inds]
                dt_ig = np.concatenate(
                    [e["dt_ignore"] for e in e_list], axis=1)[:, inds]
                gt_ig = np.concatenate([e["gt_ignore"] for e in e_list])
                npig = int(np.count_nonzero(gt_ig == 0))
                if npig == 0:
                    continue
                tps = np.logical_and(dtm, np.logical_not(dt_ig))
                fps = np.logical_and(np.logical_not(dtm), np.logical_not(dt_ig))
                tp_sum = np.cumsum(tps, axis=1).astype(np.float64)
                fp_sum = np.cumsum(fps, axis=1).astype(np.float64)
                for t in range(t_n):
                    tp = tp_sum[t]
                    fp = fp_sum[t]
                    nd = len(tp)
                    rc = tp / npig
                    pr = tp / (fp + tp + np.spacing(1))
                    recall[t, ki, ai] = rc[-1] if nd else 0.0
                    # Precision envelope (right to left).
                    for i in range(nd - 1, 0, -1):
                        if pr[i] > pr[i - 1]:
                            pr[i - 1] = pr[i]
                    q = np.zeros(r_n, dtype=np.float64)
                    if nd:
                        r_inds = np.searchsorted(rc, self.rec_thrs, side="left")
                        ok = r_inds < nd
                        q[ok] = pr[r_inds[ok]]
                    precision[t, :, ki, ai] = q
        self._precision = precision
        self._recall = recall
        return precision, recall

    def summarize(self) -> Dict[str, float]:
        """COCO endpoints; entries with no valid cells report ``-1.0``."""
        if self._precision is None:
            self.accumulate()

        def _mean(x: np.ndarray) -> float:
            x = x[x > -1]
            return float(x.mean()) if x.size else -1.0

        p = self._precision
        r = self._recall
        a_all = _AREA_INDEX.get(self.area_ranges[0][0], 0)
        a_s = _AREA_INDEX.get("small", 1)
        a_m = _AREA_INDEX.get("medium", 2)
        a_l = _AREA_INDEX.get("large", 3)
        # Resolve indices by label so custom area ranges still work.
        labels = [name for name, _ in self.area_ranges]
        a_all = labels.index(self.area_ranges[0][0]) if self.area_ranges else 0
        a_s = labels.index("small") if "small" in labels else a_all
        a_m = labels.index("medium") if "medium" in labels else a_all
        a_l = labels.index("large") if "large" in labels else a_all
        return {
            "AP": _mean(p[:, :, :, a_all]),
            "AP50": _mean(p[0, :, :, a_all]),
            "AP75": _mean(p[5, :, :, a_all]),
            "AP_S": _mean(p[:, :, :, a_s]),
            "AP_M": _mean(p[:, :, :, a_m]),
            "AP_L": _mean(p[:, :, :, a_l]),
            "AR@100": _mean(r[:, :, a_all]),
        }

    def run(self) -> Dict[str, float]:
        """evaluate + accumulate + summarize in one call."""
        self.evaluate()
        self.accumulate()
        return self.summarize()


def predictions_to_detections(predictions, image_ids: Sequence,
                              image_sizes: Optional[Sequence] = None,
                              category_ids: Optional[Sequence] = None,
                              head_type: str = "softmax",
                              score_threshold: float = 0.0,
                              max_det_per_image: int = DEFAULT_MAX_DET,
                              letterbox: Optional[Sequence] = None,
                              target_size: Optional[int] = None
                              ) -> List[dict]:
    """Convert model predictions to COCO detection entries.

    ``predictions`` is a sequence of :class:`tcsr.losses.Prediction`
    (one per image) whose ``boxes`` are normalized cxcywh.  Scores are
    the maximum foreground probability (softmax head: background is the
    last channel; sigmoid head: all channels).  Boxes are mapped back to
    absolute xywh in the *original* image frame: when ``letterbox`` is
    given (per-image ``(scale, pad_left, pad_top)`` together with
    ``target_size``), the normalized coordinates are first scaled to the
    letterboxed square, de-padded and divided by ``scale``; otherwise
    they are scaled directly by ``image_sizes`` (per-image ``(w, h)``).
    ``category_ids`` maps foreground channel indices to COCO category
    ids and defaults to ``channel + 1``.
    """
    import torch

    out: List[dict] = []
    for i, pred in enumerate(predictions):
        logits = pred.logits
        boxes = pred.boxes
        if logits.dim() != 2 or boxes.dim() != 2:
            raise ValueError("Prediction logits/boxes must be 2-D")
        n = logits.shape[0]
        if n == 0:
            continue
        if head_type == "softmax":
            probs = torch.softmax(logits, dim=-1)[:, :-1]
        elif head_type == "sigmoid":
            probs = torch.sigmoid(logits)
        else:
            raise ValueError(f"unknown head_type {head_type!r}")
        scores, channels = probs.max(dim=-1)
        keep = scores >= score_threshold
        if not bool(keep.any()):
            continue
        scores = scores[keep]
        channels = channels[keep]
        b = boxes[keep]
        order = torch.argsort(scores, descending=True)
        if max_det_per_image is not None and order.numel() > max_det_per_image:
            order = order[:max_det_per_image]
        scores = scores[order].detach().cpu().tolist()
        channels = channels[order].detach().cpu().tolist()
        b = b[order].detach().cpu().numpy()

        if category_ids is None:
            cat_of = lambda c: int(c) + 1
        else:
            cat_of = lambda c: int(category_ids[int(c)])

        if letterbox is not None:
            if target_size is None:
                raise ValueError("target_size required with letterbox")
            scale, pad_x, pad_y = letterbox[i][:3]
            cx = (b[:, 0] * target_size - pad_x) / scale
            cy = (b[:, 1] * target_size - pad_y) / scale
            w = b[:, 2] * target_size / scale
            h = b[:, 3] * target_size / scale
        else:
            if image_sizes is None:
                raise ValueError("image_sizes required without letterbox")
            w_img, h_img = image_sizes[i][:2]
            cx = b[:, 0] * w_img
            cy = b[:, 1] * h_img
            w = b[:, 2] * w_img
            h = b[:, 3] * h_img
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        image_id = image_ids[i]
        for r in range(len(scores)):
            out.append({
                "image_id": image_id,
                "category_id": cat_of(channels[r]),
                "bbox": [float(x1[r]), float(y1[r]), float(w[r]), float(h[r])],
                "score": float(scores[r]),
            })
    return out


def evaluate_detections(detections: Iterable[dict],
                        ground_truths: Iterable[dict], **kwargs) -> Dict[str, float]:
    """One-shot COCO-style evaluation of detection entries vs GT entries."""
    ev = COCOStyleEvaluator(
        iou_thrs=kwargs.pop("iou_thrs", IOU_THRS),
        rec_thrs=kwargs.pop("rec_thrs", REC_THRS),
        area_ranges=kwargs.pop("area_ranges", AREA_RANGES),
        max_det=kwargs.pop("max_det", DEFAULT_MAX_DET),
    )
    ev.add_ground_truth(ground_truths)
    ev.add_detections(detections)
    return ev.run()
