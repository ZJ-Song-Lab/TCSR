"""COCO-style evaluation tests (IoU, AP/AR, prediction conversion)."""

import numpy as np
import pytest
import torch

from tcsr.evaluation import (AREA_RANGES, IOU_THRS, REC_THRS,
                             COCOStyleEvaluator, bbox_iou_matrix,
                             evaluate_detections, evaluate_image,
                             predictions_to_detections)
from tcsr.losses import Prediction


class TestIou:
    def test_identical(self):
        iou = bbox_iou_matrix(np.array([[10., 10., 20., 20.]]),
                              np.array([[10., 10., 20., 20.]]),
                              np.array([False]))
        assert iou[0, 0] == pytest.approx(1.0)

    def test_disjoint(self):
        iou = bbox_iou_matrix(np.array([[0., 0., 10., 10.]]),
                              np.array([[50., 50., 10., 10.]]),
                              np.array([False]))
        assert iou[0, 0] == 0.0

    def test_half_overlap(self):
        iou = bbox_iou_matrix(np.array([[0., 0., 10., 10.]]),
                              np.array([[5., 0., 10., 10.]]),
                              np.array([False]))
        # inter 50, union 150
        assert iou[0, 0] == pytest.approx(1.0 / 3.0)

    def test_crowd_uses_detection_area(self):
        iou = bbox_iou_matrix(np.array([[0., 0., 10., 10.]]),
                              np.array([[0., 0., 100., 100.]]),
                              np.array([True]))
        # union replaced by the detection area -> 100/100
        assert iou[0, 0] == pytest.approx(1.0)

    def test_shapes_and_empty(self):
        m = bbox_iou_matrix(np.zeros((0, 4)), np.zeros((3, 4)),
                            np.zeros(3, dtype=bool))
        assert m.shape == (0, 3)


class TestEvaluateImage:
    def test_perfect_match(self):
        gt = np.array([[10., 10., 20., 20.]])
        res = evaluate_image(gt, gt[:, 2] * gt[:, 3], np.zeros(1, dtype=bool),
                             gt, gt[:, 2] * gt[:, 3], np.array([0.9]),
                             (0.0, 1e10))
        assert res["npig"] == 1
        assert bool(res["dt_matches"][5].all())   # IoU 1 >= 0.75

    def test_empty_cell(self):
        res = evaluate_image(np.zeros((0, 4)), np.zeros(0),
                             np.zeros(0, dtype=bool), np.zeros((0, 4)),
                             np.zeros(0), np.zeros(0), (0.0, 1e10))
        assert res is None


class TestEvaluator:
    def _populate(self, ev, images, n_det=1):
        for img, boxes in images.items():
            for b in boxes:
                ev.add_ground_truth([{"image_id": img, "category_id": 1,
                                      "bbox": b}])
                ev.add_detections([{"image_id": img, "category_id": 1,
                                    "bbox": b, "score": 0.9}])

    def test_perfect_predictions_ap_one(self):
        ev = COCOStyleEvaluator()
        boxes = {1: [[10., 10., 50., 50.], [100., 100., 40., 40.]],
                 2: [[30., 30., 60., 60.]]}
        self._populate(ev, boxes)
        s = ev.run()
        assert s["AP"] == pytest.approx(1.0)
        assert s["AP50"] == pytest.approx(1.0)
        assert s["AP75"] == pytest.approx(1.0)
        assert s["AR@100"] == pytest.approx(1.0)

    def test_wrong_boxes_ap_zero(self):
        ev = COCOStyleEvaluator()
        ev.add_ground_truth([{"image_id": 1, "category_id": 1,
                              "bbox": [10., 10., 50., 50.]}])
        ev.add_detections([{"image_id": 1, "category_id": 1,
                            "bbox": [500., 500., 50., 50.], "score": 0.9}])
        s = ev.run()
        assert s["AP"] == pytest.approx(0.0)

    def test_thresholds_layout(self):
        assert len(IOU_THRS) == 10
        assert IOU_THRS[0] == pytest.approx(0.5)
        assert IOU_THRS[5] == pytest.approx(0.75)
        assert len(REC_THRS) == 101
        assert [n for n, _ in AREA_RANGES] == ["all", "small", "medium",
                                               "large"]

    def test_evaluate_detections_one_shot(self):
        gts = [{"image_id": 1, "category_id": 1, "bbox": [10., 10., 50., 50.]}]
        dts = [{"image_id": 1, "category_id": 1,
                "bbox": [10., 10., 50., 50.], "score": 0.9}]
        s = evaluate_detections(dts, gts)
        assert s["AP"] == pytest.approx(1.0)

    def test_max_det_truncation(self):
        ev = COCOStyleEvaluator(max_det=2)
        ev.add_ground_truth([{"image_id": 1, "category_id": 1,
                              "bbox": [10., 10., 10., 10.]}])
        for i in range(5):
            ev.add_detections([{"image_id": 1, "category_id": 1,
                                "bbox": [10., 10., 10., 10.],
                                "score": 0.9 - 0.1 * i}])
        s = ev.run()
        # only the top-2 detections are kept; both match the single GT
        assert s["AP"] == pytest.approx(1.0)


class TestPredictionsToDetections:
    def test_basic_conversion(self):
        logits = torch.tensor([[5.0, -5.0], [-5.0, 5.0]])
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4],
                              [0.25, 0.75, 0.1, 0.1]])
        pred = Prediction(logits=logits, boxes=boxes)
        dets = predictions_to_detections([pred], [1],
                                         image_sizes=[(100, 200)],
                                         head_type="softmax")
        assert len(dets) == 2
        # softmax: fg = channel 0 for both rows
        assert dets[0]["image_id"] == 1 and dets[1]["image_id"] == 1
        assert dets[0]["category_id"] == 1
        # box 0: center (50, 100), size (20, 80) in a 100x200 frame
        assert dets[0]["bbox"] == pytest.approx([40.0, 60.0, 20.0, 80.0])
        # box 1: center (25, 150), size (10, 20)
        assert dets[1]["bbox"] == pytest.approx([20.0, 140.0, 10.0, 20.0])

    def test_letterbox_mapping(self):
        pred = Prediction(logits=torch.tensor([[5.0, -5.0]]),
                          boxes=torch.tensor([[0.5, 0.5, 0.2, 0.2]]))
        dets = predictions_to_detections(
            [pred], [7], letterbox=[(0.5, 0.0, 0.0)], target_size=640,
            head_type="softmax")
        # cx = 0.5*640/0.5 = 640, cy = 640, w = h = 0.2*640/0.5 = 256
        assert dets[0]["bbox"] == pytest.approx([512.0, 512.0, 256.0, 256.0])

    def test_letterbox_requires_target_size(self):
        pred = Prediction(logits=torch.tensor([[5.0, -5.0]]),
                          boxes=torch.tensor([[0.5, 0.5, 0.2, 0.2]]))
        with pytest.raises(ValueError):
            predictions_to_detections([pred], [1], letterbox=[(0.5, 0, 0)],
                                      head_type="softmax")

    def test_requires_image_sizes(self):
        pred = Prediction(logits=torch.tensor([[5.0, -5.0]]),
                          boxes=torch.tensor([[0.5, 0.5, 0.2, 0.2]]))
        with pytest.raises(ValueError):
            predictions_to_detections([pred], [1], head_type="softmax")

    def test_score_threshold_and_max_det(self):
        logits = torch.tensor([[5.0, -5.0], [1.0, -1.0]])
        boxes = torch.rand(2, 4)
        pred = Prediction(logits=logits, boxes=boxes)
        dets = predictions_to_detections([pred], [1],
                                         image_sizes=[(100, 100)],
                                         head_type="softmax",
                                         score_threshold=0.5)
        # only the first row survives (softmax fg prob of row 2 ~ 0.73... )
        scores = torch.softmax(logits, -1)[:, 0]
        assert len(dets) == int((scores >= 0.5).sum())

    def test_sigmoid_head(self):
        pred = Prediction(logits=torch.tensor([[3.0, -3.0]]),
                          boxes=torch.tensor([[0.5, 0.5, 0.5, 0.5]]))
        dets = predictions_to_detections([pred], [1],
                                         image_sizes=[(10, 10)],
                                         head_type="sigmoid")
        assert dets[0]["category_id"] == 1
        assert dets[0]["score"] == pytest.approx(float(torch.sigmoid(torch.tensor(3.0))))

    def test_empty_prediction(self):
        pred = Prediction(logits=torch.zeros(0, 2), boxes=torch.zeros(0, 4))
        assert predictions_to_detections([pred], [1],
                                         image_sizes=[(10, 10)]) == []
