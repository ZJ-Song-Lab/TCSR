"""Toy model and adapter protocol tests."""

import pytest
import torch
import torch.nn as nn

from tcsr.models import get_adapter_factory
from tcsr.models.base import (DetectorAdapter, check_logits_channels,
                              validate_block_execution)
from tcsr.models.toy import (ToyAdapter, ToyDetector, make_toy_detector)


class TestToyDetector:
    def test_forward_shape(self, toy_detector):
        out = toy_detector(torch.randn(2, 3, 8, 8))
        # softmax head with num_classes=1 -> 2 logits + 4 box params
        assert out.shape == (2, 64, 6)
        assert toy_detector.num_queries == 64

    def test_predict(self, toy_detector):
        preds = toy_detector.predict(torch.randn(2, 3, 8, 8))
        assert len(preds) == 2
        assert preds[0].logits.shape == (64, 2)
        assert preds[0].boxes.shape == (64, 4)
        assert bool((preds[0].boxes >= 0).all())
        assert bool((preds[0].boxes <= 1).all())
        assert preds[0].raw_boxes is not None

    def test_deterministic_construction(self):
        m1 = make_toy_detector(seed=3)
        m2 = make_toy_detector(seed=3)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1, p2)

    def test_sigmoid_head_channels(self):
        m = make_toy_detector(seed=0, head_type="sigmoid", num_classes=3)
        out = m(torch.randn(1, 3, 8, 8))
        assert out.shape == (1, 64, 7)     # 3 class logits + 4 box params


class TestToyAdapter:
    def test_blocks(self, toy_detector):
        adapter = ToyAdapter(toy_detector)
        blocks = adapter.blocks()
        assert [n for n, _ in blocks] == ["proj", "enc1", "enc2", "head"]
        assert blocks[0][1] is toy_detector.proj
        assert blocks[3][1] is toy_detector.head

    def test_block_execution_once(self, toy_detector):
        adapter = ToyAdapter(toy_detector)
        validate_block_execution(adapter, torch.randn(1, 3, 8, 8))

    def test_capture_and_forward_with_blocks(self, toy_detector):
        adapter = ToyAdapter(toy_detector)
        ios = adapter.capture_blocks(torch.randn(2, 3, 8, 8))
        assert len(ios) == 4
        assert all(io.output is not None for io in ios)
        preds, outputs = adapter.forward_with_blocks(torch.randn(2, 3, 8, 8))
        assert len(preds) == 2 and len(outputs) == 4

    def test_run_block_reexecutes(self, toy_detector):
        adapter = ToyAdapter(toy_detector)
        io = adapter.capture_blocks(torch.randn(1, 3, 8, 8), index=0)[0]
        out = adapter.run_block(0, io.args, io.kwargs)
        assert torch.allclose(out, io.output, atol=1e-6)


class TestAdapterFactory:
    def test_unknown_model(self):
        with pytest.raises(ValueError):
            get_adapter_factory("yolov99")

    def test_toy_factory(self, toy_factory, toy_detector):
        adapter = toy_factory(toy_detector)
        assert isinstance(adapter, ToyAdapter)
        assert adapter.head_type == "softmax"
        assert adapter.num_classes == 1


class TestCheckLogitsChannels:
    def test_softmax(self):
        check_logits_channels("softmax", 1, 2)
        check_logits_channels("softmax", 80, 81)
        with pytest.raises(ValueError):
            check_logits_channels("softmax", 1, 3)

    def test_sigmoid(self):
        check_logits_channels("sigmoid", 3, 3)
        with pytest.raises(ValueError):
            check_logits_channels("sigmoid", 3, 4)

    def test_unknown_head(self):
        with pytest.raises(ValueError):
            check_logits_channels("argmax", 1, 2)
