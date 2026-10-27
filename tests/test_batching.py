"""Batch plumbing and dataset helper tests."""

import pytest
import torch

from tcsr.batching import collect_batches, cycle, unpack_batch
from tcsr.datasets import SPLIT_SIZES, letterbox_params


class TestUnpackBatch:
    def test_dict(self):
        images = torch.zeros(2, 3, 8, 8)
        b = {"images": images, "image_ids": [5, 6]}
        out, ids = unpack_batch(b)
        assert out is images
        assert ids == ["5", "6"]

    def test_tuple(self):
        images = torch.zeros(3, 3, 8, 8)
        out, ids = unpack_batch((images, ["a", "b", "c"]))
        assert out is images
        assert ids == ["a", "b", "c"]

    def test_tuple_without_ids(self):
        images = torch.zeros(2, 3, 8, 8)
        out, ids = unpack_batch((images,))
        assert ids == [None, None]

    def test_tensor_ids(self):
        images = torch.zeros(2, 3, 8, 8)
        _, ids = unpack_batch((images, torch.tensor([10, 20])))
        assert ids == ["10", "20"]

    def test_unsupported(self):
        with pytest.raises(TypeError):
            unpack_batch("nope")


class TestCollectBatches:
    def test_fill_ids(self):
        loader = [(torch.zeros(2, 3, 8, 8), [None, "b"]),
                  (torch.zeros(3, 3, 8, 8),)]
        batches = collect_batches(loader)
        assert len(batches) == 2
        assert batches[0][1] == ["img_000000", "b"]
        assert batches[1][1] == ["img_000002", "img_000003", "img_000004"]

    def test_truncation(self):
        loader = [(torch.zeros(4, 3, 8, 8), [f"i{i}" for i in range(4)]),
                  (torch.zeros(4, 3, 8, 8), [f"j{i}" for i in range(4)])]
        batches = collect_batches(loader, max_images=6)
        assert len(batches) == 2
        assert batches[0][0].shape[0] == 4
        assert batches[1][0].shape[0] == 2
        assert batches[1][1] == ["j0", "j1"]

    def test_exact_limit(self):
        loader = [(torch.zeros(4, 3, 8, 8),)]
        batches = collect_batches(loader, max_images=4)
        assert len(batches) == 1
        assert batches[0][0].shape[0] == 4


class TestCycle:
    def test_deterministic(self):
        batches = [(torch.zeros(1), ["a"]), (torch.zeros(1), ["b"]),
                   (torch.zeros(1), ["c"])]
        it1 = cycle(batches, seed=0)
        s1 = [next(it1)[1][0] for _ in range(6)]
        it2 = cycle(batches, seed=0)
        s2 = [next(it2)[1][0] for _ in range(6)]
        assert s1 == s2
        # all items appear each epoch
        assert sorted(s1[:3]) == ["a", "b", "c"]
        assert sorted(s1[3:]) == ["a", "b", "c"]

    def test_empty(self):
        with pytest.raises(ValueError):
            next(cycle([]))


class TestLetterbox:
    def test_scale_min_side(self):
        assert letterbox_params(1280, 720, 640) == (0.5, 0.0, 0.0)
        assert letterbox_params(720, 1280, 640) == (0.5, 0.0, 0.0)
        assert letterbox_params(640, 640, 640) == (1.0, 0.0, 0.0)
        assert letterbox_params(320, 200, 640) == (2.0, 0.0, 0.0)


def test_split_sizes():
    assert SPLIT_SIZES["hrsid"]["total"] == 5604
    assert SPLIT_SIZES["ssdd"] == {"train": 800, "val": 128, "test": 232,
                                   "total": 1160}
    assert SPLIT_SIZES["coco"] == {"dev": 5000, "heldout": 5000}
