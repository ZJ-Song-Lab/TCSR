"""Shared fixtures: repo-root import path, deterministic seeds, toy model."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch

from tcsr.config import MethodConfig
from tcsr.models import get_adapter_factory
from tcsr.models.toy import make_toy_detector


@pytest.fixture(autouse=True)
def _deterministic_seed():
    torch.manual_seed(0)


@pytest.fixture()
def method():
    return MethodConfig()


@pytest.fixture()
def toy_detector():
    return make_toy_detector(seed=0, head_type="softmax", num_classes=1,
                             size=8, dim=16, hidden=32)


@pytest.fixture()
def toy_factory():
    return get_adapter_factory("toy", head_type="softmax", num_classes=1)


@pytest.fixture()
def toy_batches():
    g = torch.Generator().manual_seed(0)
    batches = []
    for b in range(2):
        images = torch.randn(2, 3, 8, 8, generator=g)
        ids = [f"img_{b}_{i}" for i in range(2)]
        batches.append((images, ids))
    return batches
