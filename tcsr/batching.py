"""Batch plumbing shared by calibration, Stage 1, and Stage 2.

A calibration loader yields either

- dicts with an ``images`` tensor and an optional ``image_ids`` entry, or
- plain tuples ``(images, image_ids?)``.

``collect_batches`` materializes the calibration view (bounded by
``calib_images``) once; every later pass (cache building, Stage 1, Stage 2)
reuses the same in-memory batches so the anchor cache and the optimization
streams see exactly the same images with stable identifiers.
"""

from __future__ import annotations

import random
from typing import Any, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch


def unpack_batch(batch: Any) -> Tuple[Any, List[Optional[str]]]:
    """Normalize one loader batch to (images, ids).

    ``ids`` has one entry per image; missing identifiers are None and are
    filled in by :func:`collect_batches` with a stable global counter.
    """
    if isinstance(batch, dict):
        images = batch.get("images")
        raw = batch.get("image_ids")
    elif isinstance(batch, (list, tuple)):
        images = batch[0]
        raw = batch[1] if len(batch) > 1 else None
    else:
        raise TypeError(f"unsupported batch type {type(batch)!r}")
    if isinstance(images, (list, tuple)):
        # Nested structures such as ((images, targets), indices).
        images = images[0]
    if torch.is_tensor(raw):
        raw = raw.tolist()
    if raw is None:
        ids: List[Optional[str]] = [None] * images.shape[0]
    else:
        ids = [None if i is None else str(i) for i in raw]
    return images, ids


def collect_batches(loader: Iterable[Any],
                    max_images: Optional[int] = None
                    ) -> List[Tuple[torch.Tensor, List[str]]]:
    """Materialize the bounded calibration view as (images, ids) batches.

    The final batch is truncated so at most ``max_images`` images are kept;
    None ids become ``img_{global index:06d}`` (stable across all passes).
    """
    batches: List[Tuple[torch.Tensor, List[str]]] = []
    n = 0
    for batch in loader:
        images, ids = unpack_batch(batch)
        b = images.shape[0]
        if max_images is not None and n + b > max_images:
            keep = max_images - n
            if keep > 0:
                ids = _fill_ids(ids, n)
                batches.append((images[:keep].contiguous(), ids[:keep]))
            break
        batches.append((images, _fill_ids(ids, n)))
        n += b
        if max_images is not None and n >= max_images:
            break
    return batches


def _fill_ids(ids: Sequence[Optional[str]], start: int) -> List[str]:
    return [i if i is not None else f"img_{start + k:06d}"
            for k, i in enumerate(ids)]


def cycle(batches: Sequence[Tuple[torch.Tensor, List[str]]],
          seed: int = 0) -> Iterator[Tuple[torch.Tensor, List[str]]]:
    """Deterministic infinite stream over the calibration batches.

    Each epoch reshuffles the batch order with ``random.Random(seed)``; the
    first epoch is shuffled too so Stage 1 and Stage 2 see the same
    deterministic stream given the same seed.
    """
    items = list(batches)
    if not items:
        raise ValueError("no calibration batches")
    rng = random.Random(seed)
    while True:
        order = list(range(len(items)))
        rng.shuffle(order)
        for i in order:
            yield items[i]
