"""Stage 1: teacher-free sequential block reconstruction (Algorithm 1).

Every quantized block is optimized in execution order against the frozen
full-precision student P.  Following the BRECQ-style cascade, at every
update the quantized block input is re-captured from the *current*
quantized model (so blocks optimized earlier feed their improved outputs
forward), while the reconstruction target is the corresponding block output
of a no-grad full-precision forward:

    L_i = || out_Qi(x) - out_Pi(x) ||^2 / (|| out_Pi(x) ||^2 + eps)
          + lambda_round * R_round(block i),

with R_round the per-weight mean AdaRound regularizer.  Only the quantizer
state theta of the block is trained (Adam, lr = stage1_lr, at most
stage1_updates_per_block updates).  The result of Stage 1 is the common
starting point Q_0 shared by every Stage-2 variant.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .config import MethodConfig
from .losses import normalized_block_loss
from .models import DetectorAdapter, _first_tensor
from .quant.fake_quant import (mean_rounding_regularizer,
                               set_soft_rounding,
                               trainable_quantizer_parameters)


class BlockReconstructor:
    """Sequential per-block optimization of the quantizer state."""

    def __init__(self, adapter_q: DetectorAdapter,
                 adapter_p: DetectorAdapter, cfg: MethodConfig,
                 device: str = "cpu", seed: int = 0,
                 log: Optional[Callable[[str], None]] = None):
        self.adapter_q = adapter_q
        self.adapter_p = adapter_p
        self.cfg = cfg
        self.device = torch.device(device)
        self.seed = seed
        self._log = log if log is not None else (lambda msg: None)

    def run(self, batches: Sequence[Tuple[torch.Tensor, List[str]]],
            updates_per_block: Optional[int] = None,
            log_every: Optional[int] = None) -> List[Dict]:
        """Optimize every quantized block; returns per-block summaries."""
        from .batching import cycle

        cfg = self.cfg
        updates = (cfg.stage1_updates_per_block if updates_per_block is None
                   else int(updates_per_block))
        log_every = log_every if log_every is not None else max(1, updates // 10)

        self.adapter_p.model.eval()
        self.adapter_q.model.eval()
        set_soft_rounding(self.adapter_q.model, True)

        stream = cycle(batches, seed=self.seed)
        summaries: List[Dict] = []
        blocks = self.adapter_q.blocks()
        for i, (name, module) in enumerate(blocks):
            params = trainable_quantizer_parameters(module)
            if not params:
                self._log(f"[stage1] block {i} ({name}): no quantized "
                          "operators, skipped")
                summaries.append({"index": i, "name": name, "params": 0,
                                  "updates": 0, "first_loss": None,
                                  "last_loss": None, "mean_loss": None})
                continue
            summaries.append(self._optimize_block(
                i, name, module, params, stream, updates, log_every))
        return summaries

    def _optimize_block(self, index: int, name: str, module,
                        params, stream, updates: int,
                        log_every: int) -> Dict:
        cfg = self.cfg
        opt = torch.optim.Adam(params, lr=cfg.stage1_lr)
        first_loss: Optional[float] = None
        last_loss: Optional[float] = None
        total_loss = 0.0
        for step in range(updates):
            images, _ = next(stream)
            images = images.to(self.device)
            # Cascade input from the current quantized model (no grad).
            io_q = self.adapter_q.capture_blocks(images, index=index)[0]
            # Frozen full-precision target (no grad).
            io_p = self.adapter_p.capture_blocks(images, index=index)[0]
            if io_p.output is None:
                raise RuntimeError(
                    f"block {index} ({name}) of the full-precision model "
                    "produced no tensor output")
            out_q = _first_tensor(self.adapter_q.run_block(
                index, io_q.args, io_q.kwargs))
            if out_q is None:
                raise RuntimeError(
                    f"block {index} ({name}) of the quantized model produced "
                    "no tensor output")
            loss = normalized_block_loss(out_q, io_p.output)
            loss = loss + cfg.lambda_round * mean_rounding_regularizer(module)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            v = float(loss.detach())
            first_loss = v if first_loss is None else first_loss
            last_loss = v
            total_loss += v
            if (step + 1) % log_every == 0 or step == 0:
                self._log(f"[stage1] block {index} ({name}) "
                          f"step {step + 1}/{updates} loss {v:.6f}")
        mean_loss = total_loss / max(1, updates)
        self._log(f"[stage1] block {index} ({name}) done: "
                  f"{first_loss:.6f} -> {last_loss:.6f} "
                  f"(mean {mean_loss:.6f})")
        return {"index": index, "name": name, "params": len(params),
                "updates": updates, "first_loss": first_loss,
                "last_loss": last_loss, "mean_loss": mean_loss}
