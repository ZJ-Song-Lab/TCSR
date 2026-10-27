"""Stage 2: teacher-corroborated student set reconstruction (paper Sec. 4,
Eqs. pq-fg-residual-corloss-final; Algorithm 1 lines 8-9).

Every update runs one minibatch through the quantized model with gradients
and through the frozen full-precision student under no-grad:

- ``L_blk``: mean normalized block-reconstruction loss over all blocks
  (targets from the full-precision forward).
- ``L_fg``: per image, the anchor assignment rho_theta* is re-solved on the
  dense P/Q discrepancy matrix (a constant in backward) and
  ``L_fg^x = (1/max(1,|A_P|)) sum_j d(y_j^P, y_{rho(j)}^Q_theta)``.
- ``L_res``: Gaussian MMD between the smoothed class probabilities of the
  residual queries ``R_P = [N_P] \\ A_P`` and ``R_Q = [N_Q] \\ rho(A_P)``
  with the frozen per-setting bandwidth.
- ``L_cor``: the variant-weighted extra term
  ``(1/max(1,|A_P|)) sum_j w_j d(target_j, y_{rho(j)}^Q_theta)``; the
  coefficients w are the frozen detached variant coefficients (for the
  headline variant G, ``w_j = c_j (1 + beta u_j^0)``) and the targets are
  the student anchor values (for variant I, the feasible teacher partners).
  The term is dropped when the teacher is absent or lambda_cor = 0, so the
  objective degenerates exactly to L_student.

``L_student = lambda_blk L_blk + lambda_fg E_x L_fg + lambda_res E_x L_res
+ lambda_round R_round`` and ``L_TCSR = L_student + lambda_cor E_x L_cor``
(Eq. final); only theta (the quantizer state) receives gradients, via
AdamW at ``stage2_lr`` for ``stage2_updates`` updates.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .config import MethodConfig
from .losses import (Prediction, block_losses, class_probs,
                     pairwise_discrepancy, residual_mmd)
from .matching import solve_anchor_assignment
from .models import DetectorAdapter
from .quant.fake_quant import (mean_rounding_regularizer,
                               set_soft_rounding,
                               trainable_quantizer_parameters)
from .teacher import TeacherCache
from .variants import VARIANT_SPECS, normalize_variant


class SetReconstructor:
    """Stage-2 optimization of the quantizer state for one variant."""

    def __init__(self, adapter_q: DetectorAdapter,
                 adapter_p: DetectorAdapter, cache: TeacherCache,
                 cfg: MethodConfig, variant: str = "G",
                 bandwidth: float = 0.084, device: str = "cpu",
                 seed: int = 0, log: Optional[Callable[[str], None]] = None):
        self.adapter_q = adapter_q
        self.adapter_p = adapter_p
        self.cache = cache
        self.cfg = cfg
        self.variant = normalize_variant(variant)
        self.spec = VARIANT_SPECS[self.variant]
        self.bandwidth = float(bandwidth)
        self.device = torch.device(device)
        self.seed = seed
        self._log = log if log is not None else (lambda msg: None)
        self.has_teacher = any(r.teacher_pred is not None for r in cache)
        self.use_teacher = (self.has_teacher and self.spec.teacher_term
                            and cfg.lambda_cor != 0.0)
        self.cache.to(self.device)

    # ------------------------------------------------------------------
    def run(self, batches: Sequence[Tuple[torch.Tensor, List[str]]],
            updates: Optional[int] = None,
            log_every: Optional[int] = None) -> List[Dict]:
        """Run the Stage-2 updates; returns the loss-component history."""
        from .batching import cycle

        cfg = self.cfg
        n_updates = cfg.stage2_updates if updates is None else int(updates)
        log_every = log_every if log_every is not None else max(1, n_updates // 10)

        params = trainable_quantizer_parameters(self.adapter_q.model)
        if not params:
            raise RuntimeError("no trainable quantizer parameters found")
        if self.use_teacher:
            for rec in self.cache:
                if (rec.anchors is not None and rec.anchors.numel()
                        and rec.w is None):
                    raise ValueError(
                        "variant coefficients missing; call "
                        "assign_variant_weights before Stage 2")
        opt = torch.optim.AdamW(params, lr=cfg.stage2_lr, weight_decay=0.0)

        self.adapter_p.model.eval()
        self.adapter_q.model.eval()
        set_soft_rounding(self.adapter_q.model, True)

        stream = cycle(batches, seed=self.seed)
        history: List[Dict] = []
        for step in range(n_updates):
            images, ids = next(stream)
            images = images.to(self.device)
            comps = self._update_step(opt, images, ids)
            comps["step"] = step + 1
            if step == 0 or (step + 1) % log_every == 0:
                history.append(comps)
                self._log(
                    f"[stage2:{self.variant}] step {step + 1}/{n_updates} "
                    f"loss {comps['loss']:.6f} "
                    f"(blk {comps['blk']:.6f} fg {comps['fg']:.6f} "
                    f"res {comps['res']:.6f} cor {comps['cor']:.6f} "
                    f"round {comps['round']:.6f})")
        return history

    # ------------------------------------------------------------------
    def _update_step(self, opt, images: torch.Tensor,
                     ids: Sequence[str]) -> Dict[str, float]:
        cfg = self.cfg
        preds_q, outs_q = self.adapter_q.forward_with_blocks(images)
        with torch.no_grad():
            outs_p = self.adapter_p.block_output_tensors(images)
        blk_total, n_blk = block_losses(outs_q, outs_p)
        l_blk = blk_total / max(1, n_blk)

        fg_sum = torch.zeros((), device=self.device)
        res_sum = torch.zeros((), device=self.device)
        cor_sum = torch.zeros((), device=self.device)
        n_img = 0
        for b, image_id in enumerate(ids):
            rec = self.cache.get(image_id)
            if rec is None:
                continue
            parts = self._image_losses(rec, preds_q[b])
            fg_sum = fg_sum + parts["fg"]
            res_sum = res_sum + parts["res"]
            cor_sum = cor_sum + parts["cor"]
            n_img += 1

        loss = cfg.lambda_blk * l_blk
        reg = mean_rounding_regularizer(self.adapter_q.model)
        loss = loss + cfg.lambda_round * reg
        if self.spec.set_losses and n_img:
            loss = loss + cfg.lambda_fg * (fg_sum / n_img)
            loss = loss + cfg.lambda_res * (res_sum / n_img)
        if self.use_teacher and n_img:
            loss = loss + cfg.lambda_cor * (cor_sum / n_img)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return {"loss": float(loss.detach()),
                "blk": float(l_blk.detach()),
                "fg": float(fg_sum.detach()),
                "res": float(res_sum.detach()),
                "cor": float(cor_sum.detach()),
                "round": float(reg.detach())}

    # ------------------------------------------------------------------
    def _image_losses(self, rec, q_pred: Prediction) -> Dict[str, torch.Tensor]:
        """Per-image set losses {fg, res, cor} (Eq. fg, residual, corloss)."""
        cfg = self.cfg
        head = self.adapter_q.head_type
        parts = {"fg": torch.zeros((), device=self.device),
                 "res": torch.zeros((), device=self.device),
                 "cor": torch.zeros((), device=self.device)}
        k = int(rec.anchors.numel()) if rec.anchors is not None else 0
        denom = max(1, k)

        need_match = (self.spec.set_losses
                      or (self.use_teacher and not self.spec.teacher_targets))
        D = rho = None
        valid = None
        if k and need_match:
            a_pred = rec.anchor_predictions()
            D = pairwise_discrepancy(a_pred, q_pred, cfg, head)   # (K, N_Q)
            rho = solve_anchor_assignment(D)                       # constant
            valid = rho >= 0
            ar = torch.arange(k, device=D.device)

        if self.spec.set_losses:
            if D is not None:
                parts["fg"] = D[ar[valid], rho[valid]].sum() / denom
            # Residual sets R_P = [N_P] \ A_P, R_Q = [N_Q] \ rho(A_P).
            n_p = rec.student_pred.logits.shape[0]
            n_q = q_pred.logits.shape[0]
            mask_p = torch.ones(n_p, dtype=torch.bool, device=self.device)
            if k:
                mask_p[rec.anchors] = False
            mask_q = torch.ones(n_q, dtype=torch.bool, device=self.device)
            if rho is not None and bool(valid.any()):
                mask_q[rho[valid]] = False
            v = class_probs(rec.student_pred.logits[mask_p], cfg, head)
            w = class_probs(q_pred.logits[mask_q], cfg, head)
            parts["res"] = residual_mmd(v, w, self.bandwidth)

        if self.use_teacher:
            if self.spec.teacher_targets:
                # Variant I: targets are the feasible teacher partners.
                if rho is not None:
                    vm = ((rec.w > 0) & valid & (rec.partner >= 0))
                    if bool(vm.any()):
                        idx = torch.nonzero(vm, as_tuple=False).flatten()
                        tp = Prediction(
                            logits=rec.teacher_pred.logits[
                                rec.partner[idx]],
                            boxes=rec.teacher_pred.boxes[rec.partner[idx]])
                        d_tq = pairwise_discrepancy(tp, q_pred, cfg, head)
                        rows = torch.arange(idx.numel(), device=d_tq.device)
                        parts["cor"] = (rec.w[idx]
                                        * d_tq[rows, rho[idx]]).sum() / denom
            elif D is not None:
                # Student-valued targets with frozen variant coefficients;
                # zero coefficients contribute nothing (variant B included).
                parts["cor"] = (rec.w[valid]
                                * D[ar[valid], rho[valid]]).sum() / denom
        return parts
