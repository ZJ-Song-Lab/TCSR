"""Ablation variants A-I (paper Sec. 5.4, mechanism table).

All variants except I retain student-valued residuals: the teacher only
enters through detached coefficients.  B-I share Q_0, A_P, the P/Q matching,
the optimizer, and the update count; A is the block-only Stage-1 baseline
(no Stage-2 set reconstruction).  Each variant fixes the coefficients w_j
that multiply the per-anchor extra term (and, for I, replaces the target
values of that term with the feasible teacher partners).

Coefficient definitions (V_x = {j in A_P : c_j > 0}, G coefficients
w_j = c_j (1 + beta u_j^0)):

- A, B: all-zero coefficients (A additionally drops the Stage-2 losses).
- C:    P-proxy -- the |V_x| most confident anchors (by s_j^P) each receive
        mass / |V_x|, matching G's corroborated count and total mass.
- C_Q:  G's coefficient multiset (zeros included) assigned hardest-first:
        the r-th largest coefficient goes to the anchor with the r-th
        largest g_Q(j) = d(y_j^P, y_{rho0(j)}^{Q0}).
- D:    uniform mass / |V_x| on V_x (identity preserved, quality flattened).
- E:    c_j only (beta = 0).
- F:    1 + beta u_j^0 on V_x, zero elsewhere (c_j replaced by 1).
- G:    full TCSR coefficients c_j (1 + beta u_j^0).
- H:    the full G coefficient vector permuted within (image, predicted
        class) strata, zeros included.
- I:    G coefficients, but the extra term's targets are the feasible
        teacher partners instead of the student anchors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from .config import MethodConfig
from .teacher import ImageRecord

VARIANTS = ("A", "B", "C", "C_Q", "D", "E", "F", "G", "H", "I")

# Variants whose coefficients require the frozen priorities u^0 / g_Q.
NEEDS_U0 = ("C", "C_Q", "D", "F", "G", "H", "I")
NEEDS_GQ = ("C_Q",)


def normalize_variant(name: str) -> str:
    v = str(name).strip().upper().replace(" ", "")
    if v == "CQ":
        v = "C_Q"
    if v not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; expected one of {VARIANTS}")
    return v


@dataclass(frozen=True)
class VariantSpec:
    """Which loss terms are active for a variant."""
    stage2: bool = True            # Stage-2 set reconstruction active (False: A)
    set_losses: bool = True        # L_fg and L_res active
    teacher_term: bool = True      # extra corroboration term active
    teacher_targets: bool = False  # extra-term targets are teacher partners (I)


VARIANT_SPECS: Dict[str, VariantSpec] = {
    "A": VariantSpec(stage2=False, set_losses=False, teacher_term=False),
    "B": VariantSpec(teacher_term=False),
    "C": VariantSpec(),
    "C_Q": VariantSpec(),
    "D": VariantSpec(),
    "E": VariantSpec(),
    "F": VariantSpec(),
    "G": VariantSpec(),
    "H": VariantSpec(),
    "I": VariantSpec(teacher_targets=True),
}

VARIANT_LABELS: Dict[str, str] = {
    "A": "A: block-only reconstruction",
    "B": "B: pure student (zero coefficients)",
    "C": "C: P-proxy count/mass-matched coefficients",
    "C_Q": "C_Q: permuted G coefficients (hardest-first)",
    "D": "D: uniform coefficients on the corroborated subset",
    "E": "E: corroboration only (beta = 0)",
    "F": "F: priority only",
    "G": "G: full TCSR",
    "H": "H: stratified permutation of the G coefficients",
    "I": "I: direct teacher-valued extra term",
}


@dataclass
class VariantContext:
    """Per-image inputs for coefficient construction."""
    anchor_scores: torch.Tensor                    # (K,) s_j^P
    anchor_labels: torch.Tensor                    # (K,) l_j^P
    c: torch.Tensor                                # (K,) detached, in [0, 1]
    u0: Optional[torch.Tensor] = None              # (K,) frozen priorities
    g_q0: Optional[torch.Tensor] = None            # (K,) Stage-1 difficulty
    cfg: MethodConfig = field(default_factory=MethodConfig)
    perm_seed: int = 0


def g_weights(ctx: VariantContext) -> torch.Tensor:
    """G coefficients: c_j (1 + beta u_j^0), detached."""
    w = ctx.c.detach().clone()
    if ctx.u0 is not None and ctx.cfg.beta != 0.0:
        w = w * (1.0 + ctx.cfg.beta * ctx.u0.detach())
    return w


def _shuffle_within_strata(w: torch.Tensor, labels: torch.Tensor,
                           seed: int) -> torch.Tensor:
    """Permute the coefficient vector within predicted-class strata.

    The permutation acts on the full vector, zeros included, so the
    multiset of coefficients within each (image, class) stratum is preserved
    exactly.  Deterministic given seed.
    """
    dev = w.device
    w_c = w.detach().to("cpu")
    lab_c = labels.detach().to("cpu")
    out = torch.empty_like(w_c)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    for l in torch.unique(lab_c):
        idx = torch.nonzero(lab_c == l, as_tuple=False).flatten()
        perm = torch.randperm(idx.numel(), generator=g)
        out[idx] = w_c[idx][perm]
    return out.to(dev)


@torch.no_grad()
def build_coefficients(variant: str, ctx: VariantContext) -> torch.Tensor:
    """Frozen variant coefficients w (K,), detached (zeros if K == 0)."""
    v = normalize_variant(variant)
    k = ctx.c.shape[0]
    dev = ctx.c.device
    if k == 0:
        return torch.zeros(0, device=dev)
    if v in NEEDS_U0 and ctx.u0 is None:
        raise ValueError(f"variant {v} requires the frozen priorities u^0")
    if v in NEEDS_GQ and ctx.g_q0 is None:
        raise ValueError(f"variant {v} requires the Stage-1 difficulty g_Q")
    if v in ("A", "B"):
        return torch.zeros(k, device=dev)
    if v in ("G", "I"):
        return g_weights(ctx)
    if v == "E":
        return ctx.c.detach().clone()
    if v == "F":
        if ctx.u0 is None:
            raise ValueError("variant F requires the frozen priorities u^0")
        w = torch.zeros(k, device=dev)
        mask = ctx.c > 0
        if bool(mask.any()):
            w[mask] = 1.0 + ctx.cfg.beta * ctx.u0[mask].detach()
        return w
    if v == "D":
        # Identity-preserving uniform coefficients: mass / |V_x| on V_x.
        w = torch.zeros(k, device=dev)
        mask = ctx.c > 0
        n = int(mask.sum())
        if n:
            w[mask] = g_weights(ctx).sum() / n
        return w
    if v == "C":
        # P-proxy: |V_x| most confident anchors receive mass / |V_x|.
        w = torch.zeros(k, device=dev)
        mask = ctx.c > 0
        n = int(mask.sum())
        if n:
            mass = g_weights(ctx).sum()
            order = torch.sort(ctx.anchor_scores.detach(), descending=True,
                               stable=True)[1]
            w[order[:n]] = mass / n
        return w
    if v == "C_Q":
        if ctx.g_q0 is None:
            raise ValueError("variant C_Q requires the Stage-1 difficulty g_Q")
        if ctx.u0 is None:
            raise ValueError("variant C_Q requires the frozen priorities u^0")
        w_g = g_weights(ctx)
        w = torch.zeros(k, device=dev)
        # r-th largest coefficient -> r-th hardest anchor (stable sorts keep
        # the ascending-index tie rule on both sides).
        coef_order = torch.sort(w_g, descending=True, stable=True)[1]
        anchor_order = torch.sort(ctx.g_q0.detach(), descending=True,
                                  stable=True)[1]
        w[anchor_order] = w_g[coef_order]
        return w
    if v == "H":
        if ctx.u0 is None:
            raise ValueError("variant H requires the frozen priorities u^0")
        w_g = g_weights(ctx)
        return _shuffle_within_strata(w_g, ctx.anchor_labels, ctx.perm_seed)
    raise AssertionError(f"unhandled variant {v}")


@torch.no_grad()
def assign_variant_weights(records: Sequence[ImageRecord], variant: str,
                           cfg: Optional[MethodConfig] = None,
                           perm_seed: int = 0) -> List[ImageRecord]:
    """Compute and store rec.w (frozen active-variant coefficients).

    Deterministic given perm_seed; the per-image shuffle seed for H is
    perm_seed + record index.
    """
    v = normalize_variant(variant)
    method = cfg if cfg is not None else MethodConfig()
    for i, rec in enumerate(records):
        if rec.anchors is None or rec.anchors.numel() == 0:
            dev = rec.student_pred.logits.device
            rec.w = torch.zeros(0, device=dev)
            continue
        if v in NEEDS_U0 and rec.u0 is None:
            raise ValueError(
                f"variant {v} requires priorities; run compute_priorities first")
        if v in NEEDS_GQ and rec.g_q0 is None:
            raise ValueError(
                f"variant {v} requires g_Q; run compute_priorities first")
        ctx = VariantContext(anchor_scores=rec.anchor_scores,
                             anchor_labels=rec.anchor_labels, c=rec.c,
                             u0=rec.u0, g_q0=rec.g_q0, cfg=method,
                             perm_seed=perm_seed + i)
        rec.w = build_coefficients(v, ctx)
    return list(records)
