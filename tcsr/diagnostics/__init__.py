"""Diagnostics: two-discrepancy gaps, direction conflict, stratification,
and structural audits (paper Sec. 5).

The diagnostics operate on the frozen per-image cache (TeacherCache /
ImageRecord lists) after tcsr.priority.compute_priorities has recorded
rho_0, g_Q, and u^0, together with the Stage-1 outputs Q_0 (and, for the
stratified table, the reconstructed student Q_f of a given run).
"""

from .audit import assignment_change, structure_audit
from .direction_conflict import (COMPONENTS, direction_conflict,
                                 kappa_components)
from .gap_statistics import collect_gaps, gap_statistics
from .gap_stratified import gap_stratified

__all__ = [
    "COMPONENTS",
    "assignment_change",
    "collect_gaps",
    "direction_conflict",
    "gap_statistics",
    "gap_stratified",
    "kappa_components",
    "structure_audit",
]
