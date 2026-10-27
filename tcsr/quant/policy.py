"""Operator-level bit-width policies.

The locked profile used by the published headline results is "mixed": decoder
weights/activations W4A8 and every other supported operator W8A8. At export
the decoder becomes weight-only INT4 with FP16 compute (W4A16) while the rest
stays W8A8; LayerNorm, softmax and I/O remain FP16. Bit-profile sweeps cover
W8A8, mixed, and W4A4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class OpBits:
    w_bits: int
    a_bits: int


PROFILES = ("w8a8", "mixed", "w4a4")

# Name fragments of operators that stay full precision in every comparison:
# detection heads, the input stem, embeddings and projections of positional
# encoders are excluded from the quantized inventory.
_SKIP_FRAGMENTS = ("head", "stem", "embed", "pos_embed", "reference", "query")


def get_bit_assignment(name: str, profile: str) -> Optional[OpBits]:
    """Bit assignment for a supported Linear/Conv operator.

    Returns None when the operator stays full precision.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {PROFILES}")
    lname = name.lower()
    if any(f in lname for f in _SKIP_FRAGMENTS):
        return None
    is_decoder = "decoder" in lname
    if profile == "w8a8":
        return OpBits(8, 8)
    if profile == "mixed":
        return OpBits(4, 8) if is_decoder else OpBits(8, 8)
    # w4a4
    return OpBits(4, 4)


def macs_shares(profile: str = "mixed") -> Tuple[float, float, float]:
    """Reported MACs shares of the exported HRSID/HR-02 engine inventory:
    W4A16 / W8A8 / FP16 = 38.7% / 52.9% / 8.4%."""
    return (0.387, 0.529, 0.084)
