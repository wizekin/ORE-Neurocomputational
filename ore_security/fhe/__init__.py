"""Homomorphic encryption adapter (FHE contract; additive reference scheme)."""
from __future__ import annotations

from .paillier import (
    ADAPTER_ID,
    PARAMETER_SETS,
    SUPPORTED_OPERATIONS,
    PaillierCiphertext,
    PaillierContext,
    PaillierPublicKey,
    homomorphic_add,
    homomorphic_multiply,
    homomorphic_negate,
    homomorphic_scalar_mul,
    homomorphic_sub,
)

__all__ = [
    "ADAPTER_ID",
    "PARAMETER_SETS",
    "SUPPORTED_OPERATIONS",
    "PaillierCiphertext",
    "PaillierContext",
    "PaillierPublicKey",
    "homomorphic_add",
    "homomorphic_multiply",
    "homomorphic_negate",
    "homomorphic_scalar_mul",
    "homomorphic_sub",
]
