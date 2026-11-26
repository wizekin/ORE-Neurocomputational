"""Zero-knowledge proof adapters behind typed contracts (Phase E)."""
from __future__ import annotations

from .schnorr import (
    ADAPTER_ID,
    DEFAULT_G,
    DEFAULT_P,
    DEFAULT_Q,
    RELATION_ASSUMPTIONS,
    RELATION_DESCRIPTION,
    RELATION_ID,
    RELATION_VERSION,
    SchnorrGroup,
    SchnorrProof,
    SchnorrProofAdapter,
    SchnorrWitness,
)
from .service import VERIFIER_VERSION, ProofService
from .statements import ProofStatementRegistry, RelationDefinition

__all__ = [
    "ADAPTER_ID",
    "DEFAULT_G",
    "DEFAULT_P",
    "DEFAULT_Q",
    "RELATION_ASSUMPTIONS",
    "RELATION_DESCRIPTION",
    "RELATION_ID",
    "RELATION_VERSION",
    "VERIFIER_VERSION",
    "ProofService",
    "ProofStatementRegistry",
    "RelationDefinition",
    "SchnorrGroup",
    "SchnorrProof",
    "SchnorrProofAdapter",
    "SchnorrWitness",
]
