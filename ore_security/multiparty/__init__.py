"""Secure aggregation, MPC and shared field/sharing primitives (Phase H)."""
from __future__ import annotations

from .aggregation import (
    PROTOCOL_REF,
    AggregationParticipant,
    SecureAggregationServer,
    run_secure_aggregation_round,
)
from .fields import MERSENNE_61, PrimeField
from .mpc import PROTOCOL, THREAT_MODEL, MPCProtocolEngine
from .secret_sharing import (
    AdditiveSecretSharer,
    BeaverTriple,
    BeaverTripleFactory,
    ShareVector,
)

__all__ = [
    "MERSENNE_61",
    "PROTOCOL",
    "PROTOCOL_REF",
    "THREAT_MODEL",
    "AdditiveSecretSharer",
    "AggregationParticipant",
    "BeaverTriple",
    "BeaverTripleFactory",
    "MPCProtocolEngine",
    "PrimeField",
    "SecureAggregationServer",
    "ShareVector",
    "run_secure_aggregation_round",
]
