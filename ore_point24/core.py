"""Shared primitives for the Point 24 unified integration layer.

Every helper here is a deterministic, dependency-free building block used by the
50-step integration algorithm: canonical JSON serialisation, SHA-256 digests,
stable identifiers, UTC timestamps and the evidence-state enumeration that the
capability ledger (Step 4) relies on.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

POINT24_VERSION = "1.0.0"
SPEC_TITLE = (
    "ORE Integrated Algorithm - Point 24: Complete Integration, Residual "
    "Coverage and the Unified ORE Architecture"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def utc_now_iso() -> str:
    """Return a timezone-aware ISO-8601 UTC timestamp (second precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonical_json(value: Any) -> str:
    """Serialise a JSON-compatible value deterministically (sorted keys)."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def sha256_digest(value: Any) -> str:
    """Return the SHA-256 hex digest of the canonical JSON form of ``value``.

    This is the deterministic fingerprint F(a) = SHA256(canonical(a)) required
    by Step 1 (baseline hashes), Step 6 (artifact identity) and Step 29
    (integrity and durable lineage).
    """
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def is_sha256(value: str) -> bool:
    """Return True when ``value`` is a lowercase 64-character SHA-256 digest."""
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def new_id(prefix: str) -> str:
    """Return a stable, unique identifier of the form ``prefix_<uuid4hex>``."""
    clean = re.sub(r"[^a-z0-9_]+", "_", prefix.strip().lower()).strip("_") or "id"
    return f"{clean}_{uuid.uuid4().hex}"


def require_unit_interval(name: str, value: float) -> float:
    """Validate that ``value`` is a finite number in [0, 1]; return it."""
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1]; got {value!r}")
    return number


class EvidenceState(str, Enum):
    """Evidence-based implementation states (Step 4).

    A capability's state is a conjunction of independently evidenced facts;
    "examined" and "implemented" are never interchangeable, and intermediate
    states stay visible instead of collapsing into one percentage.
    """

    IMPLEMENTED_AND_TESTED = "implemented_and_tested"
    IMPLEMENTED_INTEGRATION_INCOMPLETE = "implemented_integration_incomplete"
    REGISTERED_BACKEND_UNAVAILABLE = "registered_backend_unavailable"
    PROPOSED = "proposed"
    RESEARCH_OPTION = "research_option"
    REJECTED_OR_DEFERRED = "rejected_or_deferred"


# Discrete module-state vector entries (Step 2): (source, manifest, backend,
# validation, integration-test) each separately evidenced.
class ModuleStateVector:
    """The five separately evidenced Booleans of Step 2 for one module.

    ``v_m = (s_m, r_m, b_m, t_m, i_m)`` - source presence, manifest
    registration, backend binding, validation and integration-test coverage.
    """

    __slots__ = ("source_present", "manifest_registered", "backend_bound",
                 "validated", "integration_tested")

    def __init__(
        self,
        *,
        source_present: bool,
        manifest_registered: bool,
        backend_bound: bool,
        validated: bool,
        integration_tested: bool,
    ) -> None:
        self.source_present = bool(source_present)
        self.manifest_registered = bool(manifest_registered)
        self.backend_bound = bool(backend_bound)
        self.validated = bool(validated)
        self.integration_tested = bool(integration_tested)

    def to_tuple(self) -> tuple[bool, bool, bool, bool, bool]:
        return (
            self.source_present,
            self.manifest_registered,
            self.backend_bound,
            self.validated,
            self.integration_tested,
        )

    def to_dict(self) -> dict[str, bool]:
        return {
            "source_present": self.source_present,
            "manifest_registered": self.manifest_registered,
            "backend_bound": self.backend_bound,
            "validated": self.validated,
            "integration_tested": self.integration_tested,
        }

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"ModuleStateVector{self.to_tuple()}"


def derive_evidence_state(vector: ModuleStateVector) -> EvidenceState:
    """Map a module state vector onto the Step 4 evidence states.

    The strict gate g_c = I_contract * I_backend * I_tests * I_integration is 1
    only when every indicator is independently evidenced.
    """
    if not vector.source_present:
        return EvidenceState.PROPOSED
    if not vector.manifest_registered:
        return EvidenceState.REGISTERED_BACKEND_UNAVAILABLE
    if not vector.backend_bound:
        return EvidenceState.REGISTERED_BACKEND_UNAVAILABLE
    if not vector.validated:
        return EvidenceState.IMPLEMENTED_INTEGRATION_INCOMPLETE
    if not vector.integration_tested:
        return EvidenceState.IMPLEMENTED_INTEGRATION_INCOMPLETE
    return EvidenceState.IMPLEMENTED_AND_TESTED


__all__ = [
    "POINT24_VERSION",
    "SPEC_TITLE",
    "canonical_json",
    "derive_evidence_state",
    "is_sha256",
    "new_id",
    "require_unit_interval",
    "sha256_digest",
    "utc_now_iso",
    "EvidenceState",
    "ModuleStateVector",
]
