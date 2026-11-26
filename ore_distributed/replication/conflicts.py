"""Semantic conflict records preserved by CRDT merges (Step 26).

When two replicas contain semantically conflicting claims, the merge keeps
both evidence records without declaring either claim true: the conflict object
links claims, contexts, sources, timestamps, and provenance, and routes the
conflict to USG evidence evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def _non_empty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class Claim:
    """One claim observed by a replica with its evidence and provenance."""

    claim_id: str
    text: str
    origin: str
    evidence_ids: tuple[str, ...] = ()
    context: Mapping[str, Any] = None  # type: ignore[assignment]
    hlc: tuple[int, int, str] = (0, 0, "")
    confidence: float = 0.0

    def __post_init__(self) -> None:
        """Validate claim identity, provenance, and confidence."""
        object.__setattr__(self, "claim_id", _non_empty("claim_id", self.claim_id))
        object.__setattr__(self, "text", _non_empty("text", self.text))
        object.__setattr__(self, "origin", _non_empty("origin", self.origin))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "context", dict(self.context or {}))
        if not isinstance(self.hlc, tuple) or len(self.hlc) != 3:
            raise ValueError("hlc must be a (physical, logical, node) triple")
        confidence = float(self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", confidence)

    def to_dict(self) -> dict[str, Any]:
        """Return the claim as a plain dictionary."""
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "origin": self.origin,
            "evidence_ids": list(self.evidence_ids),
            "context": dict(self.context),
            "hlc": list(self.hlc),
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class SemanticConflictRecord:
    """An explicit contradiction between claims, routed to USG policy (Step 26)."""

    conflict_id: str
    subject: str
    claims: tuple[Claim, ...]
    created_at: tuple[int, int, str]
    resolution_status: str = "open"

    def __post_init__(self) -> None:
        """Validate the conflict identity and claim set."""
        object.__setattr__(self, "conflict_id", _non_empty("conflict_id", self.conflict_id))
        object.__setattr__(self, "subject", _non_empty("subject", self.subject))
        claims = tuple(self.claims)
        if len(claims) < 2:
            raise ValueError("a semantic conflict links at least two claims")
        if any(not isinstance(item, Claim) for item in claims):
            raise TypeError("claims must contain Claim values")
        object.__setattr__(self, "claims", claims)
        if self.resolution_status not in {"open", "under_evaluation", "superseded", "unresolvable"}:
            raise ValueError("unsupported resolution status")

    def claim_ids(self) -> tuple[str, ...]:
        """Return the identifiers of the linked claims."""
        return tuple(claim.claim_id for claim in self.claims)

    def evidence_ids(self) -> tuple[str, ...]:
        """Return every evidence identifier attached to the conflict."""
        seen: list[str] = []
        for claim in self.claims:
            for evidence in claim.evidence_ids:
                if evidence not in seen:
                    seen.append(evidence)
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        """Return the conflict as a plain dictionary for USG routing."""
        return {
            "conflict_id": self.conflict_id,
            "subject": self.subject,
            "claims": [claim.to_dict() for claim in self.claims],
            "created_at": list(self.created_at),
            "resolution_status": self.resolution_status,
        }


def detect_semantic_contradiction(
    subject: str,
    claims: Sequence[Claim],
    hlc_now: tuple[int, int, str] = (0, 0, ""),
) -> SemanticConflictRecord | None:
    """Link semantically contradictory claims on one subject into a conflict.

    Contradiction is detected by predicate inversion on the claim texts; the
    routine never resolves the contradiction or marks either claim true.
    """
    if len(claims) < 2:
        return None
    texts = [claim.text.strip().casefold() for claim in claims]
    contradictory = False
    for i, first in enumerate(texts):
        for j, second in enumerate(texts):
            if i >= j:
                continue
            if _inverts(first, second):
                contradictory = True
    if not contradictory:
        return None
    import uuid

    return SemanticConflictRecord(
        conflict_id=f"conflict_{uuid.uuid4().hex}",
        subject=subject,
        claims=tuple(claims),
        created_at=hlc_now,
    )


_NEGATIONS = (
    (" is ", " is not "),
    (" are ", " are not "),
    (" can ", " cannot "),
    (" has ", " does not have "),
    (" increases ", " decreases "),
)


def _inverts(first: str, second: str) -> bool:
    """Return whether two claim texts are predicate-level negations of each other."""
    for positive, negative in _NEGATIONS:
        if positive in f" {first} " and negative in f" {second} ":
            return True
        if negative in f" {first} " and positive in f" {second} ":
            return True
    return False


__all__ = [
    "Claim",
    "SemanticConflictRecord",
    "detect_semantic_contradiction",
]
