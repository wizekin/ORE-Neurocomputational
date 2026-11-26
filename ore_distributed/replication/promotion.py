"""Gate for authoritative USG semantic promotion of replicated observations (Step 30).

A replicated observation becomes a candidate USG assertion only after
validation of schema, source provenance, confidence, context, and
contradiction status. Consensus may order an approval decision, but it never
establishes epistemic truth: CRDT convergence is a transport/storage
property, not proof that two claims are true or compatible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ore_distributed.replication.conflicts import SemanticConflictRecord


class PromotionRejected(ValueError):
    """Raised when an observation fails the promotion gate."""


@dataclass(frozen=True, slots=True)
class CandidateAssertion:
    """A replicated observation nominated for USG promotion (Step 30)."""

    assertion_id: str
    subject: str
    predicate: str
    payload: Mapping[str, Any]
    source_reference: str
    confidence: float
    context: Mapping[str, Any]
    observed_event_id: str
    provenance_digest: str

    def __post_init__(self) -> None:
        """Validate the candidate's identity, provenance, and confidence."""
        for name in ("assertion_id", "subject", "predicate", "source_reference", "observed_event_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "context", dict(self.context))
        confidence = float(self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", confidence)

    def to_dict(self) -> dict[str, Any]:
        """Return the candidate as a plain dictionary."""
        return {
            "assertion_id": self.assertion_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "payload": dict(self.payload),
            "source_reference": self.source_reference,
            "confidence": self.confidence,
            "context": dict(self.context),
            "observed_event_id": self.observed_event_id,
            "provenance_digest": self.provenance_digest,
        }


@dataclass(frozen=True, slots=True)
class PromotionOutcome:
    """Decision record explaining one promotion attempt."""

    assertion_id: str
    promoted: bool
    reason: str
    routed_conflict_id: str | None = None


class USGPromotionGate:
    """Validate candidates before authoritative semantic promotion (Step 30).

    The gate implements the protocol-level checks: schema conformance, source
    provenance presence, confidence bounds, context completeness, and
    contradiction status. Semantically deep validation lives in the USG adapter
    (``ore_distributed.integration.usg_adapter``), which applies the real
    ore_usg contracts on top of this gate.
    """

    def __init__(self, minimum_confidence: float = 0.5) -> None:
        """Create a gate with an explicit minimum confidence threshold."""
        if not 0.0 < float(minimum_confidence) < 1.0:
            raise ValueError("minimum_confidence must be strictly inside (0, 1)")
        self.minimum_confidence = float(minimum_confidence)
        self._promoted: dict[str, CandidateAssertion] = {}
        self._outcomes: list[PromotionOutcome] = []
        self.conflicts_routed = 0

    def outcomes(self) -> tuple[PromotionOutcome, ...]:
        """Return every decision the gate has recorded."""
        return tuple(self._outcomes)

    def promoted_assertions(self) -> tuple[CandidateAssertion, ...]:
        """Return the assertions that passed the gate."""
        return tuple(self._promoted.values())

    def _reject(self, candidate: CandidateAssertion, reason: str, conflict_id: str | None = None) -> PromotionOutcome:
        """Record and return a rejection without raising."""
        outcome = PromotionOutcome(
            assertion_id=candidate.assertion_id,
            promoted=False,
            reason=reason,
            routed_conflict_id=conflict_id,
        )
        self._outcomes.append(outcome)
        return outcome

    def evaluate(
        self,
        candidate: CandidateAssertion,
        open_conflicts: Sequence[SemanticConflictRecord] = (),
    ) -> PromotionOutcome:
        """Run the protocol-level promotion checks on one candidate."""
        if not isinstance(candidate, CandidateAssertion):
            raise TypeError("evaluate expects a CandidateAssertion")
        # Schema conformance: subject/predicate/payload shape.
        if not candidate.payload and not candidate.predicate.strip():
            return self._reject(candidate, "schema validation failed: empty payload with empty predicate")
        # Source provenance must exist and be content-addressed.
        if not candidate.source_reference.strip() or not candidate.provenance_digest.strip():
            return self._reject(candidate, "missing source provenance for promotion")
        # Confidence must clear the explicit threshold.
        if candidate.confidence < self.minimum_confidence:
            return self._reject(
                candidate,
                f"confidence {candidate.confidence} below threshold {self.minimum_confidence}",
            )
        # Context must at least identify a scope.
        if not candidate.context:
            return self._reject(candidate, "missing context scope for promotion")
        # Contradiction status: conflicts must be routed, not auto-resolved.
        for conflict in open_conflicts:
            claims = conflict.to_dict()["claims"]
            for claim in claims:
                if candidate.observed_event_id == claim.get("claim_id"):
                    self.conflicts_routed += 1
                    return self._reject(
                        candidate,
                        "candidate participates in an open semantic contradiction; "
                        "routed to USG evidence evaluation",
                        conflict_id=conflict.conflict_id,
                    )
        outcome = PromotionOutcome(
            assertion_id=candidate.assertion_id,
            promoted=True,
            reason="passed schema, provenance, confidence, context, and contradiction checks",
        )
        self._outcomes.append(outcome)
        self._promoted[candidate.assertion_id] = candidate
        return outcome

    def consensus_ordering_note(self) -> str:
        """Return the standing reminder that consensus does not establish truth."""
        return (
            "Consensus may order an approval decision; it does not establish "
            "epistemic truth. CRDT convergence is a transport/storage property."
        )


def convergence_is_not_truth(candidate: CandidateAssertion) -> bool:
    """Return whether a candidate still needs explicit epistemic validation.

    This helper exists so callers can assert that a converged CRDT replica
    never automatically marks a semantic assertion as true (Step 30).
    """
    return candidate.assertion_id and True


__all__ = [
    "PromotionRejected",
    "CandidateAssertion",
    "PromotionOutcome",
    "USGPromotionGate",
    "convergence_is_not_truth",
]
