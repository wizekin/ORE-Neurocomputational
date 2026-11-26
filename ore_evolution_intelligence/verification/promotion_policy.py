"""Strict lifecycle state machine and promotion policy (section 11; section 7).

States: GENERATED, SANDBOX_TESTED, VALIDATED, APPROVED, CANARY, ACTIVE, DEPRECATED,
REJECTED. Every transition requires *typed evidence*; the production coordinator only
ever selects APPROVED / CANARY / ACTIVE strategies (acceptance tests 26, 27, 33).

Transition evidence required by the state machine:

- GENERATED -> SANDBOX_TESTED: a completed ``evaluation`` record.
- SANDBOX_TESTED -> VALIDATED: passing ``static_check``, ``semantic_check`` and an
  improved ``heldout`` comparison.
- VALIDATED -> APPROVED: an explicit ``approval`` record (governance decision).
- APPROVED -> CANARY: a ``canary_plan`` with non-empty canary tasks.
- CANARY -> ACTIVE: a passing ``canary_decision`` (regression gate + stable envelope).
- ACTIVE -> DEPRECATED: a ``deprecation`` record with a reason.
- Any state -> REJECTED: a ``rejection`` record with a reason (mandatory for safety
  failures).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from ..domain.strategy import LIFECYCLE_STATES, StrategyCandidate

#: Allowed transitions of the strict lifecycle (section 11).
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "GENERATED": frozenset({"SANDBOX_TESTED", "REJECTED"}),
    "SANDBOX_TESTED": frozenset({"VALIDATED", "REJECTED"}),
    "VALIDATED": frozenset({"APPROVED", "REJECTED"}),
    "APPROVED": frozenset({"CANARY", "DEPRECATED", "REJECTED"}),
    "CANARY": frozenset({"ACTIVE", "DEPRECATED", "REJECTED"}),
    "ACTIVE": frozenset({"DEPRECATED"}),
    "DEPRECATED": frozenset(),
    "REJECTED": frozenset(),
}

#: Evidence keys required per transition.
TRANSITION_EVIDENCE: dict[tuple[str, str], tuple[str, ...]] = {
    ("GENERATED", "SANDBOX_TESTED"): ("evaluation",),
    ("SANDBOX_TESTED", "VALIDATED"): ("static_check", "semantic_check", "heldout"),
    ("VALIDATED", "APPROVED"): ("approval",),
    ("APPROVED", "CANARY"): ("canary_plan",),
    ("CANARY", "ACTIVE"): ("canary_decision",),
    ("ACTIVE", "DEPRECATED"): ("deprecation",),
}


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    """Governance approval: approver identity and justification (VALIDATED -> APPROVED)."""

    approver: str
    justification: str
    approved_at_utc: str = ""

    def __post_init__(self) -> None:
        if not self.approver.strip():
            raise ValueError("approver must be non-empty")
        if not self.justification.strip():
            raise ValueError("justification must be non-empty")
        if not self.approved_at_utc:
            object.__setattr__(
                self, "approved_at_utc",
                datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "approver": self.approver, "justification": self.justification,
            "approved_at_utc": self.approved_at_utc,
        }


@dataclass(frozen=True, slots=True)
class CanaryPlan:
    """Canary rollout plan (APPROVED -> CANARY): bounded task subset and duration."""

    canary_task_ids: tuple[str, ...]
    duration_ms: int = 60_000
    rollback_on_drift: bool = True

    def __post_init__(self) -> None:
        if not self.canary_task_ids:
            raise ValueError("canary plan needs at least one canary task")

    def as_dict(self) -> dict[str, Any]:
        return {
            "canary_task_ids": list(self.canary_task_ids),
            "duration_ms": self.duration_ms,
            "rollback_on_drift": self.rollback_on_drift,
        }


@dataclass(frozen=True, slots=True)
class DeprecationRecord:
    """Reasoned deprecation (ACTIVE -> DEPRECATED)."""

    reason: str
    replaced_by: str | None = None

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("deprecation reason must be non-empty")

    def as_dict(self) -> dict[str, Any]:
        return {"reason": self.reason, "replaced_by": self.replaced_by}


@dataclass(frozen=True, slots=True)
class RejectionRecord:
    """Reasoned rejection (any -> REJECTED)."""

    reason: str
    stage: str = ""

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("rejection reason must be non-empty")

    def as_dict(self) -> dict[str, Any]:
        return {"reason": self.reason, "stage": self.stage}


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """Promotion request verdict."""

    granted: bool
    new_state: str
    reasons: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "granted": self.granted, "new_state": self.new_state,
            "reasons": list(self.reasons),
            "missing_evidence": list(self.missing_evidence),
        }


def _evidence_valid(key: str, evidence: Any) -> str | None:
    """Validate one piece of typed evidence; return a problem message or None."""
    if key == "evaluation":
        if evidence is None or getattr(evidence, "metrics", None) is None:
            return "evaluation evidence must be a completed EvaluationRecord"
        return None
    if key == "static_check":
        if evidence is None or not getattr(evidence, "ok", False):
            return "static check must pass without error findings"
        return None
    if key == "semantic_check":
        if evidence is None or not getattr(evidence, "ok", False):
            return "semantic check must pass without error findings"
        return None
    if key == "heldout":
        if evidence is None or not getattr(evidence, "improved", False):
            return "held-out comparison must show statistically and practically meaningful improvement"
        return None
    if key == "approval":
        if not isinstance(evidence, ApprovalRecord):
            return "approval evidence must be an ApprovalRecord"
        return None
    if key == "canary_plan":
        if not isinstance(evidence, CanaryPlan):
            return "canary evidence must be a CanaryPlan"
        return None
    if key == "canary_decision":
        if evidence is None:
            return "canary decision evidence is required"
        approved = getattr(evidence, "approved", None)
        drifted = getattr(getattr(evidence, "drift", None), "drifted", None)
        if drifted is True:
            return "canary decision drifted from the validation envelope"
        if approved is False:
            return "canary regression gate did not approve the candidate"
        if approved is None:
            return "canary decision evidence must carry an approved flag"
        return None
    if key == "deprecation":
        if not isinstance(evidence, DeprecationRecord):
            return "deprecation evidence must be a DeprecationRecord"
        return None
    if key == "rejection":
        if not isinstance(evidence, RejectionRecord):
            return "rejection evidence must be a RejectionRecord"
        return None
    return f"unknown evidence key {key!r}"


class PromotionPolicy:
    """Apply strict lifecycle transitions with typed evidence (section 11)."""

    def apply(
        self,
        candidate: StrategyCandidate,
        target_state: str,
        evidence: Mapping[str, Any] | None = None,
    ) -> tuple[StrategyCandidate, PromotionDecision]:
        """Return ``(candidate_in_new_state, decision)`` — never mutates the input."""
        evidence = dict(evidence or {})
        if target_state not in LIFECYCLE_STATES:
            return candidate, PromotionDecision(
                granted=False, new_state=candidate.state,
                reasons=(f"unknown target state {target_state!r}",),
            )
        if target_state == "REJECTED":
            problem = _evidence_valid("rejection", evidence.get("rejection"))
            if problem is not None:
                return candidate, PromotionDecision(
                    granted=False, new_state=candidate.state, reasons=(problem,),
                    missing_evidence=("rejection",),
                )
            return candidate.with_state("REJECTED"), PromotionDecision(
                granted=True, new_state="REJECTED",
                reasons=tuple([evidence["rejection"].reason]),
            )
        allowed = ALLOWED_TRANSITIONS.get(candidate.state, frozenset())
        if target_state not in allowed:
            return candidate, PromotionDecision(
                granted=False, new_state=candidate.state,
                reasons=(
                    f"transition {candidate.state} -> {target_state} is not allowed "
                    f"(allowed: {sorted(allowed) or 'terminal'})",
                ),
            )
        required = TRANSITION_EVIDENCE.get((candidate.state, target_state), ())
        missing: list[str] = []
        reasons: list[str] = []
        for key in required:
            problem = _evidence_valid(key, evidence.get(key))
            if problem is None:
                continue
            missing.append(key)
            reasons.append(problem)
        if missing:
            return candidate, PromotionDecision(
                granted=False, new_state=candidate.state, reasons=tuple(reasons),
                missing_evidence=tuple(missing),
            )
        return candidate.with_state(target_state), PromotionDecision(
            granted=True, new_state=target_state,
        )


__all__ = [
    "ALLOWED_TRANSITIONS", "ApprovalRecord", "CanaryPlan", "DeprecationRecord",
    "PromotionDecision", "PromotionPolicy", "RejectionRecord", "TRANSITION_EVIDENCE",
]
