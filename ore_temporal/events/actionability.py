"""Actionability gating (specification Step 32).

Neural scores may contribute evidence but can never independently authorize high-impact
actions; deterministic thresholds, severity, uncertainty, and rate limits decide.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contracts import Actionability, EventSeverity, TypedEventRecord, UncertaintyReport

__all__ = ("ActionabilityGate", "ActionabilityOutcome")


@dataclass(frozen=True, slots=True)
class ActionabilityOutcome:
    """The gate's decision with its deterministic rationale."""

    decision: Actionability
    reasons: tuple[str, ...]
    rate_limited: bool = False

    @property
    def high_impact_authorized_by_neural_alone(self) -> bool:
        """Always False by construction; neural evidence alone never authorizes."""
        return False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the outcome."""
        return {
            "decision": self.decision.value,
            "reasons": list(self.reasons),
            "rate_limited": self.rate_limited,
            "high_impact_authorized_by_neural_alone": self.high_impact_authorized_by_neural_alone,
        }


@dataclass(frozen=True, slots=True)
class ActionabilityGate:
    """Apply deterministic severity, uncertainty, and rate-limit rules to events."""

    critical_severity: EventSeverity = EventSeverity.CRITICAL
    max_actionable_events: int = 8
    max_uncertainty_for_action: float = 1.0
    high_impact_requires_deterministic_evidence: bool = True

    def evaluate(self, event: TypedEventRecord, uncertainty: UncertaintyReport | None,
                 actioned_count: int, deterministic_corroboration: bool = False
                 ) -> ActionabilityOutcome:
        """Decide actionability for one typed event under the configured constraints."""
        reasons: list[str] = []
        if actioned_count >= self.max_actionable_events:
            return ActionabilityOutcome(
                decision=Actionability.NOT_ACTIONABLE,
                reasons=(f"rate limit reached ({actioned_count}/"
                         f"{self.max_actionable_events} actionable events)",),
                rate_limited=True)
        if event.severity is EventSeverity.INFO:
            reasons.append("severity is informational")
            return ActionabilityOutcome(decision=Actionability.NOT_ACTIONABLE,
                                        reasons=tuple(reasons))
        if uncertainty is not None and uncertainty.out_of_distribution:
            reasons.append("input marked out-of-distribution; uncertainty unreliable")
            return ActionabilityOutcome(decision=Actionability.REVIEW_REQUIRED,
                                        reasons=tuple(reasons))
        if event.confidence is not None and event.confidence < 0.5:
            reasons.append("event confidence below the deterministic 0.5 floor")
            return ActionabilityOutcome(decision=Actionability.REVIEW_REQUIRED,
                                        reasons=tuple(reasons))
        if (event.severity is EventSeverity.CRITICAL
                and self.high_impact_requires_deterministic_evidence
                and not deterministic_corroboration):
            reasons.append(
                "high-impact action requires deterministic corroboration; "
                "neural evidence alone is never sufficient")
            return ActionabilityOutcome(decision=Actionability.REVIEW_REQUIRED,
                                        reasons=tuple(reasons))
        reasons.append("deterministic thresholds satisfied")
        return ActionabilityOutcome(
            decision=Actionability.ACTIONABLE_WITH_CONSTRAINTS,
            reasons=tuple(reasons))
