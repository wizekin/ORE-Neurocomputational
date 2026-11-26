"""Quarantine, rollback, and canary deployment (Point 14, Steps 51-52).

On a critical threshold: stop new assignments, preserve logs, switch to a validated
fallback, mark the artifact quarantined, roll back to a known-good version;
re-entry requires reevaluation and approval.  Canary deployment routes a small
controlled share of eligible tasks to a candidate, compares it with the incumbent on
matched task classes, and rolls back when the regression threshold is crossed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..config import LearningConfig
from ..contracts import LifecycleState, ModelManifest
from ..registry import LifecycleTransitionError, ModelRegistry

__all__ = ("CanaryAssignment", "CanaryRouter", "QuarantinePlaybook",
           "execute_quarantine_and_rollback")


class QuarantineError(ValueError):
    """Raised when a quarantine or rollback violates lifecycle rules."""


@dataclass(slots=True)
class QuarantinePlaybook:
    """Step 51 -- the deterministic quarantine and rollback procedure."""

    registry: ModelRegistry
    audit_hook: Any = None

    def quarantine(self, model_id: str, version: str, reason: str,
                   task_id: str = "") -> dict[str, Any]:
        """Stop new assignments, preserve logs, and mark the artifact quarantined."""
        manifest = self.registry.get(model_id, version)
        if manifest is None:
            raise QuarantineError(f"{model_id}@{version} is not registered")
        if manifest.lifecycle_state is LifecycleState.QUARANTINED:
            return {"status": "already-quarantined", "model_id": model_id,
                    "version": version, "reason": reason}
        event = self.registry.quarantine(model_id, version, reason, task_id=task_id)
        record = {"status": "quarantined", "model_id": model_id, "version": version,
                  "reason": reason, "new_assignments_stopped": True,
                  "logs_preserved": True, "event_id": event.event_id,
                  "reentry": "re-evaluation and approval required (Step 51)"}
        self._audit("quarantine", record)
        return record

    def rollback(self, model_id: str, version: str, fallback_model_id: str,
                 fallback_version: str, reason: str,
                 task_id: str = "") -> dict[str, Any]:
        """Switch to the validated fallback; deprecate the degraded model unless it
        is already quarantined (quarantine is then its terminal record)."""
        manifest = self.registry.get(model_id, version)
        if manifest is None:
            raise QuarantineError(f"{model_id}@{version} is not registered")
        try:
            if manifest.lifecycle_state is LifecycleState.QUARANTINED:
                deprecation = None
                activation = self.registry.activate_fallback(
                    fallback_model_id, fallback_version,
                    reason=f"known-good fallback after {reason}", task_id=task_id)
            else:
                deprecation, activation = self.registry.rollback(
                    model_id, version, fallback_model_id, fallback_version, reason,
                    task_id=task_id)
        except LifecycleTransitionError as exc:
            raise QuarantineError(str(exc)) from exc
        record = {"status": "rolled-back", "model_id": model_id, "version": version,
                  "fallback_model_id": fallback_model_id,
                  "fallback_version": fallback_version, "reason": reason,
                  "deprecation_event": (deprecation.event_id
                                        if deprecation else None),
                  "activation_event": activation.event_id,
                  "fallback_state": self.registry.get(
                      fallback_model_id, fallback_version).lifecycle_state.value}
        self._audit("rollback", record)
        return record

    def reentry(self, model_id: str, version: str, validation: Any,
                approval: Any) -> dict[str, Any]:
        """Re-entry after quarantine requires reevaluation and approval (Step 51)."""
        if approval is None:
            raise QuarantineError("quarantine re-entry requires approval")
        event = self.registry.transition(
            model_id, version, LifecycleState.VALIDATED,
            reason="quarantine re-entry after re-evaluation",
            validation=validation, approval=approval)
        return {"status": "re-validated", "event_id": event.event_id}

    def _audit(self, operation: str, record: dict[str, Any]) -> None:
        """Forward one lifecycle audit event."""
        if self.audit_hook is not None:
            self.audit_hook(operation, record)


def execute_quarantine_and_rollback(registry: ModelRegistry, model_id: str,
                                    version: str, fallback_model_id: str,
                                    fallback_version: str, reason: str,
                                    audit_hook: Any = None) -> dict[str, Any]:
    """The single-call Step 51 procedure: quarantine, then roll back."""
    playbook = QuarantinePlaybook(registry, audit_hook=audit_hook)
    quarantine = playbook.quarantine(model_id, version, reason)
    rollback = playbook.rollback(model_id, version, fallback_model_id,
                                 fallback_version, reason)
    return {"quarantine": quarantine, "rollback": rollback,
            "note": "deterministic recovery before any production re-entry"}


@dataclass(slots=True)
class CanaryAssignment:
    """One task's canary routing decision."""

    task_id: str
    routed_to: str  # candidate | incumbent
    bucket: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"task_id": self.task_id, "routed_to": self.routed_to,
                "bucket": self.bucket}


class CanaryRouter:
    """Step 52 -- bounded, deterministic canary exposure with rollback gates."""

    def __init__(self, candidate: tuple[str, str], incumbent: tuple[str, str],
                 config: LearningConfig | None = None, *,
                 fraction: float | None = None,
                 registry: ModelRegistry | None = None) -> None:
        """Configure the candidate/incumbent pair and the exposure fraction."""
        self.config = config or LearningConfig()
        self.fraction = float(fraction if fraction is not None
                              else self.config.canary_default_fraction)
        if not 0.0 < self.fraction < 1.0:
            raise ValueError("canary fraction must be in (0, 1)")
        self.candidate = candidate
        self.incumbent = incumbent
        self.registry = registry
        self.assignments: list[CanaryAssignment] = []
        self.candidate_results: list[float] = []
        self.incumbent_results: list[float] = []
        self.expanded = False
        self.rolled_back = False

    def route(self, task_id: str, ordinal: int) -> CanaryAssignment:
        """Deterministically route one eligible task (stride bucketing).

        Exposure is exact: every ``1/fraction``-th task in task order goes to the
        candidate, so the configured fraction is respected deterministically.
        """
        if self.rolled_back:
            assignment = CanaryAssignment(task_id=task_id, routed_to="incumbent",
                                          bucket=ordinal % self._period())
            self.assignments.append(assignment)
            return assignment
        period = self._period()
        bucket = ordinal % period
        routed = "candidate" if bucket == 0 else "incumbent"
        assignment = CanaryAssignment(task_id=task_id, routed_to=routed,
                                      bucket=bucket)
        self.assignments.append(assignment)
        return assignment

    def record_outcome(self, task_id: str, routed_to: str, quality: float) -> None:
        """Record one matched evaluation outcome for the routed arm."""
        if routed_to == "candidate":
            self.candidate_results.append(float(quality))
        else:
            self.incumbent_results.append(float(quality))

    def compare(self) -> dict[str, Any]:
        """Compare candidate and incumbent on matched task classes."""
        candidate_mean = (sum(self.candidate_results) / len(self.candidate_results)
                          if self.candidate_results else None)
        incumbent_mean = (sum(self.incumbent_results) / len(self.incumbent_results)
                          if self.incumbent_results else None)
        regression = None
        if candidate_mean is not None and incumbent_mean is not None:
            regression = float(incumbent_mean - candidate_mean)
        return {"candidate_mean": (round(candidate_mean, 6)
                                   if candidate_mean is not None else None),
                "incumbent_mean": (round(incumbent_mean, 6)
                                   if incumbent_mean is not None else None),
                "regression": (round(regression, 6) if regression is not None else None),
                "n_candidate": len(self.candidate_results),
                "n_incumbent": len(self.incumbent_results),
                "expanded": self.expanded, "rolled_back": self.rolled_back,
                "fraction": self.fraction}

    def gate(self) -> dict[str, Any]:
        """Expand or roll back under the predefined thresholds (Step 52)."""
        comparison = self.compare()
        regression = comparison["regression"]
        if regression is not None:
            if regression > self.config.canary_regression_rollback_threshold:
                self.rolled_back = True
                if self.registry is not None:
                    try:
                        self.registry.transition(
                            self.candidate[0], self.candidate[1],
                            LifecycleState.QUARANTINED,
                            reason="canary regression threshold crossed")
                    except LifecycleTransitionError:
                        pass
                return {"decision": "rollback",
                        "reason": (f"candidate trails incumbent by {regression:.4f} "
                                   f"(threshold "
                                   f"{self.config.canary_regression_rollback_threshold})"),
                        "comparison": comparison}
            if regression <= 0:
                self.expanded = True
                return {"decision": "expand",
                        "reason": "candidate matches or beats the incumbent",
                        "comparison": comparison}
        return {"decision": "hold", "reason": "insufficient evidence to expand or "
                                              "roll back", "comparison": comparison}

    def exposure_summary(self) -> dict[str, Any]:
        """Measured exposure statistics over the routed tasks."""
        if not self.assignments:
            return {"tasks": 0, "candidate_share": 0.0, "fraction": self.fraction}
        candidate_count = sum(1 for item in self.assignments
                              if item.routed_to == "candidate")
        return {"tasks": len(self.assignments),
                "candidate_share": round(candidate_count / len(self.assignments), 6),
                "candidate_count": candidate_count,
                "fraction": self.fraction,
                "rolled_back": self.rolled_back}

    def _period(self) -> int:
        """The deterministic stride period derived from the fraction."""
        return max(1, int(round(1.0 / self.fraction)))
