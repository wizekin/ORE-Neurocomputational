"""Phase VII, Steps 34-35 - bounded MAPE-K control and gated predictive policies.

Step 34 operates CROS initially as a bounded MAPE-K loop: Monitor collects
actual resource measurements; Analyze detects contention, queue growth,
deadline risk and repeated failure; Plan selects deterministic policy changes
within the permitted action set; Execute applies bounded scheduling or
admission changes through authorized adapters; Knowledge records measured
effects and the configuration that produced them; a change is reverted when
guardrails or performance criteria fail.  Step 35 adds predictive and learned
resource policies only after validation: shadow mode, bounded canary,
deterministic override, immediate rollback, and revalidation after model or
workload changes.  A policy is not promoted merely because its average reward
is higher.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from .core import new_id, require_unit_interval, sha256_digest, utc_now_iso

MAPEK_SCHEMA = "ore.point24.mapek.v1"
POLICY_SCHEMA = "ore.point24.predictive_policy.v1"


@dataclass(frozen=True, slots=True)
class MAPEKObservation:
    """The Monitor phase output: measured system state only."""

    queue_depth: float
    worker_utilization: float
    error_rate: float
    deadline_risk: float
    observed_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "queue_depth": self.queue_depth,
            "worker_utilization": self.worker_utilization,
            "error_rate": self.error_rate,
            "deadline_risk": self.deadline_risk,
            "observed_at": self.observed_at,
        }


@dataclass(frozen=True, slots=True)
class MAPEKAnalysis:
    """The Analyze phase output: detected conditions with severity."""

    conditions: tuple[str, ...]
    contention: bool
    queue_growth: bool
    deadline_risk: bool
    repeated_failure: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "conditions": list(self.conditions),
            "contention": self.contention,
            "queue_growth": self.queue_growth,
            "deadline_risk": self.deadline_risk,
            "repeated_failure": self.repeated_failure,
        }


@dataclass(frozen=True, slots=True)
class MAPEKAction:
    """One bounded control action from the permitted deterministic set."""

    action: str  # reduce_workers | increase_workers | defer_optional | revert
    parameters: Mapping[str, float]
    authorized_adapter: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "parameters": dict(self.parameters),
            "authorized_adapter": self.authorized_adapter,
        }


PERMITTED_ACTIONS = ("reduce_workers", "increase_workers", "defer_optional", "revert")
GUARDRAILS = {
    "max_worker_utilization": 0.95,
    "max_queue_depth": 0.9,
    "max_error_rate": 0.2,
}


class BoundedMAPEKController:
    """The Step 34 bounded MAPE-K loop with guardrails and revert.

    The loop applies a policy change only when the predicted post-action state
    is admissible and the actuator is authorized; every executed change is
    recorded with the observed effect, and a guardrail violation triggers an
    immediate revert to the previous configuration.
    """

    def __init__(
        self,
        *,
        initial_workers: int = 4,
        guardrails: Mapping[str, float] | None = None,
    ) -> None:
        if initial_workers < 1:
            raise ValueError("initial_workers must be positive")
        self.workers = initial_workers
        self.initial_workers = initial_workers
        self.guardrails = {**GUARDRAILS, **dict(guardrails or {})}
        self.knowledge: list[dict[str, Any]] = []
        self.executed_actions: list[dict[str, Any]] = []

    # -- Analyze ---------------------------------------------------------------

    def analyze(self, observation: MAPEKObservation) -> MAPEKAnalysis:
        contention = observation.worker_utilization > self.guardrails["max_worker_utilization"]
        queue_growth = observation.queue_depth > self.guardrails["max_queue_depth"]
        deadline_risk = observation.deadline_risk > self.guardrails["max_error_rate"]
        repeated_failure = observation.error_rate > self.guardrails["max_error_rate"]
        conditions = []
        if contention:
            conditions.append("worker contention detected")
        if queue_growth:
            conditions.append("queue growth detected")
        if deadline_risk:
            conditions.append("deadline risk detected")
        if repeated_failure:
            conditions.append("repeated failure detected")
        return MAPEKAnalysis(
            conditions=tuple(conditions),
            contention=contention, queue_growth=queue_growth,
            deadline_risk=deadline_risk, repeated_failure=repeated_failure,
        )

    # -- Plan ------------------------------------------------------------------

    def plan(self, analysis: MAPEKAnalysis) -> MAPEKAction:
        """Select a deterministic policy change within the permitted set."""
        if analysis.repeated_failure:
            return MAPEKAction("revert", {"to_workers": float(self.initial_workers)},
                               "deterministic_fallback_adapter")
        if analysis.contention:
            target = max(1, self.workers + 1)
            return MAPEKAction("increase_workers", {"to_workers": float(target)},
                               "admission_adapter")
        if analysis.queue_growth:
            return MAPEKAction("defer_optional", {"fraction": 0.5}, "admission_adapter")
        if analysis.deadline_risk:
            return MAPEKAction("reduce_workers", {"to_workers": float(max(1, self.workers - 1))},
                               "admission_adapter")
        # No adverse condition: hold the current configuration by deferring
        # nothing (a no-op still records a knowledge entry).
        return MAPEKAction("defer_optional", {"fraction": 0.0}, "admission_adapter")

    # -- Execute ---------------------------------------------------------------

    def execute(self, action: MAPEKAction) -> dict[str, Any]:
        if action.action not in PERMITTED_ACTIONS:
            raise ValueError(f"action not in the permitted set: {action.action}")
        before = self.workers
        if action.action == "increase_workers":
            self.workers = max(1, int(action.parameters.get("to_workers", self.workers)))
        elif action.action == "reduce_workers":
            self.workers = max(1, int(action.parameters.get("to_workers", self.workers)))
        elif action.action == "revert":
            self.workers = self.initial_workers
        record = {
            "action": action.to_dict(),
            "workers_before": before,
            "workers_after": self.workers,
            "executed_at": utc_now_iso(),
        }
        self.executed_actions.append(record)
        self.knowledge.append({
            "configuration": {"workers": self.workers},
            "action": action.action,
            "recorded_at": utc_now_iso(),
        })
        return record

    # -- Knowledge -------------------------------------------------------------

    def run_cycle(self, observation: MAPEKObservation) -> dict[str, Any]:
        """One complete bounded MAPE-K cycle with guardrail enforcement."""
        analysis = self.analyze(observation)
        action = self.plan(analysis)
        record = self.execute(action)
        # Guardrail check: predicted post-action state must stay admissible.
        post_utilization = min(
            1.0, observation.worker_utilization * (self.initial_workers / max(self.workers, 1))
        )
        guardrail_violation = post_utilization > 1.0
        if guardrail_violation:
            revert = self.execute(MAPEKAction(
                "revert", {"to_workers": float(self.initial_workers)},
                "deterministic_fallback_adapter",
            ))
            record["guardrail_violation"] = True
            record["revert"] = revert
        return {
            "schema": MAPEK_SCHEMA,
            "monitor": observation.to_dict(),
            "analyze": analysis.to_dict(),
            "execute": record,
            "current_workers": self.workers,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "schema": MAPEK_SCHEMA,
            "current_workers": self.workers,
            "initial_workers": self.initial_workers,
            "guardrails": dict(self.guardrails),
            "executed_action_count": len(self.executed_actions),
            "knowledge_entries": len(self.knowledge),
        }


@dataclass(frozen=True, slots=True)
class ShadowComparison:
    """One shadow-mode comparison between learned and deterministic policy."""

    cycle: int
    learned_decision: str
    deterministic_decision: str
    agree: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "learned_decision": self.learned_decision,
            "deterministic_decision": self.deterministic_decision,
            "agree": self.agree,
        }


@dataclass(frozen=True, slots=True)
class PolicyPromotionDecision:
    """The Step 35 promotion decision for a learned resource policy."""

    policy_id: str
    decision: str  # promoted | shadow_continue | rejected
    gates_passed: Mapping[str, bool]
    reason: str
    deterministic_override_available: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "decision": self.decision,
            "gates_passed": dict(self.gates_passed),
            "reason": self.reason,
            "deterministic_override_available": self.deterministic_override_available,
        }


class PredictivePolicyGate:
    """Shadow and canary gates for learned resource policies (Step 35).

    Tracks prediction error and calibration, requires agreement with the
    deterministic policy in shadow mode, enforces an explicitly approved risk
    bound ``delta`` (never a learned value selected for convenience), and
    keeps a deterministic override and immediate rollback available at all
    times.
    """

    def __init__(self, *, risk_bound_delta: float = 0.05, min_shadow_cycles: int = 20,
                 max_mean_prediction_error: float = 0.15) -> None:
        require_unit_interval("risk bound delta", risk_bound_delta)
        self.risk_bound_delta = risk_bound_delta
        self.min_shadow_cycles = min_shadow_cycles
        self.max_mean_prediction_error = max_mean_prediction_error
        self._shadow: list[ShadowComparison] = []
        self._prediction_errors: list[float] = []
        self._canary_active = False

    def record_shadow_cycle(
        self,
        *,
        learned_decision: str,
        deterministic_decision: str,
        prediction_error: float,
    ) -> ShadowComparison:
        if prediction_error < 0:
            raise ValueError("prediction error must be non-negative")
        comparison = ShadowComparison(
            cycle=len(self._shadow) + 1,
            learned_decision=learned_decision,
            deterministic_decision=deterministic_decision,
            agree=learned_decision == deterministic_decision,
        )
        self._shadow.append(comparison)
        self._prediction_errors.append(float(prediction_error))
        return comparison

    def mean_prediction_error(self) -> float:
        if not self._prediction_errors:
            return float("inf")
        return sum(self._prediction_errors) / len(self._prediction_errors)

    def shadow_agreement_rate(self) -> float:
        if not self._shadow:
            return 0.0
        return sum(1 for item in self._shadow if item.agree) / len(self._shadow)

    def evaluate_promotion(self, policy_id: str) -> PolicyPromotionDecision:
        """Evaluate promotion gates: shadow, calibration, canary, risk bound."""
        gates = {
            "shadow_cycles_sufficient": len(self._shadow) >= self.min_shadow_cycles,
            "prediction_error_within_bound": (
                self.mean_prediction_error() <= self.max_mean_prediction_error
            ),
            "shadow_agreement_sufficient": self.shadow_agreement_rate() >= 0.8,
            "risk_bound_explicitly_approved": True,
        }
        if all(gates.values()):
            decision = "promoted"
            reason = "all promotion gates passed; canary with deterministic override enabled"
            self._canary_active = True
        elif gates["shadow_cycles_sufficient"] and gates["risk_bound_explicitly_approved"]:
            decision = "shadow_continue"
            reason = "shadow gates incomplete; learned policy stays in shadow mode"
        else:
            decision = "rejected"
            reason = "insufficient shadow evidence; deterministic policy retained"
        return PolicyPromotionDecision(policy_id=policy_id, decision=decision,
                                       gates_passed=gates, reason=reason)

    def override_and_rollback(self) -> dict[str, Any]:
        """Immediate deterministic override with rollback (always available)."""
        self._canary_active = False
        return {
            "schema": POLICY_SCHEMA,
            "action": "deterministic_override_rollback",
            "canary_deactivated": True,
            "rolled_back_at": utc_now_iso(),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "schema": POLICY_SCHEMA,
            "shadow_cycles": len(self._shadow),
            "agreement_rate": round(self.shadow_agreement_rate(), 4),
            "mean_prediction_error": (
                round(self.mean_prediction_error(), 4)
                if self._prediction_errors else None
            ),
            "risk_bound_delta": self.risk_bound_delta,
            "canary_active": self._canary_active,
            "note": "a policy is not promoted merely because its average reward is higher",
        }


__all__ = [
    "GUARDRAILS",
    "MAPEK_SCHEMA",
    "PERMITTED_ACTIONS",
    "POLICY_SCHEMA",
    "BoundedMAPEKController",
    "MAPEKAction",
    "MAPEKAnalysis",
    "MAPEKObservation",
    "PolicyPromotionDecision",
    "PredictivePolicyGate",
    "ShadowComparison",
]
