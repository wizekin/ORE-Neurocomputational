"""Policy measurement, proposal, and promotion governance (Step 31).

Learned policy updates stay outside ordinary inference: measurements are written
as a separate transaction after a run completes, proposals carry provenance and
are evaluated against explicit gates, and promotion requires every gate to pass.
A task's accepted answer must not change merely because the system is writing
feedback statistics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ore_workspace.models import utc_now

from .contracts import sha256_json


@dataclass(frozen=True, slots=True)
class MeasurementTransaction:
    """One completed, separately-recorded measurement transaction (Step 31)."""

    task_id: str
    measurements: tuple[Mapping[str, Any], ...]
    outcomes: tuple[Mapping[str, Any], ...]
    recorded_at: str = field(default_factory=utc_now)
    inference_answer_unchanged: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a detached transaction record."""
        return {
            "task_id": self.task_id,
            "measurements": [dict(item) for item in self.measurements],
            "outcomes": [dict(item) for item in self.outcomes],
            "recorded_at": self.recorded_at,
            "inference_answer_unchanged": self.inference_answer_unchanged,
            "transaction_kind": "post_run_measurement_only",
        }


@dataclass(frozen=True, slots=True)
class PolicyProposal:
    """A separately authorized policy change proposal with provenance (Step 31)."""

    proposal_id: str
    task_family: str
    proposed_changes: Mapping[str, Mapping[str, float]]
    provenance: Mapping[str, Any]
    created_at: str = field(default_factory=utc_now)
    status: str = "proposed"

    def fingerprint(self) -> str:
        """Return the content fingerprint of the proposed changes."""
        return sha256_json({
            "task_family": self.task_family,
            "proposed_changes": dict(self.proposed_changes),
        })

    def to_dict(self) -> dict[str, Any]:
        """Return a detached proposal record."""
        return {
            "proposal_id": self.proposal_id,
            "task_family": self.task_family,
            "proposed_changes": {
                key: dict(value) for key, value in self.proposed_changes.items()
            },
            "provenance": dict(self.provenance),
            "created_at": self.created_at,
            "status": self.status,
            "fingerprint": self.fingerprint(),
        }


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """Report whether every promotion gate passed for one proposal (Step 31)."""

    proposal_id: str
    promoted: bool
    gate_results: Mapping[str, Mapping[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a detached promotion decision record."""
        return {
            "proposal_id": self.proposal_id,
            "promoted": self.promoted,
            "gate_results": {
                key: dict(value) for key, value in self.gate_results.items()
            },
        }


class RunMeasurementRecorder:
    """Record post-run measurements as explicit separate transactions (Step 31)."""

    def __init__(self, resource_coordinator: Any) -> None:
        """Bind the recorder to the shared resource history stores."""
        self.resource_coordinator = resource_coordinator
        self.transactions: list[MeasurementTransaction] = []

    def record(
        self,
        task_id: str,
        *,
        measurements: Sequence[Mapping[str, Any]],
        outcomes: Sequence[Mapping[str, Any]],
    ) -> MeasurementTransaction:
        """Record one completed run's measurements without touching task answers."""
        for item in measurements:
            self.resource_coordinator.record_measurement(
                item["module_id"], latency_ms=item.get("latency_ms", 0.0),
            )
        for item in outcomes:
            self.resource_coordinator.record_outcome(
                item["module_id"], bool(item["success"]),
            )
        transaction = MeasurementTransaction(
            task_id=task_id,
            measurements=tuple(dict(item) for item in measurements),
            outcomes=tuple(dict(item) for item in outcomes),
        )
        self.transactions.append(transaction)
        return transaction


class PolicyProposalBuilder:
    """Build provenance-complete policy proposals from observed run statistics (Step 31)."""

    def __init__(self, task_family: str) -> None:
        """Configure the proposal builder for one task family."""
        self.task_family = task_family

    def propose(
        self,
        *,
        activation_statistics: Mapping[str, Mapping[str, float]],
        measurement_provenance: Mapping[str, Any],
    ) -> PolicyProposal:
        """Propose bounded subscription threshold adjustments from observed behaviour.

        The proposal is a recommendation only: it is never applied automatically,
        and it becomes effective only after every promotion gate passes.
        """
        proposed: dict[str, dict[str, float]] = {}
        for module_id, stats in sorted(activation_statistics.items()):
            suppressed = float(stats.get("suppressed_computations", 0.0))
            total = float(stats.get("activations", 0.0))
            if total <= 0:
                continue
            suppression_ratio = suppressed / max(1.0, total + suppressed)
            threshold = float(stats.get("activation_threshold", 0.5))
            if suppression_ratio > 0.5:
                proposed[module_id] = {
                    "activation_threshold": min(1.0, threshold + 0.1),
                    "suppression_ratio_observed": round(suppression_ratio, 4),
                }
            elif suppression_ratio < 0.05 and threshold > 0.0:
                proposed[module_id] = {
                    "activation_threshold": max(0.0, threshold - 0.05),
                    "suppression_ratio_observed": round(suppression_ratio, 4),
                }
        proposal_id = sha256_json({
            "task_family": self.task_family,
            "activation_statistics": {
                key: dict(value) for key, value in activation_statistics.items()
            },
        })[:24]
        return PolicyProposal(
            proposal_id=f"proposal_{proposal_id}",
            task_family=self.task_family,
            proposed_changes=proposed,
            provenance={
                "origin": "ore_blackboard.policy_learning",
                "task_family": self.task_family,
                "basis": "observed activation and suppression statistics",
                "basis_statistics": {
                    key: dict(value) for key, value in activation_statistics.items()
                },
                "measurement_provenance": dict(measurement_provenance),
                "note": "Heuristic proposal; not applied automatically.",
            },
        )


class PromotionGateEvaluator:
    """Evaluate promotion gates for policy proposals (Step 31)."""

    def __init__(
        self,
        *,
        regression_runner: Callable[[PolicyProposal], Mapping[str, float]],
        safety_check: Callable[[PolicyProposal], tuple[bool, str]] | None = None,
    ) -> None:
        """Bind the regression and safety gates to caller-supplied callables."""
        self.regression_runner = regression_runner
        self.safety_check = safety_check or _default_safety_check

    def evaluate(self, proposal: PolicyProposal) -> PromotionDecision:
        """Return ``Promote = and_j Gate_j`` over every configured gate."""
        gate_results: dict[str, dict[str, Any]] = {}
        regression_metrics = self.regression_runner(proposal)
        regression_pass = all(
            value >= 0.0 for value in regression_metrics.values()
        ) if regression_metrics else True
        gate_results["regression"] = {
            "passed": regression_pass,
            "metrics": dict(regression_metrics),
            "note": "A positive mean score change alone is insufficient.",
        }
        safety_passed, safety_reason = self.safety_check(proposal)
        gate_results["safety"] = {
            "passed": safety_passed, "reason": safety_reason,
        }
        provenance_ok = bool(proposal.provenance) and "origin" in proposal.provenance
        gate_results["provenance"] = {
            "passed": provenance_ok,
            "fingerprint": proposal.fingerprint(),
        }
        promoted = all(
            gate.get("passed", False) for gate in gate_results.values()
        )
        return PromotionDecision(
            proposal_id=proposal.proposal_id,
            promoted=promoted,
            gate_results=gate_results,
        )


def _default_safety_check(proposal: PolicyProposal) -> tuple[bool, str]:
    """Default safety gate: proposed thresholds remain bounded (Step 31)."""
    for changes in proposal.proposed_changes.values():
        threshold = changes.get("activation_threshold")
        if threshold is not None and not 0.0 <= float(threshold) <= 1.0:
            return False, "proposed activation threshold outside [0, 1]"
    return True, "proposed changes remain within declared bounds"


__all__ = [
    "MeasurementTransaction",
    "PolicyProposal",
    "PolicyProposalBuilder",
    "PromotionDecision",
    "PromotionGateEvaluator",
    "RunMeasurementRecorder",
]
