"""MAPE-K Plan stage and Knowledge base (ORE Point 23, Steps 11-12).

The planner generates candidate actions with predicted effects, costs and
rollback conditions. The knowledge base retrieves historical outcomes,
workload profiles, prior decisions, constraints and failure modes, with
versioned policy knowledge that distinguishes measured facts from learned
estimates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from ..actions import ActionType, PolicyOrigin, ResourceAction
from ..config import CROSConfig
from ..resource_model import WorkloadClass
from .analyze import Diagnosis, ResourceCondition


@dataclass(frozen=True)
class DecisionRecord:
    """Provenance record for one executed decision (spec Steps 12 and 15)."""

    state_version: int
    policy_version: str
    action_id: str
    action_type: str
    policy_origin: str
    rationale: str
    constraints_checked: tuple[str, ...]
    predicted_outcome: Mapping[str, Any]
    measured_outcome: Mapping[str, Any] | None
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the record."""
        return {
            "state_version": self.state_version, "policy_version": self.policy_version,
            "action_id": self.action_id, "action_type": self.action_type,
            "policy_origin": self.policy_origin, "rationale": self.rationale,
            "constraints_checked": list(self.constraints_checked),
            "predicted_outcome": dict(self.predicted_outcome),
            "measured_outcome": dict(self.measured_outcome)
            if self.measured_outcome else None,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class KnowledgeEntry:
    """One knowledge entry with explicit provenance (spec Step 12)."""

    key: str
    fact: str
    provenance: str  # "measured" | "learned" | "declared"
    value: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the entry."""
        return {
            "key": self.key, "fact": self.fact, "provenance": self.provenance,
            "value": self.value,
        }


@dataclass(frozen=True)
class KnowledgeContext:
    """Policy context retrieved for one planning step (spec Step 12)."""

    entries: tuple[KnowledgeEntry, ...]
    recent_decisions: tuple[DecisionRecord, ...]
    knowledge_version: str

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the context."""
        return {
            "entries": [item.to_dict() for item in self.entries],
            "recent_decisions": [item.to_dict() for item in self.recent_decisions],
            "knowledge_version": self.knowledge_version,
        }


class KnowledgeBase:
    """Versioned store of policy history, profiles and constraints (Step 12)."""

    def __init__(self, *, knowledge_version: str = "knowledge-1.0.0") -> None:
        """Initialize the versioned knowledge store."""
        if not isinstance(knowledge_version, str) or not knowledge_version.strip():
            raise ValueError("knowledge_version must be a non-empty string")
        self.knowledge_version = knowledge_version
        self._entries: dict[str, KnowledgeEntry] = {}
        self._decisions: list[DecisionRecord] = []
        self._max_decisions = 1024

    def record_fact(self, entry: KnowledgeEntry) -> None:
        """Record one knowledge entry with its provenance."""
        if entry.provenance not in ("measured", "learned", "declared"):
            raise ValueError("provenance must be measured, learned or declared")
        self._entries[entry.key] = entry

    def record_decision(self, record: DecisionRecord) -> None:
        """Append one decision record with bounded retention."""
        self._decisions.append(record)
        if len(self._decisions) > self._max_decisions:
            del self._decisions[: len(self._decisions) - self._max_decisions]

    def retrieve(self, diagnosis: Diagnosis) -> KnowledgeContext:
        """Retrieve the policy context relevant to one diagnosis."""
        relevant: list[KnowledgeEntry] = []
        condition_key = f"condition/{diagnosis.condition.value}"
        for key, entry in sorted(self._entries.items()):
            if key.startswith("condition/") and key != condition_key:
                continue
            relevant.append(entry)
        return KnowledgeContext(
            entries=tuple(relevant),
            recent_decisions=tuple(self._decisions[-8:]),
            knowledge_version=self.knowledge_version,
        )

    def decision_count(self) -> int:
        """Return the number of retained decision records."""
        return len(self._decisions)


class Planner:
    """MAPE-K Plan stage: candidate action generation (spec Step 11)."""

    def __init__(self, config: CROSConfig) -> None:
        """Configure the planner with the shared CROS configuration."""
        self.config = config

    def generate(
        self, diagnosis: Diagnosis, knowledge: KnowledgeContext,
        *,
        current_worker_count: int,
    ) -> tuple[ResourceAction, ...]:
        """Generate candidate actions with predicted effects and rollback.

        Candidates always include the deterministic KEEP_CURRENT_POLICY
        option; resource-pressure conditions add reversible adjustments,
        ordered so optional exploration and indexing defer before any
        integrity-critical operation is touched (spec Step 47).
        """
        version = self.config.policy_version
        state_version = diagnosis.snapshot_version
        candidates: list[ResourceAction] = [
            ResourceAction(
                action_id="keep-current", action_type=ActionType.KEEP_CURRENT_POLICY,
                parameters={}, rationale="maintain current safe policy",
                policy_origin=PolicyOrigin.BASELINE, policy_version=version,
                state_version=state_version,
            )
        ]
        if diagnosis.condition is ResourceCondition.HEALTHY:
            return tuple(candidates)
        if diagnosis.condition is ResourceCondition.DEPENDENCY_BLOCKED:
            candidates.append(ResourceAction(
                action_id="defer-exploration-dependency",
                action_type=ActionType.DEFER_BACKGROUND,
                parameters={"defer_classes": [WorkloadClass.EXPLORATORY], "defer": True},
                rationale="dependency-blocked work is the bottleneck; defer optional "
                          "exploration before touching anything else",
                policy_origin=PolicyOrigin.DETERMINISTIC, policy_version=version,
                state_version=state_version,
                predicted_objective_delta=-0.05, cost=0.01,
                rollback_condition="dependency backlog clears",
            ))
            return tuple(candidates)
        if diagnosis.condition in (
            ResourceCondition.CPU_BOUND, ResourceCondition.IO_BOUND,
            ResourceCondition.NETWORK_BOUND, ResourceCondition.GPU_BOUND,
            ResourceCondition.MEMORY_BOUND,
        ):
            defer_classes = [
                WorkloadClass.EXPLORATORY,
                WorkloadClass.MAINTENANCE,
                WorkloadClass.BATCH,
            ]
            candidates.append(ResourceAction(
                action_id="defer-optional-work",
                action_type=ActionType.DEFER_BACKGROUND,
                parameters={"defer_classes": defer_classes, "defer": True},
                rationale=(
                    f"{diagnosis.condition.value} pressure: defer optional "
                    "exploration/indexing before core work is affected"
                ),
                policy_origin=PolicyOrigin.DETERMINISTIC, policy_version=version,
                state_version=state_version,
                predicted_objective_delta=-0.15, cost=0.02,
                rollback_condition="pressure reading returns below threshold",
            ))
            target = max(
                self.config.limits.min_worker_count,
                min(self.config.limits.max_worker_count, current_worker_count + 1),
            )
            candidates.append(ResourceAction(
                action_id="scale-workers-up",
                action_type=ActionType.ADJUST_WORKER_COUNT,
                parameters={"target_worker_count": target},
                rationale="add one bounded worker to absorb the measured backlog",
                policy_origin=PolicyOrigin.DETERMINISTIC, policy_version=version,
                state_version=state_version,
                predicted_objective_delta=-0.10, cost=0.05,
                rollback_condition="queue depth returns below threshold",
            ))
        if diagnosis.condition is ResourceCondition.MEMORY_BOUND:
            candidates.append(ResourceAction(
                action_id="shrink-hot-tier",
                action_type=ActionType.CHANGE_CACHE_TIER,
                parameters={"tier_sizes": {"hot": 32, "warm": 128, "cold": 512}},
                rationale="memory pressure: shrink the hot tier; retrieval semantics "
                          "and provenance are unaffected by tier residency",
                policy_origin=PolicyOrigin.DETERMINISTIC, policy_version=version,
                state_version=state_version,
                predicted_objective_delta=-0.08, cost=0.03,
                rollback_condition="memory pressure reading returns below threshold",
            ))
        return tuple(candidates)

    def estimate(self, action: ResourceAction) -> ResourceAction:
        """Complete effect, cost and rollback estimates for one action (Step 11).

        Actions generated by this planner already carry estimates; this
        method re-validates them and can be extended with predictive models.
        """
        for name in ("predicted_objective_delta", "cost"):
            value = float(getattr(action, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        return action


def select_baseline_or_constrained(
    safe_actions: Sequence[ResourceAction],
    *,
    canary_active: bool,
    bandit_proposal: str | None = None,
) -> ResourceAction:
    """Select the action to execute: baseline or constrained learned (section 5).

    Learned proposals may only win when the canary is active and the
    proposal is among the safe actions; otherwise the deterministic
    baseline ordering decides (lowest cost, then best objective delta,
    then action id for determinism).
    """
    if not safe_actions:
        raise ValueError("select_baseline_or_constrained requires safe actions")
    if canary_active and bandit_proposal is not None:
        for action in safe_actions:
            if action.action_id == bandit_proposal:
                return action
    return sorted(
        safe_actions,
        # Total predicted cost: objective delta plus execution cost; the
        # deterministic no-op has total 0, so only genuinely improving
        # actions displace it. Ties break deterministically.
        key=lambda item: (
            item.predicted_objective_delta + item.cost,
            item.cost, item.action_id,
        ),
    )[0]


__all__ = [
    "Planner", "KnowledgeBase", "KnowledgeContext", "KnowledgeEntry",
    "DecisionRecord", "select_baseline_or_constrained",
]
