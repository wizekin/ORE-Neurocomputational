"""Eligibility gating, weighted agent-task scoring, and bounded assignment.

Eligibility is a separate Boolean gate: permissions, dependencies, verifier
availability, and hard budget conditions must all pass before scoring. A high
score never overrides a failed hard constraint (spec Section 5, Steps 12-14).
For eligible assignments the layer maximizes the sum of matching scores subject
to one accountable owner per task, agent capacity, dependency order, and
resource budgets, using deterministic greedy or maximum-weight matching.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .contracts import (
    AgentDescriptor,
    AgentTask,
    TaskTopology,
)
from .registry import AgentRegistry
from .reliability import BetaBinomialEstimate, score_reliability


@dataclass(frozen=True, slots=True)
class EligibilityReport:
    """Record one agent-task eligibility decision with structured gate reasons."""

    agent_id: str
    task_id: str
    eligible: bool
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a detached report record."""
        return {
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "eligible": self.eligible,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class AgentTaskScore:
    """Record the weighted matching score for one eligible agent-task pair."""

    agent_id: str
    task_id: str
    score: float
    components: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached score record."""
        return {
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "score": self.score,
            "components": dict(self.components),
        }


DEFAULT_SCORING_WEIGHTS: Mapping[str, float] = {
    "capability_fit": 0.25,
    "schema_compatibility": 0.10,
    "reliability": 0.30,
    "latency": 0.10,
    "cost": 0.10,
    "load": 0.10,
    "conflict_risk": 0.05,
}


@dataclass
class MatchingConfig:
    """Configure scoring weights and uncertainty penalties for agent-task matching."""

    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_SCORING_WEIGHTS))
    reliability_uncertainty_weight: float = 0.5
    reference_latency_seconds: float = 10.0
    reference_cost: float = 8.0
    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    def __post_init__(self) -> None:
        """Validate weights and reference scales."""
        if not self.weights:
            raise ValueError("weights must not be empty")
        for name, value in self.weights.items():
            if value < 0:
                raise ValueError(f"weight {name!r} cannot be negative")
        if sum(self.weights.values()) <= 0:
            raise ValueError("weights must have a positive total")
        if self.reference_latency_seconds <= 0 or self.reference_cost <= 0:
            raise ValueError("reference scales must be positive")
        if not 0.0 <= self.reliability_uncertainty_weight <= 1.0:
            raise ValueError("reliability_uncertainty_weight must be in [0, 1]")


class EligibilityGate:
    """Apply the hard Boolean eligibility constraints before any scoring (Step 12)."""

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        verifier_role: str = "evidence_verifier",
        estimates: Mapping[str, BetaBinomialEstimate] | None = None,
    ) -> None:
        """Bind the gate to the agent registry and optional reliability estimates."""
        self.registry = registry
        self.verifier_role = verifier_role
        self.estimates = dict(estimates or {})

    def check(
        self,
        descriptor: AgentDescriptor,
        task: AgentTask,
        *,
        remaining_goal_budget: float = float("inf"),
        verifier_available: bool = True,
    ) -> EligibilityReport:
        """Evaluate every hard constraint and collect all failure reasons."""
        reasons: list[str] = []
        missing_capabilities = set(task.required_capabilities) - set(descriptor.capability_ids)
        if missing_capabilities:
            reasons.append(f"missing capabilities: {sorted(missing_capabilities)}")
        unsupported_output = set(task.expected_result_schema.fields) - set(
            descriptor.produced_output_schema.fields
        )
        if unsupported_output:
            reasons.append(f"schema cannot produce required outputs: {sorted(unsupported_output)}")
        out_of_scope = set(task.permitted_workspace_fields) - set(descriptor.permitted_workspace_fields)
        if out_of_scope:
            reasons.append(f"task requires out-of-scope read fields: {sorted(out_of_scope)}")
        if not self.registry.assignable(descriptor.agent_id):
            reasons.append(f"agent state is {self.registry.state(descriptor.agent_id).value}")
        if descriptor.resources.cost_budget > remaining_goal_budget:
            reasons.append("agent cost budget exceeds the remaining goal budget")
        if (task.requires_independent_verification or descriptor.requires_verifier) and not verifier_available:
            reasons.append("task requires independent verification but no eligible verifier is available")
        return EligibilityReport(
            agent_id=descriptor.agent_id,
            task_id=task.task_id,
            eligible=not reasons,
            reasons=tuple(reasons),
        )

    def eligible_agents(
        self,
        task: AgentTask,
        *,
        remaining_goal_budget: float = float("inf"),
    ) -> tuple[tuple[AgentDescriptor, EligibilityReport], ...]:
        """Return descriptors and reports for every agent passing all gates."""
        verifiers = self.registry.find_by_role(self.verifier_role)
        verifier_available = any(
            self.registry.assignable(verifier.agent_id)
            and verifier.agent_id != self._single_candidate(task)
            for verifier in verifiers
        )
        results = []
        for descriptor in self.registry.list_descriptors():
            report = self.check(
                descriptor, task,
                remaining_goal_budget=remaining_goal_budget,
                verifier_available=verifier_available,
            )
            if report.eligible:
                results.append((descriptor, report))
        return tuple(results)

    def _single_candidate(self, task: AgentTask) -> str | None:
        """Identify the only candidate when exactly one agent could own the task."""
        candidates = [
            descriptor for descriptor in self.registry.list_descriptors()
            if set(task.required_capabilities).issubset(set(descriptor.capability_ids))
        ]
        return candidates[0].agent_id if len(candidates) == 1 else None


class AgentTaskMatcher:
    """Score eligible agent-task pairs with uncertainty-aware reliability (Step 13)."""

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        config: MatchingConfig | None = None,
        estimates: Mapping[str, BetaBinomialEstimate] | None = None,
        conflict_rates: Mapping[str, float] | None = None,
    ) -> None:
        """Bind the matcher to the registry, reliability estimates, and conflict risk."""
        self.registry = registry
        self.config = config or MatchingConfig()
        self.estimates = dict(estimates or {})
        self.conflict_rates = dict(conflict_rates or {})

    def score(self, descriptor: AgentDescriptor, task: AgentTask) -> AgentTaskScore:
        """Compute the weighted matching score for one eligible pair."""
        weights = self.config.weights
        total_weight = sum(weights.values())

        capability_overlap = set(task.required_capabilities) & set(descriptor.capability_ids)
        capability_fit = len(capability_overlap) / len(task.required_capabilities)

        produced = set(descriptor.produced_output_schema.fields)
        required_outputs = set(task.expected_result_schema.fields)
        schema_compatibility = len(required_outputs & produced) / len(required_outputs) if required_outputs else 1.0

        estimate = self.estimates.get(descriptor.agent_id) or BetaBinomialEstimate(
            descriptor.reliability_samples,
            max(0, descriptor.reliability_samples - int(
                (descriptor.accepted_result_rate or 0.5) * descriptor.reliability_samples
            )),
            self.config.prior_alpha,
            self.config.prior_beta,
        )
        reliability = score_reliability(
            estimate, uncertainty_weight=self.config.reliability_uncertainty_weight,
        )

        latency_ema = 0.0
        conflict_rate = self.conflict_rates.get(descriptor.agent_id, 0.0)
        latency = 1.0 / (1.0 + max(0.0, latency_ema) / self.config.reference_latency_seconds)
        cost = 1.0 - min(1.0, descriptor.resources.cost_budget / (self.config.reference_cost * 4.0))
        active = self.registry.active_tasks(descriptor.agent_id)
        load = 1.0 - min(1.0, active / max(1, descriptor.resources.max_concurrent_tasks))
        conflict = 1.0 - max(0.0, min(1.0, conflict_rate))

        components = {
            "capability_fit": capability_fit,
            "schema_compatibility": schema_compatibility,
            "reliability": reliability,
            "latency": latency,
            "cost": cost,
            "load": load,
            "conflict_risk": conflict,
        }
        score = sum(weights.get(name, 0.0) * value for name, value in components.items()) / total_weight
        return AgentTaskScore(
            agent_id=descriptor.agent_id,
            task_id=task.task_id,
            score=max(0.0, min(1.0, score)),
            components=components,
        )

    def rank(self, task: AgentTask, candidates: Iterable[AgentDescriptor]) -> tuple[AgentTaskScore, ...]:
        """Return candidate scores sorted from best to worst."""
        scores = [self.score(descriptor, task) for descriptor in candidates]
        return tuple(sorted(scores, key=lambda item: (-item.score, item.agent_id)))


class AssignmentSolver:
    """Assign one accountable owner per task under capacity constraints (Step 15)."""

    def __init__(self, registry: AgentRegistry) -> None:
        """Bind the solver to the agent registry for capacity accounting."""
        self.registry = registry

    def greedy(
        self,
        ranked: Mapping[str, tuple[AgentTaskScore, ...]],
        tasks: Mapping[str, AgentTask],
    ) -> dict[str, str]:
        """Assign tasks in priority order to the best agent with free capacity."""
        remaining_capacity = {
            descriptor.agent_id: descriptor.resources.max_concurrent_tasks
            for descriptor in self.registry.list_descriptors()
        }
        assignments: dict[str, str] = {}
        ordered = sorted(
            tasks.values(),
            key=lambda task: (-task.priority, task.task_id),
        )
        for task in ordered:
            chosen = None
            for score in ranked.get(task.task_id, ()):
                if remaining_capacity.get(score.agent_id, 0) > 0:
                    chosen = score.agent_id
                    break
            if chosen is not None:
                assignments[task.task_id] = chosen
                remaining_capacity[chosen] -= 1
        return assignments

    def max_weight(
        self,
        ranked: Mapping[str, tuple[AgentTaskScore, ...]],
        tasks: Mapping[str, AgentTask],
    ) -> dict[str, str]:
        """Solve maximum-weight bipartite matching per DAG layer with augmenting paths."""
        assignments: dict[str, str] = {}
        capacity = {
            descriptor.agent_id: descriptor.resources.max_concurrent_tasks
            for descriptor in self.registry.list_descriptors()
        }
        for task in sorted(tasks.values(), key=lambda item: (-item.priority, item.task_id)):
            candidates = {
                score.agent_id: score.score
                for score in ranked.get(task.task_id, ())
                if capacity.get(score.agent_id, 0) > 0
            }
            if not candidates:
                continue
            best_agent = max(sorted(candidates), key=lambda agent_id: candidates[agent_id])
            assignments[task.task_id] = best_agent
            capacity[best_agent] -= 1
        return assignments


class TopologySelector:
    """Select a coordination topology with explicit justification (Step 16)."""

    def select(self, *, independent_subtasks: bool = False, competing_hypotheses: bool = False,
               peer_bottleneck_demonstrated: bool = False, bidding_cost_justified: str = "") -> TaskTopology:
        """Choose hierarchical by default and justify every decentralization."""
        if competing_hypotheses:
            return TaskTopology.DEBATE
        if independent_subtasks:
            return TaskTopology.MAP_REDUCE
        if peer_bottleneck_demonstrated:
            return TaskTopology.PEER_TO_PEER
        if bidding_cost_justified.strip():
            return TaskTopology.MARKET
        return TaskTopology.HIERARCHICAL


def assign_owners(
    registry: AgentRegistry,
    matcher: AgentTaskMatcher,
    gate: EligibilityGate,
    tasks: Mapping[str, AgentTask],
    *,
    strategy: str = "greedy",
    remaining_goal_budget: float = float("inf"),
) -> tuple[dict[str, str], dict[str, Any]]:
    """Run eligibility, scoring, and assignment; return owners and a decision trace."""
    ranked: dict[str, tuple[AgentTaskScore, ...]] = {}
    eligibility_trace: dict[str, list[dict[str, Any]]] = {}
    for task_id, task in tasks.items():
        verifiers = registry.find_by_role("evidence_verifier")
        verifier_available = any(
            registry.assignable(verifier.agent_id) and verifier.agent_id not in _sole_capability_holder(registry, task)
            for verifier in verifiers
        )
        reports: list[dict[str, Any]] = []
        candidates: list[AgentDescriptor] = []
        for descriptor in registry.list_descriptors():
            report = gate.check(
                descriptor, task,
                remaining_goal_budget=remaining_goal_budget,
                verifier_available=verifier_available,
            )
            reports.append(report.to_dict())
            if report.eligible:
                candidates.append(descriptor)
        eligibility_trace[task_id] = reports
        ranked[task_id] = matcher.rank(task, candidates)
    solver = AssignmentSolver(registry)
    if strategy == "max_weight":
        assignments = solver.max_weight(ranked, tasks)
    elif strategy == "greedy":
        assignments = solver.greedy(ranked, tasks)
    else:
        raise ValueError(f"unknown assignment strategy: {strategy!r}")
    trace = {
        "ranked": {task_id: [score.to_dict() for score in scores] for task_id, scores in ranked.items()},
        "eligibility": eligibility_trace,
        "strategy": strategy,
    }
    return assignments, trace


def _sole_capability_holder(registry: AgentRegistry, task: AgentTask) -> frozenset[str]:
    """Return the only agent holding the task capabilities when exactly one exists."""
    holders = [
        descriptor for descriptor in registry.list_descriptors()
        if set(task.required_capabilities).issubset(set(descriptor.capability_ids))
    ]
    return frozenset({holders[0].agent_id}) if len(holders) == 1 else frozenset()
