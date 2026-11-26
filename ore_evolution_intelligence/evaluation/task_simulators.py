"""Deterministic ORE coordinator simulator (integration map, section 5).

The simulator models the bounded coordinator surface that Point 9 optimizes:
AdaptiveResourceCoordinator-style module selection, activation thresholds, routing
priorities and resource budgets, executed by CognitiveTaskCoordinator and scheduled by
ModuleScheduler. It is the *evaluator-side model*: candidate genomes are data
interpreted here — they never execute code, cannot touch scoring, labels, the clock or
task selection (step 14).

Deterministic given (genome, task, seed): the same candidate, task and seed always
produce the same raw observation, which makes fixed-seed experiments reproducible
within declared tolerance (acceptance test 35).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..domain.actions import ESCALATION_MODES, ORE_MODULES, ROUTE_QUEUES, ActionCall
from ..domain.evaluation_record import OUTCOME_CREDIT
from .benchmark_manifest import TaskCase

#: Per-module profile: latency base (ms), memory (MB), cost units.
MODULE_PROFILES: dict[str, tuple[float, float, float]] = {
    "ore_lexical_access": (32.0, 12.0, 4.0),
    "ore_syntax": (44.0, 14.0, 5.0),
    "ore_semantic": (46.0, 16.0, 6.0),
    "ore_semantic_composition": (52.0, 18.0, 7.0),
    "ore_discourse": (58.0, 18.0, 7.0),
    "ore_working_memory": (26.0, 10.0, 3.0),
    "ore_vwfa": (36.0, 12.0, 4.0),
    "ore_reading": (62.0, 20.0, 8.0),
    "ore_hippocampal_encoding": (66.0, 22.0, 8.0),
    "ore_sleep_consolidation": (74.0, 24.0, 9.0),
    "verification_monitor": (22.0, 8.0, 3.0),
}

#: Module relevance per task family (0..1): which modules help which families.
FAMILY_RELEVANCE: dict[str, dict[str, float]] = {
    "lexical_lookup": {"ore_lexical_access": 0.95, "ore_vwfa": 0.7, "ore_working_memory": 0.35},
    "semantic_composition": {
        "ore_semantic": 0.95, "ore_semantic_composition": 0.95, "ore_discourse": 0.4,
    },
    "syntax_parse": {"ore_syntax": 0.95, "ore_working_memory": 0.5},
    "reading": {"ore_reading": 0.95, "ore_vwfa": 0.5, "ore_semantic": 0.4},
    "discourse": {"ore_discourse": 0.95, "ore_semantic_composition": 0.5, "ore_working_memory": 0.4},
    "consolidation": {
        "ore_hippocampal_encoding": 0.9, "ore_sleep_consolidation": 0.85,
    },
}

#: Escalation posture activation rule (verification monitor on safety-critical tasks).
ESCALATION_COMPLEXITY_GATE: dict[str, float] = {
    "conservative": 0.75, "balanced": 0.5, "aggressive": 0.0,
}

#: Base module cap granted by the resource budget share (resource-tier behavior).
BASE_MODULE_CAP = 2
MODULE_CAP_SPAN = 6


def _module_cap(cpu_share: float, task: TaskCase) -> int:
    cap = BASE_MODULE_CAP + MODULE_CAP_SPAN * cpu_share
    if task.resource_state == "pressured":
        cap -= 2
    return max(int(cap), 1)


def _queue_for_task(task: TaskCase) -> str:
    if task.latency_sensitivity > 0.66:
        return "fast"
    if task.latency_sensitivity > 0.33:
        return "standard"
    return "deep"


def _required_coverage(task: TaskCase) -> float:
    return 0.5 + 0.4 * task.complexity


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Raw simulated execution outcome for one task (steps 13, 17)."""

    task_id: str
    actions: tuple[str, ...]
    active_modules: tuple[str, ...]
    quality: float
    outcome: str
    latency_ms: float
    memory_mb: float
    cpu_ms: float
    cost: float
    safety_violations: int
    safety_notes: tuple[str, ...] = ()
    steps: int = 0
    terminated: bool = False
    invalid_reasons: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    def outcome_credit(self) -> float:
        return OUTCOME_CREDIT[self.outcome]


class CoordinatorSimulator:
    """Deterministic simulation of the ORE coordinator surface (section 5)."""

    def __init__(self, noise_scale: float = 0.02) -> None:
        if noise_scale < 0.0:
            raise ValueError("noise scale must be non-negative")
        self.noise_scale = noise_scale

    # ------------------------------------------------------------------ dispatch
    def simulate(self, genome: Any, task: TaskCase, rng: Any) -> SimulationResult:
        """Simulate one genome on one task with one RNG state."""
        dispatch = {
            "parameters": self._simulate_parameters,
            "policy_table": self._simulate_policy_table,
            "policy_ir": self._simulate_policy_ir,
        }
        handler = dispatch.get(genome.kind)
        if handler is None:
            raise ValueError(f"simulator cannot interpret genome kind {genome.kind!r}")
        return handler(genome, task, rng)

    # ------------------------------------------------------------- parameter genome
    def _simulate_parameters(self, genome: Any, task: TaskCase, rng: Any) -> SimulationResult:
        values = genome.values
        relevance = FAMILY_RELEVANCE[task.family]
        scores: list[tuple[float, str]] = []
        for module in ORE_MODULES:
            if module == "verification_monitor":
                continue
            relevance_value = relevance.get(module, 0.05)
            score = values.get(f"weight__{module}", 0.0) * (
                relevance_value * (1.0 + 0.5 * task.complexity)
            )
            threshold = values.get(f"threshold__{module}", 1.0)
            if score >= threshold and relevance_value > 0.0:
                scores.append((score, module))
        scores.sort(reverse=True)
        cap = _module_cap(values.get("cpu_budget_share", 0.6), task)
        active = [module for _score, module in scores[:cap]]
        if not active and scores:
            active = [scores[0][1]]

        # Verification monitor posture on safety-critical tasks.
        safety_violations = 0
        safety_notes: list[str] = []
        escalation_mode = values.get("escalation_mode", "balanced")
        if escalation_mode not in ESCALATION_MODES:
            escalation_mode = "balanced"
        verification_active = (
            task.safety_critical
            and task.complexity >= ESCALATION_COMPLEXITY_GATE[escalation_mode]
        )
        if task.safety_critical and not verification_active:
            safety_violations += 1
            safety_notes.append(
                "safety-critical task executed without verification monitor "
                f"(escalation_mode={escalation_mode}, complexity={task.complexity:.2f})"
            )

        actions = [ActionCall("activate_module", {"module": module}) for module in active]
        if verification_active:
            actions.append(ActionCall("escalate_verification", {}))
        queue = _queue_for_task(task)
        route_weight = values.get(f"route_weight__{queue}", 1.0 / 3.0)
        actions.append(ActionCall("route_task", {"queue": queue}))
        retrieval_depth = int(values.get("retrieval_depth", 1))
        if task.novelty > 0.6 and retrieval_depth > 0:
            actions.append(ActionCall("request_retrieval", {"depth": retrieval_depth}))

        memory_share = values.get("memory_budget_share", 0.6)
        return self._outcome_from_activation(
            genome, task, rng, active, actions, route_weight, memory_share,
            retrieval_depth, safety_violations, safety_notes,
        )

    # ------------------------------------------------------------ policy-table genome
    def _simulate_policy_table(self, genome: Any, task: TaskCase, rng: Any) -> SimulationResult:
        actions = genome.decision_for(task.context_key())
        active: list[str] = []
        route_weight = 1.0 / 3.0
        retrieval_depth = 0
        verification_active = False
        deferred = False
        safety_violations = 0
        safety_notes: list[str] = []
        for action in sorted(actions, key=lambda a: a.name):
            if action.name == "activate_module":
                active.append(action.params["module"])
            elif action.name == "route_task":
                route_weight = max(route_weight, 0.1)  # table routes by queue choice
            elif action.name == "request_retrieval":
                retrieval_depth = int(action.params["depth"])
            elif action.name == "escalate_verification":
                verification_active = True
            elif action.name == "defer_task":
                deferred = True
        if deferred:
            return SimulationResult(
                task_id=task.task_id,
                actions=tuple(a.name for a in actions),
                active_modules=(),
                quality=OUTCOME_CREDIT["ABSTAIN"],
                outcome="ABSTAIN",
                latency_ms=8.0,
                memory_mb=4.0,
                cpu_ms=5.0,
                cost=1.0,
                safety_violations=0,
                steps=1,
                evidence={"deferred": True, "active_modules": 0},
            )
        if task.safety_critical and not verification_active:
            safety_violations += 1
            safety_notes.append(
                "safety-critical task executed without verification escalation"
            )
        memory_share = 0.6
        return self._outcome_from_activation(
            genome, task, rng, active, [a for a in actions], route_weight, memory_share,
            retrieval_depth, safety_violations, safety_notes,
        )

    # ------------------------------------------------------------------ policy-IR genome
    def _simulate_policy_ir(self, genome: Any, task: TaskCase, rng: Any) -> SimulationResult:
        context = dict(task.features())
        context["task_family_index"] = float(["lexical_lookup", "semantic_composition",
                                              "syntax_parse", "reading", "discourse",
                                              "consolidation"].index(task.family))
        action_calls, steps, terminated = genome.evaluate_actions(context)
        active: list[str] = []
        route_weight = 1.0 / 3.0
        retrieval_depth = 0
        verification_active = False
        safety_violations = 0
        safety_notes: list[str] = []
        for action in action_calls:
            if action.name == "activate_module":
                active.append(action.params["module"])
            elif action.name == "request_retrieval":
                retrieval_depth = int(action.params["depth"])
            elif action.name == "escalate_verification":
                verification_active = True
        if task.safety_critical and not verification_active:
            safety_violations += 1
            safety_notes.append(
                "safety-critical task executed without verification escalation"
            )
        if terminated:
            return SimulationResult(
                task_id=task.task_id,
                actions=tuple(a.name for a in action_calls),
                active_modules=tuple(active),
                quality=0.0,
                outcome="INVALID",
                latency_ms=0.0,
                memory_mb=0.0,
                cpu_ms=0.0,
                cost=0.0,
                safety_violations=0,
                steps=steps,
                terminated=True,
                invalid_reasons=("step budget exceeded during policy evaluation",),
                evidence={"active_modules": 0},
            )
        memory_share = 0.6
        return self._outcome_from_activation(
            genome, task, rng, active, action_calls, route_weight, memory_share,
            retrieval_depth, safety_violations, safety_notes,
        )

    # ------------------------------------------------------------- shared physics
    def _outcome_from_activation(
        self,
        genome: Any,
        task: TaskCase,
        rng: Any,
        active: Sequence[str],
        actions: Sequence[ActionCall],
        route_weight: float,
        memory_share: float,
        retrieval_depth: int,
        safety_violations: int,
        safety_notes: Sequence[str],
    ) -> SimulationResult:
        active = list(active)
        # Coverage: how much of the family's relevant module space is active.
        relevance = FAMILY_RELEVANCE[task.family]
        required = _required_coverage(task)
        top = sorted(relevance.values(), reverse=True)[:3]
        denom = sum(top) or 1.0
        coverage = sum(relevance.get(m, 0.0) for m in active if m != "verification_monitor") / denom

        # Latency: per-module base, routing multiplier, pressure penalty.
        latency = 12.0  # coordinator dispatch overhead
        memory = 6.0
        cost = 1.5
        for module in active:
            base_latency, base_memory, base_cost = MODULE_PROFILES.get(
                module, (40.0, 12.0, 5.0)
            )
            latency += base_latency
            memory += base_memory * (1.4 - 0.4 * memory_share)
            cost += base_cost
        latency += retrieval_depth * 6.0
        cost += retrieval_depth * 2.0
        route_multiplier = 1.0 / (0.5 + max(route_weight, 0.0))
        latency *= route_multiplier
        if task.resource_state == "pressured":
            latency *= 1.35
            memory *= 1.3
        latency *= 1.0 + 0.25 * task.complexity
        cpu_ms = latency * (0.8 + 0.4 * len(active) / 6.0)

        # Quality: coverage against the difficulty-required coverage. The potential
        # ceiling (~0.92 + retrieval bonus) stays below 1.0 so seeded noise keeps
        # trial-level variance — a perfectly-covered strategy is still distinguishable
        # from a benchmark-hardcoding candidate whose outputs are bit-identical.
        quality = 0.32 + 0.60 * min(coverage, 1.0)
        if coverage < required:
            quality *= 0.4 + 0.6 * (coverage / required)
        critical_missing = any(
            relevance.get(m, 0.0) >= 0.8 and m not in active for m in relevance
        )
        if critical_missing:
            quality *= 0.55
        if task.novelty > 0.6:
            quality += 0.03 * retrieval_depth
        if self.noise_scale > 0:
            quality += rng.gauss(0.0, self.noise_scale)
        quality = min(max(quality, 0.0), 1.0)

        # Outcome semantics (step 5): observable criteria only.
        invalid_reasons: list[str] = []
        outcome: str
        if safety_violations > 0:
            outcome = "INVALID"
            invalid_reasons.extend(safety_notes)
        elif latency > task.latency_budget_ms:
            outcome = "TIMEOUT"
        elif len(active) < 2 and task.complexity > 0.6:
            outcome = "ABSTAIN"
            quality = OUTCOME_CREDIT["ABSTAIN"]
        elif quality >= 0.75:
            outcome = "SUCCESS"
        elif quality >= 0.35:
            outcome = "PARTIAL"
        else:
            outcome = "FAILURE"

        return SimulationResult(
            task_id=task.task_id,
            actions=tuple(a.name for a in actions),
            active_modules=tuple(active),
            quality=quality,
            outcome=outcome,
            latency_ms=latency,
            memory_mb=memory,
            cpu_ms=cpu_ms,
            cost=cost,
            safety_violations=safety_violations,
            safety_notes=tuple(safety_notes),
            steps=max(len(active) + 1, 1),
            invalid_reasons=tuple(invalid_reasons),
            evidence={
                "active_modules": len(active),
                "coverage": round(coverage, 4),
                "required_coverage": round(required, 4),
                "critical_missing": critical_missing,
                "route_multiplier": round(route_multiplier, 4),
            },
        )


__all__ = [
    "CoordinatorSimulator", "ESCALATION_COMPLEXITY_GATE", "FAMILY_RELEVANCE",
    "MODULE_PROFILES", "SimulationResult",
]
