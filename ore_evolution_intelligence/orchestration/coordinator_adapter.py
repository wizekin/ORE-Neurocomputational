"""Coordinator adapter and the controlled configuration interface (section 5).

AdaptiveResourceCoordinator is the *foundation* for resource adaptation, not evidence
that evolutionary strategy discovery exists (section 5 note). Approved strategies reach
the coordinator only through a controlled interface: the adapter converts an APPROVED /
CANARY / ACTIVE strategy's genome into a :class:`CoordinatorConfiguration`, applies it
through :class:`ControlledConfigurationInterface` with version pinning, and never
mutates production state directly (step 38; acceptance test 1: candidate generation
cannot change production configuration).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..domain.actions import ORE_MODULES, ROUTE_QUEUES
from ..domain.genome import ParameterGenome
from ..domain.strategy import PRODUCTION_ELIGIBLE_STATES, StrategyCandidate


@dataclass(frozen=True, slots=True)
class CoordinatorParameterTarget:
    """One bounded, narrow optimization target (steps 1-2).

    Evolution optimizes *one measurable behavior* of *one* bounded surface — never the
    entire architecture or unrelated components at once.
    """

    name: str
    description: str
    measurable_behavior: str
    genome_kind: str
    permitted_actions: tuple[str, ...] = (
        "activate_module", "route_task", "request_retrieval", "escalate_verification",
    )

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.measurable_behavior.strip():
            raise ValueError("target needs a name and one measurable behavior")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "measurable_behavior": self.measurable_behavior,
            "genome_kind": self.genome_kind,
            "permitted_actions": list(self.permitted_actions),
        }


#: Default target: module-selection weights of the simulated coordinator (step 10).
MODULE_SELECTION_TARGET = CoordinatorParameterTarget(
    name="module_selection_weights",
    description=(
        "Module-selection weights, activation thresholds, routing priorities and "
        "resource budgets of the simulated AdaptiveResourceCoordinator surface"
    ),
    measurable_behavior="verified task quality under latency/cost constraints",
    genome_kind="parameters",
)

#: Policy-table target (finite decision policy).
ROUTING_POLICY_TARGET = CoordinatorParameterTarget(
    name="task_routing_policy",
    description="Finite context-conditioned routing and activation policy table",
    measurable_behavior="verified task quality under latency/cost constraints",
    genome_kind="policy_table",
)

#: Constrained-GP target (typed policy IR).
PLANNING_HEURISTIC_TARGET = CoordinatorParameterTarget(
    name="planning_heuristic",
    description="Typed expression-tree planning heuristic compiled to declarative policy IR",
    measurable_behavior="verified task quality under latency/cost constraints",
    genome_kind="policy_ir",
)


@dataclass(frozen=True, slots=True)
class CoordinatorConfiguration:
    """Bounded configuration applied to the simulated coordinator (section 5)."""

    strategy_id: str | None
    weights: Mapping[str, float] = field(default_factory=dict)
    thresholds: Mapping[str, float] = field(default_factory=dict)
    route_weights: Mapping[str, float] = field(default_factory=dict)
    resource_budgets: Mapping[str, float] = field(default_factory=dict)
    discrete: Mapping[str, Any] = field(default_factory=dict)
    policy_rules: tuple[Mapping[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "weights": dict(self.weights),
            "thresholds": dict(self.thresholds),
            "route_weights": dict(self.route_weights),
            "resource_budgets": dict(self.resource_budgets),
            "discrete": dict(self.discrete),
            "policy_rules": [dict(rule) for rule in self.policy_rules],
        }


def baseline_configuration() -> CoordinatorConfiguration:
    """The deterministic baseline configuration (step 3: immutable rollback target)."""
    return CoordinatorConfiguration(
        strategy_id=None,
        weights={module: 0.6 for module in ORE_MODULES},
        thresholds={module: 0.5 for module in ORE_MODULES},
        route_weights={queue: 1.0 / 3.0 for queue in ROUTE_QUEUES},
        resource_budgets={"cpu_budget_share": 0.6, "memory_budget_share": 0.6},
        discrete={"retrieval_depth": 1, "escalation_mode": "balanced"},
    )


class ControlledConfigurationInterface:
    """The only write path from approved strategies to coordinator configuration.

    Keeps the immutable rollback target, pins versions, records every application and
    refuses strategies that are not production-eligible (acceptance test 1).
    """

    def __init__(self, *, event_recorder: Any | None = None, run_id: str = "run") -> None:
        from ..registry.versioning import RollbackTarget
        self._current = baseline_configuration()
        self._history: list[dict[str, Any]] = []
        self._event_recorder = event_recorder
        self._run_id = run_id
        self.rollback_target = RollbackTarget(
            strategy_id=None,
            content_hash="baseline",
            configuration=self._current.as_dict(),
            benchmark_version="",
            benchmark_results={},
            source_version="baseline-v1",
        )

    @property
    def current(self) -> CoordinatorConfiguration:
        return self._current

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def apply(self, configuration: CoordinatorConfiguration, candidate: StrategyCandidate) -> CoordinatorConfiguration:
        """Apply one configuration for a production-eligible strategy (step 38)."""
        if not candidate.is_production_eligible():
            raise PermissionError(
                f"strategy {candidate.candidate_id} in state {candidate.state} is not "
                "production-eligible; only APPROVED/CANARY/ACTIVE strategies may be applied"
            )
        self._current = configuration
        entry = {
            "strategy_id": configuration.strategy_id,
            "candidate_id": candidate.candidate_id,
            "state": candidate.state,
            "version": candidate.version,
            "content_hash": candidate.content_hash,
        }
        self._history.append(entry)
        if self._event_recorder is not None:
            self._event_recorder.emit(
                "promotion_decided", self._run_id, entry,
            )
        return self._current

    def restore_rollback_target(self) -> CoordinatorConfiguration:
        """Restore the pinned baseline configuration (acceptance test 34)."""
        from ..domain.genome import content_digest
        payload = dict(self.rollback_target.configuration)
        restored = CoordinatorConfiguration(
            strategy_id=None,
            weights=payload.get("weights", {}),
            thresholds=payload.get("thresholds", {}),
            route_weights=payload.get("route_weights", {}),
            resource_budgets=payload.get("resource_budgets", {}),
            discrete=payload.get("discrete", {}),
            policy_rules=tuple(payload.get("policy_rules", ())),
        )
        self._current = restored
        self._history.append({
            "strategy_id": None,
            "candidate_id": None,
            "state": "ROLLBACK",
            "content_hash": content_digest(payload),
        })
        return restored


class CoordinatorAdapter:
    """Convert approved strategy genomes into coordinator configurations (section 5)."""

    def __init__(self, controlled_interface: ControlledConfigurationInterface) -> None:
        self.interface = controlled_interface

    def configuration_for(self, candidate: StrategyCandidate) -> CoordinatorConfiguration:
        """Translate one candidate's genome into a bounded configuration."""
        if not candidate.is_production_eligible():
            raise PermissionError(
                f"strategy {candidate.candidate_id} in state {candidate.state} cannot "
                "be translated into a production configuration"
            )
        genome = candidate.genome
        if genome.kind == "parameters":
            weights = {
                module: float(genome.values.get(f"weight__{module}", 0.6))
                for module in ORE_MODULES
            }
            thresholds = {
                module: float(genome.values.get(f"threshold__{module}", 0.5))
                for module in ORE_MODULES
            }
            route_weights = {
                queue: float(genome.values.get(f"route_weight__{queue}", 1.0 / 3.0))
                for queue in ROUTE_QUEUES
            }
            resource_budgets = {
                "cpu_budget_share": float(genome.values.get("cpu_budget_share", 0.6)),
                "memory_budget_share": float(genome.values.get("memory_budget_share", 0.6)),
            }
            discrete = {
                "retrieval_depth": int(genome.values.get("retrieval_depth", 1)),
                "escalation_mode": genome.values.get("escalation_mode", "balanced"),
            }
            return CoordinatorConfiguration(
                strategy_id=candidate.candidate_id,
                weights=weights, thresholds=thresholds,
                route_weights=route_weights, resource_budgets=resource_budgets,
                discrete=discrete,
            )
        if genome.kind in ("policy_table", "policy_ir"):
            rules = tuple(
                {"cell": "|".join(key), "actions": [a.name for a in actions]}
                for key, actions in sorted(getattr(genome, "rules", {}).items())
            )
            return CoordinatorConfiguration(
                strategy_id=candidate.candidate_id, policy_rules=rules,
            )
        raise ValueError(f"cannot adapt genome kind {genome.kind!r}")

    def apply_approved(self, candidate: StrategyCandidate) -> CoordinatorConfiguration:
        """Translate and apply one production-eligible strategy (controlled path)."""
        configuration = self.configuration_for(candidate)
        return self.interface.apply(configuration, candidate)


def genome_from_configuration(configuration: CoordinatorConfiguration) -> ParameterGenome:
    """Rebuild the parameter genome matching one configuration (baseline replay)."""
    from ..domain.genome import default_search_space
    values: dict[str, Any] = {}
    for module in ORE_MODULES:
        values[f"weight__{module}"] = float(configuration.weights.get(module, 0.6))
        values[f"threshold__{module}"] = float(configuration.thresholds.get(module, 0.5))
    for queue in ROUTE_QUEUES:
        values[f"route_weight__{queue}"] = float(
            configuration.route_weights.get(queue, 1.0 / 3.0)
        )
    values.update({
        "cpu_budget_share": float(
            configuration.resource_budgets.get("cpu_budget_share", 0.6)
        ),
        "memory_budget_share": float(
            configuration.resource_budgets.get("memory_budget_share", 0.6)
        ),
    })
    values.update(dict(configuration.discrete))
    return ParameterGenome(values, default_search_space())


__all__ = [
    "MODULE_SELECTION_TARGET", "PLANNING_HEURISTIC_TARGET", "ROUTING_POLICY_TARGET",
    "ControlledConfigurationInterface", "CoordinatorAdapter",
    "CoordinatorConfiguration", "CoordinatorParameterTarget",
    "baseline_configuration", "genome_from_configuration",
]
