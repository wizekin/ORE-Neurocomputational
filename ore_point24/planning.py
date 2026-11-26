"""Phase IV, Steps 18-19 - validate the dependency DAG and select modules.

Step 18 resolves providers for required input ports by exact contract IDs and
versions, detects cycles with Kahn's algorithm, topologically sorts the graph
into execution layers and marks missing providers as planning gaps instead of
fabricating backends.  Step 19 selects a module set under relevance,
reliability and resource constraints - no score may override a hard
permission, safety, dependency or resource constraint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ore_contracts.module_registry import ModuleRegistry
from ore_contracts.resource_coordination import ModuleResourceEstimate

from .compatibility import check_port_compatibility
from .core import new_id, sha256_digest, utc_now_iso
from .readiness import ReadinessTracker
from .decomposition import TaskGraph

PLANNING_SCHEMA = "ore.point24.planning.v1"


@dataclass(frozen=True, slots=True)
class ExecutionLayer:
    """One topological layer: all nodes whose dependencies are satisfied."""

    index: int
    node_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlanningGap:
    """A missing or incompatible provider recorded as an explicit gap."""

    node_id: str
    required_contract: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "required_contract": self.required_contract,
                "reason": self.reason}


@dataclass(slots=True)
class ValidatedPlan:
    """The Step 18 result: layers, port matches and explicit gaps."""

    plan_id: str = field(default_factory=lambda: new_id("point24_plan"))
    created_at: str = field(default_factory=utc_now_iso)
    layers: list[ExecutionLayer] = field(default_factory=list)
    gaps: list[PlanningGap] = field(default_factory=list)
    port_matches: list[dict[str, Any]] = field(default_factory=list)
    cycle_detected: bool = False
    executable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLANNING_SCHEMA,
            "plan_id": self.plan_id,
            "created_at": self.created_at,
            "cycle_detected": self.cycle_detected,
            "executable": self.executable,
            "layers": [
                {"index": layer.index, "node_ids": list(layer.node_ids)}
                for layer in self.layers
            ],
            "gaps": [gap.to_dict() for gap in self.gaps],
            "port_matches": list(self.port_matches),
        }


def kahn_layers(nodes: Iterable[str], edges: Mapping[str, Iterable[str]]) -> tuple[list[ExecutionLayer], bool]:
    """Kahn's algorithm: return execution layers and whether a cycle exists.

    ``edges`` maps each node to the nodes it depends on.  At each iteration
    only nodes whose in-degree is zero relative to unfinished required
    dependencies enter the next layer; a non-empty residual graph with no
    zero-in-degree node indicates a cycle.
    """
    node_set = set(nodes)
    dependents: dict[str, set[str]] = {node: set() for node in node_set}
    indegree: dict[str, int] = {node: 0 for node in node_set}
    for consumer, producers in edges.items():
        if consumer not in node_set:
            continue
        for producer in producers:
            if producer in node_set and producer != consumer:
                if consumer not in dependents[producer]:
                    dependents[producer].add(consumer)
                    indegree[consumer] += 1
            elif producer == consumer:
                indegree[consumer] += 1  # self-loop is a cycle
    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    layers: list[ExecutionLayer] = []
    processed = 0
    index = 0
    while ready:
        layers.append(ExecutionLayer(index=index, node_ids=tuple(sorted(ready))))
        processed += len(ready)
        next_ready: list[str] = []
        for node in ready:
            for dependent in sorted(dependents[node]):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    next_ready.append(dependent)
        ready = sorted(set(next_ready))
        index += 1
    cycle = processed < len(node_set)
    return layers, cycle


class DependencyPlanValidator:
    """Step 18 - build and validate the dependency DAG against the registry."""

    def __init__(self, registry: ModuleRegistry) -> None:
        self.registry = registry
        self.readiness = ReadinessTracker(registry)

    def validate(self, graph: TaskGraph) -> ValidatedPlan:
        """Topologically order the task graph and record exact port matches."""
        plan = ValidatedPlan()
        layers, cycle = kahn_layers(graph.nodes.keys(), graph.edges)
        plan.layers = layers
        plan.cycle_detected = cycle

        for node in graph.nodes.values():
            if node.node_type == "verification":
                continue
            capability = node.required_capability
            if capability is None:
                continue
            records = self.registry.find_by_capability(capability)
            runnable = [record for record in records if record.runnable]
            if not runnable:
                plan.gaps.append(PlanningGap(
                    node_id=node.node_id,
                    required_contract=node.produces_contract,
                    reason=(
                        "no runnable provider declares capability "
                        f"{capability!r} (missing backend reported as a planning gap; a backend is never invented)"
                    ),
                ))
                continue
            for record in runnable:
                manifest = self.registry.get_manifest(record.module_id)
                if manifest is None or node.produces_contract is None:
                    continue
                for port in manifest.output_ports:
                    if port.contract_id == node.produces_contract:
                        plan.port_matches.append({
                            "node_id": node.node_id,
                            "module_id": record.module_id,
                            "port_id": port.port_id,
                            "contract": f"{port.contract_id}@{port.contract_version}",
                        })
        plan.executable = not cycle and not plan.gaps
        return plan


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    """One Step 19 selection or exclusion with its recorded reason."""

    module_id: str
    selected: bool
    reason: str
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"module_id": self.module_id, "selected": self.selected,
                "reason": self.reason, "score": self.score}


class ConstrainedModuleSelector:
    """Step 19 - select modules under relevance, reliability and budgets.

    Hard constraints (permissions, safety, dependencies, resource ceilings)
    are removed *before* scoring; the operational score follows the documented
    resource-selection form ``Score(m) = P(G_m + lambda_I I_m) / (eps + C_m)
    - lambda_F F_m`` whose information-gain term stays heuristic unless
    separately calibrated.
    """

    def __init__(self, registry: ModuleRegistry, *, epsilon: float = 1e-6) -> None:
        self.registry = registry
        self.readiness = ReadinessTracker(registry)
        self.epsilon = float(epsilon)

    def select(
        self,
        *,
        required_capabilities: Iterable[str],
        measured_costs: Mapping[str, Mapping[str, float]] | None = None,
        reliability: Mapping[str, float] | None = None,
        budget: Mapping[str, float] | None = None,
        priority_weight: float = 1.0,
        information_gain_weight: float = 0.5,
        failure_penalty_weight: float = 1.0,
        max_selected: int | None = None,
    ) -> list[SelectionDecision]:
        """Return per-capability selection decisions with recorded reasons."""
        decisions: list[SelectionDecision] = []
        selected: list[str] = []
        for capability in required_capabilities:
            records = self.registry.find_by_capability(capability)
            if not records:
                decisions.append(SelectionDecision(
                    module_id=f"<missing:{capability}>", selected=False,
                    reason="no module declares this capability",
                ))
                continue
            best: tuple[float, str] | None = None
            excluded: list[SelectionDecision] = []
            for record in records:
                readiness = self.readiness.inspect(record.module_id)
                if not readiness.runnable:
                    excluded.append(SelectionDecision(
                        module_id=record.module_id, selected=False,
                        reason="; ".join(readiness.failed_conditions()) or "not runnable",
                    ))
                    continue
                if record.disabled:
                    excluded.append(SelectionDecision(
                        module_id=record.module_id, selected=False, reason="module disabled",
                    ))
                    continue
                costs = (measured_costs or {}).get(record.module_id, {})
                cost = self._normalized_cost(costs, budget)
                reliability_value = float((reliability or {}).get(record.module_id, 0.5))
                failure_penalty = 1.0 - reliability_value
                gain = priority_weight
                info = information_gain_weight  # heuristic first-version estimate
                score = (gain + info) / (self.epsilon + cost) - failure_penalty_weight * failure_penalty
                if budget and self._exceeds_budget(costs, budget):
                    excluded.append(SelectionDecision(
                        module_id=record.module_id, selected=False,
                        reason="hard resource constraint violated; score cannot override it",
                        score=score,
                    ))
                    continue
                if best is None or score > best[0]:
                    best = (score, record.module_id)
            decisions.extend(excluded)
            if best is not None:
                decisions.append(SelectionDecision(
                    module_id=best[1], selected=True,
                    reason="highest resource-aware score among runnable, compatible providers",
                    score=best[0],
                ))
                selected.append(best[1])
                if max_selected is not None and len(selected) >= max_selected:
                    break
            else:
                decisions.append(SelectionDecision(
                    module_id=f"<no-runnable:{capability}>", selected=False,
                    reason="all providers excluded by hard constraints",
                ))
        return decisions

    @staticmethod
    def _normalized_cost(costs: Mapping[str, float], budget: Mapping[str, float] | None) -> float:
        """Normalize measured (or declared) cost into (0, 1]; unknown = 1.0."""
        if not costs:
            return 1.0
        reference = budget or {}
        ratios = []
        for key, value in costs.items():
            limit = float(reference.get(key, 0) or 0)
            if limit > 0:
                ratios.append(min(float(value) / limit, 1.0))
        if not ratios:
            # No budget reference: normalize against the largest measured cost.
            peak = max(float(v) for v in costs.values())
            return peak if peak > 0 else 1.0
        return max(0.0, min(max(ratios), 1.0)) or 1e-6

    @staticmethod
    def _exceeds_budget(costs: Mapping[str, float], budget: Mapping[str, float]) -> bool:
        for key, value in costs.items():
            limit = budget.get(key)
            if limit is not None and float(value) > float(limit):
                return True
        return False


__all__ = [
    "PLANNING_SCHEMA",
    "ConstrainedModuleSelector",
    "DependencyPlanValidator",
    "ExecutionLayer",
    "PlanningGap",
    "SelectionDecision",
    "ValidatedPlan",
    "kahn_layers",
]
