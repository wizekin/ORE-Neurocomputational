"""Partial-order planning constraints for the ORE Point 12 planning layer.

The builder constructs a directed dependency graph from preconditions, causal
links, and data dependencies. Independent operations are left unordered so they
can run concurrently; edges are added only when required. Mutex and exclusive
resource conflicts serialize only the conflicting operations, protection
conditions keep intermediate results stable until consumers run, and the
resulting plan must remain a DAG with a well-defined ready frontier.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise

from .contracts import (
    DependencyType,
    OperatorSpec,
    PlanEdge,
    PlanNode,
    PlanNodeStatus,
    Predicate,
    TaskSpec,
)
from .htn import TaskTree


@dataclass(frozen=True, slots=True)
class ProtectionInterval:
    """Keep one produced fact stable between its producer and its consumer."""

    producer: str
    consumer: str
    protected_fact: str

    def to_dict(self) -> dict:
        """Return the protection record."""
        return {
            "producer": self.producer,
            "consumer": self.consumer,
            "protected_fact": self.protected_fact,
        }


@dataclass(frozen=True, slots=True)
class PartialOrderPlan:
    """Hold one validated partial-order plan with frontier and concurrency groups."""

    plan_id: str
    goal_id: str
    nodes: tuple[PlanNode, ...]
    edges: tuple[PlanEdge, ...]
    protections: tuple[ProtectionInterval, ...]
    serialized_conflicts: tuple[tuple[str, str, str], ...]
    ready_frontier: tuple[str, ...]
    concurrency_levels: tuple[tuple[str, ...], ...]
    cycle_nodes: tuple[str, ...]
    workspace_revision: int

    def __post_init__(self) -> None:
        """Normalize plan collections."""
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "protections", tuple(self.protections))
        object.__setattr__(
            self, "serialized_conflicts", tuple(self.serialized_conflicts),
        )
        object.__setattr__(self, "ready_frontier", tuple(self.ready_frontier))
        object.__setattr__(self, "concurrency_levels", tuple(self.concurrency_levels))
        object.__setattr__(self, "cycle_nodes", tuple(self.cycle_nodes))

    @property
    def valid(self) -> bool:
        """Indicate whether the plan is a valid DAG without unresolved cycles."""
        return not self.cycle_nodes

    def node(self, node_id: str) -> PlanNode:
        """Return one plan node by identifier."""
        return next(item for item in self.nodes if item.node_id == node_id)

    def to_dict(self) -> dict:
        """Return a detached, JSON-compatible plan record."""
        return {
            "plan_id": self.plan_id,
            "goal_id": self.goal_id,
            "nodes": [item.to_dict() for item in self.nodes],
            "edges": [item.to_dict() for item in self.edges],
            "protections": [item.to_dict() for item in self.protections],
            "serialized_conflicts": [
                {"first": a, "second": b, "resource": r}
                for a, b, r in self.serialized_conflicts
            ],
            "ready_frontier": list(self.ready_frontier),
            "concurrency_levels": [list(level) for level in self.concurrency_levels],
            "cycle_nodes": list(self.cycle_nodes),
            "workspace_revision": self.workspace_revision,
            "valid": self.valid,
        }


class PartialOrderBuilder:
    """Build dependency DAGs from decomposition trees, operators, and initial facts."""

    def __init__(
        self,
        operators: Mapping[str, OperatorSpec],
        *,
        serial_ordering: bool = False,
    ) -> None:
        """Bind the operator pool and default ordering semantics for method constraints."""
        self.operators = dict(operators)
        self.serial_ordering = serial_ordering

    def _operators_for_capability(self, capability: str) -> tuple[OperatorSpec, ...]:
        """Return every eligible operator providing one capability."""
        return tuple(
            operator for operator in self.operators.values()
            if operator.module_capability == capability
        )

    def build(
        self,
        plan_id: str,
        goal_id: str,
        tree: TaskTree,
        initial_facts: frozenset[str] | set[str],
        *,
        workspace_revision: int = 0,
    ) -> PartialOrderPlan:
        """Construct the partial-order constraint graph for one decomposition tree."""
        nodes: list[PlanNode] = []
        sibling_index = 0
        for leaf in tree.leaves():
            if leaf.status is PlanNodeStatus.UNSATISFIED:
                continue
            task = leaf.task
            sibling_index += 1
            node_id = f"{plan_id}_n{len(nodes) + 1:02d}_{task.task_type.replace('.', '_')}"
            providers = self._operators_for_capability(task.task_type)
            effects: tuple[Predicate, ...] = ()
            if providers:
                effects = (Predicate("goal.satisfied", (task.task_type,)),)
            nodes.append(PlanNode(
                node_id=node_id,
                task=TaskSpec(
                    task_id=task.task_id,
                    task_type=task.task_type,
                    parameters=dict(task.parameters),
                    completion=task.completion,
                    compound=False,
                    operator_id=task.operator_id,
                ),
                operator_id=task.operator_id,
                parameters=dict(task.parameters),
                status=PlanNodeStatus.PENDING,
                expected_effects=effects,
                provenance=leaf.provenance,
            ))
        facts = frozenset(initial_facts)
        edges = self._causal_edges(nodes, self.operators, facts)
        if self.serial_ordering and len(nodes) > 1:
            edges = self._merge_edges(edges, self._sibling_ordering_edges(nodes))
        conflicts = self._resource_conflicts(nodes, self.operators)
        edges = self._merge_edges(edges, self._conflict_edges(nodes, conflicts))
        edges, protections = self._protection_intervals(nodes, edges, self.operators, facts)
        levels, cycles = _topological_levels(nodes, edges)
        frontier = levels[0] if levels else ()
        return PartialOrderPlan(
            plan_id=plan_id,
            goal_id=goal_id,
            nodes=tuple(nodes),
            edges=tuple(edges),
            protections=protections,
            serialized_conflicts=tuple(conflicts),
            ready_frontier=frontier,
            concurrency_levels=levels,
            cycle_nodes=cycles,
            workspace_revision=workspace_revision,
        )

    @staticmethod
    def _merge_edges(
        existing: Iterable[PlanEdge], added: Iterable[PlanEdge]
    ) -> list[PlanEdge]:
        """Merge edge collections deterministically, keeping the first reason per pair."""
        merged: dict[tuple[str, str], PlanEdge] = {}
        for edge in (*existing, *added):
            key = (edge.predecessor, edge.successor)
            if key not in merged:
                merged[key] = edge
        return [merged[key] for key in sorted(merged)]

    @staticmethod
    def _causal_edges(
        nodes: list[PlanNode],
        operators: Mapping[str, OperatorSpec],
        facts: frozenset[str],
    ) -> list[PlanEdge]:
        """Add only the edges required by producer-consumer causal dependencies."""
        edges: list[PlanEdge] = []
        for consumer in nodes:
            for precondition_atom in _precondition_atoms(consumer, operators, facts):
                if precondition_atom in facts:
                    continue
                producers = [
                    candidate for candidate in nodes
                    if _provides_fact(candidate, operators, precondition_atom)
                ]
                for producer in sorted(producers, key=lambda item: item.node_id):
                    if producer.node_id == consumer.node_id:
                        continue
                    edges.append(PlanEdge(
                        predecessor=producer.node_id,
                        successor=consumer.node_id,
                        dependency_type=DependencyType.CAUSAL,
                        causal_link=precondition_atom,
                    ))
        return PartialOrderBuilder._merge_edges(edges, ())

    def _sibling_ordering_edges(self, nodes: list[PlanNode]) -> list[PlanEdge]:
        """Serialize siblings sequentially when the chosen method demands ordering."""
        edges: list[PlanEdge] = []
        for first, second in pairwise(nodes):
            edges.append(PlanEdge(
                predecessor=first.node_id,
                successor=second.node_id,
                dependency_type=DependencyType.ORDERING,
                causal_link=None,
            ))
        return edges

    @staticmethod
    def _resource_conflicts(
        nodes: list[PlanNode],
        operators: Mapping[str, OperatorSpec],
    ) -> list[tuple[str, str, str]]:
        """Detect exclusive-resource conflicts between pairs of plan nodes.

        A capability-bound node holds the union of the exclusive resources declared
        by every eligible operator for its capability, so serialization is imposed
        before the search engine commits to a concrete provider.
        """
        conflicts: list[tuple[str, str, str]] = []
        by_resource: dict[str, list[PlanNode]] = {}
        for node in nodes:
            resources = {
                resource
                for operator in operators.values()
                if operator.module_capability == node.task.task_type
                for resource in operator.resource_requirements.exclusive_resources
            }
            for resource in sorted(resources):
                by_resource.setdefault(resource, []).append(node)
        for resource in sorted(by_resource):
            holders = sorted(by_resource[resource], key=lambda item: item.node_id)
            for index, first in enumerate(holders):
                for second in holders[index + 1:]:
                    conflicts.append((first.node_id, second.node_id, resource))
        return conflicts

    def _conflict_edges(
        self, nodes: list[PlanNode], conflicts: Iterable[tuple[str, str, str]]
    ) -> list[PlanEdge]:
        """Serialize only the conflicting operations in deterministic node order."""
        del nodes
        edges: list[PlanEdge] = []
        for first, second, resource in sorted(conflicts):
            edges.append(PlanEdge(
                predecessor=first,
                successor=second,
                dependency_type=DependencyType.RESOURCE,
                causal_link=resource,
            ))
        return edges

    @staticmethod
    def _protection_intervals(
        nodes: list[PlanNode],
        edges: list[PlanEdge],
        operators: Mapping[str, OperatorSpec],
        facts: frozenset[str],
    ) -> tuple[list[PlanEdge], tuple[ProtectionInterval, ...]]:
        """Protect produced facts until their consumers have executed."""
        protections: list[ProtectionInterval] = []
        for edge in edges:
            if edge.dependency_type is not DependencyType.CAUSAL or not edge.causal_link:
                continue
            protections.append(ProtectionInterval(
                producer=edge.predecessor,
                consumer=edge.successor,
                protected_fact=edge.causal_link,
            ))
        del nodes, operators, facts
        return edges, tuple(protections)


def _precondition_atoms(
    node: PlanNode,
    operators: Mapping[str, OperatorSpec],
    facts: frozenset[str],
) -> list[str]:
    """Collect the precondition atoms required by any operator of this node's capability."""
    del facts
    atoms: list[str] = []
    for operator in operators.values():
        if operator.module_capability != node.task.task_type:
            continue
        atoms.extend(precondition.to_atom() for precondition in operator.preconditions)
    return sorted(set(atoms))


def _provides_fact(
    node: PlanNode, operators: Mapping[str, OperatorSpec], fact: str
) -> bool:
    """Check whether any operator of this node's capability produces the fact."""
    return any(
        operator.provides(fact)
        for operator in operators.values()
        if operator.module_capability == node.task.task_type
    )


def _topological_levels(
    nodes: list[PlanNode], edges: list[PlanEdge]
) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...]]:
    """Apply Kahn's algorithm in parallelizable layers and expose cycle-blocked nodes."""
    remaining = {node.node_id for node in nodes}
    connections = tuple(edges)
    levels: list[tuple[str, ...]] = []
    while remaining:
        ready = sorted(
            node_id for node_id in remaining
            if not any(
                edge.successor == node_id and edge.predecessor in remaining
                for edge in connections
            )
        )
        if not ready:
            break
        levels.append(tuple(ready))
        remaining.difference_update(ready)
    return tuple(levels), tuple(sorted(remaining))


def recompute_frontier(
    plan: PartialOrderPlan,
    completed: Iterable[str],
) -> tuple[str, ...]:
    """Return the ready frontier after the given nodes completed successfully."""
    done = set(completed)
    ready: list[str] = []
    for node in plan.nodes:
        if node.node_id in done or node.status is PlanNodeStatus.UNSATISFIED:
            continue
        predecessors = {
            edge.predecessor for edge in plan.edges if edge.successor == node.node_id
        }
        if predecessors <= done:
            ready.append(node.node_id)
    return tuple(sorted(ready))


__all__ = [
    "PartialOrderBuilder", "PartialOrderPlan", "ProtectionInterval", "recompute_frontier",
]
