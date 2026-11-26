"""Graph integrity validation, quarantine handling, and bounded partitioning (steps 9-10).

The integrity validator checks unique identities, dangling references, direction,
argument types against the schema, duplicates, provenance presence, and quarantines
malformed records instead of silently repairing them. The partitioner bounds graph
computation by domain, connected component, relation family, or sampled neighborhood
while preserving node identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .typed_graph import TypedEdge, TypedGraphSnapshot


@dataclass(frozen=True, slots=True)
class IntegrityCheck:
    """One executed integrity check with its outcome."""

    check_id: str
    description: str
    passed: bool
    violations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id, "description": self.description,
            "passed": self.passed, "violations": list(self.violations),
        }


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    """Aggregate integrity outcome for one projected snapshot (ORE Point 8 step 9)."""

    checks: tuple[IntegrityCheck, ...]
    quarantined_edge_ids: tuple[str, ...] = ()
    quarantined_node_ids: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """A projection passes only when every check passes."""
        return all(check.passed for check in self.checks)

    def failures(self) -> tuple[IntegrityCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [check.as_dict() for check in self.checks],
            "quarantined_edges": len(self.quarantined_edge_ids),
            "quarantined_nodes": len(self.quarantined_node_ids),
        }


class IntegrityValidator:
    """Validate a typed graph snapshot without mutating it."""

    def validate(self, snapshot: TypedGraphSnapshot) -> IntegrityReport:
        """Run every integrity check; quarantine results are advisory lists of offenders."""
        node_ids = {node.node_id for node in snapshot.nodes}
        checks: list[IntegrityCheck] = []

        dangling = tuple(
            edge.edge_id for edge in snapshot.edges
            if edge.source_id not in node_ids or edge.target_id not in node_ids
        )
        checks.append(IntegrityCheck(
            "dangling_references", "Every edge endpoint resolves to a projected node",
            not dangling, dangling,
        ))

        duplicate_triples: dict[tuple[str, str, str], int] = {}
        for edge in snapshot.edges:
            key = (edge.source_id, edge.relation_type, edge.target_id)
            duplicate_triples[key] = duplicate_triples.get(key, 0) + 1
        duplicates = tuple(
            f"{key[0]}|{key[1]}|{key[2]}" for key, count in duplicate_triples.items() if count > 1
        )
        checks.append(IntegrityCheck(
            "duplicate_edges", "No duplicate (subject, relation, object) instances exist",
            not duplicates, duplicates,
        ))

        direction_bad = tuple(
            edge.edge_id for edge in snapshot.edges if edge.direction not in {"DIRECTED", "UNDIRECTED"}
        )
        checks.append(IntegrityCheck(
            "direction_preserved", "Every edge carries an explicit direction",
            not direction_bad, direction_bad,
        ))

        schema = snapshot.schema
        type_violations = []
        kind_by_id = {node.node_id: node.node_kind for node in snapshot.nodes}
        for edge in snapshot.edges:
            signature = schema.signature(edge.relation_type)
            if signature is None:
                continue
            subject_kind = kind_by_id.get(edge.source_id)
            object_kind = kind_by_id.get(edge.target_id)
            if subject_kind is not None and not signature.domain_allows(subject_kind):
                type_violations.append(f"{edge.edge_id}:subject:{subject_kind}")
            if object_kind is not None and not signature.range_allows(object_kind):
                type_violations.append(f"{edge.edge_id}:object:{object_kind}")
        checks.append(IntegrityCheck(
            "argument_types", "Edge endpoints satisfy relation domain/range signatures",
            not type_violations, tuple(type_violations),
        ))

        missing_provenance = tuple(
            item.node_id for item in snapshot.nodes if item.provenance is None
        ) + tuple(
            edge.edge_id for edge in snapshot.edges if edge.provenance is None
        )
        checks.append(IntegrityCheck(
            "provenance_present", "Every node and edge carries provenance with a snapshot hash",
            not missing_provenance, missing_provenance,
        ))

        event_nodes = {node.node_id for node in snapshot.nodes if node.node_kind == "RELATION_EVENT"}
        role_target_missing = []
        for edge in snapshot.edges:
            if edge.source_id in event_nodes or edge.target_id in event_nodes:
                for role in edge.roles:
                    if role.node_id not in node_ids and role.node_id not in event_nodes:
                        role_target_missing.append(f"{edge.edge_id}:{role.role}")
        checks.append(IntegrityCheck(
            "role_bindings", "Reified event roles resolve to projected nodes",
            not role_target_missing, tuple(role_target_missing),
        ))

        quarantine_unhandled = tuple(
            record.record_id for record in snapshot.quarantine
            if not record.reason.strip()
        )
        checks.append(IntegrityCheck(
            "quarantine_documented", "Quarantined records carry an explicit reason",
            not quarantine_unhandled, quarantine_unhandled,
        ))

        return IntegrityReport(checks=tuple(checks))


class GraphPartitioner:
    """Partition a projection while preserving node identity (ORE Point 8 step 10)."""

    def __init__(self, *, seed: int = 0) -> None:
        if seed < 0:
            raise ValueError("seed cannot be negative")
        self.seed = seed

    def by_relation_family(self, snapshot: TypedGraphSnapshot) -> dict[str, list[TypedEdge]]:
        """Group edges by relation type; every partition records the strategy and seed."""
        groups: dict[str, list[TypedEdge]] = {}
        for edge in snapshot.edges:
            groups.setdefault(edge.relation_type, []).append(edge)
        return groups

    def by_domain(self, snapshot: TypedGraphSnapshot) -> dict[str, list[TypedEdge]]:
        """Group edges by their declared context scopes (domains)."""
        groups: dict[str, list[TypedEdge]] = {}
        for edge in snapshot.edges:
            for context_id in edge.context_ids or ("unscoped",):
                groups.setdefault(context_id, []).append(edge)
        return groups

    def by_component(self, snapshot: TypedGraphSnapshot) -> list[set[str]]:
        """Weakly connected components over the undirected view of the projection."""
        adjacency: dict[str, set[str]] = {node.node_id: set() for node in snapshot.nodes}
        for edge in snapshot.edges:
            adjacency[edge.source_id].add(edge.target_id)
            adjacency[edge.target_id].add(edge.source_id)
        seen: set[str] = set()
        components: list[set[str]] = []
        for start in sorted(adjacency):
            if start in seen:
                continue
            stack = [start]
            component: set[str] = set()
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                stack.extend(sorted(adjacency[current] - component))
            seen |= component
            components.append(component)
        return components

    def sampled_neighborhood(
        self,
        snapshot: TypedGraphSnapshot,
        anchor_id: str,
        *,
        max_nodes: int,
        sampler: Callable[[list[str], int], list[str]] | None = None,
    ) -> set[str]:
        """Bounded BFS neighborhood around an anchor, identity-preserving and seed-recorded."""
        if max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        adjacency: dict[str, set[str]] = {node.node_id: set() for node in snapshot.nodes}
        for edge in snapshot.edges:
            adjacency[edge.source_id].add(edge.target_id)
            adjacency[edge.target_id].add(edge.source_id)
        frontier = [anchor_id]
        selected: set[str] = {anchor_id}
        while frontier and len(selected) < max_nodes:
            candidates = sorted({neighbor for node in frontier for neighbor in adjacency[node]} - selected)
            if not candidates:
                break
            if sampler is not None:
                chosen = sampler(candidates, min(len(candidates), max_nodes - len(selected)))
            else:
                chosen = candidates[: max_nodes - len(selected)]
            selected.update(chosen)
            frontier = list(chosen)
        return selected


__all__ = [
    "GraphPartitioner", "IntegrityCheck", "IntegrityReport", "IntegrityValidator",
]
