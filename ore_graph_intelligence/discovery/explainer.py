"""Bounded explanation-neighborhood retrieval (ORE Point 8 step 28).

For each candidate the explainer retrieves a bounded neighborhood: supporting paths up to
a declared depth, the relation types on those paths, source/evidence identifiers from the
edges traversed, the snapshot hash, and counter-evidence edges found along the way. The
retrieval is bounded by an explicit expansion limit so explanations stay finite.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

from ..projection.typed_graph import TypedGraphSnapshot
from ..schemas.candidate_schema import GraphCandidate


@dataclass
class ExplanationNeighborhood:
    """One bounded explanation neighborhood for a candidate hypothesis."""

    candidate_id: str
    paths: tuple[tuple[str, ...], ...] = ()
    path_relation_types: tuple[tuple[str, ...], ...] = ()
    supporting_edge_ids: tuple[str, ...] = ()
    counter_edge_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    snapshot_hash: str = ""
    expansion_nodes_bounded: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "paths": [list(path) for path in self.paths],
            "path_relation_types": [list(types) for types in self.path_relation_types],
            "supporting_edge_ids": list(self.supporting_edge_ids),
            "counter_edge_ids": list(self.counter_edge_ids),
            "evidence_ids": list(self.evidence_ids),
            "source_ids": list(self.source_ids),
            "snapshot_hash": self.snapshot_hash,
            "expansion_nodes_bounded": self.expansion_nodes_bounded,
        }


class NeighborhoodExplainer:
    """Retrieve bounded supporting paths and counter-evidence for candidates."""

    def __init__(
        self,
        snapshot: TypedGraphSnapshot,
        *,
        max_paths: int = 4,
        max_expansion_nodes: int = 256,
    ) -> None:
        if max_paths < 1 or max_expansion_nodes < 1:
            raise ValueError("max_paths and max_expansion_nodes must be positive")
        self.snapshot = snapshot
        self.max_paths = int(max_paths)
        self.max_expansion_nodes = int(max_expansion_nodes)
        self.adjacency: dict[str, list[tuple[str, str, str]]] = {}
        for edge in snapshot.edges:
            self.adjacency.setdefault(edge.source_id, []).append(
                (edge.target_id, edge.relation_type, edge.edge_id)
            )

    def explain(
        self,
        candidate: GraphCandidate,
        *,
        depth: int = 3,
    ) -> ExplanationNeighborhood:
        """Bounded BFS for connecting paths plus direct edge inspection (step 28)."""
        if depth < 1:
            raise ValueError("depth must be positive")
        subject, _relation, obj = candidate.triple()
        neighborhood = ExplanationNeighborhood(
            candidate_id=candidate.candidate_id,
            snapshot_hash=self.snapshot.graph_hash,
        )
        paths: list[tuple[str, ...]] = []
        path_types: list[tuple[str, ...]] = []
        supporting_edges: set[str] = set()
        counter_edges: set[str] = set()
        expanded = 0
        queue: deque[tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = deque()
        queue.append((subject, (subject,), (), ()))
        visited: set[tuple[str, str]] = {(subject, subject)}
        while queue and len(paths) < self.max_paths and expanded < self.max_expansion_nodes:
            current, node_path, relation_path, edge_path = queue.popleft()
            expanded += 1
            if current == obj:
                if len(node_path) >= 2:
                    paths.append(node_path)
                    path_types.append(relation_path)
                    supporting_edges.update(edge_path)
                continue
            if len(node_path) > depth:
                continue
            for target, relation_type, edge_id in sorted(
                self.adjacency.get(current, ()), key=lambda item: (item[0], item[1]),
            ):
                if (current, target) in visited:
                    continue
                visited.add((current, target))
                queue.append((
                    target,
                    node_path + (target,),
                    relation_path + (relation_type,),
                    edge_path + (edge_id,),
                ))
                if expanded >= self.max_expansion_nodes:
                    break
        neighborhood.paths = tuple(paths)
        neighborhood.path_relation_types = tuple(path_types)
        neighborhood.expansion_nodes_bounded = expanded
        edge_ids_on_paths = supporting_edges
        neighborhood.supporting_edge_ids = tuple(sorted(edge_ids_on_paths))
        evidence_ids: set[str] = set()
        source_ids: set[str] = set()
        for edge in self.snapshot.edges:
            if edge.edge_id in edge_ids_on_paths:
                evidence_ids.update(edge.evidence_ids)
                if edge.provenance is not None:
                    source_ids.update(edge.provenance.source_ids)
            if (
                edge.source_id == obj and edge.target_id == subject
                and edge.qualifiers.polarity == "negative"
            ):
                counter_edges.add(edge.edge_id)
        neighborhood.evidence_ids = tuple(sorted(evidence_ids))
        neighborhood.source_ids = tuple(sorted(source_ids))
        neighborhood.counter_edge_ids = tuple(sorted(counter_edges))
        return neighborhood

    def attach(
        self,
        candidates: Sequence[GraphCandidate],
        *,
        depth: int = 3,
    ) -> list[ExplanationNeighborhood]:
        """Explain every candidate in deterministic order."""
        return [self.explain(candidate, depth=depth) for candidate in candidates]


__all__ = ["ExplanationNeighborhood", "NeighborhoodExplainer"]
