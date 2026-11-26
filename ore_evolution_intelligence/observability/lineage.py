"""Candidate lineage tracking and diversity metrics (step 27; section 8).

Immutable IDs, hashes, manifests and event logs guard against broken lineage
(section 9); the tracker reconstructs ancestry trees, counts descendants and measures
lineage diversity in a population (acceptance test 24: candidate lineage is
reconstructable).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ..domain.strategy import LineageRecord, StrategyCandidate


@dataclass(frozen=True, slots=True)
class LineageNode:
    """One node in the reconstructed ancestry tree."""

    candidate_id: str
    parents: tuple[str, ...]
    operation: str
    generation: int
    run_id: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id, "parents": list(self.parents),
            "operation": self.operation, "generation": self.generation,
            "run_id": self.run_id,
        }


class LineageTracker:
    """Record candidates and reconstruct their ancestry (step 27)."""

    def __init__(self) -> None:
        self._nodes: dict[str, LineageNode] = {}
        self._lineage_records: dict[str, LineageRecord] = {}

    def record(self, candidate: StrategyCandidate) -> LineageNode:
        """Record one candidate; immutable by candidate ID (deduplicated)."""
        node = LineageNode(
            candidate_id=candidate.candidate_id,
            parents=tuple(candidate.lineage.parents),
            operation=candidate.lineage.operation,
            generation=candidate.lineage.generation,
            run_id=candidate.lineage.run_id or candidate.run_id,
        )
        self._nodes.setdefault(candidate.candidate_id, node)
        self._lineage_records.setdefault(candidate.candidate_id, candidate.lineage)
        return self._nodes[candidate.candidate_id]

    def record_many(self, candidates: Iterable[StrategyCandidate]) -> None:
        for candidate in candidates:
            self.record(candidate)

    def node(self, candidate_id: str) -> LineageNode | None:
        return self._nodes.get(candidate_id)

    def ancestry(self, candidate_id: str) -> dict[str, LineageNode]:
        """Reconstruct the full ancestry tree of one candidate (step 27)."""
        tree: dict[str, LineageNode] = {}
        stack = [candidate_id]
        while stack:
            current = stack.pop()
            node = self._nodes.get(current)
            if node is None or current in tree:
                continue
            tree[current] = node
            stack.extend(node.parents)
        return tree

    def ancestry_depth(self, candidate_id: str) -> int:
        """Longest chain of ancestors from one candidate back to its roots."""

        def depth_of(node_id: str, seen: frozenset[str]) -> int:
            node = self._nodes.get(node_id)
            if node is None or not node.parents or node_id in seen:
                return 0
            return 1 + max(
                depth_of(parent, seen | {node_id}) for parent in node.parents
            )

        return depth_of(candidate_id, frozenset())

    def descendants(self, candidate_id: str) -> list[str]:
        """All candidates descending (transitively) from one candidate."""
        result: list[str] = []
        for node_id, node in self._nodes.items():
            stack = list(node.parents)
            while stack:
                current = stack.pop()
                if current == candidate_id:
                    result.append(node_id)
                    break
                parent_node = self._nodes.get(current)
                if parent_node is not None:
                    stack.extend(parent_node.parents)
        return sorted(result)

    def lineage_diversity(self, candidate_ids: Sequence[str]) -> int:
        """Number of distinct ancestral roots in a set of candidates (step 26)."""
        roots: set[str] = set()
        for candidate_id in candidate_ids:
            node = self._nodes.get(candidate_id)
            if node is None:
                roots.add(candidate_id)
                continue
            if not node.parents:
                roots.add(node.candidate_id)
            else:
                roots.add(node.parents[0])
        return len(roots)

    def population_summary(self) -> dict[str, object]:
        return {
            "recorded": len(self._nodes),
            "roots": len(
                {node.candidate_id for node in self._nodes.values() if not node.parents}
            ),
            "operations": sorted({node.operation for node in self._nodes.values()}),
        }

    def nodes(self) -> Mapping[str, LineageNode]:
        return dict(self._nodes)


__all__ = ["LineageNode", "LineageTracker"]
