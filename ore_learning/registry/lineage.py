"""Reproducible model lineage (Point 14, Step 53).

Every model links to its base artifact, fine-tuning datasets, training code,
hyperparameters, evaluation suites, approvals, and deployment events.  Artifact
digests prevent silent substitution; retirement preserves the lineage graph.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Iterable

from ..contracts import LifecycleState, ModelManifest

__all__ = ("LineageGraph", "LineageLink", "LineageRecord")


@dataclass(frozen=True, slots=True)
class LineageLink:
    """One edge in the lineage graph: model -> related artifact or event."""

    related_id: str
    relation: str
    digest: str = ""
    recorded_at: float = 0.0
    metadata: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"related_id": self.related_id, "relation": self.relation,
                "digest": self.digest, "recorded_at": self.recorded_at,
                "metadata": dict(self.metadata)}


@dataclass(slots=True)
class LineageRecord:
    """The lineage of one model identity (Step 53)."""

    model_id: str
    model_version: str
    base_artifact: str = ""
    created_at: float = field(default_factory=time.time)
    links: list[LineageLink] = field(default_factory=list)
    evaluation_history: list[dict[str, Any]] = field(default_factory=list)

    def add_link(self, related_id: str, relation: str, digest: str = "",
                 metadata: dict[str, str] | None = None) -> LineageLink:
        """Append one lineage edge and return it."""
        if not related_id or not relation:
            raise ValueError("related_id and relation are required")
        link = LineageLink(related_id=related_id, relation=relation, digest=digest,
                           recorded_at=time.time(),
                           metadata=tuple(sorted((metadata or {}).items())))
        self.links.append(link)
        return link

    def add_evaluation(self, suite: str, metrics: dict[str, Any]) -> dict[str, Any]:
        """Append one evaluation entry; history is append-only (Step 56)."""
        entry = {"suite": suite, "metrics": dict(metrics), "recorded_at": time.time()}
        self.evaluation_history.append(entry)
        return entry

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"model_id": self.model_id, "model_version": self.model_version,
                "base_artifact": self.base_artifact, "created_at": self.created_at,
                "links": [link.to_dict() for link in self.links],
                "evaluation_history": list(self.evaluation_history)}


class LineageGraph:
    """The lineage graph over every registered model identity (Step 53)."""

    def __init__(self) -> None:
        """Start with an empty graph."""
        self._records: dict[tuple[str, str], LineageRecord] = {}

    def register(self, manifest: ModelManifest, base_artifact: str = "") -> LineageRecord:
        """Create or return the lineage record for a manifest."""
        key = (manifest.model_id, manifest.version)
        if key not in self._records:
            record = LineageRecord(model_id=manifest.model_id,
                                   model_version=manifest.version,
                                   base_artifact=base_artifact or manifest.artifact_digest)
            record.add_link(manifest.training_data_provenance.dataset_id
                            if manifest.training_data_provenance else "unknown-dataset",
                            "training_dataset")
            for benchmark in manifest.benchmark_results:
                record.add_evaluation(benchmark.suite, benchmark.to_dict())
            self._records[key] = record
        return self._records[key]

    def record(self, model_id: str, version: str) -> LineageRecord:
        """Return an existing record or raise."""
        key = (model_id, version)
        if key not in self._records:
            raise KeyError(f"no lineage record for {model_id}@{version}")
        return self._records[key]

    def link(self, model_id: str, version: str, related_id: str, relation: str,
             digest: str = "", metadata: dict[str, str] | None = None) -> LineageLink:
        """Add one edge to an existing record."""
        return self.record(model_id, version).add_link(
            related_id, relation, digest=digest, metadata=metadata)

    def history(self, model_id: str, version: str) -> list[dict[str, Any]]:
        """Return the full evaluation history (preserved after retirement, Step 56)."""
        return list(self.record(model_id, version).evaluation_history)

    def ancestors(self, model_id: str, version: str) -> list[str]:
        """Follow base-artifact links back to the root model."""
        chain: list[str] = []
        current = self.record(model_id, version)
        seen: set[str] = set()
        while current.base_artifact and current.base_artifact not in seen:
            chain.append(current.base_artifact)
            seen.add(current.base_artifact)
            parent = self._find_by_digest(current.base_artifact)
            if parent is None:
                break
            current = parent
        return chain

    def _find_by_digest(self, digest: str) -> LineageRecord | None:
        """Locate a record whose model identity matches a recorded base artifact."""
        for (model_id, _), record in self._records.items():
            if record.base_artifact == digest or model_id == digest:
                return record
        return None

    def describe(self) -> dict[str, Any]:
        """Summarize the graph."""
        return {"records": len(self._records),
                "links": sum(len(record.links) for record in self._records.values()),
                "evaluations": sum(len(record.evaluation_history)
                                   for record in self._records.values())}

    def iter_records(self) -> Iterable[LineageRecord]:
        """Iterate over lineage records."""
        return iter(self._records.values())


def lifecycle_states_preserving_lineage() -> tuple[LifecycleState, ...]:
    """States in which lineage must remain queryable (Step 56)."""
    return (LifecycleState.DEPRECATED, LifecycleState.RETIRED, LifecycleState.REVOKED)
