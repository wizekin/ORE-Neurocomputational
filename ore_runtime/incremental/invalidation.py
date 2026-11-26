"""Stale propagation and dependency-aware recomputation planning."""
from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..contracts import utc_now
from ..events.schemas import ARTIFACT_STALE, EventEnvelope, SCHEMA_VERSION
from .dependency_graph import ArtifactRecord, ArtifactState, DependencyGraph, DependencyRef


@dataclass(frozen=True, slots=True)
class InvalidationReport:
    """Summarize one invalidation sweep over changed source records."""

    changed_records: tuple[str, ...]
    candidate_artifacts: tuple[str, ...]
    marked_stale: tuple[str, ...]
    unaffected: tuple[str, ...]
    recomputed: tuple[str, ...]
    reused: tuple[str, ...]
    still_stale: tuple[str, ...]
    recompute_levels: tuple[tuple[str, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable invalidation report."""
        return {
            "changed_records": list(self.changed_records),
            "candidate_artifacts": list(self.candidate_artifacts),
            "marked_stale": list(self.marked_stale),
            "unaffected": list(self.unaffected),
            "recomputed": list(self.recomputed),
            "reused": list(self.reused),
            "still_stale": list(self.still_stale),
            "recompute_levels": [list(level) for level in self.recompute_levels],
        }


@dataclass(frozen=True, slots=True)
class InvalidationEvent:
    """Describe one stale-marking decision with its preserved history."""

    artifact_id: str
    reason: str
    previous_hash: str | None
    occurred_at: str


class InvalidationEngine:
    """Propagate source changes to exactly the affected derived artifacts.

    Affected does not automatically mean invalid: candidate dependents are
    filtered by comparing dependency versions and content hashes, and only
    artifacts whose dependency set no longer matches are marked STALE with
    their historical hash preserved until recomputed or revalidated.
    """

    def __init__(
        self, graph: DependencyGraph, *,
        on_stale: Callable[[EventEnvelope], None] | None = None,
    ) -> None:
        """Initialize the engine with an optional stale-event observer."""
        self._graph = graph
        self._lock = threading.RLock()
        self._on_stale = on_stale
        self._invalidation_log: list[InvalidationEvent] = []

    # ------------------------------------------------------------- sweep

    def invalidate(self, changed: Sequence[DependencyRef | str]) -> InvalidationReport:
        """Mark exactly the affected downstream artifacts stale after a change.

        Candidates are the transitive closure of dependents of the changed
        records. Within the closure, an artifact is marked stale when its own
        dependency versions mismatch the current sources or when any upstream
        dependency was already marked stale in this sweep, so downstream
        conclusions never stay current while an upstream fact changed.
        """
        changed_refs = [
            DependencyRef(record_id=item) if isinstance(item, str) else item
            for item in changed
        ]
        with self._lock:
            changed_ids = {ref.record_id for ref in changed_refs}
            candidates = self._graph.transitive_dependents(changed_ids)
            marked: list[str] = []
            unaffected: list[str] = []
            stale_set: set[str] = set()
            if candidates:
                levels = self._graph.topological_recompute_order(candidates)
                for level in levels:
                    for artifact_id in level:
                        record = self._graph.artifact(artifact_id)
                        if record is None:
                            continue
                        directly_affected = self._artifact_really_affected(record)
                        upstream_stale = any(
                            dep.record_id in stale_set
                            or dep.record_id in changed_ids
                            for dep in record.deps
                        )
                        if directly_affected or upstream_stale:
                            self._mark_stale(artifact_id, reason=self._stale_reason(record))
                            stale_set.add(artifact_id)
                            marked.append(artifact_id)
                        else:
                            unaffected.append(artifact_id)
            recompute_levels = self._graph.topological_recompute_order(marked) \
                if marked else ()
            return InvalidationReport(
                changed_records=tuple(ref.record_id for ref in changed_refs),
                candidate_artifacts=tuple(candidates),
                marked_stale=tuple(sorted(marked)),
                unaffected=tuple(sorted(unaffected)),
                recomputed=(),
                reused=(),
                still_stale=tuple(sorted(marked)),
                recompute_levels=recompute_levels,
            )

    def _artifact_really_affected(self, record: ArtifactRecord) -> bool:
        """Compare dependency versions against current sources semantically."""
        for dep in record.deps:
            source = self._graph.source(dep.record_id)
            if source is None:
                return True
            if dep.version != source.version:
                return True
            if dep.content_hash is not None and dep.content_hash != source.content_hash:
                return True
        return False

    @staticmethod
    def _stale_reason(record: ArtifactRecord) -> str:
        """Build the invalidation reason naming the first mismatched dependency."""
        return (
            f"artifact '{record.artifact_id}' depends on outdated source versions: "
            + ", ".join(
                f"{dep.record_id}@{dep.version}" for dep in record.deps
            )
        )

    def _mark_stale(self, artifact_id: str, *, reason: str) -> None:
        """Mark one artifact stale while preserving its historical hash."""
        record = self._graph.artifact(artifact_id)
        if record is None:
            return
        record.previous_hash = record.artifact_hash
        record.state = ArtifactState.STALE
        record.stale_reason = reason
        record.invalidated_at = utc_now()
        event = InvalidationEvent(
            artifact_id=artifact_id, reason=reason,
            previous_hash=record.previous_hash, occurred_at=record.invalidated_at,
        )
        self._invalidation_log.append(event)
        if self._on_stale is not None:
            envelope = EventEnvelope.create(
                event_type=ARTIFACT_STALE,
                aggregate_id=artifact_id,
                aggregate_version=1,
                payload={"artifact_id": artifact_id, "reason": reason},
                schema_version=SCHEMA_VERSION,
            )
            self._on_stale(envelope)

    # --------------------------------------------------------- recomputation

    def recompute_plan(self, stale_ids: Sequence[str]) -> tuple[tuple[str, ...], ...]:
        """Return topologically ordered levels over the stale artifact set."""
        with self._lock:
            return self._graph.topological_recompute_order(stale_ids)

    def recompute(
        self, stale_ids: Sequence[str],
        recompute_fn: Callable[[ArtifactRecord], Mapping[str, Any]],
        *,
        validity: Callable[[ArtifactRecord], bool] | None = None,
    ) -> InvalidationReport:
        """Recompute stale artifacts in dependency order, reusing fresh caches.

        A stale artifact is recomputed only when its dependency hash differs
        from the stored hash or its validity predicate fails, mirroring the
        specification's incremental recomputation rule.
        """
        with self._lock:
            levels = self.recompute_plan(stale_ids)
            recomputed: list[str] = []
            reused: list[str] = []
            still_stale: list[str] = []
            for level in levels:
                for artifact_id in level:
                    record = self._graph.artifact(artifact_id)
                    if record is None:
                        continue
                    current_deps = [
                        self._graph.source(dep.record_id).as_dependency()
                        if self._graph.source(dep.record_id) is not None else dep
                        for dep in record.deps
                    ]
                    fresh_deps = all(
                        self._graph.source(dep.record_id) is not None
                        and dep.matches(self._graph.source(dep.record_id).as_dependency())
                        for dep in record.deps
                    )
                    if fresh_deps and (validity is None or validity(record)):
                        reused.append(artifact_id)
                        continue
                    content = recompute_fn(record)
                    self._graph.update_artifact_content(
                        artifact_id, content, deps=current_deps,
                    )
                    recomputed.append(artifact_id)
            still_stale = [
                artifact_id for artifact_id in stale_ids
                if self._graph.artifact(artifact_id) is not None
                and self._graph.artifact(artifact_id).state is ArtifactState.STALE
            ]
            return InvalidationReport(
                changed_records=(),
                candidate_artifacts=tuple(stale_ids),
                marked_stale=(),
                unaffected=(),
                recomputed=tuple(sorted(recomputed)),
                reused=tuple(sorted(reused)),
                still_stale=tuple(sorted(still_stale)),
                recompute_levels=levels,
            )

    # ------------------------------------------------------------- queries

    def stale_artifacts(self) -> tuple[str, ...]:
        """Return every artifact currently marked stale."""
        return tuple(
            record.artifact_id for record in self._graph.artifacts()
            if record.state is ArtifactState.STALE
        )

    def is_current(self, artifact_id: str) -> bool:
        """Return whether an artifact is current, never silently stale."""
        return not (
            self._graph.artifact(artifact_id) is not None
            and self._graph.artifact(artifact_id).state is ArtifactState.STALE
        )

    def invalidation_log(self) -> tuple[InvalidationEvent, ...]:
        """Return the ordered stale-marking history for audit trails."""
        return tuple(self._invalidation_log)

    def stats(self) -> dict[str, Any]:
        """Return invalidation counters for the observability report."""
        recomputed_total = sum(
            1 for record in self._graph.artifacts()
            if record.computed_at is not None
        )
        changed_total = len(self._invalidation_log)
        return {
            **self._graph.staleness_report(),
            "invalidations": changed_total,
            "recomputed_artifacts": recomputed_total,
            "recompute_to_change_ratio": (
                round(recomputed_total / changed_total, 4)
                if changed_total else 0.0
            ),
        }
