"""Artifact dependency records, content hashes, and reverse dependency indexes."""
from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..contracts import content_fingerprint


class ArtifactState(str, Enum):
    """Track whether a derived artifact is current, stale, or recomputing."""

    CURRENT = "current"
    STALE = "stale"
    RECOMPUTING = "recomputing"


@dataclass(frozen=True, slots=True)
class DependencyRef:
    """Identify one source dependency by record ID, version, and content hash."""

    record_id: str
    version: int = 1
    content_hash: str | None = None

    def __post_init__(self) -> None:
        """Validate the dependency identity fields."""
        if not self.record_id.strip():
            raise ValueError("record_id must be non-empty.")
        if self.version < 1:
            raise ValueError("version must be at least 1.")

    def matches(self, other: "DependencyRef") -> bool:
        """Check version and content-hash equality against another reference."""
        return (
            self.record_id == other.record_id
            and self.version == other.version
            and self.content_hash == other.content_hash
        )


def compute_artifact_hash(
    dependencies: Sequence[DependencyRef], rules: Sequence[str],
    config: Mapping[str, Any], algorithm_version: str,
) -> str:
    """Hash inputs, versions, rules, config, and algorithm version per spec."""
    return content_fingerprint({
        "dependencies": [
            {
                "record_id": ref.record_id,
                "version": ref.version,
                "content_hash": ref.content_hash,
            }
            for ref in dependencies
        ],
        "rules": list(rules),
        "config": dict(config),
        "algorithm_version": algorithm_version,
    })


@dataclass(slots=True)
class ArtifactRecord:
    """Register one derived artifact with its exact dependency set and hash."""

    artifact_id: str
    kind: str = "derived"
    deps: tuple[DependencyRef, ...] = ()
    rules: tuple[str, ...] = ()
    config: Mapping[str, Any] = field(default_factory=dict)
    algorithm_version: str = "0.1.0"
    artifact_hash: str = ""
    state: ArtifactState = ArtifactState.CURRENT
    stale_reason: str | None = None
    previous_hash: str | None = None
    invalidated_at: str | None = None
    computed_at: str | None = None
    content: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate identity, detach mappings, and default the hash."""
        if not self.artifact_id.strip():
            raise ValueError("artifact_id must be non-empty.")
        if len(set(dep.record_id for dep in self.deps)) != len(self.deps):
            raise ValueError("artifact dependencies must have unique record IDs.")
        self.config = dict(self.config)
        self.content = dict(self.content)
        if not self.artifact_hash:
            self.artifact_hash = compute_artifact_hash(
                self.deps, self.rules, self.config, self.algorithm_version,
            )

    def dependency_versions(self) -> dict[str, int]:
        """Return the source versions this artifact was computed against."""
        return {dep.record_id: dep.version for dep in self.deps}

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable artifact record for reports and manifests."""
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "deps": [
                {"record_id": dep.record_id, "version": dep.version,
                 "content_hash": dep.content_hash}
                for dep in self.deps
            ],
            "rules": list(self.rules),
            "config": dict(self.config),
            "algorithm_version": self.algorithm_version,
            "artifact_hash": self.artifact_hash,
            "state": self.state.value,
            "stale_reason": self.stale_reason,
            "previous_hash": self.previous_hash,
        }


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """Version one USG or workspace source record consumed by derived artifacts."""

    record_id: str
    version: int
    content_hash: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the source record identity and detach its payload."""
        if not self.record_id.strip():
            raise ValueError("record_id must be non-empty.")
        if self.version < 1:
            raise ValueError("version must be at least 1.")
        object.__setattr__(self, "payload", dict(self.payload))

    def as_dependency(self) -> DependencyRef:
        """Return this record's current dependency reference."""
        return DependencyRef(
            record_id=self.record_id, version=self.version,
            content_hash=self.content_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable source record."""
        return {
            "record_id": self.record_id,
            "version": self.version,
            "content_hash": self.content_hash,
            "payload": dict(self.payload),
        }


class DependencyGraph:
    """Index derived artifacts against versioned sources with reverse edges.

    The reverse index answers which artifacts transitively depend on a changed
    source record, and freshness is decided by comparing dependency versions
    and content hashes rather than assuming affected means invalid.
    """

    def __init__(self) -> None:
        """Initialize the thread-safe artifact and source indexes."""
        self._lock = threading.RLock()
        self._artifacts: dict[str, ArtifactRecord] = {}
        self._sources: dict[str, SourceRecord] = {}
        self._reverse: dict[str, set[str]] = {}

    # -------------------------------------------------------------- sources

    def register_source(
        self, record_id: str, payload: Mapping[str, Any], *,
        validator: Callable[[Mapping[str, Any]], bool] | None = None,
    ) -> SourceRecord:
        """Register or version-up one source record through its validation contract."""
        if not record_id.strip():
            raise ValueError("record_id must be non-empty.")
        if validator is not None and not validator(dict(payload)):
            raise ValueError(f"source '{record_id}' failed its validation contract.")
        with self._lock:
            existing = self._sources.get(record_id)
            version = existing.version + 1 if existing else 1
            record = SourceRecord(
                record_id=record_id, version=version,
                content_hash=content_fingerprint(payload), payload=dict(payload),
            )
            self._sources[record_id] = record
            return record

    def source(self, record_id: str) -> SourceRecord | None:
        """Return the current source record when registered."""
        with self._lock:
            return self._sources.get(record_id)

    def sources(self) -> tuple[SourceRecord, ...]:
        """Return every registered source record."""
        with self._lock:
            return tuple(self._sources.values())

    # ------------------------------------------------------------ artifacts

    def register_artifact(
        self, artifact_id: str, deps: Sequence[DependencyRef], *,
        rules: Sequence[str] = (), config: Mapping[str, Any] | None = None,
        algorithm_version: str = "0.1.0", content: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        """Register one derived artifact with its exact dependency set."""
        with self._lock:
            if artifact_id in self._artifacts:
                raise ValueError(f"artifact '{artifact_id}' is already registered.")
            record = ArtifactRecord(
                artifact_id=artifact_id,
                deps=tuple(deps),
                rules=tuple(rules),
                config=dict(config or {}),
                algorithm_version=algorithm_version,
                content=dict(content or {}),
            )
            self._artifacts[artifact_id] = record
            for dep in record.deps:
                self._reverse.setdefault(dep.record_id, set()).add(artifact_id)
            return record

    def update_artifact_content(
        self, artifact_id: str, content: Mapping[str, Any],
        *, deps: Sequence[DependencyRef] | None = None,
        rules: Sequence[str] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        """Recompute one artifact's stored content and dependency hash in place."""
        with self._lock:
            record = self._artifacts.get(artifact_id)
            if record is None:
                raise KeyError(f"unknown artifact '{artifact_id}'.")
            record.deps = tuple(deps) if deps is not None else record.deps
            record.rules = tuple(rules) if rules is not None else record.rules
            record.config = dict(config) if config is not None else record.config
            record.artifact_hash = compute_artifact_hash(
                record.deps, record.rules, record.config, record.algorithm_version,
            )
            record.content = dict(content)
            record.state = ArtifactState.CURRENT
            record.stale_reason = None
            record.previous_hash = None
            from ..contracts import utc_now
            record.computed_at = utc_now()
            return record

    def artifact(self, artifact_id: str) -> ArtifactRecord | None:
        """Return one registered artifact record when present."""
        with self._lock:
            return self._artifacts.get(artifact_id)

    def artifacts(self) -> tuple[ArtifactRecord, ...]:
        """Return every registered artifact record."""
        with self._lock:
            return tuple(self._artifacts.values())

    # ------------------------------------------------------------- traversal

    def dependents_of(self, record_id: str) -> tuple[str, ...]:
        """Return the artifacts that directly depend on one source record."""
        with self._lock:
            return tuple(sorted(self._reverse.get(record_id, ())))

    def transitive_dependents(
        self, record_ids: Sequence[str], *,
        include_intermediate: bool = True,
    ) -> tuple[str, ...]:
        """Compute the transitive closure of dependents for changed sources."""
        with self._lock:
            visited: set[str] = set()
            frontier: deque[str] = deque()
            for record_id in record_ids:
                for dependent in self._reverse.get(record_id, ()):
                    if dependent not in visited:
                        visited.add(dependent)
                        frontier.append(dependent)
            while frontier:
                artifact_id = frontier.popleft()
                if include_intermediate:
                    for dependent in self._reverse.get(artifact_id, ()):
                        if dependent not in visited:
                            visited.add(dependent)
                            frontier.append(dependent)
            return tuple(sorted(visited))

    def is_fresh(
        self, artifact_id: str, *,
        validity: Callable[[ArtifactRecord], bool] | None = None,
    ) -> bool:
        """Check that every dependency version matches the current source version."""
        with self._lock:
            record = self._artifacts.get(artifact_id)
            if record is None:
                raise KeyError(f"unknown artifact '{artifact_id}'.")
            if record.state is ArtifactState.STALE:
                return False
            for dep in record.deps:
                source = self._sources.get(dep.record_id)
                if source is None:
                    return False
                if dep.version != source.version:
                    return False
                if dep.content_hash is not None and dep.content_hash != source.content_hash:
                    return False
            if validity is not None and not validity(record):
                return False
            return True

    def staleness_report(self) -> dict[str, Any]:
        """Return stale artifact counts and ages for observability."""
        from ..contracts import utc_now
        with self._lock:
            stale = [
                record for record in self._artifacts.values()
                if record.state is ArtifactState.STALE
            ]
            return {
                "artifact_count": len(self._artifacts),
                "stale_count": len(stale),
                "stale_ids": [record.artifact_id for record in stale],
                "source_count": len(self._sources),
            }

    def topological_recompute_order(self, artifact_ids: Sequence[str]) -> tuple[tuple[str, ...], ...]:
        """Group artifacts into topological levels for ordered recomputation.

        Artifacts in the same level are independent of one another and may be
        recomputed in parallel; a level only runs after all its predecessors.
        """
        with self._lock:
            wanted_set = set(artifact_ids)
            for artifact_id in wanted_set:
                if artifact_id not in self._artifacts:
                    raise KeyError(f"unknown artifact '{artifact_id}'.")
            upstream: dict[str, set[str]] = {}
            for artifact_id in wanted_set:
                record = self._artifacts[artifact_id]
                upstream[artifact_id] = {
                    dep.record_id for dep in record.deps
                    if dep.record_id in wanted_set and dep.record_id != artifact_id
                }
            levels: list[tuple[str, ...]] = []
            remaining = set(wanted_set)
            while remaining:
                ready = sorted(
                    artifact_id for artifact_id in remaining
                    if not (upstream[artifact_id] & remaining)
                )
                if not ready:
                    raise ValueError(
                        "artifact dependency cycle prevents topological recompute"
                    )
                levels.append(tuple(ready))
                remaining.difference_update(ready)
            return tuple(levels)
