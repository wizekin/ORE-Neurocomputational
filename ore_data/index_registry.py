"""Rebuildable derived-index registry with watermarks and reconciliation (Step 12).

Every Bloom filter, sketch, similarity index and materialized view declares
its authoritative source, schema version, build watermark, parameters,
checksum and rebuild procedure, so derived structures can be discarded and
rebuilt without losing any authoritative knowledge. Reconciliation (Step 19
and section 2) compares each index's watermark with the authoritative
watermark and samples records to detect divergence.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from ore_data.contracts import FreshnessContract


class IndexKind(str, Enum):
    """Kinds of derived structure the registry tracks (Step 12)."""

    BLOOM_FILTER = "bloom_filter"
    COUNT_MIN_SKETCH = "count_min_sketch"
    HYPERLOGLOG = "hyperloglog"
    KLL_QUANTILE = "kll_quantile"
    SPACE_SAVING = "space_saving"
    MINHASH_LSH = "minhash_lsh"
    SIMHASH = "simhash"
    MATERIALIZED_VIEW = "materialized_view"


class IndexHealth(str, Enum):
    """Health status of one derived index (source section 8.4)."""

    FRESH = "fresh"
    STALE = "stale"
    INCOMPLETE = "incomplete"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DerivedIndexSpec:
    """Metadata record of one derived structure (Step 12)."""

    index_id: str
    kind: IndexKind
    authoritative_source: str
    schema_version: str
    build_watermark: int
    parameters: Mapping[str, Any]
    checksum: str
    rebuild_procedure: str
    memory_budget_bytes: int
    freshness: FreshnessContract | None = None

    def __post_init__(self) -> None:
        """Validate the index descriptor."""
        if not self.index_id.strip():
            raise ValueError("index_id must be non-empty")
        if not isinstance(self.kind, IndexKind):
            raise TypeError("kind must be an IndexKind")
        if not self.authoritative_source.strip():
            raise ValueError("authoritative_source must be non-empty")
        if not self.schema_version.strip():
            raise ValueError("schema_version must be non-empty")
        if self.build_watermark < 0:
            raise ValueError("build_watermark must be non-negative")
        if not self.rebuild_procedure.strip():
            raise ValueError("rebuild_procedure must be non-empty")
        if self.memory_budget_bytes <= 0:
            raise ValueError("memory_budget_bytes must be positive")
        object.__setattr__(self, "parameters", dict(self.parameters))

    def with_watermark(self, watermark: int) -> "DerivedIndexSpec":
        """Return a copy of this spec with one advanced build watermark."""
        return DerivedIndexSpec(
            index_id=self.index_id,
            kind=self.kind,
            authoritative_source=self.authoritative_source,
            schema_version=self.schema_version,
            build_watermark=watermark,
            parameters=self.parameters,
            checksum=self.checksum,
            rebuild_procedure=self.rebuild_procedure,
            memory_budget_bytes=self.memory_budget_bytes,
            freshness=self.freshness,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the index record as a plain dictionary."""
        return {
            "index_id": self.index_id,
            "kind": self.kind.value,
            "authoritative_source": self.authoritative_source,
            "schema_version": self.schema_version,
            "build_watermark": self.build_watermark,
            "parameters": dict(self.parameters),
            "checksum": self.checksum,
            "rebuild_procedure": self.rebuild_procedure,
            "memory_budget_bytes": self.memory_budget_bytes,
        }


@dataclass(frozen=True, slots=True)
class IndexHealthReport:
    """Freshness, watermark, version and health of one derived index (section 8.4)."""

    index_id: str
    version: str
    watermark: int
    freshness_lag: int
    health: IndexHealth
    usable_for_precheck: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the health record as a plain dictionary."""
        return {
            "index_id": self.index_id,
            "version": self.version,
            "watermark": self.watermark,
            "freshness_lag": self.freshness_lag,
            "health": self.health.value,
            "usable_for_precheck": self.usable_for_precheck,
        }


@dataclass(frozen=True, slots=True)
class RebuildPlan:
    """Deterministic rebuild plan for one derived structure (Step 12)."""

    index_id: str
    from_watermark: int
    to_watermark: int
    rebuild_procedure: str

    def to_dict(self) -> dict[str, Any]:
        """Return the plan as a plain dictionary."""
        return {
            "index_id": self.index_id,
            "from_watermark": self.from_watermark,
            "to_watermark": self.to_watermark,
            "rebuild_procedure": self.rebuild_procedure,
        }


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    """Result of one reconciliation pass over every derived index (Step 19)."""

    authoritative_watermark: int
    index_reports: tuple[IndexHealthReport, ...]
    rebuild_plans: tuple[RebuildPlan, ...]

    @property
    def indexes_stale(self) -> int:
        """Return the number of stale or incomplete indexes."""
        return sum(
            1
            for report in self.index_reports
            if report.health in (IndexHealth.STALE, IndexHealth.INCOMPLETE)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the reconciliation report as a plain dictionary."""
        return {
            "authoritative_watermark": self.authoritative_watermark,
            "index_reports": [report.to_dict() for report in self.index_reports],
            "rebuild_plans": [plan.to_dict() for plan in self.rebuild_plans],
            "indexes_stale": self.indexes_stale,
        }


class DerivedIndexRegistry:
    """Registry of every derived structure with its rebuild metadata (Step 12)."""

    def __init__(self) -> None:
        """Create an empty index registry."""
        self._specs: dict[str, DerivedIndexSpec] = {}
        self._incomplete: set[str] = set()

    def register(self, spec: DerivedIndexSpec) -> None:
        """Register one derived index; identifiers stay unique."""
        if spec.index_id in self._specs:
            raise ValueError(f"duplicate derived index {spec.index_id!r}")
        self._specs[spec.index_id] = spec

    def require(self, index_id: str) -> DerivedIndexSpec:
        """Return one index's metadata or raise (unregistered indexes cannot serve)."""
        if index_id not in self._specs:
            raise KeyError(
                f"derived index {index_id!r} is not registered; derived structures "
                "must declare source, watermark and rebuild path (Step 12)"
            )
        return self._specs[index_id]

    def update_watermark(self, index_id: str, watermark: int) -> None:
        """Advance one index's build watermark after a verified rebuild or catch-up."""
        spec = self.require(index_id)
        if watermark < spec.build_watermark:
            raise ValueError("watermarks only advance; rebuild instead of rewinding")
        self._specs[index_id] = spec.with_watermark(watermark)
        self._incomplete.discard(index_id)

    def mark_incomplete(self, index_id: str) -> None:
        """Mark one index incomplete: its negative shortcuts must be bypassed (Step 19)."""
        self.require(index_id)
        self._incomplete.add(index_id)

    def specs(self) -> tuple[DerivedIndexSpec, ...]:
        """Return every registered index metadata record."""
        return tuple(self._specs.values())

    def health(
        self,
        index_id: str,
        authoritative_watermark: int,
    ) -> IndexHealthReport:
        """Report one index's freshness against the authoritative watermark (Step 5/12)."""
        spec = self.require(index_id)
        if index_id in self._incomplete:
            health = IndexHealth.INCOMPLETE
            usable = False
        elif spec.freshness is not None and spec.freshness.max_lag_events is not None:
            lag = authoritative_watermark - spec.build_watermark
            if lag < 0:
                health, usable = IndexHealth.UNKNOWN, False
            elif spec.freshness.within_lag(lag):
                health, usable = IndexHealth.FRESH, True
            else:
                health, usable = IndexHealth.STALE, False
        else:
            health = IndexHealth.FRESH if authoritative_watermark == spec.build_watermark else IndexHealth.STALE
            usable = health is IndexHealth.FRESH
        return IndexHealthReport(
            index_id=index_id,
            version=spec.schema_version,
            watermark=max(0, spec.build_watermark),
            freshness_lag=max(0, authoritative_watermark - spec.build_watermark),
            health=health,
            usable_for_precheck=usable,
        )

    def health_reports(self, authoritative_watermark: int) -> tuple[IndexHealthReport, ...]:
        """Report freshness, watermark, version and health of every index (section 8.4)."""
        return tuple(self.health(index_id, authoritative_watermark) for index_id in self._specs)

    def reconcile(self, authoritative_watermark: int) -> ReconciliationReport:
        """Reconcile this registry against one authoritative watermark (Step 19)."""
        return reconcile_derived_indexes(self, authoritative_watermark)


def reconcile_derived_indexes(
    registry: DerivedIndexRegistry,
    authoritative_watermark: int,
) -> ReconciliationReport:
    """Reconcile every derived index against the authoritative watermark (Step 19).

    Each index exposes its watermark and freshness status; stale or
    incomplete structures get a deterministic rebuild plan from their
    registered procedure, because derived structures can always be
    discarded and rebuilt from the authoritative source.
    """
    reports: list[IndexHealthReport] = []
    plans: list[RebuildPlan] = []
    for spec in registry.specs():
        report = registry.health(spec.index_id, authoritative_watermark)
        reports.append(report)
        if report.health in (IndexHealth.STALE, IndexHealth.INCOMPLETE):
            plans.append(
                RebuildPlan(
                    index_id=spec.index_id,
                    from_watermark=max(0, spec.build_watermark),
                    to_watermark=authoritative_watermark,
                    rebuild_procedure=spec.rebuild_procedure,
                )
            )
    return ReconciliationReport(
        authoritative_watermark=authoritative_watermark,
        index_reports=tuple(reports),
        rebuild_plans=tuple(plans),
    )


def freshness_lag_of(spec: DerivedIndexSpec, authoritative_watermark: int) -> int:
    """Return the lag between the authoritative and one index's build watermark."""
    return max(0, authoritative_watermark - spec.build_watermark)


__all__ = [
    "IndexKind",
    "IndexHealth",
    "DerivedIndexSpec",
    "IndexHealthReport",
    "RebuildPlan",
    "ReconciliationReport",
    "DerivedIndexRegistry",
    "reconcile_derived_indexes",
    "freshness_lag_of",
]
