"""Coordinator, CROS, workspace and USG integration adapters (source section 6).

The Main Brain Coordinator requests data services and decides which
retrieval or analysis operation to invoke; it receives estimate metadata
and the exactness class of every answer (section 6). The CROS adapter
converts storage telemetry into bounded adaptation of compaction, rebuild
and rotation work while protecting every correctness and integrity
parameter. The workspace connector publishes versioned data-change events
as workspace events and registers approximate indexes as derived views,
never as workspace authority. The USG adapter resolves exact semantic
identity behind a resolver protocol, so similarity candidates are verified
against authoritative concept identity before any merge, deletion or
canonicalization decision (source 6 critical boundary).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from ore_data.contracts import ExactnessClass
from ore_data.ingestion import DataChangeEvent
from ore_data.observability import estimate_metadata
from ore_data.similarity.minhash import VerifiedSimilarity


# --------------------------------------------------------------------- CROS

#: Storage parameters CROS may adapt within safe bounds (source section 6).
ADAPTABLE_PARAMETERS = frozenset(
    {
        "compaction_batch_bytes",
        "bloom_rebuild_interval_events",
        "sketch_rotation_interval_events",
        "backpressure_queue_limit",
    }
)

#: Correctness and integrity parameters CROS must never relax (Step 6 + section 6).
PROTECTED_PARAMETERS = frozenset(
    {
        "target_false_positive_rate",
        "count_min_epsilon",
        "count_min_delta",
        "hyperloglog_precision",
        "similarity_verification_required",
        "hash_algorithm",
        "canonicalization_version",
        "quorum_of_shards",  # erasure reconstruction threshold
    }
)

#: Safe bounds for every adaptable parameter.
PARAMETER_BOUNDS: Mapping[str, tuple[int, int]] = {
    "compaction_batch_bytes": (1 << 16, 1 << 24),
    "bloom_rebuild_interval_events": (100, 100_000),
    "sketch_rotation_interval_events": (100, 100_000),
    "backpressure_queue_limit": (16, 65_536),
}


@dataclass(frozen=True, slots=True)
class StorageTelemetry:
    """One sampled storage-pressure snapshot (source section 6)."""

    memory_pressure: float
    storage_pressure: float
    queue_depth: int
    compaction_load: float

    def __post_init__(self) -> None:
        """Validate pressure samples."""
        for name in ("memory_pressure", "storage_pressure", "compaction_load"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.queue_depth < 0:
            raise ValueError("queue_depth must be non-negative")

    @property
    def overall_pressure(self) -> float:
        """Return the maximum pressure across sampled resources."""
        return max(self.memory_pressure, self.storage_pressure, self.compaction_load)

    def to_dict(self) -> dict[str, object]:
        """Return the telemetry sample as a plain dictionary."""
        return {
            "memory_pressure": self.memory_pressure,
            "storage_pressure": self.storage_pressure,
            "queue_depth": self.queue_depth,
            "compaction_load": self.compaction_load,
            "overall_pressure": self.overall_pressure,
        }


def telemetry_to_capability_demand(telemetry: StorageTelemetry) -> Mapping[str, object]:
    """Convert storage telemetry into the repo's CROS demand record.

    Returns a plain payload compatible with
    ``ore_contracts.resource_profiles.CapabilityDemand``; the conversion
    imports lazily so the core data layer never depends on the contracts
    package.
    """
    try:  # pragma: no cover - import bridge to the repository's CROS
        from ore_contracts.resource_profiles import CapabilityDemand
    except ImportError:  # pragma: no cover - standalone fallback
        return {
            "capability_id": "ore.data.storage_pressure",
            "priority": round(min(1.0, telemetry.overall_pressure), 4),
            "mandatory": False,
            "target_confidence": round(1.0 - telemetry.overall_pressure, 4),
        }
    return CapabilityDemand(
        capability_id="ore.data.storage_pressure",
        priority=round(min(1.0, telemetry.overall_pressure), 4),
        mandatory=False,
        target_confidence=round(1.0 - telemetry.overall_pressure, 4),
    )


class CROSDataBudgetAdapter:
    """Bounded, invariant-preserving CROS adaptation of data-infrastructure work."""

    def __init__(self, initial: Mapping[str, int] | None = None) -> None:
        """Create one adapter with default adaptable parameters."""
        self._parameters: dict[str, int] = {
            "compaction_batch_bytes": 1 << 20,
            "bloom_rebuild_interval_events": 5_000,
            "sketch_rotation_interval_events": 5_000,
            "backpressure_queue_limit": 1_024,
        }
        for key, value in (initial or {}).items():
            self.set_parameter(key, value)
        self.telemetry_samples: list[StorageTelemetry] = []
        self.adaptations: list[tuple[str, int, int]] = []
        self.violations_blocked = 0

    def parameters(self) -> dict[str, int]:
        """Return the current adaptable parameter values."""
        return dict(self._parameters)

    def set_parameter(self, name: str, value: int) -> None:
        """Set one parameter inside its safe bounds; correctness stays protected."""
        if name in PROTECTED_PARAMETERS:
            self.violations_blocked += 1
            raise ValueError(
                f"{name!r} is a correctness or integrity parameter; CROS "
                "adaptation may never relax it (Step 6 / section 6)"
            )
        if name not in ADAPTABLE_PARAMETERS:
            raise ValueError(f"unknown parameter {name!r}")
        low, high = PARAMETER_BOUNDS[name]
        if not low <= int(value) <= high:
            raise ValueError(f"{name} must stay within [{low}, {high}]")
        self._parameters[name] = int(value)

    def record_telemetry(self, telemetry: StorageTelemetry) -> None:
        """Record one telemetry sample."""
        self.telemetry_samples.append(telemetry)

    def adapt(self, telemetry: StorageTelemetry) -> dict[str, int]:
        """Adapt data-infrastructure work within bounds under pressure (section 6)."""
        self.record_telemetry(telemetry)
        pressure = telemetry.overall_pressure
        rebuild_interval = self._parameters["bloom_rebuild_interval_events"]
        rotation_interval = self._parameters["sketch_rotation_interval_events"]
        batch = self._parameters["compaction_batch_bytes"]
        if pressure > 0.75:
            rebuild_interval = min(PARAMETER_BOUNDS["bloom_rebuild_interval_events"][1], rebuild_interval * 2)
            rotation_interval = min(PARAMETER_BOUNDS["sketch_rotation_interval_events"][1], rotation_interval * 2)
            batch = max(PARAMETER_BOUNDS["compaction_batch_bytes"][0], batch // 2)
        elif pressure < 0.25:
            rebuild_interval = max(PARAMETER_BOUNDS["bloom_rebuild_interval_events"][0], rebuild_interval // 2)
            rotation_interval = max(PARAMETER_BOUNDS["sketch_rotation_interval_events"][0], rotation_interval // 2)
            batch = min(PARAMETER_BOUNDS["compaction_batch_bytes"][1], batch * 2)
        for name, value in (
            ("bloom_rebuild_interval_events", rebuild_interval),
            ("sketch_rotation_interval_events", rotation_interval),
            ("compaction_batch_bytes", batch),
        ):
            if self._parameters[name] != value:
                self.adaptations.append((name, self._parameters[name], value))
                self._parameters[name] = value
        return self.parameters()

    def try_violate_correctness(self, name: str, value: int) -> bool:
        """Attempt to change a protected parameter; return whether it was blocked."""
        try:
            self.set_parameter(name, value)
            return False
        except ValueError:
            return True


class BackpressureController:
    """Bounded event queues: backpressure prevents unbounded growth (section 8.4)."""

    def __init__(self, queue_limit: int = 1_024) -> None:
        """Create one bounded queue controller."""
        if queue_limit <= 0:
            raise ValueError("queue_limit must be positive")
        self.queue_limit = queue_limit
        self._queue: list[Any] = []
        self.deferred: int = 0
        self.overflow_events: list[str] = []

    def offer(self, item: Any) -> bool:
        """Offer one item; overflow defers instead of growing the queue."""
        if len(self._queue) >= self.queue_limit:
            self.deferred += 1
            self.overflow_events.append(str(getattr(item, "event_id", item)))
            return False
        self._queue.append(item)
        return True

    def drain(self) -> list[Any]:
        """Return and clear the queued items."""
        drained, self._queue = self._queue, []
        return drained

    def depth(self) -> int:
        """Return the current queue depth."""
        return len(self._queue)

    def stats(self) -> dict[str, int]:
        """Return backpressure statistics."""
        return {"queue_limit": self.queue_limit, "deferred": self.deferred}


# --------------------------------------------------------------------- coordinator


@dataclass(frozen=True, slots=True)
class CoordinatorAnswer:
    """One answer to a coordinator data request with its exactness class (section 6)."""

    operation: str
    exactness: ExactnessClass
    payload: object
    estimate_metadata: dict[str, object] | None

    def to_dict(self) -> dict[str, object]:
        """Return the answer as a plain dictionary."""
        return {
            "operation": self.operation,
            "exactness": self.exactness.value,
            "payload": self.payload,
            "estimate_metadata": self.estimate_metadata,
        }


class MainBrainCoordinatorAdapter:
    """Select retrieval and analysis services by exactness class (section 6)."""

    def __init__(self) -> None:
        """Create one coordinator adapter with request counters."""
        self.requests: list[str] = []
        self.two_stage_exact_lookups = 0
        self.exact_lookups_shortcircuited = 0

    def service_for(self, operation: str, exactness: ExactnessClass) -> str:
        """Return the service the coordinator invokes for one exactness class."""
        self.requests.append(operation)
        if exactness is ExactnessClass.EXACT_MEMBERSHIP:
            return "authoritative exact lookup (bloom precheck only with complete watermark)"
        if exactness is ExactnessClass.BOUNDED_ERROR_ESTIMATE:
            return "approximate structure query with estimate metadata"
        if exactness is ExactnessClass.CANDIDATE_SIMILARITY:
            return "candidate generation followed by exact or semantic verification"
        if exactness is ExactnessClass.CRYPTOGRAPHIC_INTEGRITY:
            return "content-addressed or merkle verification"
        raise ValueError(f"unknown exactness class {exactness!r}")

    def request(
        self,
        operation: str,
        exactness: ExactnessClass,
        payload: object,
        estimate: object | None = None,
    ) -> CoordinatorAnswer:
        """Answer one data request, attaching estimate metadata when approximate."""
        if exactness is ExactnessClass.BOUNDED_ERROR_ESTIMATE and estimate is None:
            raise ValueError(
                "bounded-error answers must carry estimate metadata (section 5.1)"
            )
        metadata = estimate_metadata(estimate) if estimate is not None else None
        if metadata is not None and metadata.get("exact"):
            raise ValueError(
                "exact results must not be routed through the approximate path"
            )
        return CoordinatorAnswer(
            operation=operation,
            exactness=exactness,
            payload=payload,
            estimate_metadata=metadata,
        )

    def record_two_stage_lookup(self, shortcircuited: bool) -> None:
        """Record one two-stage exact lookup outcome (bloom metric)."""
        self.two_stage_exact_lookups += 1
        if shortcircuited:
            self.exact_lookups_shortcircuited += 1

    def stats(self) -> dict[str, int]:
        """Return coordinator statistics."""
        return {
            "requests": len(self.requests),
            "two_stage_exact_lookups": self.two_stage_exact_lookups,
            "exact_lookups_shortcircuited": self.exact_lookups_shortcircuited,
        }


# --------------------------------------------------------------------- workspace


@dataclass(slots=True)
class WorkspaceDataPublisher:
    """Publish versioned data-change events to the shared cognitive workspace (section 6).

    Approximate indexes are registered as derived views with explicit
    watermarks; they never become workspace authority (source section 6
    critical boundary)."""
    workspace_id: str = "ore-data-infrastructure"
    derived_views: dict[str, int] = field(default_factory=dict)
    published_event_ids: list[str] = field(default_factory=list)
    _revision: int = 0

    def publish_change_event(self, event: DataChangeEvent) -> Any:
        """Convert one data-change event into a typed workspace event (Step 10)."""
        try:  # pragma: no cover - repository integration
            from ore_workspace.models import WorkspaceEvent

            self._revision += 1
            workspace_event = WorkspaceEvent(
                workspace_id=self.workspace_id,
                workspace_revision=self._revision,
                event_type=f"data.{event.event_type.value}",
                changed_fields=(event.entity_key,),
                producer_module="ore_data",
                finding_ids=(),
            )
            self.published_event_ids.append(workspace_event.event_id)
            return workspace_event
        except ImportError:  # pragma: no cover - standalone fallback
            self._revision += 1
            record = {
                "workspace_id": self.workspace_id,
                "workspace_revision": self._revision,
                "event_type": f"data.{event.event_type.value}",
                "changed_fields": (event.entity_key,),
                "producer_module": "ore_data",
                "event_id": f"data-{event.event_id}",
            }
            self.published_event_ids.append(str(record["event_id"]))
            return record

    def register_derived_view(self, index_id: str, watermark: int) -> None:
        """Register one approximate index as a derived view with its watermark."""
        self.derived_views[index_id] = int(watermark)

    def view_status(self) -> dict[str, object]:
        """Return the derived-view registry for the workspace (not authority)."""
        return {
            "derived_views": dict(self.derived_views),
            "authority": "authoritative stores only; derived views carry watermarks",
        }


# --------------------------------------------------------------------- USG


@dataclass(frozen=True, slots=True)
class ConceptIdentity:
    """One exact authoritative USG concept identity (section 6)."""

    concept_id: str
    label: str
    language: str = ""
    source_reference: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return the identity as a plain dictionary."""
        return {
            "concept_id": self.concept_id,
            "label": self.label,
            "language": self.language,
            "source_reference": self.source_reference,
        }


@runtime_checkable
class ExactConceptResolver(Protocol):
    """Exact semantic identity resolution protocol (source section 6 USG row)."""

    def lookup_concept(self, label: str) -> Sequence[ConceptIdentity]:
        """Return every exact authoritative concept identity for one label."""


class InMemoryConceptResolver:
    """Deterministic exact resolver for tests and local deployments."""

    def __init__(self, identities: Sequence[ConceptIdentity]) -> None:
        """Index identities by normalized label."""
        self._by_label: dict[str, list[ConceptIdentity]] = {}
        for identity in identities:
            self._by_label.setdefault(
                identity.label.strip().casefold(), []
            ).append(identity)

    def lookup_concept(self, label: str) -> Sequence[ConceptIdentity]:
        """Return exact label matches only; no fuzzy matching ever."""
        return tuple(self._by_label.get(label.strip().casefold(), ()))


class UsgIndexResolver:
    """Wrap the repository's ``ore_usg`` retrieval index as an exact resolver."""

    def __init__(self, index: Any) -> None:
        """Adapt one ``ore_usg.retrieval.USGKnowledgeIndex`` instance."""
        if not hasattr(index, "lookup"):
            raise TypeError(
                "the USG resolver requires an ore_usg retrieval index with lookup()"
            )
        self._index = index

    def lookup_concept(self, label: str) -> Sequence[ConceptIdentity]:
        """Return exact concept matches from the USG knowledge index."""
        result = self._index.lookup(label, kind="SEMANTIC_CONCEPT")
        if not getattr(result, "exact_match", False):
            return ()
        return tuple(
            ConceptIdentity(
                concept_id=item.record_id,
                label=item.label,
                language=item.language,
                source_reference=item.source_reference,
            )
            for item in result.items
        )


@dataclass(frozen=True, slots=True)
class MergeVerificationOutcome:
    """USG-verified duplicate decision for one candidate pair (Step 42/43)."""

    pair: tuple[str, str]
    may_merge: bool
    reason: str
    resolved_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return the outcome as a plain dictionary."""
        return {
            "pair": list(self.pair),
            "may_merge": self.may_merge,
            "reason": self.reason,
            "resolved_ids": list(self.resolved_ids),
        }


class USGSemanticVerificationAdapter:
    """Verify similarity candidates against authoritative USG identity (section 6).

    Sketches, MinHash, LSH and SimHash cannot establish semantic truth
    (source 1.2/6): a merge is permitted only when exact concept identity
    resolution confirms the pair, and byte-identical or verified near
    duplicates still carry their verification provenance downstream.
    """

    def __init__(self, resolver: ExactConceptResolver) -> None:
        """Create one verification adapter over one exact resolver."""
        self.resolver = resolver
        self.verifications: list[MergeVerificationOutcome] = []

    def verify_candidate_merge(
        self,
        candidate: VerifiedSimilarity,
        *,
        first_label: str,
        second_label: str,
    ) -> MergeVerificationOutcome:
        """Resolve both labels exactly before permitting any merge (Step 42)."""
        first_ids = tuple(
            identity.concept_id for identity in self.resolver.lookup_concept(first_label)
        )
        second_ids = tuple(
            identity.concept_id for identity in self.resolver.lookup_concept(second_label)
        )
        shared = tuple(sorted(set(first_ids) & set(second_ids)))
        if shared:
            outcome = MergeVerificationOutcome(
                pair=candidate.pair,
                may_merge=True,
                reason=(
                    "authoritative USG concept identity resolves both records to "
                    f"{shared[0]!r}; similarity only nominated the candidate"
                ),
                resolved_ids=shared,
            )
        elif first_ids and second_ids:
            outcome = MergeVerificationOutcome(
                pair=candidate.pair,
                may_merge=False,
                reason=(
                    "USG resolves both records to distinct authoritative concepts; "
                    "merging distinct concepts on similarity alone is forbidden"
                ),
                resolved_ids=tuple(sorted(set(first_ids) | set(second_ids))),
            )
        elif (
            candidate.verification_method == "canonical_byte_equality"
            and candidate.verified
        ):
            outcome = MergeVerificationOutcome(
                pair=candidate.pair,
                may_merge=True,
                reason=(
                    "byte-identical duplicates confirmed by canonical byte "
                    "equality; USG identity resolution found no conflicting concepts"
                ),
            )
        else:
            outcome = MergeVerificationOutcome(
                pair=candidate.pair,
                may_merge=False,
                reason=(
                    "insufficient authoritative identity evidence; similarity "
                    "candidates are never merged without exact verification"
                ),
            )
        self.verifications.append(outcome)
        return outcome


def cros_cannot_relax_correctness(adapter: CROSDataBudgetAdapter) -> bool:
    """Acceptance check: CROS cannot relax correctness or integrity (section 6)."""
    checks = (
        ("target_false_positive_rate", 1),
        ("count_min_epsilon", 0),
        ("count_min_delta", 0),
        ("hyperloglog_precision", 0),
        ("hash_algorithm", 0),
        ("similarity_verification_required", 0),
    )
    return all(adapter.try_violate_correctness(name, value) for name, value in checks)


def estimates_reach_coordinator_with_metadata(
    coordinator: MainBrainCoordinatorAdapter,
    operation: str,
    estimate: Any,
    payload: object,
) -> CoordinatorAnswer:
    """Acceptance check: estimates reach the coordinator with metadata (section 6)."""
    answer = coordinator.request(operation, ExactnessClass.BOUNDED_ERROR_ESTIMATE, payload, estimate)
    return answer


__all__ = [
    "ADAPTABLE_PARAMETERS",
    "PROTECTED_PARAMETERS",
    "PARAMETER_BOUNDS",
    "StorageTelemetry",
    "telemetry_to_capability_demand",
    "CROSDataBudgetAdapter",
    "BackpressureController",
    "CoordinatorAnswer",
    "MainBrainCoordinatorAdapter",
    "WorkspaceDataPublisher",
    "ConceptIdentity",
    "ExactConceptResolver",
    "InMemoryConceptResolver",
    "UsgIndexResolver",
    "MergeVerificationOutcome",
    "USGSemanticVerificationAdapter",
    "cros_cannot_relax_correctness",
    "estimates_reach_coordinator_with_metadata",
]
