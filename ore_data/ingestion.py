"""Ingestion and update path for authoritative records (Phase B).

Steps 7 through 11: every incoming record is validated before it may touch
any index or integrity commitment (Step 7); durable event identities make
retries idempotent (Step 8); authoritative state commits before derived
updates through a transactional store with an outbox (Steps 9-10); records
route to the storage class their contract assigns (Step 11).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Mapping

from ore_data.contracts import (
    DataClass,
    DataFamilyContract,
    canonical_key,
)


class RecordStatus(str, Enum):
    """Outcome of one ingestion attempt."""

    COMMITTED = "committed"
    DUPLICATE_APPLIED = "duplicate_applied"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"


class ChangeEventType(str, Enum):
    """Versioned data-change event types (Step 10)."""

    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    SNAPSHOT = "snapshot"
    MERGE = "merge"
    INTEGRITY = "integrity"


def _now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True, slots=True)
class ProvenanceInfo:
    """Source provenance of one incoming record (Step 7)."""

    source: str
    actor: str
    authorization: str | None = None
    occurred_at: str | None = None

    def __post_init__(self) -> None:
        """Validate provenance identity."""
        if not self.source.strip():
            raise ValueError("provenance source must be non-empty")
        if not self.actor.strip():
            raise ValueError("provenance actor must be non-empty")


@dataclass(frozen=True, slots=True)
class ValidationPolicy:
    """Validation rules for one ingestion family (Step 7)."""

    family_id: str
    allowed_schema_versions: frozenset[str]
    required_fields: tuple[str, ...]
    max_record_bytes: int = 1 << 20
    authorized_sources: frozenset[str] | None = None
    quarantine_enabled: bool = True

    def __post_init__(self) -> None:
        """Validate the policy itself."""
        if not self.family_id.strip():
            raise ValueError("family_id must be non-empty")
        if not self.allowed_schema_versions:
            raise ValueError("allowed_schema_versions must be non-empty")
        if not self.required_fields:
            raise ValueError("required_fields must be non-empty")
        if self.max_record_bytes <= 0:
            raise ValueError("max_record_bytes must be positive")

    def source_authorized(self, source: str) -> bool:
        """Return whether one source may write into this family."""
        if self.authorized_sources is None:
            return True
        return source in self.authorized_sources


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of one record validation (Step 7)."""

    family_id: str
    valid: bool
    errors: tuple[str, ...] = ()
    quarantined: bool = False

    def __post_init__(self) -> None:
        """Keep errors consistent with validity."""
        object.__setattr__(self, "errors", tuple(self.errors))

    def to_dict(self) -> dict[str, Any]:
        """Return the validation outcome as a plain dictionary."""
        return {
            "family_id": self.family_id,
            "valid": self.valid,
            "errors": list(self.errors),
            "quarantined": self.quarantined,
        }


def validate_record(
    record: Mapping[str, Any],
    provenance: ProvenanceInfo,
    policy: ValidationPolicy,
) -> ValidationResult:
    """Validate one record against its policy (Step 7).

    Checks schema version, required fields, provenance source
    authorization, timestamps and size limits. Invalid records are rejected
    or quarantined; they must never update indexes or integrity commitments.
    """
    errors: list[str] = []
    if not isinstance(record, Mapping):
        return ValidationResult(policy.family_id, False, ("record must be a mapping",), True)
    schema_version = record.get("schema_version")
    if schema_version not in policy.allowed_schema_versions:
        errors.append(
            f"schema_version {schema_version!r} not in {sorted(policy.allowed_schema_versions)}"
        )
    for required in policy.required_fields:
        value = record.get(required)
        if value is None or (isinstance(value, str) and not value.strip()):
            errors.append(f"required field {required!r} missing or empty")
    if "entity_key" in record and not str(record["entity_key"]).strip():
        errors.append("entity_key must be non-empty text")
    if not policy.source_authorized(provenance.source):
        errors.append(f"source {provenance.source!r} is not authorized for this family")
    if "created_at" in record and not str(record["created_at"]).strip():
        errors.append("created_at must be non-empty when present")
    record_bytes = len(str(dict(record)).encode("utf-8"))
    if record_bytes > policy.max_record_bytes:
        errors.append(
            f"record size {record_bytes}B exceeds limit {policy.max_record_bytes}B"
        )
    valid = not errors
    return ValidationResult(
        policy.family_id,
        valid,
        tuple(errors),
        quarantined=(not valid) and policy.quarantine_enabled,
    )


@dataclass(frozen=True, slots=True)
class CommitReceipt:
    """Receipt of one authoritative commit (Step 9)."""

    event_id: str
    entity_key: str
    status: RecordStatus
    sequence: int
    idempotency_key: str
    committed_at: str

    @property
    def applied(self) -> bool:
        """Whether the authoritative mutation is (already) applied."""
        return self.status in (RecordStatus.COMMITTED, RecordStatus.DUPLICATE_APPLIED)

    def to_dict(self) -> dict[str, Any]:
        """Return the receipt as a plain dictionary."""
        return {
            "event_id": self.event_id,
            "entity_key": self.entity_key,
            "status": self.status.value,
            "sequence": self.sequence,
            "idempotency_key": self.idempotency_key,
            "committed_at": self.committed_at,
        }


@dataclass(frozen=True, slots=True)
class DataChangeEvent:
    """One versioned data-change event (Step 10)."""

    event_id: str
    event_type: ChangeEventType
    entity_key: str
    source: str
    schema_version: str
    sequence: int
    causal_reference: str | None = None
    payload_digest: str | None = None
    published_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        """Return the event as a plain dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "entity_key": self.entity_key,
            "source": self.source,
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "causal_reference": self.causal_reference,
            "payload_digest": self.payload_digest,
            "published_at": self.published_at,
        }


@dataclass(frozen=True, slots=True)
class EventReceipt:
    """Receipt of one published change event (Step 10)."""

    event_id: str
    sequence: int
    duplicate: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the receipt as a plain dictionary."""
        return {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "duplicate": self.duplicate,
        }


class Outbox:
    """Reliable publication of change events after authoritative commits (Steps 9-10).

    Events enter the outbox as part of the authoritative commit sequence and
    are delivered later; consumers detect duplicates by event identity, so
    at-least-once delivery yields exactly-once effects.
    """

    def __init__(self) -> None:
        """Create an empty outbox."""
        self._events: list[DataChangeEvent] = []
        self._by_event_id: dict[str, DataChangeEvent] = {}

    def append(self, event: DataChangeEvent) -> EventReceipt:
        """Append one event; duplicate event IDs are detected, not duplicated (Step 10)."""
        if event.event_id in self._by_event_id:
            existing = self._by_event_id[event.event_id]
            return EventReceipt(existing.event_id, existing.sequence, True)
        self._events.append(event)
        self._by_event_id[event.event_id] = event
        return EventReceipt(event.event_id, event.sequence, False)

    def pull(self, since_sequence: int = 0) -> tuple[DataChangeEvent, ...]:
        """Return every event after one sequence watermark for replay-safe consumption."""
        return tuple(event for event in self._events if event.sequence > since_sequence)

    def __len__(self) -> int:
        """Return the number of distinct published events."""
        return len(self._events)


class AuthoritativeStore:
    """Transactional source-of-record for mutable authoritative state (Step 9).

    Commits are all-or-nothing per event; event identities make retries
    idempotent; sequence numbers are monotonic and act as the authoritative
    watermark every derived structure reconciles against.
    """

    def __init__(self, family: str = "ore.workspace.findings") -> None:
        """Create an empty authoritative store for one family."""
        if not family.strip():
            raise ValueError("family must be non-empty")
        self.family = family
        self._records: dict[str, tuple[int, Mapping[str, Any]]] = {}
        self._tombstones: dict[str, int] = {}
        self._applied_event_ids: dict[str, int] = {}
        self._sequence = 0
        self.outbox = Outbox()

    # -- commit path -------------------------------------------------------

    def commit_authoritative(
        self,
        record: Mapping[str, Any],
        idempotency_key: str,
        *,
        event_type: ChangeEventType = ChangeEventType.INSERT,
        provenance: ProvenanceInfo | None = None,
        payload_digest: str | None = None,
    ) -> CommitReceipt:
        """Commit one record transactionally (Step 9).

        Retries carrying the same idempotency key return
        ``DUPLICATE_APPLIED`` with the original sequence and never
        double-apply the mutation (Step 8).
        """
        if not idempotency_key.strip():
            raise ValueError("idempotency_key must be non-empty")
        if "entity_key" not in record:
            raise ValueError("record must carry an entity_key")
        entity_key = str(record["entity_key"])

        if idempotency_key in self._applied_event_ids:
            original_sequence = self._applied_event_ids[idempotency_key]
            return CommitReceipt(
                event_id=f"evt-{idempotency_key}",
                entity_key=entity_key,
                status=RecordStatus.DUPLICATE_APPLIED,
                sequence=original_sequence,
                idempotency_key=idempotency_key,
                committed_at=_now(),
            )

        # Transactional commit: record write, sequence advance and outbox
        # append happen together or not at all (single-threaded atomicity).
        self._sequence += 1
        sequence = self._sequence
        event_id = f"evt-{idempotency_key}"
        self._records[entity_key] = (sequence, dict(record))
        self._tombstones.pop(entity_key, None)
        self._applied_event_ids[idempotency_key] = sequence
        receipt = CommitReceipt(
            event_id=event_id,
            entity_key=entity_key,
            status=RecordStatus.COMMITTED,
            sequence=sequence,
            idempotency_key=idempotency_key,
            committed_at=_now(),
        )
        self._publish(receipt, event_type, record, provenance, payload_digest)
        return receipt

    def delete(
        self,
        entity_key: str,
        *,
        idempotency_key: str,
        provenance: ProvenanceInfo | None = None,
    ) -> CommitReceipt:
        """Delete one record authoritatively (Step 10 delete event)."""
        if idempotency_key in self._applied_event_ids:
            original_sequence = self._applied_event_ids[idempotency_key]
            return CommitReceipt(
                event_id=f"evt-{idempotency_key}",
                entity_key=entity_key,
                status=RecordStatus.DUPLICATE_APPLIED,
                sequence=original_sequence,
                idempotency_key=idempotency_key,
                committed_at=_now(),
            )
        self._sequence += 1
        sequence = self._sequence
        self._records.pop(entity_key, None)
        self._tombstones[entity_key] = sequence
        self._applied_event_ids[idempotency_key] = sequence
        receipt = CommitReceipt(
            event_id=f"evt-{idempotency_key}",
            entity_key=entity_key,
            status=RecordStatus.COMMITTED,
            sequence=sequence,
            idempotency_key=idempotency_key,
            committed_at=_now(),
        )
        event = DataChangeEvent(
            event_id=receipt.event_id,
            event_type=ChangeEventType.DELETE,
            entity_key=entity_key,
            source=provenance.source if provenance else "unknown",
            schema_version="0.1.0",
            sequence=sequence,
        )
        self.outbox.append(event)
        return receipt

    def _publish(
        self,
        receipt: CommitReceipt,
        event_type: ChangeEventType,
        record: Mapping[str, Any],
        provenance: ProvenanceInfo | None,
        payload_digest: str | None,
    ) -> None:
        event = DataChangeEvent(
            event_id=receipt.event_id,
            event_type=event_type,
            entity_key=receipt.entity_key,
            source=provenance.source if provenance else "unknown",
            schema_version=str(record.get("schema_version", "0.1.0")),
            sequence=receipt.sequence,
            causal_reference=receipt.idempotency_key,
            payload_digest=payload_digest,
        )
        self.outbox.append(event)

    # -- read path ---------------------------------------------------------

    def lookup(self, entity_key: str) -> Mapping[str, Any] | None:
        """Return the exact authoritative record for one key (Step 4 exact membership)."""
        if entity_key in self._tombstones and entity_key not in self._records:
            return None
        entry = self._records.get(entity_key)
        return entry[1] if entry else None

    def contains(self, entity_key: str) -> bool:
        """Return exact membership for one key."""
        return self.lookup(entity_key) is not None

    def record_sequence(self, entity_key: str) -> int | None:
        """Return the sequence at which one record was last written."""
        entry = self._records.get(entity_key)
        return entry[0] if entry else None

    def keys(self) -> tuple[str, ...]:
        """Return every live authoritative key."""
        return tuple(self._records.keys())

    @property
    def watermark(self) -> int:
        """Return the authoritative watermark: the latest committed sequence."""
        return self._sequence

    def stats(self) -> dict[str, int]:
        """Return store statistics for observability."""
        return {
            "records": len(self._records),
            "tombstones": len(self._tombstones),
            "applied_events": len(self._applied_event_ids),
            "watermark": self._sequence,
        }


class Quarantine:
    """Quarantine holding for invalid records (Step 7)."""

    def __init__(self) -> None:
        """Create an empty quarantine."""
        self.entries: list[tuple[str, Mapping[str, Any], tuple[str, ...]]] = []

    def hold(
        self,
        record: Mapping[str, Any],
        errors: tuple[str, ...],
        family_id: str,
    ) -> None:
        """Hold one invalid record with its errors; it updates no index."""
        self.entries.append((family_id, dict(record), tuple(errors)))

    def size(self) -> int:
        """Return the number of quarantined records."""
        return len(self.entries)


class StorageRouter:
    """Route records to the storage class their contract assigns (Step 11)."""

    def __init__(self, lsm_justified: bool = False) -> None:
        """Create a router; the LSM path stays dormant until benchmarks justify it (Step 44)."""
        self.lsm_justified = lsm_justified
        self.routes: list[tuple[str, str]] = []

    def route(self, contract: DataFamilyContract) -> str:
        """Return the storage class for one family contract."""
        if contract.data_class is DataClass.IMMUTABLE_ARTIFACT:
            route = "content_addressed"
        elif contract.data_class is DataClass.APPEND_ONLY_EVENT and self.lsm_justified:
            route = "lsm_backed"
        elif contract.data_class in (DataClass.EXACT_TRANSACTIONAL, DataClass.APPEND_ONLY_EVENT):
            route = "transactional"
        elif contract.data_class is DataClass.DERIVED_INDEX:
            route = "derived_index"
        elif contract.data_class is DataClass.APPROXIMATE_TELEMETRY:
            route = "approximate_telemetry"
        elif contract.data_class is DataClass.CACHE:
            route = "cache"
        else:  # pragma: no cover - exhaustive enum
            route = "transactional"
        self.routes.append((contract.family_id, route))
        return route


class IngestionService:
    """The Phase B ingestion path assembled from validation, commit, outbox and routing."""

    def __init__(
        self,
        policies: Mapping[str, ValidationPolicy] | None = None,
        *,
        family: str = "ore.workspace.findings",
        lsm_justified: bool = False,
    ) -> None:
        """Create the ingestion service with its validation policies."""
        self.policies: dict[str, ValidationPolicy] = dict(policies or {})
        self.store = AuthoritativeStore(family=family)
        self.quarantine = Quarantine()
        self.router = StorageRouter(lsm_justified=lsm_justified)

    def add_policy(self, policy: ValidationPolicy) -> None:
        """Register one family's validation policy."""
        if policy.family_id in self.policies:
            raise ValueError(f"duplicate policy for family {policy.family_id!r}")
        self.policies[policy.family_id] = policy

    def ingest(
        self,
        record: Mapping[str, Any],
        provenance: ProvenanceInfo,
        idempotency_key: str,
        *,
        payload_digest: str | None = None,
    ) -> CommitReceipt | None:
        """Validate, commit and publish one record (Steps 7-10).

        Invalid records are rejected or quarantined and return ``None``:
        they never reach the authoritative store, the outbox, or any derived
        index.
        """
        family_id = str(record.get("family", self.store.family))
        if family_id not in self.policies:
            self.quarantine.hold(
                record,
                (f"no validation policy for family {family_id!r}",),
                family_id,
            )
            return None
        policy = self.policies[family_id]
        result = validate_record(record, provenance, policy)
        if not result.valid:
            if result.quarantined:
                self.quarantine.hold(record, result.errors, family_id)
            return None
        return self.store.commit_authoritative(
            record,
            idempotency_key,
            provenance=provenance,
            payload_digest=payload_digest,
        )

    def publish_change_event(self, receipt: CommitReceipt) -> EventReceipt:
        """Re-publish one commit's change event (duplicate-safe, Step 10)."""
        for event in self.store.outbox.pull(0):
            if event.event_id == receipt.event_id:
                return self.store.outbox.append(event)
        raise ValueError(f"no change event found for receipt {receipt.event_id!r}")

    def replay_events(self, consumer: Callable[[DataChangeEvent], None], since: int = 0) -> int:
        """Replay change events to one consumer; duplicates are detected upstream."""
        delivered = 0
        for event in self.store.outbox.pull(since):
            consumer(event)
            delivered += 1
        return delivered


__all__ = [
    "RecordStatus",
    "ChangeEventType",
    "ProvenanceInfo",
    "ValidationPolicy",
    "ValidationResult",
    "validate_record",
    "CommitReceipt",
    "DataChangeEvent",
    "EventReceipt",
    "Outbox",
    "AuthoritativeStore",
    "Quarantine",
    "StorageRouter",
    "IngestionService",
    "canonical_key",
]
