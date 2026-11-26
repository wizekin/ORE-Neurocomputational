"""Typed event envelopes, the event catalogue, and schema evolution rules."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..contracts import DeliveryGuarantee, content_fingerprint, new_id, utc_now
from ..exceptions import EventSchemaError

SCHEMA_VERSION = "0.1.0"

TASK_CREATED = "task.created"
TASK_STATE_CHANGED = "task.state_changed"
TASK_DISPATCHED = "task.dispatched"
TASK_SUCCEEDED = "task.succeeded"
TASK_FAILED = "task.failed"
TASK_CANCELLED = "task.cancelled"
TASK_RETRIED = "task.retried"
TASK_COMPENSATING = "task.compensating"
MODULE_ACTIVATED = "module.activated"
FINDING_PUBLISHED = "finding.published"
EVIDENCE_UPDATED = "evidence.updated"
HYPOTHESIS_CHANGED = "hypothesis.changed"
VERIFICATION_COMPLETED = "verification.completed"
GOAL_STARTED = "goal.started"
GOAL_COMPLETED = "goal.completed"
WORKSPACE_COMMITTED = "workspace.revision_committed"
USG_RECORD_CHANGED = "usg.record_changed"
ARTIFACT_STALE = "artifact.stale"
ARTIFACT_RECOMPUTED = "artifact.recomputed"
RECOVERY_CHECKPOINTED = "recovery.checkpointed"
SAGA_STEP_COMPLETED = "saga.step_completed"
SAGA_COMPENSATED = "saga.compensated"

AGGREGATE_TASK = "task"
AGGREGATE_GOAL = "goal"
AGGREGATE_WORKSPACE = "workspace"
AGGREGATE_USG = "usg"
AGGREGATE_ARTIFACT = "artifact"
AGGREGATE_SAGA = "saga"


@dataclass(frozen=True, slots=True)
class EventSchema:
    """Define one event type's required payload fields and delivery guarantee."""

    event_type: str
    schema_version: str
    aggregate_kind: str
    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = ()
    delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE


_TASK_LIFECYCLE_FIELDS = ("task_id", "from_state", "to_state")

EVENT_CATALOGUE: dict[str, tuple[EventSchema, ...]] = {
    TASK_CREATED: (
        EventSchema(TASK_CREATED, SCHEMA_VERSION, AGGREGATE_TASK,
                    ("task_id", "name", "capability")),
    ),
    TASK_STATE_CHANGED: (
        EventSchema(TASK_STATE_CHANGED, SCHEMA_VERSION, AGGREGATE_TASK,
                    _TASK_LIFECYCLE_FIELDS),
    ),
    TASK_DISPATCHED: (
        EventSchema(TASK_DISPATCHED, SCHEMA_VERSION, AGGREGATE_TASK,
                    ("task_id", "worker_id", "attempt")),
    ),
    TASK_SUCCEEDED: (
        EventSchema(TASK_SUCCEEDED, SCHEMA_VERSION, AGGREGATE_TASK,
                    ("task_id", "committed_revision")),
    ),
    TASK_FAILED: (
        EventSchema(TASK_FAILED, SCHEMA_VERSION, AGGREGATE_TASK,
                    ("task_id", "error")),
    ),
    TASK_CANCELLED: (
        EventSchema(TASK_CANCELLED, SCHEMA_VERSION, AGGREGATE_TASK,
                    ("task_id", "reason")),
    ),
    TASK_RETRIED: (
        EventSchema(TASK_RETRIED, SCHEMA_VERSION, AGGREGATE_TASK,
                    ("task_id", "attempt")),
    ),
    TASK_COMPENSATING: (
        EventSchema(TASK_COMPENSATING, SCHEMA_VERSION, AGGREGATE_TASK,
                    ("task_id", "compensation_step")),
    ),
    MODULE_ACTIVATED: (
        EventSchema(MODULE_ACTIVATED, SCHEMA_VERSION, AGGREGATE_TASK,
                    ("task_id", "module_id")),
    ),
    FINDING_PUBLISHED: (
        EventSchema(FINDING_PUBLISHED, SCHEMA_VERSION, AGGREGATE_WORKSPACE,
                    ("finding_id", "field", "workspace_revision"),
                    optional_fields=("producer_module", "confidence")),
    ),
    EVIDENCE_UPDATED: (
        EventSchema(EVIDENCE_UPDATED, SCHEMA_VERSION, AGGREGATE_WORKSPACE,
                    ("evidence_id", "change")),
    ),
    HYPOTHESIS_CHANGED: (
        EventSchema(HYPOTHESIS_CHANGED, SCHEMA_VERSION, AGGREGATE_WORKSPACE,
                    ("hypothesis_id", "change")),
    ),
    VERIFICATION_COMPLETED: (
        EventSchema(VERIFICATION_COMPLETED, SCHEMA_VERSION, AGGREGATE_WORKSPACE,
                    ("verification_id", "valid")),
    ),
    GOAL_STARTED: (
        EventSchema(GOAL_STARTED, SCHEMA_VERSION, AGGREGATE_GOAL,
                    ("goal_id", "description")),
    ),
    GOAL_COMPLETED: (
        EventSchema(GOAL_COMPLETED, SCHEMA_VERSION, AGGREGATE_GOAL,
                    ("goal_id", "status")),
    ),
    WORKSPACE_COMMITTED: (
        EventSchema(WORKSPACE_COMMITTED, SCHEMA_VERSION, AGGREGATE_WORKSPACE,
                    ("committed_revision", "read_version"),
                    optional_fields=("change_keys",)),
    ),
    USG_RECORD_CHANGED: (
        EventSchema(USG_RECORD_CHANGED, SCHEMA_VERSION, AGGREGATE_USG,
                    ("record_id", "record_version"),
                    delivery_guarantee=DeliveryGuarantee.AT_LEAST_ONCE,
        ),
    ),
    ARTIFACT_STALE: (
        EventSchema(ARTIFACT_STALE, SCHEMA_VERSION, AGGREGATE_ARTIFACT,
                    ("artifact_id", "reason")),
    ),
    ARTIFACT_RECOMPUTED: (
        EventSchema(ARTIFACT_RECOMPUTED, SCHEMA_VERSION, AGGREGATE_ARTIFACT,
                    ("artifact_id", "artifact_hash")),
    ),
    RECOVERY_CHECKPOINTED: (
        EventSchema(RECOVERY_CHECKPOINTED, SCHEMA_VERSION, AGGREGATE_TASK,
                    ("task_id",)),
    ),
    SAGA_STEP_COMPLETED: (
        EventSchema(SAGA_STEP_COMPLETED, SCHEMA_VERSION, AGGREGATE_SAGA,
                    ("saga_id", "step_name", "step_index")),
    ),
    SAGA_COMPENSATED: (
        EventSchema(SAGA_COMPENSATED, SCHEMA_VERSION, AGGREGATE_SAGA,
                    ("saga_id", "step_name")),
    ),
}

#: Explicit migrations from older supported schema versions to the current one.
SCHEMA_MIGRATIONS: dict[tuple[str, str], Callable[[Mapping[str, Any]], Mapping[str, Any]]] = {
}


def supported_versions(event_type: str) -> tuple[str, ...]:
    """Return every registered schema version for one event type."""
    return tuple(schema.schema_version for schema in EVENT_CATALOGUE.get(event_type, ()))


def schema_for(event_type: str, schema_version: str) -> EventSchema:
    """Return the exact schema for one event type and version, or raise."""
    for schema in EVENT_CATALOGUE.get(event_type, ()):
        if schema.schema_version == schema_version:
            return schema
    raise EventSchemaError(
        f"unsupported schema version '{schema_version}' for event type '{event_type}'",
        errors=("unknown_schema_version",),
    )


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    """Carry one durable, uniquely identified, causally linked event."""

    event_id: str
    event_type: str
    aggregate_id: str
    aggregate_version: int
    sequence: int = 0
    payload: Mapping[str, Any] = field(default_factory=dict)
    causation_id: str | None = None
    correlation_id: str | None = None
    occurred_at: str = field(default_factory=utc_now)
    producer: str = "ore_runtime"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate identifiers, detach the payload, and normalize enums."""
        if not self.event_id.strip():
            raise ValueError("event_id must be non-empty.")
        if not self.event_type.strip():
            raise ValueError("event_type must be non-empty.")
        if not self.aggregate_id.strip():
            raise ValueError("aggregate_id must be non-empty.")
        if self.aggregate_version < 0:
            raise ValueError("aggregate_version cannot be negative.")
        if self.sequence < 0:
            raise ValueError("sequence cannot be negative.")
        if not self.schema_version.strip():
            raise ValueError("schema_version must be non-empty.")
        object.__setattr__(self, "payload", dict(self.payload))

    @staticmethod
    def create(
        event_type: str, aggregate_id: str, aggregate_version: int,
        payload: Mapping[str, Any], *, producer: str = "ore_runtime",
        causation_id: str | None = None, correlation_id: str | None = None,
        sequence: int = 0, schema_version: str = SCHEMA_VERSION,
    ) -> "EventEnvelope":
        """Create a new unique envelope with a fresh event ID and timestamp."""
        return EventEnvelope(
            event_id=new_id("event"),
            event_type=event_type,
            aggregate_id=aggregate_id,
            aggregate_version=aggregate_version,
            sequence=sequence,
            payload=dict(payload),
            causation_id=causation_id,
            correlation_id=correlation_id,
            producer=producer,
            schema_version=schema_version,
        )

    @property
    def content_hash(self) -> str:
        """Hash the canonical payload so equal payloads share one identifier."""
        return content_fingerprint(
            {
                "event_type": self.event_type,
                "aggregate_id": self.aggregate_id,
                "aggregate_version": self.aggregate_version,
                "payload": dict(self.payload),
                "schema_version": self.schema_version,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable envelope record for the event store and outbox."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "aggregate_version": self.aggregate_version,
            "sequence": self.sequence,
            "payload": dict(self.payload),
            "causation_id": self.causation_id,
            "correlation_id": self.correlation_id,
            "occurred_at": self.occurred_at,
            "producer": self.producer,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "EventEnvelope":
        """Rebuild an envelope from its serialized record."""
        return cls(
            event_id=record["event_id"],
            event_type=record["event_type"],
            aggregate_id=record["aggregate_id"],
            aggregate_version=int(record["aggregate_version"]),
            sequence=int(record.get("sequence", 0)),
            payload=dict(record.get("payload", {})),
            causation_id=record.get("causation_id"),
            correlation_id=record.get("correlation_id"),
            occurred_at=record.get("occurred_at", utc_now()),
            producer=record.get("producer", "ore_runtime"),
            schema_version=record.get("schema_version", SCHEMA_VERSION),
        )

    def migrate(self, target_version: str = SCHEMA_VERSION) -> "EventEnvelope":
        """Migrate one envelope through registered migrations or reject explicitly."""
        if target_version == self.schema_version:
            return self
        migration = SCHEMA_MIGRATIONS.get((self.schema_version, target_version))
        if migration is None:
            raise EventSchemaError(
                "no registered migration from "
                f"{self.schema_version} to {target_version} for '{self.event_type}'",
                errors=("missing_migration",),
            )
        migrated_payload = migration(self.payload)
        return EventEnvelope(
            event_id=self.event_id,
            event_type=self.event_type,
            aggregate_id=self.aggregate_id,
            aggregate_version=self.aggregate_version,
            sequence=self.sequence,
            payload=migrated_payload,
            causation_id=self.causation_id,
            correlation_id=self.correlation_id,
            occurred_at=self.occurred_at,
            producer=self.producer,
            schema_version=target_version,
        )


def validate_envelope(envelope: EventEnvelope) -> tuple[str, ...]:
    """Validate one envelope against its catalogue schema and return its errors."""
    errors: list[str] = []
    schemas = EVENT_CATALOGUE.get(envelope.event_type)
    if not schemas:
        return (f"unknown event type '{envelope.event_type}'",)
    schema = next(
        (item for item in schemas if item.schema_version == envelope.schema_version), None,
    )
    if schema is None:
        supported = ", ".join(item.schema_version for item in schemas)
        errors.append(
            f"schema version '{envelope.schema_version}' is not supported for "
            f"'{envelope.event_type}' (supported: {supported})"
        )
        return tuple(errors)
    for required in schema.required_fields:
        if required not in envelope.payload:
            errors.append(f"missing required field '{required}'")
    unknown = set(envelope.payload) - set(schema.required_fields) - set(schema.optional_fields)
    if unknown:
        errors.append(f"unknown fields: {sorted(unknown)}")
    return tuple(errors)


def assert_valid_envelope(envelope: EventEnvelope) -> None:
    """Raise EventSchemaError when an envelope violates its schema contract."""
    errors = validate_envelope(envelope)
    if errors:
        raise EventSchemaError(
            f"invalid event envelope '{envelope.event_id}' of type "
            f"'{envelope.event_type}': " + "; ".join(errors),
            errors=errors,
        )


def event_validity(envelope: EventEnvelope) -> bool:
    """Return whether the envelope satisfies Schema, RequiredFields, and provenance."""
    return (
        not validate_envelope(envelope)
        and bool(envelope.event_id)
        and bool(envelope.producer)
    )


def delivery_guarantee_for(event_type: str) -> DeliveryGuarantee:
    """Return the documented delivery guarantee for one event type."""
    schemas = EVENT_CATALOGUE.get(event_type)
    if not schemas:
        raise KeyError(f"unknown event type '{event_type}'")
    return schemas[0].delivery_guarantee


def catalogue_as_dicts() -> dict[str, list[dict[str, Any]]]:
    """Return the catalogue as serializable records for documentation and tests."""
    return {
        event_type: [
            {
                "schema_version": schema.schema_version,
                "aggregate_kind": schema.aggregate_kind,
                "required_fields": list(schema.required_fields),
                "optional_fields": list(schema.optional_fields),
                "delivery_guarantee": schema.delivery_guarantee.value,
            }
            for schema in schemas
        ]
        for event_type, schemas in sorted(EVENT_CATALOGUE.items())
    }
