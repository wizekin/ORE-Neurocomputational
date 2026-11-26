"""Event-driven coordination: schemas, bus, outbox, idempotency, and sagas."""
from .bus import DeliveryOutcome, DeliveryStatus, EventBus, PublishRecord, Subscription
from .idempotency import ApplyStatus, IdempotencyStore
from .outbox import OutboxRow, OutboxStatus, TransactionalOutbox
from .sagas import (
    SagaCoordinator, SagaOutcome, SagaState, SagaStep, SagaStepOutcome,
    build_compensatable_workflow,
)
from .schemas import (
    EVENT_CATALOGUE, SCHEMA_VERSION, EventEnvelope, EventSchema, assert_valid_envelope,
    catalogue_as_dicts, event_validity, supported_versions, validate_envelope,
)

__all__ = [
    "ApplyStatus", "EVENT_CATALOGUE", "EventBus", "EventEnvelope", "EventSchema",
    "IdempotencyStore", "OutboxRow", "OutboxStatus", "PublishRecord",
    "SCHEMA_VERSION", "SagaCoordinator", "SagaOutcome", "SagaState", "SagaStep",
    "SagaStepOutcome", "Subscription", "TransactionalOutbox",
    "assert_valid_envelope", "build_compensatable_workflow", "catalogue_as_dicts",
    "event_validity", "supported_versions", "validate_envelope",
]
