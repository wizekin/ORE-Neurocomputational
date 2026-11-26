"""Acceptance tests for event schemas, bus delivery, idempotency, and the store."""
from __future__ import annotations

import pytest

from ore_runtime import (
    ApplyStatus, DeliveryGuarantee, EventBus, EventEnvelope, EventStore,
    IdempotencyStore,
)
from ore_runtime.events.schemas import (
    SCHEMA_VERSION, TASK_SUCCEEDED, assert_valid_envelope, delivery_guarantee_for,
    validate_envelope,
)
from ore_runtime.exceptions import EventSchemaError


def make_envelope(**overrides):
    """Build a valid task.succeeded envelope with overridable fields."""
    payload = {"task_id": "t1", "committed_revision": 1}
    fields = {
        "event_id": "event_1",
        "event_type": TASK_SUCCEEDED,
        "aggregate_id": "task::t1",
        "aggregate_version": 0,
        "payload": payload,
    }
    fields.update(overrides)
    return EventEnvelope(**fields)


def test_envelope_schema_validation_rejects_missing_fields():
    """Event schema evolution: invalid envelopes are rejected explicitly."""
    envelope = make_envelope(payload={"task_id": "t1"})
    errors = validate_envelope(envelope)
    assert "missing required field 'committed_revision'" in errors
    with pytest.raises(EventSchemaError):
        assert_valid_envelope(envelope)


def test_envelope_schema_rejects_unknown_versions():
    """Unsupported schema versions and unknown types are rejected explicitly."""
    envelope = make_envelope(schema_version="9.9.9")
    assert validate_envelope(envelope)  # non-empty error tuple
    from ore_runtime.exceptions import EventSchemaError as SchemaError
    with pytest.raises(SchemaError):
        assert_valid_envelope(envelope)
    with pytest.raises(SchemaError):
        assert_valid_envelope(make_envelope(event_type="nonsense.event"))


def test_migration_without_registered_path_is_rejected():
    """Envelope migration rejects versions without an explicit registered path."""
    envelope = make_envelope(schema_version="0.0.9")
    with pytest.raises(EventSchemaError):
        envelope.migrate(SCHEMA_VERSION)


def test_idempotent_consumer_applies_effect_exactly_once():
    """Duplicate event: redelivering an event never applies its effect twice."""
    applied = []

    def effect() -> None:
        """Count logical applications of the consumer effect."""
        applied.append(1)

    store = IdempotencyStore()
    first, _ = store.apply_once("consumer", "event", effect)
    second, _ = store.apply_once("consumer", "event", effect)
    assert first is ApplyStatus.APPLIED
    assert second is ApplyStatus.ALREADY_APPLIED
    assert len(applied) == 1


def test_bus_delivers_to_matching_subscribers_with_predicates():
    """Consumers subscribe by event type and predicate, not direct calls."""
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe("a", lambda envelope: seen.append(envelope.event_type),
                  event_types=(TASK_SUCCEEDED,))
    bus.subscribe("b", lambda envelope: seen.append("predicate"),
                  predicate=lambda envelope: envelope.payload.get("task_id") == "special")
    envelope = make_envelope(event_id="event_x")
    outcomes = bus.publish(envelope)
    assert [outcome.consumer_id for outcome in outcomes] == ["a"]
    assert seen == [TASK_SUCCEEDED]
    special = make_envelope(event_id="event_y", payload={"task_id": "special",
                                                        "committed_revision": 2})
    outcomes = bus.publish(special)
    assert {outcome.consumer_id for outcome in outcomes} == {"a", "b"}
    bus.close()


def test_bus_isolates_consumer_failures():
    """A failing consumer does not block delivery to other consumers."""
    bus = EventBus()
    received: list[str] = []

    def broken(envelope) -> None:
        """Always raise to test delivery isolation."""
        raise RuntimeError("consumer failure")

    bus.subscribe("broken", broken, event_types=(TASK_SUCCEEDED,))
    bus.subscribe("healthy",
                  lambda envelope: received.append(envelope.event_id),
                  event_types=(TASK_SUCCEEDED,))
    outcomes = bus.publish(make_envelope(event_id="event_z"))
    by_consumer = {outcome.consumer_id: outcome for outcome in outcomes}
    assert by_consumer["broken"].status.value == "failed"
    assert by_consumer["healthy"].status.value == "delivered"
    assert received == ["event_z"]
    bus.close()


def test_causal_trace_reconstructs_the_event_chain():
    """Causal trace: correlation and causation identifiers reconstruct the chain."""
    bus = EventBus()
    store = EventStore()
    bus.subscribe("history", lambda envelope: store.append(envelope))
    root = make_envelope(event_id="root", correlation_id="corr-1",
                         aggregate_version=0)
    child = make_envelope(event_id="child", causation_id="root",
                          correlation_id="corr-1", aggregate_version=1)
    grandchild = make_envelope(event_id="grandchild", causation_id="child",
                               correlation_id="corr-1", aggregate_version=2)
    for envelope in (root, child, grandchild):
        bus.publish(envelope)
    chain = store.causal_chain("corr-1")
    assert [row["event_id"] for row in chain] == ["root", "child", "grandchild"]
    assert store.by_causation("root")[0].envelope.event_id == "child"
    bus.close()


def test_event_store_rejects_out_of_order_aggregate_versions():
    """Per-aggregate order: out-of-order transitions are quarantined."""
    store = EventStore()
    first = make_envelope(event_id="e1", aggregate_id="task::x", aggregate_version=0)
    store.append(first)
    skipped = make_envelope(event_id="e2", aggregate_id="task::x", aggregate_version=5)
    with pytest.raises(ValueError):
        store.append(skipped)
    assert len(store.quarantined()) == 1
    strict = make_envelope(event_id="e3", aggregate_id="task::x", aggregate_version=5)
    quarantined = store.append(strict, strict=False)
    assert quarantined.expected_version == 1
    assert len(store.quarantined()) == 2


def test_event_sourcing_replay_reduces_committed_history():
    """Event sourcing: State_n equals Reduce(State_0, committed events)."""
    store = EventStore()
    for index in range(4):
        store.append(make_envelope(
            event_id=f"e{index}", aggregate_id="task::chain", aggregate_version=index,
            payload={"task_id": f"t{index}", "committed_revision": index},
        ))
    reducer = lambda state, envelope: {**state, envelope.payload["task_id"]: True}
    replayed = store.replay(reducer, {}, aggregate_id="task::chain")
    assert replayed == {"t0": True, "t1": True, "t2": True, "t3": True}
    store.save_snapshot("task::chain", replayed)
    assert store.load_snapshot("task::chain").state == replayed


def test_projection_reports_its_own_staleness():
    """CQRS: read projections expose their freshness instead of hiding staleness."""
    store = EventStore()
    for index in range(3):
        store.append(make_envelope(
            event_id=f"p{index}", aggregate_id="proj", aggregate_version=index,
        ))
    projection = store.build_projection("counts", lambda state, env: state, 0,
                                         upto_sequence=1)
    assert projection.current_through_sequence == 1
    assert projection.staleness(store.global_sequence) == 2


def test_delivery_guarantees_are_declared_not_claimed():
    """No hidden exactly-once claim: guarantees are labeled per channel."""
    guarantee = delivery_guarantee_for(TASK_SUCCEEDED)
    assert guarantee is DeliveryGuarantee.AT_LEAST_ONCE
    assert DeliveryGuarantee.EFFECTIVELY_ONCE.value == "effectively_once"


def test_effectively_once_delivery_deduplicates_redelivery():
    """Redelivery of one event to an effectively-once consumer applies once."""
    bus = EventBus()
    calls: list[str] = []
    bus.subscribe(
        "managed",
        lambda envelope: calls.append(envelope.event_id),
        event_types=(TASK_SUCCEEDED,),
        delivery_guarantee=DeliveryGuarantee.EFFECTIVELY_ONCE,
    )
    envelope = make_envelope(event_id="once-1")
    bus.publish(envelope)
    bus.publish(envelope)
    assert calls == ["once-1"]
    stats = bus.stats()
    assert stats["idempotency"]["duplicate_deliveries"] == 1
    bus.close()
