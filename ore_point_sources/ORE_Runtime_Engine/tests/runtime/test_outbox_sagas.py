"""Acceptance tests for the transactional outbox and saga compensation."""
from __future__ import annotations

import pytest

from ore_runtime import (
    EventEnvelope, SagaCoordinator, SagaStep, TransactionalOutbox,
)
from ore_runtime.events.schemas import TASK_SUCCEEDED
from ore_runtime.exceptions import SagaCompensationIncompleteError


def make_envelope(event_id: str = "event_1") -> EventEnvelope:
    """Build one valid envelope for outbox publication tests."""
    return EventEnvelope(
        event_id=event_id,
        event_type=TASK_SUCCEEDED,
        aggregate_id="task::t1",
        aggregate_version=0,
        payload={"task_id": "t1", "committed_revision": 1},
    )


def test_committed_state_change_always_has_durable_outbox_row():
    """Outbox reliability: a committed change keeps its row even after crash."""
    fake_now = {"value": 100.0}

    def fake_clock() -> float:
        """Advance a deterministic clock on every access."""
        fake_now["value"] += 1.0
        return fake_now["value"]

    outbox = TransactionalOutbox(max_attempts=5, backoff_seconds=0.01,
                                 clock=fake_clock)
    envelope = make_envelope()
    outbox.record(envelope)
    publish_attempts: list[int] = []

    def crashing_publisher(envelope) -> None:
        """Simulate a publisher that crashes before delivery."""
        publish_attempts.append(1)
        raise RuntimeError("publisher crashed")

    delivered = outbox.drain(crashing_publisher)
    assert delivered == ()
    assert outbox.pending_rows()
    assert publish_attempts

    delivered = outbox.drain(lambda env: None)
    assert [row.envelope.event_id for row in delivered] == [envelope.event_id]
    assert outbox.pending_rows() == ()


def test_outbox_marks_delivered_only_after_successful_publication():
    """Rows transition PENDING to DELIVERED only on publication success."""
    outbox = TransactionalOutbox()
    outbox.record(make_envelope("e1"))
    outbox.record(make_envelope("e2"))
    published: list[str] = []

    def publisher(envelope) -> None:
        """Publish and record successfully delivered events."""
        published.append(envelope.event_id)

    outbox.drain(publisher)
    assert published == ["e1", "e2"]
    assert outbox.stats()["pending"] == 0
    assert outbox.stats()["delivered"] == 2


def test_outbox_transaction_records_rows_with_the_commit():
    """The transaction callback pairs state change and outbox rows atomically."""
    outbox = TransactionalOutbox()
    transaction = outbox.as_transaction([make_envelope("e-txn")])
    rows = transaction()
    assert [row.envelope.event_id for row in rows] == ["e-txn"]
    assert outbox.pending_rows()[0].envelope.event_id == "e-txn"


def test_saga_compensation_runs_defined_actions_in_reverse():
    """Saga compensation: a failed workflow executes compensations in order."""
    log: list[str] = []

    def build_steps() -> tuple[SagaStep, ...]:
        """Build a three-step saga with compensations appending to the log."""
        return (
            SagaStep(name="reserve",
                     action=lambda context: log.append("reserve"),
                     compensate=lambda context, result: log.append("undo-reserve")),
            SagaStep(name="charge",
                     action=lambda context: log.append("charge"),
                     compensate=lambda context, result: log.append("undo-charge")),
            SagaStep(name="ship",
                     action=lambda context: (_ for _ in ()).throw(RuntimeError("ship failed"))),
        )

    coordinator = SagaCoordinator()
    outcome = coordinator.run("order-1", build_steps(), {})
    assert outcome.completed is False
    assert outcome.failed_step == "ship"
    assert log == ["reserve", "charge", "undo-charge", "undo-reserve"]
    assert outcome.compensated_steps == ("charge", "reserve")


def test_irreversible_step_records_incomplete_compensation():
    """Irreversible steps record residual effects requiring human review."""
    coordinator = SagaCoordinator()
    steps = (
        SagaStep(name="bill", action=lambda context: "billed", irreversible=True),
        SagaStep(name="pack",
                 action=lambda context: (_ for _ in ()).throw(RuntimeError("pack failed"))),
    )
    with pytest.raises(SagaCompensationIncompleteError):
        coordinator.run("order-2", steps, {})


def test_successful_saga_completes_without_compensation():
    """A fully successful saga reports completion with empty compensation."""
    coordinator = SagaCoordinator()
    steps = (
        SagaStep(name="a", action=lambda context: 1),
        SagaStep(name="b", action=lambda context: 2),
    )
    outcome = coordinator.run("order-3", steps, {})
    assert outcome.completed is True
    assert outcome.compensated_steps == ()
    assert coordinator.saga_state("order-3")["status"] == "completed"


def test_saga_resume_continues_from_persisted_step():
    """Sagas persist their position so recovery resumes the next step."""
    coordinator = SagaCoordinator()
    steps = (
        SagaStep(name="first", action=lambda context: context.setdefault("runs", []).append("first")),
        SagaStep(name="second", action=lambda context: context.setdefault("runs", []).append("second")),
    )
    state = coordinator.start("order-4", steps)
    state.next_step = 1
    context: dict = {"runs": []}
    outcome = coordinator.resume("order-4", context)
    assert outcome.completed is True
    assert context["runs"] == ["second"]
