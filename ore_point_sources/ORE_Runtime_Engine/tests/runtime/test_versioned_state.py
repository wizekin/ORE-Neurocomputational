"""Acceptance tests for versioned state, optimistic commits, and RCU reclamation."""
from __future__ import annotations

import threading

from ore_runtime import CommitDecision, VersionedStore


def test_reader_sees_exactly_one_committed_revision():
    """Snapshot consistency: a reader sees one committed revision, never partial state."""
    store = VersionedStore({"alpha": 1})
    snapshot = store.capture()
    result = store.commit(store.prepare(snapshot.revision, {"beta": 2}))
    assert result.committed
    assert snapshot.data == {"alpha": 1}
    assert store.latest().data == {"alpha": 1, "beta": 2}


def test_two_writers_from_same_revision_cannot_silently_overwrite():
    """Optimistic conflict: the second writer from a stale revision gets CONFLICT."""
    store = VersionedStore({"value": 0})
    read = store.capture()
    first = store.commit(store.prepare(read.revision, {"value": 1}))
    second = store.commit(store.prepare(read.revision, {"value": 2}))
    assert first.decision is CommitDecision.COMMITTED
    assert second.decision is CommitDecision.CONFLICT
    assert second.committed_revision is None
    assert store.latest().data["value"] == 1


def test_atomic_publication_switches_the_visible_pointer():
    """Atomic publication: readers observe either the previous or the next version."""
    store = VersionedStore({"phase": "before"})
    read = store.capture()
    seen: list[dict] = []
    barrier = threading.Barrier(2)

    def reader() -> None:
        """Sample the visible snapshot repeatedly around the commit point."""
        barrier.wait()
        for _ in range(200):
            seen.append(dict(store.latest().data))
            if seen[-1].get("phase") == "after":
                break

    thread = threading.Thread(target=reader)
    thread.start()
    barrier.wait()
    store.commit(store.prepare(read.revision, {"phase": "after"}))
    thread.join()
    assert seen and seen[-1].get("phase") == "after"
    assert all(item.get("phase") in {"before", "after"} for item in seen)


def test_commit_validation_failure_reports_conflict_without_state_change():
    """A candidate revision that fails validation never publishes."""
    store = VersionedStore({"value": 0}, validation=lambda data: data["value"] <= 10)
    read = store.capture()
    result = store.commit(store.prepare(read.revision, {"value": 99}))
    assert result.decision is CommitDecision.CONFLICT
    assert result.reason == "candidate revision failed validation"
    assert store.latest().data == {"value": 0}


def test_rcu_reclamation_waits_for_grace_period():
    """RCU: an old revision is reclaimed only after all pre-existing readers exit."""
    store = VersionedStore({"step": 0}, retained_revision_limit=2)
    pinned = store.capture()
    ticket = store.pin(pinned)
    store.commit(store.prepare(store.latest().revision, {"step": 1}))
    store.commit(store.prepare(store.latest().revision, {"step": 2}))
    assert pinned.revision in store.retained_revisions()
    store.unpin(ticket)
    store.commit(store.prepare(store.latest().revision, {"step": 3}))
    assert pinned.revision not in store.retained_revisions()


def test_commit_with_transaction_pairs_state_and_side_records():
    """The outbox-style transaction callback runs only inside a successful commit."""
    store = VersionedStore({"state": "initial"})
    read = store.capture()
    side_effects: list[str] = []

    def transaction() -> list[str]:
        """Record the side rows only when the commit proceeds."""
        side_effects.append("recorded")
        return ["outbox-row"]

    result = store.commit(store.prepare(read.revision, {"state": "committed"}),
                          transaction=transaction)
    assert result.committed
    assert side_effects == ["recorded"]

    stale = store.prepare(read.revision, {"state": "lost"})
    side_effects.clear()
    conflict = store.commit(stale, transaction=transaction)
    assert conflict.decision is CommitDecision.CONFLICT
    assert side_effects == []


def test_snapshot_detaches_caller_mappings():
    """Snapshots never hand out references that could mutate committed state."""
    store = VersionedStore({"nested": {"value": 1}})
    snapshot = store.capture()
    snapshot.data["nested"]["value"] = 999
    assert store.latest().data["nested"]["value"] == 1


def test_index_tracks_changed_keys():
    """The per-key index records which revision last changed each key."""
    store = VersionedStore({"a": 0, "b": 0})
    read = store.capture()
    store.commit(store.prepare(read.revision, {"a": 1}))
    latest = store.latest()
    assert latest.index["a"] == 1
    assert latest.index["b"] == 0
