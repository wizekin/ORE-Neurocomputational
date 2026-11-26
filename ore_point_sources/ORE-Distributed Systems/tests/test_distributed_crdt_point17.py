"""Phase D acceptance tests: CRDT merge laws, convergence, conflicts, promotion."""
from __future__ import annotations

import random

import pytest

from ore_distributed.replication.crdts import (
    GSet,
    LWWRegister,
    ORSet,
    PNCounter,
    check_merge_laws,
    merge_semilattice_documented,
)
from ore_distributed.replication.conflicts import (
    Claim,
    detect_semantic_contradiction,
)
from ore_distributed.replication.promotion import (
    CandidateAssertion,
    USGPromotionGate,
)
from ore_distributed.replication.reconciliation import (
    offline_replica_reconnect,
    reconcile_replicas,
    reconciliation_is_idempotent,
)
from ore_distributed.replication.replica import (
    CrdtReplica,
    ObservationStatus,
)
from ore_distributed.replication.tombstones import (
    TombstoneCollector,
    deletion_survives_offline_replica,
)
from ore_distributed.time.hlc import HybridLogicalClock


def _hlc(node: str, physical: int) -> object:
    clock = HybridLogicalClock(node)
    return clock.now.__self__.now.__self__ if False else _stamp(node, physical)


def _stamp(node: str, physical: int, logical: int = 0):
    from ore_distributed.time.hlc import HLCTimestamp

    return HLCTimestamp(physical=physical, logical=logical, node_id=node)


def _generated_gsets(count: int, rng: random.Random) -> list[GSet]:
    """Generate varied grow-only sets for property testing."""
    sets = []
    for index in range(count):
        item = GSet()
        for element in rng.sample(range(12), k=4):
            item.add(f"item-{element}")
        sets.append(item)
    return sets


def test_gset_merge_is_associative_commutative_idempotent():
    """CRDT merge is associative under generated test states (Step 24)."""
    report = check_merge_laws(_generated_gsets(5, random.Random(17)))
    assert report.associative is True
    assert report.violations == ()


def test_orset_merge_is_commutative_idempotent_under_concurrent_add_remove():
    """CRDT merge holds its laws under concurrent adds and removes (Steps 23-24)."""
    rng = random.Random(23)
    sets = []
    for index in range(5):
        item = ORSet(node_id=f"n{index}")
        item.add("shared")
        item.add(f"own-{index}")
        if index % 2:
            item.remove("shared")
        if index % 3:
            item.add("extra")
        sets.append(item)
    report = check_merge_laws(sets)
    assert report.associative and report.commutative and report.idempotent
    assert merge_semilattice_documented("orset")["tombstones"].startswith("remove tombstones")


def test_pncounter_and_lww_register_merge_laws():
    """Counters and registers keep associative, commutative, idempotent merges."""
    rng = random.Random(31)
    counters = []
    for index in range(4):
        counter = PNCounter()
        counter.increment(f"n{index}", rng.randint(1, 5))
        counter.decrement(f"n{index}", rng.randint(0, 2))
        counters.append(counter)
    report = check_merge_laws(counters)
    assert report.associative and report.commutative and report.idempotent

    registers = [
        LWWRegister(value=f"v{i}", timestamp=(physical, i, f"w{i}"), writer=f"w{i}")
        for i, physical in enumerate([10, 12, 12, 9])
    ]
    report = check_merge_laws(registers)
    assert report.associative and report.commutative and report.idempotent


def test_merge_law_violations_are_reported():
    """A broken merge algebra is detected rather than trusted (Step 24)."""
    class BrokenRegister:
        """A deliberately non-idempotent merge to prove detection works."""

        def __init__(self, value: int) -> None:
            self.value = value

        def merge(self, other: "BrokenRegister") -> "BrokenRegister":
            self.value = self.value + other.value
            return self

        def clone(self) -> "BrokenRegister":
            return BrokenRegister(self.value)

        def __eq__(self, other: object) -> bool:
            return isinstance(other, BrokenRegister) and self.value == other.value

    report = check_merge_laws([BrokenRegister(1), BrokenRegister(2)])
    assert report.idempotent is False
    assert report.violations


def test_reordered_and_duplicated_crdt_messages_converge():
    """Reordered and duplicated CRDT messages converge (Section 6)."""
    left = CrdtReplica("replica-left")
    right = CrdtReplica("replica-right")
    observations = [left.apply_local(f"observation-{i}") for i in range(5)]

    rng = random.Random(47)
    shuffled = list(observations)
    rng.shuffle(shuffled)
    for observation in shuffled:
        right.apply_remote(observation)
    # Deliver every observation a second time (duplication).
    for observation in shuffled:
        right.apply_remote(observation)
    assert right.duplicates_dropped == 5
    assert {o.event_id for o in right.observations()} == {o.event_id for o in observations}


def test_offline_replicas_reconcile_after_reconnect():
    """Offline replicas reconcile after reconnect (Section 6)."""
    left = CrdtReplica("replica-left")
    right = CrdtReplica("replica-right")
    # Shared pre-disconnect history.
    for text in ("alpha", "beta"):
        left.apply_local(text)
    report = reconcile_replicas(left, right)
    assert report.converged is True

    # Left goes offline; both sides accumulate independent observations.
    left_offline = [left.apply_local(f"left-only-{i}") for i in range(2)]
    right_online = [right.apply_local(f"right-only-{i}") for i in range(3)]
    reconnect = offline_replica_reconnect(left, right)
    assert reconnect.converged is True
    assert len(left.observations()) == len(right.observations()) == 7
    # Both sides' independent contributions survive reconciliation.
    ids = {o.event_id for o in left.observations()}
    assert {o.event_id for o in left_offline} <= ids
    assert {o.event_id for o in right_online} <= ids


def test_repeated_reconciliation_is_idempotent():
    """Repeated reconciliation produces no further changes (Step 28)."""
    left = CrdtReplica("replica-left")
    right = CrdtReplica("replica-right")
    left.apply_local("one")
    right.apply_local("two")
    converged, changed = reconciliation_is_idempotent(left, right)
    assert converged is True and changed == 4


def test_identical_text_from_two_agents_is_two_observations():
    """Duplicate identity is distinct from duplicate content (Step 25)."""
    left = CrdtReplica("agent-left")
    right = CrdtReplica("agent-right")
    left_id, right_id = (
        left.apply_local("membrane regulates transport").event_id,
        right.apply_local("membrane regulates transport").event_id,
    )
    assert left_id != right_id
    reconcile_replicas(left, right)
    groups = right.observations_by_content_hash()
    same_content = [group for group in groups.values() if len(group) == 2]
    assert same_content, "identical text must remain two distinct observations"
    # Re-delivery of one event is deduplicated by ID, not by content.
    before = right.duplicates_dropped
    right.apply_remote(left.observation(left_id))
    assert right.duplicates_dropped == before + 1


def test_local_acceptance_is_not_global_visibility():
    """Local acceptance, dissemination, and reconciliation stay separate (Step 27)."""
    replica = CrdtReplica("replica-1")
    observation = replica.apply_local("fresh-observation")
    assert replica.status_of(observation.event_id) is ObservationStatus.RECORDED_LOCALLY
    assert replica.pending_dissemination() and replica.pending_dissemination()[0].event_id == observation.event_id
    replica.mark_disseminated([observation.event_id])
    assert replica.status_of(observation.event_id) is ObservationStatus.DISSEMINATED
    assert replica.pending_dissemination() == ()


def test_semantic_contradictions_remain_explicit_after_merge():
    """Semantic contradictions remain explicit after CRDT merge (Step 26)."""
    left = CrdtReplica("agent-left")
    right = CrdtReplica("agent-right")
    claim_a = Claim(
        claim_id="claim-a",
        text="the membrane is selectively permeable",
        origin="agent-left",
        evidence_ids=("ev-1",),
        hlc=(10, 1, "agent-left"),
        confidence=0.8,
    )
    claim_b = Claim(
        claim_id="claim-b",
        text="the membrane is not selectively permeable",
        origin="agent-right",
        evidence_ids=("ev-2",),
        hlc=(10, 2, "agent-right"),
        confidence=0.7,
    )
    left.apply_local(claim_a.text, context={"claim_id": claim_a.claim_id})
    right.apply_local(claim_b.text, context={"claim_id": claim_b.claim_id})
    report = reconcile_replicas(left, right)
    assert report.converged is True
    # The contradiction is still representable and routed, not resolved.
    conflict = detect_semantic_contradiction(
        "membrane permeability", [claim_a, claim_b], hlc_now=(11, 0, "usg")
    )
    assert conflict is not None
    assert conflict.resolution_status == "open"
    assert set(conflict.evidence_ids()) == {"ev-1", "ev-2"}
    assert conflict.to_dict()["claims"][0]["claim_id"] == "claim-a"


def test_crdt_convergence_never_marks_assertion_true():
    """CRDT convergence never automatically marks a semantic assertion true (Step 30)."""
    gate = USGPromotionGate(minimum_confidence=0.5)
    candidate = CandidateAssertion(
        assertion_id="assert-converged",
        subject="membrane",
        predicate="regulates",
        payload={"relation": "regulates"},
        source_reference="obs://agent/7",
        confidence=0.4,  # below threshold even though the replica converged
        context={"task": "reading"},
        observed_event_id="obs-77",
        provenance_digest="b" * 64,
    )
    outcome = gate.evaluate(candidate)
    assert outcome.promoted is False
    assert "confidence" in outcome.reason
    assert gate.consensus_ordering_note().startswith("Consensus may order")


def test_promotion_gate_requires_schema_provenance_context_and_no_conflict():
    """Promotion requires schema, provenance, context, and contradiction checks (Step 30)."""
    gate = USGPromotionGate(minimum_confidence=0.5)
    base = dict(
        assertion_id="assert-1",
        subject="membrane",
        predicate="regulates",
        payload={"relation": "regulates"},
        source_reference="obs://agent/1",
        confidence=0.9,
        context={"task": "reading"},
        observed_event_id="obs-1",
        provenance_digest="c" * 64,
    )
    assert gate.evaluate(CandidateAssertion(**base)).promoted is True

    no_provenance = dict(base, assertion_id="assert-2", provenance_digest="")
    assert gate.evaluate(CandidateAssertion(**no_provenance)).promoted is False

    no_context = dict(base, assertion_id="assert-3", context={})
    assert gate.evaluate(CandidateAssertion(**no_context)).promoted is False

    conflict_linked = dict(base, assertion_id="assert-4", observed_event_id="claim-b")
    conflict = detect_semantic_contradiction(
        "subject",
        [
            Claim("claim-a", "x is true", "left", hlc=(1, 0, "left")),
            Claim("claim-b", "x is not true", "right", hlc=(2, 0, "right")),
        ],
        hlc_now=(3, 0, "usg"),
    )
    outcome = gate.evaluate(CandidateAssertion(**conflict_linked), open_conflicts=[conflict])
    assert outcome.promoted is False
    assert outcome.routed_conflict_id == conflict.conflict_id
    assert gate.conflicts_routed == 1


def test_tombstone_collection_never_resurrects_deleted_values():
    """Tombstone collection does not resurrect deleted values (Step 29)."""
    replica_a = ORSet(node_id="replica-a")
    tag = replica_a.add("contribution-1")
    replica_a.remove("contribution-1")
    assert replica_a.contains("contribution-1") is False

    collector = TombstoneCollector(known_replicas=("replica-a", "replica-b", "replica-c"))
    # Only two replicas observed the deletion; the offline replica did not.
    collector.record_observation(tag, "replica-a")
    collector.record_observation(tag, "replica-b")
    safe, reason = deletion_survives_offline_replica(collector, "replica-c", [tag])
    assert safe is False
    assert collector.collect(replica_a, [tag]) == 0  # blocked from collection

    # The offline replica rejoins and observes the deletion.
    collector.record_observation(tag, "replica-c")
    assert collector.collect(replica_a, [tag]) >= 1
    assert replica_a.contains("contribution-1") is False


def test_orset_add_wins_over_stale_remove():
    """Observed-remove semantics keep concurrent adds alive (Step 23)."""
    first = ORSet(node_id="r1")
    second = ORSet(node_id="r2")
    first.add("element")  # tag t1
    # second never observed t1, so its remove cannot kill t1.
    second.remove("element")
    first.merge(second)
    assert first.contains("element") is True
    second.merge(first)
    assert second.contains("element") is True


def test_merge_documentation_is_required_for_each_crdt_kind():
    """CRDT kinds must document assumptions and tombstone behavior (Step 23)."""
    for kind in ("gset", "orset", "pncounter", "lww_register"):
        spec = merge_semilattice_documented(kind)
        assert {"merge_algebra", "assumptions", "tombstones", "garbage_collection"} <= set(spec)
    with pytest.raises(ValueError):
        merge_semilattice_documented("arbitrary_json_blob")


def test_replica_status_lifecycle_is_visible():
    """Dissemination statuses are exposed in APIs (Step 27)."""
    replica = CrdtReplica("replica-1")
    first = replica.apply_local("status-check")
    second = replica.apply_local("another-check")
    statuses = {first.event_id: replica.status_of(first.event_id)}
    assert all(status in ObservationStatus for status in statuses.values())
    replica.mark_disseminated([first.event_id, second.event_id])
    assert replica.pending_dissemination() == ()
