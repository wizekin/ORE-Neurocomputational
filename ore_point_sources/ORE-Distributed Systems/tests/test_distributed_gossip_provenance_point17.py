"""Phase E acceptance tests: gossip, suspicion, HLC, causality, provenance."""
from __future__ import annotations

import pytest

from ore_distributed.core.identity import (
    ClusterCredentialAuthority,
    create_node_identity,
)
from ore_distributed.gossip.dissemination import (
    GossipDissemination,
    digest_of_entries,
    run_gossip_rounds,
)
from ore_distributed.gossip.peer_sampling import (
    PeerSamplingService,
    fanout_spread_estimate,
    gossip_message_budget,
)
from ore_distributed.gossip.suspicion import (
    FailureSuspicion,
    SuspicionState,
)
from ore_distributed.membership.registry import (
    CapabilityDescriptor,
    NodeHealth,
    NodeRecord,
    NodeRegistry,
)
from ore_distributed.provenance.envelope import (
    ProvenanceEnvelope,
    build_envelope,
    supersede_envelope,
)
from ore_distributed.provenance.journal import (
    ProvenanceJournal,
    VerificationFailure,
    audit_completeness,
    lineage_of,
)
from ore_distributed.time.causal import (
    CausalHistory,
    VectorClock,
    events_concurrent,
    happens_before,
    missing_dependencies,
)
from ore_distributed.time.hlc import (
    HLCTimestamp,
    HybridLogicalClock,
    monotonic_restart_state,
)


def _registry(count: int, domains: dict[str, str] | None = None) -> NodeRegistry:
    registry = NodeRegistry()
    domains = domains or {}
    for index in range(1, count + 1):
        node_id = f"node-{index}"
        registry.register(
            NodeRecord(
                node_id=node_id,
                health=NodeHealth.AUTHENTICATED,
                failure_domain=domains.get(node_id, f"rack-{index}"),
            )
        )
    return registry


def test_peer_sampling_keeps_failure_domain_diversity():
    """Peer sampling avoids one failure domain absorbing the view (Step 31)."""
    registry = _registry(8, domains={f"node-{i}": "rack-a" for i in range(1, 5)})
    sampler = PeerSamplingService("node-1", max_peers=6, per_domain_limit=2, fanout=2)
    view = sampler.sample_view(registry)
    assert 1 <= len(view) <= 6
    report = sampler.diversity_report(view, registry)
    assert max(report.values()) <= 2
    assert "node-1" not in view


def test_peer_sampling_selects_bounded_fanout():
    """Gossip fanout selects a small randomized peer subset (Step 32)."""
    registry = _registry(6)
    sampler = PeerSamplingService("node-1", max_peers=6, per_domain_limit=6, fanout=2)
    view = sampler.sample_view(registry)
    fanout = sampler.select_fanout(view)
    assert len(fanout) == 2
    assert set(fanout) <= set(view)
    assert "node-1" not in fanout


def test_gossip_disseminates_under_fanout_and_budget():
    """Gossip disseminates updates under configured fanout and budget (Step 32)."""
    members = [GossipDissemination(f"node-{i}") for i in range(1, 6)]
    members[0].add_local("entry-1", {"value": 1})
    members[1].add_local("entry-2", {"value": 2})
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (1, 3)]
    converged, rounds = run_gossip_rounds(members, pairs)
    assert converged is True
    assert rounds <= len(pairs)
    assert {entry for entry in members[0].entry_ids()} == {"entry-1", "entry-2"}
    assert fanout_spread_estimate(population=10, fanout=2, rounds=8) >= 9
    assert gossip_message_budget(rounds=5, fanout=2, payload_bytes=1024) == 5 * 2 * 1024


def test_gossip_digest_exchange_repairs_only_missing_entries():
    """Digest comparison precedes selective repair (Step 32)."""
    left = GossipDissemination("node-left")
    right = GossipDissemination("node-right")
    left.add_local("alpha", {"v": 1})
    right.add_local("beta", {"v": 2})
    report = left.exchange(right.digest(), right.entries())
    assert report["received_new"] == 1
    assert report["sent_repair"] == 1
    assert digest_of_entries(("alpha",)) != digest_of_entries(("alpha", "beta"))


def test_gossip_payload_caps_apply():
    """Payload caps bound gossip amplification (Step 32)."""
    node = GossipDissemination("node-capped", max_batch=2, max_payload_bytes=128)
    peer = GossipDissemination("node-peer")
    peer.add_local("k1", "x" * 64)
    peer.add_local("k2", "y" * 64)
    with pytest.raises(ValueError):
        node.apply_remote({"huge": "z" * 512})


def test_peer_timeout_enters_suspicion_before_removal():
    """A peer timeout enters suspicion before final removal (Step 33)."""
    tracker = FailureSuspicion(unreachable_after=3, suspected_after=6, removed_after=12)
    # A single missed heartbeat is not failure.
    tracker.record_missed_heartbeat("node-9")
    assert tracker.state_of("node-9") is SuspicionState.ALIVE
    assert tracker.may_reassign_exclusive_work("node-9") is False
    for _ in range(2):
        tracker.record_missed_heartbeat("node-9")
    assert tracker.state_of("node-9") is SuspicionState.UNREACHABLE
    for _ in range(3):
        tracker.record_missed_heartbeat("node-9")
    assert tracker.state_of("node-9") is SuspicionState.SUSPECTED
    assert tracker.may_reassign_exclusive_work("node-9") is True
    # Recovery resets suspicion before removal.
    tracker.record_heartbeat("node-9")
    assert tracker.state_of("node-9") is SuspicionState.ALIVE
    for _ in range(12):
        tracker.record_missed_heartbeat("node-9")
    assert tracker.state_of("node-9") is SuspicionState.REMOVED
    assert tracker.was_suspected_before_removal("node-9") is True


def test_hlc_remains_monotonic_across_local_events_and_messages():
    """HLC values remain monotonic across local events and observed messages (Step 34)."""
    wall = {"now": 1000}
    left = HybridLogicalClock("node-a", wall_clock=lambda: wall["now"])
    right = HybridLogicalClock("node-b", wall_clock=lambda: wall["now"])
    stamps = [left.tick() for _ in range(5)]
    assert stamps == sorted(stamps)

    # Physical clock drifts backward on the sender; HLC stays monotonic.
    wall["now"] = 900
    remote = right.tick()
    merged = left.merge(remote)
    assert merged.order_key > stamps[-1].order_key
    wall["now"] = 1001
    later = left.tick()
    assert later.order_key > merged.order_key

    # Ticks with the same wall time increment the logical component.
    wall["now"] = 1002
    first = left.tick()
    second = left.tick()
    assert second.physical == first.physical and second.logical == first.logical + 1


def test_hlc_persistence_prevents_backward_movement_on_restart():
    """Persisted HLC state keeps a restarted clock monotonic (Step 34)."""
    wall = {"now": 5000}
    clock = HybridLogicalClock("node-a", wall_clock=lambda: wall["now"])
    peak = clock.tick()
    state = clock.to_state()
    wall["now"] = 4000  # the wall clock moved backward during downtime
    restarted = HybridLogicalClock("node-a", wall_clock=lambda: wall["now"])
    restarted.restore_state(state)
    stamp = monotonic_restart_state(restarted, peak)
    assert stamp.order_key > peak.order_key


def test_causal_dependencies_are_preserved_independently_of_hlc():
    """Causality is explicit, independent of HLC ordering (Step 35)."""
    first = VectorClock().increment("node-a")
    second = VectorClock()
    second.merge(first)
    second.increment("node-a")
    assert happens_before(first.snapshot(), second.snapshot()) is True
    unrelated = VectorClock().increment("node-b")
    assert events_concurrent(first.snapshot(), unrelated.snapshot()) is True

    history = CausalHistory(event_id="ev-3", parents=frozenset({"ev-1", "ev-2"}))
    assert missing_dependencies(history, ("ev-1",)) == ("ev-2",)
    # HLC order keys may order concurrent events without implying causality.
    stamp_left = HLCTimestamp(physical=10, logical=0, node_id="a")
    stamp_right = HLCTimestamp(physical=10, logical=0, node_id="b")
    assert stamp_left.order_key < stamp_right.order_key  # tiebreak, not causality


def _envelope_pair(journal_node, clock, payload_a, payload_b, parents=()):
    """Build two signed envelopes from one node for journal tests."""
    a = build_envelope(
        event_id="ev-a",
        actor="agent-1",
        action="observe",
        object_id="obj-1",
        payload=payload_a,
        hlc=clock.tick(),
        causal_parents=parents,
        credential=journal_node.credential,
    )
    b = build_envelope(
        event_id="ev-b",
        actor="agent-1",
        action="observe",
        object_id="obj-1",
        payload=payload_b,
        hlc=clock.tick(),
        causal_parents=parents,
        credential=journal_node.credential,
    )
    return a, b


def test_event_digests_and_signatures_are_verified():
    """Event digests and authenticated origins are verified (Step 37)."""
    authority = ClusterCredentialAuthority("ore")
    node = create_node_identity("node-a", "ore", authority)
    clock = HybridLogicalClock("node-a")
    journal = ProvenanceJournal()
    journal.trust_credential(node.credential)
    envelope_a, envelope_b = _envelope_pair(node, clock, {"x": 1}, {"x": 1})
    assert journal.accept(envelope_a, {"x": 1}) == "accepted"
    # Payload tampering is caught by digest verification.
    with pytest.raises(VerificationFailure):
        journal.accept(envelope_b, {"x": 2})
    # Duplicate event IDs are rejected.
    with pytest.raises(VerificationFailure):
        journal.accept(envelope_a, {"x": 1})


def test_untrusted_origin_is_rejected():
    """Unsigned or untrusted origins never enter the journal (Step 37)."""
    authority = ClusterCredentialAuthority("ore")
    trusted = create_node_identity("node-a", "ore", authority)
    stranger = create_node_identity("node-b", "ore", authority)
    clock = HybridLogicalClock("node-a")
    journal = ProvenanceJournal()
    journal.trust_credential(trusted.credential)
    envelope = build_envelope(
        event_id="ev-stranger",
        actor="agent-9",
        action="observe",
        object_id="obj-9",
        payload={"x": 1},
        hlc=clock.tick(),
        credential=stranger.credential,  # signed, but by an untrusted origin
    )
    with pytest.raises(VerificationFailure):
        journal.accept(envelope)
    assert journal.verification_failures() == ("ev-stranger",)


def test_missing_causal_dependencies_are_quarantined_then_repaired():
    """Missing dependencies are quarantined or repaired, never discarded (Step 37)."""
    authority = ClusterCredentialAuthority("ore")
    node = create_node_identity("node-a", "ore", authority)
    clock = HybridLogicalClock("node-a")
    journal = ProvenanceJournal()
    journal.trust_credential(node.credential)

    parent = build_envelope(
        event_id="ev-parent",
        actor="agent-1",
        action="observe",
        object_id="obj-1",
        payload={"p": 1},
        hlc=clock.tick(),
        credential=node.credential,
    )
    child = build_envelope(
        event_id="ev-child",
        actor="agent-1",
        action="observe",
        object_id="obj-1",
        payload={"c": 1},
        hlc=clock.tick(),
        causal_parents=("ev-parent",),
        credential=node.credential,
    )
    # Child arrives first: it is quarantined because its parent is missing.
    assert journal.accept(child) == "quarantined"
    assert [entry.event_id for entry in journal.quarantined()] == ["ev-child"]
    assert {entry.missing_parents for entry in journal.quarantined()} == {("ev-parent",)}
    # The audit trail records the quarantine, not silent discard.
    outcomes = {record.event_id: record.outcome for record in journal.audit_trail()}
    assert outcomes["ev-child"] == "quarantined"
    # Repair delivers the missing parent and releases the quarantine.
    repaired = journal.repair_quarantine([parent])
    assert repaired == 1
    assert journal.quarantined() == ()
    assert "ev-child" in {record.event_id for record in journal.audit_trail()}


def test_supersede_records_lineage_and_audit():
    """Superseded events remain in the audit trail with lineage (Step 37)."""
    authority = ClusterCredentialAuthority("ore")
    node = create_node_identity("node-a", "ore", authority)
    clock = HybridLogicalClock("node-a")
    journal = ProvenanceJournal()
    journal.trust_credential(node.credential)
    original, _ = _envelope_pair(node, clock, {"v": 1}, {"v": 2})
    journal.accept(original, {"v": 1})
    replacement = build_envelope(
        event_id="ev-replacement",
        actor="agent-1",
        action="observe",
        object_id="obj-1",
        payload={"v": 3},
        hlc=clock.tick(),
        causal_parents=("ev-a",),
        credential=node.credential,
    )
    record = journal.supersede("ev-a", replacement)
    assert record.outcome == "superseded"
    assert journal.superseded_map()["ev-a"] == "ev-replacement"
    assert audit_completeness(journal, ("ev-a",)) >= 0.5


def test_envelope_immutable_fields_and_content_addressing():
    """Envelopes carry immutable identity fields and content references (Step 36)."""
    authority = ClusterCredentialAuthority("ore")
    node = create_node_identity("node-a", "ore", authority)
    clock = HybridLogicalClock("node-a")
    envelope = build_envelope(
        event_id="ev-1",
        actor="agent-1",
        action="observe",
        object_id="obj-1",
        payload={"payload": "data"},
        hlc=clock.tick(),
        source_references=("src-1",),
        credential=node.credential,
    )
    assert envelope.content_reference.startswith("sha256:")
    assert envelope.verify_signature(node.credential)
    superseded = supersede_envelope(envelope, envelope)
    assert "ev-1" in superseded.causal_parents
    with pytest.raises(ValueError):
        ProvenanceEnvelope(
            event_id="bad",
            actor="a",
            action="act",
            object_id="o",
            payload_digest="not-a-digest",
            hlc=clock.tick(),
        )


def test_lineage_walks_causal_ancestry():
    """Journal lineage returns the transitive causal ancestry (Step 37)."""
    authority = ClusterCredentialAuthority("ore")
    node = create_node_identity("node-a", "ore", authority)
    clock = HybridLogicalClock("node-a")
    journal = ProvenanceJournal()
    journal.trust_credential(node.credential)
    root = build_envelope(
        event_id="ev-root", actor="a", action="act", object_id="o",
        payload={"r": 1}, hlc=clock.tick(), credential=node.credential,
    )
    middle = build_envelope(
        event_id="ev-mid", actor="a", action="act", object_id="o",
        payload={"m": 1}, hlc=clock.tick(), causal_parents=("ev-root",),
        credential=node.credential,
    )
    leaf = build_envelope(
        event_id="ev-leaf", actor="a", action="act", object_id="o",
        payload={"l": 1}, hlc=clock.tick(), causal_parents=("ev-mid",),
        credential=node.credential,
    )
    for envelope, payload in ((root, {"r": 1}), (middle, {"m": 1}), (leaf, {"l": 1})):
        journal.accept(envelope, payload)
    assert lineage_of(journal, "ev-leaf") == ("ev-root", "ev-mid", "ev-leaf")
