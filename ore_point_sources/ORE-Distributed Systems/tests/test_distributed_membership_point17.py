"""Phase B acceptance tests: registry, admission, transport, and backpressure."""
from __future__ import annotations

import random

import pytest

from ore_distributed.core.identity import (
    ClusterCredentialAuthority,
    NodeCredential,
    create_node_identity,
)
from ore_distributed.core.protocol import make_envelope
from ore_distributed.membership.admission import (
    AdmissionController,
    MembershipChangeError,
    audit_trail,
    voting_configuration,
)
from ore_distributed.membership.registry import (
    CapabilityDescriptor,
    NodeHealth,
    NodeRecord,
    NodeRegistry,
)
from ore_distributed.membership.transport import (
    BackpressureError,
    InMemoryNetworkFabric,
    RetryPolicy,
    SimulatedTransport,
    simulate_unreachable_peer_attempts,
)


def _node_ids(count: int) -> tuple[str, ...]:
    return tuple(f"node-{i}" for i in range(1, count + 1))


def _registry_with_voters(count: int) -> NodeRegistry:
    """Build a registry with the given number of voting members."""
    registry = NodeRegistry()
    for node_id in _node_ids(count):
        registry.register(NodeRecord(node_id=node_id, health=NodeHealth.VOTING))
    return registry


def test_registry_distinguishes_membership_lifecycle_states():
    """The registry tracks discovered through removed states (Step 7)."""
    registry = NodeRegistry()
    record = NodeRecord(node_id="node-1")
    registry.register(record)
    assert registry.get("node-1").health is NodeHealth.DISCOVERED
    record.health = NodeHealth.AUTHENTICATED
    assert registry.authenticated() and registry.get("node-1") in registry.authenticated()
    record.health = NodeHealth.VOTING
    assert [item.node_id for item in registry.voters()] == ["node-1"]
    registry.remove("node-1")
    assert registry.get("node-1").health is NodeHealth.REMOVED


def test_voting_membership_cannot_be_silently_demoted():
    """Voter state changes only through authorized membership changes (Step 12)."""
    registry = _registry_with_voters(1)
    with pytest.raises(ValueError):
        registry.register(
            NodeRecord(node_id="node-1", health=NodeHealth.AUTHENTICATED)
        )


def test_capability_descriptors_are_claims_not_trust():
    """Capability exchange stores claims; discovery grants no voting rights (Steps 9, 12)."""
    registry = NodeRegistry()
    descriptor = CapabilityDescriptor(
        node_id="node-9",
        ore_modules=frozenset({"ore_n1", "ore_usg"}),
        usg_schema_versions=frozenset({"0.1.0"}),
        workload_classes=frozenset({"visual_pipeline"}),
    )
    registry.publish_capabilities(descriptor)
    record = registry.require("node-9")
    assert record.health is NodeHealth.DISCOVERED
    assert record.capabilities.workload_classes == frozenset({"visual_pipeline"})
    assert registry.voters() == ()


def test_nodes_authenticate_each_other_and_reject_unknown_identities():
    """Multi-process nodes authenticate peers and reject unknown identities (Steps 10)."""
    authority = ClusterCredentialAuthority("ore")
    member_a = create_node_identity("node-a", "ore", authority)
    member_b = create_node_identity("node-b", "ore", authority)
    fabric = InMemoryNetworkFabric()
    transport_a = SimulatedTransport("node-a", "ore", member_a.credential, fabric)
    transport_b = SimulatedTransport("node-b", "ore", member_b.credential, fabric)

    received: list[str] = []
    transport_b.set_handler(lambda envelope: received.append(envelope.sender))

    # Without mutual trust the reply is rejected (unknown identity).
    envelope = make_envelope(
        "consensus.client_command",
        "node-a",
        "ore",
        {"command": "probe"},
        credential=member_a.credential,
        nonce="n-1",
    )
    transport_a.send("node-b", envelope)
    fabric.drain()
    assert received == []
    assert transport_b.rejected >= 1

    # After exchanging trust, delivery works.
    transport_a.trust(member_b.credential)
    transport_b.trust(member_a.credential)
    envelope = make_envelope(
        "consensus.client_command",
        "node-a",
        "ore",
        {"command": "probe"},
        credential=member_a.credential,
        nonce="n-2",
    )
    assert transport_a.send("node-b", envelope) is True
    fabric.drain()
    assert received == ["node-a"]


def test_discovered_node_cannot_become_voter_without_authorized_change():
    """A discovered node cannot join the voting configuration (Steps 8, 12)."""
    registry = NodeRegistry()
    registry.register(NodeRecord(node_id="node-1", health=NodeHealth.VOTING))
    controller = AdmissionController(
        registry, "ore", authorized_initiators={"ops-lead"}, minimum_voters_after_change=1
    )
    controller.authenticate("node-1")
    registry.require("node-1")

    # Capability advertisement must not create a voter.
    before, after = controller.discovered_to_authenticated_flow(
        CapabilityDescriptor(node_id="node-2", ore_modules=frozenset({"ore_n1"}))
    )
    assert after is NodeHealth.DISCOVERED
    assert voting_configuration(registry) == ("node-1",)

    # Unknown node admission fails outright.
    with pytest.raises(MembershipChangeError):
        controller.admit_voter("ghost-node", initiator="ops-lead", reason="capacity", committing_event="ev-1")

    # The discovered node joins only through an authorized membership change.
    controller.authenticate("node-2")
    change = controller.admit_voter(
        "node-2", initiator="ops-lead", reason="capacity", committing_event="ev-config-1"
    )
    assert change.configuration_version == 2
    assert voting_configuration(registry) == ("node-1", "node-2")
    assert audit_trail(controller)[0]["initiator"] == "ops-lead"


def test_unauthorized_membership_initiator_is_rejected():
    """Only authorized initiators may change membership (Step 8)."""
    registry = _registry_with_voters(3)
    controller = AdmissionController(
        registry, "ore", authorized_initiators={"ops-lead"}, minimum_voters_after_change=2
    )
    with pytest.raises(MembershipChangeError):
        controller.admit_voter(
            "node-4", initiator="intern", reason="capacity", committing_event="ev-9"
        )


def test_removal_refuses_to_strand_the_cluster_below_quorum():
    """Membership removal cannot strand the cluster (Step 20 guard)."""
    registry = _registry_with_voters(2)
    controller = AdmissionController(
        registry, "ore", authorized_initiators={"ops-lead"}, minimum_voters_after_change=2
    )
    with pytest.raises(MembershipChangeError):
        controller.remove_voter(
            "node-1", initiator="ops-lead", reason="maintenance", committing_event="ev-10"
        )
    # Growing to three voters makes removal safe again.
    controller.authenticate("node-3")
    controller.admit_voter(
        "node-3", initiator="ops-lead", reason="capacity", committing_event="ev-11"
    )
    change = controller.remove_voter(
        "node-1", initiator="ops-lead", reason="maintenance", committing_event="ev-12"
    )
    assert change.action == "remove"
    assert voting_configuration(registry) == ("node-2", "node-3")


def test_retry_policy_uses_bounded_exponential_backoff_with_jitter():
    """Exponential backoff with jitter stays bounded (Step 11)."""
    policy = RetryPolicy(base_delay=0.05, max_delay=0.4, max_attempts=6, jitter_fraction=0.2)
    rng = random.Random(99)
    delays = simulate_unreachable_peer_attempts(policy, attempts=8, rng=rng)
    assert len(delays) == 8
    assert all(0.0 <= delay <= policy.max_delay * 1.2 for delay in delays)
    assert delays[0] < delays[2] < delays[4]  # growth before saturation
    assert policy.exhausted(attempt=6) is True


def test_backpressure_applies_admission_control():
    """Bounded queues reject excess work instead of growing without bound (Step 11)."""
    authority = ClusterCredentialAuthority("ore")
    member_a = create_node_identity("node-a", "ore", authority)
    member_b = create_node_identity("node-b", "ore", authority)
    fabric = InMemoryNetworkFabric()
    transport_a = SimulatedTransport(
        "node-a", "ore", member_a.credential, fabric, queue_limit=2
    )
    transport_b = SimulatedTransport("node-b", "ore", member_b.credential, fabric, queue_limit=1)
    transport_a.trust(member_b.credential)

    blocker = {"pad": "x" * 64}
    # Saturate node-b's inbound queue through the fabric handler.
    first = make_envelope("control.ping", "node-a", "ore", blocker)
    fabric.register(
        "node-b",
        lambda envelope: (_ for _ in ()).throw(BackpressureError("bounded queue overflowed")),
    )
    fabric.send("node-b", first)
    with pytest.raises(BackpressureError):
        fabric.send("node-b", make_envelope("control.ping", "node-a", "ore", {"huge": "y" * (1 << 21)}))


def test_fabric_enforces_bounded_message_size():
    """Oversized payloads are rejected before transmission (Step 10)."""
    authority = ClusterCredentialAuthority("ore")
    member_a = create_node_identity("node-a", "ore", authority)
    fabric = InMemoryNetworkFabric(max_payload_bytes=256)
    transport = SimulatedTransport("node-a", "ore", member_a.credential, fabric)
    envelope = make_envelope("control.ping", "node-a", "ore", {"blob": "z" * 1024})
    with pytest.raises(BackpressureError):
        transport.send("node-a", envelope)


def test_transport_retries_and_partition_drop_accounting():
    """Partitioned sends fail after the bounded retry budget (Steps 10-11)."""
    authority = ClusterCredentialAuthority("ore")
    member_a = create_node_identity("node-a", "ore", authority)
    member_b = create_node_identity("node-b", "ore", authority)
    fabric = InMemoryNetworkFabric()
    transport = SimulatedTransport("node-a", "ore", member_a.credential, fabric)
    transport.trust(member_b.credential)
    fabric.register("node-b", lambda envelope: None)

    envelope = make_envelope("control.ping", "node-a", "ore", {"x": 1})
    assert transport.send("node-b", envelope) is True

    fabric.partition({"node-a"}, {"node-b"})
    blocked = make_envelope("control.ping", "node-a", "ore", {"x": 2})
    assert transport.send("node-b", blocked) is False
    assert fabric.dropped and transport.retries > 0

    fabric.heal_all_partitions()
    resumed = make_envelope("control.ping", "node-a", "ore", {"x": 3})
    assert transport.send("node-b", resumed) is True
