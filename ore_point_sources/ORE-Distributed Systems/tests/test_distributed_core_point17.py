"""Phase A acceptance tests: deployment, state classification, identity, protocol."""
from __future__ import annotations

from pathlib import Path

import pytest

from ore_distributed.core.deployment import (
    DeploymentMode,
    TopologyDescriptor,
    detect_deployment_mode,
    deployment_allows_consensus,
)
from ore_distributed.core.identity import (
    ClusterCredentialAuthority,
    create_node_identity,
    load_identity,
    persist_identity,
)
from ore_distributed.core.protocol import (
    PROTOCOL_VERSION,
    MessageEnvelope,
    ProtocolGate,
    ProtocolViolation,
    check_rolling_upgrade,
    make_envelope,
    protocol_compatible,
)
from ore_distributed.core.state_class import (
    ConsistencyModel,
    FailureModel,
    StateClass,
    StateClassificationError,
    StateObjectSpec,
    StateSchemaRegistry,
    ThreatModel,
    default_state_schema_registry,
    recommended_protocol_family,
)
from ore_distributed.consensus.log import majority_quorum, quorum_intersection, quorum_size
from ore_distributed.adapters import LocalDeployment
from ore_distributed.pipeline import build_pipeline
from ore_distributed.registry import operation_count, phases


def test_single_process_mode_operates_without_consensus_cluster():
    """Single-process mode starts and operates without a consensus cluster."""
    pipeline = build_pipeline("local-1", "ore-local", process_count=1, machine_count=1)
    assert pipeline.capabilities.consensus_active is False
    local = pipeline.build_local_deployment()
    assert local.is_consensus_free()
    local.publish_local_event({"kind": "scratch"})
    assert len(local.local_events()) == 1


def test_deployment_mode_classification():
    """Topology detection separates process and machine boundaries (Step 1)."""
    assert detect_deployment_mode(process_count=1) is DeploymentMode.SINGLE_PROCESS
    assert detect_deployment_mode(process_count=4, machine_count=1) is DeploymentMode.MULTI_PROCESS_LOCAL
    assert detect_deployment_mode(process_count=4, machine_count=3) is DeploymentMode.MULTI_MACHINE_CLUSTER
    assert deployment_allows_consensus(DeploymentMode.SINGLE_PROCESS) is False
    assert deployment_allows_consensus(DeploymentMode.MULTI_MACHINE_CLUSTER) is True


def test_topology_descriptor_round_trip():
    """Topology descriptors validate identity and survive serialization."""
    topology = TopologyDescriptor(
        node_id="node-a", cluster_id="ore", mode=DeploymentMode.MULTI_MACHINE_CLUSTER
    )
    restored = TopologyDescriptor.from_dict(topology.to_dict())
    assert restored.node_id == topology.node_id
    assert restored.mode is DeploymentMode.MULTI_MACHINE_CLUSTER
    with pytest.raises(ValueError):
        TopologyDescriptor(node_id="", cluster_id="ore", mode=DeploymentMode.SINGLE_PROCESS)


def test_unclassified_state_schemas_are_rejected():
    """Schemas without a classification are rejected (Step 2)."""
    registry = StateSchemaRegistry()
    with pytest.raises(StateClassificationError):
        registry.require("ore.some.unknown.schema")
    spec = StateObjectSpec(
        schema_id="ore.test.record",
        state_class=StateClass.MERGEABLE_COLLABORATIVE,
        consistency=ConsistencyModel.EVENTUAL,
    )
    registry.register(spec)
    assert registry.require("ore.test.record").schema_id == "ore.test.record"


def test_consistency_model_must_agree_with_state_class():
    """Illegal consistency requirements for a class are rejected (Step 3)."""
    with pytest.raises(ValueError):
        StateObjectSpec(
            schema_id="ore.bad",
            state_class=StateClass.AGENT_LOCAL_TRANSIENT,
            consistency=ConsistencyModel.LINEARIZABLE,
        )
    with pytest.raises(ValueError):
        StateObjectSpec(
            schema_id="ore.bad2",
            state_class=StateClass.AUTHORITATIVE_OPERATIONAL,
            consistency=ConsistencyModel.EVENTUAL,
        )


def test_default_state_classification_covers_the_core_separation():
    """The canonical classification separates the four state classes (Step 2)."""
    registry = default_state_schema_registry()
    classes = {spec.state_class for spec in registry.specs()}
    assert classes == {
        StateClass.AUTHORITATIVE_OPERATIONAL,
        StateClass.MERGEABLE_COLLABORATIVE,
        StateClass.IMMUTABLE_EVENT_PROVENANCE,
        StateClass.AGENT_LOCAL_TRANSIENT,
    }
    assert registry.require("ore.task.ownership").consistency is ConsistencyModel.LINEARIZABLE
    assert registry.require("ore.agent.scratch").consistency is ConsistencyModel.LOCAL_ONLY


def test_threat_model_selects_protocol_family():
    """Crash threats select Raft; Byzantine threats demand BFT (Step 4)."""
    crash = ThreatModel(tolerated=(FailureModel.CRASH_RECOVERY, FailureModel.NETWORK_PARTITION))
    byzantine = ThreatModel(tolerated=(FailureModel.CRASH_STOP,), byzantine_participants=True)
    assert recommended_protocol_family(crash) == "raft"
    assert recommended_protocol_family(byzantine) == "bft"


def test_majority_quorums_intersect():
    """Any two majorities of one voter set intersect (Section 2.1)."""
    voters = tuple(f"n{i}" for i in range(5))
    q = quorum_size(len(voters))
    assert q == 3
    quorum_a = set(voters[:3])
    quorum_b = set(voters[2:])
    assert quorum_intersection(voters, quorum_a, quorum_b) >= 2 * q - len(voters)
    assert quorum_intersection(voters, quorum_a, quorum_b) > 0


def test_node_identity_is_durable_across_restart(tmp_path: Path):
    """Node identity persists and reloads; it never derives from an address (Step 6)."""
    authority = ClusterCredentialAuthority("ore")
    identity = create_node_identity("node-7", "ore", authority)
    path = tmp_path / "identity.json"
    persist_identity(identity, path)
    restored = load_identity(path, cluster_id="ore")
    assert restored.node_id == "node-7"
    assert restored.credential is not None
    assert restored.credential.verify({"probe": 1}, restored.credential.sign({"probe": 1}))
    with pytest.raises(ValueError):
        load_identity(path, cluster_id="other-cluster")


def test_protocol_gate_rejects_incompatible_and_unauthenticated_messages():
    """Incompatible or unauthenticated messages are rejected before mutation (Step 5)."""
    authority = ClusterCredentialAuthority("ore")
    member = create_node_identity("member-a", "ore", authority)
    gate = ProtocolGate("ore", known_credentials={member.node_id: member.credential})

    foreign = make_envelope(
        "control.ping", "member-a", "other-cluster", {"x": 1}
    )
    with pytest.raises(ProtocolViolation):
        gate.validate(foreign)

    unsigned = MessageEnvelope(
        message_type="consensus.client_command",
        sender="member-a",
        cluster_id="ore",
        payload={"x": 1},
    )
    with pytest.raises(ProtocolViolation):
        gate.validate(unsigned)

    unknown_sender = MessageEnvelope(
        message_type="consensus.client_command",
        sender="intruder",
        cluster_id="ore",
        payload={"x": 1},
        signature="0" * 64,
    )
    with pytest.raises(ProtocolViolation):
        gate.validate(unknown_sender)

    valid = make_envelope(
        "consensus.client_command",
        "member-a",
        "ore",
        {"x": 1},
        credential=member.credential,
        nonce="nonce-1",
    )
    assert gate.validate(valid) is valid


def test_protocol_gate_rejects_tampered_payload_and_replayed_nonce():
    """Digest mismatch and replayed nonces are both rejected (Steps 5, 11)."""
    authority = ClusterCredentialAuthority("ore")
    member = create_node_identity("member-a", "ore", authority)
    gate = ProtocolGate("ore")
    gate.register_credential(member.credential)

    envelope = make_envelope(
        "consensus.client_command",
        "member-a",
        "ore",
        {"x": 1},
        credential=member.credential,
        nonce="once",
    )
    gate.validate(envelope)

    tampered = MessageEnvelope(
        message_type=envelope.message_type,
        sender=envelope.sender,
        cluster_id=envelope.cluster_id,
        nonce=envelope.nonce,
        payload={"x": 2},
        signature=envelope.signature,
    )
    with pytest.raises(ProtocolViolation):
        gate.validate(tampered)

    with pytest.raises(ProtocolViolation):
        gate.validate(envelope)  # replay of the same nonce


def test_rolling_upgrade_keeps_a_common_protocol_version():
    """Rolling upgrades require a mutually supported protocol version (Step 5)."""
    assert protocol_compatible(PROTOCOL_VERSION, "ORE.DIST/0.1.0")
    assert check_rolling_upgrade("ORE.DIST/0.1.0", "ORE.DIST/0.2.0") is True
    assert check_rolling_upgrade("ORE.DIST/0.0.9", "ORE.DIST/0.1.0") is True


def test_local_deployment_has_no_consensus_dependency():
    """Local in-memory adapters satisfy the Step 45 interfaces without a cluster."""
    deployment = LocalDeployment("node-local", "ore")
    assert deployment.is_consensus_free()
    from ore_distributed.adapters import verify_protocol_implementations

    checks = verify_protocol_implementations()
    assert all(checks.values()), checks


def test_operation_registry_covers_48_steps_in_7_phases():
    """The registry maps all 48 algorithmic steps across the 7 phases."""
    assert operation_count() == 48
    assert phases() == tuple(f"Phase {letter}" for letter in "ABCDEFG")
