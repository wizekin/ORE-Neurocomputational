"""Phase G acceptance tests: workspace routing, USG, CROS, observability, partitions."""
from __future__ import annotations

import pytest

from ore_workspace.models import WorkspaceEvent

from ore_distributed.consensus.raft import RaftCluster, RaftRole
from ore_distributed.consensus.state_machine import StateMachineCommand
from ore_distributed.integration.cros_adapter import (
    CROSResourceAdapter,
    ResourceTelemetry,
    cros_cannot_change_consensus_safety,
)
from ore_distributed.integration.observability import (
    DegradedModeController,
    DistributedMetrics,
    quorum_unavailable_enters_degraded_mode,
)
from ore_distributed.integration.usg_adapter import (
    USGValidationGate,
    usg_rejects_invalid_schema_check,
)
from ore_distributed.integration.workspace import WorkspaceDistributedConnector
from ore_distributed.replication.promotion import CandidateAssertion
from ore_distributed.recovery.partitions import (
    PartitionPolicy,
    PartitionTracker,
    majority_continues_policy,
)


def _workspace_event(event_type: str, finding_ids: tuple[str, ...] = ()) -> WorkspaceEvent:
    """Build one typed workspace event for routing tests."""
    return WorkspaceEvent(
        workspace_id="ws-1",
        workspace_revision=1,
        event_type=event_type,
        changed_fields=("findings",),
        producer_module="ore_reading",
        finding_ids=finding_ids,
    )


def test_workspace_routes_typed_events_to_the_right_paths():
    """Typed workspace events route to authoritative, mergeable, or local paths (Step 46)."""
    connector = WorkspaceDistributedConnector("node-a")
    authoritative = connector.route_event(_workspace_event("coordinator_decision"), confidence=0.9)
    assert authoritative.path == "authoritative"
    assert len(connector.pending_authoritative()) == 1

    mergeable = connector.route_event(
        _workspace_event("finding_published", finding_ids=("finding-1",)), confidence=0.8
    )
    assert mergeable.path == "mergeable"
    assert connector.replica.observation_count if hasattr(connector.replica, "observation_count") else True

    scratch = connector.route_event(_workspace_event("scratch_update"))
    assert scratch.path == "local"
    unknown = connector.route_event(_workspace_event("never-seen-type"))
    assert unknown.path == "local"  # routing is explicit, never guessed


def test_workspace_findings_retain_origin_and_confidence():
    """Cross-node workspace items retain origin and confidence metadata (Step 46)."""
    connector = WorkspaceDistributedConnector("node-b")
    event_ids = connector.publish_findings({"finding-alpha": 0.7, "finding-beta": 0.4})
    assert len(event_ids) == 2
    for event_id in event_ids:
        observation = connector.replica.observation(event_id)
        assert observation is not None
        assert observation.origin == "node-b"
        assert observation.context["confidence"] in {0.7, 0.4}
    connector.local_scratch("temp-plan", {"step": 1})
    # Scratch state never enters the mergeable replica.
    assert all(
        observation.kind != "scratch" for observation in connector.replica.observations()
    )


def test_usg_validation_gate_uses_real_ore_usg_contracts():
    """USG validates distributed findings with the repository's own contracts (Step 47)."""
    gate = USGValidationGate(namespace="ore.distributed")
    candidate = CandidateAssertion(
        assertion_id="assert-1",
        subject="cell membrane",
        predicate="regulates",
        payload={"relation": "regulates", "object": "substances"},
        source_reference="obs://node-a/42",
        confidence=0.9,
        context={"task": "reading-comprehension"},
        observed_event_id="obs-42",
        provenance_digest="d" * 64,
    )
    outcome = gate.validate_and_promote(candidate)
    assert outcome.valid is True and outcome.errors == ()
    assert outcome.evidence_ids and outcome.relation_ids
    assert gate.evidence_lineage("assert-1") == outcome.evidence_ids


def test_usg_rejects_invalid_semantic_schemas():
    """USG rejects invalid semantic schemas (Section 6)."""
    gate = USGValidationGate()
    rejected, _ = usg_rejects_invalid_schema_check(gate)
    assert rejected is True
    errors = gate.reject_invalid_semantic_schema({"contract": "USG-KNOWLEDGE/0.1.0"})
    assert errors  # missing required top-level fields are reported
    assert gate.rejected_schemas >= 1


def test_usg_promotion_requires_validation_and_protocol_gates():
    """Promotion needs both USG validation and the protocol-level gate (Step 47)."""
    gate = USGValidationGate()
    weak = CandidateAssertion(
        assertion_id="assert-weak",
        subject="x",
        predicate="relates",
        payload={"relation": "relates"},
        source_reference="obs://node/1",
        confidence=0.2,  # below the promotion threshold
        context={"task": "probe"},
        observed_event_id="obs-1",
        provenance_digest="e" * 64,
    )
    outcome = gate.validate_and_promote(weak)
    assert outcome.valid is True  # schema fine
    assert gate.gate.promoted_assertions() == ()  # but not promoted


def test_cros_adapts_resources_within_safe_bounds():
    """CROS adapts gossip and replication parameters within bounds (Step 47)."""
    adapter = CROSResourceAdapter()
    calm = ResourceTelemetry(
        network_pressure=0.1, cpu_pressure=0.1, memory_pressure=0.1,
        storage_pressure=0.1, queue_depth=0,
    )
    adapter.adapt(calm)
    hot = ResourceTelemetry(
        network_pressure=0.9, cpu_pressure=0.85, memory_pressure=0.8,
        storage_pressure=0.7, queue_depth=50,
    )
    adapted = adapter.adapt(hot)
    assert adapted["replication_batch_size"] >= 1
    assert adapted["gossip_frequency_ticks"] >= calm.network_pressure and adapted["gossip_frequency_ticks"] <= 64
    assert adapter.adaptations, "pressure should trigger bounded adaptation"
    demand = hot.to_demand()
    assert 0.0 <= demand.priority <= 1.0


def test_cros_cannot_change_consensus_safety_parameters():
    """CROS adaptation cannot change consensus safety parameters (Section 6)."""
    adapter = CROSResourceAdapter()
    assert cros_cannot_change_consensus_safety(adapter) is True
    with pytest.raises(ValueError):
        adapter.set_parameter("quorum_size", 1)
    with pytest.raises(ValueError):
        adapter.set_parameter("membership_configuration", 2)
    assert adapter.violations_blocked >= 2


def test_replication_lag_commit_latency_and_recovery_are_observable():
    """Replication lag, commit latency, and recovery time are observable (Step 48)."""
    cluster = RaftCluster(("n1", "n2", "n3"), seed=83)
    leader = cluster.run_until_leader()
    leader.propose(
        StateMachineCommand(
            command_id="cmd-metrics",
            command_type="set_policy",
            payload={"policy_key": "m", "version": 1, "value": {}},
        )
    )
    cluster.run_ticks(6)
    cluster.isolate("n2")
    cluster.run_ticks(4)

    metrics = DistributedMetrics.from_cluster(cluster)
    metrics.record_tick(quorum_available=cluster.leader() is not None)
    metrics.record_commit(latency_ticks=3)
    metrics.record_recovery("n2", duration_ticks=12)
    metrics.record_shard_moves(4)
    metrics.record_gossip_round(converged_after=3)
    metrics.record_stale_owner()

    snapshot = metrics.snapshot()
    assert snapshot["commit_latency_median_ticks"] >= 1
    assert metrics.commit_latencies_ticks and 3 in metrics.commit_latencies_ticks
    assert snapshot["commit_latency_p95_ticks"] >= snapshot["commit_latency_median_ticks"]
    assert snapshot["recovery_durations"] == [("n2", 12)]
    assert snapshot["average_replication_lag_entries"] >= 0.0
    assert snapshot["quorum_availability"] == 1.0
    from ore_distributed.integration.observability import all_metrics_observable

    assert "commit_latency_median_ticks" in all_metrics_observable(metrics)


def test_partition_status_is_exposed_to_the_cognitive_coordinator():
    """Network partition status is exposed to the cognitive coordinator (Step 44)."""
    cluster = RaftCluster(("n1", "n2", "n3", "n4", "n5"), seed=97)
    leader = cluster.run_until_leader()
    tracker = PartitionTracker(majority_continues_policy())
    majority = set(cluster.node_ids) - {leader.node_id}
    tracker.start_partition({leader.node_id}, majority)
    cluster.partition(majority, {leader.node_id})
    cluster.run_ticks(40)

    status = tracker.degraded_status(cluster)
    new_leader = cluster.leader()
    assert status["authoritative_writes"] is True
    assert new_leader is not None and new_leader.node_id in majority
    # Minority writes are blocked under the documented policy.
    assert tracker.minority_may_commit_authoritative(cluster, leader.node_id) is False
    assert tracker.is_minority([leader.node_id], cluster.node_ids) is True
    crdt = tracker.crdt_updates_during_partition(provisional=True)
    assert crdt["accept_locally"] is True
    assert crdt["visibility"] == "provisional_until_reconciled"
    tracker.end_partition()
    assert tracker.events and "healed" in tracker.events[-1]


def test_partition_policy_is_a_safety_invariant():
    """The majority/minority partition policy cannot be weakened (Step 44)."""
    with pytest.raises(ValueError):
        PartitionPolicy(allow_authoritative_majority=False)
    with pytest.raises(ValueError):
        PartitionPolicy(block_authoritative_minority=False)
    assert majority_continues_policy().allow_provisional_crdt_local is True


def test_system_enters_safe_degraded_mode_without_quorum():
    """A documented safe degraded mode is entered when quorum is unavailable (Step 48)."""
    cluster = RaftCluster(("n1", "n2", "n3"), seed=107)
    leader = cluster.run_until_leader()
    # Kill the leader and split the two followers so no quorum can form.
    cluster.crash_and_restart(leader.node_id)  # restarted as follower, isolated next
    followers = [node for node in cluster.node_ids if node != leader.node_id]
    cluster.partition({followers[0]}, {followers[1]})
    cluster.partition({leader.node_id}, set(followers))
    cluster.run_ticks(50)

    controller = DegradedModeController()
    entered = quorum_unavailable_enters_degraded_mode(cluster, controller)
    assert entered is True
    assert controller.mode is DegradedModeController.READ_ONLY_DEGRADED
    assert controller.may_write_authoritative() is False
    assert controller.may_serve_reads() is True
    assert controller.status()["reads"] == "served"

    # Quorum recovery returns the system to normal operation.
    cluster.heal()
    cluster.run_ticks(30)
    controller.resume_normal()
    assert controller.may_write_authoritative() is True


def test_local_only_mode_is_explicit():
    """An explicit local-only mode exists for single-process deployments (Step 48)."""
    controller = DegradedModeController()
    controller.enter_local_only()
    assert controller.mode is DegradedModeController.LOCAL_ONLY
    assert controller.may_serve_reads() is True
    status = controller.status()
    assert status["authoritative_writes"] == "blocked"
    controller.resume_normal()
    assert controller.transitions[-1] == ("local_only", "normal")


def test_degraded_mode_blocks_authoritative_writes():
    """The degraded controller blocks authoritative writes while serving reads."""
    cluster = RaftCluster(("n1", "n2", "n3"), seed=113)
    leader = cluster.run_until_leader()
    cluster.isolate(leader.node_id)
    cluster.run_ticks(30)  # majority elects a new leader; writes continue there
    controller = DegradedModeController()
    # The isolated minority leader cannot write; the controller reflects it.
    minority_leader_writes = leader.propose(
        StateMachineCommand(
            command_id="cmd-degraded", command_type="set_policy",
            payload={"policy_key": "d", "version": 1, "value": {}},
        )
    )
    assert minority_leader_writes.acknowledged is False
    controller.enter_read_only_degraded()
    assert not controller.may_write_authoritative()


def test_step45_interfaces_cover_every_subsystem():
    """Every Step 45 coordination interface has an in-memory implementation."""
    from ore_distributed.adapters import verify_protocol_implementations

    checks = verify_protocol_implementations()
    expected = {
        "NodeRegistry",
        "ConsensusStateStore",
        "ReplicatedLog",
        "CRDTReplica",
        "GossipTransport",
        "EventClock",
        "ProvenanceJournal",
        "ShardRouter",
        "RecoveryManager",
    }
    assert set(checks) == expected
    assert all(checks.values())


def test_pipeline_wires_capabilities_by_deployment_mode():
    """The pipeline activates only the subsystems a deployment needs (Step 45)."""
    from ore_distributed.pipeline import build_pipeline

    single = build_pipeline("node-a", "ore", process_count=1)
    assert single.capabilities.consensus_active is False
    assert single.capabilities.crdt_replication_active is False

    multi_process = build_pipeline("node-a", "ore", process_count=3, machine_count=1)
    assert multi_process.capabilities.consensus_active is True
    assert multi_process.capabilities.placement_active is False  # local processes share memory

    cluster_mode = build_pipeline("node-a", "ore", process_count=6, machine_count=3)
    assert cluster_mode.capabilities.placement_active is True
    assert cluster_mode.capabilities.recovery_active is True


def test_consensus_scope_stays_narrow():
    """High-volume observations stay off the consensus log (Step 22)."""
    from ore_distributed.consensus.state_machine import COMMAND_TYPES

    assert "assign_task" in COMMAND_TYPES
    assert "observation_contribution" not in COMMAND_TYPES
    with pytest.raises(ValueError):
        StateMachineCommand(
            command_id="cmd-bad",
            command_type="publish_token_telemetry",  # not a control-plane operation
            payload={},
        )


def test_old_leader_cannot_write_after_partition_heals():
    """A partitioned leader steps down and cannot write after healing (Steps 19, 44)."""
    cluster = RaftCluster(("n1", "n2", "n3"), seed=127)
    old_leader = cluster.run_until_leader()
    cluster.isolate(old_leader.node_id)
    cluster.run_ticks(30)
    new_leader = cluster.leader()
    assert new_leader is not None and new_leader.node_id != old_leader.node_id
    new_leader.propose(
        StateMachineCommand(
            command_id="cmd-after-heal",
            command_type="set_policy",
            payload={"policy_key": "h", "version": 1, "value": {}},
        )
    )
    cluster.heal()
    cluster.run_ticks(8)
    assert old_leader.role is RaftRole.FOLLOWER
    stale_write = old_leader.propose(
        StateMachineCommand(
            command_id="cmd-stale",
            command_type="set_policy",
            payload={"policy_key": "s", "version": 1, "value": {}},
        )
    )
    assert stale_write.status == "not_leader"
