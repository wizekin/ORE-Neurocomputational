"""End-to-end demonstration of the Point 17 distributed coordination layer.

The example walks one small multi-machine deployment through the full stack:
deployment classification, state classification, membership admission,
Raft-backed authoritative coordination with fencing-checked task execution,
CRDT observations with semantic conflict preservation, HLC-stamped
provenance, USG promotion, and safe degradation on quorum loss.
"""
from __future__ import annotations

from ore_distributed.core.identity import (
    ClusterCredentialAuthority,
    create_node_identity,
)
from ore_distributed.consensus.raft import RaftCluster
from ore_distributed.consensus.state_machine import StateMachineCommand
from ore_distributed.execution.tasks import TaskCoordinator, StaleOwner
from ore_distributed.gossip.suspicion import FailureSuspicion
from ore_distributed.integration.cros_adapter import (
    CROSResourceAdapter,
    ResourceTelemetry,
)
from ore_distributed.integration.observability import (
    DegradedModeController,
    DistributedMetrics,
)
from ore_distributed.integration.usg_adapter import USGValidationGate
from ore_distributed.pipeline import build_pipeline
from ore_distributed.replication.conflicts import (
    Claim,
    detect_semantic_contradiction,
)
from ore_distributed.replication.promotion import CandidateAssertion
from ore_distributed.replication.reconciliation import offline_replica_reconnect
from ore_distributed.replication.replica import CrdtReplica
from ore_distributed.time.hlc import HybridLogicalClock


def main() -> None:
    """Run the demonstration end to end."""
    print("== ORE Point 17: distributed coordination demonstration ==")

    # Phase A: deployment classification and state boundaries.
    pipeline = build_pipeline(
        "node-a", "ore-demo", process_count=3, machine_count=3
    )
    print("deployment mode:", pipeline.capabilities.to_dict()["topology"])
    print("task ownership class:", pipeline.classify_schema("ore.task.ownership")["state_class"])

    # Phase B: durable node identities and credentials.
    authority = ClusterCredentialAuthority("ore-demo")
    for node_id in ("node-a", "node-b", "node-c"):
        create_node_identity(node_id, "ore-demo", authority)
    print("issued credentials for", ", ".join(("node-a", "node-b", "node-c")))

    # Phase C: Raft authoritative coordination.
    cluster = RaftCluster(("node-a", "node-b", "node-c"), seed=17)
    leader = cluster.run_until_leader()
    print("elected leader:", leader.node_id, "term", leader.current_term)

    coordinator = TaskCoordinator(cluster)
    grant = coordinator.assign("demo-task", "worker-a", lease_expiry=10.0)
    print("task assigned:", grant.task_id, "owner", grant.owner, "epoch", grant.fencing_epoch)

    # A stale worker is fenced out of committing results.
    reassigned = coordinator.reassign_after_timeout("demo-task", "worker-b", lease_expiry=20.0)
    try:
        coordinator.commit_output("demo-task", "worker-a", reassigned.fencing_epoch - 1, "stale")
    except StaleOwner as exc:
        print("stale worker rejected:", exc)

    # Phase D: CRDT observations with preserved contradictions.
    left = CrdtReplica("agent-left")
    right = CrdtReplica("agent-right")
    claim_a = Claim("claim-a", "the membrane is selectively permeable", "agent-left",
                    evidence_ids=("ev-1",), hlc=(1, 0, "agent-left"), confidence=0.8)
    claim_b = Claim("claim-b", "the membrane is not selectively permeable", "agent-right",
                    evidence_ids=("ev-2",), hlc=(1, 1, "agent-right"), confidence=0.7)
    left.apply_local(claim_a.text)
    right.apply_local(claim_b.text)
    report = offline_replica_reconnect(left, right)
    conflict = detect_semantic_contradiction("permeability", [claim_a, claim_b], (2, 0, "usg"))
    print("replicas converged:", report.converged, "| open conflict:", conflict is not None)

    # Phase E: HLC-stamped provenance and suspicion.
    clock = HybridLogicalClock("node-a")
    print("hlc stamps:", [clock.tick().to_dict() for _ in range(2)])
    suspicion = FailureSuspicion(unreachable_after=3, suspected_after=6, removed_after=12)
    for _ in range(6):
        suspicion.record_missed_heartbeat("node-z")
    print("peer state after misses:", suspicion.state_of("node-z").value)

    # Phase G: USG promotion through the repository's own contracts.
    gate = USGValidationGate(namespace="ore.demo")
    candidate = CandidateAssertion(
        assertion_id="assert-demo",
        subject="cell membrane",
        predicate="regulates",
        payload={"relation": "regulates", "object": "substances"},
        source_reference="obs://agent-left/7",
        confidence=0.9,
        context={"task": "reading"},
        observed_event_id=left.observations()[0].event_id,
        provenance_digest="f" * 64,
    )
    outcome = gate.validate_and_promote(candidate)
    print("usg validation:", outcome.valid, "| evidence lineage:", outcome.evidence_ids)

    cros = CROSResourceAdapter()
    adapted = cros.adapt(ResourceTelemetry(
        network_pressure=0.9, cpu_pressure=0.5, memory_pressure=0.4,
        storage_pressure=0.3, queue_depth=20,
    ))
    print("cros adapted:", adapted)

    metrics = DistributedMetrics.from_cluster(cluster)
    print("observable metrics:", {
        key: value for key, value in metrics.snapshot().items()
        if key in ("leader_changes", "commit_latency_median_ticks",
                   "average_replication_lag_entries", "quorum_availability")
    })

    # Safe degradation on quorum loss.
    controller = DegradedModeController()
    cluster.partition({"node-a", "node-b"}, {"node-c"})
    cluster.crash_and_restart(leader.node_id)
    cluster.partition({"node-a"}, {"node-b"})
    cluster.run_ticks(30)
    if cluster.leader() is None:
        controller.enter_read_only_degraded()
    print("degraded status:", controller.status())

    cluster.heal()
    cluster.run_ticks(30)
    cluster.leader().propose(StateMachineCommand(
        command_id="demo-close",
        command_type="set_policy",
        payload={"policy_key": "demo", "version": 1, "value": {}},
    ))
    print("final leader writes:", cluster.leader() is not None)


if __name__ == "__main__":
    main()
