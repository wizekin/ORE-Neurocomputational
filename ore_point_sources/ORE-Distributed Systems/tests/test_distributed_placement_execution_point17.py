"""Phase F acceptance tests: placement, task leases, idempotency, checkpoints."""
from __future__ import annotations

import pytest

from ore_distributed.core.identity import digest_hex
from ore_distributed.consensus.raft import RaftCluster
from ore_distributed.consensus.state_machine import StateMachineCommand
from ore_distributed.execution.checkpoints import (
    CheckpointStore,
    build_checkpoint_from_state,
)
from ore_distributed.execution.idempotency import (
    IdempotencyRegistry,
    TransactionalInbox,
    TransactionalOutbox,
    at_least_once_vs_exactly_once_note,
    retries_do_not_duplicate_effects,
)
from ore_distributed.execution.tasks import (
    StaleOwner,
    TaskCoordinator,
    delayed_worker_write_check,
    stale_worker_cannot_overwrite,
)
from ore_distributed.placement.consistent_hashing import (
    ConsistentHashRing,
    rebalance_plan,
    shard_balance,
)
from ore_distributed.placement.router import DataClass, ShardRouter, distinct_key_spaces
from ore_distributed.recovery.recovery import RecoveryManager


def _elected_cluster(node_count: int = 3, seed: int = 61) -> tuple[RaftCluster, object]:
    cluster = RaftCluster(tuple(f"n{i}" for i in range(1, node_count + 1)), seed=seed)
    return cluster, cluster.run_until_leader()


def test_consistent_hash_placement_moves_only_expected_keys_on_churn():
    """Placement changes only the expected subset of keys during churn (Step 38)."""
    nodes = tuple(f"node-{i}" for i in range(1, 6))
    ring = ConsistentHashRing(nodes, virtual_nodes=64, replication_factor=3)
    keys = [f"key-{i}" for i in range(120)]
    before = {key: ring.owners_of(key).owners for key in keys}

    # Adding one node moves only the keys that gained the new node; never all.
    ring.add_node("node-6")
    moved_after_add = rebalance_plan(ring, before, keys)
    assert 0 < len(moved_after_add) < len(keys)
    assert all("node-6" in ring.owners_of(key).owners for key in moved_after_add)

    after_add = {key: ring.owners_of(key).owners for key in keys}
    ring.remove_node("node-3")
    moved_after_remove = rebalance_plan(ring, after_add, keys)
    # Removal touches exactly the keys the removed node used to own.
    assert set(moved_after_remove) == {key for key in keys if "node-3" in after_add[key]}
    assert len(moved_after_remove) < len(keys)
    # Every key remains placed with a full replication set.
    assert all(len(ring.owners_of(key).owners) == 3 for key in keys)


def test_shard_rebalancing_respects_failure_domains():
    """Shard rebalancing respects configured failure-domain constraints (Step 38)."""
    domains = {
        "node-1": "rack-a",
        "node-2": "rack-a",
        "node-3": "rack-b",
        "node-4": "rack-b",
        "node-5": "rack-c",
    }
    ring = ConsistentHashRing(
        tuple(domains), virtual_nodes=64, replication_factor=3, failure_domains=domains
    )
    multi_domain_hits = 0
    for index in range(60):
        placement = ring.owners_of(f"domain-key-{index}")
        owner_domains = {domains[owner] for owner in placement.owners}
        if len(owner_domains) >= 2:
            multi_domain_hits += 1
    assert multi_domain_hits >= 55  # replicas spread across domains
    balance = shard_balance(
        {f"key-{i}": ring.owners_of(f"key-{i}").owners for i in range(60)}
    )
    assert len(balance) == 5 and max(balance.values()) / min(balance.values()) < 3.0


def test_placement_balance_uses_virtual_nodes():
    """Virtual nodes reduce placement imbalance (Step 38)."""
    ring = ConsistentHashRing(tuple(f"node-{i}" for i in range(1, 5)), virtual_nodes=128)
    counts = ring.placement_balance()
    assert len(counts) == 4
    assert set(counts.values()) == {128}


def test_distinct_partition_keys_per_data_class():
    """Data classes use distinct partition keys; control metadata stays in consensus (Step 39)."""
    ring = ConsistentHashRing(("node-1", "node-2", "node-3", "node-4"))
    router = ShardRouter(ring)
    keys = distinct_key_spaces(router, "membrane-lesson")
    assert len(set(keys)) == 4
    assert {key.split(":")[0] for key in keys} == {"episodic", "semantic", "queue", "artifact"}
    placement = router.route(DataClass.EPISODIC_RECORD, "lesson-7")
    assert placement.owners and len(placement.owners) == 3
    assert router.route_control("task-ownership") == "consensus_state_machine"
    query = router.resolve_usg_query(("concept-1", "concept-2"))
    assert [p.data_class for p in query] == [DataClass.SEMANTIC_INDEX] * 2
    placement_data, provenance = router.retrieve_with_provenance(placement, {})
    assert placement_data["partition_key"] == placement.partition_key
    assert provenance is None


def test_task_assignment_uses_authoritative_state_machine():
    """Task assignment commits through the authoritative path (Step 40)."""
    cluster, _ = _elected_cluster()
    coordinator = TaskCoordinator(cluster)
    record = coordinator.assign("task-1", "worker-a", lease_expiry=100.0)
    assert record.owner == "worker-a" and record.fencing_epoch >= 1
    ownership = coordinator.current_ownership("task-1")
    assert ownership.owner == "worker-a"
    states = cluster.state_machines()
    assert states["n1"]["tasks"]["task-1"]["owner"] == "worker-a"
    assert states["n1"] == states["n2"] == states["n3"]


def test_stale_lease_holder_cannot_write_task_output():
    """A stale lease holder is rejected by fencing-token validation (Step 40)."""
    cluster, _ = _elected_cluster()
    coordinator = TaskCoordinator(cluster)
    coordinator.assign("task-2", "worker-a", lease_expiry=5.0)
    # Reassignment advances ownership with a strictly higher fencing epoch.
    reassigned = coordinator.reassign_after_timeout("task-2", "worker-b", lease_expiry=50.0)
    assert reassigned.owner == "worker-b" and reassigned.attempt >= 2

    accepted, reason = stale_worker_cannot_overwrite(
        coordinator, "task-2", "worker-a", reassigned.fencing_epoch - 1, "stale-digest"
    )
    assert accepted is True
    assert "stale fencing epoch" in reason


def test_delayed_worker_cannot_overwrite_newer_result():
    """A delayed worker cannot overwrite a newer task result (Step 40)."""
    cluster, _ = _elected_cluster()
    coordinator = TaskCoordinator(cluster)
    coordinator.assign("task-3", "worker-slow", lease_expiry=5.0)
    reassigned = coordinator.reassign_after_timeout("task-3", "worker-fast", lease_expiry=50.0)
    coordinator.commit_output("task-3", "worker-fast", reassigned.fencing_epoch, "digest-fast")
    with pytest.raises(StaleOwner):
        coordinator.commit_output("task-3", "worker-slow", 1, "digest-slow")
    delayed_ok, _ = delayed_worker_write_check(coordinator, "task-3", "worker-slow", 1)
    assert delayed_ok is True
    task = cluster.leader().state_machine.tasks["task-3"]
    assert task["output_digest"] == "digest-fast"


def test_worker_must_validate_ownership_before_commit():
    """Workers validate current ownership before committing outputs (Step 40)."""
    cluster, _ = _elected_cluster()
    coordinator = TaskCoordinator(cluster)
    record = coordinator.assign("task-4", "worker-a", lease_expiry=100.0)
    assert coordinator.validate_worker_output("task-4", "worker-a", record.fencing_epoch)
    with pytest.raises(StaleOwner):
        coordinator.validate_worker_output("task-4", "worker-a", record.fencing_epoch - 1)
    with pytest.raises(StaleOwner):
        coordinator.validate_worker_output("task-4", "worker-imposter", record.fencing_epoch)


def test_task_retries_do_not_duplicate_protected_effects():
    """Task retries do not duplicate protected external effects (Step 41)."""
    registry = IdempotencyRegistry()
    executions, duplicates = retries_do_not_duplicate_effects(
        registry, "task-protected", delivery_attempts=5
    )
    assert executions == 1
    assert duplicates == 4
    note = at_least_once_vs_exactly_once_note()
    assert "at-least-once" in note and "exactly-once" in note


def test_transactional_outbox_and_inbox_deduplicate_deliveries():
    """Outbox/inbox patterns bound external effect duplication (Step 41)."""
    outbox = TransactionalOutbox()
    inbox = TransactionalInbox()
    outbox.enqueue("cmd-1", {"effect": "publish"})
    assert len(outbox.drain()) == 1
    assert outbox.pending() == ()
    assert inbox.receive("msg-1", {"effect": "publish"}) == "new"
    assert inbox.receive("msg-1", {"effect": "publish"}) == "duplicate"
    assert inbox.duplicates_dropped == 1
    with pytest.raises(ValueError):
        outbox.enqueue("", {"effect": "x"})


def test_checkpoint_validity_requires_durable_artifacts():
    """Checkpoints are valid only when referenced artifacts verify (Step 42)."""
    store = CheckpointStore()
    inputs = {"lesson": b"lesson-bytes"}
    outputs = {"summary": b"summary-bytes"}
    checkpoint = build_checkpoint_from_state(
        checkpoint_id="ckpt-1",
        workflow_id="workflow-1",
        workflow_state={"stage": "midway"},
        committed_task_ids=("task-1",),
        inputs=inputs,
        outputs=outputs,
        dependency_graph={"task-1": ()},
        coordinator_epoch=3,
        recovery_cursor=2,
    )
    for payload in list(inputs.values()) + list(outputs.values()):
        store.register_artifact(digest_hex(payload), payload)
    store.save(checkpoint)
    assert store.latest_for("workflow-1").checkpoint_id == "ckpt-1"

    # A stored artifact whose payload no longer matches its digest is rejected.
    tampered_payload = b"tampered-summary"
    tampered_digest = digest_hex(tampered_payload)
    store.artifacts[tampered_digest] = b"silently-corrupted-bytes"
    tampered = build_checkpoint_from_state(
        checkpoint_id="ckpt-2",
        workflow_id="workflow-1",
        workflow_state={"stage": "final"},
        committed_task_ids=("task-1", "task-2"),
        inputs={"lesson": b"lesson-bytes"},
        outputs={"summary": tampered_payload},
        dependency_graph={"task-1": (), "task-2": ("task-1",)},
        coordinator_epoch=4,
        recovery_cursor=5,
    )
    with pytest.raises(ValueError):
        store.save(tampered)
    assert store.get("ckpt-2") is None


def test_recovery_restores_committed_workflow_state():
    """Recovery restores committed workflow state and verifies artifacts (Step 43)."""
    cluster, leader = _elected_cluster()
    transition = StateMachineCommand(
        command_id="cmd-transition",
        command_type="transition_workflow",
        payload={"workflow_id": "workflow-9", "state": "processing"},
    )
    leader.propose(transition)
    cluster.run_ticks(4)

    store = CheckpointStore()
    payload = b"workflow-snapshot"
    digest = digest_hex(payload)
    store.register_artifact(digest, payload)
    checkpoint = build_checkpoint_from_state(
        checkpoint_id="ckpt-r",
        workflow_id="workflow-9",
        workflow_state={"stage": "processing", "artifact": digest},
        committed_task_ids=(),
        inputs={"snapshot": payload},
        outputs={},
        dependency_graph={},
        coordinator_epoch=leader.current_term,
        recovery_cursor=cluster.committed_indices()[leader.node_id],
    )
    store.save(checkpoint)

    recovery = RecoveryManager(cluster, store)
    # The node's process is lost first; recovery then rejoins from storage.
    cluster.crash_and_restart(leader.node_id)
    report = recovery.recover_node(leader.node_id, workflow_id="workflow-9")
    assert "authenticated_rejoin" in report.steps
    assert report.checkpoint_restored == "ckpt-r"
    assert report.authorized is True
    states = cluster.state_machines()
    assert states[leader.node_id]["workflows"]["workflow-9"]["state"] == "processing"
    assert states[leader.node_id] == states["n2"] == states["n3"]


def test_recovered_local_state_is_not_authoritative_until_reconciled():
    """Recovered local state publishes only after reconciliation (Step 43)."""
    cluster, leader = _elected_cluster()
    leader.propose(
        StateMachineCommand(
            command_id="cmd-x",
            command_type="set_policy",
            payload={"policy_key": "k", "version": 1, "value": {}},
        )
    )
    cluster.run_ticks(4)
    follower = next(node for node in cluster.nodes.values() if not node.is_leader())
    stale_local_state = follower.state_machine.snapshot()
    # The follower never reconciled the latest committed entry.
    stale_local_state["applied_commands"].pop("cmd-x", None)
    recovery = RecoveryManager(cluster, CheckpointStore())
    assert recovery.recovered_state_is_authoritative(
        stale_local_state, leader.state_machine.snapshot()
    ) is False


def test_missing_events_are_replayed_after_downtime():
    """A recovered node replays the events it missed (Step 43)."""
    journal_events = {"ev-1": {}, "ev-2": {}, "ev-3": {}}
    received_while_down = {"ev-1"}
    recovery = RecoveryManager(*_elected_cluster()[0:1], CheckpointStore())
    missing = recovery.replay_missing_events(journal_events, received_while_down)
    assert missing == ("ev-2", "ev-3")
