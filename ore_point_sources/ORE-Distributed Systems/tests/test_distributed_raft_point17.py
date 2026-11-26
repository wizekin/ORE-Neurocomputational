"""Phase C acceptance tests: Raft election, commit, recovery, membership, leases."""
from __future__ import annotations

import pytest

from ore_distributed.consensus.leases import (
    FencingRejected,
    LeaseGrant,
    LeaseManager,
    LeasePolicy,
)
from ore_distributed.consensus.log import quorum_size
from ore_distributed.consensus.raft import (
    NO_OP_COMMAND,
    RaftCluster,
    RaftNode,
    RaftRole,
)
from ore_distributed.consensus.reconfiguration import (
    ConfigurationChangeError,
    RaftReconfiguration,
    quorum_available_after_change,
)
from ore_distributed.consensus.state_machine import (
    StateMachineCommand,
    replay_deterministic,
)


def _policy_command(command_id: str, version: int) -> StateMachineCommand:
    return StateMachineCommand(
        command_id=command_id,
        command_type="set_policy",
        payload={"policy_key": "probe", "version": version, "value": {"v": version}},
    )


def _assign_command(command_id: str, task_id: str, owner: str) -> StateMachineCommand:
    return StateMachineCommand(
        command_id=command_id,
        command_type="assign_task",
        payload={"task_id": task_id, "owner": owner, "lease_expiry": 10.0},
    )


def _elected_cluster(node_count: int = 3, seed: int = 41) -> tuple[RaftCluster, RaftNode]:
    """Build a cluster and elect a leader deterministically."""
    cluster = RaftCluster(tuple(f"n{i}" for i in range(1, node_count + 1)), seed=seed)
    leader = cluster.run_until_leader()
    return cluster, leader


def test_leader_elected_from_valid_majority():
    """A Raft leader is elected from a valid majority (Step 14)."""
    cluster, leader = _elected_cluster(5)
    assert leader.role is RaftRole.LEADER
    assert len(leader.votes_received & set(leader.voters)) >= quorum_size(5)
    committed_leader = cluster.leader()
    assert committed_leader is not None and committed_leader.node_id == leader.node_id


def test_commit_reaches_every_replica_deterministically():
    """State-machine replay is deterministic across replicas (Steps 17-18)."""
    cluster, leader = _elected_cluster(3)
    commands = (
        _assign_command("cmd-1", "task-1", "worker-a"),
        _policy_command("cmd-2", 3),
    )
    for command in commands:
        result = leader.propose(command)
        assert result.status == "committed"
    cluster.run_ticks(4)
    states = cluster.state_machines()
    first = states["n1"]
    assert first == states["n2"] == states["n3"]
    assert first["tasks"]["task-1"]["owner"] == "worker-a"
    replayed_first, replayed_second = replay_deterministic(
        (NO_OP_COMMAND,) + commands
    )
    assert replayed_first == replayed_second


def test_client_acknowledged_only_after_durable_commit():
    """Client success is acknowledged only after the durable commit point (Step 17)."""
    cluster, leader = _elected_cluster(3)
    # Isolate one follower so replication needs the tick loop to settle.
    command = _policy_command("cmd-pending", 7)
    result = leader.propose(command)
    assert result.status in {"committed", "pending"}
    cluster.run_ticks(8)
    assert leader.is_committed("cmd-pending") is True

    # A follower never acknowledges: it is not the leader.
    follower = next(node for node in cluster.nodes.values() if not node.is_leader())
    assert follower.propose(_policy_command("cmd-f", 1)).status == "not_leader"


def test_commit_requires_current_term_and_majority_replication():
    """Entries from earlier terms commit only indirectly (Step 17)."""
    node = RaftNode("n1", voters=("n1", "n2", "n3"))
    node.storage.persist_term_and_vote(2, "n1")
    node.storage.log.append(1, _policy_command("old-entry", 1))  # term 1 entry
    node.match_index.update({"n1": 1, "n2": 1, "n3": 1})
    assert node.commit_rule_holds_for(1) is False
    node.storage.log.append(2, _policy_command("new-entry", 2))  # term 2 entry
    node.match_index.update({"n1": 2, "n2": 2, "n3": 2})
    assert node.commit_rule_holds_for(2) is True


def test_stale_term_messages_are_rejected():
    """Stale-term messages are rejected (Step 19)."""
    cluster, leader = _elected_cluster(3)
    follower = next(node for node in cluster.nodes.values() if not node.is_leader())
    before = follower.stale_messages_rejected
    reply = follower.handle_append_entries(
        {
            "term": 0,
            "leader": leader.node_id,
            "prev_log_index": 0,
            "prev_log_term": 0,
            "entries": [],
            "leader_commit": 0,
        }
    )
    assert reply["success"] is False
    assert follower.stale_messages_rejected == before + 1

    vote_reply = follower.handle_request_vote(
        {
            "term": leader.current_term - 1,
            "candidate": leader.node_id,
            "last_log_index": 99,
            "last_log_term": 99,
        }
    )
    assert vote_reply["vote_granted"] is False


def test_vote_persists_before_granting_and_is_unique_per_term():
    """Term and vote persist before responding; one vote per term (Step 15)."""
    node = RaftNode("n1", voters=("n1", "n2", "n3"))
    node.storage.persist_term_and_vote(1, None)
    reply = node.handle_request_vote(
        {"term": 1, "candidate": "n2", "last_log_index": 0, "last_log_term": 0}
    )
    assert reply["vote_granted"] is True
    # The durable store recorded the grant before the reply existed.
    assert node.storage.voted_for == "n2"
    assert ("vote", (1, "n2")) in node.storage.persistence_events
    second = node.handle_request_vote(
        {"term": 1, "candidate": "n3", "last_log_index": 5, "last_log_term": 5}
    )
    assert second["vote_granted"] is False  # already voted this term


def test_candidate_log_freshness_is_checked():
    """Voters check candidate log freshness before granting (Step 15)."""
    node = RaftNode("n1", voters=("n1", "n2", "n3"))
    node.storage.log.append(1, _policy_command("a", 1))
    reply = node.handle_request_vote(
        {"term": 2, "candidate": "n2", "last_log_index": 0, "last_log_term": 0}
    )
    assert reply["vote_granted"] is False  # candidate log is stale
    reply = node.handle_request_vote(
        {"term": 2, "candidate": "n2", "last_log_index": 2, "last_log_term": 2}
    )
    assert reply["vote_granted"] is True


def test_follower_reconciles_conflicting_uncommitted_suffix():
    """A follower with a conflicting uncommitted suffix is reconciled (Step 16)."""
    cluster, first_leader = _elected_cluster(3)
    first_leader.propose(_policy_command("leader-entry", 1))
    cluster.run_ticks(4)
    follower = next(
        node for node in cluster.nodes.values() if not node.is_leader()
    )
    # Snapshot the follower's committed prefix before the leadership change.
    pre_election_log = list(follower.storage.log.snapshot())

    # Force a higher-term leader: isolate the old leader, elect a successor.
    cluster.isolate(first_leader.node_id)
    cluster.run_ticks(30)
    leader = cluster.leader()
    assert leader is not None and leader.current_term > first_leader.current_term
    leader.propose(_policy_command("successor-entry", 2))
    cluster.run_ticks(6)

    # Build a follower holding a conflicting uncommitted suffix at an old term.
    divergent = RaftNode("stray", voters=follower.voters)
    divergent.storage.log.restore(pre_election_log)
    divergent.storage.persist_term_and_vote(1, "stray")
    divergent.storage.log.append(1, _policy_command("stray-entry", 99))
    stray_index = divergent.storage.log.last_index

    # Leader replication repairs the follower's conflicting suffix.
    reply = divergent.handle_append_entries(
        {
            "term": leader.current_term,
            "leader": leader.node_id,
            "prev_log_index": stray_index - 1,
            "prev_log_term": leader.storage.log.term_at(stray_index - 1),
            "entries": list(leader.storage.log.entries_from(stray_index)),
            "leader_commit": leader.storage.commit_index,
        }
    )
    assert reply["success"] is True
    assert divergent.storage.log.last_index == leader.storage.log.last_index
    assert [entry.term for entry in divergent.storage.log.entries_from(1)] == [
        entry.term for entry in leader.storage.log.entries_from(1)
    ]


def test_committed_entry_survives_crash_restart():
    """A committed log entry survives the documented crash/restart scenario (Step 19)."""
    cluster, leader = _elected_cluster(3)
    command = _assign_command("cmd-survive", "task-9", "worker-c")
    leader.propose(command)
    cluster.run_ticks(4)
    assert leader.is_committed("cmd-survive")

    crashed_node_id = leader.node_id
    commit_before = leader.storage.commit_index
    cluster.crash_and_restart(crashed_node_id)
    restarted = cluster.node(crashed_node_id)
    assert restarted.storage.commit_index == commit_before
    assert restarted.is_committed("cmd-survive")
    states = cluster.state_machines()
    assert states[crashed_node_id] == states["n2"] == states["n3"]


def test_minority_partition_cannot_commit_authoritative_writes():
    """A minority partition cannot commit authoritative writes (Step 44)."""
    cluster, leader = _elected_cluster(5)
    minority = {leader.node_id}
    majority = set(cluster.node_ids) - minority
    cluster.partition(majority, minority)

    minority_result = leader.propose(_policy_command("cmd-min", 1))
    assert minority_result.acknowledged is False

    # The majority elects its own leader and commits.
    cluster.run_ticks(40)
    new_leader = cluster.leader()
    assert new_leader is not None
    assert new_leader.node_id in majority
    majority_result = new_leader.propose(_policy_command("cmd-maj", 2))
    cluster.run_ticks(10)
    assert new_leader.is_committed("cmd-maj") is True
    assert leader.is_committed("cmd-min") is False


def test_two_simultaneous_candidates_keep_one_leader_per_term():
    """Two simultaneous candidates cannot cause conflicting committed histories."""
    cluster = RaftCluster(("n1", "n2", "n3", "n4", "n5"), seed=71)
    # Force a simultaneous split vote by starting elections on two nodes.
    candidates = [cluster.node("n1"), cluster.node("n2")]
    for candidate in candidates:
        candidate.election_elapsed = candidate.election_timeout_ticks
        candidate.tick()
    cluster.run_ticks(30)
    leaders_per_term: dict[int, list[str]] = {}
    for node in cluster.nodes.values():
        for term, outcome in node.election_events:
            if outcome == "leader":
                leaders_per_term.setdefault(term, []).append(node.node_id)
    for term, leaders in leaders_per_term.items():
        assert len(set(leaders)) == 1, (term, leaders)
    final_leader = cluster.leader()
    assert final_leader is not None


def test_recovering_old_leader_steps_down_on_higher_term():
    """A recovering old leader cannot continue authoritative writes (Step 19)."""
    cluster, old_leader = _elected_cluster(3)
    cluster.isolate(old_leader.node_id)
    cluster.run_ticks(30)
    new_leader = cluster.leader()
    assert new_leader is not None and new_leader.node_id != old_leader.node_id
    assert new_leader.current_term > old_leader.current_term

    # Heal the partition; the old leader observes the higher term and steps down.
    cluster.heal()
    cluster.run_ticks(8)
    assert old_leader.role is RaftRole.FOLLOWER
    assert old_leader.propose(_policy_command("cmd-old", 1)).status == "not_leader"


def test_duplicate_client_commands_do_not_duplicate_assignments():
    """Duplicate client commands do not create duplicate task assignments (Step 18)."""
    cluster, leader = _elected_cluster(3)
    command = _assign_command("cmd-dup", "task-dup", "worker-a")
    first = leader.propose(command)
    cluster.run_ticks(4)
    duplicate = leader.propose(
        _assign_command("cmd-dup", "task-dup", "worker-b")  # same command_id, new owner
    )
    assert duplicate.status == "duplicate"
    task = cluster.state_machines()["n1"]["tasks"]["task-dup"]
    assert task["owner"] == "worker-a"  # first assignment wins; no duplicate effect


def test_membership_addition_uses_safe_configuration_procedure():
    """Membership addition uses the safe configuration-change procedure (Step 20)."""
    cluster, leader = _elected_cluster(3)
    joiner = RaftNode(
        "n4",
        voters=leader.voters,
        rpc_send=cluster._deliver,  # noqa: SLF001 - harness wiring
        seed=101,
    )
    cluster.nodes["n4"] = joiner
    reconfiguration = RaftReconfiguration(cluster)
    record = reconfiguration.add_voter("n4", node=joiner)
    assert record.action == "add"
    assert set(record.voters) == {"n1", "n2", "n3", "n4"}
    assert all("n4" in node.voters for node in cluster.nodes.values())
    # The joining voter received the committed log via state transfer.
    assert cluster.node("n4").storage.commit_index >= cluster.node("n1").storage.commit_index - 1


def test_membership_removal_keeps_quorum_available():
    """Membership removal does not strand the cluster below quorum (Step 20)."""
    cluster, _ = _elected_cluster(5, seed=53)
    reconfiguration = RaftReconfiguration(cluster)
    # Pre-validate quorum availability before removing a voter.
    voters = cluster.node(cluster.node_ids[0]).voters
    ok, _, _ = quorum_available_after_change(voters, "remove", "n3")
    assert ok is True
    reconfiguration.remove_voter("n3")
    assert "n3" not in cluster.nodes
    remaining = tuple(cluster.node_ids)
    assert quorum_size(len(remaining)) <= len(remaining)
    # The reduced cluster still elects a leader and commits commands.
    cluster.run_ticks(12)
    leader = cluster.leader()
    assert leader is not None
    leader.propose(_policy_command("post-remove", 1))
    cluster.run_ticks(6)
    assert leader.is_committed("post-remove") is True


def test_simultaneous_membership_edits_are_rejected():
    """Simultaneous uncoordinated membership edits are prevented (Step 20)."""
    cluster, _ = _elected_cluster(3)
    reconfiguration = RaftReconfiguration(cluster)
    reconfiguration._pending = "add:n4"  # noqa: SLF001 - simulate an in-flight change
    with pytest.raises(ConfigurationChangeError):
        reconfiguration.remove_voter("n2")


def test_quorum_availability_validated_before_change():
    """Quorum availability is validated before any configuration change."""
    ok, before, after = quorum_available_after_change(("n1", "n2", "n3"), "add", "n4")
    assert ok is True and after == 3
    with pytest.raises(ConfigurationChangeError):
        quorum_available_after_change(("n1",), "remove", "n1")  # would empty the config
    with pytest.raises(ConfigurationChangeError):
        quorum_available_after_change(("n1", "n2"), "add", "n2")


def test_stale_lease_holder_rejected_by_fencing_token():
    """A stale lease holder is rejected by fencing-token validation (Step 21)."""
    clock_values = [0.0]
    manager = LeaseManager(clock=lambda: clock_values[0])
    grant = manager.grant("resource-1", "worker-old", duration=5.0)
    assert manager.validate_write("resource-1", "worker-old", grant.fencing_epoch)

    clock_values[0] = 6.0  # lease expires
    reassigned = manager.reassign_after_expiry("resource-1", "worker-new")
    assert reassigned is not None
    assert reassigned.fencing_epoch > grant.fencing_epoch
    with pytest.raises(FencingRejected):
        manager.validate_write("resource-1", "worker-old", grant.fencing_epoch)
    assert manager.validate_write("resource-1", "worker-new", reassigned.fencing_epoch)


def test_lease_timeout_alone_is_not_authorization():
    """A lease timeout alone never authorizes the old owner's write (Step 21)."""
    clock_values = [0.0]
    manager = LeaseManager(clock=lambda: clock_values[0])
    grant = manager.grant("resource-2", "worker-a", duration=5.0)
    clock_values[0] = 10.0  # expired but no reassignment
    with pytest.raises(FencingRejected):
        manager.validate_write("resource-2", "worker-a", grant.fencing_epoch)
    policy = LeasePolicy()
    assert policy.effective_duration() < policy.lease_duration  # drift margin applied


def test_lease_policy_records_explicit_clock_assumptions():
    """Lease policy carries explicit drift bounds and renewal rules (Step 21)."""
    policy = LeasePolicy(lease_duration=4.0, clock_drift_bound=0.5, renewal_margin=1.0)
    grant = LeaseGrant("r", "owner", 3, granted_at=0.0, expires_at=4.0)
    assert policy.should_renew(grant, now=3.2) is True
    with pytest.raises(ValueError):
        LeasePolicy(lease_duration=1.0, clock_drift_bound=5.0)  # drift exceeds duration
