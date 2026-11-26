"""Acceptance tests for distributed placement, migration and failure (Steps 36-41)."""
import pytest

from ore_cros.distributed.placement import (
    CLUSTER_DESIGN_REFERENCES, ClusterResourceMap, FailureDetector,
    NodeAdvertisement, PlacementScorer, check_eligibility, decide_migration,
    filter_eligible_nodes,
)
from ore_cros.resource_model import (
    CheckpointSupport, ResourceDemand, SideEffectPolicy, TaskRecord, WorkloadClass,
)

NOW = 1_700_000_000.0


def _node(node_id, *, capabilities=("cap",), healthy=True, load_cpu=1.0,
          capacity_cpu=8.0, locality=("dataset-a",), version=1):
    """Build one node advertisement for cluster tests."""
    return NodeAdvertisement(
        node_id=node_id, capabilities=tuple(capabilities), healthy=healthy,
        capacity=ResourceDemand(cpu=capacity_cpu, ram=8192.0),
        load=ResourceDemand(cpu=load_cpu, ram=1024.0),
        data_locality=tuple(locality), version=version,
        published_at=NOW, expires_after_seconds=30.0,
    )


def _task(task_id="t1", *, capabilities=("cap",), locality=("dataset-a",),
          checkpoint=CheckpointSupport.CHECKPOINT, cpu=1.0,
          side_effects=SideEffectPolicy.NON_IDEMPOTENT_CHECKPOINTED):
    """Build one distributed task record."""
    return TaskRecord(
        task_id=task_id, parent_goal_id=None, module_id="m1",
        workload_class=WorkloadClass.CORE_COGNITIVE,
        demand=ResourceDemand(cpu=cpu), remaining_work=1.0,
        capabilities=tuple(capabilities), input_data_locality=tuple(locality),
        input_size_mib=100.0, checkpoint=checkpoint,
        side_effects=side_effects,
    )


def test_advertisements_expire_and_refresh():
    """Step 36: advertisements expire; stale versions are rejected."""
    cluster = ClusterResourceMap()
    cluster.publish(_node("n1", version=2))
    cluster.publish(_node("n1", version=3))
    with pytest.raises(ValueError):
        cluster.publish(_node("n1", version=1))
    assert cluster.nodes()[0].version == 3
    assert cluster.prune(now=NOW + 31.0) == ("n1",)
    assert cluster.nodes() == ()


def test_eligibility_filters_incapable_and_unhealthy_nodes():
    """Acceptance: distributed placement never selects incapable or
    unhealthy nodes (Step 37)."""
    task = _task()
    cluster = ClusterResourceMap()
    cluster.publish(_node("capable"))
    cluster.publish(_node("no-capability", capabilities=("other",)))
    cluster.publish(_node("unhealthy", healthy=False))
    cluster.publish(_node("no-data", locality=("dataset-z",)))
    cluster.publish(_node("overloaded", load_cpu=8.0, capacity_cpu=8.0))
    eligible = filter_eligible_nodes(task, cluster)
    assert [node.node_id for node in eligible] == ["capable"]
    check = check_eligibility(task, _node("unhealthy", healthy=False))
    assert not check.health_ok and not check.eligible
    assert "health" in check.reason


def test_eligibility_respects_policy_allowlist():
    """Policy constraints restrict placement to allowed nodes (Step 37)."""
    task = _task()
    cluster = ClusterResourceMap()
    cluster.publish(_node("n1"))
    cluster.publish(_node("n2"))
    allowed = filter_eligible_nodes(task, cluster, policy_allowed_nodes=("n2",))
    assert [node.node_id for node in allowed] == ["n2"]


def test_placement_scores_rank_by_cost_with_explanations():
    """Step 38: placement minimizes weighted completion cost with explanations."""
    task = _task()
    scorer = PlacementScorer(network_bandwidth_mbit=1000.0)
    local_light = _node("local-light", load_cpu=0.5, locality=("dataset-a",))
    local_loaded = _node("local-loaded", load_cpu=7.0, locality=("dataset-a",))
    remote_light = _node("remote-light", load_cpu=0.5, locality=("dataset-b",))
    decision = scorer.score(task, [local_loaded, remote_light, local_light])
    assert decision.selected_node == "local-light"
    assert decision.ranked_nodes[0] == "local-light"
    assert decision.scores["local-light"] < decision.scores["local-loaded"]
    # The remote node pays a transfer cost the local nodes do not.
    assert "transfer=" in decision.explanation["remote-light"]
    assert decision.explanation["local-light"].count("transfer=0.000") == 1
    empty = scorer.score(task, [])
    assert empty.selected_node is None


def test_migration_rejected_when_costs_exceed_benefit():
    """Acceptance: migration is rejected when transfer/restart costs exceed
    expected benefit (Step 40)."""
    task = _task()
    rejected = decide_migration(
        task, expected_benefit=1.0, transfer_cost=2.0,
        restart_cost=1.0, risk_penalty=0.5,
    )
    assert not rejected.advisable
    assert rejected.total_cost == pytest.approx(3.5)
    accepted = decide_migration(
        task, expected_benefit=5.0, transfer_cost=1.0,
        restart_cost=1.0, risk_penalty=0.5,
    )
    assert accepted.advisable
    # An advisable migration still requires a checkpoint or safe restart.
    unprotected = _task(
        "unprotected", checkpoint=CheckpointSupport.NONE,
        side_effects=SideEffectPolicy.NON_IDEMPOTENT_UNPROTECTED,
    )
    with pytest.raises(ValueError):
        decide_migration(
            unprotected, expected_benefit=5.0, transfer_cost=0.1,
            restart_cost=0.1, risk_penalty=0.1,
        )
    # Tasks declaring checkpointed side effects must actually carry a
    # checkpoint; the model rejects the contradiction at construction.
    with pytest.raises(ValueError):
        _task("contradictory", checkpoint=CheckpointSupport.NONE)


def test_node_failure_recovery_is_side_effect_aware():
    """Acceptance: node failure triggers side-effect-aware recovery (Step 41)."""
    idempotent = TaskRecord(
        task_id="idem", parent_goal_id=None, module_id="m",
        workload_class=WorkloadClass.BATCH, demand=ResourceDemand(cpu=0.1),
        remaining_work=1.0, checkpoint=CheckpointSupport.NONE,
        side_effects=SideEffectPolicy.IDEMPOTENT,
    )
    checkpointed = _task("ckpt")
    unprotected = _task(
        "raw", checkpoint=CheckpointSupport.NONE,
        side_effects=SideEffectPolicy.NON_IDEMPOTENT_UNPROTECTED,
    )
    detector = FailureDetector(heartbeat_timeout_seconds=5.0)
    detector.record_heartbeat("n1", now=NOW)
    assert detector.unavailable_nodes(now=NOW + 10.0) == ("n1",)
    presumed = detector.recover(idempotent, node_presumed_alive=True)
    assert presumed.presumed_running and presumed.action == "retry"
    retry = detector.recover(idempotent, node_presumed_alive=False)
    assert retry.action == "retry"
    restart = detector.recover(checkpointed, node_presumed_alive=False)
    assert restart.action == "restart-from-checkpoint"
    cancelled = detector.recover(unprotected, node_presumed_alive=False)
    assert cancelled.action == "cancel"


def test_timeout_does_not_prove_task_stopped():
    """A heartbeat timeout marks nodes unavailable without claiming proof."""
    detector = FailureDetector(heartbeat_timeout_seconds=5.0)
    detector.record_heartbeat("n1", now=NOW)
    detector.record_heartbeat("n2", now=NOW)
    assert detector.unavailable_nodes(now=NOW + 4.0) == ()  # within the timeout
    detector.isolate("n2")
    assert detector.unavailable_nodes(now=NOW + 10.0) == ("n1",)


def test_cluster_design_references_documented():
    """Step 39: Borg, Kubernetes and Decima remain documented references."""
    assert {item["reference"] for item in CLUSTER_DESIGN_REFERENCES} == {
        "Borg", "Kubernetes Scheduler", "Decima",
    }
    assert all("principle_adopted" in item for item in CLUSTER_DESIGN_REFERENCES)
