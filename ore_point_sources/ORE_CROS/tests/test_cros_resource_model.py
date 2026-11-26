"""Acceptance tests for the CROS unified resource model (Steps 1-2, 47)."""
import pytest

from ore_cros.config import (
    AdaptationLimits, CROSConfig, ObjectiveWeights, WindowConfig,
)
from ore_cros.resource_model import (
    CriticalityLevel, EnvironmentProfile, ModuleBudget, ResourceDemand,
    ResourceDimension, ResourceKind, ResourceScope, ResourceState, ScopeLevel,
    TaskRecord, TaskState, WorkloadClass, build_resource_state, jain_fairness_index,
    machine_scope,
)


def _profile(**capacity):
    """Build a test environment profile with default test capacities."""
    base = {
        ResourceKind.CPU: 8.0, ResourceKind.RAM: 4096.0,
        ResourceKind.IO: None, ResourceKind.NET: None,
        ResourceKind.GPU: None, ResourceKind.STORAGE: 100_000.0,
        ResourceKind.QUEUE: 1024.0,
    }
    base.update(capacity)
    return EnvironmentProfile(
        profile_version="test-1", host_os="TestOS", kernel_release="1.0.0",
        capacity=base,
    )


def test_resource_vector_covers_all_seven_dimensions():
    """R_t covers CPU, RAM, IO, NET, GPU, STORAGE and QUEUE (spec 2.1)."""
    state = build_resource_state(_profile(), machine_scope(), version=3)
    assert set(state.dimensions) == set(ResourceKind)
    assert state.version == 3
    # A resource state missing any dimension is rejected at construction.
    incomplete = {kind: dim for kind, dim in state.dimensions.items()
                   if kind is not ResourceKind.QUEUE}
    with pytest.raises(ValueError):
        ResourceState(
            version=0, timestamp=0.0, scope=machine_scope(), dimensions=incomplete,
        )


def test_headroom_is_capacity_minus_reserved_minus_allocated():
    """Headroom_r = Capacity_r - Reserved_r - Allocated_r (spec 2.1)."""
    state = build_resource_state(
        _profile(), machine_scope(),
        allocations={ResourceKind.CPU: 3.0},
        reservations={ResourceKind.CPU: 2.0},
    )
    assert state.headroom(ResourceKind.CPU) == pytest.approx(3.0)
    assert state.headroom(ResourceKind.IO) is None  # unknown capacity


def test_unknown_capacity_is_not_zero():
    """Unknown dimensions keep headroom None rather than pretending empty."""
    dimension = ResourceDimension(kind=ResourceKind.GPU, capacity=None)
    assert dimension.unknown
    assert dimension.headroom is None


def test_demand_vector_math():
    """Demand vectors scale, dominate and reject negative components."""
    demand = ResourceDemand(cpu=1.0, ram=512.0)
    assert demand.scaled_by(2.0).cpu == 2.0
    assert demand.dominates(ResourceDemand(cpu=0.5, ram=512.0))
    with pytest.raises(ValueError):
        ResourceDemand(cpu=-1.0)


def test_task_records_cannot_self_promote_criticality():
    """Step 2.2: class assignment is policy-controlled, no self-promotion."""
    task = TaskRecord(
        task_id="t1", parent_goal_id="g1", module_id="m1",
        workload_class=WorkloadClass.BATCH, demand=ResourceDemand(cpu=0.5),
        remaining_work=1.0,
    )
    assert task.criticality is CriticalityLevel.NORMAL
    with pytest.raises(ValueError):
        task.reclassify(WorkloadClass.INTEGRITY_CRITICAL)
    promoted = task.reclassify(
        WorkloadClass.INTEGRITY_CRITICAL, policy_authority=True,
    )
    assert promoted.criticality is CriticalityLevel.CRITICAL
    # Demotion is always allowed.
    assert task.reclassify(WorkloadClass.EXPLORATORY).criticality is CriticalityLevel.OPTIONAL


def test_deadline_slack_computation():
    """slack_i = deadline_i - now - estimated_remaining_i (spec 3.3)."""
    task = TaskRecord(
        task_id="t2", parent_goal_id=None, module_id="m1",
        workload_class=WorkloadClass.INTERACTIVE, demand=ResourceDemand(),
        remaining_work=2.0, ready_time=0.0, deadline=10.0,
    )
    assert task.slack(now=5.0) == pytest.approx(3.0)
    assert TaskRecord(
        task_id="t3", parent_goal_id=None, module_id="m1",
        workload_class=WorkloadClass.BATCH, demand=ResourceDemand(),
        remaining_work=1.0,
    ).slack(now=0.0) is None


def test_jain_fairness_index_bounds():
    """The Jain index is 1.0 for equal shares and lower otherwise."""
    assert jain_fairness_index([1.0, 1.0, 1.0]) == pytest.approx(1.0)
    assert jain_fairness_index([3.0, 1.0]) == pytest.approx(0.8)
    assert jain_fairness_index([0.0, 0.0]) == 1.0
    with pytest.raises(ValueError):
        jain_fairness_index([-1.0])


def test_module_budget_validation():
    """A cap below the reservation is contradictory and rejected."""
    reserved = ResourceDemand(cpu=1.0)
    assert ModuleBudget("m", reserved, ResourceDemand(cpu=2.0)).borrowable is False
    with pytest.raises(ValueError):
        ModuleBudget("m", reserved, ResourceDemand(cpu=0.5))
    with pytest.raises(ValueError):
        ModuleBudget(" ", reserved, ResourceDemand(cpu=2.0))


def test_scope_topology_paths():
    """Scopes form a joinable topology without scope confusion (Step 2)."""
    machine = machine_scope("m1")
    process = ResourceScope(ScopeLevel.PROCESS, "p1", machine)
    task_scope = ResourceScope(ScopeLevel.TASK, "t1", process)
    assert task_scope.is_descendant_of(machine)
    assert process.path() == (machine, process)
    with pytest.raises(ValueError):
        ResourceScope(ScopeLevel.MODULE, " ")


def test_config_validates_objective_weights_for_every_class():
    """CROSConfig requires objective weights for every workload class."""
    config = CROSConfig()
    assert config.weights_for(WorkloadClass.INTERACTIVE).latency > 0.0
    incomplete = dict(config.objective_weights)
    del incomplete[WorkloadClass.BATCH]
    with pytest.raises(ValueError):
        CROSConfig(objective_weights=incomplete)
    with pytest.raises(ValueError):
        ObjectiveWeights(
            latency=-1.0, deadline_misses=0.0, cost=0.0,
            energy=0.0, unfairness=0.0, instability_risk=0.0,
        )


def test_adaptation_limits_bounds():
    """Adaptation limits reject out-of-range canary and fairness floors."""
    with pytest.raises(ValueError):
        AdaptationLimits(canary_fraction=1.5)
    with pytest.raises(ValueError):
        AdaptationLimits(fairness_floor=0.0)
    with pytest.raises(ValueError):
        WindowConfig(short_window_seconds=10.0, medium_window_seconds=1.0)
