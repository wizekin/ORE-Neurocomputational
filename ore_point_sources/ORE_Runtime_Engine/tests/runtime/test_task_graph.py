"""Acceptance tests for the task graph: ordering, cycles, and iterative regions."""
from __future__ import annotations

import pytest

from ore_runtime import (
    IterativeRegionSpec, TaskGraph, TaskSpec, TaskState,
)
from ore_runtime.exceptions import CycleRejectedError


def test_task_with_unfinished_predecessor_is_never_ready():
    """Dependency ordering: a task with an unfinished predecessor never becomes READY."""
    graph = TaskGraph((
        TaskSpec(task_id="a", name="first"),
        TaskSpec(task_id="b", name="second", dependencies=("a",)),
    ))
    assert [node.task_id for node in graph.ready_tasks()] == ["a"]
    graph.node("a").state = TaskState.RUNNING
    assert graph.ready_tasks() == ()
    graph.node("a").state = TaskState.SUCCEEDED
    assert [node.task_id for node in graph.ready_tasks()] == ["b"]


def test_accidental_cycle_is_rejected_with_diagnostic_path():
    """Cycle handling: an accidental cycle is rejected with a diagnostic path."""
    graph = TaskGraph((
        TaskSpec(task_id="a", name="first", dependencies=("c",)),
        TaskSpec(task_id="b", name="second", dependencies=("a",)),
        TaskSpec(task_id="c", name="third", dependencies=("b",)),
    ))
    with pytest.raises(CycleRejectedError) as excinfo:
        graph.validate_plan()
    assert excinfo.value.diagnostic_path
    assert set(excinfo.value.diagnostic_path) == {"a", "b", "c"}


def test_self_dependency_is_rejected_immediately():
    """A task can never list itself as a dependency."""
    with pytest.raises(CycleRejectedError):
        TaskGraph((TaskSpec(task_id="a", name="first", dependencies=("a",)),))


def test_unknown_dependency_is_rejected_by_manifest_validation():
    """Tasks referencing unknown dependencies fail manifest validation."""
    with pytest.raises(ValueError):
        TaskGraph((TaskSpec(task_id="b", name="second", dependencies=("missing",)),))


def test_critical_path_length_sums_estimated_costs():
    """The critical path length is the maximum summed cost along any path."""
    graph = TaskGraph((
        TaskSpec(task_id="a", name="first", estimated_cost=2.0),
        TaskSpec(task_id="b", name="second", estimated_cost=5.0, dependencies=("a",)),
        TaskSpec(task_id="c", name="third", estimated_cost=1.0, dependencies=("a",)),
    ))
    assert graph.critical_path_length() == 7.0


def test_iterative_region_converges_below_epsilon():
    """Iterative regions stop when successive states converge below epsilon."""
    graph = TaskGraph((TaskSpec(task_id="fixed", name="iterate"),))
    region = IterativeRegionSpec(
        region_id="refine",
        member_task_ids=("fixed",),
        update=lambda state: {key: value / 2.0 for key, value in state.items()},
        initial_state={"activation": 1.0},
        max_iterations=64,
        epsilon=1e-6,
    )
    result = graph.run_iterative_region(region)
    assert result.converged is True
    assert result.stopped_because == "converged_below_epsilon"
    assert result.iterations <= 64
    assert result.final_state["activation"] < 1e-5


def test_iterative_region_respects_iteration_limit():
    """Iterative regions respect the maximum iteration count when diverging."""
    graph = TaskGraph((TaskSpec(task_id="spin", name="spin"),))
    region = IterativeRegionSpec(
        region_id="spin",
        member_task_ids=("spin",),
        update=lambda state: {"value": state.get("value", 1.0) * 2.0},
        max_iterations=10,
        epsilon=1e-9,
    )
    result = graph.run_iterative_region(region)
    assert result.converged is False
    assert result.stopped_because == "iteration_limit_reached"
    assert result.iterations == 10


def test_iterative_region_stops_on_validation_predicate_failure():
    """A failing validation predicate stops iteration before divergence."""
    graph = TaskGraph((TaskSpec(task_id="gate", name="gate"),))
    region = IterativeRegionSpec(
        region_id="gated",
        member_task_ids=("gate",),
        update=lambda state: {"value": state.get("value", 0.0) + 1.0},
        validation_predicate=lambda state: state.get("value", 0.0) < 3,
        initial_state={"value": 0.0},
        max_iterations=32,
    )
    result = graph.run_iterative_region(region)
    assert result.converged is False
    assert result.stopped_because == "validation_predicate_failed"


def test_topological_order_respects_dependencies():
    """The Kahn order places every task after its dependencies."""
    graph = TaskGraph((
        TaskSpec(task_id="a", name="first"),
        TaskSpec(task_id="b", name="second", dependencies=("a",)),
        TaskSpec(task_id="c", name="third", dependencies=("a",)),
        TaskSpec(task_id="d", name="fourth", dependencies=("b", "c")),
    ))
    order = graph.topological_order()
    assert order.index("a") < order.index("b")
    assert order.index("a") < order.index("c")
    assert order.index("b") < order.index("d")
    assert order.index("c") < order.index("d")


def test_declared_iterative_region_is_registered():
    """Declared iterative regions are retained on the graph for execution."""
    graph = TaskGraph((TaskSpec(task_id="loop", name="loop"),))
    region = IterativeRegionSpec(region_id="r1", member_task_ids=("loop",))
    graph.declare_iterative_region(region)
    assert [item.region_id for item in graph.iterative_regions] == ["r1"]
    with pytest.raises(ValueError):
        graph.declare_iterative_region(region)
