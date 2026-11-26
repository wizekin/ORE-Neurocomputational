"""Acceptance tests for task and module resource attribution (Step 5)."""
import pytest

from ore_cros.accounting.task_costs import TaskUsageAccountant
from ore_cros.resource_model import (
    ResourceDemand, ResourceKind, ResourceState, TaskRecord, WorkloadClass,
)
from ore_cros.telemetry.schemas import Measurement, MetricKind

NOW = 1_700_000_000.0


def _task(task_id, module_id="m1", workload_class=WorkloadClass.CORE_COGNITIVE):
    """Build a minimal task record for accounting tests."""
    return TaskRecord(
        task_id=task_id, parent_goal_id="goal-1", module_id=module_id,
        workload_class=workload_class, demand=ResourceDemand(cpu=1.0, ram=100.0),
        remaining_work=1.0,
    )


def _cpu_measurement(value, timestamp, scope):
    """Build one CPU allocation measurement at the given scope."""
    return Measurement(
        resource=ResourceKind.CPU, metric=MetricKind.ALLOCATED, value=value,
        unit="cores", timestamp=timestamp, scope=scope, source="test",
    )


def test_directly_attributed_usage_is_exact():
    """Usage attributed to the correct task is not marked estimated."""
    from ore_cros.resource_model import machine_scope, ScopeLevel, ResourceScope

    accountant = TaskUsageAccountant()
    task = _task("t1")
    accountant.open_task(task, started_at=NOW)
    for index in range(4):
        accountant.record_sample(
            "t1", "m1", ResourceKind.CPU, 0.5, timestamp=NOW + index,
        )
    accountant.close_task("t1", ended_at=NOW + 4.0)
    rows = accountant.accounting_table()
    assert len(rows) == 1
    assert rows[0].task_id == "t1"
    assert rows[0].module_id == "m1"
    assert rows[0].estimated is False
    # Rate metric: mean rate (0.5 cores) integrated over 4 seconds.
    assert rows[0].usage.cpu == pytest.approx(2.0)


def test_machine_scope_distribution_is_labelled_estimated():
    """Bounded estimates from machine-scope data are labelled as estimates."""
    from ore_cros.resource_model import machine_scope

    accountant = TaskUsageAccountant()
    running = [_task("t1"), _task("t2")]
    for task in running:
        accountant.open_task(task, started_at=NOW)
    measurements = [
        _cpu_measurement(2.0, NOW, machine_scope()),
        Measurement(
            resource=ResourceKind.RAM, metric=MetricKind.ALLOCATED,
            value=512.0, unit="MiB", timestamp=NOW, scope=machine_scope(),
            source="test",
        ),
    ]
    created = accountant.attribute_machine_measurements(measurements, running, now=NOW)
    assert created > 0
    for task in running:
        accountant.close_task(task.task_id, ended_at=NOW + 2.0)
    rows = accountant.accounting_table()
    assert all(row.estimated for row in rows)
    # Equal declared demands split the 2-core machine usage evenly: each task
    # holds 1.0 core for the 2-second interval (2 core-seconds integrated).
    assert rows[0].usage.cpu == pytest.approx(2.0)
    assert rows[1].usage.cpu == pytest.approx(2.0)
    # Level metric: peak RAM per task (half of the machine's 512 MiB).
    assert rows[0].usage.ram == pytest.approx(256.0)


def test_unattributed_kinds_are_explicit():
    """Kinds without any samples appear in unattributed_kinds, not as zeros."""
    accountant = TaskUsageAccountant()
    accountant.open_task(_task("t1"), started_at=NOW)
    accountant.record_sample("t1", "m1", ResourceKind.CPU, 0.5, timestamp=NOW)
    accountant.close_task("t1", ended_at=NOW + 1.0)
    row = accountant.accounting_table()[0]
    assert ResourceKind.GPU.value in row.unattributed_kinds
    assert row.usage.gpu == 0.0


def test_module_totals_aggregate_across_tasks():
    """Accounting aggregates usage per module for cost attribution."""
    accountant = TaskUsageAccountant()
    for index in range(3):
        task = _task(f"t{index}", module_id="indexer")
        accountant.open_task(task, started_at=NOW)
        accountant.record_sample(
            f"t{index}", "indexer", ResourceKind.CPU, 1.0, timestamp=NOW,
        )
        accountant.close_task(f"t{index}", ended_at=NOW + 2.0)
    totals = accountant.module_totals()
    assert totals["indexer"].cpu == pytest.approx(6.0)


def test_interval_validation_rejects_backwards_time():
    """Accounting intervals reject unknown tasks and backwards time."""
    accountant = TaskUsageAccountant()
    with pytest.raises(ValueError):
        accountant.close_task("missing", ended_at=NOW)
    accountant.open_task(_task("t1"), started_at=NOW)
    with pytest.raises(ValueError):
        accountant.close_task("t1", ended_at=NOW - 1.0)
    with pytest.raises(ValueError):
        accountant.record_sample("t1", "m1", ResourceKind.CPU, -1.0, timestamp=NOW)
