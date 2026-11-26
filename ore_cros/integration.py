"""Integration boundaries with existing ORE components (ORE Point 23, section 6).

CROS works *with* existing ORE components rather than duplicating them:
AdaptiveResourceCoordinator stays the central policy authority for module
selection, CognitiveTaskCoordinator publishes task demand, SharedWorkspace
is not a telemetry database, ore_usg facts are never mutated by resource
decisions, and ore_logger stores decision provenance with bounded
retention. All imports of sibling ORE packages are optional: CROS runs
standalone when they are absent.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence

from .actions import ActionType, ResourceAction
from .mape.plan import DecisionRecord
from .resource_model import TaskRecord, WorkloadClass

#: Responsibility map from the specification (section 6).
INTEGRATION_BOUNDARIES: tuple[dict[str, str], ...] = (
    {
        "component": "AdaptiveResourceCoordinator",
        "responsibility": "global resource policy, budgets and allocation",
        "boundary": "central policy authority; does not implement OS mechanisms",
    },
    {
        "component": "CognitiveTaskCoordinator",
        "responsibility": "goal decomposition and task state",
        "boundary": "publishes task demand, deadlines and dependencies",
    },
    {
        "component": "ModuleScheduler",
        "responsibility": "eligible modules and workers",
        "boundary": "applies capability, priority and execution constraints",
    },
    {
        "component": "SharedWorkspace",
        "responsibility": "cognitive state and findings",
        "boundary": "not a general telemetry database",
    },
    {
        "component": "ore_usg",
        "responsibility": "semantic knowledge and evidence",
        "boundary": "supplies context; resource decisions cannot mutate USG facts",
    },
    {
        "component": "ore_logger",
        "responsibility": "historical decisions and annotations",
        "boundary": "stores provenance and outcomes with bounded retention",
    },
    {
        "component": "CROS resource-state model",
        "responsibility": "CPU, memory, I/O, network, accelerators, queues, topology",
        "boundary": "versioned resource state and accounting",
    },
    {
        "component": "Resource execution manager",
        "responsibility": "applies approved changes",
        "boundary": "narrow permissions, validation, rollback and audit",
    },
    {
        "component": "Policy engine",
        "responsibility": "deterministic, predictive and learned selection",
        "boundary": "cannot override hard constraints",
    },
    {
        "component": "Safety/verification layer",
        "responsibility": "budget, integrity, isolation and fallback checks",
        "boundary": "independent validation before and after execution",
    },
)


class OreWorkloadClassMapper:
    """Policy-controlled mapping from ORE metadata to CROS workload classes.

    Class assignment is owned by this mapper (the policy authority); a
    task cannot promote its own criticality. Recognized classes come from
    task metadata first, then module-id conventions, then a conservative
    default of CORE_COGNITIVE.
    """

    _MODULE_HINTS: tuple[tuple[str, WorkloadClass], ...] = (
        ("workspace", WorkloadClass.INTEGRITY_CRITICAL),
        ("verify", WorkloadClass.INTEGRITY_CRITICAL),
        ("safety", WorkloadClass.INTEGRITY_CRITICAL),
        ("interactive", WorkloadClass.INTERACTIVE),
        ("dialogue", WorkloadClass.INTERACTIVE),
        ("index", WorkloadClass.BATCH),
        ("bulk", WorkloadClass.BATCH),
        ("conversion", WorkloadClass.BATCH),
        ("explore", WorkloadClass.EXPLORATORY),
        ("hypothesis", WorkloadClass.EXPLORATORY),
        ("maintenance", WorkloadClass.MAINTENANCE),
        ("compaction", WorkloadClass.MAINTENANCE),
        ("backup", WorkloadClass.MAINTENANCE),
    )

    def classify(
        self, *, module_id: str, metadata: Mapping[str, Any] | None = None,
    ) -> WorkloadClass:
        """Return the workload class for one ORE module invocation."""
        metadata = metadata or {}
        declared = metadata.get("workload_class")
        if isinstance(declared, str):
            try:
                return WorkloadClass(declared)
            except ValueError:
                pass
        lowered = module_id.lower()
        for hint, workload_class in self._MODULE_HINTS:
            if hint in lowered:
                return workload_class
        return WorkloadClass.CORE_COGNITIVE


def cros_task_from_metadata(
    *,
    task_id: str,
    module_id: str,
    mapper: OreWorkloadClassMapper | None = None,
    metadata: Mapping[str, Any] | None = None,
    remaining_work: float = 1.0,
    ready_time: float = 0.0,
    deadline: float | None = None,
    dependencies: Sequence[str] = (),
) -> TaskRecord:
    """Build a CROS TaskRecord from ORE task metadata (section 6 boundary).

    ``CognitiveTaskCoordinator`` publishes task demand, deadlines and
    dependencies; this adapter projects them into the CROS task model.
    """
    from .resource_model import ResourceDemand

    metadata = dict(metadata or {})
    mapper = mapper or OreWorkloadClassMapper()
    demand = ResourceDemand(
        cpu=float(metadata.get("demand_cpu", 0.1)),
        ram=float(metadata.get("demand_ram", 0.0)),
        io=float(metadata.get("demand_io", 0.0)),
        net=float(metadata.get("demand_net", 0.0)),
        gpu=float(metadata.get("demand_gpu", 0.0)),
        storage=float(metadata.get("demand_storage", 0.0)),
    )
    return TaskRecord(
        task_id=task_id, parent_goal_id=metadata.get("parent_goal_id"),
        module_id=module_id,
        workload_class=mapper.classify(module_id=module_id, metadata=metadata),
        demand=demand, remaining_work=float(remaining_work),
        ready_time=float(ready_time), deadline=deadline,
        dependencies=tuple(dependencies),
    )


class SqliteDecisionStore:
    """Persist CROS decision provenance in an ore_logger-style database.

    Decision records are stored in their own ``cros_decisions`` table with
    bounded retention, independent of the cognitive annotations ore_logger
    already manages.
    """

    _MAX_ROWS = 4096

    def __init__(self, database_path: str | Path) -> None:
        """Open (or create) the decision store inside a SQLite file."""
        self.path = Path(database_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS cros_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at TEXT NOT NULL,
                    state_version INTEGER NOT NULL,
                    policy_version TEXT NOT NULL,
                    action_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    policy_origin TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    constraints TEXT NOT NULL,
                    predicted_outcome TEXT NOT NULL,
                    measured_outcome TEXT
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with a bounded busy timeout."""
        connection = sqlite3.connect(str(self.path), timeout=20.0)
        connection.execute("PRAGMA busy_timeout=20000")
        return connection

    def store(self, record: DecisionRecord) -> None:
        """Append one decision record with bounded retention."""
        import json

        with self._connect() as connection:
            connection.execute(
                """INSERT INTO cros_decisions (
                       recorded_at, state_version, policy_version, action_id,
                       action_type, policy_origin, rationale, constraints,
                       predicted_outcome, measured_outcome
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(record.timestamp), record.state_version,
                    record.policy_version, record.action_id, record.action_type,
                    record.policy_origin, record.rationale,
                    ",".join(record.constraints_checked),
                    json.dumps(dict(record.predicted_outcome)),
                    json.dumps(dict(record.measured_outcome))
                    if record.measured_outcome else None,
                ),
            )
            connection.execute(
                "DELETE FROM cros_decisions WHERE id <= "
                "(SELECT COALESCE(MAX(id), 0) FROM cros_decisions) - ?",
                (self._MAX_ROWS,),
            )

    def count(self) -> int:
        """Return the number of stored decision records."""
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM cros_decisions").fetchone()
        return int(row[0])


def describe_boundaries() -> tuple[dict[str, str], ...]:
    """Return the section-6 responsibility map as structured data."""
    return tuple(dict(item) for item in INTEGRATION_BOUNDARIES)


__all__ = [
    "OreWorkloadClassMapper", "cros_task_from_metadata", "SqliteDecisionStore",
    "describe_boundaries", "INTEGRATION_BOUNDARIES",
]
