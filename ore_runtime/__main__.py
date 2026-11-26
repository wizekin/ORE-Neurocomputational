"""Run a small synthetic goal through the Point 19 execution engine."""
from __future__ import annotations

import json

from ore_runtime import (
    EngineConfig, ExecutionRequest, TaskSpec, WorkspaceRuntimeAdapter,
    build_workspace_permissions,
)
from ore_runtime.engine import CognitiveExecutionEngine


def main() -> None:
    """Execute one demo goal with a three-task dependency DAG."""
    engine = CognitiveExecutionEngine(EngineConfig(max_workers=3, max_active_tasks=3))
    specs = (
        TaskSpec(task_id="perceive", name="perceive input",
                 capability="generic", priority=0.8, payload={"sentence": "run demo"}),
        TaskSpec(task_id="analyze", name="analyze content",
                 dependencies=("perceive",), capability="generic", priority=0.6),
        TaskSpec(task_id="conclude", name="draw conclusion",
                 dependencies=("analyze",), capability="generic", priority=0.4,
                 payload={"mode": "summary"}),
    )

    def executor(context):
        """Produce one deterministic output derived from the snapshot."""
        return {"step": context.task_spec.task_id, "snapshot_keys": sorted(context.snapshot)}

    report = engine.execute_goal(
        ExecutionRequest(
            goal_id="demo_goal", description="demo Point 19 execution",
            inputs={}, expected_outputs=("conclude",),
        ),
        specs,
        {"perceive": executor, "analyze": executor, "conclude": executor},
    )
    print(json.dumps(report.to_dict(), indent=2, default=str)[:2000])
    print("events:", report.events_emitted)
    print("metrics:", json.dumps(engine.metrics_report()["task_throughput"]))
    engine.close()


if __name__ == "__main__":
    main()
