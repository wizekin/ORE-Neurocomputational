"""Synthetic demonstration of the Point 15 agentic workflow and memory layer.

Run with ``python -m ore_agentic`` from the repository root.  The demo builds
a goal with three independent subgoals and one reduce join, registers two
deterministic local workers (one with a transient fault that succeeds on
retry), executes the 50-step pipeline with a temporary SQLite store, and
prints the typed result envelope summary including the operation trace, the
distinct outcome categories and the recorded memory artifacts.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

from ore_agentic import (
    AgenticConfig,
    AgenticEngine,
    DeliverableSpec,
    GoalRequest,
    LocalWorker,
    ResourceVector,
    SubgoalSpec,
    WorkerManifest,
)
from ore_agentic.registry.operations import registry_as_dicts

__all__ = ("main", "build_demo_request", "register_demo_workers")


def build_demo_request() -> GoalRequest:
    """Build a demo goal with three map subgoals and a reduce join."""
    return GoalRequest(
        objective="Summarize evidence about attention and publish a brief.",
        requester="demo-operator",
        subgoals=(
            SubgoalSpec(name="collect", capability="demo.collect",
                        description="Collect source evidence",
                        inputs={"value": "collected evidence"}),
            SubgoalSpec(name="analyze", capability="demo.analyze",
                        description="Analyze the collected evidence",
                        inputs={"value": "analysis"}, depends_on=("collect",)),
            SubgoalSpec(name="brief", capability="demo.brief",
                        description="Draft the brief",
                        inputs={"value": "brief draft"}, depends_on=("analyze",)),
        ),
        deliverables=(DeliverableSpec(name="brief"),
                      DeliverableSpec(name="analysis")),
        constraints={"intent": "brief"},
        priority=0.6,
        idempotency_key="demo-run-001",
    )


def register_demo_workers(engine: AgenticEngine, *, flaky: bool = True) -> None:
    """Register two deterministic demo workers, one with a transient fault."""
    state = {"collect_attempts": 0}

    def _collect(envelope):  # type: ignore[no-untyped-def]
        if flaky:
            state["collect_attempts"] += 1
            if state["collect_attempts"] == 1:
                from ore_agentic.delegation.workers import TransientTaskError
                raise TransientTaskError("demo transient store hiccup")
        return {"value": envelope.inputs["payload"].get("value", "collected"),
                "evidence": ["demo-source-1"]}

    def _generic(envelope):  # type: ignore[no-untyped-def]
        return {"value": envelope.inputs["payload"].get("value", "result"),
                "evidence": ["demo-source-2"]}

    engine.register_worker(
        WorkerManifest(worker_id="demo-worker-1", version="1.0.0",
                       capabilities=("demo.collect", "demo.analyze", "demo.brief"),
                       resource_profile=ResourceVector(seconds=2.0),
                       reliability=0.95, latency_seconds=0.2, cost_per_task=0.5),
        LocalWorker("demo-worker-1", "1.0.0",
                    ("demo.collect", "demo.analyze", "demo.brief"), _collect))
    engine.register_worker(
        WorkerManifest(worker_id="demo-worker-2", version="1.1.0",
                       capabilities=("demo.analyze", "demo.brief"),
                       resource_profile=ResourceVector(seconds=1.0),
                       reliability=0.90, latency_seconds=0.1, cost_per_task=0.3),
        LocalWorker("demo-worker-2", "1.1.0", ("demo.analyze", "demo.brief"),
                    _generic))


def main(argv: list[str] | None = None) -> int:
    """Run the Point 15 demonstration end to end."""
    parser = argparse.ArgumentParser(description="Point 15 agentic demonstration")
    parser.add_argument("--no-flaky", action="store_true",
                        help="disable the injected transient fault")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="ore-agentic-demo-") as tmp:
        engine = AgenticEngine(
            AgenticConfig(retry_initial_delay_seconds=0.01,
                          retry_max_delay_seconds=0.05),
            database_path=str(Path(tmp) / "agentic.sqlite3"))
        register_demo_workers(engine, flaky=not args.no_flaky)
        result = engine.run(build_demo_request())

        print("=== Point 15 Agentic Workflow and Memory Layer ===")
        print(f"workflow_id: {result.workflow_id}")
        print(f"status:      {result.status.value}")
        print(f"outcome:     {result.outcome_status.value}")
        print(f"pattern:     {result.pattern.value}  plan: {result.plan_hash}")
        print(f"duration:    {result.duration_ms:.1f} ms")
        print(f"risk:        score={result.risk.score:.3f} "
              f"decision={result.risk.decision.value}")
        print("outcomes:")
        print(f"  completed:  {list(result.outcomes.completed_task_ids)}")
        print(f"  failed:     {list(result.outcomes.failed_task_ids)}")
        print(f"  cancelled:  {list(result.outcomes.cancelled_task_ids)}")
        print(f"  unverified: {list(result.outcomes.unverified_task_ids)}")
        print("deliverables:")
        for task_id, payload in result.deliverables.items():
            print(f"  {task_id}: {payload}")
        print(f"trace steps recorded: {len(result.trace)} "
              f"(expected 50 + fenced/deferral annotations)")
        phases: dict[str, int] = {}
        for entry in result.trace:
            phases[entry["phase"]] = phases.get(entry["phase"], 0) + 1
        print(f"trace by phase: {phases}")
        print(f"episodes recorded: {len(result.episode_ids)}")
        print(f"procedure proposals (draft, test-gated): "
              f"{len(result.procedure_proposal_ids)}")
        print(f"semantic proposals (candidate, non-authoritative): "
              f"{len(result.semantic_proposal_ids)}")
        print(f"verification: "
              f"{'passed' if result.verification and result.verification.passed else 'failed'}"
              f" ({result.verification.detail if result.verification else ''})")
        print("limitations:")
        for limitation in result.limitations:
            print(f"  - {limitation}")
        print(f"database: {Path(tmp) / 'agentic.sqlite3'} "
              f"(temporary; workflow, tasks, results, episodes persisted)")
    print("operation registry: "
          f"{len(registry_as_dicts())} source steps defined (P15-01..P15-50)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
