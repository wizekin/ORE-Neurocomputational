"""Runnable example: ORE Point 11 multi-agent coordination with swarm experiments.

Runs the full MACL loop over a workspace seeded with an input sentence, prints
per-stage results, and persists the complete lineage (annotations plus the
nine-stage operation trace) into an ORE SQLite logger database.

Run from the repository root (for example with the repo on PYTHONPATH):

    PYTHONPATH=. python3 examples/multi_agent_swarm_coordination.py

Note: the repository's automatic script tracer (sitecustomize) makes call-dense
coordination runs substantially slower. For fast interactive runs disable it:

    ORE_AUTO_LOG=0 PYTHONPATH=. python3 examples/multi_agent_swarm_coordination.py
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from ore_macl import (
    MACLRunConfig,
    record_result_to_sqlite,
    register_annotations,
    run_multi_agent_coordination,
)

INPUT_TEXT = "the reporter ignored the scientist and argued about the study"


def main() -> None:
    """Run the nine-stage MACL pipeline and print a compact report."""
    parser = argparse.ArgumentParser(description="ORE Point 11 MACL example")
    parser.add_argument(
        "--database", default="",
        help="optional SQLite path for annotations and the operation trace",
    )
    arguments = parser.parse_args()

    print("=== ORE Point 11: Multi-Agent Cognitive Coordination (MACL) ===")
    print(f"input: {INPUT_TEXT!r}\n")

    result = run_multi_agent_coordination(MACLRunConfig(
        goal_id="example-goal",
        input_text=INPUT_TEXT,
        max_rounds=8,
    ))

    for stage in result.operation_trace:
        summary = stage["functional_output"]
        print(f"[{stage['operation_id']}] {stage['operation']}: {stage['status']}")
        if stage["operation_id"] == "MACL-2":
            for task_id, item in summary["tasks"].items():
                owner = item["owner"] or "deterministic fallback"
                print(f"    {task_id}: {item['outcome']} (owner: {owner})")
            print(f"    rounds={summary['rounds']} stop={summary['stop_reason']!r}")
        elif stage["operation_id"] == "MACL-4":
            print(
                f"    findings={summary['findings_accepted']} "
                f"revision={summary['workspace_revision']} "
                f"lexical_tokens={summary['lexical_tokens']}"
            )
        elif stage["operation_id"] == "MACL-7":
            print(
                f"    goal_outcome={summary['goal_outcome']} "
                f"score={summary['goal_score']:.3f} "
                f"completed_tasks={summary['completed_tasks']}"
            )
        elif stage["operation_id"] == "MACL-8":
            for agent_id, item in summary["reliability"].items():
                low, high = item["credible_interval"]
                print(
                    f"    {agent_id}: mean={item['posterior_mean']:.3f} "
                    f"CI=[{low:.3f}, {high:.3f}] samples={item['samples']}"
                )
        elif stage["operation_id"] == "MACL-9":
            pso = summary["pso"]
            aco = summary["aco"]
            print(
                f"    PSO best_score={pso['best_score']:.4f} "
                f"evaluations={pso['evaluations']} feasible={pso['feasible']}"
            )
            print(
                f"    ACO best_path={aco['best_path']} "
                f"validated={aco['validated_best']} isolated={summary['isolated']}"
            )

    report = result.goal_report
    print(f"\ngoal outcome : {report.goal_evaluation.outcome.value}")
    print(f"stop reason  : {report.stop_reason}")
    print(f"message stats: {report.message_stats['messages_sent']} sent, "
          f"{report.message_stats['by_outcome'].get('delivered', 0)} delivered")
    print(f"elapsed      : {result.elapsed_seconds:.3f}s")

    if arguments.database:
        database_path = Path(arguments.database).expanduser().resolve()
    else:
        database_path = Path(tempfile.gettempdir()) / "ore_macl_example.db"

    catalogue = register_annotations(database_path)
    recorded = record_result_to_sqlite(result, database_path)
    print(f"\nSQLite lineage: {catalogue['symbols_written']} symbols indexed, "
          f"run {recorded['run_id']} with {recorded['database_summary']['events']} "
          f"stage events -> {database_path}")


if __name__ == "__main__":
    main()
