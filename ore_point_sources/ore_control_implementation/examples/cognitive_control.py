"""End-to-end demonstration of the ORE cognitive control layer (Point 02).

The demo exercises the 40-step algorithm on three triggers: an ambiguous
sentence with a USG bundle (decomposition, retrieval competition, preserved
uncertainty), a plain sentence, and repeated runs that promote a reusable
procedure through the Phase H gates. It then runs the paired benchmark
against the sequential module-coordination baseline.

The demonstration uses deterministic reference operators and a toy learned
ranker: it validates the control algorithm and its traceability, not
real language-understanding quality.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ore_blackboard import BlackboardCoordinator  # noqa: E402

from ore_control import (  # noqa: E402
    CognitiveControlCoordinator,
    TriggerEnvelope,
    TriggerKind,
    run_benchmark,
    paired_bootstrap_latency,
)
from ore_control.registry import CONTROL_OPERATIONS  # noqa: E402


def _demo_usg_bundle() -> dict:
    """Reuse the blackboard demo bundle (bank polysemy)."""
    from blackboard_coordination import _demo_usg_bundle as bundle

    return bundle()


def main() -> None:
    """Run the cognitive-control demonstration end to end."""
    print("=== ORE cognitive control layer (Point 02) ===")

    coordinator = CognitiveControlCoordinator(
        blackboard=BlackboardCoordinator(),
        usg_bundle=_demo_usg_bundle(),
    )

    print("\n--- Task 1: ambiguous sentence with USG knowledge ---")
    result = coordinator.process_trigger(TriggerEnvelope(
        kind=TriggerKind.USER_TASK,
        idempotency_key="demo_ambiguous",
        event_reference="evt:demo_ambiguous",
        payload={"text": "The fisherman sat near the bank."},
    ))
    print(f"  status:            {result.status} ({result.goal_outcome})")
    print(f"  cycles:            {result.cycles}")
    print(f"  operators applied:{result.operator_applications}")
    print(f"  child contexts:    {result.child_contexts}")
    print(f"  retrievals:        {result.retrievals} "
          f"(cost {result.retrieval_cost:.2f})")
    print(f"  learned proposals: {result.learned_proposals_used}")
    print(f"  uncertainty kept:  {list(result.uncertainty_reasons)}")
    for trace in result.decision_traces:
        selection = trace["selection"]
        selected = selection["selected_operator_id"] if selection else "(none)"
        print(f"  cycle {trace['cycle']}: selected {selected}; "
              f"{trace['next_cycle_reason'][:58]}")

    print("\n--- Task 2: plain sentence (no ambiguity) + procedure promotion ---")
    result_two = coordinator.process_trigger(TriggerEnvelope(
        kind=TriggerKind.USER_TASK,
        idempotency_key="demo_plain",
        event_reference="evt:demo_plain",
        payload={"text": "A boy opened the door."},
    ))
    print(f"  status: {result_two.status} ({result_two.goal_outcome})")
    for event in result_two.procedure_events:
        print(f"  procedure event:  {event['event']}")

    print("\n--- Task 3: idempotent replay and procedure reuse ---")
    replay = coordinator.process_trigger(TriggerEnvelope(
        kind=TriggerKind.USER_TASK,
        idempotency_key="demo_plain",
        event_reference="evt:demo_plain",
        payload={"text": "A boy opened the door."},
    ))
    print(f"  idempotent replay: {replay.run_id == result_two.run_id}")

    print("\n--- Task 4: third distinct verified run ---")
    result_three = coordinator.process_trigger(TriggerEnvelope(
        kind=TriggerKind.USER_TASK,
        idempotency_key="demo_third",
        event_reference="evt:demo_third",
        payload={"text": "The captain saw the boat."},
    ))
    for event in result_three.procedure_events:
        print(f"  procedure event:  {event['event']}")
    procedures = coordinator.procedure_registry.records()
    if procedures:
        record = procedures[0]
        print(f"  procedure:        {record.procedure_id} v{record.version} "
              f"(non-authoritative: {not record.to_dict()['authoritative']})")
        print(f"  registered steps: {[s.operator_id for s in record.operator_sequence]}")

    print("\n--- Step §7: paired benchmark vs sequential baseline ---")
    report = run_benchmark(
        coordinator,
        ("The captain saw the boat.", "A boy opened the door."),
        baseline=BlackboardCoordinator(),
        task_labels=("bench_1", "bench_2"),
    )
    summary = report.summary()
    control = summary["control"]
    baseline = summary["baseline"]
    print(f"  control:   success={control['success_rate']:.2f}, "
          f"ops={control['mean_operator_applications']:.1f}, "
          f"latency={control['mean_latency_ms']:.1f} ms")
    print(f"  baseline: success={baseline['success_rate']:.2f}, "
          f"module_calls={baseline['mean_module_calls']:.1f}, "
          f"latency={baseline['mean_latency_ms']:.1f} ms")
    mean, low, high = paired_bootstrap_latency(report, iterations=500)
    print(f"  paired latency difference: {mean:.1f} ms (bootstrap CI {low:.1f}..{high:.1f})")
    print("  (harness statistic only; no improvement claim from a single demonstration)")

    covered = {
        step["step"]
        for trace in result.decision_traces + result_two.decision_traces
        + result_three.decision_traces
        for step in trace["operation_steps"]
    }
    print(f"\n  operations covered: {len(covered)}/40 (runtime; procedure steps "
          f"in Phase H events)")

    print(f"\nOperation registry: {len(CONTROL_OPERATIONS)} stable IDs "
          f"(CCA-001..CCA-040); see docs/COGNITIVE_CONTROL_IMPLEMENTATION.md.")
    print("This demonstration uses deterministic reference operators; it "
          "validates the\ncontrol algorithm, not real language-understanding quality.")


if __name__ == "__main__":
    main()
