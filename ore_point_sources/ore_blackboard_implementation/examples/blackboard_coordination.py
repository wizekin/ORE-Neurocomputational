"""Runnable demonstration of the ORE blackboard coordination algorithm (Algorithm 1).

This example runs two sentence-understanding tasks through
``BlackboardCoordinator``: an unambiguous sentence, and an ambiguous sentence
whose lexical polysemy comes from a read-only USG-KNOWLEDGE bundle with
competing episodic memory evidence. It then prints the epistemic result
sections, the ranked alternatives, and the operation-coverage summary, and
finally runs a small benchmark and ablation to show the Step 29-30 harnesses.

Run from the repository root:

    python examples/blackboard_coordination.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("ORE_AUTO_LOG", "0")

from ore_blackboard import (  # noqa: E402
    BLACKBOARD_OPERATION_IDS,
    BlackboardCoordinator,
    BlackboardTaskRequest,
    run_ablation,
    run_benchmark,
)


def _demo_usg_bundle() -> dict:
    """Build a minimal USG-KNOWLEDGE bundle with bank polysemy (Step 5 demo).

    The bundle schema follows ``ore_usg.validation``: records, sources,
    evidence, contexts, mappings, and relations with lifecycle states and
    rights status. This toy bundle is demonstration data, not real lexical
    knowledge.
    """
    timestamp = "2026-11-21T00:00:00Z"
    return {
        "contract": "USG-KNOWLEDGE/0.1.0",
        "namespace": "urn:usg:blackboard-demo",
        "bundle_id": "ore-blackboard-demo",
        "created_at": timestamp,
        "records": [
            {
                "id": "lemma:en:bank", "kind": "LEMMA", "version": 1,
                "state": "ACCEPTED", "created_at": timestamp,
                "source_ids": ["source:permitted"], "evidence_ids": ["evidence:lemma"],
                "context_ids": ["context:english"],
                "payload": {"lemma": "bank", "language": "en", "part_of_speech": "NOUN"},
            },
            {
                "id": "sense:en:bank:financial", "kind": "LEXICAL_SENSE", "version": 1,
                "state": "ACCEPTED", "created_at": timestamp,
                "source_ids": ["source:permitted"], "evidence_ids": ["evidence:financial"],
                "context_ids": ["context:finance"],
                "payload": {
                    "lemma_record_id": "lemma:en:bank", "language": "en",
                    "definition": "A financial institution.",
                    "sense_source_id": "source:permitted",
                },
            },
            {
                "id": "sense:en:bank:river", "kind": "LEXICAL_SENSE", "version": 1,
                "state": "VALIDATED_FOR_PROFILE", "created_at": timestamp,
                "source_ids": ["source:internal"], "evidence_ids": ["evidence:river"],
                "context_ids": ["context:geography"],
                "payload": {
                    "lemma_record_id": "lemma:en:bank", "language": "en",
                    "definition": "The land alongside a river.",
                    "sense_source_id": "source:internal",
                },
            },
        ],
        "sources": [
            {
                "id": "source:permitted", "provider": "Demo fixture",
                "source_type": "OTHER", "retrieved_at": timestamp,
                "rights_status": "KNOWN_PERMITTED", "redistribution": "ALLOWED",
                "attribution": "Demo fixture source",
            },
            {
                "id": "source:internal", "provider": "Demo internal",
                "source_type": "USG_DOCUMENT", "retrieved_at": timestamp,
                "rights_status": "INTERNAL", "redistribution": "NOT_APPLICABLE",
                "attribution": "Demo internal source",
            },
        ],
        "evidence": [
            {
                "id": "evidence:lemma", "source_id": "source:permitted",
                "evidence_type": "DEFINITION", "support_status": "SUPPORTS",
                "captured_at": timestamp, "transformation_ids": [],
            },
            {
                "id": "evidence:financial", "source_id": "source:permitted",
                "evidence_type": "DEFINITION", "support_status": "SUPPORTS",
                "captured_at": timestamp, "transformation_ids": [],
            },
            {
                "id": "evidence:river", "source_id": "source:internal",
                "evidence_type": "DEFINITION", "support_status": "SUPPORTS",
                "captured_at": timestamp, "transformation_ids": [],
            },
        ],
        "contexts": [
            {"id": "context:english", "context_type": "LANGUAGE", "value": "en"},
            {"id": "context:finance", "context_type": "DOMAIN", "value": "finance"},
            {"id": "context:geography", "context_type": "DOMAIN", "value": "geography"},
        ],
        "mappings": [],
        "relations": [],
    }


def _print_result(result) -> None:
    """Print one blackboard run result with its epistemic sections."""
    print(f"  status:            {result.status}")
    print(f"  reason:            {result.status_reason}")
    print(f"  goal score:        {result.goal_score:.3f} (threshold {result.goal_threshold})")
    print(f"  rounds/calls:      {result.rounds}/{result.module_calls}")
    print(f"  findings:          accepted={result.accepted_finding_count}, "
          f"pending={result.pending_finding_count}, "
          f"rejected={result.rejected_finding_count}, "
          f"conflicts={result.conflict_count}")
    print(f"  trace completeness: {result.trace_completeness:.3f} "
          f"(integrity ok: {result.integrity_ok})")
    for observation in result.observations:
        print(f"  observation:       {observation['observation_type']}"
              + (f" [{observation.get('token')}]" if observation.get("token") else ""))
    for reference in result.imported_knowledge_references[:3]:
        print(f"  imported:          {reference['reference_type']} "
              f"{reference.get('bundle_id', reference.get('record_id', ''))[:44]}")
    for hypothesis in result.supported_hypotheses:
        print(f"  hypothesis:        E(h)={hypothesis.score:.3f} "
              f"uncalibrated | {hypothesis.label[:56]}")
    for alternative in result.unresolved_alternatives[:3]:
        kind = alternative.get("conflict_type") or alternative.get("alternative_type")
        print(f"  unresolved:        {kind}")
    for conclusion in result.verified_conclusions:
        print(f"  verified:          {conclusion.get('question_id')}: "
              f"{conclusion.get('status')}")
    for action in result.recommended_next_actions[:2]:
        print(f"  next action:       {action[:70]}")
    print(f"  operations covered: {len(result.operation_coverage)}/32 "
          f"(runtime; steps 29-30 in the evaluation harness)")


def main() -> None:
    """Run the demonstration tasks, benchmark, and ablation."""
    coordinator = BlackboardCoordinator()

    print("=== Task 1: unambiguous sentence (built-in fallback lexicon) ===")
    plain = BlackboardTaskRequest(
        input="The boy opened the door.",
        goal="understand_sentence",
        task_id="example-plain",
    )
    _print_result(coordinator.run(plain))

    print()
    print("=== Task 2: USG polysemy with competing memory evidence ===")
    ambiguous = BlackboardTaskRequest(
        input="The captain saw the bank.",
        goal="understand_sentence",
        task_id="example-ambiguous",
        usg_bundle=_demo_usg_bundle(),
        memory_store=(
            {"memory_id": "m1", "text": "The fisherman moored his boat by the river bank.",
             "keywords": ("river", "land", "water")},
            {"memory_id": "m2", "text": "The captain deposited the money at the bank.",
             "keywords": ("financial", "money")},
        ),
    )
    _print_result(coordinator.run(ambiguous))

    print()
    print("=== Idempotent replay of Task 1 (same task identity) ===")
    replay = coordinator.run(plain)
    print(f"  idempotent replay: {replay.idempotent_replay} "
          f"(workspace {replay.workspace_id[:20]}...)")

    print()
    print("=== Step 29: benchmark against a sequential baseline ===")
    report, _blackboard, _baseline = run_benchmark(
        lambda: BlackboardCoordinator(), (plain, ambiguous), iterations=200,
    )
    print(f"  blackboard: success={report.blackboard['success_rate']:.2f}, "
          f"calls={report.blackboard['mean_module_calls']:.1f}, "
          f"unnecessary={report.blackboard['unnecessary_activation_rate']:.2f}")
    print(f"  baseline:   success={report.baseline['success_rate']:.2f}, "
          f"calls={report.baseline['mean_module_calls']:.1f}, "
          f"unnecessary={report.baseline['unnecessary_activation_rate']:.2f}")
    mean, low, high = report.paired_differences["latency_ms"]
    print(f"  paired latency difference: {mean:.1f} ms "
          f"(bootstrap CI {low:.1f}..{high:.1f})")

    print()
    print("=== Step 30: single-mechanism ablation of the ambiguous task ===")

    from ore_blackboard import AblationSwitches, BlackboardPolicy

    ablation = run_ablation(
        lambda policy: BlackboardCoordinator(policy=policy),
        ambiguous,
        base_policy=BlackboardPolicy(
            ablations=AblationSwitches(),
        ),
    )
    for mechanism in (
        "selective_subscriptions", "usg_retrieval", "dependency_deferral",
    ):
        ablated = ablation.ablations[mechanism]
        print(f"  {mechanism:24s} calls={ablated['module_calls']:.0f}, "
              f"conflicts={ablated['conflict_count']:.0f}, "
              f"usg={ablated['usg_contribution']:.0f}, "
              f"provenance={ablated['provenance_completeness']:.2f}")

    print()
    print(f"Operation registry: {len(BLACKBOARD_OPERATION_IDS)} stable IDs "
          "(BBA-001..BBA-032); see docs/BLACKBOARD_COORDINATION_IMPLEMENTATION.md.")
    print("This demonstration uses reference toy specialists; it validates the")
    print("coordination algorithm, not real language-understanding quality.")


if __name__ == "__main__":
    main()
