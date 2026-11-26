"""Runnable demonstration of the ORE neuro-symbolic reasoning layer (Point 3).

This example runs the full 45-step algorithm through
``NeuroSymbolicReasoningCoordinator`` over three synthetic tasks:

1. a multi-hop strict derivation with independent proof verification and
   workspace publication;
2. a defeasible task with an exception (general rule defeated by the specific
   exception) plus preserved ambiguity over competing entity readings from a
   read-only USG-KNOWLEDGE bundle;
3. a contradictory task whose derived conclusion clashes with an explicit
   stored fact, so the conflict is recorded instead of overwriting anything.

It then prints the epistemic result sections, the proof classification, the
operation-coverage summary, and finally runs a small benchmark and ablation to
show the Section 10 harnesses. No learned model, GPU or optional framework is
required: the deterministic core carries the whole run.

Run from the repository root:

    python examples/neurosymbolic_reasoning.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("ORE_AUTO_LOG", "0")

from ore_neurosymbolic import (  # noqa: E402
    EpistemicStatus,
    EvidenceItem,
    IndependenceDeclaration,
    InferenceMode,
    LearnedCandidateAdapter,
    Literal,
    LogicalTerm,
    LogicalVariable,
    ModelDescriptor,
    NSRL_OPERATION_IDS,
    NeuroSymbolicReasoningCoordinator,
    ProbabilisticEvidenceAdapter,
    ProofClass,
    PublicationState,
    ReasoningRequest,
    RuleClass,
    RuleDefinition,
    SemanticFact,
    VerificationLevel,
    evaluate_corpus,
    run_ablation,
)


X = LogicalVariable("X", "entity")
Y = LogicalVariable("Y", "entity")
Z = LogicalVariable("Z", "entity")
BIRD = LogicalTerm("bird", "class")
PENGUIN = LogicalTerm("penguin", "class")
FLY = LogicalTerm("fly", "activity")


def _fact(fact_id: str, subject: str, predicate: str, object_value: str, **kw) -> SemanticFact:
    """Build one typed fact with an explicit epistemic status."""
    return SemanticFact(
        fact_id=fact_id, subject=subject, predicate=predicate,
        object_value=object_value,
        epistemic_status=kw.pop("epistemic_status", EpistemicStatus.ASSERTED),
        **kw,
    )


def _demo_usg_bundle() -> dict:
    """Build a minimal read-only USG-KNOWLEDGE bundle for entity ambiguity."""
    timestamp = "2026-11-22T00:00:00Z"
    return {
        "contract": "USG-KNOWLEDGE/0.1.0",
        "namespace": "urn:usg:nsrl-demo",
        "bundle_id": "ore-nsrl-demo",
        "created_at": timestamp,
        "records": [
            {
                "id": "lemma:en:bank", "kind": "LEMMA", "version": 1,
                "state": "ACCEPTED", "created_at": timestamp,
                "source_ids": ["source:permitted"],
                "evidence_ids": ["evidence:lemma"], "context_ids": ["context:english"],
                "payload": {"lemma": "bank", "language": "en", "part_of_speech": "NOUN"},
            },
            {
                "id": "concept:bank_financial", "kind": "CONCEPT", "version": 1,
                "state": "ACCEPTED", "created_at": timestamp,
                "source_ids": ["source:permitted"],
                "evidence_ids": ["evidence:lemma"], "context_ids": ["context:english"],
                "payload": {"label": "bank", "language": "en",
                            "concept_type": "ENTITY", "namespace": "urn:usg:nsrl-demo"},
            },
            {
                "id": "concept:bank_river", "kind": "CONCEPT", "version": 1,
                "state": "ACCEPTED", "created_at": timestamp,
                "source_ids": ["source:permitted"],
                "evidence_ids": ["evidence:lemma"], "context_ids": ["context:english"],
                "payload": {"label": "bank", "language": "en",
                            "concept_type": "ENTITY", "namespace": "urn:usg:nsrl-demo"},
            },
        ],
        "sources": [{
            "id": "source:permitted", "provider": "demo", "source_type": "OTHER",
            "retrieved_at": timestamp, "rights_status": "KNOWN_PERMITTED",
            "redistribution": "ALLOWED", "attribution": "ORE demo corpus",
        }],
        "evidence": [{
            "id": "evidence:lemma", "source_id": "source:permitted",
            "evidence_type": "ATTESTATION", "support_status": "SUPPORTS",
            "locator": "demo://bank", "text_digest": "demo",
            "captured_at": timestamp, "transformation_ids": [],
        }],
        "contexts": [{
            "id": "context:english", "context_type": "LANGUAGE", "value": "en",
        }],
        "mappings": [
            {
                "id": "map:bank:financial", "source_record_id": "lemma:en:bank",
                "target_record_id": "concept:bank_financial", "mapping_type": "RELATED_TO",
                "state": "ACCEPTED", "created_at": timestamp,
                "evidence_ids": ["evidence:lemma"], "context_ids": ["context:english"],
            },
            {
                "id": "map:bank:river", "source_record_id": "lemma:en:bank",
                "target_record_id": "concept:bank_river", "mapping_type": "RELATED_TO",
                "state": "ACCEPTED", "created_at": timestamp,
                "evidence_ids": ["evidence:lemma"], "context_ids": ["context:english"],
            },
        ],
        "relations": [],
    }


def _superior_rule() -> RuleDefinition:
    """Strict multi-hop rule: commanded_by(X,Y) and commands(Y,Z) -> indirect_command."""
    return RuleDefinition(
        rule_id="r_superior", version="1.0.0",
        typed_variables={"?X": "entity", "?Y": "entity", "?Z": "entity"},
        premises=(Literal("commanded_by", (X, Y)), Literal("commands", (Y, Z))),
        conclusion=Literal("indirect_command", (X, Z)),
        rule_class=RuleClass.STRICT, priority=1.0,
        provenance={"source": "Point 3 demo ruleset"},
    )


def _bird_rules() -> tuple[RuleDefinition, RuleDefinition]:
    """Defeasible bird-fly rule with an explicit penguin exception."""
    fly_rule = RuleDefinition(
        rule_id="r_bird_flies", version="1.0.0", typed_variables={"?X": "entity"},
        premises=(Literal("is_a", (X, BIRD)),),
        conclusion=Literal("ability", (X, FLY)),
        rule_class=RuleClass.DEFEASIBLE, priority=0.5,
        scope={"domain": "zoology", "rule_strength": 0.8},
        exceptions=(Literal("is_a", (X, PENGUIN)),),
        provenance={"source": "Point 3 demo ruleset"},
    )
    flightless_rule = RuleDefinition(
        rule_id="r_penguin_flightless", version="1.0.0", typed_variables={"?X": "entity"},
        premises=(Literal("is_a", (X, PENGUIN)),),
        conclusion=Literal("inability", (X, FLY)),
        rule_class=RuleClass.STRICT, priority=1.0,
        scope={"domain": "zoology"},
        provenance={"source": "Point 3 demo ruleset"},
    )
    return fly_rule, flightless_rule


def _print_result(title: str, result) -> None:
    """Print one run's epistemic sections without collapsing categories."""
    print(f"\n=== {title} ===")
    print(f"status: {result.status} ({result.status_reason})")
    print(f"operation coverage: {len(result.operation_coverage)}/45 steps")
    for finding in result.conclusions:
        print(
            f"  conclusion {finding.conclusion.signature()}"
            f"  status={finding.epistemic_status.value}"
            f"  proof={finding.proof_class.value}"
            f"  publication={finding.publication_state.value}"
        )
    for conflict in result.conflicts:
        print(f"  conflict {conflict.conflict_id}: {conflict.claims[0]['claim']} vs "
              f"{conflict.claims[1]['claim']} [{conflict.status}]")
    for group in result.ambiguity_groups:
        print(f"  ambiguity preserved: {group['member_ids']}")
    for support in result.probabilistic_support:
        print(f"  probabilistic support: posterior={support.get('posterior')} "
              f"semantics={support.get('semantics')}")
    print(f"  proof nodes: {result.proof_summary.get('node_count')}, "
          f"verified: {result.proof_summary.get('verified_nodes')}")
    print(f"  workspace revision: {result.workspace_revision}, "
          f"trace completeness: {result.trace_completeness:.2f}")


def demo_strict_derivation() -> None:
    """Task 1: multi-hop strict derivation, verified and published (Phases A-C,G)."""
    request = ReasoningRequest(
        goal_predicate="indirect_command",
        goal_args=("captain_harris", "fleet_7"),
        facts=(
            _fact("f1", "captain_harris", "commanded_by", "admiral_reyes"),
            _fact("f2", "admiral_reyes", "commands", "fleet_7"),
        ),
        rules=(_superior_rule(),),
        entity_references=("captain_harris", "fleet_7"),
        inference_mode=InferenceMode.HYBRID,
        verification_level=VerificationLevel.STRICT,
        task_id="demo_strict",
    )
    result = NeuroSymbolicReasoningCoordinator().run(request)
    _print_result("Task 1: verified multi-hop derivation", result)


def demo_defeasible_and_ambiguity() -> None:
    """Task 2: exception-precedence plus preserved USG ambiguity (Phases B,D)."""
    fly_rule, flightless_rule = _bird_rules()
    request = ReasoningRequest(
        goal_predicate="inability",
        goal_args=("tweety", "fly"),
        facts=(
            _fact("f1", "tweety", "is_a", "bird"),
            _fact("f2", "tweety", "is_a", "penguin"),
        ),
        rules=(fly_rule, flightless_rule),
        entity_references=("bank",),
        usg_bundle=_demo_usg_bundle(),
        inference_mode=InferenceMode.FORWARD,
        verification_level=VerificationLevel.BASIC,
        task_id="demo_defeasible",
    )
    result = NeuroSymbolicReasoningCoordinator().run(request)
    _print_result("Task 2: defeasible exception + preserved ambiguity", result)


def demo_conflict_and_probabilistic() -> None:
    """Task 3: recorded contradiction plus declared-dependence evidence (E,G).

    The probabilistic witnesses stay qualitative on purpose: their shared
    source is declared dependent and no joint likelihood is supplied, so the
    adapter refuses to multiply by default (Section 6.4) and reports
    qualitative support instead of fabricated precision.
    """
    adapter = ProbabilisticEvidenceAdapter()
    witness_evidence = tuple(
        adapter.wrap_model_evidence(
            source=f"witness_{index}",
            observation={"probability": 0.8, "hypothesis": "status(ship,active)",
                         "prior": 0.5},
            reliability=0.9,
            model_id="status_classifier", model_version="1.0.0",
            input_hash="demo-input-hash",
            score_semantics="calibrated_probability",
            calibration={"method": "temperature", "ece": 0.03},
        )
        for index in range(2)
    )
    request = ReasoningRequest(
        goal_predicate="indirect_command",
        goal_args=("captain_harris", "fleet_7"),
        facts=(
            _fact("f1", "captain_harris", "commanded_by", "admiral_reyes"),
            _fact("f2", "admiral_reyes", "commands", "fleet_7"),
            # Explicit stored contradiction: the derived triple asserted false.
            _fact("f3", "captain_harris", "indirect_command", "fleet_7",
                  polarity=False),
        ),
        rules=(_superior_rule(),),
        evidence_items=witness_evidence,
        inference_mode=InferenceMode.FORWARD,
        verification_level=VerificationLevel.STRICT,
        task_id="demo_conflict",
    )
    result = NeuroSymbolicReasoningCoordinator().run(request)
    _print_result("Task 3: preserved conflict + probabilistic support", result)


def demo_learned_assistance() -> None:
    """Task 4: replaceable learned model proposes, symbolic layer disposes (F)."""

    class DemoLinker:
        """Toy model backend: returns calibrated candidates for 'bank'."""

        def invoke(self, capability: str, payload) -> dict:
            return {
                "output_schema": "NSRL.LINK/1.0.0",
                "model_version": "1.0.0",
                "score_semantics": "calibrated_probability",
                "candidates": [
                    {
                        "candidate_id": f"proposal_{concept}",
                        "semantic_structure": {"reference": "bank",
                                               "concept_id": concept},
                        "neural_scores": {"similarity": score},
                    }
                    for concept, score in (
                        ("concept:bank_river", 0.82), ("concept:bank_financial", 0.79),
                    )
                ],
            }

    learned = LearnedCandidateAdapter()
    learned.register(
        ModelDescriptor(
            model_id="demo_linker", model_version="1.0.0",
            capability="entity_linking", output_schema="NSRL.LINK/1.0.0",
            score_semantics="calibrated_probability",
            quality=0.6, cost_per_call=0.4,
            limitations=("toy lexical model", "no real embeddings"),
        ),
        DemoLinker(),
    )
    fly_rule, _flightless = _bird_rules()
    request = ReasoningRequest(
        goal_predicate="ability",
        goal_args=("tweety", "fly"),
        facts=(
            _fact("f1", "tweety", "is_a", "bird"),
            _fact("f2", "penguin_toto", "is_a", "penguin"),
        ),
        rules=(fly_rule,),
        entity_references=("bank",),
        usg_bundle=_demo_usg_bundle(),
        inference_mode=InferenceMode.FORWARD,
        verification_level=VerificationLevel.BASIC,
        task_id="demo_learned",
    )
    coordinator = NeuroSymbolicReasoningCoordinator(learned_adapter=learned)
    result = coordinator.run(request)
    _print_result("Task 4: learned proposals -> symbolic filtering", result)
    for proposal in result.model_proposals:
        if proposal.get("model_id"):
            print(f"  model proposal: {proposal['model_id']} "
                  f"@{proposal['model_version']} candidates={proposal['candidates']}")
    ledger = learned.ledger.to_dict()
    print(f"  adaptation ledger version {ledger['version']}: "
          f"{len(ledger['updates'])} verified updates, "
          f"{len(ledger['rejected_updates'])} rejected")


def demo_benchmark_and_ablation() -> None:
    """Section 10 harnesses on the synthetic demo corpus."""
    def factory(index: int) -> ReasoningRequest:
        return ReasoningRequest(
            goal_predicate="indirect_command",
            goal_args=("captain_harris", "fleet_7"),
            facts=(
                _fact(f"f1_{index}", "captain_harris", "commanded_by", "admiral_reyes"),
                _fact(f"f2_{index}", "admiral_reyes", "commands", "fleet_7"),
            ),
            rules=(_superior_rule(),),
            inference_mode=InferenceMode.HYBRID,
            task_id=f"bench_{index}",
        )

    report = evaluate_corpus(
        factory,
        tuple({"indirect_command(captain_harris,fleet_7)": True} for _ in range(3)),
    )
    print("\n=== Benchmark (synthetic demo corpus) ===")
    print(f"  summary: {dict(sorted(report.summary.items()))}")
    for note in report.notes:
        print(f"  note: {note}")

    ablation = run_ablation(factory(0))
    print("\n=== Ablation (one mechanism at a time) ===")
    contributions = ablation.to_dict()["contributions"]
    for mechanism in sorted(contributions):
        print(f"  {mechanism}: published delta = "
              f"{contributions[mechanism]['published_count']}")


def main() -> None:
    """Run every demonstration section in order."""
    demo_strict_derivation()
    demo_defeasible_and_ambiguity()
    demo_conflict_and_probabilistic()
    demo_learned_assistance()
    demo_benchmark_and_ablation()
    print(f"\noperation registry: {len(NSRL_OPERATION_IDS)} stable IDs "
          f"({NSRL_OPERATION_IDS[0]}..{NSRL_OPERATION_IDS[-1]})")


if __name__ == "__main__":
    main()
