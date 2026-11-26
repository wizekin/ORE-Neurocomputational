"""End-to-end demonstration of the ORE Formal Reasoning and Constraint Engine.

This example runs the complete 52-step algorithm of ``ORE Integrated Algorithm
— Point 5`` over one integrated workload:

1. typed USG-style knowledge is mapped into the versioned logical IR;
2. Datalog inference derives transitive ancestor facts with full provenance;
3. the consistency layer detects and preserves a contradiction without repair;
4. Z3 checks satisfiability, entailment and a countermodel;
5. OR-Tools CP-SAT (with an exact reference fallback) selects a module set;
6. a proof obligation is generated, searched, and kernel-checked;
7. typed result envelopes are validated, published to a shared cognitive
   workspace under the permission policy, and audited to SQLite;
8. the versioned benchmark suite reports correctness and operational metrics.

Run from the repository root:

    python examples/formal_reasoning.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ore_formal_reasoning import (  # noqa: E402
    AssuranceLevel,
    FormalReasoningCoordinator,
    FormalizationRequest,
    ModuleCandidate,
    NamedAssumption,
    PermittedTheory,
    Premise,
    ProofObligation,
    PropImplies,
    PropVar,
    SelectionProblem,
    TaskClass,
    TypedKnowledgeRecord,
    run_benchmark_suite,
)


def build_usg_records() -> tuple[TypedKnowledgeRecord, ...]:
    """Typed USG-style knowledge: relations plus safe Datalog rules."""
    return (
        TypedKnowledgeRecord(
            "obj-tom", "object", "tom", {"entity": "tom"}, version_id="usg-v1",
        ),
        TypedKnowledgeRecord(
            "rel-parent-1", "relation", "parent_of",
            {"predicate": "parent", "subject": "tom", "object": "bob",
             "arg_sorts": ("String", "String")},
            version_id="usg-v1",
        ),
        TypedKnowledgeRecord(
            "rel-parent-2", "relation", "parent_of",
            {"predicate": "parent", "subject": "bob", "object": "ann",
             "arg_sorts": ("String", "String")},
            version_id="usg-v1",
        ),
        TypedKnowledgeRecord(
            "rel-parent-3", "relation", "parent_of",
            {"predicate": "parent", "subject": "ann", "object": "zed",
             "arg_sorts": ("String", "String")},
            version_id="usg-v1",
        ),
        TypedKnowledgeRecord(
            "rule-direct", "rule", "direct_ancestor",
            {
                "head_predicate": "ancestor",
                "head_arg_sorts": ("String", "String"),
                "head_variables": ["X", "Y"],
                "body_predicates": ["parent"],
                "body_0_variables": ["X", "Y"],
                "variables": [("X", "String"), ("Y", "String")],
            },
            version_id="usg-v1",
        ),
        TypedKnowledgeRecord(
            "rule-transitive", "rule", "transitive_ancestor",
            {
                "head_predicate": "ancestor",
                "head_arg_sorts": ("String", "String"),
                "head_variables": ["X", "Z"],
                "body_predicates": ["parent", "ancestor"],
                "body_0_variables": ["X", "Y"],
                "body_1_variables": ["Y", "Z"],
                "variables": [("X", "String"), ("Y", "String"), ("Z", "String")],
            },
            version_id="usg-v1",
        ),
        TypedKnowledgeRecord(
            "attr-capacity", "attribute", "capacity_of",
            {"predicate": "capacity", "subject": "tom", "value": "3",
             "value_sort": "Int"},
            version_id="usg-v1",
        ),
    )


def build_proof_obligation() -> ProofObligation:
    """A small proof obligation: coverage follows from eligibility."""
    selected = PropVar("module_alpha_selected")
    eligible = PropVar("module_alpha_eligible")
    covered = PropVar("requirement_vision_covered")
    return ProofObligation(
        obligation_id="example-proof-1",
        theorem_name="selection_implies_coverage",
        goal=PropImplies(selected, covered),
        premises=(
            Premise("h1", PropImplies(selected, eligible), "stipulated"),
            Premise("h2", PropImplies(eligible, covered), "stipulated"),
        ),
        dependencies=("none",),
    )


def build_selection_problem() -> SelectionProblem:
    """One concrete module-selection task with budgets and exclusivity."""
    return SelectionProblem(
        problem_id="example-selection",
        candidates=(
            ModuleCandidate("vision_a", cost=10.0, capabilities=("vision",),
                            resources={"cpu": 2.0, "memory_gb": 4.0},
                            latency_ms=120.0, reliability=0.92),
            ModuleCandidate("vision_b", cost=7.0, capabilities=("vision",),
                            resources={"cpu": 3.0, "memory_gb": 2.0},
                            latency_ms=90.0, reliability=0.85),
            ModuleCandidate("planner", cost=6.0, capabilities=("planning",),
                            resources={"cpu": 2.0, "memory_gb": 2.0},
                            depends_on=("vision_b",)),
            ModuleCandidate("legacy_hook", cost=2.0, capabilities=("planning",),
                            resources={"cpu": 0.5, "memory_gb": 0.5},
                            mutually_exclusive_with=("planner",)),
        ),
        required_capabilities=("vision", "planning"),
        resource_limits={"cpu": 5.0, "memory_gb": 6.0},
        objective_weights={"cost": 1.0},
        units={"cost": "usd", "cpu": "cores", "memory_gb": "GB", "latency_ms": "ms"},
        deadline_ms=250.0,
    )


def build_workspace():
    """Create one shared cognitive workspace through the ORE components."""
    from ore_workspace.workspace import SharedWorkspace

    return SharedWorkspace(
        current_input="formal reasoning task: derive, verify, optimize, prove",
        current_goal="establish typed results with full provenance",
    )


def build_conflict_request() -> FormalizationRequest:
    """A request carrying a genuinely contradictory knowledge fragment."""
    return FormalizationRequest(
        request_id="example-conflict-1",
        objective="test consistency of a contradictory observation pair",
        task_classes=(TaskClass.BOOLEAN_CONSISTENCY,),
        required_property="room1 availability",
        assumptions=(NamedAssumption(
            "room_authority", "rooms follow one authority per scope",
        ),),
        permitted_theories=frozenset({PermittedTheory.BOOL}),
    )


def build_conflict_ir():
    """An IR whose facts contradict each other under one authority scope."""
    from ore_formal_reasoning import (
        Atom,
        Const,
        DomainSemantics,
        FormalIRBuilder,
        NotF,
        PredicateDecl,
        Statement,
        StatementKind,
    )

    builder = FormalIRBuilder()
    builder.declare_domain(DomainSemantics("default"))
    builder.declare_predicate(PredicateDecl("available", ("String",)))
    builder.add_statement(Statement(
        "obs-a", StatementKind.FACT,
        Atom("available", (Const("room1", "String"),)),
        origin="usg",
    ))
    builder.add_statement(Statement(
        "obs-b", StatementKind.FACT,
        NotF(Atom("available", (Const("room1", "String"),))),
        origin="usg",
    ))
    return builder.build()


def main() -> int:
    print("=" * 72)
    print("ORE Formal Reasoning and Constraint Engine (Point 5) - full run")
    print("=" * 72)

    with tempfile.TemporaryDirectory(prefix="frce-example-") as work_dir:
        # A real shared cognitive workspace wired through the ORE permission
        # policy: FRCE may publish into the granted fields only.
        workspace = build_workspace()
        permissions = workspace.permissions
        permissions.grant(
            "ore_formal_reasoning",
            {"semantic_hypotheses", "artifact_references", "unresolved_questions"},
        )

        coordinator = FormalReasoningCoordinator(
            workspace=workspace,
            permission_policy=permissions,
            audit_path=Path(work_dir) / "frce_audit.sqlite3",
        )

        request = FormalizationRequest(
            request_id="example-request-1",
            objective=(
                "derive ancestor facts from typed USG knowledge, verify "
                "consistency, check the coverage entailment, select modules "
                "under budgets, and machine-check the coverage theorem"
            ),
            task_classes=(TaskClass.MIXED,),
            required_property="ancestor and coverage",
            usg_entity_ids=("obj-tom",),
            assumptions=(
                NamedAssumption(
                    "open_world_knowledge",
                    "the knowledge base is open-world: absence is not falsity",
                ),
            ),
            permitted_theories=frozenset({
                PermittedTheory.BOOL,
                PermittedTheory.LINEAR_INTEGER_ARITHMETIC,
                PermittedTheory.LINEAR_REAL_ARITHMETIC,
                PermittedTheory.BITVECTORS,
            }),
            assurance_level=AssuranceLevel.HIGH,
            optimization_problem=build_selection_problem(),
            proof_obligation=build_proof_obligation(),
        )

        run = coordinator.run(request, usg_records=build_usg_records())

        # A second request exercises the contradiction policy: both sides
        # preserved, resolution requested, no destructive repair.
        conflict_run = coordinator.run(
            build_conflict_request(), ir=build_conflict_ir(),
        )

        audit_path = Path(work_dir) / "frce_audit.sqlite3"

    # -- report ---------------------------------------------------------------
    print(f"\nrun id            : {run.run_id}")
    print(f"overall status    : {run.status}")
    print(f"steps recorded    : {len(run.step_trace)} of 52 (FRC-001..FRC-052)")
    phases: dict[str, list[int]] = {}
    for record in run.step_trace:
        phases.setdefault(record["phase"], []).append(record["step"])
    for phase in sorted(phases):
        steps = phases[phase]
        print(f"  phase {phase}         : {len(steps)} step records "
              f"(first FRC-{steps[0]:03d}, last FRC-{steps[-1]:03d})")

    if run.retrieval is not None:
        print(f"\nretrieval (Step 4) : {len(run.retrieval.selected_record_ids)} records, "
              f"query digest {run.retrieval.query_digest[:16]}...")
    if run.ir is not None:
        print(f"IR (Step 15)       : {len(run.ir.statements)} statements, "
              f"{len(run.ir.rules)} rules, hash {run.ir_hash[:16]}...")
    if run.datalog_result is not None:
        pairs = sorted(
            (f.args[0], f.args[1]) for f in run.datalog_result.derived_facts
            if f.predicate == "ancestor"
        )
        print(f"Datalog (19/20)    : {len(run.datalog_result.derived_facts)} derived "
              f"in {run.datalog_result.iterations} iterations")
        for a, c in pairs:
            print(f"  ancestor({a}, {c})")
        multi = [f for f in run.datalog_result.derived_facts if len(f.derivations) > 1]
        if multi:
            fact = multi[0]
            print(f"  multi-derivation example: {fact.predicate}{tuple(fact.args)} "
                  f"has {len(fact.derivations)} independent derivations")
    if run.consistency is not None:
        print(f"consistency (21/22): {len(run.consistency.conflicts)} conflicts, "
              f"{run.consistency.preserved_sides} sides preserved, "
              f"{run.consistency.destructive_repairs} destructive repairs")

    for index, outcome in enumerate((*run.smt_outcomes, *run.entailment_outcomes), 1):
        print(f"SMT #{index} (26)      : {outcome.solver_status.value} "
              f"({outcome.wall_clock_ms:.1f} ms)")
        if outcome.model is not None:
            print(f"  model independently validated: "
                  f"{outcome.model_independently_validated}")
        if outcome.counterexample is not None:
            print("  countermodel: satisfies the premises, violates the goal")
        if outcome.unsat_core:
            print(f"  unsat core: {outcome.unsat_core}")

    if run.optimization is not None:
        outcome = run.optimization
        print(f"CP-SAT (36)        : {outcome.status.value} "
              f"selection={list(outcome.selection)} "
              f"objective={outcome.objective_value} "
              f"gap={outcome.optimality_gap} backend={outcome.solver.backend}")

    for outcome in run.proof_outcomes:
        if outcome.artifact is not None:
            artifact = outcome.artifact
            print(f"proof (40/41)      : {artifact.theorem_name} "
                  f"[{artifact.checker_name} {artifact.checker_version}] "
                  f"conditional on {artifact.conditional_on}")
        else:
            print(f"proof (40)         : {outcome.detail}")

    print(f"envelopes (43)     : {len(run.results)} typed results, all valid = "
          f"{all(v.valid for v in run.envelope_validations)}")
    print(f"results by kind    : {sorted(r.kind.value for r in run.results)}")
    print(f"publications (44)  : {[p.finding_type for p in run.publications]}")
    if run.promotion_proposals:
        print(f"governance (45)    : {len(run.promotion_proposals)} "
              "promotion proposal(s); USG stays read-only")
    print(f"audit (46)         : record id {run.audit_id} at {audit_path.name}")
    print(f"capabilities (52)  : "
          f"{ {name: state['enabled'] for name, state in run.capabilities.items()} }")
    print(f"metrics (50)       : {run.metrics['wall_clock_ms']} ms wall clock, "
          f"{run.metrics['smt_checks']} SMT checks, "
          f"{run.metrics['proofs_verified']} proof(s) verified, "
          f"unknown-never-coerced={run.metrics['unknown_never_coerced']}")

    if conflict_run.consistency is not None and conflict_run.consistency.conflicts:
        print("\ncontradiction demo (Steps 21-22):")
        for conflict in conflict_run.consistency.conflicts:
            print(f"  {conflict.conflict_id} [{conflict.kind}] "
                  f"side_a={list(conflict.side_a)} side_b={list(conflict.side_b)}")
        print(f"  preserved sides: {conflict_run.consistency.preserved_sides}, "
              f"destructive repairs: {conflict_run.consistency.destructive_repairs}")

    print("\n" + "=" * 72)
    print("Versioned benchmark suite (Step 49)")
    print("=" * 72)
    report = run_benchmark_suite()
    print(f"suite version      : {report.suite_version}")
    print(f"cases              : {report.passed}/{report.total} passed")
    print(f"formalization validity      : {report.formalization_validity:.2f}")
    print(f"proof-check pass rate       : {report.proof_check_pass_rate:.2f}")
    print(f"model validity rate         : {report.model_validity_rate:.2f}")
    print(f"contradiction detection     : {report.contradiction_detection_rate:.2f} "
          f"(false positives {report.contradiction_false_positive_rate:.2f})")
    print(f"optimization feasibility    : {report.optimization_feasibility_rate:.2f}")
    if report.mean_optimization_gap is not None:
        print(f"mean optimization gap       : {report.mean_optimization_gap:.4f}")
    print(f"reproducibility             : {report.reproducibility_rate:.2f}")

    ok = report.passed == report.total and all(
        v.valid for v in run.envelope_validations
    )
    print("\nEXAMPLE RESULT:", "SUCCESS" if ok else "FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
