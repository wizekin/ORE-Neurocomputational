"""End-to-end pipeline tests: gates, lineage, determinism, all 52 steps."""
from __future__ import annotations

import pytest
from test_ore_hic_helpers import (
    ASSIGNMENT_IDEA,
    build_compiler,
    fast_settings,
    run_to_completion,
)

from ore_hic import (
    ArtifactKind,
    compile_idea,
)


def test_first_pass_pauses_on_blocking_ambiguity():
    compiler = build_compiler()
    result = compiler.compile(ASSIGNMENT_IDEA)
    assert result.status == "AWAITING_CLARIFICATION"
    # The pipeline stopped before requirements and code generation.
    assert result.requirement_spec is None
    assert result.candidate_set is None
    assert result.code_artifact is None
    # Ranked clarification questions are available.
    assert result.clarification_questions
    statuses = {s.status for s in result.step_log}
    assert "BLOCKED" in statuses


def test_answered_questions_unblock_the_pipeline():
    compiler = build_compiler()
    first = compiler.compile(ASSIGNMENT_IDEA)
    answers = {q.question_id: q.options[0] for q in first.clarification_questions}
    answers["as-model-linearity"] = "confirmed by the user"
    compiler.answer_clarifications(answers)
    second = compiler.compile(ASSIGNMENT_IDEA)
    assert second.status == "AWAITING_APPROVAL"
    assert second.requirement_spec is not None
    assert second.candidate_set is not None
    # The clarified ambiguity is resolved in the new ambiguity set.
    blocking = [a for a in second.ambiguity_set.ambiguities if a.blocking]
    assert blocking == []
    unconfirmed = second.ambiguity_set.unconfirmed_high_impact_assumptions
    assert unconfirmed == ()


def test_codegen_requires_explicit_approval():
    compiler = build_compiler()
    result = run_to_completion(compiler)
    assert result.status == "COMPLETED"
    # Without allow_codegen_without_approval, an approval record exists.
    assert result.approval is not None
    assert result.approval.approved_by == "test-reviewer"


def test_auto_approval_is_explicit_and_audited():
    compiler = build_compiler(allow_codegen_without_approval=True)
    result = run_to_completion(compiler)
    assert result.status == "COMPLETED"
    assert result.approval is not None
    audits = [a.event for a in result.session.ledger.audits]
    assert "AUTO_APPROVAL" in audits


def test_all_52_steps_are_executed_and_logged(completed_result):
    step_ids = [s.step_id for s in completed_result.step_log]
    expected = {f"HIC{i:02d}" for i in range(1, 53)}
    assert expected <= set(step_ids), f"missing steps: {expected - set(step_ids)}"
    executed = [s for s in completed_result.step_log if s.status == "EXECUTED"]
    assert len(executed) >= 52
    phases = {s.phase for s in completed_result.step_log}
    assert phases == {"A", "B", "C", "D", "E", "F", "G", "H"}


def test_twelve_canonical_artifacts_are_produced(completed_result):
    session = completed_result.session
    expected_kinds = {
        ArtifactKind.INPUT_RECORD,
        ArtifactKind.SEMANTIC_GRAPH,
        ArtifactKind.AMBIGUITY_SET,
        ArtifactKind.REQUIREMENT_SPEC,
        ArtifactKind.MATH_MODEL,
        ArtifactKind.ALGO_IR,
        ArtifactKind.CANDIDATE_SET,
        ArtifactKind.VERIFICATION_REPORT,
        ArtifactKind.CODE_ARTIFACT,
        ArtifactKind.EXECUTION_REPORT,
        ArtifactKind.APPROVAL_RECORD,
        ArtifactKind.LEARNING_RECORD,
    }
    present = {kind for kind in expected_kinds if session.versions_of(kind)}
    assert present == expected_kinds, f"missing artifact kinds: {expected_kinds - present}"


def test_full_lineage_is_reconstructable(completed_result):
    lineage = completed_result.lineage()
    assert len(lineage) >= 10
    # The chain starts at the original input record and reaches the code.
    first_outputs = {d for entry in lineage for d in entry.output_digests}
    assert completed_result.input_record.content_digest in first_outputs
    # Every lineage entry cites rule references.
    assert all(entry.rule_references or entry.usg_references for entry in lineage)


def test_provenance_ledger_records_gates_and_approvals(completed_result):
    audits = completed_result.session.ledger.audits
    events = {a.event for a in audits}
    assert "IR_VALIDATION" in events
    assert "VERIFICATION" in events
    assert "SANDBOX_EXECUTION" in events
    assert "APPROVAL_GRANTED" in events


def test_deterministic_recompilation_yields_identical_artifacts():
    idea = ASSIGNMENT_IDEA

    def full_run():
        compiler = build_compiler()
        result = run_to_completion(compiler, idea)
        assert result.status == "COMPLETED"
        return result

    first = full_run()
    second = full_run()
    pairs = (
        ("input_record", first.input_record, second.input_record),
        ("semantic_graph", first.semantic_graph, second.semantic_graph),
        ("ambiguity_set", first.ambiguity_set, second.ambiguity_set),
        ("requirement_spec", first.requirement_spec, second.requirement_spec),
        ("math_model", first.math_model, second.math_model),
        ("algo_ir", first.algo_ir, second.algo_ir),
        ("candidate_set", first.candidate_set, second.candidate_set),
        ("code_artifact", first.code_artifact, second.code_artifact),
        ("learning_record", first.learning_record, second.learning_record),
    )
    for name, a, b in pairs:
        assert a.semantic_digest == b.semantic_digest, f"{name} digest differs across runs"
    # Generated source is byte-identical under deterministic settings.
    assert first.code_artifact.source == second.code_artifact.source


def test_compile_idea_convenience_entry_point():
    result = compile_idea(ASSIGNMENT_IDEA, user="conv-user", settings=fast_settings())
    assert result.status in {"AWAITING_CLARIFICATION", "AWAITING_APPROVAL", "COMPLETED"}


def test_state_machine_of_the_spec_rejects_illegal_runs(completed_result):
    machine = completed_result.requirement_spec.state_machine
    assert machine.can_transition("RECEIVED", "validate_input")
    assert machine.can_transition("VALIDATED", "run_assignment")
    assert not machine.can_transition("COMPLETED", "run_assignment")
    assert not machine.can_transition("ASSIGNED", "validate_input")
    # Recovery paths exist from intermediate states.
    assert machine.can_transition("VALIDATED", "infeasible")


def test_sandboxed_generated_code_matches_in_process_candidate(completed_result):
    """Behavioral conformance: generated code equals the in-process procedure."""
    import json
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    from ore_hic.discovery import AssignmentProblem, ModuleSpec, TaskSpec

    problem = AssignmentProblem(
        tasks=(TaskSpec("t1", 1.0), TaskSpec("t2", 2.0)),
        modules=(ModuleSpec("m1", 4.0, 0.0), ModuleSpec("m2", 3.0, 1.0)),
        eligibility={("t1", "m1"): True, ("t1", "m2"): True, ("t2", "m1"): True, ("t2", "m2"): True},
        capability={("t1", "m1"): 0.9, ("t1", "m2"): 0.3, ("t2", "m1"): 0.4, ("t2", "m2"): 0.8},
        costs={("t1", "m1"): 0.1, ("t1", "m2"): 0.2, ("t2", "m1"): 0.2, ("t2", "m2"): 0.1},
    )
    expected = completed_result.candidate_set.selected().procedure(problem)

    with tempfile.TemporaryDirectory() as tmp:
        module_path = Path(tmp) / "genmod.py"
        module_path.write_text(completed_result.code_artifact.source, encoding="utf-8")
        runner = Path(tmp) / "runner.py"
        runner_source = (
            "import json, sys\n"
            f"sys.path.insert(0, r'{tmp}')\n"
            "import genmod as g\n"
            "r = g.assign_tasks(\n"
            "    (g.Task('t1', 1.0), g.Task('t2', 2.0)),\n"
            "    (g.Module('m1', 4.0, 0.0), g.Module('m2', 3.0, 1.0)),\n"
            "    {('t1','m1'): True, ('t1','m2'): True, ('t2','m1'): True, ('t2','m2'): True},\n"
            "    {('t1','m1'): 0.9, ('t1','m2'): 0.3, ('t2','m1'): 0.4, ('t2','m2'): 0.8},\n"
            "    {('t1','m1'): 0.1, ('t1','m2'): 0.2, ('t2','m1'): 0.2, ('t2','m2'): 0.1},\n"
            ")\n"
            "print(json.dumps({'status': r.status, 'assignment': r.assignment,\n"
            "                  'objective_value': r.objective_value}))\n"
        )
        runner.write_text(runner_source, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(runner)], check=False,
            capture_output=True, text=True, timeout=15,
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        assert payload["status"] == expected.status
        assert payload["assignment"] == dict(expected.assignment)
        assert payload["objective_value"] == pytest.approx(expected.objective_value)


def test_explanation_is_consistent_with_canonical_ir(completed_result):
    explanation = completed_result.explanation
    assert "GOAL" in explanation
    assert "assign_tasks" in explanation
    # Node ids in the explanation exist in the IR.
    import re

    for node_id in re.findall(r"\b(n\d{4})\b", explanation):
        completed_result.algo_ir.node(node_id)  # raises if unknown
