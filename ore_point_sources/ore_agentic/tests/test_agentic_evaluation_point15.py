"""Acceptance tests for output contracts, evidence and the critic loop (Point 15).

Covers specification acceptance tests 24-30: contracts declared before
execution, structural and semantic validation, evidence provenance,
correlated-source suppression, critic defects mapped to criteria, bounded
revision loops and final independent verification.
"""
from __future__ import annotations

import pytest

from ore_agentic import (
    ApprovalRequirement,
    ApprovalStatus,
    ContractValidator,
    CriterionKind,
    CriterionSpec,
    DeterministicCheckRunner,
    DeterministicCritic,
    EvidenceBundle,
    EvidenceItem,
    EvidenceKind,
    FinalVerifier,
    MaterialConflict,
    OutputContract,
    OutcomeStatus,
    SchemaValidator,
    VerificationReport,
    assess_evidence,
    validate_schema,
)
from ore_agentic.evaluation.critic import CriticLoop, RevisionBudget

from agentic_point15_helpers import CAP_A, CAP_B, make_engine, make_request


def make_contract() -> OutputContract:
    """Build the standard test contract with criteria of every kind."""
    return OutputContract(
        contract_id="test.contract", version="1.0.0",
        criteria=(
            CriterionSpec("summary_present", CriterionKind.COMPLETENESS,
                          "summary key present",
                          {"required_keys": ("summary",)}),
            CriterionSpec("summary_type", CriterionKind.SCHEMA,
                          "summary is a non-empty string",
                          {"schema": {"type": "object",
                                      "properties": {"summary": {"type": "string",
                                                                 "min_length": 1}}}}),
            CriterionSpec("word_count", CriterionKind.DETERMINISTIC,
                          "deterministic word count check",
                          {"check_id": "word_count"}),
            CriterionSpec("evidence_backed", CriterionKind.EVIDENCE,
                          "at least two evidence refs",
                          {"minimum_refs": 2}),
        ))


# ------------------------------------------------------- contracts (24-25)

def test_output_contract_declared_in_advance_and_versioned():
    """Acceptance 24: output contracts are declared in advance and versioned."""
    contract = make_contract()
    assert contract.contract_id == "test.contract"
    assert contract.version == "1.0.0"
    assert contract.declared_at  # declared before execution, timestamped
    assert len(contract.criteria_of_kind(CriterionKind.SCHEMA)) == 1
    with pytest.raises(ValueError):
        OutputContract(contract_id="", version="1.0.0")
    with pytest.raises(ValueError):
        OutputContract(contract_id="dup", version="1.0.0", criteria=(
            CriterionSpec("c", CriterionKind.COMPLETENESS),
            CriterionSpec("c", CriterionKind.SCHEMA)))


def test_output_schema_and_domain_invariants_are_validated():
    """Acceptance 25: output schema and domain invariants are validated."""
    schema = {"type": "object",
              "required": ["summary", "count"],
              "properties": {"summary": {"type": "string", "min_length": 1},
                             "count": {"type": "integer", "minimum": 0,
                                       "maximum": 10},
                             "mode": {"enum": ["fast", "slow"]}}}
    assert validate_schema({"summary": "ok", "count": 3, "mode": "fast"},
                           schema) == []
    errors = validate_schema({"summary": "", "count": 99, "mode": "other"}, schema)
    assert len(errors) == 3  # min_length, above maximum, enum
    validator = SchemaValidator()
    with pytest.raises(Exception):
        validator.validate_strict({"count": -1}, schema)
    # Engine-level: schema violations fail the task before aggregation.
    from ore_agentic import LocalWorker, SubgoalSpec, WorkerManifest

    engine = make_engine()
    engine.register_worker(
        WorkerManifest(worker_id="bad-output", version="1.0.0",
                       capabilities=(CAP_A, CAP_B), latency_seconds=0.1,
                       cost_per_task=0.1),
        LocalWorker("bad-output", "1.0.0", (CAP_A, CAP_B),
                    lambda envelope: {"value": "missing field"}))
    request = make_request(subgoals=(
        SubgoalSpec(
            name="alpha", capability=CAP_A,
            output_schema={"type": "object",
                           "required": ["value", "quality"],
                           "properties": {"quality": {"type": "number"}}}),))
    result = engine.run(request)
    task = result.task_results["task-alpha"]
    assert task.status.value == "failed"
    assert "required field missing" in task.error


# ----------------------------------------------------------- evidence (26-27)

def test_evidence_retains_source_identity_and_provenance():
    """Acceptance 26: evidence retains source identity and provenance."""
    item = EvidenceItem(item_id="ev-1", source_id="usg:bundle-a",
                        claim="attention modulates reading speed",
                        kind=EvidenceKind.PRIMARY,
                        source_locator="usg:record:123", version="1.0.0",
                        reliability=0.9)
    serialized = item.to_dict()
    assert serialized["source_id"] == "usg:bundle-a"
    assert serialized["source_locator"] == "usg:record:123"
    assert serialized["version"] == "1.0.0"
    assert serialized["kind"] == "primary"
    assert serialized["timestamp"]
    assessment = assess_evidence(EvidenceBundle([item]), topic="attention")
    assert assessment.independent_confirmations == 1
    assert assessment.supporting == ("ev-1",)


def test_correlated_sources_are_not_counted_as_independent_confirmation():
    """Acceptance 27: correlated sources are not counted as independent confirmation."""
    bundle = EvidenceBundle((
        EvidenceItem(item_id="ev-1", source_id="wire-a", claim="claim",
                     correlation_group="wire-service"),
        EvidenceItem(item_id="ev-2", source_id="wire-a-copy", claim="claim",
                     correlation_group="wire-service"),
        EvidenceItem(item_id="ev-3", source_id="independent-journal",
                     claim="claim"),
    ))
    assessment = assess_evidence(bundle, topic="claim")
    assert assessment.independent_confirmations == 2  # 1 group + 1 ungrouped
    assert set(assessment.suppressed_correlated) == {"ev-1", "ev-2"}
    # Conflict is preserved, never averaged away.
    conflicting = EvidenceBundle((
        EvidenceItem(item_id="s-1", source_id="journal-x", claim="claim holds"),
        EvidenceItem(item_id="o-1", source_id="journal-y",
                     claim="claim does not hold", supports=False),
    ))
    assessment = assess_evidence(conflicting, topic="claim")
    assert len(assessment.conflicts) == 1
    conflict = assessment.conflicts[0]
    assert isinstance(conflict, MaterialConflict)
    assert len(conflict.positions) == 2
    assert assessment.bayesian_posterior is None  # withheld without justification
    assert assessment.assumptions  # assumptions preserved


def test_bayesian_update_only_with_justified_independence():
    """Bayesian updating requires justified independence of primary evidence."""
    independent = EvidenceBundle((
        EvidenceItem(item_id="p-1", source_id="primary-a", claim="c",
                     kind=EvidenceKind.PRIMARY, reliability=0.9),
        EvidenceItem(item_id="p-2", source_id="primary-b", claim="c",
                     kind=EvidenceKind.PRIMARY, reliability=0.9),
    ))
    justified = assess_evidence(independent, topic="c", prior=0.5,
                                allow_bayes=True)
    assert justified.bayesian_posterior is not None
    assert justified.bayesian_posterior > 0.5  # two reliable confirmations
    correlated = EvidenceBundle((
        EvidenceItem(item_id="p-1", source_id="wire-a", claim="c",
                     kind=EvidenceKind.PRIMARY, correlation_group="wire"),
        EvidenceItem(item_id="p-2", source_id="wire-b", claim="c",
                     kind=EvidenceKind.PRIMARY, correlation_group="wire"),
    ))
    unjustified = assess_evidence(correlated, topic="c", prior=0.5,
                                  allow_bayes=True)
    assert unjustified.bayesian_posterior is None  # correlated: withheld
    from ore_agentic import bayes_posterior
    assert bayes_posterior(0.5, 0.8, 0.4) == pytest.approx(0.667, abs=0.01)
    with pytest.raises(ValueError):
        bayes_posterior(1.5, 0.8, 0.4)


# ------------------------------------------------------------ critic (28-29)

def test_critic_defects_map_to_contract_criteria():
    """Acceptance 28: critic defects map to contract criteria."""
    runner = DeterministicCheckRunner()
    runner.register("word_count", lambda payload: (
        len(str(payload.get("summary", "")).split()) >= 3,
        "summary must have at least three words"))
    validator = ContractValidator(deterministic_runner=runner)
    critic = DeterministicCritic(validator)
    artifact = {"summary": "", "evidence_refs": ["only-one"]}
    report = critic.review(artifact, make_contract())
    criterion_ids = {defect.criterion_id for defect in report.defects}
    assert criterion_ids == {"summary_type", "word_count", "evidence_backed"}
    for defect in report.defects:
        assert defect.severity in ("info", "minor", "major", "critical")
        assert defect.rationale
        assert defect.suggested_correction


def test_revision_loop_stops_at_iteration_cost_or_stall_budget():
    """Acceptance 29: revision loop stops at iteration/cost budget."""
    runner = DeterministicCheckRunner()
    runner.register("word_count", lambda payload: (False, "always failing"))
    validator = ContractValidator(deterministic_runner=runner)
    critic = DeterministicCritic(validator)
    contract = make_contract()

    def never_fixes(artifact, defects):
        return dict(artifact)  # no improvement

    iteration_limited = CriticLoop(critic, RevisionBudget(max_iterations=2))
    report = iteration_limited.run({"summary": "x"}, contract,
                                   revise=never_fixes)
    assert report.stop_reason == "stalled"  # same defect count twice

    cost_limited = CriticLoop(critic, RevisionBudget(max_iterations=10,
                                                     cost_budget=0.5))
    report = cost_limited.run({"summary": "x"}, contract, revise=never_fixes,
                              revision_cost=1.0)
    assert report.stop_reason == "cost_budget"

    no_revision = CriticLoop(critic, RevisionBudget(max_iterations=5))
    report = no_revision.run({"summary": "x"}, contract, revise=None)
    assert report.stop_reason == "no_revision_capability"
    assert len(report.artifact_versions) == 1  # evidence trail never overwritten


def test_revision_loop_passes_when_defects_are_corrected():
    """A bounded revision loop converges when revisions fix the defects."""
    runner = DeterministicCheckRunner()
    runner.register("word_count", lambda payload: (
        len(str(payload.get("summary", "")).split()) >= 3, "too few words"))
    validator = ContractValidator(deterministic_runner=runner)
    critic = DeterministicCritic(validator)
    contract = make_contract()

    def fixes(artifact, defects):
        return {"summary": "now a valid summary text",
                "evidence_refs": ["e1", "e2"]}

    loop = CriticLoop(critic, RevisionBudget(max_iterations=3))
    report = loop.run({"summary": "x", "evidence_refs": []}, contract,
                      revise=fixes)
    assert report.stop_reason == "passed"
    assert report.final_passed
    assert len(report.artifact_versions) == 2  # original + revised version


# ---------------------------------------------------- final verification (30)

def test_final_verifier_rejects_failed_deterministic_test():
    """Acceptance 30: final verifier rejects failed deterministic test."""
    runner = DeterministicCheckRunner()
    runner.register("math_check", lambda payload: (2 + 2 == 4, "arithmetic ok"))
    runner.register("failing_check", lambda payload: (False, "broken invariant"))
    verifier = FinalVerifier(runner)
    report = verifier.verify({}, ("math_check", "failing_check"))
    assert isinstance(report, VerificationReport)
    assert not report.passed
    assert report.failed_checks == ("failing_check",)
    # Unknown checks fail closed.
    report = verifier.verify({}, ("unknown_check",))
    assert not report.passed
    assert "unknown deterministic check" in report.detail


def test_final_verifier_enforces_approval_gates():
    """Consequential actions require human approval (Step 30)."""
    runner = DeterministicCheckRunner()
    verifier = FinalVerifier(runner)
    contract = OutputContract(contract_id="consequential", version="1.0.0",
                              approval=ApprovalRequirement(
                                  consequential=True, approver_role="operator"))
    pending = verifier.verify({}, (), contract)
    assert pending.approval is ApprovalStatus.PENDING
    assert not pending.passed
    approved = FinalVerifier(runner, approver=lambda cid: ApprovalStatus.APPROVED)
    report = approved.verify({}, (), contract)
    assert report.passed
    rejected = FinalVerifier(runner, approver=lambda cid: ApprovalStatus.REJECTED)
    assert not rejected.verify({}, (), contract).passed


def test_engine_final_verification_gates_unverified_outcomes():
    """End-to-end: failed final verification marks outputs unverified (Step 50)."""
    engine = make_engine(deterministic_checks={
        "deliverables_present": lambda payload: (False, "forced verification "
                                                         "failure")})
    result = engine.run(make_request())
    assert result.verification is not None and not result.verification.passed
    assert result.outcomes.unverified_task_ids  # unverified reported distinctly
    assert result.outcome_status is not OutcomeStatus.COMPLETED
    assert "deliverables_present" in result.verification.detail


def test_performance_criterion_enforces_latency_budget():
    """Performance criteria compare elapsed time against the declared budget."""
    runner = DeterministicCheckRunner()
    validator = ContractValidator(deterministic_runner=runner)
    contract = OutputContract(
        contract_id="perf", version="1.0.0",
        criteria=(CriterionSpec("latency", CriterionKind.PERFORMANCE,
                                "under one second",
                                {"max_seconds": 1.0}),))
    fast = validator.validate({"elapsed_seconds": 0.5}, contract)
    assert fast.passed
    slow = validator.validate({"elapsed_seconds": 2.0}, contract)
    assert not slow.passed
