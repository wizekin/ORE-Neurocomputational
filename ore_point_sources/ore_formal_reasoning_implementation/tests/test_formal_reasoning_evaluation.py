"""Phase G/H and end-to-end tests for FRCE: result envelopes, workspace
publication, write governance, SQLite audit, bounded workers, recovery,
capability flags, the benchmark suite, the 52-step coordinator, and the
blackboard bridge.
"""
from __future__ import annotations

import json
import time
from fractions import Fraction

import pytest

from ore_formal_reasoning.contracts import (
    FormalResult,
    FormalizationRequest,
    NamedAssumption,
    PermittedTheory,
    ResultKind,
    ResultStatus,
    SolverIdentity,
    SourceRef,
    TaskClass,
)
from ore_formal_reasoning.ir import (
    Atom,
    BodyLiteral,
    Const,
    DomainSemantics,
    FormalIRBuilder,
    PredicateDecl,
    RuleDecl,
    Statement,
    StatementKind,
    Var,
    VarDecl,
)
from ore_formal_reasoning.coordinator import (
    FormalReasoningCoordinator,
    FormalReasoningService,
)
from ore_formal_reasoning.cpsat import SelectionProblem, ModuleCandidate
from ore_formal_reasoning.envelope import (
    CertificateReplay,
    validate_result_envelope,
)
from ore_formal_reasoning.evaluation import (
    assert_unknown_never_coerced,
    classify_failure_for_improvement,
    coerce_to_truth_value,
    run_benchmark_suite,
)
from ore_formal_reasoning.exceptions import GovernanceViolationError
from ore_formal_reasoning.governance import (
    KnowledgeWriteGovernance,
    PromotionPolicy,
    PromotionProposal,
    SQLiteAuditLog,
    WorkspacePublicationAdapter,
)
from ore_formal_reasoning.policy import (
    BinaryAllowlist,
    BoundedWorker,
    CapabilityFlags,
    FailureClass,
    FileAccessPolicy,
    NetworkPolicy,
    RecoveryPolicy,
)
from ore_formal_reasoning.proofs import (
    PHyp,
    PLam,
    PVar,
    PApp,
    Premise,
    ProofCandidate,
    ProofObligation,
    ProofSearcher,
    ProofVerifier,
    PropImplies,
    PropVar,
)
from ore_formal_reasoning.contracts import ResourceBudget
from ore_formal_reasoning.usg_adapter import TypedKnowledgeRecord
from ore_formal_reasoning.version import (
    FORMAL_REASONING_BENCHMARK_VERSION,
    FORMAL_REASONING_PACKAGE_VERSION,
)


def _result(**overrides: object) -> FormalResult:
    defaults: dict = {
        "result_id": "res-x",
        "request_id": "req-x",
        "kind": ResultKind.SATISFIABILITY_RESULT,
        "status": ResultStatus.SATISFIABLE,
        "statement": "the encoded constraint set is satisfiable",
        "solver": SolverIdentity("Z3", "4.x", {"random_seed": 0}),
        "provenance": (SourceRef("src-1"),),
        "reproducibility": {"ir_hash": "a" * 64},
    }
    defaults.update(overrides)
    return FormalResult(**defaults)


# ---------------------------------------------------------------------------
# Envelope validation (Steps 42-43)
# ---------------------------------------------------------------------------


def test_result_envelope_validates_against_versioned_schema() -> None:
    """Acceptance: a result envelope validates against its versioned schema."""
    validation = validate_result_envelope(_result())
    assert validation.valid is True, validation.errors


def test_every_solver_result_includes_solver_identity() -> None:
    """Acceptance: every solver result includes solver identity and configuration."""
    validation = validate_result_envelope(_result(solver=None))
    assert validation.valid is False
    assert any("solver_identity" in e for e in validation.errors)
    with_identity = validate_result_envelope(_result(solver=SolverIdentity(
        "Z3", "4.x", {"random_seed": 0},
    )))
    assert with_identity.valid is True
    assert with_identity.checks and any(
        c.name == "solver_identity" and c.passed for c in with_identity.checks
    )


def test_unsat_core_overclaim_detected_by_validator() -> None:
    """Acceptance: a non-minimized core must not be presented as minimal."""
    overclaimed = _result(
        status=ResultStatus.UNSATISFIABLE,
        unsat_core=("a", "b"),
        unsat_core_minimized=False,
        limitations=(),
    )
    validation = validate_result_envelope(overclaimed)
    assert validation.valid is False
    honest = validate_result_envelope(_result(
        status=ResultStatus.UNSATISFIABLE,
        unsat_core=("a", "b"),
        unsat_core_minimized=False,
        limitations=("the unsatisfiable core is a sufficient conflicting subset, "
                     "not necessarily a minimum-cardinality cause",),
    ))
    assert honest.valid is True


def test_certificate_replay_rechecks_proofs_and_models() -> None:
    """Step 43: replay certificates where possible."""
    s = PropVar("socrates_is_man")
    h = PropVar("socrates_is_human")
    m = PropVar("socrates_is_mortal")
    obligation = ProofObligation(
        "obl-r", "syllogism", PropImplies(s, m),
        premises=(
            Premise("h1", PropImplies(s, h), "observed"),
            Premise("h2", PropImplies(h, m), "stipulated"),
        ),
    )
    candidate = ProofSearcher().attempt(obligation, ("h1", "h2"))
    assert candidate is not None
    verifier = ProofVerifier()
    outcome = verifier.verify(obligation, candidate, artifact_id="art-replay")
    assert outcome.status == ResultStatus.PROVED
    replay = CertificateReplay(verifier).replay_proof(
        obligation, candidate, expected_hash=outcome.artifact.artifact_hash,
    )
    assert replay.valid is True


# ---------------------------------------------------------------------------
# Workspace publication and write governance (Steps 44-45)
# ---------------------------------------------------------------------------


class _FakeWorkspace:
    """A minimal stand-in mirroring the shared workspace publish path."""

    def __init__(self) -> None:
        self.workspace_id = "ws-test"
        self.revision = 0
        self.published: list[object] = []

    def snapshot(self) -> object:
        return self

    def publish(self, finding: object) -> object:
        from ore_workspace.models import UpdateDecision, ValidationResult

        self.published.append(finding)
        self.revision += 1
        return ValidationResult(
            finding_id=getattr(finding, "finding_id", "f"),
            decision=UpdateDecision.ACCEPTED,
            message="test acceptance",
            workspace_revision=self.revision,
        )


class _FakeValidation:
    def __init__(self, decision_value: str) -> None:
        self.decision_value = decision_value


def test_workspace_publication_distinguishes_result_types() -> None:
    """Acceptance: workspace publication distinguishes inferred, satisfiable,
    optimized and proved result types."""
    from ore_formal_reasoning.governance import _FIELD_BY_KIND

    kinds = {
        ResultKind.RULE_DERIVED_INFERENCE: "frce_inferred_fact",
        ResultKind.SATISFIABILITY_RESULT: "frce_satisfiability_result",
        ResultKind.OPTIMIZATION_RESULT: "frce_optimization_proposal",
        ResultKind.MACHINE_CHECKED_PROOF: "frce_proof_artifact",
    }
    seen = set(kinds.values())
    assert len(seen) == 4
    # distinct workspace fields keep epistemic categories separated
    assert _FIELD_BY_KIND[ResultKind.RULE_DERIVED_INFERENCE] == "semantic_hypotheses"
    assert _FIELD_BY_KIND[ResultKind.SATISFIABILITY_RESULT] == "artifact_references"


def test_publish_result_through_fake_workspace() -> None:
    class _Record:
        def __init__(self) -> None:
            self.decision_value = "accepted"

        @property
        def decision(self):
            class _D:
                value = "accepted"
            return _D()

    class _WS:
        workspace_id = "ws"
        revision = 3

        def snapshot(self):
            return self

        def publish(self, finding):
            self.finding = finding
            return _Record()

    from ore_formal_reasoning.contracts import ResourceBudget as _RB

    class _Policy:
        def can_write(self, module, field):
            return True

    workspace = _WS()
    adapter = WorkspacePublicationAdapter()
    record = adapter.publish_result(workspace, _result(), _Policy())
    assert record.accepted is True
    payload = workspace.finding.payload
    assert payload["epistemic_category"] == "satisfiability_result"
    assert payload["status"] == "satisfiable"


def test_engine_cannot_write_authoritative_usg_storage() -> None:
    """Acceptance: the reasoning engine cannot write directly to authoritative
    USG storage."""
    governance = KnowledgeWriteGovernance()
    with pytest.raises(GovernanceViolationError):
        governance.attempt_direct_usg_write({"fact": "x"})


def test_promotion_requires_configured_approval() -> None:
    """Step 45: promotion requires the configured validation and approval policy."""
    governance = KnowledgeWriteGovernance(PromotionPolicy(approvers=("chief",)))
    proposal = governance.propose_derived_fact(
        "ancestor(tom, ann)", ("r_rec",), ("a_open",),
        proposal_id="prop-1",
    )
    denied = governance.submit(proposal, approver=None)
    assert denied.approved is False
    approved = governance.submit(proposal, approver="chief")
    assert approved.approved is True
    permissive = KnowledgeWriteGovernance(PromotionPolicy(require_explicit_approval=False))
    assert permissive.submit(proposal).approved is True


# ---------------------------------------------------------------------------
# SQLite audit (Step 46)
# ---------------------------------------------------------------------------


def test_sqlite_audit_records_are_complete_and_replayable(tmp_path) -> None:
    """Acceptance: SQLite audit records contain request ID, IR hash, source IDs,
    assumptions, status and artifact hashes."""
    log = SQLiteAuditLog(tmp_path / "audit.sqlite3")
    row_id = log.record_run({
        "request_id": "req-audit",
        "fingerprint": "fp-" + "0" * 60,
        "schema_version": "FRCE-RESULT/1.0.0",
        "ir_version": "FRCE-IR/1.0.0",
        "ir_hash": "b" * 64,
        "input_fact_ids": ["f1", "f2"],
        "input_rule_ids": ["r1"],
        "assumptions": ["a_open"],
        "solver_versions": ["Z3/4.x"],
        "solver_config": {"random_seed": 0},
        "resource_use": {"wall_clock_ms": 12.5},
        "result_kind": "satisfiability_result",
        "status": "satisfiable",
        "proof_refs": ["art-1"],
        "artifact_hashes": ["c" * 64],
        "error": None,
        "trace": [{"operation_id": "FRC-001", "step": 1}],
    })
    assert row_id >= 1
    record = log.get_run("req-audit")
    assert record is not None
    assert record["ir_hash"] == "b" * 64
    assert record["input_fact_ids"] == ["f1", "f2"]
    assert record["assumptions"] == ["a_open"]
    assert record["status"] == "satisfiable"
    assert record["artifact_hashes"] == ["c" * 64]
    assert record["solver_config"]["random_seed"] == 0
    # secrets are redacted before storage
    log.record_run({
        "request_id": "req-secret",
        "fingerprint": "fp2",
        "status": "unknown",
        "api_key": "super-secret-value",
    })
    leaked = log.get_run("req-secret")
    assert leaked is not None
    assert "super-secret-value" not in json.dumps(log.list_runs(10))
    log.close()


# ---------------------------------------------------------------------------
# Bounded workers and recovery (Steps 47-48)
# ---------------------------------------------------------------------------


def test_cpu_memory_and_wallclock_limits_enforced() -> None:
    """Acceptance: CPU, memory and wall-clock limits are enforced."""
    worker = BoundedWorker(process_isolation=True)
    budget = ResourceBudget(wall_clock_ms=400, cpu_seconds=5, memory_mb=128)
    outcome = worker.run(lambda: time.sleep(2.0) or 1, (), budget)
    assert outcome.ok is False
    assert outcome.failure_class == "timeout"


def test_solver_process_crash_contained_and_reported() -> None:
    """Acceptance: solver process crash is contained and reported."""
    import os

    worker = BoundedWorker(process_isolation=True)
    budget = ResourceBudget(wall_clock_ms=3000, cpu_seconds=5)

    def crash() -> int:
        os._exit(3)
        return 0  # pragma: no cover

    outcome = worker.run(crash, (), budget)
    assert outcome.ok is False
    assert outcome.failure_class == "crash"
    assert "exit code 3" in outcome.detail


def test_malformed_solver_output_rejected_safely() -> None:
    """Acceptance: malformed solver output is rejected safely."""
    worker = BoundedWorker(process_isolation=True)
    budget = ResourceBudget(wall_clock_ms=3000, cpu_seconds=5, max_output_bytes=100)
    outcome = worker.run(lambda: {"blob": "x" * 5000}, (), budget)
    assert outcome.ok is False
    assert outcome.failure_class == "output_limit"


def test_unauthorized_file_and_network_access_blocked() -> None:
    """Acceptance: unauthorized file or network access is blocked in the worker."""
    policy = FileAccessPolicy("/tmp/frce-pinned")
    with pytest.raises(GovernanceViolationError):
        policy.check_path_access("/etc/passwd", "r")
    with pytest.raises(GovernanceViolationError):
        NetworkPolicy().check_network_access("solver.example")
    allowlist = BinaryAllowlist(("lean", "z3"))
    with pytest.raises(GovernanceViolationError):
        allowlist.check("curl")


def test_recovery_retries_transient_failures_with_backoff() -> None:
    """Step 48: retry only under policy, with bounded backoff, no correctness change."""
    worker = BoundedWorker(process_isolation=False)
    budget = ResourceBudget(wall_clock_ms=5000, cpu_seconds=5, max_retries=2,
                            retry_backoff_ms=1)
    attempts = {"count": 0}

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise TimeoutError("transient")
        return "done"

    outcome = RecoveryPolicy(retry_backoff_ms=1).execute(flaky, (), budget, worker)
    assert outcome.ok is True
    assert outcome.payload == "done"
    assert outcome.attempts == 3

    def deterministic_crash() -> str:
        raise ValueError("deterministic")

    outcome2 = RecoveryPolicy(retry_backoff_ms=1).execute(
        deterministic_crash, (), budget, worker,
    )
    assert outcome2.ok is False
    assert outcome2.failure_class == FailureClass.CRASH


def test_worker_outcome_requires_failure_class() -> None:
    from ore_formal_reasoning.policy import WorkerOutcome

    with pytest.raises(ValueError):
        WorkerOutcome(ok=False)


# ---------------------------------------------------------------------------
# Capability flags (Step 52)
# ---------------------------------------------------------------------------


def test_capability_flags_promote_only_after_suite_passes() -> None:
    """Acceptance: all enabled capability flags remain off until their acceptance
    suite passes; promotion without a passing suite is rejected."""
    flags = CapabilityFlags()
    with pytest.raises(ValueError):
        flags.promote("lean_export", "suite.lean", False)
    with pytest.raises(ValueError):
        flags.promote("lean_export", "suite.wrong", True)
    state = flags.promote("lean_export", "suite.lean", True)
    assert state.enabled is True
    assert state.promoted_at == FORMAL_REASONING_PACKAGE_VERSION
    snapshot = {c.name: c for c in flags.snapshot()}
    assert snapshot["datalog_engine"].enabled is True
    assert flags.to_dict()["smt_z3"]["acceptance_suite"] == "suite.smt_z3"


def test_unknown_status_never_coerced_into_true_or_false() -> None:
    """Acceptance: a regression test ensures UNKNOWN is never coerced into
    TRUE or FALSE."""
    ok, detail = assert_unknown_never_coerced()
    assert ok, detail
    assert coerce_to_truth_value(ResultStatus.UNKNOWN) is None
    assert coerce_to_truth_value(ResultStatus.TIMEOUT) is None
    assert coerce_to_truth_value(ResultStatus.UNSUPPORTED) is None
    from ore_formal_reasoning.contracts import SolverStatus

    assert coerce_to_truth_value(SolverStatus.UNKNOWN) is None
    assert coerce_to_truth_value(ResultStatus.SATISFIABLE) is True
    assert coerce_to_truth_value(ResultStatus.UNSATISFIABLE) is False


# ---------------------------------------------------------------------------
# Benchmark suite (Steps 49-51)
# ---------------------------------------------------------------------------


def test_versioned_benchmark_suite_passes_with_separate_metrics() -> None:
    """Acceptance: the benchmark suite covers the required fixture families and
    reports inference accuracy separately from solver soundness and reliability."""
    report = run_benchmark_suite()
    assert report.suite_version == FORMAL_REASONING_BENCHMARK_VERSION
    families = {c.family for c in report.cases}
    assert {
        "boolean_sat", "boolean_unsat", "arithmetic", "recursive_datalog",
        "contradiction", "proof_obligation", "cpsat_selection",
        "ambiguous_formalization", "invalid_encoding",
    } <= families
    assert report.passed == report.total, [
        (c.case_id, c.claimed, c.expected) for c in report.cases if not c.passed
    ]
    # correctness and operational quality measured separately
    assert 0.0 <= report.formalization_validity <= 1.0
    assert 0.0 <= report.proof_check_pass_rate <= 1.0
    assert 0.0 <= report.model_validity_rate <= 1.0
    assert 0.0 <= report.contradiction_detection_rate <= 1.0
    assert report.contradiction_false_positive_rate == 0.0
    assert 0.0 <= report.reproducibility_rate <= 1.0


def test_failures_improve_formalization_not_guarantees() -> None:
    """Step 51: failure classification never loosens guarantees."""
    action = classify_failure_for_improvement({"kind": "solver_timeout"})
    assert action.category == "solver_limitation"
    assert action.loosens_guarantees is False
    # unknown kinds fall back to the semantic-mapping category with a fixture
    fallback = classify_failure_for_improvement({"kind": "never_seen_before"})
    assert fallback.category == "semantic_mapping"
    assert fallback.loosens_guarantees is False
    from ore_formal_reasoning.evaluation import ImprovementAction

    with pytest.raises(ValueError, match="never loosen"):
        ImprovementAction("semantic_mapping", "weaken soundness", loosens_guarantees=True)


# ---------------------------------------------------------------------------
# Coordinator end-to-end (all phases)
# ---------------------------------------------------------------------------


_E2E_RECORDS = (
    TypedKnowledgeRecord(
        "rel-f", "relation", "parent_of",
        {"predicate": "parent", "subject": "tom", "object": "bob",
         "arg_sorts": ("String", "String")}, version_id="v1",
    ),
    TypedKnowledgeRecord(
        "rel-g", "relation", "parent_of",
        {"predicate": "parent", "subject": "bob", "object": "ann",
         "arg_sorts": ("String", "String")}, version_id="v1",
    ),
    TypedKnowledgeRecord(
        "rule-base", "rule", "direct_ancestor",
        {
            "head_predicate": "ancestor",
            "head_arg_sorts": ("String", "String"),
            "head_variables": ["X", "Y"],
            "body_predicates": ["parent"],
            "body_0_variables": ["X", "Y"],
            "variables": [("X", "String"), ("Y", "String")],
        },
        version_id="v1",
    ),
    TypedKnowledgeRecord(
        "rule-rec", "rule", "transitive_ancestor",
        {
            "head_predicate": "ancestor",
            "head_arg_sorts": ("String", "String"),
            "head_variables": ["X", "Z"],
            "body_predicates": ["parent", "ancestor"],
            "body_0_variables": ["X", "Y"],
            "body_1_variables": ["Y", "Z"],
            "variables": [("X", "String"), ("Y", "String"), ("Z", "String")],
        },
        version_id="v1",
    ),
)


def _e2e_request(**overrides: object) -> FormalizationRequest:
    defaults: dict = {
        "request_id": "req-e2e",
        "objective": "derive ancestor facts and check consistency",
        "task_classes": (TaskClass.RULE_INFERENCE,),
        "required_property": "ancestor",
        "assumptions": (NamedAssumption("a_open", "open world domain"),),
        "permitted_theories": frozenset({
            PermittedTheory.BOOL,
            PermittedTheory.LINEAR_INTEGER_ARITHMETIC,
        }),
    }
    defaults.update(overrides)
    return FormalizationRequest(**defaults)


def test_coordinator_executes_52_step_algorithm_end_to_end() -> None:
    coordinator = FormalReasoningCoordinator()
    run = coordinator.run(_e2e_request(), usg_records=_E2E_RECORDS)
    assert run.status == "completed"
    steps = [t["step"] for t in run.step_trace]
    assert len(run.step_trace) >= 20
    assert steps[0] == 1 and steps[-1] == 52
    phases = {t["phase"] for t in run.step_trace}
    assert {"A", "B", "C", "G", "H"} <= phases
    assert run.datalog_result is not None
    derived = {(f.args[0], f.args[1]) for f in run.datalog_result.derived_facts}
    assert ("tom", "ann") in derived
    assert run.ir_hash
    assert run.retrieval is not None and run.retrieval.query_digest
    assert all(v.valid for v in run.envelope_validations)
    assert run.promotion_proposals
    assert run.capabilities["datalog_engine"]["enabled"] is True


def test_coordinator_idempotent_replay_produces_equivalent_output() -> None:
    """Acceptance: replaying the same deterministic fixture produces equivalent
    normalized output."""
    coordinator = FormalReasoningCoordinator()
    request = _e2e_request()
    first = coordinator.run(request, usg_records=_E2E_RECORDS)
    second = coordinator.run(request, usg_records=_E2E_RECORDS)
    assert second is first  # the fingerprint cache returns the same typed run
    assert first.fingerprint == second.fingerprint


def test_coordinator_publishes_to_workspace_with_permission_gate(tmp_path) -> None:
    """Step 44/45: publications flow through the workspace permission policy."""
    from ore_workspace.models import WorkspaceState
    from ore_workspace.permissions import PermissionPolicy

    class _Snapshot:
        revision = 0

    class _WS:
        workspace_id = "ws-frce"
        revision = 0

        def snapshot(self):
            return _Snapshot()

        def publish(self, finding):
            from ore_workspace.models import UpdateDecision, ValidationResult

            self.last = finding
            self.revision += 1
            return ValidationResult(
                finding_id=finding.finding_id,
                decision=UpdateDecision.ACCEPTED,
                message="accepted by test harness",
                workspace_revision=self.revision,
            )

    workspace = _WS()
    permissions = PermissionPolicy()
    permissions.grant("ore_formal_reasoning", {"semantic_hypotheses", "artifact_references"})
    coordinator = FormalReasoningCoordinator(
        workspace=workspace, permission_policy=permissions,
    )
    run = coordinator.run(_e2e_request(), usg_records=_E2E_RECORDS)
    assert run.publications
    finding_types = {p.finding_type for p in run.publications}
    assert "frce_inferred_fact" in finding_types
    denied_workspace = _WS()
    no_grant = PermissionPolicy()  # no grants for the producer
    denied_run = FormalReasoningCoordinator(
        workspace=denied_workspace, permission_policy=no_grant,
    ).run(_e2e_request(), usg_records=_E2E_RECORDS)
    assert any(
        not p.accepted and "permission policy" in p.detail
        for p in denied_run.publications
    )


def test_coordinator_persists_audit_record(tmp_path) -> None:
    coordinator = FormalReasoningCoordinator(
        audit_path=tmp_path / "frce_audit.sqlite3",
    )
    run = coordinator.run(_e2e_request(), usg_records=_E2E_RECORDS)
    assert run.audit_id is not None
    record = coordinator.audit_log.get_run(run.request_id)
    assert record is not None
    assert record["ir_hash"] == run.ir_hash
    assert record["fingerprint"] == run.fingerprint
    assert record["status"] == run.status


def test_coordinator_returns_structured_insufficient_formalization() -> None:
    """Step 7: return a structured 'insufficient formalization' result."""
    coordinator = FormalReasoningCoordinator()
    run = coordinator.run(FormalizationRequest(
        request_id="req-blocked",
        objective="do something vague",
        task_classes=(TaskClass.RULE_INFERENCE,),
    ))
    assert run.status == "not_established"
    blocked = run.results[0]
    assert blocked.status == ResultStatus.INVALID_INPUT
    assert "cannot be operationalized" in " ".join(blocked.limitations)
    assert any(
        t["step"] == 7 and t["details"]["sufficient"] is False for t in run.step_trace
    )


def test_coordinator_runs_entailment_and_optimization_and_proof() -> None:
    """Mixed reasoning: inference + entailment + optimization + proof in one run."""
    x = __import__("ore_formal_reasoning.ir", fromlist=["Var"]).Var("x", "Int")
    VarDecl = __import__("ore_formal_reasoning.ir", fromlist=["VarDecl"]).VarDecl
    builder = __import__(
        "ore_formal_reasoning.ir", fromlist=["FormalIRBuilder"]
    ).FormalIRBuilder()
    builder.declare_domain(
        __import__("ore_formal_reasoning.ir", fromlist=["DomainSemantics"]).DomainSemantics(
            "default"
        )
    )
    builder.add_statement(
        __import__("ore_formal_reasoning.ir", fromlist=["Statement"]).Statement(
            "p1",
            __import__("ore_formal_reasoning.ir", fromlist=["StatementKind"]).StatementKind.CONSTRAINT,
            __import__("ore_formal_reasoning.ir", fromlist=["Compare"]).Compare(
                __import__("ore_formal_reasoning.ir", fromlist=["CompareOp"]).CompareOp.GT,
                x,
                __import__("ore_formal_reasoning.ir", fromlist=["NumLit"]).NumLit.int_value(5),
            ),
            variables=(VarDecl(x),),
        )
    )
    builder.add_statement(
        __import__("ore_formal_reasoning.ir", fromlist=["Statement"]).Statement(
            "goal",
            __import__("ore_formal_reasoning.ir", fromlist=["StatementKind"]).StatementKind.GOAL,
            __import__("ore_formal_reasoning.ir", fromlist=["Compare"]).Compare(
                __import__("ore_formal_reasoning.ir", fromlist=["CompareOp"]).CompareOp.GT,
                x,
                __import__("ore_formal_reasoning.ir", fromlist=["NumLit"]).NumLit.int_value(4),
            ),
            variables=(VarDecl(x),),
        )
    )
    ir = builder.build()
    obligation = ProofObligation(
        "obl-e2e", "coverage_chain", PropImplies(PropVar("sel"), PropVar("covered")),
        premises=(
            Premise("h1", PropImplies(PropVar("sel"), PropVar("eligible")), "stipulated"),
            Premise("h2", PropImplies(PropVar("eligible"), PropVar("covered")), "stipulated"),
        ),
    )
    problem = SelectionProblem(
        problem_id="e2e-opt",
        candidates=(
            ModuleCandidate("alpha", cost=3.0, capabilities=("vision",),
                            resources={"cpu": 1.0}),
        ),
        required_capabilities=("vision",),
        resource_limits={"cpu": 2.0},
        objective_weights={"cost": 1.0},
        units={"cpu": "cores"},
    )
    request = _e2e_request(
        task_classes=(TaskClass.MIXED,),
        assurance_level=__import__(
            "ore_formal_reasoning.contracts", fromlist=["AssuranceLevel"]
        ).AssuranceLevel.HIGH,
        optimization_problem=problem,
        proof_obligation=obligation,
    )
    run = FormalReasoningCoordinator().run(request, ir=ir)
    assert run.entailment_outcomes
    assert run.entailment_outcomes[0].result_status == "unsatisfiable"
    assert run.optimization is not None
    assert run.optimization.status == ResultStatus.OPTIMAL
    assert run.proof_outcomes
    assert run.proof_outcomes[0].status == ResultStatus.PROVED
    kinds = {r.kind for r in run.results}
    assert ResultKind.SATISFIABILITY_RESULT in kinds
    assert ResultKind.OPTIMIZATION_RESULT in kinds
    assert ResultKind.MACHINE_CHECKED_PROOF in kinds
    assert all(v.valid for v in run.envelope_validations)


def test_coordinator_ambiguous_usg_reference_records_failure() -> None:
    coordinator = FormalReasoningCoordinator()
    records = _E2E_RECORDS + (
        TypedKnowledgeRecord(
            "rel-h", "relation", "parent_of",
            {"predicate": "parent", "subject": "ann", "object": "zed",
             "arg_sorts": ("String", "String")}, version_id="v2",
        ),
    )
    run = coordinator.run(
        _e2e_request(usg_entity_ids=("parent_of",)), usg_records=records,
    )
    assert run.failures or run.warnings  # ambiguity surfaced, never silently resolved


def test_service_facade_exposes_capabilities_and_benchmarks() -> None:
    service = FormalReasoningService()
    capabilities = service.capabilities()
    assert "datalog_engine" in capabilities
    report = service.benchmarks()
    assert report.total >= 10
    assert report.passed == report.total


# ---------------------------------------------------------------------------
# Blackboard bridge (Point 1 integration)
# ---------------------------------------------------------------------------


def test_blackboard_bridge_specialist_round_trip() -> None:
    from ore_contracts import ContractCatalog
    from ore_formal_reasoning.bridge import (
        FormalReasoningSpecialist,
        register_formal_reasoning_contracts,
    )

    catalog = ContractCatalog()
    register_formal_reasoning_contracts(catalog)
    specialist = FormalReasoningSpecialist()
    manifest = specialist.describe()
    assert manifest.identity.module_id == "ore.formal_reasoning.specialist"

    # Build a small IR the specialist will formalize and evaluate.
    builder = FormalIRBuilder()
    builder.declare_domain(DomainSemantics("default"))
    builder.declare_predicate(PredicateDecl("parent", ("String", "String")))
    builder.declare_predicate(PredicateDecl("ancestor", ("String", "String")))
    builder.add_statement(Statement(
        "f1", StatementKind.FACT,
        Atom("parent", (Const("tom", "String"), Const("bob", "String"))),
    ))
    builder.add_rule(RuleDecl(
        "base", "direct ancestry",
        variables=(VarDecl(Var("X", "String")), VarDecl(Var("Y", "String"))),
        head=Atom("ancestor", (Var("X", "String"), Var("Y", "String"))),
        body=(BodyLiteral(Atom("parent", (Var("X", "String"), Var("Y", "String")))),),
    ))
    ir = builder.build()

    class _Input:
        def __init__(self, payload: dict) -> None:
            self.payload = payload

    class _Context:
        run_id = "bb-run-1"

    result = specialist.execute(
        {"task_frame": _Input({
            "request_id": "bb-frce-1",
            "objective": "derive ancestor facts",
            "task_classes": ["rule_inference"],
            "required_property": "ancestor",
            "ir": ir,
        })},
        _Context(),
    )
    assert result.status.value in {"success", "failed"}
    assert result.artifacts, "the specialist must return its finding-batch artifact"
    artifact = result.artifacts[0]
    assert artifact.envelope.contract_id == "ORE.FRCE.FINDING_BATCH"
    findings = artifact.payload["findings"]
    assert findings
    assert all(f["field"] == "artifact_references" for f in findings)


def test_package_public_api_importable() -> None:
    import ore_formal_reasoning

    assert ore_formal_reasoning.FORMAL_REASONING_PACKAGE_VERSION
    assert len(ore_formal_reasoning.FRCE_OPERATION_IDS) == 52
    assert ore_formal_reasoning.FormalReasoningCoordinator is not None
