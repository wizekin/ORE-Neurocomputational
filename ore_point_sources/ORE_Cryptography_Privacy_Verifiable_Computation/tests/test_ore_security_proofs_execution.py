"""Acceptance tests: zero-knowledge proofs and verifiable execution (Phases E-F)."""
from __future__ import annotations

import pytest

from ore_security.exceptions import StatementVersionError
from ore_security.proofs import (
    ProofService,
    ProofStatementRegistry,
    RelationDefinition,
    SchnorrWitness,
)
from ore_security.proofs.service import VERIFIER_VERSION
from ore_security.zkvm import (
    ExecutionVerifier,
    ExecutionWorker,
    Instruction,
    Program,
    benchmark_execution_verification,
)


class FrozenClock:
    def __init__(self, start: float = 3000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


@pytest.fixture()
def clock():
    return FrozenClock()


@pytest.fixture()
def service(clock):
    return ProofService(clock=clock)


@pytest.fixture()
def witness():
    return SchnorrWitness(987654321)


def _valid_proof_case(service, witness, *, task_id="task-42", nonce="nonce-1"):
    y = witness.public_value(service.adapter.group)
    statement = service.build_statement(
        "stmt-1", {"y": y}, task_id=task_id, task_nonce=nonce, ttl=100.0,
    )
    record = service.prove(statement, witness)
    return statement, record


def test_zkp_verifier_accepts_valid_proof(service, witness):
    """Acceptance: ZKP verifier accepts valid proof for the correct statement."""
    statement, record = _valid_proof_case(service, witness)
    result = service.verify(record, statement)
    assert result.verified
    assert "cryptographic verification passed" in result.reason


def test_zkp_verifier_rejects_invalid_proof(service, witness):
    """Acceptance: ZKP verifier rejects invalid proof."""
    statement, record = _valid_proof_case(service, witness)
    tampered = dict(record)
    tampered["proof"] = dict(record["proof"])
    tampered["proof"]["s"] = str((int(record["proof"]["s"]) + 1)
                                 % int(record["proof"]["group"]["q"]))
    result = service.verify(tampered, statement)
    assert not result.verified
    assert "cryptographic verification failed" in result.reason


def test_proof_bound_to_another_task_or_nonce_is_rejected(service, witness):
    """Acceptance: proof bound to another task or nonce is rejected."""
    y = witness.public_value(service.adapter.group)
    statement = service.build_statement(
        "stmt-1", {"y": y}, task_id="task-42", task_nonce="nonce-1", ttl=100.0,
    )
    record = service.prove(statement, witness)
    other_task = service.build_statement(
        "stmt-other", {"y": y}, task_id="task-ATTACK", task_nonce="nonce-2", ttl=100.0,
    )
    assert not service.verify(record, other_task).verified
    same_id_other_nonce = service.build_statement(
        "stmt-1", {"y": y}, task_id="task-42", task_nonce="nonce-REPLAY", ttl=100.0,
    )
    assert not service.verify(record, same_id_other_nonce).verified


def test_expired_statement_proof_is_rejected(service, witness, clock):
    """Proofs for expired statements are rejected (binding includes expiry)."""
    statement, record = _valid_proof_case(service, witness)
    clock.now += 200.0
    result = service.verify(record, statement)
    assert not result.verified
    assert "expired" in result.reason


def test_proof_for_obsolete_circuit_version_is_rejected(service, witness):
    """Acceptance: proof for obsolete circuit version is rejected unless supported."""
    statement, record = _valid_proof_case(service, witness)
    assert service.verify(record, statement).verified
    # retire the relation version through a controlled upgrade (step 31)
    service.registry.register_relation(RelationDefinition(
        relation_id=statement.relation_id, version=statement.relation_version,
        description="retired", public_input_schema={"y": "int"},
        assumptions="retired", motivation="upgrade", supported=False,
    ), replace=True)
    result = service.verify(record, statement)
    assert not result.verified
    assert "retired" in result.reason
    # building new statements on the retired version is refused outright
    y = witness.public_value(service.adapter.group)
    with pytest.raises(StatementVersionError):
        service.build_statement("stmt-new", {"y": y}, task_id="t", task_nonce="n",
                                relation_version=statement.relation_version)


def test_proof_result_records_verifier_and_statement_versions(service, witness):
    """Acceptance: proof result records verifier and statement versions."""
    statement, record = _valid_proof_case(service, witness)
    result = service.verify(record, statement)
    assert result.verifier_version == VERIFIER_VERSION
    assert result.relation_version == statement.relation_version
    assert result.statement_id == statement.statement_id
    assert result.proof_ref == record["proof_id"]
    as_dict = result.to_dict()
    assert as_dict["verifier_version"] and as_dict["relation_version"]


def test_statement_registry_rejects_unknown_versions():
    """Unknown relation ids and versions cannot produce statements."""
    registry = ProofStatementRegistry()
    with pytest.raises(StatementVersionError):
        registry.build_statement("s", "no-such-relation", "1.0.0", {},
                                 task_id="t", task_nonce="n", expires_at=1.0)


# ---------------------------------------------------------------------------
# zkVM: committed program execution
# ---------------------------------------------------------------------------

FIB_PROGRAM = Program(
    instructions=(
        Instruction("PUSHC", 0), Instruction("POPM", 0),   # a = 1
        Instruction("PUSHC", 0), Instruction("POPM", 1),   # b = 1
        Instruction("PUSHC", 1), Instruction("POPM", 2),   # i = 20 (loop bound)
        Instruction("PUSHM", 0), Instruction("PUSHM", 1), Instruction("ADD"),
        Instruction("PUSHM", 0), Instruction("ADD"),       # t = a + 2b? -> simple sum demo
        Instruction("POPM", 3), Instruction("HALT"),
    ),
    constants=(1, 20),
)


def _worker_execution(program=FIB_PROGRAM, public_inputs=(5,)):
    worker = ExecutionWorker()
    return worker.run(program, public_inputs, spot_checks=8)


def test_zkvm_proof_is_bound_to_committed_program_and_output():
    """Acceptance: zkVM proof is bound to committed program and public output."""
    execution = _worker_execution()
    verifier = ExecutionVerifier()
    full = verifier.verify(FIB_PROGRAM, (5,), execution, mode="full")
    spot = verifier.verify(FIB_PROGRAM, (5,), execution, mode="spot")
    assert full["verified"] and spot["verified"]
    assert full["program_digest"] == execution["commitment"]["program_digest"]
    assert full["outputs"] == execution["outputs"]


def test_modified_program_fails_zkvm_verification():
    """Acceptance: modified program fails zkVM verification."""
    execution = _worker_execution()
    verifier = ExecutionVerifier()
    different = Program(
        instructions=FIB_PROGRAM.instructions + (Instruction("PUSHC", 0), Instruction("HALT")),
        constants=FIB_PROGRAM.constants,
    )
    result = verifier.verify(different, (5,), execution, mode="full")
    assert not result["verified"]
    assert "program digest mismatch" in result["reason"]


def test_modified_output_fails_zkvm_verification():
    """Acceptance: modified output fails zkVM verification."""
    execution = _worker_execution()
    verifier = ExecutionVerifier()
    execution["outputs"] = [execution["outputs"][0] + 1] + execution["outputs"][1:]
    assert not verifier.verify(FIB_PROGRAM, (5,), execution, mode="full")["verified"]
    assert not verifier.verify(FIB_PROGRAM, (5,), execution, mode="spot")["verified"]


def test_tampered_trace_root_fails_spot_verification():
    """A forged trace Merkle root is caught by the spot-check openings."""
    execution = _worker_execution()
    execution["proof"] = dict(execution["proof"])
    execution["proof"]["trace_root"] = "ab" * 32
    verifier = ExecutionVerifier()
    result = verifier.verify(FIB_PROGRAM, (5,), execution, mode="spot")
    assert not result["verified"]


def test_zkvm_adapter_does_not_claim_privacy():
    """Acceptance: zkVM adapter does not claim privacy absent a separate mechanism."""
    from ore_security.zkvm.executor import PRIVACY_NOTE, PROVIDES_PRIVACY
    execution = _worker_execution()
    assert PROVIDES_PRIVACY is False
    assert execution["provides_privacy"] is False
    assert "not hidden" in execution["privacy_note"]
    assert PRIVACY_NOTE


def test_zkvm_acceptance_does_not_bypass_semantic_validation():
    """Acceptance: zkVM proof acceptance does not bypass semantic validation."""
    from ore_security import build_default_trust_stack
    from ore_security.contracts import PrincipalKind
    from ore_security.policy import DataClassification, PolicyRule, ResourceDescriptor

    clock = FrozenClock(4000.0)
    stack = build_default_trust_stack(clock=clock)
    ids = stack["identity_service"]
    engine = stack["policy_engine"]
    ids.register_identity("svc-executor", PrincipalKind.SERVICE, roles=("EXECUTOR",))
    engine.resource_registry.register(ResourceDescriptor(
        "compute://external", DataClassification.INTERNAL, "compute-owner"))
    engine.add_rule(PolicyRule(
        "allow-executor", "svc-executor", frozenset({"compute.external.run"}),
        ("compute://external",), DataClassification.INTERNAL,
    ))
    token, _ = ids.issue_credential("svc-executor", "ore-service", ("compute:run",), ttl=600)

    execution = _worker_execution()
    statement, record = _valid_proof_case(stack["proof_service"], SchnorrWitness(42),
                                          task_id="compute-1", nonce="n")
    verifier = ExecutionVerifier()

    # execution proof verified, but the semantic validator rejects the result:
    # the operation must not be treated as semantically valid
    result = stack["pipeline"].execute(
        action="compute.external.run", resource="compute://external/job-1",
        purpose="external-computation", credential_token=token,
        expected_audience="ore-service",
        operation=lambda: execution["outputs"],
        proof=(record, statement),
        semantic_validator=lambda outputs: isinstance(outputs, list) and all(v < 0 for v in outputs),
    )
    assert result.verified is True          # cryptographic proof passed
    assert result.semantic_valid is False   # semantics refused the content
    assert "semantic validation failed" in result.failure_reason


def test_execution_benchmark_reports_costs():
    """Step 37: proof cost is measured against plain recomputation."""
    report = benchmark_execution_verification(FIB_PROGRAM, (5,), spot_checks=4)
    assert report["spot_verified"] and report["full_verified"]
    assert report["steps"] > 0
    assert report["proof_size_bytes"] > 0
    assert "recommendation" in report
