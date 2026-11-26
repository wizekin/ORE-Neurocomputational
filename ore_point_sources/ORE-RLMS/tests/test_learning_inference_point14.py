"""Acceptance tests: bounded execution, sandboxing, and the USG authority boundary."""
from __future__ import annotations

import time

import pytest

from ore_learning.adapters import (
    BrokenAdapter,
    CountingAdapter,
    ExecutionContext,
    MockRegressionAdapter,
)
from ore_learning.contracts import (
    AuthorityBoundary,
    CheckOutcome,
    DeterministicCheck,
    InferenceRequest,
    ModelProposal,
    ProposalStatus,
    UncertaintyMetadata,
    InputProvenance,
    ValidationReport,
)
from ore_learning.inference import InferenceExecutor, LeastPrivilegeSandbox


def make_request(**overrides) -> InferenceRequest:
    """A fully populated inference request."""
    values = dict(task_id="task-01", model_id="reg.model",
                  inputs=({"values": [0.1, 0.2, -0.3, 0.4], "timestamp": 1.0},),
                  budget_ms=10000.0, deadline_seconds=5.0)
    values.update(overrides)
    return InferenceRequest(**values)


class BusyAdapter:
    """An adapter that ignores checkpoints and busy-loops (hard-timeout path)."""

    adapter_kind = CountingAdapter.adapter_kind
    input_schema = CountingAdapter.input_schema
    output_schema = CountingAdapter.output_schema

    def invoke(self, payload, execution):  # noqa: ANN001
        """Busy-loop far past the deadline."""
        end = time.perf_counter() + 5.0
        while time.perf_counter() < end:
            pass
        return {"values": [1.0], "intervals": [[0.0, 1.0]], "confidence": 0.5}


def test_inference_timeout_cancels_execution_and_releases_resources():
    """Acceptance: inference timeout cancels execution and releases resources."""
    adapter = CountingAdapter(steps=200, sleep_seconds=0.02)
    request = make_request(deadline_seconds=0.06)
    releases: list[str] = []
    result = InferenceExecutor().execute(
        request, adapter, artifact_digest="a" * 16, model_version="1.0.0",
        release_hook=lambda status: releases.append(status))
    assert result.status == "cancelled"
    assert result.failure_category == "timeout"
    assert result.execution.cancelled is True
    assert result.execution.steps_executed < 200  # stopped early
    assert releases == ["cancelled"]  # resources released on cancellation
    assert result.validation.verified is False
    assert any(check.outcome is CheckOutcome.FAIL
               for check in result.validation.checks)


def test_hard_deadline_guards_non_cooperating_adapters():
    """The SIGALRM guard cancels adapters that never call checkpoint()."""
    request = make_request(deadline_seconds=0.15)
    result = InferenceExecutor().execute(request, BusyAdapter(),
                                         artifact_digest="a" * 16,
                                         model_version="1.0.0")
    assert result.status == "cancelled"
    assert result.failure_category == "timeout"


def test_adapter_output_is_validated_against_its_schema():
    """Acceptance: adapter output is validated against its schema."""
    result = InferenceExecutor().execute(make_request(), BrokenAdapter(),
                                         artifact_digest="a" * 16,
                                         model_version="1.0.0")
    assert result.status == "failed"
    assert result.failure_category == "schema_violation"
    assert result.typed_output == {} or "intervals" not in result.typed_output
    assert not result.validation.verified


def test_every_result_retains_identity_and_input_provenance():
    """Acceptance: every result retains model ID, version, digest, provenance."""
    adapter = MockRegressionAdapter()
    request = make_request()
    result = InferenceExecutor().execute(request, adapter,
                                         artifact_digest="feedbeefcafe1234",
                                         model_version="2.1.0")
    assert result.status == "completed"
    assert result.model_id == "reg.model"
    assert result.model_version == "2.1.0"
    assert result.artifact_digest == "feedbeefcafecafe"[:16] or \
        result.artifact_digest == "feedbeefcafe1234"
    assert result.input_provenance.input_refs
    assert len(result.input_provenance.input_digests) == \
        len(request.inputs)
    assert result.input_provenance.timestamps == (1.0,)
    payload = result.to_dict()
    assert payload["model_id"] == "reg.model"
    assert payload["artifact_digest"] == result.artifact_digest
    assert payload["input_provenance"]["input_refs"]


def test_artifact_digest_mismatch_fails_execution():
    """Supply-chain check: the runtime artifact must match its registered digest."""
    verifier_calls: list[tuple[str, str, str]] = []

    def verifier(model_id: str, version: str, digest: str) -> bool:
        verifier_calls.append((model_id, version, digest))
        return False

    executor = InferenceExecutor(digest_verifier=verifier)
    result = executor.execute(make_request(), MockRegressionAdapter(),
                              artifact_digest="tampered123456",
                              model_version="1.0.0")
    assert result.status == "failed"
    assert result.failure_category == "artifact_digest_mismatch"
    assert verifier_calls and verifier_calls[0][0] == "reg.model"


def test_raw_model_output_cannot_write_directly_to_usg():
    """Acceptance: raw model output cannot write directly to authoritative USG."""
    boundary = AuthorityBoundary()
    # 1) a raw inference result is not even the right type
    raw_result = InferenceExecutor().execute(
        make_request(), MockRegressionAdapter(), artifact_digest="a" * 16,
        model_version="1.0.0")
    refused = boundary.request_canonical_update(raw_result)
    assert refused["decision"] == "refused"
    assert "raw model output" in refused["reason"]
    # 2) a proposal whose deterministic validation failed is refused
    from learning_point14_helpers import digest_for
    unvalidated = ModelProposal(
        proposal_id="prop-1", task_id="task-01", model_id="reg.model",
        model_version="1.0.0", artifact_digest=digest_for("m"),
        payload={"values": [1.0]},
        uncertainty=UncertaintyMetadata(confidence=0.8),
        input_provenance=InputProvenance(input_refs=("in:1",),
                                         input_digests=("d:1",)),
        status=ProposalStatus.PROPOSED,
        validation=ValidationReport(checks=(
            DeterministicCheck(check_id="c", validator="unit-test",
                               outcome=CheckOutcome.FAIL, detail="assertion failed"),)))
    refused_two = boundary.request_canonical_update(unvalidated)
    assert refused_two["decision"] == "refused"
    assert "validation" in refused_two["reason"]
    # 3) a validated proposal without approval is refused (no self-approval)
    from learning_point14_helpers import digest_for
    validated = ModelProposal(
        proposal_id="prop-2", task_id="task-01", model_id="reg.model",
        model_version="1.0.0", artifact_digest=digest_for("m"),
        payload={"values": [1.0]},
        uncertainty=UncertaintyMetadata(confidence=0.8),
        input_provenance=InputProvenance(input_refs=("in:1",),
                                         input_digests=("d:1",)),
        status=ProposalStatus.VALIDATED,
        validation=ValidationReport(checks=(
            DeterministicCheck(check_id="c", validator="unit-test",
                               outcome=CheckOutcome.PASS),)))
    refused_three = boundary.request_canonical_update(validated)
    assert refused_three["decision"] == "refused"
    assert "approval" in refused_three["reason"]
    assert len(boundary.refusals) == 3
    # 4) only validated + coordinator-approved proposals queue a USG update request
    approved = ModelProposal(
        proposal_id="prop-3", task_id="task-01", model_id="reg.model",
        model_version="1.0.0", artifact_digest=digest_for("m"),
        payload={"values": [1.0]},
        uncertainty=UncertaintyMetadata(confidence=0.8),
        input_provenance=InputProvenance(input_refs=("in:1",),
                                         input_digests=("d:1",)),
        status=ProposalStatus.APPROVED,
        validation=validated.validation)
    queued = boundary.request_canonical_update(approved)
    assert queued["decision"] == "queued_for_usg_workflow"
    assert len(boundary.update_requests) == 1
    summary = boundary.describe()
    assert summary["refusals"] == 3 and summary["update_requests"] == 1


def test_executions_record_structured_audit_events():
    """The executor's audit hook records one event per execution (Step 54)."""
    events: list[dict] = []
    executor = InferenceExecutor(
        audit_hook=lambda operation, metadata: events.append(
            {"operation": operation, **metadata}))
    executor.execute(make_request(), MockRegressionAdapter(),
                     artifact_digest="a" * 16, model_version="1.0.0")
    executor.execute(make_request(), BrokenAdapter(), artifact_digest="a" * 16,
                     model_version="1.0.0")
    assert len(events) == 2
    assert events[0]["operation"] == "execute" and events[0]["status"] == "completed"
    assert events[1]["status"] == "failed"
    assert "model_id" in events[0] and "duration_ms" in events[0]


def test_sandbox_rejects_forbidden_operations_statically():
    """Acceptance: generated code cannot execute outside the approved sandbox."""
    sandbox = LeastPrivilegeSandbox()
    rejected = [
        "import subprocess\nresult = 1",
        "import socket\nresult = 2",
        "result = open('/etc/passwd')",
        "result = eval('1+1')",
        "import os\nresult = os.system('ls')",
    ]
    for code in rejected:
        outcome = sandbox.execute(code)
        assert outcome.status == "rejected", code
        assert outcome.violations, code
    allowed = sandbox.execute("import math\nresult = math.factorial(5)")
    assert allowed.status == "executed"
    assert allowed.result == 120


def test_sandbox_runtime_import_guard():
    """Runtime layer: even dynamically constructed imports stay inside the allowlist."""
    sandbox = LeastPrivilegeSandbox()
    outcome = sandbox.execute(
        "builtins = __import__('sys')\nresult = 1")
    assert outcome.status in ("rejected", "error")
    assert outcome.violations
    # an allowed module imports fine at runtime
    ok = sandbox.execute("import json\nresult = json.dumps([1, 2])")
    assert ok.status == "executed" and ok.result == "[1, 2]"


def test_execution_context_cooperative_cancellation():
    """ExecutionContext checkpoints honour cancellation signals."""
    class Signal:
        cancelled = True
        reason = "coordinator abort"

    context = ExecutionContext(cancellation=Signal())
    with pytest.raises(TimeoutError, match="coordinator abort"):
        context.checkpoint()
    # step budget exhaustion also cancels
    limited = ExecutionContext(max_steps=2)
    limited.checkpoint()
    limited.checkpoint()
    with pytest.raises(TimeoutError, match="deadline"):
        limited.checkpoint()
