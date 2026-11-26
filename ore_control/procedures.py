"""Conservative procedure learning for the control layer (Phase H).

Implements Point 02 Steps 37-40: candidate procedure extraction from verified
decision traces, generalization and safety evaluation on held-out examples
and counterexamples, versioned registration as non-authoritative procedures
behind promotion gates, and cautious reuse with monitoring, disabling,
rollback, and quarantine. A single successful run is never sufficient to
promote a procedure, and procedures never rewrite authoritative knowledge.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ore_workspace.models import utc_now

from .config import ControlPolicy
from .contracts import (
    OperatorSpec,
    RiskClass,
    RetryPolicy,
    sha256_json,
)
from .operators import (
    ControlServices,
    OperatorAction,
    OperatorOutcome,
    OperatorRegistry,
)

# Replay executor: (task_text) -> ReplayResult(success, latency_ms, produced_verified)
ReplayExecutor = Callable[[str], "ReplayResult"]


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Outcome of replaying a procedure on one held-out input (Step 38)."""

    success: bool
    latency_ms: float
    produced_verified: bool
    notes: str = ""


@dataclass(frozen=True, slots=True)
class ProcedureSequenceStep:
    """One step of a candidate procedure's operator sequence (Step 37)."""

    operator_id: str
    normalized_arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Freeze the step arguments."""
        object.__setattr__(self, "normalized_arguments", dict(self.normalized_arguments))

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable step."""
        return {
            "operator_id": self.operator_id,
            "normalized_arguments": dict(self.normalized_arguments),
        }


@dataclass(frozen=True, slots=True)
class VerifiedRunRecord:
    """One verified run considered during procedure extraction (Step 37)."""

    run_id: str
    goal_fingerprint: str
    task_fingerprint: str
    operator_sequence: tuple[ProcedureSequenceStep, ...]
    status: str
    mean_latency_ms: float

    def __post_init__(self) -> None:
        """Freeze the sequence."""
        object.__setattr__(self, "operator_sequence", tuple(self.operator_sequence))


@dataclass(frozen=True, slots=True)
class CandidateProcedure:
    """A reusable-procedure candidate extracted from verified runs (Step 37)."""

    procedure_id: str
    goal_fingerprint: str
    operator_sequence: tuple[ProcedureSequenceStep, ...]
    repeatable_runs: int
    preconditions: tuple[str, ...] = ("task_text_available",)
    assumptions: tuple[str, ...] = ()
    extraction_provenance: Mapping[str, Any] = field(default_factory=dict)
    extracted_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate the candidate."""
        if not self.operator_sequence:
            raise ValueError("a candidate procedure needs at least one step")
        if self.repeatable_runs < 1:
            raise ValueError("repeatable_runs must be positive")
        object.__setattr__(self, "operator_sequence", tuple(self.operator_sequence))
        object.__setattr__(self, "preconditions", tuple(self.preconditions))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "extraction_provenance", dict(self.extraction_provenance))

    def sequence_fingerprint(self) -> str:
        """Return the stable fingerprint of the operator-id sequence.

        Only operator identities enter the fingerprint: recorded arguments are
        parameters of the procedure, not part of its identity (Step 37).
        """
        return sha256_json({
            "goal_fingerprint": self.goal_fingerprint,
            "sequence": [step.operator_id for step in self.operator_sequence],
        })

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable candidate record."""
        return {
            "procedure_id": self.procedure_id,
            "goal_fingerprint": self.goal_fingerprint,
            "operator_sequence": [step.to_dict() for step in self.operator_sequence],
            "repeatable_runs": int(self.repeatable_runs),
            "preconditions": list(self.preconditions),
            "assumptions": list(self.assumptions),
            "extraction_provenance": dict(self.extraction_provenance),
            "extracted_at": self.extracted_at,
            "sequence_fingerprint": self.sequence_fingerprint(),
        }


@dataclass(frozen=True, slots=True)
class GeneralizationReport:
    """Held-out and counterexample evaluation of one candidate (Step 38)."""

    held_out_successes: int
    held_out_total: int
    counterexample_rejections: int
    counterexample_total: int
    mean_replay_latency_ms: float
    reference_latency_ms: float
    resource_regression_ratio: float
    memorization_detected: bool
    notes: tuple[str, ...] = ()

    @property
    def success_rate(self) -> float:
        """Return the held-out success rate (0.0 when nothing was tested)."""
        if not self.held_out_total:
            return 0.0
        return self.held_out_successes / self.held_out_total

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable report."""
        return {
            "held_out_successes": int(self.held_out_successes),
            "held_out_total": int(self.held_out_total),
            "success_rate": self.success_rate,
            "counterexample_rejections": int(self.counterexample_rejections),
            "counterexample_total": int(self.counterexample_total),
            "mean_replay_latency_ms": float(self.mean_replay_latency_ms),
            "reference_latency_ms": float(self.reference_latency_ms),
            "resource_regression_ratio": float(self.resource_regression_ratio),
            "memorization_detected": bool(self.memorization_detected),
            "notes": list(self.notes),
        }


@dataclass
class ProcedureRecord:
    """One registered, versioned, non-authoritative procedure (Step 39)."""

    procedure_id: str
    version: str
    operator_sequence: tuple[ProcedureSequenceStep, ...]
    goal_fingerprint: str
    preconditions: tuple[str, ...]
    parameters: Mapping[str, Any]
    provenance: Mapping[str, Any]
    test_results: Mapping[str, Any]
    limitations: tuple[str, ...]
    rollback_information: Mapping[str, Any]
    risk_class: RiskClass
    approved: bool
    enabled: bool = True
    quarantined: bool = False
    created_at: str = field(default_factory=utc_now)
    executions: list[Mapping[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable procedure record."""
        return {
            "procedure_id": self.procedure_id,
            "version": self.version,
            "operator_sequence": [step.to_dict() for step in self.operator_sequence],
            "goal_fingerprint": self.goal_fingerprint,
            "preconditions": list(self.preconditions),
            "parameters": dict(self.parameters),
            "provenance": dict(self.provenance),
            "test_results": dict(self.test_results),
            "limitations": list(self.limitations),
            "rollback_information": dict(self.rollback_information),
            "risk_class": self.risk_class.value,
            "approved": bool(self.approved),
            "enabled": bool(self.enabled),
            "quarantined": bool(self.quarantined),
            "created_at": self.created_at,
            "execution_count": len(self.executions),
            "authoritative": False,
        }


def extract_candidate_procedure(
    verified_runs: Sequence[VerifiedRunRecord],
    *,
    policy: ControlPolicy,
) -> CandidateProcedure | None:
    """Extract a candidate procedure from verified runs (Step 37).

    A candidate is extracted only when the same goal fingerprint and the same
    stable operator sequence appear in at least ``min_repeatable_runs``
    verified runs; a single successful execution never generalizes.
    """
    by_sequence: dict[str, list[VerifiedRunRecord]] = {}
    for record in verified_runs:
        if record.status not in {"SUCCEEDED", "COMPLETED_WITH_UNCERTAINTY"}:
            continue
        fingerprint = sha256_json({
            "goal_fingerprint": record.goal_fingerprint,
            "sequence": [step.operator_id for step in record.operator_sequence],
        })
        by_sequence.setdefault(fingerprint, []).append(record)
    for fingerprint, runs in sorted(by_sequence.items()):
        if len(runs) < policy.procedures.min_repeatable_runs:
            continue
        first = runs[0]
        reference = runs[0].mean_latency_ms
        task_fingerprints = {run.task_fingerprint for run in runs}
        return CandidateProcedure(
            procedure_id=f"procedure:{fingerprint[:24]}",
            goal_fingerprint=first.goal_fingerprint,
            operator_sequence=first.operator_sequence,
            repeatable_runs=len(runs),
            preconditions=(
                "task_text_available", "retrieval_domains_available",
                "not_exact_extraction_input",
            ),
            assumptions=(
                "operator sequence stability across verified runs",
                f"extracted from {len(runs)} verified runs with "
                f"{len(task_fingerprints)} distinct task fingerprints",
            ),
            extraction_provenance={
                "run_ids": [run.run_id for run in runs],
                "reference_latency_ms": reference,
                "extracted_at": utc_now(),
                "extraction": "ore_control.procedures.extract_candidate_procedure",
            },
        )
    return None


def evaluate_generalization(
    candidate: CandidateProcedure,
    *,
    held_out_inputs: Sequence[str],
    counterexample_inputs: Sequence[str],
    replay_executor: ReplayExecutor,
    policy: ControlPolicy,
) -> GeneralizationReport:
    """Evaluate a candidate on held-out inputs and counterexamples (Step 38).

    Rejections on counterexamples are required: a procedure that succeeds on
    inputs it should not handle is a memorizer or an over-generalizer and is
    rejected. Resource regression compares mean replay latency against the
    extraction reference latency.
    """
    held_out = list(held_out_inputs)[: policy.procedures.held_out_examples]
    successes = 0
    latencies: list[float] = []
    for text in held_out:
        result = replay_executor(text)
        if result.success:
            successes += 1
        latencies.append(result.latency_ms)
    counterexamples = list(counterexample_inputs)[
        : policy.procedures.counterexample_checks
    ]
    rejections = 0
    for text in counterexamples:
        result = replay_executor(text)
        if not result.success:
            rejections += 1
    reference = float(
        candidate.extraction_provenance.get("reference_latency_ms", 0.0) or 0.0,
    )
    mean_latency = statistics.fmean(latencies) if latencies else 0.0
    regression = (
        mean_latency / reference if reference > 0 else (0.0 if not latencies else 1.0)
    )
    memorization = len(candidate.extraction_provenance.get("run_ids", [])) < 2 or (
        not held_out and not counterexamples
    )
    notes: list[str] = []
    if successes < len(held_out):
        notes.append("some held-out inputs failed replay")
    if rejections < len(counterexamples):
        notes.append("procedure accepted a counterexample it must reject")
    return GeneralizationReport(
        held_out_successes=successes,
        held_out_total=len(held_out),
        counterexample_rejections=rejections,
        counterexample_total=len(counterexamples),
        mean_replay_latency_ms=mean_latency,
        reference_latency_ms=reference,
        resource_regression_ratio=regression,
        memorization_detected=bool(memorization),
        notes=tuple(notes),
    )


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """Gate-by-gate promotion verdict for one candidate (Step 39)."""

    procedure_id: str
    promoted: bool
    gate_results: Mapping[str, Mapping[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable promotion decision."""
        return {
            "procedure_id": self.procedure_id,
            "promoted": bool(self.promoted),
            "gate_results": {
                name: dict(result) for name, result in self.gate_results.items()
            },
        }


def evaluate_promotion_gates(
    candidate: CandidateProcedure,
    report: GeneralizationReport,
    *,
    policy: ControlPolicy,
    risk_class: RiskClass = RiskClass.LOW,
    manual_approval: bool | None = None,
) -> PromotionDecision:
    """Evaluate every promotion gate; promotion is the conjunction (Step 39).

    A positive mean score alone is insufficient: repeatability, held-out
    success, counterexample rejection, no memorization, resource regression,
    and risk-appropriate approval must all pass.
    """
    gates: dict[str, Mapping[str, Any]] = {}

    def gate(name: str, passed: bool, detail: Mapping[str, Any]) -> bool:
        gates[name] = {"passed": bool(passed), **dict(detail)}
        return passed

    overall = True
    overall &= gate(
        "repeatable_runs",
        candidate.repeatable_runs >= policy.procedures.min_repeatable_runs,
        {"runs": candidate.repeatable_runs,
         "required": policy.procedures.min_repeatable_runs},
    )
    overall &= gate(
        "held_out_success",
        report.held_out_total > 0
        and report.success_rate >= policy.procedures.held_out_success_rate,
        {"success_rate": report.success_rate,
         "required": policy.procedures.held_out_success_rate,
         "tested": report.held_out_total},
    )
    overall &= gate(
        "counterexamples_rejected",
        report.counterexample_total > 0
        and report.counterexample_rejections == report.counterexample_total,
        {"rejections": report.counterexample_rejections,
         "tested": report.counterexample_total},
    )
    overall &= gate(
        "no_memorization",
        not report.memorization_detected,
        {"memorization_detected": report.memorization_detected},
    )
    overall &= gate(
        "resource_regression",
        report.resource_regression_ratio <= policy.procedures.max_resource_regression,
        {"ratio": report.resource_regression_ratio,
         "allowed": policy.procedures.max_resource_regression},
    )
    requires_manual = (
        risk_class in {RiskClass.MEDIUM, RiskClass.HIGH}
        and policy.procedures.high_risk_requires_manual_approval
    )
    approved = (
        (not requires_manual) if manual_approval is None else bool(manual_approval)
    )
    overall &= gate(
        "risk_approval",
        approved,
        {"risk_class": risk_class.value, "requires_manual": requires_manual},
    )
    return PromotionDecision(
        procedure_id=candidate.procedure_id, promoted=overall, gate_results=gates,
    )


class ProcedureRegistry:
    """Versioned, non-authoritative procedure store (Steps 39-40)."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._procedures: dict[str, ProcedureRecord] = {}
        self.events: list[Mapping[str, Any]] = []

    def register(self, record: ProcedureRecord) -> None:
        """Register one promoted procedure with version identity."""
        if record.procedure_id in self._procedures:
            raise ValueError(
                f"procedure already registered: {record.procedure_id}"
            )
        self._procedures[record.procedure_id] = record
        self.events.append({
            "event": "procedure_registered",
            "procedure_id": record.procedure_id,
            "version": record.version,
            "at": utc_now(),
        })

    def get(self, procedure_id: str) -> ProcedureRecord | None:
        """Return one procedure record."""
        return self._procedures.get(procedure_id)

    def records(self) -> tuple[ProcedureRecord, ...]:
        """Return all procedure records in stable ID order."""
        return tuple(
            self._procedures[key] for key in sorted(self._procedures)
        )

    def record_execution(
        self, procedure_id: str, *, success: bool, latency_ms: float,
    ) -> None:
        """Record one monitored procedure execution (Step 40)."""
        record = self._procedures.get(procedure_id)
        if record is None:
            return
        record.executions.append({
            "success": bool(success), "latency_ms": float(latency_ms),
            "at": utc_now(),
        })

    def check_health(
        self, procedure_id: str, *, policy: ControlPolicy,
    ) -> tuple[bool, str]:
        """Check monitored health; disable degraded procedures (Step 40).

        Success-rate degradation over the monitoring window disables the
        procedure. Disabling is a registry-local action: it touches no
        workspace state and no authoritative knowledge.
        """
        record = self._procedures.get(procedure_id)
        if record is None:
            return False, "procedure_not_registered"
        if not record.enabled:
            return False, "procedure_disabled"
        if not record.approved:
            return False, "procedure_not_approved"
        window = record.executions[-policy.procedures.monitoring_window:]
        if len(window) >= policy.procedures.monitoring_window:
            successes = sum(1 for item in window if item.get("success"))
            rate = successes / len(window)
            if rate < policy.procedures.disable_success_rate:
                self.disable(procedure_id, reason="degraded_success_rate")
                return False, "degraded_success_rate"
        return True, "healthy"

    def disable(self, procedure_id: str, *, reason: str) -> None:
        """Disable one procedure without corrupting authoritative knowledge."""
        record = self._procedures.get(procedure_id)
        if record is None:
            return
        record.enabled = False
        self.events.append({
            "event": "procedure_disabled",
            "procedure_id": procedure_id,
            "reason": reason,
            "rollback_available": bool(record.rollback_information),
            "at": utc_now(),
        })

    def rollback(self, procedure_id: str) -> bool:
        """Roll back a procedure registration to its prior version state."""
        record = self._procedures.get(procedure_id)
        if record is None or not record.rollback_information:
            return False
        prior = record.rollback_information.get("prior_version")
        record.enabled = False
        self.events.append({
            "event": "procedure_rolled_back",
            "procedure_id": procedure_id,
            "prior_version": prior,
            "at": utc_now(),
        })
        return True

    def quarantine(self, procedure_id: str, *, reason: str) -> None:
        """Quarantine a procedure after a safety condition is violated."""
        record = self._procedures.get(procedure_id)
        if record is None:
            return
        record.quarantined = True
        record.enabled = False
        self.events.append({
            "event": "procedure_quarantined",
            "procedure_id": procedure_id,
            "reason": reason,
            "at": utc_now(),
        })


class ProcedureOperatorHandler:
    """Cautious procedure-reuse handler (Step 40).

    The handler re-checks each recorded step's operator preconditions and
    version, replays the sequence with per-step validation, and reports
    execution to the monitor. Any step failure fails the whole procedure
    execution instead of forcing a result.
    """

    def __init__(
        self,
        *,
        procedure: ProcedureRecord,
        registry: OperatorRegistry,
        procedure_registry: ProcedureRegistry,
    ) -> None:
        """Bind the handler to its procedure and operator registries."""
        self.procedure = procedure
        self.registry = registry
        self.procedure_registry = procedure_registry

    def __call__(
        self,
        context,
        state: Mapping[str, Any],
        action: OperatorAction,
        services: ControlServices,
    ) -> OperatorOutcome:
        """Replay the procedure sequence with re-checked preconditions."""
        import time as _time

        from .operators import evaluate_preconditions

        started = _time.perf_counter()
        findings: list[Any] = []
        checkpoint_events: list[Mapping[str, Any]] = []
        working_state: dict[str, Any] = dict(state)
        for index, step in enumerate(self.procedure.operator_sequence):
            spec = self.registry.get_spec(step.operator_id)
            if spec is None or spec.version != step.normalized_arguments.get(
                "_operator_version", spec.version,
            ):
                self.procedure_registry.record_execution(
                    self.procedure.procedure_id, success=False,
                    latency_ms=(_time.perf_counter() - started) * 1000,
                )
                return OperatorOutcome(
                    status="failed",
                    payload={"failed_step": index},
                    diagnostics={"reason": "operator_missing_or_version_changed"},
                    failure_classification="procedure_operator_unavailable",
                )
            ok, failures = evaluate_preconditions(spec, working_state, services)
            if not ok:
                self.procedure_registry.record_execution(
                    self.procedure.procedure_id, success=False,
                    latency_ms=(_time.perf_counter() - started) * 1000,
                )
                return OperatorOutcome(
                    status="deferred",
                    payload={"failed_step": index},
                    diagnostics={
                        "reason": "precondition_failed",
                        "precondition_failures": list(failures),
                    },
                )
            handler = self.registry.handler(step.operator_id)
            if handler is None:
                return OperatorOutcome(
                    status="failed",
                    payload={"failed_step": index},
                    diagnostics={"reason": "handler_unavailable"},
                    failure_classification="procedure_handler_missing",
                )
            step_action = OperatorAction(
                operator_id=step.operator_id,
                arguments={
                    key: value for key, value in step.normalized_arguments.items()
                    if not key.startswith("_")
                },
            )
            outcome: OperatorOutcome = handler(
                context, working_state, step_action, services,
            )
            if outcome.status == "failed":
                self.procedure_registry.record_execution(
                    self.procedure.procedure_id, success=False,
                    latency_ms=(_time.perf_counter() - started) * 1000,
                )
                return outcome
            findings.extend(outcome.findings)
            checkpoint_events.extend(outcome.checkpoint_events)
            # Carry published payloads forward for later steps' preconditions.
            for finding in outcome.findings:
                bucket = working_state.setdefault(finding.field, [])
                if isinstance(bucket, list):
                    bucket.append(dict(finding.payload))
        elapsed_ms = (_time.perf_counter() - started) * 1000
        self.procedure_registry.record_execution(
            self.procedure.procedure_id, success=True, latency_ms=elapsed_ms,
        )
        return OperatorOutcome(
            status="completed",
            payload={"procedure_id": self.procedure.procedure_id,
                     "steps_replayed": len(self.procedure.operator_sequence)},
            findings=tuple(findings),
            diagnostics={
                "procedure_id": self.procedure.procedure_id,
                "version": self.procedure.version,
                "non_authoritative": True,
            },
            checkpoint_events=tuple(checkpoint_events),
            elapsed_ms=elapsed_ms,
        )


def procedure_operator_spec(record: ProcedureRecord) -> OperatorSpec:
    """Build the candidate-operator spec for one procedure (Step 40)."""
    return OperatorSpec(
        operator_id=record.procedure_id,
        version=record.version,
        purpose=f"Replay the verified procedure '{record.procedure_id}'.",
        preconditions=record.preconditions,
        required_capabilities=("procedural.reuse",),
        estimated_time_seconds=0.05,
        estimated_memory_mb=2.0,
        expected_progress=0.6,
        expected_information_gain=0.3,
        risk_class=record.risk_class,
        writes_workspace=True,
        write_fields=tuple(sorted({
            finding_field
            for step in record.operator_sequence
            for finding_field in ()  # conservative default: references only
        }) or ("artifact_references",)),
        read_fields=("current_input",),
        retry_policy=RetryPolicy(timeout_seconds=10.0, max_retries=1),
        handler_reference="ore_control.procedures:ProcedureOperatorHandler",
        processing_path="EXPLICIT_SYMBOLIC",
        provenance_annotation="ore.control.procedure",
    )


def wrap_procedure(
    record: ProcedureRecord,
    *,
    registry: OperatorRegistry,
    procedure_registry: ProcedureRegistry,
) -> tuple[OperatorSpec, ProcedureOperatorHandler]:
    """Wrap one procedure as a reusable candidate operator (Step 40)."""
    spec = procedure_operator_spec(record)
    handler = ProcedureOperatorHandler(
        procedure=record, registry=registry, procedure_registry=procedure_registry,
    )
    return spec, handler


__all__ = [
    "CandidateProcedure",
    "GeneralizationReport",
    "ProcedureOperatorHandler",
    "ProcedureRecord",
    "ProcedureRegistry",
    "PromotionDecision",
    "ReplayResult",
    "VerifiedRunRecord",
    "evaluate_generalization",
    "evaluate_promotion_gates",
    "extract_candidate_procedure",
    "procedure_operator_spec",
    "wrap_procedure",
]
