"""Resource-aware inference execution (Point 14, Section 2.1 and acceptance 8-12).

The executor runs adapters under time, step, and cancellation limits, verifies
artifact digests before execution, validates every output against the adapter's
declared schema, and always releases reserved resources -- on success, on schema
violation, and on timeout/cancellation.  A hard wall-clock guard (SIGALRM where
available) backs the cooperative checkpoints for adapters that misbehave.
"""
from __future__ import annotations

import signal
import threading
import time
from typing import Any, Callable, Mapping

from ..adapters import ExecutionContext, SpecialistAdapter, validate_payload_against_schema
from ..config import LearningConfig
from ..contracts import (
    CheckOutcome,
    DeterministicCheck,
    ExecutionMetrics,
    InferenceRequest,
    InferenceResult,
    InputProvenance,
    SchemaDescriptor,
    UncertaintyMetadata,
    ValidationReport,
)
from ..contracts.manifests import canonical_payload_digest

__all__ = ("InferenceExecutor", "execute_with_limits")


class InferenceExecutor:
    """Runs one specialist behind typed limits and returns an InferenceResult."""

    def __init__(self, config: LearningConfig | None = None,
                 audit_hook: Callable[[str, Mapping[str, Any]], None] | None = None,
                 digest_verifier: Callable[[str, str, str], bool] | None = None) -> None:
        """Configure governance limits, an optional audit hook and digest verifier.

        ``digest_verifier(model_id, version, digest) -> bool`` supplies the
        supply-chain check that the runtime artifact still matches its registered
        digest (Section 9: supply-chain compromise).
        """
        self.config = config or LearningConfig()
        self.audit_hook = audit_hook
        self.digest_verifier = digest_verifier
        self.executions: list[InferenceResult] = []

    def execute(self, request: InferenceRequest, adapter: SpecialistAdapter,
                *, artifact_digest: str = "", model_version: str = "",
                release_hook: Callable[[str], bool] | None = None) -> InferenceResult:
        """Execute one request under its limits (acceptance 9-12).

        The happy path returns a completed result whose payload passed the adapter's
        output-schema validation.  Timeouts cancel execution and release resources;
        schema violations produce a failed result with category ``schema_violation``.
        """
        started = time.perf_counter()
        provenance = self._provenance(request)
        metrics_holder = {"steps": 0, "cancelled": False}

        def finish(status: str, output: dict[str, Any] | None, category: str = "",
                   uncertainty: UncertaintyMetadata | None = None,
                   checks: tuple[DeterministicCheck, ...] = ()) -> InferenceResult:
            wall_ms = (time.perf_counter() - started) * 1000.0
            result = InferenceResult(
                task_id=request.task_id, model_id=request.model_id,
                model_version=model_version or "unversioned",
                artifact_digest=artifact_digest or "unverified",
                status=status, typed_output=output or {},
                uncertainty=uncertainty or UncertaintyMetadata(
                    confidence=0.0, abstained=True,
                    notes="no output; execution did not complete"),
                execution=ExecutionMetrics(
                    wall_clock_ms=wall_ms, steps_executed=metrics_holder["steps"],
                    cancelled=metrics_holder["cancelled"]),
                input_provenance=provenance,
                validation=(ValidationReport(checks=checks) if checks
                            else _schema_check_payload(output, adapter.output_schema,
                                                       status)),
                failure_category=category)
            self.executions.append(result)
            if release_hook is not None:
                release_hook(status)
            self._audit(result)
            return result

        # supply-chain check: the artifact must match its registered digest
        if self.digest_verifier is not None and artifact_digest:
            if not self.digest_verifier(request.model_id, model_version,
                                        artifact_digest):
                return finish("failed", None, category="artifact_digest_mismatch")

        execution = ExecutionContext(
            deadline_seconds=request.deadline_seconds, budget_ms=request.budget_ms,
            cancellation=request.cancellation)
        metrics_holder_ref = metrics_holder

        original = _HardTimeout.install(request.deadline_seconds)
        try:
            output = adapter.invoke(request.inputs[0], execution)
        except TimeoutError as exc:
            metrics_holder_ref["cancelled"] = True
            metrics_holder_ref["steps"] = execution.steps
            reason = str(exc) or "timeout"
            return finish("cancelled", None, category="timeout",
                          checks=(DeterministicCheck(
                              check_id="timeout", validator="executor-deadline",
                              outcome=CheckOutcome.FAIL, detail=reason),))
        except Exception as exc:  # noqa: BLE001 - execution isolation by design
            metrics_holder_ref["steps"] = execution.steps
            return finish("failed", None, category=type(exc).__name__)
        finally:
            metrics_holder_ref["steps"] = execution.steps
            _HardTimeout.restore(original)

        ok, reason = validate_payload_against_schema(output, adapter.output_schema)
        if not ok:
            return finish("failed", output, category="schema_violation")
        confidence = _payload_confidence(output)
        return finish("completed", output,
                      uncertainty=UncertaintyMetadata(
                          confidence=confidence,
                          calibration_method="adapter-declared",
                          notes="confidence is a proposal signal, never proof"))

    # ------------------------------------------------------------- internals

    def _provenance(self, request: InferenceRequest) -> InputProvenance:
        """Build input provenance (refs + digests) for the whole request."""
        refs, digests, timestamps = [], [], []
        for index, item in enumerate(request.inputs):
            refs.append(f"input:{index}:{request.request_digest[:8]}")
            digests.append(canonical_payload_digest(dict(item)))
            timestamps.append(float(item.get("timestamp", 0.0)))
        return InputProvenance(input_refs=tuple(refs), input_digests=tuple(digests),
                               timestamps=tuple(timestamps))

    def _audit(self, result: InferenceResult) -> None:
        """Emit one structured audit event per execution (Step 54)."""
        if self.audit_hook is not None:
            self.audit_hook("execute", {
                "operation": "execute", "status": result.status,
                "task_id": result.task_id, "model_id": result.model_id,
                "model_version": result.model_version,
                "artifact_digest": result.artifact_digest,
                "failure_category": result.failure_category,
                "duration_ms": result.execution.wall_clock_ms,
                "detail": f"latency={result.execution.wall_clock_ms:.1f}ms"})


def _schema_check_payload(output: Mapping[str, Any] | None,
                          schema: SchemaDescriptor,
                          status: str) -> "Any":  # ValidationReport
    """Build the schema-validation portion of the result's validation report."""
    from ..contracts import ValidationReport

    if status == "completed":
        return ValidationReport(checks=(DeterministicCheck(
            check_id="output-schema", validator="adapter-schema-validator",
            outcome=CheckOutcome.PASS, detail="output satisfies the declared schema"),))
    return ValidationReport(checks=(DeterministicCheck(
        check_id="output-schema", validator="adapter-schema-validator",
        outcome=CheckOutcome.FAIL,
        detail="no schema-valid output; status " + status),))


def _payload_confidence(output: Mapping[str, Any]) -> float:
    """Extract a declared confidence from a validated payload, else a neutral 0.5."""
    for key in ("confidence", "score", "probability"):
        value = output.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(min(1.0, max(0.0, float(value))))
    return 0.5


class _HardTimeout:
    """A best-effort SIGALRM guard for non-cooperating adapters (main thread only)."""

    @classmethod
    def install(cls, seconds: float) -> tuple[bool, Any]:
        """Arm the alarm when possible; returns the restore token."""
        try:
            if threading.current_thread() is not threading.main_thread():
                return (False, None)
            if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
                return (False, None)

            def handler(signum: int, frame: Any) -> None:
                raise TimeoutError("hard deadline exceeded")

            previous = signal.signal(signal.SIGALRM, handler)
            signal.setitimer(signal.ITIMER_REAL, max(seconds, 0.001))
            return (True, previous)
        except (ValueError, OSError):
            return (False, None)

    @classmethod
    def restore(cls, token: tuple[bool, Any]) -> None:
        """Disarm the alarm and restore the previous handler."""
        armed, previous = token
        if not armed:
            return
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous)  # type: ignore[arg-type]
        except (ValueError, OSError):
            pass


def execute_with_limits(request: InferenceRequest, adapter: SpecialistAdapter,
                        config: LearningConfig | None = None,
                        **kwargs: Any) -> InferenceResult:
    """Convenience wrapper: build an executor and run one request."""
    return InferenceExecutor(config).execute(request, adapter, **kwargs)
