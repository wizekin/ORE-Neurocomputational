"""Gating of generated outputs (Point 14, Step 26).

Generated text, code, data, and hypotheses enter the workspace as untrusted
candidates: schema, provenance, safety, and task-specific validation apply, and
generated code executes only inside the least-privilege sandbox.  Nothing generated
is ever treated as a validated fact or a canonical record.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..adapters import validate_payload_against_schema
from ..contracts import CheckOutcome, DeterministicCheck, SchemaDescriptor, ValidationReport
from ..inference import LeastPrivilegeSandbox

__all__ = ("GeneratedCandidate", "GeneratedOutputGate", "generate_id")


def generate_id(kind: str, sequence: int) -> str:
    """A stable candidate identifier."""
    return f"gen-{kind}-{sequence:05d}"


@dataclass(slots=True)
class GeneratedCandidate:
    """One untrusted generated candidate with its gate verdict."""

    candidate_id: str
    kind: str  # text | code | data | hypothesis
    payload: dict[str, Any]
    objective_kind: str
    generator_digest: str
    checks: list[DeterministicCheck]
    sandboxed: bool = False
    sandbox_result: Any = None
    admitted: bool = False

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view (code text is referenced, not copied, in summaries)."""
        return {"candidate_id": self.candidate_id, "kind": self.kind,
                "objective_kind": self.objective_kind,
                "generator_digest": self.generator_digest,
                "admitted": self.admitted, "sandboxed": self.sandboxed,
                "checks": [check.to_dict() for check in self.checks],
                "sandbox_status": (self.sandbox_result.status
                                   if self.sandbox_result else None)}


class GeneratedOutputGate:
    """Step 26 -- gate every generated output before it can travel anywhere."""

    def __init__(self, sandbox: LeastPrivilegeSandbox | None = None) -> None:
        """Configure the sandbox used for generated code."""
        self.sandbox = sandbox or LeastPrivilegeSandbox()
        self.sequence = 0
        self.gated: list[GeneratedCandidate] = []

    def gate(self, kind: str, payload: Mapping[str, Any], *,
             objective_kind: str, generator_digest: str,
             schema: SchemaDescriptor | None = None,
             safety_rules: tuple[str, ...] = ("no_canonical_claims",)) -> GeneratedCandidate:
        """Validate one generated payload as an untrusted candidate.

        Schema validation, provenance attachment, and safety rules run as
        deterministic checks; generated code additionally runs only in the sandbox.
        The candidate never becomes a fact: it enters the workspace as a proposal.
        """
        self.sequence += 1
        candidate_id = generate_id(kind, self.sequence)
        checks: list[DeterministicCheck] = [
            DeterministicCheck(check_id="provenance", validator="generation-gate",
                               outcome=CheckOutcome.PASS,
                               detail=f"objective={objective_kind}; "
                                      f"generator={generator_digest}")]

        if schema is not None:
            ok, reason = validate_payload_against_schema(payload, schema)
            checks.append(DeterministicCheck(
                check_id="schema", validator="adapter-schema-validator",
                outcome=CheckOutcome.PASS if ok else CheckOutcome.FAIL, detail=reason))

        safety_failures = [rule for rule in safety_rules
                           if rule == "no_canonical_claims"
                           and str(payload.get("claim", "")).lower().startswith("fact:")]
        checks.append(DeterministicCheck(
            check_id="safety", validator="generation-gate-safety",
            outcome=CheckOutcome.FAIL if safety_failures else CheckOutcome.PASS,
            detail=("generated candidates may not assert canonical facts"
                    if safety_failures else "safety rules satisfied")))

        sandboxed = False
        sandbox_result = None
        if kind == "code":
            sandboxed = True
            sandbox_result = self.sandbox.execute(str(payload.get("code_text", "")))
            checks.append(DeterministicCheck(
                check_id="sandbox", validator="least-privilege-sandbox",
                outcome=(CheckOutcome.PASS if sandbox_result.status == "executed"
                         else CheckOutcome.FAIL),
                detail=f"sandbox status {sandbox_result.status}; "
                       f"violations={sandbox_result.violations}"))

        admitted = all(check.outcome is not CheckOutcome.PASS or True
                       for check in checks) and all(
                check.outcome is not CheckOutcome.FAIL for check in checks)
        candidate = GeneratedCandidate(
            candidate_id=candidate_id, kind=kind, payload=dict(payload),
            objective_kind=objective_kind, generator_digest=generator_digest,
            checks=checks, sandboxed=sandboxed, sandbox_result=sandbox_result,
            admitted=admitted)
        self.gated.append(candidate)
        return candidate

    def report(self, candidate: GeneratedCandidate) -> ValidationReport:
        """Wrap a candidate's checks as a ValidationReport for the workspace."""
        return ValidationReport(checks=tuple(candidate.checks))

    def describe(self) -> dict[str, Any]:
        """Summarize gate activity."""
        return {"gated": len(self.gated),
                "admitted": sum(1 for item in self.gated if item.admitted),
                "rejected": sum(1 for item in self.gated if not item.admitted),
                "code_sandboxed": sum(1 for item in self.gated if item.sandboxed),
                "note": "generated outputs are untrusted candidates by construction"}
