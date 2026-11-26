"""Unified result envelopes and independent integrity validation (Steps 42-43).

Outputs from Datalog, SAT/SMT, CP-SAT and the proof kernel are normalized into
one typed result schema without flattening their distinct semantics. The
validator re-checks schema conformance, source links, proof references, solver
identity, model decoding, assumption disclosure, and provenance — and replays
certificates where possible (proof hashes are recomputed; models are
re-evaluated by the independent evaluator).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import (
    FormalResult,
    ResultKind,
    ResultStatus,
    sha256_json,
)
from .proofs import ProofCandidate, ProofObligation, ProofVerifier, TrustedKernel

_SUPPORTED_SCHEMA_VERSIONS = frozenset({"FRCE-RESULT/1.0.0"})

_KIND_STATUSES: Mapping[ResultKind, frozenset[ResultStatus]] = {
    ResultKind.RETRIEVED_SOURCE_FACT: frozenset({
        ResultStatus.SATISFIABLE, ResultStatus.UNKNOWN,
        ResultStatus.INVALID_INPUT, ResultStatus.ERROR,
    }),
    ResultKind.RULE_DERIVED_INFERENCE: frozenset({
        ResultStatus.SATISFIABLE, ResultStatus.UNKNOWN,
        ResultStatus.TIMEOUT, ResultStatus.UNSUPPORTED,
        ResultStatus.INVALID_INPUT, ResultStatus.ERROR,
    }),
    ResultKind.SATISFIABILITY_RESULT: frozenset({
        ResultStatus.SATISFIABLE, ResultStatus.UNSATISFIABLE,
        ResultStatus.UNKNOWN, ResultStatus.TIMEOUT, ResultStatus.UNSUPPORTED,
        ResultStatus.INVALID_INPUT, ResultStatus.ERROR,
    }),
    ResultKind.OPTIMIZATION_RESULT: frozenset({
        ResultStatus.OPTIMAL, ResultStatus.FEASIBLE, ResultStatus.INFEASIBLE,
        ResultStatus.UNKNOWN, ResultStatus.TIMEOUT, ResultStatus.UNSUPPORTED,
        ResultStatus.INVALID_INPUT, ResultStatus.ERROR,
    }),
    ResultKind.MACHINE_CHECKED_PROOF: frozenset({
        ResultStatus.PROVED, ResultStatus.UNKNOWN, ResultStatus.TIMEOUT,
        ResultStatus.UNSUPPORTED, ResultStatus.INVALID_INPUT, ResultStatus.ERROR,
    }),
    ResultKind.EMPIRICAL_OBSERVATION: frozenset({
        ResultStatus.SATISFIABLE, ResultStatus.UNKNOWN, ResultStatus.INVALID_INPUT,
        ResultStatus.ERROR,
    }),
}


@dataclass(frozen=True, slots=True)
class EnvelopeCheck:
    """One validation check with its outcome (Step 43)."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class EnvelopeValidation:
    """The aggregate validation report for one result envelope."""

    valid: bool
    checks: tuple[EnvelopeCheck, ...]
    errors: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(self.checks))
        object.__setattr__(self, "errors", tuple(self.errors))


def validate_result_envelope(result: FormalResult) -> EnvelopeValidation:
    """Validate one result envelope against its versioned schema (Step 43)."""
    checks: list[EnvelopeCheck] = []
    errors: list[str] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append(EnvelopeCheck(name, passed, detail))
        if not passed:
            errors.append(f"{name}: {detail}")

    add(
        "schema_version",
        result.schema_version in _SUPPORTED_SCHEMA_VERSIONS,
        f"envelope declares {result.schema_version!r}",
    )
    allowed = _KIND_STATUSES.get(result.kind, frozenset())
    add(
        "kind_status_compatibility",
        result.status in allowed,
        f"kind {result.kind.value} with status {result.status.value}",
    )
    add(
        "non_success_never_false",
        not (result.status.value in {"unknown", "timeout", "unsupported", "error", "invalid-input"}
             and result.establishes_false),
        "non-success statuses never carry falsity claims",
    )
    provenance_count = len(result.provenance) + len(result.statement_ir_refs)
    add(
        "provenance_present",
        provenance_count > 0,
        f"{len(result.provenance)} source links, "
        f"{len(result.statement_ir_refs)} IR references",
    )
    if result.solver is not None:
        add(
            "solver_identity",
            bool(result.solver.name) and bool(result.solver.version),
            f"{result.solver.name} {result.solver.version}",
        )
    else:
        solver_backed = result.kind in {
            ResultKind.SATISFIABILITY_RESULT, ResultKind.OPTIMIZATION_RESULT,
        }
        add(
            "solver_identity",
            not solver_backed,
            "solver-backed results must record solver identity",
        )
    if result.model is not None:
        add(
            "model_independently_validated",
            result.model_independently_validated is True,
            "decoded models must be independently validated before use",
        )
    if result.kind == ResultKind.MACHINE_CHECKED_PROOF and result.status == ResultStatus.PROVED:
        add(
            "proof_kernel_checked",
            result.proof_check_status == "kernel_checked" and result.proof_ref is not None,
            f"proof_check_status={result.proof_check_status!r}",
        )
        add(
            "conditional_disclosure",
            len(result.assumptions) > 0 or result.statement.startswith("unconditional"),
            "proofs must disclose the assumptions they are conditional on",
        )
    if result.unsat_core and not result.unsat_core_minimized:
        add(
            "core_not_overclaimed",
            "not necessarily a minimum-cardinality cause" in " ".join(result.limitations),
            "a non-minimized core must be labeled as sufficient, not minimal",
        )
    add(
        "reproducibility_present",
        bool(result.reproducibility),
        f"keys: {sorted(result.reproducibility)}",
    )
    return EnvelopeValidation(
        valid=not errors, checks=tuple(checks), errors=tuple(errors),
    )


def result_fingerprint(result: FormalResult) -> str:
    """Compute the reproducibility fingerprint of one result (Section 3)."""
    return sha256_json(result.to_dict())


class CertificateReplay:
    """Replay certificates to re-verify claimed results (Step 43)."""

    def __init__(self, verifier: ProofVerifier | None = None) -> None:
        self._verifier = verifier or ProofVerifier()

    def replay_proof(
        self,
        obligation: ProofObligation,
        candidate: ProofCandidate,
        expected_hash: str | None = None,
    ) -> EnvelopeValidation:
        """Re-check a proof candidate with the trusted kernel and compare hashes."""
        kernel = TrustedKernel()
        context = {p.name: p for p in obligation.premises}
        verdict = kernel.check(context, candidate.term, obligation.goal)
        checks = [EnvelopeCheck(
            "proof_kernel_replay", verdict.accepted, verdict.detail,
        )]
        from .proofs import proof_term_text, proposition_text

        # The recomputed hash must match the verifier's artifact hash formula:
        # sha256(proposition_text(goal) + proof_term_text(term)).
        recomputed = __import__("hashlib").sha256(
            (proposition_text(obligation.goal)
             + proof_term_text(candidate.term)).encode("utf-8")
        ).hexdigest()
        if expected_hash is not None:
            checks.append(EnvelopeCheck(
                "proof_hash_match",
                recomputed == expected_hash,
                f"recomputed {recomputed[:12]}… vs recorded {expected_hash[:12]}…",
            ))
        errors = tuple(
            f"{c.name}: {c.detail}" for c in checks if not c.passed
        )
        return EnvelopeValidation(
            valid=not errors, checks=tuple(checks), errors=errors,
        )

    def replay_model(
        self,
        result: FormalResult,
        statements: Sequence[Any],
    ) -> EnvelopeValidation:
        """Independently re-evaluate a decoded model against the IR statements."""
        from .smt import IndependentEvaluator

        evaluator = IndependentEvaluator()
        if result.model is None:
            return EnvelopeValidation(
                valid=False,
                checks=(EnvelopeCheck(
                    "model_replay", False, "no decoded model present",
                ),),
                errors=("model_replay: no decoded model present",),
            )
        assignment = {}
        for key, value in result.model.items():
            if isinstance(value, Mapping) and "numerator" in value:
                from fractions import Fraction

                assignment[key] = Fraction(value["numerator"], value["denominator"])
            else:
                assignment[key] = value
        checks: list[EnvelopeCheck] = []
        for statement in statements:
            outcome = evaluator.evaluate_formula(statement.formula, assignment)
            checks.append(EnvelopeCheck(
                f"model_replay_{statement.statement_id}",
                outcome is True,
                f"independent evaluation returned {outcome}",
            ))
        errors = tuple(f"{c.name}: {c.detail}" for c in checks if not c.passed)
        return EnvelopeValidation(
            valid=not errors, checks=tuple(checks), errors=errors,
        )


__all__ = [
    "EnvelopeCheck",
    "EnvelopeValidation",
    "validate_result_envelope",
    "result_fingerprint",
    "CertificateReplay",
]
