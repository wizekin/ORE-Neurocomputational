"""Typed reasoning steps and PRM screening (Point 14, Steps 32-33).

Multi-step tasks are represented as typed transitions -- evidence selection, semantic
mapping, rule application, calculation, intermediate result, conclusion -- each with
input references, expected invariants, and a validator where possible.  A Process
Reward Model scores whether each step looks promising; the score is a learned
screening signal stored separately from validator results, never the final verdict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..contracts import PRMDiagnostic, ValidationReport

__all__ = ("PRMModel", "ReasoningStep", "ReasoningTrace", "StepKind", "StepStatus")


class StepKind(str, __import__("enum").Enum):
    """The typed transitions of a reasoning process (Step 32)."""

    EVIDENCE_SELECTION = "evidence_selection"
    SEMANTIC_MAPPING = "semantic_mapping"
    RULE_APPLICATION = "rule_application"
    CALCULATION = "calculation"
    INTERMEDIATE_RESULT = "intermediate_result"
    CONCLUSION = "conclusion"


class StepStatus(str, __import__("enum").Enum):
    """Per-step status; only deterministic PASS yields ``verified`` (Step 34)."""

    PENDING = "pending"
    MODEL_ASSESSED = "model_assessed"
    HUMAN_REVIEWED = "human_reviewed"
    VERIFIED = "verified"
    FAILED = "failed"


@dataclass(slots=True)
class ReasoningStep:
    """One typed transition with inputs, invariants, and its validators (Step 32)."""

    step_id: str
    step_kind: StepKind
    description: str
    input_refs: tuple[str, ...]
    expected_invariants: tuple[str, ...]
    output_ref: str = ""
    validator_name: str = ""
    validator_result: str = ""  # pass | fail | unavailable
    prm_score: float | None = None
    status: StepStatus = StepStatus.PENDING

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view; PRM scores and validator results stay separate fields."""
        return {"step_id": self.step_id, "step_kind": self.step_kind.value,
                "description": self.description,
                "input_refs": list(self.input_refs),
                "expected_invariants": list(self.expected_invariants),
                "output_ref": self.output_ref,
                "validator_name": self.validator_name,
                "validator_result": self.validator_result,
                "prm_score": self.prm_score,
                "status": self.status.value}


@dataclass(slots=True)
class ReasoningTrace:
    """A typed multi-step reasoning process (Step 32)."""

    task_id: str
    steps: list[ReasoningStep] = field(default_factory=list)

    def add(self, step: ReasoningStep) -> ReasoningStep:
        """Append one step."""
        self.steps.append(step)
        return step

    @property
    def verified(self) -> bool:
        """True only when every step passed its deterministic validator (Step 34)."""
        return bool(self.steps) and all(
            step.validator_result == "pass" for step in self.steps)

    def prm_diagnostics(self, prm_model_id: str = "prm.reference@1.0.0"
                        ) -> tuple[Any, ...]:
        """The PRM scores as separate diagnostics, never mixed into verification."""
        return tuple(PRMDiagnostic(step_id=step.step_id, score=step.prm_score,
                                   calibrated=False, prm_model_id=prm_model_id)
                     for step in self.steps if step.prm_score is not None)

    def status_counts(self) -> dict[str, int]:
        """Counts by step status."""
        counts: dict[str, int] = {}
        for step in self.steps:
            counts[step.status.value] = counts.get(step.status.value, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"task_id": self.task_id, "steps": [step.to_dict() for step in self.steps],
                "verified": self.verified,
                "status_counts": self.status_counts()}


class PRMModel:
    """Step 33 -- a Process Reward Model used purely as a screening signal.

    The reference implementation scores steps from declared signal features (invariant
    count, validator presence, step kind priors); the important property is the
    boundary: PRM output lands in ``prm_score`` and in PRM diagnostics, never in
    ``validator_result`` or the verified flag.
    """

    model_id: str = "prm.reference@1.0.0"

    #: Documented kind priors: steps with cheap deterministic validators are easier
    #: to screen reliably than open-ended conclusions.
    KIND_PRIORS: dict[str, float] = {
        StepKind.EVIDENCE_SELECTION.value: 0.6,
        StepKind.SEMANTIC_MAPPING.value: 0.55,
        StepKind.RULE_APPLICATION.value: 0.7,
        StepKind.CALCULATION.value: 0.8,
        StepKind.INTERMEDIATE_RESULT.value: 0.65,
        StepKind.CONCLUSION.value: 0.5,
    }

    def score_step(self, step: ReasoningStep) -> float:
        """Estimate whether the step is promising; a screening signal only."""
        prior = self.KIND_PRIORS.get(step.step_kind.value, 0.5)
        invariant_bonus = 0.05 * min(len(step.expected_invariants), 3)
        validator_bonus = 0.1 if step.validator_name else 0.0
        evidence_bonus = 0.05 if len(step.input_refs) >= 2 else 0.0
        return float(min(1.0, max(0.0, prior + invariant_bonus + validator_bonus
                                  + evidence_bonus)))

    def screen(self, trace: ReasoningTrace, *,
               threshold: float = 0.5) -> dict[str, Any]:
        """Score every step and return the screening summary with calibration metadata."""
        for step in trace.steps:
            step.prm_score = self.score_step(step)
        flagged = [step.step_id for step in trace.steps
                   if (step.prm_score or 0.0) < threshold]
        return {"model_id": self.model_id, "threshold": threshold,
                "scored_steps": len(trace.steps),
                "screening_flags": flagged,
                "calibration_metadata": {"calibrated": False,
                                         "method": "reference priors",
                                         "note": "screening signal; deterministic "
                                                 "checks take precedence"},
                "note": "PRM scores are stored per-step and never override validator "
                        "results"}
