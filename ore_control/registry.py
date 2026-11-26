"""Stable operation registry for the ORE integrated cognitive control algorithm.

Every operation ID below corresponds to one numbered step of the algorithm in
``docs/COGNITIVE_ARCHITECTURES_POINT02_ALGORITHM.md`` (ORE Integrated Algorithm
Point 02: cognitive architectures — Soar, ACT-R and CLARION). The IDs follow the
repository-wide traceability convention: stable, never recycled, and usable for
runtime tracing alongside the blackboard registry ``BBA-001..BBA-032``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one cognitive-control operation and its source step."""

    operation_id: str
    step: int
    title: str
    purpose: str


_CONTROL_OPERATIONS: tuple[tuple[int, str, str, str], ...] = (
    (1, "Receive a cognitive trigger",
     "Normalize a trigger into a typed envelope with an idempotency key; reject malformed triggers."),
    (2, "Read an authoritative workspace snapshot",
     "Decide from one recorded revision; later publication uses the workspace revision protocol."),
    (3, "Resolve the governing goal",
     "Map the trigger to an authoritative goal with success conditions and critical requirements."),
    (4, "Build or resume the CognitiveContext",
     "Create or resume a version-compatible context; reconcile stale contexts before continuing."),
    (5, "Derive the current cognitive state",
     "Summarize satisfied and unsatisfied conditions, open questions, and a stable state fingerprint."),
    (6, "Query the operator registry",
     "Load operators compatible with the task; unavailable operators are never eligible."),
    (7, "Evaluate operator preconditions",
     "Evaluate applicability predicates and record machine-readable failure reasons."),
    (8, "Generate candidate cognitive actions",
     "Create typed candidate actions across retrieval, hypothesis, verification, and planning."),
    (9, "Deduplicate and constrain candidates",
     "Deduplicate by operator identity, arguments, and revision; remove unsafe or satisfied work."),
    (10, "Construct retrieval cues",
     "Build traceable cues from the goal, open questions, hypotheses, and evidence."),
    (11, "Identify retrieval domains",
     "Select only relevant stores and prefer cheap cache or index coverage first."),
    (12, "Generate candidate memories",
     "Retrieve a bounded candidate set with permissions, versions, and provenance checks."),
    (13, "Estimate base-level activation",
     "Compute the ACT-R-inspired B_i recency/frequency activation or mark history unavailable."),
    (14, "Estimate cue association",
     "Compute S_i from weighted cue associations with per-cue contribution traces."),
    (15, "Combine relevance, confidence and cost",
     "Score candidates with R_i while handling missing features explicitly."),
    (16, "Preserve retrieval competition and ambiguity",
     "Retain a bounded shortlist when scores are close; record ambiguity instead of forcing a winner."),
    (17, "Choose a retrieval depth under budget",
     "Stop when marginal expected gain no longer justifies cost; escalate stores only when justified."),
    (18, "Estimate operator outcomes",
     "Estimate progress probability, value, cost, and risk from measured data or labelled priors."),
    (19, "Calculate expected utility",
     "Compute EU(o) and the richer normalized U(o) as labelled ORE design scores."),
    (20, "Apply hard constraints before ranking",
     "Reject operators that violate budgets, permissions, or critical goal conditions first."),
    (21, "Select one action or a safe parallel set",
     "Select deterministically with stable tie-breaking; log alternatives and rationale."),
    (22, "Detect whether the action requires decomposition",
     "Create child contexts only when dependencies and expected benefit justify splitting."),
    (23, "Define each child goal and contract",
     "Give each child a narrow goal, references, allowed operators, budget, and success predicate."),
    (24, "Enforce subproblem boundaries",
     "Enforce depth, child-count, time, and operation limits; keep children inside their contracts."),
    (25, "Reconcile child outcomes with the parent",
     "Validate child contracts, merge compatible findings, and preserve conflicts explicitly."),
    (26, "Classify the processing path",
     "Route each action to explicit symbolic, learned statistical, or hybrid processing."),
    (27, "Check learned-component suitability",
     "Verify model identity, schemas, task support, provenance, and calibration before invocation."),
    (28, "Keep proposals separate from authoritative knowledge",
     "Treat learned outputs as proposals that require validation before publication."),
    (29, "Apply meta-cognitive monitoring",
     "Track uncertainty, disagreement, failures, and stale evidence; escalate instead of forcing certainty."),
    (30, "Reserve resources and dispatch",
     "Reserve resources and delegate planning and execution to the existing coordinators."),
    (31, "Execute with checkpoints and event capture",
     "Run the operator handler with start, progress, completion, timeout, and failure events."),
    (32, "Validate outputs against contracts",
     "Check schema, provenance, postconditions, and operator-specific validation."),
    (33, "Integrate validated findings atomically",
     "Submit findings through the authoritative workspace path and record the exact decision."),
    (34, "Recompute goal progress",
     "Evaluate weighted progress while keeping critical requirements as hard gates."),
    (35, "Detect stagnation and cycles",
     "Compare state fingerprints and repeated patterns; apply retry limits and backoff."),
    (36, "Decide whether to continue, revise, wait or stop",
     "Continue only while useful eligible action and budget remain; stop with structured reasons."),
    (37, "Identify a candidate reusable procedure",
     "Extract stable operator sequences only from verified, repeatable tasks."),
    (38, "Evaluate generalization and safety",
     "Test candidates on held-out examples and counterexamples; reject memorization."),
    (39, "Version and register the procedure as non-authoritative",
     "Store versioned procedures with provenance, test results, and rollback information."),
    (40, "Reuse cautiously and monitor outcomes",
     "Re-check preconditions on each reuse; disable, roll back, or quarantine degraded procedures."),
)


def _operation_id(step: int) -> str:
    """Format the stable operation ID for one source step."""
    return f"CCA-{step:03d}"


CONTROL_OPERATIONS: tuple[OperationSpec, ...] = tuple(
    OperationSpec(
        operation_id=_operation_id(step), step=step, title=title, purpose=purpose,
    )
    for step, title, purpose in _CONTROL_OPERATIONS
)

CONTROL_OPERATION_IDS: tuple[str, ...] = tuple(
    item.operation_id for item in CONTROL_OPERATIONS
)


def operation_by_step(step: int) -> OperationSpec:
    """Return the operation specification for one algorithm step number."""
    if not isinstance(step, int) or not 1 <= step <= len(CONTROL_OPERATIONS):
        raise ValueError(f"unknown cognitive control algorithm step: {step!r}")
    return CONTROL_OPERATIONS[step - 1]


def operation_by_id(operation_id: str) -> OperationSpec:
    """Return the operation specification for one stable operation ID."""
    for item in CONTROL_OPERATIONS:
        if item.operation_id == operation_id:
            return item
    raise ValueError(f"unknown cognitive control operation ID: {operation_id!r}")


__all__ = [
    "CONTROL_OPERATION_IDS",
    "CONTROL_OPERATIONS",
    "OperationSpec",
    "operation_by_id",
    "operation_by_step",
]
