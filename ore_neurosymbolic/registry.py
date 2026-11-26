"""Stable operation registry for the ORE Neuro-Symbolic Reasoning Layer (Point 3).

Every operation ID below corresponds to one ordered step of the 45-step
integrated algorithm in ``ORE —Neurocomputational Point 03 Neuro-Symbolic
Reasoning Algorithm.txt`` (Section 5, Phases A-H). IDs follow the
repository-wide traceability convention: stable, never recycled, usable for
runtime tracing (``NSR-001`` .. ``NSR-045``).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one NSRL operation and its source step."""

    operation_id: str
    step: int
    phase: str
    title: str
    purpose: str


_NSRL_OPERATIONS: tuple[tuple[int, str, str, str], ...] = (
    (1, "A", "Receive and validate the reasoning request",
     "Parse goal predicate, scope, entity references, mode, deadline and verification level; reject invalid scope or absent budgets."),
    (2, "A", "Bind one consistent workspace snapshot revision",
     "Read one revision, record it in the run, and never mix revisions implicitly."),
    (3, "A", "Resolve semantic IDs while preserving ambiguity",
     "Map words/references to USG concepts and predicates, keeping multiple mappings."),
    (4, "A", "Retrieve facts and rules through exact indexes",
     "Use exact keys and relation indexes first; similarity expansion carries source/version references."),
    (5, "A", "Normalize canonical forms and retain source form",
     "Canonicalize IDs, direction, types, units, polarity and temporal qualifiers."),
    (6, "A", "Assign explicit epistemic status",
     "Distinguish observed evidence, asserted fact, derived fact, hypothesis, probabilistic claim and unknown."),
    (7, "A", "Assemble the bounded reasoning context",
     "Bind snapshot reference, selected facts/rules, goal, constraints and budgets over existing workspace context."),
    (8, "B", "Validate the ruleset against the safe profile",
     "Check arity, bound variables, type compatibility, operators, scope, exceptions, priority and recursion policy."),
    (9, "B", "Select goal-relevant rules with recorded rationale",
     "Index by predicate/type and goal relevance; record why each rule was selected."),
    (10, "B", "Unify premises with typed substitutions",
     "Perform typed substitutions, reject type mismatches, and preserve bounded diagnostics."),
    (11, "B", "Evaluate rule conjunctions through the strict firing gate",
     "Fire(r,theta)=1 iff every premise is supported and type-valid; defeasible rules also need valid scope and no exception."),
    (12, "B", "Instantiate conclusions with rule version and premises",
     "Attach rule version and premise references to every candidate conclusion immediately."),
    (13, "B", "Resolve defeasible exceptions by specificity and priority",
     "Apply the explicit policy; incomparable conflicting rules produce conflict records."),
    (14, "B", "Forward chain to fixed point under hard budgets",
     "Process new facts in a deduplicating queue until fixed point, goal completion or budget exhaustion."),
    (15, "B", "Backward chain goal-directed queries with memoization",
     "Expand rules able to produce the goal; memoize subgoals with shared proof semantics."),
    (16, "B", "Detect and stop unsafe cycles with explicit partial status",
     "Track active goals/substitutions/states and return partial status instead of unsafe recursion."),
    (17, "C", "Build the content-addressed derivation DAG",
     "Reference applied rules and premise nodes; reuse identical content-addressed nodes."),
    (18, "C", "Validate the proof graph structurally",
     "Ensure every premise and rule version resolves and every substitution is type-correct."),
    (19, "C", "Independently re-execute strict rule applications",
     "A separate verifier recomputes each strict application without trusting engine claims."),
    (20, "C", "Trace source lineage to original records",
     "Recursively resolve premises to source/evidence records and the workspace revision."),
    (21, "C", "Classify proofs into non-conflatable labels",
     "Assign strict verified, defeasible, probabilistic support, neural proposal or unverified candidate."),
    (22, "D", "Generate the finite candidate set",
     "Combine lexical/USG alternatives with grammar, type, context and relation constraints."),
    (23, "D", "Apply hard constraints to proven violations only",
     "Eliminate candidates proven to violate explicit constraints; keep everything else."),
    (24, "D", "Apply soft constraints and rank candidates",
     "Score context fit, coherence and evidence support; soft preferences rank but never prove."),
    (25, "D", "Preserve ambiguity with alternatives and discriminating evidence",
     "Publish close alternatives instead of forcing a single interpretation."),
    (26, "E", "Wrap model predictions as explicit evidence",
     "Include model/version, input hash, score semantics and calibration; raw softmax is not a probability."),
    (27, "E", "Separate probabilistic observations from crisp facts",
     "Probabilistic observations are never stored as ordinary true facts."),
    (28, "E", "Combine evidence under declared dependence assumptions",
     "Multiply only under justified conditional independence; dependent evidence uses joint models."),
    (29, "E", "Propagate uncertainty without fabricated precision",
     "Use documented probabilistic rule semantics; otherwise report qualitative support."),
    (30, "E", "Publish the decomposed confidence vector",
     "Keep source, observation, rule, coverage and conflict dimensions separate (Section 6.5)."),
    (31, "F", "Invoke learned models only when needed",
     "Request similarity/retrieval when deterministic indexes are insufficient; choose by capability, quality, cost and policy."),
    (32, "F", "Validate model responses against policy",
     "Enforce schema, version, input hash, score range/meaning, calibration, latency and policy."),
    (33, "F", "Convert model outputs to hypotheses",
     "Learned outputs become proposals with model-proposal epistemic status, never authoritative facts."),
    (34, "F", "Filter proposals symbolically with bounded reasons",
     "Apply type/rule/constraint/evidence checks and retain bounded rejection reasons."),
    (35, "F", "Learn only from verified feedback",
     "Update routing/ranking from confirmed outcomes; never self-label unverified inferences as ground truth."),
    (36, "G", "Detect contradictions",
     "Compare findings with incompatible facts, rules, temporal scopes and exceptions."),
    (37, "G", "Record conflicts with provenance and resolution options",
     "Include claims, provenance, priority, time scope and options; never overwrite source facts."),
    (38, "G", "Apply publication gates",
     "Require proof validity, evidence sufficiency, permitted epistemic status, critical constraints and conflict levels."),
    (39, "G", "Publish through existing workspace APIs",
     "Send structured findings, proof references, lineage, support vectors and the snapshot revision through the atomic path."),
    (40, "G", "Request independent review for high-impact findings",
     "Send premises and proof to another module for high-impact, novel or contested conclusions."),
    (41, "H", "Log annotations and events through the existing logger",
     "Record rule fires, model calls, proof verification, constraint decisions, conflicts, resource use and publication status."),
    (42, "H", "Measure quality and cost",
     "Compute proof validity, contradiction rate, ambiguity, precision/recall where labeled, latency, fires, memory and overhead."),
    (43, "H", "Compare predictions with verified outcomes only",
     "Record prediction error only when genuine ground truth or independently verified feedback exists."),
    (44, "H", "Update approved parameters with bounded versioned changes",
     "Adapt retrieval/routing/ranking weights from verified feedback; authoritative rules need separate governance."),
    (45, "H", "Checkpoint safe run state and recover explicitly",
     "Persist safe state; on timeout return explicit partial status/resume information; never publish incomplete work."),
)

_OPERATION_ID_PREFIX = "NSR"


def _operation_id(step: int) -> str:
    """Format the stable operation ID for one source step."""
    return f"{_OPERATION_ID_PREFIX}-{step:03d}"


NSRL_OPERATIONS: tuple[OperationSpec, ...] = tuple(
    OperationSpec(
        operation_id=_operation_id(step), step=step, phase=phase,
        title=title, purpose=purpose,
    )
    for step, phase, title, purpose in _NSRL_OPERATIONS
)

NSRL_OPERATION_IDS: tuple[str, ...] = tuple(item.operation_id for item in NSRL_OPERATIONS)

PHASE_STEP_RANGES: dict[str, tuple[int, int]] = {
    "A": (1, 7),
    "B": (8, 16),
    "C": (17, 21),
    "D": (22, 25),
    "E": (26, 30),
    "F": (31, 35),
    "G": (36, 40),
    "H": (41, 45),
}


def operation_by_step(step: int) -> OperationSpec:
    """Return the operation specification for one algorithm step number."""
    if not isinstance(step, int) or not 1 <= step <= len(NSRL_OPERATIONS):
        raise ValueError(f"unknown neuro-symbolic algorithm step: {step!r}")
    return NSRL_OPERATIONS[step - 1]


def operation_by_id(operation_id: str) -> OperationSpec:
    """Return the operation specification for one stable operation ID."""
    for item in NSRL_OPERATIONS:
        if item.operation_id == operation_id:
            return item
    raise ValueError(f"unknown neuro-symbolic operation ID: {operation_id!r}")


__all__ = [
    "NSRL_OPERATIONS",
    "NSRL_OPERATION_IDS",
    "OperationSpec",
    "PHASE_STEP_RANGES",
    "operation_by_id",
    "operation_by_step",
]
