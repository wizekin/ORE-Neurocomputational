"""Stable operation registry for the ORE Formal Reasoning and Constraint Engine.

Every operation ID below corresponds to one numbered step of ``ORE Integrated
Algorithm — Point 5: Automated Reasoning, Theorem Proving, SAT/SMT Solvers and
Constraint-Based Intelligence`` (52 steps in 8 phases A-H). The IDs follow the
repository-wide traceability convention: stable, never recycled, and usable for
runtime tracing, audit records, and capability-flag gating.
"""
from __future__ import annotations

from dataclasses import dataclass

PHASES: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H")


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one FRCE operation, its source step, phase, and purpose."""

    operation_id: str
    step: int
    phase: str
    title: str
    purpose: str


# (step, phase, title, purpose) — one entry per numbered step of the spec.
_FRCE_OPERATIONS: tuple[tuple[int, str, str, str], ...] = (
    (1, "A", "Receive a formal reasoning task",
     "Accept a narrowly stated objective and reject goals that cannot be operationalized."),
    (2, "A", "Classify the reasoning problem",
     "Route the request into rule inference, consistency, theory constraints, optimization, proof, verification, or mixed reasoning."),
    (3, "A", "Resolve semantic entities against USG",
     "Map referenced concepts, objects, relations, predicates and rules to stable USG identifiers while preserving ambiguity."),
    (4, "A", "Retrieve only relevant knowledge",
     "Assemble the smallest sufficient fact/rule set via typed relations, neighborhoods and rule dependencies; record query parameters."),
    (5, "A", "Separate facts from assumptions and hypotheses",
     "Tag inputs as observed facts, stipulated assumptions, inferred results, hypotheses, or policy constraints."),
    (6, "A", "Preserve provenance and temporal scope",
     "Attach source IDs, timestamps, versions, authority metadata and validity intervals; never collapse scoped facts."),
    (7, "A", "Check completeness and ambiguity",
     "Detect missing values, undefined predicates, ambiguous quantifiers, unit mismatches and underspecified domains; request clarification."),
    (8, "B", "Construct a typed logical intermediate representation",
     "Map USG objects to constants, classes to predicates, relations to predicates/functions, rules to formulas with a stable IR version."),
    (9, "B", "Declare sorts and domains",
     "Declare Boolean, integer, real, bit-vector, string, enumeration and domain sorts with explicit finiteness."),
    (10, "B", "Formalize predicates and relations",
     "Declare arity, argument types, interpretation and constraints; reject accidental type coercions."),
    (11, "B", "Formalize quantifiers and scope",
     "Translate all/some/none/exactly-one into explicit quantifiers and cardinality constraints with recorded scope."),
    (12, "B", "Formalize rules with explicit premises and conclusions",
     "Represent each rule as a named implication preserving rule IDs and substitution mappings."),
    (13, "B", "Attach units and dimensional constraints",
     "Represent quantities with units, convert only through validated rules, and require dimension compatibility."),
    (14, "B", "Select the semantic assumption profile",
     "Choose open/closed-world behavior, explicit negation, unique names and equality semantics per knowledge domain."),
    (15, "B", "Normalize and validate the IR",
     "Canonicalize names, alpha-rename, simplify safe expressions, type-check terms and validate references with source mapping."),
    (16, "C", "Select the rule engine",
     "Use Datalog for finite relational inference; route richer first-order problems to theorem provers."),
    (17, "C", "Build the dependency graph",
     "Construct the fact/rule dependency graph, identify recursive strongly connected components and stratification."),
    (18, "C", "Evaluate non-recursive rules",
     "Apply rules with known premises using indexed joins and semi-naive evaluation, emitting rule and parent provenance."),
    (19, "C", "Evaluate recursive rules to a fixed point",
     "Derive only newly enabled facts until a least fixed point with explicit limits for unsafe extensions."),
    (20, "C", "Construct an explanation graph",
     "Store premises, rule, substitutions and derivation path per derived fact; preserve multiple distinct derivations."),
    (21, "C", "Detect conflicting conclusions",
     "Check positive/negative assertions, exclusivity, cardinality and incompatible values; classify each conflict type."),
    (22, "C", "Apply contradiction policy without destructive repair",
     "Preserve both sides, mark conflicts, identify minimal evidence sets and request authority/temporal resolution."),
    (23, "D", "Decide whether SAT or SMT is required",
     "Route Boolean-only problems to SAT and theory problems to SMT; prefer one capable backend."),
    (24, "D", "Encode the formal IR",
     "Translate predicates and constraints into solver variables with a bidirectional map to IR nodes and USG provenance."),
    (25, "D", "Add assumptions as named constraints",
     "Keep assumptions separately addressable so inconsistency can be attributed; never embed undocumented assumptions."),
    (26, "D", "Check satisfiability",
     "Call the solver with deterministic configuration, explicit timeout and memory budget; return SAT/UNSAT/UNKNOWN/timeout/error distinctly."),
    (27, "D", "Interpret satisfying models cautiously",
     "Decode models as one satisfying assignment and validate them independently; never claim uniqueness or reality."),
    (28, "D", "Explain unsatisfiability",
     "Request unsatisfiable cores, map them back to named assumptions and label them as sufficient, not minimal, causes."),
    (29, "D", "Check entailment using counterexample search",
     "Test Gamma entails phi by checking Gamma and not-phi; return countermodels, and 'not established' on unknown."),
    (30, "D", "Validate arithmetic and numeric semantics",
     "Check integer/real semantics, overflow, division-by-zero, nonlinear support and precision; prefer exact arithmetic."),
    (31, "E", "Recognize finite decision and optimization tasks",
     "Route scheduling, assignment and discrete allocation to CP-SAT; never use it for semantic inference."),
    (32, "E", "Define decision variables and domains",
     "Define binary selection, assignment, ordering and start-time variables with explicit domains and units."),
    (33, "E", "Encode task coverage and eligibility",
     "Require at least one eligible module per required capability plus dependency, authorization and compatibility constraints."),
    (34, "E", "Encode resource and safety limits",
     "Constrain CPU, memory, I/O, latency, deadline, concurrency, isolation and policy limits."),
    (35, "E", "Define a transparent objective",
     "Minimize weighted cost with justified extensions, normalized scales and exposed weights."),
    (36, "E", "Solve and report solution quality",
     "Distinguish proven optimal, feasible incumbent, infeasible, unknown and time-limited outcomes with bound and gap."),
    (37, "F", "Identify claims requiring proof-level assurance",
     "Select mathematical statements, critical invariants and kernel-level checks for proof assistants; never export routine inferences."),
    (38, "F", "Generate a proof obligation",
     "Translate the target theorem and assumptions into a well-scoped proposition; reject incomplete or unsound translations."),
    (39, "F", "Construct or import a proof candidate",
     "Accept searcher, tactic-engine or library proofs as untrusted until checked by the proof kernel."),
    (40, "F", "Check the proof independently",
     "Invoke the trusted checker with pinned versions; store the theorem, proof artifact hash, dependency versions and result."),
    (41, "F", "Separate proof validity from premise truth",
     "Record that premise truth, model adequacy and real-world correspondence remain separate questions."),
    (42, "G", "Create a unified result envelope",
     "Normalize Datalog, SAT/SMT, CP-SAT and proof outputs into one typed schema without flattening semantics."),
    (43, "G", "Independently validate result integrity",
     "Check schema, source links, proof references, statuses, model decoding, assumptions and provenance; replay certificates."),
    (44, "G", "Publish to the shared cognitive workspace",
     "Publish findings, conflicts, counterexamples, proposals and proofs as separate typed entries."),
    (45, "G", "Enforce knowledge-write governance",
     "Stay read-only against authoritative USG; propose derived facts with provenance and require approval for promotion."),
    (46, "G", "Persist an auditable execution record",
     "Log request ID, IR hash, inputs, assumptions, solver versions, configurations, resource use, status and artifact hashes."),
    (47, "G", "Apply resource budgets and isolation",
     "Run solvers in bounded workers with CPU, memory, wall-clock and output limits, allowlisted binaries and no unnecessary network."),
    (48, "G", "Handle failures and recovery",
     "Terminate safely on timeout, crash, malformed output or unsupported theory; retry only under policy with bounded backoff."),
    (49, "H", "Build a versioned benchmark suite",
     "Include satisfiable/unsatisfiable formulas, arithmetic, recursive Datalog, contradictions, proofs, CP-SAT, ambiguity and invalid encodings."),
    (50, "H", "Measure correctness and operational quality separately",
     "Track soundness, proof-check rate, model validity, precision/recall, detection, gap, latency, memory, timeouts and reproducibility."),
    (51, "H", "Use failures to improve formalization, not weaken guarantees",
     "Classify failures as semantic mapping, missing assumption, type/unit error, solver limitation, resource limit or incorrect rule."),
    (52, "H", "Release incrementally behind capability flags",
     "Promote Datalog, Z3, CP-SAT and Lean capabilities only after their acceptance suites pass."),
)

_PHASE_RANGES: tuple[tuple[str, int, int], ...] = (
    ("A", 1, 7), ("B", 8, 15), ("C", 16, 22), ("D", 23, 30),
    ("E", 31, 36), ("F", 37, 41), ("G", 42, 48), ("H", 49, 52),
)

PHASE_OF_STEP: dict[int, str] = {
    step: phase for phase, low, high in _PHASE_RANGES for step in range(low, high + 1)
}


def _operation_id(step: int) -> str:
    """Format the stable operation ID for one source step."""
    return f"FRC-{step:03d}"


FRCE_OPERATIONS: tuple[OperationSpec, ...] = tuple(
    OperationSpec(
        operation_id=_operation_id(step),
        step=step,
        phase=PHASE_OF_STEP[step],
        title=title,
        purpose=purpose,
    )
    for step, phase, title, purpose in _FRCE_OPERATIONS
)

FRCE_OPERATION_IDS: tuple[str, ...] = tuple(item.operation_id for item in FRCE_OPERATIONS)


def operation_by_step(step: int) -> OperationSpec:
    """Return the operation specification for one algorithm step number."""
    if not isinstance(step, int) or not 1 <= step <= len(FRCE_OPERATIONS):
        raise ValueError(f"unknown formal reasoning algorithm step: {step!r}")
    return FRCE_OPERATIONS[step - 1]


def operation_by_id(operation_id: str) -> OperationSpec:
    """Return the operation specification for one stable operation ID."""
    for item in FRCE_OPERATIONS:
        if item.operation_id == operation_id:
            return item
    raise ValueError(f"unknown formal reasoning operation ID: {operation_id!r}")


def operations_by_phase(phase: str) -> tuple[OperationSpec, ...]:
    """Return the ordered operations belonging to one phase letter A-H."""
    if phase not in PHASES:
        raise ValueError(f"unknown phase: {phase!r}")
    return tuple(item for item in FRCE_OPERATIONS if item.phase == phase)


__all__ = [
    "PHASES",
    "PHASE_OF_STEP",
    "OperationSpec",
    "FRCE_OPERATIONS",
    "FRCE_OPERATION_IDS",
    "operation_by_step",
    "operation_by_id",
    "operations_by_phase",
]
