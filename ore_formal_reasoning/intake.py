"""Request intake, classification, and evidence normalization (Phase A, Steps 1-7).

The intake pipeline validates that a request can be operationalized, routes it
into task classes without calling everything ``theorem proving``, separates
facts from assumptions and hypotheses, preserves provenance and temporal scope,
and detects incompleteness or ambiguity before any solver is invoked. When the
goal cannot be operationalized the pipeline returns a structured
``insufficient formalization`` report with explicit clarification requests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .contracts import (
    AssuranceLevel,
    EvidenceTag,
    FormalizationRequest,
    NamedAssumption,
    TaskClass,
)
from .ir import (
    AndF,
    ArithExpr,
    Atom,
    BvExpr,
    CardinalityF,
    Compare,
    ExistsF,
    Finiteness,
    FormalIR,
    ForallF,
    FuncApp,
    IffF,
    ImpliesF,
    NotF,
    OrF,
    SortKind,
    Statement,
    StrExpr,
    ToReal,
)


@dataclass(frozen=True, slots=True)
class ClarificationRequest:
    """One explicit clarification the coordinator must answer (Steps 1, 7)."""

    code: str
    message: str
    statement_id: str | None = None


@dataclass(frozen=True, slots=True)
class TaskClassification:
    """Routed task classes for one request (Step 2)."""

    declared: tuple[TaskClass, ...]
    detected: tuple[TaskClass, ...]
    routed: tuple[TaskClass, ...]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "notes", tuple(self.notes))


@dataclass(frozen=True, slots=True)
class EvidenceSeparation:
    """Fact/assumption/hypothesis/policy separation over statements (Step 5)."""

    observed_facts: tuple[Statement, ...]
    stipulated_assumptions: tuple[Statement, ...]
    inferred_results: tuple[Statement, ...]
    hypotheses: tuple[Statement, ...]
    policy_constraints: tuple[Statement, ...]
    named_assumptions: tuple[NamedAssumption, ...] = ()


@dataclass(frozen=True, slots=True)
class ProvenanceLedger:
    """Provenance and temporal-scope ledger over all statements (Step 6)."""

    statement_count: int
    source_ids: tuple[str, ...]
    scoped_statements: tuple[str, ...]
    time_scoped_statements: tuple[str, ...]
    authority_records: tuple[str, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CompletenessReport:
    """Completeness and ambiguity findings before formal execution (Step 7)."""

    sufficient: bool
    clarification_requests: tuple[ClarificationRequest, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IntakeReport:
    """Aggregated Phase-A output consumed by the coordinator."""

    request: FormalizationRequest
    classification: TaskClassification
    separation: EvidenceSeparation
    provenance: ProvenanceLedger
    completeness: CompletenessReport
    operationalizable: bool


def validate_request(request: FormalizationRequest) -> tuple[ClarificationRequest, ...]:
    """Check that the objective can be operationalized (Step 1).

    A request must state what should be established (a required property), or
    carry an optimization problem, or carry a proof obligation. Requests that
    only narrate a goal are rejected until clarification is accepted.
    """
    clarifications: list[ClarificationRequest] = []
    has_property = bool(request.required_property and request.required_property.strip())
    has_optimization = request.optimization_problem is not None
    has_proof = request.proof_obligation is not None
    if not (has_property or has_optimization or has_proof):
        clarifications.append(ClarificationRequest(
            code="goal_not_operationalizable",
            message=(
                "the objective does not state a derivable fact, a property to verify, "
                "an optimization problem, or a proof obligation"
            ),
        ))
    if TaskClass.COMBINATORIAL_OPTIMIZATION in request.task_classes and not has_optimization:
        clarifications.append(ClarificationRequest(
            code="missing_optimization_problem",
            message="optimization tasks must carry a SelectionProblem payload",
        ))
    if TaskClass.MATHEMATICAL_PROOF in request.task_classes and not has_proof and not has_property:
        clarifications.append(ClarificationRequest(
            code="missing_proof_obligation",
            message="mathematical proof tasks must carry a proof obligation or property",
        ))
    return tuple(clarifications)


def classify_problem(request: FormalizationRequest, ir: FormalIR) -> TaskClassification:
    """Detect task classes from IR content and merge with declared classes (Step 2)."""
    detected: set[TaskClass] = set()
    notes: list[str] = []

    if ir.rules:
        detected.add(TaskClass.RULE_INFERENCE)

    has_arithmetic = False
    has_theory_terms = False
    has_boolean_structure = False
    for statement in ir.statements:
        kind = _classify_formula(statement.formula)
        if kind == "arithmetic":
            has_arithmetic = True
        elif kind == "theory":
            has_theory_terms = True
        elif kind == "boolean":
            has_boolean_structure = True

    if has_arithmetic:
        detected.add(TaskClass.ARITHMETIC_CONSTRAINTS)
    if has_boolean_structure and not has_arithmetic and not has_theory_terms:
        detected.add(TaskClass.BOOLEAN_CONSISTENCY)
    if request.optimization_problem is not None:
        detected.add(TaskClass.COMBINATORIAL_OPTIMIZATION)
    if request.proof_obligation is not None:
        detected.add(TaskClass.MATHEMATICAL_PROOF)
    if request.required_property and request.assurance_level == AssuranceLevel.HIGH:
        detected.add(TaskClass.PROGRAM_PROPERTY_VERIFICATION)

    merged: list[TaskClass] = []
    for task_class in request.task_classes:
        if task_class != TaskClass.MIXED:
            merged.append(task_class)
    for task_class in sorted(detected, key=lambda item: item.value):
        if task_class not in merged:
            merged.append(task_class)
    if len(merged) > 1:
        notes.append("multiple task classes routed; not every task is theorem proving")
    return TaskClassification(
        declared=tuple(request.task_classes),
        detected=tuple(sorted(detected, key=lambda item: item.value)),
        routed=tuple(dict.fromkeys(merged)),
        notes=tuple(notes),
    )


def _classify_formula(formula: Any) -> str:
    """Classify one formula: arithmetic, theory-laden, boolean, or plain."""
    if isinstance(formula, Atom):
        for arg in formula.args:
            found = _classify_term(arg)
            if found != "none":
                return found
        return "none"
    if isinstance(formula, Compare):
        left_kind = _classify_term(formula.left)
        right_kind = _classify_term(formula.right)
        if "arithmetic" in {left_kind, right_kind}:
            return "arithmetic"
        if "theory" in {left_kind, right_kind}:
            return "theory"
        left_sort = getattr(formula.left, "sort", None)
        if left_sort in {"Int", "Real"}:
            return "arithmetic"
        if left_sort is not None:
            return "theory"
        return "none"
    if isinstance(formula, (NotF, ForallF, ExistsF)):
        return _classify_formula(formula.body)
    if isinstance(formula, (AndF, OrF)):
        kinds = {_classify_formula(arg) for arg in formula.args}
        return _worst_kind(kinds)
    if isinstance(formula, ImpliesF):
        return _worst_kind({_classify_formula(formula.premise), _classify_formula(formula.conclusion)})
    if isinstance(formula, IffF):
        return _worst_kind({_classify_formula(formula.left), _classify_formula(formula.right)})
    if isinstance(formula, CardinalityF):
        return _worst_kind({_classify_formula(arg) for arg in formula.args})
    return "none"


def _worst_kind(kinds: set[str]) -> str:
    """Combine child classifications with the strongest kind winning."""
    for priority in ("arithmetic", "theory", "boolean", "none"):
        if priority in kinds:
            return priority
    return "none"


def _classify_term(term: Any) -> str:
    """Classify one term as arithmetic, theory-laden, or plain."""
    if isinstance(term, (ArithExpr, ToReal)):
        return "arithmetic"
    if isinstance(term, (BvExpr, StrExpr, FuncApp)):
        return "theory"
    inner_args = getattr(term, "args", None)
    if isinstance(inner_args, tuple):
        for arg in inner_args:
            found = _classify_term(arg)
            if found != "none":
                return found
    operand = getattr(term, "operand", None)
    if operand is not None:
        return _classify_term(operand)
    return "none"


def separate_evidence(
    statements: Iterable[Statement],
    named_assumptions: Iterable[NamedAssumption] = (),
) -> EvidenceSeparation:
    """Tag and partition every input by its evidence category (Step 5)."""
    groups: dict[EvidenceTag, list[Statement]] = {tag: [] for tag in EvidenceTag}
    for statement in statements:
        groups[statement.tag].append(statement)
    return EvidenceSeparation(
        observed_facts=tuple(groups[EvidenceTag.OBSERVED_FACT]),
        stipulated_assumptions=tuple(groups[EvidenceTag.STIPULATED_ASSUMPTION]),
        inferred_results=tuple(groups[EvidenceTag.INFERRED_RESULT]),
        hypotheses=tuple(groups[EvidenceTag.HYPOTHESIS]),
        policy_constraints=tuple(groups[EvidenceTag.POLICY_CONSTRAINT]),
        named_assumptions=tuple(named_assumptions),
    )


def preserve_provenance(statements: Iterable[Statement]) -> ProvenanceLedger:
    """Collect provenance, authority, and temporal scopes without merging facts (Step 6)."""
    all_statements = tuple(statements)
    source_ids: list[str] = []
    scoped: list[str] = []
    time_scoped: list[str] = []
    authorities: list[str] = []
    notes: list[str] = []
    for statement in all_statements:
        if statement.context_scope != "default" or statement.domain != "default":
            scoped.append(statement.statement_id)
        if statement.temporal_scope is not None:
            time_scoped.append(statement.statement_id)
        for ref in statement.source_refs:
            if ref.source_id not in source_ids:
                source_ids.append(ref.source_id)
            if ref.authority and ref.authority not in authorities:
                authorities.append(ref.authority)
    if time_scoped:
        notes.append(
            "time-scoped facts are preserved with their validity intervals; conflicting "
            "time-scoped facts are not treated as simultaneous contradictions without "
            "checking their scopes"
        )
    return ProvenanceLedger(
        statement_count=len(all_statements),
        source_ids=tuple(source_ids),
        scoped_statements=tuple(scoped),
        time_scoped_statements=tuple(time_scoped),
        authority_records=tuple(authorities),
        notes=tuple(notes),
    )


def check_completeness(request: FormalizationRequest, ir: FormalIR) -> CompletenessReport:
    """Detect missing values, ambiguous quantifiers, unit issues, and open domains (Step 7)."""
    requests: list[ClarificationRequest] = []
    warnings: list[str] = []
    sorts = ir.sort_map()
    domain_map = ir.domain_map()

    for statement in ir.statements:
        _check_quantifier_scope(statement, sorts, requests)
        _check_underspecified_domains(statement, sorts, requests)

    # Open-world versus closed-world: absence is never converted into explicit
    # false unless the domain profile explicitly declares closed-world (Step 14).
    for name, semantics in domain_map.items():
        if not semantics.closed_world:
            warnings.append(
                f"domain {name!r} uses open-world semantics: absence of a fact is not falsity"
            )

    if not ir.facts() and TaskClass.RULE_INFERENCE in request.task_classes:
        requests.append(ClarificationRequest(
            code="no_facts_available",
            message="rule inference requires at least one observed fact; retrieval found none",
        ))

    return CompletenessReport(
        sufficient=not requests,
        clarification_requests=tuple(requests),
        warnings=tuple(warnings),
    )


def _check_quantifier_scope(
    statement: Statement,
    sorts: Mapping[str, Any],
    requests: list[ClarificationRequest],
) -> None:
    """Flag quantifiers whose domain sort is partially known or underspecified (Step 7)."""

    def visit(formula: Any) -> None:
        if isinstance(formula, (ForallF, ExistsF)):
            for decl in formula.variables:
                sort = sorts.get(decl.variable.sort)
                if sort is None:
                    continue
                if sort.finiteness == Finiteness.PARTIALLY_KNOWN:
                    requests.append(ClarificationRequest(
                        code="missing_quantifier_scope",
                        message=(
                            f"quantifier over {decl.variable.name!r} ranges a partially "
                            f"known sort {decl.variable.sort!r}; the scope must be completed"
                        ),
                        statement_id=statement.statement_id,
                    ))
            visit(formula.body)
            return
        body = getattr(formula, "body", None)
        if body is not None:
            visit(body)
        args = getattr(formula, "args", None)
        if isinstance(args, tuple):
            for arg in args:
                if hasattr(arg, "predicate") or hasattr(arg, "body") or hasattr(arg, "op"):
                    visit(arg)

    visit(statement.formula)


def _check_underspecified_domains(
    statement: Statement,
    sorts: Mapping[str, Any],
    requests: list[ClarificationRequest],
) -> None:
    """Flag enum sorts without values and numeric sorts with incomplete bounds."""

    def visit_formula(formula: Any) -> None:
        if isinstance(formula, Atom):
            for arg in formula.args:
                visit_term(arg)
            return
        if isinstance(formula, Compare):
            visit_term(formula.left)
            visit_term(formula.right)
            return
        body = getattr(formula, "body", None)
        if body is not None:
            visit_formula(body)
        args = getattr(formula, "args", None)
        if isinstance(args, tuple):
            for arg in args:
                if hasattr(arg, "predicate") or hasattr(arg, "body") or hasattr(arg, "op"):
                    visit_formula(arg)

    def visit_term(term: Any) -> None:
        sort_name = getattr(term, "sort", None)
        if not sort_name or sort_name not in sorts:
            return
        sort = sorts[sort_name]
        if sort.kind == SortKind.ENUM and not sort.values:
            requests.append(ClarificationRequest(
                code="underspecified_domain",
                message=f"enum sort {sort_name!r} declares no domain values",
                statement_id=statement.statement_id,
            ))
        if (
            sort.kind == SortKind.INT
            and sort.finiteness == Finiteness.FINITE
            and (sort.lower is None or sort.upper is None)
        ):
            requests.append(ClarificationRequest(
                code="underspecified_domain",
                message=f"finite numeric sort {sort_name!r} lacks lower/upper bounds",
                statement_id=statement.statement_id,
            ))

    visit_formula(statement.formula)


def run_intake(request: FormalizationRequest, ir: FormalIR | None = None) -> IntakeReport:
    """Execute Phase A checks with IR-aware passes when the IR is available (Steps 1-7)."""
    clarifications = validate_request(request)
    sufficient = not clarifications
    classification = TaskClassification(
        declared=tuple(request.task_classes),
        detected=(),
        routed=tuple(request.task_classes),
    )
    separation = EvidenceSeparation(
        observed_facts=(), stipulated_assumptions=(), inferred_results=(),
        hypotheses=(), policy_constraints=(),
        named_assumptions=tuple(request.assumptions),
    )
    provenance = ProvenanceLedger(
        statement_count=0,
        source_ids=tuple(r.source_id for r in request.source_refs),
        scoped_statements=(), time_scoped_statements=(),
        authority_records=tuple(
            r.authority for r in request.source_refs if r.authority
        ),
    )
    completeness = CompletenessReport(
        sufficient=sufficient,
        clarification_requests=clarifications,
    )
    if ir is not None:
        classification = classify_problem(request, ir)
        separation = separate_evidence(ir.statements, request.assumptions)
        provenance = preserve_provenance(ir.statements)
        completeness = check_completeness(request, ir)
        sufficient = sufficient and completeness.sufficient
    return IntakeReport(
        request=request,
        classification=classification,
        separation=separation,
        provenance=provenance,
        completeness=completeness,
        operationalizable=sufficient,
    )


__all__ = [
    "ClarificationRequest",
    "TaskClassification",
    "EvidenceSeparation",
    "ProvenanceLedger",
    "CompletenessReport",
    "IntakeReport",
    "validate_request",
    "classify_problem",
    "separate_evidence",
    "preserve_provenance",
    "check_completeness",
    "run_intake",
]
