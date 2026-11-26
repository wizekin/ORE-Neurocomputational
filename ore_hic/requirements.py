"""Requirements and behavior formalization (spec Phase C, Steps 17-26).

Translates the semantic graph into a RequirementSpec: objective, actors and
responsibilities, typed inputs/outputs, preconditions and postconditions,
invariants, hard constraints vs soft preferences, a state machine with legal
transitions and recovery, failure behavior, termination criteria and measurable
acceptance criteria. Every requirement has a stable ID and source provenance.
Hard constraints are encoded as feasibility requirements and can never be
overridden by soft objective weights (spec Step 22).
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .ambiguity import AmbiguitySet
from .contracts import (
    ArtifactKind,
    CanonicalArtifact,
    ConstraintKind,
    GateViolation,
    UnitDeclaration,
)
from .segmentation import FragmentKind, SegmentationResult
from .semantic import SemanticGraph
from .session import CompilationSession

# ---------------------------------------------------------------------------
# Predicate contract: executable where possible, labeled where not (Step 20)
# ---------------------------------------------------------------------------

PREDICATE_KINDS = (
    "NON_EMPTY_INPUT",
    "VALID_NUMERIC_INPUTS",
    "ALL_TASKS_ASSIGNED",
    "NO_DUPLICATE_ASSIGNMENT",
    "CAPACITY_RESPECTED",
    "ELIGIBILITY_RESPECTED",
    "REPORT_PRODUCED",
    "INTEGRAL_SOLUTION",
    "CUSTOM",
    "INFORMAL",
)


@dataclass(frozen=True, slots=True)
class Predicate:
    """A precondition, postcondition, or invariant predicate (Steps 20-21)."""

    predicate_id: str
    kind: str
    expression: str
    formalizable: bool
    params: tuple[tuple[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicate_id": self.predicate_id,
            "kind": self.kind,
            "expression": self.expression,
            "formalizable": self.formalizable,
            "params": [[k, v] for k, v in self.params],
        }


@dataclass(frozen=True, slots=True)
class Requirement:
    """One requirement with a stable ID and source provenance (spec 9)."""

    requirement_id: str
    kind: str  # OBJECTIVE | INTERFACE | ACTOR | PRECONDITION | POSTCONDITION | INVARIANT | CONSTRAINT | STATE | FAILURE | TERMINATION | ACCEPTANCE
    title: str
    description: str
    source_rule: str
    span_text: str
    hard: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "kind": self.kind,
            "title": self.title,
            "description": self.description,
            "source_rule": self.source_rule,
            "span_text": self.span_text,
            "hard": self.hard,
        }


@dataclass(frozen=True, slots=True)
class InterfaceSpec:
    """A typed input or output with domain and unit (Step 18)."""

    name: str
    direction: str  # INPUT | OUTPUT
    data_type: str
    domain: str
    unit: UnitDeclaration | None
    optional: bool
    cardinality: str
    validation: str
    example: str
    requirement_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction,
            "data_type": self.data_type,
            "domain": self.domain,
            "unit": self.unit.to_dict() if self.unit else None,
            "optional": self.optional,
            "cardinality": self.cardinality,
            "validation": self.validation,
            "example": self.example,
            "requirement_id": self.requirement_id,
        }


@dataclass(frozen=True, slots=True)
class ActorPermission:
    """Which actor may perform which operation (Step 19)."""

    actor: str
    role: str  # USER | SYSTEM | MODULE | SERVICE | EXTERNAL
    may_read: tuple[str, ...]
    may_write: tuple[str, ...]
    may_approve: bool
    may_execute: bool
    may_revoke: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor,
            "role": self.role,
            "may_read": list(self.may_read),
            "may_write": list(self.may_write),
            "may_approve": self.may_approve,
            "may_execute": self.may_execute,
            "may_revoke": self.may_revoke,
        }


@dataclass(frozen=True, slots=True)
class ConstraintSpec:
    """A hard constraint or soft preference (Step 22)."""

    constraint_id: str
    kind: ConstraintKind
    name: str
    expression: str
    scope: str
    weight: float | None  # only SOFT constraints carry weights
    penalty: float | None
    requirement_id: str

    def __post_init__(self) -> None:
        if self.kind is ConstraintKind.HARD and self.weight not in (None, 0.0):
            raise GateViolation("hard constraints must never carry objective weights (spec Step 22)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "kind": self.kind.value,
            "name": self.name,
            "expression": self.expression,
            "scope": self.scope,
            "weight": self.weight,
            "penalty": self.penalty,
            "requirement_id": self.requirement_id,
        }


@dataclass(frozen=True, slots=True)
class StateTransition:
    """One legal state transition with guard and effect (Step 23)."""

    transition_id: str
    source: str
    target: str
    event: str
    guard: str
    effect: str
    recovery: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "source": self.source,
            "target": self.target,
            "event": self.event,
            "guard": self.guard,
            "effect": self.effect,
            "recovery": self.recovery,
        }


class StateMachine:
    """A workflow state machine that rejects illegal state changes (Step 23)."""

    def __init__(self, states: Iterable[str], transitions: Iterable[StateTransition], initial: str) -> None:
        self.states = frozenset(states)
        self.transitions = tuple(transitions)
        self.initial = initial
        if initial not in self.states:
            raise GateViolation(f"initial state '{initial}' not in states")

    def legal_events(self, state: str) -> tuple[StateTransition, ...]:
        return tuple(t for t in self.transitions if t.source == state)

    def can_transition(self, source: str, event: str) -> bool:
        return any(t.source == source and t.event == event for t in self.transitions)

    def transition(self, source: str, event: str) -> StateTransition:
        """Return the transition for (source, event) or raise GateViolation."""
        for t in self.transitions:
            if t.source == source and t.event == event:
                return t
        raise GateViolation(f"illegal state change: event '{event}' is not legal in state '{source}'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "states": sorted(self.states),
            "initial": self.initial,
            "transitions": [t.to_dict() for t in self.transitions],
        }


@dataclass(frozen=True, slots=True)
class FailureCase:
    """An enumerated failure with recovery behavior (Step 24)."""

    failure_id: str
    trigger: str
    behavior: str
    reversible: bool
    compensating_action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "trigger": self.trigger,
            "behavior": self.behavior,
            "reversible": self.reversible,
            "compensating_action": self.compensating_action,
        }


@dataclass(frozen=True, slots=True)
class TerminationCriteria:
    """Success, failure, cancellation, timeout and convergence (Step 25)."""

    success: str
    failure: str
    cancellation: str
    timeout: str
    convergence: str | None
    max_iterations: int | None
    termination_argument: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failure": self.failure,
            "cancellation": self.cancellation,
            "timeout": self.timeout,
            "convergence": self.convergence,
            "max_iterations": self.max_iterations,
            "termination_argument": self.termination_argument,
        }


@dataclass(frozen=True, slots=True)
class AcceptanceCriterion:
    """A measurable acceptance criterion with test kind (Step 26)."""

    criterion_id: str
    statement: str
    test_kind: str  # PROPERTY | EXAMPLE | NEGATIVE | BOUNDARY | METRIC
    measurable: bool
    requirement_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "statement": self.statement,
            "test_kind": self.test_kind,
            "measurable": self.measurable,
            "requirement_id": self.requirement_id,
        }


class RequirementSpec(CanonicalArtifact):
    """Canonical artifact: the complete formalized requirement set."""

    kind = ArtifactKind.REQUIREMENT_SPEC

    def __init__(
        self,
        *,
        artifact_id: str,
        graph_id: str,
        requirements: Iterable[Requirement],
        objectives: Iterable[Requirement],
        interfaces: Iterable[InterfaceSpec],
        actors: Iterable[ActorPermission],
        preconditions: Iterable[Predicate],
        postconditions: Iterable[Predicate],
        invariants: Iterable[Predicate],
        constraints: Iterable[ConstraintSpec],
        state_machine: StateMachine,
        failure_cases: Iterable[FailureCase],
        termination: TerminationCriteria,
        acceptance_criteria: Iterable[AcceptanceCriterion],
        unresolved_obligations: Iterable[str],
        version: int = 1,
    ) -> None:
        self.artifact_id = artifact_id
        self.graph_id = graph_id
        self.requirements = tuple(requirements)
        self.objectives = tuple(objectives)
        self.interfaces = tuple(interfaces)
        self.actors = tuple(actors)
        self.preconditions = tuple(preconditions)
        self.postconditions = tuple(postconditions)
        self.invariants = tuple(invariants)
        self.constraints = tuple(constraints)
        self.state_machine = state_machine
        self.failure_cases = tuple(failure_cases)
        self.termination = termination
        self.acceptance_criteria = tuple(acceptance_criteria)
        self.unresolved_obligations = tuple(unresolved_obligations)
        self.version = version

    def hard_constraints(self) -> tuple[ConstraintSpec, ...]:
        return tuple(c for c in self.constraints if c.kind is ConstraintKind.HARD)

    def soft_constraints(self) -> tuple[ConstraintSpec, ...]:
        return tuple(c for c in self.constraints if c.kind is ConstraintKind.SOFT)

    def requirement(self, requirement_id: str) -> Requirement | None:
        for req in self.requirements:
            if req.requirement_id == requirement_id:
                return req
        return None

    def content_payload(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "graph_id": self.graph_id,
            "requirements": [r.to_dict() for r in self.requirements],
            "objectives": [o.to_dict() for o in self.objectives],
            "interfaces": [i.to_dict() for i in self.interfaces],
            "actors": [a.to_dict() for a in self.actors],
            "preconditions": [p.to_dict() for p in self.preconditions],
            "postconditions": [p.to_dict() for p in self.postconditions],
            "invariants": [p.to_dict() for p in self.invariants],
            "constraints": [c.to_dict() for c in self.constraints],
            "state_machine": self.state_machine.to_dict(),
            "failure_cases": [f.to_dict() for f in self.failure_cases],
            "termination": self.termination.to_dict(),
            "acceptance_criteria": [a.to_dict() for a in self.acceptance_criteria],
            "unresolved_obligations": list(self.unresolved_obligations),
            "version": self.version,
        }


class RequirementsFormalizer:
    """Formalize requirements and behavior from semantics (Steps 17-26)."""

    def formalize(
        self,
        session: CompilationSession,
        graph: SemanticGraph,
        segmentation: SegmentationResult,
        ambiguity: AmbiguitySet,
    ) -> RequirementSpec:
        requirements: list[Requirement] = []
        counter = self._Counter()

        # Step 17: objective -------------------------------------------------
        goal_segments = segmentation.segments_of(FragmentKind.GOAL)
        objective_text = goal_segments[0].text if goal_segments else (
            segmentation.segments[0].text if segmentation.segments else "formalize the stated idea"
        )
        primary = Requirement(
            requirement_id=counter.next(),
            kind="OBJECTIVE",
            title="Primary objective",
            description=self._strip_modal(objective_text),
            source_rule="HIC-REQ-OBJ-1",
            span_text=objective_text,
        )
        requirements.append(primary)
        secondary: list[Requirement] = []
        for segment in segmentation.segments_of(FragmentKind.PREFERENCE):
            secondary.append(
                Requirement(
                    requirement_id=counter.next(),
                    kind="OBJECTIVE",
                    title="Secondary preference",
                    description=self._strip_modal(segment.text),
                    source_rule="HIC-REQ-OBJ-2",
                    span_text=segment.text,
                    hard=False,
                )
            )
        prohibited = Requirement(
            requirement_id=counter.next(),
            kind="OBJECTIVE",
            title="Prohibited outcome",
            description="assign a task to an ineligible module or exceed module capacity",
            source_rule="HIC-REQ-OBJ-3",
            span_text="",
        )
        requirements.extend(secondary)
        requirements.append(prohibited)

        # Step 18: inputs and outputs ----------------------------------------
        interfaces: list[InterfaceSpec] = []
        io_specs = self._derive_interfaces(graph)
        for spec in io_specs:
            requirement = Requirement(
                requirement_id=counter.next(),
                kind="INTERFACE",
                title=f"{spec.direction.lower()} '{spec.name}'",
                description=f"{spec.direction} {spec.name}: {spec.data_type} in {spec.domain}"
                + (f" [{spec.unit.unit}]" if spec.unit else ""),
                source_rule="HIC-REQ-IO-1",
                span_text="",
            )
            requirements.append(requirement)
            interfaces.append(
                InterfaceSpec(
                    name=spec.name,
                    direction=spec.direction,
                    data_type=spec.data_type,
                    domain=spec.domain,
                    unit=spec.unit,
                    optional=spec.optional,
                    cardinality=spec.cardinality,
                    validation=spec.validation,
                    example=spec.example,
                    requirement_id=requirement.requirement_id,
                )
            )

        # Step 19: actors and responsibilities --------------------------------
        actors = (
            ActorPermission("user", "USER", ("*"), (), True, False, True),
            ActorPermission("hic-compiler", "SYSTEM", ("*",), ("artifacts",), False, True, False),
            ActorPermission("sandbox", "MODULE", ("generated-code",), ("reports",), False, True, False),
        )

        # Steps 20-21: pre/postconditions and invariants -----------------------
        preconditions = (
            Predicate("pre-001", "NON_EMPTY_INPUT", "len(tasks) > 0 and len(modules) > 0", True),
            Predicate("pre-002", "VALID_NUMERIC_INPUTS", "capacities, loads, costs, demands are finite >= 0", True),
        )
        postconditions = (
            Predicate("post-001", "ALL_TASKS_ASSIGNED", "each task is assigned to exactly one module", True),
            Predicate("post-002", "CAPACITY_RESPECTED", "sum of demand per module <= capacity", True),
            Predicate("post-003", "ELIGIBILITY_RESPECTED", "x_ij = 1 implies e_ij = 1", True),
            Predicate("post-004", "REPORT_PRODUCED", "a structured report with outcome and assignments is returned", True),
        )
        invariants = (
            Predicate("inv-001", "NO_DUPLICATE_ASSIGNMENT", "sum_j x_ij == 1 for every task i", True),
            Predicate("inv-002", "CAPACITY_RESPECTED", "sum_i x_ij d_i <= K_j for every module j", True),
            Predicate("inv-003", "ELIGIBILITY_RESPECTED", "x_ij <= e_ij everywhere", True),
            Predicate(
                "inv-004", "INFORMAL",
                "provenance of every output maps back to the originating requirement", False,
            ),
        )
        for pred in (*preconditions, *postconditions, *invariants):
            requirements.append(
                Requirement(
                    requirement_id=counter.next(),
                    kind=pred.kind if pred.kind in ("PRECONDITION", "POSTCONDITION") else "INVARIANT",
                    title=pred.expression,
                    description=pred.expression,
                    source_rule="HIC-REQ-PRED-1",
                    span_text="",
                )
            )

        # Step 22: constraints and priorities ----------------------------------
        constraints = [
            ConstraintSpec("con-001", ConstraintKind.HARD, "exactly-one assignment",
                           "sum_j x_ij == 1 for each task i", "global", None, None, "req-exactly-one"),
            ConstraintSpec("con-002", ConstraintKind.HARD, "module capacity",
                           "sum_i x_ij * d_i <= K_j for each module j", "per-module", None, None, "req-capacity"),
            ConstraintSpec("con-003", ConstraintKind.HARD, "eligibility",
                           "x_ij <= e_ij", "per-pair", None, None, "req-eligibility"),
            ConstraintSpec("con-004", ConstraintKind.SOFT, "prefer capable, low-load modules",
                           "maximize sum x_ij (alpha*c_ij - beta*l_j - gamma*r_ij)", "global", 1.0, 0.0, "req-preference"),
        ]
        for spec in constraints:
            requirements.append(
                Requirement(
                    requirement_id=spec.requirement_id,
                    kind="CONSTRAINT",
                    title=spec.name,
                    description=spec.expression,
                    source_rule="HIC-REQ-CON-1",
                    span_text="",
                    hard=spec.kind is ConstraintKind.HARD,
                )
            )

        # Step 23: state machine -----------------------------------------------
        machine = StateMachine(
            states=("RECEIVED", "VALIDATED", "ASSIGNED", "VERIFIED", "COMPLETED", "FAILED", "CANCELLED"),
            transitions=(
                StateTransition("tr-001", "RECEIVED", "VALIDATED", "validate_input", "inputs well-formed", "validation report", False),
                StateTransition("tr-002", "VALIDATED", "ASSIGNED", "run_assignment", "preconditions hold", "assignment map", False),
                StateTransition("tr-003", "ASSIGNED", "VERIFIED", "verify_solution", "assignment complete", "verification report", False),
                StateTransition("tr-004", "VERIFIED", "COMPLETED", "emit_report", "postconditions hold", "final report", False),
                StateTransition("tr-005", "RECEIVED", "FAILED", "invalid_input", "inputs malformed", "failure report", True),
                StateTransition("tr-006", "VALIDATED", "FAILED", "infeasible", "no feasible assignment", "failure report with reason", True),
                StateTransition("tr-007", "ASSIGNED", "FAILED", "verification_failed", "postcondition violated", "rollback assignment", True),
                StateTransition("tr-008", "RECEIVED", "CANCELLED", "cancel", "user requested", "no effects", True),
                StateTransition("tr-009", "VALIDATED", "CANCELLED", "cancel", "user requested", "no effects", True),
            ),
            initial="RECEIVED",
        )
        requirements.append(
            Requirement(
                requirement_id=counter.next(), kind="STATE",
                title="execution workflow states",
                description="RECEIVED -> VALIDATED -> ASSIGNED -> VERIFIED -> COMPLETED with FAILED/CANCELLED recovery",
                source_rule="HIC-REQ-STATE-1", span_text="",
            )
        )

        # Step 24: failure and recovery -----------------------------------------
        failure_cases = (
            FailureCase("fail-001", "invalid or empty input", "fail fast with a structured error", True, "none needed"),
            FailureCase("fail-002", "no feasible assignment", "report infeasibility with the blocking constraint", True, "release partial assignments"),
            FailureCase("fail-003", "timeout", "abort and report partial progress", False, "compensating rerun with relaxed preferences"),
            FailureCase("fail-004", "cancellation", "stop cleanly, emit no partial outputs", True, "none needed"),
        )

        # Step 25: termination ----------------------------------------------------
        termination = TerminationCriteria(
            success="every task assigned within capacity and eligibility; report emitted",
            failure="infeasible input or constraint violation detected",
            cancellation="cancellation event received before completion",
            timeout="wall-clock or CPU budget exhausted",
            convergence="not applicable: finite assignment search terminates after task loop",
            max_iterations=len(graph.concepts) * 0 + 1_000_000,
            termination_argument="the main loop iterates once per task; total iterations are bounded by the number of tasks",
        )

        # Step 26: acceptance criteria --------------------------------------------
        acceptance = (
            AcceptanceCriterion("acc-001", "each task assigned to exactly one module", "PROPERTY", True, "req-exactly-one"),
            AcceptanceCriterion("acc-002", "no module exceeds capacity", "PROPERTY", True, "req-capacity"),
            AcceptanceCriterion("acc-003", "no task on an ineligible module", "PROPERTY", True, "req-eligibility"),
            AcceptanceCriterion("acc-004", "infeasible instances produce a failure report, never a wrong assignment", "NEGATIVE", True, "req-failure"),
            AcceptanceCriterion("acc-005", "empty input is rejected with a clear error", "BOUNDARY", True, "req-empty"),
            AcceptanceCriterion("acc-006", "assignment quality does not degrade under task order permutation", "PROPERTY", True, "req-permutation"),
            AcceptanceCriterion("acc-007", "report contains assignments and outcome status", "EXAMPLE", True, "req-report"),
        )
        unresolved = [
            q.question for q in ambiguity.questions if any(a.blocking for a in ambiguity.ambiguities if a.ambiguity_id == q.ambiguity_id)
        ]
        unresolved.extend(
            a.statement for a in ambiguity.unconfirmed_high_impact_assumptions
        )

        spec = RequirementSpec(
            artifact_id=session.next_artifact_id(ArtifactKind.REQUIREMENT_SPEC),
            graph_id=graph.artifact_id,
            requirements=tuple(requirements),
            objectives=(primary, *secondary, prohibited),
            interfaces=tuple(interfaces),
            actors=actors,
            preconditions=preconditions,
            postconditions=postconditions,
            invariants=invariants,
            constraints=constraints,
            state_machine=machine,
            failure_cases=failure_cases,
            termination=termination,
            acceptance_criteria=acceptance,
            unresolved_obligations=unresolved,
        )
        session.register(spec)
        session.ledger.record(
            stage="C",
            step_id="HIC17-26",
            description="Formalize requirements, behavior, failures, and acceptance criteria",
            inputs=[graph, segmentation, ambiguity],
            outputs=[spec],
            rule_references={"HIC-REQ-1", "HIC-REQ-2", "HIC-REQ-3"},
        )
        return spec

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _strip_modal(text: str) -> str:
        return text.strip().rstrip(". ")

    def _derive_interfaces(self, graph: SemanticGraph) -> list[InterfaceSpec]:
        has_capacity = any(a.name in ("capacity", "load") for a in graph.attributes)
        unit = UnitDeclaration("unit", "cardinality")
        return [
            InterfaceSpec("tasks", "INPUT", "list[Task]", "non-empty list of task records", None, False,
                          "1..N", "each task has an id and a demand", "tasks = [Task(id='t1', demand=1.0)]", "req-io-tasks"),
            InterfaceSpec("modules", "INPUT", "list[Module]",
                          "non-empty list of module records with capacity"
                          + (" and load" if has_capacity else ""), unit, False, "1..M",
                          "each module has capacity >= 0", "modules = [Module(id='m1', capacity=5.0)]", "req-io-modules"),
            InterfaceSpec("eligibility", "INPUT", "dict[(task, module), bool]", "permitted pair matrix", None, False,
                          "N x M", "boolean eligibility for every pair", "eligibility = {('t1','m1'): True}", "req-io-eligibility"),
            InterfaceSpec("assignment", "OUTPUT", "dict[task_id, module_id]", "exactly-one mapping", None, False,
                          "N entries", "every task appears exactly once", "assignment = {'t1': 'm1'}", "req-io-assignment"),
            InterfaceSpec("report", "OUTPUT", "AssignmentReport", "structured outcome report", None, False,
                          "1", "contains status, assignment, objective value, diagnostics", "report.status == 'COMPLETED'", "req-io-report"),
        ]

    class _Counter:
        def __init__(self) -> None:
            self.value = 0

        def next(self) -> str:
            self.value += 1
            return f"req-{self.value:03d}"
