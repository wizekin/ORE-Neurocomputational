"""AlgoIR: the canonical, language-independent intermediate representation.

Implements spec Phase E (Steps 34-39) and Section 4: a versioned typed graph
with all required node types (GOAL, ENTITY, TYPE, VARIABLE, INPUT, OUTPUT,
PRECONDITION, POSTCONDITION, OPERATION, CONDITION, LOOP, DEPENDENCY,
CONSTRAINT, INVARIANT, FAILURE_CASE, COMPLEXITY, TEST, PROVENANCE), required
metadata on every node, representation invariants, schema validation,
well-formedness validation, deterministic serialization with round trips, and
a human-readable explanation derived from the canonical IR itself (never an
independent paraphrase that could conflict with it).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .contracts import (
    HIC_SCHEMA_VERSION,
    ArtifactKind,
    CanonicalArtifact,
    NodeType,
    SchemaError,
    SourceSpan,
    TerminationPolicy,
    UnitDeclaration,
    USGReference,
    ValidationState,
    digest_payload,
)
from .mathmodel import MathModel
from .requirements import RequirementSpec
from .semantic import SemanticGraph
from .session import CompilationSession

# ---------------------------------------------------------------------------
# Node and graph schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AlgoIRNode:
    """One typed AlgoIR node with complete required metadata (spec 4.2)."""

    node_id: str
    node_type: NodeType
    name: str
    scope: str
    description: str
    content: Mapping[str, Any]
    confidence: float
    source_spans: tuple[SourceSpan, ...]
    parents: tuple[str, ...]
    children: tuple[str, ...]
    dependencies: tuple[str, ...]
    assumptions: tuple[str, ...]
    usg_references: tuple[USGReference, ...]
    creation_stage: str
    validation_state: ValidationState
    derived_construction: bool
    units: UnitDeclaration | None = None

    def __post_init__(self) -> None:
        if not self.node_id or not self.node_id.startswith("n"):
            raise SchemaError(f"node id '{self.node_id}' must be a stable n-prefixed id")
        if not 0.0 <= self.confidence <= 1.0:
            raise SchemaError(f"node {self.node_id}: confidence must be in [0, 1]")
        if not self.source_spans and not self.derived_construction:
            raise SchemaError(
                f"node {self.node_id}: every node needs provenance or an explicit "
                "derived-construction label (spec 4.3)"
            )

    @property
    def payload_digest(self) -> str:
        return digest_payload(
            {
                "node_id": self.node_id,
                "node_type": self.node_type.value,
                "name": self.name,
                "content": dict(sorted(self.content.items())),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "schema_version": HIC_SCHEMA_VERSION,
            "node_type": self.node_type.value,
            "name": self.name,
            "scope": self.scope,
            "description": self.description,
            "content": dict(sorted(self.content.items())),
            "units": self.units.to_dict() if self.units else None,
            "confidence": self.confidence,
            "source_spans": [s.to_dict() for s in self.source_spans],
            "parents": list(self.parents),
            "children": list(self.children),
            "dependencies": list(self.dependencies),
            "assumptions": list(self.assumptions),
            "usg_references": [r.to_dict() for r in self.usg_references],
            "creation_stage": self.creation_stage,
            "validation_state": self.validation_state.value,
            "derived_construction": self.derived_construction,
            "payload_digest": self.payload_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AlgoIRNode:
        return cls(
            node_id=payload["node_id"],
            node_type=NodeType(payload["node_type"]),
            name=payload["name"],
            scope=payload["scope"],
            description=payload["description"],
            content=dict(payload["content"]),
            confidence=float(payload["confidence"]),
            source_spans=tuple(
                SourceSpan(s["artifact_id"], s["start"], s["end"], s.get("text", "")) for s in payload["source_spans"]
            ),
            parents=tuple(payload["parents"]),
            children=tuple(payload["children"]),
            dependencies=tuple(payload["dependencies"]),
            assumptions=tuple(payload["assumptions"]),
            usg_references=tuple(
                USGReference(r["concept_id"], r["rule_id"], r.get("mapping_type", "UNRESOLVED"), r.get("lifecycle_state", "UNRESOLVED"))
                for r in payload["usg_references"]
            ),
            creation_stage=payload["creation_stage"],
            validation_state=ValidationState(payload["validation_state"]),
            derived_construction=bool(payload["derived_construction"]),
            units=UnitDeclaration(**payload["units"]) if payload.get("units") else None,
        )


class AlgoIRGraph(CanonicalArtifact):
    """Canonical artifact: the versioned, typed, executable-intent graph."""

    kind = ArtifactKind.ALGO_IR

    def __init__(self, *, artifact_id: str, model_id: str, version: int = 1) -> None:
        self.artifact_id = artifact_id
        self.model_id = model_id
        self.version = version
        self._nodes: dict[str, AlgoIRNode] = {}
        self._counter = 0
        self.schema_version = HIC_SCHEMA_VERSION

    # -- construction --------------------------------------------------------

    def next_node_id(self) -> str:
        self._counter += 1
        return f"n{self._counter:04d}"

    def add_node(self, node: AlgoIRNode) -> AlgoIRNode:
        """Add a node after schema checks; unknown types and ids are rejected."""
        if not isinstance(node.node_type, NodeType):
            raise SchemaError(f"unknown or invalid node type: {node.node_type!r}")
        if node.node_id in self._nodes:
            raise SchemaError(f"duplicate node id {node.node_id}")
        self._nodes[node.node_id] = node
        return node

    def node(self, node_id: str) -> AlgoIRNode:
        if node_id not in self._nodes:
            raise SchemaError(f"undefined node {node_id}")
        return self._nodes[node_id]

    def nodes_of(self, node_type: NodeType) -> tuple[AlgoIRNode, ...]:
        return tuple(n for n in self._nodes.values() if n.node_type is node_type)

    @property
    def nodes(self) -> tuple[AlgoIRNode, ...]:
        return tuple(self._nodes.values())

    def __len__(self) -> int:
        return len(self._nodes)

    # -- serialization -------------------------------------------------------

    def content_payload(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "model_id": self.model_id,
            "schema_version": self.schema_version,
            "node_count": len(self._nodes),
            "nodes": [n.to_dict() for n in sorted(self._nodes.values(), key=lambda n: n.node_id)],
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AlgoIRGraph:
        """Rebuild a graph from its canonical serialization (round trip)."""
        graph = cls(
            artifact_id=payload["artifact_id"],
            model_id=payload["model_id"],
            version=int(payload.get("version", 1)),
        )
        for node_payload in payload["nodes"]:
            graph.add_node(AlgoIRNode.from_dict(node_payload))
        graph._counter = len(graph._nodes)
        return graph


# ---------------------------------------------------------------------------
# Validation (Steps 34, 38)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class IRViolation:
    """One well-formedness violation found by the validator."""

    code: str
    node_id: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "node_id": self.node_id, "message": self.message}


class AlgoIRValidator:
    """Validate AlgoIR schema and well-formedness (spec Steps 34, 38)."""

    def validate(self, graph: AlgoIRGraph) -> tuple[IRViolation, ...]:
        violations: list[IRViolation] = []
        violations.extend(self._check_type_consistency(graph))
        violations.extend(self._check_branches(graph))
        violations.extend(self._check_loops(graph))
        violations.extend(self._check_dependency_cycles(graph))
        violations.extend(self._check_units(graph))
        violations.extend(self._check_reachability(graph))
        violations.extend(self._check_outputs(graph))
        violations.extend(self._check_constraint_scopes(graph))
        violations.extend(self._check_blocking_assumptions(graph))
        return tuple(violations)

    # -- checks ---------------------------------------------------------------

    @staticmethod
    def _declared_symbols(graph: AlgoIRGraph) -> set[str]:
        return {
            node.name
            for node in graph.nodes
            if node.node_type in (NodeType.TYPE, NodeType.VARIABLE, NodeType.INPUT, NodeType.OUTPUT, NodeType.ENTITY)
        }

    def _check_type_consistency(self, graph: AlgoIRGraph) -> list[IRViolation]:
        violations: list[IRViolation] = []
        declared = self._declared_symbols(graph)
        for node in graph.nodes_of(NodeType.OPERATION):
            for port in ("inputs", "outputs"):
                for symbol in node.content.get(port, ()):  # type: ignore[union-attr]
                    if symbol not in declared and symbol not in ("tasks", "modules"):
                        violations.append(
                            IRViolation("UNDEFINED_SYMBOL", node.node_id,
                                        f"operation '{node.name}' references undeclared {port[:-1]} '{symbol}'")
                        )
        return violations

    def _check_branches(self, graph: AlgoIRGraph) -> list[IRViolation]:
        violations: list[IRViolation] = []
        declared = self._declared_symbols(graph)
        for node in graph.nodes_of(NodeType.CONDITION):
            predicate = str(node.content.get("predicate", "")).strip()  # type: ignore[union-attr]
            if not predicate:
                violations.append(
                    IRViolation("MALFORMED_PREDICATE", node.node_id, "condition has an empty predicate")
                )
                continue
            referenced = {str(tok) for tok in node.content.get("variables", ())}  # type: ignore[union-attr]
            unknown = referenced - declared
            if unknown:
                violations.append(
                    IRViolation("UNDEFINED_SYMBOL", node.node_id,
                                f"condition '{node.name}' references undefined variables {sorted(unknown)}")
                )
            paths = node.content.get("paths", {})  # type: ignore[union-attr]
            if not isinstance(paths, Mapping) or set(paths) != {"then", "else"}:
                violations.append(
                    IRViolation("INCOMPLETE_BRANCH", node.node_id,
                                "condition must define exactly 'then' and 'else' paths")
                )
        return violations

    def _check_loops(self, graph: AlgoIRGraph) -> list[IRViolation]:
        violations: list[IRViolation] = []
        for node in graph.nodes_of(NodeType.LOOP):
            policy = node.content.get("termination_policy")  # type: ignore[union-attr]
            if policy not in (TerminationPolicy.BOUNDED.value, TerminationPolicy.CONVERGENCE.value,
                              TerminationPolicy.NONTERMINATING_DECLARED.value):
                violations.append(
                    IRViolation("LOOP_TERMINATION", node.node_id,
                                f"loop '{node.name}' lacks a termination argument, bound, or declared policy")
                )
                continue
            if policy == TerminationPolicy.BOUNDED.value and "bound" not in node.content:  # type: ignore[operator]
                violations.append(
                    IRViolation("LOOP_TERMINATION", node.node_id,
                                f"bounded loop '{node.name}' lacks an explicit iteration bound")
                )
            if policy == TerminationPolicy.CONVERGENCE.value and "convergence_criterion" not in node.content:  # type: ignore[operator]
                violations.append(
                    IRViolation("LOOP_TERMINATION", node.node_id,
                                f"convergence loop '{node.name}' lacks a convergence criterion")
                )
        return violations

    def _check_dependency_cycles(self, graph: AlgoIRGraph) -> list[IRViolation]:
        """Detect cycles outside declared iterative constructs (spec 4.3)."""
        violations: list[IRViolation] = []
        iterative_roots = {n.node_id for n in graph.nodes_of(NodeType.LOOP)}
        edges: dict[str, list[str]] = {}
        for node in graph.nodes:
            if node.node_type is NodeType.DEPENDENCY:
                source = str(node.content.get("from", ""))  # type: ignore[union-attr]
                target = str(node.content.get("to", ""))  # type: ignore[union-attr]
                # Ignore edges declared inside a LOOP construct (iterative by design).
                if node.scope in iterative_roots:
                    continue
                edges.setdefault(source, []).append(target)
            else:
                for dep in node.dependencies:
                    if node.node_type is NodeType.LOOP or node.scope in iterative_roots:
                        continue
                    edges.setdefault(node.node_id, []).append(dep)
        state: dict[str, int] = {}

        def visit(node_id: str, stack: tuple[str, ...]) -> None:
            mark = state.get(node_id, 0)
            if mark == 2:
                return
            if mark == 1:
                cycle = stack[stack.index(node_id):] + (node_id,)
                violations.append(
                    IRViolation("DEPENDENCY_CYCLE", node_id,
                                "dependency cycle outside declared iterative constructs: " + " -> ".join(cycle))
                )
                return
            state[node_id] = 1
            for nxt in edges.get(node_id, ()):
                if nxt in graph._nodes:
                    visit(nxt, stack + (nxt,))
            state[node_id] = 2

        for node_id in sorted(graph._nodes):
            visit(node_id, (node_id,))
        return violations

    def _check_units(self, graph: AlgoIRGraph) -> list[IRViolation]:
        violations: list[IRViolation] = []
        unit_by_symbol: dict[str, str] = {}
        for node in graph.nodes:
            if node.units is not None and node.node_type in (NodeType.VARIABLE, NodeType.INPUT, NodeType.OUTPUT):
                unit_by_symbol[node.name] = node.units.unit
        for node in graph.nodes_of(NodeType.OPERATION):
            inputs = list(node.content.get("inputs", ()))  # type: ignore[union-attr]
            outputs = list(node.content.get("outputs", ()))  # type: ignore[union-attr]
            in_units = {unit_by_symbol[s] for s in inputs if s in unit_by_symbol}
            out_units = {unit_by_symbol[s] for s in outputs if s in unit_by_symbol}
            if len(in_units) > 1:
                violations.append(
                    IRViolation("UNIT_MISMATCH", node.node_id,
                                f"operation '{node.name}' mixes input units {sorted(in_units)}")
                )
            if len(out_units) > 1:
                violations.append(
                    IRViolation("UNIT_MISMATCH", node.node_id,
                                f"operation '{node.name}' mixes output units {sorted(out_units)}")
                )
        return violations

    def _check_reachability(self, graph: AlgoIRGraph) -> list[IRViolation]:
        violations: list[IRViolation] = []
        goals = {n.node_id for n in graph.nodes_of(NodeType.GOAL)}
        if not goals:
            violations.append(IRViolation("NO_GOAL", "-", "graph has no GOAL node"))
            return violations
        reachable: set[str] = set()

        # Edges: explicit children/dependencies plus DEPENDENCY node from/to links.
        successors: dict[str, list[str]] = {}
        for node in graph.nodes:
            if node.node_type is NodeType.DEPENDENCY:
                source = str(node.content.get("from", ""))
                target = str(node.content.get("to", ""))
                successors.setdefault(source, []).append(target)
            else:
                successors.setdefault(node.node_id, []).extend((*node.children, *node.dependencies))

        def walk(node_id: str) -> None:
            if node_id in reachable or node_id not in graph._nodes:
                return
            reachable.add(node_id)
            for nxt in successors.get(node_id, ()):
                walk(nxt)

        for goal in goals:
            walk(goal)
        for node in graph.nodes_of(NodeType.OPERATION):
            if node.node_id not in reachable:
                violations.append(
                    IRViolation("UNREACHABLE_OPERATION", node.node_id,
                                f"operation '{node.name}' is unreachable from any GOAL")
                )
        return violations

    def _check_outputs(self, graph: AlgoIRGraph) -> list[IRViolation]:
        violations: list[IRViolation] = []
        outputs = {n.name for n in graph.nodes_of(NodeType.OUTPUT)}
        for node in graph.nodes_of(NodeType.POSTCONDITION):
            for symbol in node.content.get("produces", ()):  # type: ignore[union-attr]
                if symbol not in outputs:
                    violations.append(
                        IRViolation("MISSING_OUTPUT", node.node_id,
                                    f"postcondition expects output '{symbol}' which is not declared")
                    )
        return violations

    def _check_constraint_scopes(self, graph: AlgoIRGraph) -> list[IRViolation]:
        violations: list[IRViolation] = []
        for node in graph.nodes_of(NodeType.CONSTRAINT):
            scope = node.scope
            if scope != "global" and scope not in graph._nodes:
                violations.append(
                    IRViolation("CONSTRAINT_SCOPE", node.node_id,
                                f"constraint '{node.name}' is attached to unknown scope '{scope}'")
                )
        return violations

    def _check_blocking_assumptions(self, graph: AlgoIRGraph) -> list[IRViolation]:
        """Unresolved assumptions stay visible and block unsafe generation."""
        violations: list[IRViolation] = []
        for node in graph.nodes:
            if node.content.get("unresolved"):
                violations.append(
                    IRViolation("UNRESOLVED_ASSUMPTION", node.node_id,
                                f"node '{node.name}' is unresolved and blocks downstream generation")
                )
        return violations


# ---------------------------------------------------------------------------
# Builder (Steps 34-37)
# ---------------------------------------------------------------------------

class AlgoIRBuilder:
    """Instantiate the AlgoIR schema from spec + math model (Steps 34-37)."""

    def build(
        self,
        session: CompilationSession,
        graph_semantic: SemanticGraph,
        spec: RequirementSpec,
        model: MathModel,
    ) -> AlgoIRGraph:
        graph = AlgoIRGraph(
            artifact_id=session.next_artifact_id(ArtifactKind.ALGO_IR),
            model_id=model.artifact_id,
        )
        concept_span = graph_semantic.concepts[0].span if graph_semantic.concepts else SourceSpan("input", 0, 0)

        def node(
            node_type: NodeType,
            name: str,
            description: str,
            content: Mapping[str, Any] | None = None,
            *,
            scope: str = "global",
            confidence: float = 0.9,
            spans: tuple[SourceSpan, ...] = (),
            children: tuple[str, ...] = (),
            dependencies: tuple[str, ...] = (),
            assumptions: tuple[str, ...] = (),
            usg_refs: tuple[USGReference, ...] = (),
            derived: bool = False,
            units: UnitDeclaration | None = None,
            stage: str = "E",
        ) -> AlgoIRNode:
            created = AlgoIRNode(
                node_id=graph.next_node_id(),
                node_type=node_type,
                name=name,
                scope=scope,
                description=description,
                content=dict(content or {}),
                confidence=confidence,
                source_spans=spans or (concept_span,),
                parents=(),
                children=children,
                dependencies=dependencies,
                assumptions=assumptions,
                usg_references=usg_refs,
                creation_stage=stage,
                validation_state=ValidationState.UNVALIDATED,
                derived_construction=derived,
                units=units,
            )
            return graph.add_node(created)

        # Step 34-35: types, entities, variables -----------------------------
        goal = node(NodeType.GOAL, "assign_tasks", "Assign every task to exactly one eligible module "
                     "within capacity, maximizing the weighted preference score.",
                     {"objective": "task-to-module assignment", "direction": "MAXIMIZE"})
        node(NodeType.ENTITY, "task", "A unit of work with a demand.", {"usg": "USG-TASK"},
             usg_refs=(USGReference("USG-TASK", "HIC-USG-R01"),), spans=(concept_span,))
        node(NodeType.ENTITY, "module", "A processing resource with capacity.", {"usg": "USG-MODULE"},
             usg_refs=(USGReference("USG-MODULE", "HIC-USG-R01"),), spans=(concept_span,))
        node(NodeType.TYPE, "Task", "record {id: str, demand: float}", {"fields": {"id": "str", "demand": "float"}})
        node(NodeType.TYPE, "Module", "record {id: str, capacity: float, load: float}",
             {"fields": {"id": "str", "capacity": "float", "load": "float"}})
        node(NodeType.TYPE, "AssignmentReport", "record {status, assignment, objective_value, diagnostics}",
             {"fields": {"status": "str", "assignment": "dict", "objective_value": "float", "diagnostics": "list"}})
        node(NodeType.VARIABLE, "x_ij", "binary decision: task i on module j",
             {"domain": "{0, 1}", "role": "DECISION"}, dependencies=(), derived=True)
        node(NodeType.VARIABLE, "remaining_capacity", "capacity left per module",
             {"domain": "float >= 0", "role": "DERIVED"}, units=UnitDeclaration("unit", "capacity"), derived=True)
        node(NodeType.VARIABLE, "validated", "input validation verdict",
             {"domain": "bool", "role": "DERIVED"}, derived=True)
        node(NodeType.VARIABLE, "scores", "preference score per (task, module)",
             {"domain": "float", "role": "DERIVED"}, derived=True)
        node(NodeType.VARIABLE, "feasible", "feasibility verdict",
             {"domain": "bool", "role": "DERIVED"}, derived=True)

        # INPUT / OUTPUT -------------------------------------------------------
        inp_tasks = node(NodeType.INPUT, "tasks", "list of task records", {"type": "list[Task]"},
                         dependencies=())
        inp_modules = node(NodeType.INPUT, "modules", "list of module records", {"type": "list[Module]"})
        node(NodeType.INPUT, "eligibility", "permitted (task, module) pairs",
             {"type": "dict[tuple[str, str], bool]"})
        inp_cap = node(NodeType.INPUT, "capability", "suitability score per (task, module)",
                       {"type": "dict[tuple[str, str], float]"})
        out_assignment = node(NodeType.OUTPUT, "assignment", "task -> module map", {"type": "dict[str, str]"})
        out_report = node(NodeType.OUTPUT, "report", "structured outcome report", {"type": "AssignmentReport"})

        # PRECONDITION / POSTCONDITION / INVARIANT ------------------------------
        node(NodeType.PRECONDITION, "inputs_present", "tasks and modules are non-empty",
             {"predicate": "len(tasks) > 0 and len(modules) > 0", "kind": "NON_EMPTY_INPUT"},
             dependencies=(inp_tasks.node_id, inp_modules.node_id))
        node(NodeType.POSTCONDITION, "all_tasks_assigned", "each task assigned exactly once",
             {"predicate": "all(t in assignment for t in tasks)", "produces": ("assignment",)},
             dependencies=(out_assignment.node_id,))
        node(NodeType.POSTCONDITION, "capacity_respected", "no module over capacity",
             {"predicate": "per-module demand <= capacity", "produces": ("assignment", "report")},
             dependencies=(out_report.node_id,))
        node(NodeType.INVARIANT, "eligibility_invariant", "x_ij <= e_ij at all times",
             {"predicate": "assignment[t] eligible for every t"})

        # OPERATIONS with typed inputs/outputs (Step 36) -------------------------
        op_validate = node(NodeType.OPERATION, "validate_inputs", "fail fast on malformed input",
                            {"inputs": ["tasks", "modules"], "outputs": ["validated"],
                             "body": "raise on empty lists or negative numbers"},
                            dependencies=(inp_tasks.node_id, inp_modules.node_id), derived=True)
        op_score = node(NodeType.OPERATION, "score_pairs", "compute alpha*c_ij - beta*l_j - gamma*r_ij",
                        {"inputs": ["capability", "modules"], "outputs": ["scores"],
                         "body": "weighted preference score per (task, module)"},
                        dependencies=(inp_cap.node_id, inp_modules.node_id), derived=True)
        loop = node(NodeType.LOOP, "for_each_task", "iterate tasks once",
                    {"termination_policy": "BOUNDED", "bound": "len(tasks)", "iterates": "tasks",
                     "termination_argument": "one iteration per task; total bounded by len(tasks)"},
                    children=(), derived=True)
        op_assign = node(NodeType.OPERATION, "select_module", "choose best feasible module for the task",
                         {"inputs": ["scores", "modules"], "outputs": ["assignment"],
                          "body": "argmax score among eligible modules with remaining capacity"},
                         scope=loop.node_id, dependencies=(op_score.node_id,), derived=True)
        op_check = node(NodeType.OPERATION, "check_feasibility", "verify exactly-one and capacity",
                        {"inputs": ["assignment"], "outputs": ["feasible"],
                         "body": "evaluate postconditions"},
                        scope=loop.node_id, dependencies=(op_assign.node_id,), derived=True)
        cond = node(NodeType.CONDITION, "feasible?", "branch on feasibility",
                    {"predicate": "feasible and all tasks assigned", "variables": ["assignment"],
                     "paths": {"then": "build report", "else": "emit failure report"}},
                    scope=loop.node_id, dependencies=(op_check.node_id,), derived=True)
        # The loop body contains scoring, selection, checking, and the branch.
        object.__setattr__(loop, "children", (op_score.node_id, op_assign.node_id, op_check.node_id, cond.node_id))
        op_report = node(NodeType.OPERATION, "build_report", "assemble the structured report",
                         {"inputs": ["assignment"], "outputs": ["report"],
                          "body": "status, assignment, objective value, diagnostics"},
                         dependencies=(out_report.node_id,), derived=True)
        op_fail = node(NodeType.OPERATION, "emit_failure", "structured failure report",
                       {"inputs": [], "outputs": ["report"], "body": "status=FAILED with reason"},
                       dependencies=(out_report.node_id,), derived=True)

        # DEPENDENCY nodes -------------------------------------------------------
        def dependency(name: str, source: str, target: str) -> None:
            node(NodeType.DEPENDENCY, name, f"{source} before {target}",
                 {"from": source, "to": target}, derived=True)

        dependency("dep-validate-score", op_validate.node_id, op_score.node_id)
        dependency("dep-score-assign", op_score.node_id, op_assign.node_id)
        dependency("dep-assign-check", op_assign.node_id, op_check.node_id)
        dependency("dep-check-cond", op_check.node_id, cond.node_id)
        dependency("dep-cond-report", cond.node_id, op_report.node_id)
        dependency("dep-cond-fail", cond.node_id, op_fail.node_id)

        # CONSTRAINTS / FAILURE / COMPLEXITY / TEST / PROVENANCE -----------------
        node(NodeType.CONSTRAINT, "exactly_one", "sum_j x_ij == 1 for each task",
             {"expression": "exactly-one assignment", "hard": True}, scope="global",
             spans=(concept_span,))
        node(NodeType.CONSTRAINT, "capacity", "sum_i x_ij d_i <= K_j",
             {"expression": "per-module capacity", "hard": True}, scope=op_assign.node_id)
        node(NodeType.CONSTRAINT, "eligibility", "x_ij <= e_ij",
             {"expression": "eligible pairs only", "hard": True}, scope=op_assign.node_id)
        for failure in spec.failure_cases:
            node(NodeType.FAILURE_CASE, failure.failure_id, failure.trigger,
                 {"behavior": failure.behavior, "reversible": failure.reversible,
                  "compensating_action": failure.compensating_action},
                 spans=(concept_span,), derived=True)
        for estimate in model.complexity:
            node(NodeType.COMPLEXITY, estimate.function, estimate.expression,
                 {"case": estimate.case, "input_size_assumption": estimate.input_size_assumption,
                  "confidence": estimate.confidence}, derived=True)
        for criterion in spec.acceptance_criteria:
            node(NodeType.TEST, criterion.criterion_id, criterion.statement,
                 {"test_kind": criterion.test_kind, "requirement": criterion.requirement_id},
                 spans=(concept_span,), derived=True)
        node(NodeType.PROVENANCE, "source_lineage", "links this IR to the original input record",
             {"input_record": graph_semantic.record_id, "semantic_graph": graph_semantic.artifact_id,
              "requirement_spec": spec.artifact_id, "math_model": model.artifact_id},
             spans=(concept_span,), derived=False)

        # goal wiring: goal -> validate
        goal_node = graph.node(goal.node_id)
        object.__setattr__(goal_node, "children", (op_validate.node_id,))

        session.register(graph)
        session.ledger.record(
            stage="E",
            step_id="HIC34-37",
            description="Instantiate AlgoIR schema and map semantics to executable types",
            inputs=[graph_semantic, spec, model],
            outputs=[graph],
            rule_references={"HIC-IR-1", "HIC-IR-2"},
        )
        return graph

    def validate_and_explain(
        self,
        session: CompilationSession,
        graph: AlgoIRGraph,
    ) -> tuple[AlgoIRGraph, tuple[IRViolation, ...], str]:
        """Steps 38-39: validate well-formedness, render explanation."""
        validator = AlgoIRValidator()
        violations = validator.validate(graph)
        for node in graph.nodes:
            object.__setattr__(
                node,
                "validation_state",
                ValidationState.VALID if not any(v.node_id == node.node_id for v in violations)
                else ValidationState.INVALID,
            )
        renderer = AlgoIRRenderer()
        explanation = renderer.render(graph)
        session.ledger.record(
            stage="E",
            step_id="HIC38-39",
            description="Validate AlgoIR well-formedness and produce human-readable explanation",
            inputs=[graph],
            outputs=[graph],
            rule_references={"HIC-IR-VALIDATE"},
        )
        session.ledger.audit(
            "IR_VALIDATION",
            graph.content_digest,
            f"violations={len(violations)}",
            *(f"{v.code}:{v.node_id}" for v in violations),
        )
        return graph, violations, explanation


# ---------------------------------------------------------------------------
# Human-readable explanation (Step 39)
# ---------------------------------------------------------------------------

class AlgoIRRenderer:
    """Render an explanation derived from the canonical IR (spec Step 39)."""

    _SECTION_ORDER = (
        NodeType.GOAL, NodeType.ENTITY, NodeType.TYPE, NodeType.VARIABLE, NodeType.INPUT,
        NodeType.OUTPUT, NodeType.PRECONDITION, NodeType.POSTCONDITION, NodeType.INVARIANT,
        NodeType.OPERATION, NodeType.CONDITION, NodeType.LOOP, NodeType.DEPENDENCY,
        NodeType.CONSTRAINT, NodeType.FAILURE_CASE, NodeType.COMPLEXITY, NodeType.TEST,
        NodeType.PROVENANCE,
    )

    def render(self, graph: AlgoIRGraph) -> str:
        lines: list[str] = [f"# AlgoIR explanation ({graph.artifact_id})", ""]
        lines.append(f"Nodes: {len(graph)}; schema {graph.schema_version}.")
        lines.append("")
        for node_type in self._SECTION_ORDER:
            nodes = graph.nodes_of(node_type)
            if not nodes:
                continue
            lines.append(f"## {node_type.value}")
            lines.append("")
            for node in nodes:
                provenance = (
                    ", ".join(f"{s.artifact_id}[{s.start}:{s.end}]" for s in node.source_spans)
                    if node.source_spans
                    else "derived construction"
                )
                lines.append(
                    f"- **{node.name}** ({node.node_id}, scope={node.scope}, "
                    f"confidence={node.confidence:.2f}, state={node.validation_state.value}): "
                    f"{node.description} [provenance: {provenance}]"
                )
            lines.append("")
        return "\n".join(lines)
