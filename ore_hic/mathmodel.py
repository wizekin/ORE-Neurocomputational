"""Mathematical formalization (spec Phase D, Steps 27-33; Section 3).

Identifies mathematical structure, declares typed variables with domains,
units and roles, formulates equations, constraints and objectives with
derivation provenance, validates modeling assumptions, selects mathematical
tools whose structure fits the problem, and estimates complexity with explicit
input-size assumptions and confidence. The task-to-module assignment model of
spec 3.3 is implemented as the built-in worked model:

    maximize   sum_i sum_j x_ij (alpha c_ij - beta l_j - gamma r_ij)
    subject to sum_j x_ij = 1                    for each task i      (exactly-one)
               sum_i x_ij d_i <= K_j             for each module j    (capacity)
               x_ij <= e_ij,  x_ij in {0,1}                          (eligibility)

The model is valid only if tasks require one module and costs are additive;
these assumptions are stored explicitly with the model (spec 3.3).
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .contracts import (
    ArtifactKind,
    CanonicalArtifact,
    MathStructureKind,
    SolverToolKind,
    UnitDeclaration,
    VariableRole,
)
from .requirements import RequirementSpec
from .semantic import SemanticGraph
from .session import CompilationSession


@dataclass(frozen=True, slots=True)
class MathVariable:
    """A typed mathematical variable with domain, unit, scope, role (Step 28)."""

    symbol: str
    role: VariableRole
    domain: str
    unit: UnitDeclaration | None
    scope: str
    description: str
    origin_requirement: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "role": self.role.value,
            "domain": self.domain,
            "unit": self.unit.to_dict() if self.unit else None,
            "scope": self.scope,
            "description": self.description,
            "origin_requirement": self.origin_requirement,
        }


@dataclass(frozen=True, slots=True)
class MathEquation:
    """An equation, inequality, or transition with derivation (Step 29)."""

    equation_id: str
    kind: str  # EQUALITY | INEQUALITY | LOGICAL | OBJECTIVE
    expression: str
    derivation: str
    origin_requirement: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "equation_id": self.equation_id,
            "kind": self.kind,
            "expression": self.expression,
            "derivation": self.derivation,
            "origin_requirement": self.origin_requirement,
        }


@dataclass(frozen=True, slots=True)
class MathObjective:
    """An objective function with terms, weights, units (Step 30)."""

    objective_id: str
    direction: str  # MAXIMIZE | MINIMIZE
    expression: str
    terms: tuple[str, ...]
    weights: tuple[float, ...]
    unit_note: str
    origin_requirement: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_id": self.objective_id,
            "direction": self.direction,
            "expression": self.expression,
            "terms": list(self.terms),
            "weights": list(self.weights),
            "unit_note": self.unit_note,
            "origin_requirement": self.origin_requirement,
        }


@dataclass(frozen=True, slots=True)
class ModelAssumption:
    """A modeling assumption with validation status (Step 31)."""

    assumption_id: str
    statement: str
    supported_by_evidence: bool
    consequence_if_false: str
    blocks_solver: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "assumption_id": self.assumption_id,
            "statement": self.statement,
            "supported_by_evidence": self.supported_by_evidence,
            "consequence_if_false": self.consequence_if_false,
            "blocks_solver": self.blocks_solver,
        }


@dataclass(frozen=True, slots=True)
class ToolSelection:
    """A selected mathematical tool with fit rationale (Step 32)."""

    tool: SolverToolKind
    rationale: str
    fits_structure: bool

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool.value, "rationale": self.rationale, "fits_structure": self.fits_structure}


@dataclass(frozen=True, slots=True)
class ComplexityEstimate:
    """A time/space/resource estimate with assumptions and confidence (Step 33)."""

    estimate_id: str
    function: str  # T_A(n), S_A(n), R_A(n)
    expression: str
    input_size_assumption: str
    case: str  # WORST | AVERAGE | EXPECTED
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate_id": self.estimate_id,
            "function": self.function,
            "expression": self.expression,
            "input_size_assumption": self.input_size_assumption,
            "case": self.case,
            "confidence": self.confidence,
        }


class MathModel(CanonicalArtifact):
    """Canonical artifact: the formalized mathematical model."""

    kind = ArtifactKind.MATH_MODEL

    def __init__(
        self,
        *,
        artifact_id: str,
        spec_id: str,
        structures: Iterable[MathStructureKind],
        variables: Iterable[MathVariable],
        equations: Iterable[MathEquation],
        objectives: Iterable[MathObjective],
        assumptions: Iterable[ModelAssumption],
        tools: Iterable[ToolSelection],
        complexity: Iterable[ComplexityEstimate],
        model_family: str,
        version: int = 1,
    ) -> None:
        self.artifact_id = artifact_id
        self.spec_id = spec_id
        self.structures = tuple(structures)
        self.variables = tuple(variables)
        self.equations = tuple(equations)
        self.objectives = tuple(objectives)
        self.assumptions = tuple(assumptions)
        self.tools = tuple(tools)
        self.complexity = tuple(complexity)
        self.model_family = model_family
        self.version = version

    @property
    def invalid_assumptions(self) -> tuple[ModelAssumption, ...]:
        return tuple(a for a in self.assumptions if not a.supported_by_evidence)

    def content_payload(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "spec_id": self.spec_id,
            "structures": [s.value for s in self.structures],
            "variables": [v.to_dict() for v in self.variables],
            "equations": [e.to_dict() for e in self.equations],
            "objectives": [o.to_dict() for o in self.objectives],
            "assumptions": [a.to_dict() for a in self.assumptions],
            "tools": [t.to_dict() for t in self.tools],
            "complexity": [c.to_dict() for c in self.complexity],
            "model_family": self.model_family,
            "version": self.version,
        }


class MathFormalizer:
    """Formalize the mathematical model from requirements (Steps 27-33)."""

    def identify_structure(self, graph: SemanticGraph, spec: RequirementSpec) -> tuple[MathStructureKind, ...]:
        """Step 27: detect classification/search/optimization/... structure."""
        structures: list[MathStructureKind] = []
        text = " ".join(r.description for r in spec.requirements).casefold()
        if any(word in text for word in ("assign", "allocation", "mapping")):
            structures.append(MathStructureKind.ASSIGNMENT)
        if any(word in text for word in ("maximize", "minimize", "prefer", "objective")):
            structures.append(MathStructureKind.OPTIMIZATION)
        if any(word in text for word in ("schedule", "scheduling", "slot", "deadline")):
            structures.append(MathStructureKind.SCHEDULING)
        has_relation = bool(graph.relations)
        if has_relation:
            structures.append(MathStructureKind.GRAPH)
        if not structures:
            structures.append(MathStructureKind.UNKNOWN)
        return tuple(structures)

    def build_assignment_model(self, spec: RequirementSpec) -> MathModel:
        """Build the spec 3.3 task-to-module assignment model."""
        variables = (
            MathVariable("x_ij", VariableRole.DECISION, "binary {0, 1}", None, "task i, module j",
                         "1 if task i is assigned to module j", "req-exactly-one"),
            MathVariable("c_ij", VariableRole.PARAMETER, "real [0, 1]", None, "task i, module j",
                         "capability suitability of module j for task i", "req-preference"),
            MathVariable("l_j", VariableRole.OBSERVED, "real >= 0", UnitDeclaration("unit", "load"), "module j",
                         "current load of module j", "req-preference"),
            MathVariable("r_ij", VariableRole.PARAMETER, "real >= 0", None, "task i, module j",
                         "resource cost of running task i on module j", "req-preference"),
            MathVariable("d_i", VariableRole.OBSERVED, "real >= 0", UnitDeclaration("unit", "demand"), "task i",
                         "resource demand of task i", "req-capacity"),
            MathVariable("K_j", VariableRole.OBSERVED, "real > 0", UnitDeclaration("unit", "capacity"), "module j",
                         "capacity of module j", "req-capacity"),
            MathVariable("e_ij", VariableRole.PARAMETER, "binary {0, 1}", None, "task i, module j",
                         "eligibility of module j for task i", "req-eligibility"),
            MathVariable("alpha", VariableRole.PARAMETER, "real > 0", None, "global",
                         "weight of capability suitability", "req-preference"),
            MathVariable("beta", VariableRole.PARAMETER, "real >= 0", None, "global",
                         "weight of load penalty", "req-preference"),
            MathVariable("gamma", VariableRole.PARAMETER, "real >= 0", None, "global",
                         "weight of resource cost penalty", "req-preference"),
        )
        equations = (
            MathEquation("eq-001", "EQUALITY", "sum_j x_ij = 1 for each task i",
                         "each task must be assigned to exactly one module", "req-exactly-one"),
            MathEquation("eq-002", "INEQUALITY", "sum_i x_ij * d_i <= K_j for each module j",
                         "total assigned demand must not exceed module capacity", "req-capacity"),
            MathEquation("eq-003", "INEQUALITY", "x_ij <= e_ij",
                         "assignments only to eligible modules", "req-eligibility"),
            MathEquation("eq-004", "LOGICAL", "infeasible -> report failure, emit no assignment",
                         "failure behavior for unsatisfiable instances", "req-failure"),
        )
        objectives = (
            MathObjective("obj-001", "MAXIMIZE",
                          "sum_i sum_j x_ij (alpha c_ij - beta l_j - gamma r_ij)",
                          ("capability gain", "load penalty", "resource cost penalty"),
                          (1.0, 1.0, 1.0),
                          "all terms are unitless scores; alpha/beta/gamma trade them off",
                          "req-preference"),
        )
        assumptions = (
            ModelAssumption(
                "as-001", "each task requires exactly one module", True,
                "a set-cover or split-assignment formulation would be required", True,
            ),
            ModelAssumption(
                "as-002", "costs and demands are additive across tasks", True,
                "nonlinear or packing constraints would invalidate the model", True,
            ),
            ModelAssumption(
                "as-003", "no precedence constraints between tasks", True,
                "scheduling formulations with ordering variables would be required", False,
            ),
            ModelAssumption(
                "as-004", "eligibility is static during assignment", True,
                "dynamic eligibility would require a fixed-point or iterative method", False,
            ),
            ModelAssumption(
                "as-005", "tasks cannot be split across modules", True,
                "split tasks would need fractional x_ij and a different formulation", True,
            ),
        )
        tools = (
            ToolSelection(SolverToolKind.LINEAR_INTEGER_PROGRAMMING,
                          "binary decision variables with linear objective and linear constraints", True),
            ToolSelection(SolverToolKind.EXHAUSTIVE_ENUMERATION,
                          "exact optimum for small N*M by enumerating assignment maps", True),
            ToolSelection(SolverToolKind.GRAPH_ALGORITHM,
                          "bipartite matching applies when capacities are unit-sized", True),
            ToolSelection(SolverToolKind.BAYESIAN_INFERENCE,
                          "no probabilistic structure present; does not fit", False),
        )
        complexity = (
            ComplexityEstimate("cx-001", "T_A(n)", "O(N * M) greedy; O(N * M^2) Hungarian; O(M^N) exhaustive",
                               "N tasks, M modules, N, M <= 20", "WORST", 0.8),
            ComplexityEstimate("cx-002", "S_A(n)", "O(N + M) bookkeeping plus O(N * M) inputs",
                               "N tasks, M modules", "WORST", 0.9),
            ComplexityEstimate("cx-003", "R_A(n)", "no network or accelerator required; CPU only",
                               "commodity CPU", "EXPECTED", 0.7),
        )
        model = MathModel(
            artifact_id="pending",
            spec_id=spec.artifact_id,
            structures=(MathStructureKind.ASSIGNMENT, MathStructureKind.OPTIMIZATION),
            variables=variables,
            equations=equations,
            objectives=objectives,
            assumptions=assumptions,
            tools=tools,
            complexity=complexity,
            model_family="TASK_ASSIGNMENT_ILP",
        )
        return model

    def validate_assumptions(self, model: MathModel) -> tuple[ModelAssumption, ...]:
        """Step 31: flag assumptions not supported by evidence before solving."""
        return model.invalid_assumptions

    def formalize(
        self,
        session: CompilationSession,
        graph: SemanticGraph,
        spec: RequirementSpec,
    ) -> MathModel:
        """Run Steps 27-33 and register the MathModel artifact."""
        structures = self.identify_structure(graph, spec)
        if MathStructureKind.ASSIGNMENT in structures:
            model = self.build_assignment_model(spec)
            model = MathModel(
                artifact_id=session.next_artifact_id(ArtifactKind.MATH_MODEL),
                spec_id=spec.artifact_id,
                structures=model.structures,
                variables=model.variables,
                equations=model.equations,
                objectives=model.objectives,
                assumptions=model.assumptions,
                tools=model.tools,
                complexity=model.complexity,
                model_family=model.model_family,
            )
        else:
            model = MathModel(
                artifact_id=session.next_artifact_id(ArtifactKind.MATH_MODEL),
                spec_id=spec.artifact_id,
                structures=structures,
                variables=(),
                equations=(),
                objectives=(),
                assumptions=(
                    ModelAssumption("as-unknown", "problem structure not recognized", False,
                                    "no solver can be selected safely", True),
                ),
                tools=(ToolSelection(SolverToolKind.HEURISTIC,
                                     "structure unknown; no tool forced onto the problem", False),),
                complexity=(ComplexityEstimate("cx-unknown", "T_A(n)", "unknown",
                                               "input sizes unknown", "WORST", 0.0),),
                model_family="UNRECOGNIZED",
            )
        session.register(model)
        session.ledger.record(
            stage="D",
            step_id="HIC27-33",
            description="Mathematical formalization with explicit assumptions and complexity",
            inputs=[graph, spec],
            outputs=[model],
            rule_references={"HIC-MATH-1", "HIC-MATH-2"},
        )
        return model


def assignment_objective_value(
    assignment: dict[str, str],
    *,
    capability: dict[tuple[str, str], float],
    loads: dict[str, float],
    costs: dict[tuple[str, str], float],
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> float:
    """Evaluate the spec 3.3 objective for one concrete assignment."""
    total = 0.0
    for task, module in assignment.items():
        total += (
            alpha * capability.get((task, module), 0.0)
            - beta * loads.get(module, 0.0)
            - gamma * costs.get((task, module), 0.0)
        )
    if not math.isfinite(total):
        raise ValueError("objective value is not finite")
    return total
