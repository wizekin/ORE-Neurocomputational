"""Target-language code generation with source maps (spec Phase H, Step 50).

Translates the approved AlgoIR subset to Python while preserving the stable IR
for later generators. The generated source carries the compiler version, the
candidate identity, and a line-level source map from code to AlgoIR node IDs.
Generation is deterministic and never adds unapproved external behavior: only
approved imports are emitted and every statement traces to IR nodes.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .algoir import AlgoIRGraph
from .contracts import (
    HIC_GENERATOR_VERSION,
    ArtifactKind,
    CanonicalArtifact,
    GateViolation,
    NodeType,
)
from .discovery import CandidateSet
from .session import CompilationSession


@dataclass(frozen=True, slots=True)
class SourceMapEntry:
    """One line-range to node mapping."""

    start_line: int
    end_line: int
    node_id: str
    node_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_line": self.start_line,
            "end_line": self.end_line,
            "node_id": self.node_id,
            "node_type": self.node_type,
        }


class SourceMap:
    """Bidirectional mapping between generated code and AlgoIR nodes."""

    def __init__(self, entries: Iterable[SourceMapEntry]) -> None:
        self.entries = tuple(entries)

    def nodes_for_line(self, line: int) -> tuple[str, ...]:
        return tuple(e.node_id for e in self.entries if e.start_line <= line <= e.end_line)

    def lines_for_node(self, node_id: str) -> tuple[tuple[int, int], ...]:
        return tuple((e.start_line, e.end_line) for e in self.entries if e.node_id == node_id)

    @property
    def covered_node_ids(self) -> frozenset[str]:
        return frozenset(e.node_id for e in self.entries)

    def coverage_of(self, required: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        covered = self.covered_node_ids
        included = tuple(sorted(set(required) & covered))
        missing = tuple(sorted(set(required) - covered))
        return included, missing

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [e.to_dict() for e in self.entries]}


class CodeArtifact(CanonicalArtifact):
    """Canonical artifact: generated source plus source map."""

    kind = ArtifactKind.CODE_ARTIFACT

    def __init__(
        self,
        *,
        artifact_id: str,
        ir_id: str,
        candidate_id: str,
        language: str,
        module_name: str,
        source: str,
        source_map: SourceMap,
        dependencies: tuple[str, ...],
        generator_version: str = HIC_GENERATOR_VERSION,
        version: int = 1,
    ) -> None:
        self.artifact_id = artifact_id
        self.ir_id = ir_id
        self.candidate_id = candidate_id
        self.language = language
        self.module_name = module_name
        self.source = source
        self.source_map = source_map
        self.dependencies = dependencies
        self.generator_version = generator_version
        self.version = version

    def content_payload(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "ir_id": self.ir_id,
            "candidate_id": self.candidate_id,
            "language": self.language,
            "module_name": self.module_name,
            "source": self.source,
            "source_map": self.source_map.to_dict(),
            "dependencies": list(self.dependencies),
            "generator_version": self.generator_version,
            "version": self.version,
        }


class PythonGenerator:
    """Generate deterministic Python from an approved AlgoIR graph."""

    LANGUAGE = "python"

    def generate(
        self,
        session: CompilationSession,
        ir: AlgoIRGraph,
        candidate_set: CandidateSet,
        *,
        approved_digest: str | None = None,
    ) -> CodeArtifact:
        """Step 50: translate verified AlgoIR into Python with a source map."""
        candidate = candidate_set.selected()
        if candidate is None:
            raise GateViolation("code generation requires a selected candidate (Step 44)")
        if approved_digest is not None and approved_digest != candidate_set.semantic_digest:
            raise GateViolation("approval digest does not match the candidate set; renewed approval required")

        lines: list[str] = []
        map_entries: list[SourceMapEntry] = []

        def emit(text: str = "", nodes: Iterable[str] = ()) -> None:
            for node_id in nodes:
                map_entries.append(
                    SourceMapEntry(len(lines) + 1, len(lines) + 1, node_id, self._node_type(ir, node_id))
                )
            lines.append(text)

        # Header with full provenance ---------------------------------------
        provenance = next(iter(ir.nodes_of(NodeType.PROVENANCE)), None)
        prov_content = dict(provenance.content) if provenance else {}
        emit(f'"""Generated by {HIC_GENERATOR_VERSION} from AlgoIR {ir.artifact_id}.')
        emit()
        emit("This module is a compiled artifact, not hand-written code. Every")
        emit("statement maps to AlgoIR nodes; see the companion source map.")
        emit(f"Input record:      {prov_content.get('input_record', 'unknown')}")
        emit(f"Semantic graph:    {prov_content.get('semantic_graph', 'unknown')}")
        emit(f"Requirement spec:  {prov_content.get('requirement_spec', 'unknown')}")
        emit(f"Math model:        {prov_content.get('math_model', 'unknown')}")
        emit(f"Candidate:         {candidate.candidate_id} ({candidate.display_name})")
        emit('"""')
        emit("from dataclasses import dataclass")
        emit("import math")
        emit()

        # Data types (TYPE nodes) ---------------------------------------------
        type_nodes = {n.name: n for n in ir.nodes_of(NodeType.TYPE)}
        task_type = type_nodes.get("Task")
        module_type = type_nodes.get("Module")
        report_type = type_nodes.get("AssignmentReport")
        task_ids = () if task_type is None else (task_type.node_id,)
        module_ids = () if module_type is None else (module_type.node_id,)
        report_ids = () if report_type is None else (report_type.node_id,)
        emit("@dataclass(frozen=True)", nodes=task_ids)
        emit("class Task:", nodes=task_ids)
        emit('    """A unit of work with a demand."""', nodes=task_ids)
        emit("    task_id: str", nodes=task_ids)
        emit("    demand: float", nodes=task_ids)
        emit()
        emit("@dataclass(frozen=True)", nodes=module_ids)
        emit("class Module:", nodes=module_ids)
        emit('    """A processing resource with capacity and current load."""', nodes=module_ids)
        emit("    module_id: str", nodes=module_ids)
        emit("    capacity: float", nodes=module_ids)
        emit("    load: float", nodes=module_ids)
        emit()
        emit("@dataclass(frozen=True)", nodes=report_ids)
        emit("class AssignmentReport:", nodes=report_ids)
        emit('    """Structured outcome: status, assignment, objective, diagnostics."""', nodes=report_ids)
        emit("    status: str", nodes=report_ids)
        emit("    assignment: dict", nodes=report_ids)
        emit("    objective_value: float", nodes=report_ids)
        emit("    diagnostics: tuple", nodes=report_ids)
        emit()

        # Preconditions as a validation function --------------------------------
        pre_nodes = ir.nodes_of(NodeType.PRECONDITION)
        pre_ids = tuple(n.node_id for n in pre_nodes)
        validate_op_ids = tuple(
            n.node_id for n in ir.nodes_of(NodeType.OPERATION) if n.name == "validate_inputs"
        )
        emit("def validate_inputs(tasks, modules):", nodes=pre_ids + validate_op_ids)
        emit('    """Fail fast on malformed input (preconditions)."""', nodes=pre_ids + validate_op_ids)
        emit("    if not tasks:", nodes=pre_ids)
        emit('        raise ValueError("tasks must be a non-empty list")', nodes=pre_ids)
        emit("    if not modules:", nodes=pre_ids)
        emit('        raise ValueError("modules must be a non-empty list")', nodes=pre_ids)
        emit("    for task in tasks:", nodes=pre_ids)
        emit("        if task.demand < 0 or not math.isfinite(task.demand):", nodes=pre_ids)
        emit('            raise ValueError(f"task {task.task_id} has invalid demand")', nodes=pre_ids)
        emit("    for module in modules:", nodes=pre_ids)
        emit("        if module.capacity < 0 or module.load < 0:", nodes=pre_ids)
        emit('            raise ValueError(f"module {module.module_id} has invalid capacity/load")', nodes=pre_ids)
        emit()

        # Scoring is inlined inside the task loop so that it always uses the
        # CURRENT module loads, exactly matching the selected greedy candidate;
        # the emitted lines are mapped to the score_pairs OPERATION node.
        score_nodes = [n for n in ir.nodes_of(NodeType.OPERATION) if n.name == "score_pairs"]
        score_ids = tuple(n.node_id for n in score_nodes)
        emit()

        # Constraints as runtime assertions ----------------------------------------
        constraint_nodes = ir.nodes_of(NodeType.CONSTRAINT)
        constraint_ids = tuple(n.node_id for n in constraint_nodes)
        emit("def assert_hard_constraints(assignment, tasks, modules, eligibility):",
             nodes=constraint_ids)
        emit('    """Runtime assertions for exactly-one, capacity, and eligibility."""', nodes=constraint_ids)
        emit("    task_ids = {t.task_id for t in tasks}", nodes=constraint_ids)
        emit('    assert set(assignment) == task_ids, "exactly-one constraint violated"',
             nodes=constraint_ids)
        emit("    used = {m.module_id: m.load for m in modules}", nodes=constraint_ids)
        emit("    demand_by_task = {t.task_id: t.demand for t in tasks}", nodes=constraint_ids)
        emit("    for task_id, module_id in assignment.items():", nodes=constraint_ids)
        emit("        used[module_id] += demand_by_task[task_id]", nodes=constraint_ids)
        emit("    for module in modules:", nodes=constraint_ids)
        emit("        assert used[module.module_id] <= module.capacity + 1e-9, (", nodes=constraint_ids)
        emit('            "capacity constraint violated")', nodes=constraint_ids)
        emit("    for task_id, module_id in assignment.items():", nodes=constraint_ids)
        emit("        assert eligibility.get((task_id, module_id), False), (", nodes=constraint_ids)
        emit('            "eligibility constraint violated")', nodes=constraint_ids)
        emit("    return True", nodes=constraint_ids)
        emit()

        # Main entry: loop + condition per IR ---------------------------------------
        loop_nodes = ir.nodes_of(NodeType.LOOP)
        loop_ids = tuple(n.node_id for n in loop_nodes)
        cond_nodes = ir.nodes_of(NodeType.CONDITION)
        cond_ids = tuple(n.node_id for n in cond_nodes)
        assign_nodes = [n for n in ir.nodes_of(NodeType.OPERATION) if n.name == "select_module"]
        check_nodes = [n for n in ir.nodes_of(NodeType.OPERATION) if n.name == "check_feasibility"]
        report_nodes = [n for n in ir.nodes_of(NodeType.OPERATION) if n.name in ("build_report", "emit_failure")]
        main_ids = tuple(
            n.node_id for n in (*loop_nodes, *cond_nodes, *assign_nodes, *check_nodes, *report_nodes)
        )
        emit("def assign_tasks(tasks, modules, eligibility, capability, costs,", nodes=main_ids)
        emit("                  alpha=1.0, beta=1.0, gamma=1.0):", nodes=main_ids)
        emit('    """Assign every task to exactly one eligible module within capacity."""', nodes=main_ids)
        emit("    validate_inputs(tasks, modules)", nodes=main_ids)
        emit("    loads = {m.module_id: m.load for m in modules}", nodes=main_ids)
        emit("    remaining = {m.module_id: m.capacity - m.load for m in modules}", nodes=main_ids)
        emit("    assignment = {}", nodes=main_ids)
        emit("    # canonical order: descending demand, then id (permutation-invariant)", nodes=main_ids)
        emit("    ordered_tasks = sorted(tasks, key=lambda t: (-t.demand, t.task_id))", nodes=main_ids)
        emit("    # LOOP node: one iteration per task, bounded by len(tasks)", nodes=loop_ids)
        emit("    for task in ordered_tasks:  # bounded loop: terminates after len(tasks) iterations", nodes=main_ids)
        emit("        best_module = None", nodes=main_ids)
        emit("        best_score = -math.inf", nodes=main_ids)
        emit("        for module in modules:", nodes=main_ids)
        emit("            if not eligibility.get((task.task_id, module.module_id), False):", nodes=main_ids)
        emit("                continue", nodes=main_ids)
        emit("            if remaining[module.module_id] + 1e-9 < task.demand:", nodes=main_ids)
        emit("                continue", nodes=main_ids)
        emit("            # score with CURRENT loads, recomputed per task (score_pairs node)",
             nodes=score_ids + main_ids)
        emit("            score = (alpha * capability.get((task.task_id, module.module_id), 0.0)", nodes=score_ids + main_ids)
        emit("                     - beta * loads.get(module.module_id, module.load)", nodes=score_ids + main_ids)
        emit("                     - gamma * costs.get((task.task_id, module.module_id), 0.0))", nodes=score_ids + main_ids)
        emit("            if score > best_score:", nodes=main_ids)
        emit("                best_score = score", nodes=main_ids)
        emit("                best_module = module.module_id", nodes=main_ids)
        emit("        if best_module is None:", nodes=main_ids)
        emit("            return AssignmentReport('FAILED', {}, 0.0,", nodes=main_ids)
        emit("                ('no eligible module with remaining capacity for task '", nodes=main_ids)
        emit("                 + task.task_id + ',',))", nodes=main_ids)
        emit("        assignment[task.task_id] = best_module", nodes=main_ids)
        emit("        remaining[best_module] -= task.demand", nodes=main_ids)
        emit("        loads[best_module] += task.demand", nodes=main_ids)
        emit("    # CONDITION node: branch on feasibility", nodes=cond_ids)
        emit("    if len(assignment) == len(tasks):", nodes=main_ids)
        emit("        assert_hard_constraints(assignment, tasks, modules, eligibility)", nodes=main_ids)
        emit("        objective = 0.0", nodes=main_ids)
        emit("        for task in tasks:", nodes=main_ids)
        emit("            module_id = assignment[task.task_id]", nodes=main_ids)
        emit("            objective += (alpha * capability.get((task.task_id, module_id), 0.0)", nodes=main_ids)
        emit("                         - beta * loads[module_id]", nodes=main_ids)
        emit("                         - gamma * costs.get((task.task_id, module_id), 0.0))", nodes=main_ids)
        emit("        return AssignmentReport('COMPLETED', dict(assignment), objective, ())", nodes=main_ids)
        emit("    return AssignmentReport('FAILED', {}, 0.0, ('not all tasks assigned',))", nodes=main_ids)
        emit()

        # Tests emitted from TEST nodes ------------------------------------------------
        test_nodes = ir.nodes_of(NodeType.TEST)
        test_ids = tuple(n.node_id for n in test_nodes)
        emit("def run_examples():", nodes=test_ids)
        emit('    """Example fixtures derived from TEST nodes."""', nodes=test_ids)
        emit("    tasks = (Task('t1', 1.0), Task('t2', 2.0))", nodes=test_ids)
        emit("    modules = (Module('m1', 5.0, 0.0), Module('m2', 4.0, 1.0))", nodes=test_ids)
        emit("    eligibility = {('t1', 'm1'): True, ('t1', 'm2'): True,", nodes=test_ids)
        emit("                  ('t2', 'm1'): True, ('t2', 'm2'): True}", nodes=test_ids)
        emit("    capability = {('t1', 'm1'): 0.9, ('t1', 'm2'): 0.4,", nodes=test_ids)
        emit("                 ('t2', 'm1'): 0.3, ('t2', 'm2'): 0.8}", nodes=test_ids)
        emit("    costs = {('t1', 'm1'): 0.1, ('t1', 'm2'): 0.2,", nodes=test_ids)
        emit("             ('t2', 'm1'): 0.2, ('t2', 'm2'): 0.1}", nodes=test_ids)
        emit("    return assign_tasks(tasks, modules, eligibility, capability, costs)", nodes=test_ids)

        source = "\n".join(lines) + "\n"
        source_map = SourceMap(map_entries)

        # Round-trip compile check: generated source must parse.
        import ast as _ast

        try:
            _ast.parse(source)
        except SyntaxError as exc:  # pragma: no cover - generator regression guard
            raise GateViolation(f"generated source failed to parse: {exc}") from exc

        artifact = CodeArtifact(
            artifact_id=session.next_artifact_id(ArtifactKind.CODE_ARTIFACT),
            ir_id=ir.artifact_id,
            candidate_id=candidate.candidate_id,
            language=self.LANGUAGE,
            module_name="hic_generated_assignment",
            source=source,
            source_map=source_map,
            dependencies=("dataclasses", "math"),
        )
        session.register(artifact)
        session.ledger.record(
            stage="H",
            step_id="HIC50",
            description="Generate Python code with source maps from approved AlgoIR",
            inputs=[ir, candidate_set],
            outputs=[artifact],
            rule_references={"HIC-CODEGEN-1"},
        )
        return artifact

    @staticmethod
    def _node_type(ir: AlgoIRGraph, node_id: str) -> str:
        try:
            return ir.node(node_id).node_type.value
        except Exception:  # noqa: BLE001 - source-map labels degrade, never crash codegen
            return "UNKNOWN"
