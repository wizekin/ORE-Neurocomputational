"""Phase D tests: mathematical formalization (Steps 27-33, spec Section 3)."""
from __future__ import annotations

import pytest
from test_ore_hic_helpers import ASSIGNMENT_IDEA

from ore_hic import (
    AmbiguityAnalyzer,
    AssignmentProblem,
    CompilationSession,
    MathFormalizer,
    MathStructureKind,
    ModuleSpec,
    Normalizer,
    RequirementsFormalizer,
    SemanticInterpreter,
    TaskSpec,
    TextInputAdapter,
    VariableRole,
    assignment_objective_value,
)
from ore_hic.segmentation import IdeaSegmenter


@pytest.fixture(scope="module")
def model():
    session = CompilationSession(user="u")
    record = TextInputAdapter().capture(session, ASSIGNMENT_IDEA, source_identity="u")
    view = Normalizer().normalize(record)
    segmentation = IdeaSegmenter().run(session, record, view)
    graph = SemanticInterpreter().interpret(session, record, segmentation)
    ambiguity = AmbiguityAnalyzer().analyze(session, graph, resolved={"amb-001"}, confirmed={"as-model-linearity"})
    spec = RequirementsFormalizer().formalize(session, graph, segmentation, ambiguity)
    return MathFormalizer().formalize(session, graph, spec)


def test_step27_structure_is_identified(model):
    assert MathStructureKind.ASSIGNMENT in model.structures
    assert MathStructureKind.OPTIMIZATION in model.structures


def test_step28_variables_have_domains_units_roles(model):
    by_symbol = {v.symbol: v for v in model.variables}
    x = by_symbol["x_ij"]
    assert x.role is VariableRole.DECISION
    assert "{0, 1}" in x.domain
    k = by_symbol["K_j"]
    assert k.unit is not None  # capacity unit declared
    assert k.origin_requirement  # provenance to the originating requirement
    roles = {v.role for v in model.variables}
    assert VariableRole.OBSERVED in roles and VariableRole.PARAMETER in roles


def test_step29_equations_map_to_requirements(model):
    by_id = {e.equation_id: e for e in model.equations}
    assert "sum_j x_ij = 1" in by_id["eq-001"].expression
    assert by_id["eq-001"].origin_requirement == "req-exactly-one"
    assert by_id["eq-002"].origin_requirement == "req-capacity"
    assert by_id["eq-003"].origin_requirement == "req-eligibility"
    for equation in model.equations:
        assert equation.derivation, "every formula keeps its derivation"


def test_step30_objective_matches_spec_section_33(model):
    objective = model.objectives[0]
    assert objective.direction == "MAXIMIZE"
    assert "alpha c_ij" in objective.expression
    assert "beta l_j" in objective.expression
    assert "gamma r_ij" in objective.expression
    assert len(objective.terms) == 3
    assert objective.unit_note  # scale compatibility considered


def test_step31_assumptions_are_validated(model):
    unsupported = model.invalid_assumptions
    # The built-in model's assumptions are all supported by the spec text.
    assert unsupported == ()
    statements = " ".join(a.statement for a in model.assumptions)
    assert "additive" in statements, "the 3.3 additivity assumption must be stored"
    for assumption in model.assumptions:
        assert assumption.consequence_if_false, "consequences must be recorded"


def test_step32_tools_fit_the_structure(model):
    fitting = [t for t in model.tools if t.fits_structure]
    kinds = {t.tool.value for t in fitting}
    assert "LINEAR_INTEGER_PROGRAMMING" in kinds
    assert "GRAPH_ALGORITHM" in kinds
    not_fitting = [t for t in model.tools if not t.fits_structure]
    assert not_fitting, "misfitting tools must be explicitly rejected, not forced"


def test_step33_complexity_includes_assumptions_and_confidence(model):
    functions = {c.function for c in model.complexity}
    assert {"T_A(n)", "S_A(n)", "R_A(n)"} <= functions
    for estimate in model.complexity:
        assert estimate.input_size_assumption
        assert 0.0 <= estimate.confidence <= 1.0
        assert estimate.case in {"WORST", "AVERAGE", "EXPECTED"}


def test_objective_evaluation_is_deterministic():
    problem_kwargs = {
        "capability": {("t1", "m1"): 0.9, ("t1", "m2"): 0.2},
        "loads": {"m1": 1.0, "m2": 0.0},
        "costs": {("t1", "m1"): 0.1, ("t1", "m2"): 0.05},
    }
    value = assignment_objective_value({"t1": "m1"}, **problem_kwargs)
    assert value == pytest.approx(0.9 - 1.0 - 0.1)
    value_m2 = assignment_objective_value({"t1": "m2"}, **problem_kwargs)
    assert value_m2 == pytest.approx(0.2 - 0.0 - 0.05)


def test_unrecognized_structure_is_honest():
    session = CompilationSession(user="u")
    record = TextInputAdapter().capture(session, "Explain the history of poetry.", source_identity="u")
    view = Normalizer().normalize(record)
    segmentation = IdeaSegmenter().run(session, record, view)
    graph = SemanticInterpreter().interpret(session, record, segmentation)
    ambiguity = AmbiguityAnalyzer().analyze(session, graph, resolved={"amb-001"}, confirmed={"as-model-linearity"})
    spec = RequirementsFormalizer().formalize(session, graph, segmentation, ambiguity)
    model = MathFormalizer().formalize(session, graph, spec)
    if model.model_family == "UNRECOGNIZED":
        # Never invent a complexity bound when the structure is unknown.
        assert model.complexity[0].expression == "unknown"
        assert model.complexity[0].confidence == 0.0
        assert all(not t.fits_structure for t in model.tools)


def test_assignment_problem_validates_inputs():
    with pytest.raises(ValueError):
        AssignmentProblem(
            tasks=(TaskSpec("t1", -1.0),),
            modules=(ModuleSpec("m1", 5.0, 0.0),),
            eligibility={}, capability={}, costs={},
        )
    with pytest.raises(ValueError):
        AssignmentProblem(
            tasks=(), modules=(ModuleSpec("m1", 5.0, 0.0),),
            eligibility={}, capability={}, costs={},
        )
