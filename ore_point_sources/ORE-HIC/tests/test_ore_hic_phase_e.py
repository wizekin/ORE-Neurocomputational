"""Phase E tests: AlgoIR schema, invariants, validation, explanation (Steps 34-39)."""
from __future__ import annotations

import pytest
from test_ore_hic_helpers import ASSIGNMENT_IDEA

from ore_hic import (
    AlgoIRBuilder,
    AlgoIRGraph,
    AlgoIRNode,
    AlgoIRValidator,
    AmbiguityAnalyzer,
    CompilationSession,
    NodeType,
    Normalizer,
    RequirementsFormalizer,
    SemanticInterpreter,
    SourceSpan,
    TextInputAdapter,
    ValidationState,
)
from ore_hic.contracts import SchemaError
from ore_hic.segmentation import IdeaSegmenter


@pytest.fixture(scope="module")
def built_ir():
    session = CompilationSession(user="u")
    record = TextInputAdapter().capture(session, ASSIGNMENT_IDEA, source_identity="u")
    view = Normalizer().normalize(record)
    segmentation = IdeaSegmenter().run(session, record, view)
    graph = SemanticInterpreter().interpret(session, record, segmentation)
    ambiguity = AmbiguityAnalyzer().analyze(session, graph, resolved={"amb-001"}, confirmed={"as-model-linearity"})
    spec = RequirementsFormalizer().formalize(session, graph, segmentation, ambiguity)
    from ore_hic import MathFormalizer

    model = MathFormalizer().formalize(session, graph, spec)
    builder = AlgoIRBuilder()
    ir = builder.build(session, graph, spec, model)
    return session, ir


def test_step34_all_required_node_types_exist(built_ir):
    _, ir = built_ir
    present = {n.node_type for n in ir.nodes}
    required = set(NodeType)
    assert required <= present, f"missing node types: {required - present}"


def test_step34_nodes_carry_required_metadata(built_ir):
    _, ir = built_ir
    for node in ir.nodes:
        assert node.node_id.startswith("n")
        assert node.scope
        assert 0.0 <= node.confidence <= 1.0
        assert node.creation_stage
        assert node.validation_state in set(ValidationState)
        # Provenance or explicit derived-construction label (spec 4.3).
        assert node.source_spans or node.derived_construction
        assert node.payload_digest


def test_schema_rejects_unknown_node_types(built_ir):
    _, ir = built_ir
    span = SourceSpan("in-1", 0, 3, "abc")
    with pytest.raises(SchemaError):
        ir.add_node(
            AlgoIRNode(
                node_id="n9999", node_type="NOT_A_TYPE", name="x", scope="global",
                description="", content={}, confidence=0.5, source_spans=(span,),
                parents=(), children=(), dependencies=(), assumptions=(),
                usg_references=(), creation_stage="E",
                validation_state=ValidationState.UNVALIDATED, derived_construction=True,
            )
        )


def test_schema_rejects_missing_provenance(built_ir):
    _, ir = built_ir
    with pytest.raises(SchemaError):
        ir.add_node(
            AlgoIRNode(
                node_id="n9998", node_type=NodeType.ENTITY, name="orphan", scope="global",
                description="", content={}, confidence=0.5, source_spans=(),
                parents=(), children=(), dependencies=(), assumptions=(),
                usg_references=(), creation_stage="E",
                validation_state=ValidationState.UNVALIDATED, derived_construction=False,
            )
        )


def test_step38_wellformed_graph_validates_cleanly(built_ir):
    _, ir = built_ir
    violations = AlgoIRValidator().validate(ir)
    assert violations == (), f"expected clean IR, got {[v.code for v in violations]}"


def test_step38_undefined_variables_are_detected():
    graph = AlgoIRGraph(artifact_id="ir-test", model_id="m")
    span = SourceSpan("in-1", 0, 3, "abc")

    def add(node_id: str, node_type: NodeType, name: str, content: dict, dependencies=()):
        graph.add_node(
            AlgoIRNode(
                node_id=node_id, node_type=node_type, name=name, scope="global",
                description="", content=content, confidence=0.9,
                source_spans=(span,), parents=(), children=(), dependencies=dependencies,
                assumptions=(), usg_references=(), creation_stage="E",
                validation_state=ValidationState.UNVALIDATED, derived_construction=True,
            )
        )

    add("n1", NodeType.GOAL, "goal", {}, dependencies=("n2",))
    add("n2", NodeType.OPERATION, "op", {"inputs": ["undefined_symbol"], "outputs": []})
    violations = AlgoIRValidator().validate(graph)
    codes = {v.code for v in violations}
    assert "UNDEFINED_SYMBOL" in codes


def test_step38_dependency_cycles_outside_iterative_constructs_detected():
    graph = AlgoIRGraph(artifact_id="ir-cycle", model_id="m")
    span = SourceSpan("in-1", 0, 3, "abc")

    def add(node_id, node_type, name, content, dependencies=()):
        graph.add_node(
            AlgoIRNode(
                node_id=node_id, node_type=node_type, name=name, scope="global",
                description="", content=content, confidence=0.9, source_spans=(span,),
                parents=(), children=(), dependencies=dependencies, assumptions=(),
                usg_references=(), creation_stage="E",
                validation_state=ValidationState.UNVALIDATED, derived_construction=True,
            )
        )

    add("n1", NodeType.GOAL, "goal", {}, dependencies=("n2",))
    add("n2", NodeType.OPERATION, "op_a", {}, dependencies=("n3",))
    add("n3", NodeType.OPERATION, "op_b", {}, dependencies=("n2",))  # cycle
    violations = AlgoIRValidator().validate(graph)
    assert any(v.code == "DEPENDENCY_CYCLE" for v in violations)


def test_step38_loops_need_termination_arguments():
    graph = AlgoIRGraph(artifact_id="ir-loop", model_id="m")
    span = SourceSpan("in-1", 0, 3, "abc")
    graph.add_node(
        AlgoIRNode(
            node_id="n1", node_type=NodeType.LOOP, name="loop", scope="global",
            description="", content={"termination_policy": "NOT_DECLARED"}, confidence=0.9,
            source_spans=(span,), parents=(), children=(), dependencies=(), assumptions=(),
            usg_references=(), creation_stage="E",
            validation_state=ValidationState.UNVALIDATED, derived_construction=True,
        )
    )
    violations = AlgoIRValidator().validate(graph)
    assert any(v.code == "LOOP_TERMINATION" for v in violations)


def test_step38_branches_need_predicates_and_paths():
    graph = AlgoIRGraph(artifact_id="ir-cond", model_id="m")
    span = SourceSpan("in-1", 0, 3, "abc")
    graph.add_node(
        AlgoIRNode(
            node_id="n1", node_type=NodeType.CONDITION, name="cond", scope="global",
            description="", content={"predicate": "", "variables": [], "paths": {"then": "x"}},
            confidence=0.9, source_spans=(span,), parents=(), children=(), dependencies=(),
            assumptions=(), usg_references=(), creation_stage="E",
            validation_state=ValidationState.UNVALIDATED, derived_construction=True,
        )
    )
    violations = AlgoIRValidator().validate(graph)
    codes = {v.code for v in violations}
    assert "MALFORMED_PREDICATE" in codes or "INCOMPLETE_BRANCH" in codes


def test_step38_constraints_need_known_scopes():
    graph = AlgoIRGraph(artifact_id="ir-scope", model_id="m")
    span = SourceSpan("in-1", 0, 3, "abc")
    graph.add_node(
        AlgoIRNode(
            node_id="n1", node_type=NodeType.CONSTRAINT, name="con", scope="nonexistent-scope",
            description="", content={"expression": "x", "hard": True}, confidence=0.9,
            source_spans=(span,), parents=(), children=(), dependencies=(), assumptions=(),
            usg_references=(), creation_stage="E",
            validation_state=ValidationState.UNVALIDATED, derived_construction=True,
        )
    )
    violations = AlgoIRValidator().validate(graph)
    assert any(v.code == "CONSTRAINT_SCOPE" for v in violations)


def test_serialization_round_trip_preserves_semantics(built_ir):
    _, ir = built_ir
    from ore_hic.contracts import canonical_json

    payload = ir.to_dict()
    restored = AlgoIRGraph.from_dict(payload)
    assert len(restored) == len(ir)
    assert restored.content_payload() == ir.content_payload()
    assert restored.content_digest == ir.content_digest
    # Canonical JSON is stable across a dump/load cycle.
    assert canonical_json(restored.to_dict()) == canonical_json(payload)


def test_step39_explanation_is_derived_from_canonical_ir(built_ir):
    session, ir = built_ir
    builder = AlgoIRBuilder()
    graph, violations, explanation = builder.validate_and_explain(session, ir)
    assert violations == ()
    assert explanation.startswith("# AlgoIR explanation")
    # Every node type appears in the explanation, derived from the IR itself.
    for node_type in NodeType:
        assert f"## {node_type.value}" in explanation
    # Validation states are updated on the graph.
    assert all(n.validation_state is ValidationState.VALID for n in graph.nodes)
