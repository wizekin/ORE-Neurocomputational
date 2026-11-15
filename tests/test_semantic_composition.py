"""Software-contract tests for ORE sentence-level semantic composition."""
from __future__ import annotations

from ore_semantic import SemanticConcept
from ore_semantic_composition import (
    OPERATION_REGISTRY,
    STAGE_REGISTRY,
    PredicateFrame,
    SemanticCompositionConfig,
    SemanticCompositionProcessor,
    SemanticCompositionRequest,
)


def _concepts() -> list[SemanticConcept]:
    """Create a tiny provenance-labelled inventory for deterministic synthetic tests."""
    return [
        SemanticConcept(
            concept_id="scientist", label="scientist",
            semantic_features={"HUMAN": 1.0, "AGENTIVE": 0.9},
            lexical_category="noun", frequency=15.0,
            verified_by="test", source_reference="test://scientist",
        ),
        SemanticConcept(
            concept_id="study", label="studies", aliases=("study",),
            semantic_features={"EVENT": 0.8, "COGNITION": 1.0},
            lexical_category="verb", frequency=12.0,
            verified_by="test", source_reference="test://study",
        ),
        SemanticConcept(
            concept_id="neuron", label="neurons", aliases=("neuron",),
            semantic_features={"BIOLOGICAL_ENTITY": 1.0, "CELL": 0.9},
            lexical_category="noun", frequency=18.0,
            verified_by="test", source_reference="test://neuron",
        ),
        SemanticConcept(
            concept_id="fire", label="fire",
            semantic_features={"EVENT": 0.8, "ACTIVITY": 0.9},
            lexical_category="verb", frequency=8.0,
            verified_by="test", source_reference="test://fire",
        ),
        SemanticConcept(
            concept_id="rain", label="rain",
            semantic_features={"WEATHER_EVENT": 1.0},
            lexical_category="noun", frequency=9.0,
            verified_by="test", source_reference="test://rain",
        ),
        SemanticConcept(
            concept_id="fell", label="fell",
            semantic_features={"MOTION_EVENT": 0.8},
            lexical_category="verb", frequency=8.0,
            verified_by="test", source_reference="test://fell",
        ),
        SemanticConcept(
            concept_id="flood", label="floods",
            semantic_features={"WATER_EVENT": 0.9},
            lexical_category="noun", frequency=5.0,
            verified_by="test", source_reference="test://flood",
        ),
        SemanticConcept(
            concept_id="form", label="formed",
            semantic_features={"CHANGE_EVENT": 0.8},
            lexical_category="verb", frequency=5.0,
            verified_by="test", source_reference="test://form",
        ),
    ]


def _processor(simulate_biology: bool = False) -> SemanticCompositionProcessor:
    """Create a processor with one explicit verb frame for the main example."""
    frame = PredicateFrame(
        predicate="studies",
        predicate_concept_id="study",
        roles=("Agent", "Patient"),
        required_roles=("Agent", "Patient"),
        selectional_preferences={
            "Agent": {"HUMAN": 1.0, "AGENTIVE": 0.8},
            "Patient": {"BIOLOGICAL_ENTITY": 1.0},
        },
        source_reference="test://frames/studies",
    )
    return SemanticCompositionProcessor(
        _concepts(),
        (frame,),
        SemanticCompositionConfig(simulate_biology=simulate_biology),
    )


def test_registry_contains_nine_stages_and_fifty_four_ordered_operations() -> None:
    """Ensure the public registry has six stable operation IDs for each requested stage."""
    assert len(STAGE_REGISTRY) == 9
    assert len(OPERATION_REGISTRY) == 54
    assert [item.operation_id for item in OPERATION_REGISTRY] == [
        f"SMC-{index:03d}" for index in range(1, 55)
    ]
    assert all(len(stage.operation_ids) == 6 for stage in STAGE_REGISTRY)


def test_predicate_arguments_roles_and_proposition_composition() -> None:
    """Check that syntax links and semantic preferences produce Agent/Patient bindings."""
    processor = _processor()
    syntax = {
        "tokens": ["scientist", "studies", "neurons"],
        "selected_pos": [
            {"token_index": 0, "tag": "NOUN"},
            {"token_index": 1, "tag": "VERB"},
            {"token_index": 2, "tag": "NOUN"},
        ],
        "dependency_arcs": [
            {"dependent_index": 0, "head_index": 1, "relation": "nsubj"},
            {"dependent_index": 1, "head_index": None, "relation": "root"},
            {"dependent_index": 2, "head_index": 1, "relation": "obj"},
        ],
    }
    result = processor.process(SemanticCompositionRequest(
        sequence_id="test-roles",
        tokens=("scientist", "studies", "neurons"),
        syntax_result=syntax,
    ))
    assert len(result.operation_trace) == 54
    assert result.diagnostics["valid"]
    assert len(result.propositions) == 1
    assigned = {row["argument"]: row["selected_role"] for row in result.thematic_roles}
    assert assigned == {"scientist": "Agent", "neurons": "Patient"}
    assert result.propositions[0]["well_formed"]
    assert result.sentence_representation["representation_type"] == "typed_sentence_semantic_graph"


def test_negation_modality_and_quantifier_scope_are_preserved() -> None:
    """Check that not/may/every are represented as explicit scoped semantic operators."""
    processor = _processor()
    syntax = {
        "tokens": ["every", "neuron", "may", "fire"],
        "selected_pos": [
            {"token_index": 0, "tag": "DET"},
            {"token_index": 1, "tag": "NOUN"},
            {"token_index": 2, "tag": "AUX"},
            {"token_index": 3, "tag": "VERB"},
        ],
        "dependency_arcs": [
            {"dependent_index": 0, "head_index": 1, "relation": "det"},
            {"dependent_index": 1, "head_index": 3, "relation": "nsubj"},
            {"dependent_index": 2, "head_index": 3, "relation": "aux"},
            {"dependent_index": 3, "head_index": None, "relation": "root"},
        ],
    }
    result = processor.process(SemanticCompositionRequest(
        sequence_id="test-operators",
        tokens=("every", "neuron", "may", "fire"),
        syntax_result=syntax,
    ))
    assert result.quantifier_scopes[0]["symbol"] == "∀"
    assert result.quantifier_scopes[0]["domain_head"] == "neuron"
    assert result.propositions[0]["modal_operators"][0]["token"] == "may"

    negative = processor.process(SemanticCompositionRequest(
        sequence_id="test-negation",
        tokens=("neuron", "does", "not", "fire"),
        syntax_result={
            "tokens": ["neuron", "does", "not", "fire"],
            "selected_pos": [
                {"token_index": 0, "tag": "NOUN"},
                {"token_index": 1, "tag": "AUX"},
                {"token_index": 2, "tag": "PART"},
                {"token_index": 3, "tag": "VERB"},
            ],
            "dependency_arcs": [
                {"dependent_index": 0, "head_index": 3, "relation": "nsubj"},
                {"dependent_index": 1, "head_index": 3, "relation": "aux"},
                {"dependent_index": 3, "head_index": None, "relation": "root"},
            ],
        },
    ))
    assert negative.propositions[0]["polarity"] == "negative"


def test_explicit_because_marker_creates_directed_causal_relation() -> None:
    """Check that an explicit causal connective creates a cause-to-effect relation."""
    result = _processor().process(SemanticCompositionRequest(
        sequence_id="test-causality",
        tokens=("rain", "fell", "because", "floods", "formed"),
    ))
    assert result.causal_relations
    relation = result.causal_relations[0]
    assert relation["relation"] == "CAUSES"
    assert relation["cue"] == "because"
    assert relation["cause_proposition_id"] == "prop:4"
    assert relation["effect_proposition_id"] == "prop:1"


def test_biological_reference_is_separate_and_reports_physical_units() -> None:
    """Check that the optional nine-population LIF reference returns explicit units and traces."""
    result = _processor(simulate_biology=True).process(SemanticCompositionRequest(
        sequence_id="test-biology",
        tokens=("scientist", "studies", "neurons"),
    ))
    assert len(result.biological["layers"]) == 9
    assert len(result.biological["operation_states"]) == 54
    assert result.biological["units"]["membrane_voltage"] == "mV"
    assert result.biological["units"]["synaptic_current"] == "pA"
    assert result.biological["units"]["firing_rate"] == "Hz"
    assert result.biological["layers"]["concept_retrieval"]["spike_count"] > 0
