from ore_syntax import (
    OPERATION_REGISTRY, STAGE_REGISTRY, SyntaxConfig, SyntaxProcessor, SyntaxRequest,
)


def processor(**kwargs):
    return SyntaxProcessor(config=SyntaxConfig(simulate_biology=False, **kwargs))


def test_registry_has_nine_stages_and_56_ordered_operations():
    assert len(STAGE_REGISTRY) == 9
    assert [stage.operation_count for stage in STAGE_REGISTRY] == [7, 6, 7, 6, 5, 6, 6, 7, 6]
    assert len(OPERATION_REGISTRY) == 56
    assert [op.operation_id for op in OPERATION_REGISTRY] == [
        f"SYN-{index:03d}" for index in range(1, 57)
    ]
    assert all(op.functional_responsibility and op.mathematical_form and op.biological_reference
               for op in OPERATION_REGISTRY)


def test_active_sentence_runs_all_stages_and_assigns_arguments():
    result = processor().process(SyntaxRequest(
        sequence_id="active-01", text="The cat eats fish.", language="en"
    ))
    assert result.diagnostics["valid"] is True
    assert result.diagnostics["operation_count"] == 56
    assert result.diagnostics["operation_order_valid"] is True
    assert result.diagnostics["dependency_tree_valid"] is True
    assert result.constituency_tree["label"] == "S"
    assert [item["tag"] for item in result.selected_pos][2] == "VERB"
    assert any(item["type"] == "NP" for item in result.phrases)
    eats = next(frame for frame in result.predicate_arguments if frame["predicate"].casefold() == "eats")
    assert eats["subjects"][0]["token"].casefold() == "cat"
    assert eats["objects"][0]["token"].casefold() == "fish"
    assert any(role["role"] == "agent" for role in result.thematic_roles)
    assert any(role["role"] == "patient" for role in result.thematic_roles)


def test_passive_voice_maps_subject_to_patient_and_by_phrase_to_agent():
    result = processor().process(SyntaxRequest(
        sequence_id="passive-01", text="The cake was eaten by John.", language="en",
        semantic_features={1: {"FOOD": 1.0}, 5: {"ANIMATE": 1.0, "HUMAN": 1.0}},
    ))
    frame = next(frame for frame in result.predicate_arguments if frame["predicate"].casefold() == "eaten")
    assert frame["voice"] == "passive"
    assert frame["subjects"][0]["token"].casefold() == "cake"
    assert frame["subjects"][0]["provisional_role"] == "patient"
    assert frame["oblique_agents"][0]["token"].casefold() == "john"
    role_by_token = {item["argument"].casefold(): item["role"] for item in result.thematic_roles}
    assert role_by_token["cake"] == "patient"
    assert role_by_token["john"] == "agent"


def test_wh_filler_creates_enhanced_remote_dependency_without_changing_tree():
    result = processor().process(SyntaxRequest(
        sequence_id="wh-01", text="What did John say Mary bought?", language="en"
    ))
    links = result.long_distance_dependencies
    assert links
    assert links[0]["filler"].casefold() == "what"
    assert links[0]["status"] == "linked"
    assert links[0]["relation"] == "obj"
    assert links[0]["head"].casefold() == "bought"
    assert links[0]["basic_dependency_tree_modified"] is False


def test_lexical_access_evidence_can_override_default_category_inventory():
    result = processor().process(SyntaxRequest(
        sequence_id="lexical-bridge-01",
        tokens=("The", "quokka", "runs", "."),
        lexical_evidence={1: {"NOUN": 0.99, "VERB": 0.01}},
        language="en",
    ))
    assert result.selected_pos[1]["tag"] == "NOUN"
    assert result.operation_trace[0]["operation_id"] == "SYN-001"
    assert result.operation_trace[-1]["operation_id"] == "SYN-056"


def test_lif_reference_track_is_optional_and_reports_physical_units():
    enabled = SyntaxProcessor(config=SyntaxConfig(simulate_biology=True, steps_per_operation=1)).process(
        SyntaxRequest(sequence_id="lif-01", text="The dog chased the cat.", language="en")
    )
    disabled = processor().process(
        SyntaxRequest(sequence_id="lif-02", text="The dog chased the cat.", language="en")
    )
    assert enabled.biological["model"] == "current_based_LIF_syntax_reference"
    assert enabled.biological["units"]["voltage"] == "mV"
    assert enabled.biological["operation_count"] == 56
    assert disabled.biological["enabled"] is False
