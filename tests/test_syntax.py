from ore_syntax import (
    OPERATION_REGISTRY, STAGE_REGISTRY, SyntaxConfig, SyntaxProcessor, SyntaxRequest,
)


def processor(**kwargs):
    return SyntaxProcessor(config=SyntaxConfig(simulate_biology=False, **kwargs))


def test_registry_has_nine_stages_and_58_ordered_operations():
    assert len(STAGE_REGISTRY) == 9
    assert [stage.operation_count for stage in STAGE_REGISTRY] == [7, 6, 7, 6, 5, 6, 6, 7, 8]
    assert len(OPERATION_REGISTRY) == 58
    assert [op.operation_id for op in OPERATION_REGISTRY] == [
        f"SYN-{index:03d}" for index in range(1, 59)
    ]
    assert all(op.functional_responsibility and op.mathematical_form and op.biological_reference
               for op in OPERATION_REGISTRY)


def test_active_sentence_runs_all_stages_and_assigns_arguments():
    result = processor().process(SyntaxRequest(
        sequence_id="active-01", text="The cat eats fish.", language="en"
    ))
    assert result.diagnostics["valid"] is True
    assert result.diagnostics["operation_count"] == 58
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



def test_dependency_memory_release_and_nested_competition_preserve_open_alternatives():
    """Release only resolved records and group nested/competing dependencies explicitly."""
    worker = processor()
    links = [
        {
            "filler_index": 1, "head_index": 8, "status": "linked", "score": 0.94,
            "candidate_sites": [{"predicate_index": 8, "score": 0.94},
                                {"predicate_index": 10, "score": 0.63}],
        },
        {
            "filler_index": 3, "head_index": 6, "status": "linked", "score": 0.72,
            "candidate_sites": [{"predicate_index": 6, "score": 0.72}],
        },
        {
            "filler_index": 3, "head_index": None, "status": "unresolved", "score": 0.20,
            "candidate_sites": [{"predicate_index": 8, "score": 0.20}],
        },
    ]
    release = worker._release_resolved_dependency_memory(links)
    nested = worker._manage_nested_competing_dependencies(links)

    assert release["released_dependency_count"] == 1
    assert release["retained_dependency_count"] == 2
    assert links[0]["memory_status"] == "released_after_resolution"
    assert links[2]["memory_status"] == "retained_for_retrieval"
    assert nested["nested_pair_count"] >= 1
    assert nested["competing_pair_count"] >= 1
    assert links[0]["alternatives_preserved"] is True
    assert "competition_probability" in links[0]["candidate_sites"][0]


def test_full_parser_executes_two_additional_source_gap_operations():
    """Expose source steps 9.9 and 9.10 as actual registry-backed runtime operations."""
    result = processor().process(SyntaxRequest(
        sequence_id="source-gap-9-9-9-10",
        text="What did the scientist say the student bought?",
        language="en",
    ))
    assert [row["operation_id"] for row in result.operation_trace][-2:] == [
        "SYN-057", "SYN-058"
    ]
    assert result.operation_trace[-2]["operation_name"] == "Release resolved long-distance dependency memory"
    assert result.operation_trace[-1]["operation_name"] == "Manage nested and competing long-distance dependencies"
def test_lexical_access_evidence_can_override_default_category_inventory():
    result = processor().process(SyntaxRequest(
        sequence_id="lexical-bridge-01",
        tokens=("The", "quokka", "runs", "."),
        lexical_evidence={1: {"NOUN": 0.99, "VERB": 0.01}},
        language="en",
    ))
    assert result.selected_pos[1]["tag"] == "NOUN"
    assert result.operation_trace[0]["operation_id"] == "SYN-001"
    assert result.operation_trace[-1]["operation_id"] == "SYN-058"


def test_lif_reference_track_is_optional_and_reports_physical_units():
    enabled = SyntaxProcessor(config=SyntaxConfig(simulate_biology=True, steps_per_operation=1)).process(
        SyntaxRequest(sequence_id="lif-01", text="The dog chased the cat.", language="en")
    )
    disabled = processor().process(
        SyntaxRequest(sequence_id="lif-02", text="The dog chased the cat.", language="en")
    )
    assert enabled.biological["model"] == "current_based_LIF_syntax_reference"
    assert enabled.biological["units"]["voltage"] == "mV"
    assert enabled.biological["operation_count"] == 58
    assert disabled.biological["enabled"] is False
