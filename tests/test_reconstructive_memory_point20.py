"""Focused acceptance tests for Point 20 schema-driven inference and reconstructive memory."""
from __future__ import annotations

import sqlite3

import pytest

from ore_reconstructive_memory import (
    DetailCandidate, EpisodicFragment, EpisodicRelation, FeedbackRecord, Point20Config,
    Point20Request, Point20State, ReconstructiveMemoryProcessor,
    SchemaProfile, SourceRecord, OPERATION_REGISTRY,
    activate_candidate_schemas, assess_familiarity_and_recollection, brier_score,
    build_feature_ledger,
    build_provenance_graph, calibrate_detail_confidence, choose_response_policy,
    generate_inferred_completions,
    generate_inferred_completions, reactivate_episodic_fragments,
    search_competing_episodes, update_general_schema_knowledge,
    update_reconstruction_from_feedback,
)


def make_schema(schema_id: str = "travel", prior: float = 0.7) -> SchemaProfile:
    """Create a small illustrative schema with several plausible event completions."""
    return SchemaProfile(
        schema_id=schema_id,
        name="water journey script",
        prior_probability=prior,
        feature_probabilities={
            "object_type": {"canoe": 0.7, "boat": 0.3},
            "cause": {"storm": 0.6, "collision": 0.4},
        },
        context_profile={"setting": 0.9},
        goal_relevance=0.8,
        cultural_scope="generic-example",
    )


def make_request(**overrides) -> Point20Request:
    """Build an incomplete story recollection with one source-labelled episodic fragment."""
    values = {
        "event_id": "story-001",
        "cue": {"setting": "river"},
        "target_features": ["setting", "object_type", "cause"],
        "episodic_fragments": [
            EpisodicFragment(
                detail_id="detail-setting",
                feature_type="setting",
                content="riverbank",
                status="episodically_retrieved",
                retrieval_strength=0.92,
                episodic_context=0.90,
                attention_encoding=0.90,
                centrality=0.85,
                familiarity=0.25,
                recollection=0.90,
                source_type="episodic_retrieval",
                source_id="source-original-1",
                episode_id="story-001",
                cue_features={"setting": "river"},
                source_features={"spatial": 0.9, "temporal": 0.5},
                timestamp=1.0,
            )
        ],
        "schemas": [make_schema()],
        "context": {
            "setting": "river",
            "context_support": 0.7,
            "gist": "People travelled across water before an unusual event interrupted the journey.",
            "constraints": [
                {"constraint_id": "setting-check", "feature_type": "setting",
                 "allowed_values": ["riverbank"], "hard": True, "weight": 1.0}
            ],
        },
        "source_history": [
            SourceRecord(
                source_id="source-original-1",
                source_type="episodic_retrieval",
                detail_id="detail-setting",
                content="riverbank",
                reliability=0.9,
                timestamp=1.0,
                independence_group="original-story",
            )
        ],
        "competing_episodes": [],
        "feedback": [],
        "task_type": "story_retelling",
        "precision": "exact",
        "target_episode_id": "story-001",
    }
    values.update(overrides)
    return Point20Request(**values)


def test_registry_contains_all_sixteen_source_ordered_operations() -> None:
    """Keep Point 20's complete source sequence stable."""
    assert len(OPERATION_REGISTRY) == 16
    assert [item.operation_id for item in OPERATION_REGISTRY] == [
        f"P20-{index:02d}" for index in range(1, 17)
    ]


def test_unknown_is_not_converted_to_absent_or_contradicted() -> None:
    """A feature with no recalled fragment stays not-retrieved rather than false."""
    request = make_request()
    retrieval = reactivate_episodic_fragments(request, {"cue": request.cue, "cue_features": request.cue,
        "precision": "exact"}, Point20Config(simulate_biology=False))
    ledger = build_feature_ledger(request, retrieval)
    assert ledger["features"]["object_type"]["status"] == "not_retrieved"
    assert ledger["features"]["object_type"]["unavailable_not_absent"] is True
    assert ledger["features"]["cause"]["status"] == "not_retrieved"


def test_schema_completions_are_marked_as_inferred() -> None:
    """Schema proposals may fill candidate lists but cannot become episodic facts."""
    request = make_request()
    config = Point20Config(simulate_biology=False)
    retrieval = reactivate_episodic_fragments(
        request, {"cue": request.cue, "cue_features": dict(request.cue)}, config,
    )
    ledger = build_feature_ledger(request, retrieval)
    from ore_reconstructive_memory.functional import activate_candidate_schemas
    activation = activate_candidate_schemas(request, ledger, config)
    generated = generate_inferred_completions(request, ledger, activation, config)
    assert generated["candidate_count"] > 0
    assert all(item.status == "inferred" for item in generated["candidates"])
    assert all(item.provenance[-1] == "generated_as_inference" for item in generated["candidates"])


def test_schema_prior_can_rank_alternatives_but_not_create_evidence() -> None:
    """High schema typicality does not manufacture an event-specific observation."""
    result = ReconstructiveMemoryProcessor(
        Point20Config(simulate_biology=False)
    ).process(make_request())
    inferred = result.reconstruction["inferred_details"]
    assert inferred
    assert all(item["status"] == "inferred" for item in inferred)
    assert result.reconstruction["supported_details"][0]["content"] == "riverbank"
    assert all(item["content"] != "riverbank" for item in inferred)


def test_familiarity_and_recollection_are_separate_dimensions() -> None:
    """Repeated later mentions may increase familiarity without providing event recollection."""
    candidate = DetailCandidate(
        candidate_id="travel:object_type:canoe",
        feature_type="object_type",
        content="canoe",
        schema_id="travel",
        status="inferred",
        schema_prior=0.7,
    )
    sources = [
        SourceRecord(
            source_id=f"later-{index}",
            source_type="later_testimony",
            detail_id="",
            content="canoe",
            timestamp=float(index),
            independence_group=f"later-group-{index}",
            reliability=0.8,
        )
        for index in range(5)
    ]
    result = assess_familiarity_and_recollection([candidate], [], sources, "exact", Point20Config(simulate_biology=False))
    row = result["dimensions"][0]
    assert row["familiarity"] > 0.0
    assert row["source_specific_recollection"] == 0.0
    assert row["familiarity_not_proof"] is True


def test_competing_episode_overlap_is_flagged_not_assumed_as_target_fact() -> None:
    """A detail found only in another episode is treated as possible blending."""
    request = make_request(competing_episodes=[{
        "episode_id": "story-older",
        "fragments": [{"feature_type": "object_type", "content": "canoe"}],
    }])
    candidate = DetailCandidate(
        candidate_id="travel:object_type:canoe",
        feature_type="object_type",
        content="canoe",
        schema_id="travel",
    )
    result = search_competing_episodes([candidate], request, request.episodic_fragments)
    comparison = result["comparisons"][0]
    assert comparison["possible_episode_blend"] is True
    assert comparison["target_episode_support"] is False
    assert comparison["absence_of_target_match_is_not_a_contradiction"] is True


def test_confidence_is_not_derived_from_story_coherence() -> None:
    """A schema-only completion remains low-confidence without source-specific evidence."""
    inferred = DetailCandidate(
        candidate_id="schema:item",
        feature_type="object_type",
        content="canoe",
        status="inferred",
        schema_support=1.0,
        schema_prior=0.99,
        source_uncertainty=0.1,
        inference_proportion=1.0,
    )
    output = calibrate_detail_confidence([inferred], Point20Config(simulate_biology=False))
    assert output["per_detail"][0]["confidence"] < 0.50
    assert output["per_detail"][0]["coherence_used_as_confidence"] is False


def test_exact_policy_can_request_clarification_for_an_unknown_feature() -> None:
    """When exact detail is required, an unavailable central feature is not silently invented."""
    request = make_request()
    config = Point20Config(simulate_biology=False)
    retrieval = reactivate_episodic_fragments(
        request, {"cue": request.cue, "cue_features": dict(request.cue)}, config,
    )
    ledger = build_feature_ledger(request, retrieval)
    policy = choose_response_policy(request, ledger, [], config)
    object_decision = next(item for item in policy["decisions"] if item["feature_type"] == "object_type")
    assert object_decision["selected"]["action"] in {"clarify", "omit"}
    assert object_decision["selected"]["status"] == "unknown"


def test_provenance_graph_preserves_sources_and_rejects_cycles() -> None:
    """Source lineage remains inspectable and must form a directed acyclic graph."""
    request = make_request()
    graph = build_provenance_graph(request, [])
    assert graph["earliest_source_preserved"] is True
    assert any(node["node_type"] == "original_event" for node in graph["nodes"].values())
    cyclic = make_request(source_history=[
        SourceRecord("a", "later_testimony", parent_source_ids=("b",)),
        SourceRecord("b", "later_testimony", parent_source_ids=("a",)),
    ])
    with pytest.raises(ValueError, match="directed acyclic"):
        build_provenance_graph(cyclic, [])


def test_likelihood_ratio_feedback_updates_posterior_and_deduplicates_sources() -> None:
    """A reliable independent correction changes odds, while repeated dependent reports count once."""
    candidates = [
        DetailCandidate("c1", "object_type", "canoe", posterior=0.30, confidence=0.3),
        DetailCandidate("c2", "object_type", "boat", posterior=0.70, confidence=0.7),
    ]
    feedback = [
        FeedbackRecord("f1", "object_type", "canoe", True, "interview-1",
                       reliability=0.90, independence_group="witness-a", candidate_id="c1"),
        FeedbackRecord("f2", "object_type", "canoe", True, "interview-2",
                       reliability=0.90, independence_group="witness-a", candidate_id="c1"),
    ]
    state = Point20State()
    result = update_reconstruction_from_feedback(
        candidates, feedback, state, Point20Config(simulate_biology=False),
    )
    assert len(result["updates"]) == 1
    assert result["ignored_feedback"][0]["reason"] == "duplicate_independence_group"
    assert candidates[0].feedback_probability > 0.30
    assert candidates[0].posterior == pytest.approx(0.30)


def test_schema_learning_requires_verified_independent_feedback() -> None:
    """Only authorized verified examples change the general schema beta counts."""
    request = make_request(feedback=[
        FeedbackRecord(
            "verified-1", "object_type", "canoe", True, "archival-record",
            reliability=0.98, independence_group="archive-independent",
            schema_id="travel", verified=True, use_for_schema_update=True,
        )
    ])
    state = Point20State()
    result = update_general_schema_knowledge(
        request, state, Point20Config(simulate_biology=False),
    )
    assert len(result["updates"]) == 1
    row = result["updates"][0]
    assert row["counts_after"]["alpha"] > row["counts_before"]["alpha"]
    assert state.episode_records == {}


def test_brier_score_checks_calibration_outcomes() -> None:
    """Use external correctness labels to assess probability calibration."""
    assert brier_score([0.9, 0.2, 0.7], [1, 0, 0]) == pytest.approx((0.01 + 0.04 + 0.49) / 3)
    with pytest.raises(ValueError, match="equal lengths"):
        brier_score([0.5], [0, 1])


def test_complete_processor_emits_sixteen_operations_and_separate_biology() -> None:
    """Run the whole reconstruction pipeline and verify both execution tracks."""
    processor = ReconstructiveMemoryProcessor(Point20Config(simulate_biology=True))
    result = processor.process(make_request())
    assert len(result.operation_trace) == 16
    assert [row["operation_id"] for row in result.operation_trace] == [
        f"P20-{index:02d}" for index in range(1, 17)
    ]
    assert result.biological["operation_count"] == 16
    assert result.metadata["source_sequence_steps"] == 16
    assert result.reconstruction["not_retrieved_does_not_mean_absent"] is True
    assert result.reconstruction["supported_details"][0]["content"] == "riverbank"
    assert result.reconstruction["inferred_details"]


def test_processor_preserves_multiple_versions_of_the_same_event() -> None:
    """Later reconstruction rounds append versions rather than overwriting the earliest report."""
    processor = ReconstructiveMemoryProcessor(Point20Config(simulate_biology=False))
    request = make_request()
    first = processor.process(request)
    second = processor.process(request)
    versions = processor.state.episode_records[request.event_id]["versions"]
    assert len(versions) == 2
    assert versions[0]["version"] == 1
    assert versions[1]["version"] == 2
    assert len(first.operation_trace) == len(second.operation_trace) == 16


def test_biological_track_can_be_disabled_without_disabling_functional_reconstruction() -> None:
    """The functional track is independent of the optional population-rate reference."""
    result = ReconstructiveMemoryProcessor(
        Point20Config(simulate_biology=False)
    ).process(make_request())
    assert len(result.operation_trace) == 16
    assert result.biological["model"] == "disabled"
    assert all(row["biological_output"]["model"] == "disabled" for row in result.operation_trace)


def test_sqlite_annotation_and_sixteen_event_logging_when_logger_is_available(tmp_path) -> None:
    """Index all Point 20 symbols and persist one event for each of the sixteen operations."""
    from ore_reconstructive_memory import register_annotations, record_result_to_sqlite

    pytest.importorskip("ore_logger")
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    database_path = tmp_path / "ore_logger.sqlite3"
    result = ReconstructiveMemoryProcessor(
        Point20Config(simulate_biology=False)
    ).process(make_request())
    report = record_result_to_sqlite(result, database_path, repository_root=root)
    assert report["annotation_catalogue"]["files_seen"] >= 1
    assert report["database_summary"]["events"] == 16
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT COUNT(*) FROM annotations WHERE module_name LIKE 'ore_reconstructive_memory%'"
        ).fetchone()[0]
        events = connection.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=?", (report["run_id"],)
        ).fetchone()[0]
        status = connection.execute(
            "SELECT status FROM runs WHERE run_id=?", (report["run_id"],)
        ).fetchone()[0]
    assert rows >= 30
    assert events == 16
    assert status == "completed"

def test_repeated_schema_retellings_increase_familiarity_not_recollection() -> None:
    """A prior inference can feel more familiar after retelling without gaining episodic context."""
    processor = ReconstructiveMemoryProcessor(Point20Config(simulate_biology=False))
    request = make_request()
    first = processor.process(request)
    second = processor.process(request)
    first_canoe = next(
        item for item in first.candidate_evaluation["generation"]["candidates"]
        if item["content"] == "canoe"
    )
    second_canoe = next(
        item for item in second.candidate_evaluation["generation"]["candidates"]
        if item["content"] == "canoe"
    )
    assert second_canoe["familiarity"] > first_canoe["familiarity"]
    assert second_canoe["recollection"] == first_canoe["recollection"]
    assert second_canoe["status"] == "inferred"

def test_verified_schema_learning_changes_later_feature_prior_not_event_record() -> None:
    """A verified schema update changes later priors while remaining separate from episode memory."""
    request = make_request()
    config = Point20Config(simulate_biology=False)
    retrieval = reactivate_episodic_fragments(
        request, {"cue": request.cue, "cue_features": dict(request.cue)}, config,
    )
    ledger = build_feature_ledger(request, retrieval)
    baseline_activation = activate_candidate_schemas(request, ledger, config, {})
    baseline = generate_inferred_completions(request, ledger, baseline_activation, config, {})
    learned_counts = {
        "travel|object_type|canoe": {"alpha": 30.0, "beta": 1.0},
        "travel|object_type|boat": {"alpha": 1.0, "beta": 30.0},
    }
    learned_activation = activate_candidate_schemas(request, ledger, config, learned_counts)
    learned = generate_inferred_completions(
        request, ledger, learned_activation, config, learned_counts,
    )
    baseline_canoe = next(item for item in baseline["candidates"] if item.content == "canoe")
    learned_canoe = next(item for item in learned["candidates"] if item.content == "canoe")
    assert learned_canoe.schema_prior > baseline_canoe.schema_prior
    assert learned_canoe.status == "inferred"


def test_feedback_adjusted_probability_can_change_final_reporting_policy() -> None:
    """Reliable feedback affects response caution without rewriting the source/status label."""
    from ore_reconstructive_memory import choose_response_policy

    request = make_request(precision="exact")
    candidate = DetailCandidate(
        candidate_id="source-detail",
        feature_type="object_type",
        content="canoe",
        status="episodically_retrieved",
        confidence=0.20,
        feedback_probability=0.95,
        assigned_source="episodic_retrieval",
    )
    ledger = {
        "features": {
            "object_type": {
                "centrality": 0.90,
                "status": "not_retrieved",
                "likely_encoded": 0.70,
                "unavailable_not_absent": True,
                "values": [],
            }
        }
    }
    policy = choose_response_policy(
        request, ledger, [candidate], Point20Config(simulate_biology=False),
    )
    selected = policy["decisions"][0]["selected"]
    assert selected["action"] == "report"
    assert selected["confidence"] == pytest.approx(0.95)
    assert candidate.status == "episodically_retrieved"


def test_explicit_unknown_evidence_status_is_accepted() -> None:
    """Unknown is a legal evidence label and must not be converted into contradiction."""
    fragment = EpisodicFragment(
        detail_id="unknown-detail",
        feature_type="cause",
        content=None,
        status="unknown",
        retrieval_strength=0.0,
        episodic_context=0.0,
        source_type="unknown",
    )
    assert fragment.status == "unknown"

def test_schema_relation_templates_generate_explicit_inferred_relation_edges() -> None:
    """Schema scripts can propose missing participant/action/object relations without asserting them."""
    base = make_request()
    schema = SchemaProfile(
        schema_id="journey-relations",
        name="water journey relations",
        prior_probability=0.8,
        feature_probabilities={"object_type": {"canoe": 0.7, "boat": 0.3}},
        relation_templates=[{
            "subject": "traveler",
            "predicate": "uses",
            "object": "canoe",
            "probability": 0.85,
        }],
    )
    request = make_request(schemas=[schema])
    result = ReconstructiveMemoryProcessor(
        Point20Config(simulate_biology=False)
    ).process(request)
    relations = result.reconstruction["inferred_relations"]
    assert len(relations) == 1
    assert relations[0]["feature_type"] == "relation"
    assert relations[0]["content"] == {
        "subject": "traveler", "predicate": "uses", "object": "canoe"
    }
    assert relations[0]["status"] == "inferred"
    assert result.reconstruction["supported_relations"] == []


def test_observed_event_relations_are_preserved_in_event_graph() -> None:
    """A supported relation remains attached to its event and is not regenerated as a schema fact."""
    relation = EpisodicRelation(
        relation_id="rel-traveler-uses-vessel",
        subject="traveler",
        predicate="uses",
        object="vessel",
        status="episodically_retrieved",
        retrieval_strength=0.95,
        episodic_context=0.90,
        source_type="episodic_retrieval",
        source_id="source-rel-001",
        episode_id="story-001",
        timestamp=2.0,
    )
    schema = SchemaProfile(
        schema_id="travel-relations",
        name="travel relations",
        prior_probability=0.8,
        feature_probabilities={"object_type": {"canoe": 0.7, "boat": 0.3}},
        relation_templates=[{
            "subject": "traveler", "predicate": "uses", "object": "canoe",
            "probability": 0.8,
        }],
    )
    request = make_request(schemas=[schema], episodic_relations=[relation])
    result = ReconstructiveMemoryProcessor(
        Point20Config(simulate_biology=False)
    ).process(request)
    assert result.reconstruction["supported_relations"][0]["relation_id"] == relation.relation_id
    assert result.candidate_evaluation["generation"]["candidates"]
    assert result.reconstruction["inferred_relations"]
    assert result.reconstruction["provenance_graph"]["nodes"]


def test_relation_constraints_reject_an_incompatible_inferred_edge() -> None:
    """Temporal/spatial/causal relation constraints penalize inconsistent proposed triples."""
    request = make_request(
        context={
            **make_request().context,
            "constraints": [{
                "constraint_id": "agent-role",
                "constraint_type": "relation",
                "feature_type": "relation",
                "expected_relation": {
                    "subject": "traveler", "predicate": "uses", "object": "boat"
                },
                "hard": True,
                "weight": 1.0,
            }],
        }
    )
    candidate = DetailCandidate(
        candidate_id="schema-relation-1",
        feature_type="relation",
        content={"subject": "traveler", "predicate": "uses", "object": "canoe"},
        status="inferred",
        schema_prior=0.8,
    )
    from ore_reconstructive_memory import check_spatiotemporal_causal_constraints
    result = check_spatiotemporal_causal_constraints([candidate], request)
    assert result["violations"]
    assert result["inferred_relation_candidates"][0]["object"] == "canoe"

def test_contradicted_detail_does_not_validate_every_alternative() -> None:
    """A contradicted value is penalized without turning all other values into positive evidence."""
    from ore_reconstructive_memory import evaluate_candidates_bayesian

    fragment = EpisodicFragment(
        detail_id="rejected-red",
        feature_type="color",
        content="red",
        status="contradicted",
        retrieval_strength=0.95,
        episodic_context=0.90,
    )
    red = DetailCandidate(
        candidate_id="red", feature_type="color", content="red",
        status="inferred", schema_prior=0.5,
    )
    blue = DetailCandidate(
        candidate_id="blue", feature_type="color", content="blue",
        status="inferred", schema_prior=0.5,
    )
    result = evaluate_candidates_bayesian(
        [red, blue], [fragment], Point20Config(simulate_biology=False),
    )
    assert red.contradiction > 0.0
    assert blue.evidence_strength == 0.0
    assert blue.contradiction == 0.0


def test_schema_learning_uses_existing_schema_distribution_as_prior() -> None:
    """A positive verified observation refines rather than resets the schema's base prior."""
    request = make_request(feedback=[
        FeedbackRecord(
            "verified-canoe", "object_type", "canoe", True, "verified-source",
            reliability=0.99, independence_group="schema-source",
            schema_id="travel", verified=True, use_for_schema_update=True,
        )
    ])
    state = Point20State()
    result = update_general_schema_knowledge(
        request, state, Point20Config(simulate_biology=False),
    )
    row = result["updates"][0]
    base_prior = 0.70
    updated_mean = row["posterior_mean"]
    assert row["counts_before"]["alpha"] == pytest.approx(base_prior * 4.0)
    assert updated_mean > base_prior


def test_schema_updates_are_scoped_to_schema_feature_and_value() -> None:
    """One witness can contribute distinct verified observations to separate schema dimensions."""
    request = make_request(feedback=[
        FeedbackRecord(
            "f-canoe", "object_type", "canoe", True, "archive",
            reliability=0.95, independence_group="same-document",
            schema_id="travel", verified=True, use_for_schema_update=True,
        ),
        FeedbackRecord(
            "f-storm", "cause", "storm", True, "archive",
            reliability=0.95, independence_group="same-document",
            schema_id="travel", verified=True, use_for_schema_update=True,
        ),
    ])
    state = Point20State()
    output = update_general_schema_knowledge(
        request, state, Point20Config(simulate_biology=False),
    )
    assert len(output["updates"]) == 2

def test_verified_relation_feedback_refines_later_schema_relation_prior() -> None:
    """Verified relation feedback updates a general script prior for a later event reconstruction."""
    schema = SchemaProfile(
        schema_id="travel-relations",
        name="travel relation schema",
        prior_probability=0.8,
        feature_probabilities={"object_type": {"canoe": 0.7, "boat": 0.3}},
        relation_templates=[{
            "subject": "traveler", "predicate": "uses", "object": "canoe",
            "probability": 0.75,
        }],
    )
    feedback = FeedbackRecord(
        "verified-relation",
        "relation",
        {"subject": "traveler", "predicate": "uses", "object": "canoe"},
        True,
        "archival-source",
        reliability=0.99,
        independence_group="archive-relation-group",
        schema_id="travel-relations",
        verified=True,
        use_for_schema_update=True,
    )
    request = make_request(schemas=[schema], feedback=[feedback])
    processor = ReconstructiveMemoryProcessor(Point20Config(simulate_biology=False))
    first = processor.process(request)
    second = processor.process(request)
    first_relation = next(
        item for item in first.candidate_evaluation["generation"]["candidates"]
        if item["feature_type"] == "relation"
    )
    second_relation = next(
        item for item in second.candidate_evaluation["generation"]["candidates"]
        if item["feature_type"] == "relation"
    )
    first_prior = first_relation["schema_prior"] / first_relation["schema_support"]
    second_prior = second_relation["schema_prior"] / second_relation["schema_support"]
    assert second_prior > first_prior

def test_similar_episode_fragment_is_competitor_not_target_event_evidence() -> None:
    """A fragment from another episode cannot substantiate the target story's missing detail."""
    other_episode = EpisodicFragment(
        detail_id="older-story-canoe",
        feature_type="object_type",
        content="canoe",
        status="episodically_retrieved",
        retrieval_strength=0.98,
        episodic_context=0.95,
        attention_encoding=0.95,
        centrality=0.8,
        familiarity=0.8,
        recollection=0.95,
        source_type="episodic_retrieval",
        source_id="older-story-source",
        episode_id="story-older",
        timestamp=0.5,
    )
    base = make_request()
    request = make_request(
        episodic_fragments=[*base.episodic_fragments, other_episode],
    )
    result = ReconstructiveMemoryProcessor(
        Point20Config(simulate_biology=False)
    ).process(request)
    object_ledger = result.feature_ledger["features"]["object_type"]
    assert object_ledger["status"] == "not_retrieved"
    assert object_ledger["competing_episode_values"][0]["episode_id"] == "story-older"
    assert all(item.get("content") != "canoe" for item in result.reconstruction["supported_details"])
    assert result.reconstruction["competing_episode_details"]
    canoe_comparison = next(
        item for item in result.candidate_evaluation["competing_episodes"]["comparisons"]
        if isinstance(item, dict) and item.get("candidate_id", "").endswith(":object_type:canoe")
    )
    assert canoe_comparison["possible_episode_blend"] is True
    assert result.reconstruction["not_retrieved_does_not_mean_absent"] is True

def test_similar_episode_relation_is_competitor_not_target_event_support() -> None:
    """An event relation retrieved from a different episode remains a competing relation."""
    other_relation = EpisodicRelation(
        relation_id="older-uses-canoe",
        subject="traveler",
        predicate="uses",
        object="canoe",
        status="episodically_retrieved",
        retrieval_strength=0.99,
        episodic_context=0.99,
        source_type="episodic_retrieval",
        source_id="older-relation-source",
        episode_id="story-older",
        timestamp=0.5,
    )
    request = make_request(episodic_relations=[other_relation])
    result = ReconstructiveMemoryProcessor(
        Point20Config(simulate_biology=False)
    ).process(request)
    assert result.reconstruction["supported_relations"] == []
    assert result.reconstruction["competing_relations"]
    assert result.reconstruction["competing_relations"][0]["relation_id"] == other_relation.relation_id
    assert result.feature_ledger["event_graph"]["edges"] == []


def test_post_event_relation_suggestion_keeps_a_distinct_output_label() -> None:
    """A relation introduced after the event is shown as suggestion, never episodic support."""
    suggestion = EpisodicRelation(
        relation_id="suggested-cause",
        subject="traveler",
        predicate="caused_by",
        object="storm",
        status="suggested_after_event",
        retrieval_strength=0.7,
        episodic_context=0.7,
        source_type="later_testimony",
        source_id="later-witness",
        episode_id="story-001",
    )
    result = ReconstructiveMemoryProcessor(
        Point20Config(simulate_biology=False)
    ).process(make_request(episodic_relations=[suggestion]))
    assert result.reconstruction["supported_relations"] == []
    assert any(
        item["relation_id"] == suggestion.relation_id
        for item in result.reconstruction["post_event_suggested_relations"]
    )

def test_unknown_source_fragment_is_reported_as_source_uncertain() -> None:
    """A fragment explicitly marked unknown keeps that label instead of becoming direct evidence."""
    fragment = EpisodicFragment(
        detail_id="unattributed-object",
        feature_type="object_type",
        content="vessel",
        status="unknown",
        retrieval_strength=0.30,
        episodic_context=0.20,
        attention_encoding=0.60,
        source_type="unknown",
        source_id="unattributed-source",
        episode_id="story-001",
    )
    result = ReconstructiveMemoryProcessor(
        Point20Config(simulate_biology=False)
    ).process(make_request(episodic_fragments=[fragment]))
    assert result.feature_ledger["features"]["object_type"]["status"] == "weakly_retrieved"
    assert result.reconstruction["episode_uncertain_details"]
    assert result.reconstruction["supported_details"] == []


def test_unknown_source_relation_is_not_added_to_observed_event_graph() -> None:
    """A relation with unknown source remains uncertain and does not become an observed edge."""
    relation = EpisodicRelation(
        relation_id="unattributed-edge",
        subject="traveler",
        predicate="uses",
        object="canoe",
        status="unknown",
        retrieval_strength=0.7,
        episodic_context=0.6,
        source_type="unknown",
        source_id="unattributed-relation-source",
        episode_id="story-001",
    )
    result = ReconstructiveMemoryProcessor(
        Point20Config(simulate_biology=False)
    ).process(make_request(episodic_relations=[relation]))
    assert result.reconstruction["supported_relations"] == []
    assert result.reconstruction["episode_uncertain_relations"]
    assert result.feature_ledger["event_graph"]["edges"] == []
