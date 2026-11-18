"""Focused tests for Point 10 schema-guided inference and generalization."""

from __future__ import annotations

from pathlib import Path

import pytest

from ore_logger.database import LoggerDatabase
from ore_schema_formation.contracts import SchemaFrame, SchemaRelation, SchemaSlot
from ore_schema_formation.functional import context_signature
from ore_schema_guided_inference import (
    ActionOption,
    CaseEvidence,
    EvidenceItem,
    InferenceConfig,
    QueryOption,
    RelationEvidence,
    SchemaGuidedInferenceProcessor,
    SchemaProfile,
    OPERATION_REGISTRY,
    register_annotations,
)
from ore_schema_guided_inference.functional import (
    expected_information_gain,
    rank_schema_profiles,
    select_hierarchical_category,
)


def make_frame(
    schema_id: str,
    name: str,
    sound: str,
    relation_confidence: float,
    prior: float,
    context: dict | None = None,
) -> SchemaFrame:
    """Build a Point 09 schema frame for a familiar animal class."""
    context_values = context or {"environment=park": 1.0}
    key = context_signature({"environment": "park"})
    slots = {
        "animal": SchemaSlot(
            "slot:animal", "animal", {"animal": 1.0}, (schema_id + "-animal",),
            6, 1.0, "core", 0.9,
        ),
        "person": SchemaSlot(
            "slot:person", "person", {"person": 1.0}, (schema_id + "-person",),
            4, 0.8, "typical", 0.75,
        ),
    }
    relations = (
        SchemaRelation(
            "animal", "approaches", "person", "observed", 5, 6,
            relation_confidence, "typical", tuple(schema_id + "-ep" + str(i) for i in range(5)),
        ),
    )
    transitions = (
        {"from_event": "approach", "to_event": sound, "context_key": key,
         "count": 5, "eligible_count": 6, "probability": 0.84,
         "mandatory": False, "status": "common"},
        {"from_event": sound, "to_event": "move_closer", "context_key": key,
         "count": 4, "eligible_count": 6, "probability": 0.70,
         "mandatory": False, "status": "common"},
        {"from_event": sound, "to_event": "retreat", "context_key": "*",
         "count": 1, "eligible_count": 6, "probability": 0.10,
         "mandatory": False, "status": "alternative"},
    )
    return SchemaFrame(
        schema_id=schema_id, name=name, slots=slots, relations=relations,
        transitions=transitions, goals={"classify_animal": 1.0},
        outcomes={"identified": 1.0}, context_profile=context_values,
        event_types=("approach", sound, "move_closer", "retreat"),
        supporting_episode_ids=tuple(schema_id + "-ep" + str(i) for i in range(6)),
        confidence=0.85,
    )


def make_profiles() -> tuple[SchemaProfile, SchemaProfile]:
    """Create dog and cat profiles with overlapping surface cues and discriminating sound cues."""
    dog = SchemaProfile(
        frame=make_frame("dog", "dog schema", "bark", 0.92, 0.65),
        feature_probabilities={
            "fur": {"true": 0.95, "false": 0.05},
            "legs": {"four": 0.92, "two": 0.08},
            "sound": {"bark": 0.85, "meow": 0.03, "purr": 0.02, "other": 0.10},
            "activity": {"chase": 0.60, "sleep": 0.15, "play": 0.25},
        },
        feature_diagnosticity={"sound": 1.0, "fur": 0.5, "legs": 0.5},
        prior_probability=0.65,
    )
    cat = SchemaProfile(
        frame=make_frame("cat", "cat schema", "meow", 0.20, 0.35),
        feature_probabilities={
            "fur": {"true": 0.90, "false": 0.10},
            "legs": {"four": 0.95, "two": 0.05},
            "sound": {"bark": 0.03, "meow": 0.85, "purr": 0.10, "other": 0.02},
            "activity": {"chase": 0.25, "sleep": 0.50, "play": 0.25},
        },
        feature_diagnosticity={"sound": 1.0, "fur": 0.5, "legs": 0.5},
        prior_probability=0.35,
    )
    return dog, cat


def dog_case(**overrides) -> CaseEvidence:
    """Build an unfamiliar dog-like case with an explicitly observed sound and temporal context."""
    features = {
        "fur": EvidenceItem("fur", True, evidence_id="fur-observation"),
        "legs": EvidenceItem("legs", "four", evidence_id="legs-observation"),
        "sound": EvidenceItem("sound", "bark", evidence_id="sound-observation"),
        "role:animal": EvidenceItem("role:animal", "animal", evidence_id="animal-role"),
        "unused_color": EvidenceItem("unused_color", None, status="not_observed"),
    }
    values = {
        "case_id": "new-dog-like-case",
        "features": features,
        "relations": (RelationEvidence("animal", "approaches", "person", evidence_id="edge-1"),),
        "event_history": ("approach", "bark"),
        "context": {"environment": "park"},
        "goals": ("classify_animal",),
    }
    values.update(overrides)
    return CaseEvidence(**values)


def processor(config: InferenceConfig | None = None) -> SchemaGuidedInferenceProcessor:
    """Register a small schema library for all Point 10 tests."""
    return SchemaGuidedInferenceProcessor(make_profiles(), config=config)


def test_registry_contains_exact_17_ordered_source_stages():
    """Verify every source step is represented once and in the correct order."""
    assert len(OPERATION_REGISTRY) == 17
    assert [item.operation_id for item in OPERATION_REGISTRY] == [
        f"P10-{index:02d}" for index in range(1, 18)
    ]


def test_unknown_and_not_observed_features_are_neutral_not_absent():
    """Confirm that missing measurements do not lower a schema's posterior."""
    profiles = make_profiles()
    case = dog_case()
    baseline = rank_schema_profiles(case, profiles, InferenceConfig())
    with_unknown = CaseEvidence(
        case_id="same-case", features={
            "fur": case.features["fur"], "legs": case.features["legs"],
            "sound": case.features["sound"],
            "role:animal": case.features["role:animal"],
            "unused_color": EvidenceItem("unused_color", False, status="unknown"),
        },
        relations=case.relations, event_history=case.event_history,
        context=case.context, goals=case.goals,
    )
    with_unknown_scores = rank_schema_profiles(with_unknown, profiles, InferenceConfig())
    base = {item.schema_id: item.posterior for item in baseline}
    actual = {item.schema_id: item.posterior for item in with_unknown_scores}
    assert actual == pytest.approx(base)


def test_candidate_ranking_selects_supported_specific_category():
    """Combine priors, likelihood ratios, context, goals, and temporal transitions."""
    result = processor().infer(dog_case())
    assert result.selected_schema_id == "dog"
    assert result.selected_category == "dog schema"
    assert result.confidence >= 0.60
    assert result.candidate_scores[0].schema_id == "dog"
    assert len(result.operation_trace) == 17


def test_unknown_candidate_is_retained_when_schema_library_exceeds_top_k():
    """Keep an open-set alternative even when known schemas fill the candidate limit."""
    profiles = make_profiles()
    ranked = rank_schema_profiles(
        dog_case(), profiles, InferenceConfig(max_candidates=1)
    )
    assert len(ranked) == 1
    assert ranked[0].schema_id == "unknown" or any(
        item.schema_id == "unknown" for item in ranked
    )


def test_relation_prediction_outcome_is_scored_separately_from_feature_prediction():
    """Evaluate a later observed relation and include its calibration score."""
    model = processor()
    result = model.infer(dog_case(relations=()))
    assert result.inferred_relations
    feedback = model.evaluate_outcome(
        result,
        observed_evidence=(),
        observed_relations=(
            RelationEvidence("animal", "approaches", "person", status="observed",
                            source="later_observation"),
        ),
    )
    assert any(item["kind"] == "relation" and item["correct"]
               for item in feedback.tested_predictions)
    assert feedback.mean_brier_score is not None


def test_explicit_contradictory_cue_reduces_dog_evidence_score():
    """An explicit incompatible observation is negative evidence, unlike an unknown value."""
    profiles = make_profiles()
    positive = dog_case()
    contradictory = CaseEvidence(
        case_id="contradictory", features={
            "fur": positive.features["fur"], "legs": positive.features["legs"],
            "sound": EvidenceItem("sound", "meow", status="contradicted", evidence_id="negative-bark"),
        },
        context=positive.context, goals=positive.goals,
    )
    base = {x.schema_id: x for x in rank_schema_profiles(positive, profiles, InferenceConfig())}
    new = {x.schema_id: x for x in rank_schema_profiles(contradictory, profiles, InferenceConfig())}
    assert new["dog"].feature_score < base["dog"].feature_score
    assert "negative-bark" in new["dog"].conflicting_evidence_ids


def test_predictions_are_inferred_and_never_added_to_current_case():
    """Infer likely properties and missing slots without mutating the evidence record."""
    case = dog_case()
    before = case.to_dict()
    result = processor().infer(case)
    assert any(item.name == "activity" for item in result.inferred_properties)
    assert any(item.name == "person" for item in result.inferred_properties)
    assert all(item.status == "inferred" for item in result.inferred_properties)
    assert all(item.provenance == "schema_prediction" for item in result.inferred_properties)
    assert all(item["provenance"] == "schema_prediction"
               for item in result.evidence_ledger if item.get("status") == "inferred")
    assert case.to_dict() == before


def test_relation_completion_preserves_direction_and_uncertainty():
    """Predict a missing schema edge as inferred while retaining its direction and edge status."""
    case = dog_case(relations=())
    result = processor().infer(case)
    prediction = next(item for item in result.inferred_relations if item.name == "approaches")
    assert prediction.value == {
        "subject": "animal", "predicate": "approaches", "object": "person"
    }
    assert prediction.status == "inferred"
    assert prediction.provenance == "schema_prediction"


def test_next_event_prediction_uses_context_specific_transition_before_global():
    """Prefer the park-conditioned next event and keep it separate from causation."""
    result = processor().infer(dog_case())
    assert result.next_event_predictions
    first = result.next_event_predictions[0]
    assert first.name == "move_closer"
    assert first.status == "inferred"
    trace = result.operation_trace[9]["functional_output"]
    assert trace["temporal_and_causal_claims_separated"] is True


def test_source_ledger_distinguishes_current_observation_episode_and_prediction():
    """Keep current observations, retrieved cases, and schema completions under separate labels."""
    model = processor()
    prior = CaseEvidence(
        "remembered-dog", {"fur": EvidenceItem("fur", True, source="prior_episode")},
        context={"environment": "park"}, goals=("classify_animal",),
    )
    assert model.biological_network.observe_case(prior)
    result = model.infer(dog_case())
    provenances = {item.get("provenance") for item in result.evidence_ledger}
    assert "current_case" in provenances
    assert "specific_episode" in provenances
    assert "schema_prediction" in provenances
    recalled = [item for item in result.evidence_ledger
                if item.get("content_type") == "episodic_memory"]
    assert recalled and recalled[0]["case_id"] == "remembered-dog"


def test_ambiguous_case_can_abstain_and_choose_diagnostic_query():
    """Use expected information gain to recommend a discriminating cue rather than force a label."""
    config = InferenceConfig(posterior_threshold=0.95, candidate_margin=0.20)
    model = processor(config)
    case = CaseEvidence(
        "ambiguous-animal",
        features={
            "fur": EvidenceItem("fur", True),
            "legs": EvidenceItem("legs", "four"),
        },
        context={"environment": "park"}, goals=("classify_animal",),
        query_options=(QueryOption("sound", cost=0.0, risk=0.0, action_label="listen to sound"),),
    )
    result = model.infer(case)
    assert result.selected_schema_id is None
    assert result.selected_category == "uncertain/unknown"
    assert result.query_recommendation is not None
    assert result.query_recommendation.feature == "sound"
    assert result.query_recommendation.expected_information_gain > 0.0


def test_context_goal_and_hierarchical_parent_fallback():
    """Select a broader parent when a specific child category does not meet the threshold."""
    dog, cat = make_profiles()
    parent_frame = make_frame("animal", "animal", "bark", 0.5, 0.7)
    parent_frame.child_schema_ids = ("dog", "cat")
    parent = SchemaProfile(parent_frame, prior_probability=0.7)
    dog.frame.parent_schema_ids = ("animal",)
    cat.frame.parent_schema_ids = ("animal",)
    profiles = [parent, dog, cat]
    scores = (
        # Deliberately force a child to be too uncertain while its parent remains plausible.
        __import__("ore_schema_guided_inference").CandidateScore(
            "dog", "dog schema", 0.48, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            parent_schema_ids=("animal",),
        ),
        __import__("ore_schema_guided_inference").CandidateScore(
            "animal", "animal", 0.42, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ),
        __import__("ore_schema_guided_inference").CandidateScore(
            "unknown", "unknown/other", 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ),
    )
    selected, category, confidence, reason = select_hierarchical_category(
        scores, profiles, InferenceConfig(posterior_threshold=0.70, unknown_confidence_threshold=0.40)
    )
    assert selected == "animal"
    assert category == "animal"
    assert reason == "broader_parent_selected"


def test_expected_information_gain_rewards_features_that_separate_schemas():
    """Verify a feature with divergent candidate distributions has positive information gain."""
    dog, cat = make_profiles()
    case = CaseEvidence(
        "query-case", features={"fur": EvidenceItem("fur", True)}
    )
    ranked = rank_schema_profiles(case, [dog, cat], InferenceConfig())
    query = expected_information_gain(
        QueryOption("sound"), ranked, [dog, cat], InferenceConfig()
    )
    assert query is not None
    assert query.expected_information_gain > 0.0
    assert set(query.predicted_answers) >= {"bark", "meow"}


def test_action_selection_subtracts_cost_and_risk_from_expected_utility():
    """Prefer a low-risk action when the higher nominal reward carries a larger risk penalty."""
    case = dog_case(action_options=(
        ActionOption("approach_animal", {"dog": 4.0, "cat": 3.0}, cost=0.5, risk=0.95),
        ActionOption("observe_from_distance", {"dog": 2.0, "cat": 2.0}, risk=0.05),
    ))
    result = processor().infer(case)
    assert result.action_decision is not None
    assert result.action_decision.action_id == "observe_from_distance"
    assert result.action_decision.selected


def test_case_refinement_preserves_superseded_observations():
    """Keep old and new observations separately when a fresh sensory cue contradicts a prior one."""
    model = processor()
    previous = CaseEvidence(
        "refine-case", {"sound": EvidenceItem("sound", "bark", evidence_id="old-sound")},
    )
    revised = model.refine(
        previous, [EvidenceItem("sound", "meow", evidence_id="new-sound")]
    )
    assert revised.features["sound"].value == "meow"
    assert revised.evidence_history[0].value == "bark"
    assert revised.evidence_history[0].status == "observed"
    assert revised.evidence_history[0].evidence_id == "old-sound"


def test_feedback_scores_only_tested_predictions_and_does_not_learn_without_permission():
    """Evaluate observed outcomes while leaving reusable schema probabilities unchanged by default."""
    model = processor()
    result = model.infer(dog_case())
    feature_prediction = next(item for item in result.inferred_properties if item.name == "activity")
    profile = model.profiles["dog"]
    original = dict(profile.feature_probabilities["activity"])
    feedback = model.evaluate_outcome(
        result, [EvidenceItem("activity", feature_prediction.value, source="later_observation")],
        learning_authorized=False,
    )
    assert feedback.tested_predictions
    assert feedback.mean_brier_score is not None
    assert feedback.schema_update_authorized is False
    assert feedback.update_status == "not_authorized"
    assert model.profiles["dog"].feature_probabilities["activity"] == original


def test_authorized_schema_update_requires_repeated_distinct_validated_cases():
    """Update feature likelihoods only after explicit permission and enough independent cases."""
    config = InferenceConfig(minimum_update_cases=3)
    model = processor(config)
    result = model.infer(dog_case())
    evidence = [EvidenceItem("activity", "sleep", reliability=0.9, source="validated_outcome")]
    blocked = model.evaluate_outcome(
        result, evidence, learning_authorized=True, validated_case_ids=("case-1", "case-2")
    )
    assert blocked.update_status == "insufficient_distinct_validated_cases"
    before = model.profiles["dog"].feature_probabilities["activity"]["sleep"]
    updated = model.evaluate_outcome(
        result, evidence, learning_authorized=True,
        validated_case_ids=("case-1", "case-2", "case-3"),
    )
    assert updated.update_status == "feature_likelihoods_updated_after_authorized_validation"
    after = model.profiles["dog"].feature_probabilities["activity"]["sleep"]
    assert after > before
    assert model.profile_update_history[-1]["validated_case_ids"] == ["case-1", "case-2", "case-3"]


def test_episodic_memory_duplicate_guard_and_retrieval_identity():
    """Retain two distinct cases and avoid duplicate hippocampal evidence inflation."""
    model = processor()
    prior = CaseEvidence(
        "old-case", {"fur": EvidenceItem("fur", True)},
        context={"environment": "park"}, goals=("classify_animal",),
    )
    assert model.biological_network.observe_case(prior)
    assert not model.biological_network.observe_case(prior)
    result = model.infer(dog_case())
    assert "old-case" in result.diagnostics["retrieved_case_ids"]
    assert result.biological["hippocampus"]["case_count"] == 2


def test_biological_network_models_expose_separate_functional_contributions():
    """Check semantic association, prefrontal gain, episodic comparison, and conflict monitoring."""
    model = processor()
    model.biological_network.prepare(list(model.profiles.values()))
    result = model.infer(dog_case())
    assert result.biological["temporal_semantic"]["association_count"] > 0
    assert result.biological["medial_prefrontal"]["selection_count"] > 0
    assert result.biological["hippocampus"]["case_count"] == 1
    assert result.biological["attention_conflict"]["selection_count"] > 0
    assert "surrogate" in result.biological["scientific_scope"]


def test_sqlite_indexes_point10_symbols_and_records_all_17_operations(tmp_path):
    """Index Point 10 docstrings and persist the full operation trace in ORE SQLite."""
    model = processor()
    result = model.infer(dog_case())
    database_path = tmp_path / "point10.sqlite3"
    root = Path(__file__).resolve().parents[1]
    catalog = register_annotations(database_path, root)
    assert catalog["symbols_written"] > 0
    database = LoggerDatabase(database_path)
    rows = database.search_annotations("inference", limit=1000)
    assert any(row["module_name"] == "ore_schema_guided_inference.pipeline"
               and row["symbol_kind"] == "class" for row in rows)
    from ore_schema_guided_inference.pipeline import record_result_to_sqlite
    stored = record_result_to_sqlite(result, database_path, root)
    assert stored["database_summary"]["events"] == 17
    events = database.list_events(stored["run_id"])
    assert len(events) == 17
    assert events[0]["metadata"]["operation_id"] == "P10-01"
    assert events[-1]["metadata"]["operation_id"] == "P10-17"


def test_schema_frame_and_profile_round_trip_keep_source_provenance():
    """Verify inference profiles reuse existing Point 09 frame IDs and source episode lineage."""
    dog, _ = make_profiles()
    serialized = dog.to_dict()
    assert serialized["schema_id"] == "dog"
    assert serialized["frame"]["supporting_episode_ids"] == list(dog.frame.supporting_episode_ids)
    assert serialized["feature_probabilities"]["sound"]["bark"] == pytest.approx(0.85)


def test_inference_result_serializes_all_predictions_and_17_stage_trace():
    """Confirm the public result is JSON-compatible and contains every sequential operation."""
    result = processor().infer(dog_case())
    payload = result.to_dict()
    assert payload["case_id"] == "new-dog-like-case"
    assert len(payload["operation_trace"]) == 17
    assert payload["operation_trace"][0]["operation_id"] == "P10-01"
    assert payload["operation_trace"][-1]["operation_id"] == "P10-17"
    assert all(item["status"] == "inferred" for item in payload["inferred_properties"])
