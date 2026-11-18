"""Tests for Point 06 commonality detection and revisable concept formation."""

from __future__ import annotations

from pathlib import Path

import pytest

from ore_commonality_concepts import (
    AttentionControlNetwork,
    BiologicalCommonalityNetwork,
    CorticostriatalFeedbackNetwork,
    CommonalityDetectionConceptFormationProcessor,
    ConceptLearningConfig,
    ConceptLearningRequest,
    DistributedCorticalConceptNetwork,
    EpisodeInput,
    HippocampalEpisodeNetwork,
    register_annotations,
    registry_as_dicts,
)
from ore_commonality_concepts.functional import (
    build_concept_candidate,
    canonical_features,
    canonical_relations,
    estimate_contrastive_diagnosticity,
    estimate_feature_statistics,
    estimate_pairwise_associations,
    estimate_relation_strengths,
    predict_episode,
    retrieve_related_episodes,
)
from ore_logger.database import LoggerDatabase


def _bird_examples() -> tuple[list[EpisodeInput], list[EpisodeInput]]:
    """Build three bird episodes and two non-bird comparison episodes with controlled evidence."""
    target = [
        EpisodeInput(
            "sparrow",
            {"feathers": 1.0, "wings": 1.0, "beak": 1.0, "tree": 1.0, "small": 1.0},
            context={"location": "garden", "time": "morning"},
            relations=(("feathers", "cover", "bird"), ("wings", "part_of", "bird")),
            feature_modalities={"feathers": "visual", "bird_call": "auditory"},
            reliability=1.0,
        ),
        EpisodeInput(
            "robin",
            {"feathers": 1.0, "wings": 1.0, "beak": 1.0, "tree": 0.0, "small": 0.0},
            context={"location": "field", "time": "afternoon"},
            relations=(("feathers", "cover", "bird"), ("wings", "part_of", "bird")),
            reliability=0.9,
        ),
        EpisodeInput(
            "pigeon",
            {"feathers": 1.0, "wings": 1.0, "beak": 1.0, "tree": 1.0, "small": 0.0},
            context={"location": "square", "time": "evening"},
            relations=(("feathers", "cover", "bird"), ("wings", "part_of", "bird")),
            reliability=0.95,
        ),
    ]
    comparison = [
        EpisodeInput(
            "airplane",
            {"feathers": 0.0, "wings": 1.0, "beak": 0.0, "tree": 0.0, "small": 0.0, "metal": 1.0},
            relations=(("wings", "part_of", "airplane"),),
        ),
        EpisodeInput(
            "butterfly",
            {"feathers": 0.0, "wings": 1.0, "beak": 0.0, "tree": 1.0, "small": 0.0, "metal": 0.0},
            relations=(("wings", "part_of", "insect"),),
        ),
    ]
    return target, comparison


def _request(
    target: list[EpisodeInput] | None = None,
    comparison: list[EpisodeInput] | None = None,
    predictions: list[EpisodeInput] | None = None,
) -> ConceptLearningRequest:
    """Create a bird concept request with explicit task attention and test examples."""
    positive, negative = _bird_examples()
    return ConceptLearningRequest(
        concept_id="bird",
        target_episodes=target if target is not None else positive,
        comparison_episodes=comparison if comparison is not None else negative,
        prediction_episodes=predictions or [],
        attention_weights={"feathers": 1.0, "wings": 0.9, "beak": 0.9, "tree": 0.2},
        goal_weights={"feathers": 0.9, "wings": 0.9, "beak": 0.8},
        prior_weights={"feathers": 0.8, "wings": 0.8, "beak": 0.8},
    )


def test_registry_contains_sixteen_source_ordered_commonality_stages() -> None:
    """Verify operation names and IDs follow the exact Point 06 sequence."""
    operations = registry_as_dicts()
    assert len(operations) == 16
    assert [item["operation_id"] for item in operations] == [
        f"CC{index:02d}" for index in range(1, 17)
    ]
    assert "prototype" in operations[15]["purpose"].lower() or "shared features" in operations[15]["purpose"].lower()
    assert "cortical" in operations[11]["name"].lower()


def test_episode_contract_preserves_unknown_separately_from_absence() -> None:
    """Accept None as unobserved and numeric zero as explicitly observed absence."""
    episode = EpisodeInput("observations", {"feathers": None, "wings": 0.0, "beak": 1.0})
    assert episode.features["feathers"] is None
    assert episode.features["wings"] == 0.0
    assert episode.features["beak"] == 1.0


@pytest.mark.parametrize(
    "features",
    [
        {"": 1.0},
        {"wing": float("nan")},
        {"wing": float("inf")},
        {"wing": -0.1},
        {"wing": 1.1},
        {},
    ],
)
def test_episode_contract_rejects_invalid_feature_observations(features: dict) -> None:
    """Reject invalid feature names, non-finite values, out-of-range values, and empty feature maps."""
    with pytest.raises(ValueError):
        EpisodeInput("invalid", features)


def test_request_rejects_duplicate_ids_and_wrong_category_roles() -> None:
    """Prevent ambiguous training membership and require labelled roles to be consistent."""
    target, comparison = _bird_examples()
    with pytest.raises(ValueError, match="unique"):
        ConceptLearningRequest(
            "bird",
            target_episodes=[target[0]],
            prediction_episodes=[EpisodeInput("sparrow", {"feathers": 1.0})],
        )
    with pytest.raises(ValueError, match="cannot contain category_label=False"):
        ConceptLearningRequest(
            "bird",
            target_episodes=[EpisodeInput("negative", {"wing": 1.0}, category_label=False)],
        )
    with pytest.raises(ValueError, match="cannot contain category_label=True"):
        ConceptLearningRequest(
            "bird",
            comparison_episodes=[EpisodeInput("positive", {"wing": 1.0}, category_label=True)],
        )


def test_canonical_feature_aliases_combine_observed_values_without_inventing_missing_data() -> None:
    """Normalize feature aliases and combine duplicate observations while ignoring unknowns."""
    episode = EpisodeInput(
        "alias",
        {"wing_surface": 1.0, "wings": 0.5, "winged": None},
    )
    result = canonical_features(
        episode,
        {"wing_surface": "wings", "winged": "wings"},
    )
    assert result["wings"] == pytest.approx(0.75)


def test_canonical_relations_apply_aliases_and_remove_exact_duplicates() -> None:
    """Normalize relationship endpoints so equivalent typed relations can co-occur."""
    episode = EpisodeInput(
        "relations",
        {"wing_surface": 1.0},
        relations=(
            ("wing_surface", "part_of", "bird"),
            ("wings", "part_of", "bird"),
        ),
    )
    result = canonical_relations(episode, {"wing_surface": "wings"})
    assert result == (("wings", "part_of", "bird"),)


def test_feature_statistics_exclude_unknowns_but_count_explicit_zero() -> None:
    """Compute prevalence from observed episodes only with zero retained as an observation."""
    examples = [
        EpisodeInput("one", {"feathers": 1.0, "wings": 1.0}),
        EpisodeInput("two", {"feathers": None, "wings": 0.0}),
        EpisodeInput("three", {"feathers": 0.0, "wings": None}),
    ]
    stats = estimate_feature_statistics(examples, {})
    assert stats["feathers"]["observation_count"] == 2
    assert stats["feathers"]["prevalence"] == pytest.approx(0.5)
    assert stats["wings"]["observation_count"] == 2
    assert stats["wings"]["prevalence"] == pytest.approx(0.5)
    assert stats["feathers"]["effective_sample_size"] == pytest.approx(2.0)


def test_feature_statistics_weight_unreliable_examples_less() -> None:
    """Verify episode reliability controls weighted feature prevalence."""
    examples = [
        EpisodeInput("reliable-present", {"wings": 1.0}, reliability=1.0),
        EpisodeInput("uncertain-absent", {"wings": 0.0}, reliability=0.1),
    ]
    stats = estimate_feature_statistics(examples, {})
    assert stats["wings"]["prevalence"] > 0.9
    assert stats["wings"]["coverage"] == pytest.approx(1.0)


def test_contrastive_diagnosticity_downweights_shared_background_features() -> None:
    """Check that bird-specific features score above properties shared with alternatives."""
    target, comparison = _bird_examples()
    scores = estimate_contrastive_diagnosticity(target, comparison, {}, beta_prior=1.0)
    feather = scores["feathers"]["log_odds_difference"]
    wing = scores["wings"]["log_odds_difference"]
    tree = scores["tree"]["log_odds_difference"]
    assert feather is not None and wing is not None and tree is not None
    assert feather > wing
    assert feather > tree


def test_cooccurrence_and_typed_relations_are_detected() -> None:
    """Estimate feature conjunctions and repeated part relations across distinct episodes."""
    target, _ = _bird_examples()
    associations = estimate_pairwise_associations(target, {})
    relation_strengths = estimate_relation_strengths(target, {})
    assert associations["beak::feathers"] > 0.0
    key = "wings :: part_of :: bird"
    assert relation_strengths[key]["support"] == pytest.approx(1.0)
    assert len(relation_strengths[key]["supporting_episode_ids"]) == 3


def test_retrieval_returns_related_episodes_without_merging_episode_identity() -> None:
    """Retrieve similar episodes by partial cues while preserving each memory identifier."""
    target, comparison = _bird_examples()
    retrieved = retrieve_related_episodes(target[0], target[1:] + comparison, {}, limit=10)
    assert retrieved
    assert retrieved[0]["episode_id"] in {"robin", "pigeon"}
    assert all(row["episode_id"] != "sparrow" for row in retrieved)


def test_processor_runs_all_sixteen_stages_and_forms_diagnostic_shared_structure() -> None:
    """Run the integrated pipeline and verify recurring category features outrank background details."""
    processor = CommonalityDetectionConceptFormationProcessor(
        ConceptLearningConfig(cortical_units=12, recurrent_iterations=4)
    )
    result = processor.process(_request())
    assert len(result.operation_trace) == 16
    assert [row["operation_id"] for row in result.operation_trace] == [
        f"CC{index:02d}" for index in range(1, 17)
    ]
    assert all(row["operation"] and row["explanation"] for row in result.operation_trace)
    assert result.concept["feature_weights"]["feathers"] > result.concept["feature_weights"]["tree"]
    assert "feathers" in result.concept["shared_features"]
    assert "feathers" in result.associative_index["feature_to_concept"]
    assert result.biological["distributed_cortical_network"]["population_size"] == 12
    assert result.concept["representation_scope"].startswith("shared features and relations only")


def test_no_comparison_examples_produce_fit_only_not_fabricated_membership_posterior() -> None:
    """Refuse to report category posterior membership when no contrastive examples exist."""
    target, _ = _bird_examples()
    test_episode = EpisodeInput("new-bird", {"feathers": 1.0, "wings": 1.0, "beak": 1.0})
    processor = CommonalityDetectionConceptFormationProcessor(
        ConceptLearningConfig(cortical_units=8)
    )
    result = processor.process(_request(target=target, comparison=[], predictions=[test_episode]))
    assert result.predictions[0]["status"] == "fit_only_no_contrastive_examples"
    assert result.predictions[0]["posterior_category_probability"] is None
    assert result.predictions[0]["fit_score"] > 0.8


def test_prediction_uses_contrastive_evidence_for_new_examples() -> None:
    """Predict category membership for a held-out example from target and comparison likelihoods."""
    target, comparison = _bird_examples()
    example = EpisodeInput(
        "unseen-bird",
        {"feathers": 1.0, "wings": 1.0, "beak": 1.0, "tree": 0.0},
    )
    processor = CommonalityDetectionConceptFormationProcessor(
        ConceptLearningConfig(cortical_units=8)
    )
    result = processor.process(
        _request(target=target, comparison=comparison, predictions=[example])
    )
    prediction = result.predictions[0]
    assert prediction["status"] == "contrastive_posterior"
    assert prediction["posterior_category_probability"] > 0.8
    assert prediction["predicted_category"] is True


def test_labelled_prediction_feedback_revises_concept_only_after_prediction() -> None:
    """Record the initial prediction first, then incorporate explicitly labelled feedback."""
    target, comparison = _bird_examples()
    feedback = EpisodeInput(
        "new-category-member",
        {"feathers": 1.0, "wings": 1.0, "beak": 1.0, "tree": 0.0},
        category_label=True,
    )
    processor = CommonalityDetectionConceptFormationProcessor(
        ConceptLearningConfig(cortical_units=8)
    )
    result = processor.process(
        _request(target=target, comparison=comparison, predictions=[feedback])
    )
    assert len(result.predictions) == 1
    assert result.predictions[0]["expected_category"] is True
    assert result.mismatch_and_revision["feedback_added"] == 1
    assert result.operation_trace[13]["functional_output"][
        "corticostriatal_feedback_gate"
    ]["mean_learning_gate"] > 0.0
    assert "new-category-member" in result.concept["supporting_episode_ids"]
    assert result.operation_trace[13]["status"] == "completed_with_feedback_revision"


def test_labelled_negative_feedback_updates_comparison_store() -> None:
    """Use labelled non-members as contrastive evidence only after measuring their prediction error."""
    target, comparison = _bird_examples()
    feedback = EpisodeInput(
        "new-aircraft",
        {"feathers": 0.0, "wings": 1.0, "beak": 0.0, "metal": 1.0},
        category_label=False,
    )
    processor = CommonalityDetectionConceptFormationProcessor(
        ConceptLearningConfig(cortical_units=8)
    )
    result = processor.process(
        _request(target=target, comparison=comparison, predictions=[feedback])
    )
    assert result.mismatch_and_revision["negative_feedback_ids"] == ["new-aircraft"]
    assert "new-aircraft" in result.concept["comparison_episode_ids"]


def test_duplicate_training_examples_are_idempotent_but_content_conflicts_raise() -> None:
    """Do not duplicate repeated episode IDs and reject attempts to rewrite an existing episode."""
    target, comparison = _bird_examples()
    processor = CommonalityDetectionConceptFormationProcessor(
        ConceptLearningConfig(cortical_units=8)
    )
    first = processor.process(_request(target=target, comparison=comparison))
    updates = first.biological["distributed_cortical_network"]["learning_update_count"]
    second = processor.process(_request(target=target, comparison=comparison))
    assert len(second.concept["supporting_episode_ids"]) == 3
    assert second.biological["distributed_cortical_network"]["learning_update_count"] == updates
    changed = EpisodeInput("sparrow", {"feathers": 0.0, "wings": 0.0})
    with pytest.raises(ValueError, match="different content"):
        processor.process(_request(target=[changed], comparison=[]))


def test_episode_cannot_change_from_target_to_comparison() -> None:
    """Keep episode category provenance stable when a concept is incrementally updated."""
    target, comparison = _bird_examples()
    processor = CommonalityDetectionConceptFormationProcessor(
        ConceptLearningConfig(cortical_units=8)
    )
    processor.process(_request(target=target, comparison=comparison))
    with pytest.raises(ValueError, match="both target and comparison"):
        processor.process(_request(target=[], comparison=[target[0]]))


def test_cortical_network_learns_new_episodes_once_and_keeps_distributed_state() -> None:
    """Verify recurrent cortical activity and Hebbian updates are stable under duplicate calls."""
    config = ConceptLearningConfig(cortical_units=10)
    model = DistributedCorticalConceptNetwork(config)
    target, _ = _bird_examples()
    first = model.learn(target, {}, {"feathers": 1.0, "wings": 1.0, "beak": 1.0})
    second = model.learn(target, {}, {"feathers": 1.0, "wings": 1.0, "beak": 1.0})
    assert first["learning_update_performed"] is True
    assert second["learning_update_performed"] is False
    assert second["learning_update_count"] == 1
    assert len(second["cortical_pattern"]) == 10


def test_biological_subsystems_keep_attention_episode_and_cortical_roles_separate() -> None:
    """Check the population model exposes distinct functional systems rather than one concept centre."""
    target, _ = _bird_examples()
    network = BiologicalCommonalityNetwork(
        ConceptLearningConfig(cortical_units=8)
    )
    network.register_episodes(target)
    gains = network.attention.select_feature_gains(
        ["feathers", "wings"], {"feathers": 1.0}, {"wings": 1.0}, {}
    )
    related = network.hippocampus.retrieve(target[0], {})
    cortical = network.learn_cortical_commonality(
        target, {}, {"feathers": 1.0, "wings": 1.0, "beak": 1.0}
    )
    snapshot = network.snapshot()
    assert gains["feathers"] > 0.0
    assert related
    assert cortical["learning_update_performed"] is True
    assert "attention_control" in snapshot
    assert "hippocampal_episode_network" in snapshot
    assert "distributed_cortical_network" in snapshot


def test_annotation_catalog_indexes_point06_classes_and_functions(tmp_path: Path) -> None:
    """Register docstrings and check that the reusable SQLite catalogue resolves Point 06 symbols."""
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "ore_point06_annotations.sqlite"
    report = register_annotations(database_path, root)
    assert report["parse_errors"] == 0
    database = LoggerDatabase(database_path)
    expected = (
        ("ore_commonality_concepts.pipeline",
         "CommonalityDetectionConceptFormationProcessor", "class"),
        ("ore_commonality_concepts.pipeline",
         "CommonalityDetectionConceptFormationProcessor.process", "function"),
        ("ore_commonality_concepts.functional", "estimate_feature_statistics", "function"),
        ("ore_commonality_concepts.biology", "DistributedCorticalConceptNetwork", "class"),
        ("ore_commonality_concepts.biology",
         "DistributedCorticalConceptNetwork.learn", "function"),
        ("ore_commonality_concepts.contracts", "EpisodeInput", "class"),
        ("ore_commonality_concepts.registry", "OperationSpec", "class"),
    )
    for module, qualname, kind in expected:
        row = database.get_annotation(module, qualname, kind)
        assert row is not None and row["annotation"].strip()
        assert row["annotation_source"] == "docstring"
    assert database.summary()["symbols"] >= 45


def test_processor_records_sixteen_sqlite_operation_events(tmp_path: Path) -> None:
    """Verify a full Point 06 run stores one annotated SQLite event per ordered operation."""
    database_path = tmp_path / "ore_point06_run.sqlite"
    root = Path(__file__).resolve().parents[1]
    processor = CommonalityDetectionConceptFormationProcessor(
        ConceptLearningConfig(cortical_units=8),
        database_path=database_path,
        repository_root=root,
    )
    result = processor.process(_request())
    assert result.logger["enabled"] is True
    database = LoggerDatabase(database_path)
    runs = database.list_runs()
    assert len(runs) == 1 and runs[0]["status"] == "completed"
    events = database.list_events(runs[0]["run_id"])
    operations = [event for event in events if event["event_kind"] == "operation"]
    assert len(operations) == 16
    assert all(event.get("annotation") for event in operations)



def test_labelled_feedback_is_idempotent_and_cannot_be_reused_with_changed_content() -> None:
    """Do not relearn identical feedback and reject conflicting reuse of a feedback episode ID."""
    target, comparison = _bird_examples()
    feedback = EpisodeInput(
        "feedback-member",
        {"feathers": 1.0, "wings": 1.0, "beak": 1.0},
        category_label=True,
    )
    processor = CommonalityDetectionConceptFormationProcessor(
        ConceptLearningConfig(cortical_units=8)
    )
    first = processor.process(
        _request(target=target, comparison=comparison, predictions=[feedback])
    )
    updates = first.biological["distributed_cortical_network"]["learning_update_count"]
    second = processor.process(
        _request(target=target, comparison=comparison, predictions=[feedback])
    )
    assert second.mismatch_and_revision["feedback_added"] == 0
    assert len(second.concept["supporting_episode_ids"]) == 4
    assert second.biological["distributed_cortical_network"]["learning_update_count"] == updates

    changed = EpisodeInput(
        "feedback-member",
        {"feathers": 0.0, "wings": 0.0, "beak": 0.0},
        category_label=True,
    )
    with pytest.raises(ValueError, match="different prediction content"):
        processor.process(
            _request(target=target, comparison=comparison, predictions=[changed])
        )



def test_corticostriatal_gate_requires_labelled_feedback_and_scales_with_error() -> None:
    """Gate cortical learning only when a labelled outcome is available and error is informative."""
    network = CorticostriatalFeedbackNetwork()
    absent = network.compute_learning_gate(0.9, 1.0, feedback_available=False)
    low_error = network.compute_learning_gate(0.05, 1.0, feedback_available=True)
    high_error = network.compute_learning_gate(0.9, 1.0, feedback_available=True)
    assert absent["learning_gate"] == 0.0
    assert high_error["learning_gate"] > low_error["learning_gate"]
    assert high_error["learning_gate"] <= 1.0


def test_unlabelled_mismatch_does_not_create_a_corticostriatal_teaching_signal() -> None:
    """Keep prediction surprise distinct from reward or explicit category feedback."""
    target, comparison = _bird_examples()
    unlabelled = EpisodeInput(
        "unlabelled-surprise",
        {"feathers": 0.0, "wings": 1.0, "beak": 0.0},
    )
    processor = CommonalityDetectionConceptFormationProcessor(
        ConceptLearningConfig(cortical_units=8)
    )
    result = processor.process(
        _request(target=target, comparison=comparison, predictions=[unlabelled])
    )
    gate = result.operation_trace[13]["functional_output"]["corticostriatal_feedback_gate"]
    assert gate["labelled_feedback_count"] == 0
    assert gate["mean_learning_gate"] == 0.0
    assert gate["unlabelled_mismatch_is_not_a_teaching_signal"] is True
