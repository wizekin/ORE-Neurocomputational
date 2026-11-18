"""Functional, dual-store, provenance, uncertainty, biological, and SQLite tests for Point 04."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from ore_memory_transformation import (
    EpisodeInput,
    EpisodicToSemanticTransformationProcessor,
    MemoryTransformationConfig,
    MemoryTransformationRequest,
    OPERATION_REGISTRY,
    RelationEvidence,
    register_annotations,
)
from ore_memory_transformation.functional import (
    accessibility_decay,
    estimate_feature_statistics,
    information_gain,
    weighted_feature_similarity,
)


def _bird(
    episode_id: str,
    location: str,
    time_of_day: str,
    *,
    source_type: str = "direct_experience",
    concept_hint: str = "bird",
    habitat_importance: float = 0.1,
) -> EpisodeInput:
    """Build a bird episode with common semantic features and distinct context."""
    return EpisodeInput(
        episode_id=episode_id,
        concept_hint=concept_hint,
        content_features={
            "feathers": 1.0,
            "beak": 1.0,
            "animal_bird": 1.0,
            "sings": 0.8,
            "red_marking": 0.0 if episode_id.endswith("a") else 0.2,
        },
        spatial_features={
            "habitat": 0.8,
            f"location_{location}": 1.0,
        },
        temporal_features={f"time_{time_of_day}": 1.0},
        sensory_features={"birdsong": 0.9, "visible_wings": 1.0},
        affective_features={"pleasant": 0.6},
        causal_features={"wing_motion_supports_flight": 0.9},
        relations=(
            RelationEvidence("bird", "has_property", "feathers", 0.95, source_type),
            RelationEvidence("bird", "has_part", "beak", 0.95, source_type),
            RelationEvidence("bird", "can", "sing", 0.8, source_type),
        ),
        feature_confidence={"content": {"feathers": 0.95, "beak": 0.98}},
        context_importance={"spatial": {"habitat": habitat_importance}},
        source_type=source_type,
        source_id=f"{source_type}-source",
        source_confidence=0.95,
        overall_confidence=0.95,
        timestamp_ms=1000.0,
    )


def _fish(episode_id: str = "fish-1") -> EpisodeInput:
    """Build a contrastive non-bird episode with explicit counterevidence."""
    return EpisodeInput(
        episode_id=episode_id,
        concept_hint="fish",
        content_features={
            "feathers": 0.0,
            "beak": 0.0,
            "animal_bird": 0.0,
            "fins": 1.0,
            "gills": 1.0,
        },
        spatial_features={"underwater": 1.0},
        sensory_features={"visible_fins": 1.0},
        relations=(
            RelationEvidence("fish", "has_part", "fins"),
            RelationEvidence("fish", "uses", "gills"),
        ),
        source_type="direct_experience",
        source_confidence=0.95,
        overall_confidence=0.95,
    )


def _transform(
    processor: EpisodicToSemanticTransformationProcessor,
    request_id: str = "bird-transform",
    episodes: tuple[EpisodeInput, ...] = (),
    **kwargs: object,
):
    """Run one deterministic memory transformation request."""
    return processor.process(MemoryTransformationRequest(
        request_id=request_id, episodes=episodes, **kwargs
    ))


def test_registry_has_all_sixteen_source_ordered_operations() -> None:
    """Protect the exact source order and require equations and biological interpretation."""
    assert len(OPERATION_REGISTRY) == 16
    assert [row.operation_id for row in OPERATION_REGISTRY] == [
        f"ES{index:02d}" for index in range(1, 17)
    ]
    assert all(row.mathematical_form and row.biological_reference and row.evidence_level
               for row in OPERATION_REGISTRY)


def test_contracts_preserve_unknown_separately_from_explicit_zero() -> None:
    """Verify absent feature keys remain unknown while supplied zeroes are valid observations."""
    episode = EpisodeInput(
        episode_id="known-zero",
        content_features={"present": 1.0, "observed_absent": 0.0},
    )
    groups = episode.feature_groups()
    assert groups["content"]["observed_absent"] == 0.0
    assert "not_observed" not in groups["content"]
    with pytest.raises(ValueError):
        EpisodeInput(episode_id="bad", content_features={"invalid": 1.1})
    with pytest.raises(ValueError):
        EpisodeInput(episode_id="bad-source", source_type="guessed_as_fact")


def test_information_gain_and_reliability_statistics_are_contrast_sensitive() -> None:
    """A discriminating feature should carry more information than a feature shared by every class."""
    assert information_gain([1, 1, 0, 0], ["bird", "bird", "fish", "fish"]) > 0.99
    assert information_gain([1, 1, 1, 1], ["bird", "bird", "fish", "fish"]) == 0.0
    target = (_bird("bird-a", "park", "morning"),
              _bird("bird-b", "garden", "evening"))
    contrast = (_fish("fish-a"),)
    config = MemoryTransformationConfig()
    stats = estimate_feature_statistics(
        target, ["bird", "bird"], config, contrast, ["fish"]
    )
    assert stats["content:feathers"]["is_invariant"] is True
    assert stats["content:feathers"]["diagnosticity"] > 0.9
    assert stats["content:red_marking"]["observed_episode_count"] == 2


def test_similarity_does_not_treat_missing_features_as_explicit_absence() -> None:
    """Compare common observations only, while retaining explicit zero-valued contrasts."""
    first = {"content": {"feathers": 1.0}}
    missing = {"content": {}}
    observed_zero = {"content": {"feathers": 0.0}}
    assert weighted_feature_similarity(first, missing) == 0.0
    assert weighted_feature_similarity(first, observed_zero) == 0.0


def test_transform_reactivates_events_and_runs_all_sixteen_stages() -> None:
    """Run a cue-targeted event and check the complete source-ordered processing trace."""
    processor = EpisodicToSemanticTransformationProcessor()
    result = _transform(
        processor, episodes=(
            _bird("bird-a", "park", "morning"),
            _bird("bird-b", "garden", "evening"),
            _bird("bird-c", "woodland", "noon"),
            _fish(),
        ),
        target_episode_id="bird-a",
        retrieval_type="mixed",
        cue_features={"content": {"feathers": 1.0, "beak": 1.0}},
    )
    assert len(result.operation_trace) == 16
    assert [row["operation_id"] for row in result.operation_trace] == [
        row.operation_id for row in OPERATION_REGISTRY
    ]
    assert result.selected_episode_ids == ["bird-a"]
    assert set(result.related_episode_ids) == {"bird-a", "bird-b", "bird-c"}
    assert "bird" in result.concept_ids[0]
    assert result.retrieved_result["episodic_detail"]["episode_id"] == "bird-a"
    assert result.retrieved_result["semantic_knowledge"] is not None
    assert result.biological["semantic_and_episodic_tracks_coexist"] is True


def test_gist_retains_recurring_meaning_but_drops_incidental_one_off_context() -> None:
    """Shared semantic properties generalize while unique location and time remain episode details."""
    processor = EpisodicToSemanticTransformationProcessor()
    result = _transform(
        processor,
        episodes=(
            _bird("bird-a", "park", "morning"),
            _bird("bird-b", "garden", "evening"),
            _bird("bird-c", "woodland", "noon"),
        ),
        target_episode_id="bird-a",
        retrieval_type="mixed",
    )
    gist = result.semantic_representations[0]
    assert gist["features"]["content"]["feathers"] == pytest.approx(1.0)
    assert gist["features"]["content"]["beak"] == pytest.approx(1.0)
    assert "location_park" not in gist["features"]["spatial"]
    assert "time_morning" not in gist["features"]["temporal"]
    assert "location_park" in result.episodic_representation["event_records"]["bird-a"]["episode"]["spatial_features"]
    assert gist["context_weights"]["spatial:habitat"] < 0.6


def test_explicitly_important_context_is_retained_more_than_incidental_context() -> None:
    """Context with high diagnostic/retrieval importance receives a larger semantic weight."""
    low = EpisodicToSemanticTransformationProcessor()
    high = EpisodicToSemanticTransformationProcessor()
    low_result = _transform(
        low, "low-context",
        episodes=(_bird("low-1", "park", "morning", habitat_importance=0.0),
                  _bird("low-2", "garden", "evening", habitat_importance=0.0)),
        target_episode_id="low-1",
    )
    high_result = _transform(
        high, "high-context",
        episodes=(_bird("high-1", "park", "morning", habitat_importance=1.0),
                  _bird("high-2", "garden", "evening", habitat_importance=1.0)),
        target_episode_id="high-1",
    )
    low_weight = low_result.semantic_representations[0]["context_weights"]["spatial:habitat"]
    high_weight = high_result.semantic_representations[0]["context_weights"]["spatial:habitat"]
    assert high_weight > low_weight


def test_dual_store_preserves_event_specific_bindings_while_building_semantics() -> None:
    """Semantic integration must never replace original event details or relation bindings."""
    processor = EpisodicToSemanticTransformationProcessor()
    episode_a = _bird("bird-a", "park", "morning")
    episode_b = _bird("bird-b", "garden", "evening")
    result = _transform(processor, episodes=(episode_a, episode_b), target_episode_id="bird-a")
    stored = result.episodic_representation["event_records"]["bird-a"]["episode"]
    assert stored["spatial_features"]["location_park"] == 1.0
    assert stored["relations"][0]["subject"] == "bird"
    assert result.semantic_representations[0]["features"]["content"]["feathers"] == 1.0
    assert result.biological["episodic_trace_erasure_inferred_from_low_accessibility"] is False


def test_recurring_relations_are_generalized_with_supporting_episode_ids() -> None:
    """Common typed relations should enter the gist while one-off relations remain episodic."""
    processor = EpisodicToSemanticTransformationProcessor()
    first = _bird("bird-a", "park", "morning")
    second = _bird("bird-b", "garden", "evening")
    one_off = EpisodeInput(
        episode_id="bird-c", concept_hint="bird",
        content_features={"feathers": 1.0, "beak": 1.0},
        relations=(RelationEvidence("bird", "nested_in", "old_oak"),),
    )
    result = _transform(processor, episodes=(first, second, one_off), target_episode_id="bird-a")
    gist = result.semantic_representations[0]
    assert "bird|has_property|feathers" in gist["relations"]
    assert "bird|nested_in|old_oak" not in gist["relations"]
    assert "bird-c" in gist["exceptions"].get("content:red_marking", {}).get(
        "episode_evidence", {}
    ) or "bird-c" in result.related_episode_ids


def test_contrastive_concepts_are_not_automatically_merged() -> None:
    """Shared population similarity may create a schema link but never erases concept identities."""
    processor = EpisodicToSemanticTransformationProcessor()
    result = _transform(
        processor, episodes=(
            _bird("bird-a", "park", "morning"),
            _bird("bird-b", "garden", "evening"),
            _fish("fish-a"),
            _fish("fish-b"),
        ),
        target_episode_id="bird-a",
    )
    assert processor.episode_concept_ids["bird-a"] != processor.episode_concept_ids["fish-a"]
    assert set(result.concept_ids) == {
        processor.episode_concept_ids["bird-a"],
        processor.episode_concept_ids["fish-a"],
    }
    assert all(link["identities_merged"] is False for link in result.schema_links)


def test_similarity_cluster_can_generalize_unhinted_related_examples() -> None:
    """Repeated high-overlap events without labels may be assigned to one provisional cluster."""
    processor = EpisodicToSemanticTransformationProcessor()
    first = EpisodeInput(
        episode_id="unhinted-1",
        content_features={"feathers": 1.0, "beak": 1.0, "sings": 0.9},
        relations=(RelationEvidence("bird", "has_part", "beak"),),
    )
    second = EpisodeInput(
        episode_id="unhinted-2",
        content_features={"feathers": 1.0, "beak": 1.0, "sings": 0.8},
        spatial_features={"different_place": 1.0},
        relations=(RelationEvidence("bird", "has_part", "beak"),),
    )
    processor.add_episode(first)
    processor.add_episode(second)
    assert processor.episode_concept_ids[first.episode_id] == processor.episode_concept_ids[second.episode_id]


def test_episode_accessibility_decays_without_deleting_original_content() -> None:
    """Spatial/temporal access may decline over time while original feature records remain intact."""
    processor = EpisodicToSemanticTransformationProcessor()
    initial = _transform(
        processor, "initial",
        episodes=(_bird("bird-a", "park", "morning"),
                  _bird("bird-b", "garden", "evening")),
        target_episode_id="bird-a",
        now_ms=0.0,
        reactivation_gain=0.0,
    )
    before = initial.accessibility["per_episode"]["bird-a"]["after"]
    later = _transform(
        processor, "one-year-later",
        target_episode_id="bird-a",
        now_ms=365.0 * 24.0 * 60.0 * 60.0 * 1000.0,
        reactivation_gain=0.0,
    )
    after = later.accessibility["per_episode"]["bird-a"]["after"]
    assert after["spatial:location_park"] < before["spatial:location_park"]
    assert after["content:feathers"] < before["content:feathers"]
    assert after["spatial:location_park"] < after["content:feathers"]
    saved = later.episodic_representation["event_records"]["bird-a"]["episode"]
    assert saved["spatial_features"]["location_park"] == 1.0
    assert later.accessibility["trace_content_deleted"] is False


def test_reactivation_boosts_accessibility_without_mutating_event_content() -> None:
    """A retrieval opportunity improves modeled access but keeps stored episode details unchanged."""
    processor = EpisodicToSemanticTransformationProcessor()
    _transform(processor, "add", episodes=(_bird("bird-a", "park", "morning"),),
               target_episode_id="bird-a", now_ms=10_000.0, reactivation_gain=0.0)
    before_episode = processor.episodic_store["bird-a"].episode.to_dict()
    before_access = processor.episodic_store["bird-a"].feature_accessibility["content:feathers"]
    result = _transform(processor, "reactivate", target_episode_id="bird-a",
                        now_ms=10_000.0, reactivation_gain=1.0)
    after_access = processor.episodic_store["bird-a"].feature_accessibility["content:feathers"]
    assert after_access > before_access
    assert processor.episodic_store["bird-a"].episode.to_dict() == before_episode
    assert result.retrieved_result["episodic_detail"]["episode_id"] == "bird-a"


def test_query_type_routes_to_distinct_memory_tracks() -> None:
    """Episodic, semantic, and mixed goals must use the requested representation, not one universal route."""
    processor = EpisodicToSemanticTransformationProcessor()
    episodes = (_bird("bird-a", "park", "morning"), _bird("bird-b", "garden", "evening"))
    _transform(processor, "train", episodes=episodes, target_episode_id="bird-a")
    episodic = _transform(processor, "episodic", target_episode_id="bird-a", retrieval_type="episodic")
    semantic = _transform(processor, "semantic", target_concept_id="hint:bird",
                          retrieval_type="semantic")
    mixed = _transform(processor, "mixed", target_episode_id="bird-b", retrieval_type="mixed")
    assert episodic.retrieved_result["episodic_detail"] is not None
    assert episodic.retrieved_result["semantic_knowledge"] is None
    assert semantic.retrieved_result["episodic_detail"] is None
    assert semantic.retrieved_result["semantic_knowledge"] is not None
    assert mixed.retrieved_result["episodic_detail"] is not None
    assert mixed.retrieved_result["semantic_knowledge"] is not None


def test_source_monitoring_keeps_inference_and_testimony_distinct_from_experience() -> None:
    """Do not relabel an inferred or reported fact as a directly experienced episode detail."""
    processor = EpisodicToSemanticTransformationProcessor()
    testimony = EpisodeInput(
        episode_id="reported-bird",
        concept_hint="bird",
        content_features={"feathers": 1.0},
        source_type="testimony",
        source_id="witness-7",
        source_confidence=0.8,
    )
    inferred = EpisodeInput(
        episode_id="inferred-bird",
        concept_hint="bird",
        content_features={"beak": 1.0},
        source_type="inference",
        source_id="reasoning",
        source_confidence=0.5,
    )
    result = _transform(processor, "source-check", episodes=(testimony, inferred),
                        target_episode_id="reported-bird", retrieval_type="mixed")
    episodic = result.retrieved_result["provenance"]["episodic_features"]
    assert episodic["content:feathers"]["source_type"] == "testimony"
    assert episodic["content:feathers"]["source_id"] == "witness-7"
    semantic = result.retrieved_result["provenance"]["semantic_features"]
    assert semantic["content:feathers"]["status"] in {
        "generalized_across_multiple_episodes", "single_non_direct_source"
    }


def test_new_evidence_refines_semantic_support_but_does_not_rewrite_prior_events() -> None:
    """Incremental learning adds a new episode and recomputes gist from evidence without editing old records."""
    processor = EpisodicToSemanticTransformationProcessor()
    first = _bird("bird-a", "park", "morning")
    second = _bird("bird-b", "garden", "evening")
    _transform(processor, "initial", episodes=(first, second), target_episode_id="bird-a")
    old_record = processor.episodic_store["bird-a"].episode.to_dict()
    later = _transform(processor, "later", episodes=(_bird("bird-c", "woodland", "noon"),),
                       target_episode_id="bird-c")
    gist = later.semantic_representations[0]
    assert set(gist["supporting_episode_ids"]) == {"bird-a", "bird-b", "bird-c"}
    assert processor.episodic_store["bird-a"].episode.to_dict() == old_record
    assert gist["features"]["content"]["feathers"] == pytest.approx(1.0)


def test_duplicate_episode_id_is_idempotent_but_conflicting_content_is_rejected() -> None:
    """Resubmitting identical event content does not double-count evidence or overwrite the trace."""
    processor = EpisodicToSemanticTransformationProcessor()
    first = _bird("bird-a", "park", "morning")
    processor.add_episode(first)
    duplicate = processor.add_episode(first)
    assert duplicate["status"] == "already_registered"
    changed = _bird("bird-a", "desert", "night")
    with pytest.raises(ValueError):
        processor.add_episode(changed)
    assert len(processor.episodic_store) == 1


def test_held_out_gist_evaluation_excludes_the_episode_from_its_prototype() -> None:
    """Leave-one-out generalization evaluates new examples without self-matching."""
    from ore_memory_transformation.functional import leave_one_out_gist_similarity

    episodes = (_bird("bird-a", "park", "morning"),
                _bird("bird-b", "garden", "evening"),
                _bird("bird-c", "woodland", "noon"))
    report = leave_one_out_gist_similarity(episodes)
    assert report["status"] == "leave_one_episode_out"
    assert set(report["scores"]) == {"bird-a", "bird-b", "bird-c"}
    assert 0.0 <= report["mean_similarity"] <= 1.0


def test_brier_calibration_is_not_fabricated_without_external_outcomes() -> None:
    """Quality evaluation must report calibration as unmeasured rather than inventing correctness labels."""
    processor = EpisodicToSemanticTransformationProcessor()
    result = _transform(processor, episodes=(_bird("bird-a", "park", "morning"),),
                        target_episode_id="bird-a")
    assert result.quality["confidence_brier_score"] is None
    assert result.quality["calibration_status"] == "not_measured_no_external_outcomes"


def test_every_package_class_and_function_has_a_docstring() -> None:
    """Require explanatory docstrings for all package classes, functions, and nested helpers."""
    package_root = Path(__file__).resolve().parents[1] / "ore_memory_transformation"
    for source_path in package_root.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                assert ast.get_docstring(node), f"missing docstring: {source_path.name}:{node.name}"


def test_package_annotations_are_indexed_in_logger_sqlite(tmp_path: Path) -> None:
    """Index public and nested Point 04 symbols as docstring annotations in SQLite."""
    from ore_logger.database import LoggerDatabase

    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "ore_point04_annotations.sqlite"
    report = register_annotations(database_path, root)
    assert report["parse_errors"] == 0
    database = LoggerDatabase(database_path)
    expected = (
        ("ore_memory_transformation.pipeline", "EpisodicToSemanticTransformationProcessor", "class"),
        ("ore_memory_transformation.pipeline", "EpisodicToSemanticTransformationProcessor.process", "function"),
        ("ore_memory_transformation.pipeline", "EpisodicToSemanticTransformationProcessor.process.<locals>.record", "function"),
        ("ore_memory_transformation.biology", "HippocampalIndexNetwork.complete", "function"),
        ("ore_memory_transformation.biology", "DistributedSemanticNetwork.integrate", "function"),
        ("ore_memory_transformation.contracts", "EpisodeInput", "class"),
        ("ore_memory_transformation.registry", "OperationSpec", "class"),
    )
    for module, qualname, kind in expected:
        row = database.get_annotation(module, qualname, kind)
        assert row is not None and row["annotation"].strip()
        assert row["annotation_source"] == "docstring"
    assert database.summary()["symbols"] >= 40


def test_processor_records_sixteen_operations_in_logger_sqlite(tmp_path: Path) -> None:
    """Persist exactly 16 source-ordered operation events and completed run status."""
    from ore_logger.database import LoggerDatabase

    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "ore_point04_run.sqlite"
    processor = EpisodicToSemanticTransformationProcessor(
        database_path=database_path, repository_root=root
    )
    result = processor.process(MemoryTransformationRequest(
        request_id="sqlite-test",
        episodes=(_bird("bird-a", "park", "morning"), _bird("bird-b", "garden", "evening")),
        target_episode_id="bird-a",
    ))
    assert result.logger["enabled"] is True
    database = LoggerDatabase(database_path)
    runs = database.list_runs()
    assert len(runs) == 1 and runs[0]["status"] == "completed"
    events = database.list_events(runs[0]["run_id"])
    operations = [event for event in events if event["event_kind"] == "operation"]
    assert len(operations) == 16
    assert all(event.get("annotation") for event in operations)
