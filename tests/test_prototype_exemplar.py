"""Tests for Point 07 prototype/exemplar representations and their SQLite annotations."""
from pathlib import Path

import pytest

from ore_prototype_exemplar import (
    EpisodeInput,
    LearningRequest,
    OPERATION_REGISTRY,
    PrototypeExemplarConfig,
    PrototypeExemplarProcessor,
    register_annotations,
)
from ore_logger.database import LoggerDatabase


def _bird_and_aircraft_examples():
    birds = [
        EpisodeInput("sparrow", {"feathers": True, "wings": True, "beak": True,
                                  "metal": False, "mass": 0.15, "colour": "brown"},
                     context={"habitat": "garden"}, relations=(("wings", "part_of", "bird"),)),
        EpisodeInput("robin", {"feathers": True, "wings": True, "beak": True,
                                "metal": False, "mass": 0.22, "colour": "red"},
                     context={"habitat": "woodland"}, relations=(("wings", "part_of", "bird"),)),
        EpisodeInput("penguin", {"feathers": True, "wings": True, "beak": True,
                                  "metal": False, "mass": 0.70, "colour": "black"},
                     context={"habitat": "coast"}, relations=(("wings", "part_of", "bird"),)),
    ]
    aircraft = [
        EpisodeInput("glider", {"feathers": False, "wings": True, "beak": False,
                                 "metal": False, "mass": 70.0, "colour": "white"},
                     relations=(("wings", "part_of", "aircraft"),)),
        EpisodeInput("jet", {"feathers": False, "wings": True, "beak": False,
                              "metal": True, "mass": 40000.0, "colour": "white"},
                     relations=(("wings", "part_of", "aircraft"),)),
        EpisodeInput("helicopter", {"feathers": False, "wings": False, "beak": False,
                                     "metal": True, "mass": 2500.0, "colour": "grey"}),
    ]
    return birds, aircraft


def test_registry_has_sixteen_source_ordered_operations():
    """Expose all sixteen Point 07 algorithm steps in the correct source sequence."""
    assert len(OPERATION_REGISTRY) == 16
    assert [row.operation_id for row in OPERATION_REGISTRY] == [
        f"PE{i:02d}" for i in range(1, 17)
    ]
    assert all(row.mathematical_form and row.biological_reference for row in OPERATION_REGISTRY)


def test_training_keeps_prototype_and_exemplar_representations_separate():
    """Summarize recurring feature statistics while preserving episode-specific identity and context."""
    birds, aircraft = _bird_and_aircraft_examples()
    processor = PrototypeExemplarProcessor(PrototypeExemplarConfig(cortical_units=12))
    result = processor.process(LearningRequest({"bird": birds, "aircraft": aircraft}))
    bird = result.categories["bird"]
    assert bird["prototype"]["categorical_distribution"]["feathers"]
    assert bird["exemplar_count"] == 3
    assert set(bird["exemplar_ids"]) == {"sparrow", "robin", "penguin"}
    assert "mass" in bird["prototype"]["numeric_mean"]
    assert result.biological["hippocampal_episode_network"]["episode_identity_preserved"]
    assert result.biological["distributed_cortical_network"]["learning_update_count"] == 6
    assert len(result.operation_trace) == 6 * 16
    assert all(row["explanation"] and row["mathematical_form"] for row in result.operation_trace)


def test_hybrid_classifier_compares_both_pathways_and_can_generalize():
    """Classify a held-out typical bird and expose prototype-, exemplar-, and hybrid-only evidence."""
    birds, aircraft = _bird_and_aircraft_examples()
    held_out = EpisodeInput("new-robin", {"feathers": True, "wings": True, "beak": True,
                                           "metal": False, "mass": 0.20, "colour": "red"})
    processor = PrototypeExemplarProcessor(PrototypeExemplarConfig(cortical_units=12))
    result = processor.process(LearningRequest(
        {"bird": birds, "aircraft": aircraft}, prediction_episodes=(held_out,)
    ))
    prediction = result.predictions[0]
    assert prediction["predicted_category"] == "bird"
    assert set(prediction["prototype_only_probabilities"]) == {"bird", "aircraft"}
    assert set(prediction["exemplar_only_probabilities"]) == {"bird", "aircraft"}
    assert set(prediction["hybrid_probabilities"]) == {"bird", "aircraft"}
    assert len(result.operation_trace) == 7 * 16


def test_distinctive_exception_is_not_automatically_rejected_or_averaged_away():
    """Retain an unusual labelled member as a protected exemplar even when it is far from the mean."""
    birds, aircraft = _bird_and_aircraft_examples()
    unusual = EpisodeInput("large-flightless-bird", {
        "feathers": True, "wings": True, "beak": True, "metal": False,
        "mass": 0.95, "colour": "blue",
    }, context={"habitat": "island"})
    processor = PrototypeExemplarProcessor(PrototypeExemplarConfig(
        cortical_units=12, outlier_threshold=0.8, exception_similarity_threshold=0.25,
    ))
    processor.fit({"bird": birds, "aircraft": aircraft})
    result = processor.process(LearningRequest(
        {"bird": [unusual]}, task_context="exception_sensitive_category_learning"
    ))
    state = processor.categories["bird"]
    assert "large-flightless-bird" in state.exemplars
    assert state.exemplars["large-flightless-bird"].is_exception is True
    assert state.prototype.update_count == 4
    assert result.operation_trace[12]["functional_output"][
        "unusual_member_not_automatically_rejected"
    ] is True


def test_explicit_feedback_is_applied_after_prediction_and_logs_gate():
    """Keep expected labels out of classification until prediction is recorded, then update memory."""
    birds, aircraft = _bird_and_aircraft_examples()
    query = EpisodeInput("ambiguous-case", {"feathers": True, "wings": True,
                                            "beak": True, "metal": False, "mass": 0.3})
    processor = PrototypeExemplarProcessor(PrototypeExemplarConfig(cortical_units=12))
    result = processor.process(LearningRequest(
        {"bird": birds, "aircraft": aircraft}, prediction_episodes=(query,),
        feedback_labels={"ambiguous-case": "bird"},
    ))
    feedback_row = [row for row in result.operation_trace
                    if row["episode_id"] == "ambiguous-case"]
    assert result.predictions[0]["candidate_category"] in {"bird", "aircraft"}
    step14 = next(row for row in feedback_row if row["step"] == 14)
    assert step14["functional_output"]["feedback_available"] is True
    assert step14["functional_output"]["learning_gate"] > 0.0
    assert "ambiguous-case" in processor.categories["bird"].exemplars


def test_missing_feature_is_not_confused_with_observed_false():
    """Exclude None from a prototype's category frequencies but retain observed False evidence."""
    first = EpisodeInput("a", {"wings": True, "feathers": False, "beak": None})
    second = EpisodeInput("b", {"wings": True, "feathers": True, "beak": True})
    processor = PrototypeExemplarProcessor(PrototypeExemplarConfig(cortical_units=8))
    processor.fit({"bird": [first, second]})
    prototype = processor.categories["bird"].prototype
    assert prototype.feature_observations["feathers"] == 2
    assert "beak" not in prototype.feature_observations or prototype.feature_observations["beak"] == 1
    assert prototype.categorical_counts["feathers"]["bool:False"] == 1.0


def test_repeated_training_is_idempotent_and_conflicting_episode_content_is_rejected():
    """Do not duplicate an exemplar when the same episode is replayed and reject altered reuse."""
    birds, _ = _bird_and_aircraft_examples()
    processor = PrototypeExemplarProcessor(PrototypeExemplarConfig(cortical_units=8))
    processor.fit({"bird": birds})
    processor.fit({"bird": birds})
    assert len(processor.categories["bird"].exemplars) == 3
    changed = EpisodeInput("sparrow", {"feathers": False, "wings": False})
    with pytest.raises(ValueError, match="different content"):
        processor.fit({"bird": [changed]})


def test_sqlite_records_all_step_events_and_docstring_annotations(tmp_path: Path):
    """Register every package class/function and record its operation events in SQLite."""
    birds, aircraft = _bird_and_aircraft_examples()
    database_path = tmp_path / "point07.sqlite3"
    root = Path(__file__).resolve().parents[1]
    report = register_annotations(database_path, root)
    assert report["parse_errors"] == 0
    database = LoggerDatabase(database_path)
    assert database.get_annotation("ore_prototype_exemplar.pipeline",
                                   "PrototypeExemplarProcessor", "class")
    assert database.get_annotation("ore_prototype_exemplar.pipeline",
                                   "PrototypeExemplarProcessor.process", "function")
    assert database.get_annotation("ore_prototype_exemplar.functional",
                                   "prototype_similarity", "function")
    processor = PrototypeExemplarProcessor(database_path=database_path, repository_root=root,
                                           config=PrototypeExemplarConfig(cortical_units=8))
    result = processor.process(LearningRequest({"bird": birds, "aircraft": aircraft}))
    assert result.logger["enabled"] is True
    runs = database.list_runs()
    assert len(runs) == 1 and runs[0]["status"] == "completed"
    events = database.list_events(runs[0]["run_id"])
    operations = [row for row in events if row["event_kind"] == "operation"]
    assert len(operations) == 6 * 16
    assert all(row.get("annotation") for row in operations)


def test_unknown_only_input_is_rejected_and_classifier_requires_training():
    """Guard invalid empty evidence and prevent classification without learned categories."""
    with pytest.raises(ValueError, match="at least one candidate feature"):
        EpisodeInput("empty", {})
    processor = PrototypeExemplarProcessor()
    with pytest.raises(RuntimeError, match="fit at least one category"):
        processor.classify(EpisodeInput("query", {"feature": 1.0}))
