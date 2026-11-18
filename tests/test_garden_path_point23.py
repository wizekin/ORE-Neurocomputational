"""Acceptance tests for Point 23 incremental parsing and garden-path recovery."""
from __future__ import annotations
import sqlite3
from pathlib import Path
import pytest
from ore_garden_path import (
    LexicalCandidate, ParseChoicePoint, ParseDependency, ParseHypothesis,
    Point23Config, Point23GardenPathProcessor, Point23State, SentenceRequest,
    record_result_to_sqlite, register_annotations,
)

def make_lexicon():
    """Provide lexical-category alternatives for the demo ambiguity."""
    return {
        "while": (LexicalCandidate("subordinator"),),
        "the": (LexicalCandidate("determiner"),),
        "man": (LexicalCandidate("noun", prior=0.9), LexicalCandidate("verb", prior=0.1)),
        "hunted": (LexicalCandidate("verb"),),
        "deer": (LexicalCandidate("noun"),),
        "ran": (LexicalCandidate("verb"),),
        "into": (LexicalCandidate("preposition"),),
        "woods": (LexicalCandidate("noun"),),
    }

def make_garden_path_request(**overrides):
    """Create an initially preferred object attachment contradicted by a later verb."""
    sentence = "While the man hunted the deer ran into the woods."
    categories = {
        0:"subordinator", 1:"determiner", 2:"noun", 3:"verb", 4:"determiner",
        5:"noun", 6:"verb", 7:"preposition", 8:"determiner", 9:"noun",
    }
    initial = ParseHypothesis(
        parse_id="deer-object",
        description="The deer is the object of hunted.",
        prior=0.85, token_categories=categories,
        structural_likelihoods={6:0.01}, semantic_plausibility=0.95,
        dependencies=(ParseDependency(3,2,"subject","agent"),
                      ParseDependency(3,5,"direct_object","patient")),
        thematic_roles={"hunted.agent":2,"hunted.patient":5},
        choice_points=(ParseChoicePoint("deer-role",5,"object-of-hunted"),),
    )
    revised = ParseHypothesis(
        parse_id="deer-matrix-subject",
        description="The deer is the subject of ran after the subordinate clause.",
        prior=0.15, token_categories=categories,
        structural_likelihoods={6:0.99}, semantic_plausibility=0.95,
        dependencies=(ParseDependency(3,2,"subject","agent"),
                      ParseDependency(6,5,"subject","agent")),
        thematic_roles={"hunted.agent":2,"ran.agent":5},
        choice_points=(ParseChoicePoint("deer-role",5,"subject-of-ran"),),
    )
    values = {
        "sentence_id":"while-man-hunted-deer",
        "text":sentence,
        "lexicon":make_lexicon(),
        "parse_hypotheses":(initial,revised),
        "task_goal":"precise",
        "construction_id":"while-subordinate-clause",
    }
    values.update(overrides)
    return SentenceRequest(**values)

def test_pipeline_executes_seventeen_ordered_operations_with_two_tracks():
    """Every source stage creates an aligned functional and biological-reference trace."""
    result = Point23GardenPathProcessor().process(make_garden_path_request())
    assert len(result.operation_trace) == 17
    assert [row["operation_id"] for row in result.operation_trace] == [
        f"P23-{i:02d}" for i in range(1,18)
    ]
    assert all(row["status"] == "completed" for row in result.operation_trace)
    assert all("functional_output" in row and "biological_output" in row
               for row in result.operation_trace)
    assert result.diagnostics["validation"]["operation_order_verified"] is True

def test_garden_path_revision_changes_thematic_roles():
    """The deer first fills the object slot, then becomes the subject of the matrix verb."""
    result = Point23GardenPathProcessor().process(make_garden_path_request())
    assert result.initial_parse_id == "deer-object"
    assert result.final_parse_id == "deer-matrix-subject"
    assert result.revisions[0]["revision_attempted"] is True
    assert result.revisions[0]["revision_succeeded"] is True
    assert result.revisions[0]["earliest_revision_index"] == 5
    assert "hunted.patient" not in result.thematic_roles
    assert result.thematic_roles["ran.agent"]["token"] == "deer"
    assert result.interpretation_status == "full"

def test_surprisal_and_conflict_peak_at_disambiguating_verb():
    """A low-likelihood continuation produces a cost peak and parse-posterior shift."""
    result = Point23GardenPathProcessor().process(make_garden_path_request())
    rows = result.diagnostics["structural_expectations"]["token_trace"]
    ran = next(row for row in rows if row["token"] == "ran")
    hunted = next(row for row in rows if row["token"] == "hunted")
    assert ran["surprisal"] > hunted["surprisal"]
    assert result.diagnostics["conflict"]["leading_parse_changed"] is True
    assert result.diagnostics["conflict"]["conflict_score"] > 0.2
    assert result.behavioral_predictions["highest_surprisal_region"]["token"] == "ran"

def test_revision_point_identifies_the_earliest_changed_choice():
    """The model records the earlier structural decision it must reconsider."""
    result = Point23GardenPathProcessor().process(make_garden_path_request())
    choice = result.diagnostics["revision_points"]["earliest_revision_choice"]
    assert choice["decision_id"] == "deer-role"
    assert choice["before"] == "object-of-hunted"
    assert choice["after"] == "subject-of-ran"

def test_compatible_single_parse_does_not_trigger_reanalysis():
    """A compatible sentence with one candidate does not reparse by default."""
    parse = ParseHypothesis(
        parse_id="simple", description="The man hunted the deer.", prior=1.0,
        token_categories={0:"determiner",1:"noun",2:"verb",3:"determiner",4:"noun"},
        thematic_roles={"agent":1,"patient":4},
    )
    request = SentenceRequest(
        sentence_id="simple-control", text="The man hunted the deer.",
        parse_hypotheses=(parse,), task_goal="precise",
    )
    result = Point23GardenPathProcessor().process(request)
    assert result.revisions[0]["revision_attempted"] is False
    assert result.final_parse_id == "simple"

def test_no_grammar_candidate_returns_unresolved_surface_fallback():
    """A surface baseline is not reported as a complete syntactic analysis."""
    result = Point23GardenPathProcessor().process(
        SentenceRequest(sentence_id="no-grammar", text="The old man the boats.")
    )
    assert result.metadata["grammar_candidates_supplied"] is False
    assert result.interpretation_status == "unresolved"
    assert "no_grammar_parse_hypotheses_supplied" in result.unresolved_ambiguities

def test_context_cue_can_change_the_final_readout():
    """Late discourse evidence can change the selected parse after the initial commitment."""
    from dataclasses import replace
    base = make_garden_path_request()
    neutral_parses = tuple(replace(item, structural_likelihoods={})
                           for item in base.parse_hypotheses)
    request = replace(
        base,
        parse_hypotheses=neutral_parses,
        context_likelihoods={"deer-object":0.0001,"deer-matrix-subject":1.0},
    )
    result = Point23GardenPathProcessor().process(request)
    assert result.initial_parse_id == "deer-object"
    assert result.final_parse_id == "deer-matrix-subject"
    assert "context" in result.diagnostics["surface_cue_integration"]["cue_sources_used"]

def test_gist_goal_can_accept_good_enough_interpretation():
    """A low-precision task can accept a plausible parse without claiming a complete one."""
    parses = (
        ParseHypothesis(parse_id="a",description="Candidate A",prior=0.6,
                        token_categories={0:"determiner",1:"noun"}),
        ParseHypothesis(parse_id="b",description="Candidate B",prior=0.4,
                        token_categories={0:"determiner",1:"noun"}),
    )
    request = SentenceRequest(
        sentence_id="gist",text="The visitor",parse_hypotheses=parses,task_goal="gist"
    )
    assert Point23GardenPathProcessor().process(request).interpretation_status == "good_enough"

def test_unresolved_dependency_prevents_precise_completion():
    """An open grammatical slot prevents a high-confidence parse from being called complete."""
    parse = ParseHypothesis(
        parse_id="open",description="Open clause",prior=1.0,
        token_categories={0:"determiner",1:"noun"},
        unresolved_dependencies=("verb-object-slot",),
    )
    result = Point23GardenPathProcessor().process(SentenceRequest(
        sentence_id="open-slot",text="The hunter",parse_hypotheses=(parse,)
    ))
    assert result.interpretation_status == "unresolved"
    assert "verb-object-slot" in result.unresolved_ambiguities

def test_working_memory_capacity_pressure_is_visible():
    """Dependency maintenance exposes resource pressure under high modeled load."""
    result = Point23GardenPathProcessor(
        Point23Config(working_memory_capacity=0.4)
    ).process(make_garden_path_request(working_memory_load=0.95))
    assert result.diagnostics["working_memory"]["capacity_pressure"] is True

def test_exposure_does_not_count_as_correctness_without_explicit_feedback():
    """Only an explicit parse outcome updates construction-specific counts."""
    state = Point23State()
    processor = Point23GardenPathProcessor(state=state)
    first = processor.process(make_garden_path_request())
    assert first.final_state["construction_exposures"]["while-subordinate-clause"] == 1
    assert first.final_state["construction_counts"] == {}
    second = processor.process(make_garden_path_request(
        sentence_id="feedback",feedback_parse_id="deer-matrix-subject"
    ))
    assert second.final_state["construction_counts"]["while-subordinate-clause"][
        "deer-matrix-subject"
    ] > 0
    learned = second.diagnostics["learning"]["learned_construction_expectations"]
    assert sum(learned.values()) == pytest.approx(1.0)

def test_spoken_prosody_and_written_punctuation_are_separate_cue_sources():
    """Spoken and written surface information remains modality-labelled."""
    spoken = Point23GardenPathProcessor().process(make_garden_path_request(
        modality="spoken",prosody_likelihoods={"deer-object":0.2,"deer-matrix-subject":0.9}
    ))
    written = Point23GardenPathProcessor().process(make_garden_path_request(
        modality="written",punctuation_likelihoods={"deer-object":0.2,"deer-matrix-subject":0.9}
    ))
    assert spoken.diagnostics["surface_cue_integration"]["cue_sources_used"] == ["prosody"]
    assert written.diagnostics["surface_cue_integration"]["cue_sources_used"] == ["punctuation"]

def test_biological_track_can_be_disabled_without_disabling_functional_algorithm():
    """The seventeen functional operations remain when population activity is disabled."""
    result = Point23GardenPathProcessor(
        Point23Config(simulate_biology=False)
    ).process(make_garden_path_request())
    assert len(result.operation_trace) == 17
    assert result.biological["model"] == "disabled"
    assert all(row["biological_output"]["model"] == "disabled" for row in result.operation_trace)

def test_invalid_lexical_probability_and_parse_index_are_rejected():
    """The input contract prevents invalid probabilities and out-of-range token references."""
    with pytest.raises(ValueError,match="lexical candidate prior"):
        LexicalCandidate("noun",prior=1.2)
    invalid = ParseHypothesis(
        parse_id="bad",description="Out-of-range token",token_categories={99:"noun"}
    )
    request = SentenceRequest(
        sentence_id="bad-index",text="The deer",parse_hypotheses=(invalid,)
    )
    with pytest.raises(ValueError,match="beyond sentence length"):
        Point23GardenPathProcessor().process(request)

def test_sqlite_logs_seventeen_events_and_registers_package_symbols(tmp_path: Path):
    """The shared SQLite annotations and run/event tables persist the complete trace."""
    root = Path(__file__).resolve().parents[1]
    target = tmp_path / "ore_logger.sqlite3"
    result = Point23GardenPathProcessor(
        Point23Config(simulate_biology=False)
    ).process(make_garden_path_request())
    report = record_result_to_sqlite(result,target,repository_root=root)
    assert report["database_summary"]["events"] == 17
    with sqlite3.connect(target) as connection:
        annotations = connection.execute(
            "SELECT COUNT(*) FROM annotations WHERE module_name LIKE 'ore_garden_path%'"
        ).fetchone()[0]
        events = connection.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=?",(report["run_id"],)
        ).fetchone()[0]
        status = connection.execute(
            "SELECT status FROM runs WHERE run_id=?",(report["run_id"],)
        ).fetchone()[0]
    assert annotations >= 20
    assert events == 17
    assert status == "completed"

def test_annotation_registration_is_scoped_to_the_point23_package(tmp_path: Path):
    """Manual source annotation sync touches only the Point 23 package."""
    root = Path(__file__).resolve().parents[1]
    report = register_annotations(tmp_path / "catalog.sqlite3",repository_root=root)
    assert report["scope"] == "ore_garden_path"
    assert report["annotation_table"] == "annotations"
