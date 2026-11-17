"""Acceptance tests for source-ordered Point 3 syntactic parsing."""

from __future__ import annotations

import ast
from pathlib import Path
import shutil

from ore_logger import LoggerDatabase
from ore_syntax import SyntaxConfig
from ore_language_comprehension import (
    POINT3_STEP_SPECS,
    Point3Request,
    Point3SyntacticParser,
    register_point03_annotations,
)


def _item(form: str, category: str, *, probability: float = 0.98, morphology=None) -> dict:
    """Create one Point-2-like lexical token with explicit category and feature evidence."""
    return {
        "form": form,
        "category_candidates": {category: probability},
        "confidence": 0.95,
        "morphology": dict(morphology or {}),
    }


def _parser(simulate_biology: bool = False) -> Point3SyntacticParser:
    """Create a deterministic syntax adapter with optional reference LIF simulation."""
    return Point3SyntacticParser(
        config=SyntaxConfig(
            simulate_biology=simulate_biology,
            population_size=4,
            steps_per_operation=1,
            max_parse_alternatives=4,
        ),
        register_annotations=False,
    )


def test_point3_registry_contains_twelve_ordered_steps_with_math_and_biology() -> None:
    """Check stable 12-step identifiers and explicit computational/biological descriptions."""
    assert len(POINT3_STEP_SPECS) == 12
    assert [item["operation_id"] for item in POINT3_STEP_SPECS] == [
        f"LC-P03-{index:02d}" for index in range(1, 13)
    ]
    assert all(item["mathematical_operations"] for item in POINT3_STEP_SPECS)
    assert all(item["biological_reference"] for item in POINT3_STEP_SPECS)


def test_active_sentence_builds_phrases_and_grammatical_dependencies() -> None:
    """Parse a simple transitive sentence into constituent groups and grammatical relations."""
    request = Point3Request(
        sequence_id="point3-dog-cat",
        lexical_items=[
            _item("The", "DET"), _item("dog", "NOUN"),
            _item("chased", "VERB"), _item("the", "DET"), _item("cat", "NOUN"),
            _item(".", "PUNCT"),
        ],
    )
    result = _parser().process(request)

    assert result.diagnostics["valid"] is True
    assert len(result.functional_trace) == 12
    assert [item["operation_id"] for item in result.functional_trace] == [
        f"LC-P03-{index:02d}" for index in range(1, 13)
    ]
    assert any(item["type"] == "NP" for item in result.constituent_groups)
    assert any(item["relation"] == "nsubj" for item in result.dependency_relations)
    assert any(item["relation"] == "obj" for item in result.dependency_relations)
    assert result.constituency_tree
    assert result.diagnostics["sentence_proposition_composed"] is False
    assert result.diagnostics["thematic_roles_assigned"] is False
    assert "thematic_roles" not in result.to_dict()


def test_category_ambiguity_is_preserved_and_ranked() -> None:
    """Retain multiple supplied part-of-speech candidates instead of forcing a category early."""
    request = Point3Request(
        sequence_id="point3-ambiguous-book",
        lexical_items=[
            _item("They", "PRON"),
            {
                "form": "book",
                "category_candidates": {"NOUN": 0.55, "VERB": 0.45},
                "confidence": 0.75,
            },
        ],
    )
    result = _parser().process(request)
    candidates = result.pos_candidates[1]

    assert len(candidates) >= 2
    assert {item["tag"] for item in candidates} >= {"NOUN", "VERB"}
    assert result.constituency_alternatives
    assert all(0.0 <= item["posterior"] <= 1.0 for item in result.constituency_alternatives)


def test_supplied_morphology_generates_explicit_agreement_mismatch_penalty() -> None:
    """Compare only caller-supplied number features on determiner and nominal head."""
    request = Point3Request(
        sequence_id="point3-agreement",
        lexical_items=[
            _item("These", "DET", morphology={"number": "plural"}),
            _item("dog", "NOUN", morphology={"number": "singular"}),
            _item("runs", "VERB", morphology={"number": "singular"}),
            _item(".", "PUNCT"),
        ],
    )
    result = _parser().process(request)

    mismatch = next(item for item in result.agreement_checks if item["relation"] == "det")
    assert mismatch["status"] == "mismatch"
    assert mismatch["checks"][0]["feature"] == "number"
    assert mismatch["penalty"] > 0.0
    arc = next(
        item for item in result.dependency_relations
        if item["relation"] == "det" and item["dependent"].casefold() == "these"
    )
    assert arc["score"] < arc["score_before_agreement"]


def test_supplied_prosodic_and_boundary_cues_are_retained_without_inference() -> None:
    """Expose caller-supplied prosody and punctuation as separate, modality-aware cues."""
    request = Point3Request(
        sequence_id="point3-boundary-cues",
        lexical_items=[
            _item("The", "DET"), _item("dog", "NOUN"),
            _item("runs", "VERB"), _item(".", "PUNCT"),
        ],
        boundary_cues={
            2: {"modality": "auditory", "pause_strength": 0.6, "pitch_boundary": 0.7},
        },
    )
    result = _parser().process(request)

    assert result.cue_evidence["punctuation_and_visual_boundaries"]
    assert result.cue_evidence["caller_supplied_speech_prosody_or_boundary_cues"][2]["pause_strength"] == 0.6
    assert "Prosody is used only when supplied" in result.cue_evidence["limitations"]


def test_point2_result_adapter_flattens_lexical_features_for_syntax() -> None:
    """Convert Point 2 records into ordered tokens with their lexical category and morphology."""
    point2 = {
        "lexical_candidates": [{
            "entry_id": "dog-n",
            "canonical_word_form": "dog",
            "lexical_category": "noun",
            "posterior": 0.96,
            "form_reliability": 0.91,
            "lexical_properties": {
                "morphological_features": {"number": "singular"},
                "grammatical_features": {"person": "third"},
            },
        }],
        "selected_candidate": None,
        "confidence": 0.96,
        "form_candidates": [{"form": "dog", "confidence": 0.91}],
    }
    request = Point3Request.from_point2_results("point3-from-point2", [point2])

    assert request.lexical_items[0]["form"] == "dog"
    assert request.lexical_items[0]["category_candidates"] == {"NOUN": 1.0}
    assert request.lexical_items[0]["morphology"]["number"] == "singular"
    assert request.lexical_items[0]["morphology"]["person"] == "third"


def test_lif_reference_track_is_separate_and_not_claimed_as_human_measurement() -> None:
    """Run the syntax LIF reference and validate the 12 mapped source-level biological records."""
    result = _parser(True).process(Point3Request(
        sequence_id="point3-lif",
        lexical_items=[
            _item("The", "DET"), _item("cat", "NOUN"),
            _item("runs", "VERB"), _item(".", "PUNCT"),
        ],
    ))

    assert result.biological_summary["model"] == "current_based_LIF_syntax_reference"
    assert result.biological_summary["units"]["voltage"] == "mV"
    assert len(result.biological_trace) == 12
    assert result.biological_trace[0]["status"] == "simulated_reference_not_measured_human_data"
    assert result.diagnostics["biological_reference_is_measured_data"] is False


def test_every_language_comprehension_class_and_function_has_a_docstring() -> None:
    """Ensure the logger can index every Point 3 class, helper, method, and package definition."""
    root = Path(__file__).resolve().parents[1] / "ore_language_comprehension"
    missing = []
    for source in sorted(root.glob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if ast.get_docstring(node) is None:
                    missing.append(f"{source.name}:{node.lineno}:{node.name}")
    assert missing == []


def test_point3_annotations_use_the_existing_sqlite_catalogue(tmp_path: Path) -> None:
    """Index syntax and language-comprehension definitions into the shared annotations table."""
    source_root = Path(__file__).resolve().parents[1]
    isolated = tmp_path / "ore-project"
    isolated.mkdir()
    for package in ("ore_logger", "ore_language_comprehension", "ore_syntax"):
        shutil.copytree(source_root / package, isolated / package)
    (isolated / "pyproject.toml").write_text(
        "[project]\nname='point3-annotation-test'\n", encoding="utf-8"
    )
    database_path = tmp_path / "point3-annotations.sqlite3"
    report = register_point03_annotations(isolated, database_path)
    rows = LoggerDatabase(database_path).search_annotations("ore_language_comprehension", limit=1000)

    assert report["symbols_written"] > 0
    assert report["database_path"] == str(database_path)
    assert rows
    assert any(row["qualified_name"] == "Point3SyntacticParser.process" for row in rows)
    assert any(row["qualified_name"] == "Point3Request.from_point2_results" for row in rows)
    assert all(row["source_docstring"] for row in rows)
