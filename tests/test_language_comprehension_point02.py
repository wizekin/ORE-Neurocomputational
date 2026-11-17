"""Acceptance tests for source-ordered Point 2 lexical access."""

from __future__ import annotations

import ast
from pathlib import Path
import shutil

from ore_logger import LoggerDatabase
from ore_lexical_access import LexicalAccessConfig
from ore_lexical_context import LexicalEntry, LexicalMeaning, MorphemeEntry
from ore_language_comprehension import (
    POINT2_STEP_SPECS,
    Point2LexicalAccessProcessor,
    Point2Request,
    register_point02_annotations,
)


def _visual_hypotheses(word: str, confidence: float = 0.98) -> dict:
    """Build Point 1-shaped ordered grapheme alternatives for a test word."""
    glyphs = []
    for index, character in enumerate(word):
        glyphs.append({
            "glyph_id": f"glyph-{index}",
            "region_id": "region-1",
            "position_code": {"ordinal_position": index},
            "candidates": [{"grapheme": character, "probability": confidence}],
            "confidence": confidence,
        })
    return {
        "visual": {"form_hypotheses": glyphs},
        "auditory": {"form_hypotheses": []},
    }


def _auditory_hypotheses(phonemes: tuple[str, ...], confidence: float = 0.98) -> dict:
    """Build Point 1-shaped time-ordered phoneme alternatives for a test pronunciation."""
    segments = []
    for index, phoneme in enumerate(phonemes):
        segments.append({
            "selected_symbol": phoneme,
            "alternatives": [{"symbol": phoneme, "probability": confidence}],
            "confidence": confidence,
            "frame_span": [index * 3, index * 3 + 3],
            "onset_seconds": index * 0.03,
            "offset_seconds": (index + 1) * 0.03,
        })
    return {
        "visual": {"form_hypotheses": []},
        "auditory": {"form_hypotheses": segments},
    }


def _processor(simulate_biology: bool = False) -> Point2LexicalAccessProcessor:
    """Create a deterministic Point 2 processor without touching the live annotation database."""
    return Point2LexicalAccessProcessor(
        config=LexicalAccessConfig(
            simulate_biology=simulate_biology,
            population_size=4,
            steps_per_character=1,
        ),
        register_annotations=False,
        top_k=4,
        max_form_candidates=8,
    )


def _cat_entry() -> LexicalEntry:
    """Return one explicit English lexical entry with a pronunciation and word-level features."""
    return LexicalEntry(
        "cat",
        "cat",
        frequency=24,
        phonemes=("k", "æ", "t"),
        meanings=(
            LexicalMeaning(
                "feline",
                {"ANIMAL": 1.0, "FELINE": 1.0, "DOMESTIC": 0.7},
                gloss="a small domesticated feline",
                verified_by="test-fixture",
                source_reference="test://cat",
            ),
        ),
        lexical_category="noun",
        language="en",
        verified_by="test-fixture",
        source_reference="test://cat",
    )


def test_point2_registry_contains_twelve_ordered_steps_with_math_and_biology() -> None:
    """Check stable source operation IDs and both computational and biological annotations."""
    assert len(POINT2_STEP_SPECS) == 12
    assert [item["operation_id"] for item in POINT2_STEP_SPECS] == [
        f"LC-P02-{index:02d}" for index in range(1, 13)
    ]
    assert all(item["mathematical_operations"] for item in POINT2_STEP_SPECS)
    assert all(item["biological_reference"] for item in POINT2_STEP_SPECS)


def test_visual_point1_handoff_retrieves_word_identity_meaning_and_category() -> None:
    """Process ordered grapheme candidates and return an explicit word-level lexical-semantic record."""
    request = Point2Request(
        sequence_id="point2-visual-cat",
        form_hypotheses=_visual_hypotheses("cat"),
        lexicon=(_cat_entry(),),
    )
    result = _processor().process(request)

    assert result.modality_routes["visual_active"] is True
    assert result.modality_routes["auditory_active"] is False
    assert result.form_candidates[0]["form"] == "cat"
    assert result.lexical_candidates[0]["entry_id"] == "cat"
    assert result.selected_candidate is not None
    assert result.word_level_semantic_features["ANIMAL"] > 0.0
    assert result.meaning_candidates[0]["meaning_id"] == "feline"
    assert result.lexical_properties["candidate_properties"][0]["lexical_category"] == "noun"
    assert len(result.functional_trace) == 12
    assert result.functional_trace[-1]["functional_result"]["sentence_structure_built"] is False
    assert result.diagnostics["valid"] is True


def test_auditory_point1_handoff_uses_only_caller_supplied_pronunciations() -> None:
    """Match a time-ordered phoneme sequence to stored entry pronunciations without inventing G2P."""
    request = Point2Request(
        sequence_id="point2-auditory-cat",
        form_hypotheses=_auditory_hypotheses(("k", "æ", "t")),
        lexicon=(_cat_entry(),),
    )
    result = _processor().process(request)

    assert result.modality_routes["auditory_active"] is True
    assert result.modality_routes["visual_active"] is False
    assert result.lexical_candidates[0]["entry_id"] == "cat"
    assert result.lexical_candidates[0]["modality"] == "auditory"
    assert result.lexical_candidates[0]["form_match_score"] == 1.0
    assert result.lexical_candidates[0]["phonological_form"] == ["k", "æ", "t"]


def test_homonyms_and_polysemous_senses_remain_separate_hypotheses() -> None:
    """Keep different lexical identities and multiple senses within one entry distinct."""
    river = LexicalEntry(
        "bank-river",
        "bank",
        frequency=20,
        meanings=(
            LexicalMeaning("river-edge", {"RIVER": 1.0, "LAND": 0.8}, prior=1.0),
            LexicalMeaning("river-slope", {"RIVER": 0.9, "SLOPE": 0.8}, prior=0.7),
        ),
        lexical_category="noun",
        language="en",
    )
    finance = LexicalEntry(
        "bank-finance",
        "bank",
        frequency=20,
        meanings=(
            LexicalMeaning("financial-institution", {"MONEY": 1.0, "INSTITUTION": 0.9}),
        ),
        lexical_category="noun",
        language="en",
    )
    result = _processor().process(Point2Request(
        sequence_id="point2-bank",
        form_hypotheses=_visual_hypotheses("bank"),
        lexicon=(river, finance),
        context_features={"RIVER": 1.0},
    ))

    ambiguity_types = {item["type"] for item in result.unresolved_ambiguities}
    assert {item["entry_id"] for item in result.lexical_candidates} == {"bank-river", "bank-finance"}
    assert "homonym_or_lexical_identity_ambiguity" in ambiguity_types
    assert "polysemy_or_sense_ambiguity" in ambiguity_types
    assert len(result.sense_candidates) == 3


def test_morphological_analysis_is_inventory_supported_and_keeps_whole_word_route() -> None:
    """Retrieve a supplied stem and plural suffix while retaining legal segmentation evidence."""
    suffix = MorphemeEntry(
        "plural-s",
        "s",
        "suffix",
        {"PLURAL": 1.0},
        frequency=20,
        productivity=0.95,
        verified_by="test-fixture",
        source_reference="test://plural-s",
    )
    result = _processor().process(Point2Request(
        sequence_id="point2-cats",
        form_hypotheses=_visual_hypotheses("cats"),
        lexicon=(_cat_entry(),),
        morphemes=(suffix,),
        morpheme_category_cues={"plural-s": {"noun": 0.9}},
    ))

    cat_candidate = next(item for item in result.lexical_candidates if item["entry_id"] == "cat")
    assert any(item["kind"] == "segmented" for item in cat_candidate["morphological_alternatives"])
    assert any(item["stem"] == "cat" for item in cat_candidate["stem_affix_retrieval"])
    assert any("PLURAL" in item["composed_features"] for item in cat_candidate["stem_affix_retrieval"])


def test_new_sensory_evidence_revises_existing_candidates_and_emits_feedback() -> None:
    """Re-run candidate evidence with an additional sensory form and report the before/after comparison."""
    cat = _cat_entry()
    bat = LexicalEntry(
        "bat",
        "bat",
        frequency=18,
        phonemes=("b", "æ", "t"),
        meanings=(LexicalMeaning("flying-mammal", {"ANIMAL": 1.0, "FLIGHT": 0.8}),),
        lexical_category="noun",
        language="en",
    )
    result = _processor().process(Point2Request(
        sequence_id="point2-update",
        form_hypotheses=_visual_hypotheses("bat", confidence=0.75),
        lexicon=(cat, bat),
        sensory_updates=({"modality": "visual", "form": "cat", "confidence": 0.99},),
    ))

    update = result.functional_trace[10]["functional_result"]
    assert update["new_evidence_received"] is True
    assert update["existing_candidates_re_scored"] is True
    assert update["updates_processed"] == 1
    assert any(item["entry_id"] == "cat" for item in result.lexical_candidates)


def test_biological_reference_runs_seven_populations_and_returns_source_step_trace() -> None:
    """Validate finite recurrent LIF outputs and distinguish simulation from measured brain data."""
    result = _processor(True).process(Point2Request(
        sequence_id="point2-lif",
        form_hypotheses=_visual_hypotheses("cat"),
        lexicon=(_cat_entry(),),
    ))

    assert result.diagnostics["biological_simulation_enabled"] is True
    assert result.diagnostics["biological_reference_numerically_valid"] is True
    assert len(result.biological_trace) == 12
    assert result.biological_trace[0]["status"] == "simulated_reference_not_measured_human_data"
    assert result.biological_trace[0]["population_names"]
    assert len(result.low_level_operation_trace) == 36


def test_point1_result_adapter_accepts_result_like_object() -> None:
    """Construct a Point 2 request directly from a Point1Result-like object."""
    class ResultLike:
        """Minimal Point 1 result container for the adapter boundary."""

        form_hypotheses = _visual_hypotheses("cat")

    request = Point2Request.from_point1_result(
        "point2-from-point1",
        ResultLike(),
        (_cat_entry(),),
    )
    assert request.form_hypotheses["visual"]["form_hypotheses"]


def test_every_point2_class_and_function_has_a_logger_ready_docstring() -> None:
    """AST-check every language-comprehension definition for the SQLite annotation index."""
    root = Path(__file__).resolve().parents[1] / "ore_language_comprehension"
    missing = []
    for source in sorted(root.glob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if ast.get_docstring(node) is None:
                    missing.append(f"{source.name}:{node.lineno}:{node.name}")
    assert missing == []


def test_point2_annotations_are_indexed_in_the_existing_sqlite_catalogue(tmp_path: Path) -> None:
    """Index an isolated repository copy and verify Point 2 functions/classes in the shared annotation table."""
    source_root = Path(__file__).resolve().parents[1]
    isolated = tmp_path / "ore-project"
    isolated.mkdir()
    packages = (
        "ore_logger",
        "ore_language_comprehension",
        "ore_lexical_access",
        "ore_lexical_context",
    )
    for package in packages:
        shutil.copytree(source_root / package, isolated / package)
    (isolated / "pyproject.toml").write_text("[project]\nname='point2-annotation-test'\n", encoding="utf-8")
    database_path = tmp_path / "point2-annotations.sqlite3"
    report = register_point02_annotations(isolated, database_path)
    rows = LoggerDatabase(database_path).search_annotations("ore_language_comprehension", limit=1000)
    assert report["symbols_written"] > 0
    assert rows
    assert any(row["symbol_kind"] == "class" for row in rows)
    assert any(row["symbol_kind"] == "function" for row in rows)
    assert any(row["qualified_name"] == "Point2LexicalAccessProcessor.process" for row in rows)
    assert all(row["source_docstring"] for row in rows)
