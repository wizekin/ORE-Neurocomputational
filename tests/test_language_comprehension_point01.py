"""Acceptance tests for the source-ordered 12-step language-comprehension Point 1 implementation."""
from __future__ import annotations

import ast
from pathlib import Path
import shutil

import numpy as np

from ore_logger import LoggerDatabase
from ore_language_comprehension import (
    POINT1_STEP_SPECS,
    Point1PerceptualProcessor,
    register_point01_annotations,
)
from ore_sensory_perception import PhonemeSpec, SensoryConfig, SensoryRequest


def _audio_fixture() -> np.ndarray:
    """Construct a deterministic harmonic waveform with a smooth speech-like amplitude envelope."""
    sample_rate = 16000.0
    time = np.arange(4800, dtype=np.float64) / sample_rate
    envelope = np.clip(time / 0.025, 0.0, 1.0) * np.clip((0.3 - time) / 0.025, 0.0, 1.0)
    return envelope * (0.4 * np.sin(2 * np.pi * 180 * time) + 0.12 * np.sin(2 * np.pi * 540 * time))


def _page_fixture() -> np.ndarray:
    """Build a deterministic high-contrast pseudo-page with repeated glyph-like strokes."""
    page = np.full((88, 288), 0.94, dtype=np.float64)
    for row, start, stop in ((16, 10, 258), (40, 18, 250), (64, 12, 270)):
        page[row:row + 5, start:stop:9] = 0.06
        page[row - 4:row + 8:4, start + 3:stop:18] = 0.12
    return page


def _processor(simulate_biology: bool = False) -> Point1PerceptualProcessor:
    """Create a deterministic Point 1 processor without registering into the live SQLite database."""
    return Point1PerceptualProcessor(
        config=SensoryConfig(
            simulate_biology=simulate_biology,
            lif_population_size=4,
            lif_steps_per_operation=1,
        ),
        register_annotations=False,
        top_k=3,
    )


def test_point1_registry_has_twelve_ordered_source_operations() -> None:
    """Verify that every source-level step has its own stable identifier, equation and biological reference."""
    assert len(POINT1_STEP_SPECS) == 12
    assert [item["operation_id"] for item in POINT1_STEP_SPECS] == [f"LC-P01-{index:02d}" for index in range(1, 13)]
    assert all(item["mathematical_operations"] and item["biological_reference"] for item in POINT1_STEP_SPECS)


def test_audio_path_returns_phonetic_forms_without_lexical_meaning_and_runs_lif_reference() -> None:
    """Exercise the auditory path and confirm distinct 12-step functional and simulated biological traces."""
    request = SensoryRequest(
        sample_id="point1-audio-only",
        auditory_waveform=_audio_fixture(),
        sample_rate_hz=16000.0,
        phoneme_inventory=(
            PhonemeSpec("a", "vowel", voiced=True, vowel_height="open", vowel_backness="front"),
            PhonemeSpec("m", "nasal", place="bilabial", voiced=True),
            PhonemeSpec("s", "fricative", place="alveolar", voiced=False),
        ),
    )
    result = _processor(True).process(request)
    assert len(result.functional_trace) == 12
    assert len(result.biological_trace) == 12
    assert len(result.sensory_operation_trace) == 39
    assert len(result.sensory_biological_trace) == 39
    assert result.modality_route["auditory_pathway_active"] is True
    assert result.modality_route["visual_pathway_active"] is False
    assert result.biological_trace[0]["status"] == "simulated_reference_not_measured_brain_data"
    assert result.form_hypotheses["lexical_meaning_retrieved"] is False
    assert result.form_hypotheses["syntactic_structure_built"] is False
    assert result.form_hypotheses["auditory"]["representation"] == "ordered_phonetic_segment_candidates"


def test_visual_path_preserves_modality_and_never_fabricates_orthographic_labels() -> None:
    """Check visual-only processing and keep unlabelled shapes when no grapheme prototypes are provided."""
    result = _processor(False).process(SensoryRequest(
        sample_id="point1-visual-only",
        visual_image=_page_fixture(),
    ))
    assert result.modality_route["visual_pathway_active"] is True
    assert result.modality_route["auditory_pathway_active"] is False
    assert result.form_hypotheses["visual"]["representation"] == "position_ordered_grapheme_candidates"
    assert result.form_hypotheses["cross_modal_alignment_source"] == "not_inferred"
    assert len(result.functional_trace) == 12
    assert all(item["status"] == "biological_simulation_disabled" for item in result.biological_trace)


def test_every_point1_class_function_and_method_has_a_docstring() -> None:
    """AST-check every package definition so the SQLite catalogue can annotate every class and callable."""
    root = Path(__file__).resolve().parents[1] / "ore_language_comprehension"
    missing: list[str] = []
    for source in sorted(root.glob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node) is None:
                missing.append(f"{source.name}:{node.lineno}:{node.name}")
    assert missing == []


def test_point1_annotations_are_written_to_the_shared_sqlite_catalogue(tmp_path: Path) -> None:
    """Index an isolated repository copy and verify that Point 1 classes/functions appear in SQLite."""
    source_root = Path(__file__).resolve().parents[1]
    isolated = tmp_path / "ore-project"
    isolated.mkdir()
    for package in ("ore_logger", "ore_sensory_perception", "ore_language_comprehension"):
        shutil.copytree(source_root / package, isolated / package)
    (isolated / "pyproject.toml").write_text("[project]\nname='annotation-test'\n", encoding="utf-8")
    database_path = tmp_path / "point1-annotations.sqlite3"
    report = register_point01_annotations(isolated, database_path)
    rows = LoggerDatabase(database_path).search_annotations("ore_language_comprehension", limit=1000)
    assert report["symbols_written"] > 0
    assert rows
    assert any(row["symbol_kind"] == "class" for row in rows)
    assert any(row["symbol_kind"] == "function" for row in rows)
    assert all(row["source_docstring"] for row in rows)
