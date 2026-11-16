"""Focused tests for visual-orthographic Stages A-E and SQLite documentation."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from ore_logger import LoggerDatabase
from ore_visual_orthography import (
    BIOLOGICAL_POPULATIONS,
    BiologicalReferenceSimulator,
    LetterPrototype,
    LexicalCandidate,
    OPERATION_REGISTRY,
    VisualOrthographyConfig,
    VisualOrthographyProcessor,
    VisualOrthographyRequest,
    compare_orthographic_strings,
    register_visual_orthography_annotations,
)


def _synthetic_word_image() -> np.ndarray:
    """Create a deterministic high-contrast array containing multiple stroke-like regions."""
    image = np.zeros((64, 128), dtype=np.float64)
    image[10:54, 12:17] = 1.0
    image[10:15, 12:42] = 1.0
    image[30:35, 12:42] = 1.0
    image[10:54, 58:63] = 1.0
    image[10:15, 58:88] = 1.0
    image[49:54, 58:88] = 1.0
    image[10:54, 103:108] = 1.0
    return image


def test_registry_contains_sixteen_ordered_steps_in_five_stages() -> None:
    """Require stable OVO operation IDs and the exact A-E step distribution."""
    assert len(OPERATION_REGISTRY) == 16
    assert [item.operation_id for item in OPERATION_REGISTRY] == [
        f"OVO-{index:03d}" for index in range(1, 17)
    ]
    assert {
        stage: sum(item.stage == stage for item in OPERATION_REGISTRY)
        for stage in "ABCDE"
    } == {"A": 3, "B": 2, "C": 4, "D": 3, "E": 4}
    assert all(item.mathematical_operations and item.biological_reference for item in OPERATION_REGISTRY)


def test_transposed_letter_effect_preserves_more_open_bigrams_than_replacement() -> None:
    """Check that JUGDE shares nine of eleven bigram-set elements with JUDGE."""
    transposed = compare_orthographic_strings("JUDGE", "JUGDE")
    replacement = compare_orthographic_strings("JUDGE", "JUPTE")
    assert transposed["bigram_jaccard_similarity"] == pytest.approx(9.0 / 11.0)
    assert transposed["combined_similarity"] > replacement["combined_similarity"]


def test_all_sixteen_steps_run_from_an_explicit_orthographic_sequence() -> None:
    """Run A-E without an image and verify recurrent, lexical and stabilization outputs."""
    config = VisualOrthographyConfig(simulate_biology=False)
    processor = VisualOrthographyProcessor(config, register_annotations=False)
    request = VisualOrthographyRequest(
        sample_id="sequence-smoke",
        orthographic_sequence=("j", "u", "d", "g", "e"),
        comparison_strings=("jugde", "jupte"),
        lexicon=(
            LexicalCandidate("judge", frequency=30.0),
            LexicalCandidate("jugde", frequency=1.0),
            LexicalCandidate("jungle", frequency=20.0),
        ),
    )
    result = processor.process(request)
    assert result.diagnostics["operation_count"] == 16
    assert result.diagnostics["operation_ids_complete"] is True
    assert result.diagnostics["stage_operation_counts"] == {
        "A": 3, "B": 2, "C": 4, "D": 3, "E": 4,
    }
    assert result.functional["stage_d"]["parallel_letter_populations"]["activation_matrix"].shape == (5, 5)
    assert result.functional["stage_d"]["recurrent_competition"]["iterations"] == config.recurrent_steps
    assert result.functional["stage_e"]["transposition_similarity"]["comparisons"]["jugde"][
        "bigram_jaccard_similarity"
    ] == pytest.approx(9.0 / 11.0)
    assert result.functional["stage_e"]["lexical_candidates"]["candidates"][0]["word"] == "judge"
    assert result.functional["stage_e"]["stabilization"]["selected_word"] == "judge"
    assert result.biological["enabled"] is False


def test_image_path_extracts_finite_visual_features_without_fabricating_ocr() -> None:
    """Run visual feature extraction while confirming unlabeled pixels are not guessed as text."""
    processor = VisualOrthographyProcessor(
        VisualOrthographyConfig(simulate_biology=False),
        register_annotations=False,
    )
    result = processor.process(
        VisualOrthographyRequest(
            sample_id="image-only",
            visual_image=_synthetic_word_image(),
        )
    )
    feature_stack = result.functional["stage_a"]["early_visual_features"]["feature_stack"]
    assert feature_stack.ndim == 3
    assert np.all(np.isfinite(feature_stack))
    assert result.functional["stage_b"]["letter_identity"]["sequence"] == []
    assert result.operation_trace[3]["functional_status"] == "awaiting_labelled_prototypes_or_explicit_sequence"
    assert result.metadata["scientific_limitations"]


def test_labelled_prototypes_decode_and_keep_visual_position_order() -> None:
    """Use simple synthetic visual prototypes to confirm labelled decoding and position tracking."""
    processor = VisualOrthographyProcessor(
        VisualOrthographyConfig(simulate_biology=False),
        register_annotations=False,
    )
    gabor = processor.visual_processor
    image_a = np.zeros((48, 48), dtype=np.float64)
    image_a[8:40, 10:14] = 1.0
    image_a[8:12, 10:38] = 1.0
    image_a[24:28, 10:38] = 1.0
    prototype = processor.make_letter_prototype("a", image_a, form_id="test-a")
    assert prototype.label == "a"
    assert len(prototype.mean) == 2 * len(gabor.config.orientations) * len(gabor.config.spatial_frequencies) + 7
    observed = processor.update_letter_prototype(prototype, prototype.mean, learning_rate=0.2)
    assert observed.label == "a"
    assert np.all(np.isfinite(observed.variance))


def test_biological_reference_advances_all_populations_with_finite_units() -> None:
    """Advance current-based LIF populations and check voltage, current, rate and elapsed time."""
    config = VisualOrthographyConfig(lif_steps_per_operation=2)
    simulator = BiologicalReferenceSimulator(config)
    snapshot = simulator.advance({"retina_lgn": 1.0}, steps=30)
    assert snapshot["time_ms"] == pytest.approx(30.0)
    assert set(snapshot["populations"]) == set(BIOLOGICAL_POPULATIONS)
    assert all(np.isfinite(item["mean_voltage_mv"]) for item in snapshot["populations"].values())
    assert all(np.isfinite(item["mean_current_pa"]) for item in snapshot["populations"].values())
    assert all(np.isfinite(item["firing_rate_hz"]) for item in snapshot["populations"].values())


def test_every_class_and_callable_has_a_docstring_for_sqlite_annotations() -> None:
    """Require explicit descriptions for each Python class and function in the new package."""
    package = Path(__file__).resolve().parents[1] / "ore_visual_orthography"
    missing: list[str] = []
    for source in package.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if ast.get_docstring(node) is None:
                    missing.append(f"{source.name}:{node.lineno}:{node.name}")
    assert not missing, "Missing docstrings: " + ", ".join(missing)


def test_class_and_function_annotations_are_persisted_in_ore_sqlite(tmp_path: Path) -> None:
    """Synchronize package docstrings into the existing SQLite annotation table and verify entries."""
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "ore_logger.sqlite3"
    report = register_visual_orthography_annotations(root, database_path)
    database = LoggerDatabase(database_path)
    class_annotation = database.get_annotation(
        "ore_visual_orthography.pipeline",
        "VisualOrthographyProcessor",
        "class",
        "ore_visual_orthography/pipeline.py",
    )
    method_annotation = database.get_annotation(
        "ore_visual_orthography.pipeline",
        "VisualOrthographyProcessor.process",
        "function",
        "ore_visual_orthography/pipeline.py",
    )
    function_annotation = database.get_annotation(
        "ore_visual_orthography.math_ops",
        "open_bigram_activations",
        "function",
        "ore_visual_orthography/math_ops.py",
    )
    assert report["database_path"] == str(database_path.resolve())
    assert class_annotation and class_annotation["annotation_source"] == "docstring"
    assert method_annotation and method_annotation["annotation_source"] == "docstring"
    assert function_annotation and function_annotation["annotation_source"] == "docstring"
    package_rows = database.search_annotations("ore_visual_orthography", limit=500)
    assert len(package_rows) >= 30
    assert all(row["annotation"].strip() for row in package_rows)
