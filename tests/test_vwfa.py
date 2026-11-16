"""Tests for the 24-operation VWFA subsystem and its SQLite annotation integration."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from ore_vwfa import (
    GraphemePrototype, LexicalEntry, MorphemeEntry, OPERATION_REGISTRY,
    STAGE_STEP_COUNTS, VWFAConfig, VWFAExample, VWFAProcessor, VWFARequest,
    VWFABiologicalSimulator,
)
from ore_vwfa.math_ops import morphology_segmentations, ordered_edit_similarity


def _image() -> np.ndarray:
    """Create a deterministic high-contrast image with separate printed-stroke-like structures."""
    image = np.zeros((72, 144), dtype=np.float64)
    image[12:60, 15:20] = 1.0
    image[12:17, 15:45] = 1.0
    image[34:39, 15:45] = 1.0
    image[12:60, 62:67] = 1.0
    image[12:17, 62:92] = 1.0
    image[55:60, 62:92] = 1.0
    return image


def test_registry_contains_exactly_24_ordered_operations_across_six_stages() -> None:
    """Require ordered VWFA-001..024 identifiers and the requested 4/4/4/5/4/3 stage split."""
    assert len(OPERATION_REGISTRY) == 24
    assert [item.operation_id for item in OPERATION_REGISTRY] == [f"VWFA-{i:03d}" for i in range(1, 25)]
    assert STAGE_STEP_COUNTS == {"A": 4, "B": 4, "C": 4, "D": 5, "E": 4, "F": 3}
    assert {item.stage for item in OPERATION_REGISTRY} == set("ABCDEF")
    assert [item.step for item in OPERATION_REGISTRY] == list(range(1, 25))
    assert all(item.mathematical_operations and item.biological_reference for item in OPERATION_REGISTRY)


def test_math_primitives_extract_finite_oriented_features_and_ordered_form() -> None:
    """Check Gabor feature flow, cosine/edit metrics, and explicit order sensitivity."""
    from ore_vwfa.math_ops import contrast_normalize, gabor_energy_bank

    normalized = contrast_normalize(_image())["normalized_image"]
    energy, responses = gabor_energy_bank(normalized, (0.0, 0.8), (0.08, 0.16), 15, 3.0, 0.7)
    assert energy.shape == (4, 72, 144)
    assert responses["even_responses"].shape == energy.shape
    assert np.all(np.isfinite(energy))
    assert ordered_edit_similarity("form", "from") < ordered_edit_similarity("form", "form")


def test_upstream_orthography_runs_all_24_operations_and_preserves_order() -> None:
    """Run the six-stage pipeline from an explicit upstream sequence without claiming image OCR."""
    request = VWFARequest(
        sample_id="upstream-sequence",
        orthographic_sequence=("f", "o", "r", "m"),
        lexicon=(LexicalEntry("form", 30.0), LexicalEntry("from", 10.0)),
    )
    result = VWFAProcessor(VWFAConfig(simulate_biology=False), register_annotations=False).process(request)
    assert result.diagnostics["valid"] is True
    assert result.diagnostics["operation_count"] == 24
    assert result.diagnostics["operation_ids_complete"] is True
    assert result.functional["orthographic_sequence"] == ["f", "o", "r", "m"]
    assert result.functional["selected_candidate"] == "form"
    assert result.biological["enabled"] is False
    assert result.operation_trace[7]["functional_output"]["status"] == "upstream_orthography_preserved"


def test_image_path_keeps_visual_regions_separate_from_ocr_and_runs_lif_populations() -> None:
    """Exercise pixel-level visual stages and verify neural-reference activity is separately reported."""
    result = VWFAProcessor(VWFAConfig(lif_steps_per_operation=2), register_annotations=False).process(
        VWFARequest(sample_id="image-only", visual_image=_image())
    )
    assert result.functional["visual_available"] is True
    assert len(result.functional["candidate_regions"]) > 0
    assert result.functional["grapheme_recognition_status"] == "awaiting_labelled_grapheme_prototypes"
    assert result.functional["selected_candidate"] is None
    assert result.biological["enabled"] is True
    assert result.biological["final_snapshot"]["time_ms"] == pytest.approx(48.0)
    assert set(result.biological["final_snapshot"]["populations"]) >= {
        "v1", "vwfa_posterior", "vwfa_intermediate", "vwfa_anterior", "ifg", "mtg", "atl"
    }
    assert result.biological["final_snapshot"]["units"] == {
        "voltage": "mV", "current": "pA", "time": "ms", "firing_rate": "Hz"
    }


def test_grapheme_prototype_matching_requires_compatible_supplied_features() -> None:
    """Permit a supplied prototype to classify image regions only in matching feature space."""
    prototype = GraphemePrototype("A", np.zeros(144), np.ones(144), "test://prototype-A")
    result = VWFAProcessor(VWFAConfig(simulate_biology=False), register_annotations=False).process(
        VWFARequest(sample_id="prototype-match", visual_image=_image(), grapheme_prototypes=(prototype,))
    )
    assert result.functional["grapheme_recognition_status"] in {"prototype_match", "unknown"}
    if result.functional["grapheme_recognition_status"] == "prototype_match":
        assert result.functional["grapheme_posteriors"]
        assert all(item["candidates"][0]["grapheme"] == "A" for item in result.functional["grapheme_posteriors"])


def test_morphology_uses_only_explicit_inventory_and_keeps_whole_word_parallel() -> None:
    """Find cat+s only from a provided inventory and preserve whole-word and morpheme channels together."""
    inventory = (
        MorphemeEntry("cat", "stem", 20.0, {"ANIMAL": 1.0}),
        MorphemeEntry("s", "inflection", 30.0, {"PLURAL": 1.0}, {"NUMBER_PLURAL": 1.0}),
    )
    assert any(item["morphemes"] == ["cat", "s"] for item in morphology_segmentations("cats", inventory))
    assert morphology_segmentations("cats", ()) == []
    result = VWFAProcessor(VWFAConfig(simulate_biology=False), register_annotations=False).process(
        VWFARequest(
            sample_id="morphology", orthographic_sequence=tuple("cats"),
            lexicon=(LexicalEntry("cats", 25.0, {"ANIMAL": 1.0}),),
            morphology_inventory=inventory, context_features={"ANIMAL": 0.7},
        )
    )
    assert result.functional["selected_candidate"] == "cats"
    assert result.functional["morphology_candidates"]
    assert result.functional["parallel_morphology_activity"]["whole_word"] > 0.0
    assert result.functional["composed_language_features"]["ANIMAL"] > 0.0
    assert result.functional["composed_language_features"]["PLURAL"] > 0.0


def test_stage_f_learning_is_preview_only_until_explicitly_authorized() -> None:
    """Verify that learning examples do not mutate association weights or memories without authorization."""
    processor = VWFAProcessor(VWFAConfig(simulate_biology=True), register_annotations=False)
    examples = (
        VWFAExample("cat", (1.0, 0.0, 0.0), "word"),
        VWFAExample("dog", (0.0, 1.0, 0.0), "word"),
        VWFAExample("xyz", (0.0, 0.0, 1.0), "control"),
    )
    preview = processor.process(VWFARequest(
        sample_id="preview", orthographic_sequence=("c", "a", "t"), learning_examples=examples,
    ))
    assert preview.diagnostics["learning_applied"] is False
    assert processor.learning_state["visual_phonological_weights"] is None
    assert processor.learning_state["fast_prototypes"] == {}
    assert preview.operation_trace[21]["functional_output"]["weights_updated"] is False
    trained = processor.process(VWFARequest(
        sample_id="authorized", orthographic_sequence=("c", "a", "t"),
        learning_examples=examples, learning_authorized=True,
    ))
    assert trained.diagnostics["learning_applied"] is True
    assert processor.learning_state["visual_phonological_weights"].shape == (64, 3)
    assert processor.learning_state["fast_prototypes"] and processor.learning_state["slow_prototypes"]
    assert trained.operation_trace[21]["functional_output"]["weights_updated"] is True
    assert trained.biological["operation_trace"][-1]["consolidation"]


def test_lif_reference_resets_per_sample_but_preserves_learned_connectivity() -> None:
    """Check membrane updates, per-sample reset, and persistent learned connectivity weights."""
    simulator = VWFABiologicalSimulator(VWFAConfig(lif_steps_per_operation=4))
    initial = simulator.snapshot()["connectivity"]
    simulator.advance_operation(1, 1.0)
    assert simulator.snapshot()["time_ms"] == pytest.approx(4.0)
    simulator.apply_hebbian_update({"vwfa_posterior": 1.0}, {"phonological_association": 1.0})
    learned = simulator.snapshot()["connectivity"]
    assert not np.allclose(initial, learned)
    simulator.reset_activity()
    reset = simulator.snapshot()
    assert reset["time_ms"] == 0.0
    assert all(item["spike_count"] == 0 for item in reset["populations"].values())
    assert np.allclose(reset["connectivity"], learned)


def test_every_class_and_function_has_a_docstring_for_sqlite_annotations() -> None:
    """Require explanatory docstrings on modules, classes, methods, helper functions, and nested functions."""
    package = Path(__file__).resolve().parents[1] / "ore_vwfa"
    missing: list[str] = []
    for source in package.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        if ast.get_docstring(tree) is None:
            missing.append(f"{source.name}:module")
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node) is None:
                missing.append(f"{source.name}:{node.lineno}:{node.name}")
    assert not missing, "Missing SQLite annotation docstrings: " + ", ".join(missing)


def test_vwfa_docstrings_are_persisted_in_the_existing_ore_sqlite_logger(tmp_path: Path) -> None:
    """Synchronize the new package and check its class, method, and numerical function annotation rows."""
    try:
        from ore_logger import LoggerDatabase
        from ore_vwfa import register_vwfa_annotations
    except ImportError:
        pytest.skip("The complete ORE checkout supplies ore_logger; this package-only test run does not.")
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "ore_logger.sqlite3"
    report = register_vwfa_annotations(root, database_path)
    database = LoggerDatabase(database_path)
    class_annotation = database.get_annotation("ore_vwfa.pipeline", "VWFAProcessor", "class", "ore_vwfa/pipeline.py")
    method_annotation = database.get_annotation("ore_vwfa.pipeline", "VWFAProcessor.process", "function", "ore_vwfa/pipeline.py")
    function_annotation = database.get_annotation("ore_vwfa.math_ops", "gabor_energy_bank", "function", "ore_vwfa/math_ops.py")
    assert report["files_indexed"] >= 1
    assert class_annotation and class_annotation["annotation_source"] == "docstring"
    assert method_annotation and method_annotation["annotation_source"] == "docstring"
    assert function_annotation and function_annotation["annotation_source"] == "docstring"
    rows = database.search_annotations("ore_vwfa", limit=1000)
    assert len(rows) >= 100
    assert all(row["annotation"].strip() for row in rows)
