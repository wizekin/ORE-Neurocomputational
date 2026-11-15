"""Focused tests for BNBEA's twelve operation interfaces and independent evidence tracks."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from ore_brain_networks import (
    BNBEAConfig,
    BNBEAProcessor,
    BNBEARequest,
    GraphemePrototype,
    LexicalEntry,
    OPERATION_REGISTRY,
    register_bnbea_annotations,
)
from ore_brain_networks.auditory_streams import AuditoryStreamProcessor
from ore_brain_networks.biology import NeuralDynamicsSimulator, POPULATIONS
from ore_brain_networks.eeg_meg import forward_model, inverse_model
from ore_brain_networks.fmri import build_rdm, rdm_correlation, fit_glm
from ore_brain_networks.grapheme_populations import GraphemePopulationDecoder
from ore_brain_networks.human_validation import HumanReadingValidation, split_subjects
from ore_brain_networks.lesion_validation import (
    apply_virtual_lesion, graph_global_efficiency, lesion_symptom_regression,
)
from ore_brain_networks.tms_validation import simulate_virtual_tms
from ore_brain_networks.visual_gabor import GaborVisualProcessor, extract_region_feature_vector
from ore_brain_networks.visual_hierarchy import VisualHierarchyProcessor
from ore_brain_networks.vwfa_hierarchy import VWFAHierarchyProcessor, ordered_similarity


def _image() -> np.ndarray:
    """Build a deterministic high-contrast image with separate stroke-like regions for smoke tests."""
    image = np.zeros((64, 96), dtype=np.float64)
    image[10:54, 12:17] = 1.0
    image[10:15, 12:42] = 1.0
    image[30:35, 12:42] = 1.0
    image[10:54, 55:60] = 1.0
    image[10:15, 55:85] = 1.0
    image[49:54, 55:85] = 1.0
    return image


def test_registry_contains_exactly_twelve_ordered_operations() -> None:
    """Require a stable, duplicate-free BNBEA-001 through BNBEA-012 operation registry."""
    assert len(OPERATION_REGISTRY) == 12
    assert [item.operation_id for item in OPERATION_REGISTRY] == [
        f"BNBEA-{index:03d}" for index in range(1, 13)
    ]
    assert all(item.mathematical_operations for item in OPERATION_REGISTRY)


def test_gabor_stage_and_region_features_are_finite() -> None:
    """Exercise contrast normalization, quadrature Gabor energy and fixed-size region features."""
    output = GaborVisualProcessor().process(_image())
    stack = output["feature_stack"]
    assert stack.ndim == 3
    assert np.all(np.isfinite(stack))
    mask = np.zeros(stack.shape[1:], dtype=bool)
    mask[12:44, 12:42] = True
    vector = extract_region_feature_vector(stack, mask)
    assert vector.shape == (stack.shape[0] * 2 + 7,)
    assert np.all(np.isfinite(vector))


def test_population_decoder_preserves_candidate_order_and_competes_over_prototypes() -> None:
    """Check candidate segmentation, distance-based population responses and labelled prototypes."""
    stack = GaborVisualProcessor().process(_image())["feature_stack"]
    mask = np.zeros(stack.shape[1:], dtype=bool)
    mask[12:44, 12:42] = True
    vector = extract_region_feature_vector(stack, mask)
    prototype = GraphemePrototype("A", vector, np.ones_like(vector))
    decoded = GraphemePopulationDecoder().process(stack, [prototype])
    assert decoded["prototype_count"] == 1
    assert decoded["recognition_status"] == "recognized_with_supplied_prototypes"
    assert all("position" in item for item in decoded["position_preserving_sequence"])


def test_lif_network_advances_and_returns_physiological_units() -> None:
    """Advance current-based neural populations and ensure voltage/current/rate remain finite."""
    config = BNBEAConfig(lif_steps_per_stage=4)
    simulator = NeuralDynamicsSimulator(config)
    simulator.advance({"retina_lgn": 1.0}, steps=40)
    snapshot = simulator.snapshot()
    assert snapshot["time_ms"] == pytest.approx(40.0)
    assert set(snapshot["populations"]) == set(POPULATIONS)
    assert all(np.isfinite(item["mean_voltage_mv"]) for item in snapshot["populations"].values())
    assert all(np.isfinite(item["mean_current_pa"]) for item in snapshot["populations"].values())


def test_visual_hierarchy_and_vwfa_keep_hierarchical_and_ordered_representations() -> None:
    """Check multi-level visual maps and that order-sensitive matching distinguishes letter swaps."""
    stack = GaborVisualProcessor().process(_image())["feature_stack"]
    hierarchy = VisualHierarchyProcessor().process(stack)
    assert {"v1", "v2", "v4", "ventral_temporal", "dorsal_spatial"}.issubset(hierarchy["layers"])
    vwfa = VWFAHierarchyProcessor().process(["f", "o", "r", "m"], lexicon=[LexicalEntry("form")])
    assert vwfa["order_preserved"] is True
    assert vwfa["best_lexical_candidate"] == "form"
    assert ordered_similarity("form", "from") < 1.0


def test_auditory_stream_produces_separate_dorsal_and_ventral_features() -> None:
    """Run a synthetic sound through STFT and cochlear-inspired frequency pooling."""
    times = np.arange(4096) / 16000.0
    waveform = np.sin(2 * np.pi * 240 * times)
    output = AuditoryStreamProcessor().process(waveform, 16000)
    assert output["stft_power"].ndim == 2
    assert output["cochlear_band_energies"].shape[0] == 16
    assert output["ventral_comprehension_vector"].shape == output["dorsal_auditory_motor_vector"].shape


def test_eeg_meg_forward_and_regularized_inverse_have_expected_dimensions() -> None:
    """Verify lead-field projection and regularized source reconstruction algebra."""
    rng = np.random.default_rng(5)
    source = rng.normal(size=(80, 4))
    lead = rng.normal(size=(8, 4))
    sensor = forward_model(source, lead)
    reconstructed = inverse_model(sensor, lead, regularization=0.01)
    assert sensor.shape == (80, 8)
    assert reconstructed.shape == source.shape
    assert np.all(np.isfinite(reconstructed))


def test_fmri_rdm_and_glm_metrics_are_finite() -> None:
    """Check representational geometry, correlation and least-squares GLM fitting."""
    patterns = np.asarray([[0, 0], [1, 0], [0, 2]], dtype=float)
    rdm = build_rdm(patterns)
    assert rdm.shape == (3, 3)
    assert rdm_correlation(rdm, rdm) == pytest.approx(1.0)
    design = np.column_stack((np.ones(10), np.linspace(0, 1, 10)))
    observed = design @ np.asarray([2.0, 3.0])
    result = fit_glm(design, observed)
    assert result["residual_mse"] < 1e-10


def test_lesion_model_and_network_efficiency_are_bounded() -> None:
    """Apply a virtual region lesion and fit a supplied lesion-symptom model."""
    activity = {"VWFA": np.ones(5), "ATL": np.full(5, 2.0)}
    result = apply_virtual_lesion(activity, {"VWFA": 0.5})
    assert np.mean(result["lesioned_activity"]["VWFA"]) == pytest.approx(0.5)
    assert 0.0 <= graph_global_efficiency(np.asarray([[0, 1], [1, 0]], dtype=float)) <= 1.0
    regression = lesion_symptom_regression(np.asarray([[0], [0.5], [1.0]]), np.asarray([1, 0.5, 0]))
    assert regression["residual_mse"] < 1e-10


def test_virtual_tms_is_unitless_and_target_specific() -> None:
    """Simulate active, sham and control activity without creating human TMS parameters."""
    source = np.ones((40, 3), dtype=float)
    result = simulate_virtual_tms(source, ("VWFA", "MTG", "ATL"), "VWFA", "ATL", strength=0.4)
    assert result["active_target"].shape == source.shape
    assert np.allclose(result["sham"], source)
    assert result["strength_unitless"] == pytest.approx(0.4)
    assert result["active_minus_sham_activity_effect"] < 0.0


def test_human_validation_waits_for_data_and_splits_subjects_without_leakage() -> None:
    """Require real independent inputs before scoring and enforce subject-disjoint partitions."""
    validator = HumanReadingValidation()
    assert validator.process(None)["status"] == "awaiting_independent_human_data"
    splits = split_subjects([f"S{i}" for i in range(10)])
    assert set(splits["train"]).isdisjoint(splits["validation"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert all(splits[part] for part in ("train", "validation", "test"))
    assert all(split_subjects(["A", "B", "C"])[part] for part in ("train", "validation", "test"))
    score = validator.process({
        "human_accuracy": [1.0, 0.0, 1.0], "model_accuracy": [1.0, 1.0, 0.0],
        "human_reading_time_ms": [400, 500], "model_reading_time_ms": [420, 450],
    })
    assert score["status"] == "evaluated_supplied_independent_data"
    assert score["metrics"]["accuracy_mae"] == pytest.approx(2 / 3)


def test_integrated_pipeline_runs_all_twelve_operations_and_separates_evidence() -> None:
    """Run an end-to-end visual/audio example and preserve distinct functional/biological/experimental data."""
    times = np.arange(2048) / 16000.0
    request = BNBEARequest(
        sample_id="integrated-test",
        visual_image=_image(),
        auditory_waveform=np.sin(2 * np.pi * 220 * times),
        orthographic_sequence=("c", "a", "t"),
        lexicon=(LexicalEntry("cat", frequency=30.0),),
        semantic_features={"ANIMAL": 1.0},
        context_features={"PET": 0.7},
    )
    processor = BNBEAProcessor(
        BNBEAConfig(lif_steps_per_stage=3, hrf_duration_seconds=3.0),
        register_annotations=False,
    )
    result = processor.process(request)
    payload = result.to_dict()
    assert result.diagnostics["operation_count"] == 12
    assert result.diagnostics["operation_ids_complete"] is True
    assert result.functional["vwfa_hierarchy"]["best_lexical_candidate"] == "cat"
    assert result.biological["enabled"] is True
    assert result.experimental["human_reading_validation"]["status"] == "awaiting_independent_human_data"
    assert payload["functional"] is not payload["biological"]
    assert len(result.biological["source_time_series"]) > 0


def test_every_public_class_and_callable_has_a_docstring_for_sqlite_annotation_catalogue() -> None:
    """Require explicit docstrings so ore_logger.catalog stores reviewed descriptions, not inferred stubs."""
    package = Path(__file__).resolve().parents[1] / "ore_brain_networks"
    missing: list[str] = []
    for source in package.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if ast.get_docstring(node) is None:
                    missing.append(f"{source.name}:{node.lineno}:{node.name}")
    assert not missing, "Missing SQLite annotation docstrings: " + ", ".join(missing)


def test_bnbea_annotations_are_persisted_in_ore_sqlite_catalogue(tmp_path: Path) -> None:
    """Synchronize all BNBEA docstrings into the repository logger's SQLite annotations table."""
    try:
        from ore_logger import LoggerDatabase
    except ImportError:
        pytest.skip("ore_logger is available in the complete ORE repository, not this isolated test copy")
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "ore_logger.sqlite3"
    report = register_bnbea_annotations(root, database_path)
    database = LoggerDatabase(database_path)
    config_annotation = database.get_annotation(
        "ore_brain_networks.contracts", "BNBEAConfig", "class", "ore_brain_networks/contracts.py"
    )
    processor_annotation = database.get_annotation(
        "ore_brain_networks.pipeline", "BNBEAProcessor", "class", "ore_brain_networks/pipeline.py"
    )
    callable_annotation = database.get_annotation(
        "ore_brain_networks.visual_gabor", "gabor_kernel", "function", "ore_brain_networks/visual_gabor.py"
    )
    assert report["files_indexed"] >= 1
    assert config_annotation and config_annotation["annotation_source"] == "docstring"
    assert processor_annotation and processor_annotation["annotation_source"] == "docstring"
    assert callable_annotation and callable_annotation["annotation_source"] == "docstring"
    rows = database.search_annotations("ore_brain_networks", limit=1000)
    assert len(rows) >= 50
    assert all(row["annotation"].strip() for row in rows)
