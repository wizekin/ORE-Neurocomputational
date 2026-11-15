"""Functional, biological, validation, and SQLite tests for the BNBEA master layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ore_bnbea import (
    BNBEAConfig,
    BNBEAProcessor,
    BNBEARequest,
    BrainValidationInput,
    LIFNeuron,
    NeuralSimulationEngine,
    OPERATION_SPECS,
    STAGES,
    STAGE_OPERATION_COUNTS,
)
from ore_bnbea.math_ops import (
    bayesian_posterior,
    difference_of_gaussians,
    edit_distance,
    gabor_kernel,
    information_gain,
    ordinal_position_code,
    orientation_energy_bank,
    probability_entropy,
)
from ore_bnbea.validation import (
    bootstrap_confidence_interval,
    event_related_potential,
    forward_project_sources,
    ridge_inverse_sources,
    validate_brain_data,
)
from ore_logger.database import LoggerDatabase
from ore_n9 import N9Patch, N9Request


def test_registry_contains_all_detailed_operations_for_nine_stages():
    """The detailed source stage lists expand to a unique, stable 139-operation trace."""
    assert len(STAGES) == 9
    assert list(STAGE_OPERATION_COUNTS.values()) == [11, 12, 12, 14, 13, 16, 18, 19, 24]
    assert len(OPERATION_SPECS) == 139
    assert len({spec.operation_id for spec in OPERATION_SPECS}) == 139
    assert len({spec.name for spec in OPERATION_SPECS}) == 139


def test_lif_neuron_integrates_current_and_obeys_refractory_period():
    """A high current depolarizes the model neuron and the spike resets its membrane."""
    neuron = LIFNeuron()
    emitted = False
    for _ in range(20):
        emitted = neuron.advance(0.001, external_current_a=500e-12)
        if emitted:
            break
    assert emitted
    assert neuron.voltage_v == neuron.reset_potential_v
    assert neuron.refractory_remaining_s > 0
    assert neuron.advance(0.001, external_current_a=500e-12) is False


def test_recurrent_network_runs_and_updates_bounded_synapses():
    """The recurrent population model returns finite spike rasters and bounded weights."""
    config = BNBEAConfig(neurons_per_population=6, simulation_steps=8, seed=5)
    engine = NeuralSimulationEngine.build_default_network(config)
    current = np.full(6, config.external_current_gain_a, dtype=np.float64)
    history = engine.advance({
        name: current for name in engine.POPULATION_NAMES
    })
    assert set(history) == set(engine.POPULATION_NAMES)
    assert all(raster.shape == (8, 6) for raster in history.values())
    metrics = engine.apply_learning()
    assert all(np.isfinite(value) for value in metrics.values())
    assert all(
        np.isfinite(projection.weights_s).all()
        and np.all(projection.weights_s >= 0)
        and np.all(projection.weights_s <= config.weight_max_s)
        for projection in engine.projections
    )


def test_visual_orthographic_and_bayesian_math():
    """Visual filters, ordinal coding, edit distance, and Bayesian updates are numeric."""
    image = np.zeros((24, 24), dtype=np.float32)
    image[8:16, 11:13] = 1.0
    dog = difference_of_gaussians(image, 1.0, 2.0)
    assert dog.shape == image.shape
    kernel = gabor_kernel(9, 0.0, 0.15, 2.0)
    assert np.isclose(np.linalg.norm(kernel), 1.0, atol=1e-5)
    energy = orientation_energy_bank(image, orientations=(0.0, 1.57), kernel_size=9)
    assert energy.shape == (2, 24, 24)
    assert ordinal_position_code(4).shape == (4, 4)
    assert edit_distance(("c", "a", "t"), ("c", "u", "t")) == 1
    likelihood = np.asarray([[0.8, 0.2], [0.2, 0.8]])
    prior = np.asarray([0.9, 0.1])
    posterior = bayesian_posterior(likelihood, prior)
    assert np.allclose(posterior.sum(axis=1), 1.0)
    assert posterior[0, 0] > likelihood[0, 0]
    assert np.all(probability_entropy(posterior) >= 0)
    assert np.isfinite(information_gain(np.tile(prior, (2, 1)), posterior)).all()


def test_experimental_validation_helpers_and_missing_data_status():
    """Forward/inverse, ERP, bootstrap, and optional-modality statuses are reproducible."""
    rng = np.random.default_rng(9)
    source = rng.normal(size=(64, 3))
    leadfield = rng.normal(size=(6, 3))
    sensor = forward_project_sources(source, leadfield)
    inverse = ridge_inverse_sources(sensor, leadfield)
    assert sensor.shape == (64, 6)
    assert inverse.shape == (64, 3)
    trials = rng.normal(size=(4, 64, 6))
    assert event_related_potential(trials).shape == (64, 6)
    low, high = bootstrap_confidence_interval(np.arange(20, dtype=float), resamples=300)
    assert low <= high

    empty = validate_brain_data(None)
    assert empty["status"] == "not_tested"
    assert empty["evidence_status"] == "inconclusive"
    assert all(value == "not_tested" for value in empty["operation_statuses"].values())


def test_brain_data_validation_runs_only_available_measurements():
    """A caller-supplied multimodal sample yields metrics without claiming brain equivalence."""
    rng = np.random.default_rng(11)
    source = rng.normal(size=(64, 3))
    features = rng.normal(size=(5, 4))
    data = BrainValidationInput(
        source_activity=source,
        eeg_leadfield=rng.normal(size=(6, 3)),
        observed_eeg_trials=rng.normal(size=(4, 64, 6)),
        eeg_sampling_rate_hz=250.0,
        hrf=np.exp(-np.linspace(0, 2, 12)),
        observed_bold=rng.normal(size=64),
        design_matrix=rng.normal(size=(64, 3)),
        regional_observations=rng.normal(size=(64, 2)),
        model_features=features,
        brain_features=features + rng.normal(scale=0.1, size=features.shape),
        human_labels=("a", "b", "c"),
        predicted_labels=("a", "b", "d"),
        heldout_labels=("a", "c"),
        heldout_predictions=("a", "c"),
        bootstrap_values=rng.normal(size=24),
        lesion_baseline_score=0.90,
        lesion_perturbed_score=0.70,
        ablation_baseline_score=0.90,
        ablation_scores={"visual": 0.65},
    )
    report = validate_brain_data(data)
    assert report["status"] == "completed"
    assert report["evidence_status"] == "inconclusive"
    assert "eeg_forward_prediction" in report["metrics"]
    assert "regional_glm" in report["metrics"]
    assert report["metrics"]["behavioral_accuracy"] == 2 / 3
    assert report["operation_statuses"]["project_sources_to_eeg_sensors"] == "executed"


def test_bnbea_pipeline_logs_operations_and_python_symbol_annotations(tmp_path):
    """A full N9 pass and its BNBEA biology are indexed in the logger SQLite database."""
    database_path = tmp_path / "bnbea.sqlite3"
    n9_request = N9Request(
        sequence_id="bnbea-test",
        patches=[
            N9Patch("patch-0", grapheme_probabilities={"a": 0.85, "o": 0.15}),
            N9Patch("patch-1", grapheme_probabilities={"b": 0.90, "d": 0.10}),
        ],
        script_profile={"script": "Latin", "reading_direction": "ltr"},
    )
    config = BNBEAConfig(
        simulation_steps=6,
        neurons_per_population=5,
        seed=3,
    )
    processor = BNBEAProcessor(
        config=config,
        database_path=database_path,
        repository_root=Path(__file__).resolve().parents[1],
    )
    result = processor.process(BNBEARequest(
        n9_request=n9_request,
        context_priors={"a": 0.8, "o": 0.2, "b": 0.8, "d": 0.2},
    ))

    assert len(result.operation_trace) == 139
    assert len(result.functional_result.operation_trace) == 120
    assert result.diagnostics["stage_count"] == 9
    assert result.logger_metadata["master_operation_annotation_count"] == 139
    assert result.diagnostics["biological_acceptance"] == "not_established"

    database = LoggerDatabase(database_path)
    assert database.get_annotation("ore_bnbea.pipeline", "BNBEAProcessor", "class")
    assert database.get_annotation("ore_bnbea.pipeline", "BNBEAProcessor.process", "function")
    assert database.get_annotation(
        "ore_bnbea.operation_registry", "define_individual_neuron_models", "operation"
    )
    run = database.get_run(result.logger_metadata["run_id"])
    assert run is not None and run["status"] == "completed"
    assert run["event_count"] >= 139
    summary = database.summary()
    assert summary["classes"] > 0
    assert summary["functions"] > 0
