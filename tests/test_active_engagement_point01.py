"""Contract, functional, biological, pipeline, and SQLite tests for Point 01."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from ore_active_engagement import (
    ActiveEngagementConfig,
    ActiveEngagementProcessor,
    ActiveEngagementRequest,
    ActiveEngagementState,
    BiologicalActiveEngagementNetwork,
    LearningSegment,
    NeuralPopulation,
    OPERATION_REGISTRY,
    SynapticPlasticityNetwork,
    allocate_attention,
    generate_questions,
    segment_material,
)
from ore_active_engagement.contracts import ActiveEngagementResult
from ore_active_engagement.functional import (
    extract_propositions,
    shannon_entropy,
    softmax,
)


def build_request(**overrides) -> ActiveEngagementRequest:
    """Create a reproducible active-engagement request for a short science explanation."""
    values = {
        "learner_id": "learner-test",
        "goal": "understand active engagement, questions, attention, and explanation",
        "material": (
            "Active engagement changes how the learner processes information.",
            "Questions direct attention toward uncertainty and relevant evidence.",
            "Self-explanation reveals gaps and supports richer encoding.",
        ),
        "learner_explanation": (
            "Active engagement uses questions and self-explanation to direct attention, "
            "identify gaps, and create a more structured memory."
        ),
        "required_concepts": ("questions", "attention", "explanation", "gaps"),
        "listener_level": "novice",
        "enable_teaching": True,
        "enable_imagery": True,
    }
    values.update(overrides)
    return ActiveEngagementRequest(**values)


def test_registry_contains_all_17_source_ordered_steps():
    """Keep registry identifiers and source-step numbering aligned with the algorithm file."""
    assert len(OPERATION_REGISTRY) == 17
    assert [item.operation_id for item in OPERATION_REGISTRY] == [
        f"P01-{index:02d}" for index in range(1, 18)
    ]
    assert [item.source_step for item in OPERATION_REGISTRY] == list(range(17))
    assert all(item.mathematical_form for item in OPERATION_REGISTRY)
    assert all(item.biological_reference for item in OPERATION_REGISTRY)


def test_softmax_is_normalized_and_entropy_is_nonnegative():
    """Verify competitive normalization and entropy boundary cases."""
    probabilities = softmax((1.0, 2.0, 3.0), temperature=0.7)
    assert sum(probabilities) == pytest.approx(1.0)
    assert all(0.0 < item < 1.0 for item in probabilities)
    assert shannon_entropy(probabilities) >= 0.0
    assert shannon_entropy(()) == 0.0


def test_material_segmentation_retains_source_provenance():
    """Keep sentence fragments traceable to their original segment identifiers."""
    units = segment_material((
        LearningSegment("s-1", "First statement. Second statement.", reliability=0.9),
    ))
    assert len(units) == 2
    assert {item["segment_id"] for item in units} == {"s-1"}
    propositions = extract_propositions(units)
    assert [item["proposition_id"] for item in propositions] == ["prop-001", "prop-002"]
    assert all(item["explicit_source_statement"] for item in propositions)


def test_question_and_attention_generation_produce_bounded_weights():
    """Ensure active questions are ranked and selected evidence has normalized attention."""
    request = build_request()
    config = ActiveEngagementConfig()
    from ore_active_engagement.functional import select_target

    units = segment_material(request.material)
    propositions = extract_propositions(units)
    target = select_target(propositions, request.goal, config)["target"]
    questions = generate_questions(target, propositions, request.goal, 4, config)
    assert questions
    assert questions[0]["utility"] >= questions[-1]["utility"]
    attention = allocate_attention(questions[0], units, config)
    assert attention["weights_sum"] == pytest.approx(1.0)
    assert attention["selected_evidence"]


def test_missing_learner_explanation_is_not_misrepresented_as_generation():
    """Keep machine-created extractive scaffolds distinct from learner-generated explanations."""
    request = build_request(learner_explanation=None)
    processor = ActiveEngagementProcessor(request.learner_id)
    result = processor.process(request)
    assert result.explanation["origin"] == "extractive_scaffold"
    assert result.explanation["is_learner_generated"] is False
    assert result.closure["episode_status"] == "incomplete_learner_explanation_required"
    assert result.closure["stop_condition_met"] is False


def test_full_pipeline_emits_17_ordered_operations_and_optional_outputs():
    """Run all source stages and inspect the functional and biological result channels."""
    request = build_request()
    result = ActiveEngagementProcessor(request.learner_id).process(request)
    assert isinstance(result, ActiveEngagementResult)
    assert len(result.operation_trace) == 17
    assert [item["operation_id"] for item in result.operation_trace] == [
        item.operation_id for item in OPERATION_REGISTRY
    ]
    assert result.explanation["origin"] == "learner_generated"
    assert result.teaching["enabled"] is True
    assert result.imagery["enabled"] is True
    assert result.episode_memory["hippocampal_binding"]["source_context_binding_preserved"]
    assert result.plasticity["weights_bounded"] is True
    assert result.output["spaced_repetition_created"] is False
    assert result.output["formal_testing_program_created"] is False
    assert result.biological["literal_neuron_simulation"] is False


def test_disabled_teaching_and_imagery_are_honored():
    """Treat teaching and imagery as optional pathways rather than mandatory output stages."""
    request = build_request(enable_teaching=False, enable_imagery=False)
    result = ActiveEngagementProcessor(request.learner_id).process(request)
    assert result.teaching["enabled"] is False
    assert result.imagery["enabled"] is False
    assert result.episode_memory["imagery_bound"] is False


def test_neural_population_has_recurrent_bounded_activity():
    """Check that the neural-population surrogate updates its state with finite activations."""
    population = NeuralPopulation("test-population", dimensions=16)
    first = population.update(np.ones(16), gain=1.0, inhibition=0.1)
    second = population.update(np.ones(16), gain=1.0, inhibition=0.1)
    assert first.shape == (16,)
    assert second.shape == (16,)
    assert np.all(np.isfinite(second))
    assert np.all(np.abs(second) <= 1.0)
    assert population.snapshot()["processing_count"] == 2


def test_synaptic_plasticity_models_potentiation_and_depression_with_bounds():
    """Verify timing sign, bounded weights, and optional persistent associative-state updates."""
    network = SynapticPlasticityNetwork(dimensions=16)
    state = ActiveEngagementState("learner-test")
    pre = np.linspace(-1.0, 1.0, 16)
    post = np.roll(pre, 2)
    potentiation = network.update(
        pre, post, learning_rate=0.05, gate=0.8,
        delta_t_seconds=0.01, state=state,
    )
    depression = network.update(
        pre, post, learning_rate=0.05, gate=0.8,
        delta_t_seconds=-0.01, state=state,
    )
    assert potentiation["timing_direction"] == "potentiation"
    assert depression["timing_direction"] == "depression"
    assert potentiation["weights_bounded"] and depression["weights_bounded"]
    assert len(state.associative_weights) > 0
    assert all(0.0 <= value <= 1.0 for value in state.associative_weights.values())


def test_biological_network_exposes_all_distributed_surrogates():
    """Ensure temporal, attention, control, hippocampal, and plasticity modules are available."""
    network = BiologicalActiveEngagementNetwork(dimensions=16)
    snapshot = network.snapshot()
    assert set(snapshot) >= {
        "temporal_language", "frontoparietal_attention", "prefrontal_control",
        "hippocampal_episode", "synaptic_plasticity",
    }
    assert snapshot["literal_neuron_simulation"] is False


def test_request_contract_rejects_invalid_material_and_configuration():
    """Reject empty segments and impossible numerical model settings early."""
    with pytest.raises(ValueError):
        ActiveEngagementRequest(
            learner_id="learner", goal="learn", material=()
        )
    with pytest.raises(ValueError):
        ActiveEngagementConfig(attention_temperature=0.0)
    with pytest.raises(ValueError):
        LearningSegment("s-1", "")


def test_sqlite_registers_every_package_definition_and_logs_17_events(tmp_path):
    """Index Point 01 classes/functions and persist the full ordered trace in shared ORE SQLite."""
    from ore_active_engagement import record_result_to_sqlite
    from ore_logger.database import LoggerDatabase

    request = build_request()
    result = ActiveEngagementProcessor(request.learner_id).process(request)
    database_path = tmp_path / "point01.sqlite3"
    repository_root = Path(__file__).resolve().parents[1]

    report = record_result_to_sqlite(
        result, database_path, repository_root=repository_root
    )
    database = LoggerDatabase(database_path)
    annotations = database.search_annotations("ore_active_engagement", limit=1000)
    class_rows = [item for item in annotations if item["symbol_kind"] == "class"]
    function_rows = [
        item for item in annotations
        if item["symbol_kind"] in {"function", "async_function"}
    ]
    assert class_rows
    assert function_rows
    assert all(item["annotation"].strip() for item in class_rows + function_rows)
    assert all(item["source_docstring"] for item in class_rows + function_rows)

    with sqlite3.connect(database_path) as connection:
        run = connection.execute(
            "SELECT status, event_count FROM runs WHERE run_id = ?",
            (report["run_id"],),
        ).fetchone()
        events = connection.execute(
            "SELECT sequence_no, event_kind, metadata_json FROM events "
            "WHERE run_id = ? ORDER BY sequence_no",
            (report["run_id"],),
        ).fetchall()
    assert run == ("completed", 17)
    assert len(events) == 17
    assert [item[0] for item in events] == list(range(1, 18))
    assert all(item[1] == "operation" for item in events)
    assert all("P01-" in item[2] for item in events)
    summary = database.summary()
    assert summary["classes"] >= len(class_rows)
    assert summary["functions"] >= len(function_rows)
