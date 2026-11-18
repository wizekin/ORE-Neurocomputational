"""Tests for Point 05 novelty-modulated hippocampal consolidation."""

from __future__ import annotations

from pathlib import Path

import pytest

from ore_novelty_consolidation import (
    BiologicalConsolidationNetwork,
    CompetingTrace,
    ConsolidationConfig,
    ConsolidationRequest,
    InternalState,
    MemoryTrace,
    NoveltyModulatedHippocampalConsolidationProcessor,
    PostLearningEvent,
    SleepEvidence,
    register_annotations,
    registry_as_dicts,
)
from ore_novelty_consolidation.functional import (
    classify_novelty,
    decay_state,
    estimate_interference,
    hill_occupancy,
    softmax_probabilities,
    tagged_resource_capture,
    temporal_eligibility,
)
from ore_logger.database import LoggerDatabase


def _trace(trace_id: str = "episode-1") -> MemoryTrace:
    """Construct a small, two-synapse hippocampal trace for deterministic tests."""
    return MemoryTrace(
        trace_id=trace_id,
        hippocampal_pattern=(1.0, 0.0, 0.0),
        encoding_time_ms=0.0,
        active_synapses={"syn-1": 0.8, "syn-2": 0.6},
        synaptic_changes={"syn-1": -0.10, "syn-2": 0.20},
        synapse_feature_indices={"syn-1": 0, "syn-2": 1},
    )


def _request(
    trace: MemoryTrace | None = None,
    *,
    event_id: str = "novel-after-learning",
    event_time_ms: float = 120.0,
    competitor: bool = False,
    sleep: SleepEvidence | None = None,
    with_event: bool = True,
) -> ConsolidationRequest:
    """Create a request with controlled novelty, cue, context, and optional sleep/competitors."""
    memory = trace or _trace()
    state = InternalState(
        prior_knowledge_patterns=((0.0, 1.0, 0.0),),
        predicted_features=(0.0, 1.0, 0.0),
        goal_relevance=0.8,
        arousal=0.55,
        motivational_value=0.7,
        reward_expectancy=0.6,
        stress=0.0,
    )
    events = (
        PostLearningEvent(
            event_id=event_id,
            timestamp_ms=event_time_ms,
            features=(1.0, 0.0, 0.0),
            sensory_intensity=0.6,
            reward_value=0.5,
        ),
    ) if with_event else ()
    competitors = (
        CompetingTrace(
            trace_id="overlapping-episode",
            hippocampal_pattern=(1.0, 0.0, 0.0),
            strength=0.8,
            conflict=1.0,
        ),
    ) if competitor else ()
    return ConsolidationRequest(
        trace=memory,
        post_learning_events=events,
        internal_state=state,
        sleep_evidence=sleep,
        competing_traces=competitors,
        retrieval_cue=(1.0, 0.0, 0.0),
        context_match=0.9,
    )


def test_registry_contains_all_sixteen_source_ordered_operations() -> None:
    """Verify the 16 canonical stages preserve the Point 05 order and identifiers."""
    operations = registry_as_dicts()
    assert len(operations) == 16
    assert [item["operation_id"] for item in operations] == [
        f"NC{index:02d}" for index in range(1, 17)
    ]
    assert "VTA" in operations[5]["biological_reference"]
    assert "LC" in operations[5]["biological_reference"]
    assert "parallel" in operations[14]["name"].lower()


def test_contract_rejects_inconsistent_feature_dimensions() -> None:
    """Reject event, retrieval, or competitor feature vectors that cannot match the trace."""
    memory = _trace()
    with pytest.raises(ValueError, match="event features"):
        ConsolidationRequest(
            trace=memory,
            post_learning_events=(
                PostLearningEvent("bad", 1.0, (1.0, 0.0)),
            ),
        )
    with pytest.raises(ValueError, match="retrieval_cue"):
        ConsolidationRequest(trace=memory, retrieval_cue=(1.0, 0.0))
    late_trace = MemoryTrace(
        trace_id="late-encoding",
        hippocampal_pattern=(1.0, 0.0, 0.0),
        encoding_time_ms=100.0,
        active_synapses={"syn-1": 0.8},
    )
    with pytest.raises(ValueError, match="must occur at or after encoding"):
        ConsolidationRequest(
            trace=late_trace,
            post_learning_events=(PostLearningEvent("early", 50.0, (1.0, 0.0, 0.0)),),
        )


def test_novelty_classifier_distinguishes_familiar_common_and_distinct() -> None:
    """Classify a familiar pattern, related novelty, and a dissimilar novel pattern."""
    prior = ((1.0, 0.0, 0.0),)
    familiar = classify_novelty(
        (1.0, 0.0, 0.0), prior, (1.0, 0.0, 0.0), 0.18, 0.60
    )
    common = classify_novelty(
        (0.5, 0.8660254037844386, 0.0), prior, None, 0.18, 0.60
    )
    distinct = classify_novelty(
        (0.0, 1.0, 0.0), prior, None, 0.18, 0.60
    )
    assert familiar["novelty_type"] == "familiar"
    assert common["novelty_type"] == "common"
    assert distinct["novelty_type"] == "distinct"
    assert common["novelty_score"] > familiar["novelty_score"]


def test_missing_prior_knowledge_does_not_fabricate_a_novelty_class() -> None:
    """Return an uncertain classification when neither a prior nor a prediction is supplied."""
    result = classify_novelty((1.0, 0.0), (), None, 0.18, 0.60)
    assert result["novelty_type"] == "undetermined"
    assert result["classification_confident"] is False
    assert result["familiarity"] is None


def test_temporal_and_biochemical_decay_are_monotonic() -> None:
    """Verify exponential eligibility and state decay diminish with elapsed time."""
    assert temporal_eligibility(100.0, 100.0) == pytest.approx(0.367879, rel=1e-5)
    assert temporal_eligibility(200.0, 100.0) < temporal_eligibility(100.0, 100.0)
    assert decay_state(0.8, 100.0, 100.0) == pytest.approx(0.8 * temporal_eligibility(100.0, 100.0))


def test_receptor_occupancy_and_tagged_capture_behave_nonlinearly_and_locally() -> None:
    """Check Hill occupancy saturation and zero capture without an eligible tag."""
    assert hill_occupancy(0.0, 0.5, 2.0) == 0.0
    assert hill_occupancy(5.0, 0.5, 2.0) > hill_occupancy(0.5, 0.5, 2.0)
    assert tagged_resource_capture(0.0, 1.0, 1.0) == 0.0
    assert tagged_resource_capture(0.8, 0.5, 0.5) == pytest.approx(0.2)


def test_softmax_probabilities_are_normalized_and_temperature_sensitive() -> None:
    """Verify retrieval competition returns a distribution and sharper choice at low temperature."""
    diffuse = softmax_probabilities((1.0, 0.5, 0.0), 1.0)
    sharp = softmax_probabilities((1.0, 0.5, 0.0), 0.1)
    assert sum(diffuse) == pytest.approx(1.0)
    assert sum(sharp) == pytest.approx(1.0)
    assert sharp[0] > diffuse[0]


def test_vta_and_lc_populations_remain_distinct_and_lc_contributes_dopamine() -> None:
    """Test rate-coded pathway separation and the modeled LC contribution to hippocampal DA."""
    network = BiologicalConsolidationNetwork(ConsolidationConfig(input_size=3))
    state = InternalState(
        prior_knowledge_patterns=((0.0, 1.0, 0.0),),
        goal_relevance=0.8,
        arousal=0.6,
    )
    result = network.recruit_and_modulate(
        {"novelty_score": 0.8, "novelty_type": "distinct"},
        0.9,
        state,
        0.5,
        100.0,
    )
    pathway = result["pathways"]
    modulators = result["neuromodulators"]
    assert len(pathway["vta_population_activity"]) == network.config.vta_units
    assert len(pathway["lc_population_activity"]) == network.config.lc_units
    assert pathway["vta_mean_activity"] != pathway["lc_mean_activity"]
    assert modulators["hippocampal_dopamine_estimate"] > 0.0
    assert modulators["hippocampal_noradrenaline_estimate"] > 0.0
    assert modulators["lc_dopamine_contribution_estimate"] > 0.0


def test_processor_runs_all_sixteen_stages_and_stabilizes_only_tagged_synapses() -> None:
    """Run the full sequence and check signed plasticity, tagging specificity, and provenance."""
    processor = NoveltyModulatedHippocampalConsolidationProcessor(
        ConsolidationConfig(input_size=3)
    )
    result = processor.process(_request())
    assert len(result.operation_trace) == 16
    assert [row["operation_id"] for row in result.operation_trace] == [
        f"NC{index:02d}" for index in range(1, 17)
    ]
    assert all(row["operation"] for row in result.operation_trace)
    assert result.operation_trace[9]["functional_output"]["captured_synapse_count"] > 0
    changes = result.final_trace["synaptic_changes"]
    assert changes["syn-1"] < 0.0
    assert changes["syn-2"] > 0.0
    assert result.final_trace["hippocampal_persistence"] >= 0.25
    assert result.novelty["events"][0]["novelty_type"] == "distinct"
    assert result.multidimensional_state["dimensions_are_not_collapsed"] is True
    assert len(result.final_trace["provenance"]) == 1


def test_processing_the_same_event_twice_is_idempotent() -> None:
    """Avoid applying novelty-related resource capture more than once for one event identifier."""
    processor = NoveltyModulatedHippocampalConsolidationProcessor(
        ConsolidationConfig(input_size=3)
    )
    first = processor.process(_request())
    after_first = first.final_trace["synaptic_persistence"]
    second = processor.process(_request())
    assert second.novelty["events"] == []
    assert second.final_trace["synaptic_persistence"] == after_first
    assert len(second.final_trace["provenance"]) == 1
    assert processor.snapshot()["processed_event_counts"]["episode-1"] == 1


def test_competing_trace_increases_interference_without_deleting_latent_memory() -> None:
    """Keep both traces and latent persistence while representing cue competition separately."""
    no_competition = NoveltyModulatedHippocampalConsolidationProcessor(
        ConsolidationConfig(input_size=3)
    ).process(_request(competitor=False))
    with_competition = NoveltyModulatedHippocampalConsolidationProcessor(
        ConsolidationConfig(input_size=3)
    ).process(_request(competitor=True))
    assert with_competition.interference["risk"] > no_competition.interference["risk"]
    assert with_competition.multidimensional_state["hippocampal_trace_persistence"] == pytest.approx(
        no_competition.multidimensional_state["hippocampal_trace_persistence"]
    )
    assert with_competition.retrieval["target_retrieval_probability"] < (
        no_competition.retrieval["target_retrieval_probability"]
    )
    assert with_competition.interference["latent_trace_persistence_unchanged_by_competition"] is True


def test_sleep_updates_cortical_integration_without_erasing_hippocampal_trace() -> None:
    """Allow NREM replay to support cortex while leaving hippocampal persistence as a separate axis."""
    sleep = SleepEvidence(
        sleep_state="nrem",
        state_confidence=1.0,
        slow_oscillation_strength=1.0,
        spindle_strength=1.0,
        ripple_strength=1.0,
        replay_strength=0.9,
        coordination_strength=1.0,
        prior_knowledge_support=0.9,
    )
    processor = NoveltyModulatedHippocampalConsolidationProcessor(
        ConsolidationConfig(input_size=3)
    )
    memory = _trace()
    before = processor.process(_request(memory, with_event=False)).final_trace
    after = processor.process(_request(memory, with_event=False, sleep=sleep))
    assert after.sleep["cortical_integration_delta"] > 0.0
    assert after.multidimensional_state["cortical_integration"] > before["cortical_integration"]
    assert after.multidimensional_state["hippocampal_trace_persistence"] == pytest.approx(
        before["hippocampal_persistence"]
    )
    assert after.sleep["hippocampal_persistence_unchanged_by_sleep_stage"] is True


def test_wake_or_missing_sleep_evidence_does_not_fabricate_replay() -> None:
    """Skip unavailable replay evidence and avoid incrementing the cortical integration state."""
    processor = NoveltyModulatedHippocampalConsolidationProcessor(
        ConsolidationConfig(input_size=3)
    )
    result = processor.process(_request(with_event=False))
    assert result.sleep["status"] == "not_supplied"
    assert result.sleep["cortical_integration_delta"] == 0.0
    assert result.operation_trace[14]["status"] == "skipped_no_sleep_evidence"


def test_interference_function_reports_overlapping_competitors() -> None:
    """Quantify cue overlap while leaving the target representation untouched."""
    result = estimate_interference(
        (1.0, 0.0),
        [((1.0, 0.0), 0.8, 1.0), ((0.0, 1.0), 0.8, 1.0)],
    )
    assert result["risk"] > 0.0
    assert result["competitors"][0]["overlap"] == pytest.approx(1.0)
    assert result["competitors"][1]["overlap"] == pytest.approx(0.0)


def test_annotation_catalog_indexes_point05_classes_and_callables(tmp_path: Path) -> None:
    """Register package docstrings and verify the logger can resolve the processor annotations."""
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "ore_point05_annotations.sqlite"
    report = register_annotations(database_path, root)
    assert report["parse_errors"] == 0
    database = LoggerDatabase(database_path)
    expected = (
        ("ore_novelty_consolidation.pipeline",
         "NoveltyModulatedHippocampalConsolidationProcessor", "class"),
        ("ore_novelty_consolidation.pipeline",
         "NoveltyModulatedHippocampalConsolidationProcessor.process", "function"),
        ("ore_novelty_consolidation.biology", "BiologicalConsolidationNetwork", "class"),
        ("ore_novelty_consolidation.biology",
         "HippocampalPlasticityModel.sleep_cortical_integration", "function"),
        ("ore_novelty_consolidation.contracts", "MemoryTrace", "class"),
        ("ore_novelty_consolidation.registry", "OperationSpec", "class"),
    )
    for module, qualname, kind in expected:
        row = database.get_annotation(module, qualname, kind)
        assert row is not None and row["annotation"].strip()
        assert row["annotation_source"] == "docstring"
    assert database.summary()["symbols"] >= 40


def test_processor_records_sixteen_sqlite_operation_events(tmp_path: Path) -> None:
    """Ensure a full Point 05 run writes all 16 operation events with reusable annotations."""
    database_path = tmp_path / "ore_point05_run.sqlite"
    root = Path(__file__).resolve().parents[1]
    processor = NoveltyModulatedHippocampalConsolidationProcessor(
        ConsolidationConfig(input_size=3),
        database_path=database_path,
        repository_root=root,
    )
    result = processor.process(_request())
    assert result.logger["enabled"] is True
    database = LoggerDatabase(database_path)
    runs = database.list_runs()
    assert len(runs) == 1 and runs[0]["status"] == "completed"
    events = database.list_events(runs[0]["run_id"])
    operations = [event for event in events if event["event_kind"] == "operation"]
    assert len(operations) == 16
    assert all(event.get("annotation") for event in operations)
