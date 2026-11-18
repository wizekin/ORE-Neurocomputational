"""Functional, biological-surrogate, rhythm, persistence, and SQLite tests for Point 03."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from ore_sleep_consolidation import (
    HippocampalTrace,
    ReplayOpportunity,
    SleepConsolidationConfig,
    SleepDependentSystemsConsolidationProcessor,
    OPERATION_REGISTRY,
    register_annotations,
)
from ore_sleep_consolidation.functional import (
    compressed_replay_times,
    detect_sharp_wave_ripple,
    detect_spindle,
    estimate_oscillation_phase,
    phase_locking_value,
    rhythm_alignment,
    stdp_timing_kernel,
)


def _trace(trace_id: str = "episode-a", features: tuple[float, ...] | None = None) -> HippocampalTrace:
    """Build a deterministic episodic trace with two ordered items."""
    vector = features or tuple(float(value) / 16.0 for value in range(1, 17))
    return HippocampalTrace(
        trace_id=trace_id,
        features=vector,
        sequence=(vector, tuple(reversed(vector))),
        recency=0.9,
        salience=0.8,
        value=0.7,
        goal_relevance=0.85,
        interference_risk=0.1,
        context_id="context-a",
        baseline_performance=0.55,
    )


def _opportunity(
    opportunity_id: str,
    phase: float = 0.0,
    sleep_state: str = "nrem",
    ripple: bool = True,
    spindle: bool = True,
    pre: float | None = None,
    post: float | None = None,
) -> ReplayOpportunity:
    """Build an offline event with optional SWR/spindle evidence and external performance."""
    return ReplayOpportunity(
        opportunity_id=opportunity_id,
        sleep_state=sleep_state,
        sleep_state_confidence=0.95,
        timestamp_ms=1000.0,
        slow_oscillation_phase_rad=phase,
        spindle_envelope=0.8 if spindle else 0.0,
        spindle_duration_ms=500.0 if spindle else 0.0,
        spindle_phase_rad=phase,
        ripple_band_z=2.2 if ripple else 0.0,
        sharp_wave_z=2.0 if ripple else 0.0,
        ripple_duration_ms=65.0 if ripple else 0.0,
        partial_cue=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        pre_post_interval_ms=12.0,
        observed_pre_performance=pre,
        observed_post_performance=post,
    )


def test_registry_has_all_eighteen_source_ordered_operations() -> None:
    """Protect the 18-stage sequence and require a mathematical and biological account."""
    assert len(OPERATION_REGISTRY) == 18
    assert [item.operation_id for item in OPERATION_REGISTRY] == [
        f"SC{index:02d}" for index in range(1, 19)
    ]
    assert all(item.mathematical_form and item.biological_reference and item.evidence_level
               for item in OPERATION_REGISTRY)


def test_event_detectors_keep_spindle_and_swr_evidence_distinct() -> None:
    """Verify spindle and ripple event detectors use separate observations and duration checks."""
    config = SleepConsolidationConfig()
    assert detect_spindle(0.8, 400.0, config)["detected"] is True
    assert detect_spindle(0.8, 50.0, config)["detected"] is False
    assert detect_sharp_wave_ripple(2.0, 2.0, 60.0, config)["detected"] is True
    assert detect_sharp_wave_ripple(2.0, 2.0, 500.0, config)["detected"] is False


def test_phase_estimation_and_timing_math_are_finite_and_bounded() -> None:
    """Check signal-derived phase, signed STDP timing, compressed time, and phase-locking."""
    sample_rate = 100.0
    time_axis = np.arange(500, dtype=np.float64) / sample_rate
    signal = np.sin(2.0 * np.pi * 0.8 * time_axis)
    phase_report = estimate_oscillation_phase(signal, sample_rate)
    assert -np.pi <= phase_report["phase_rad"] <= np.pi
    assert stdp_timing_kernel(10.0, 50.0) > 0.0
    assert stdp_timing_kernel(-10.0, 50.0) < 0.0
    assert stdp_timing_kernel(0.0, 50.0) == 1.0
    timestamps = compressed_replay_times(4, 0.25, 100.0)
    assert np.allclose(timestamps, [0.0, 25.0, 50.0, 75.0])
    assert phase_locking_value([0.0, 0.0, 0.0]) > 0.99


def test_unsuitable_sleep_state_or_missing_ripple_skips_replay_and_plasticity() -> None:
    """Prevent replay-stage changes when sleep state or SWR evidence is unavailable."""
    processor = SleepDependentSystemsConsolidationProcessor()
    processor.add_trace(_trace())
    weights_before = processor.network.cortical_weights.copy()
    wake = processor.process(_opportunity("wake", sleep_state="wake"))
    no_ripple = processor.process(_opportunity("no-ripple", ripple=False))
    assert wake.selected_trace_id is None and no_ripple.selected_trace_id is None
    assert all(row["status"] == "skipped" for row in wake.operation_trace[4:])
    assert all(row["status"] == "skipped" for row in no_ripple.operation_trace[4:])
    assert np.allclose(processor.network.cortical_weights, weights_before)
    assert processor.network.plasticity_event_count == 0


def test_nrem_replay_runs_all_stages_and_updates_separate_trace_state() -> None:
    """Run an aligned replay and verify all 18 outputs, sequence replay, retrieval, and logging flags."""
    processor = SleepDependentSystemsConsolidationProcessor()
    processor.add_trace(_trace())
    result = processor.process(_opportunity("replay-1", pre=0.50, post=0.63))
    assert len(result.operation_trace) == 18
    assert [row["operation_id"] for row in result.operation_trace] == [
        item.operation_id for item in OPERATION_REGISTRY
    ]
    assert result.selected_trace_id == "episode-a"
    assert result.rhythm_events["spindle"]["detected"] is True
    assert result.rhythm_events["ripple"]["detected"] is True
    assert result.alignment["score"] > 0.0
    assert len(result.replay_sequence) == 2
    assert result.replay_timestamps_ms == [0.0, 25.0]
    assert result.retrieval_support["status"] == "retrieval_probe_completed"
    assert result.validation["retention_change"] == 0.13
    assert result.validation["causal_improvement_claimed"] is False
    assert result.logger["enabled"] is False
    assert processor.replay_counts["episode-a"] == 1
    assert processor.consolidation_indices["episode-a"] > 0.0


def test_reverse_replay_changes_order_and_low_phase_alignment_reduces_plasticity_gate() -> None:
    """Verify sequence direction and the modulatory effect of slow-phase/spindle alignment."""
    processor = SleepDependentSystemsConsolidationProcessor()
    processor.add_trace(_trace())
    poor = ReplayOpportunity(
        opportunity_id="poor-coupling",
        sleep_state="nrem",
        sleep_state_confidence=0.95,
        slow_oscillation_phase_rad=np.pi,
        spindle_envelope=0.8,
        spindle_duration_ms=500.0,
        spindle_phase_rad=0.0,
        ripple_band_z=2.2,
        sharp_wave_z=2.0,
        ripple_duration_ms=65.0,
        replay_direction="reverse",
    )
    result = processor.process(poor)
    assert result.replay_sequence[0] != result.replay_sequence[-1]
    assert result.alignment["score"] < 0.01
    assert result.plasticity["status"] == "no_update_weak_rhythm_coupling"


def test_replay_across_opportunities_accumulates_weights_and_homeostasis() -> None:
    """Ensure repeated replay modifies connections gradually and keeps neural surrogates bounded."""
    processor = SleepDependentSystemsConsolidationProcessor()
    processor.add_trace(_trace())
    initial = processor.network.cortical_weights.copy()
    first = processor.process(_opportunity("night-1"))
    second = processor.process(_opportunity("night-2"))
    third = processor.process(_opportunity("night-3"))
    assert processor.network.plasticity_event_count == 3
    assert processor.replay_counts["episode-a"] == 3
    assert np.all(np.isfinite(processor.network.cortical_weights))
    assert np.max(np.abs(processor.network.cortical_weights)) <= processor.config.max_abs_weight
    assert np.all(processor.network.homeostatic_gain >= processor.config.homeostatic_gain_min)
    assert np.all(processor.network.homeostatic_gain <= processor.config.homeostatic_gain_max)
    assert np.any(np.abs(processor.network.cortical_weights - initial) > 0.0)
    assert third.consolidation_state["updated_index"] >= second.consolidation_state["updated_index"]
    assert first.integration["trace_id"] == "episode-a"


def test_distinct_episodes_keep_independent_cortical_prototypes() -> None:
    """Preserve episode identity while allowing the network to quantify cross-trace overlap."""
    processor = SleepDependentSystemsConsolidationProcessor()
    one = _trace("one")
    two = _trace("two", tuple(reversed(one.features)))
    processor.add_trace(one)
    processor.add_trace(two)
    processor.process(_opportunity("one-replay"))
    assert set(processor.network.cortical_memories) == {"one", "two"}
    assert not np.allclose(processor.network.cortical_memories["one"],
                           processor.network.cortical_memories["two"])


def test_all_package_classes_and_functions_have_docstrings() -> None:
    """Require a human-readable annotation for every package function, nested helper, and class."""
    package_root = Path(__file__).resolve().parents[1] / "ore_sleep_consolidation"
    for source_path in package_root.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                assert ast.get_docstring(node), f"missing docstring: {source_path.name}:{node.name}"


def test_package_annotations_are_indexed_in_logger_sqlite(tmp_path: Path) -> None:
    """Verify Point 03 symbols are discoverable through the repository's SQLite annotation catalog."""
    from ore_logger.database import LoggerDatabase

    repository_root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "ore_point03_annotations.sqlite"
    report = register_annotations(database_path, repository_root)
    assert isinstance(report, dict)
    database = LoggerDatabase(database_path)
    entry = database.get_annotation(
        "ore_sleep_consolidation.pipeline",
        "SleepDependentSystemsConsolidationProcessor.process",
        "function",
    )
    assert entry is not None
    assert "18" in entry["annotation"] or "offline" in entry["annotation"].lower()
