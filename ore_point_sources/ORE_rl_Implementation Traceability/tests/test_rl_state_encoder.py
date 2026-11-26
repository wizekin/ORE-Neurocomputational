"""Deterministic state encoding: stability, normalization, missingness (Steps 7-14)."""
import math


from ore_rl.state_encoder import (
    CapabilityStateSignal,
    DependencyStateSignal,
    GoalProgressSignal,
    RawStateSignals,
    ResourceStateSignal,
    StateEncoder,
    TaskContextSignal,
    UncertaintyHistorySignal,
    WorkspaceEvidenceSignal,
)


def _signals(step: int = 0, difficulty: float = 0.5) -> RawStateSignals:
    return RawStateSignals(
        task=TaskContextSignal(
            task_family="fact_verification",
            domain="science",
            requested_output="verified_claim",
            difficulty=difficulty,
            urgency=0.5,
            constraints=("verify_before_answer",),
        ),
        goal=GoalProgressSignal(
            completed_subgoals=("cognition.rl.input_analysis",),
            remaining_subgoals=("cognition.rl.verification",),
            unresolved_requirements=("evidence_gap",),
            completion_confidence=0.4,
        ),
        workspace=WorkspaceEvidenceSignal(
            workspace_revision=step,
            finding_count=3,
            verified_finding_count=1,
            unverified_finding_count=2,
            contradiction_count=0,
            mean_source_reliability=0.7,
            evidence_quality_score=0.5,
            evidence_quantity_score=0.5,
        ),
        capabilities=CapabilityStateSignal(
            module_ids=("m1", "m2"),
            readiness=(1.0, 0.0),
            health=(0.9, 0.8),
            queue_lengths=(0, 1),
            estimated_cost=(1.0, 2.0),
        ),
        dependencies=DependencyStateSignal(
            action_ids=("a", "b", "c"),
            dependency_satisfied=(True, False, True),
        ),
        resources=ResourceStateSignal(
            remaining_wall_clock_fraction=0.8,
            remaining_cpu_fraction=0.7,
            remaining_memory_fraction=0.6,
            remaining_module_calls_fraction=0.5,
            reserved_wall_clock_fraction=0.1,
        ),
        uncertainty=UncertaintyHistorySignal(
            evaluator_confidence=0.75,
            action_sample_counts={"a": 2, "b": 0},
            action_mean_outcomes={"a": 0.5},
            similar_task_success_rate=None,
        ),
        task_id="task-1",
        episode_id="ep-1",
        step_index=step,
        provenance={"workspace_id": "ws-1"},
    )


def test_encoding_is_deterministic_for_fixed_inputs():
    """Step 14: identical signals produce identical state ids and feature values."""
    encoder = StateEncoder(task_families=("fact_verification", "story"), domains=("science",))
    first = encoder.encode(_signals())
    second = encoder.encode(_signals())
    assert first.state_id == second.state_id
    assert first.features == second.features


def test_encoding_changes_when_signals_change():
    """Different workspace revisions and difficulties yield different states."""
    encoder = StateEncoder(task_families=("fact_verification",), domains=("science",))
    first = encoder.encode(_signals(step=0, difficulty=0.3))
    second = encoder.encode(_signals(step=1, difficulty=0.8))
    assert first.state_id != second.state_id
    assert first.features["workspace.revision"] == 0.0
    assert second.features["workspace.revision"] == 1.0


def test_one_hot_and_numeric_features_are_well_formed():
    """Step 7: categorical features use one-hot; numerics are preserved."""
    encoder = StateEncoder(task_families=("fact_verification", "story"), domains=("science",))
    state = encoder.encode(_signals())
    assert state.features["task_family::fact_verification"] == 1.0
    assert state.features["task_family::story"] == 0.0
    assert state.features["domain::science"] == 1.0
    assert state.features["task.difficulty"] == 0.5
    assert state.features["task.constraint_count"] == 1.0


def test_goal_and_dependency_features_are_typed_counters():
    """Steps 8, 11: explicit counters, dependency mask statistics."""
    encoder = StateEncoder()
    state = encoder.encode(_signals())
    assert state.features["goal.completed_count"] == 1.0
    assert state.features["goal.remaining_count"] == 1.0
    assert state.features["dependency.satisfied_fraction"] == 2.0 / 3.0
    assert state.features["dependency.eligible_count"] == 2.0


def test_missing_similar_task_history_is_explicit():
    """Step 13: missing history is represented explicitly, not as zero performance."""
    encoder = StateEncoder()
    state = encoder.encode(_signals())
    assert state.features["uncertainty.similar_task_available"] == 0.0
    assert state.features["uncertainty.similar_task_success"] == 0.0
    from dataclasses import replace

    present = replace(_signals().uncertainty, similar_task_success_rate=0.6)
    signals = _signals()
    signals = replace(signals, uncertainty=present)
    encoded = encoder.encode(signals)
    assert encoded.features["uncertainty.similar_task_available"] == 1.0
    assert encoded.features["uncertainty.similar_task_success"] == 0.6


def test_normalization_uses_persisted_statistics_only():
    """Step 7: normalization is fitted on a sample, persisted, and applied deterministically."""
    encoder = StateEncoder(task_families=("fact_verification",), domains=("science",))
    rows = [encoder.encode(_signals(difficulty=d)) for d in (0.1, 0.3, 0.5, 0.7)]
    raw = [row.features["task.difficulty"] for row in rows]
    stats = encoder.fit_normalization([_signals(difficulty=d) for d in (0.1, 0.3, 0.5, 0.7)])
    assert set(stats.feature_names) == set(encoder.feature_names)
    restored = stats.to_dict()
    assert abs(restored["mu"][list(encoder.feature_names).index("task.difficulty")]
               - sum(raw) / len(raw)) < 1e-9
    state = encoder.encode(_signals(difficulty=0.5))
    normalized = state.features["task.difficulty"]
    assert not math.isclose(normalized, 0.5)
    assert math.isclose(normalized, (0.5 - stats.mu[list(encoder.feature_names).index("task.difficulty")])
                        / (stats.sigma[list(encoder.feature_names).index("task.difficulty")] + 1e-8))


def test_vector_layout_matches_feature_names():
    """to_vector uses the declared layout; explicit missing entries become NaN."""
    encoder = StateEncoder(task_families=("fact_verification",), domains=("science",))
    state = encoder.encode(_signals())
    vector = encoder.to_vector(state)
    assert vector.shape[0] == len(encoder.feature_names)
    names = encoder.feature_names
    index = names.index("task.difficulty")
    assert vector[index] == state.features["task.difficulty"]
    state_missing = type(state)(
        state_id="s2", task_id="t", encoder_version="1",
        feature_names=names, features={}, missing_features=names[:2],
    )
    assert all(math.isnan(value) for value in encoder.to_vector(state_missing)[:2])


def test_encoder_version_is_persisted_on_states():
    """Every state records the encoder identity for provenance."""
    encoder = StateEncoder()
    state = encoder.encode(_signals())
    assert state.encoder_version == encoder.version.version
    assert encoder.version.fingerprint
    spec = encoder.spec()
    assert spec.dim == len(encoder.feature_names)
    assert spec.encoder_version == "1.0.0"
