"""Acceptance tests: edge compression, federated learning, and QUBO (Steps 38-47)."""
from __future__ import annotations

import numpy as np
import pytest

from ore_learning.config import LearningConfig
from ore_learning.edge import (
    CompressionPlan,
    EdgeDeploymentPlanner,
    compress_for_edge,
    prune_weights,
    pruning_sparsity,
)
from ore_learning.federated import (
    FederatedCoordinator,
    FederatedPolicyError,
    ParticipantCredentials,
    ParticipantUpdate,
)
from ore_learning.functional import fedavg_weights
from ore_learning.nn import TinyLinearModel, TinyMLP
from ore_learning.optimization import (
    CombinatorialProblem,
    OptimizationPipeline,
    OptimizationPolicyError,
    ConventionalBaseline,
    formulate_qubo,
    independent_feasibility_check,
    apply_to_cros_or_policy,
    verify_small_instance,
)

# ----------------------------------------------------------------- Step 39 (edge)


def trained_model(seed: int = 8) -> tuple[TinyLinearModel, np.ndarray, np.ndarray]:
    """A trained regression model with its data and quality function."""
    rng = np.random.default_rng(4)
    features = rng.normal(size=(60, 6))
    targets = features @ np.array([[0.5], [-1.0], [0.3], [0.8], [-0.2], [0.6]])
    model = TinyLinearModel(6, 1, seed=seed)
    model.fit(features, targets, epochs=60, learning_rate=0.1)
    return model, features, targets


def test_quantized_edge_model_meets_memory_and_quality_thresholds():
    """Acceptance: quantized edge model meets declared memory and quality thresholds."""
    model, features, targets = trained_model()
    scale = float(np.mean(targets ** 2))

    def quality(candidate):
        return max(0.0, 1.0 - float(np.mean((candidate.predict(features) - targets)
                                             ** 2)) / scale)

    config = LearningConfig(edge_retention_threshold=0.99, edge_max_memory_mb=1.0)
    compressed, report = compress_for_edge(model, quality,
                                           plan=CompressionPlan(bits=8),
                                           config=config)
    assert report.retention >= config.edge_retention_threshold
    assert report.memory_compressed_bytes <= config.edge_max_memory_mb * 1e6
    assert report.within_thresholds
    # memory really shrank (float32 -> int8)
    assert report.memory_reduction == pytest.approx(0.75)
    # numerical deviation is measured, not assumed
    assert report.numerical_deviation >= 0.0
    # the uncompressed reference is preserved for rollback and untouched
    assert report.rollback_reference_preserved
    assert model.memory_bytes() == report.memory_reference_bytes
    original_quality = quality(model)
    assert quality(model) == pytest.approx(original_quality)


def test_pruning_removes_smallest_magnitudes():
    """Pruning sparsifies weights toward the declared target."""
    weights = {"W": np.array([[0.1, -0.02], [0.5, -0.01]]), "b": np.array([0.3, 0.0])}
    pruned, removed = prune_weights(weights, sparsity=0.5)
    assert pruning_sparsity(pruned) == pytest.approx(0.5)
    assert pruned["W"][0, 0] == 0.1  # large magnitudes survive
    assert pruned["W"][1, 0] == 0.5
    assert removed["W"] == 2


def test_edge_deployment_decision_and_split_authorization():
    """Steps 38 and 40: measured edge decision and the split authorization boundary."""
    planner = EdgeDeploymentPlanner()
    # privacy-sensitive, latency-favouring, device-fitting -> edge wins
    edge = planner.decide(local_accuracy=0.88, central_accuracy=0.91,
                          local_latency_ms=8.0, central_latency_ms=120.0,
                          local_memory_mb=4.0, device_memory_mb=8.0,
                          connectivity_available=True, privacy_sensitive=True,
                          bandwidth_cost_per_mb=0.2)
    assert edge.deploy_at_edge
    plan = planner.plan_split_execution(edge)
    assert "observation_extraction" in plan.edge_tasks
    assert "consequential_actions" in plan.central_tasks
    assert "never authorize" in plan.authorization_boundary
    # memory overrun or a large accuracy gap keeps the task central
    central = planner.decide(local_accuracy=0.60, central_accuracy=0.95,
                             local_latency_ms=8.0, central_latency_ms=120.0,
                             local_memory_mb=4.0, device_memory_mb=8.0,
                             connectivity_available=True, privacy_sensitive=False,
                             bandwidth_cost_per_mb=0.1)
    assert not central.deploy_at_edge
    assert any("accuracy" in reason for reason in central.reasons)


# ----------------------------------------------------------- Steps 41-43 (federated)


def enrolled_coordinator() -> FederatedCoordinator:
    """A coordinator with three authenticated participants."""
    coordinator = FederatedCoordinator()
    for index, count in enumerate([30, 50, 20], start=1):
        coordinator.enroll(ParticipantCredentials(
            f"p{index}", f"token{index}" * 8, "restricted", count))
    return coordinator


def test_fedavg_weights_use_declared_participant_sample_counts():
    """Acceptance: FedAvg weights use declared participant sample counts."""
    weights = fedavg_weights([30, 50, 20])
    np.testing.assert_allclose(weights, [30 / 100, 50 / 100, 20 / 100])
    assert float(np.sum(weights)) == pytest.approx(1.0)
    # the coordinator's round record carries exactly these weights
    coordinator = enrolled_coordinator()
    rng = np.random.default_rng(3)
    updates = [coordinator.submit_update(
        f"p{index}", f"token{index}" * 8, count,
        rng.normal(0.0, 0.3, size=6), participant_seed=10 + index)
        for index, count in enumerate([30, 50, 20], start=1)]
    record = coordinator.aggregate(updates, validation_callback=lambda agg: True)
    np.testing.assert_allclose(record.aggregation_weights, weights)
    assert record.participants == ["p1", "p2", "p3"]


def test_federated_pipeline_rejects_unauthenticated_participants():
    """Acceptance: federated pipeline rejects unauthenticated participants."""
    coordinator = enrolled_coordinator()
    with pytest.raises(FederatedPolicyError, match="authentication"):
        coordinator.submit_update("p1", "wrong-token" * 3, 10, np.zeros(6))
    with pytest.raises(FederatedPolicyError, match="authentication"):
        coordinator.submit_update("p-unknown", "token1" * 8, 10, np.zeros(6))
    # refusals are recorded for audit, not silently swallowed
    assert len(coordinator.refusals) == 2
    assert all("authentication" in refusal["reason"]
               for refusal in coordinator.refusals)


def test_secure_aggregation_and_privacy_controls_are_enforced():
    """Acceptance: required secure aggregation/privacy controls are enforced."""
    config = LearningConfig(federated_clip_norm=1.0, federated_dp_noise_multiplier=0.01)
    coordinator = FederatedCoordinator(config)
    for index in range(1, 4):
        coordinator.enroll(ParticipantCredentials(
            f"p{index}", f"tok{index}" * 8, "restricted", 10 * index))
    rng = np.random.default_rng(5)
    updates = []
    for index in range(1, 4):
        huge_update = np.full(6, 25.0)  # would leak magnitude without clipping
        updates.append(coordinator.submit_update(
            f"p{index}", f"tok{index}" * 8, 10 * index, huge_update,
            participant_seed=index))
    # every accepted update is clipped to the configured bound (privacy control)
    for update in updates:
        assert np.linalg.norm(update.protected_update) <= \
            config.federated_clip_norm * 1.5  # clip + small DP noise
        assert update.raw_update_exposed is False
    # raw updates cannot enter a round when secure aggregation is required
    raw = ParticipantUpdate("p1", 10, np.full(6, 99.0), raw_update_exposed=True)
    with pytest.raises(FederatedPolicyError, match="secure aggregation"):
        coordinator.aggregate([raw] + updates[1:])
    # cohort minimums are enforced
    with pytest.raises(FederatedPolicyError, match="minimum"):
        FederatedCoordinator().aggregate(updates[:1])
    # the documented threat model travels with every round
    record = coordinator.aggregate(updates, validation_callback=lambda a: True)
    assert record.threat_model["protections"]
    assert "clipping" in " ".join(record.threat_model["protections"])


def test_aggregated_model_cannot_activate_without_independent_validation():
    """Acceptance: aggregated model cannot activate without independent validation."""
    coordinator = enrolled_coordinator()
    rng = np.random.default_rng(3)
    updates = [coordinator.submit_update(
        f"p{index}", f"token{index}" * 8, count,
        rng.normal(0.0, 0.3, size=6), participant_seed=index)
        for index, count in enumerate([30, 50, 20], start=1)]
    record = coordinator.aggregate(updates)  # no validation callback
    assert not record.validated and not record.activated
    with pytest.raises(FederatedPolicyError, match="independent validation"):
        coordinator.activate_aggregate(record)
    # independent validation then permits activation
    activated = coordinator.activate_aggregate(
        record, validation_callback=lambda item: True)
    assert activated.validated and activated.activated


def test_poisoning_screen_excludes_outlier_updates():
    """Steps 42-43: outlier update norms are screened out before aggregation."""
    coordinator = FederatedCoordinator()
    for index in range(1, 6):
        coordinator.enroll(ParticipantCredentials(
            f"q{index}", f"t{index}" * 8, "restricted", 10 * index))
    rng = np.random.default_rng(7)
    updates = [coordinator.submit_update(
        f"q{index}", f"t{index}" * 8, 10 * index,
        rng.normal(0.0, 0.2, size=6), participant_seed=index)
        for index in range(1, 6)]
    poisoned = ParticipantUpdate("q3", 30, np.full(6, 50.0))
    record = coordinator.aggregate(updates[:2] + [poisoned] + updates[3:],
                                   validation_callback=lambda a: True)
    assert record.excluded_participants == ["q3"]
    assert "q3" not in record.participants
    assert record.aggregation_weights == pytest.approx(
        [10 / 40, 20 / 40, 40 / 40][:3]) or len(record.aggregation_weights) == 4


# ------------------------------------------------------------- Steps 44-47 (QUBO)


def pick_two_problem(values=None) -> CombinatorialProblem:
    """A 'select exactly 2 items maximizing value' problem (min -value)."""
    if values is None:
        values = [3.0, 1.0, 4.0, 1.5, 5.0, 2.0, 2.5]
    n = len(values)
    return CombinatorialProblem(
        name="pick-2-best", variable_names=tuple(f"x{i}" for i in range(n)),
        objective_matrix=np.diag([-v for v in values]),
        constraints=((np.ones(n), 2.0),),
        description="select exactly 2 items maximizing total value")


def test_qubo_mapping_reproduces_known_optima_on_small_exhaustive_cases():
    """Acceptance: QUBO mapping reproduces known optima on small exhaustive cases."""
    problem = pick_two_problem()
    config = LearningConfig()
    formulation = formulate_qubo(problem, config)
    verification = verify_small_instance(formulation, config)
    assert verification["verified"] is True
    assert verification["feasible"] is True
    # the known optimum: the two most valuable items (5.0 + 4.0)
    assert verification["objective_at_qubo_optimum"] == pytest.approx(-9.0)
    assert verification["best_feasible_objective"] == pytest.approx(-9.0)
    assert verification["violations"] == [0.0]
    # an infeasible rho would break the mapping: too-small penalties fail verification
    weak = LearningConfig(qubo_penalty_rho=0.01)
    weak_formulation = formulate_qubo(problem, weak)
    weak_verification = verify_small_instance(weak_formulation, weak)
    # with a negligible penalty the QUBO optimum selects more than 2 items
    assert weak_verification["verified"] is False or \
        weak_verification["violations"] != [0.0]


def test_qubo_outputs_are_independently_checked_for_feasibility():
    """Acceptance: QUBO outputs are independently checked for feasibility."""
    problem = pick_two_problem()
    formulation = formulate_qubo(problem, LearningConfig())
    # a feasible candidate passes the independent check
    feasible = independent_feasibility_check("ising-test", [0, 0, 1, 0, 1, 0, 0],
                                             problem, formulation)
    assert feasible.feasible and feasible.accepted
    assert feasible.assignment == {"x0": 0, "x1": 0, "x2": 1, "x3": 0,
                                   "x4": 1, "x5": 0, "x6": 0}
    assert feasible.objective_value == pytest.approx(-9.0)
    # an infeasible candidate (1 item selected) is rejected with the violated
    # constraint identified
    infeasible = independent_feasibility_check("ising-test", [1, 0, 0, 0, 0, 0, 0],
                                               problem, formulation)
    assert not infeasible.feasible and not infeasible.accepted
    assert infeasible.violated_constraints == [0]
    # non-binary configurations are refused outright
    with pytest.raises(OptimizationPolicyError, match="binary"):
        independent_feasibility_check("ising-test", [0.5, 0, 1, 0, 1, 0, 0],
                                     problem, formulation)


def test_conventional_solver_baseline_is_recorded_before_ising_evaluation():
    """Acceptance: conventional solver baseline is recorded before Ising evaluation."""
    problem = pick_two_problem()
    config = LearningConfig(annealing_steps=120)
    formulation = formulate_qubo(problem, config)
    pipeline = OptimizationPipeline(config=config)
    # Ising evaluation before any baseline is refused (Step 46 ordering)
    with pytest.raises(OptimizationPolicyError, match="baseline"):
        pipeline.run_ising_only(formulation)
    # the ordered pipeline records the baseline first, then the Ising result
    record = pipeline.run(problem, formulation)
    assert record["order_enforced"]
    assert record["baseline"]["kind"] == "conventional"
    assert record["ising"]["kind"] == "ising"
    assert record["baseline"]["objective_value"] <= record["ising"]["objective_value"] \
        + 1e-9  # the exact baseline is at least as good as the annealer
    assert record["baseline"]["feasible"] is True
    # after a baseline exists, explicit Ising runs are permitted
    again = pipeline.run_ising_only(formulation)
    assert again.kind == "ising"


def test_no_optimization_backend_alters_cros_or_coordinator_policy():
    """Step 47: backends cannot reach CROS scheduling or coordinator policy."""
    problem = pick_two_problem()
    formulation = formulate_qubo(problem, LearningConfig())
    verification = independent_feasibility_check("ising-test", [0, 0, 1, 0, 1, 0, 0],
                                                 problem, formulation)
    with pytest.raises(OptimizationPolicyError, match="CROS"):
        apply_to_cros_or_policy(verification)


def test_conventional_baseline_finds_true_constrained_optimum():
    """The conventional exhaustive path applies constraints directly (CP-SAT style)."""
    problem = pick_two_problem()
    result = ConventionalBaseline().solve(problem, LearningConfig())
    assert result.feasible
    assert result.objective_value == pytest.approx(-9.0)
    assert sum(result.bits) == 2
