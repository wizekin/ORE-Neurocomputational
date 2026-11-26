"""Acceptance tests for the ore_science domain adapters (Steps 16-34).

Each test maps to acceptance criteria in Section 8 of the integrated algorithm
specification for PINNs, neural operators, equivariant geometric learning, ensemble
forecasting, factor-graph state estimation, and molecular/materials intelligence.
"""
from __future__ import annotations

import numpy as np
import pytest

from ore_science import (
    ConditionalMaterialGenerator,
    CrankNicolsonHeatSolver,
    EquivarianceTestProtocol,
    EquivariantPointModule,
    Factor,
    FourierModeOperator,
    LocalReferenceDesignEngine,
    LocalReferenceInteractionService,
    LocalReferenceStabilityModel,
    LocalReferenceStructureService,
    NonEquivariantPointModule,
    PDEProblem,
    PinnBasisSurrogate,
    PinnWeights,
    RepresentationBuilder,
    ScienceConfig,
    ScienceRequest,
    ScientificReasoner,
    StochasticForecastSimulator,
    analytic_heat_solution,
    brier_score,
    build_synthetic_slam_problem,
    calibration_report,
    check_charge_balance,
    evaluate_plausibility,
    heat_residual_grid,
    structure_fingerprint,
)
from ore_science.adapters.factor_graphs import FactorGraphBuilder
from ore_science.adapters.forecasting import EnsembleForecast
from ore_science.interpretation import ScientificTaskParser
from ore_science.knowledge import ScienceKnowledgeBase
from ore_science.representations import MaterialsGraph, MolecularGraph


def _pde_problem() -> PDEProblem:
    """Return the reference heat-diffusion PDE problem in SI units."""
    return PDEProblem(
        name="heat",
        alpha=1.0e-4,
        length=1.0,
        time_horizon=60.0,
        u_boundary=273.15,
        u_initial=373.15,
    )


def _config() -> ScienceConfig:
    """Return a fast test configuration."""
    return ScienceConfig(
        collocation_points=41, time_steps=160, pin_basis_size=6, pin_max_iterations=60
    )


# ---------------------------------------------------------------------------
# PINN (Steps 23-24, Section 5.1-5.2)
# ---------------------------------------------------------------------------
def test_pinn_residual_computation_matches_analytic_test_case() -> None:
    """PINN residual computation matches the analytic solution of the test case."""
    problem = _pde_problem()
    surrogate = PinnBasisSurrogate(problem, basis_size=8)
    x_grid = np.linspace(0.0, 1.0, 41)
    times = np.linspace(0.0, 60.0, 13)
    training = surrogate.train(x_grid, times, PinnWeights(), max_iterations=80)
    # The trained surrogate must reproduce the analytic modal decay closely.
    assert training["after"]["physics_loss"] < 1.0e-3
    final_field = surrogate.field_on_grid(x_grid, np.array([60.0]))[0]
    analytic = (
        analytic_heat_solution(x_grid, 60.0, length=1.0, alpha=1.0e-4, u_initial=100.0)
        + 273.15
    )
    assert float(np.max(np.abs(final_field - analytic))) < 2.0


def test_pinn_boundary_and_initial_losses_reported_independently() -> None:
    """PINN boundary and initial-condition losses are reported independently."""
    surrogate = PinnBasisSurrogate(_pde_problem(), basis_size=6)
    x_grid = np.linspace(0.0, 1.0, 33)
    times = np.linspace(0.0, 60.0, 9)
    report = surrogate.evaluate_losses(x_grid, times, PinnWeights())
    payload = report.to_dict()
    for key in ("data_loss", "physics_loss", "boundary_loss", "initial_loss", "total_loss"):
        assert key in payload
    assert payload["weights"]["lambda_b"] == PinnWeights().boundary


def test_pinn_evaluation_detects_incorrect_pde() -> None:
    """PINN evaluation detects a deliberately incorrect PDE parameter."""
    surrogate = PinnBasisSurrogate(_pde_problem(), basis_size=8)
    x_grid = np.linspace(0.0, 1.0, 41)
    times = np.linspace(0.0, 60.0, 13)
    surrogate.train(x_grid, times, PinnWeights(), max_iterations=80)
    correct = surrogate.evaluate_losses(x_grid, times, PinnWeights())
    wrong = surrogate.evaluate_losses(x_grid, times, PinnWeights(), alpha_override=1.0e-3)
    assert wrong.physics_loss > 100.0 * correct.physics_loss


def test_pinn_compared_with_numerical_reference_solution() -> None:
    """PINN results are compared with a numerical reference solution."""
    problem = _pde_problem()
    surrogate = PinnBasisSurrogate(problem, basis_size=8)
    x_grid = np.linspace(0.0, 1.0, 41)
    times = np.linspace(0.0, 60.0, 9)
    surrogate.train(x_grid, times, PinnWeights(), max_iterations=80)
    reference = CrankNicolsonHeatSolver().solve(
        length=1.0, time_horizon=60.0, alpha=1.0e-4,
        u_initial=373.15, u_boundary=273.15, grid_size=41, time_steps=320,
    )
    pinn_field = surrogate.field_on_grid(reference.x_grid, np.array([60.0]))[0]
    relative = float(
        np.sqrt(np.mean((pinn_field - reference.field) ** 2))
        / max(float(np.max(np.abs(reference.field))), 1.0e-12)
    )
    assert relative < 0.05


def test_heat_residual_grid_shapes_match_collocation() -> None:
    """The PDE residual evaluator returns interior collocation shapes."""
    x_grid = np.linspace(0.0, 1.0, 21)
    times = np.linspace(0.0, 10.0, 5)
    field = np.outer(np.ones_like(times), np.sin(np.pi * x_grid) * 0.5)
    residual = heat_residual_grid(field, x_grid, times, alpha=1.0e-4)
    assert residual.shape == (len(times), len(x_grid) - 2)
    assert np.all(np.isfinite(residual))


# ---------------------------------------------------------------------------
# Neural operators (Steps 25-26)
# ---------------------------------------------------------------------------
def _operator_training_pairs() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build periodic heat-semigroup training pairs at resolution 32."""
    generator = np.random.default_rng(3)
    alpha, horizon = 1.0e-4, 60.0
    inputs, targets = [], []
    for _ in range(10):
        spectrum = generator.normal(size=17) * (np.arange(17) < 8)
        initial_field = np.fft.irfft(spectrum, n=32)
        decay = np.exp(-alpha * (2.0 * np.pi * np.arange(17)) ** 2 * horizon)
        inputs.append(initial_field)
        targets.append(np.fft.irfft(np.fft.rfft(initial_field) * decay, n=32))
    return inputs, targets


def test_neural_operator_evaluates_multiple_resolutions() -> None:
    """Neural-operator evaluation tests more than one discretization resolution."""
    inputs, targets = _operator_training_pairs()
    operator = FourierModeOperator(domain_length=1.0, retained_modes=8)
    operator.fit(inputs, targets)
    transfer = operator.evaluate_resolution_transfer(
        list(zip(inputs, targets)), [32, 64, 128]
    )
    assert set(transfer.keys()) == {"32", "64", "128"}
    for resolution, entry in transfer.items():
        assert entry["rmse"] < 0.01, f"resolution {resolution} transfer failed"
    assert operator.fit_report is not None
    assert tuple(operator.fit_report.resolution_transfer_evaluations) == (32, 64, 128)


def test_fno_adapter_records_retained_modes_and_boundary_convention() -> None:
    """The FNO adapter records retained spectral modes and boundary conventions."""
    inputs, targets = _operator_training_pairs()
    operator = FourierModeOperator(domain_length=1.0, retained_modes=6)
    operator.fit(inputs, targets)
    payload = operator.to_dict()
    assert payload["retained_modes"] == 6
    assert payload["boundary_convention"] == "periodic"
    assert payload["disclosure"]
    untrained = FourierModeOperator(domain_length=1.0, retained_modes=6)
    with pytest.raises(ValueError):
        untrained.apply(np.zeros(16))


# ---------------------------------------------------------------------------
# Equivariant geometric learning (Steps 28-29, Section 5.4)
# ---------------------------------------------------------------------------
def test_equivariant_model_measures_transformation_error() -> None:
    """Equivariant model tests measure the transformation error for supported transformations."""
    rng = np.random.default_rng(11)
    points = rng.normal(size=(7, 2))
    protocol = EquivarianceTestProtocol()
    worst, details = protocol.equivariance_error(EquivariantPointModule(), points)
    assert worst < 1.0e-8
    assert len(details) == len(protocol.angles)
    assert all(item.eps_eq >= 0.0 for item in details)


def test_non_equivariant_model_is_detected() -> None:
    """A non-equivariant module fails the same transformation test loudly."""
    rng = np.random.default_rng(11)
    points = rng.normal(size=(7, 2))
    protocol = EquivarianceTestProtocol()
    worst, _ = protocol.equivariance_error(NonEquivariantPointModule(), points)
    assert worst > 0.1


def test_invariance_test_verifies_scalar_outputs_unchanged() -> None:
    """Invariance tests verify that scalar outputs remain unchanged under the group action."""

    class ScalarEnergyModule:
        """Rotation-invariant scalar module: sum of pairwise distances."""

        def forward(self, point_set: np.ndarray) -> np.ndarray:
            distances = np.linalg.norm(
                point_set[:, None, :] - point_set[None, :, :], axis=2
            )
            total = float(np.sum(distances)) * np.ones((1, 1))
            return total

    rng = np.random.default_rng(5)
    points = rng.normal(size=(6, 2))
    protocol = EquivarianceTestProtocol()
    worst, details = protocol.invariance_error(ScalarEnergyModule(), points)
    assert worst < 1.0e-10
    assert all(item.eps_eq < 1.0e-10 for item in details)


# ---------------------------------------------------------------------------
# Ensemble forecasting (Steps 30-31, Section 5.6)
# ---------------------------------------------------------------------------
def test_ensemble_outputs_retain_member_identity_and_metadata() -> None:
    """Weather ensemble outputs retain member identity and forecast metadata."""
    simulator = StochasticForecastSimulator(seed=101)
    forecast, _ = simulator.simulate(
        initial_state=8.0, members=20, lead_times=[6.0 * index for index in range(1, 9)]
    )
    assert forecast.members.shape == (20, 8)
    assert forecast.lead_times[0] == pytest.approx(6.0)
    payload = forecast.to_dict()
    assert payload["member_count"] == 20
    assert payload["model_id"] == "forecast.ensemble_ar1"
    assert payload["initialization"]


def test_ensemble_statistics_follow_section_5_6_formulas() -> None:
    """Ensemble mean and variance follow the Section 5.6 estimators."""
    rng = np.random.default_rng(9)
    members = rng.normal(size=(16, 5))
    forecast = EnsembleForecast(
        members=members,
        lead_times=np.arange(1.0, 6.0),
        quantity_name="wind speed",
        unit_symbol="m/s",
        model_id="m",
        model_version="1.0.0",
        initialization="test",
        metadata={},
    )
    assert np.allclose(forecast.mean(), members.mean(axis=0))
    assert np.allclose(forecast.variance(), members.var(axis=0, ddof=1))


def test_probabilistic_event_forecasts_evaluated_with_calibration_metrics() -> None:
    """Probabilistic event forecasts are evaluated with calibration metrics."""
    simulator = StochasticForecastSimulator(seed=21)
    forecast, observations = simulator.simulate(
        initial_state=8.0, members=32, lead_times=[6.0 * index for index in range(1, 9)]
    )
    probabilities = forecast.event_probability(lambda members: members > 7.0)
    outcomes = (observations > 7.0).astype(float)
    report = calibration_report(probabilities, outcomes)
    assert 0.0 <= report["brier_score"] <= 1.0
    assert report["reliability_bins"]
    assert report["climatological_base_rate"] == pytest.approx(float(outcomes.mean()))
    assert brier_score(probabilities, outcomes) == pytest.approx(report["brier_score"])


def test_calibration_tests_detect_overconfident_probabilities() -> None:
    """Calibration tests detect deliberately overconfident probabilities."""
    rng = np.random.default_rng(33)
    # A forecaster that always says probability 0.9 for events occurring 50% of the
    # time is overconfident and must be caught by the Brier score and reliability gap.
    outcomes = (rng.random(200) < 0.5).astype(float)
    overconfident = np.full(200, 0.9)
    honest = np.full(200, 0.5)
    report_bad = calibration_report(overconfident, outcomes)
    report_good = calibration_report(honest, outcomes)
    assert report_bad["brier_score"] > report_good["brier_score"] + 0.1
    assert report_bad["max_abs_reliability_gap"] > 0.35
    assert report_good["max_abs_reliability_gap"] < 0.1


# ---------------------------------------------------------------------------
# Factor graphs (Steps 32-34, Section 5.5)
# ---------------------------------------------------------------------------
def test_factor_graph_converges_on_known_synthetic_problem() -> None:
    """Factor-graph optimization converges on a known synthetic estimation problem."""
    adapter, ground_truth = build_synthetic_slam_problem()
    solution = adapter.optimize()
    assert solution.converged
    for name, truth in ground_truth.items():
        assert solution.values[name] == pytest.approx(truth, abs=0.15)
    assert solution.final_chi_squared < 10.0
    assert all(variance > 0.0 for variance in solution.marginal_variances.values())


def test_factor_graph_detects_inconsistent_coordinate_frames() -> None:
    """Factor-graph construction detects inconsistent coordinate frames."""
    with pytest.raises(ValueError, match="frame"):
        build_synthetic_slam_problem(inconsistent_frames=True)


def test_robust_loss_reduces_outlier_influence() -> None:
    """Robust loss reduces the influence of a designated synthetic outlier."""
    clean_adapter, truth = build_synthetic_slam_problem()
    clean = clean_adapter.optimize()
    outlier_adapter, _ = build_synthetic_slam_problem(outlier=True)
    robust = outlier_adapter.optimize()
    # The Huber-weighted solution must stay far closer to truth than a quadratic
    # loss would allow with a 5 m odometry outlier against 0.05 m noise.
    robust_error = abs(robust.values["x1"] - truth["x1"])
    assert robust_error < 0.5
    assert robust.robust_loss_active.get("odom_x0_x1_outlier") is True
    assert robust.values["x1"] != pytest.approx(clean.values["x1"], abs=1.0e-9)


def test_factor_graph_adapter_exposes_gtsam_compatible_interface() -> None:
    """The GTSAM-compatible adapter exposes optimize and marginal covariance queries."""
    adapter, _ = build_synthetic_slam_problem()
    adapter.optimize()
    variance = adapter.marginal_covariance("x1")
    assert variance > 0.0
    with pytest.raises(KeyError):
        adapter.marginal_covariance("x9")
    identity = adapter.to_dict()
    assert identity["backend"] == "local_numpy"
    assert identity["gtsam_available"] is False


def test_factor_scope_validation_rejects_undefined_variables() -> None:
    """Factors referencing undefined variables are rejected by the builder."""

    builder = FactorGraphBuilder()
    builder.add_variable("x0")
    factor = Factor(
        factor_id="bad",
        factor_type="odometry",
        scope=("x0", "x1"),
        residual=lambda state: np.zeros(1),
        covariance=np.array([[0.01]]),
    )
    with pytest.raises(ValueError):
        builder.add_factor(factor)


# ---------------------------------------------------------------------------
# Molecular intelligence (Steps 16-19)
# ---------------------------------------------------------------------------
def _molecular_graph() -> MolecularGraph:
    """Return the reference peptide-like molecular graph."""
    parser = ScientificTaskParser(ScienceKnowledgeBase())
    task = parser.parse(ScienceRequest(question="Predict the structure of a protein backbone"))
    artifact, _ = RepresentationBuilder(ScienceConfig()).build(task)
    return artifact.payload


def test_structure_checks_validate_reference_geometry() -> None:
    """Independent structure checks validate the reference molecular geometry."""
    graph = _molecular_graph()
    report = LocalReferenceStructureService().check_structure(graph)
    assert report.checks_passed
    assert not report.bond_length_violations
    assert not report.clash_pairs


def test_structure_checks_detect_bad_bond_lengths() -> None:
    """Structure checks detect a deliberately distorted bond length."""
    atoms = (
        ("N", 0.0, (0.0, 0.0, 0.0)),
        ("C", 0.0, (0.4, 0.0, 0.0)),
    )
    bonds = ((0, 1, 1, "peptide"),)
    broken = MolecularGraph(atoms=atoms, bonds=bonds)
    report = LocalReferenceStructureService().check_structure(broken)
    assert not report.checks_passed
    assert report.bond_length_violations


def test_interaction_service_enforces_declared_classes() -> None:
    """The interaction evaluator enforces its declared interaction classes."""
    graph = _molecular_graph()
    service = LocalReferenceInteractionService()
    report = service.evaluate(graph, interaction_class="protein_small_molecule")
    assert report.interaction_class == "protein_small_molecule"
    assert report.pair_count > 0
    with pytest.raises(ValueError):
        service.evaluate(graph, interaction_class="protein_protein")


def test_generated_protein_candidates_remain_candidates() -> None:
    """Generated protein candidates remain marked as candidates until validation."""
    design = LocalReferenceDesignEngine().generate(candidate_count=3, random_seed=7)
    payload = design.to_dict()
    assert payload["status"].startswith("CANDIDATE")
    assert payload["random_seed"] == 7
    check = LocalReferenceStructureService().check_structure(design.candidates[0])
    assert check.checks_passed, "reference candidates must satisfy independent checks"


def test_design_generation_is_reproducible_from_seed() -> None:
    """Candidate generation is reproducible from the recorded seed."""
    first = LocalReferenceDesignEngine().generate(candidate_count=2, random_seed=42)
    second = LocalReferenceDesignEngine().generate(candidate_count=2, random_seed=42)
    assert first.candidates[0].atoms == second.candidates[0].atoms


# ---------------------------------------------------------------------------
# Materials intelligence (Steps 20-21)
# ---------------------------------------------------------------------------
def _materials_graph():
    """Return the reference BaTiO3 crystal graph."""
    parser = ScientificTaskParser(ScienceKnowledgeBase())
    task = parser.parse(
        ScienceRequest(question="Discover stable crystal materials with a perovskite oxide lattice")
    )
    artifact, _ = RepresentationBuilder(ScienceConfig()).build(task)
    return artifact.payload


def test_charge_balance_admits_neutral_compositions() -> None:
    """Charge-balance checking admits neutral and rejects charged compositions."""
    balanced, net = check_charge_balance((("Ba", 1), ("Ti", 1), ("O", 3)))
    assert balanced and net == 0
    unbalanced, net_charge = check_charge_balance((("Ba", 1), ("O", 3)))
    assert not unbalanced


def test_materials_candidates_pass_plausibility_filters() -> None:
    """Generated materials candidates pass composition and geometry checks before downstream validation."""
    generator = ConditionalMaterialGenerator()
    candidates = generator.generate(candidate_count=5, random_seed=11)
    reports = [evaluate_plausibility(candidate.graph) for candidate in candidates]
    assert all(report.passed for report in reports)
    assert all(report.charge_balanced for report in reports)
    assert all(report.lattice_parameter_valid for report in reports)


def test_duplicate_structures_are_detected() -> None:
    """Duplicate structures are detected before expensive downstream validation."""
    graph = _materials_graph()
    fingerprint = structure_fingerprint(graph)
    report = evaluate_plausibility(graph, known_fingerprints=[fingerprint])
    assert not report.passed
    assert report.duplicate_of == (fingerprint,)


def test_predicted_stability_is_reported_as_prediction() -> None:
    """Predicted stability is reported as a model prediction, not a laboratory result."""
    prediction = LocalReferenceStabilityModel().predict(_materials_graph())
    payload = prediction.to_dict()
    assert payload["status"].startswith("PREDICTED")
    assert "not a laboratory result" in payload["status"]
    assert prediction.prediction_uncertainty_ev > 0.0


def test_out_of_bounds_lattice_is_rejected() -> None:
    """Physically inadmissible lattice parameters fail the plausibility filters."""

    weird = MaterialsGraph(
        composition=(("Ba", 1), ("Ti", 1), ("O", 3)),
        lattice=((0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.01)),
        sites=(("Ba", 0.0, 0.0, 0.0), ("Ti", 0.5, 0.5, 0.5),
               ("O", 0.5, 0.5, 0.0), ("O", 0.5, 0.0, 0.5), ("O", 0.0, 0.5, 0.5)),
    )
    report = evaluate_plausibility(weird, lattice_bounds_nm=(0.15, 1.2))
    assert not report.lattice_parameter_valid
    assert not report.passed


# ---------------------------------------------------------------------------
# Numerical reference solver (Step 27)
# ---------------------------------------------------------------------------
def test_numerical_solver_matches_analytic_solution() -> None:
    """The numerical reference solver agrees with the analytic solution."""
    solver = CrankNicolsonHeatSolver()
    result = solver.solve(
        length=1.0, time_horizon=60.0, alpha=1.0e-4,
        u_initial=373.15, u_boundary=273.15, grid_size=81, time_steps=640,
    )
    analytic = (
        analytic_heat_solution(result.x_grid, 60.0, length=1.0, alpha=1.0e-4, u_initial=100.0)
        + 273.15
    )
    assert float(np.max(np.abs(result.field - analytic))) < 0.05


def test_numerical_solver_converges_under_refinement() -> None:
    """The solver's center probe converges under grid and time refinement."""
    solver = CrankNicolsonHeatSolver()
    probes = []
    for grid in (41, 81, 161):
        solution = solver.solve(
            length=1.0, time_horizon=60.0, alpha=1.0e-4,
            u_initial=373.15, u_boundary=273.15,
            grid_size=grid, time_steps=4 * grid,
        )
        probes.append(float(solution.field[grid // 2]))
    errors = [abs(probes[i] - probes[i + 1]) for i in range(2)]
    assert errors[1] < errors[0], "error must shrink under refinement"


# ---------------------------------------------------------------------------
# Pipeline runs for every domain (Phase III routing)
# ---------------------------------------------------------------------------
def test_pipeline_forecasting_run_preserves_ensemble() -> None:
    """The forecasting pipeline run preserves ensemble members and calibration evidence."""
    result = ScientificReasoner(_config()).process(
        ScienceRequest(question="Forecast the ensemble weather distribution of wind speed 48 h ahead")
    )
    assert result.diagnostics["valid"]
    assert result.predictions[0]["payload"]["member_count"] >= 2
    assert result.uncertainty_reports[0]["alternative_futures"] >= 2


def test_pipeline_localization_run_recovers_ground_truth() -> None:
    """The localization pipeline run recovers the synthetic ground truth."""
    result = ScientificReasoner(_config()).process(
        ScienceRequest(question="Use SLAM with odometry and landmarks to localize a robot pose")
    )
    assert result.diagnostics["valid"]
    values = result.predictions[0]["payload"]["values"]
    assert values["x1"] == pytest.approx(1.0, abs=0.15)
    assert values["l0"] == pytest.approx(0.5, abs=0.15)


def test_pipeline_materials_run_keeps_candidates_separate() -> None:
    """The materials pipeline run keeps candidate and predicted statuses separate."""
    result = ScientificReasoner(_config()).process(
        ScienceRequest(
            question="Discover stable crystal materials with a perovskite oxide lattice for batteries"
        )
    )
    assert result.diagnostics["valid"]
    statuses = {prediction["status"] for prediction in result.predictions}
    assert "CANDIDATE" in statuses
    assert "PREDICTED" in statuses


def test_pipeline_molecular_run_keeps_design_candidates() -> None:
    """The molecular pipeline run keeps generated design output in CANDIDATE status."""
    result = ScientificReasoner(_config()).process(
        ScienceRequest(
            question="Predict the structure of a protein backbone and its interaction with a ligand"
        )
    )
    assert result.diagnostics["valid"]
    design = next(p for p in result.predictions if p["model_id"] == "molecular.rfdiffusion_reference")
    assert design["status"] == "CANDIDATE"


def test_pipeline_geometric_run_reports_equivariance() -> None:
    """The geometric pipeline run reports measured equivariance error."""
    result = ScientificReasoner(_config()).process(
        ScienceRequest(
            question="Check the rotation equivariance of the geometric transform for a point cloud"
        )
    )
    assert result.diagnostics["valid"]
    prediction = result.predictions[0]
    assert prediction["confidence_information"]["worst_eps_eq"] < 1.0e-6
