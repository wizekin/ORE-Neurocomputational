"""Acceptance tests for the HACOL Point 21 integrated algorithm.

Each test maps to one or more entries of the acceptance-test list in
"ORE Integrated Algorithm -- Point 21: Hardware-Aware Numerical Computing and
Quantum Algorithms", section 10. Fixtures are seeded and dimensions stay
small so the suite remains fast.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from ore_hacol import (
    AssignmentProblem,
    CancellationToken,
    CorrectnessContract,
    ConsequenceLevel,
    ExternalCROSAdapter,
    FeatureFlags,
    HardwareAwareOptimizer,
    LocalCROSGate,
    OperationKind,
    OPERATION_REGISTRY,
    OptimizationRequest,
    RequestValidationError,
    ValidationStatus,
    adversarial_attention_fixture,
    adversarial_least_squares,
    adversarial_overflow_matrix,
    adversarial_rank_deficient_matrix,
    attention_eligibility,
    attention_fixture,
    butterfly_certificate,
    ButterflyFactors,
    build_blackboard_envelope,
    capability_summary,
    check_range,
    choose_precision,
    compare_arrays,
    compare_solvers,
    decode_assignment,
    detect_structure,
    discover_hardware_profile,
    encode_assignment_qubo,
    estimate_optimization_value,
    exact_qubo_solver,
    grover_query_bound,
    hungarian_assignment,
    ising_energy,
    ising_from_qubo,
    iterative_refinement,
    least_squares_fixture,
    logical_error_rate_per_cycle,
    low_rank_matrix,
    low_rank_opportunity,
    measure_amortization,
    mixed_precision_matmul,
    online_softmax_attention,
    publish_to_blackboard,
    qec_overhead_record,
    quantum_research_gate,
    qubo_assignment_fixture,
    qubo_energy,
    randomized_range,
    randomized_svd,
    rank_deficiency_report,
    record_result_to_sqlite,
    reference_attention,
    register_hacol_annotations,
    reconstruction_error,
    research_catalog,
    run_adversarial_suite,
    select_tile_dimensions,
    shor_record,
    simulated_annealing,
    sketched_least_squares,
    stable_softmax,
    structured_transform_fixture,
    surface_code_physical_qubits,
    synthetic_bandwidth_bound_workload,
    synthetic_compute_bound_workload,
    tabu_search,
    tiled_attention,
    to_bfloat16,
    ToeplitzOperator,
    verify_high_precision,
    verify_penalty_preserves_optimum,
)
from ore_hacol.profiling import classify_bottleneck, profile_function
from ore_hacol.validation import (
    RegressionRecord,
    determinism_check,
    downstream_decision_stability,
    load_regression_records,
    save_regression_records,
)


# ---------------------------------------------------------------------------
# Request admission (steps 1-2, contract gate)
# ---------------------------------------------------------------------------


def test_rejects_request_without_operation_identity():
    with pytest.raises(RequestValidationError):
        OptimizationRequest("", "op", "attention", (4, 8, 8))
    with pytest.raises(RequestValidationError):
        OptimizationRequest("task", "", "attention", (4, 8, 8))


def test_rejects_invalid_shapes_and_dtype_declarations():
    with pytest.raises(RequestValidationError):
        OptimizationRequest("t", "op", "attention", (0, 8))
    with pytest.raises(RequestValidationError):
        OptimizationRequest("t", "op", "attention", (4, 8), dtype="int32")
    with pytest.raises(RequestValidationError):
        OptimizationRequest("t", "op", "attention", (4, 8), dtype="float64", layout="scrambled")


def test_rejects_candidates_without_correctness_contract():
    optimizer = HardwareAwareOptimizer(calibrate_hardware=True)
    fixture = attention_fixture(sequence=16, dim=8, seed=3)
    request = OptimizationRequest("t", "att", "attention", (16, 16, 8))
    result = optimizer.run(request, fixture)
    assert result.validation_status is ValidationStatus.DEFERRED
    assert any("correctness contract" in note for note in result.metrics.get("notes", []))


# ---------------------------------------------------------------------------
# Phase A: profiling, bottleneck classification, fixtures
# ---------------------------------------------------------------------------


def test_records_reproducible_reference_results():
    optimizer = HardwareAwareOptimizer(calibrate_hardware=True)
    fixture = attention_fixture(sequence=24, dim=8, seed=11)
    request = OptimizationRequest(
        "t", "att", "attention", (24, 24, 8),
        correctness_contract=CorrectnessContract(rel_tolerance=1e-9),
    )
    first = optimizer.run(request, fixture)
    second = optimizer.run(request, fixture)
    reference_a = np.asarray(first.reference_output)
    reference_b = np.asarray(second.reference_output)
    assert np.array_equal(reference_a, reference_b)


def test_benchmark_fixtures_are_seeded_and_realistic():
    fixture = low_rank_matrix(rows=48, columns=32, rank=4, seed=17)
    twin = low_rank_matrix(rows=48, columns=32, rank=4, seed=17)
    assert np.array_equal(fixture["A"], twin["A"])
    assert fixture["A"].shape == (48, 32)


def test_detects_synthetic_compute_bound_workload():
    hardware = discover_hardware_profile(calibrate=True, bandwidth_bytes=1 << 21, compute_size=96)
    workload = synthetic_compute_bound_workload(size=256, seed=7)
    report = profile_function(workload["callable"], "compute-probe", flops=workload["flops"],
                              bytes_moved=workload["bytes_moved"], runs=3)
    assert classify_bottleneck(report, hardware) == "compute_bound"


def test_detects_synthetic_bandwidth_bound_workload():
    hardware = discover_hardware_profile(calibrate=True, bandwidth_bytes=1 << 21, compute_size=96)
    workload = synthetic_bandwidth_bound_workload(elements=1 << 18, seed=8)
    report = profile_function(workload["callable"], "memory-probe", flops=workload["flops"],
                              bytes_moved=workload["bytes_moved"], runs=3)
    assert classify_bottleneck(report, hardware) == "bandwidth_bound"


def test_records_hardware_and_runtime_capability_identity():
    hardware = discover_hardware_profile(calibrate=False)
    assert hardware.hardware_identity().startswith("hw:")
    summary = capability_summary(hardware)
    assert summary["runtime_versions"]["numpy"]
    assert "float64" in hardware.supported_dtypes


# ---------------------------------------------------------------------------
# Phase B: CROS admission, value gate
# ---------------------------------------------------------------------------


def test_enforces_cros_resource_admission_and_deferral():
    optimizer = HardwareAwareOptimizer(
        cros=LocalCROSGate(memory_cap_bytes=256, max_concurrent=1), calibrate_hardware=True
    )
    fixture = attention_fixture(sequence=16, dim=8, seed=3)
    request = OptimizationRequest(
        "t", "att", "attention", (16, 16, 8),
        correctness_contract=CorrectnessContract(rel_tolerance=1e-9),
    )
    result = optimizer.run(request, fixture)
    assert result.validation_status is ValidationStatus.DEFERRED
    assert any("CROS" in str(entry.get("detail", "")) for entry in result.step_trace)


def test_external_cros_adapter_delegates_authority():
    class TinyCoordinator:
        def __init__(self):
            self.calls = 0

        def admit(self, ask, task_id=""):
            self.calls += 1

            class Decision:
                granted = False
                reason = "global coordinator retains authority"

            return Decision()

    coordinator = TinyCoordinator()
    gate = ExternalCROSAdapter(coordinator)
    from ore_hacol import ResourceAsk

    decision = gate.admit(ResourceAsk(memory_bytes=10))
    assert decision.granted is False
    assert decision.reason.startswith("[external_cros]")
    assert coordinator.calls == 1


def test_value_estimate_rejects_promotion_below_complexity_cost():
    estimate = estimate_optimization_value(
        call_frequency=1.0, per_call_savings_seconds=1e-6,
        setup_cost_seconds=0.0, maintenance_cost_seconds=0.0,
        complexity_penalty_seconds=1e-3,
    )
    assert estimate.recommend_optimization is False
    assert estimate.net_value_seconds < 0


# ---------------------------------------------------------------------------
# Phase C: IO-aware attention (steps 17-23)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mask_kind", ["none", "causal", "padding"])
def test_online_softmax_matches_stable_reference(mask_kind):
    fixture = attention_fixture(sequence=48, dim=12, mask_kind=mask_kind, seed=13)
    reference, _ = reference_attention(fixture["q"], fixture["k"], fixture["v"], fixture["mask"])
    online, _ = online_softmax_attention(fixture["q"], fixture["k"], fixture["v"], fixture["mask"], block=7)
    np.testing.assert_allclose(online, reference, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("mask_kind", ["none", "causal", "padding"])
def test_tiled_attention_within_tolerance(mask_kind):
    fixture = attention_fixture(sequence=64, dim=16, mask_kind=mask_kind, seed=13)
    reference, _ = reference_attention(fixture["q"], fixture["k"], fixture["v"], fixture["mask"])
    tiled, record = tiled_attention(
        fixture["q"], fixture["k"], fixture["v"], fixture["mask"], tile_query=16, tile_key=16
    )
    np.testing.assert_allclose(tiled, reference, atol=1e-10, rtol=1e-9)
    assert record.score_matrix_materialized is False


def test_handles_extreme_logits_without_overflow():
    fixture = attention_fixture(sequence=32, dim=8, seed=9, extreme_logits=True)
    reference, _ = reference_attention(fixture["q"], fixture["k"], fixture["v"])
    tiled, record = tiled_attention(fixture["q"], fixture["k"], fixture["v"], tile_query=8, tile_key=8)
    assert np.all(np.isfinite(tiled))
    np.testing.assert_allclose(tiled, reference, atol=1e-9, rtol=1e-9)


def test_handles_all_masked_rows_per_contract():
    mask = np.zeros((6, 6), dtype=bool)
    mask[0, :2] = True  # row 0 has keys; rows 1-5 are fully masked
    fixture = attention_fixture(sequence=6, dim=4, seed=3)
    zeros_ref, _ = reference_attention(fixture["q"], fixture["k"], fixture["v"], mask, all_masked_policy="zeros")
    zeros_tiled, _ = tiled_attention(fixture["q"], fixture["k"], fixture["v"], mask, all_masked_policy="zeros")
    np.testing.assert_allclose(zeros_tiled[1:], 0.0, atol=0.0)
    np.testing.assert_allclose(zeros_tiled[0], zeros_ref[0], atol=1e-12)
    uniform_ref, _ = reference_attention(fixture["q"], fixture["k"], fixture["v"], mask, all_masked_policy="uniform")
    uniform_tiled, _ = tiled_attention(fixture["q"], fixture["k"], fixture["v"], mask, all_masked_policy="uniform")
    mean_row = np.broadcast_to(np.mean(fixture["v"], axis=0), (5, fixture["v"].shape[1]))
    np.testing.assert_allclose(uniform_tiled[1:], mean_row, atol=1e-12)
    np.testing.assert_allclose(uniform_ref[1:], mean_row, atol=1e-12)
    with pytest.raises(ValueError):
        tiled_attention(fixture["q"], fixture["k"], fixture["v"], mask, all_masked_policy="reject")


def test_preserves_causal_and_padding_masks_semantics():
    from ore_hacol import causal_mask, padding_mask

    fixture = attention_fixture(sequence=20, dim=6, seed=5)
    causal = causal_mask(20, 20)
    out, _ = tiled_attention(fixture["q"], fixture["k"], fixture["v"], causal)
    ref, _ = reference_attention(fixture["q"], fixture["k"], fixture["v"], causal)
    np.testing.assert_allclose(out, ref, atol=1e-10)
    padding = padding_mask([5, 12, 20, 3] + [20] * 16, 20)
    out2, _ = tiled_attention(fixture["q"], fixture["k"], fixture["v"], padding)
    ref2, _ = reference_attention(fixture["q"], fixture["k"], fixture["v"], padding)
    np.testing.assert_allclose(out2, ref2, atol=1e-10)


def test_fallback_on_unsupported_attention_shape():
    fixture = attention_fixture(sequence=16, dim=8, seed=3)
    incompatible_mask = np.zeros((9, 9), dtype=bool)
    eligible, reason = attention_eligibility(fixture["q"], fixture["k"], fixture["v"], incompatible_mask)
    assert eligible is False
    # The pipeline path falls back to trusted attention rather than erroring.
    optimizer = HardwareAwareOptimizer(calibrate_hardware=True)
    request = OptimizationRequest(
        "t", "att", "attention", (16, 16, 8),
        correctness_contract=CorrectnessContract(rel_tolerance=1e-9),
    )
    result = optimizer.run(request, {**fixture, "mask": incompatible_mask})
    assert result.validation_status is ValidationStatus.DEFERRED
    assert any("mask" in note for note in result.metrics.get("notes", []))


def test_tile_dimensions_respect_memory_budget():
    tiles = select_tile_dimensions(1024, 1024, 64, memory_budget_bytes=1 << 16)
    assert 8 <= tiles[0] <= 128 and 8 <= tiles[1] <= 128


# ---------------------------------------------------------------------------
# Phase D: randomized numerical linear algebra (steps 24-32)
# ---------------------------------------------------------------------------


def test_randomized_svd_reproduces_with_fixed_seed():
    fixture = low_rank_matrix(rows=64, columns=48, rank=5, noise=1e-5, seed=17)
    first = randomized_svd(fixture["A"], rank=5, oversampling=5, seed=42)
    second = randomized_svd(fixture["A"], rank=5, oversampling=5, seed=42)
    np.testing.assert_allclose(first["U"], second["U"])
    np.testing.assert_allclose(first["singular_values"], second["singular_values"])
    assert first["metadata"].seed == 42 == second["metadata"].seed


def test_randomized_range_dimensions_and_rank():
    fixture = low_rank_matrix(rows=64, columns=48, rank=5, noise=1e-5, seed=17)
    Q, meta = randomized_range(fixture["A"], rank=5, oversampling=4, seed=1)
    assert Q.shape == (64, 9)
    assert meta.detected_rank == 9
    residual = reconstruction_error(fixture["A"], Q)
    assert residual < 1e-4


def test_detects_rank_deficient_sketches():
    rank_deficient = adversarial_rank_deficient_matrix(seed=31)
    report = rank_deficiency_report(rank_deficient)
    assert report["detected_rank"] == 2
    assert report["rank_deficient"] is True
    Q, meta = randomized_range(rank_deficient, rank=8, oversampling=2, seed=3)
    assert meta.rank_deficient is True


def test_measures_low_rank_reconstruction_residual_and_budget():
    fixture = low_rank_matrix(rows=64, columns=48, rank=5, noise=1e-5, seed=17)
    result = randomized_svd(fixture["A"], rank=5, oversampling=5, seed=42)
    budget = 1e-3
    assert result["exact_relative_error"] < budget
    assert result["error_estimate"] < budget * 5
    rejected = randomized_svd(fixture["A"], rank=1, oversampling=0, seed=42)
    # Rank 1 truncation of a rank-5-plus-noise matrix exceeds a tight budget.
    assert rejected["exact_relative_error"] > 1e-2


def test_low_rank_opportunity_requires_savings_above_decomposition():
    fixture = low_rank_matrix(rows=64, columns=48, rank=5, noise=1e-5, seed=17)
    low_reuse = low_rank_opportunity(fixture["A"], reuse_count=1, tolerance=5e-2)
    assert low_reuse["opportunity"] is False
    heavy_reuse = low_rank_opportunity(fixture["A"], reuse_count=500, tolerance=5e-2, workload_columns=8)
    assert heavy_reuse["opportunity"] is True
    assert heavy_reuse["estimated_flops_savings_fraction"] > 0


def test_sketch_least_squares_validates_original_system_residual():
    fixture = least_squares_fixture(rows=80, columns=8, condition=1e2, seed=17)
    result = sketched_least_squares(fixture["A"], fixture["b"], sketch_rows=24, seed=7)
    assert result["original_residual"] < 5 * result["reference_residual"] + 1e-9
    assert "original system" in result["notes"][0]
    adversarial = adversarial_least_squares()
    ill = sketched_least_squares(adversarial["A"], adversarial["b"], sketch_rows=24, seed=7)
    assert ill["conditioning"]["original_condition"] > 1e10


# ---------------------------------------------------------------------------
# Phase E: structured transforms (steps 33-36) and precision (steps 37-42)
# ---------------------------------------------------------------------------


def test_structured_factorization_reconstruction_validates():
    factors = ButterflyFactors.build(64, seed=13)
    dense = factors.to_dense()
    rng = np.random.default_rng(4)
    vectors = tuple(rng.standard_normal(64) for _ in range(3))
    for vector in vectors:
        np.testing.assert_allclose(factors.apply(vector), dense @ vector, atol=1e-10)
    assert butterfly_certificate(dense, factors) is True
    perturbed = dense.copy()
    perturbed[0, 0] += 1e-3
    assert butterfly_certificate(perturbed, factors) is False


def test_toeplitz_structured_apply_matches_dense():
    rng = np.random.default_rng(6)
    toeplitz = ToeplitzOperator.build(rng.standard_normal(33))
    dense = toeplitz.to_dense()
    vector = rng.standard_normal(33)
    np.testing.assert_allclose(toeplitz.matvec(vector), dense @ vector, atol=1e-10)
    report = detect_structure(dense)
    assert report.toeplitz_like is True


def test_rejects_unsupported_structure_assumptions():
    rng = np.random.default_rng(6)
    dense = rng.standard_normal((16, 16))
    report = detect_structure(dense)
    assert report.toeplitz_like is False


def test_measures_preprocessing_amortization():
    fixture = structured_transform_fixture(size=2048, seed=13)
    factors = fixture["factors"]
    dense = fixture["dense"]
    rng = np.random.default_rng(2)
    vectors = tuple(rng.standard_normal(2048) for _ in range(5))
    amortization = measure_amortization(
        lambda: ButterflyFactors.build(2048, seed=13),
        factors.apply,
        lambda vector: dense @ vector,
        vectors,
        applications=100,
    )
    assert amortization.net_benefit is True
    assert amortization.break_even_applications < 100


def test_fp16_bf16_fp32_fp64_selection_policy():
    loose = CorrectnessContract(rel_tolerance=0.05)
    wide = choose_precision(10.0, 10.0, loose, accumulation_depth=64)
    assert wide.product_dtype == "float32"
    tight = CorrectnessContract(rel_tolerance=1e-9)
    strict = choose_precision(10.0, 1e3, tight)
    assert strict.product_dtype == "float64"
    exact = CorrectnessContract(exactness_required=True, rel_tolerance=1e-12)
    forced = choose_precision(10.0, 1.0, exact)
    assert forced.product_dtype == "float64" and forced.accumulate_dtype == "float64"
    overflowing = choose_precision(1e5, 10.0, loose)
    assert overflowing.product_dtype != "float16"


def test_detects_overflow_and_underflow():
    values = np.array([1e5, 60000.0, 1e-8, 1.0])
    report = check_range(values, "float16")
    assert report.overflow_count == 2
    assert report.underflow_subnormal_count == 1
    assert report.safe is False
    comfortable = check_range(np.array([1.0, 2.0, 3.0]), "float16")
    assert comfortable.safe is True


def test_bfloat16_emulation_rounds_correctly():
    values = np.array([1.0, -2.5, 0.1, 3.14159265], dtype=np.float32)
    rounded = to_bfloat16(values)
    assert float(rounded[0]) == 1.0
    assert float(rounded[1]) == -2.5
    relative = abs(float(rounded[2]) - 0.1) / 0.1
    assert relative < 1e-2


def test_mixed_precision_matmul_and_high_precision_verification():
    rng = np.random.default_rng(3)
    left = rng.uniform(0.5, 1.5, size=(40, 24))
    right = rng.uniform(0.5, 1.5, size=(24, 20))
    reference = left @ right
    output, report = mixed_precision_matmul(
        left, right, product_dtype="float32", accumulate_dtype="float32", block=8
    )
    assert report["overflow_blocks"] == 0
    contract = CorrectnessContract(rel_tolerance=5e-3)
    error = verify_high_precision(output, reference, contract)
    assert error.within_tolerance is True
    big = adversarial_overflow_matrix()
    _, overflow_report = mixed_precision_matmul(big, big, product_dtype="float16",
                                                accumulate_dtype="float32", block=8)
    assert overflow_report["overflow_blocks"] > 0


def test_iterative_refinement_converges_for_eligible_systems():
    rng = np.random.default_rng(17)
    left, _ = np.linalg.qr(rng.standard_normal((12, 12)))
    right, _ = np.linalg.qr(rng.standard_normal((12, 12)))
    singular = np.logspace(0, -3, 12)
    matrix = (left * singular) @ right.T
    solution_true = rng.standard_normal(12)
    target = matrix @ solution_true  # consistent system: refinement can converge
    solution, report = iterative_refinement(
        matrix, target, solve_dtype="float32", residual_dtype="float64"
    )
    assert report["converged"] is True
    assert report["final_residual"] <= report["initial_residual"]
    assert report["eligible"] is True
    adversarial = adversarial_least_squares()
    _, ill_report = iterative_refinement(
        adversarial["A"], adversarial["b"], solve_dtype="float32", max_iterations=1
    )
    assert ill_report["condition_estimate"] > 1e10


def test_high_precision_check_detects_injected_errors():
    rng = np.random.default_rng(5)
    reference = rng.standard_normal((8, 8))
    clean = reference + 1e-12
    contract = CorrectnessContract(rel_tolerance=1e-6)
    assert verify_high_precision(clean, reference, contract).within_tolerance is True
    injected = reference.copy()
    injected[0, 0] += 0.5
    assert verify_high_precision(injected, reference, contract).within_tolerance is False


def test_downstream_decision_stability_under_perturbation():
    fixture = attention_fixture(sequence=16, dim=8, seed=7)

    def optimized(inputs):
        return tiled_attention(inputs["q"], inputs["k"], inputs["v"], inputs["mask"], tile_query=8, tile_key=8)[0]

    def reference(inputs):
        return reference_attention(inputs["q"], inputs["k"], inputs["v"], inputs["mask"])[0]

    report = downstream_decision_stability(optimized, reference, fixture, epsilon=1e-6)
    assert report["stable"] is True


# ---------------------------------------------------------------------------
# Phase F: classical optimization and quantum research gate (steps 43-48)
# ---------------------------------------------------------------------------


def test_classical_baselines_are_established_first():
    fixture = qubo_assignment_fixture(tasks=4, resources=5, seed=17)
    problem = AssignmentProblem(fixture["cost"])
    hungarian = hungarian_assignment(problem)
    comparison = compare_solvers(problem, seed=17)
    assert hungarian["feasible"] is True
    assert comparison["hungarian_optimum"] == hungarian["objective"]
    solvers = {record["solver"] for record in comparison["records"]}
    assert {"hungarian_baseline", "greedy_heuristic", "simulated_annealing",
            "tabu_search", "exact_enumeration", "cp_sat"} <= solvers


def test_qubo_maps_feasible_assignments_correctly():
    fixture = qubo_assignment_fixture(tasks=3, resources=4, seed=17)
    problem = AssignmentProblem(fixture["cost"])
    encoded = encode_assignment_qubo(problem, penalty=100.0)
    Q, c = encoded["Q"], encoded["c"]
    rng = np.random.default_rng(2)
    for _ in range(5):
        z = rng.integers(0, 2, size=Q.shape[0])
        decode = decode_assignment(z, 3, 4)
        counts = np.asarray(decode["task_assignments"])
        manual = float(np.sum(counts * fixture["cost"]))
        penalty = 100.0 * (
            sum((sum(row) - 1) ** 2 for row in decode["task_assignments"])
            + sum(load * (load - 1) // 2 for load in decode["resource_load"])
        )
        np.testing.assert_allclose(qubo_energy(Q, c, z), manual + penalty - 300.0, atol=1e-9)


def test_ising_mapping_corresponds_to_qubo():
    fixture = qubo_assignment_fixture(tasks=3, resources=4, seed=17)
    problem = AssignmentProblem(fixture["cost"])
    encoded = encode_assignment_qubo(problem, penalty=100.0)
    ising = ising_from_qubo(encoded["Q"], encoded["c"])
    rng = np.random.default_rng(3)
    for _ in range(5):
        z = rng.integers(0, 2, size=encoded["Q"].shape[0])
        spins = 2 * z - 1
        np.testing.assert_allclose(
            ising_energy(ising["J"], ising["h"], ising["constant"], spins),
            qubo_energy(encoded["Q"], encoded["c"], z),
            atol=1e-9,
        )


def test_penalties_preserve_intended_feasible_optimum():
    fixture = qubo_assignment_fixture(tasks=3, resources=4, seed=17)
    problem = AssignmentProblem(fixture["cost"])
    well_scaled = verify_penalty_preserves_optimum(problem, 100.0)
    assert well_scaled["verified"] is True
    assert well_scaled["qubo_optimum_objective"] == well_scaled["classical_optimum"]
    mis_scaled = verify_penalty_preserves_optimum(problem, 0.001)
    assert mis_scaled["verified"] is False


def test_records_feasibility_and_optimality_gap():
    fixture = qubo_assignment_fixture(tasks=3, resources=4, seed=17)
    comparison = compare_solvers(AssignmentProblem(fixture["cost"]), seed=17)
    gaps = {
        record["solver"]: record.get("optimality_gap")
        for record in comparison["records"]
    }
    assert gaps["hungarian_baseline"] == 0.0
    assert gaps["exact_enumeration"] == 0.0
    assert gaps["simulated_annealing"] is None or gaps["simulated_annealing"] >= 0.0


def test_quantum_runtime_optional_and_never_selected_for_production():
    catalog = research_catalog()
    assert catalog["gate"]["selected"] is False
    assert catalog["gate"]["quantum_runtime_in_core"] is False
    for kind in ("database_search", "factorization", "scheduling"):
        gate = quantum_research_gate(kind)
        assert gate["selected"] is False
        assert gate["decision"] == "research_only"


def test_grover_and_shor_research_records_with_assumptions():
    classical, quantum = grover_query_bound(1_000_000)
    assert classical == 1_000_000
    assert quantum < 1000
    grover = research_catalog()["grover"]
    assert grover["status"] == "research_only"
    assert grover["error_correction_overhead_factor"] > 1
    shor = shor_record(2048)
    assert shor.logical_qubits_estimate == 2 * 2048 + 3
    assert shor.status == "research_awareness_only"


def test_quantum_error_correction_overhead_model():
    assert surface_code_physical_qubits(2, 17) == 2 * 17**2 * 2
    assert logical_error_rate_per_cycle(1e-3, 25) < 1e-12
    record = qec_overhead_record(logical_qubits=4, code_distance=17)
    assert record.physical_qubits == 4 * 2 * 17**2
    assert record.magic_state_overhead_factor == 15
    with pytest.raises(ValueError):
        surface_code_physical_qubits(1, 4)  # even distance rejected
    with pytest.raises(ValueError):
        logical_error_rate_per_cycle(0.5, 25)  # above threshold rejected


# ---------------------------------------------------------------------------
# Pipeline integration, Phase G lifecycle
# ---------------------------------------------------------------------------


def make_optimizer(**kwargs) -> HardwareAwareOptimizer:
    return HardwareAwareOptimizer(calibrate_hardware=True, **kwargs)


def test_pipeline_attention_passes_with_tiled_attention():
    optimizer = make_optimizer()
    fixture = attention_fixture(sequence=48, dim=16, mask_kind="causal", seed=21)
    request = OptimizationRequest(
        "t", "att", "attention", (48, 48, 16), call_frequency=20.0,
        correctness_contract=CorrectnessContract(rel_tolerance=1e-9),
    )
    result = optimizer.run(request, fixture)
    assert result.validation_status is ValidationStatus.PASSED
    assert result.algorithm_identity == "tiled_attention"
    assert result.measured_error.within_tolerance is True
    assert result.metrics["latency_p95_seconds"] >= 0.0
    envelope = result.to_envelope()
    assert envelope["algorithm_identity"] == "tiled_attention"
    assert envelope["step_count"] == len(result.step_trace)
    assert envelope["provenance"]["module_version"]
    executed_steps = {entry["step"] for entry in result.step_trace}
    assert {17, 18, 19, 20, 21, 22} <= executed_steps


def test_pipeline_records_algorithm_backend_dtype_and_configuration():
    optimizer = make_optimizer()
    rng = np.random.default_rng(2)
    request = OptimizationRequest(
        "t", "mm", "matrix_multiply", (32, 24, 16), call_frequency=10.0,
        correctness_contract=CorrectnessContract(rel_tolerance=5e-3),
    )
    result = optimizer.run(
        request, {"A": rng.uniform(0.5, 1.5, (32, 24)), "B": rng.uniform(0.5, 1.5, (24, 16))}
    )
    envelope = result.to_envelope()
    assert envelope["configuration"]["algorithm"] == result.algorithm_identity
    assert envelope["configuration"]["backend"] == result.plan.backend
    assert envelope["dtype"] == result.plan.precision
    assert envelope["configuration"]["config_hash"] == result.plan.config_hash
    assert envelope["error"]["within_tolerance"] is True


def test_feature_flag_rollback_restores_baseline():
    optimizer = make_optimizer()
    fixture = attention_fixture(sequence=48, dim=16, mask_kind="causal", seed=21)
    request = OptimizationRequest(
        "t", "att", "attention", (48, 48, 16), call_frequency=20.0,
        correctness_contract=CorrectnessContract(rel_tolerance=1e-9),
    )
    first = optimizer.run(request, fixture)
    assert first.algorithm_identity == "tiled_attention"
    optimizer.flags.rollback("tiled_attention")
    second = optimizer.run(request, fixture)
    assert second.algorithm_identity == "reference_attention"
    assert second.validation_status is ValidationStatus.REFERENCE_ONLY


def test_cancellation_returns_safely():
    optimizer = make_optimizer()
    fixture = attention_fixture(sequence=32, dim=8, seed=5)
    request = OptimizationRequest(
        "t", "att", "attention", (32, 32, 8),
        correctness_contract=CorrectnessContract(rel_tolerance=1e-9),
    )
    token = CancellationToken()
    token.cancel()
    result = optimizer.run(request, fixture, cancellation=token)
    assert result.validation_status is ValidationStatus.CANCELLED
    assert any("cancelled" in note for note in result.metrics.get("notes", []))


def test_exhaustive_step_registry_covers_48_source_steps():
    assert len(OPERATION_REGISTRY) == 48
    steps = [spec.step for spec in OPERATION_REGISTRY]
    assert steps == list(range(1, 49))
    phases = {"A": 8, "B": 8, "C": 7, "D": 9, "E": 10, "F": 6}
    counts: dict[str, int] = {}
    for spec in OPERATION_REGISTRY:
        counts[spec.phase] = counts.get(spec.phase, 0) + 1
    assert counts == phases


def test_determinism_of_pipeline_outputs():
    optimizer = make_optimizer()
    fixture = attention_fixture(sequence=32, dim=12, seed=21)
    request = OptimizationRequest(
        "t", "att", "attention", (32, 32, 12),
        correctness_contract=CorrectnessContract(rel_tolerance=1e-9),
    )
    first = optimizer.run(request, fixture)
    second = optimizer.run(request, fixture)
    assert np.array_equal(np.asarray(first.output_reference), np.asarray(second.output_reference))


def test_adversarial_fixture_comparison_for_attention():
    fixture = adversarial_attention_fixture(sequence=24, dim=8)
    optimized = lambda data: tiled_attention(data["q"], data["k"], data["v"], data["mask"], tile_query=8, tile_key=8)[0]  # noqa: E731
    reference = lambda data: reference_attention(data["q"], data["k"], data["v"], data["mask"])[0]  # noqa: E731
    report = run_adversarial_suite(optimized, reference)
    assert report["within_tolerance"] is True
    assert report["finite"] is True


def test_regression_records_round_trip(tmp_path: Path):
    record = RegressionRecord(
        name="attention", config_hash="abc123", metrics={"p95": 0.001},
        fixture_seed=21, notes=("regression evidence",),
    )
    target = tmp_path / "regression.json"
    save = save_regression_records((record,), target)
    assert save["record_count"] == 1
    loaded = load_regression_records(target)
    assert loaded[0].name == "attention"
    assert loaded[0].config_hash == "abc123"
    assert loaded[0].metrics["p95"] == 0.001


# ---------------------------------------------------------------------------
# ORE integration: Blackboard publication and ore_logger SQLite
# ---------------------------------------------------------------------------


def test_publishes_metadata_envelope_to_blackboard():
    try:
        from ore_workspace import SharedWorkspace
    except ImportError:  # pragma: no cover - repository optional
        pytest.skip("ore_workspace not importable")
    workspace = SharedWorkspace("hacol-demo-input", "verify HACOL evidence publication")
    optimizer = make_optimizer(workspace=workspace)
    fixture = attention_fixture(sequence=32, dim=8, seed=5)
    request = OptimizationRequest(
        "t", "att", "attention", (32, 32, 8),
        correctness_contract=CorrectnessContract(rel_tolerance=1e-9),
    )
    result = optimizer.run(request, fixture)
    report = publish_to_blackboard(build_blackboard_envelope(result), workspace)
    assert report["published"] is True
    assert report["workspace_id"] == workspace.workspace_id


def test_envelope_is_metadata_only():
    optimizer = make_optimizer()
    fixture = attention_fixture(sequence=32, dim=8, seed=5)
    request = OptimizationRequest(
        "t", "att", "attention", (32, 32, 8),
        correctness_contract=CorrectnessContract(rel_tolerance=1e-9),
    )
    result = optimizer.run(request, fixture)
    envelope = build_blackboard_envelope(result)
    assert isinstance(envelope["validation_status"], str)
    assert envelope["step_count"] >= 8


def test_logs_steps_to_ore_logger_sqlite(tmp_path: Path):
    optimizer = make_optimizer()
    fixture = attention_fixture(sequence=32, dim=8, seed=5)
    request = OptimizationRequest(
        "t", "att", "attention", (32, 32, 8),
        correctness_contract=CorrectnessContract(rel_tolerance=1e-9),
    )
    result = optimizer.run(request, fixture)
    database_path = tmp_path / "ore_logger.sqlite3"
    report = record_result_to_sqlite(result, database_path=database_path)
    assert report["stored_in_sqlite"] is True
    assert report["run_id"]
    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            "SELECT COUNT(*) FROM events WHERE run_id = ?", (report["run_id"],)
        ).fetchone()
        kinds = connection.execute(
            "SELECT DISTINCT event_kind FROM events WHERE run_id = ?", (report["run_id"],)
        ).fetchall()
    finally:
        connection.close()
    assert rows[0] == len(result.step_trace) + 1
    assert {row[0] for row in kinds} == {"hacol_step", "hacol_result"}


def test_register_hacol_annotations_in_shared_catalogue(tmp_path: Path):
    report = register_hacol_annotations(database_path=tmp_path / "logger.sqlite3")
    assert report.get("registered", report.get("symbol_count", 0)) is not False
    if report.get("database_path"):
        connection = sqlite3.connect(report["database_path"])
        try:
            count = connection.execute(
                "SELECT COUNT(*) FROM annotations WHERE module_name LIKE 'ore_hacol%'"
            ).fetchone()[0]
        finally:
            connection.close()
        assert count > 50  # classes, methods, and functions of the package


# ---------------------------------------------------------------------------
# Core numeric primitives kept under direct test
# ---------------------------------------------------------------------------


def test_stable_softmax_handles_wide_dynamic_range():
    logits = np.array([[1000.0, 999.0, -1000.0], [1.0, 2.0, 3.0]])
    result = stable_softmax(logits)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-12)


def test_compare_arrays_reports_absolute_and_relative_error():
    reference = np.array([1.0, 2.0, 4.0])
    candidate = np.array([1.1, 2.0, 4.0])
    contract = CorrectnessContract(abs_tolerance=0.0, rel_tolerance=0.2)
    report = compare_arrays(candidate, reference, contract)
    assert report.within_tolerance is True
    assert report.max_abs == pytest.approx(0.1)
    assert report.max_rel == pytest.approx(0.1, rel=1e-6)


def test_exact_and_heuristic_qubo_solvers_agree_on_optimum():
    fixture = qubo_assignment_fixture(tasks=3, resources=4, seed=17)
    problem = AssignmentProblem(fixture["cost"])
    encoded = encode_assignment_qubo(problem, penalty=100.0)
    exact = exact_qubo_solver(encoded["Q"], encoded["c"])
    annealed = simulated_annealing(encoded["Q"], encoded["c"], seed=7, sweeps=60)
    tabu = tabu_search(encoded["Q"], encoded["c"], seed=7)
    assert exact["energy"] <= annealed["energy"] + 1e-9
    assert exact["energy"] <= tabu["energy"] + 1e-9
    decode = decode_assignment(exact["z"], 3, 4)
    assert decode["feasible"] is True
