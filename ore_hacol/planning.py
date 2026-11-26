"""Candidate generation, cost estimation, and plan selection (spec section 5
policy table; integrated steps 15-16).

Implements the workload-to-candidate policy exactly as the source specifies:
large attention -> FlashAttention/IO-aware tiling; approximately low-rank
matrix -> randomized SVD/sketching; repeated structured transform ->
butterfly/fast/hierarchical; numerically tolerant neural math -> mixed
precision; constrained assignment -> CP-SAT/MIP/heuristics first with QUBO
experimental; integer factorization research -> Shor awareness; unstructured
oracle search -> Grover awareness. A baseline is always chosen before any
optimization; newness is never a selection criterion.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np

from .contracts import (
    AdmissionDecision,
    ContractAssessment,
    CostEstimate,
    HardwareProfile,
    OperationKind,
    OptimizationPlan,
    OptimizationRequest,
)

#: The algorithm-selection policy table (spec section 5, transcribed verbatim).
POLICY_TABLE: tuple[dict[str, str], ...] = (
    {"workload": "Large attention", "candidate": "FlashAttention/IO-aware tiling",
     "preconditions": "Compatible attention and device", "control": "Stable reference attention"},
    {"workload": "Approximately low-rank matrix", "candidate": "Randomized SVD/sketching",
     "preconditions": "Approximation permitted", "control": "Dense residual"},
    {"workload": "Repeated structured transform", "candidate": "Butterfly/fast/hierarchical",
     "preconditions": "Demonstrated structure", "control": "Dense reference"},
    {"workload": "Numerically tolerant neural math", "candidate": "Mixed precision",
     "preconditions": "Error budget and range checks", "control": "Higher-precision validation"},
    {"workload": "Constrained assignment", "candidate": "CP-SAT/MIP/heuristics first; QUBO experimental",
     "preconditions": "Explicit objective and constraints", "control": "Feasibility and gap"},
    {"workload": "Integer factorization research", "candidate": "Shor awareness",
     "preconditions": "Fault-tolerant quantum capability", "control": "Classical baseline"},
    {"workload": "Unstructured oracle search", "candidate": "Grover awareness",
     "preconditions": "Valid oracle and hardware", "control": "End-to-end oracle cost"},
)


@dataclass(frozen=True)
class Candidate:
    """One algorithm/backend/precision candidate from the policy table."""

    algorithm: str
    backend: str
    precision: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    estimated_cost: CostEstimate | None = None
    expected_error: float | None = None
    preconditions: tuple[str, ...] = ()
    controls: tuple[str, ...] = ()
    tier: str = "optimized"  # "reference" | "optimized" | "experimental"

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "backend": self.backend,
            "precision": self.precision,
            "parameters": dict(self.parameters),
            "estimated_cost": self.estimated_cost.to_dict() if self.estimated_cost else None,
            "expected_error": self.expected_error,
            "preconditions": list(self.preconditions),
            "controls": list(self.controls),
            "tier": self.tier,
        }


def _time_estimate(
    flops: float, bytes_moved: float, profile: HardwareProfile, setup: float
) -> float:
    # Additive (serial) time model: no compute/memory overlap is assumed, so
    # estimated savings from reduced memory traffic are counted conservatively.
    compute = (
        flops / profile.compute_rate_flops_per_second
        if profile.compute_rate_flops_per_second > 0
        else 0.0
    )
    memory = (
        bytes_moved / profile.bandwidth_bytes_per_second
        if profile.bandwidth_bytes_per_second > 0
        else 0.0
    )
    return compute + memory + setup


def _attention_cost(request: OptimizationRequest, hardware: HardwareProfile) -> tuple[CostEstimate, CostEstimate]:
    rows, keys, dim = request.shape[0], request.shape[1], request.shape[2]
    flops = 2.0 * rows * keys * dim + 2.0 * rows * keys * dim + 5.0 * rows * keys
    streams = (rows + keys) * dim * 8 + rows * dim * 8
    reference_bytes = streams + rows * keys * 8
    tiled_bytes = streams + 2 * (32 * 32 * 8)
    reference = CostEstimate(
        flops=flops, bytes_moved=reference_bytes,
        temporary_bytes=rows * keys * 8,
        estimated_seconds=_time_estimate(flops, reference_bytes, hardware, 0.0),
        rationale="materializes the full score matrix in global memory",
    )
    tiled = CostEstimate(
        flops=flops, bytes_moved=tiled_bytes,
        temporary_bytes=2 * (32 * 32 * 8),
        estimated_seconds=_time_estimate(flops, tiled_bytes, hardware, 0.0),
        rationale="IO-aware tiling avoids the full score matrix",
    )
    return reference, tiled


def candidates_for(
    request: OptimizationRequest,
    hardware: HardwareProfile,
    assessment: ContractAssessment | None = None,
) -> tuple[Candidate, ...]:
    """Generate the candidate set for one request from the policy table."""
    candidates: list[Candidate] = []
    kind = request.kind
    approximation_permitted = (
        assessment.approximation_permitted if assessment else request.approximation_allowed
    )

    if kind is OperationKind.ATTENTION:
        reference_cost, tiled_cost = _attention_cost(request, hardware)
        candidates.append(
            Candidate(
                algorithm="reference_attention",
                backend="cpu_reference",
                precision="float64",
                estimated_cost=reference_cost,
                expected_error=0.0,
                preconditions=("attention semantics confirmed",),
                controls=("stable reference attention",),
                tier="reference",
            )
        )
        candidates.append(
            Candidate(
                algorithm="tiled_attention",
                backend="cpu_optimized",
                precision=request.dtype,
                parameters={"tile_query": 32, "tile_key": 32},
                estimated_cost=tiled_cost,
                expected_error=1e-12,
                preconditions=("attention semantics confirmed", "supported shape/dtype/mask"),
                controls=("compare with stable reference within tolerance",),
            )
        )
        return tuple(candidates)

    if kind is OperationKind.MATRIX_MULTIPLY:
        rows, inner, columns = request.shape
        flops = 2.0 * rows * inner * columns
        bytes64 = (rows * inner + inner * columns + rows * columns) * 8
        bytes32 = (rows * inner + inner * columns + rows * columns) * 4
        candidates.append(
            Candidate(
                algorithm="reference_matmul",
                backend="cpu_reference",
                precision="float64",
                estimated_cost=CostEstimate(
                    flops=flops, bytes_moved=bytes64,
                    estimated_seconds=_time_estimate(flops, bytes64, hardware, 0.0),
                    rationale="dense float64 GEMM reference",
                ),
                expected_error=0.0,
                controls=("always chosen before an optimization",),
                tier="reference",
            )
        )
        candidates.append(
            Candidate(
                algorithm="mixed_precision_matmul",
                backend="cpu_optimized",
                precision="float32",
                parameters={"product_dtype": "float32", "accumulate_dtype": "float32", "block": 64},
                estimated_cost=CostEstimate(
                    flops=flops, bytes_moved=bytes32,
                    temporary_bytes=64 * 64 * 4,
                    estimated_seconds=_time_estimate(flops, bytes32, hardware, 0.0),
                    rationale="halved operand traffic relative to float64",
                ),
                expected_error=1e-5,
                preconditions=("error budget and range checks", "not exactness-required"),
                controls=("higher-precision validation", "fallback on overflow"),
            )
        )
        return tuple(candidates)

    if kind is OperationKind.FACTORIZATION:
        rows, columns = request.shape[0], request.shape[1]
        dense_flops = 2.0 * rows * columns * min(rows, columns)
        candidates.append(
            Candidate(
                algorithm="reference_svd",
                backend="cpu_reference",
                precision="float64",
                estimated_cost=CostEstimate(
                    flops=dense_flops, bytes_moved=rows * columns * 8,
                    estimated_seconds=_time_estimate(dense_flops, rows * columns * 8, hardware, 0.0),
                    rationale="dense deterministic SVD",
                ),
                expected_error=0.0,
                controls=("dense residual authority",),
                tier="reference",
            )
        )
        if approximation_permitted:
            width = min(rows, columns, 8) + 5
            sketch_flops = 2.0 * rows * columns * width
            candidates.append(
                Candidate(
                    algorithm="randomized_svd",
                    backend="cpu_optimized",
                    precision="float64",
                    parameters={"rank": min(rows, columns, 8), "oversampling": 5,
                                "seed": 21, "power_iterations": 0},
                    estimated_cost=CostEstimate(
                        flops=sketch_flops, bytes_moved=rows * columns * 8,
                        temporary_bytes=(rows + columns) * width * 8,
                        estimated_seconds=_time_estimate(sketch_flops, rows * columns * 8, hardware, 0.0),
                        rationale="randomized range finding with declared error budget",
                    ),
                    expected_error=None,
                    preconditions=("approximation permitted",
                                    "savings exceed decomposition cost"),
                    controls=("dense residual",
                              "reject approximation exceeding the error budget"),
                )
            )
        return tuple(candidates)

    if kind is OperationKind.LEAST_SQUARES:
        rows, columns = request.shape[0], request.shape[1]
        dense_flops = 2.0 * rows * columns * columns
        candidates.append(
            Candidate(
                algorithm="reference_lstsq",
                backend="cpu_reference",
                precision="float64",
                estimated_cost=CostEstimate(
                    flops=dense_flops, bytes_moved=(rows * columns + rows) * 8,
                    estimated_seconds=_time_estimate(dense_flops, rows * columns * 8, hardware, 0.0),
                    rationale="dense QR least squares",
                ),
                expected_error=0.0,
                controls=("original-system residual authority",),
                tier="reference",
            )
        )
        if approximation_permitted:
            sketch_rows = max(2 * columns + 10, 1)
            # Honest model: the sketch must be applied to A (2 s m n) before the
            # smaller solve (2 s n^2); for small dense problems the reference QR
            # solve is genuinely cheaper and the baseline is kept.
            sketch_flops = 2.0 * sketch_rows * rows * columns + 2.0 * sketch_rows * columns * columns
            candidates.append(
                Candidate(
                    algorithm="sketched_least_squares",
                    backend="cpu_optimized",
                    precision="float64",
                    parameters={"sketch_rows": sketch_rows, "seed": 21, "family": "gaussian"},
                    estimated_cost=CostEstimate(
                        flops=sketch_flops,
                        bytes_moved=(sketch_rows * rows + rows * columns) * 8,
                        temporary_bytes=sketch_rows * columns * 8,
                        estimated_seconds=_time_estimate(
                            sketch_flops, (sketch_rows * rows + rows * columns) * 8, hardware, 0.0
                        ),
                        rationale="sketch-and-solve with original-system residual validation",
                    ),
                    expected_error=None,
                    preconditions=("approximation permitted", "conditioning inspected"),
                    controls=("residual evaluated on the original system",),
                )
            )
        return tuple(candidates)

    if kind is OperationKind.TRANSFORM:
        size = int(request.shape[-1])
        dense_flops = 2.0 * size * size
        candidates.append(
            Candidate(
                algorithm="dense_matvec",
                backend="cpu_reference",
                precision="float64",
                estimated_cost=CostEstimate(
                    flops=dense_flops, bytes_moved=size * size * 8 + size * 8,
                    estimated_seconds=_time_estimate(dense_flops, size * 8, hardware, 0.0),
                    rationale="dense O(n^2) reference apply",
                ),
                expected_error=0.0,
                controls=("dense reference",),
                tier="reference",
            )
        )
        if size >= 8 and size & (size - 1) == 0:
            stages = float(max(1, int(np.log2(size))))
            candidates.append(
                Candidate(
                    algorithm="butterfly_transform",
                    backend="cpu_optimized",
                    precision="float64",
                    parameters={"size": size, "seed": 21},
                    estimated_cost=CostEstimate(
                        flops=2.0 * size * stages,
                        bytes_moved=size * 8,
                        temporary_bytes=size * 8,
                        estimated_seconds=_time_estimate(2.0 * size * stages, size * 8, hardware, 0.0),
                        rationale="O(n log n) structured apply after demonstrated structure",
                    ),
                    expected_error=1e-12,
                    preconditions=("demonstrated structure with verified reconstruction",),
                    controls=("dense reference comparison", "reject if runtime is worse"),
                )
            )
        return tuple(candidates)

    if kind is OperationKind.OPTIMIZATION:
        tasks, resources = request.shape[0], request.shape[1]
        variables = tasks * resources
        candidates.append(
            Candidate(
                algorithm="classical_assignment_baselines",
                backend="cpu_reference",
                precision="float64",
                parameters={"solver": "hungarian+greedy", "seed": 21},
                estimated_cost=CostEstimate(
                    flops=float(2 * variables),
                    bytes_moved=variables * 8,
                    estimated_seconds=_time_estimate(2.0 * variables, variables * 8, hardware, 0.0),
                    rationale="classical baselines established first (Hungarian + greedy)",
                ),
                expected_error=0.0,
                preconditions=("explicit objective and constraints",),
                controls=("feasibility and optimality gap recorded",),
                tier="reference",
            )
        )
        candidates.append(
            Candidate(
                algorithm="qubo_simulated_annealing",
                backend="cpu_optimized",
                precision="float64",
                parameters={"penalty": "auto", "seed": 21, "sweeps": 60},
                estimated_cost=CostEstimate(
                    flops=float(variables * 60 * variables),
                    bytes_moved=float(variables * variables * 8),
                    estimated_seconds=_time_estimate(
                        float(variables * 60 * variables), float(variables**2 * 8), hardware, 0.0
                    ),
                    rationale="experimental QUBO solver; never promoted above classical "
                              "baselines without end-to-end evidence",
                ),
                expected_error=None,
                preconditions=("explicit objective and constraints",
                               "feasible mapping verified"),
                controls=("compare with CP-SAT/MIP/heuristics",
                          "record feasibility and gap"),
                tier="experimental",
            )
        )
        return tuple(candidates)

    # Unknown or research workloads: reference only, with research records kept.
    candidates.append(
        Candidate(
            algorithm="reference_implementation",
            backend="cpu_reference",
            precision="float64",
            estimated_cost=CostEstimate(
                flops=float(np.prod(request.shape)) * 2.0,
                bytes_moved=float(np.prod(request.shape)) * 8,
                estimated_seconds=0.0,
                rationale="reference-only path for unclassified workloads",
            ),
            expected_error=0.0,
            controls=("baseline before any optimization",),
            tier="reference",
        )
    )
    return tuple(candidates)


def estimate_candidate_cost(
    candidate: Candidate,
    request: OptimizationRequest,
    hardware: HardwareProfile,
) -> Candidate:
    """Attach a coarse CostEstimate to candidates that lack one (step 15)."""
    if candidate.estimated_cost is not None:
        return candidate
    elements = float(np.prod(request.shape))
    flops = elements * 2.0
    bytes_moved = elements * 8
    estimate = CostEstimate(
        flops=flops,
        bytes_moved=bytes_moved,
        estimated_seconds=_time_estimate(flops, bytes_moved, hardware, 0.0),
        rationale=f"coarse model for {candidate.algorithm} on {candidate.backend}",
    )
    return replace(candidate, estimated_cost=estimate)


def choose_plan(
    candidates: tuple[Candidate, ...],
    request: OptimizationRequest,
    assessment: ContractAssessment,
    admission: AdmissionDecision,
    *,
    feature_flags: Mapping[str, bool] | None = None,
    value_threshold_seconds: float = 0.0,
    evidence: Mapping[str, bool] | None = None,
) -> OptimizationPlan:
    """Select the execution plan (integrated step 16: admit or defer).

    Order of authority: correctness contract (approximation permission and
    preconditions), CROS admission, feature flags, then estimated end-to-end
    benefit. The reference plan is always available and is returned whenever
    no optimized candidate passes every gate.
    """
    flags = feature_flags or {}
    proof = evidence or {}
    reference = next(
        (item for item in candidates if item.tier == "reference"),
        candidates[0] if candidates else None,
    )
    if not admission.granted:
        plan = _plan_from(reference, request)
        return _plan_with_rationale(plan, f"deferred by CROS admission: {admission.reason}")
    reference_cost = (
        reference.estimated_cost.estimated_seconds if reference.estimated_cost else 0.0
    )
    best: Candidate | None = None
    best_savings = value_threshold_seconds
    for candidate in candidates:
        if candidate.tier == "reference" or candidate.backend == "research_only":
            continue
        if candidate.algorithm in flags and not flags[candidate.algorithm]:
            continue
        if not _preconditions_satisfied(candidate, assessment, proof):
            continue
        cost = candidate.estimated_cost.estimated_seconds if candidate.estimated_cost else 0.0
        savings = reference_cost - cost
        expected = candidate.expected_error
        if expected is not None and expected > assessment.contract.rel_tolerance:
            continue
        if savings > best_savings:
            best_savings = savings
            best = candidate
    if best is None:
        return _plan_with_rationale(
            _plan_from(reference, request),
            "no optimized candidate passed contract, admission, flags, or value gates; "
            "baseline kept (a baseline is always chosen before an optimization)",
        )
    return _plan_with_rationale(
        _plan_from(best, request),
        f"selected {best.algorithm}: estimated saving {best_savings:.6f}s vs reference "
        f"under contract tolerance {assessment.contract.rel_tolerance:g}; "
        f"fallback reference_implementation",
    )


def _preconditions_satisfied(
    candidate: Candidate,
    assessment: ContractAssessment,
    evidence: Mapping[str, bool],
) -> bool:
    for precondition in candidate.preconditions:
        normalized = precondition.lower()
        if "approximation permitted" in normalized and not assessment.approximation_permitted:
            return False
        if "not exactness-required" in normalized and assessment.contract.exactness_required:
            return False
        if "demonstrated structure" in normalized and not evidence.get("structure_demonstrated", False):
            return False
    return True


def _plan_from(candidate: Candidate, request: OptimizationRequest) -> OptimizationPlan:
    return OptimizationPlan(
        algorithm=candidate.algorithm,
        backend=candidate.backend,
        precision=candidate.precision,
        parameters=dict(candidate.parameters),
        estimated_cost=candidate.estimated_cost,
        expected_error=candidate.expected_error,
        fallback_algorithm="reference_implementation",
        rationale="; ".join(candidate.controls) or "baseline",
    )


def _plan_with_rationale(plan: OptimizationPlan, rationale: str) -> OptimizationPlan:
    combined = f"{plan.rationale} | {rationale}" if plan.rationale else rationale
    return replace(plan, rationale=combined)
