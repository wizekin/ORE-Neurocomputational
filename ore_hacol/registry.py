"""Source-ordered registry of the 48 integrated HACOL steps (spec section 4).

Each entry records the step number, phase, name, purpose, mathematical form
(where applicable), and the control/safeguard that must remain active. The
pipeline appends one trace record per executed step so results stay auditable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HACOLStepSpec:
    """Document one integrated step and its control (safeguard)."""

    step: int
    phase: str
    name: str
    purpose: str
    mathematical_form: str
    control: str

    @property
    def step_id(self) -> str:
        return f"HACOL-{self.step:02d}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"step_id": self.step_id}


_ROWS: tuple[tuple[str, str, str, str, str], ...] = (
    # Phase A -- Discover and characterize workloads
    ("A", "Receive an optimization request",
     "Require operation identity, input schema/dimensions, correctness contract, latency target, memory limit, precision requirements, and approximation permission.",
     "request = (operation_id, shape, dtype, contract, budgets)",
     "Reject requests without operation identity or executable workload definition."),
    ("A", "Identify the mathematical primitive",
     "Classify the workload as attention, matrix multiplication, similarity, factorization, least squares, transform, reduction, optimization, or another explicit kernel.",
     "kind = classify(operation)",
     "Reject requests without an executable workload definition."),
    ("A", "Trace the call path",
     "Record invoking module, class/function, task ID, data lineage, call frequency, and downstream consumers using structured ore_logger annotations.",
     "trace = (module, class, function, task_id, frequency)",
     "Structured annotations only; no raw sensitive payloads."),
    ("A", "Establish a reference implementation",
     "Identify or create a trusted conventional implementation with deterministic settings where available.",
     "y_ref = f_ref(x)",
     "Reference behavior recorded before any optimization."),
    ("A", "Build representative benchmark fixtures",
     "Capture realistic dimensions, sparsity, distributions, batching and layout with synthetic fixtures where production data cannot be retained.",
     "x ~ F(shape, distribution, seed)",
     "Avoid toy-only benchmarks; fixtures are seeded and reproducible."),
    ("A", "Measure end-to-end cost",
     "Measure latency, warm/cold startup, throughput, CPU/GPU time, peak memory, allocations, transfers, queue delay, and energy where measurable.",
     "t = t_kernel + t_orchestration; peak = max(memory)",
     "Separate kernel time from orchestration."),
    ("A", "Identify the bottleneck",
     "Determine compute-, bandwidth-, launch-, transfer-, synchronization-, or parallelism-bound behavior using profiler evidence.",
     "AI = flops / bytes; bound = argmax(t_compute, t_memory)",
     "Classification requires measured evidence, not assumption."),
    ("A", "Estimate optimization value",
     "Estimate call frequency times per-call savings, subtracting setup, conversion, compilation and maintenance cost.",
     "value = frequency x savings - setup - maintenance",
     "Prioritize end-to-end benefit; reject complexity without benefit."),
    # Phase B -- Numerical contract and admission
    ("B", "Define correctness requirements",
     "Specify exactness, absolute/relative error, conservation, monotonicity, feasibility, stability, determinism, and downstream sensitivity as relevant.",
     "eps_abs = |y_hat - y|; eps_rel = |y_hat - y| / max(|y|, eps)",
     "Reject candidates without correctness contracts."),
    ("B", "Classify consequence level",
     "Label output exploratory, advisory, decision-support, scientific, or high-assurance and tighten tolerances accordingly.",
     "level in {exploratory, ..., high_assurance}",
     "High-consequence use requires independent verification."),
    ("B", "Decide whether approximation is permitted",
     "If exactness is required, disable randomized/low-rank methods unless equivalence or a valid certificate is available.",
     "approx_permitted = not exactness_required",
     "Record the decision in the plan rationale."),
    ("B", "Build a sensitivity profile",
     "Identify ill-conditioning, cancellation, dynamic range, accumulation depth, near-singularity, and downstream decision sensitivity.",
     "kappa(A); range = max|x| / min|x|",
     "Sensitivity evidence gates precision and approximation choices."),
    ("B", "Discover hardware capabilities",
     "Query CPU vector features, cores, memory capacity/bandwidth, accelerator availability, supported dtypes, runtime versions and utilization.",
     "H = (features, cores, mem, bw, dtypes, runtimes)",
     "Record hardware/runtime capability identity."),
    ("B", "Query CROS",
     "Ask AdaptiveResourceCoordinator/CROS for CPU, memory, accelerator slots, queue pressure, deadline slack and reserved capacity.",
     "ask = (cpu_share, memory, slots); decision = CROS(ask)",
     "HACOL recommends; CROS retains global allocation authority."),
    ("B", "Estimate candidate cost",
     "Estimate arithmetic, memory traffic, temporary storage, transfers, compilation and concurrency effects for each algorithm/backend pair.",
     "t_est = flops / rate + bytes / bandwidth + setup",
     "Cost models are estimates; measurements remain authoritative."),
    ("B", "Admit or defer",
     "Execute only when budgets and deadline are credible; otherwise queue, reduce batch size, use a cheaper baseline, or return a controlled deferral.",
     "admit iff C_used <= C_budget",
     "Hard caps and controlled deferral protect ORE."),
    # Phase C -- IO-aware attention
    ("C", "Check attention eligibility",
     "Confirm actual query/key/value attention semantics and supported dimensions.",
     "is_attention(Q, K, V) and dims supported",
     "Never apply FlashAttention to generic matrix operations."),
    ("C", "Select tile dimensions",
     "Respect device memory hierarchy, shared-memory/register limits, occupancy and supported kernel configuration.",
     "(tile_q, tile_k) = argmax throughput subject to memory limits",
     "Device-aware tuning; poor tiles regress occupancy."),
    ("C", "Compute tiled query-key scores",
     "Compute blocks of QK^T / sqrt(d_k) without materializing the full score matrix in global memory.",
     "S_block = (Q_i K_j^T) / sqrt(d_k)",
     "Preserve masks and scale order."),
    ("C", "Update online softmax",
     "Maintain running (m, l, a), rescaling previous accumulators when the maximum changes.",
     "m' = max(m, m_b); l' = e^{m-m'} l + sum e^{x-m'}; a' = e^{m-m'} a + sum e^{x-m'} v",
     "Stable running maximum; handle all-masked rows explicitly."),
    ("C", "Accumulate value contributions",
     "Add each block's weighted values and normalize at completion, preserving the mathematical attention semantics.",
     "o = a' / l'",
     "Attention semantics preserved (rows sum to one)."),
    ("C", "Validate attention output",
     "Compare against reference for causal/padding masks, extreme logits, long sequences and mixed precision.",
     "||o_tiled - o_ref|| <= eps",
     "Record memory and data movement."),
    ("C", "Fallback on incompatibility",
     "If shape, device, dtype, mask or kernel is unsupported, use trusted attention.",
     "o = f_ref(Q, K, V)",
     "Never silently change semantics."),
    # Phase D -- Randomized numerical linear algebra
    ("D", "Detect low-rank opportunity",
     "Estimate dimensions, spectrum/rank indicators, reuse count and required accuracy; confirm savings exceed decomposition and reconstruction cost.",
     "spectrum decay + reuse count vs decomposition cost",
     "Savings must exceed decomposition and reconstruction cost."),
    ("D", "Select rank and oversampling",
     "Choose k and p from accuracy/memory objectives; record random seed and rationale.",
     "(k, p) = select(accuracy, memory)",
     "Seed and rationale recorded."),
    ("D", "Generate a test matrix",
     "Construct Omega using a documented distribution or structured transform.",
     "Omega ~ D(n, k + p)",
     "Preserve reproducibility where required."),
    ("D", "Compute range sketch",
     "Calculate Y = A Omega, using blocks when needed; apply power iterations only when error tests justify extra work.",
     "Y = A Omega",
     "Power iterations require error justification."),
    ("D", "Orthonormalize",
     "Compute Q = orth(Y) with stable QR/SVD; check rank deficiency and finite values.",
     "Q = orth(Y)",
     "Rank check and finite values."),
    ("D", "Form compressed matrix",
     "Compute B = Q^T A, factorize the smaller matrix, and reconstruct only required outputs.",
     "B = Q^T A; A ~ QB",
     "Reconstruct only required outputs."),
    ("D", "Measure approximation error",
     "Evaluate ||A - QB|| or a justified estimator and compare downstream outputs against reference.",
     "||A - QB||_F / ||A||_F <= eps_allowed",
     "Reject excess error."),
    ("D", "Apply sketching to eligible least squares",
     "Solve the sketched system, then evaluate ||Ax - b|| on the original system and inspect conditioning.",
     "min ||S A x - S b||^2; check ||A x - b||",
     "Original-system residual is authoritative."),
    ("D", "Preserve approximation metadata",
     "Store rank, oversampling, sketch family, seed, power iterations, error estimate and validation status.",
     "meta = (k, p, family, seed, iters, err, status)",
     "Metadata travels with every approximate result."),
    # Phase E -- Structured transforms and mixed precision
    ("E", "Detect exploitable structure",
     "Test for butterfly, Toeplitz-like, hierarchical, sparse or low-rank-block structure.",
     "structure tests with tolerance",
     "A hoped-for speedup is not evidence."),
    ("E", "Choose a structure-preserving representation",
     "Define factor/block layout, dimensions, assumptions and reconstruction semantics; retain a reference path.",
     "A ~ prod_s B_s",
     "Reference path retained."),
    ("E", "Construct and execute factors",
     "Build the representation with a validated method; order operations for locality and stability; measure preprocessing amortization.",
     "y = B_L ... B_1 x",
     "Measure preprocessing amortization."),
    ("E", "Verify structured results",
     "Compare reconstruction and operation outputs with dense/reference computation on representative and adversarial inputs.",
     "||y_struct - y_dense|| <= eps",
     "Reject if accuracy or runtime is worse."),
    ("E", "Assign precision per operation",
     "Consider FP16, BF16, FP32 and FP64 based on hardware, dynamic range, conditioning and error contract.",
     "dtype = policy(range, kappa, eps_allowed)",
     "Avoid global dtype changes."),
    ("E", "Separate product and accumulation precision",
     "Use lower-precision products only where safe; accumulate sensitive sums at higher precision where required.",
     "C = sum_blocks (A_b B_b)_low + ... acc_high",
     "Record actual library behavior."),
    ("E", "Protect numerical range",
     "Apply justified scaling and overflow/underflow checks.",
     "|x| <= max(dtype); x finite",
     "Do not silently clip unless specified."),
    ("E", "Apply iterative refinement where valid",
     "For eligible linear systems, solve initially at lower precision, compute residual at higher precision, solve correction and update.",
     "r = b - A x; x <- x + A^{-1}_low r",
     "Check conditioning and convergence."),
    ("E", "Verify with higher precision",
     "Recompute selected outputs/residuals in FP64 or a trusted reference and test downstream decision stability.",
     "||y_low - y_fp64|| <= eps",
     "Test downstream decision stability."),
    ("E", "Reject unsafe precision",
     "Fall back to higher precision or a stable algorithm when error, range, conditioning or constraint violations exceed limits.",
     "if error/range fails: dtype <- higher",
     "Fall back; never corrupt silently."),
    # Phase F -- Classical optimization and quantum research gate
    ("F", "Formulate combinatorial problems classically first",
     "Define variables, objective, constraints and feasibility; establish OR-Tools CP-SAT, MIP, or strong heuristic baselines.",
     "min c^T x s.t. Ax <= b",
     "Classical baselines established first."),
    ("F", "Create QUBO/Ising candidate only when useful",
     "Encode binary decisions and penalties; verify correspondence between feasible original solutions and low-energy encoded solutions.",
     "min z^T Q z + c^T z; s = 2z - 1",
     "Penalties must not distort the objective."),
    ("F", "Compare classical QUBO solvers",
     "Test simulated annealing, tabu/local search, exact, CP-SAT and MIP baselines; record feasibility, optimality gap, runtime and scaling.",
     "gap = (E_best - E_opt) / |E_opt|",
     "Record feasibility and optimality gap."),
    ("F", "Evaluate quantum annealing conditionally",
     "Proceed only when access, problem size, embedding/connectivity, precision, queue time and end-to-end costs are known.",
     "backend available and end-to-end cost known",
     "Hardware is an optional solver backend."),
    ("F", "Keep Shor and Grover as research records",
     "Record mathematical scope and assumptions; do not route ordinary ORE search, factorization or scheduling to quantum algorithms.",
     "Grover O(sqrt(N)) queries (ideal model); Shor poly(n)",
     "No quantum routing for ordinary production tasks."),
    ("F", "Include quantum error-correction overhead",
     "Evaluate logical error targets, code distance, physical/logical qubit overhead, syndrome extraction, decoding, magic-state costs and fault-tolerant runtime.",
     "physical/logical ~ 2 d^2; p_L ~ 0.1 (100 p)^{(d+1)/2}",
     "Never compare ideal query complexity directly with end-to-end classical runtime."),
    # Phase G is the unnumbered post-execution lifecycle of spec section 4 and is
    # represented by PHASE_G_ACTIONS below, not by numbered steps.
)

#: The 48 integrated steps, in source order (A=8, B=8, C=7, D=9, E=10, F=6).
OPERATION_REGISTRY = tuple(
    HACOLStepSpec(index, *row)
    for index, row in enumerate(_ROWS, start=1)
)

#: Phase G lifecycle actions (spec section 4, final paragraph -- unnumbered).
PHASE_G_ACTIONS: tuple[str, ...] = (
    "Compare optimized and reference outputs on deterministic fixtures, property tests, "
    "edge cases and workload-scale samples; check absolute/relative error, constraints "
    "and downstream decisions.",
    "Benchmark end-to-end including conversion, transfers, compilation, warmup, "
    "synchronization and materialization; report median, p95 and p99 latency.",
    "Publish a versioned result envelope containing operation, algorithm/backend, "
    "hardware, dtype, configuration, approximation metadata, error, validation status "
    "and provenance to the Blackboard.",
    "Send useful summaries to the Blackboard; Main Brain Coordinator and "
    "CognitiveTaskCoordinator decide reasoning relevance.",
    "Log module/class/function, task, dimensions, algorithm, precision, device, "
    "duration, memory, transfers, error, status and provenance to ore_logger/SQLite.",
    "Promote behind feature flags, monitor regressions, and roll back to reference. "
    "HACOL must never bypass CROS.",
)

#: Steps that are executed once per request for every operation kind.
ALWAYS_STEPS = tuple(spec.step for spec in OPERATION_REGISTRY if spec.phase in ("A", "B"))


def registry_as_dicts() -> list[dict[str, Any]]:
    """Return all 48 integrated steps in source order."""
    return [item.to_dict() for item in OPERATION_REGISTRY]


def step_by_number(step: int) -> HACOLStepSpec:
    """Return the step spec for a 1-based step number."""
    if not 1 <= step <= len(OPERATION_REGISTRY):
        raise IndexError(f"step {step} outside 1..{len(OPERATION_REGISTRY)}")
    return OPERATION_REGISTRY[step - 1]


def steps_for_phase(phase: str) -> tuple[HACOLStepSpec, ...]:
    """Return all step specs within one phase letter (A-G)."""
    return tuple(spec for spec in OPERATION_REGISTRY if spec.phase == phase)
