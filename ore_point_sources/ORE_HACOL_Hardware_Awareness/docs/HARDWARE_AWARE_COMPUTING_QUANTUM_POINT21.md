# HACOL — Point 21: Hardware-Aware Numerical Computing and Quantum Algorithms

Implementation of "ORE Integrated Algorithm — Point 21: Hardware-Aware Numerical
Computing and Quantum Algorithms" (source: *Most Intelligent Systems 01-24.txt*,
Point 21, sections 21.1–21.8; canonical relevance rank 24 of 24).

The module implements a **Hardware-Aware Computational Optimization Layer
(HACOL)** that improves ORE's actual mathematical workloads by jointly
considering algorithmic structure, memory movement, processor capabilities,
numerical precision, computational cost, and correctness. Governing rule:
optimize only a measured bottleneck, preserve a trusted reference
implementation, and accept a faster implementation only when accuracy and
correctness satisfy explicit contracts. HACOL recommends; CROS retains global
allocation authority.

## Package layout

Proposed package boundaries from the source spec, reconciled with the live
repository's flat-package convention as one package `ore_hacol`:

| Spec boundary | Module | Content |
|---|---|---|
| `ore_compute` (contracts) | `contracts.py` | `OptimizationRequest`, `HardwareProfile`, `OptimizationPlan`, `OptimizationResult`, `CorrectnessContract`, `ErrorReport`, `ProvenanceRecord`, admission/cost/value records |
| `profiling` | `profiling.py` | Seeded benchmark fixtures, latency statistics (median/p95/p99), bottleneck classification, optimization-value estimation |
| `capabilities` | `capabilities.py` | CPU feature/core/memory discovery, measured bandwidth and GEMM-rate calibration, dtype support, accelerator detection (honest absence) |
| `planning` | `planning.py` | Algorithm-selection policy table (spec section 5), candidate generation, analytical cost models, plan choice with admission gates |
| `attention` | `attention.py` | Stable reference attention, online-softmax streaming primitive (m, l, a), FlashAttention-style tiled attention, mask policies, eligibility, fallback |
| `randomized_linalg` | `randomized_linalg.py` | Randomized range finding, randomized SVD, Gaussian/SRHT test matrices, power iterations, rank-deficiency detection, error probes, sketched least squares validated on the original system |
| `structured` | `structured.py` | Butterfly factorization (O(n log n) apply), Toeplitz fast apply via FFT convolution, structure detection, dense-reference verification, amortization measurement |
| `precision` | `precision.py` | FP16/BF16/FP32/FP64 selection policy, CPU bfloat16 emulation (RNE), product/accumulation precision separation, range checks, iterative refinement, FP64 verification, unsafe-precision rejection |
| `optimization` | `optimization.py` | Classical-first assignment baselines (Hungarian, greedy, optional CP-SAT), QUBO/Ising encoding with penalty validation, simulated annealing, tabu search, exact enumeration, solver comparison with optimality gaps |
| `quantum_research` | `quantum_research.py` | Grover/Shor research records with explicit assumptions, surface-code overhead model, magic-state distillation record, research-only gate (never routes production tasks) |
| `backends` | `backends.py` | CPU reference and optimized backends, optional accelerator declarations, request eligibility |
| `validation` | `validation.py` | Output comparison with combined tolerance, adversarial fixtures (extreme logits, ill-conditioning, rank deficiency, overflow scale), determinism, downstream decision stability, regression records |
| `workspace` | `workspace.py` | Blackboard publication of versioned metadata-only envelopes via `ore_workspace.FindingPublisher` |
| `logging` | `logging_adapter.py` | `ore_logger` SQLite integration: annotation sync, per-step events, run lifecycle (repo pattern from `ore_confirmation_bias.pipeline`) |
| — | `cros.py` | CROS admission gate: `LocalCROSGate` (in-process budgets) and `ExternalCROSAdapter` (delegates to the caller's coordinator) |
| — | `registry.py` | The 48 integrated steps with phases, purposes, mathematical forms, and controls (safeguards) |
| — | `biology.py` | Distributed rate-coded biological-reference network with honest evidence levels (quantum layers explicitly not biologically grounded) |
| — | `pipeline.py` | `HardwareAwareOptimizer`: the complete 48-step orchestration with feature flags, rollback, cancellation, deferral, and fallback |
| — | `__main__.py` | Demo CLI (`python -m ore_hacol`) |

## The 48-step integrated algorithm

Phase A — Discover and characterize workloads (steps 1–8): request
validation with operation identity; mathematical-primitive classification;
call-path tracing; trusted reference execution; seeded fixtures; end-to-end
cost measurement separating kernel from orchestration; measured-roofline
bottleneck classification; value estimation (frequency × savings − setup −
maintenance).

Phase B — Numerical contract and admission (steps 9–16): correctness contract
required (candidates without contracts are rejected); consequence levels with
tightened tolerances; approximation permission; sensitivity profile; hardware
discovery; CROS admission; candidate cost estimation; admit-or-defer.

Phase C — IO-aware attention (steps 17–23): eligibility; tile selection under
memory budgets; tiled QKᵀ blocks without materializing the full score matrix;
online-softmax (m, l, a) updates with rescaling; value accumulation and
normalization; validation against the reference; trusted fallback.

Phase D — Randomized numerical linear algebra (steps 24–32): low-rank
opportunity; rank/oversampling selection with recorded seed; documented test
matrices; range sketch with justified power iterations; orthonormalization
with rank checks; compressed B = Qᵀ A; measured approximation error; sketched
least squares with original-system residual; preserved approximation metadata.

Phase E — Structured transforms and mixed precision (steps 33–42): structure
detection with evidence; structure-preserving representation; factor
construction with amortization measurement; dense-reference verification;
per-operation dtype policy; product/accumulation precision separation; range
protection; iterative refinement; FP64 verification; unsafe-precision
rejection.

Phase F — Classical optimization and quantum research gate (steps 43–48):
classical formulation first; QUBO/Ising candidate with penalty validation;
classical solver comparison with optimality gaps; conditional quantum
annealing evaluation; Shor/Grover as research records only; QEC overhead
model (surface code, magic-state distillation).

Phase G — Verification, integration and lifecycle (unnumbered): optimized
vs. reference comparison on fixtures and adversarial inputs; end-to-end
benchmarking including warmup/conversion with median/p95/p99; versioned
envelope publication to the Blackboard; ore_logger SQLite logging;
feature-flag promotion, monitoring, and rollback; cancellation; HACOL never
bypasses CROS.

## Integration responsibilities (spec section 7)

- **CROS / AdaptiveResourceCoordinator** — `ore_hacol.cros.LocalCROSGate`
  (default) or `ExternalCROSAdapter` wrapping the repo's coordinator object;
  every `run()` passes admission through the gate and releases granted slots.
- **Shared Cognitive Workspace / Blackboard** — `workspace.publish_result`
  submits metadata-only envelopes through `FindingPublisher`; the Main Brain
  Coordinator and CognitiveTaskCoordinator decide reasoning relevance.
- **ore_logger / SQLite** — `register_hacol_annotations` syncs the package
  catalogue; every result records per-step events with module/class/function,
  task, dimensions, algorithm/backend, precision/device, duration, memory,
  transfers, error, status, and provenance. Raw sensitive payloads are never
  logged.
- **Cancellation** — `CancellationToken` checked between phases; cancelled
  runs return partial traces with status `cancelled`.

## Installation

The package is part of the repository (`pyproject.toml` includes `ore_hacol*`).
Standard repository installation applies:

    python -m pip install -e ".[dev]"
    python -m pytest tests/test_hacol.py
    python -m ore_hacol --demo all

## Acceptance tests

`tests/test_hacol.py` covers the acceptance-test list of spec section 10 for
the environment-supported items: request rejection rules, reproducible
references, compute/bandwidth-bound detection, hardware identity, CROS
admission and deferral, attention numerics (online softmax, tiling, extreme
logits, all-masked rows, causal/padding masks, fallback), randomized SVD
reproducibility and budgets, rank-deficiency detection, sketched LS
original-system residuals, structured verification and amortization, dtype
policies, overflow/underflow detection, iterative refinement, injected-error
detection, QUBO feasibility/penalty/gap checks, quantum research gating,
pipeline envelopes, rollback, cancellation, Blackboard publication, and SQLite
logging.

## Scientific status

A biologically motivated computational synthesis (see `biology.py` for the
per-phase references and evidence levels). It is not a cellular simulation, a
fitted model of a particular species, or evidence that ORE reproduces the
complete human visual or cognitive system. Quantum algorithms are research
awareness only and are never executed.
