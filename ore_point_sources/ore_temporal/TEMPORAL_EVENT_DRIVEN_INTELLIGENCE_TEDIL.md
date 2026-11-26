# ORE Temporal and Event-Driven Intelligence Layer (TEDIL) — Point 13 Architecture

Canonical relevance rank: 23 of 24

Source: "ORE Integrated Algorithm — Point 13: Continuous-Time Neural Networks and
Neuromorphic Computing" (from Most Intelligent Systems 01-24.txt, sections 13.1-13.9).

Status: Implemented as the `ore_temporal` package, software-first, CPU-reference only.
This document maps the specification to the implementation, tests, and integration
seams in the live repository.

## 1. Objective and design principle

`ore_temporal` implements the optional Temporal and Event-Driven Intelligence Layer for
observations whose meaning depends on time, irregular sampling, continuous dynamics, or
sparse asynchronous events.  It complements USG, symbolic reasoning, the shared
cognitive workspace, and ordinary discrete algorithms.

Implemented candidate families (all deterministic, seeded, pure numpy):

| Family | Module | Core equation |
|---|---|---|
| Liquid Neural Network (LNN) | `ore_temporal/models/lnn.py` | dx/dt = (tanh(W_x x + W_u u + b) - x) / tau |
| Closed-form Continuous-time (CfC) | `ore_temporal/models/cfc.py` | x(t+dt) = x_inf + (x - x_inf) exp(-dt/tau) |
| Neural ODE | `ore_temporal/models/neural_ode.py` | dh/dt = W2 tanh(W1 [h;u] + b1) + b2 |
| Selective SSM | `ore_temporal/models/ssm.py` | h_t = A(x_t) h_{t-1} + B(x_t) x_t |
| Spiking NN (LIF) | `ore_temporal/models/snn.py` | tau_m dV/dt = -(V - V_rest) + R I(t) |
| Deterministic baselines | `ore_temporal/models/baseline.py` | moving average / exponential smoothing / Kalman / threshold |

Core principle, implemented literally: classify the temporal problem first, then select
the least expensive representation that meets accuracy, latency, interpretability, and
reliability requirements. The deterministic baseline always runs as the control
condition for every model-benefit claim.

## 2. Package architecture (reconciled with the live repository)

The specification proposed twelve package boundaries "to reconcile with the live
repository before coding". The reconciliation:

```
ore_temporal/
├── contracts.py       # typed request/observation/result contracts (spec §6)
├── config.py          # versioned, validated configuration; every governed parameter
├── functional.py      # pure math: dt, Euler, RK4, relaxation, SSM, LIF, CV, CUSUM, PSI inputs
├── biology.py         # cautious biological references + LIF reference population
├── pipeline.py        # TemporalEngine: 50-step orchestration, phases A-G
├── annotations.py     # ore_logger/SQLite structured logging (spec's logging/ role)
├── registration.py    # central ModuleRegistry registration (ore_usg pattern)
├── __main__.py        # synthetic demo: python -m ore_temporal
├── ingestion/         # Steps 8-16: timestamps, units, ordering, missing, normalize, windows
├── representations/   # temporal windows, event records, evidence, NIR-inspired graphs
├── models/            # Steps 17-23: five families + baselines + state isolation
├── solvers/           # bounded Euler/RK4 integration with evaluation budgets
├── events/            # Steps 26-32: change/anomaly detection, relationships, actionability
├── selection/         # Steps 4-6: profiling, baseline-first policy, resource admission
├── monitoring/        # metrics, conformal calibration, drift, numerical health
├── backends/          # Steps 40-45: CPU reference, portable graphs, lazy Lava, benchmarks
├── workspace/         # Steps 33-39: Blackboard publication, USG mapping, notifications
└── registry/          # 50-step operation registry + model artifact lifecycle
```

Reconciliation notes:

- The spec's `ore_temporal/logging/` role is filled by `annotations.py`, matching the
  repository-wide convention (`register_annotations`, `record_result_to_sqlite`).
- The spec's `ore_temporal/registry/` role (model metadata, artifacts, lifecycle) is a
  subpackage that also carries the repository-conventional 50-step `OPERATION_REGISTRY`
  (`P13-01` … `P13-50`), so both `from ore_temporal.registry import OPERATION_REGISTRY`
  and the model-registry API live in one import path.
- Root-level `contracts.py`, `functional.py`, `biology.py`, `pipeline.py`,
  `annotations.py` follow the flat-module conventions shared by every point module.

## 3. The 50-step algorithm

`TemporalEngine.run(request, observations, ...)` executes phases A-G in order and emits
an operation trace with exactly one entry per source step (verified by test):

| Phase | Steps | Engine method |
|---|---|---|
| A — Admission and representation selection | 1-7 | `_phase_a` |
| B — Ingestion, time alignment, and quality | 8-16 | `_phase_b` |
| C — Model execution | 17-25 | `_phase_c` |
| D — Temporal interpretation and event detection | 26-32 | `_phase_d` |
| E — Cognitive integration | 33-39 | `_phase_e` |
| F — Portability and hardware abstraction | 40-45 | `_phase_f` |
| G — Governance and lifecycle | 46-50 | `_phase_g` |

The conceptual interface from spec §6 is preserved as public methods:
`classify(request)`, `select_model(profile, budget)`, `detect_events(...)`,
`publish(result, workspace)`, `checkpoint_or_expire(task_id, source_id, model)`.

## 4. Mathematical foundation

Every governed quantity of spec §8 is a pure function in `functional.py` with a test:

| Operation | Function | Guard |
|---|---|---|
| Time delta | `compute_delta_times` | rejects negative; flags zero |
| Euler integration | `euler_step`, `integrate_interval` | step/stability/eval-budget checks |
| Exponential relaxation | `exponential_relaxation` | tau > 0 |
| State-space update | `ssm_step`, `SelectiveStateSpaceModel` | dimension/version checks |
| LIF | `lif_advance`, `lif_derivative` | threshold/reset/refractory explicit |
| Sampling CV | `coefficient_of_variation` | zero-mean handled explicitly |
| Event density/sparsity | `event_density`, `event_sparsity` | consistent interval |
| Ensemble spread | `ensemble_mean_variance` | calibrated downstream |
| Residual | `residual` | aligned time/units |
| CUSUM | `cusum_update` | bounded statistic, drift, threshold |
| Resource budget | `ResourceAdmission.admit` | hard caps, baseline fallback |

Solver behaviour (spec §3.2): explicit Euler and RK4 with bounded steps, exposed
tolerances, exact evaluation-budget enforcement, and divergence/non-finite detection
that raises `SolverFailure` — the engine catches it and falls back to the baseline with
`ValidationStatus.FALLBACK` (never a silently clipped output).

## 5. Model-selection policy (spec §5)

`ModelSelectionPolicy.select` implements the specification table: irregular samples
favour CfC/LNN (with exponential smoothing / Kalman as the alternative), continuous
learned dynamics favour Neural ODE (solver cost caution), long sequences favour the
selective SSM (benchmark on real lengths), sparse events favour SNN (encoding-loss
caution), high-assurance monitoring prefers the deterministic detector, and hardware
research stays on the CPU simulator. Selection is explicit, testable, and logged in the
operation trace and result diagnostics.

## 6. ORE integration responsibilities

| Component | How TEDIL integrates |
|---|---|
| USG | `USGConceptMapper` with an explicit label→id table; unknown labels map to `usg:temporal:unknown@1`, never an invented class |
| Shared Cognitive Workspace | `WorkspaceBridge` publishes versioned findings to the append-only `artifact_references` field via `FindingPublisher` with `Provenance("module_inference", ...)`; earlier evidence is never overwritten; field writes go through the documented permission-grant handshake |
| Main Brain Coordinator | `build_coordinator_notification` — bounded summaries only, never raw streams |
| CognitiveTaskCoordinator | `build_downstream_task_hints` — explanation / cross-source / resource-prediction tasks, each with bounded evidence and one explicit question; the module also registers as a backend through `registration.py` |
| ModuleScheduler | cooperative `CancellationSignal` checked between phases; safe partial results |
| AdaptiveResourceCoordinator/CROS | `ResourceAdmission` with a `ResourceCoordinatorPort`; the local estimator is the default when CROS is not wired in; admission defers/downsamples/falls back before weights load |
| ore_logger/SQLite | `register_annotations`, `record_result_to_sqlite`, and the engine-time `TemporalAnnotationRecorder` (module/class/function, task/model identity, config hash, durations, solver statistics, event counts, status, exception category, provenance; secrets and raw payloads stripped) |

Central registration follows the `ore_usg/registration.py` pattern:

```
from ore_contracts import ContractCatalog, ModuleRegistry
from ore_temporal import register_ore_temporal_module

registry = ModuleRegistry(ContractCatalog())
backend = register_ore_temporal_module(registry)
```

This registers the `ORE.TEMPORAL.REQUEST/OBSERVATIONS/RESULT@1.0.0` contracts, the
validated manifest (`ore.temporal_intelligence` v1.0.0, capabilities
`temporal.irregular_sampling.forecast`, `temporal.event_detection`,
`temporal.anomaly_detection`, `temporal.continuous_state`,
`temporal.baseline_comparison`), and binds a `describe()/execute()` backend that
reports `capability_support` for the AdaptiveResourceCoordinator.

## 7. Portability and hardware abstraction (spec §3.6, Steps 40-45)

- `PortableGraph` (NIR-inspired): versioned nodes over the supported-operation contract
  (`linear`, `tanh`, `sigmoid`, `exp_relax`, `ssm_cell`, `lif_cell`, `euler_step`,
  `rk4_step`, `threshold`, `gate`), state nodes, event channels, parameter tensors,
  metadata; unsupported operators raise `UnsupportedGraphOperationError`.
- `CPUReferenceBackend`: deterministic numpy kernels; the engine validates that the
  exported graph reproduces the model step (observed deviations <= 1e-9 for
  CfC/LNN/SSM).
- `LavaBackendAdapter`: lazy import, `OptionalDependencyError` when absent, never
  imported during core startup (test-verified).
- `negotiate_capabilities`: capability mismatch triggers safe CPU fallback.
- `benchmark_identical_workload`: identical windows across backends; hardware adoption
  is a separate decision.

## 8. Governance and lifecycle (spec §12 safeguards)

- Training is separated from production inference: production uses frozen seeded
  artifacts; `ModelArtifactRecord` carries hash, lineage, training config, metrics,
  license, limitations, and approval state; `TemporalModelRegistry` enforces the
  PROPOSED → VALIDATED → APPROVED → DEPRECATED lifecycle.
- `chronological_split` and held-out `CalibrationData` enforce temporal generalization;
  normalization statistics are fitted on training data only (leakage test-verified).
- `DriftMonitor` scores PSI against the reference statistics and triggers review at the
  pre-registered threshold; no automatic retraining.
- Controlled adaptation is disabled by default; the learning-rate bound lives in
  versioned configuration.
- `evaluate_promotion` implements the §11 gates: a candidate is promoted only when it
  beats or materially complements the baseline AND meets latency/memory budgets AND
  has acceptable calibration AND passes safety tests — lower error alone is
  insufficient.
- `ActionabilityGate`: neural evidence never independently authorizes high-impact
  actions; deterministic corroboration, severity, uncertainty, and rate limits decide.

## 9. Acceptance tests

The specification's §10 acceptance list is covered by 91 tests in four files:

- `tests/test_temporal_ingestion_point13.py` — Steps 1-16: request rejection,
  malformed timestamps, negative elapsed time, duplicates, timestamp/clock-basis
  preservation, unit conversion accept/reject, out-of-range flagging, imputation masks
  and every missing-data policy, irregularity, gap detection, normalization and
  leakage, reorder buffer, beyond-window flagging, backpressure.
- `tests/test_temporal_models_point13.py` — Steps 17-25, 40-45: deterministic
  initialization under fixed seed, checkpoint restore and version-mismatch rejection,
  cross-source state isolation, explicit reset, state-size limits, stable-system
  solver accuracy, evaluation-budget enforcement, divergence detection, actual-dt CfC
  verification and numerical-reference comparison, chronological SSM with bounded
  memory, LIF threshold/reset/refractory, encoding-ordering preservation, spike
  budgets, unsupported graph operators, CPU determinism, graph/model equivalence,
  optional Lava, capability fallback, identical-window benchmarks.
- `tests/test_temporal_events_monitoring_point13.py` — Steps 26-32, 46-50: labelled
  anomaly detection with FP/FN measurement, chronological conformal calibration and
  coverage, OOD marking, predicted-not-measured labels, imputed-value
  distinguishability, non-causal correlation serialization, health checks that
  invalidate, drift review triggers, actionability (including the no-neural-actuation
  rule and rate limits), promotion gates, lifecycle transitions, baseline-only mode,
  feature-flag rollback, adaptation-disabled default, metrics.
- `tests/test_temporal_integration_point13.py` — Steps 33-39, 50 plus the full
  pipeline: USG known/unknown mapping, Blackboard publication with provenance,
  versioned non-overwriting publication, bounded coordinator summaries, CROS budget
  fallback, solver-failure fallback, safe cancellation, reproducibility from config and
  fixture, central ModuleRegistry registration with capability support, SQLite
  annotation indexing and event logging, secret/raw-payload exclusion, state expiration
  under retention, all-fifty-steps trace, end-to-end level-shift detection.

Run with: `python -m pytest tests/test_temporal_*_point13.py`.

## 10. Failure modes and safeguards (spec §12)

| Failure | Safeguard implemented |
|---|---|
| Clock mismatch | `TimestampValidator` preserves clock basis; conversion only with metadata |
| Irregularity ignored | actual dt everywhere; CV/jitter measured and recorded |
| Solver instability | bounded solver, diagnostics, baseline fallback |
| State contamination | per-(task, source, family, version) isolation keys |
| Hidden imputation | masks, IMPUTED flags, epistemic status |
| Lossy spike encoding | encoding-loss recorded; ordering preservation test |
| Event overload | event/spike caps, backpressure policies |
| Model drift | PSI monitor with pre-registered review gate |
| Uncalibrated uncertainty | conformal calibration with version and coverage lineage |
| Hardware lock-in | portable contract + CPU reference; Lava optional |
| Resource overrun | admission with hard caps and baseline fallback |
| Prediction treated as fact | epistemic-status labels on every estimate |
| Excessive logging | payload minimization; secrets stripped |
| Optional dependency failure | lazy adapters; core startup never imports Lava |

## 11. Roadmap status (spec §9)

| Stage | Status |
|---|---|
| 0 Repository/dependency audit | complete (this document records the reconciliation) |
| 1 Contracts and baseline | complete |
| 2 Ingestion/event pipeline | complete (synthetic-stream tests) |
| 3 SSM baseline | complete (selective SSM + bounded-memory tests) |
| 4 CfC/LNN experiment | complete (irregular-time validation, numerical-reference deviation) |
| 5 Neural ODE experiment | complete (solver abstraction, budgets, tolerance sensitivity API) |
| 6 SNN simulator | complete (LIF, rate/TTFS encoding, event budgets) |
| 7 Portability | complete (portable graph, CPU reference, lazy Lava) |
| 8 Controlled integration | complete (registry registration, workspace publication, budgets, SQLite) |
| 9 Benchmark and decision | complete (identical-window benchmark, promotion gates) |

Neuromorphic hardware (Loihi 2, Hala Point, SpiNNaker) remains explicitly out of scope:
simulation is the default development path, and each model family must earn inclusion
through measured improvement over the deterministic baseline.

## 12. Honest limitations

- The seeded reference models use engineering default parameters; they are not fitted
  to empirical data, and the demo's CfC does not beat the Kalman baseline on the
  synthetic workload — promotion gates correctly report that.
- Event density and sparsity characterize a workload; neither proves energy savings.
- Correlation and temporal precedence are never treated as causal proof.
- The biological-reference track is a labelled engineering abstraction, not a brain
  simulation.
