"""Source-ordered operation definitions for the 50-step Point 13 TEDIL algorithm."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

__all__ = ("OPERATION_REGISTRY", "OperationSpec", "PHASE_OF_STEP", "STEP_PHASES",
           "registry_as_dicts")


@dataclass(frozen=True)
class OperationSpec:
    """Describe one source step, its equation, and its cautious biological reference."""

    operation_id: str
    source_step: int
    phase: str
    name: str
    purpose: str
    mathematical_form: str
    biological_reference: str
    evidence_level: str = (
        "functional_computational_model_with_distributed_biological_correspondence")

    def to_dict(self) -> dict[str, Any]:
        """Return the operation definition as a plain dictionary."""
        return asdict(self)


_ROWS: tuple[tuple[str, str, str, str, str], ...] = (
    # Phase A -- Admission and representation selection (Steps 1-7)
    ("receive_temporal_request",
     "Receive a temporal-analysis request with task/source IDs, objective, schema, "
     "timestamp semantics, budgets, and output contract; reject malformed requests.",
     "valid(request) -> accepted | rejected",
     "Coordinated task admission parallels prefrontal gating of task relevance."),
    ("classify_temporal_problem",
     "Classify the problem as static, regular, irregular, continuous-dynamic, or sparse "
     "events, recording confidence and evidence.",
     "argmax_c P(c | metadata, timestamps)",
     "Temporal processing in cortex spans multiple timescales and representation regimes."),
    ("establish_deterministic_baseline",
     "Run a deterministic baseline (moving average, smoothing, FSM, threshold, Kalman) "
     "as the control condition for every benefit claim.",
     "baseline: s_t = alpha*x_t + (1-alpha)*s_{t-1}",
     "Baselines carry no biological claim; they are engineering control conditions."),
    ("build_selection_profile",
     "Record sampling regularity, event density, sequence length, state dimension, "
     "nonlinearity, horizon, accuracy, missingness, and resource ceiling.",
     "profile = normalize(workload_statistics; train_stats)",
     "Task profiling parallels demand estimation in frontoparietal control systems."),
    ("select_candidate_family",
     "Select the least expensive candidate family that meets the requirements, comparing "
     "multiple candidates only when a benchmark justifies it.",
     "family = argmin cost(f) s.t. accuracy(f) >= target",
     "Different neural systems trade accuracy against latency and energy."),
    ("perform_resource_admission",
     "Query resource coordination, estimate cost before loading weights, and defer, "
     "downsample, or fall back when necessary.",
     "admit(family) iff C_est <= C_budget",
     "Metabolic and attentional budgets constrain biological computation."),
    ("create_execution_record",
     "Record model version, configuration hash, data window, normalization version, seed, "
     "solver settings, backend, and software versions.",
     "record = hash(model, config, window, seed)",
     "Auditable execution records are an engineering governance construct."),
    # Phase B -- Ingestion, time alignment, and quality (Steps 8-16)
    ("receive_timestamped_observations",
     "Receive observations carrying source ID, event ID, timestamp, values, units, quality "
     "flags, and provenance.",
     "obs = (source, event, t, x, u, flags, provenance)",
     "Sensory systems carry modality-specific timing and reliability signals."),
    ("validate_timestamp_semantics",
     "Distinguish wall-clock, monotonic, device, and simulation time; convert only with "
     "valid metadata and preserve the original clock basis.",
     "basis(obs) must equal basis(request)",
     "Biological timing relies on reliable, clock-like oscillatory processes."),
    ("detect_ordering_errors_and_duplicates",
     "Deduplicate by stable identity; buffer late observations within a bounded reorder "
     "window; never rewrite historical ordering.",
     "dup iff (source, event) seen; buffer when t_obs < t_latest",
     "Temporal order is fundamental to causal interpretation of sensory streams."),
    ("compute_elapsed_time",
     "Compute actual elapsed times, reject negative intervals, treat zero intervals as "
     "simultaneous, and apply the maximum-gap policy.",
     "Delta_t_k = t_k - t_{k-1} >= 0",
     "Effective processing depends on actual intervals, not assumed uniform steps."),
    ("validate_units_and_bounds",
     "Check dimensional compatibility, ranges, missing values, infinities, and sensor "
     "flags; conversions must be explicit and reversible.",
     "ok(x, unit) iff lower <= x <= upper and unit in dimension",
     "Peripheral signals are normalized and range-checked before cortical use."),
    ("handle_missing_observations",
     "Apply a documented missing-data policy and record the imputation mask; imputed values "
     "stay distinguishable from measurements.",
     "mask[i] = 1 iff x_i imputed",
     "Missing evidence reduces confidence; brains interpolate but remain uncertain."),
    ("estimate_sampling_irregularity",
     "Compute delta-t mean/variance, coefficient of variation, gap quantiles, and "
     "missingness rate with explicit zero-mean handling.",
     "CV = sigma_dt / mu_dt",
     "Irregular sampling is the norm in biological sensing, not the exception."),
    ("normalize_safely",
     "Apply a versioned transform using training statistics only, guard near-zero sigma, "
     "and preserve the inverse transform.",
     "z = (x - mu_train) / max(sigma_train, eps)",
     "Adaptive normalization parallels gain control in sensory pathways."),
    ("form_temporal_window",
     "Build the model input as an ordered window, control segment, state update, or sparse "
     "event packet, preserving elapsed times and source boundaries.",
     "W = [(t_k, x_k, Delta_t_k)] ordered",
     "Working memory maintains bounded, time-structured context."),
    # Phase C -- Model execution (Steps 17-25)
    ("initialize_or_restore_state",
     "Restore state only when model version, schema, source identity, and state timestamp "
     "are compatible; otherwise apply the cold-start policy.",
     "restore iff compatible(model, schema, source, t_state)",
     "Memory reactivation depends on matching context and content."),
    ("execute_lnn_candidate",
     "Integrate continuous latent dynamics between observations with a bounded solver.",
     "dx/dt = f_theta(x, u, t); x(t+dt) via solver",
     "Liquid networks echo continuously evolving recurrent circuit dynamics."),
    ("execute_cfc_candidate",
     "Pass actual elapsed time into the closed-form update and record compute time and "
     "numerical deviation versus a numerical reference.",
     "x(t+dt) = x_inf + (x - x_inf) exp(-dt/tau)",
     "Membrane integration toward a driving target follows exponential time constants."),
    ("execute_neural_ode_candidate",
     "Define the vector field, interval, solver, tolerances, and evaluation cap; monitor "
     "failures, excessive evaluations, and exploding states.",
     "dh/dt = f_theta(h, t); solve h(t1)",
     "Learned differential systems approximate smooth cortical population dynamics."),
    ("execute_selective_ssm_candidate",
     "Process ordered observations through recurrent state updates with input-dependent "
     "parameters and bounded memory.",
     "h_t = A(x_t) h_{t-1} + B(x_t) x_t",
     "Gated recurrent microcircuits carry context across long sequences."),
    ("execute_snn_candidate",
     "Encode input with documented rate or TTFS encoding, update membranes, apply "
     "threshold/reset rules, and record encoding loss and event density.",
     "tau_m dV/dt = -(V - V_rest) + R I(t)",
     "Spiking neurons compute with thresholds, resets, and refractory periods."),
    ("isolate_model_state",
     "Separate state by source, task, model version, and security boundary; define "
     "expiration, checkpoint, and reset semantics.",
     "state keyed by (task, source, family, version)",
     "Memory systems separate streams by context and source."),
    ("enforce_compute_bounds",
     "Limit sequence length, event count, integration interval, evaluations, state size, "
     "duration, and retries; return partial results with clear status on exhaustion.",
     "C_used <= C_budget (hard caps)",
     "Biological computation is bounded by time, memory, and energy."),
    ("run_numerical_health_checks",
     "Check finite values, state norms, derivative magnitudes, convergence, spike rates, "
     "and output bounds; failures invalidate output rather than clipping it.",
     "healthy iff finite and within invariants",
     "Failure of a neural population invalidates its readout, not a clipped version."),
    # Phase D -- Temporal interpretation and event detection (Steps 26-32)
    ("convert_output_to_typed_estimates",
     "Represent values, latent summaries, event probabilities, horizons, and intervals as "
     "typed objects; hidden vectors are not semantic conclusions.",
     "Estimate(channel, value, unit, horizon, CI, status)",
     "Perception reports contents, not internal activity patterns."),
    ("detect_significant_changes",
     "Combine domain thresholds with statistical change detection such as rolling-residual "
     "or CUSUM scores with calibrated false positives.",
     "CUSUM: S_t = max(0, S_{t-1} + r_t - k); change iff S_t > h",
     "Change detection is central to prediction-error signalling."),
    ("detect_anomalies_against_baseline",
     "Compute residual magnitude, robust z-scores, or likelihood scores; label cause "
     "hypotheses only when evidence permits.",
     "z = 0.6745 (r - median) / MAD",
     "Prediction-error systems flag unexpected sensory evidence."),
    ("estimate_forecast_uncertainty",
     "Use predictive distributions, ensembles, quantile regression, or conformal "
     "calibration; include calibration version and mark OOD inputs.",
     "interval = y_hat +/- q_alpha; coverage on held-out data",
     "Confidence signals accompany perceptual decisions in biological circuits."),
    ("identify_temporal_relationships",
     "Extract ordering, duration, recurrence, co-occurrence, and lagged correlation; "
     "correlation and precedence are not causation.",
     "corr(x_{t-lag}, y_t); explicit non-causal note",
     "Statistical co-occurrence is weaker evidence than mechanism."),
    ("build_compact_event_summaries",
     "Summarize event type, time range, magnitude, confidence, source, and evidence "
     "references while preserving access to the original window.",
     "summary = fold(events, window_ref)",
     "Episodic memory stores structured summaries with retrieval cues."),
    ("decide_event_actionability",
     "Apply deterministic thresholds, severity, uncertainty, and rate limits; neural "
     "scores never independently authorize high-impact actions.",
     "actionable iff deterministic_rules(event) and rate_ok",
     "Action control requires evidence beyond statistical pattern detection."),
    # Phase E -- Cognitive integration (Steps 33-39)
    ("map_outputs_to_usg_concepts",
     "Translate validated event labels into USG identifiers through explicit mappings; "
     "publish unknown/uncategorized when no mapping exists.",
     "usg(label) explicit or usg:temporal:unknown@1",
     "Conceptual classification depends on learned, explicit category systems."),
    ("attach_evidence_and_provenance",
     "Attach source IDs, time range, preprocessing version, model identity, uncertainty, "
     "baseline comparison, and validation status; distinguish epistemic kinds.",
     "Provenance(source, versions, model, uncertainty, status)",
     "Source memory distinguishes how knowledge was acquired."),
    ("publish_to_shared_workspace",
     "Publish versioned records with task, producer, validity interval, confidence, "
     "evidence, and status without overwriting earlier observations.",
     "publish(field, key, payload) append-only",
     "Shared workspaces parallel communication through a global blackboard."),
    ("notify_main_brain_coordinator",
     "Send a concise bounded event/result message; the coordinator chooses downstream "
     "consumers; raw streams are never broadcast.",
     "notify(small summary)",
     "Broadcast suppression parallels attentional gating of awareness."),
    ("permit_downstream_task_decomposition",
     "Allow explanation, comparison, resource-prediction, or analysis tasks with bounded "
     "evidence and one explicit question each.",
     "tasks = decompose(result, bounds)",
     "Cognitive control decomposes goals into subtasks."),
    ("coordinate_execution_and_backpressure",
     "Honour scheduler priorities, concurrency, and cancellation; apply documented "
     "backpressure, aggregation, or sampling when ingestion exceeds capacity.",
     "apply(policy) on overload",
     "Processing bottlenecks trigger adaptive selection of inputs."),
    ("preserve_symbolic_boundaries",
     "Keep USG and deterministic modules responsible for semantic rules; temporal models "
     "provide evidence and predictions, never rule rewrites.",
     "semantic_rules owned by USG modules",
     "Separation of evidence from belief revision is a design safeguard."),
    # Phase F -- Portability and hardware abstraction (Steps 40-45)
    ("define_hardware_independent_contract",
     "Specify inputs, outputs, state shape, timestamp/event semantics, precision, reset "
     "behaviour, and operations independently of the backend.",
     "contract(graph) independent of device",
     "The brain's computations are not tied to one substrate description."),
    ("build_portable_intermediate_graph",
     "Represent supported operations, state nodes, event channels, parameters, and "
     "metadata in a versioned graph; reject unsupported operators explicitly.",
     "graph = nodes(ops, params) versioned",
     "NIR's architectural lesson: separate model semantics from device scheduling."),
    ("create_cpu_reference_backend",
     "Implement deterministic simulator behaviour for correctness tests; accelerated "
     "backends come only after reference behaviour is stable.",
     "reference: numpy kernels, deterministic",
     "Simulation remains the default development path."),
    ("add_optional_lava_adapter",
     "Isolate Lava as an optional framework adapter; dependencies stay optional and are "
     "never imported during core startup.",
     "import lava lazily; optional",
     "Hardware-specific adaptations are isolated from core function."),
    ("negotiate_hardware_capabilities",
     "Query supported operations, precision, memory, event routing, runtime, and "
     "availability; mismatch triggers safe fallback.",
     "select(backend) iff caps >= required",
     "Behaviour adapts to the available substrate without changing intent."),
    ("benchmark_before_adoption",
     "Compare identical workloads across backends, measuring accuracy, latency, "
     "throughput, energy where measurable, overhead, and maintenance.",
     "metrics(backend) on identical windows",
     "Neural systems are shaped by measurable energetic trade-offs."),
    # Phase G -- Governance and lifecycle (Steps 46-50)
    ("separate_training_from_inference",
     "Train offline with versioned datasets; production uses an approved immutable "
     "artifact unless controlled adaptation is explicitly enabled.",
     "prod uses frozen artifact",
     "Memory consolidation separates fast learning from stable retrieval."),
    ("validate_temporal_generalization",
     "Use chronological splits, rolling-origin evaluation, and source-separated tests; "
     "never leak adjacent windows between train and test.",
     "t_train < t_test (chronological)",
     "Learning and evaluation are separated in time in natural settings."),
    ("monitor_distribution_shift",
     "Track input statistics, missingness, event rates, residuals, and calibration drift; "
     "trigger review at pre-registered limits without automatic retraining.",
     "PSI(current || reference) <= threshold",
     "Drift in the environment requires re-evaluation of learned models."),
    ("apply_controlled_adaptation",
     "If approved, use bounded learning rates, replay windows, rollback checkpoints, and "
     "independent evaluation; keep online adaptation disabled by default.",
     "lr <= lr_bound; rollback available",
     "Plasticity is bounded and stabilized in biological learning."),
    ("maintain_registry_privacy_logging_closure",
     "Store artifact hashes, lineage, metrics, licenses, limitations, and approval state; "
     "minimize raw retention; log structured events; expire state; return the explainable "
     "result envelope.",
     "registry + ore_logger/SQLite + envelope",
     "Governance constructs are engineering safeguards, not biological claims."),
)


def _phase_for(step: int) -> str:
    """Return the phase label that owns a step number."""
    if step <= 7:
        return "Phase A -- Admission and representation selection"
    if step <= 16:
        return "Phase B -- Ingestion, time alignment, and quality"
    if step <= 25:
        return "Phase C -- Model execution"
    if step <= 32:
        return "Phase D -- Temporal interpretation and event detection"
    if step <= 39:
        return "Phase E -- Cognitive integration"
    if step <= 45:
        return "Phase F -- Portability and hardware abstraction"
    return "Phase G -- Governance and lifecycle"


OPERATION_REGISTRY = tuple(
    OperationSpec(
        operation_id=f"P13-{index:02d}",
        source_step=index,
        phase=_phase_for(index),
        name=name,
        purpose=purpose,
        mathematical_form=formula,
        biological_reference=biology,
    )
    for index, (name, purpose, formula, biology) in enumerate(_ROWS, start=1)
)


STEP_PHASES: dict[str, tuple[int, int]] = {
    "A": (1, 7),
    "B": (8, 16),
    "C": (17, 25),
    "D": (26, 32),
    "E": (33, 39),
    "F": (40, 45),
    "G": (46, 50),
}

PHASE_OF_STEP: dict[int, str] = {
    step: phase
    for phase, (start, end) in STEP_PHASES.items()
    for step in range(start, end + 1)
}


def registry_as_dicts() -> list[dict[str, Any]]:
    """Return the complete 50-operation registry in source order."""
    return [item.to_dict() for item in OPERATION_REGISTRY]
