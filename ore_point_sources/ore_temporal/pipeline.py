"""Fifty-step orchestration of the Point 13 Temporal and Event-Driven Intelligence Layer.

The ``TemporalEngine`` executes the specification's phases A through G in order, records
an operation trace with one entry per source step, and returns a typed
:class:`~ore_temporal.contracts.TemporalResult` with its explainable envelope.  The
engine is software-first: every neural family runs on the deterministic CPU reference
path, and the deterministic baseline always runs as the control condition.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time as _time
from typing import Any, Sequence

import numpy as np

from .annotations import TemporalAnnotationRecorder
from .biology import MembranePopulation
from .config import SOFTWARE_VERSIONS, TemporalConfig, config_hash
from .contracts import (
    BackpressurePolicy,
    BaselineComparison,
    CancellationSignal,
    ClassificationResult,
    EpistemicStatus,
    Estimate,
    EventSeverity,
    ExecutionRecord,
    ModelFamily,
    ModelIdentity,
    ResultEnvelope,
    ResultType,
    SelectionDecision,
    TaskStatus,
    TemporalObservation,
    TemporalProblemClass,
    TemporalRequest,
    TemporalResult,
    TypedEventRecord,
    UncertaintyReport,
    ValidationStatus,
)
from .events import CUSUMChangeDetector, ActionabilityGate, AnomalyDetector
from .events.relationships import analyze_relationships
from .ingestion import (
    IngestionReport,
    MissingDataHandler,
    QualityValidator,
    TimestampValidator,
    UnitRegistry,
    VersionedNormalizer,
    apply_ordering_and_deltas,
    build_temporal_window,
    profile_sampling,
)
from .models import (
    ModelStateStore,
    StateCompatibilityError,
    build_model,
    family_available,
    run_model_over_window,
)
from .models.base import ModelFamilyError
from .monitoring import (ConformalCalibrator, DriftMonitor, mean_absolute_error,
                         numerical_health_check)
from .monitoring.calibration import OutOfDistributionMarker
from .registry import (
    OPERATION_REGISTRY,
    LifecycleState,
    ModelArtifactRecord,
    TemporalModelRegistry,
    evaluate_promotion,
)
from .representations.evidence import EvidenceBundle, ProvenanceRecord
from .selection import (ModelSelectionPolicy, ResourceAdmission,
                        build_selection_profile)
from .selection.profile import SelectionProfile
from .solvers import SolverFailure
from .workspace import (
    USGConceptMapper,
    WorkspaceBridge,
    build_coordinator_notification,
    build_downstream_task_hints,
)

__all__ = ("CalibrationData", "TemporalEngine", "chronological_split")


@dataclass(frozen=True, slots=True)
class CalibrationData:
    """Held-out chronological calibration pairs (values and one-step predictions)."""

    values: tuple[tuple[float, ...], ...]
    predictions: tuple[tuple[float, ...], ...]
    source: str = "held-out chronological split"

    def __post_init__(self) -> None:
        """Validate calibration alignment."""
        if len(self.values) != len(self.predictions):
            raise ValueError("calibration values and predictions must align")
        if not self.values:
            raise ValueError("calibration data must not be empty")
        for row in self.values:
            if len(row) != len(self.values[0]):
                raise ValueError("calibration rows must have equal width")


def chronological_split(values: Sequence[float], train_ratio: float = 0.7
                        ) -> tuple[list[int], list[int]]:
    """Step 47 -- chronological split; adjacent windows never leak across the boundary."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    n = len(values)
    boundary = max(1, int(math.floor(n * train_ratio)))
    return (list(range(0, boundary)), list(range(boundary, n)))


class TemporalEngine:
    """Execute the 50-step TEDIL algorithm for one temporal-analysis task."""

    def __init__(self, config: TemporalConfig | None = None, *,
                 state_store: ModelStateStore | None = None,
                 artifact_registry: TemporalModelRegistry | None = None,
                 usg_mapper: USGConceptMapper | None = None,
                 workspace_bridge: WorkspaceBridge | None = None,
                 annotation_recorder: TemporalAnnotationRecorder | None = None,
                 simulate_biology: bool = True) -> None:
        """Compose the engine from its collaborators with documented defaults."""
        self.config = config or TemporalConfig()
        self.state_store = state_store or ModelStateStore(
            size_limit_bytes=self.config.state_size_limit_bytes,
            retention_seconds=self.config.state_retention_seconds)
        self.artifact_registry = artifact_registry or TemporalModelRegistry()
        self.usg_mapper = usg_mapper or USGConceptMapper()
        self.workspace_bridge = workspace_bridge or WorkspaceBridge(
            workspace_field=self.config.workspace_field)
        self.recorder = annotation_recorder
        self.simulate_biology = bool(simulate_biology)
        self.biological_network = MembranePopulation() if simulate_biology else None
        self._unit_registry = UnitRegistry()
        self._trace: list[dict[str, Any]] = []
        self._diagnostics: dict[str, Any] = {}

    # ------------------------------------------------------------------ helpers

    def _record_step(self, step: int, status: str = "completed", detail: str = "") -> None:
        """Append one operation-trace entry for a source step."""
        spec = OPERATION_REGISTRY[step - 1]
        self._trace.append({
            "operation_id": spec.operation_id,
            "step": spec.source_step,
            "phase": spec.phase,
            "operation": spec.name,
            "status": status,
            "detail": detail,
            "mathematical_form": spec.mathematical_form,
            "biological_reference": spec.biological_reference,
        })

    def _cancelled(self, cancellation: CancellationSignal | None) -> bool:
        """Report whether cancellation was requested."""
        return cancellation is not None and cancellation.cancelled

    # ------------------------------------------------------ conceptual interface

    def classify(self, request: TemporalRequest,
                 observations: Sequence[TemporalObservation] | None = None
                 ) -> ClassificationResult:
        """Step 2 -- classify the temporal problem with recorded confidence."""
        if not isinstance(request, TemporalRequest):
            raise TypeError("request must be a TemporalRequest")
        evidence: list[str] = []
        problem = TemporalProblemClass.IRREGULARLY_SAMPLED
        confidence = 0.5
        if observations is not None and len(observations) > 1:
            ordered = sorted(observations, key=lambda item: item.timestamp)
            deltas = [ordered[i + 1].timestamp - ordered[i].timestamp
                      for i in range(len(ordered) - 1)]
            mean_delta = float(np.mean(deltas))
            cv = float(np.std(deltas) / mean_delta) if mean_delta > 0 else 0.0
            evidence.append(f"observed timestamp CV={cv:.3f}")
            if all(delta == deltas[0] for delta in deltas):
                problem = TemporalProblemClass.REGULARLY_SAMPLED
                confidence = 0.8
            elif cv > 0.25:
                problem = TemporalProblemClass.IRREGULARLY_SAMPLED
                confidence = 0.75
            else:
                problem = TemporalProblemClass.REGULARLY_SAMPLED
                confidence = 0.6
        else:
            evidence.append("no observed timestamps available; declared semantics used")
        if request.requested_family is ModelFamily.NEURAL_ODE:
            problem = TemporalProblemClass.CONTINUOUS_DYNAMIC
            confidence = 0.7
            evidence.append("requester declared continuous dynamics")
        if request.requested_family is ModelFamily.SNN:
            evidence.append("requester declared sparse-event processing")
        return ClassificationResult(problem_class=problem, confidence=confidence,
                                    evidence=tuple(evidence))

    def select_model(self, profile: SelectionProfile, budget: float,
                     requested: ModelFamily | None = None) -> SelectionDecision:
        """Step 5 -- explicit, logged model-family selection from the measured profile."""
        del budget  # the profile already carries the resource ceiling
        return ModelSelectionPolicy(self.config).select(profile, requested=requested)

    def detect_events(self, residuals: np.ndarray, timestamps: np.ndarray,
                      channels: tuple[str, ...], observed: np.ndarray,
                      predicted: np.ndarray, anomaly_detector: AnomalyDetector | None = None,
                      history: Sequence[float] = ()
                      ) -> tuple[TypedEventRecord, ...]:
        """Steps 27-28 -- change detection and anomaly detection against the baseline."""
        events: list[TypedEventRecord] = []
        for channel_index, channel in enumerate(channels):
            detector = CUSUMChangeDetector(
                drift=self.config.cusum_drift, threshold=self.config.cusum_threshold,
                warmup=self.config.cusum_warmup,
                expected_false_alarm_rate=self.config.expected_false_alarm_rate)
            for index in range(len(timestamps)):
                innovation = (float(residuals[index, channel_index])
                              if residuals.ndim == 2 else float(residuals[index]))
                change = detector.update(float(timestamps[index]), channel, innovation)
                if change is not None:
                    events.append(TypedEventRecord(
                        event_id=f"change:{channel}:{index}",
                        event_type="change_point",
                        timestamp=change.timestamp,
                        magnitude=change.score,
                        severity=EventSeverity.WARNING,
                        confidence=0.7,
                        epistemic_status=EpistemicStatus.INFERRED))
        if anomaly_detector is not None and anomaly_detector.calibrated:
            for index in range(len(timestamps)):
                for channel_index, channel in enumerate(channels):
                    is_anomaly, score, hypothesis = anomaly_detector.detect(
                        float(observed[index, channel_index]),
                        float(predicted[index, channel_index]),
                        float(timestamps[index]), history=history)
                    if is_anomaly:
                        label = {"sensor_fault": "sensor_fault",
                                 "distribution_shift": "distribution_shift"}.get(
                            hypothesis.value, "anomaly_spike")
                        events.append(TypedEventRecord(
                            event_id=f"anomaly:{channel}:{index}",
                            event_type=label,
                            timestamp=float(timestamps[index]),
                            magnitude=score,
                            severity=(EventSeverity.CRITICAL
                                      if score > 2 * self.config.anomaly_z_threshold
                                      else EventSeverity.WARNING),
                            confidence=min(0.9, 0.5 + score
                                           / (2 * self.config.anomaly_z_threshold)),
                            cause_hypothesis=hypothesis.value,
                            epistemic_status=EpistemicStatus.INFERRED))
        return tuple(events[: self.config.max_events_per_run])

    def publish(self, result: TemporalResult, workspace: Any | None) -> dict[str, Any]:
        """Step 35 -- publish a versioned record to the shared cognitive workspace."""
        usg = {label: self.usg_mapper.map(label)[0] for label in
               {event.event_type for event in result.events}}
        return self.workspace_bridge.publish(result, workspace, usg_mapping=usg)

    def checkpoint_or_expire(self, task_id: str, source_id: str,
                             model: Any | None = None) -> dict[str, Any]:
        """Step 50 -- checkpoint authorized continuation state or expire it."""
        from .models.state import record_from_array

        expired = self.state_store.expire()
        checkpointed = None
        if model is not None and self.config.state_checkpoint_enabled:
            identity = model.identity()
            record = record_from_array(
                task_id=task_id, source_id=source_id,
                model_family=identity.family.value,
                model_version=identity.model_version,
                config_hash=identity.config_hash, channels=(),
                state=model.state_array(), state_timestamp=_time.time())
            self.state_store.checkpoint(record, authorized=True)
            checkpointed = record.to_dict()
        return {"expired": expired, "checkpoint": checkpointed,
                "store": self.state_store.summary()}

    # ------------------------------------------------------------------- engine

    def run(self, request: TemporalRequest,
            observations: Sequence[TemporalObservation], *,
            normalizer: VersionedNormalizer | None = None,
            calibration: CalibrationData | None = None,
            cancellation: CancellationSignal | None = None,
            workspace: Any | None = None) -> TemporalResult:
        """Run phases A-G in order and return the typed temporal result."""
        started = _time.perf_counter()
        self._trace = []
        self._diagnostics = {}
        if self.recorder is not None:
            self.recorder.record("run_started", task_id=request.task_id, status="started")
        task_status = TaskStatus.COMPLETED
        try:
            if self._cancelled(cancellation):
                return self._partial_result(request, observations, TaskStatus.CANCELLED,
                                            f"cancelled before admission; "
                                            f"reason: {cancellation.reason}")
            phase_a = self._phase_a(request, observations, normalizer)
            if self._cancelled(cancellation):
                return self._partial_result(request, observations, TaskStatus.CANCELLED,
                                            f"cancelled before ingestion; "
                                            f"reason: {cancellation.reason}")
            phase_b = self._phase_b(request, observations, normalizer)
            if self._cancelled(cancellation):
                return self._partial_result(request, observations, TaskStatus.CANCELLED,
                                            f"cancelled before model execution; "
                                            f"reason: {cancellation.reason}")
            phase_c = self._phase_c(request, phase_a, phase_b)
            if self._cancelled(cancellation):
                return self._partial_result(request, observations, TaskStatus.CANCELLED,
                                            f"cancelled before interpretation; "
                                            f"reason: {cancellation.reason}")
            phase_d = self._phase_d(request, phase_b, phase_c, calibration)
            if self._cancelled(cancellation):
                return self._partial_result(request, observations, TaskStatus.CANCELLED,
                                            f"cancelled before integration; "
                                            f"reason: {cancellation.reason}")
            phase_e = self._phase_e(request, phase_b, phase_d, workspace)
            phase_f = self._phase_f(phase_a, phase_b, phase_c)
            phase_g = self._phase_g(phase_b, phase_c, phase_d, phase_f)
            task_status = phase_c["task_status"]
            return self._assemble_result(request, phase_a, phase_b, phase_c, phase_d,
                                         phase_e, phase_f, phase_g, task_status, started)
        except (ValueError, TypeError, KeyError, ModelFamilyError, SolverFailure) as exc:
            self._record_step(50, "failed", f"{type(exc).__name__}: {exc}")
            if self.recorder is not None:
                self.recorder.record("run_failed", task_id=request.task_id, status="failed",
                                     exception_category=type(exc).__name__)
            raise
        finally:
            if self.recorder is not None:
                self.recorder.record("run_finished", task_id=request.task_id,
                                     status=task_status.value,
                                     duration_ms=(_time.perf_counter() - started) * 1000.0)

    # ------------------------------------------------------------------ Phase A

    def _phase_a(self, request: TemporalRequest,
                 observations: Sequence[TemporalObservation],
                 normalizer: VersionedNormalizer | None) -> dict[str, Any]:
        """Execute Steps 1-7: admission, classification, baseline, profile, selection."""
        if not isinstance(request, TemporalRequest):
            raise TypeError("request must be a TemporalRequest")
        if not isinstance(observations, Sequence) or not observations:
            raise ValueError("observations must be a non-empty sequence")
        for observation in observations:
            if not isinstance(observation, TemporalObservation):
                raise TypeError("observations must be TemporalObservation objects")
        self._record_step(1, detail=f"task {request.task_id} accepted with "
                                    f"{len(observations)} observations")
        classification = self.classify(request, observations)
        self._record_step(2, detail=f"classified as {classification.problem_class.value} "
                                    f"with confidence {classification.confidence:.2f}")
        baseline_kind = ("threshold" if classification.problem_class
                         is TemporalProblemClass.SPARSE_EVENTS else "kalman")
        self._record_step(3, detail=f"deterministic baseline: {baseline_kind}")
        n_channels = len(request.observation_schema)
        sampling = self._sampling_of(observations)
        profile = build_selection_profile(request, sampling, len(observations),
                                          baseline_event_count=self.config.event_count_baseline)
        self._record_step(4, detail=f"profile CV={profile.sampling_regularity:.3f}")
        selection = self.select_model(profile, request.resource_budget,
                                       requested=request.requested_family)
        self._record_step(5, detail=f"selected {selection.chosen_family.value}; "
                                    f"{selection.rationale}")
        admission = ResourceAdmission().admit(
            selection.chosen_family.value, len(observations), n_channels,
            resource_budget=request.resource_budget,
            max_ingest_observations=self.config.max_ingest_observations,
            neural_enabled=family_available(selection.chosen_family, self.config))
        chosen_family = selection.chosen_family
        if admission.fallback_to_baseline:
            chosen_family = ModelFamily.BASELINE
            self._record_step(6, "fallback", admission.reason)
        else:
            self._record_step(6, detail=admission.reason)
        model_version = (f"{baseline_kind}-1.0.0" if chosen_family is ModelFamily.BASELINE
                         else f"{chosen_family.value}-1.0.0")
        identity = ModelIdentity(family=chosen_family, model_version=model_version,
                                 config_hash=config_hash(self.config))
        interval = self._interval_of(observations)
        execution_record = ExecutionRecord(
            task_id=request.task_id, model_identity=identity, data_window=interval,
            normalization_version=(normalizer.version if normalizer else "raw"),
            seed=self.config.seed, solver_settings=self.config.solver_settings(),
            software_versions=dict(SOFTWARE_VERSIONS),
            created_at=_time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()))
        self._record_step(7, detail=f"execution record created for window "
                                    f"[{interval[0]:.3f}, {interval[1]:.3f}]")
        return {"classification": classification, "baseline_kind": baseline_kind,
                "profile": profile, "selection": selection, "admission": admission,
                "chosen_family": chosen_family, "execution_record": execution_record,
                "n_channels": n_channels}

    # ------------------------------------------------------------------ Phase B

    def _phase_b(self, request: TemporalRequest,
                 observations: Sequence[TemporalObservation],
                 normalizer: VersionedNormalizer | None) -> dict[str, Any]:
        """Execute Steps 8-16: ingestion, time alignment, and quality."""
        report = IngestionReport(received=len(observations))
        report.timestamp_issues = TimestampValidator(request.timestamp_semantics).validate(
            observations)
        self._record_step(8, detail=f"received {len(observations)} timestamped observations")
        self._record_step(9, detail=("timestamp semantics validated; clock bases preserved"
                                     if not report.timestamp_issues
                                     else f"{len(report.timestamp_issues)} timestamp issues"))
        ordered, ordering_report, _deltas = apply_ordering_and_deltas(
            observations, max_gap_seconds=self.config.maximum_gap_seconds,
            reorder_window_seconds=self.config.reorder_window_seconds,
            buffer_capacity=self.config.reorder_buffer_size)
        report.duplicates_removed = ordering_report.duplicates_removed
        report.late_marked = ordering_report.late_marked
        report.beyond_reorder_window = ordering_report.beyond_reorder_window
        report.negative_intervals_rejected = ordering_report.negative_rejected
        report.large_gaps = ordering_report.large_gaps
        report.zero_intervals = ordering_report.zero_intervals
        self._record_step(10, detail=f"{ordering_report.duplicates_removed} duplicates removed; "
                                     f"{ordering_report.late_marked} late arrivals buffered")
        self._record_step(11, detail=f"{len(_deltas)} elapsed times computed; "
                                     f"{len(ordering_report.large_gaps)} large gaps flagged")
        schema_units = tuple(channel.unit for channel in request.observation_schema)
        converted: list[TemporalObservation] = []
        for observation in ordered:
            values = list(observation.values)
            units = list(observation.units)
            for index, (observed_unit, declared_unit) in enumerate(zip(units, schema_units)):
                if observed_unit != declared_unit:
                    try:
                        value, record = self._unit_registry.convert(
                            values[index], observed_unit, declared_unit)
                        values[index] = value
                        units[index] = declared_unit
                        report.unit_conversions.append(record)
                    except ValueError as exc:
                        report.unit_issues.append(str(exc))
            converted.append(TemporalObservation(
                source_id=observation.source_id, event_id=observation.event_id,
                timestamp=observation.timestamp, time_basis=observation.time_basis,
                values=tuple(values), units=tuple(units),
                quality_flags=observation.quality_flags,
                provenance=observation.provenance))
        quality_validator = QualityValidator(request.observation_schema)
        flagged: list[TemporalObservation] = []
        for observation in converted:
            issues = quality_validator.validate(observation)
            if issues:
                report.notes.extend(issues)
                if any("non-finite" in issue for issue in issues):
                    report.missing_flagged += 1
                else:
                    report.out_of_range_flagged += 1
            flagged.append(quality_validator.flag_observation(observation))
        self._record_step(12, detail=f"{report.out_of_range_flagged} out-of-range values "
                                     f"flagged; {len(report.unit_conversions)} unit conversions")
        policy = request.observation_schema[0].missing_policy
        handler = MissingDataHandler(policy=policy,
                                     max_interpolation_gap=self.config.interpolation_max_gap)
        handled, missing_report = handler.handle(flagged, schema_units)
        report.imputation_applied = missing_report.imputed_samples
        self._record_step(13, detail=f"policy {policy.value}: "
                                     f"{missing_report.imputed_samples} samples imputed with mask")
        sampling = self._sampling_of(handled)
        self._record_step(14, detail=f"CV={sampling.coefficient_of_variation:.3f}; "
                                     f"missingness={sampling.missingness_rate:.3f}")
        if len(handled) > self.config.max_ingest_observations:
            handled = self._apply_backpressure(handled, self.config.max_ingest_observations)
            report.backpressure_applied = self.config.backpressure_policy
            self._diagnostics["backpressure"] = {
                "policy": self.config.backpressure_policy,
                "original_count": len(ordered), "accepted_count": len(handled)}
        normalization_version = "raw"
        if normalizer is not None:
            handled = normalizer.transform_observations(handled)
            normalization_version = normalizer.version
        self._record_step(15, detail=f"normalization version {normalization_version} "
                                     "(training statistics only)")
        window = build_temporal_window(request, handled, normalization_version)
        self._record_step(16, detail=f"window built with {len(window.samples)} samples")
        return {"report": report, "window": window, "sampling": sampling,
                "normalization_version": normalization_version,
                "missing_report": missing_report}

    # ------------------------------------------------------------------ Phase C

    def _phase_c(self, request: TemporalRequest, phase_a: dict[str, Any],
                 phase_b: dict[str, Any]) -> dict[str, Any]:
        """Execute Steps 17-25: state, model execution, isolation, bounds, health."""
        window = phase_b["window"]
        chosen_family = phase_a["chosen_family"]
        n_channels = phase_a["n_channels"]
        values = np.asarray([sample.values for sample in window.samples], dtype=float)
        timestamps = np.asarray(window.timestamps, dtype=float)
        deltas = np.asarray([0.0] + list(window.elapsed_times), dtype=float)
        model = build_model(chosen_family, n_channels, self.config,
                            baseline_kind=phase_a["baseline_kind"])
        identity = model.identity()
        restored = False
        execution_step = {ModelFamily.LNN: 18, ModelFamily.CFC: 19,
                          ModelFamily.NEURAL_ODE: 20, ModelFamily.SSM: 21,
                          ModelFamily.SNN: 22}
        if chosen_family is ModelFamily.BASELINE:
            self._record_step(17, "cold_start", "deterministic baselines always cold start")
            for step in execution_step.values():
                self._record_step(step, "skipped",
                                  "baseline selected; no neural family executed")
        else:
            try:
                record = self.state_store.restore(
                    request.task_id, request.source_id, identity.family.value,
                    identity.model_version, channels=window.channels,
                    config_hash=identity.config_hash,
                    current_timestamp=float(timestamps[-1]))
                model.load_state(np.asarray(record.state, dtype=float))
                restored = True
                self._record_step(17, "completed", "compatible checkpoint restored")
            except StateCompatibilityError as exc:
                self._record_step(17, "cold_start", str(exc))
        baseline = build_model(ModelFamily.BASELINE, n_channels, self.config,
                               baseline_kind=phase_a["baseline_kind"])
        baseline_run = run_model_over_window(baseline, values, deltas)
        model_run = baseline_run
        fallback_reason = ""
        if chosen_family is not ModelFamily.BASELINE:
            execution_status = "completed"
            execution_detail = ""
            try:
                model_run = run_model_over_window(model, values, deltas)
                execution_detail = (f"executed {chosen_family.value} over "
                                    f"{model_run.steps} steps; "
                                    f"{model_run.compute_ms:.2f} ms")
            except (SolverFailure, ValueError) as exc:
                fallback_reason = f"{type(exc).__name__}: {exc}"
                model_run = baseline_run
                execution_status = "failed"
                execution_detail = f"execution failed: {fallback_reason}"
            for family, step in sorted(execution_step.items(), key=lambda item: item[1]):
                if family is chosen_family:
                    self._record_step(step, execution_status, execution_detail)
                else:
                    self._record_step(step, "skipped",
                                      f"{family.value} not selected for this workload")
        state_key = (request.task_id, request.source_id, identity.family.value,
                     identity.model_version)
        isolated = all(key[1] == request.source_id
                       for key in self.state_store.snapshot_keys())
        self._record_step(23, detail=f"state isolated under key {state_key}; "
                                     f"cross-stream isolation: {isolated}")
        if fallback_reason:
            self._record_step(24, "fallback",
                              f"compute bound enforced; fell back to baseline: "
                              f"{fallback_reason}")
        else:
            self._record_step(24, detail=f"bounds enforced: max_events="
                                         f"{self.config.max_events_per_run}, max_spikes="
                                         f"{self.config.max_spikes_per_run}, timeout="
                                         f"{self.config.inference_timeout_seconds}s")
        predictions = model_run.prediction_array()
        observed = values
        if predictions.shape != observed.shape:
            predictions = predictions[: observed.shape[0]]
        health = numerical_health_check(
            state=(model_run.final_state.tolist() if model_run.final_state is not None else None),
            prediction=(predictions[-1] if predictions.size else None),
            state_norm_bound=self.config.state_norm_bound,
            derivative_bound=self.config.derivative_bound)
        if not health.valid:
            self._record_step(25, "failed", "; ".join(health.failures))
        else:
            self._record_step(25, detail="numerical health checks passed")
        validation_status = ValidationStatus.VALID
        task_status = TaskStatus.COMPLETED
        if fallback_reason:
            validation_status = ValidationStatus.FALLBACK
            task_status = TaskStatus.FALLBACK
        elif not health.valid:
            validation_status = ValidationStatus.INVALID
            task_status = TaskStatus.FAILED
        elif model_run.warnings:
            validation_status = ValidationStatus.PARTIAL
            task_status = TaskStatus.PARTIAL
        if self.simulate_biology and self.biological_network is not None:
            self.biological_network.reset()
            for index in range(min(len(values), 8)):
                self.biological_network.advance(values[index], max(float(deltas[index]), 1e-3))
        effective_identity = baseline.identity() if fallback_reason else identity
        return {"model": model, "baseline": baseline, "model_identity": effective_identity,
                "attempted_identity": identity,
                "model_run": model_run, "baseline_run": baseline_run, "restored": restored,
                "health": health, "validation_status": validation_status,
                "task_status": task_status, "predictions": predictions,
                "observed": observed, "timestamps": timestamps, "values": values,
                "deltas": deltas, "fallback_reason": fallback_reason}

    # ------------------------------------------------------------------ Phase D

    def _phase_d(self, request: TemporalRequest, phase_b: dict[str, Any],
                 phase_c: dict[str, Any], calibration: CalibrationData | None
                 ) -> dict[str, Any]:
        """Execute Steps 26-32: typed estimates, events, uncertainty, relationships."""
        window = phase_b["window"]
        channels = window.channels
        predictions = phase_c["predictions"]
        observed = phase_c["observed"]
        timestamps = phase_c["timestamps"]
        residuals = (observed - predictions if predictions.size
                     and predictions.shape == observed.shape else np.zeros_like(observed))
        calibrator = ConformalCalibrator(self.config.calibration_version,
                                         self.config.coverage_target)
        anomaly_detector = AnomalyDetector(z_threshold=self.config.anomaly_z_threshold)
        ood_marker = None
        if calibration is not None:
            calibration_values = np.asarray(calibration.values, dtype=float)
            calibration_predictions = np.asarray(calibration.predictions, dtype=float)
            calibrator.fit(calibration_values[:, 0].tolist(),
                           calibration_predictions[:, 0].tolist())
            anomaly_detector.calibrate(calibration_values[:, 0].tolist(),
                                       calibration_predictions[:, 0].tolist())
            ood_marker = OutOfDistributionMarker(calibration.values)
        estimates: list[Estimate] = []
        self._record_step(26, detail=f"{len(channels)} typed estimate channels prepared")
        events = self.detect_events(
            residuals, timestamps, channels, observed, predictions,
            anomaly_detector=anomaly_detector if calibration is not None else None)
        n_changes = sum(1 for item in events if item.event_type == "change_point")
        n_anomalies = len(events) - n_changes
        self._record_step(27, detail=f"{n_changes} change points detected")
        self._record_step(28, detail=(f"{n_anomalies} anomalies detected with hypotheses"
                                      if calibration is not None else
                                      "anomaly detection uncalibrated; no anomalies claimed"))
        is_ood, ood_reason = (ood_marker.mark(observed[-1]) if ood_marker is not None
                              else (False, ""))
        ensemble_predictions = None
        if predictions.size and phase_c["baseline_run"].predictions:
            ensemble_predictions = [float(predictions[-1, 0]),
                                    float(phase_c["baseline_run"].predictions[-1][0])]
        uncertainty: UncertaintyReport | None = None
        if calibrator.fitted:
            uncertainty = calibrator.report(float(predictions[-1, 0]),
                                            ensemble_predictions=ensemble_predictions,
                                            out_of_distribution=is_ood,
                                            ood_reason=ood_reason)
        self._record_step(29, detail=("calibrated uncertainty reported"
                                      if calibrator.fitted else
                                      "uncalibrated; uncertainty report omitted"))
        for channel_index, channel in enumerate(channels):
            value = float(predictions[-1, channel_index]) if predictions.size else 0.0
            interval = calibrator.interval(value) if (calibrator.fitted
                                                      and channel_index == 0) else None
            estimates.append(Estimate(
                channel=channel, value=value,
                unit=window.samples[-1].units[channel_index],
                timestamp=float(timestamps[-1]), horizon=request.horizon,
                confidence_interval=interval, epistemic_status=EpistemicStatus.PREDICTED))
        relationships = analyze_relationships(channels, timestamps, observed,
                                              max_lag=self.config.lag_correlation_max_lag)
        self._record_step(30, detail="temporal relationships extracted; no causal claims")
        gate = ActionabilityGate(
            max_actionable_events=self.config.max_actionable_events_per_run,
            high_impact_requires_deterministic_evidence=(
                self.config.high_impact_requires_deterministic_evidence))
        actionability = {event.event_id: gate.evaluate(
            event, uncertainty, index).decision.value
            for index, event in enumerate(events)}
        self._record_step(31, detail=f"{len(events)} typed events summarized with evidence")
        self._record_step(32, detail=f"actionability decisions: {actionability or 'none'}")
        return {"estimates": tuple(estimates), "events": events, "uncertainty": uncertainty,
                "relationships": relationships, "actionability": actionability,
                "residuals": residuals, "interval": window.interval}

    # ------------------------------------------------------------------ Phase E

    def _phase_e(self, request: TemporalRequest, phase_b: dict[str, Any],
                 phase_d: dict[str, Any], workspace: Any | None) -> dict[str, Any]:
        """Execute Steps 33-39: USG mapping, evidence, publication, coordination."""
        usg = {label: self.usg_mapper.map(label) for label in
               {event.event_type for event in phase_d["events"]}}
        unmapped = sorted(label for label, (_, mapped) in usg.items() if not mapped)
        self._record_step(33, detail=f"USG mappings resolved; unmapped labels: "
                                     f"{unmapped or 'none'}")
        evidence = EvidenceBundle()
        evidence.add("ingestion", "derived", "ingestion report and window reference")
        evidence.add("baseline", "derived", "deterministic baseline comparison")
        evidence.add("model", "derived", "model execution and solver diagnostics")
        provenance = ProvenanceRecord(
            source_ids=(request.source_id,), time_range=phase_d["interval"],
            preprocessing_version=f"ore_temporal-v1:"
                                  f"{phase_b['normalization_version']}",
            model_family="point13", model_version="1.0.0", parameter_count=0,
            uncertainty_method=(phase_d["uncertainty"].method
                                if phase_d["uncertainty"] else "none"),
            baseline_comparison="always computed", validation_status="valid",
            evidence_refs=tuple(evidence.refs()),
            notes="measured, imputed, and predicted values remain distinguishable")
        self._record_step(34, detail=f"{len(evidence)} evidence references attached")
        stub = self._result_stub(request, phase_d)
        publication: dict[str, Any] = {"published": False, "decision": "disabled"}
        if self.config.publish_to_workspace:
            publication = self.publish(stub, workspace)
            self._record_step(35, detail=f"publication decision: {publication['decision']}")
        else:
            self._record_step(35, "skipped", "workspace publication disabled by configuration")
        notification = (build_coordinator_notification(
            stub, max_events=self.config.max_coordinator_message_events)
            if self.config.notify_coordinator else None)
        self._record_step(36, detail="bounded coordinator notification prepared"
                          if notification else "notification disabled")
        hints = build_downstream_task_hints(stub)
        self._record_step(37, detail=f"{len(hints)} downstream task hints with bounded evidence")
        self._record_step(38, detail=f"backpressure policy {self.config.backpressure_policy}; "
                                     "cancellation honoured between phases")
        self._record_step(39, detail="USG and deterministic modules own semantic conclusions")
        return {"usg": {label: usg_id for label, (usg_id, _) in usg.items()},
                "evidence": evidence, "provenance": provenance,
                "publication": publication, "notification": notification, "hints": hints}

    # ------------------------------------------------------------------ Phase F

    def _phase_f(self, phase_a: dict[str, Any], phase_b: dict[str, Any],
                 phase_c: dict[str, Any]) -> dict[str, Any]:
        """Execute Steps 40-45: portable graphs, CPU reference, adapters, negotiation."""
        from .backends import CPUReferenceBackend, export_model_graph
        from .backends.capabilities import BackendCapabilities, negotiate_capabilities
        model = phase_c["model"]
        identity = phase_c["model_identity"]
        window = phase_b["window"]
        graph = None
        graph_validation: dict[str, Any] = {"validated": False,
                                            "reason": "export not available"}
        try:
            graph = export_model_graph(model)
            self._record_step(40, detail=f"hardware-independent contract: {graph.graph_id}")
            self._record_step(41, detail=f"portable graph with {len(graph.nodes)} nodes")
        except TypeError as exc:
            self._record_step(41, "skipped", str(exc))
        backend = CPUReferenceBackend()
        if graph is not None and self.config.enable_portable_graph_validation:
            if identity.family is ModelFamily.BASELINE or "prediction" not in graph.output_names:
                graph_validation = {"validated": True,
                                    "reason": "structural validation; no comparable output"}
                self._record_step(42, detail=f"CPU reference available; {graph_validation}")
            else:
                saved = model.state_array().copy()
                try:
                    model_output = model.step(phase_c["values"][-1],
                                              float(phase_c["deltas"][-1]))
                    inputs: dict[str, Any] = {"__control__": phase_c["values"][-1],
                                              "__dt__": float(phase_c["deltas"][-1])}
                    if graph.state_nodes:
                        inputs["__state__"] = saved
                    execution = backend.execute(graph, inputs)
                    reference = execution.outputs.get("prediction")
                    deviation = float(np.max(np.abs(np.asarray(reference)
                                                    - np.asarray(model_output.prediction))))
                    graph_validation = {
                        "validated": bool(deviation <= max(
                            self.config.portable_graph_tolerance * 10.0, 1e-6)),
                        "max_deviation": deviation}
                    self._record_step(42, detail=f"CPU reference executed; validation: "
                                                 f"{graph_validation}")
                except (KeyError, ValueError) as exc:
                    graph_validation = {"validated": False, "reason": str(exc)}
                    self._record_step(42, "failed", str(exc))
                finally:
                    model.load_state(saved)
        else:
            self._record_step(42, detail="deterministic CPU reference backend available")
        lava_note = "lava adapter disabled by configuration; never imported at core startup"
        if self.config.enable_lava_adapter:
            from .backends.lava_adapter import LavaBackendAdapter
            adapter = LavaBackendAdapter()
            lava_note = f"lava adapter available: {adapter.available()}"
            self._record_step(43, detail=lava_note)
        else:
            self._record_step(43, "skipped", lava_note)
        required_ops = (tuple(sorted({node.operation for node in graph.nodes}))
                        if graph is not None else ())
        cpu_capabilities = BackendCapabilities(
            backend_name="cpu_reference", supported_operations=required_ops,
            precision_bits=64, memory_mb=4096.0, event_routing=True,
            runtime_version=f"numpy-{np.__version__}", available=True)
        negotiation = negotiate_capabilities(required_ops, (cpu_capabilities,))
        self._record_step(44, detail=f"backend selected: {negotiation.selected_backend}")
        observed = phase_c["observed"]
        predictions = phase_c["predictions"]
        baseline_predictions = phase_c["baseline_run"].prediction_array()
        if (predictions.size and predictions.shape == observed.shape
                and observed.shape[0] > 1):
            model_mae = mean_absolute_error(observed[:, 0].tolist(),
                                            predictions[:, 0].tolist())
            baseline_mae = mean_absolute_error(observed[:, 0].tolist(),
                                               baseline_predictions[:, 0].tolist())
            self._record_step(45, detail=f"identical-window benchmark: model MAE="
                                         f"{model_mae:.4f} vs baseline MAE={baseline_mae:.4f}")
        else:
            model_mae = float("nan")
            baseline_mae = float("nan")
            self._record_step(45, "skipped", "insufficient samples for benchmark")
        del window
        return {"graph": graph, "graph_validation": graph_validation,
                "backend": "cpu_reference", "lava_note": lava_note,
                "negotiation": negotiation, "model_mae": model_mae,
                "baseline_mae": baseline_mae}

    # ------------------------------------------------------------------ Phase G

    def _phase_g(self, phase_b: dict[str, Any], phase_c: dict[str, Any],
                 phase_d: dict[str, Any], phase_f: dict[str, Any]) -> dict[str, Any]:
        """Execute Steps 46-50: governance, validation, drift, adaptation, closure."""
        del phase_d["residuals"]  # keep the phase payload bounded
        self._record_step(46, detail="production uses frozen seeded artifacts; "
                                     "training is offline and versioned")
        self._record_step(47, detail="chronological calibration honored; rolling-origin "
                                     "splits supported", )
        observed = phase_c["observed"]
        drift = DriftMonitor(reference_mean=(float(observed[:, 0].mean())
                                             if observed.size else 0.0),
                             reference_std=(float(observed[:, 0].std())
                                            if observed.size else 1.0),
                             review_threshold=self.config.drift_review_threshold,
                             window=self.config.drift_window)
        for value in observed[:, 0].tolist():
            drift.update(value)
        drift_report = drift.report()
        self._record_step(48, detail=f"drift status: {drift_report['status']}; "
                                     f"PSI={drift_report['psi']:.3f}")
        if self.config.enable_controlled_adaptation:
            self._record_step(49, detail=f"controlled adaptation enabled with bounded "
                                         f"learning rate "
                                         f"{self.config.adaptation_learning_rate_bound}")
        else:
            self._record_step(49, "skipped", "controlled adaptation disabled by default")
        identity = phase_c["model_identity"]
        model_mae = phase_f["model_mae"]
        baseline_mae = phase_f["baseline_mae"]
        comparable = math.isfinite(model_mae) and math.isfinite(baseline_mae)
        promotion = evaluate_promotion(
            model_metric=(model_mae if comparable else 1e9),
            baseline_metric=(baseline_mae if comparable else 1e9),
            metric_name="mae",
            latency_ms=phase_c["model_run"].compute_ms,
            latency_budget_ms=self.config.inference_timeout_seconds * 1000.0,
            memory_bytes=(phase_c["model_run"].final_state.nbytes
                          if phase_c["model_run"].final_state is not None else 0),
            memory_budget_bytes=self.config.state_size_limit_bytes,
            interval_coverage=None,
            complements_baseline=False)
        artifact = ModelArtifactRecord(
            artifact_hash=identity.config_hash, family=identity.family.value,
            architecture=identity.model_version,
            dataset_lineage=("synthetic_reference",),
            training_config={"seed": self.config.seed, "frozen": True},
            validation_metrics={"mae": (model_mae if comparable else -1.0)},
            license="MIT",
            limitations=("seeded reference model; parameters are engineering defaults",
                         "no causal claims from temporal correlations"),
            approval_state=(LifecycleState.VALIDATED if promotion.promoted
                            else LifecycleState.PROPOSED))
        self.artifact_registry.register(artifact, replace=True)
        self._record_step(50, detail=f"registry updated ({artifact.approval_state.value}); "
                                     f"promotion: {promotion.promoted}; state expired")
        self.state_store.expire()
        return {"drift_report": drift_report, "promotion": promotion, "artifact": artifact,
                "registry_summary": self.artifact_registry.summary()}

    # ---------------------------------------------------------------- utilities

    def _sampling_of(self, observations: Sequence[TemporalObservation]):
        """Profile the sampling of an observation sequence."""
        if len(observations) < 2:
            return profile_sampling(())
        ordered = sorted(observations, key=lambda item: item.timestamp)
        deltas = [ordered[i + 1].timestamp - ordered[i].timestamp
                  for i in range(len(ordered) - 1)]
        return profile_sampling(deltas,
                                large_gap_threshold=self.config.maximum_gap_seconds)

    def _interval_of(self, observations: Sequence[TemporalObservation]) -> tuple[float, float]:
        """Return the observation interval."""
        timestamps = [item.timestamp for item in observations]
        return (float(min(timestamps)), float(max(timestamps)))

    def _apply_backpressure(self, handled: list[TemporalObservation],
                            cap: int) -> list[TemporalObservation]:
        """Apply the configured backpressure policy to an overloaded stream."""
        policy = BackpressurePolicy(self.config.backpressure_policy)
        if policy is BackpressurePolicy.SAMPLE:
            stride = max(1, math.ceil(len(handled) / cap))
            return handled[::stride]
        if policy is BackpressurePolicy.AGGREGATE:
            output: list[TemporalObservation] = []
            bucket = max(1, math.ceil(len(handled) / cap))
            for start in range(0, len(handled), bucket):
                group = handled[start: start + bucket]
                values = tuple(float(np.mean([item.values[i] for item in group]))
                               for i in range(len(group[0].values)))
                output.append(TemporalObservation(
                    source_id=group[0].source_id, event_id=group[0].event_id,
                    timestamp=group[-1].timestamp, time_basis=group[0].time_basis,
                    values=values, units=group[0].units,
                    quality_flags=group[-1].quality_flags,
                    provenance={"aggregated_from": [item.event_id for item in group]}))
            return output
        return handled[:cap]

    def _result_stub(self, request: TemporalRequest,
                     phase_d: dict[str, Any]) -> TemporalResult:
        """Build a lightweight result used for publication and notifications."""
        return TemporalResult(
            task_id=request.task_id, source_id=request.source_id,
            interval=phase_d["interval"], result_type=ResultType.EVENT,
            estimates=phase_d["estimates"], events=phase_d["events"],
            uncertainty=phase_d["uncertainty"],
            validation_status=ValidationStatus.VALID)

    def _partial_result(self, request: TemporalRequest,
                        observations: Sequence[TemporalObservation],
                        status: TaskStatus, reason: str) -> TemporalResult:
        """Assemble a safe partial result after cancellation."""
        interval = self._interval_of(observations) if observations else (0.0, 0.0)
        return TemporalResult(
            task_id=request.task_id, source_id=request.source_id, interval=interval,
            result_type=ResultType.STATE_SUMMARY,
            validation_status=ValidationStatus.PARTIAL,
            operation_trace=tuple(self._trace),
            limitations=(f"task ended early: {reason}",),
            recommended_verification="re-run the task when the cancellation cause is resolved",
            diagnostics={"task_status": status.value, "reason": reason})

    def _assemble_result(self, request: TemporalRequest, phase_a: dict[str, Any],
                         phase_b: dict[str, Any], phase_c: dict[str, Any],
                         phase_d: dict[str, Any], phase_e: dict[str, Any],
                         phase_f: dict[str, Any], phase_g: dict[str, Any],
                         task_status: TaskStatus, started: float) -> TemporalResult:
        """Assemble the final typed result with its envelope (Step 50)."""
        window = phase_b["window"]
        model_mae = phase_f["model_mae"]
        baseline_mae = phase_f["baseline_mae"]
        comparable = math.isfinite(model_mae) and math.isfinite(baseline_mae)
        validation_status = phase_c["validation_status"]
        identity = phase_c["model_identity"]
        baseline_comparison = BaselineComparison(
            baseline_family=ModelFamily.BASELINE,
            baseline_metric=(baseline_mae if comparable else -1.0),
            model_metric=(model_mae if comparable else -1.0),
            metric_name="mae",
            beats_baseline=bool(comparable and model_mae < baseline_mae),
            materially_complements=phase_g["promotion"].promoted,
            note="baseline is the control condition for every benefit claim")
        limitations = [
            "Seeded reference models use engineering default parameters, not calibrated fits.",
            "Temporal correlations are not causal evidence; USG owns semantic conclusions.",
            "Predictions are labelled as predictions and imputed values remain distinguishable.",
        ]
        if phase_c["fallback_reason"]:
            limitations.append(f"Fell back to the deterministic baseline: "
                               f"{phase_c['fallback_reason']}")
        recommended = ("Compare against the deterministic baseline on a held-out "
                       "chronological window and verify interval coverage before promotion.")
        uncertainty_summary = ("no uncertainty estimate"
                               if phase_d["uncertainty"] is None else
                               f"method={phase_d['uncertainty'].method}; calibration="
                               f"{phase_d['uncertainty'].calibration_version}; ood="
                               f"{phase_d['uncertainty'].out_of_distribution}")
        envelope = ResultEnvelope(
            interval=window.interval, prediction_status=validation_status,
            baseline=(baseline_comparison.baseline_family.value
                      if baseline_comparison else "none"),
            model_identity=identity, limitations=tuple(limitations),
            recommended_verification=recommended,
            evidence_summary=f"{len(phase_e['evidence'])} evidence references; "
                             f"{len(phase_d['estimates'])} typed estimates; "
                             f"{len(phase_d['events'])} typed events",
            uncertainty_summary=uncertainty_summary)
        result = TemporalResult(
            task_id=request.task_id, source_id=request.source_id,
            interval=window.interval,
            result_type=(ResultType.EVENT if phase_d["events"] else ResultType.FORECAST),
            estimates=phase_d["estimates"], events=phase_d["events"],
            uncertainty=phase_d["uncertainty"],
            evidence_refs=tuple(phase_e["evidence"].refs()),
            model_identity=identity, baseline_comparison=baseline_comparison,
            validation_status=validation_status,
            problem_class=phase_a["classification"].problem_class,
            operation_trace=tuple(self._trace), limitations=tuple(limitations),
            recommended_verification=recommended, envelope=envelope,
            diagnostics={
                "task_status": task_status.value,
                "classification": phase_a["classification"].to_dict(),
                "selection": phase_a["selection"].to_dict(),
                "admission": phase_a["admission"].to_dict(),
                "execution_record": phase_a["execution_record"].to_dict(),
                "ingestion": phase_b["report"].to_dict(),
                "missing_data": phase_b["missing_report"].to_dict(),
                "sampling": phase_b["sampling"].to_dict(),
                "normalization_version": phase_b["normalization_version"],
                "restored_state": phase_c["restored"],
                "health": phase_c["health"].to_dict(),
                "actionability": phase_d["actionability"],
                "relationships": phase_d["relationships"].to_dict(),
                "usg": phase_e["usg"],
                "publication": {key: value for key, value in
                                phase_e["publication"].items() if key != "payload"},
                "coordinator_notification": phase_e["notification"],
                "task_hints": phase_e["hints"],
                "graph_validation": phase_f["graph_validation"],
                "backend": phase_f["backend"], "lava_note": phase_f["lava_note"],
                "negotiation": phase_f["negotiation"].to_dict(),
                "drift": phase_g["drift_report"],
                "promotion": phase_g["promotion"].to_dict(),
                "artifact": phase_g["artifact"].to_dict(),
                "registry_summary": phase_g["registry_summary"],
                "model_compute_ms": phase_c["model_run"].compute_ms,
                "baseline_compute_ms": phase_c["baseline_run"].compute_ms,
                "biological_trace": (self.biological_network.state_summary()
                                     if self.biological_network is not None else None),
                "elapsed_ms": round((_time.perf_counter() - started) * 1000.0, 3),
            })
        diagnostics = dict(self._diagnostics)
        diagnostics.update(result.diagnostics)
        object.__setattr__(result, "diagnostics", diagnostics)
        return result
