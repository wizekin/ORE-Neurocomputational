"""Persistent 18-stage hippocampal replay and sleep systems-consolidation processor."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
import uuid
from typing import Any, Iterable

import numpy as np

from .biology import HippocampalCorticalNetwork
from .contracts import (
    HippocampalTrace,
    ReplayOpportunity,
    SleepConsolidationConfig,
    SleepConsolidationResult,
)
from .functional import (
    classify_offline_state,
    compressed_replay_times,
    detect_sharp_wave_ripple,
    detect_spindle,
    estimate_oscillation_phase,
    pearson_correlation,
    replay_priority,
    rhythm_alignment,
    softmax,
    reinstatement_similarity,
)
from .registry import OPERATION_REGISTRY


class SleepDependentSystemsConsolidationProcessor:
    """Run source-ordered replay operations while retaining state across offline opportunities.

    Traces are registered explicitly. Each process call represents one offline interval/event;
    repeated calls reuse cortical weights, eligibility, memory prototypes, and consolidation
    estimates. Without an eligible state, SWR candidate, and stored trace, later replay stages are
    explicitly marked skipped rather than fabricating neural events.
    """

    def __init__(
        self,
        config: SleepConsolidationConfig | None = None,
        database_path: str | Path | None = None,
        repository_root: str | Path | None = None,
    ) -> None:
        """Initialize the population model, trace stores, replay history, and optional SQLite path."""
        self.config = config or SleepConsolidationConfig()
        self.network = HippocampalCorticalNetwork(self.config)
        self.traces: dict[str, HippocampalTrace] = {}
        self.replay_counts: dict[str, int] = {}
        self.consolidation_indices: dict[str, float] = {}
        self.replay_history: list[dict[str, Any]] = []
        self.coupling_history: list[float] = []
        self.performance_changes: list[float] = []
        self.opportunity_count = 0
        self.evaluation_count = 0
        self.database_path = Path(database_path).expanduser().resolve() if database_path else None
        self.repository_root = (Path(repository_root).expanduser().resolve() if repository_root
                                else Path(__file__).resolve().parents[1])

    def add_trace(self, trace: HippocampalTrace) -> dict[str, object]:
        """Register an episodic trace and construct its initial hippocampal-cortical codes."""
        if trace.trace_id not in self.traces and len(self.traces) >= self.config.max_trace_count:
            raise ValueError("configured trace capacity has been reached")
        self.traces[trace.trace_id] = trace
        self.replay_counts.setdefault(trace.trace_id, 0)
        self.consolidation_indices.setdefault(trace.trace_id, 0.0)
        return self.network.register_trace(trace)

    def _log_sqlite(self, trace: list[dict[str, Any]], elapsed_ms: float) -> dict[str, Any]:
        """Persist operation descriptions and execution outcomes through the repository logger."""
        if self.database_path is None:
            return {"enabled": False, "reason": "database_path_not_configured"}
        from ore_logger.recorder import SQLiteRunRecorder

        run_id = uuid.uuid4().hex
        recorder = SQLiteRunRecorder(
            database_path=self.database_path,
            repository_root=self.repository_root,
            run_id=run_id,
            component="ore_sleep_consolidation",
            callable_name="SleepDependentSystemsConsolidationProcessor.process",
            started_at_utc=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            entry_kind="neurocomputational_component",
        )
        for index, (spec, row) in enumerate(zip(OPERATION_REGISTRY, trace), start=1):
            recorder.log_operation({
                "step": index, "operation": spec.operation_id, "name": spec.name,
                "status": row.get("status", "completed"), "phase": "sleep_systems_consolidation",
                "explanation": spec.purpose, "mathematical_form": spec.mathematical_form,
                "biological_reference": spec.biological_reference,
                "evidence_level": spec.evidence_level, "inputs": list(spec.inputs),
                "outputs": list(spec.outputs), "functional_output": row.get("functional_output", {}),
                "biological_output": row.get("biological_output", {}),
            })
        return {"enabled": True, **recorder.finish("completed", elapsed_ms,
                                                     output_type="SleepConsolidationResult")}

    def process(
        self, opportunity: ReplayOpportunity,
        traces: Iterable[HippocampalTrace] | None = None,
    ) -> SleepConsolidationResult:
        """Process one offline opportunity through all 18 replay and consolidation operations."""
        started = time.perf_counter()
        for trace in traces or ():
            self.add_trace(trace)
        self.opportunity_count += 1
        rows: list[dict[str, Any]] = []

        def record(index: int, functional: dict[str, Any],
                   biological: dict[str, Any] | None = None, status: str = "completed") -> None:
            """Append one inspectable operation result in the exact source order."""
            spec = OPERATION_REGISTRY[index]
            rows.append({"operation_id": spec.operation_id, "operation": spec.name,
                         "status": status, "evidence_level": spec.evidence_level,
                         "functional_output": functional, "biological_output": biological or {}})

        def skipped(index: int, reason: str) -> None:
            """Record an operation that cannot run because its required evidence is unavailable."""
            record(index, {"status": "skipped", "reason": reason}, status="skipped")

        cfg = self.config
        gate = classify_offline_state(opportunity.sleep_state, opportunity.sleep_state_confidence,
                                      cfg.sleep_state_threshold, cfg.allow_awake_rest)
        record(0, gate, {"candidate_traces": len(self.traces)})

        phase = opportunity.slow_oscillation_phase_rad
        phase_report: dict[str, Any] = {"phase_rad": phase, "source": "supplied_phase"}
        if phase is None and opportunity.slow_oscillation_signal:
            phase_report = estimate_oscillation_phase(opportunity.slow_oscillation_signal,
                                                       opportunity.sample_rate_hz)
            phase = float(phase_report["phase_rad"])
            phase_report["source"] = "filtered_signal_and_hilbert_transform"
            phase_report.pop("filtered_signal", None)
        if phase is None:
            phase_report = {"phase_rad": None, "source": "not_observed",
                            "status": "slow_phase_unavailable"}
        record(1, phase_report)

        spindle = detect_spindle(opportunity.spindle_envelope, opportunity.spindle_duration_ms, cfg)
        record(2, spindle, {"thalamocortical_spindle_proxy": spindle["detected"]})
        ripple = detect_sharp_wave_ripple(opportunity.ripple_band_z, opportunity.sharp_wave_z,
                                          opportunity.ripple_duration_ms, cfg)
        record(3, ripple, {"hippocampal_SWR_candidate": ripple["detected"]})

        selected: HippocampalTrace | None = None
        selection: dict[str, Any] = {"status": "no_candidate_selected", "probabilities": {}}
        completed_pattern: np.ndarray | None = None
        replay_states: list[np.ndarray] = []
        replay_times = np.asarray([], dtype=np.float64)
        alignment: dict[str, Any] = {"score": 0.0, "status": "replay_not_available"}
        reinstatement: dict[str, Any] = {"status": "replay_not_available"}
        plasticity: dict[str, Any] = {"status": "not_applied"}
        integration: dict[str, Any] = {"status": "not_applied"}
        homeostasis: dict[str, Any] = {"status": "not_applied"}
        retrieval: dict[str, Any] = {"status": "trace_not_selected"}
        consolidation: dict[str, Any] = {"status": "trace_not_selected"}
        validation: dict[str, Any] = {"status": "no_external_performance_measurements"}
        cortical_activity = self.network.cortical_activity.copy()

        replay_possible = bool(gate["eligible"] and ripple["detected"] and self.traces)
        if replay_possible:
            candidates = list(self.traces.values())
            cue = opportunity.partial_cue
            priority_scores: list[float] = []
            for candidate in candidates:
                cue_match = (0.0 if cue is None else
                             0.15 * (1.0 + self._trace_similarity(candidate, cue)) * 0.5)
                priority_scores.append(replay_priority(
                    candidate.recency, candidate.salience, candidate.value,
                    candidate.goal_relevance, candidate.interference_risk,
                ) + cue_match)
            probabilities = softmax(priority_scores, cfg.replay_temperature)
            winner = int(np.argmax(probabilities))
            selected = candidates[winner]
            selection = {"status": "selected", "trace_id": selected.trace_id,
                         "priority_scores": priority_scores,
                         "probabilities": {item.trace_id: float(probabilities[index])
                                           for index, item in enumerate(candidates)},
                         "selection_method": "maximum_probability_model_choice"}
            record(4, selection, {"candidate_count": len(candidates)})
            completed_pattern, completion_report = self.network.pattern_complete(
                selected.trace_id, opportunity.partial_cue)
            record(5, completion_report,
                   {"completed_hippocampal_population_norm": float(np.linalg.norm(completed_pattern))})
            replay_states, sequence_report = self.network.replay_sequence(
                selected.trace_id, opportunity.replay_direction)
            replay_times = compressed_replay_times(len(replay_states), cfg.replay_compression)
            sequence_output = {"direction": opportunity.replay_direction,
                               "element_count": len(replay_states),
                               "compressed_timestamps_ms": replay_times.tolist(),
                               "compression_ratio": cfg.replay_compression}
            record(6, sequence_output, sequence_report)
            phases = [float(item["slow_phase_rad"]) for item in self.replay_history
                      if item.get("slow_phase_rad") is not None]
            alignment = rhythm_alignment(
                phase, bool(spindle["detected"]), float(spindle["strength"]),
                opportunity.spindle_phase_rad, cfg.preferred_up_phase_rad,
                bool(ripple["detected"]), phases,
            )
            alignment["phase_source"] = phase_report.get("source")
            record(7, alignment, {"timing_gate": float(alignment["score"])})
            prior_cortical = self.network.cortical_activity.copy()
            cortical_activity, propagation = self.network.propagate_to_cortex(
                replay_states, float(alignment["score"]))
            record(8, propagation, {"cortical_population_size": int(cortical_activity.size)})
            target_cortical = self.network.cortical_memories[selected.trace_id]
            reinstatement = reinstatement_similarity(cortical_activity, target_cortical)
            reinstatement["status"] = "partial_or_full_cortical_reinstatement_estimated"
            record(9, reinstatement)
            eligibility = self.network.update_eligibility(
                prior_cortical, cortical_activity, opportunity.pre_post_interval_ms)
            record(10, eligibility, {"eligibility_trace_norm": eligibility["eligibility_norm"]})
            if float(alignment["score"]) > 0.0:
                plasticity = self.network.apply_plasticity(
                    float(alignment["score"]), opportunity.sleep_state_confidence)
            else:
                plasticity = {"status": "no_update_weak_rhythm_coupling",
                              "weight_delta_norm": 0.0}
            record(11, plasticity)
            integration = self.network.integrate_cortical_memory(selected.trace_id, cortical_activity)
            integration["shared_feature_integration"] = "trace-specific_prototype_updated"
            record(12, integration)
            self.replay_counts[selected.trace_id] += 1
            self.replay_history.append({
                "opportunity_id": opportunity.opportunity_id,
                "trace_id": selected.trace_id,
                "timestamp_ms": opportunity.timestamp_ms,
                "slow_phase_rad": phase,
                "alignment_score": float(alignment["score"]),
                "reinstatement_similarity": float(reinstatement["overall_similarity"]),
            })
            self.coupling_history.append(float(alignment["score"]))
            record(13, {"trace_id": selected.trace_id,
                        "replay_count": self.replay_counts[selected.trace_id],
                        "total_replay_events": len(self.replay_history)},
                   {"cumulative_plasticity_events": self.network.plasticity_event_count})
            homeostasis = self.network.stabilize_homeostasis()
            record(14, homeostasis)
            retrieval = self.network.retrieval_probe(selected.trace_id, opportunity.partial_cue)
            retrieval["reinstatement_score_after_replay"] = float(reinstatement["overall_similarity"])
            record(15, retrieval)
            prior_index = self.consolidation_indices[selected.trace_id]
            drive = cfg.consolidation_rate * float(alignment["score"]) * max(
                0.0, float(reinstatement["overall_similarity"]))
            interference = cfg.destabilization_rate * selected.interference_risk
            updated_index = float(np.clip(
                prior_index + drive * (1.0 - prior_index) - interference * prior_index, 0.0, 1.0))
            self.consolidation_indices[selected.trace_id] = updated_index
            consolidation = {"trace_id": selected.trace_id, "previous_index": prior_index,
                             "updated_index": updated_index, "consolidation_drive": drive,
                             "destabilization_term": interference,
                             "replay_count": self.replay_counts[selected.trace_id]}
            record(16, consolidation)
            pre = opportunity.observed_pre_performance
            post = opportunity.observed_post_performance
            change = None if pre is None or post is None else float(post - pre)
            if change is not None:
                self.performance_changes.append(change)
            validation = {"status": "external_pre_post_comparison" if change is not None
                          else "performance_not_fully_observed",
                          "observed_pre_performance": pre, "observed_post_performance": post,
                          "retention_change": change,
                          "coupling_performance_correlation": pearson_correlation(
                              self.coupling_history, self.performance_changes),
                          "causal_improvement_claimed": False}
            record(17, validation)
        else:
            reason = ("offline_state_not_eligible" if not gate["eligible"] else
                      "no_SWR_candidate_detected" if not ripple["detected"] else
                      "no_registered_hippocampal_traces")
            for index in range(4, 18):
                skipped(index, reason)

        if len(rows) != len(OPERATION_REGISTRY):
            raise RuntimeError("operation trace does not cover all 18 registered stages")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger = self._log_sqlite(rows, elapsed_ms)
        biological = self.network.snapshot()
        biological.update({"model_scope": "rate_population_surrogate_not_cell_level_digital_twin",
                           "literal_neuronal_arithmetic_claimed": False,
                           "trace_specific_storage_preserved": True})
        return SleepConsolidationResult(
            opportunity_id=opportunity.opportunity_id,
            selected_trace_id=selected.trace_id if selected else None,
            operation_trace=rows,
            sleep_gate=gate,
            rhythm_events={"spindle": spindle, "ripple": ripple, "phase": phase_report},
            replay_sequence=[item.tolist() for item in replay_states],
            replay_timestamps_ms=replay_times.tolist(),
            alignment=alignment,
            reinstatement=reinstatement,
            plasticity=plasticity,
            integration=integration,
            homeostasis=homeostasis,
            retrieval_support=retrieval,
            consolidation_state=consolidation,
            validation=validation,
            cortical_activity=cortical_activity.tolist(),
            biological=biological,
            logger=logger,
        )

    def _trace_similarity(self, trace: HippocampalTrace, cue: list[float] | tuple[float, ...]) -> float:
        """Compare supplied cue features with a trace without changing stored trace content."""
        from .functional import cosine_similarity
        return cosine_similarity(trace.features, cue)

    def snapshot(self) -> dict[str, Any]:
        """Return accumulated trace, replay, consolidation, and biological model state."""
        return {"opportunity_count": self.opportunity_count, "trace_count": len(self.traces),
                "replay_event_count": len(self.replay_history),
                "replay_counts": dict(self.replay_counts),
                "consolidation_indices": dict(self.consolidation_indices),
                "mean_alignment": (float(np.mean(self.coupling_history))
                                   if self.coupling_history else None),
                "mean_observed_performance_change": (
                    float(np.mean(self.performance_changes)) if self.performance_changes else None),
                "biological": self.network.snapshot()}
