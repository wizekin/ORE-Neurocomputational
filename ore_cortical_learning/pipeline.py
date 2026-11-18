"""Integrated 14-operation gradual cortical-learning pipeline with persistent encounter state."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
import uuid
from typing import Any, Iterable

import numpy as np

from .biology import CorticalPopulationModel
from .contracts import CorticalLearningConfig, CorticalLearningResult, EncounterInput
from .functional import (
    beta_uncertainty,
    compare_with_history,
    compose_encounter_vector,
    compute_prediction_error,
    conditional_reliability,
    coactivity_statistics,
    calculate_modulatory_gate,
    estimate_reliability_for_activity,
    mean_squared_error,
    relationship_observations,
    timing_eligibility,
)
from .registry import OPERATION_REGISTRY


class GradualCorticalLearningProcessor:
    """Learn recurring cortical regularities incrementally across a stream of encounters.

    Each call to ``process`` reuses recurrent weights, accumulated co-activity statistics,
    relationship reliability, context branches, and homeostatic state. A caller-marked held-out
    encounter is evaluated but never contributes to the learner's stored evidence or weights.
    """

    def __init__(
        self,
        config: CorticalLearningConfig | None = None,
        database_path: str | Path | None = None,
        repository_root: str | Path | None = None,
    ) -> None:
        """Initialize cortical population dynamics, long-run statistics, and optional SQLite logging."""
        self.config = config or CorticalLearningConfig()
        self.model = CorticalPopulationModel(self.config)
        units = self.config.cortical_units
        self.feature_mean = np.zeros(units, dtype=np.float64)
        self.feature_ema = np.zeros(units, dtype=np.float64)
        self.coactivity_ema = np.zeros((units, units), dtype=np.float64)
        self.covariance_ema = np.zeros((units, units), dtype=np.float64)
        self.relationship_success = np.zeros((units, units), dtype=np.float64)
        self.relationship_failure = np.zeros((units, units), dtype=np.float64)
        self.context_relationships: dict[str, dict[str, np.ndarray]] = {}
        self.context_prototypes: dict[str, np.ndarray] = {}
        self.context_encounter_counts: dict[str, int] = {}
        self.encounter_count = 0
        self.evaluation_count = 0
        self.history: list[dict[str, Any]] = []
        self.last_context_id: str | None = None
        self.database_path = Path(database_path).expanduser().resolve() if database_path else None
        self.repository_root = (
            Path(repository_root).expanduser().resolve()
            if repository_root else Path(__file__).resolve().parents[1]
        )

    def _compose_input(self, encounter: EncounterInput) -> np.ndarray:
        """Build the configured fixed-width input vector without inventing omitted channel values."""
        cfg = self.config
        return compose_encounter_vector(
            encounter.sensory_activity, encounter.action_activity, encounter.context,
            encounter.goal_activity, encounter.episode_trace,
            (cfg.sensory_size, cfg.action_size, cfg.context_size,
             cfg.goal_size, cfg.episode_trace_size),
        )

    def _relationship_state(self, context_id: str) -> tuple[np.ndarray, np.ndarray, str]:
        """Return global or context-specific relationship counts without merging distinct situations."""
        branch = self.context_relationships.get(context_id)
        if branch is None:
            return self.relationship_success, self.relationship_failure, "global_prior_for_unseen_context"
        return branch["success"], branch["failure"], "context_specific_branch"

    def _update_relationship_state(
        self, context_id: str, activity: np.ndarray,
    ) -> dict[str, Any]:
        """Accumulate global and context-specific conditional relationship evidence for training events."""
        successes, failures = relationship_observations(activity)
        self.relationship_success += successes
        self.relationship_failure += failures
        branch = self.context_relationships.get(context_id)
        if branch is None and len(self.context_relationships) < self.config.max_context_branches:
            branch = {"success": np.zeros_like(successes), "failure": np.zeros_like(failures)}
            self.context_relationships[context_id] = branch
        if branch is not None:
            branch["success"] += successes
            branch["failure"] += failures
        return {"positive_relation_observations": int(np.sum(successes)),
                "negative_relation_observations": int(np.sum(failures)),
                "global_observations": int(np.sum(self.relationship_success + self.relationship_failure)),
                "context_branch_recorded": branch is not None,
                "context_branch_count": len(self.context_relationships)}

    def _update_running_statistics(self, activity: np.ndarray) -> dict[str, Any]:
        """Update exponential feature, co-activity, and covariance summaries across encounters."""
        coactivity, covariance = coactivity_statistics(activity)
        decay = self.config.statistics_decay
        self.feature_ema = (1.0 - decay) * self.feature_ema + decay * activity
        self.coactivity_ema = (1.0 - decay) * self.coactivity_ema + decay * coactivity
        self.covariance_ema = (1.0 - decay) * self.covariance_ema + decay * covariance
        return {"feature_ema_mean": float(np.mean(self.feature_ema)),
                "coactivity_ema_mean": float(np.mean(self.coactivity_ema)),
                "covariance_ema_mean": float(np.mean(self.covariance_ema)),
                "coactivity_matrix_shape": list(self.coactivity_ema.shape)}

    def _update_context_prototype(self, context_id: str, activity: np.ndarray) -> dict[str, Any]:
        """Update an equal-weight context-specific activity mean for cross-encounter integration."""
        count = self.context_encounter_counts.get(context_id, 0)
        previous = self.context_prototypes.get(context_id)
        if previous is None:
            prototype = activity.copy()
        else:
            prototype = previous + (activity - previous) / float(count + 1)
        self.context_prototypes[context_id] = prototype
        self.context_encounter_counts[context_id] = count + 1
        self.feature_mean = self.feature_mean + (activity - self.feature_mean) / float(self.encounter_count + 1)
        return {"context_encounter_count": self.context_encounter_counts[context_id],
                "context_prototype_mean": float(np.mean(prototype)),
                "global_encounter_mean": float(np.mean(self.feature_mean)),
                "distinct_context_count": len(self.context_prototypes)}

    def _log_sqlite(self, trace: list[dict[str, Any]], elapsed_ms: float) -> dict[str, Any]:
        """Persist all operation annotations and the current encounter's execution records in SQLite."""
        if self.database_path is None:
            return {"enabled": False, "reason": "database_path_not_configured"}
        from ore_logger.recorder import SQLiteRunRecorder

        run_id = uuid.uuid4().hex
        recorder = SQLiteRunRecorder(
            database_path=self.database_path,
            repository_root=self.repository_root,
            run_id=run_id,
            component="ore_cortical_learning",
            callable_name="GradualCorticalLearningProcessor.process",
            started_at_utc=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            entry_kind="neurocomputational_component",
        )
        for index, (spec, row) in enumerate(zip(OPERATION_REGISTRY, trace), start=1):
            recorder.log_operation({
                "step": index, "operation": spec.operation_id, "name": spec.name,
                "status": row.get("status", "completed"), "phase": "gradual_cortical_learning",
                "explanation": spec.purpose, "mathematical_form": spec.mathematical_form,
                "biological_reference": spec.biological_reference,
                "evidence_level": spec.evidence_level, "inputs": list(spec.inputs),
                "outputs": list(spec.outputs), "functional_output": row.get("functional_output", {}),
                "biological_output": row.get("biological_output", {}),
            })
        return {"enabled": True, **recorder.finish("completed", elapsed_ms,
                                                     output_type="CorticalLearningResult")}

    def process(self, encounter: EncounterInput) -> CorticalLearningResult:
        """Process one encounter through representation, reliability, plasticity, and stability stages."""
        started = time.perf_counter()
        trace: list[dict[str, Any]] = []
        cfg = self.config
        training = not encounter.evaluation_only

        def record(
            operation_index: int, functional_output: dict[str, Any],
            biological_output: dict[str, Any] | None = None, status: str = "completed",
        ) -> None:
            """Append one inspectable operation row in canonical source order."""
            spec = OPERATION_REGISTRY[operation_index]
            trace.append({"operation_id": spec.operation_id, "operation": spec.name,
                          "status": status, "evidence_level": spec.evidence_level,
                          "functional_output": functional_output,
                          "biological_output": biological_output or {}})

        # CL01: retain all explicitly provided encounter channels and the context identity.
        input_vector = self._compose_input(encounter)
        record(0, {"encounter_id": encounter.encounter_id, "context_id": encounter.context_id,
                   "input_size": int(input_vector.size), "channel_dimensions": {
                       "sensory": len(encounter.sensory_activity), "action": len(encounter.action_activity),
                       "context": len(encounter.context), "goal": len(encounter.goal_activity),
                       "episode_trace": len(encounter.episode_trace)},
                   "episodic_trace_role": "optional_input_only_no_sleep_replay"})

        # CL02: recurrent cortical population transformation with lateral inhibition and prediction.
        activity, prediction, population_report = self.model.forward(input_vector)
        record(1, {"cortical_activity": activity.tolist(), "prediction": prediction.tolist()},
               population_report)

        # CL03: compare this activity to retained encounters without collapsing context identity.
        similarities = compare_with_history(activity, self.history, encounter.context_id,
                                            cfg.similarity_threshold)
        record(2, similarities)

        # CL04: collect recurring feature and relationship summaries across the encounter stream.
        if training:
            recurring_features = self._update_running_statistics(activity)
        else:
            coactivity, covariance = coactivity_statistics(activity)
            recurring_features = {"coactivity_mean_for_evaluation": float(np.mean(coactivity)),
                                  "covariance_mean_for_evaluation": float(np.mean(covariance)),
                                  "stored_statistics_updated": False}
        relation_successes, relation_failures = relationship_observations(activity)
        recurring_relationships = {
            "current_coactivity_pair_count": int(np.sum(relation_successes)),
            "current_noncoactivity_pair_count": int(np.sum(relation_failures)),
            "accumulated_coactivity_mean": float(np.mean(self.coactivity_ema)),
            "relationship_source": "current_population_pattern_and_prior_stream_statistics",
        }
        record(3, {"features": recurring_features, "relationships": recurring_relationships})

        # CL05: use only previously accumulated reliability, with context-specific evidence if available.
        success_state, failure_state, reliability_source = self._relationship_state(encounter.context_id)
        reliability_matrix, reliability_counts = conditional_reliability(
            success_state, failure_state, cfg.reliability_prior_alpha, cfg.reliability_prior_beta
        )
        reliability_report = estimate_reliability_for_activity(reliability_matrix, activity)
        reliability_report.update({"source": reliability_source,
                                   "mean_prior_observations": float(np.mean(reliability_counts)),
                                   "context_id": encounter.context_id})
        relationship_reliability = reliability_report
        record(4, relationship_reliability)

        # CL06: calculate target error only if downstream activity was explicitly observed.
        error_info = compute_prediction_error(encounter.observed_activity, prediction)
        prediction_error = error_info["error"]
        record(5, {"status": error_info["status"], "prediction_mse": error_info["mse"],
                   "prediction_error": prediction_error.tolist() if prediction_error is not None else None})

        # CL07: combine eligibility, timing, modulation, and relationship support.
        old_eligibility = self.model.eligibility_trace.copy()
        eligibility_trace, eligibility_biology = self.model.compute_eligibility(
            encounter.pre_post_interval_ms
        )
        gate_report = calculate_modulatory_gate(encounter.modulatory_state, cfg.modulatory_floor)
        timing_factor = timing_eligibility(encounter.pre_post_interval_ms, cfg.eligibility_tau_ms)
        gate_report["timing_factor"] = timing_factor
        gate_report["reliability_factor"] = relationship_reliability["mean_reliability"]
        record(6, {"mean_absolute_eligibility": float(np.mean(np.abs(eligibility_trace))),
                   "timing_factor": timing_factor, "modulatory_gate": gate_report["gate"]},
               eligibility_biology)

        # CL08: apply a small, gated, local update; held-out evaluation cannot change synapses.
        before_recurrent = self.model.recurrent_weights.copy()
        before_prediction = self.model.prediction_weights.copy()
        if training:
            synaptic_update = self.model.apply_plasticity(
                activity, prediction_error, float(gate_report["gate"]),
                float(relationship_reliability["mean_reliability"]), timing_factor,
            )
            update_status = "applied"
        else:
            self.model.eligibility_trace = old_eligibility
            synaptic_update = {"rule": "not_applied_to_held_out_encounter",
                               "prediction_update_applied": False, "update_count": self.model.update_count}
            update_status = "skipped_evaluation_only"
        record(7, synaptic_update, {"recurrent_weight_change": float(np.mean(np.abs(
            self.model.recurrent_weights - before_recurrent))),
            "prediction_weight_change": float(np.mean(np.abs(
                self.model.prediction_weights - before_prediction)))}, update_status)

        # CL09: expose update locality, clipping, importance protection, and deviations from reference weights.
        weighted_deviation = float(np.mean(self.model.recurrent_importance * np.abs(
            self.model.recurrent_weights - self.model.reference_recurrent_weights)))
        interference_control = {
            "recurrent_update_mean_absolute": float(np.mean(np.abs(
                self.model.recurrent_weights - before_recurrent))),
            "prediction_update_mean_absolute": float(np.mean(np.abs(
                self.model.prediction_weights - before_prediction))),
            "importance_weighted_reference_deviation": weighted_deviation,
            "max_absolute_recurrent_weight": float(np.max(np.abs(self.model.recurrent_weights))),
            "max_absolute_prediction_weight": float(np.max(np.abs(self.model.prediction_weights))),
            "bounded_local_updates": True,
            "training_update": training,
            "interpretation": "engineering interference-control diagnostic, not measured biological penalty",
        }
        record(8, interference_control)

        # CL10: integrate current examples into exact-count means and context-specific prototypes.
        if training:
            accumulated_statistics = self._update_context_prototype(encounter.context_id, activity)
        else:
            prototype = self.context_prototypes.get(encounter.context_id)
            accumulated_statistics = {"context_encounter_count": self.context_encounter_counts.get(encounter.context_id, 0),
                                      "prototype_similarity": (float(np.dot(activity, prototype) /
                                          max(np.linalg.norm(activity) * np.linalg.norm(prototype), 1e-12))
                                          if prototype is not None else None),
                                      "stored_statistics_updated": False}
        record(9, accumulated_statistics)

        # CL11: regulate average population responses after learning, not on held-out examples.
        if training:
            homeostasis = self.model.regulate_homeostasis(activity)
            homeostasis_status = "regulated"
        else:
            homeostasis = {"target_activity": cfg.target_activity,
                           "mean_recent_activity": float(np.mean(self.model.activity_mean)),
                           "stored_homeostasis_updated": False}
            homeostasis_status = "skipped_evaluation_only"
        record(10, homeostasis, {"mean_homeostatic_gain": float(np.mean(self.model.homeostatic_gain)),
                                 "mean_homeostatic_bias": float(np.mean(self.model.homeostatic_bias))},
               homeostasis_status)

        # CL12: distinguish declared held-out tests from ordinary in-stream training measurements.
        if encounter.evaluation_only and prediction_error is not None:
            generalization_evaluation = {
                "status": "held_out_evaluation_completed",
                "mean_squared_error": mean_squared_error(encounter.observed_activity or (), prediction),
                "held_out": True,
                "training_state_changed": False,
                "context_match": encounter.context_id in self.context_prototypes,
            }
            self.evaluation_count += 1
        elif encounter.evaluation_only:
            generalization_evaluation = {"status": "held_out_target_not_observed",
                                         "mean_squared_error": None, "held_out": True,
                                         "training_state_changed": False}
            self.evaluation_count += 1
        else:
            generalization_evaluation = {
                "status": "in_stream_monitor_not_generalization_proof",
                "mean_squared_error": error_info["mse"], "held_out": False,
                "training_state_changed": True,
            }
        record(11, generalization_evaluation)

        # CL13: record positive/negative relation evidence in global and context-specific branches.
        if training:
            relation_update = self._update_relationship_state(encounter.context_id, activity)
        else:
            relation_update = {"status": "not_updated_for_held_out_encounter",
                               "context_branch_count": len(self.context_relationships)}
        current_success, current_failure, current_source = self._relationship_state(encounter.context_id)
        alpha = cfg.reliability_prior_alpha + current_success
        beta = cfg.reliability_prior_beta + current_failure
        uncertainty = beta_uncertainty(alpha, beta)
        uncertainty.update({"context_id": encounter.context_id, "source": current_source,
                            "positive_observations": int(np.sum(current_success)),
                            "negative_observations": int(np.sum(current_failure)),
                            "relation_update": relation_update})
        record(12, uncertainty, {"context_specific_branch_available": current_source == "context_specific_branch"},
               "completed" if training else "skipped_evaluation_only")

        # CL14: retain the updated state for the next encounter, but do not train on held-out data.
        if training:
            self.model.finish_encounter()
            self.encounter_count += 1
            self.last_context_id = encounter.context_id
            self.history.append({"encounter_id": encounter.encounter_id,
                                 "context_id": encounter.context_id,
                                 "activity": activity.copy()})
            if len(self.history) > cfg.max_encounter_history:
                self.history.pop(0)
            next_state_status = "updated_and_ready_for_next_encounter"
        else:
            next_state_status = "held_out_state_not_added_to_training_history"
        readiness = {"status": next_state_status,
                     "training_encounter_count": self.encounter_count,
                     "evaluation_encounter_count": self.evaluation_count,
                     "history_size": len(self.history),
                     "context_branch_count": len(self.context_relationships),
                     "episodic_memory_replaced": False,
                     "sleep_replay_or_systems_consolidation": "not_implemented_by_point_02"}
        record(13, readiness, self.model.snapshot(),
               "completed" if training else "evaluation_only")

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger_metadata = self._log_sqlite(trace, elapsed_ms)
        biological = {
            "cortical_population": population_report,
            "plasticity_eligibility": eligibility_biology,
            "homeostasis": homeostasis,
            "network_snapshot": self.model.snapshot(),
            "scope": "biologically_motivated_population_surrogate_not_physiological_validation",
        }
        return CorticalLearningResult(
            encounter_id=encounter.encounter_id,
            context_id=encounter.context_id,
            evaluation_only=encounter.evaluation_only,
            cortical_activity=activity.tolist(),
            prediction=prediction.tolist(),
            observed_activity=list(encounter.observed_activity) if encounter.observed_activity is not None else None,
            prediction_error=prediction_error.tolist() if prediction_error is not None else None,
            prediction_mse=error_info["mse"],
            similarities=similarities,
            recurring_features=recurring_features,
            recurring_relationships=recurring_relationships,
            relationship_reliability=relationship_reliability,
            plasticity_eligibility={"trace_mean_absolute": float(np.mean(np.abs(eligibility_trace))),
                                    "timing_factor": timing_factor},
            modulatory_gate=gate_report,
            synaptic_update=synaptic_update,
            interference_control=interference_control,
            accumulated_statistics=accumulated_statistics,
            homeostasis=homeostasis,
            generalization_evaluation=generalization_evaluation,
            uncertainty=uncertainty,
            operation_trace=trace,
            biological=biological,
            metadata={"schema": "ore.gradual_cortical_learning.1.0",
                      "operation_count": len(trace), "configuration": cfg.to_dict(),
                      "input_vector_size": int(input_vector.size),
                      "sqlite_logger": logger_metadata},
        )

    def process_stream(self, encounters: Iterable[EncounterInput]) -> list[CorticalLearningResult]:
        """Process encounters in order while preserving the evolving cortical learner between calls."""
        return [self.process(encounter) for encounter in encounters]

    def snapshot(self) -> dict[str, Any]:
        """Return all persistent statistics, context branches, and cortical model parameters."""
        reliability, counts = conditional_reliability(
            self.relationship_success, self.relationship_failure,
            self.config.reliability_prior_alpha, self.config.reliability_prior_beta,
        )
        return {"schema": "ore.gradual_cortical_learning.state.1.0",
                "encounter_count": self.encounter_count,
                "evaluation_count": self.evaluation_count,
                "feature_mean": self.feature_mean.tolist(),
                "feature_ema": self.feature_ema.tolist(),
                "coactivity_ema": self.coactivity_ema.tolist(),
                "covariance_ema": self.covariance_ema.tolist(),
                "relationship_reliability_mean": float(np.mean(reliability)),
                "relationship_observation_mean": float(np.mean(counts)),
                "context_encounter_counts": dict(self.context_encounter_counts),
                "context_branch_count": len(self.context_relationships),
                "model": self.model.snapshot()}
