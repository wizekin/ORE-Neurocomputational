"""End-to-end 16-stage novelty-modulated hippocampal consolidation processor."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
from pathlib import Path
import time
import uuid
from typing import Any, Sequence

from .biology import BiologicalConsolidationNetwork
from .contracts import (
    ConsolidationConfig,
    ConsolidationRequest,
    ConsolidationResult,
    MemoryTrace,
)
from .functional import (
    aggregate_feature_support,
    clamp,
    classify_novelty,
    cosine_similarity,
    estimate_interference,
    retrieval_score,
    sigmoid,
    softmax_probabilities,
    stabilization_probability,
    tagged_resource_capture,
    temporal_eligibility,
)
from .registry import OPERATION_REGISTRY


class NoveltyModulatedHippocampalConsolidationProcessor:
    """Run post-learning novelty, synaptic tagging/capture, retrieval, interference, and sleep stages.

    The processor retains each episode's mutable model state across calls. Neuromodulator,
    receptor, synaptic, and cortical values remain explicit estimates. Sleep-related cortical
    integration is a parallel process and does not erase or replace hippocampal persistence.
    """

    def __init__(
        self,
        config: ConsolidationConfig | None = None,
        database_path: str | Path | None = None,
        repository_root: str | Path | None = None,
    ) -> None:
        """Initialize deterministic population models, trace stores, and optional SQLite logging."""
        self.config = config or ConsolidationConfig()
        self.network = BiologicalConsolidationNetwork(self.config)
        self.traces: dict[str, MemoryTrace] = {}
        self._processed_event_ids: dict[str, set[str]] = {}
        self.database_path = Path(database_path).expanduser().resolve() if database_path else None
        self.repository_root = (
            Path(repository_root).expanduser().resolve()
            if repository_root else Path(__file__).resolve().parents[1]
        )

    def add_trace(self, trace: MemoryTrace) -> dict[str, Any]:
        """Register an episode once and preserve its identity and existing synaptic state."""
        if len(trace.hippocampal_pattern) != self.config.input_size:
            raise ValueError("trace pattern width must match configured input_size")
        existing = self.traces.get(trace.trace_id)
        if existing is not None:
            if existing.hippocampal_pattern != trace.hippocampal_pattern:
                raise ValueError("trace_id is already registered with different episode content")
            return {"trace_id": trace.trace_id, "status": "already_registered"}
        self.traces[trace.trace_id] = deepcopy(trace)
        self._processed_event_ids[trace.trace_id] = set()
        return {"trace_id": trace.trace_id, "status": "registered"}

    def _make_operation(
        self,
        index: int,
        functional: dict[str, Any],
        biological: dict[str, Any] | None = None,
        status: str = "completed",
    ) -> dict[str, Any]:
        """Create one source-ordered operation row with its registry explanation and evidence label."""
        spec = OPERATION_REGISTRY[index]
        return {
            "step": index + 1,
            "operation_id": spec.operation_id,
            "operation": spec.name,
            "status": status,
            "explanation": spec.purpose,
            "evidence_level": spec.evidence_level,
            "mathematical_form": spec.mathematical_form,
            "functional_output": functional,
            "biological_output": biological or {},
        }

    def _salience(self, novelty: dict[str, Any], request: ConsolidationRequest, event: Any) -> float:
        """Combine novelty, goal relevance, moderate arousal, value, and stress into a bounded signal."""
        novelty_score = clamp(float(novelty.get("novelty_score", 0.5)))
        state = request.internal_state
        arousal_efficiency = clamp(1.0 - abs(state.arousal - 0.55) / 0.55)
        value = clamp(
            0.4 * state.motivational_value
            + 0.3 * state.reward_expectancy
            + 0.3 * event.reward_value
        )
        raw = (
            0.42 * novelty_score
            + 0.22 * state.goal_relevance
            + 0.14 * arousal_efficiency
            + 0.16 * value
            + 0.06 * event.sensory_intensity
            - 0.12 * state.stress
        )
        return clamp(raw)

    def _update_tags(
        self,
        trace: MemoryTrace,
        event: Any,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Decay local tags to the event time and return residual eligibility per synapse."""
        residual: dict[str, float] = {}
        for synapse, prior_tag in trace.synaptic_tags.items():
            elapsed = max(0.0, event.timestamp_ms - trace.tag_times_ms[synapse])
            value = prior_tag * temporal_eligibility(elapsed, self.config.tag_tau_ms)
            trace.synaptic_tags[synapse] = value
            trace.tag_times_ms[synapse] = event.timestamp_ms
            residual[synapse] = value
        return residual, dict(trace.synaptic_tags)

    def _update_synaptic_persistence(
        self, trace: MemoryTrace, captures: dict[str, float]
    ) -> dict[str, Any]:
        """Stabilize captured synaptic changes while retaining their original signed direction."""
        probabilities: dict[str, float] = {}
        for synapse, capture in captures.items():
            probability = stabilization_probability(
                capture, self.config.capture_threshold, self.config.capture_slope
            )
            probabilities[synapse] = probability
            old = trace.synaptic_persistence[synapse]
            new = (
                old
                + self.config.stabilization_rate * probability * (1.0 - old)
                - self.config.synaptic_decay_rate * (1.0 - probability) * old
            )
            trace.synaptic_persistence[synapse] = clamp(new)
            change = trace.synaptic_changes[synapse]
            magnitude = min(
                self.config.max_synaptic_change,
                abs(change) + self.config.stabilization_rate * probability * (1.0 - abs(change)),
            )
            trace.synaptic_changes[synapse] = math.copysign(magnitude, change if change != 0.0 else 1.0)
        return {
            "stabilization_probabilities": probabilities,
            "synaptic_persistence": dict(trace.synaptic_persistence),
            "signed_synaptic_changes": dict(trace.synaptic_changes),
        }

    def _update_ensemble(self, trace: MemoryTrace) -> dict[str, float]:
        """Pool synaptic lifetimes into a non-uniform hippocampal ensemble persistence estimate."""
        synapse_ids = list(trace.active_synapses)
        values = [trace.synaptic_persistence[key] for key in synapse_ids]
        indices = [trace.synapse_feature_indices[key] for key in synapse_ids]
        counts = [0] * len(trace.hippocampal_pattern)
        for index in indices:
            counts[index] += 1
        pooled = aggregate_feature_support(values, indices, len(trace.hippocampal_pattern))
        support = tuple(
            clamp(0.55 * old + 0.45 * new) if counts[i] else old
            for i, (old, new) in enumerate(zip(trace.pattern_support, pooled))
        )
        trace.pattern_support = support
        total_activity = sum(trace.active_synapses.values())
        weighted_persistence = (
            sum(trace.active_synapses[key] * trace.synaptic_persistence[key] for key in synapse_ids)
            / max(total_activity, 1e-12)
        )
        mean_support = sum(support) / len(support)
        old_persistence = trace.hippocampal_persistence
        trace.hippocampal_persistence = clamp(
            old_persistence * (1.0 - self.config.persistence_decay)
            + 0.22 * weighted_persistence * (1.0 - old_persistence)
        )
        current_pattern = tuple(
            value * factor for value, factor in zip(trace.hippocampal_pattern, support)
        )
        coherence = clamp((cosine_similarity(trace.hippocampal_pattern, current_pattern) + 1.0) / 2.0)
        trace.episodic_detail = clamp(
            0.92 * trace.episodic_detail + 0.08 * trace.hippocampal_persistence
        )
        return {
            "weighted_synaptic_persistence": weighted_persistence,
            "mean_feature_support": mean_support,
            "hippocampal_persistence_before": old_persistence,
            "hippocampal_persistence_after": trace.hippocampal_persistence,
            "pattern_coherence": coherence,
        }

    def _retrieval_report(
        self,
        trace: MemoryTrace,
        request: ConsolidationRequest,
        interference_risk: float,
    ) -> dict[str, Any]:
        """Estimate target expression against cue and context without treating it as trace existence."""
        cue = request.retrieval_cue
        cue_match = (
            clamp((cosine_similarity(trace.hippocampal_pattern, cue) + 1.0) / 2.0)
            if cue is not None else 0.5
        )
        cfg = self.config
        target_score = retrieval_score(
            trace.hippocampal_persistence, cue_match, request.context_match,
            interference_risk, cfg.retrieval_persistence_weight,
            cfg.retrieval_match_weight, cfg.retrieval_context_weight,
            cfg.retrieval_interference_weight,
        )
        candidates = [target_score]
        candidate_ids = [trace.trace_id]
        for competitor in request.competing_traces:
            match = clamp((
                cosine_similarity(
                    request.retrieval_cue or trace.hippocampal_pattern,
                    competitor.hippocampal_pattern,
                ) + 1.0
            ) / 2.0)
            score = retrieval_score(
                competitor.strength, match, request.context_match,
                0.0, cfg.retrieval_persistence_weight, cfg.retrieval_match_weight,
                cfg.retrieval_context_weight, 0.0,
            )
            candidates.append(score)
            candidate_ids.append(competitor.trace_id)
        probabilities = softmax_probabilities(candidates, cfg.retrieval_temperature)
        return {
            "target_trace_id": trace.trace_id,
            "cue_match": cue_match,
            "context_match": request.context_match,
            "target_retrieval_score": target_score,
            "candidate_trace_ids": candidate_ids,
            "candidate_scores": candidates,
            "candidate_probabilities": list(probabilities),
            "target_retrieval_probability": probabilities[0],
            "latent_trace_persistence": trace.hippocampal_persistence,
            "interference_risk_used": interference_risk,
        }

    def _log_sqlite(self, rows: list[dict[str, Any]], elapsed_ms: float) -> dict[str, Any]:
        """Persist the 16 operation annotations and outcomes in the reusable ORE SQLite logger."""
        if self.database_path is None:
            return {"enabled": False, "reason": "database_path_not_configured"}
        from ore_logger.recorder import SQLiteRunRecorder

        run_id = uuid.uuid4().hex
        started_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        recorder = SQLiteRunRecorder(
            database_path=self.database_path,
            repository_root=self.repository_root,
            run_id=run_id,
            component="ore_novelty_consolidation",
            callable_name="NoveltyModulatedHippocampalConsolidationProcessor.process",
            started_at_utc=started_at,
        )
        for row in rows:
            recorder.log_operation({
                "step": row["step"],
                "operation": row["operation"],
                "status": row["status"],
                "explanation": row["explanation"],
                "functional_output": row["functional_output"],
                "biological_output": row["biological_output"],
                "evidence_level": row["evidence_level"],
                "mathematical_form": row["mathematical_form"],
            })
        return {
            "enabled": True,
            **recorder.finish("completed", elapsed_ms, output_type="ConsolidationResult"),
        }

    def process(self, request: ConsolidationRequest) -> ConsolidationResult:
        """Process one post-learning interval through all 16 ordered novelty and consolidation stages."""
        started = time.perf_counter()
        if len(request.trace.hippocampal_pattern) != self.config.input_size:
            raise ValueError("trace pattern width must match configured input_size")
        trace_id = request.trace.trace_id
        if trace_id not in self.traces:
            self.add_trace(request.trace)
        trace = self.traces[trace_id]
        if any(len(event.features) != self.config.input_size for event in request.post_learning_events):
            raise ValueError("event feature width must match configured input_size")
        rows: list[dict[str, Any]] = []
        seen_events = self._processed_event_ids.setdefault(trace_id, set())
        pending_events = [
            event for event in sorted(request.post_learning_events, key=lambda item: item.timestamp_ms)
            if event.event_id not in seen_events
        ]
        novelty_rows: list[dict[str, Any]] = []
        salience_rows: list[float] = []
        pathway_rows: list[dict[str, Any]] = []
        neuromodulator_rows: list[dict[str, float]] = []
        gate_rows: list[dict[str, float]] = []
        resource_rows: list[dict[str, float]] = []
        capture_totals = {key: 0.0 for key in trace.active_synapses}
        capture_event_count = 0

        # NC01: select the episode-specific hippocampal trace as the target.
        rows.append(self._make_operation(0, {
            "trace_id": trace.trace_id,
            "encoding_time_ms": trace.encoding_time_ms,
            "pattern_width": len(trace.hippocampal_pattern),
            "eligible_synapse_count": len(trace.active_synapses),
            "existing_hippocampal_persistence": trace.hippocampal_persistence,
        }, {
            "represented_system": "hippocampal_formation",
            "interpretation": "Retained rate-population code and encoding-linked synapse identifiers.",
        }))

        # NC02: establish temporary tags on encoding-active synapses.
        # MemoryTrace initializes tags from encoding activity only when no explicit tag
        # map was supplied; explicit zeros remain ineligible at this stage.
        rows.append(self._make_operation(1, {
            "synaptic_tags": dict(trace.synaptic_tags),
            "tag_times_ms": dict(trace.tag_times_ms),
            "tagged_synapse_count": sum(value > 0.0 for value in trace.synaptic_tags.values()),
        }, {
            "mechanism": "transient_local_synaptic_eligibility",
            "not_a_literal_synaptic_label": True,
        }))

        # NC03: monitor new events, time since encoding, and temporal eligibility.
        timing_rows = []
        for event in pending_events:
            elapsed = event.timestamp_ms - trace.encoding_time_ms
            timing_rows.append({
                "event_id": event.event_id,
                "elapsed_since_encoding_ms": elapsed,
                "temporal_eligibility": temporal_eligibility(
                    elapsed, self.config.temporal_eligibility_tau_ms
                ),
            })
        rows.append(self._make_operation(2, {
            "new_event_count": len(pending_events),
            "time_weighted_events": timing_rows,
            "sleep_evidence_supplied": request.sleep_evidence is not None,
        }, {
            "post_encoding_monitoring": "event and internal-state estimates",
            "temporal_window_is_task_dependent": True,
        }, "completed" if pending_events else "completed_no_new_event"))

        # NC04: compare post-learning events with stored familiarity and predictions.
        for event in pending_events:
            profile = classify_novelty(
                event.features,
                request.internal_state.prior_knowledge_patterns,
                request.internal_state.predicted_features,
                self.config.novelty_threshold,
                self.config.common_novelty_similarity_threshold,
            )
            novelty_rows.append({"event_id": event.event_id, **profile})
        rows.append(self._make_operation(3, {
            "novelty_evidence": novelty_rows,
            "events_compared": len(novelty_rows),
        }, {
            "networks": ["hippocampal_formation", "medial_temporal_cortex", "prefrontal_cortex"],
            "equations_are_model_operations": True,
        }, "completed" if pending_events else "skipped_no_new_post_learning_event"))

        # NC05: estimate novelty category and salience separately from physical intensity.
        for event, profile in zip(pending_events, novelty_rows):
            salience_rows.append(self._salience(profile, request, event))
        salience_report = [
            {"event_id": event.event_id, "salience": salience}
            for event, salience in zip(pending_events, salience_rows)
        ]
        rows.append(self._make_operation(4, {
            "novelty_and_salience": [
                {**profile, "salience": salience}
                for profile, salience in zip(novelty_rows, salience_rows)
            ],
            "salience_components": salience_report,
        }, {
            "novelty_intensity_separated_from_relevance": True,
            "common_distinct_mapping_is_hypothesis": True,
        }, "completed" if pending_events else "skipped_no_new_post_learning_event"))

        # NC06-NC07: recruit VTA/LC ensembles, then estimate distinct DA and NE dynamics.
        for event, profile, salience, timing in zip(
            pending_events, novelty_rows, salience_rows, timing_rows
        ):
            elapsed = float(timing["elapsed_since_encoding_ms"])
            pathway = self.network.recruit_and_modulate(
                profile, salience, request.internal_state, event.reward_value, elapsed
            )
            pathway_rows.append({
                "event_id": event.event_id,
                **pathway["pathways"],
            })
            neuromodulator_rows.append({
                "event_id": event.event_id,
                **pathway["neuromodulators"],
            })
        rows.append(self._make_operation(5, {
            "pathway_estimates": pathway_rows,
            "event_count": len(pathway_rows),
        }, {
            "distinct_populations": ["VTA_rate_population", "LC_rate_population"],
            "lc_can_contribute_hippocampal_dopamine": True,
        }, "completed" if pending_events else "skipped_no_new_post_learning_event"))
        mean_dopamine = (
            sum(row["hippocampal_dopamine_estimate"] for row in neuromodulator_rows)
            / len(neuromodulator_rows) if neuromodulator_rows else 0.0
        )
        mean_noradrenaline = (
            sum(row["hippocampal_noradrenaline_estimate"] for row in neuromodulator_rows)
            / len(neuromodulator_rows) if neuromodulator_rows else 0.0
        )
        rows.append(self._make_operation(6, {
            "neuromodulator_estimates": neuromodulator_rows,
            "mean_hippocampal_dopamine": mean_dopamine,
            "mean_hippocampal_noradrenaline": mean_noradrenaline,
            "dopamine_sources": ["VTA_estimate", "LC_contribution_estimate"],
        }, {
            "clearance_and_temporal_decay_included": True,
            "dopamine_and_noradrenaline_kept_separate": True,
        }, "completed" if pending_events else "skipped_no_new_post_learning_event"))

        # NC08-NC09: engage receptor gates and produce a transient plasticity-resource pool.
        synaptic_activity = list(trace.active_synapses.values())
        for event, neuromodulators, timing in zip(
            pending_events, neuromodulator_rows, timing_rows
        ):
            state = self.network.receptor_and_resource_state(
                neuromodulators["hippocampal_dopamine_estimate"],
                neuromodulators["hippocampal_noradrenaline_estimate"],
                synaptic_activity,
                float(timing["elapsed_since_encoding_ms"]),
                float(timing["temporal_eligibility"]),
            )
            gate_rows.append({"event_id": event.event_id, **state["gates"]})
            resource_rows.append({"event_id": event.event_id, **state["resources"]})
        rows.append(self._make_operation(7, {
            "receptor_gate_estimates": gate_rows,
            "mean_d1_d5_gate": (
                sum(item["d1_d5_plasticity_gate"] for item in gate_rows) / len(gate_rows)
                if gate_rows else 0.0
            ),
            "mean_adrenergic_gate": (
                sum(item["adrenergic_plasticity_gate"] for item in gate_rows) / len(gate_rows)
                if gate_rows else 0.0
            ),
        }, {
            "receptor_occupancy_model": "Hill-type approximation",
            "local_activity_coincidence_required": True,
        }, "completed" if pending_events else "skipped_no_new_post_learning_event"))
        rows.append(self._make_operation(8, {
            "resource_estimates": resource_rows,
            "mean_plasticity_resource_pool": (
                sum(item["plasticity_resource_pool"] for item in resource_rows) / len(resource_rows)
                if resource_rows else 0.0
            ),
            "resource_is_transient": True,
        }, {
            "candidate_pathways": ["cAMP_PKA", "ERK_MAPK", "CREB", "calcium_dependent", "protein_synthesis"],
            "not_one_fixed_linear_cascade": True,
        }, "completed" if pending_events else "skipped_no_new_post_learning_event"))

        # NC10: capture resources only where a residual learning tag and local compatibility remain.
        if pending_events:
            for event, resources, timing in zip(pending_events, resource_rows, timing_rows):
                tags, _ = self._update_tags(trace, event)
                resource_pool = float(resources["plasticity_resource_pool"])
                capture_event_count += 1
                for synapse, activity in trace.active_synapses.items():
                    local_resource = clamp(resource_pool * (0.5 + 0.5 * activity))
                    compatibility = 0.5 + 0.5 * activity
                    capture = tagged_resource_capture(
                        tags[synapse], local_resource, compatibility
                    )
                    capture_totals[synapse] += capture
                seen_events.add(event.event_id)
        mean_capture = {
            key: clamp(value / max(1, capture_event_count))
            for key, value in capture_totals.items()
        }
        rows.append(self._make_operation(9, {
            "capture_scores": mean_capture,
            "captured_synapse_count": sum(value > 0.0 for value in mean_capture.values()),
            "tagged_resource_coincidence_events": capture_event_count,
        }, {
            "specificity_rule": "learning-active tag AND local resource AND compatibility",
            "untagged_synapses_receive_no_tag_capture": True,
        }, "completed" if pending_events else "skipped_no_new_post_learning_event"))

        # NC11: convert sufficient capture into longer-lived signed synaptic modifications.
        if pending_events:
            stabilization = self._update_synaptic_persistence(trace, mean_capture)
        else:
            stabilization = {
                "stabilization_probabilities": {key: 0.0 for key in trace.active_synapses},
                "synaptic_persistence": dict(trace.synaptic_persistence),
                "signed_synaptic_changes": dict(trace.synaptic_changes),
            }
        rows.append(self._make_operation(10, stabilization, {
            "plasticity_direction_preserved": True,
            "unsupported_changes_undergo_decay": True,
        }, "completed" if pending_events else "skipped_no_new_post_learning_event"))

        # NC12: pool synaptic state into a distributed hippocampal ensemble state.
        if pending_events:
            ensemble = self._update_ensemble(trace)
        else:
            ensemble = {
                "weighted_synaptic_persistence": sum(trace.synaptic_persistence.values())
                / len(trace.synaptic_persistence),
                "mean_feature_support": sum(trace.pattern_support) / len(trace.pattern_support),
                "hippocampal_persistence_before": trace.hippocampal_persistence,
                "hippocampal_persistence_after": trace.hippocampal_persistence,
                "pattern_coherence": 1.0,
            }
        rows.append(self._make_operation(11, ensemble, {
            "representation": "distributed_hippocampal_ensemble",
            "non_uniform_synapse_and_feature_stabilization": True,
        }, "completed"))

        # NC13: compute retrieval expression before and after the new interference estimate.
        preliminary_retrieval = self._retrieval_report(
            trace, request, trace.interference_risk
        )
        rows.append(self._make_operation(12, {
            **preliminary_retrieval,
            "expression_is_separate_from_latent_trace_persistence": True,
        }, {
            "retrieval_networks": ["hippocampal_cue_matching", "cortical_context"],
            "competition_rule": "temperature-scaled softmax surrogate",
        }, "completed" if request.retrieval_cue is not None else "completed_neutral_cue_estimate"))

        # NC14: update competition risk without deleting the target or competitor representations.
        interference = estimate_interference(
            trace.hippocampal_pattern,
            [
                (item.hippocampal_pattern, item.strength, item.conflict)
                for item in request.competing_traces
            ],
            self.config.interference_sensitivity,
        )
        trace.interference_risk = float(interference["risk"])
        final_retrieval = self._retrieval_report(trace, request, trace.interference_risk)
        interference["latent_trace_persistence_unchanged_by_competition"] = (
            trace.hippocampal_persistence == preliminary_retrieval["latent_trace_persistence"]
        )
        rows.append(self._make_operation(13, {
            **interference,
            "retrieval_probability_after_interference": final_retrieval["target_retrieval_probability"],
            "trace_store_count": 1 + len(request.competing_traces),
        }, {
            "trace_weakening_separated_from_retrieval_masking": True,
            "both_target_and_competing_traces_retained": True,
        }, "completed" if request.competing_traces else "completed_no_competitors"))

        # NC15: use supplied sleep/replay evidence to update a distinct cortical integration state.
        if request.sleep_evidence is None:
            sleep_report: dict[str, Any] = {
                "status": "not_supplied",
                "cortical_integration_before": trace.cortical_integration,
                "cortical_integration_after": trace.cortical_integration,
                "cortical_integration_delta": 0.0,
                "hippocampal_persistence_unchanged_by_sleep_stage": True,
            }
            sleep_status = "skipped_no_sleep_evidence"
        else:
            sleep_evidence = request.sleep_evidence
            replay_estimate = clamp(
                sleep_evidence.replay_strength
                * (0.5 + 0.5 * trace.hippocampal_persistence)
            )
            sleep_report = self.network.integrate_sleep(
                trace.cortical_integration, sleep_evidence, replay_estimate
            )
            trace.cortical_integration = float(sleep_report["cortical_integration_after"])
            trace.semantic_structure = clamp(
                trace.semantic_structure
                + 0.20 * float(sleep_report["cortical_integration_delta"])
                * sleep_evidence.prior_knowledge_support
            )
            sleep_report.update({
                "sleep_state": sleep_evidence.sleep_state,
                "replay_estimate_used": replay_estimate,
                "hippocampal_persistence_unchanged_by_sleep_stage": True,
            })
            sleep_status = (
                "completed_with_sleep_or_rest_evidence"
                if float(sleep_report["sleep_state_gate"]) > 0.0
                else "completed_no_eligible_sleep_state"
            )
        rows.append(self._make_operation(14, sleep_report, {
            "systems_consolidation_track": "parallel_to_novelty_related_cellular_consolidation",
            "rhythm_sources_are_supplied_evidence": True,
        }, sleep_status))

        # NC16: assemble the multi-dimensional trace and preserve event-level provenance.
        for event, profile, salience, pathway, neuromodulators, resources in zip(
            pending_events, novelty_rows, salience_rows, pathway_rows,
            neuromodulator_rows, resource_rows,
        ):
            trace.provenance.append({
                "event_id": event.event_id,
                "event_timestamp_ms": event.timestamp_ms,
                "elapsed_since_encoding_ms": event.timestamp_ms - trace.encoding_time_ms,
                "novelty_type_estimate": profile["novelty_type"],
                "novelty_score_estimate": profile["novelty_score"],
                "novelty_classification_confident": profile["classification_confident"],
                "salience_estimate": salience,
                "vta_activity_estimate": pathway["vta_mean_activity"],
                "lc_activity_estimate": pathway["lc_mean_activity"],
                "hippocampal_dopamine_estimate": neuromodulators["hippocampal_dopamine_estimate"],
                "hippocampal_noradrenaline_estimate": neuromodulators["hippocampal_noradrenaline_estimate"],
                "plasticity_resource_pool_estimate": resources["plasticity_resource_pool"],
                "capture_scores": dict(mean_capture),
                "uncertainty": (
                    "Latent neuromodulator and molecular states are model estimates, not direct observations."
                ),
            })
        multidimensional = {
            "hippocampal_trace_persistence": trace.hippocampal_persistence,
            "episodic_detail": trace.episodic_detail,
            "cortical_integration": trace.cortical_integration,
            "semantic_structure": trace.semantic_structure,
            "interference_risk": trace.interference_risk,
            "retrieval_expression_probability": final_retrieval["target_retrieval_probability"],
            "pattern_support": list(trace.pattern_support),
            "dimensions_are_not_collapsed": True,
        }
        rows.append(self._make_operation(15, {
            "memory_state": multidimensional,
            "provenance_record_count": len(trace.provenance),
            "uncertainty_preserved": True,
        }, {
            "hippocampal_cellular_and_cortical_systems_tracks_coexist": True,
            "event_history_is_append_only": True,
        }, "completed"))

        self.traces[trace_id] = trace
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger = self._log_sqlite(rows, elapsed_ms)
        return ConsolidationResult(
            trace_id=trace.trace_id,
            operation_trace=rows,
            final_trace=trace.to_dict(),
            novelty={
                "events": novelty_rows,
                "salience": salience_rows,
                "novel_event_count": sum(
                    item["novelty_type"] in {"common", "distinct", "unclassified_novelty"}
                    for item in novelty_rows
                ),
            },
            neuromodulation={
                "events": neuromodulator_rows,
                "mean_dopamine": mean_dopamine,
                "mean_noradrenaline": mean_noradrenaline,
                "pathways": pathway_rows,
            },
            synaptic_capture={
                "capture_scores": mean_capture,
                "stabilization": stabilization,
                "ensemble": ensemble,
            },
            retrieval=final_retrieval,
            interference=interference,
            sleep=sleep_report,
            multidimensional_state=multidimensional,
            biological=self.network.snapshot(),
            logger=logger,
        )

    def snapshot(self) -> dict[str, Any]:
        """Return retained traces, processed events, and the current surrogate population state."""
        return {
            "trace_count": len(self.traces),
            "trace_ids": sorted(self.traces),
            "processed_event_counts": {
                key: len(value) for key, value in self._processed_event_ids.items()
            },
            "biological_network": self.network.snapshot(),
        }
