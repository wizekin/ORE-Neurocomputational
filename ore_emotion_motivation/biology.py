"""Distributed biological-function surrogates for Point 13 emotion and memory modulation.

The components are functional models of interacting systems, not anatomical replicas. NE,
glucocorticoid, dopamine, and neural-gain variables are abstract model states, not measured
concentrations or claims of literal neuronal arithmetic.
"""
from __future__ import annotations

from typing import Any, Mapping

from .contracts import (
    AffectiveAppraisal, EmotionLearningConfig, LearningExperience,
    LearningState, MemoryTrace, RewardFeedback,
)
from .functional import (
    allocate_effort, consolidate_memory_trace, encode_memory_trace,
    estimate_affective_appraisal, estimate_control_capacity,
    estimate_motivational_value, estimate_neuromodulatory_gain,
    update_reward_values,
)


class AmygdalaAppraisalMemoryNetwork:
    """Model appraisal of event significance and BLA-like memory modulation."""

    def __init__(self) -> None:
        """Initialize appraisal and modulatory activity counters."""
        self.appraisal_count = 0
        self.modulation_count = 0
        self.last_significance = 0.0
        self.last_modulation = 0.0

    def appraise(self, experience: LearningExperience) -> AffectiveAppraisal:
        """Estimate valence, arousal, and significance from event and goal-related cues."""
        self.appraisal_count += 1
        appraisal = estimate_affective_appraisal(experience)
        self.last_significance = appraisal.significance
        return appraisal

    def modulate_memory(self, appraisal: AffectiveAppraisal, feature_salience: float) -> float:
        """Return a bounded memory-modulation proxy for arousal and salient-event significance."""
        self.modulation_count += 1
        value = appraisal.arousal * appraisal.significance * max(0.0, min(1.0, feature_salience))
        self.last_modulation = max(0.0, min(1.0, value))
        return self.last_modulation

    def snapshot(self) -> dict[str, Any]:
        """Expose appraisal and memory-modulation counters without implying an amygdala memory store."""
        return {
            "model": "amygdala_appraisal_and_memory_modulation_surrogate",
            "appraisal_count": self.appraisal_count,
            "modulation_count": self.modulation_count,
            "last_significance": self.last_significance,
            "last_modulation": self.last_modulation,
            "scope": "Appraisal and modulation, not storage of the entire emotional episode.",
        }


class LocusCoeruleusNorepinephrineNetwork:
    """Model a fast arousal-linked noradrenergic adaptive-gain proxy."""

    def __init__(self) -> None:
        """Initialize LC activity and gain counters."""
        self.update_count = 0
        self.last_gain = 0.5

    def update(self, state: LearningState, config: EmotionLearningConfig) -> float:
        """Estimate state-dependent gain from the fast arousal and LC-NE proxy state."""
        self.update_count += 1
        self.last_gain = estimate_neuromodulatory_gain(state.arousal.to_dict(), config)
        return self.last_gain

    def snapshot(self) -> dict[str, Any]:
        """Return fast gain-proxy diagnostics with a receptor/dose dependence caution."""
        return {
            "model": "locus_coeruleus_norepinephrine_gain_surrogate",
            "update_count": self.update_count,
            "last_gain": self.last_gain,
            "scope": "Gain proxy only; not norepinephrine concentration or a universal activation multiplier.",
        }


class HypothalamicPituitaryAdrenalNetwork:
    """Model a slower HPA/glucocorticoid load separately from fast LC-NE dynamics."""

    def __init__(self, retention: float = 0.97) -> None:
        """Initialize slow stress retention and update diagnostics."""
        if not 0.0 <= retention < 1.0:
            raise ValueError("retention must be in [0, 1)")
        self.retention = float(retention)
        self.update_count = 0
        self.last_load = 0.0

    def update(self, state: LearningState, appraisal: AffectiveAppraisal) -> float:
        """Accumulate a slow threat/arousal load rather than copying the fast NE state."""
        self.update_count += 1
        drive = min(1.0, max(0.0, 0.75 * appraisal.threat + 0.25 * appraisal.arousal))
        self.last_load = self.retention * state.arousal.glucocorticoid_load + (1.0 - self.retention) * drive
        state.arousal.glucocorticoid_load = self.last_load
        return self.last_load

    def snapshot(self) -> dict[str, Any]:
        """Return slow endocrine-load proxy state with a timescale limitation."""
        return {
            "model": "HPA_glucocorticoid_slow_load_surrogate",
            "update_count": self.update_count,
            "last_load": self.last_load,
            "scope": "Slow load proxy; not cortisol concentration or a physiological endocrine simulation.",
        }


class DopamineStriatalRewardNetwork:
    """Model reward valuation and temporal-difference learning for state/action outcomes."""

    def __init__(self) -> None:
        """Initialize reward update counters and last prediction error."""
        self.update_count = 0
        self.last_prediction_error = 0.0
        self.last_values: dict[str, float] = {}

    def update(
        self, feedback: RewardFeedback | None,
        state: LearningState, config: EmotionLearningConfig,
    ) -> tuple[float, dict[str, float]]:
        """Update learned values from observed reward feedback using a TD-style rule."""
        if feedback is not None:
            self.update_count += 1
        delta, values = update_reward_values(feedback, state, config)
        self.last_prediction_error = delta
        self.last_values = dict(values)
        return delta, values

    def snapshot(self) -> dict[str, Any]:
        """Expose value-learning diagnostics without reducing dopamine to pleasure."""
        return {
            "model": "dopamine_striatal_reward_learning_surrogate",
            "update_count": self.update_count,
            "last_prediction_error": self.last_prediction_error,
            "last_values": dict(self.last_values),
            "scope": "TD-inspired reward-learning abstraction; dopamine populations are heterogeneous.",
        }


class HippocampalEpisodicEncodingNetwork:
    """Model event-element binding, episodic trace formation, and later consolidation."""

    def __init__(self) -> None:
        """Initialize episodic encoding and consolidation counters."""
        self.encoding_count = 0
        self.consolidation_count = 0
        self.last_event_id = ""

    def encode(
        self, experience: LearningExperience, attention: Any,
        appraisal: AffectiveAppraisal, state: LearningState,
        config: EmotionLearningConfig,
    ) -> MemoryTrace:
        """Bind attended features and relations into a selectively weighted episodic trace."""
        self.encoding_count += 1
        trace = encode_memory_trace(experience, attention, appraisal, state, config)
        self.last_event_id = trace.event_id
        return trace

    def consolidate(
        self, trace: MemoryTrace, elapsed_hours: float,
        replay_support: float, config: EmotionLearningConfig,
    ) -> MemoryTrace:
        """Apply a delayed state-dependent consolidation update to an existing trace."""
        self.consolidation_count += 1
        return consolidate_memory_trace(trace, elapsed_hours, replay_support, config)

    def snapshot(self) -> dict[str, Any]:
        """Return episodic trace and consolidation activity counts."""
        return {
            "model": "hippocampal_episodic_binding_surrogate",
            "encoding_count": self.encoding_count,
            "consolidation_count": self.consolidation_count,
            "last_event_id": self.last_event_id,
        }


class PrefrontalGoalControlNetwork:
    """Model task-goal maintenance, attentional control, and effort allocation."""

    def __init__(self) -> None:
        """Initialize task-control diagnostics."""
        self.goal_update_count = 0
        self.effort_decision_count = 0
        self.active_goal = ""

    def maintain_goal(self, goal: str, state: LearningState) -> float:
        """Maintain the experience's explicit goal and return its current strength."""
        self.goal_update_count += 1
        if goal:
            self.active_goal = goal
            state.motivation.goal = goal
        return state.motivation.goal_strength

    def allocate_effort(
        self, experience: LearningExperience, state: LearningState,
        expected_value: float, config: EmotionLearningConfig,
    ) -> Any:
        """Choose an effort level using expected progress, capacity, and cost."""
        self.effort_decision_count += 1
        capacity = estimate_control_capacity(
            state.arousal.to_dict(), state.arousal.fatigue,
            experience.task_demands, config,
        )
        return allocate_effort(experience, state, capacity, expected_value, config)

    def snapshot(self) -> dict[str, Any]:
        """Return current goal and task-control operation counters."""
        return {
            "model": "prefrontal_goal_and_control_surrogate",
            "goal_update_count": self.goal_update_count,
            "effort_decision_count": self.effort_decision_count,
            "active_goal": self.active_goal,
        }


class CinguloOpercularEffortMonitoringNetwork:
    """Model task progress, conflict, and the need to persist or change strategy."""

    def __init__(self) -> None:
        """Initialize monitoring counters and the last effort recommendation."""
        self.monitor_count = 0
        self.last_recommendation = "not_evaluated"

    def monitor(self, effort_decision: Any, prediction_error: float) -> dict[str, Any]:
        """Combine progress and outcome mismatch into a task-monitoring record."""
        self.monitor_count += 1
        self.last_recommendation = effort_decision.recommended_strategy
        return {
            "expected_progress": effort_decision.expected_progress,
            "net_utility": effort_decision.net_utility,
            "prediction_error": float(prediction_error),
            "recommended_strategy": self.last_recommendation,
            "continue_task": effort_decision.continue_task,
        }

    def snapshot(self) -> dict[str, Any]:
        """Expose monitoring activity and last strategy recommendation."""
        return {
            "model": "cingulo_opercular_effort_monitoring_surrogate",
            "monitor_count": self.monitor_count,
            "last_recommendation": self.last_recommendation,
        }


class DistributedCorticalElaborationNetwork:
    """Model semantic association and elaborative linking of new and prior information."""

    def __init__(self) -> None:
        """Initialize cortical association and elaboration counters."""
        self.elaboration_count = 0
        self.last_link_count = 0

    def elaborate(
        self, experience: LearningExperience, trace: MemoryTrace,
        state: LearningState, effort_decision: Any,
    ) -> MemoryTrace:
        """Create relevant retrieval links when expected benefit justifies elaborative effort."""
        from .functional import elaborate_processing

        self.elaboration_count += 1
        enriched = elaborate_processing(experience, trace, state, effort_decision)
        self.last_link_count = len(enriched.elaboration_links)
        return enriched

    def snapshot(self) -> dict[str, Any]:
        """Return association and elaboration activity."""
        return {
            "model": "distributed_cortical_semantic_elaboration_surrogate",
            "elaboration_count": self.elaboration_count,
            "last_link_count": self.last_link_count,
        }


class BiologicalEmotionNetwork:
    """Coordinate affective appraisal, neuromodulation, reward, control, and memory surrogates."""

    def __init__(self) -> None:
        """Initialize the interacting biological-function surrogates."""
        self.amygdala = AmygdalaAppraisalMemoryNetwork()
        self.locus_coeruleus = LocusCoeruleusNorepinephrineNetwork()
        self.hpa_axis = HypothalamicPituitaryAdrenalNetwork()
        self.dopamine_striatal = DopamineStriatalRewardNetwork()
        self.hippocampus = HippocampalEpisodicEncodingNetwork()
        self.prefrontal = PrefrontalGoalControlNetwork()
        self.cingulo_opercular = CinguloOpercularEffortMonitoringNetwork()
        self.cortical_elaboration = DistributedCorticalElaborationNetwork()

    def snapshot(self) -> dict[str, Any]:
        """Return each subsystem's proxy state and the distributed-network scope limitation."""
        return {
            "amygdala": self.amygdala.snapshot(),
            "locus_coeruleus": self.locus_coeruleus.snapshot(),
            "hpa_axis": self.hpa_axis.snapshot(),
            "dopamine_striatal": self.dopamine_striatal.snapshot(),
            "hippocampus": self.hippocampus.snapshot(),
            "prefrontal": self.prefrontal.snapshot(),
            "cingulo_opercular": self.cingulo_opercular.snapshot(),
            "cortical_elaboration": self.cortical_elaboration.snapshot(),
            "scientific_scope": (
                "Interacting functional surrogates, not anatomical replicas or direct measurements "
                "of neurotransmitters, hormones, or neural firing."
            ),
        }
