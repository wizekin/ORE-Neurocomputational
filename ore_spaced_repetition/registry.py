"""Canonical 12-operation registry for Point 03 spaced repetition."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one source-aligned operation and its mathematical/biological scope."""

    operation_id: str
    source_step: int
    name: str
    purpose: str
    mathematical_form: str
    biological_reference: str
    evidence_level: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return this operation specification as a plain dictionary."""
        return asdict(self)


_ROWS = (
    (1, "Detect and register a learning episode",
     "Segment one learning target, retain its essential components and relations, detect duplicates, and establish a stable item identity.",
     "sim(x,m)=cos(x,m); match=argmax_m sim(x,m); X={x_1,...,x_N}",
     "Perceptual/language systems, attention, working memory, hippocampal binding, and cortical semantic systems contribute to encoding.",
     "learning_and_memory_encoding_supported; symbolic item record and similarity test are software abstractions",
     ("learning_material", "source", "context"), ("memory_item", "item_identity", "source_context")),
    (2, "Estimate initial encoding quality and uncertainty",
     "Estimate attention, comprehension, relational organization, and meaningful practice without pretending immediate success proves durable retention.",
     "E=w_a Atn+w_c Compr+w_r Rel+w_p Prac; S_0=S_min+E(S_max−S_min)",
     "Attention and elaborative encoding influence distributed neural activation and plasticity; the weighted score is a model variable.",
     "encoding_quality_factors_supported; scalar quality/stability initialization is computational",
     ("memory_item", "encoding_measurements"), ("encoding_quality", "encoding_uncertainty", "initial_stability")),
    (3, "Register temporal origin and create a meaningful delay",
     "Record the last meaningful learning event and choose a nonzero initial interval constrained by difficulty and retention horizon.",
     "Δt=t_now−t_last; I_0=clip(k·h·g(d,E),I_min,I_max)",
     "Temporal separation can change the effectiveness of repeated learning; the explicit calendar and due timestamp are external scheduler constructs.",
     "spacing_effect_supported; interval-selection equation is an empirical scheduler hypothesis",
     ("encoding_quality", "difficulty", "target_retention_horizon", "current_time"), ("last_learning_time", "initial_interval", "next_due_time")),
    (4, "Estimate accessibility after elapsed time",
     "Estimate likely recall accessibility and uncertainty from time since learning, stability, and the item's review history.",
     "A(Δt)=A_0 exp(−Δt/τ) or A_0(1+αΔt)^(−β); P_recall=σ((A−θ)/σ_r)",
     "Access depends on distributed representations, cue availability, interference, and hippocampal/cortical reinstatement; one decay curve is only a behavioral approximation.",
     "forgetting_and_accessibility_changes_supported; decay and probability functions require calibration",
     ("stability", "elapsed_time", "review_history"), ("accessibility", "predicted_recall", "uncertainty")),
    (5, "Decide whether review is due and appropriately difficult",
     "Choose review, defer, relearn-with-cue, or urgent review based on predicted recall, its target band, and the retention deadline.",
     "Action∈{REVIEW,DEFER,RELEARN_WITH_CUE,URGENT_REVIEW}; Δt_target=−S ln(p*)",
     "Retrieval effort and task control are cognitive processes; the numeric difficulty band and external timing policy are scheduler decisions.",
     "retrieval_difficulty_is_task_dependent; thresholds are tunable engineering assumptions",
     ("predicted_recall", "target_band", "deadline", "next_due_time"), ("review_decision", "cue_policy", "deadline_margin")),
    (6, "Attempt retrieval before re-exposure",
     "Capture a free, cued, procedural, or explanatory attempt before revealing the answer, and keep recall separate from recognition.",
     "Q=w_1 Accuracy+w_2 Completeness+w_3 Relation+w_4 Fluency; q∈{failure,partial,success}",
     "A cue can reactivate a partial representation through hippocampal pattern completion and distributed cortical reinstatement.",
     "retrieval_and_reactivation_supported; rubric weights are task-dependent software parameters",
     ("memory_item", "review_cue", "response"), ("retrieval_result", "effort", "cue_condition")),
    (7, "Correct incomplete or inaccurate retrieval",
     "Compare the first response to the target, identify missing/incorrect relations, provide correction, and record corrected relearning separately from unaided success.",
     "Loss=λ_1 L_missing+λ_2 L_incorrect+λ_3 L_relation; x_new=x_old+η·CorrectionSignal",
     "Feedback and error-sensitive learning can alter later encoding; reconsolidation is not presumed to occur after every recall.",
     "corrective_learning_supported; loss and update rule are computational",
     ("first_response", "correct_representation", "feedback"), ("corrected_representation", "correction_record", "event_type")),
    (8, "Update the modeled memory state",
     "Revise stability and uncertainty according to independent retrieval success, partial performance, failure, and corrected relearning.",
     "S_new=clip(S_old+η·G(q,e,feedback),S_min,S_max); p~Beta(a,b)",
     "Reactivation and learning can coincide with distributed synaptic plasticity; a single stability scalar is a coarse model.",
     "activity_dependent_learning_supported; scalar update and Beta model summarize outcomes",
     ("review_outcome", "effort", "feedback", "prior_stability"), ("updated_stability", "updated_uncertainty", "event_history")),
    (9, "Choose the next interval adaptively",
     "Set a new review interval from estimated stability, target recall probability, observed performance, difficulty, uncertainty, and remaining horizon.",
     "I_next=clip(−S ln(p*),I_min,I_max); I_next≤ρ·h_remaining",
     "People can adapt study behavior to prior success and difficulty, but the calendar scheduler is an external tool.",
     "adaptive_review_scheduling_is_engineering_policy; optimal intervals are learner/task-specific",
     ("updated_stability", "review_history", "target_horizon", "uncertainty"), ("next_interval", "next_due_time", "schedule_reason")),
    (10, "Repeat reviews and monitor maintenance criteria",
     "Retain an item-specific timeline and mark knowledge only provisionally stable after repeated independent success across meaningful delays.",
     "L_i={(t_1,q_1),...,(t_n,q_n)}; STOP=I[Q≥θ_Q and delay≥h and independent]",
     "Repeated learning may stabilize and reorganize distributed representations; no single test proves that future forgetting is impossible.",
     "distributed_practice_and_delayed_retention_supported; stop rule is operational",
     ("review_history", "retention_horizon", "accuracy_threshold"), ("maintenance_status", "retention_trend", "future_due_policy")),
    (11, "Represent offline consolidation separately and cautiously",
     "Allow optional sleep/offline observations to influence a model only when evidence is supplied; do not apply a guaranteed memory-strength multiplier.",
     "S(t+Δt)=S(t)+C(Δt,state,trace)−D(Δt,interference)",
     "Hippocampal replay and coordinated hippocampal, thalamic, and cortical activity during sleep are candidate consolidation mechanisms.",
     "sleep_associated_consolidation_supported; modelled effect size remains uncertain and configurable",
     ("offline_context", "interference", "trace_state"), ("consolidation_record", "optional_stability_adjustment")),
    (12, "Guard against familiarity and calibrate predictions",
     "Separate free recall, cued recall, and recognition; compare predicted with observed success and increase uncertainty when estimates are inconsistent.",
     "P_total=w_r P_recall+w_c P_cued+w_n P_recognition; ε=observed−predicted; Brier=mean((p−y)^2)",
     "Familiarity and recollection can be dissociable; behavioral calibration is not a direct neural-state measurement.",
     "familiarity_recollection_distinction_supported; calibration metrics are external behavioral tools",
     ("predicted_recall", "observed_outcome", "response_mode"), ("calibration_metrics", "uncertainty_adjustment", "model_caveats")),
)

OPERATION_REGISTRY = tuple(
    OperationSpec(f"P03-{index:02d}", *row)
    for index, row in enumerate(_ROWS, start=1)
)


def registry_as_dicts() -> list[dict[str, object]]:
    """Return the canonical 12-stage Point 03 operation registry."""
    return [item.to_dict() for item in OPERATION_REGISTRY]
