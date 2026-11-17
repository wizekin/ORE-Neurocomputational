"""Point 13: individualized and development-sensitive language comprehension.

This is an adaptive computational profile, not a diagnostic classifier or a literal
simulation of neuronal calculations. Research statistics are kept distinct from
biologically plausible candidate operations.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


POINT13_COMPONENTS = (
    "lexical_access_efficiency",
    "vocabulary_knowledge",
    "perceptual_reliability",
    "working_memory_support",
    "inhibitory_control",
    "world_knowledge",
    "pragmatic_perspective_taking",
    "contextual_cue_use",
    "reanalysis_efficiency",
)

POINT13_STEP_NAMES = (
    "Define task-specific baseline",
    "Estimate component abilities separately",
    "Characterize linguistic experience and proficiency",
    "Estimate contextual facilitation",
    "Estimate individual cue weighting",
    "Separate context compensation from core efficiency",
    "Model developmental trajectories",
    "Model pragmatic development conditionally",
    "Model multilingual processing by language",
    "Test aging-related compensation and interference",
    "Estimate memory and executive support demands",
    "Separate stable differences from momentary state",
    "Estimate individual and group variation",
    "Compare compensatory processing strategies",
    "Adapt downstream processing parameters",
    "Preserve uncertainty and alternative explanations",
    "Evaluate calibration and generalization",
    "Update the profile reversibly",
)

POINT13_STEP_SPECS = tuple(
    {
        "operation_id": f"LC-P13-{index:02d}",
        "name": name,
        "objective": name + " from participant-level evidence without demographic stereotyping.",
        "mathematical_operations": (
            "component measurement and standard error",
            "cue-weighted evidence and softmax",
            "longitudinal slope and hierarchical variation",
            "held-out loss and confidence calibration",
        ),
        "biological_reference": (
            "Candidate biological processes are discussed separately; statistical "
            "estimation is not a literal neuronal calculation."
        ),
    }
    for index, name in enumerate(POINT13_STEP_NAMES, 1)
)


def _plain(value: Any) -> Any:
    """Convert NumPy values, dataclasses, and nested containers to JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _snapshot(value: Any, name: str) -> dict[str, Any]:
    """Validate and detach a mapping so later caller mutation cannot change a request."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return deepcopy(dict(value))


def _numbers(values: Any) -> np.ndarray:
    """Return finite numeric values from a scalar or collection."""
    if values is None:
        return np.asarray([], dtype=float)
    if isinstance(values, (int, float, np.number)):
        values = [values]
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    return array[np.isfinite(array)]


def _summary(values: Any) -> dict[str, Any]:
    """Summarize finite observations and retain sampling uncertainty."""
    array = _numbers(values)
    count = int(array.size)
    if not count:
        return {"available": False, "n": 0, "mean": None, "sd": None, "se": None}
    sd = float(np.std(array, ddof=1)) if count > 1 else None
    se = sd / math.sqrt(count) if sd is not None else None
    return {
        "available": True,
        "n": count,
        "mean": float(np.mean(array)),
        "sd": sd,
        "se": se,
    }


def _softmax(scores: Mapping[str, Any]) -> dict[str, float]:
    """Normalize finite candidate scores into relative weights with numerical stability."""
    valid = {}
    for key, value in scores.items():
        array = _numbers([value])
        if array.size:
            valid[str(key)] = float(array[0])
    if not valid:
        return {}
    peak = max(valid.values())
    weights = {key: math.exp(value - peak) for key, value in valid.items()}
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def _regression_slope(times: Any, values: Any) -> dict[str, Any]:
    """Estimate a descriptive linear slope only when at least two distinct times exist."""
    x, y = _numbers(times), _numbers(values)
    size = min(x.size, y.size)
    if size < 2:
        return {"available": False, "slope": None, "n": int(size)}
    x, y = x[:size], y[:size]
    if np.unique(x).size < 2:
        return {"available": False, "slope": None, "n": int(size)}
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "available": True,
        "slope": float(slope),
        "intercept": float(intercept),
        "n": int(size),
        "interpretation": "descriptive trajectory; not a biological growth law",
    }


@dataclass(frozen=True, slots=True)
class Point13Request:
    """Immutable participant-specific request for adaptive language-comprehension profiling."""

    participant_id: str
    target_operation: str
    stimulus: Mapping[str, Any]
    discourse_context: Mapping[str, Any] = field(default_factory=dict)
    task_requirements: Mapping[str, Any] = field(default_factory=dict)
    observations: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    component_measures: Mapping[str, Any] = field(default_factory=dict)
    experience_profile: Mapping[str, Any] = field(default_factory=dict)
    baseline_conditions: Mapping[str, Any] = field(default_factory=dict)
    longitudinal_measures: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    cohort_observations: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    candidate_strategies: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    previous_profile: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate required identifiers and deep-copy all nested request data."""
        if not str(self.participant_id).strip():
            raise ValueError("participant_id must be non-empty")
        if not str(self.target_operation).strip():
            raise ValueError("target_operation must be non-empty")
        for key in (
            "stimulus",
            "discourse_context",
            "task_requirements",
            "component_measures",
            "experience_profile",
            "baseline_conditions",
            "previous_profile",
            "metadata",
        ):
            object.__setattr__(self, key, _snapshot(getattr(self, key), key))
        for key in (
            "observations",
            "longitudinal_measures",
            "cohort_observations",
            "candidate_strategies",
        ):
            raw = getattr(self, key)
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise TypeError(f"{key} must be a sequence of mappings")
            copied = tuple(_snapshot(item, f"{key}[{index}]") for index, item in enumerate(raw))
            object.__setattr__(self, key, copied)


@dataclass(slots=True)
class Point13Result:
    """Auditable individualized profile, adapted settings, uncertainty, and 18-step trace."""

    participant_id: str
    target_operation: str
    profile: dict[str, Any]
    adapted_parameters: dict[str, Any]
    uncertainty_estimates: dict[str, Any]
    unresolved_hypotheses: list[str]
    trace: list[dict[str, Any]]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible representation of the result."""
        return _plain(asdict(self))


class Point13IndividualizedProcessor:
    """Estimate a reversible, task-sensitive comprehension profile from supplied observations."""

    def __init__(self, minimum_trials: int = 3, default_learning_rate: float = 0.25) -> None:
        """Configure descriptive evidence thresholds and a bounded profile update rate."""
        if minimum_trials < 1:
            raise ValueError("minimum_trials must be positive")
        if not 0.0 < default_learning_rate <= 1.0:
            raise ValueError("default_learning_rate must be in (0, 1]")
        self.minimum_trials = int(minimum_trials)
        self.default_learning_rate = float(default_learning_rate)

    def _record(
        self, trace: list[dict[str, Any]], step: int, result: Mapping[str, Any]
    ) -> None:
        """Append one detached, source-ordered operation result."""
        spec = POINT13_STEP_SPECS[step - 1]
        trace.append(
            {
                "step": step,
                "operation_id": spec["operation_id"],
                "name": spec["name"],
                "result": _plain(dict(result)),
            }
        )

    def _baseline(self, request: Point13Request) -> dict[str, Any]:
        """Compare supplied low- and high-context baseline conditions."""
        output = {}
        for condition in ("low_context", "high_context"):
            record = request.baseline_conditions.get(condition, {})
            record = record if isinstance(record, Mapping) else {}
            output[condition] = {
                "response_time": _summary(record.get("response_time", [])),
                "accuracy": _summary(record.get("accuracy", [])),
                "error_types": _plain(record.get("error_types", [])),
                "neural_measures": _plain(record.get("neural_measures", {})),
            }
        output["task_operation"] = request.target_operation
        output["modality"] = request.stimulus.get("modality", "unspecified")
        output["speed_accuracy_tradeoff_must_be_checked"] = True
        return output

    def _components(self, measures: Mapping[str, Any]) -> dict[str, Any]:
        """Estimate each component independently and retain sample size and uncertainty."""
        output = {}
        for component in POINT13_COMPONENTS:
            supplied = measures.get(component)
            if isinstance(supplied, Mapping):
                values = supplied.get("values", supplied.get("score", []))
                summary = _summary(values)
                summary["measurement_notes"] = supplied.get("notes")
                summary["measurement_quality"] = supplied.get("quality", "unspecified")
            else:
                summary = _summary(supplied)
                summary["measurement_notes"] = None
                summary["measurement_quality"] = "unspecified"
            summary["not_inferred_from_demographic_label"] = True
            output[component] = summary
        return output

    def _language_profile(self, experience: Mapping[str, Any]) -> dict[str, Any]:
        """Keep exposure, acquisition history, proficiency, and use separate by language."""
        languages = experience.get("languages", {})
        if not isinstance(languages, Mapping):
            languages = {}
        output = {}
        for language, values in languages.items():
            values = values if isinstance(values, Mapping) else {}
            output[str(language)] = {
                "age_of_acquisition": values.get("age_of_acquisition"),
                "exposure_hours": values.get("exposure_hours"),
                "recency_of_use": values.get("recency_of_use"),
                "use_contexts": _plain(values.get("use_contexts", [])),
                "proficiency": _plain(values.get("proficiency", {})),
                "target_word_familiarity": _plain(values.get("target_word_familiarity", {})),
                "parameters_are_language_specific": True,
            }
        return {
            "languages": output,
            "education_and_literacy": _plain(experience.get("education_and_literacy", {})),
            "domain_experience": _plain(experience.get("domain_experience", {})),
            "task_familiarity": experience.get("task_familiarity"),
            "age_is_descriptor_not_mechanism": True,
        }

    def _context_benefit(self, baseline: Mapping[str, Any]) -> dict[str, Any]:
        """Quantify context benefit only when matched baseline summaries are available."""
        low = baseline.get("low_context", {})
        high = baseline.get("high_context", {})
        low_rt = low.get("response_time", {}).get("mean")
        high_rt = high.get("response_time", {}).get("mean")
        low_acc = low.get("accuracy", {}).get("mean")
        high_acc = high.get("accuracy", {}).get("mean")
        rt_benefit = low_rt - high_rt if low_rt is not None and high_rt is not None else None
        accuracy_benefit = (
            high_acc - low_acc if low_acc is not None and high_acc is not None else None
        )
        return {
            "response_time_benefit": rt_benefit,
            "accuracy_benefit": accuracy_benefit,
            "available": rt_benefit is not None or accuracy_benefit is not None,
            "raw_response_time_units_preserved": True,
            "speed_accuracy_tradeoff_requires_joint_interpretation": True,
            "not_a_direct_measure_of_region_activity": True,
        }

    def _cue_weights(self, observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Preserve estimated cue weights and normalize supplied candidate evidence when possible."""
        supplied = {}
        for row in observations:
            weights = row.get("cue_weight_estimates", {})
            if isinstance(weights, Mapping):
                for key, value in weights.items():
                    supplied.setdefault(str(key), []).extend(_numbers([value]).tolist())
        means = {key: float(np.mean(values)) for key, values in supplied.items() if values}
        candidate = {}
        for row in observations:
            scores = row.get("candidate_scores", {})
            if isinstance(scores, Mapping):
                for key, value in scores.items():
                    candidate.setdefault(str(key), []).extend(_numbers([value]).tolist())
        candidate_means = {
            key: float(np.mean(values)) for key, values in candidate.items() if values
        }
        return {
            "cue_weight_estimates": means,
            "cue_weight_sample_counts": {key: len(values) for key, values in supplied.items()},
            "candidate_probabilities": _softmax(candidate_means),
            "weights_are_empirical_inputs_not_assumed_truth": True,
        }

    def _development(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Fit component-specific descriptive trajectories from repeated measurements."""
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            component = str(row.get("component", "")).strip()
            if component:
                grouped.setdefault(component, []).append(row)
        output = {}
        for component, values in grouped.items():
            output[component] = _regression_slope(
                [row.get("time", row.get("age")) for row in values],
                [row.get("score") for row in values],
            )
        return {
            "component_trajectories": output,
            "age_or_time_is_not_a_fixed_threshold": True,
            "trajectory_is_descriptive_not_a_neural_growth_equation": True,
        }

    def _pragmatic_profile(
        self, requirements: Mapping[str, Any], measures: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Describe the pragmatic operation requested without collapsing distinct tasks."""
        task_types = requirements.get(
            "pragmatic_operations",
            ["reference", "indirect_request", "implicature", "irony", "belief_inference"],
        )
        if not isinstance(task_types, (list, tuple)):
            task_types = []
        observed = measures.get("pragmatic_tasks", {})
        return {
            "required_operations": _plain(task_types),
            "task_specific_measures": _plain(observed if isinstance(observed, Mapping) else {}),
            "theory_of_mind_is_not_assumed_for_every_pragmatic_task": True,
            "literal_comprehension_and_intention_inference_are_separate": True,
        }

    def _aging_profile(
        self, experience: Mapping[str, Any], components: Mapping[str, Any], context: Mapping[str, Any],
        observations: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Test supplied context support and interference without inferring a universal aging effect."""
        lexical = components["lexical_access_efficiency"].get("mean")
        world = components["world_knowledge"].get("mean")
        interference = []
        for row in observations:
            value = _numbers([row.get("interference_cost")])
            if value.size:
                interference.extend(value.tolist())
        age = experience.get("chronological_age")
        return {
            "age_descriptor": age,
            "lexical_efficiency_estimate": lexical,
            "world_knowledge_estimate": world,
            "context_benefit": context,
            "interference_cost": _summary(interference),
            "aging_relevance_explicitly_supplied": bool(
                experience.get("aging_relevant", False) or experience.get("age_group") == "older"
            ),
            "no_uniform_aging_profile_assumed": True,
        }

    def _resource_demands(self, requirements: Mapping[str, Any]) -> dict[str, Any]:
        """Combine supplied task-demand dimensions without inventing a capacity threshold."""
        keys = ("length", "dependency_load", "competitor_count", "revision_demand")
        values = {}
        for key in keys:
            numeric = _numbers([requirements.get(key)])
            values[key] = float(numeric[0]) if numeric.size else None
        weights = requirements.get("demand_weights", {})
        weights = weights if isinstance(weights, Mapping) else {}
        total = 0.0
        used = []
        for key, value in values.items():
            weight_values = _numbers([weights.get(key, 1.0)])
            if value is not None and weight_values.size:
                total += value * float(weight_values[0])
                used.append(key)
        return {
            "dimensions": values,
            "weighted_demand": total if used else None,
            "included_dimensions": used,
            "capacity_threshold_not_assumed": True,
        }

    def _state_stability(self, observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Separate repeated-session performance variation from supplied temporary state labels."""
        sessions: dict[str, list[float]] = {}
        states: dict[str, list[float]] = {}
        for index, row in enumerate(observations):
            session = str(row.get("session_id", f"trial-{index}"))
            rt = _numbers([row.get("response_time")])
            if rt.size:
                sessions.setdefault(session, []).append(float(rt[0]))
            state = row.get("state")
            if state is not None and rt.size:
                states.setdefault(str(state), []).append(float(rt[0]))
        return {
            "session_response_times": {
                key: _summary(values) for key, values in sessions.items()
            },
            "state_response_times": {key: _summary(values) for key, values in states.items()},
            "repeated_sessions": len(sessions),
            "stable_trait_not_inferred_from_single_trial": True,
        }

    def _heterogeneity(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Summarize supplied participant-level effects without treating group means as universal."""
        values = []
        for row in rows:
            numeric = _numbers([row.get("effect", row.get("score"))])
            if numeric.size:
                values.append(float(numeric[0]))
        return {
            "participant_summary": _summary(values),
            "participant_count": len(rows),
            "moderators": _plain(
                {
                    key: sorted(
                        {str(row[key]) for row in rows if row.get(key) is not None}
                    )
                    for key in ("age_group", "proficiency_group", "language", "culture")
                }
            ),
            "group_average_does_not_describe_every_person": True,
        }

    def _strategy_comparison(
        self, strategies: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Compare supplied strategy predictions on paired held-out outcomes and complexity."""
        results = []
        for strategy in strategies:
            observed = _numbers(strategy.get("held_out_observed", []))
            predicted = _numbers(strategy.get("held_out_predicted", []))
            if observed.size == 0 or observed.size != predicted.size:
                mse = None
            else:
                mse = float(np.mean((observed - predicted) ** 2))
            complexity = _numbers([strategy.get("complexity", 0.0)])
            penalty = float(complexity[0]) if complexity.size else 0.0
            penalty_weight = _numbers([strategy.get("penalty_weight", 0.0)])
            weight = float(penalty_weight[0]) if penalty_weight.size else 0.0
            results.append(
                {
                    "strategy_id": str(strategy.get("strategy_id", "unnamed")),
                    "held_out_mse": mse,
                    "complexity": penalty,
                    "score": mse + weight * penalty if mse is not None else None,
                    "supported_by_held_out_data": mse is not None,
                }
            )
        ranked = sorted(
            (item for item in results if item["score"] is not None),
            key=lambda item: item["score"],
        )
        return {
            "strategies": results,
            "ranked_strategy_ids": [item["strategy_id"] for item in ranked],
            "best_score_is_not_proof_of_biological_truth": True,
        }

    def _adapt(
        self, components: Mapping[str, Any], context: Mapping[str, Any],
        demands: Mapping[str, Any], requirements: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Translate measured profile dimensions into reversible downstream model settings."""
        perceptual = components["perceptual_reliability"].get("mean")
        lexical = components["lexical_access_efficiency"].get("mean")
        memory = components["working_memory_support"].get("mean")
        inhibition = components["inhibitory_control"].get("mean")
        context_benefit = context.get("response_time_benefit")
        return {
            "retain_alternative_word_forms": perceptual is not None
            and perceptual < float(requirements.get("low_perceptual_threshold", 0.5)),
            "allow_contextual_candidate_support": context_benefit is not None
            and context_benefit > 0,
            "preserve_uncertainty_for_unfamiliar_words": components[
                "vocabulary_knowledge"
            ].get("mean") is None
            or components["vocabulary_knowledge"].get("mean")
            < float(requirements.get("vocabulary_threshold", 0.5)),
            "increase_referent_maintenance": (
                demands.get("weighted_demand") is not None
                and memory is not None
                and demands["weighted_demand"] > memory
            ),
            "retain_previous_interpretation_trace": inhibition is not None
            and inhibition < float(requirements.get("inhibition_threshold", 0.5)),
            "lexical_efficiency_estimate": lexical,
            "adaptation_is_reversible": True,
            "thresholds_are_configurable_modelling_choices": True,
        }

    def _calibration(self, observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Calculate confidence calibration and held-out error from supplied paired observations."""
        confidence, correct = [], []
        observed, predicted = [], []
        for row in observations:
            c, y = _numbers([row.get("confidence")]), _numbers([row.get("correct")])
            if c.size and y.size and 0 <= c[0] <= 1 and 0 <= y[0] <= 1:
                confidence.append(float(c[0]))
                correct.append(float(y[0]))
            o, p = _numbers([row.get("held_out_observed")]), _numbers(
                [row.get("held_out_predicted")]
            )
            if o.size and p.size:
                observed.append(float(o[0]))
                predicted.append(float(p[0]))
        calibration_error = None
        if confidence:
            bins = np.linspace(0.0, 1.0, 6)
            terms = []
            c_array, y_array = np.asarray(confidence), np.asarray(correct)
            for index in range(5):
                mask = (c_array >= bins[index]) & (
                    c_array <= bins[index + 1] if index == 4 else c_array < bins[index + 1]
                )
                if np.any(mask):
                    terms.append(float(np.mean(mask)) * abs(
                        float(np.mean(y_array[mask])) - float(np.mean(c_array[mask]))
                    ))
            calibration_error = float(sum(terms))
        held_out_mse = (
            float(np.mean((np.asarray(observed) - np.asarray(predicted)) ** 2))
            if observed and len(observed) == len(predicted)
            else None
        )
        return {
            "calibration_error": calibration_error,
            "calibration_n": len(confidence),
            "held_out_mse": held_out_mse,
            "held_out_n": len(observed),
            "generalization_supported": bool(
                held_out_mse is not None
                and all(row.get("model_frozen", False) for row in observations if
                        row.get("held_out_observed") is not None)
                and any(row.get("independent_test_set", False) for row in observations)
            ),
            "task_specificity_must_be_reported": True,
        }

    def process(self, request: Point13Request) -> Point13Result:
        """Run the 18-step individualized workflow without demographic stereotyping."""
        if not isinstance(request, Point13Request):
            raise TypeError("request must be a Point13Request")
        trace: list[dict[str, Any]] = []
        baseline = self._baseline(request)
        self._record(trace, 1, baseline)
        components = self._components(request.component_measures)
        self._record(trace, 2, components)
        languages = self._language_profile(request.experience_profile)
        self._record(trace, 3, languages)
        context = self._context_benefit(baseline)
        self._record(trace, 4, context)
        cues = self._cue_weights(request.observations)
        self._record(trace, 5, cues)
        compensation = {
            "core_component_estimates": {
                "lexical_access_efficiency": components[
                    "lexical_access_efficiency"
                ],
                "perceptual_reliability": components["perceptual_reliability"],
            },
            "context_benefit": context,
            "core_efficiency_and_context_benefit_are_distinct": True,
        }
        self._record(trace, 6, compensation)
        development = self._development(request.longitudinal_measures)
        self._record(trace, 7, development)
        pragmatic = self._pragmatic_profile(request.task_requirements, request.component_measures)
        self._record(trace, 8, pragmatic)
        self._record(trace, 9, languages)
        aging = self._aging_profile(
            request.experience_profile, components, context, request.observations
        )
        self._record(trace, 10, aging)
        demands = self._resource_demands(request.task_requirements)
        self._record(trace, 11, demands)
        state = self._state_stability(request.observations)
        self._record(trace, 12, state)
        heterogeneity = self._heterogeneity(request.cohort_observations)
        self._record(trace, 13, heterogeneity)
        strategies = self._strategy_comparison(request.candidate_strategies)
        self._record(trace, 14, strategies)
        adapted = self._adapt(
            components, context, demands, request.task_requirements
        )
        self._record(trace, 15, adapted)
        unresolved = []
        if not request.observations:
            unresolved.append("no_trial_level_observations_supplied")
        if not any(item["available"] for item in components.values()):
            unresolved.append("component_abilities_not_measured")
        if not languages["languages"]:
            unresolved.append("language_specific_experience_not_supplied")
        if not strategies["ranked_strategy_ids"]:
            unresolved.append("competing_strategies_not_evaluated_on_held_out_data")
        uncertainty = {
            "component_standard_errors": {
                key: value["se"] for key, value in components.items()
            },
            "missing_components": [
                key for key, value in components.items() if not value["available"]
            ],
            "unresolved_hypotheses": list(unresolved),
            "missing_measurement_is_not_zero_ability": True,
            "demographic_label_is_not_a_substitute_for_measurement": True,
        }
        self._record(trace, 16, uncertainty)
        calibration = self._calibration(request.observations)
        self._record(trace, 17, calibration)
        learning_rate_values = _numbers(
            [request.metadata.get("learning_rate", self.default_learning_rate)]
        )
        learning_rate = (
            float(learning_rate_values[0])
            if learning_rate_values.size
            else self.default_learning_rate
        )
        learning_rate = min(max(learning_rate, 0.0), 1.0)
        previous = request.previous_profile.get("components", {})
        previous = previous if isinstance(previous, Mapping) else {}
        updated = {}
        for key, current in components.items():
            new_value = current["mean"]
            old_values = _numbers([previous.get(key)])
            if new_value is not None and old_values.size:
                new_value = (
                    (1.0 - learning_rate) * float(old_values[0])
                    + learning_rate * new_value
                )
            # A previous estimate is not presented as a current measurement when
            # this component has no new evidence. Missing remains unknown.
            updated[key] = new_value
        profile = {
            "participant_id": request.participant_id,
            "target_operation": request.target_operation,
            "components": updated,
            "component_measurements": components,
            "language_experience": languages,
            "contextual_facilitation": context,
            "cue_profile": cues,
            "developmental_trajectories": development,
            "pragmatic_profile": pragmatic,
            "aging_and_interference_profile": aging,
            "resource_demands": demands,
            "state_stability": state,
            "group_heterogeneity": heterogeneity,
            "strategy_comparison": strategies,
            "calibration": calibration,
            "profile_is_task_sensitive_not_permanent_identity": True,
        }
        self._record(
            trace,
            18,
            {
                "learning_rate": learning_rate,
                "updated_components": updated,
                "previous_profile_used": bool(previous),
                "update_is_reversible": True,
            },
        )
        return Point13Result(
            participant_id=request.participant_id,
            target_operation=request.target_operation,
            profile=profile,
            adapted_parameters=adapted,
            uncertainty_estimates=uncertainty,
            unresolved_hypotheses=unresolved,
            trace=trace,
            diagnostics={
                "valid": True,
                "step_count": len(trace),
                "component_count": len(POINT13_COMPONENTS),
                "research_statistics_are_not_neural_calculations": True,
                "clinical_diagnosis_not_performed": True,
            },
        )


def register_point13_annotations(
    repository_root: str | Path | None = None,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Index Point 13 public definitions in ORE's shared SQLite annotation catalogue."""
    from ore_logger import LoggerDatabase, sync_repository

    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    target = Path(
        database_path
        or os.environ.get("ORE_LOGGER_DB")
        or root / "ore_logs" / "ore_logger.sqlite3"
    ).expanduser().resolve()
    report = sync_repository(
        LoggerDatabase(target), root, include_paths={"ore_language_comprehension"}
    )
    return {
        **report,
        "database_path": str(target),
        "modules": ["ore_language_comprehension"],
        "annotation_source": "python_docstring_and_ast_signature",
        "storage": "ore_logger.annotations SQLite table",
    }
