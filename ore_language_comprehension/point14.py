"""Point 14: clinically relevant language-comprehension profiling without diagnosis.

This module implements a task-specific research model for separating language,
semantic, discourse, pragmatic, memory, executive, sensory, and social-inference
demands. Its equations are computational/psychometric abstractions; the biological
track names plausible distributed operations and never asserts literal neural
execution of logistic regression, Bayesian inference, graph metrics, or cross-validation.
"""
from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


POINT14_COMPONENTS = (
    "perceptual_access",
    "lexical_access",
    "syntactic_parsing",
    "semantic_composition",
    "working_memory",
    "discourse_coherence",
    "pragmatic_inference",
    "social_state_inference",
    "world_knowledge",
    "executive_control",
)

POINT14_PROFILE_FAMILIES = (
    "right_hemisphere",
    "autism_related",
    "psychosis_spectrum",
    "neurodegenerative",
    "aphasia",
)

POINT14_STEP_SPECS = (
    ("define_outcome", "Define the observable comprehension outcome", "Operationalize the task before interpreting errors", "Attention and task-set control select relevant behaviour", "Outcome coding and latency distributions", "Analyst-side measurement, not a neuronal equation"),
    ("task_demands", "Build a task-demand matrix", "Map task items onto distinct functional operations", "Distributed language, memory, control, and social networks cooperate", "Demand matrix, rank, and condition number", "Matrix diagnostics assess identifiability, not neural calculation"),
    ("basic_access", "Establish sensory and basic language access", "Check modality access, word recognition, vocabulary, and low-load grammar", "Auditory/visual sensory pathways and language-selective temporal/frontal populations", "Modality-specific accuracy and matched contrasts", "Behavioural access checks; no reverse inference from brain activation"),
    ("literal_inference", "Separate literal meaning from inferred meaning", "Compare explicit content with implied or nonliteral meaning", "Lexical-semantic representations interact with discourse and control systems", "Separate means and literal-minus-inference cost", "A contrast does not identify a unique cause"),
    ("cue_integration", "Test prosody and communicative cue use", "Measure how linguistic, prosodic, scene, and speaker cues jointly contribute", "Parallel cue-sensitive populations and recurrent integration", "Reliability-weighted cue scores and normalized cue weights", "Weighted/Bayesian equations are normative models"),
    ("social_nonsocial", "Separate social from nonsocial inference", "Compare beliefs/intentions with matched physical or causal inference", "Language and theory-of-mind systems can cooperate while remaining distinguishable", "Separate domain scores and explicit-knowledge benefit", "Pragmatics is not reducible to theory of mind"),
    ("discourse_coherence", "Measure discourse maintenance and coherence", "Represent entities/events and test required, incorrect, and unsupported links", "Recurrent temporal, parietal, frontal, and memory interactions", "Graph edge precision/recall and distractor cost", "Graph metrics are analysis tools, not neuronal graph calculations"),
    ("memory_executive_load", "Test working-memory and executive contributions", "Vary length, load, interference, and switching demand", "Recurrent maintenance and distributed control/inhibitory interactions", "Load slopes, high-low contrasts, and switching costs", "Regression is a behavioural model"),
    ("right_hemisphere_profile", "Profile discourse after right-hemisphere injury when relevant", "Compare sentence-level performance with discourse, inference, prosody, and nonverbal cues", "Distributed right-hemisphere and bilateral discourse/cognitive-communication systems", "Within-person contrasts by task family", "No single right-hemisphere pragmatics module is assumed"),
    ("autism_profile", "Profile autism-related pragmatic tasks when relevant", "Test individual cue use, explicitness effects, language, and communication preferences", "Heterogeneous language, sensory, social-cognitive, and executive networks", "Task-specific cue and context effects", "No uniform autistic profile or individual neural biomarker is inferred"),
    ("psychosis_profile", "Profile psychosis-spectrum language and discourse when relevant", "Separate comprehension, production, semantic selection, coherence, and executive/social demands", "Distributed language, semantic, default-mode, social, and control networks", "Semantic/context and discourse measures kept separate", "No diagnosis-specific circuit or mechanism is assumed"),
    ("neurodegeneration_profile", "Differentiate neurodegenerative profiles", "Track semantic knowledge, episodic memory, executive function, and pragmatics separately", "Condition- and person-dependent distributed network change", "Component-specific longitudinal slopes", "No single dementia-severity mechanism is imposed"),
    ("aphasia_profile", "Separate aphasic language access from nonlinguistic reasoning", "Compare linguistic and matched nonlinguistic tasks", "Language-selective circuitry interacts with conceptual and domain-general systems", "Language/concept contrasts and matched-task effects", "Preservation of nonlinguistic ability is possible, not guaranteed"),
    ("component_profile", "Estimate cross-domain components and preservation", "Combine repeated measures, retaining stronger and weaker measured abilities", "Distributed population coding and interacting networks", "Component means, sample spread, and measurement counts", "Scores are task-bound estimates, not fixed traits"),
    ("competing_models", "Compare competing explanations", "Generate alternatives and test their distinct predictions", "Prediction/error-driven revision is a plausible broad principle", "Held-out Brier/MSE and complexity-aware comparison", "Model comparison is researcher-side analysis"),
    ("support_conditions", "Test supports and compensatory routes", "Vary context, memory load, rate, speaker knowledge, and cues", "Attention and recurrent feedback can alter effective cue availability", "Matched support benefit and generalization", "Improvement does not prove that an underlying mechanism was repaired"),
    ("independent_validation", "Validate against independent behaviour and neural evidence", "Test predictions on held-out items/sessions and interpret neural evidence cautiously", "Neural measures may constrain models at appropriate levels", "Held-out accuracy, Brier score, and calibration", "Cross-validation is not a neural operation"),
    ("longitudinal_update", "Update the profile over time", "Update only components with new evidence and preserve uncertainty", "Plasticity, state variation, and disease/development-related change", "Component-wise state updates and slopes", "A missing new measurement stays unknown; prior values remain historical"),
    ("transparent_report", "Produce a transparent functional explanation", "Report evidence, preserved abilities, alternatives, support effects, limits, and next tests", "No single biological module explains a whole profile", "Relative rankings and confidence/uncertainty summaries", "Output is a research profile, never a diagnosis"),
)

POINT14_STEP_NAMES = tuple(item[1] for item in POINT14_STEP_SPECS)


def _finite(value: Any) -> float | None:
    """Return a finite numeric value or None for missing, malformed, or non-finite input."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _score(record: Mapping[str, Any]) -> float | None:
    """Read a normalized task score without turning missing responses into zero."""
    for key in ("score", "accuracy", "correct"):
        value = record.get(key)
        if key == "correct" and isinstance(value, bool):
            return float(value)
        number = _finite(value)
        if number is not None:
            return min(1.0, max(0.0, number))
    return None


def _mean(values: Iterable[Any]) -> float | None:
    """Compute a finite arithmetic mean while preserving an empty sample as unknown."""
    numbers = [number for value in values if (number := _finite(value)) is not None]
    return float(sum(numbers) / len(numbers)) if numbers else None


def _slope(points: Sequence[tuple[float, float]]) -> float | None:
    """Estimate a least-squares slope only when at least two distinct times are observed."""
    if len(points) < 2 or len({time for time, _ in points}) < 2:
        return None
    x = np.asarray([point[0] for point in points], dtype=float)
    y = np.asarray([point[1] for point in points], dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def _as_values(value: Any) -> list[float]:
    """Normalize a scalar, sequence, or values mapping into finite numeric observations."""
    if isinstance(value, Mapping):
        value = value.get("values", value.get("score"))
    if isinstance(value, (list, tuple)):
        candidates = value
    else:
        candidates = [value]
    return [number for item in candidates if (number := _finite(item)) is not None]


def _edges(value: Any) -> set[str]:
    """Normalize graph edges into stable strings for exact relation comparison."""
    result: set[str] = set()
    if not isinstance(value, (list, tuple, set)):
        return result
    for edge in value:
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            result.add("::".join(str(part) for part in edge))
        elif isinstance(edge, str) and edge.strip():
            result.add(edge.strip())
    return result


@dataclass
class Point14Request:
    """Validated inputs for the Point 14 task-specific research profile.

    The clinical-context field is descriptive metadata supplied by the caller. It is
    never used as a diagnostic classifier or as evidence of a particular deficit.
    """

    participant_id: str
    target_outcome: str
    task_set: list[dict[str, Any]] = field(default_factory=list)
    component_measures: dict[str, Any] = field(default_factory=dict)
    observations: list[dict[str, Any]] = field(default_factory=list)
    clinical_contexts: list[str] = field(default_factory=list)
    candidate_explanations: list[dict[str, Any]] = field(default_factory=list)
    longitudinal_observations: list[dict[str, Any]] = field(default_factory=list)
    previous_profile: dict[str, Any] = field(default_factory=dict)
    optional_neural_data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate identifiers and deep-copy nested input so processing is reproducible."""
        if not isinstance(self.participant_id, str) or not self.participant_id.strip():
            raise ValueError("participant_id must be a non-empty string")
        if not isinstance(self.target_outcome, str) or not self.target_outcome.strip():
            raise ValueError("target_outcome must be a non-empty string")
        for name in (
            "task_set", "observations", "clinical_contexts",
            "candidate_explanations", "longitudinal_observations",
        ):
            value = getattr(self, name)
            if not isinstance(value, list):
                raise TypeError(f"{name} must be a list")
            setattr(self, name, copy.deepcopy(value))
        for name in (
            "component_measures", "previous_profile", "optional_neural_data", "metadata",
        ):
            value = getattr(self, name)
            if not isinstance(value, dict):
                raise TypeError(f"{name} must be a dictionary")
            setattr(self, name, copy.deepcopy(value))
        if not all(isinstance(item, dict) for item in self.task_set + self.observations):
            raise TypeError("task_set and observations must contain dictionaries")
        if not all(isinstance(item, str) and item.strip() for item in self.clinical_contexts):
            raise ValueError("clinical_contexts must contain non-empty strings")


@dataclass
class Point14Result:
    """Structured functional profile, alternatives, predictions, trace, and safeguards."""

    participant_id: str
    target_outcome: str
    profile: dict[str, Any]
    candidate_explanations: list[dict[str, Any]]
    uncertainty_estimates: dict[str, Any]
    support_predictions: dict[str, Any]
    validation: dict[str, Any]
    trace: list[dict[str, Any]]
    diagnostics: dict[str, Any]


class Point14ClinicalProfileProcessor:
    """Execute the 19-step Point 14 functional and biological-reference workflow.

    The processor is descriptive and hypothesis-testing oriented. It does not assign
    diagnoses, infer a unique neural cause, or treat clinical group labels as parameters.
    """

    def __init__(self, *, relative_split: float = 0.5) -> None:
        """Create a processor with a configurable within-profile ranking split."""
        if not 0.0 < relative_split < 1.0:
            raise ValueError("relative_split must be between 0 and 1")
        self.relative_split = float(relative_split)

    def _record(self, trace: list[dict[str, Any]], index: int, result: dict[str, Any]) -> None:
        """Append one source-ordered step with functional, biological, and mathematical notes."""
        operation_id, name, functional, biological, mathematics, status = POINT14_STEP_SPECS[index - 1]
        trace.append({
            "step": index,
            "operation_id": f"LC-P14-{index:02d}",
            "name": name,
            "functional_operation": functional,
            "possible_biological_implementation": biological,
            "mathematical_operations": mathematics,
            "scientific_status": status,
            "result": result,
        })

    def _task_records(self, request: Point14Request) -> list[dict[str, Any]]:
        """Combine declared task metadata with observed responses without mutating the request."""
        tasks = copy.deepcopy(request.task_set)
        by_id = {
            str(task.get("task_id")): task
            for task in tasks if task.get("task_id") is not None
        }
        for index, observation in enumerate(request.observations):
            task_id = observation.get("task_id")
            if task_id is not None and str(task_id) in by_id:
                by_id[str(task_id)].update(copy.deepcopy(observation))
            else:
                item = copy.deepcopy(observation)
                item.setdefault("task_id", f"observation-{index + 1}")
                tasks.append(item)
        for index, task in enumerate(tasks):
            task.setdefault("task_id", f"task-{index + 1}")
        return tasks

    def _components(self, request: Point14Request, tasks: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Estimate measured components and their uncertainty using independent observations."""
        components: dict[str, Any] = {}
        uncertainty: dict[str, Any] = {"missing_components": [], "component_measurement_counts": {}}
        for component in POINT14_COMPONENTS:
            values = _as_values(request.component_measures.get(component))
            values.extend(
                score for task in tasks
                if task.get("component") == component
                and (score := _score(task)) is not None
            )
            mean = _mean(values)
            components[component] = mean
            uncertainty["component_measurement_counts"][component] = len(values)
            if mean is None:
                uncertainty["missing_components"].append(component)
                uncertainty.setdefault("component_standard_error", {})[component] = None
            elif len(values) > 1:
                uncertainty.setdefault("component_standard_error", {})[component] = float(
                    np.std(values, ddof=1) / math.sqrt(len(values))
                )
            else:
                uncertainty.setdefault("component_standard_error", {})[component] = None
        return components, uncertainty

    def process(self, request: Point14Request) -> Point14Result:
        """Run all 19 algorithm steps and return an auditable, non-diagnostic profile."""
        if not isinstance(request, Point14Request):
            raise TypeError("request must be a Point14Request")
        tasks = self._task_records(request)
        trace: list[dict[str, Any]] = []
        uncertainty: dict[str, Any] = {}

        # 1. Define the behaviour before interpreting it.
        self._record(trace, 1, {
            "participant_id": request.participant_id,
            "target_outcome": request.target_outcome,
            "task_count": len(tasks),
            "outcome_is_operationalized": True,
        })

        # 2. Build a task-by-component demand matrix and expose confounding.
        matrix = np.asarray([
            [_finite(task.get("demands", {}).get(component)) or 0.0
             for component in POINT14_COMPONENTS]
            for task in tasks
        ], dtype=float) if tasks else np.zeros((0, len(POINT14_COMPONENTS)), dtype=float)
        observed_demand_cells = sum(
            _finite(task.get("demands", {}).get(component)) is not None
            for task in tasks for component in POINT14_COMPONENTS
        )
        demand_rank = int(np.linalg.matrix_rank(matrix)) if matrix.size else 0
        demand_condition = (
            float(np.linalg.cond(matrix)) if matrix.size and demand_rank else None
        )
        task_demands = {
            "matrix": matrix.tolist(),
            "component_order": list(POINT14_COMPONENTS),
            "rank": demand_rank,
            "maximum_rank": min(matrix.shape) if matrix.ndim == 2 else 0,
            "condition_number": demand_condition if demand_condition is None or math.isfinite(demand_condition) else None,
            "observed_demand_cells": observed_demand_cells,
            "identifiability_warning": bool(tasks) and demand_rank < min(matrix.shape),
        }
        self._record(trace, 2, task_demands)

        # 3. Confirm basic access before attributing later-level errors.
        access = {
            "perceptual_access": _mean(
                _score(task) for task in tasks
                if task.get("stage") in {"perceptual", "sensory", "word_recognition"}
            ),
            "lexical_access": _mean(
                _score(task) for task in tasks
                if task.get("stage") in {"lexical", "vocabulary", "word_meaning"}
            ),
            "syntactic_parsing": _mean(
                _score(task) for task in tasks
                if task.get("stage") in {"syntax", "sentence_grammar"}
            ),
            "modality_scores": {},
        }
        for modality in sorted({str(task.get("modality")) for task in tasks if task.get("modality")}):
            access["modality_scores"][modality] = _mean(
                _score(task) for task in tasks if task.get("modality") == modality
            )
        access["basic_access_unmeasured"] = all(
            access[key] is None for key in ("perceptual_access", "lexical_access", "syntactic_parsing")
        )
        self._record(trace, 3, access)

        # 4. Compare explicit/literal meaning with inferred meaning.
        literal = _mean(
            _score(task) for task in tasks
            if task.get("meaning_level", task.get("task_type")) in {"literal", "explicit"}
        )
        inferred = _mean(
            _score(task) for task in tasks
            if task.get("meaning_level", task.get("task_type")) in {"inference", "pragmatic", "implied"}
        )
        inference_cost = literal - inferred if literal is not None and inferred is not None else None
        literal_inference = {
            "literal_comprehension": literal,
            "inferred_meaning_comprehension": inferred,
            "inference_cost": inference_cost,
            "comparison_available": inference_cost is not None,
        }
        self._record(trace, 4, literal_inference)

        # 5. Combine cue weights only when there is measured evidence.
        cue_values: dict[str, list[float]] = {}
        for task in tasks:
            raw = task.get("cue_weight_estimates", task.get("cue_weights", {}))
            if isinstance(raw, Mapping):
                for cue, value in raw.items():
                    number = _finite(value)
                    if number is not None and number >= 0:
                        cue_values.setdefault(str(cue), []).append(number)
        cue_means = {cue: float(np.mean(values)) for cue, values in cue_values.items()}
        cue_total = sum(cue_means.values())
        cue_weights = {
            cue: value / cue_total for cue, value in cue_means.items()
        } if cue_total > 0 else {}
        cue_profile = {
            "mean_cue_estimates": cue_means,
            "normalized_cue_weights": cue_weights,
            "cue_weight_evidence_count": {cue: len(values) for cue, values in cue_values.items()},
            "available": bool(cue_weights),
        }
        self._record(trace, 5, cue_profile)

        # 6. Keep social-state and nonsocial causal reasoning distinct.
        social = _mean(
            _score(task) for task in tasks
            if task.get("inference_domain") in {"social", "belief", "intention", "emotion"}
        )
        nonsocial = _mean(
            _score(task) for task in tasks
            if task.get("inference_domain") in {"nonsocial", "physical", "causal"}
        )
        explicit_speaker = _mean(
            _score(task) for task in tasks if task.get("speaker_knowledge_explicit") is True
        )
        implicit_speaker = _mean(
            _score(task) for task in tasks if task.get("speaker_knowledge_explicit") is False
        )
        social_nonsocial = {
            "social_inference": social,
            "nonsocial_inference": nonsocial,
            "explicit_speaker_knowledge_performance": explicit_speaker,
            "implicit_speaker_knowledge_performance": implicit_speaker,
            "explicit_knowledge_benefit": (
                explicit_speaker - implicit_speaker
                if explicit_speaker is not None and implicit_speaker is not None else None
            ),
            "theory_of_mind_not_assumed_for_all_pragmatics": True,
        }
        self._record(trace, 6, social_nonsocial)

        # 7. Measure discourse links and interference without treating semantic similarity as truth.
        graph_scores = []
        for task in tasks:
            graph = task.get("discourse_graph", {})
            if not isinstance(graph, Mapping):
                continue
            expected = _edges(graph.get("required_edges", graph.get("expected_edges", [])))
            produced = _edges(graph.get("produced_edges", graph.get("observed_edges", [])))
            if expected:
                correct_edges = expected & produced
                graph_scores.append({
                    "task_id": task["task_id"],
                    "precision": len(correct_edges) / len(produced) if produced else 0.0,
                    "recall": len(correct_edges) / len(expected),
                    "missing_required_edges": sorted(expected - produced),
                    "unsupported_edges": sorted(produced - expected),
                })
        distractor_scores = {
            state: _mean(_score(task) for task in tasks if task.get("distractor_present") is state)
            for state in (True, False)
        }
        discourse = {
            "graph_task_scores": graph_scores,
            "mean_edge_precision": _mean(item["precision"] for item in graph_scores),
            "mean_edge_recall": _mean(item["recall"] for item in graph_scores),
            "with_distractor": distractor_scores[True],
            "without_distractor": distractor_scores[False],
            "distractor_cost": (
                distractor_scores[False] - distractor_scores[True]
                if all(distractor_scores[state] is not None for state in (True, False)) else None
            ),
        }
        self._record(trace, 7, discourse)

        # 8. Quantify load effects and interpretation-switch demand.
        load_points = [
            (_finite(task.get("memory_load")), _score(task))
            for task in tasks
            if _finite(task.get("memory_load")) is not None and _score(task) is not None
        ]
        load_slope = _slope([(x, y) for x, y in load_points if x is not None and y is not None])
        load_values = sorted({x for x, _ in load_points if x is not None})
        low_load = _mean(y for x, y in load_points if x == load_values[0]) if load_values else None
        high_load = _mean(y for x, y in load_points if x == load_values[-1]) if load_values else None
        executive = {
            "memory_load_accuracy_slope": load_slope,
            "low_load_performance": low_load,
            "high_load_performance": high_load,
            "high_minus_low_load": high_load - low_load if high_load is not None and low_load is not None else None,
            "switch_cost": _mean(
                task.get("switch_cost") for task in tasks if task.get("switch_cost") is not None
            ),
            "executive_load_observation_count": len(load_points),
        }
        self._record(trace, 8, executive)

        # 9-13. Context-specific views are enabled by supplied context or explicitly tagged tasks.
        context_aliases = {
            "right_hemisphere": {"right_hemisphere", "right_hemisphere_injury", "rhd"},
            "autism_related": {"autism_related", "autism", "asd"},
            "psychosis_spectrum": {"psychosis_spectrum", "schizophrenia_spectrum", "psychosis"},
            "neurodegenerative": {"neurodegenerative", "dementia"},
            "aphasia": {"aphasia", "aphasia_related"},
        }
        requested_contexts = {item.lower().replace("-", "_") for item in request.clinical_contexts}
        family_profiles: dict[str, Any] = {}
        for index, family in enumerate(POINT14_PROFILE_FAMILIES, start=9):
            aliases = context_aliases[family]
            selected = [
                task for task in tasks
                if str(task.get("profile_family", task.get("profile_context", ""))).lower().replace("-", "_") in aliases
            ]
            enabled = bool(requested_contexts & aliases) or bool(selected)
            family_profiles[family] = {
                "available": enabled and bool(selected),
                "context_was_explicitly_supplied": bool(requested_contexts & aliases),
                "task_count": len(selected),
                "performance_by_task_type": {
                    task_type: _mean(
                        _score(task) for task in selected
                        if task.get("task_type") == task_type
                    )
                    for task_type in sorted({str(task.get("task_type")) for task in selected if task.get("task_type")})
                },
                "task_ids": [task["task_id"] for task in selected],
                "heterogeneity_preserved": True,
                "diagnostic_inference_performed": False,
            }
            if family == "right_hemisphere":
                family_profiles[family]["sentence_vs_discourse_contrast"] = (
                    family_profiles[family]["performance_by_task_type"].get("sentence")
                    - family_profiles[family]["performance_by_task_type"].get("discourse")
                    if "sentence" in family_profiles[family]["performance_by_task_type"]
                    and "discourse" in family_profiles[family]["performance_by_task_type"] else None
                )
            if family == "autism_related":
                family_profiles[family]["cue_profile_is_individual_and_task_specific"] = True
                family_profiles[family]["communication_preferences_recorded"] = any(
                    task.get("communication_preferences") is not None for task in selected
                )
            if family == "psychosis_spectrum":
                family_profiles[family]["comprehension_and_production_separated"] = True
                family_profiles[family]["comprehension_score"] = _mean(
                    _score(task) for task in selected if task.get("response_mode") != "production"
                )
                family_profiles[family]["production_score"] = _mean(
                    _score(task) for task in selected if task.get("response_mode") == "production"
                )
            if family == "neurodegenerative":
                family_profiles[family]["component_trajectories_are_separate"] = True
            if family == "aphasia":
                family_profiles[family]["linguistic_score"] = _mean(
                    _score(task) for task in selected if task.get("presentation") == "linguistic"
                )
                family_profiles[family]["nonlinguistic_score"] = _mean(
                    _score(task) for task in selected if task.get("presentation") == "nonlinguistic"
                )
                family_profiles[family]["language_concept_dissociation"] = (
                    family_profiles[family]["linguistic_score"] - family_profiles[family]["nonlinguistic_score"]
                    if family_profiles[family]["linguistic_score"] is not None
                    and family_profiles[family]["nonlinguistic_score"] is not None else None
                )
            self._record(trace, index, family_profiles[family])

        # 12 is also the component-specific longitudinal trajectory analysis.
        trajectory_points: dict[str, list[tuple[float, float]]] = {}
        for item in request.longitudinal_observations:
            component = item.get("component")
            time_value = _finite(item.get("time"))
            score_value = _finite(item.get("score"))
            if component in POINT14_COMPONENTS and time_value is not None and score_value is not None:
                trajectory_points.setdefault(str(component), []).append((time_value, min(1.0, max(0.0, score_value))))
        trajectories = {}
        for component, points in trajectory_points.items():
            trajectories[component] = {
                "available": _slope(points) is not None,
                "slope_per_time_unit": _slope(points),
                "observation_count": len(points),
                "distinct_time_points": len({time for time, _ in points}),
                "time_ordered_points": sorted(points),
            }
        family_profiles["neurodegenerative"]["component_trajectories"] = trajectories

        # 14. Combine repeated direct measures and task-derived component evidence.
        components, component_uncertainty = self._components(request, tasks)
        uncertainty.update(component_uncertainty)
        measured = {key: value for key, value in components.items() if value is not None}
        ranked = sorted(measured, key=measured.get)
        split_at = max(1, min(len(ranked), int(math.ceil(len(ranked) * self.relative_split)))) if ranked else 0
        component_profile = {
            "components": components,
            "relatively_lower_measured_components": ranked[:split_at],
            "relatively_higher_measured_components": ranked[split_at:],
            "relative_ranking_is_not_a_normative_or_diagnostic_cutoff": True,
            "unmeasured_components_remain_unknown": True,
        }
        self._record(trace, 14, component_profile)

        # 15. Score candidate explanations only when explicit predictions are supplied.
        scored_hypotheses = []
        task_by_id = {str(task["task_id"]): task for task in tasks}
        for hypothesis in request.candidate_explanations:
            predictions = hypothesis.get("predictions", {})
            errors = []
            if isinstance(predictions, Mapping):
                for task_id, probability in predictions.items():
                    task = task_by_id.get(str(task_id))
                    predicted = _finite(probability)
                    observed = _score(task) if task is not None else None
                    if predicted is not None and observed is not None:
                        errors.append((min(1.0, max(0.0, predicted)) - observed) ** 2)
            scored_hypotheses.append({
                "hypothesis_id": str(hypothesis.get("hypothesis_id", f"hypothesis-{len(scored_hypotheses) + 1}")),
                "description": str(hypothesis.get("description", "")),
                "brier_score": _mean(errors),
                "prediction_count": len(errors),
                "complexity": _finite(hypothesis.get("complexity")),
                "score_available": bool(errors),
            })
        scored_hypotheses.sort(
            key=lambda item: item["brier_score"] if item["brier_score"] is not None else math.inf
        )
        self._record(trace, 15, {
            "scored_hypotheses": scored_hypotheses,
            "unresolved_when_predictions_are_missing": True,
            "best_supported_hypothesis": next(
                (item["hypothesis_id"] for item in scored_hypotheses if item["score_available"]), None
            ),
        })

        # 16. Estimate matched support benefits using pair identifiers and explicit support flags.
        pairs: dict[str, dict[bool, list[float]]] = {}
        for task in tasks:
            pair_id = task.get("pair_id")
            support_flag = task.get("support_present")
            value = _score(task)
            if pair_id is not None and isinstance(support_flag, bool) and value is not None:
                pairs.setdefault(str(pair_id), {True: [], False: []})[support_flag].append(value)
        support_comparisons = []
        for pair_id, conditions in pairs.items():
            with_support = _mean(conditions[True])
            without_support = _mean(conditions[False])
            if with_support is not None and without_support is not None:
                support_comparisons.append({
                    "pair_id": pair_id,
                    "with_support": with_support,
                    "without_support": without_support,
                    "benefit": with_support - without_support,
                })
        support_predictions = {
            "matched_comparisons": support_comparisons,
            "mean_support_benefit": _mean(item["benefit"] for item in support_comparisons),
            "generalization_items": [
                task["task_id"] for task in tasks
                if task.get("support_generalization") is True
            ],
            "benefit_does_not_prove_mechanism_repair": True,
        }
        self._record(trace, 16, support_predictions)

        # 17. Use only explicit independent validation observations.
        held_out = [
            task for task in tasks
            if task.get("split") in {"validation", "held_out", "test"}
            and _finite(task.get("predicted_probability")) is not None
            and _score(task) is not None
        ]
        validation_errors = [
            (_finite(task["predicted_probability"]) - _score(task)) ** 2
            for task in held_out
        ]
        validation = {
            "validation_n": len(held_out),
            "held_out_mse": _mean(validation_errors),
            "held_out_brier_score": _mean(validation_errors),
            "held_out_accuracy": _mean(
                float((_finite(task["predicted_probability"]) >= 0.5) == (_score(task) >= 0.5))
                for task in held_out
            ),
            "validation_available": bool(held_out),
            "neural_data_supplied": bool(request.optional_neural_data),
            "neural_evidence_interpreted_without_reverse_inference": True,
            "independent_validation_required": True,
        }
        self._record(trace, 17, validation)

        # 18. Keep previous estimates historical unless a current measurement exists.
        learning_rate = _finite(request.metadata.get("learning_rate"))
        learning_rate = min(1.0, max(0.0, learning_rate if learning_rate is not None else 0.5))
        previous_components = request.previous_profile.get("components", {})
        previous_components = previous_components if isinstance(previous_components, Mapping) else {}
        historical = {
            component: _finite(previous_components.get(component))
            for component in POINT14_COMPONENTS
            if _finite(previous_components.get(component)) is not None
        }
        updated = {}
        for component, current in components.items():
            prior = _finite(previous_components.get(component))
            if current is not None:
                updated[component] = (
                    current if prior is None
                    else prior + learning_rate * (current - prior)
                )
        adaptation = {
            "historical_previous_components": historical,
            "updated_components_with_current_evidence_only": updated,
            "components_without_current_evidence_remain_unknown": [
                component for component in POINT14_COMPONENTS if components[component] is None
            ],
            "learning_rate": learning_rate,
            "update_is_bounded": True,
            "previous_estimates_are_not_current_measurements": True,
        }
        self._record(trace, 18, adaptation)

        # 19. Assemble an explanatory report that preserves uncertainty and alternatives.
        measured_order = ranked
        report = {
            "participant_id": request.participant_id,
            "target_outcome": request.target_outcome,
            "tasks_tested": [task["task_id"] for task in tasks],
            "components_with_converging_measures": [
                component for component in POINT14_COMPONENTS
                if uncertainty["component_measurement_counts"][component] >= 2
            ],
            "relatively_lower_measured_components": component_profile["relatively_lower_measured_components"],
            "relatively_higher_measured_components": component_profile["relatively_higher_measured_components"],
            "components_without_sufficient_measurement": uncertainty["missing_components"],
            "alternative_explanations": scored_hypotheses,
            "support_effect": support_predictions["mean_support_benefit"],
            "validation_available": validation["validation_available"],
            "confidence_is_not_a_diagnosis": True,
            "limitations": [
                "This is a task-specific research profile, not a clinical diagnosis.",
                "Group-level findings do not establish an individual's neural organization.",
                "Task confounding and measurement reliability limit component attribution.",
                "Biological descriptions are distributed functional hypotheses, not one-to-one region labels.",
            ],
            "recommended_next_measurements": [
                component for component in POINT14_COMPONENTS if components[component] is None
            ],
            "relative_ordering_only": measured_order,
        }
        self._record(trace, 19, report)

        profile = {
            "functional_components": component_profile,
            "task_demands": task_demands,
            "basic_access": access,
            "literal_vs_inferred_meaning": literal_inference,
            "cue_integration": cue_profile,
            "social_vs_nonsocial_inference": social_nonsocial,
            "discourse_coherence": discourse,
            "working_memory_and_executive_load": executive,
            "context_specific_profiles": family_profiles,
            "longitudinal_trajectories": trajectories,
            "profile_update": adaptation,
            "transparent_report": report,
            "clinical_contexts_are_descriptive_only": list(request.clinical_contexts),
            "diagnosis_not_performed": True,
        }
        uncertainty["profile_confidence_is_component_specific"] = True
        uncertainty["unmeasured_is_not_zero"] = True
        return Point14Result(
            participant_id=request.participant_id,
            target_outcome=request.target_outcome,
            profile=profile,
            candidate_explanations=scored_hypotheses,
            uncertainty_estimates=uncertainty,
            support_predictions=support_predictions,
            validation=validation,
            trace=trace,
            diagnostics={
                "valid": True,
                "step_count": len(trace),
                "component_count": len(POINT14_COMPONENTS),
                "profile_family_count": len(POINT14_PROFILE_FAMILIES),
                "clinical_diagnosis_not_performed": True,
                "neural_equations_are_not_asserted": True,
                "missing_data_preserved": True,
            },
        )


def register_point14_annotations(
    repository_root: str | Path | None = None,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Index every public Point 14 class, function, and method in ORE's shared SQLite catalogue.

    Scoped synchronization refreshes only ore_language_comprehension, preserving annotations
    owned by unrelated packages and reusing the repository-wide ore_logger.annotations table.
    """
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
