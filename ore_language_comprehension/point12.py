"""Point 12: methodological evaluation and model selection for language neuroscience.

Researcher-facing workflow: statistical operations analyze supplied studies and are not
presented as computations performed by the brain.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np


_NAMES = (
    "Define observable phenomenon", "Generate competing hypotheses", "Audit causal structure",
    "Evaluate contrast identifiability", "Preserve participant and item records",
    "Separate spatial temporal and functional claims", "Interpret ERP patterns conservatively",
    "Test language selectivity", "Evaluate network boundaries", "Separate association from causation",
    "Constrain neural interpretation with behavior", "Audit population generalization",
    "Prevent overfitting and selective reporting", "Evaluate out-of-sample prediction",
    "Compare candidate models", "Test boundary conditions", "Inspect individual heterogeneity",
    "Audit measurement quality and missingness", "Evaluate evidence dependence", "Check model adequacy",
    "Calibrate claim strength", "Preserve uncertainty and alternatives",
    "Choose discriminating experiments", "Construct calibrated mechanistic account",
    "Return auditable research report",
)
POINT12_METHODS = ("behavior", "fmri", "eeg", "meg", "connectivity", "lesion", "perturbation")
POINT12_STEP_SPECS = tuple({
    "operation_id": f"LC-P12-{i:02d}", "name": name,
    "objective": name + " using only supplied observations and design metadata.",
    "mathematical_operations": ("P(H|D) ∝ P(D|H)P(H)", "MSE=mean((y-yhat)^2)", "G=(V,E,W)"),
    "biological_reference": "Researcher-side analysis; not a literal neuronal calculation.",
} for i, name in enumerate(_NAMES, 1))


def _plain(value: Any) -> Any:
    """Convert arrays, NumPy scalars, dataclasses, and nested containers to plain Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    """Validate a mapping input and return a detached dictionary snapshot."""
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return deepcopy(dict(value))


def _finite(value: Any) -> np.ndarray:
    """Return finite numeric observations as a one-dimensional array."""
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    return arr[np.isfinite(arr)]


def _mean(value: Any) -> float | None:
    """Calculate a finite-value mean or return None when observations are absent."""
    arr = _finite(value)
    return float(np.mean(arr)) if arr.size else None


def _dag_audit(edges: Sequence[Any]) -> dict[str, Any]:
    """Detect cycles in a supplied directed graph using Kahn's topological-sort algorithm."""
    graph: dict[str, set[str]] = {}
    degree: dict[str, int] = {}
    clean = []
    for edge in edges:
        if isinstance(edge, Mapping):
            a, b = str(edge.get("source", "")).strip(), str(edge.get("target", "")).strip()
        elif isinstance(edge, (list, tuple)) and len(edge) == 2:
            a, b = str(edge[0]).strip(), str(edge[1]).strip()
        else:
            continue
        if not a or not b:
            continue
        graph.setdefault(a, set())
        graph.setdefault(b, set())
        degree.setdefault(a, 0)
        degree.setdefault(b, 0)
        if b not in graph[a]:
            graph[a].add(b)
            degree[b] += 1
        clean.append({"source": a, "target": b})
    queue = [n for n, d in degree.items() if d == 0]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for target in sorted(graph[node]):
            degree[target] -= 1
            if degree[target] == 0:
                queue.append(target)
    cyclic = len(order) != len(degree)
    return {"edges": clean, "node_count": len(degree), "acyclic": not cyclic,
            "cycle_detected": cyclic, "topological_order": [] if cyclic else order,
            "direction_not_inferred_from_correlation": True}


@dataclass(frozen=True, slots=True)
class Point12Request:
    """Immutable snapshot of a research-methodology evaluation request."""
    target_phenomenon: str
    observable_target: Mapping[str, Any]
    candidate_hypotheses: Sequence[Mapping[str, Any]]
    conditions: Mapping[str, Any] = field(default_factory=dict)
    behavioral_data: Mapping[str, Any] = field(default_factory=dict)
    neural_data: Mapping[str, Any] = field(default_factory=dict)
    participant_metadata: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    analysis_plan: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate required inputs and deep-copy nested structures to prevent external mutation."""
        if not str(self.target_phenomenon).strip():
            raise ValueError("target_phenomenon must be non-empty")
        if not isinstance(self.observable_target, Mapping):
            raise TypeError("observable_target must be a mapping")
        if not isinstance(self.candidate_hypotheses, Sequence) or isinstance(self.candidate_hypotheses, (str, bytes)):
            raise TypeError("candidate_hypotheses must be a sequence of mappings")
        hypotheses = tuple(_mapping(h, f"candidate_hypotheses[{i}]")
                           for i, h in enumerate(self.candidate_hypotheses))
        if not hypotheses:
            raise ValueError("at least one candidate hypothesis is required")
        for key in ("conditions", "behavioral_data", "neural_data", "analysis_plan", "metadata"):
            object.__setattr__(self, key, _mapping(getattr(self, key), key))
        participants = tuple(_mapping(p, f"participant_metadata[{i}]")
                             for i, p in enumerate(self.participant_metadata))
        object.__setattr__(self, "observable_target", deepcopy(dict(self.observable_target)))
        object.__setattr__(self, "candidate_hypotheses", hypotheses)
        object.__setattr__(self, "participant_metadata", participants)


@dataclass(slots=True)
class Point12Result:
    """Auditable 25-step result with separate evidence, confounds, uncertainty, and hypotheses."""
    target_phenomenon: str
    hypotheses: list[dict[str, Any]]
    ranked_hypotheses: list[dict[str, Any]]
    evidence_by_method: dict[str, Any]
    confound_audit: dict[str, Any]
    design_audit: dict[str, Any]
    generalization_report: dict[str, Any]
    uncertainty_report: dict[str, Any]
    next_experiments: list[dict[str, Any]]
    mechanistic_account: dict[str, Any]
    trace: list[dict[str, Any]]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible result."""
        return _plain(asdict(self))


class Point12ModelEvaluator:
    """Evaluate competing language-neuroscience hypotheses against supplied study records."""

    def __init__(self, minimum_temporal_trials: int = 20) -> None:
        """Set the descriptive minimum EEG/MEG trial count; reject non-positive thresholds."""
        if minimum_temporal_trials < 1:
            raise ValueError("minimum_temporal_trials must be positive")
        self.minimum_temporal_trials = int(minimum_temporal_trials)

    def _record(self, trace: list[dict[str, Any]], step: int, result: Mapping[str, Any]) -> None:
        """Append a detached result using the stable source-ordered operation identifier."""
        spec = POINT12_STEP_SPECS[step - 1]
        trace.append({"step": step, "operation_id": spec["operation_id"],
                      "name": spec["name"], "result": _plain(dict(result))})

    def _hypotheses(self, items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Normalize hypotheses while preserving predictions, falsifiers, boundaries, and supplied likelihoods."""
        out = []
        for i, item in enumerate(items):
            predictions = item.get("predictions", [])
            falsifiers = item.get("falsifiers", item.get("disconfirming_predictions", []))
            out.append({"hypothesis_id": str(item.get("hypothesis_id", item.get("name", f"H{i+1}"))),
                "operation": str(item.get("operation", "")),
                "predictions": _plain(list(predictions) if isinstance(predictions, (list, tuple)) else []),
                "falsifiers": _plain(list(falsifiers) if isinstance(falsifiers, (list, tuple)) else []),
                "boundary_conditions": _plain(item.get("boundary_conditions", [])),
                "prior_probability": item.get("prior_probability"),
                "likelihoods": _plain(item.get("likelihoods", {})),
                "falsifiable": bool(predictions and falsifiers)})
        return out

    def _design(self, conditions: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
        """Compare target/control feature summaries and identify unmatched non-manipulated dimensions."""
        target, control = conditions.get("target", {}), conditions.get("control", {})
        target = target if isinstance(target, Mapping) else {}
        control = control if isinstance(control, Mapping) else {}
        contrasts = {}
        for key in sorted(set(target) | set(control)):
            a, b = _mean(target.get(key, [])), _mean(control.get(key, []))
            contrasts[str(key)] = {"target_mean": a, "control_mean": b,
                                   "difference": a-b if a is not None and b is not None else None}
        manipulated, matched = list(plan.get("manipulated_variables", [])), list(plan.get("matched_variables", []))
        unmatched = [k for k, v in contrasts.items()
                     if v["difference"] not in (None, 0.0) and k not in manipulated]
        confounds = [k for k in unmatched if k not in matched]
        return {"feature_contrasts": contrasts, "manipulated_variables": manipulated,
            "matched_variables": matched, "unmatched_nonmanipulated_features": unmatched,
            "potentially_confounded_features": confounds,
            "contrast_identifiability": "limited" if confounds else "no_numeric_mismatch_detected",
            "factorial_design": bool(plan.get("factorial_design", False)),
            "participant_item_effects_planned": bool(plan.get("participant_and_item_effects", False))}


    def _evidence(self, behavior: Mapping[str, Any], neural: Mapping[str, Any]) -> dict[str, Any]:
        """Summarize behavioral and neural evidence while keeping measurement types distinct."""
        target = behavior.get("target", {}) if isinstance(behavior.get("target", {}), Mapping) else {}
        control = behavior.get("control", {}) if isinstance(behavior.get("control", {}), Mapping) else {}
        ta, ca = _mean(target.get("accuracy", [])), _mean(control.get("accuracy", []))
        tr, cr = _mean(target.get("response_time", [])), _mean(control.get("response_time", []))
        out = {"behavior": {"available": bool(target or control), "target_accuracy": ta,
            "control_accuracy": ca, "accuracy_difference": ta-ca if ta is not None and ca is not None else None,
            "target_rt": tr, "control_rt": cr, "response_time_difference": tr-cr if tr is not None and cr is not None else None,
            "interpretation": "behavioral outcome; does not localize a neural mechanism"}}
        for method in ("fmri", "eeg", "meg"):
            data = neural.get(method, {})
            data = data if isinstance(data, Mapping) else {}
            a, b = _finite(data.get("target", [])), _finite(data.get("control", []))
            trials = min(int(data.get("target_trial_count", a.size)), int(data.get("control_trial_count", b.size)))
            out[method] = {"available": bool(a.size or b.size), "target_n": int(a.size), "control_n": int(b.size),
                "target_mean": _mean(a), "control_mean": _mean(b),
                "contrast": _mean(a)-_mean(b) if a.size and b.size else None,
                "trial_count_lower_bound": trials,
                "sufficient_trials": trials >= self.minimum_temporal_trials if method in ("eeg", "meg") else None,
                "artifact_controlled": bool(data.get("artifact_controlled", False)),
                "independent_localizer": bool(data.get("independent_localizer", data.get("independent_roi", False))),
                "motion_controlled": bool(data.get("motion_controlled", False)),
                "multiple_comparison_control": bool(data.get("multiple_comparison_control", False)),
                "component_label": data.get("component_label", "unspecified"),
                "interpretation": "measured condition response; not a unique process label"}
        conn = neural.get("connectivity", {})
        conn = conn if isinstance(conn, Mapping) else {}
        edges = conn.get("edges", [])
        within, between = [], []
        for edge in edges if isinstance(edges, (list, tuple)) else []:
            if not isinstance(edge, Mapping):
                continue
            try:
                source = str(edge.get("source_network", edge.get("source", "")))
                target_name = str(edge.get("target_network", edge.get("target", "")))
                weight = float(edge.get("weight"))
            except (TypeError, ValueError):
                continue
            (within if source and source == target_name else between).append(weight)
        out["connectivity"] = {"available": bool(within or between), "edge_count": len(within)+len(between),
            "within_network_mean": float(np.mean(within)) if within else None,
            "between_network_mean": float(np.mean(between)) if between else None,
            "parcellation": conn.get("parcellation", "unspecified"),
            "interpretation": "statistical network relationship; not direct anatomical wiring"}
        for method in ("lesion", "perturbation"):
            data = neural.get(method, {})
            data = data if isinstance(data, Mapping) else {}
            out[method] = {"available": bool(data), "supplied_evidence": _plain(data),
                "interpretation": "potential causal contribution subject to design assumptions" if data else "not supplied"}
        return out

    def _comparison(self, hypotheses: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate posterior weights only when every candidate supplies comparable valid likelihoods."""
        keysets = [set(h["likelihoods"]) if isinstance(h["likelihoods"], Mapping) else set() for h in hypotheses]
        comparable = bool(keysets and keysets[0] and all(keys == keysets[0] for keys in keysets))
        priors, log_likelihoods = [], []
        if comparable:
            for hypothesis in hypotheses:
                prior = float(hypothesis["prior_probability"]) if hypothesis["prior_probability"] is not None else 1/len(hypotheses)
                likelihoods = [float(hypothesis["likelihoods"][k]) for k in sorted(keysets[0])]
                if not math.isfinite(prior) or prior <= 0 or any(not math.isfinite(x) or x <= 0 or x > 1 for x in likelihoods):
                    comparable = False
                    break
                priors.append(prior)
                log_likelihoods.append(sum(math.log(x) for x in likelihoods))
        if not comparable:
            return {"available": False, "reason": "comparable explicit likelihoods for every hypothesis were not supplied",
                "posterior_probabilities": {}, "ranked_hypotheses": [], "likelihood_source": "caller_supplied_only"}
        total_prior = sum(priors)
        logw = [math.log(p/total_prior)+ll for p, ll in zip(priors, log_likelihoods)]
        peak = max(logw)
        weights = [math.exp(x-peak) for x in logw]
        total = sum(weights)
        posterior = {h["hypothesis_id"]: w/total for h, w in zip(hypotheses, weights)}
        return {"available": True, "posterior_probabilities": posterior,
            "ranked_hypotheses": sorted(posterior, key=posterior.get, reverse=True),
            "likelihood_source": "caller_supplied_only", "dataset_ids": sorted(keysets[0]),
            "posterior_is_not_neural_measurement": True}

    def _generalization(self, participants: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any],
                        plan: Mapping[str, Any]) -> dict[str, Any]:
        """Describe sample diversity and score held-out predictions only when paired values are supplied."""
        groups: dict[str, set[str]] = {}
        for person in participants:
            for key in ("language_background", "culture", "developmental_group", "age_group"):
                if person.get(key) is not None:
                    groups.setdefault(key, set()).add(str(person[key]))
        observed, predicted = _finite(plan.get("held_out_observed", [])), _finite(plan.get("held_out_predicted", []))
        valid = bool(observed.size and observed.size == predicted.size)
        error = observed-predicted if valid else np.asarray([], dtype=float)
        corr = None
        if valid and observed.size >= 3 and np.std(observed) and np.std(predicted):
            corr = float(np.corrcoef(observed, predicted)[0, 1])
        return {"sample_description": _plain(metadata.get("sample_description", {})),
            "participant_count": len(participants), "group_variation": {k: sorted(v) for k, v in groups.items()},
            "held_out_available": valid, "held_out_n": int(observed.size) if valid else 0,
            "mse": float(np.mean(error**2)) if valid else None,
            "mae": float(np.mean(np.abs(error))) if valid else None, "correlation": corr,
            "model_was_frozen": bool(plan.get("model_was_frozen", False)),
            "independent_test_set": bool(plan.get("independent_test_set", False)),
            "generalization_supported": bool(valid and plan.get("model_was_frozen") and plan.get("independent_test_set")),
            "cultural_or_language_universality_assumed": False}

    def process(self, request: Point12Request) -> Point12Result:
        """Execute the complete 25-step research workflow without inventing missing study evidence."""
        if not isinstance(request, Point12Request):
            raise TypeError("request must be a Point12Request")
        trace: list[dict[str, Any]] = []
        plan, neural = request.analysis_plan, request.neural_data
        hypotheses = self._hypotheses(request.candidate_hypotheses)
        observable = {"phenomenon": request.target_phenomenon, "target": _plain(request.observable_target),
            "measure": request.observable_target.get("measure"),
            "comprehension_separated_from_response": bool(request.observable_target.get("separates_comprehension_from_response", False))}
        self._record(trace, 1, observable)
        self._record(trace, 2, {"hypotheses": hypotheses, "all_falsifiable": all(h["falsifiable"] for h in hypotheses)})
        dag = _dag_audit(plan.get("causal_edges", []))
        self._record(trace, 3, dag)
        design = self._design(request.conditions, plan)
        self._record(trace, 4, design)
        expected = int(request.metadata.get("expected_participant_count", len(request.participant_metadata)))
        participants = {"observed_records": len(request.participant_metadata), "expected_records": expected,
            "coverage": len(request.participant_metadata)/expected if expected > 0 else None,
            "individual_records_preserved": True, "group_aggregation_performed": False}
        self._record(trace, 5, participants)
        evidence = self._evidence(request.behavioral_data, neural)
        self._record(trace, 6, {"claims_separated": True,
            "spatial": evidence["fmri"], "temporal": {"eeg": evidence["eeg"], "meg": evidence["meg"]},
            "functional": evidence["connectivity"]})
        self._record(trace, 7, {"eeg_component": evidence["eeg"]["component_label"],
            "meg_component": evidence["meg"]["component_label"], "component_is_unique_process_label": False,
            "trial_sufficiency": {"eeg": evidence["eeg"]["sufficient_trials"], "meg": evidence["meg"]["sufficient_trials"]}})
        profile = neural.get("task_response_profile", {})
        profile = profile if isinstance(profile, Mapping) else {}
        lang, nonlang = _mean(profile.get("language", [])), _mean(profile.get("non_language_control", []))
        self._record(trace, 8, {"language_mean": lang, "nonlanguage_mean": nonlang,
            "selectivity_contrast": lang-nonlang if lang is not None and nonlang is not None else None,
            "absolute_exclusivity_claimed": False})
        self._record(trace, 9, evidence["connectivity"])
        self._record(trace, 10, {"lesion": evidence["lesion"], "perturbation": evidence["perturbation"],
            "coactivation_is_not_causation": True, "causal_claim_requires_design_review": True})
        self._record(trace, 11, {"behavior": evidence["behavior"],
            "behavioral_outcome_does_not_uniquely_identify_neural_cause": True})
        generalization = self._generalization(request.participant_metadata, request.metadata, plan)
        self._record(trace, 12, generalization)
        flags = []
        if not observable["measure"]:
            flags.append("observable_outcome_measure_not_specified")
        if not all(h["falsifiable"] for h in hypotheses):
            flags.append("hypothesis_missing_prediction_or_falsifier")
        if dag["cycle_detected"]:
            flags.append("causal_graph_contains_cycle")
        flags.extend("potential_confound:"+name for name in design["potentially_confounded_features"])
        if not plan.get("preregistered", False):
            flags.append("analysis_not_marked_preregistered")
        if plan.get("exploratory_only", False):
            flags.append("analysis_is_exploratory")
        if not plan.get("multiple_comparison_control", False):
            flags.append("multiple_comparison_control_not_documented")
        if evidence["fmri"]["available"] and not evidence["fmri"]["independent_localizer"]:
            flags.append("fmri_independent_localizer_not_documented")
        for method in ("eeg", "meg"):
            if evidence[method]["available"]:
                if not evidence[method]["sufficient_trials"]:
                    flags.append(method+"_trial_count_below_configured_minimum")
                if not evidence[method]["artifact_controlled"]:
                    flags.append(method+"_artifact_control_not_documented")
        if generalization["held_out_available"] and not generalization["generalization_supported"]:
            flags.append("held_out_independence_or_model_freeze_not_established")
        self._record(trace, 13, {"preregistered": bool(plan.get("preregistered", False)),
            "exploratory": bool(plan.get("exploratory_only", False)),
            "multiplicity_control": bool(plan.get("multiple_comparison_control", False)), "flags": list(flags)})
        self._record(trace, 14, generalization)
        comparison = self._comparison(hypotheses)
        self._record(trace, 15, comparison)
        boundaries = {h["hypothesis_id"]: h["boundary_conditions"] for h in hypotheses}
        self._record(trace, 16, {"boundary_conditions": boundaries, "outside_scope_not_assumed": True})
        self._record(trace, 17, {"individual_effects_supplied": bool(request.metadata.get("individual_effects")),
            "participant_count": len(request.participant_metadata), "heterogeneity_preserved": True,
            "population_mean_not_assumed_to_describe_each_person": True})
        self._record(trace, 18, {"data_quality": _plain(request.metadata.get("data_quality", {})),
            "missingness_reasons": _plain(request.metadata.get("missingness_reasons", [])),
            "unreported_quality_not_assumed_good": True})
        source_groups: dict[str, list[str]] = {}
        for method in POINT12_METHODS:
            if evidence[method]["available"]:
                supplied = neural.get(method, {})
                source = supplied.get("independence_group", request.behavioral_data.get("independence_group",
                    request.metadata.get("study_id", "unspecified"))) if isinstance(supplied, Mapping) else "unspecified"
                source_groups.setdefault(str(source), []).append(method)
        self._record(trace, 19, {"methods_by_source_group": source_groups,
            "independent_group_count": len(source_groups),
            "double_counting_warning": any(len(v) > 1 for v in source_groups.values())})
        self._record(trace, 20, {"posterior_comparison_available": comparison["available"],
            "all_candidates_may_be_inadequate": True, "least_bad_model_not_forced": True})
        methods = [m for m in POINT12_METHODS if evidence[m]["available"]]
        claim = "model_comparison_available" if comparison["available"] else (
            "behavioral_or_neural_association_only" if methods else "unresolved_no_evidence")
        self._record(trace, 21, {"claim_level": claim, "association_is_not_necessity": True,
            "causal_claim_not_auto_promoted": True})
        uncertainty = {"quality_control_flags": flags,
            "unresolved_hypotheses": [h["hypothesis_id"] for h in hypotheses if not h["falsifiable"] or not h["likelihoods"]],
            "posterior_probabilities": comparison["posterior_probabilities"],
            "missing_evidence_is_not_evidence_of_absence": True, "unmeasured_confounding_remains_possible": True}
        self._record(trace, 22, uncertainty)
        next_experiments = [{"hypothesis_id": h["hypothesis_id"], "discriminating_test": item}
            for h in hypotheses for item in h["falsifiers"]]
        supplied = request.metadata.get("next_experiments", [])
        if isinstance(supplied, (list, tuple)):
            next_experiments.extend(_plain(supplied))
        self._record(trace, 23, {"candidate_experiments": next_experiments,
            "information_gain_not_computed_without_outcome_model": True})
        account = {"phenomenon": request.target_phenomenon, "evidence_types": methods,
            "hypotheses": [h["hypothesis_id"] for h in hypotheses],
            "ranked_hypotheses": comparison["ranked_hypotheses"], "boundaries": boundaries,
            "claim_level": claim, "research_workflow_not_brain_simulation": True}
        self._record(trace, 24, account)
        report = {"observable_target": observable, "ranked_hypotheses": comparison["ranked_hypotheses"],
            "hypothesis_comparison": comparison, "evidence_by_method": evidence,
            "confound_audit": {"causal_graph": dag, "design": design},
            "generalization": generalization, "uncertainty": uncertainty,
            "next_experiments": next_experiments,
            "scientific_scope": "Statistical analysis is researcher-side, not literal neuronal calculation.",
            "all_candidate_models_may_be_inadequate": True}
        self._record(trace, 25, {"report_sections": list(report), "auditable": True,
            "ranking_requires_comparable_likelihoods": True})
        return Point12Result(request.target_phenomenon, hypotheses, comparison["ranked_hypotheses"],
            evidence, {"causal_graph": dag, "design": design}, design, generalization, uncertainty,
            next_experiments, account, trace,
            {"valid": True, "step_count": len(trace), "method_count": len(POINT12_METHODS),
             "research_analysis_not_brain_simulation": True})


def register_point12_annotations(repository_root: str | None = None,
                                 database_path: str | None = None) -> dict[str, Any]:
    """Index Point 12 public definitions in ORE's shared SQLite annotation catalogue."""
    import os
    from pathlib import Path
    from ore_logger import LoggerDatabase, sync_repository
    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    target = Path(database_path or os.environ.get("ORE_LOGGER_DB") or root/"ore_logs"/"ore_logger.sqlite3").expanduser().resolve()
    report = sync_repository(LoggerDatabase(target), root, include_paths={"ore_language_comprehension"})
    return {**report, "database_path": str(target), "modules": ["ore_language_comprehension"],
        "annotation_source": "python_docstring_and_ast_signature",
        "storage": "ore_logger.annotations SQLite table"}
