"""Point 10: evidence-based identification and validation of language-comprehension mechanisms.

This module implements a reproducible research-analysis workflow. Its statistical operations
analyze supplied study data; they are not presented as calculations performed by the brain.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


POINT10_METHODS = ("behavior", "fmri", "eeg", "meg", "lesion", "stimulation", "computational")
POINT10_STEP_SPECS: tuple[dict[str, Any], ...] = (
    {"operation_id": "LC-P10-01", "name": "Operationalize one mechanism", "objective": "Define a bounded cognitive operation, inputs, outputs, triggers, positive predictions, and disconfirming predictions.", "mathematical_operations": ("M=(inputs,outputs,trigger,observable_predictions)", "P(Y|C1,M) versus P(Y|C0,M)"), "biological_reference": "Research design operation; not a neural computation."},
    {"operation_id": "LC-P10-02", "name": "Construct controlled contrasts", "objective": "Specify target/control conditions, matching, counterbalancing, participant and item variation, and nuisance variables.", "mathematical_operations": ("Y_ij=beta0+beta1*C_ij+u_i+v_j+epsilon_ij", "confound_balance and design completeness"), "biological_reference": "Experimental-design operation; not a neural computation."},
    {"operation_id": "LC-P10-03", "name": "Establish behavioral signature", "objective": "Summarize accuracy, response time, error types, confidence, and selective versus general difficulty.", "mathematical_operations": ("accuracy=N_correct/N_valid", "mean_RT=sum(RT)/N", "Cohen_d=(mean_target-mean_control)/SD_pooled"), "biological_reference": "Behavior is an observed outcome; its summary statistics are computed by the researcher."},
    {"operation_id": "LC-P10-04", "name": "Evaluate spatial fMRI evidence", "objective": "Evaluate supplied BOLD contrasts, localizer independence, ROI provenance, motion, nuisance regressors, and multiplicity safeguards.", "mathematical_operations": ("y_v=X*beta_v+epsilon_v", "delta_BOLD=beta_target-beta_control", "selectivity_index=(R_L-R_C)/(|R_L|+|R_C|+epsilon)"), "biological_reference": "BOLD is an indirect hemodynamic measurement, not a direct readout of computation."},
    {"operation_id": "LC-P10-05", "name": "Evaluate temporal EEG/MEG evidence", "objective": "Summarize event-locked signals, trial counts, artifact control, time windows, and component ambiguity.", "mathematical_operations": ("ERP_c(t)=mean_k EEG_c,k(t)", "delta_ERP(t)=ERP_target(t)-ERP_control(t)"), "biological_reference": "EEG/MEG measure electrical or magnetic fields; source and process interpretations remain inferential."},
    {"operation_id": "LC-P10-06", "name": "Evaluate lesion evidence", "objective": "Relate supplied lesion/disconnection features to performance while checking lesion volume and co-occurring deficits.", "mathematical_operations": ("performance=beta0+beta1*lesion_feature+gamma*covariates+epsilon", "P(deficit)=sigmoid(X*beta)"), "biological_reference": "Lesion associations constrain network contribution but do not prove a one-region/one-function map."},
    {"operation_id": "LC-P10-07", "name": "Evaluate perturbational evidence", "objective": "Compare active stimulation with sham/control-site outcomes and check selectivity and non-specific effects.", "mathematical_operations": ("delta_RT=RT_active-RT_sham", "delta_accuracy=accuracy_active-accuracy_sham"), "biological_reference": "Perturbation can support causal contribution, but stimulation is not perfectly focal."},
    {"operation_id": "LC-P10-08", "name": "Build a method-by-prediction matrix", "objective": "Keep behavioral, spatial, temporal, lesion, perturbational, and computational contributions distinct.", "mathematical_operations": ("evidence_matrix[method,prediction]", "method-specific uncertainty"), "biological_reference": "Cross-method synthesis is a scientific inference, not a biological process."},
    {"operation_id": "LC-P10-09", "name": "Evaluate competing explanations", "objective": "Record alternatives such as difficulty, memory, attention, perception, response selection, and prediction violation.", "mathematical_operations": ("support(H_j|D) compared across explicit alternatives", "discriminating_contrast selection"), "biological_reference": "Alternative-hypothesis testing prevents over-localization."},
    {"operation_id": "LC-P10-10", "name": "Estimate effect size and uncertainty", "objective": "Report effect magnitudes and intervals where supplied; flag multiplicity and exploratory analyses.", "mathematical_operations": ("d=(mean1-mean0)/SD_pooled", "CI=estimate +/- critical_value*SE"), "biological_reference": "Statistical uncertainty quantifies evidence quality, not neural confidence."},
    {"operation_id": "LC-P10-11", "name": "Test individual-level reliability", "objective": "Compare participant-level estimates, repeated runs/sessions, and group versus individual localization.", "mathematical_operations": ("test_retest_correlation", "ICC when design metadata permit"), "biological_reference": "Individual variability is retained; group averages are not treated as universal anatomy."},
    {"operation_id": "LC-P10-12", "name": "Integrate evidence without double-counting", "objective": "Track source, sample, cohort, and dataset dependence before synthesizing evidence.", "mathematical_operations": ("log_odds_post=log_odds_prior+sum(independent_log_likelihood_ratios)", "group by independent evidence source"), "biological_reference": "Bayesian updating is an analyst's framework, not proof of neural probability calculations."},
    {"operation_id": "LC-P10-13", "name": "Revise the mechanistic account", "objective": "Retain replicated findings, revise regional assignments, preserve unresolved alternatives, and create new predictions.", "mathematical_operations": ("model_t+1=revise(model_t,evidence,uncertainty)", "prediction novelty and falsifiability"), "biological_reference": "The output is a revisable scientific model, not a fixed brain atlas."},
    {"operation_id": "LC-P10-14", "name": "Validate out of sample", "objective": "Evaluate frozen model predictions on held-out participants, items, sessions, or independent datasets.", "mathematical_operations": ("MSE=mean((y-y_hat)^2)", "accuracy, log_loss, calibration_error"), "biological_reference": "Generalization is a model-evaluation operation."},
    {"operation_id": "LC-P10-15", "name": "Produce calibrated mechanism report", "objective": "Report evidence by method, competing accounts, uncertainty, generalization, and next discriminating experiments.", "mathematical_operations": ("evidence_profile=(association,temporal,lesion,perturbation,replication)", "confidence labels remain qualitative and traceable"), "biological_reference": "The report distinguishes measured findings from computational assumptions."},
)


def _plain(value: Any) -> Any:
    """Convert NumPy and dataclass values into detached JSON-compatible Python data."""
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


def _finite_values(value: Any) -> np.ndarray:
    """Return finite numeric observations as a one-dimensional float array."""
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    return arr[np.isfinite(arr)]


def _mean(value: Any) -> float | None:
    """Compute a finite-value mean, returning None when no observations are available."""
    arr = _finite_values(value)
    return float(np.mean(arr)) if arr.size else None


def _cohen_d(target: Any, control: Any) -> float | None:
    """Compute pooled-standard-deviation Cohen's d when both samples support it."""
    a, b = _finite_values(target), _finite_values(control)
    if a.size < 2 or b.size < 2:
        return None
    pooled_var = ((a.size - 1) * np.var(a, ddof=1) + (b.size - 1) * np.var(b, ddof=1)) / (a.size + b.size - 2)
    if pooled_var <= 0 or not math.isfinite(float(pooled_var)):
        return None
    return float((np.mean(a) - np.mean(b)) / math.sqrt(float(pooled_var)))


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    """Normalize a mapping or dataclass record into a detached dictionary."""
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping or dataclass")
    return deepcopy(dict(value))


@dataclass(frozen=True, slots=True)
class Point10Request:
    """One evidence-validation job with a hypothesis, study plan, and supplied datasets."""

    mechanism: Mapping[str, Any]
    study_plan: Mapping[str, Any]
    datasets: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and snapshot every input so later caller mutations cannot alter the run."""
        for name in ("mechanism", "study_plan", "datasets", "metadata"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            object.__setattr__(self, name, deepcopy(dict(value)))


@dataclass(slots=True)
class Point10Result:
    """Calibrated report with method-specific evidence, QC flags, and a 15-step audit trace."""

    mechanism_name: str
    operation: str
    evidence: dict[str, Any]
    evidence_matrix: dict[str, Any]
    alternatives: dict[str, Any]
    quality_control: dict[str, Any]
    synthesis: dict[str, Any]
    generalization: dict[str, Any]
    report: dict[str, Any]
    trace: list[dict[str, Any]]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible representation of the validation result."""
        return _plain(asdict(self))


class Point10EvidenceValidator:
    """Validate a language-mechanism hypothesis against multimethod study evidence."""

    def __init__(self, minimum_temporal_trials: int = 20, minimum_behavior_trials: int = 2) -> None:
        """Configure conservative data sufficiency thresholds for descriptive validation."""
        if minimum_temporal_trials < 1 or minimum_behavior_trials < 1:
            raise ValueError("minimum trial thresholds must be positive")
        self.minimum_temporal_trials = int(minimum_temporal_trials)
        self.minimum_behavior_trials = int(minimum_behavior_trials)

    def _step(self, trace: list[dict[str, Any]], number: int, name: str, result: Mapping[str, Any]) -> None:
        """Append one ordered, detached operation record to the validation trace."""
        trace.append({"step": number, "operation_id": POINT10_STEP_SPECS[number - 1]["operation_id"],
                      "name": name, "result": _plain(dict(result))})

    def _behavior(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Summarize target/control behavior without treating response time as a unique process marker."""
        target = data.get("target", {})
        control = data.get("control", {})
        if not isinstance(target, Mapping) or not isinstance(control, Mapping):
            return {"available": False, "reason": "target/control behavior must be mappings"}
        result: dict[str, Any] = {"available": bool(target or control), "target": {}, "control": {}}
        for label, group in (("target", target), ("control", control)):
            for measure in ("accuracy", "response_time", "confidence"):
                vals = _finite_values(group.get(measure, []))
                result[label][measure] = {"n": int(vals.size), "mean": float(np.mean(vals)) if vals.size else None}
            errors = group.get("error_types", [])
            result[label]["error_count"] = len(errors) if isinstance(errors, (list, tuple)) else None
        result["accuracy_difference"] = (
            (result["target"]["accuracy"]["mean"] - result["control"]["accuracy"]["mean"])
            if result["target"]["accuracy"]["mean"] is not None and result["control"]["accuracy"]["mean"] is not None else None
        )
        result["response_time_difference"] = (
            (result["target"]["response_time"]["mean"] - result["control"]["response_time"]["mean"])
            if result["target"]["response_time"]["mean"] is not None and result["control"]["response_time"]["mean"] is not None else None
        )
        result["accuracy_effect_size_d"] = _cohen_d(target.get("accuracy", []), control.get("accuracy", []))
        result["response_time_effect_size_d"] = _cohen_d(target.get("response_time", []), control.get("response_time", []))
        return result

    def _fmri(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Summarize supplied fMRI contrast values and flag localizer, motion, and inference limitations."""
        target, control = _finite_values(data.get("target", [])), _finite_values(data.get("control", []))
        result = {"available": bool(target.size or control.size), "target_n": int(target.size),
                  "control_n": int(control.size), "target_mean": _mean(target), "control_mean": _mean(control),
                  "contrast": (_mean(target) - _mean(control)) if target.size and control.size else None,
                  "effect_size_d": _cohen_d(target, control),
                  "independent_roi": bool(data.get("independent_roi", False)),
                  "motion_controlled": bool(data.get("motion_controlled", False)),
                  "multiple_comparison_control": bool(data.get("multiple_comparison_control", False)),
                  "localizer_source": data.get("localizer_source", "unspecified"),
                  "inference": "regional/network association; not necessity or causal direction"}
        return result

    def _temporal(self, data: Mapping[str, Any], method: str) -> dict[str, Any]:
        """Summarize event-locked EEG/MEG values and preserve component-interpretation uncertainty."""
        target, control = _finite_values(data.get("target", [])), _finite_values(data.get("control", []))
        trial_count = min(int(data.get("target_trial_count", target.size)),
                          int(data.get("control_trial_count", control.size)))
        return {"available": bool(target.size or control.size), "method": method,
                "target_mean": _mean(target), "control_mean": _mean(control),
                "mean_difference": (_mean(target) - _mean(control)) if target.size and control.size else None,
                "target_values_n": int(target.size), "control_values_n": int(control.size),
                "trial_count_lower_bound": trial_count,
                "sufficient_trials_by_config": trial_count >= self.minimum_temporal_trials,
                "artifact_controlled": bool(data.get("artifact_controlled", False)),
                "pre_registered_window": bool(data.get("pre_registered_window", False)),
                "time_window_ms": _plain(data.get("time_window_ms")),
                "component_label": data.get("component_label", "unspecified"),
                "interpretation": "temporal condition sensitivity; component is not a unique process label"}

    def _lesion(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Summarize lesion-behavior evidence while retaining volume and disconnection caveats."""
        feature, performance = _finite_values(data.get("lesion_feature", [])), _finite_values(data.get("performance", []))
        association = None
        if feature.size == performance.size and feature.size >= 3 and np.std(feature) > 0 and np.std(performance) > 0:
            association = float(np.corrcoef(feature, performance)[0, 1])
        return {"available": bool(feature.size or performance.size), "n": int(min(feature.size, performance.size)),
                "lesion_performance_correlation": association,
                "lesion_volume_controlled": bool(data.get("lesion_volume_controlled", False)),
                "disconnection_assessed": bool(data.get("disconnection_assessed", False)),
                "non_language_abilities_assessed": bool(data.get("non_language_abilities_assessed", False)),
                "interpretation": "lesion association; network effects, vascular constraints, and compensation remain possible"}

    def _stimulation(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Compare active and sham/control stimulation outcomes without asserting perfect focality."""
        active_rt, sham_rt = _mean(data.get("active_response_time", [])), _mean(data.get("sham_response_time", []))
        active_acc, sham_acc = _mean(data.get("active_accuracy", [])), _mean(data.get("sham_accuracy", []))
        return {"available": any(k in data for k in ("active_response_time", "sham_response_time", "active_accuracy", "sham_accuracy")),
                "response_time_difference": active_rt - sham_rt if active_rt is not None and sham_rt is not None else None,
                "accuracy_difference": active_acc - sham_acc if active_acc is not None and sham_acc is not None else None,
                "sham_or_control_site": bool(data.get("sham_or_control_site", False)),
                "randomized_or_counterbalanced": bool(data.get("randomized_or_counterbalanced", False)),
                "non_specific_effects_checked": bool(data.get("non_specific_effects_checked", False)),
                "safety_protocol_documented": bool(data.get("safety_protocol_documented", False)),
                "interpretation": "perturbational evidence for causal contribution under tested conditions; not a uniquely localized computation"}

    def _quality_control(self, mechanism: Mapping[str, Any], plan: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
        """Collect explicit design and measurement warnings rather than silently assuming controls."""
        flags: list[str] = []
        if not str(mechanism.get("operation", "")).strip():
            flags.append("target_operation_not_operationally_defined")
        if not mechanism.get("predictions"):
            flags.append("no_falsifiable_predictions")
        if not mechanism.get("alternatives"):
            flags.append("no_competing_explanations")
        if not plan.get("target_condition") or not plan.get("control_condition"):
            flags.append("target_or_control_condition_missing")
        if not plan.get("counterbalanced", False):
            flags.append("counterbalancing_not_documented")
        if not plan.get("confounds_controlled"):
            flags.append("confound_controls_not_documented")
        if evidence.get("fmri", {}).get("available"):
            f = evidence["fmri"]
            if not f["independent_roi"]:
                flags.append("fMRI_ROI_independence_not_documented")
            if not f["motion_controlled"]:
                flags.append("fMRI_motion_control_not_documented")
            if not f["multiple_comparison_control"]:
                flags.append("fMRI_multiple_comparison_control_not_documented")
        for method in ("eeg", "meg"):
            if evidence.get(method, {}).get("available"):
                t = evidence[method]
                if not t["artifact_controlled"]:
                    flags.append(f"{method}_artifact_control_not_documented")
                if not t["sufficient_trials_by_config"]:
                    flags.append(f"{method}_trial_count_below_configured_minimum")
                if not t["pre_registered_window"]:
                    flags.append(f"{method}_time_window_not_pre_registered")
        if evidence.get("lesion", {}).get("available"):
            l = evidence["lesion"]
            if not l["lesion_volume_controlled"]:
                flags.append("lesion_volume_not_controlled")
            if not l["disconnection_assessed"]:
                flags.append("lesion_disconnection_not_assessed")
            if not l["non_language_abilities_assessed"]:
                flags.append("patient_non_language_abilities_not_assessed")
        if evidence.get("stimulation", {}).get("available"):
            s = evidence["stimulation"]
            if not s["sham_or_control_site"]:
                flags.append("stimulation_sham_or_control_site_missing")
            if not s["non_specific_effects_checked"]:
                flags.append("stimulation_non_specific_effects_not_checked")
            if not s["safety_protocol_documented"]:
                flags.append("stimulation_safety_protocol_not_documented")
        if plan.get("roi_selected_from_test_contrast", False):
            flags.append("circular_ROI_selection_possible")
        if plan.get("multiple_comparisons_ignored", False):
            flags.append("multiple_comparisons_ignored")
        if plan.get("null_result_claimed_as_absence", False):
            flags.append("null_result_overinterpreted_as_proof_of_absence")
        return {"passed": not flags, "flags": flags, "flag_count": len(flags),
                "interpretation": "flags indicate missing safeguards or overclaim risks, not proof that a study is invalid"}

    def _alternatives(self, mechanism: Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, Any]:
        """Retain all named alternatives and attach caller-supplied discriminating evidence."""
        names = mechanism.get("alternatives", [])
        if isinstance(names, str):
            names = [names]
        supplied = data.get("alternative_evidence", {})
        supplied = supplied if isinstance(supplied, Mapping) else {}
        return {str(name): {"evidence": _plain(supplied.get(str(name), {})),
                            "status": "evaluated_from_supplied_evidence" if str(name) in supplied else "unresolved"}
                for name in names}

    def _integrate(self, evidence: Mapping[str, Any], datasets: Mapping[str, Any]) -> dict[str, Any]:
        """Group evidence by independent source/cohort so multiple measures from one sample are not counted twice."""
        source_groups: dict[str, list[str]] = {}
        method_status: dict[str, str] = {}
        for method in POINT10_METHODS:
            item = evidence.get(method, {})
            if not isinstance(item, Mapping) or not item.get("available", False):
                method_status[method] = "not_supplied"
                continue
            source = str(datasets.get(method, {}).get("independence_group", datasets.get(method, {}).get("source_id", method)))
            source_groups.setdefault(source, []).append(method)
            method_status[method] = "available_but_not_independent_until_source_checked"
        independent_groups = [methods for methods in source_groups.values()]
        available = [m for m in POINT10_METHODS if method_status[m] != "not_supplied"]
        independent_count = len(independent_groups)
        return {"available_methods": available, "method_status": method_status,
                "independence_groups": independent_groups, "independent_source_group_count": independent_count,
                "double_counting_warning": any(len(group) > 1 for group in independent_groups),
                "bayesian_interpretation": "posterior updating requires likelihoods and dependence assumptions not estimated by this descriptive validator"}

    def _generalization(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Evaluate supplied held-out predictions using metrics appropriate to numeric outcomes."""
        held = data.get("held_out")
        if not isinstance(held, Mapping):
            return {"available": False, "status": "no_held_out_dataset_supplied"}
        observed, predicted = _finite_values(held.get("observed", [])), _finite_values(held.get("predicted", []))
        if not observed.size or observed.size != predicted.size:
            return {"available": False, "status": "observed_and_predicted_arrays_missing_or_mismatched"}
        mse = float(np.mean((observed - predicted) ** 2))
        return {"available": True, "status": "evaluated", "n": int(observed.size), "mse": mse,
                "mean_absolute_error": float(np.mean(np.abs(observed - predicted))),
                "correlation": float(np.corrcoef(observed, predicted)[0, 1])
                if observed.size >= 2 and np.std(observed) > 0 and np.std(predicted) > 0 else None,
                "model_was_frozen": bool(held.get("model_was_frozen", False)),
                "independent_test_set": bool(held.get("independent_test_set", False))}

    def process(self, request: Point10Request) -> Point10Result:
        """Run the complete 15-operation research validation and return a calibrated evidence report."""
        mechanism, plan, datasets = map(dict, (request.mechanism, request.study_plan, request.datasets))
        trace: list[dict[str, Any]] = []
        operation = str(mechanism.get("operation", "")).strip()
        name = str(mechanism.get("name", "unnamed_mechanism"))
        definition = {"name": name, "operation": operation, "operationally_defined": bool(operation),
                      "predictions": _plain(mechanism.get("predictions", [])),
                      "alternatives": _plain(mechanism.get("alternatives", []))}
        self._step(trace, 1, "Select and operationalize mechanism", definition)
        design = {"target_condition_present": bool(plan.get("target_condition")),
                  "control_condition_present": bool(plan.get("control_condition")),
                  "counterbalanced": bool(plan.get("counterbalanced", False)),
                  "confounds_controlled": _plain(plan.get("confounds_controlled", [])),
                  "participant_and_item_effects_planned": bool(plan.get("participant_and_item_effects", False))}
        self._step(trace, 2, "Construct controlled contrasts", design)
        evidence = {
            "behavior": self._behavior(datasets.get("behavior", {})),
            "fmri": self._fmri(datasets.get("fmri", {})),
            "eeg": self._temporal(datasets.get("eeg", {}), "EEG/ERP"),
            "meg": self._temporal(datasets.get("meg", {}), "MEG"),
            "lesion": self._lesion(datasets.get("lesion", {})),
            "stimulation": self._stimulation(datasets.get("stimulation", {})),
            "computational": {"available": bool(datasets.get("computational")),
                              "supplied": _plain(datasets.get("computational", {})),
                              "interpretation": "model fit is not biological proof"},
        }
        self._step(trace, 3, "Establish behavioral signature", evidence["behavior"])
        self._step(trace, 4, "Evaluate fMRI spatial evidence", evidence["fmri"])
        self._step(trace, 5, "Evaluate EEG/MEG temporal evidence", {"eeg": evidence["eeg"], "meg": evidence["meg"]})
        self._step(trace, 6, "Evaluate lesion evidence", evidence["lesion"])
        self._step(trace, 7, "Evaluate perturbational evidence", evidence["stimulation"])
        matrix = {method: {"available": bool(evidence[method].get("available", False)),
                           "contribution": evidence[method].get("interpretation", "not supplied")}
                  for method in POINT10_METHODS}
        self._step(trace, 8, "Build method-by-prediction matrix", matrix)
        alternatives = self._alternatives(mechanism, datasets)
        self._step(trace, 9, "Evaluate competing explanations", alternatives)
        effects = {method: evidence[method].get("effect_size_d") for method in ("behavior", "fmri")}
        self._step(trace, 10, "Estimate effect sizes and uncertainty", {"available_standardized_effects": effects,
                    "confidence_intervals": "not estimated unless supplied by the study analysis",
                    "multiple_comparison_control": bool(plan.get("multiple_comparison_control", False))})
        individual = _plain(datasets.get("individual_reliability", {}))
        reliability = {"available": bool(individual), "participant_level_results": individual,
                       "group_average_is_not_individual_localization": True}
        self._step(trace, 11, "Test individual-level reliability", reliability)
        synthesis = self._integrate(evidence, datasets)
        self._step(trace, 12, "Integrate evidence without double-counting", synthesis)
        available_count = sum(bool(evidence[m].get("available", False)) for m in POINT10_METHODS)
        synthesis["evidence_coverage"] = available_count / float(len(POINT10_METHODS))
        synthesis["calibration_label"] = (
            "no_empirical_evidence_supplied" if available_count == 0 else
            "preliminary_single_source" if synthesis["independent_source_group_count"] <= 1 else
            "multimethod_evidence_requires_quality_review"
        )
        synthesis["causal_evidence_available"] = bool(
            evidence["lesion"].get("available") or evidence["stimulation"].get("available")
        )
        synthesis["causal_claim_permitted_by_presence_alone"] = False
        self._step(trace, 13, "Revise mechanistic account", {"calibration_label": synthesis["calibration_label"],
                    "unresolved_alternatives": [k for k, v in alternatives.items() if v["status"] == "unresolved"],
                    "new_prediction_required": True})
        generalization = self._generalization(datasets)
        self._step(trace, 14, "Validate out of sample", generalization)
        qc = self._quality_control(mechanism, plan, evidence)
        confidence = "insufficient_evidence" if available_count == 0 else (
            "provisional" if qc["flag_count"] or synthesis["independent_source_group_count"] < 2 else
            "convergent_but_revisable"
        )
        report = {"mechanism_name": name, "operation": operation, "confidence_label": confidence,
                  "evidence_by_method": _plain(evidence), "quality_control_flags": qc["flags"],
                  "competing_explanations": _plain(alternatives),
                  "independent_source_groups": synthesis["independence_groups"],
                  "generalization": _plain(generalization),
                  "measured_findings_vs_assumptions": "Only supplied observations are summarized; mathematical formulations are analysis models, not proof of biological implementation.",
                  "next_experiments": _plain(mechanism.get("next_discriminating_experiments", []))}
        self._step(trace, 15, "Produce calibrated mechanism report", {"confidence_label": confidence,
                    "evidence_method_count": available_count, "quality_control_flag_count": qc["flag_count"],
                    "no_unsupported_certainty": True})
        return Point10Result(name, operation, _plain(evidence), _plain(matrix), _plain(alternatives),
                             _plain(qc), _plain(synthesis), _plain(generalization), _plain(report), trace,
                             {"valid": bool(operation and mechanism.get("predictions")),
                              "step_count": len(trace), "method_count": len(POINT10_METHODS),
                              "research_analysis_not_brain_simulation": True})


def register_point10_annotations(repository_root: str | Path | None = None,
                                 database_path: str | Path | None = None) -> dict[str, Any]:
    """Index Point 10 classes and functions in the shared ORE SQLite annotation catalogue."""
    import os
    from ore_logger import LoggerDatabase, sync_repository

    root = Path(repository_root).expanduser().resolve() if repository_root is not None else Path(__file__).resolve().parents[1]
    target = Path(database_path or os.environ.get("ORE_LOGGER_DB") or root / "ore_logs" / "ore_logger.sqlite3").expanduser().resolve()
    database = LoggerDatabase(target)
    modules = ["ore_language_comprehension"]
    report = sync_repository(database, root, include_paths=set(modules))
    return {**report, "database_path": str(target), "modules": modules,
            "annotation_source": "python_docstring_and_ast_signature",
            "storage": "ore_logger.annotations SQLite table"}
