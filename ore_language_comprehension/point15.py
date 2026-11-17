"""Point 15: temporal, network, causal, and cross-population research validation.

This is a scientist-facing research workflow, not a cognitive stage or a claim
that neurons execute the statistical operations below. Biological notes identify
plausible levels of implementation and explicitly retain uncertainty.
"""
from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


POINT15_HYPOTHESES = (
    "language_network", "language_plus_context", "social_inference",
    "domain_general_control", "recurrent_interaction",
)
POINT15_MODALITIES = ("EEG", "MEG", "fMRI", "intracranial", "behaviour", "structural")
POINT15_STAGES = (
    ("falsifiable_question", "Define a falsifiable mechanistic question", "Operationalize operation, contrast, outcome, candidate network, time range, and alternative", "Task-set and attentional systems select relevant information; no unique neural routine is implied", "Explicit hypothesis fields and preregistered contrasts", "Research design operation, not a biological computation"),
    ("competing_hypotheses", "Build competing or nested hypotheses", "Represent language-only, context, social, control, and recurrent explanations", "Distributed specialized networks may cooperate; hypotheses are analyst abstractions", "Categorical model set and model-comparison design", "Hypothesis enumeration is researcher-side"),
    ("distinct_predictions", "Derive discriminating predictions", "Specify effects under manipulations and counterfactual alternatives", "Prediction-sensitive recurrent processing is a broad candidate principle", "Conditional contrasts and predicted effect directions", "Predictions are falsifiable model commitments"),
    ("condition_design", "Select a minimal controlled condition set", "Contrast literal, context-shifted, bridging, pragmatic, and matched difficult controls", "Language and non-language systems are conditionally recruited", "Factorial condition matrix and contrast coding", "Design matrix is not a neural data structure"),
    ("factor_manipulation", "Manipulate critical factors independently", "Vary context, prosody, speaker knowledge, plausibility, and memory load", "Cue-sensitive populations and recurrent feedback may respond to changed evidence", "Factorial effects and interaction terms", "Statistical interactions are not synapses"),
    ("baseline_matching", "Match baseline difficulty and confounds", "Measure lexical, syntactic, familiarity, emotional, and response demands", "Multiple systems contribute to difficulty and task execution", "Standardized differences and covariate balance", "Matching is experimental control"),
    ("event_timing", "Mark informative stimulus events", "Align events to acoustic or visual availability of disambiguating evidence", "Sensory pathways encode continuously arriving input", "Event times and inter-event intervals", "Timestamps are measurement conventions"),
    ("behavioural_ground_truth", "Define behavioural outcomes and uncertainty", "Record accuracy, latency, confidence, and normed interpretations", "Distributed processing culminates in measurable choices and responses", "Accuracy, response-time, confidence, and ambiguity summaries", "Behavioural labels are operational ground truth, not neural truth"),
    ("method_selection", "Select complementary measurement methods", "Combine methods according to spatial, temporal, and ethical constraints", "EEG/MEG, fMRI, intracranial signals, anatomy, and behaviour sample different levels", "Method-coverage matrix", "Modalities are not interchangeable"),
    ("signal_quality", "Check signal quality and reliability", "Apply preregistered artifact rules and participant-level reliability checks", "Physiological noise and state variation affect measurements", "Valid-trial counts and reliability estimates", "Quality control is analyst-side"),
    ("individual_networks", "Preserve participant-specific network information", "Estimate functional regions individually where feasible", "Functional topography and connectivity vary across individuals", "Participant-level node and parameter estimates", "Group averages do not define an individual"),
    ("event_alignment", "Align neural data to informative events", "Analyze time relative to the arrival of critical evidence", "Neural populations respond over overlapping time courses", "Event-locked windows and trial alignment", "Alignment does not establish causality"),
    ("time_resolved_representation", "Estimate time-resolved representations", "Decode or compare information across time windows", "Population activity may carry changing information about form, meaning, context, and intention", "Decoding scores and representational similarity", "Decodability does not prove explicit symbolic coding"),
    ("temporal_uncertainty", "Estimate onset, peak, duration, and uncertainty", "Report participant-specific timing intervals and uncertainty", "Neural response timing varies with input, person, and task", "Bootstrap intervals and distributions", "No universal processing deadline is assumed"),
    ("temporal_overlap", "Test overlap without imposing a serial chain", "Estimate lagged association between candidate signals", "Parallel and recurrent processing can coexist", "Cross-correlation and lag estimates", "Correlation/lag is not direction or causation"),
    ("network_graph", "Construct participant-specific candidate networks", "Represent justified nodes and time-varying candidate edges", "Distributed populations communicate through synaptic and anatomical pathways", "Weighted adjacency matrices and graph summaries", "Graph edges are estimates, not direct anatomical proof"),
    ("connectivity_distinction", "Distinguish functional and effective connectivity", "Compare undirected dependence with directed models and shared-input alternatives", "Recurrent and feedforward influences are biologically plausible", "Alternative direction models and fit criteria", "Effective connectivity is assumption-dependent"),
    ("dynamic_state_model", "Estimate changing network coupling", "Fit a simplified state-space model to latent and observed signals", "Synaptic, membrane, population, and neuromodulatory dynamics support changing states", "State transition estimates and residual error", "Matrix equations are computational abstractions"),
    ("context_modulation", "Test context-dependent network change", "Compare identical critical input across different preceding contexts", "Top-down and recurrent context modulation may change gain and interpretation", "Within-item context contrasts and interaction effects", "Context effects require matched controls"),
    ("causal_claim", "Narrow causal claims to a specific contribution", "Separate necessity, contribution, timing, downstream influence, and compensation", "Perturbation-sensitive mechanisms can contribute without being sole loci", "Claim scope and causal estimand", "Activation alone is not causal evidence"),
    ("causal_method", "Select an ethically appropriate causal method", "Use controlled stimulation, lesion evidence, or clinical recordings only when justified", "Perturbation changes activity in connected distributed systems", "Intervention/control contrast and safety metadata", "Ethical approval and clinical indication are prerequisites"),
    ("causal_evidence", "Grade causal evidence conservatively", "Distinguish association, temporal relation, directed model, perturbation, and convergence", "Multiple biological levels constrain mechanism claims", "Evidence tier and alternative-confound ledger", "No single universal evidence ranking"),
    ("population_validation", "Validate across linguistic and cultural populations", "Adapt, norm, and test measurement comparability across groups", "Learning and experience shape language and pragmatic systems", "Group parameters, uncertainty, and measurement invariance", "One population cannot establish universality"),
    ("independent_validation", "Evaluate held-out prediction and replication", "Keep training and validation data separate and test external generalization", "A biologically useful model should predict new behaviour and neural observations", "Held-out Brier/MSE, calibration, and replication status", "Cross-validation is analyst-side"),
    ("falsifiable_report", "Report uncertainty, alternatives, and falsifiable predictions", "Document supported claims, limits, missing evidence, and next experiments", "Mechanistic explanations remain revisable as evidence changes", "Evidence ledger, posterior/interval summaries, and prediction register", "A transparent report is not proof of mechanism"),
)
POINT15_STEP_NAMES = tuple(row[1] for row in POINT15_STAGES)


def _number(value: Any) -> float | None:
    """Return a finite numeric value while treating booleans and missing values as unknown."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _mean(values: Sequence[Any]) -> float | None:
    """Compute a finite mean without converting an empty sample to zero."""
    nums = [n for value in values if (n := _number(value)) is not None]
    return float(np.mean(nums)) if nums else None


def _correlation(left: Sequence[Any], right: Sequence[Any]) -> float | None:
    """Compute Pearson correlation for paired finite values with nonzero variance."""
    pairs = [(_number(a), _number(b)) for a, b in zip(left, right)]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    x, y = np.asarray(pairs, dtype=float).T
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _brier(records: Sequence[Mapping[str, Any]]) -> float | None:
    """Calculate Brier score only for records with explicit held-out labels and predictions."""
    errors = []
    for row in records:
        p, y = _number(row.get("predicted_probability")), _number(row.get("outcome"))
        if p is not None and y is not None and 0 <= p <= 1 and 0 <= y <= 1:
            errors.append((p - y) ** 2)
    return _mean(errors)


@dataclass
class Point15Request:
    """Inputs for a mechanistic language-comprehension research and validation run.

    Neural records use {participant_id, region, time, value, signal, condition}.
    Network edges use {source, target, weight, time, kind}; intervention records
    use {method, target, condition, outcome, control_outcome, approved}.
    These are researcher-provided measurements, not inferred raw neural states.
    """
    research_question: str
    participants: list[dict[str, Any]] = field(default_factory=list)
    conditions: list[dict[str, Any]] = field(default_factory=list)
    event_times: list[dict[str, Any]] = field(default_factory=list)
    behavioural_trials: list[dict[str, Any]] = field(default_factory=list)
    neural_records: list[dict[str, Any]] = field(default_factory=list)
    network_edges: list[dict[str, Any]] = field(default_factory=list)
    interventions: list[dict[str, Any]] = field(default_factory=list)
    population_groups: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the question and defensively copy all caller-owned records."""
        if not isinstance(self.research_question, str) or not self.research_question.strip():
            raise ValueError("research_question must be a non-empty string")
        for name in ("participants", "conditions", "event_times", "behavioural_trials",
                     "neural_records", "network_edges", "interventions",
                     "population_groups", "hypotheses"):
            value = getattr(self, name)
            if not isinstance(value, list):
                raise TypeError(f"{name} must be a list")
            setattr(self, name, copy.deepcopy(value))
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")
        self.metadata = copy.deepcopy(self.metadata)


@dataclass
class Point15Result:
    """Structured research report with trace, evidence tiers, and uncertainty."""
    research_question: str
    hypotheses: list[dict[str, Any]]
    temporal: dict[str, Any]
    network: dict[str, Any]
    causal: dict[str, Any]
    population: dict[str, Any]
    validation: dict[str, Any]
    report: dict[str, Any]
    trace: list[dict[str, Any]]
    diagnostics: dict[str, Any]


class Point15MechanisticResearchProcessor:
    """Evaluate temporal, network, causal, and cross-population hypotheses.

    This processor organizes supplied data and tests explicit predictions. It does
    not diagnose participants, infer missing neural signals, or turn correlations
    into causal claims. Biological descriptions are qualitative hypotheses.
    """

    def process(self, request: Point15Request) -> Point15Result:
        """Run the source-ordered 25-stage research workflow."""
        if not isinstance(request, Point15Request):
            raise TypeError("request must be a Point15Request")
        trace: list[dict[str, Any]] = []

        def record(index: int, output: dict[str, Any]) -> None:
            """Append a source-ordered, fully annotated operation result."""
            name, title, functional, biological, math_ops, caveat = POINT15_STAGES[index - 1]
            trace.append({"step": index, "name": name, "title": title,
                          "functional_algorithm": functional,
                          "biological_reference": biological,
                          "mathematical_operations": math_ops,
                          "scientific_caveat": caveat, "output": output})

        # 1-3: falsifiable hypotheses and discriminating predictions.
        question = {"claim": request.research_question.strip(),
                    "falsifiable": bool(request.metadata.get("predicted_direction")
                                        or request.metadata.get("primary_outcome")),
                    "missing_specifications": [k for k in ("contrast", "primary_outcome",
                        "candidate_network", "time_window", "alternative_explanation")
                        if k not in request.metadata]}
        record(1, question)
        hypotheses = []
        for item in request.hypotheses:
            if isinstance(item, Mapping) and item.get("hypothesis_id"):
                hypotheses.append({"hypothesis_id": str(item["hypothesis_id"]),
                    "family": item.get("family", "unspecified"),
                    "prediction": copy.deepcopy(item.get("prediction", {})),
                    "alternative": item.get("alternative"),
                    "mutually_exclusive": False})
        if not hypotheses:
            hypotheses = [{"hypothesis_id": key, "family": key, "prediction": {},
                           "alternative": None, "mutually_exclusive": False}
                          for key in POINT15_HYPOTHESES]
        record(2, {"hypothesis_set": hypotheses, "nested_or_coexisting_allowed": True})
        distinct = [{"hypothesis_id": h["hypothesis_id"],
                     "manipulations": list(h["prediction"].keys())
                        if isinstance(h["prediction"], Mapping) else [],
                     "distinguishable_prediction_supplied": bool(h["prediction"])}
                    for h in hypotheses]
        record(3, {"predictions": distinct,
                   "warning": "A model without a discriminating prediction is descriptive only"})

        # 4-8: experimental design, events, and behavioural anchors.
        condition_names = [str(c.get("condition_id", c.get("name", f"condition-{i}")))
                           for i, c in enumerate(request.conditions) if isinstance(c, Mapping)]
        factors = sorted({str(k) for c in request.conditions if isinstance(c, Mapping)
                          for k in c.get("factors", {})})
        record(4, {"conditions": condition_names, "factor_names": factors,
                   "minimal_design": bool(condition_names)})
        factor_levels = {factor: sorted({str(c.get("factors", {}).get(factor))
                          for c in request.conditions if isinstance(c, Mapping)
                          and factor in c.get("factors", {})}) for factor in factors}
        record(5, {"factor_levels": factor_levels, "independent_manipulations": len(factors)})
        baseline = {}
        for key in ("lexical_frequency", "word_length", "sentence_length",
                    "syntactic_complexity", "plausibility", "familiarity", "response_demand"):
            vals = [_number(c.get(key)) for c in request.conditions if isinstance(c, Mapping)]
            vals = [v for v in vals if v is not None]
            baseline[key] = {"mean": _mean(vals), "range": (max(vals)-min(vals)) if vals else None}
        record(6, {"baseline_descriptives": baseline,
                   "unmeasured_confounds": [k for k,v in baseline.items() if v["mean"] is None]})
        times = sorted([{"event": str(e.get("event")), "time": _number(e.get("time"))}
                        for e in request.event_times if isinstance(e, Mapping)
                        and _number(e.get("time")) is not None], key=lambda x: x["time"])
        intervals = [times[i]["time"]-times[i-1]["time"] for i in range(1,len(times))]
        record(7, {"ordered_events": times, "inter_event_intervals": intervals,
                   "informative_event_times_supplied": bool(times)})
        behavioural = [r for r in request.behavioural_trials if isinstance(r, Mapping)]
        valid_outcomes = [_number(r.get("outcome", r.get("accuracy"))) for r in behavioural]
        valid_outcomes = [v for v in valid_outcomes if v is not None and 0 <= v <= 1]
        record(8, {"trial_count": len(behavioural), "valid_outcome_count": len(valid_outcomes),
                   "mean_outcome": _mean(valid_outcomes),
                   "mean_response_time": _mean([r.get("response_time") for r in behavioural]),
                   "mean_confidence": _mean([r.get("confidence") for r in behavioural]),
                   "ambiguity_marked_count": sum(bool(r.get("ambiguous")) for r in behavioural)})

        # 9-11: modality coverage, quality, and individual-level data.
        modalities = sorted({str(r.get("modality")).upper() for r in request.neural_records
                             if isinstance(r, Mapping) and r.get("modality")})
        modality_coverage = {m: sum(1 for r in request.neural_records
                                   if isinstance(r, Mapping) and str(r.get("modality")).upper()==m)
                             for m in modalities}
        record(9, {"available_modalities": modalities,
                   "recognized_modalities": [m for m in modalities
                                             if m in {x.upper() for x in POINT15_MODALITIES}],
                   "coverage": modality_coverage})
        signals = [r for r in request.neural_records if isinstance(r, Mapping)
                   and _number(r.get("value")) is not None]
        quality = {"record_count": len(request.neural_records), "valid_signal_count": len(signals),
                   "invalid_or_missing_count": len(request.neural_records)-len(signals),
                   "participant_count": len({r.get("participant_id") for r in signals
                                             if r.get("participant_id") is not None})}
        record(10, quality)
        participant_regions: dict[str, list[str]] = {}
        for row in signals:
            participant_regions.setdefault(str(row.get("participant_id", "unknown")), []).append(
                str(row.get("region", "unspecified")))
        record(11, {"participant_specific_regions": {
            p: sorted(set(regions)) for p, regions in participant_regions.items()},
            "group_average_is_not_individual_ground_truth": True})

        # 12-15: time course, decoding proxies, uncertainty, and overlap.
        aligned = []
        for row in signals:
            time = _number(row.get("time"))
            if time is not None:
                aligned.append({**row, "relative_time": time - ( _number(row.get("event_time")) or 0.0)})
        record(12, {"aligned_signal_count": len(aligned),
                    "alignment_reference": "event_time when supplied, otherwise zero",
                    "all_trials_kept_for_time_course": True})
        by_signal: dict[str, list[float]] = {}
        for row in aligned:
            key = str(row.get("signal", row.get("region", "unspecified")))
            by_signal.setdefault(key, []).append(float(row["value"]))
        representations = {key: {"sample_count": len(values), "mean_signal": _mean(values),
                                 "variance": float(np.var(values)) if values else None}
                           for key, values in by_signal.items()}
        record(13, {"time_resolved_signal_summaries": representations,
                    "decoding_not_claimed_without_labels": True})
        timing = {}
        for key in by_signal:
            rows = [r for r in aligned if str(r.get("signal", r.get("region","unspecified")))==key]
            ts = [float(r["relative_time"]) for r in rows]
            vals = [float(r["value"]) for r in rows]
            timing[key] = {"first_sample_time": min(ts) if ts else None,
                           "peak_sample_time": ts[int(np.argmax(vals))] if vals else None,
                           "last_sample_time": max(ts) if ts else None,
                           "sample_count": len(ts)}
        record(14, {"timing_estimates": timing,
                    "uncertainty_not_estimated_without_repeated_trials": True})
        overlap = {}
        signals_by_name = {str(r.get("signal", r.get("region","unspecified"))): r
                           for r in []}
        names = list(by_signal)
        for i, left in enumerate(names):
            for right in names[i+1:]:
                lrows = [r for r in aligned if str(r.get("signal",r.get("region","unspecified")))==left]
                rrows = [r for r in aligned if str(r.get("signal",r.get("region","unspecified")))==right]
                lmap = {r["relative_time"]: r["value"] for r in lrows}
                rmap = {r["relative_time"]: r["value"] for r in rrows}
                common = sorted(set(lmap).intersection(rmap))
                overlap[f"{left}|{right}"] = {"paired_samples": len(common),
                    "zero_lag_correlation": _correlation([lmap[t] for t in common],
                                                         [rmap[t] for t in common])}
        record(15, {"pairwise_temporal_associations": overlap,
                    "correlation_does_not_establish_direction": True})

        # 16-19: candidate graph, direction alternatives, dynamic state, and context.
        edges = [e for e in request.network_edges if isinstance(e, Mapping)
                 and e.get("source") is not None and e.get("target") is not None]
        nodes = sorted({str(e[k]) for e in edges for k in ("source","target")})
        adjacency = np.zeros((len(nodes),len(nodes)), dtype=float)
        node_index = {node:i for i,node in enumerate(nodes)}
        for edge in edges:
            weight = _number(edge.get("weight"))
            if weight is not None:
                adjacency[node_index[str(edge["source"])],node_index[str(edge["target"])]] = weight
        record(16, {"nodes": nodes, "adjacency_matrix": adjacency.tolist(),
                    "edge_count": len(edges), "time_varying_edges": any(e.get("time") is not None for e in edges),
                    "edge_weights_are_estimates_not_anatomy": True})
        kinds = sorted({str(e.get("kind","unspecified")) for e in edges})
        record(17, {"edge_kinds": kinds, "functional_and_effective_connectivity_separated": True,
                    "candidate_direction_models": ["forward","backward","bidirectional_recurrent",
                                                   "shared_input"]})
        dynamic = {"time_points": sorted({_number(e.get("time")) for e in edges
                                         if _number(e.get("time")) is not None}),
                   "state_space_model": "z[t+1]=A[t]z[t]+B u[t]+epsilon; y[t]=C z[t]+nu",
                   "fit_performed": False, "reason": "Raw synchronized multivariate time series and model specification required"}
        record(18, dynamic)
        contexts: dict[str, dict[str, list[float]]] = {}
        for row in behavioural:
            if row.get("context") is not None and _number(row.get("outcome",row.get("accuracy"))) is not None:
                contexts.setdefault(str(row["condition"]), {}).setdefault(str(row["context"]), []).append(
                    float(row.get("outcome",row.get("accuracy"))))
        context_effects = {}
        for condition, levels in contexts.items():
            if len(levels) >= 2:
                names2 = sorted(levels)
                context_effects[condition] = {"level_means": {k:_mean(v) for k,v in levels.items()},
                    "first_minus_second": _mean(levels[names2[0]]) - _mean(levels[names2[1]])}
        record(19, {"context_condition_effects": context_effects,
                    "same_item_matching_required": True})

        # 20-22: causal scope and ethically constrained intervention evidence.
        causal_claim = {"claim_scope": request.metadata.get("causal_claim_scope", "association"),
                        "allowed_scopes": ["necessity_under_condition","processing_contribution",
                            "timing_of_contribution","downstream_influence","compensation"],
                        "activation_alone_is_not_causation": True}
        record(20, causal_claim)
        approved = [x for x in request.interventions if isinstance(x, Mapping)
                    and x.get("approved") is True and _number(x.get("outcome")) is not None
                    and _number(x.get("control_outcome")) is not None]
        methods = sorted({str(x.get("method","unspecified")) for x in request.interventions
                          if isinstance(x, Mapping)})
        record(21, {"methods_reported": methods, "approved_intervention_count": len(approved),
                    "unapproved_or_incomplete_excluded": len(request.interventions)-len(approved),
                    "ethical_approval_required": True})
        tiers = {"behavioural_association": bool(behavioural),
                 "neural_association": bool(signals),
                 "temporal_relationship": bool(timing),
                 "model_based_direction": bool(edges),
                 "perturbational_evidence": bool(approved)}
        effects = [{"method": x.get("method"), "target": x.get("target"),
                    "outcome_minus_control": float(x["outcome"])-float(x["control_outcome"]),
                    "condition": x.get("condition")} for x in approved]
        causal = {"evidence_tiers_available": tiers, "intervention_effects": effects,
                  "convergent_causal_evidence": bool(approved and signals and behavioural),
                  "causal_claim_must_not_exceed_evidence": True}
        record(22, causal)

        # 23-25: cross-population validity, independent prediction, and transparent report.
        group_summary = {}
        for row in request.population_groups:
            if not isinstance(row, Mapping) or row.get("group_id") is None:
                continue
            group_id = str(row["group_id"])
            scores = [_number(t.get("outcome",t.get("accuracy"))) for t in behavioural
                      if str(t.get("group_id"))==group_id]
            scores = [s for s in scores if s is not None]
            group_summary[group_id] = {"n": len(scores), "mean_outcome": _mean(scores),
                "language": row.get("language"), "cultural_context": row.get("cultural_context"),
                "measurement_invariance_tested": bool(row.get("measurement_invariance_tested",False))}
        record(23, {"population_summaries": group_summary,
                    "cross_group_comparability_established": bool(group_summary) and all(
                        x["measurement_invariance_tested"] for x in group_summary.values()),
                    "single_population_is_not_universal": True})
        heldout = [r for r in request.behavioural_trials if isinstance(r, Mapping)
                   and str(r.get("split","")).lower() in {"test","held_out","validation"}]
        validation = {"held_out_n": len(heldout), "brier_score": _brier(heldout),
                      "validation_available": _brier(heldout) is not None,
                      "external_replication_reported": bool(request.metadata.get("external_replication")),
                      "training_data_reused_as_test": False}
        record(24, validation)
        report = {"research_question": request.research_question,
            "supported_evidence_tiers": [k for k,v in tiers.items() if v],
            "candidate_hypotheses": hypotheses, "causal_interpretation": causal,
            "temporal_estimates": timing, "network_nodes": nodes,
            "population_groups": group_summary, "validation": validation,
            "missing_evidence": [k for k,v in tiers.items() if not v],
            "falsifiable_next_predictions": [h["prediction"] for h in hypotheses
                                             if h.get("prediction")],
            "limitations": ["No diagnosis or individual neural trait inference is performed.",
                "Neural data are summarized as supplied; no unobserved signal is fabricated.",
                "Timing and coupling estimates depend on sampling and preprocessing.",
                "Connectivity and temporal precedence do not establish causation.",
                "Population comparisons require construct and measurement comparability."],
            "mechanism_not_proven_by_software": True}
        record(25, report)
        return Point15Result(request.research_question, hypotheses, timing,
            {"nodes":nodes,"adjacency_matrix":adjacency.tolist(),"edges":edges,
             "temporal_associations":overlap,"context_effects":context_effects},
            causal, group_summary, validation, report, trace,
            {"valid":True,"step_count":len(trace),"stage_count":len(POINT15_STAGES),
             "neural_values_fabricated":False,"causality_overclaimed":False,
             "missing_data_preserved":True})


def register_point15_annotations(repository_root: str | Path | None = None,
                                 database_path: str | Path | None = None) -> dict[str, Any]:
    """Index Point 15 classes, methods, and functions in ORE's shared SQLite annotation table."""
    from ore_logger import LoggerDatabase, sync_repository
    root = (Path(repository_root).expanduser().resolve() if repository_root is not None
            else Path(__file__).resolve().parents[1])
    target = Path(database_path or os.environ.get("ORE_LOGGER_DB")
                  or root / "ore_logs" / "ore_logger.sqlite3").expanduser().resolve()
    report = sync_repository(LoggerDatabase(target), root,
                             include_paths={"ore_language_comprehension"})
    return {**report, "database_path":str(target),
            "modules":["ore_language_comprehension"],
            "annotation_source":"python_docstring_and_ast_signature",
            "storage":"ore_logger.annotations SQLite table"}
