"""End-to-end 40-stage Point 11 Bayesian, probabilistic and causal reasoning processor."""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .biology import BiologicalBayesianCausalNetwork
from .contracts import (
    BayesianCausalConfig,
    BayesianCausalRequest,
    BayesianCausalResult,
    BayesianCausalState,
    EvidenceBinding,
)
from .functional import (
    backdoor_adjustment_query,
    classify_causal_roles,
    compare_hypotheses,
    conditional_joint,
    content_digest,
    counterfactual_query,
    detect_dependent_evidence,
    evidence_conflict_analysis,
    expected_information_gain,
    find_backdoor_adjustment,
    hypothesis_prior_sensitivity,
    intervention_query,
    network_domains,
    network_fingerprint,
    network_parents,
    posterior_predictive_check,
    prior_predictive_check,
    prior_sensitivity_analysis,
    probabilistic_programming_status,
    run_posterior_inference,
    select_inference_method,
    summarize_evidence_intake,
    validate_model_schema,
    validate_network,
    validate_scm,
)
from .math_ops import (
    backward_smoothing,
    brier_score,
    detect_distribution_shift,
    expected_calibration_error,
    forward_filter,
    log_loss,
    reliability_diagram,
    validate_hmm_spec,
    viterbi_decode,
)
from .registry import OPERATION_REGISTRY


class _ResourceBudget:
    """Track bounded work units so runs finish as partial outcomes instead of diverging."""

    def __init__(self, total: int) -> None:
        self.total = max(1, int(total))
        self.used = 0
        self.exhausted = False

    def charge(self, units: int) -> bool:
        """Consume units if available; otherwise mark the budget exhausted."""
        units = max(0, int(units))
        if self.used + units > self.total:
            self.exhausted = True
            return False
        self.used += units
        return True


def _append_operation(
    trace: list[dict[str, Any]], index: int, functional_output: Mapping[str, Any],
    biological_output: Mapping[str, Any] | None = None, status: str = "completed",
) -> None:
    """Append a source-ordered operation with its mathematical and biological metadata."""
    spec = OPERATION_REGISTRY[index]
    trace.append({
        "step": index + 1, "source_step": spec.source_step,
        "operation_id": spec.operation_id, "operation": spec.name,
        "status": status, "purpose": spec.purpose,
        "evidence_level": spec.evidence_level,
        "mathematical_form": spec.mathematical_form,
        "functional_output": dict(functional_output),
        "biological_output": dict(biological_output or {}),
    })


def _bool_hook(hook: Any, payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Call an external verification hook defensively and normalize its verdict."""
    if hook is None:
        return None
    try:
        verdict = hook(dict(payload))
        if isinstance(verdict, Mapping):
            return {"valid": bool(verdict.get("valid", True)),
                    "violations": list(verdict.get("violations", ())),
                    "source": "external_verifier_hook"}
    except Exception as exc:  # pragma: no cover - external hook failures are reported
        return {"valid": False, "violations": [f"hook_error:{type(exc).__name__}"],
                "source": "external_verifier_hook"}
    return None


class BayesianCausalProcessor:
    """Run the 40-step Point 11 algorithm over one inference request.

    The processor keeps probability, source reliability, measurement uncertainty,
    logical truth and causal confidence separate at every stage. Missing evidence is
    never treated as negative evidence, correlated reports are not double counted, and
    counterfactual results are always published as hypothetical.
    """

    def __init__(
        self, runner_id: str, config: BayesianCausalConfig | None = None,
        state: BayesianCausalState | None = None,
        biological: BiologicalBayesianCausalNetwork | None = None,
        workspace: Any = None, finding_publisher: Any = None,
        coordinator_handoff: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> None:
        """Initialize one reasoning agent with configuration, state and integration hooks."""
        if not runner_id or not runner_id.strip():
            raise ValueError("runner_id must be non-empty")
        self.runner_id = runner_id.strip()
        self.config = config or BayesianCausalConfig()
        self.state = state or BayesianCausalState(self.runner_id)
        if self.state.runner_id != self.runner_id:
            raise ValueError("processor runner_id must match state runner_id")
        self.biological = biological or BiologicalBayesianCausalNetwork()
        self.workspace = workspace
        self.finding_publisher = finding_publisher
        self.coordinator_handoff = coordinator_handoff

    # ------------------------------------------------------------------
    # Main 40-step entry point
    # ------------------------------------------------------------------

    def process(
        self, request: BayesianCausalRequest, state: BayesianCausalState | None = None,
    ) -> BayesianCausalResult:
        """Execute the complete ordered 40-step Point 11 pipeline for one request."""
        active = state if state is not None else self.state
        if active.runner_id != self.runner_id:
            raise ValueError("supplied state runner_id must match processor runner_id")
        config = self.config
        trace: list[dict[str, Any]] = []
        bio = self.biological
        budget = _ResourceBudget(request.resource_budget or config.resource_budget)
        decision_threshold = request.decision_threshold or config.decision_threshold
        started = time.perf_counter()
        run_status = "completed"

        def cancelled() -> bool:
            check = request.cancel_check
            if check is None:
                return False
            try:
                return bool(check())
            except Exception:
                return False

        # -- Section A: intake and evidence model (steps 1-8) -----------------

        request_frame = {
            "question": request.question,
            "time_horizon": request.time_horizon,
            "scope": request.scope,
            "target_variables": [variable.name for variable in request.variables],
            "decision_threshold": decision_threshold,
            "resource_budget": budget.total,
        }
        _append_operation(trace, 0, {"request_frame": request_frame, "budget": budget.total},
                          bio.process_step(1, [request.question], 0.0, 0.8))

        bound_revision = None
        run_record: dict[str, Any] = {
            "run_index": active.run_count + 1,
            "question": request.question,
            "started_at": time.time(),
        }
        workspace_handle = self._workspace_handle(request.question)
        if workspace_handle is not None:
            try:
                bound_revision = int(workspace_handle.snapshot().revision)
            except Exception as exc:
                run_record["revision_binding"] = f"failed:{type(exc).__name__}"
        run_record["bound_revision"] = bound_revision
        active.runs.append(run_record)
        _append_operation(trace, 1, {"run_record": run_record,
                                     "workspace_bound": bound_revision is not None},
                          bio.process_step(2, [request.question], 0.0, 0.8))

        identity_resolution = self._resolve_usg_identities(request)
        _append_operation(trace, 2, {"identity_resolution": identity_resolution},
                          bio.process_step(3, [request.question], 0.05, 0.8))

        variables_map = {variable.name: variable for variable in request.variables}
        if request.network is not None:
            for variable in request.network.variables:
                variables_map.setdefault(variable.name, variable)
        retrieved = {
            "request_evidence_count": len(request.evidence),
            "request_hypothesis_count": len(request.hypotheses),
            "state_network_count": len(active.networks),
            "state_causal_model_count": len(active.causal_models),
            "state_scm_count": len(active.scms),
            "has_network": request.network is not None,
            "has_hmm": request.hmm is not None,
            "has_causal_dag": request.causal_dag is not None,
            "has_scm": request.scm is not None,
            "provenance": {
                "network": request.network.provenance if request.network else None,
                "evidence_sources": sorted({
                    record.source.source_id for record in request.evidence if record.source
                }),
            },
        }
        _append_operation(trace, 3, {"retrieved_inputs": retrieved},
                          bio.process_step(4, [request.question], 0.05, 0.8))

        for record in request.evidence:
            existing = active.evidence.get(record.evidence_id)
            if existing is not None and existing != record:
                raise ValueError(f"conflicting evidence id in state: {record.evidence_id}")
            active.evidence[record.evidence_id] = record
            if record.source is not None:
                active.sources[record.source.source_id] = record.source
        for record in request.hypotheses:
            active.hypotheses.setdefault(record.hypothesis_id, record)
        normalization = {
            "normalized_count": len(request.evidence),
            "unit_declarations": sorted({record.unit for record in request.evidence
                                         if record.unit is not None}),
            "timestamp_coverage": sum(1 for record in request.evidence if record.timestamp),
            "missing_value_codes": sorted({record.missing_value_code for record in request.evidence
                                           if record.missing_value_code is not None}),
            "measurement_intervals": sum(1 for record in request.evidence
                                         if record.measurement_interval is not None),
            "entity_identity": identity_resolution,
        }
        _append_operation(trace, 4, {"normalized_evidence": normalization},
                          bio.process_step(5, [request.question], 0.05, 0.8))

        intake = summarize_evidence_intake(request.evidence, variables_map)
        if not intake["valid"]:
            raise ValueError(f"evidence intake failed validation: {intake['issues']}")
        _append_operation(trace, 5, {"classified_evidence": intake},
                          bio.process_step(6, [request.question], 0.05, 0.8))

        separation = {
            "dimensions": ["probability", "source_reliability", "measurement_uncertainty",
                           "logical_truth", "causal_confidence"],
            "report": intake["separation_report"],
            "never_merged": True,
        }
        _append_operation(trace, 6, {"separated_confidence_dimensions": separation},
                          bio.process_step(7, [request.question], 0.05, 0.8))

        dependence = detect_dependent_evidence(request.evidence, config.dependence_threshold)
        for record in request.evidence:
            if record.evidence_id not in bio.hippocampus.episodes:
                bio.bind_evidence(record.evidence_id, [record.variable, record.classification],
                                  {"source": record.source.source_id if record.source else None,
                                   "classification": record.classification})
        _append_operation(trace, 7, {"source_dependence_report": dependence},
                          bio.process_step(8, [request.question], 0.10, 0.8))

        # -- Section B: Bayesian hypothesis evaluation (steps 9-20) -----------

        domains_report: dict[str, Any]
        network = request.network
        if network is not None:
            ambiguous = [variable.name for variable in network.variables if len(variable.states) < 2]
            if ambiguous:
                raise ValueError(f"variables need at least two states: {ambiguous}")
            domains_report = {"domains": {name: list(states) for name, states in
                                          network_domains(network).items()},
                              "ambiguous_rejected": []}
        elif request.variables:
            ambiguous = [variable.name for variable in request.variables if len(variable.states) < 2]
            if ambiguous:
                raise ValueError(f"variables need at least two states: {ambiguous}")
            domains_report = {"domains": {variable.name: list(variable.states)
                                          for variable in request.variables},
                              "ambiguous_rejected": []}
        else:
            domains_report = {"domains": {}, "not_applicable": "no variables declared"}
        _append_operation(trace, 8, {"variable_domains": domains_report},
                          bio.process_step(9, [request.question], 0.05, 0.8))

        validation: dict[str, Any] = {"not_applicable": "no Bayesian network supplied"}
        if network is not None:
            validation = validate_network(network)
            if not validation["valid"]:
                raise ValueError(f"Bayesian network rejected: {validation['errors']}")
        _append_operation(trace, 9, {"validated_graph": validation},
                          bio.process_step(10, [request.question], 0.10, 0.8))

        parents_map = network_parents(network) if network is not None else {}
        _append_operation(trace, 10, {"specified_cpts": {
            "parent_map": {name: list(parents) for name, parents in parents_map.items()},
            "cpt_count": len(network.cpts) if network is not None else 0,
        }}, bio.process_step(11, [request.question], 0.10, 0.8))

        verifier_verdict = _bool_hook(request.verifier_hook, {
            "stage": "probability_validation", "network_validation": validation})
        probability_validation = {
            "internal": validation.get("valid", True),
            "errors": validation.get("errors", []),
            "external_verifier": verifier_verdict,
        }
        if verifier_verdict is not None and not verifier_verdict["valid"]:
            raise ValueError(f"external verifier rejected the model: {verifier_verdict['violations']}")
        _append_operation(trace, 11, {"probability_validation": probability_validation},
                          bio.process_step(12, [request.question], 0.10, 0.8))

        prior_documentation = validation.get("prior_documentation", {}) if network else {}
        undocumented = [variable for variable, source in prior_documentation.items()
                        if source == "unspecified"]
        _append_operation(trace, 12, {"documented_priors": {
            "prior_sources": prior_documentation,
            "undocumented_priors": undocumented,
            "warning": "priors without a documented origin weaken provenance" if undocumented else None,
        }}, bio.process_step(13, [request.question], 0.10, 0.8))

        hard_evidence: dict[str, str] = {}
        soft_evidence: dict[str, dict[str, float]] = {}
        binding_report: list[dict[str, Any]] = []
        if network is not None:
            domains = network_domains(network)
            binding_by_variable: dict[str, list[EvidenceBinding]] = {}
            for binding in request.bindings:
                if binding.variable not in domains:
                    raise ValueError(f"binding references unknown variable {binding.variable}")
                binding_by_variable.setdefault(binding.variable, []).append(binding)
            for binding in request.bindings:
                if binding.hard_value is not None:
                    if binding.hard_value not in domains[binding.variable]:
                        raise ValueError(
                            f"hard evidence state {binding.hard_value} is outside the domain "
                            f"of {binding.variable}")
                    hard_evidence[binding.variable] = binding.hard_value
                    binding_report.append({"variable": binding.variable,
                                           "binding": "hard",
                                           "evidence_id": binding.evidence_id,
                                           "value": binding.hard_value})
                else:
                    weights = {state: float(binding.likelihood.get(state, 0.0))
                               for state in domains[binding.variable]}
                    soft_evidence[binding.variable] = weights
                    binding_report.append({"variable": binding.variable,
                                           "binding": "soft",
                                           "evidence_id": binding.evidence_id})
            missing_records = [record for record in request.evidence
                               if record.missing_value_code is not None]
            binding_summary = {
                "bindings": binding_report,
                "missing_evidence_is_not_negative": True,
                "explicitly_missing_evidence": [record.evidence_id
                                                for record in missing_records],
            }
        else:
            binding_summary = {"not_applicable": "no Bayesian network supplied"}
        _append_operation(trace, 13, {"evidence_bindings": binding_summary},
                          bio.process_step(14, [request.question], 0.10, 0.8))

        # Deduplicate dependent evidence before counting it as independent support.
        representative_ids = {group["representative"] for group in dependence["groups"]}
        effective_hard: dict[str, str] = {}
        for binding in request.bindings:
            if binding.hard_value is None:
                continue
            record = active.evidence.get(binding.evidence_id)
            if record is None or binding.evidence_id in representative_ids:
                effective_hard[binding.variable] = binding.hard_value

        hypothesis_variables = sorted({record.target_variable for record in request.hypotheses})
        method_selection: dict[str, Any] = {"not_applicable": "no network or hypotheses"}
        if network is not None and hypothesis_variables:
            method_selection = {
                "selected": select_inference_method(network, config.enumeration_state_limit),
                "joint_state_space": _joint_size(network),
                "per_query": {},
            }
        _append_operation(trace, 14, {"inference_method_selection": method_selection},
                          bio.process_step(15, [request.question], 0.10, 0.8))

        posteriors: dict[str, dict[str, float]] = {}
        reproducibility: dict[str, Any] = {
            "random_seed": config.random_seed,
            "config_digest": content_digest({
                "decision_threshold": config.decision_threshold,
                "enumeration_state_limit": config.enumeration_state_limit,
                "sensitivity_perturbation": config.sensitivity_perturbation,
                "calibration_bins": config.calibration_bins,
                "random_seed": config.random_seed,
            }),
            "network_digest": network_fingerprint(network) if network is not None else None,
            "evidence_digest": content_digest({
                "hard": hard_evidence, "soft": soft_evidence,
                "groups": dependence["independent_group_count"]}),
        }
        inference_runs: list[dict[str, Any]] = []
        inference_skipped = False
        if network is not None and hypothesis_variables:
            for variable in hypothesis_variables:
                cost = _joint_size(network)
                if not budget.charge(cost):
                    inference_skipped = True
                    break
                try:
                    result = run_posterior_inference(
                        network, variable, effective_hard, soft_evidence,
                        config.enumeration_state_limit)
                except ValueError:
                    result = None
                if result is None:
                    continue
                posteriors[variable] = result["posterior"]
                inference_runs.append({"query": variable, "method": result["method"],
                                       "converged": result.get("converged", True)})
        if inference_skipped or (network is not None and hypothesis_variables
                                 and not posteriors):
            if budget.exhausted:
                run_status = "partial_budget_exhausted"
        _append_operation(trace, 15, {
            "posterior_distributions": posteriors,
            "reproducibility_metadata": reproducibility,
            "inference_runs": inference_runs,
        }, bio.process_step(16, [request.question], 0.15, 0.8),
            status="skipped_budget_exhausted" if inference_skipped else "completed")

        likelihoods: dict[str, dict[str, float]] = {}
        coverage: dict[str, float] = {}
        markov_blankets = _markov_blankets(parents_map) if parents_map else {}
        total_groups = max(1, dependence["independent_group_count"])
        for record in request.hypotheses:
            if network is None or inference_skipped:
                likelihoods[record.hypothesis_id] = {"value": 0.0, "unavailable": True}
                coverage[record.hypothesis_id] = 0.0
                continue
            prior = run_posterior_inference(network, record.target_variable, {}, {})
            prior_value = prior["posterior"].get(record.target_state, 0.0)
            joint = conditional_joint(network, {**effective_hard,
                                                record.target_variable: record.target_state})
            likelihood_value = joint / prior_value if prior_value > 0 else 0.0
            likelihoods[record.hypothesis_id] = {"value": likelihood_value}
            relevant = {record.target_variable} | markov_blankets.get(record.target_variable, set())
            supporting = sum(1 for group in dependence["groups"]
                             if any(active.evidence[eid].variable in relevant
                                    for eid in group["members"]
                                    if eid in active.evidence))
            coverage[record.hypothesis_id] = supporting / total_groups
        constraint_verdicts = self._apply_symbolic_constraints(request)
        evaluations = compare_hypotheses(
            request.hypotheses, {record.hypothesis_id: posteriors.get(record.target_variable, {})
                                 for record in request.hypotheses},
            likelihoods, coverage)
        for row in evaluations:
            row["constraint_verdict"] = constraint_verdicts.get(row["hypothesis_id"], "allowed")
        if request.hypotheses and network is not None:
            strengths = {record.hypothesis_id:
                         posteriors.get(record.target_variable, {}).get(record.target_state, 0.0)
                         for record in request.hypotheses}
            eligible = {key: value for key, value in strengths.items()
                        if constraint_verdicts.get(key, "allowed") != "forbidden"}
            competition = (bio.compete_hypotheses(eligible) if eligible
                           else {"winner": None, "shares": {},
                                 "note": "no hypotheses eligible after symbolic constraints"})
            hypothesis_summary = {"ranking": evaluations, "competition": competition}
        else:
            hypothesis_summary = {"ranking": evaluations,
                                  "not_applicable": "no hypotheses or network supplied"}
        _append_operation(trace, 16, {"hypothesis_ranking": hypothesis_summary},
                          bio.process_step(17, [request.question], 0.15, 0.8))

        sensitivity: dict[str, Any] = {"not_applicable": "no network supplied"}
        if network is not None and hypothesis_variables and not inference_skipped:
            target = hypothesis_variables[0]
            sensitivity = prior_sensitivity_analysis(
                network, target, effective_hard, soft_evidence,
                config.sensitivity_perturbation, decision_threshold)
            if request.hypotheses and likelihoods:
                hypothesis_level = hypothesis_prior_sensitivity(
                    request.hypotheses, likelihoods, config.sensitivity_perturbation)
                sensitivity["hypothesis_level"] = hypothesis_level
        _append_operation(trace, 17, {"sensitivity_report": sensitivity},
                          bio.process_step(18, [request.question], 0.15, 0.8))

        ig_ranking: list[dict[str, Any]] = []
        if network is not None and request.observation_candidates and not inference_skipped:
            if budget.charge(_joint_size(network) * len(request.observation_candidates)):
                try:
                    ig_ranking = expected_information_gain(
                        network, hypothesis_variables[0] if hypothesis_variables else
                        request.observation_candidates[0], effective_hard, soft_evidence,
                        request.observation_candidates, request.acquisition_costs)
                except ValueError:
                    ig_ranking = []
        _append_operation(trace, 18, {"information_gain_ranking": ig_ranking},
                          bio.process_step(19, [request.question], 0.15, 0.8))

        calibration: dict[str, Any] = {"status": "uncalibrated",
                                       "reason": "no held-out labelled predictions supplied"}
        approximate = any(run["method"].endswith("sampling") for run in inference_runs)
        if request.calibration_labels:
            predictions = [float(p) for p, _ in request.calibration_labels]
            labels = [int(y) for _, y in request.calibration_labels]
            calibration = {
                "status": "calibrated_measured",
                "brier_score": brier_score(predictions, labels),
                "log_loss": log_loss(predictions, labels),
                "expected_calibration_error": expected_calibration_error(
                    predictions, labels, config.calibration_bins),
                "reliability_diagram": reliability_diagram(predictions, labels,
                                                           config.calibration_bins),
                "held_out_labelled_data": True,
            }
        elif approximate:
            calibration["status"] = "approximate_uncalibrated"
        _append_operation(trace, 19, {
            "calibration_status": calibration,
            "final_distributions": {name: dict(dist) for name, dist in posteriors.items()},
        }, bio.process_step(20, [request.question], 0.15, 0.8))

        # -- Section C: probabilistic programming and model validation --------

        pp_routing = probabilistic_programming_status()
        pp_routing["routed"] = False
        pp_routing["reason"] = "discrete network inference is adequate for this request"
        if inference_skipped and pp_routing["any_available"]:
            pp_routing["routed"] = True
            pp_routing["reason"] = ("exact inference infeasible under the budget and an "
                                    "adapter is installed")
        _append_operation(trace, 20, {"pp_routing_decision": pp_routing},
                          bio.process_step(21, [request.question], 0.10, 0.8))

        schema_payload = self._network_schema_payload(network, request)
        schema_validation = validate_model_schema(schema_payload)
        _append_operation(trace, 21, {"schema_validation": schema_validation},
                          bio.process_step(22, [request.question], 0.10, 0.8))

        predictive_checks: dict[str, Any] = {"not_applicable": "no network supplied"}
        if network is not None and not inference_skipped:
            if budget.charge(config.prior_predictive_samples + config.posterior_predictive_samples):
                predictive_checks = {
                    "prior_predictive": prior_predictive_check(
                        network, config.random_seed, config.prior_predictive_samples),
                    "posterior_predictive": posterior_predictive_check(
                        network, effective_hard, soft_evidence),
                }
        _append_operation(trace, 22, {"predictive_check_reports": predictive_checks},
                          bio.process_step(23, [request.question], 0.15, 0.8))

        convergence: dict[str, Any] = {
            "inference_runs": inference_runs,
            "all_converged": all(run.get("converged", True) for run in inference_runs) and
            (not inference_skipped),
            "predictive_checks_passed": (predictive_checks.get("prior_predictive", {}).get("passed")
                                         and predictive_checks.get("posterior_predictive",
                                                                   {}).get("passed", False))
            if predictive_checks != {"not_applicable": "no network supplied"} else None,
        }
        if inference_skipped:
            convergence["inconclusive_reason"] = "resource budget exhausted"
        _append_operation(trace, 23, {"convergence_report": convergence},
                          bio.process_step(24, [request.question], 0.15, 0.8))

        model_entry: dict[str, Any] = {"not_applicable": "no network supplied"}
        if network is not None:
            model_entry = {
                "network_id": network.network_id,
                "model_version": network.version,
                "model_digest": network_fingerprint(network),
                "data_snapshot_digest": reproducibility["evidence_digest"],
                "assumptions": list(network.assumptions),
                "prior_sources": dict(prior_documentation),
                "parameters": {"variables": len(network.variables), "cpts": len(network.cpts)},
                "reproducibility": reproducibility,
                "content_addressed": True,
            }
            active.networks[network.network_id] = network
        _append_operation(trace, 24, {"model_registry_entry": model_entry},
                          bio.process_step(25, [request.question], 0.10, 0.8))

        # -- Section D: sequential hidden-state reasoning (steps 26-30) -------

        hmm_report: dict[str, Any] = {"not_applicable": "no HMM supplied"}
        if request.hmm is not None:
            hmm_report = {"model_id": request.hmm.model_id,
                          "hidden_states": list(request.hmm.hidden_states)}
        _append_operation(trace, 25, {"hmm_model": hmm_report},
                          bio.process_step(26, [request.question], 0.10, 0.8))

        hmm_validation: dict[str, Any] = {"not_applicable": "no HMM supplied"}
        if request.hmm is not None:
            try:
                validated = validate_hmm_spec(
                    request.hmm.hidden_states, request.hmm.observation_states,
                    request.hmm.initial_distribution, request.hmm.transition_matrix,
                    request.hmm.emission_matrix)
                hmm_validation = {"valid": True, "states": validated["hidden_states"]}
            except ValueError as exc:
                raise ValueError(f"HMM validation failed: {exc}") from exc
        _append_operation(trace, 26, {"hmm_validation": hmm_validation},
                          bio.process_step(27, [request.question], 0.10, 0.8))

        sequential: dict[str, Any] = {"not_applicable": "no HMM supplied"}
        shift_report: dict[str, Any] = {"not_applicable": "no HMM supplied"}
        if request.hmm is not None and request.hmm_observations:
            if not budget.charge(len(request.hmm_observations) * len(request.hmm.hidden_states) ** 2):
                sequential = {"status": "skipped_budget_exhausted"}
                run_status = run_status if run_status != "completed" else "partial_budget_exhausted"
            else:
                filtering = forward_filter(
                    request.hmm.hidden_states, request.hmm.transition_matrix,
                    request.hmm.emission_matrix, request.hmm.initial_distribution,
                    request.hmm_observations)
                sequential = {"filtering": filtering}
                if request.hmm_goal in {"smooth", "monitor"} and all(
                        observation is not None for observation in request.hmm_observations):
                    sequential["smoothing"] = backward_smoothing(
                        request.hmm.hidden_states, request.hmm.transition_matrix,
                        request.hmm.emission_matrix, request.hmm_observations,
                        request.hmm.initial_distribution)
                elif request.hmm_goal == "smooth":
                    sequential["smoothing"] = {
                        "skipped": "smoothing requires complete future observations"}
                if request.hmm_goal == "decode":
                    sequential["viterbi"] = viterbi_decode(
                        request.hmm.hidden_states, request.hmm.transition_matrix,
                        request.hmm.emission_matrix, request.hmm.initial_distribution,
                        request.hmm_observations)
                shift_report = detect_distribution_shift(filtering["filtered"])
                sequential["explicit_missingness"] = {
                    "missing_steps": filtering["missing_steps"],
                    "missing_evidence_is_not_negative": True,
                }
        _append_operation(trace, 27, {"filtering_results": sequential},
                          bio.process_step(28, [request.question], 0.15, 0.8))

        viterbi_report: dict[str, Any] = {"not_applicable": "decoding was not the declared goal"}
        if request.hmm is not None and request.hmm_goal == "decode" and "viterbi" in sequential:
            viterbi_report = sequential["viterbi"]
        _append_operation(trace, 28, {"viterbi_results": viterbi_report},
                          bio.process_step(29, [request.question], 0.15, 0.8))

        _append_operation(trace, 29, {"shift_report": shift_report,
                                      "explicit_missingness": sequential.get(
                                          "explicit_missingness", {})},
                          bio.process_step(30, [request.question], 0.15, 0.8))

        # -- Section E: causal and counterfactual reasoning (steps 31-36) -----

        causal_representation: dict[str, Any] = {
            "not_applicable": "no causal DAG supplied",
            "semantic_relation_types_are_not_causal_edges": True,
        }
        if request.causal_dag is not None:
            active.causal_models[request.causal_dag.model_id] = request.causal_dag
            causal_representation = {
                "model_id": request.causal_dag.model_id,
                "variables": list(request.causal_dag.variables),
                "edges": {f"{edge.source}->{edge.target}": {
                    "confidence_class": edge.confidence_class,
                    "evidence_basis": list(edge.evidence_basis),
                    "assumptions": list(edge.assumptions),
                } for edge in request.causal_dag.edges},
                "kept_separate_from_usg_semantic_relations": True,
            }
        _append_operation(trace, 30, {"causal_dag_representation": causal_representation},
                          bio.process_step(31, [request.question], 0.10, 0.8))

        role_analysis: dict[str, Any] = {"not_applicable": "no causal DAG supplied"}
        if request.causal_dag is not None:
            exposure = next((query.interventions and next(iter(query.interventions))
                             for query in request.queries if query.kind == "intervention"
                             and query.interventions), None)
            outcome = next((query.target for query in request.queries
                            if query.kind == "intervention"), None)
            if exposure is None or outcome is None:
                exposure, outcome = self._default_causal_pair(request.causal_dag)
            if exposure and outcome and exposure != outcome:
                role_analysis = classify_causal_roles(request.causal_dag, exposure, outcome)
            else:
                role_analysis = {"not_applicable": "causal DAG has no eligible pair"}
        _append_operation(trace, 31, {"causal_role_analysis": role_analysis},
                          bio.process_step(32, [request.question], 0.10, 0.8))

        structural_model: dict[str, Any] = {"not_applicable": "no SCM supplied"}
        if request.scm is not None:
            scm_validation = validate_scm(request.scm)
            if not scm_validation["valid"]:
                raise ValueError(f"SCM rejected: {scm_validation['errors']}")
            active.scms[request.scm.model_id] = request.scm
            structural_model = {
                "model_id": request.scm.model_id,
                "endogenous": list(request.scm.variables),
                "exogenous": list(request.scm.exogenous),
                "equations": [f"{equation.variable} = f({', '.join(equation.parents)})"
                              for equation in request.scm.equations],
                "assumptions": list(request.scm.assumptions),
            }
        _append_operation(trace, 32, {"structural_model": structural_model},
                          bio.process_step(33, [request.question], 0.10, 0.8))

        causal_findings: list[dict[str, Any]] = []
        query_classification: list[dict[str, Any]] = []
        intervention_results: dict[str, Any] = {}
        for query in request.queries:
            entry = {"query_id": query.query_id, "kind": query.kind, "target": query.target}
            if query.kind == "observational" and network is not None:
                entry["note"] = "P(Y|X) is an associational quantity, not an intervention effect"
                if not inference_skipped and budget.charge(_joint_size(network)):
                    observational = run_posterior_inference(
                        network, query.target, {**effective_hard, **query.conditions}, soft_evidence)
                    entry["posterior"] = observational["posterior"]
                    entry["method"] = observational["method"]
                    causal_findings.append(entry)
                query_classification.append(entry)
            elif query.kind == "intervention":
                interventions = query.interventions
                entry["interventions"] = dict(interventions)
                entry["note"] = "P(Y|do(X)) is computed by graph surgery, not conditioning"
                if network is not None and not inference_skipped:
                    exposure = next(iter(interventions))
                    state = interventions[exposure]
                    adjustment = (find_backdoor_adjustment(request.causal_dag, exposure,
                                                           query.target)
                                  if request.causal_dag is not None else
                                  {"adjustment_set": None, "identifiable": False,
                                   "reason": "no causal DAG supplied"})
                    entry["identifiability"] = adjustment
                    if adjustment["identifiable"]:
                        entry["backdoor_adjustment"] = backdoor_adjustment_query(
                            network, exposure, state, query.target, adjustment["adjustment_set"])
                    mutilated = intervention_query(network, interventions, query.target)
                    entry["mutilated_graph"] = mutilated
                    causal_findings.append(entry)
                query_classification.append(entry)
            elif query.kind == "counterfactual":
                entry["evidence"] = dict(query.evidence)
                entry["interventions"] = dict(query.interventions)
                query_classification.append(entry)
            else:
                query_classification.append(entry)
        intervention_results["queries"] = query_classification
        _append_operation(trace, 33, {"query_classification": query_classification,
                                      "intervention_results": intervention_results},
                          bio.process_step(34, [request.question], 0.15, 0.8))

        counterfactuals: list[dict[str, Any]] = []
        if request.scm is not None:
            for query in request.queries:
                if query.kind != "counterfactual":
                    continue
                if not budget.charge(64):
                    run_status = run_status if run_status != "completed" else "partial_budget_exhausted"
                    break
                result = counterfactual_query(
                    request.scm, query.evidence, query.interventions, query.target)
                result["query_id"] = query.query_id
                counterfactuals.append(result)
        _append_operation(trace, 34, {"counterfactual_results": counterfactuals},
                          bio.process_step(35, [request.question], 0.15, 0.8))

        counterfactual_comparison = {
            "not_applicable": "no counterfactual queries"} if not counterfactuals else {
            "comparisons": [{
                "query_id": item["query_id"],
                "differences": item["differences"],
                "status": item["status"],
                "assumptions": item["assumptions"],
                "not_reported_as_observed_fact": True,
            } for item in counterfactuals],
        }
        _append_operation(trace, 35, {"counterfactual_comparison": counterfactual_comparison},
                          bio.process_step(36, [request.question], 0.15, 0.8))

        # -- Section F: conflict, publication and learning (steps 37-40) ------

        conflict_report = evidence_conflict_analysis(request.evidence, variables_map)
        conflict_report["source_dependence"] = dependence
        _append_operation(trace, 36, {"conflict_report": conflict_report},
                          bio.process_step(37, [request.question], 0.10, 0.8))

        gates = self._evaluate_publication_gates(
            request, validation, calibration, convergence, conflict_report, sensitivity,
            role_analysis, budget, schema_validation)
        _append_operation(trace, 37, {"publication_gates": gates,
                                      "gate_decision": gates["decision"]},
                          bio.process_step(38, [request.question], 0.10, 0.8))

        publication_payload = {
            "question": request.question,
            "hypothesis_evaluations": evaluations,
            "posterior_distributions": posteriors,
            "calibration_status": calibration["status"],
            "causal_findings": causal_findings,
            "counterfactuals": counterfactuals,
            "conflict_report": {name: report for name, report in
                                conflict_report["variables"].items()},
            "sensitivity_any_reversal": bool(sensitivity.get("any_reversal", False)),
            "epistemic_status": "probabilistic_inference_not_proven_fact",
        }
        publication_report = self._publish_findings(
            request, gates, publication_payload, posteriors, evaluations, decision_threshold,
            workspace_handle)
        _append_operation(trace, 38, {"publication_report": publication_report},
                          bio.process_step(39, [request.question], 0.10, 0.8))

        learning_governance = self._record_predictions_and_governance(
            active, request, evaluations, posteriors, trace, started)
        _append_operation(trace, 39, {"learning_governance": learning_governance},
                          bio.process_step(40, [request.question], 0.10, 0.8))

        if cancelled():
            run_status = "cancelled_partial"

        active.run_count += 1
        confidence = {
            "probability_is_not_truth": True,
            "posterior_support": {name: max(dist.items(), key=lambda item: item[1])
                                  for name, dist in posteriors.items()} if posteriors else {},
            "calibration_status": calibration["status"],
            "sensitivity_any_reversal": bool(sensitivity.get("any_reversal", False)),
            "conflict_detected": conflict_report["any_hard_conflict"],
            "dependent_duplicate_evidence": dependence["dependent_duplicate_count"],
            "causal_identifiability": role_analysis.get("warnings", []) if role_analysis else [],
            "uncertainty_sources": ["model assumptions", "prior sensitivity", "source dependence",
                                    "measurement uncertainty"],
        }
        output = {
            "run_status": run_status,
            "budget_used": budget.used,
            "budget_total": budget.total,
            "hypothesis_count": len(evaluations),
            "query_count": len(request.queries),
            "counterfactual_count": len(counterfactuals),
            "inference_method": inference_runs[0]["method"] if inference_runs else None,
            "publication_decision": publication_report.get("decision"),
            "software_correctness_is_separate_from_scientific_validity": True,
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        return BayesianCausalResult(
            runner_id=self.runner_id, operation_trace=tuple(trace),
            intake_report={"request_frame": request_frame, "run_record": run_record,
                           "identity_resolution": identity_resolution,
                           "retrieved_inputs": retrieved, "normalization": normalization,
                           "classification": intake, "confidence_separation": separation,
                           "source_dependence": dependence},
            hypothesis_evaluations=tuple(evaluations),
            posterior_distributions={"posteriors": posteriors,
                                     "reproducibility": reproducibility},
            diagnostics={"calibration": calibration, "sensitivity": sensitivity,
                         "predictive_checks": predictive_checks, "convergence": convergence,
                         "information_gain_ranking": ig_ranking,
                         "model_registry_entry": model_entry,
                         "schema_validation": schema_validation, "pp_routing": pp_routing},
            sequential_results={"hmm": sequential, "shift_report": shift_report,
                                "viterbi": viterbi_report},
            causal_findings=tuple(causal_findings),
            counterfactual_results=tuple(counterfactuals),
            conflict_report=conflict_report,
            publication_report=publication_report,
            confidence_and_uncertainty=confidence,
            biological=bio.snapshot(),
            final_state=active.to_dict(),
            output=output,
        )

    # ------------------------------------------------------------------
    # Integration helpers
    # ------------------------------------------------------------------

    def _workspace_handle(self, question: str) -> Any:
        """Return a SharedWorkspace, creating one when a WorkspaceManager is supplied."""
        if self.workspace is None:
            return None
        if hasattr(self.workspace, "create_workspace"):
            return self.workspace.create_workspace(question, "bayesian_causal_reasoning")
        return self.workspace

    def _default_causal_pair(self, dag: Any) -> tuple[str | None, str | None]:
        """Pick an exposure/outcome pair from edge structure when no query declares one."""
        if dag.edges:
            first = dag.edges[0]
            return first.source, first.target
        names = list(dag.variables)
        if len(names) >= 2:
            return names[0], names[1]
        return None, None

    def _resolve_usg_identities(self, request: BayesianCausalRequest) -> dict[str, Any]:
        """Resolve variables to USG identifiers while preserving unresolved alternatives."""
        names = [variable.name for variable in request.variables]
        if not names and request.network is not None:
            names = [variable.name for variable in request.network.variables]
        if request.usg_resource is None:
            return {"resolved": {}, "unresolved": {name: ["no_usg_resource_supplied"]
                                                   for name in names},
                    "alternatives_preserved": True}
        resolved: dict[str, Any] = {}
        unresolved: dict[str, Any] = {}
        try:
            from ore_usg.retrieval import USGKnowledgeIndex
            index = USGKnowledgeIndex(request.usg_resource)
            for name in names:
                result = index.lookup(name)
                if result.items:
                    resolved[name] = [item.record_id for item in result.items]
                else:
                    unresolved[name] = [f"no_exact_match_for_{name}"]
        except Exception as exc:
            unresolved = {name: [f"usg_resolution_error:{type(exc).__name__}"] for name in names}
        return {"resolved": resolved, "unresolved": unresolved,
                "alternatives_preserved": True}

    def _apply_symbolic_constraints(self, request: BayesianCausalRequest) -> dict[str, str]:
        """Apply Point 3 neuro-symbolic constraints without erasing probabilistic semantics."""
        verdicts: dict[str, str] = {record.hypothesis_id: "allowed"
                                    for record in request.hypotheses}
        for constraint in request.symbolic_constraints:
            hypothesis_id = str(constraint.get("hypothesis_id", ""))
            verdict = str(constraint.get("verdict", "allowed"))
            if hypothesis_id in verdicts and verdict in {"allowed", "forbidden"}:
                verdicts[hypothesis_id] = verdict
        return verdicts

    def _network_schema_payload(self, network: Any, request: BayesianCausalRequest) -> dict[str, Any]:
        """Build a JSON-safe structural description for sandbox validation."""
        if network is None:
            return {"kind": "point11_request", "question": request.question,
                    "variables": {}, "observed": {}}
        return {
            "kind": "discrete_bayesian_network",
            "variables": {variable.name: {"family": "categorical",
                                          "states": list(variable.states)}
                          for variable in network.variables},
            "observed": {name: list(cpt.parents) for name, cpt in
                         ((cpt.variable, cpt) for cpt in network.cpts)},
        }

    def _evaluate_publication_gates(
        self, request: BayesianCausalRequest, validation: Mapping[str, Any],
        calibration: Mapping[str, Any], convergence: Mapping[str, Any],
        conflict_report: Mapping[str, Any], sensitivity: Mapping[str, Any],
        role_analysis: Mapping[str, Any], budget: _ResourceBudget,
        schema_validation: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Evaluate every publication gate from section F of the specification."""
        gates = {
            "model_validity": bool(validation.get("valid", True)) if validation != {
                "not_applicable": "no Bayesian network supplied"} else True,
            "schema_validity": bool(schema_validation.get("valid", True)),
            "calibration_or_convergence": bool(calibration.get("status") == "calibrated_measured"
                                                or convergence.get("all_converged", True)),
            "evidence_lineage": all(record.source is not None or record.lineage
                                    for record in request.evidence),
            "causal_identifiability": not bool(role_analysis.get("warnings")) if isinstance(
                role_analysis, Mapping) and "warnings" in role_analysis else True,
            "sensitivity": not bool(sensitivity.get("any_reversal", False)) if isinstance(
                sensitivity, Mapping) and "any_reversal" in sensitivity else True,
            "budget_completion": not budget.exhausted,
            "unresolved_conflicts": not conflict_report.get("any_hard_conflict", False),
        }
        decision = "pass" if all(gates.values()) else "blocked"
        return {"gates": gates, "decision": decision,
                "blocked_reasons": [name for name, passed in gates.items() if not passed]}

    def _publish_findings(
        self, request: BayesianCausalRequest, gates: Mapping[str, Any],
        payload: Mapping[str, Any], posteriors: Mapping[str, Any],
        evaluations: Sequence[Mapping[str, Any]], decision_threshold: float,
        workspace_handle: Any = None,
    ) -> dict[str, Any]:
        """Publish typed findings through the existing workspace and coordinator APIs."""
        report: dict[str, Any] = {"decision": gates["decision"], "published": False,
                                  "workspace": "not_configured",
                                  "gates": dict(gates.get("gates", {})),
                                  "blocked_reasons": gates.get("blocked_reasons", [])}
        if gates["decision"] != "pass":
            report["withheld_reasons"] = gates["blocked_reasons"]
            report["note"] = "publication gates blocked this finding set"
            return report
        best_confidence = None
        if evaluations:
            best = evaluations[0]
            best_confidence = float(best.get("posterior", 0.0))
        report["independent_review_requested"] = bool(
            best_confidence is not None and best_confidence >= decision_threshold)
        report["coordinator_handoff"] = "not_configured"
        if self.coordinator_handoff is not None:
            try:
                self.coordinator_handoff({
                    "producer_module": "ore_bayesian_causal",
                    "question": request.question,
                    "finding": dict(payload),
                })
                report["coordinator_handoff"] = "sent"
            except Exception as exc:
                report["coordinator_handoff"] = f"failed:{type(exc).__name__}"
        if self.finding_publisher is not None or workspace_handle is not None:
            try:
                publisher = self.finding_publisher
                if publisher is None:
                    from ore_workspace import FindingPublisher, Provenance
                    publisher = FindingPublisher(workspace_handle, "ore_bayesian_causal")
                else:
                    from ore_workspace import Provenance
                provenance = Provenance(
                    source="ore_bayesian_causal.BayesianCausalProcessor",
                    rules_applied=(
                        "bayes_rule_with_explicit_normalization",
                        "bayesian_network_factorization",
                        "separate_confidence_dimensions",
                        "backdoor_criterion_before_intervention_claims",
                        "abduction_action_prediction_counterfactuals",
                    ),
                    notes="Probabilistic and causal inference output; posterior confidence "
                          "is not logical truth and association is not causation.",
                )
                decision = publisher.publish(
                    "bayesian_causal_reasoning", dict(payload),
                    finding_type="probabilistic_causal_hypothesis",
                    provenance=provenance,
                    confidence=best_confidence,
                )
                decision_value = getattr(decision, "decision", None)
                report["workspace"] = str(getattr(decision_value, "value", decision_value))
                report["published"] = True
            except Exception as exc:
                report["workspace"] = f"failed:{type(exc).__name__}"
        return report

    def _record_predictions_and_governance(
        self, state: BayesianCausalState, request: BayesianCausalRequest,
        evaluations: Sequence[Mapping[str, Any]], posteriors: Mapping[str, Any],
        trace: list[dict[str, Any]], started: float,
    ) -> dict[str, Any]:
        """Log events and register governed, versioned learning (never self-labelling)."""
        for row in evaluations:
            state.predictions.append({
                "hypothesis_id": row["hypothesis_id"],
                "predicted_probability": row["posterior"],
                "verified_outcome": None,
                "prediction_time": time.time(),
                "question": request.question,
            })
        governance = {
            "events_logged": len(trace),
            "predictions_awaiting_external_verification": len(evaluations),
            "learning_updates": [],
            "unverified_ore_outputs_used_as_labels": False,
            "policy": "priors and model parameters update only through governed, "
                      "versioned learning with externally verified outcomes",
        }
        for prediction in state.predictions:
            if prediction.get("verified_outcome") is not None:
                governance["learning_updates"].append({
                    "hypothesis_id": prediction["hypothesis_id"],
                    "verified_outcome": prediction["verified_outcome"],
                    "governed": True,
                })
        state.learning_updates.extend(governance["learning_updates"])
        return governance


def _joint_size(network: Any) -> int:
    """Return the joint assignment count of a network for budget accounting."""
    from .functional import assignment_space
    return assignment_space(network) if network is not None else 1


def _markov_blankets(parents_map: Mapping[str, Sequence[str]]) -> dict[str, set[str]]:
    """Return the Markov blanket (parents, children, co-parents) of every variable."""
    children: dict[str, set[str]] = {name: set() for name in parents_map}
    for name, parent_list in parents_map.items():
        for parent in parent_list:
            children.setdefault(parent, set()).add(name)
    blankets: dict[str, set[str]] = {}
    for name in parents_map:
        blanket = set(parents_map.get(name, ())) | children.get(name, set())
        for child in children.get(name, set()):
            blanket |= set(parents_map.get(child, ()))
        blanket.discard(name)
        blankets[name] = blanket
    return blankets


def register_annotations(
    database_path: str | Path, repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index every Point 11 class and callable through the shared AST annotation catalogue."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase
    root = Path(repository_root).expanduser().resolve() if repository_root else Path(
        __file__).resolve().parents[1]
    return sync_repository(LoggerDatabase(database_path), root, include_paths={"ore_bayesian_causal"})


def record_result_to_sqlite(
    result: BayesianCausalResult, database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist all 40 operation records and index Point 11 class/function annotations in SQLite."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase
    root = Path(repository_root).expanduser().resolve() if repository_root else Path(
        __file__).resolve().parents[1]
    database = LoggerDatabase(database_path)
    catalogue = sync_repository(database, root, include_paths={"ore_bayesian_causal"})
    run_id = database.create_run(
        script_path="ore_bayesian_causal/point11", module_name="ore_bayesian_causal.pipeline",
        entry_kind="bayesian_causal_inference", working_directory=str(root),
    )
    started = time.perf_counter()
    try:
        for sequence, item in enumerate(result.operation_trace, start=1):
            database.record_event(
                run_id=run_id, sequence_no=sequence, event_kind="operation",
                source_path="ore_bayesian_causal/pipeline.py",
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                metadata={"operation_id": item["operation_id"], "operation": item["operation"],
                          "status": item["status"], "source_step": item["source_step"],
                          "evidence_level": item["evidence_level"],
                          "mathematical_form": item["mathematical_form"],
                          "functional_output": item["functional_output"],
                          "biological_output": item["biological_output"]},
            )
        database.finish_run(run_id, "completed",
                            duration_ms=round((time.perf_counter() - started) * 1000.0, 3))
    except Exception as exc:
        database.finish_run(run_id, "failed",
                            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
                            error_summary=type(exc).__name__ + ": " + str(exc))
        raise
    return {"run_id": run_id, "annotation_catalogue": catalogue,
            "database_summary": database.summary()}
