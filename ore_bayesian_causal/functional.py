"""Functional operations for Point 11: DAG inference, causal reasoning and audits.

This module implements the mathematical operations of the Point 11 specification:
Bayesian-network factorization and exact inference, information gain, prior
sensitivity, causal structure analysis, backdoor adjustment, mutilation-based
interventions, abduction-action-prediction counterfactuals, evidence conflict
handling with Dempster-Shafer bounds, and model validation.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .contracts import (
    BayesianNetworkSpec,
    CausalDAGSpec,
    CPTSpec,
    EvidenceRecord,
    HypothesisRecord,
    SCMSpec,
    StructuralEquation,
    VariableSpec,
)
from .math_ops import (
    DempsterShaferResult,
    bayes_rule,
    combine_mass_functions,
    effective_sample_size,
    evidence_to_mass,
    normalize_weights,
    sample_from_distribution,
    seeded_random,
    shannon_entropy,
    validate_distribution,
    weighted_std_error,
)

# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def stable_id(prefix: str, *parts: str) -> str:
    """Return a deterministic content-addressed identifier from the supplied parts."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}"


def content_digest(value: Any) -> str:
    """Return a SHA-256 digest of the canonical JSON form of a JSON-compatible value."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                           default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def network_fingerprint(network: BayesianNetworkSpec) -> str:
    """Content-address a network by its structure and parameters, ignoring its label."""
    payload = {
        "variables": [{"name": variable.name, "states": list(variable.states)}
                      for variable in network.variables],
        "cpts": [{"variable": cpt.variable, "parents": list(cpt.parents),
                  "table": {key: dict(row) for key, row in cpt.table.items()}}
                 for cpt in network.cpts],
        "dynamic_unrolled": network.dynamic_unrolled,
    }
    return content_digest(payload)


# ---------------------------------------------------------------------------
# DAG structure and parameter validation (steps 9-13)
# ---------------------------------------------------------------------------


def cpt_row_key(parent_states: Sequence[str]) -> str:
    """Encode a parent-state tuple as the canonical CPT row key."""
    return "|".join(parent_states)


def topological_order(variables: Sequence[str], parents: Mapping[str, Sequence[str]],
                      dynamic_unrolled: bool = False) -> list[str]:
    """Return a topological order or raise on cycles (Kahn's algorithm)."""
    remaining = {name: set(parents.get(name, ())) & set(variables) for name in variables}
    order: list[str] = []
    ready = sorted(name for name, pending in remaining.items() if not pending)
    while ready:
        current = ready.pop(0)
        order.append(current)
        for name in sorted(remaining):
            if current in remaining[name]:
                remaining[name].discard(current)
                if not remaining[name] and name not in order and name not in ready:
                    ready.append(name)
        ready = sorted(set(ready))
    if len(order) != len(variables):
        if dynamic_unrolled:
            raise ValueError("dynamic/unrolled models must still be acyclic within one slice")
        raise ValueError("the graph contains a cycle; causal/Bayesian DAGs must be acyclic")
    return order


def network_parents(network: BayesianNetworkSpec) -> dict[str, tuple[str, ...]]:
    """Return the parent map implied by the network's CPTs."""
    return {variable.name: () for variable in network.variables} | {
        cpt.variable: tuple(cpt.parents) for cpt in network.cpts
    }


def network_domains(network: BayesianNetworkSpec) -> dict[str, tuple[str, ...]]:
    """Return the state domains of every network variable."""
    return {variable.name: tuple(variable.states) for variable in network.variables}


def validate_network(network: BayesianNetworkSpec) -> dict[str, Any]:
    """Validate DAG structure and every CPT row (steps 10-13 of the specification).

    Checks acyclicity, parent references, complete parent-state coverage, exact state
    coverage, finiteness, valid ranges and row normalization. Root CPT rows are the
    documented priors, so a missing root row is a missing prior, not a default.
    """
    domains = network_domains(network)
    parents = network_parents(network)
    errors: list[str] = []
    variables = [variable.name for variable in network.variables]

    for cpt in network.cpts:
        if any(parent not in domains for parent in cpt.parents):
            missing = [parent for parent in cpt.parents if parent not in domains]
            errors.append(f"CPT for {cpt.variable} references unknown parents {missing}")
            continue
        if cpt.parents:
            expected_keys = {cpt_row_key(values) for values in itertools.product(
                *(domains[parent] for parent in cpt.parents))}
        else:
            expected_keys = {""}
        actual_keys = set(cpt.table)
        if actual_keys != expected_keys:
            missing_rows = sorted(expected_keys - actual_keys)
            extra_rows = sorted(actual_keys - expected_keys)
            errors.append(
                f"CPT for {cpt.variable} must cover every parent combination; "
                f"missing={missing_rows} unexpected={extra_rows}")
            continue
        for key in sorted(actual_keys):
            try:
                validate_distribution(f"CPT[{cpt.variable}][{key}]", cpt.table[key],
                                      list(domains[cpt.variable]))
            except ValueError as exc:
                errors.append(str(exc))

    missing_cpts = [name for name in variables
                    if name not in {cpt.variable for cpt in network.cpts}]
    if missing_cpts:
        errors.append(f"variables without CPT/prior definition: {sorted(missing_cpts)}")

    order: list[str] = []
    if not any(parent not in domains for cpt in network.cpts for parent in cpt.parents):
        try:
            order = topological_order(variables, parents, network.dynamic_unrolled)
        except ValueError as exc:
            errors.append(str(exc))

    return {
        "valid": not errors,
        "errors": errors,
        "topological_order": order,
        "parent_map": {name: list(parents[name]) for name in variables},
        "root_variables": sorted(name for name in variables if not parents[name]),
        "prior_documentation": {
            cpt.variable: cpt.prior_source for cpt in network.cpts if not cpt.parents
        },
        "network_digest": network_fingerprint(network),
    }


# ---------------------------------------------------------------------------
# Factor algebra and exact inference (steps 14-16)
# ---------------------------------------------------------------------------


@dataclass
class Factor:
    """One discrete factor over a tuple of variables with its full table."""

    variables: tuple[str, ...]
    domains: dict[str, tuple[str, ...]]
    table: dict[tuple[str, ...], float] = field(default_factory=dict)

    def value(self, assignment: Mapping[str, str]) -> float:
        """Return the factor value for one full assignment of its variables."""
        key = tuple(assignment[name] for name in self.variables)
        return self.table.get(key, 0.0)

    def restrict(self, variable: str, state: str) -> Factor:
        """Fix one variable to a state (hard evidence conditioning)."""
        if variable not in self.variables:
            return self
        index = self.variables.index(variable)
        new_variables = tuple(name for name in self.variables if name != variable)
        table = {key[:index] + key[index + 1:]: value
                 for key, value in self.table.items() if key[index] == state}
        return Factor(new_variables, self.domains, table)

    def multiply(self, other: Factor) -> Factor:
        """Pointwise multiply two factors over the union of their variables."""
        union = tuple(dict.fromkeys(self.variables + other.variables))
        merged_domains = {**self.domains, **other.domains}
        table: dict[tuple[str, ...], float] = {}
        for combo in itertools.product(*(merged_domains[name] for name in union)):
            assignment = dict(zip(union, combo))
            left_key = tuple(assignment[name] for name in self.variables)
            right_key = tuple(assignment[name] for name in other.variables)
            product = self.table.get(left_key, 0.0) * other.table.get(right_key, 0.0)
            if product != 0.0:
                table[combo] = product
        return Factor(union, merged_domains, table)

    def sum_out(self, variable: str) -> Factor:
        """Marginalize one variable out of the factor."""
        if variable not in self.variables:
            return self
        index = self.variables.index(variable)
        new_variables = self.variables[:index] + self.variables[index + 1:]
        table: dict[tuple[str, ...], float] = {}
        for key, value in self.table.items():
            reduced = key[:index] + key[index + 1:]
            table[reduced] = table.get(reduced, 0.0) + value
        return Factor(new_variables, self.domains, table)

    def size(self) -> int:
        """Return the number of table cells the factor could occupy."""
        product = 1
        for name in self.variables:
            product *= len(self.domains[name])
        return product


def factor_from_cpt(cpt: CPTSpec, domains: Mapping[str, Sequence[str]]) -> Factor:
    """Build a factor from one CPT with variables ordered child-first."""
    variables = (cpt.variable, *cpt.parents)
    domains_full = {name: tuple(domains[name]) for name in variables}
    table: dict[tuple[str, ...], float] = {}
    for key, row in cpt.table.items():
        parent_states = key.split("|") if key else []
        for state in domains[cpt.variable]:
            table[(state, *parent_states)] = float(row.get(state, 0.0))
    return Factor(variables, domains_full, table)


def factor_from_likelihood(variable: str, likelihood: Mapping[str, float],
                           domains: Mapping[str, Sequence[str]]) -> Factor:
    """Build a virtual-evidence factor from soft evidence weights."""
    return Factor((variable,), {variable: tuple(domains[variable])},
                  {(state,): float(likelihood.get(state, 0.0))
                   for state in domains[variable]})


def joint_probability(network: BayesianNetworkSpec, assignment: Mapping[str, str]) -> float:
    """Return P(X) = prod_i P(x_i | Pa(x_i)) for one full assignment (factorization).

    States absent from a row (point-mass intervention mechanisms) get probability zero.
    """
    parents = network_parents(network)
    product = 1.0
    for cpt in network.cpts:
        parent_states = [assignment[parent] for parent in parents[cpt.variable]]
        row = cpt.table.get(cpt_row_key(parent_states), {})
        product *= float(row.get(assignment[cpt.variable], 0.0))
        if product == 0.0:
            return 0.0
    return product


def assignment_space(network: BayesianNetworkSpec) -> int:
    """Return the number of joint assignments of the network's variables."""
    product = 1
    for variable in network.variables:
        product *= len(variable.states)
    return product


def select_inference_method(network: BayesianNetworkSpec,
                            enumeration_state_limit: int = 20000) -> str:
    """Select an inference method justified by graph/model size (step 15)."""
    if assignment_space(network) <= enumeration_state_limit:
        return "enumeration"
    return "variable_elimination"


def enumerate_posterior(network: BayesianNetworkSpec, query: str,
                        hard: Mapping[str, str],
                        soft: Mapping[str, Mapping[str, float]] | None = None,
                        ) -> dict[str, Any]:
    """Compute P(query | evidence) by exact enumeration of the joint."""
    domains = network_domains(network)
    if query not in domains:
        raise ValueError(f"query variable {query} is not part of the network")
    soft = soft or {}
    for variable, state in hard.items():
        if variable not in domains or state not in domains[variable]:
            raise ValueError(f"hard evidence {variable}={state} is outside the network domains")
    total_space = assignment_space(network)
    if total_space > 10_000_000:
        raise ValueError("enumeration refused: joint state space exceeds the audit limit")
    accumulator: dict[str, float] = {state: 0.0 for state in domains[query]}
    evidence_probability = 0.0
    names = [variable.name for variable in network.variables]
    for combo in itertools.product(*(domains[name] for name in names)):
        assignment = dict(zip(names, combo))
        if any(assignment[variable] != state for variable, state in hard.items()):
            continue
        weight = joint_probability(network, assignment)
        for variable, likelihood in soft.items():
            weight *= float(likelihood.get(assignment[variable], 0.0))
        accumulator[assignment[query]] += weight
        evidence_probability += weight
    if evidence_probability <= 0.0:
        raise ZeroDivisionError(
            "the evidence has zero probability under the model; missing evidence is not "
            "negative evidence, so this indicates a model/evidence mismatch")
    posterior = {state: value / evidence_probability for state, value in accumulator.items()}
    return {"posterior": posterior, "method": "enumeration",
            "evidence_probability": evidence_probability,
            "joint_state_space": total_space, "converged": True}


def variable_elimination(network: BayesianNetworkSpec, query: str,
                         hard: Mapping[str, str],
                         soft: Mapping[str, Mapping[str, float]] | None = None,
                         max_factor_cells: int = 250_000,
                         ) -> dict[str, Any]:
    """Compute P(query | evidence) by variable elimination with a min-degree ordering."""
    domains = network_domains(network)
    if query not in domains:
        raise ValueError(f"query variable {query} is not part of the network")
    soft = soft or {}
    for variable, state in hard.items():
        if state not in domains[variable]:
            raise ValueError(f"hard evidence {variable}={state} is outside the network domains")
    factors = [factor_from_cpt(cpt, domains) for cpt in network.cpts]
    for variable, likelihood in soft.items():
        factors.append(factor_from_likelihood(variable, likelihood, domains))
    for variable, state in hard.items():
        factors = [factor.restrict(variable, state) for factor in factors]

    eliminate = [name for name in domains if name != query and name not in hard]
    # Min-degree heuristic: repeatedly eliminate the variable touching the fewest factors.
    while eliminate:
        best = min(eliminate,
                   key=lambda name: sum(1 for factor in factors if name in factor.variables))
        eliminate.remove(best)
        related = [factor for factor in factors if best in factor.variables]
        others = [factor for factor in factors if best not in factor.variables]
        if not related:
            continue
        product = related[0]
        for factor in related[1:]:
            product = product.multiply(factor)
        if product.size() > max_factor_cells:
            raise ValueError(
                f"variable elimination exceeded the factor-cell budget ({product.size()}); "
                "approve an approximation or reduce the model")
        factors = others + [product.sum_out(best)]

    result = Factor((), {}, {(): 1.0})
    for factor in factors:
        result = result.multiply(factor)
    if result.variables != (query,):
        raise ValueError(
            f"elimination did not reduce to the query variable: {result.variables}")
    posterior = normalize_weights("posterior_factor", {
        key[0]: value for key, value in result.table.items()})
    return {"posterior": posterior, "method": "variable_elimination",
            "eliminated": [name for name in domains if name != query and name not in hard],
            "converged": True}


def run_posterior_inference(network: BayesianNetworkSpec, query: str,
                            hard: Mapping[str, str],
                            soft: Mapping[str, Mapping[str, float]] | None = None,
                            enumeration_state_limit: int = 20000,
                            ) -> dict[str, Any]:
    """Dispatch to enumeration or variable elimination (step 15-16)."""
    method = select_inference_method(network, enumeration_state_limit)
    if method == "enumeration":
        return enumerate_posterior(network, query, hard, soft)
    return variable_elimination(network, query, hard, soft)


def likelihood_weighting(network: BayesianNetworkSpec, query: str,
                         hard: Mapping[str, str],
                         soft: Mapping[str, Mapping[str, float]] | None = None,
                         samples: int = 500, seed: int = 11) -> dict[str, Any]:
    """Approximate P(query | evidence) with likelihood weighting (approved approximation).

    Every sample keeps its weight; the report includes effective sample size and a
    standard error per state so uncertainty from sampling is explicit, not hidden.
    """
    domains = network_domains(network)
    parents = network_parents(network)
    order = topological_order([variable.name for variable in network.variables], parents,
                              network.dynamic_unrolled)
    soft = soft or {}
    rng = seeded_random(seed)
    cpts = {cpt.variable: cpt for cpt in network.cpts}
    weights: list[float] = []
    query_states: list[str] = []
    for _ in range(max(1, samples)):
        assignment: dict[str, str] = {}
        weight = 1.0
        for name in order:
            cpt = cpts[name]
            parent_states = [assignment[parent] for parent in parents[name]]
            row = cpt.table.get(cpt_row_key(parent_states), {})
            if name in hard:
                assignment[name] = hard[name]
                weight *= float(row.get(hard[name], 0.0))
            else:
                assignment[name] = sample_from_distribution(
                    {state: float(row.get(state, 0.0)) for state in domains[name]}, rng)
            if name in soft:
                weight *= float(soft[name].get(assignment[name], 0.0))
        weights.append(weight)
        query_states.append(assignment[query])
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("likelihood weighting produced zero total weight")
    posterior = {state: sum(w for w, s in zip(weights, query_states) if s == state) / total
                 for state in domains[query]}
    ess = effective_sample_size(weights)
    standard_errors = {
        state: weighted_std_error([1.0 if s == state else 0.0 for s in query_states], weights)
        for state in domains[query]}
    return {"posterior": posterior, "method": "likelihood_weighting",
            "samples": samples, "seed": seed, "effective_sample_size": ess,
            "standard_errors": standard_errors, "converged": ess >= 10,
            "standard_error_status": (
                "estimated" if ess > 1.0 else "unavailable_insufficient_effective_sample_size"
            ),
            "approximate": True}


# ---------------------------------------------------------------------------
# Hypothesis comparison, sensitivity and information gain (steps 17-20)
# ---------------------------------------------------------------------------


def compare_hypotheses(hypotheses: Sequence[HypothesisRecord],
                       posteriors: Mapping[str, Mapping[str, float]],
                       likelihoods: Mapping[str, Mapping[str, float]],
                       evidence_coverage: Mapping[str, float],
                       sensitivity_flags: Mapping[str, bool] | None = None) -> list[dict[str, Any]]:
    """Rank competing hypotheses by posterior, likelihood, coverage and stability."""
    sensitivity_flags = sensitivity_flags or {}
    rows: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        hypothesis_posterior = posteriors.get(hypothesis.hypothesis_id, {})
        posterior = float(hypothesis_posterior.get(hypothesis.target_state, 0.0))
        likelihood = float(likelihoods[hypothesis.hypothesis_id].get("value", 0.0))
        rows.append({
            "hypothesis_id": hypothesis.hypothesis_id,
            "statement": hypothesis.statement,
            "target": f"{hypothesis.target_variable}={hypothesis.target_state}",
            "prior": hypothesis.prior,
            "posterior": posterior,
            "posterior_available": bool(hypothesis_posterior),
            "marginal_likelihood": likelihood,
            "evidence_coverage": float(evidence_coverage.get(hypothesis.hypothesis_id, 0.0)),
            "sensitivity_reversal_flagged": bool(
                sensitivity_flags.get(hypothesis.hypothesis_id, False)),
        })
    return sorted(rows, key=lambda row: (-row["posterior"], row["hypothesis_id"]))


def prior_sensitivity_analysis(network: BayesianNetworkSpec, query: str,
                               hard: Mapping[str, str],
                               soft: Mapping[str, Mapping[str, float]] | None = None,
                               perturbation: float = 0.2,
                               decision_threshold: float = 0.75) -> dict[str, Any]:
    """Perturb documented priors and flag conclusions that reverse (step 18)."""
    domains = network_domains(network)
    baseline = run_posterior_inference(network, query, hard, soft)
    baseline_posterior = baseline["posterior"]
    baseline_winner = max(baseline_posterior, key=baseline_posterior.get)
    scenarios: list[dict[str, Any]] = []
    for cpt in network.cpts:
        if cpt.parents:
            continue
        row = cpt.table[""]
        for state in domains[cpt.variable]:
            perturbed = {
                key: value * (1.0 + perturbation if key == state else 1.0 - perturbation / 2.0)
                for key, value in row.items()}
            perturbed = {key: max(value, 1e-9) for key, value in perturbed.items()}
            total = sum(perturbed.values())
            perturbed = {key: value / total for key, value in perturbed.items()}
            table = {"": perturbed}
            new_cpt = CPTSpec(cpt.variable, (), table, cpt.prior_source)
            replaced = tuple(
                new_cpt if existing.variable == cpt.variable else existing
                for existing in network.cpts)
            trial_network = BayesianNetworkSpec(
                network.network_id + "-sensitivity", network.variables, replaced,
                network.assumptions, network.dynamic_unrolled, network.version + 1,
                "sensitivity_analysis")
            try:
                trial = run_posterior_inference(trial_network, query, hard, soft)
            except (ValueError, ZeroDivisionError) as exc:
                scenarios.append({"perturbed_prior": f"{cpt.variable}->{state}",
                                  "error": str(exc), "reversal": True})
                continue
            trial_posterior = trial["posterior"]
            winner = max(trial_posterior, key=trial_posterior.get)
            threshold_crossing = (
                (baseline_posterior[baseline_winner] < decision_threshold <= trial_posterior[winner])
                or (baseline_posterior[baseline_winner] >= decision_threshold > trial_posterior[winner]))
            reversal = winner != baseline_winner or threshold_crossing
            scenarios.append({
                "perturbed_prior": f"{cpt.variable}->{state}",
                "direction": "boost",
                "baseline_winner": baseline_winner,
                "perturbed_winner": winner,
                "baseline_best_posterior": baseline_posterior[baseline_winner],
                "perturbed_best_posterior": trial_posterior[winner],
                "reversal": reversal,
            })
    return {"baseline_winner": baseline_winner,
            "baseline_posterior": baseline_posterior,
            "scenarios": scenarios,
            "any_reversal": any(item.get("reversal", False) for item in scenarios)}


def hypothesis_prior_sensitivity(hypotheses: Sequence[HypothesisRecord],
                                 likelihoods: Mapping[str, Mapping[str, float]],
                                 perturbation: float = 0.2) -> dict[str, Any]:
    """Check whether prior perturbation flips the winning hypothesis (Bayes level)."""
    prior = {record.hypothesis_id: record.prior for record in hypotheses}
    total = sum(prior.values())
    if total <= 0:
        raise ValueError("hypothesis priors must have positive total mass")
    normalized = {key: value / total for key, value in prior.items()}
    likelihood = {record.hypothesis_id: float(likelihoods[record.hypothesis_id]["value"])
                  for record in hypotheses}
    baseline = bayes_rule(normalized, likelihood)
    baseline_winner = max(baseline, key=baseline.get)
    scenarios = []
    for record in hypotheses:
        boosted = dict(normalized)
        boosted[record.hypothesis_id] = min(1.0, boosted[record.hypothesis_id] * (1.0 + perturbation))
        others = [key for key in boosted if key != record.hypothesis_id]
        remaining = 1.0 - boosted[record.hypothesis_id]
        share = sum(normalized[key] for key in others)
        if share > 0 and remaining >= 0:
            for key in others:
                boosted[key] = remaining * normalized[key] / share
            try:
                trial = bayes_rule(boosted, likelihood)
            except ValueError:
                continue
            winner = max(trial, key=trial.get)
            scenarios.append({
                "perturbed_hypothesis": record.hypothesis_id,
                "baseline_winner": baseline_winner,
                "perturbed_winner": winner,
                "reversal": winner != baseline_winner,
            })
    return {"baseline_winner": baseline_winner, "baseline_posterior": baseline,
            "scenarios": scenarios,
            "any_reversal": any(item["reversal"] for item in scenarios)}


def expected_information_gain(network: BayesianNetworkSpec, target: str,
                              hard: Mapping[str, str],
                              soft: Mapping[str, Mapping[str, float]] | None = None,
                              candidates: Sequence[str] = (),
                              acquisition_costs: Mapping[str, float] | None = None,
                              ) -> list[dict[str, Any]]:
    """Rank candidate observations by IG(E) = H(H) - E_e[H(H | e)] relative to cost."""
    domains = network_domains(network)
    acquisition_costs = acquisition_costs or {}
    current = run_posterior_inference(network, target, hard, soft)
    current_posterior = current["posterior"]
    current_entropy = shannon_entropy(current_posterior)
    rankings: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate not in domains:
            raise ValueError(f"observation candidate {candidate} is not a network variable")
        if candidate in hard or candidate == target:
            continue
        predictive = run_posterior_inference(network, candidate, hard, soft)["posterior"]
        expected_entropy = 0.0
        for state, probability in predictive.items():
            if probability <= 0.0:
                continue
            posterior_given = run_posterior_inference(network, target, {**hard, candidate: state}, soft)
            expected_entropy += probability * shannon_entropy(posterior_given["posterior"])
        information_gain = max(0.0, current_entropy - expected_entropy)
        cost = max(1e-9, float(acquisition_costs.get(candidate, 1.0)))
        rankings.append({
            "variable": candidate,
            "expected_information_gain_bits": information_gain,
            "acquisition_cost": cost,
            "value_per_cost": information_gain / cost,
            "predictive_distribution": predictive,
        })
    return sorted(rankings, key=lambda row: (-row["value_per_cost"], row["variable"]))


# ---------------------------------------------------------------------------
# Evidence normalization, classification and dependence (steps 4-8)
# ---------------------------------------------------------------------------


def summarize_evidence_intake(evidence: Sequence[EvidenceRecord],
                              variables: Mapping[str, VariableSpec],
                              ) -> dict[str, Any]:
    """Classify inputs and record reliability separately from measurement uncertainty."""
    classifications: dict[str, int] = {}
    for record in evidence:
        classifications[record.classification] = classifications.get(record.classification, 0) + 1
    issues: list[str] = []
    for record in evidence:
        if record.variable not in variables:
            issues.append(f"evidence {record.evidence_id} references unknown variable "
                          f"{record.variable}")
        elif record.hard_value is not None and record.hard_value not in variables[record.variable].states:
            issues.append(f"evidence {record.evidence_id} binds state {record.hard_value} outside "
                          f"the domain of {record.variable}")
        if record.source is None:
            issues.append(f"evidence {record.evidence_id} has no source provenance")
    return {
        "evidence_count": len(evidence),
        "classifications": dict(sorted(classifications.items())),
        "missing_evidence_count": sum(1 for record in evidence if record.missing_value_code),
        "separation_report": {
            "source_reliability": [record.source.reliability for record in evidence if record.source],
            "measurement_uncertainty": [record.measurement_uncertainty for record in evidence
                                        if record.measurement_uncertainty is not None],
            "note": "reliability, measurement uncertainty and model confidence are separate dimensions",
        },
        "issues": issues,
        "valid": not issues,
    }


def detect_dependent_evidence(evidence: Sequence[EvidenceRecord],
                              dependence_threshold: float = 0.5) -> dict[str, Any]:
    """Detect duplicate or correlated sources and count evidence groups once (step 8).

    Records that share a source, or whose sources declare dependence on each other, are
    grouped. Only the highest-reliability record of each group is counted as independent
    evidence; the others are retained but marked as dependent duplicates.
    """
    by_id = {record.evidence_id: record for record in evidence}
    adjacency: dict[str, set[str]] = {record.evidence_id: set() for record in evidence}
    for first in evidence:
        for second in evidence:
            if first.evidence_id >= second.evidence_id:
                continue
            same_source = (first.source is not None and second.source is not None
                           and first.source.source_id == second.source.source_id)
            declared = (first.source is not None and second.source is not None and (
                second.source.source_id in first.source.depends_on
                or first.source.source_id in second.source.depends_on))
            correlated = (first.source is not None and second.source is not None
                          and first.source.dependence_strength >= dependence_threshold
                          and first.variable == second.variable)
            identical_content = (first.variable == second.variable
                                 and first.hard_value is not None
                                 and first.hard_value == second.hard_value)
            if same_source or declared or (correlated and identical_content):
                adjacency[first.evidence_id].add(second.evidence_id)
                adjacency[second.evidence_id].add(first.evidence_id)

    groups: list[dict[str, Any]] = []
    visited: set[str] = set()
    for record in evidence:
        if record.evidence_id in visited:
            continue
        stack = [record.evidence_id]
        component: list[str] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(adjacency[current] - visited)
        members = [by_id[evidence_id] for evidence_id in component]
        representative = max(members, key=lambda item: (
            item.source.reliability if item.source else 0.0, item.evidence_id))
        groups.append({
            "group_id": stable_id("p11-evidence-group-", *sorted(component)),
            "members": sorted(component),
            "representative": representative.evidence_id,
            "dependent_duplicates": sorted(set(component) - {representative.evidence_id}),
            "reliability": representative.source.reliability if representative.source else None,
        })
    return {
        "groups": groups,
        "independent_group_count": len(groups),
        "dependent_duplicate_count": len(evidence) - len(groups),
        "source_dependence_declared": any(
            record.source is not None and record.source.depends_on for record in evidence),
    }


def evidence_conflict_analysis(evidence: Sequence[EvidenceRecord],
                               variables: Mapping[str, VariableSpec],
                               ) -> dict[str, Any]:
    """Preserve support, opposition and conflict for each variable (step 37)."""
    per_variable: dict[str, dict[str, Any]] = {}
    for variable_name in {record.variable for record in evidence}:
        domain = list(variables[variable_name].states) if variable_name in variables else None
        members = [record for record in evidence if record.variable == variable_name]
        hard_values = sorted({record.hard_value for record in members if record.hard_value})
        direct_conflict = len(hard_values) > 1
        support: dict[str, list[str]] = {}
        for record in members:
            if record.hard_value is not None:
                support.setdefault(record.hard_value, []).append(record.evidence_id)
            elif record.likelihood is not None:
                best = max(record.likelihood, key=lambda state: record.likelihood[state])
                support.setdefault(best, []).append(record.evidence_id)
        report: dict[str, Any] = {
            "evidence_ids": [record.evidence_id for record in members],
            "hard_values": hard_values,
            "direct_hard_conflict": direct_conflict,
            "support": support,
            "missing_evidence_ids": [record.evidence_id for record in members
                                     if record.missing_value_code],
        }
        if domain:
            combined: Mapping[frozenset[str], float] = {frozenset(domain): 1.0}
            conflict_total = 0.0
            combinations = 0
            for record in members:
                if record.hard_value is None or record.source is None:
                    continue
                mass = evidence_to_mass(domain, record.hard_value, record.source.reliability)
                result: DempsterShaferResult = combine_mass_functions(combined, mass)
                conflict_total = result.conflict_mass
                combined = result.combined if result.combined else combined
                combinations += 1
            report["dempster_shafer"] = {
                "combination_rounds": combinations,
                "conflict_mass": conflict_total,
                "strongly_conflicted": conflict_total >= 0.5,
            }
            if combinations:
                belief = {state: 0.0 for state in domain}
                for subset, mass in combined.items():
                    if len(subset) == 1:
                        belief[next(iter(subset))] = mass
                report["dempster_shafer"]["state_belief"] = belief
                report["dempster_shafer"]["state_plausibility"] = {
                    state: sum(mass for subset, mass in combined.items() if state in subset)
                    for state in domain}
        per_variable[variable_name] = report
    return {
        "variables": per_variable,
        "any_hard_conflict": any(report["direct_hard_conflict"]
                                 for report in per_variable.values()),
        "conflicted_variables": sorted(name for name, report in per_variable.items()
                                       if report["direct_hard_conflict"]),
    }


# ---------------------------------------------------------------------------
# Probabilistic-programming adapter and model validation (steps 21-25)
# ---------------------------------------------------------------------------


ALLOWED_DISTRIBUTION_FAMILIES = {
    "categorical", "bernoulli", "binomial", "poisson", "beta", "dirichlet",
    "normal", "gamma", "exponential", "lognormal",
}
FORBIDDEN_KEY_TOKENS = ("__", "import", "exec", "eval", "compile", "open", "system")


def validate_model_schema(model_description: Mapping[str, Any],
                          max_depth: int = 8, max_nodes: int = 2000) -> dict[str, Any]:
    """Validate a model schema structurally; never execute model content (step 22)."""
    errors: list[str] = []

    def walk(value: Any, depth: int, path: str) -> int:
        if depth > max_depth:
            errors.append(f"{path}: nesting exceeds the maximum depth {max_depth}")
            return 0
        if isinstance(value, Mapping):
            count = 1
            for key in value:
                text_key = str(key)
                if any(token in text_key.casefold() for token in FORBIDDEN_KEY_TOKENS):
                    errors.append(f"{path}: forbidden key {text_key!r}")
                count += walk(value[key], depth + 1, f"{path}.{text_key}")
            return count
        if isinstance(value, (list, tuple)):
            count = 1
            for index, item in enumerate(value):
                count += walk(item, depth + 1, f"{path}[{index}]")
            return count
        if isinstance(value, (str, int, float, bool)) or value is None:
            return 1
        errors.append(f"{path}: type {type(value).__name__} is not allowed in a model schema")
        return 0

    nodes = walk(model_description, 0, "model")
    if nodes > max_nodes:
        errors.append(f"model schema exceeds the node budget ({nodes} > {max_nodes})")
    if isinstance(model_description, Mapping):
        for scope in ("variables", "parameters", "observed"):
            block = model_description.get(scope)
            if isinstance(block, Mapping):
                for name, spec in block.items():
                    if isinstance(spec, Mapping) and "family" in spec:
                        family = str(spec["family"]).casefold()
                        if family not in ALLOWED_DISTRIBUTION_FAMILIES:
                            errors.append(
                                f"model.{scope}.{name}: distribution family {family!r} "
                                "is not on the allowed list")
    return {"valid": not errors, "errors": errors, "node_count": nodes,
            "execution_attempted": False}


def probabilistic_programming_status() -> dict[str, Any]:
    """Probe optional probabilistic-programming adapters without executing any model."""
    available: dict[str, bool] = {}
    for module_name in ("pymc", "pyro", "numpyro", "stan"):
        try:  # pragma: no cover - environment-dependent import probing
            __import__(module_name)
            available[module_name] = True
        except Exception:
            available[module_name] = False
    any_available = any(available.values())
    return {
        "adapters": available,
        "any_available": any_available,
        "routing_note": "complex generative models are routed to the adapter only when "
                        "simpler models are inadequate and the adapter is installed",
        "execution_status": "not_attempted",
    }


def prior_predictive_check(network: BayesianNetworkSpec, seed: int = 11,
                           samples: int = 200) -> dict[str, Any]:
    """Compare observed prior marginals with forward-sampled frequencies (step 23)."""
    domains = network_domains(network)
    parents = network_parents(network)
    order = topological_order([variable.name for variable in network.variables], parents,
                              network.dynamic_unrolled)
    cpts = {cpt.variable: cpt for cpt in network.cpts}
    rng = seeded_random(seed)
    counts = {name: {state: 0 for state in domains[name]} for name in domains}
    for _ in range(samples):
        assignment: dict[str, str] = {}
        for name in order:
            row = cpts[name].table[cpt_row_key([assignment[parent] for parent in parents[name]])]
            assignment[name] = sample_from_distribution(row, rng)
        for name in domains:
            counts[name][assignment[name]] += 1
    frequencies = {name: {state: value / samples for state, value in row.items()}
                   for name, row in counts.items()}
    exact_marginals = {}
    for name in domains:
        try:
            exact_marginals[name] = run_posterior_inference(network, name, {}, {})["posterior"]
        except (ValueError, ZeroDivisionError):
            exact_marginals[name] = None
    discrepancies = {}
    for name in domains:
        if exact_marginals[name] is None:
            continue
        discrepancies[name] = {
            state: abs(frequencies[name][state] - exact_marginals[name][state])
            for state in domains[name]}
    return {
        "samples": samples, "seed": seed,
        "sampled_frequencies": frequencies,
        "exact_marginals": exact_marginals,
        "max_discrepancy": max((value for row in discrepancies.values() for value in row.values()),
                               default=0.0),
        "passed": max((value for row in discrepancies.values() for value in row.values()),
                      default=0.0) < 0.25,
    }


def posterior_predictive_check(network: BayesianNetworkSpec,
                               hard: Mapping[str, str],
                               soft: Mapping[str, Mapping[str, float]] | None = None,
                               ) -> dict[str, Any]:
    """Leave-one-out posterior predictive probabilities for every bound evidence variable."""
    soft = soft or {}
    reports: dict[str, Any] = {}
    for variable, state in hard.items():
        reduced_hard = {key: value for key, value in hard.items() if key != variable}
        reduced_soft = {key: value for key, value in soft.items() if key != variable}
        try:
            predictive = run_posterior_inference(network, variable, reduced_hard, reduced_soft)
            reports[variable] = {
                "observed_state": state,
                "posterior_predictive_probability": predictive["posterior"][state],
                "surprising": predictive["posterior"][state] < 0.05,
            }
        except (ValueError, ZeroDivisionError) as exc:
            reports[variable] = {"observed_state": state, "error": str(exc)}
    surprising = [name for name, report in reports.items() if report.get("surprising")]
    return {"variables": reports, "surprising_variables": surprising,
            "passed": not surprising}


# ---------------------------------------------------------------------------
# Causal structure analysis (steps 31-34)
# ---------------------------------------------------------------------------


def causal_adjacency(dag: CausalDAGSpec) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Return children and parent maps for a causal DAG."""
    children: dict[str, set[str]] = {name: set() for name in dag.variables}
    parents: dict[str, set[str]] = {name: set() for name in dag.variables}
    for edge in dag.edges:
        children[edge.source].add(edge.target)
        parents[edge.target].add(edge.source)
    return children, parents


def causal_ancestors(node: str, parents: Mapping[str, set[str]]) -> set[str]:
    """Return every ancestor of a node in a parent map."""
    seen: set[str] = set()
    stack = list(parents.get(node, ()))
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(parents.get(current, ()))
    return seen


def causal_descendants(node: str, children: Mapping[str, set[str]]) -> set[str]:
    """Return every descendant of a node in a children map."""
    seen: set[str] = set()
    stack = list(children.get(node, ()))
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(children.get(current, ()))
    return seen


def classify_causal_roles(dag: CausalDAGSpec, exposure: str, outcome: str,
                          observed: Sequence[str] | None = None) -> dict[str, Any]:
    """Identify confounders, mediators and colliders (step 32)."""
    children, parents = causal_adjacency(dag)
    all_names = set(dag.variables)
    observed_set = set(observed) if observed is not None else all_names
    exposure_ancestors = causal_ancestors(exposure, parents)
    outcome_ancestors = causal_ancestors(outcome, parents)
    confounders = sorted(exposure_ancestors & outcome_ancestors)
    mediators = sorted(causal_descendants(exposure, children) & causal_ancestors(outcome, parents)
                       - {exposure, outcome})
    colliders = sorted((causal_descendants(exposure, children) & causal_descendants(outcome, children))
                       - {exposure, outcome})
    unobserved = sorted((set(confounders) | set(mediators)) - observed_set)
    warnings = []
    if any(node not in observed_set for node in confounders):
        warnings.append("unobserved confounders: causal effects are not identifiable by "
                        "adjustment alone; report the limitation instead of an effect")
    return {
        "exposure": exposure, "outcome": outcome,
        "confounders": confounders, "mediators": mediators, "colliders": colliders,
        "unobserved_required_variables": unobserved,
        "warnings": warnings,
        "edge_confidence_classes": {
            f"{edge.source}->{edge.target}": edge.confidence_class for edge in dag.edges},
    }


def d_separated(children: Mapping[str, set[str]], parents: Mapping[str, set[str]],
                x: str, y: str, z: Sequence[str]) -> bool:
    """Test whether x and y are d-separated given z (reachability algorithm)."""
    z_set = set(z)
    ancestors_of_z: set[str] = set(z_set)
    for node in z_set:
        ancestors_of_z |= causal_ancestors(node, parents)
    agenda: list[tuple[str, str]] = [(x, "up")]
    visited: set[tuple[str, str]] = set()
    while agenda:
        node, direction = agenda.pop()
        if (node, direction) in visited:
            continue
        visited.add((node, direction))
        if node == y and node not in z_set:
            return False
        if direction == "up" and node not in z_set:
            agenda.extend((parent, "up") for parent in parents.get(node, ()))
            agenda.extend((child, "down") for child in children.get(node, ()))
        elif direction == "down":
            if node not in z_set:
                agenda.extend((child, "down") for child in children.get(node, ()))
            if node in ancestors_of_z:
                agenda.extend((parent, "up") for parent in parents.get(node, ()))
    return True


def satisfies_backdoor(children: Mapping[str, set[str]], parents: Mapping[str, set[str]],
                       exposure: str, outcome: str, adjustment: Sequence[str]) -> bool:
    """Check the backdoor criterion for one adjustment set."""
    adjustment_set = set(adjustment)
    if adjustment_set & causal_descendants(exposure, children):
        return False
    reduced_children = {name: set(targets) for name, targets in children.items()}
    for name in list(reduced_children):
        if name == exposure:
            reduced_children[name] = set()
    reduced_parents: dict[str, set[str]] = {name: set() for name in children}
    for name, targets in reduced_children.items():
        for target in targets:
            reduced_parents[target].add(name)
    return d_separated(reduced_children, reduced_parents, exposure, outcome, list(adjustment_set))


def find_backdoor_adjustment(dag: CausalDAGSpec, exposure: str, outcome: str,
                             observed: Sequence[str] | None = None,
                             max_search: int = 1024) -> dict[str, Any]:
    """Search for a minimal backdoor adjustment set among observed variables."""
    children, parents = causal_adjacency(dag)
    all_names = set(dag.variables)
    observed_set = set(observed) if observed is not None else all_names
    forbidden = causal_descendants(exposure, children) | {exposure, outcome}
    candidates = sorted(observed_set - forbidden)
    searched = 0
    for size in range(len(candidates) + 1):
        for combination in itertools.combinations(candidates, size):
            searched += 1
            if searched > max_search:
                return {"adjustment_set": None, "identifiable": False,
                        "reason": "adjustment search exceeded the budget",
                        "searched": searched}
            if satisfies_backdoor(children, parents, exposure, outcome, combination):
                return {"adjustment_set": list(combination), "identifiable": True,
                        "reason": "backdoor_criterion_satisfied", "searched": searched}
    unobserved_available = sorted(all_names - observed_set - forbidden)
    return {"adjustment_set": None, "identifiable": False,
            "reason": "no observed backdoor adjustment set exists"
                      + (f"; unobserved candidates: {unobserved_available}" if unobserved_available
                         else ""),
            "searched": searched}


def backdoor_adjustment_query(network: BayesianNetworkSpec, exposure: str, exposure_state: str,
                              outcome: str, adjustment_set: Sequence[str]) -> dict[str, Any]:
    """Compute P(Y | do(X=x)) = sum_z P(Y | X=x, z) P(z) with the adjustment formula."""
    domains = network_domains(network)
    z_variables = list(adjustment_set)
    if any(name not in domains for name in [exposure, outcome, *z_variables]):
        raise ValueError("adjustment query variables must all exist in the network")
    if exposure_state not in domains[exposure]:
        raise ValueError(f"exposure state {exposure_state} is outside the domain")
    combinations = list(itertools.product(*(domains[name] for name in z_variables))) or [()]
    posterior: dict[str, float] = {state: 0.0 for state in domains[outcome]}
    for combo in combinations:
        z_assignment = dict(zip(z_variables, combo))
        conditional = run_posterior_inference(
            network, outcome, {**z_assignment, exposure: exposure_state}, {})
        z_probability = conditional_joint(network, z_assignment) if z_assignment else 1.0
        for state in posterior:
            posterior[state] += conditional["posterior"][state] * z_probability
    total = sum(posterior.values())
    if total <= 0:
        raise ValueError("backdoor adjustment produced zero mass")
    return {"posterior": {state: value / total for state, value in posterior.items()},
            "method": "backdoor_adjustment", "adjustment_set": z_variables,
            "formula": "P(Y|do(X=x)) = sum_z P(Y|X=x,z) P(z)"}


def conditional_joint(network: BayesianNetworkSpec, assignment: Mapping[str, str]) -> float:
    """Compute the exact joint probability of a partial assignment by enumeration."""
    domains = network_domains(network)
    names = [variable.name for variable in network.variables]
    total = 0.0
    for combo in itertools.product(*(domains[name] for name in names)):
        full = dict(zip(names, combo))
        if all(full[key] == value for key, value in assignment.items()):
            total += joint_probability(network, full)
    return total


def mutilate_network(network: BayesianNetworkSpec,
                     interventions: Mapping[str, str]) -> BayesianNetworkSpec:
    """Replace intervened mechanisms with point mass (graph surgery for do(...))."""
    cpts = []
    for cpt in network.cpts:
        if cpt.variable in interventions:
            state = interventions[cpt.variable]
            cpts.append(CPTSpec(cpt.variable, (), {"": {state: 1.0}}, "do_intervention"))
        else:
            cpts.append(cpt)
    return BayesianNetworkSpec(
        network.network_id + "-mutilated", network.variables, tuple(cpts),
        network.assumptions + (f"do({', '.join(f'{k}={v}' for k, v in sorted(interventions.items()))})",),
        network.dynamic_unrolled, network.version + 1, "intervention_mutilation")


def intervention_query(network: BayesianNetworkSpec, interventions: Mapping[str, str],
                       outcome: str) -> dict[str, Any]:
    """Compute P(Y | do(X)) on the mutilated network (step 34)."""
    if not interventions:
        raise ValueError("intervention queries require at least one do(...) assignment")
    mutilated = mutilate_network(network, interventions)
    result = run_posterior_inference(mutilated, outcome, {}, {})
    return {
        "posterior": result["posterior"],
        "method": "mutilated_graph_inference",
        "interventions": dict(interventions),
        "note": "observational conditioning P(Y|X) is not interchangeable with P(Y|do(X))",
        "mutilated_network_digest": network_fingerprint(mutilated),
    }


# ---------------------------------------------------------------------------
# Structural causal models and counterfactuals (steps 33, 35, 36)
# ---------------------------------------------------------------------------


def scm_domains(scm: SCMSpec) -> dict[str, tuple[str, ...]]:
    """Derive the domain of every endogenous variable from its structural equation."""
    domains: dict[str, tuple[str, ...]] = {}
    for equation in scm.equations:
        state_sets = {tuple(sorted(row)) for row in equation.mechanism.values()}
        if len(state_sets) != 1:
            raise ValueError(
                f"structural equation for {equation.variable} declares inconsistent states")
        domains[equation.variable] = state_sets.pop()
    for name, distribution in scm.exogenous_distributions.items():
        domains[name] = tuple(sorted(distribution))
    return domains


def validate_scm(scm: SCMSpec) -> dict[str, Any]:
    """Validate structural equations, exogenous coverage and acyclicity (step 33)."""
    errors: list[str] = []
    domains = {}
    try:
        domains = scm_domains(scm)
    except ValueError as exc:
        errors.append(str(exc))
    equation_map = {equation.variable: equation for equation in scm.equations}
    missing_equations = sorted(set(scm.variables) - set(equation_map))
    if missing_equations:
        errors.append(f"variables without structural equations: {missing_equations}")
    for equation in scm.equations:
        for parent in equation.parents:
            if parent not in domains:
                errors.append(f"equation for {equation.variable} references unknown parent {parent}")
        if equation.exogenous is not None and equation.exogenous not in scm.exogenous_distributions:
            errors.append(f"equation for {equation.variable} references exogenous "
                          f"{equation.exogenous} without a distribution")
        if equation.parents:
            parent_domains = [domains[parent] for parent in equation.parents]
            expected = {cpt_row_key(combo) for combo in itertools.product(*parent_domains)}
            if set(equation.mechanism) != expected:
                errors.append(
                    f"equation for {equation.variable} must cover every parent combination")
                continue
        elif set(equation.mechanism) != {""}:
            errors.append(f"root equation for {equation.variable} must have exactly one row")
            continue
        for key, row in equation.mechanism.items():
            try:
                validate_distribution(f"mechanism[{equation.variable}][{key}]", row,
                                      list(domains.get(equation.variable, ())))
            except ValueError as exc:
                errors.append(str(exc))
    parents = {name: set(equation_map[name].parents) if name in equation_map else set()
               for name in scm.variables}
    order: list[str] = []
    if not errors:
        try:
            order = topological_order(list(scm.variables), parents)
        except ValueError as exc:
            errors.append(str(exc))
    return {"valid": not errors, "errors": errors, "domains": domains,
            "topological_order": order}


def scm_enumerate_exogenous(scm: SCMSpec, domains: Mapping[str, Sequence[str]]) -> list[dict[str, str]]:
    """Enumerate every complete exogenous assignment."""
    names = list(scm.exogenous)
    combos = list(itertools.product(*(domains[name] for name in names))) if names else [()]
    return [dict(zip(names, combo)) for combo in combos]


def scm_endogenous_probability(scm: SCMSpec, equation: StructuralEquation,
                               parents: Mapping[str, str], state: str) -> float:
    """Return f_i's probability for one state given parent values (exogenous included)."""
    key = cpt_row_key([parents[name] for name in equation.parents])
    row = equation.mechanism.get(key)
    if row is None:
        return 0.0
    return float(row[state])


def scm_joint_given_exogenous(scm: SCMSpec, domains: Mapping[str, Sequence[str]],
                              exogenous_assignment: Mapping[str, str],
                              ) -> dict[str, float]:
    """Return the joint distribution over endogenous assignments given one exogenous state.

    Deterministic mechanisms make this a point mass; probabilistic mechanisms keep their
    full conditional semantics, so abduction remains well defined either way.
    """
    equations = {equation.variable: equation for equation in scm.equations}
    names = list(scm.variables)
    joint: dict[str, float] = {}
    for combo in itertools.product(*(domains[name] for name in names)):
        assignment = dict(zip(names, combo))
        parents = {**assignment, **exogenous_assignment}
        probability = 1.0
        for name in names:
            key = cpt_row_key([parents[parent] for parent in equations[name].parents])
            row = equations[name].mechanism.get(key, {})
            probability *= float(row.get(assignment[name], 0.0))
            if probability == 0.0:
                break
        if probability > 0.0:
            joint[json.dumps(assignment, sort_keys=True)] = probability
    return joint


def scm_abduction(scm: SCMSpec, domains: Mapping[str, Sequence[str]],
                  evidence: Mapping[str, str]) -> dict[str, Any]:
    """Infer the posterior over exogenous states given observed evidence (abduction)."""
    exogenous_assignments = scm_enumerate_exogenous(scm, domains)
    endogenous_names = list(scm.variables)
    weights: dict[str, float] = {}
    prior_probability: dict[str, float] = {}
    for assignment in exogenous_assignments:
        prior = 1.0
        for name in exogenous_assignment_names(scm):
            prior *= float(scm.exogenous_distributions[name][assignment[name]])
        key = json.dumps(assignment, sort_keys=True)
        prior_probability[key] = prior
        # Exogenous evidence must match directly; endogenous evidence uses the likelihood.
        exogenous_match = all(
            assignment.get(name, state) == state
            for name, state in evidence.items() if name not in endogenous_names)
        if not exogenous_match:
            weights[key] = 0.0
            continue
        joint = scm_joint_given_exogenous(scm, domains, assignment)
        likelihood = 0.0
        for joint_key, probability in joint.items():
            full = json.loads(joint_key)
            if all(full.get(name) == state for name, state in evidence.items()
                   if name in endogenous_names):
                likelihood += probability
        weights[key] = prior * likelihood
    total = sum(weights.values())
    if total <= 0.0:
        raise ZeroDivisionError(
            "the observed evidence has zero probability under the structural model")
    posterior = {key: value / total for key, value in weights.items()}
    return {"exogenous_posterior": posterior, "evidence_probability": total,
            "exogenous_prior": prior_probability}


def exogenous_assignment_names(scm: SCMSpec) -> list[str]:
    """Return the exogenous variable names in canonical enumeration order."""
    return list(scm.exogenous)


def scm_prediction(scm: SCMSpec, domains: Mapping[str, Sequence[str]],
                   exogenous_assignment: Mapping[str, str],
                   interventions: Mapping[str, str], target: str) -> dict[str, float]:
    """Predict the target distribution under modified mechanisms (action + prediction)."""
    equations = {equation.variable: equation for equation in scm.equations}
    names = list(scm.variables)
    posterior: dict[str, float] = {state: 0.0 for state in domains[target]}
    for combo in itertools.product(*(domains[name] for name in names)):
        assignment = dict(zip(names, combo))
        if any(assignment.get(name, None) != state for name, state in interventions.items()):
            continue
        parents = {**assignment, **exogenous_assignment}
        probability = 1.0
        for name in names:
            if name in interventions:
                continue
            key = cpt_row_key([parents[parent] for parent in equations[name].parents])
            row = equations[name].mechanism.get(key, {})
            probability *= float(row.get(assignment[name], 0.0))
            if probability == 0.0:
                break
        if probability > 0.0:
            posterior[assignment[target]] += probability
    total = sum(posterior.values())
    if total <= 0.0:
        raise ValueError("counterfactual prediction produced zero probability mass")
    return {state: value / total for state, value in posterior.items()}


def counterfactual_query(scm: SCMSpec, evidence: Mapping[str, str],
                         interventions: Mapping[str, str], target: str,
                         ) -> dict[str, Any]:
    """Run abduction -> action -> prediction and compare with the factual outcome (steps 35-36)."""
    validation = validate_scm(scm)
    if not validation["valid"]:
        raise ValueError(f"SCM validation failed: {validation['errors']}")
    domains = validation["domains"]
    abduction = scm_abduction(scm, domains, evidence)
    counterfactual: dict[str, float] = {state: 0.0 for state in domains[target]}
    factual: dict[str, float] = {state: 0.0 for state in domains[target]}
    for key, weight in abduction["exogenous_posterior"].items():
        if weight <= 0.0:
            continue
        assignment = json.loads(key)
        for target_distribution, intervened in ((counterfactual, interventions), (factual, {})):
            prediction = scm_prediction(scm, domains, assignment, intervened, target)
            for state, probability in prediction.items():
                target_distribution[state] += weight * probability
    differences = {state: counterfactual[state] - factual.get(state, 0.0)
                   for state in counterfactual}
    return {
        "abduction": {"exogenous_posterior": abduction["exogenous_posterior"]},
        "action": dict(interventions),
        "prediction": dict(counterfactual),
        "factual_prediction": dict(factual),
        "differences": differences,
        "status": "hypothetical",
        "notes": [
            "the counterfactual result is conditional on the structural model and inferred "
            "exogenous state, not a direct observation",
            "uncertainty intervals follow from the exogenous posterior, not from new data",
        ],
        "assumptions": list(scm.assumptions),
    }


__all__ = [
    "Factor",
    "assignment_space",
    "backdoor_adjustment_query",
    "causal_adjacency",
    "causal_ancestors",
    "causal_descendants",
    "classify_causal_roles",
    "compare_hypotheses",
    "conditional_joint",
    "content_digest",
    "counterfactual_query",
    "cpt_row_key",
    "d_separated",
    "detect_dependent_evidence",
    "enumerate_posterior",
    "evidence_conflict_analysis",
    "expected_information_gain",
    "factor_from_cpt",
    "factor_from_likelihood",
    "find_backdoor_adjustment",
    "hypothesis_prior_sensitivity",
    "intervention_query",
    "joint_probability",
    "likelihood_weighting",
    "mutilate_network",
    "network_domains",
    "network_fingerprint",
    "network_parents",
    "posterior_predictive_check",
    "prior_predictive_check",
    "prior_sensitivity_analysis",
    "probabilistic_programming_status",
    "run_posterior_inference",
    "satisfies_backdoor",
    "scm_abduction",
    "scm_domains",
    "scm_prediction",
    "select_inference_method",
    "stable_id",
    "summarize_evidence_intake",
    "topological_order",
    "validate_model_schema",
    "validate_network",
    "validate_scm",
    "variable_elimination",
]
