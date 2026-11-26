"""Functional, biological, causal and integration tests for Point 11.

Every test maps to an acceptance criterion of the Point 11 specification: Bayes'
rule toy cases, factorization, rejection of invalid models, missing-evidence
semantics, dependent-evidence handling, reproducibility, sensitivity, information
gain, HMM filtering/Viterbi, causal versus associational quantities, confounding,
counterfactuals, Dempster-Shafer bounds, provenance, workspace publication, logger
correlation, resource limits, cancellation, calibration and honest reporting.
"""
from __future__ import annotations

import pytest

from ore_bayesian_causal import (
    OPERATION_REGISTRY,
    BayesianCausalProcessor,
    BayesianCausalRequest,
    BayesianCausalResult,
    BayesianNetworkSpec,
    BiologicalBayesianCausalNetwork,
    CausalDAGSpec,
    CausalEdge,
    CPTSpec,
    EvidenceBinding,
    EvidenceRecord,
    HMMModelSpec,
    HypothesisRecord,
    QuerySpec,
    SCMSpec,
    SourceProfile,
    StructuralEquation,
    VariableSpec,
    bayes_rule,
    brier_score,
    combine_mass_functions,
    conditional_joint,
    content_digest,
    counterfactual_query,
    d_separated,
    detect_dependent_evidence,
    evidence_to_mass,
    expected_calibration_error,
    expected_information_gain,
    forward_filter,
    joint_probability,
    likelihood_weighting,
    log_loss,
    posterior_predictive_check,
    prior_predictive_check,
    prior_sensitivity_analysis,
    record_result_to_sqlite,
    registry_as_dicts,
    run_posterior_inference,
    topological_order,
    validate_model_schema,
    validate_network,
    variable_elimination,
    viterbi_decode,
)
from ore_bayesian_causal.math_ops import detect_distribution_shift

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def grass_network() -> BayesianNetworkSpec:
    """Build the classic rain/sprinkler/grass Bayesian network."""
    rain = VariableSpec("rain", ("yes", "no"))
    sprinkler = VariableSpec("sprinkler", ("on", "off"))
    grass = VariableSpec("grass_wet", ("yes", "no"))
    return BayesianNetworkSpec(
        "grass-network", (rain, sprinkler, grass),
        (
            CPTSpec("rain", (), {"": {"yes": 0.2, "no": 0.8}}, "weather_history"),
            CPTSpec("sprinkler", ("rain",), {
                "yes": {"on": 0.01, "off": 0.99},
                "no": {"on": 0.4, "off": 0.6},
            }, "owner_schedule"),
            CPTSpec("grass_wet", ("sprinkler", "rain"), {
                "on|yes": {"yes": 0.99, "no": 0.01},
                "on|no": {"yes": 0.9, "no": 0.1},
                "off|yes": {"yes": 0.8, "no": 0.2},
                "off|no": {"yes": 0.0, "no": 1.0},
            }, "groundskeeping_manual"),
        ),
        ("causal sufficiency within the garden",),
    )


def grass_causal_dag() -> CausalDAGSpec:
    """Build the causal DAG for the garden with annotated edge confidence classes."""
    return CausalDAGSpec("grass-causal", ("rain", "sprinkler", "grass_wet"), (
        CausalEdge("rain", "sprinkler", "observational_associational", ("ev-schedule",),
                   ("owner behaviour observed only associationally",)),
        CausalEdge("rain", "grass_wet", "interventional", ("ev-rain-exp",),
                   ("randomized watering experiment",)),
        CausalEdge("sprinkler", "grass_wet", "interventional", ("ev-spr-exp",),
                   ("randomized watering experiment",)),
    ))


def grass_scm() -> SCMSpec:
    """Build the discrete structural causal model for the garden."""
    return SCMSpec(
        "grass-scm", ("sprinkler", "grass_wet"), ("rain_u",),
        (
            StructuralEquation("sprinkler", (), {"": {"on": 0.3, "off": 0.7}}),
            StructuralEquation("grass_wet", ("sprinkler", "rain_u"), {
                "on|yes": {"yes": 1.0, "no": 0.0},
                "on|no": {"yes": 0.9, "no": 0.1},
                "off|yes": {"yes": 1.0, "no": 0.0},
                "off|no": {"yes": 0.0, "no": 1.0},
            }),
        ),
        {"rain_u": {"yes": 0.3, "no": 0.7}},
        ("lawn wetness has exactly the causes sprinkler and rain",),
    )


def base_request(**overrides) -> BayesianCausalRequest:
    """Create the standard garden inference request."""
    rain = VariableSpec("rain", ("yes", "no"))
    sprinkler = VariableSpec("sprinkler", ("on", "off"))
    grass = VariableSpec("grass_wet", ("yes", "no"))
    source = SourceProfile("sensor-a", 0.9, "campus_sensor_network")
    payload = dict(
        question="did it rain, and would the sprinkler make the grass wet?",
        variables=(rain, sprinkler, grass),
        network=grass_network(),
        evidence=(EvidenceRecord("ev-1", "sprinkler", "observation", hard_value="on",
                                 source=source, timestamp="t1", unit="state"),),
        bindings=(EvidenceBinding("sprinkler", "ev-1", hard_value="on"),),
        hypotheses=(HypothesisRecord("h1", "it rained", "rain", "yes", 0.2, "weather_history"),),
        queries=(
            QuerySpec("q-obs", "observational", "grass_wet", conditions={"sprinkler": "on"}),
            QuerySpec("q-do", "intervention", "grass_wet", interventions={"sprinkler": "on"}),
        ),
        causal_dag=grass_causal_dag(),
        observation_candidates=("grass_wet",),
    )
    payload.update(overrides)
    return BayesianCausalRequest(**payload)


def hmm_fixture() -> HMMModelSpec:
    """Build a two-state Markov model with distinctive emissions."""
    return HMMModelSpec(
        "switching-hmm", ("A", "B"), ("x", "y"),
        {"A": 0.5, "B": 0.5},
        {"A": {"A": 0.9, "B": 0.1}, "B": {"A": 0.2, "B": 0.8}},
        {"A": {"x": 0.9, "y": 0.1}, "B": {"x": 0.1, "y": 0.9}},
        ("first-order Markov dynamics",),
    )


# ---------------------------------------------------------------------------
# Registry and contracts
# ---------------------------------------------------------------------------


def test_registry_contains_exactly_forty_ordered_operations() -> None:
    """The Point 11 registry must define the complete ordered 40-step algorithm."""
    assert [item.operation_id for item in OPERATION_REGISTRY] == [
        f"P11-{index:02d}" for index in range(1, 41)]
    assert [item.source_step for item in OPERATION_REGISTRY] == list(range(1, 41))
    assert all(item.mathematical_form and item.biological_reference
               for item in OPERATION_REGISTRY)
    assert registry_as_dicts()[0]["operation_id"] == "P11-01"


def test_result_contract_requires_forty_step_trace() -> None:
    """A result with a broken trace must be rejected by the data contract."""
    with pytest.raises(ValueError, match="exactly 40 ordered operations"):
        BayesianCausalResult(
            runner_id="runner", operation_trace=(),
            intake_report={}, hypothesis_evaluations=(), posterior_distributions={},
            diagnostics={}, sequential_results={}, causal_findings=(),
            counterfactual_results=(), conflict_report={}, publication_report={},
            confidence_and_uncertainty={}, biological={}, final_state={}, output={},
        )


def test_evidence_classification_and_confidence_separation() -> None:
    """Classification, reliability and measurement uncertainty are separate fields."""
    record = EvidenceRecord(
        "ev-x", "rain", "observation", hard_value="yes",
        measurement_uncertainty=0.1,
        source=SourceProfile("met-station", 0.85, "regional network"),
    )
    assert record.classification == "observation"
    assert record.source.reliability == 0.85
    assert record.measurement_uncertainty == 0.1
    with pytest.raises(ValueError, match="classification"):
        EvidenceRecord("ev-y", "rain", "hunch", hard_value="yes")
    with pytest.raises(ValueError, match="both hard and soft"):
        EvidenceRecord("ev-z", "rain", "observation", hard_value="yes",
                       likelihood={"yes": 0.5, "no": 0.5})
    with pytest.raises(ValueError, match="hard, soft, or explicitly missing"):
        EvidenceRecord("ev-w", "rain", "observation")


# ---------------------------------------------------------------------------
# Acceptance 1: Bayes' rule on analytically solvable cases
# ---------------------------------------------------------------------------


def test_bayes_rule_matches_analytic_medical_test_case() -> None:
    """P(D|+) for a 1% base rate, 90% sensitivity and 80% specificity is 0.043478."""
    posterior = bayes_rule({"disease": 0.01, "healthy": 0.99},
                           {"disease": 0.9, "healthy": 0.2})
    assert posterior["disease"] == pytest.approx(0.009 / 0.207, abs=1e-12)
    assert posterior["healthy"] == pytest.approx(0.198 / 0.207, abs=1e-12)
    assert sum(posterior.values()) == pytest.approx(1.0)


def test_bayes_rule_rejects_unjustified_conditional_independence() -> None:
    """Repeated updates require a documented independence justification."""
    from ore_bayesian_causal import bayes_update_sequential
    with pytest.raises(ValueError, match="conditional-independence"):
        bayes_update_sequential({"a": 0.5, "b": 0.5}, [{"a": 0.9, "b": 0.1}] * 2)
    posterior, history, justified = bayes_update_sequential(
        {"a": 0.5, "b": 0.5}, [{"a": 0.9, "b": 0.1}] * 2,
        conditional_independence_justified=True)
    assert len(history) == 2 and justified
    assert posterior["a"] > 0.9


# ---------------------------------------------------------------------------
# Acceptance 2: factorization and conditional independence
# ---------------------------------------------------------------------------


def test_joint_factorization_sums_to_one_and_factorizes() -> None:
    """sum_x prod_i P(x_i|Pa_i) = 1 and joint entries match the factorization."""
    network = grass_network()
    names = ["rain", "sprinkler", "grass_wet"]
    import itertools
    domains = {"rain": ("yes", "no"), "sprinkler": ("on", "off"), "grass_wet": ("yes", "no")}
    total = 0.0
    for combo in itertools.product(*(domains[name] for name in names)):
        assignment = dict(zip(names, combo))
        total += joint_probability(network, assignment)
    assert total == pytest.approx(1.0, abs=1e-12)
    # Factorization: joint = prior(rain) * P(sprinkler|rain) * P(wet|sprinkler,rain)
    value = joint_probability(network, {"rain": "yes", "sprinkler": "on", "grass_wet": "yes"})
    assert value == pytest.approx(0.2 * 0.01 * 0.99, abs=1e-15)


def test_chain_graph_conditional_independence() -> None:
    """In A -> C -> B, A and B are dependent marginally and independent given C."""
    a = VariableSpec("a", ("t", "f"))
    b = VariableSpec("b", ("t", "f"))
    c = VariableSpec("c", ("t", "f"))
    network = BayesianNetworkSpec(
        "chain", (a, c, b), (
            CPTSpec("a", (), {"": {"t": 0.4, "f": 0.6}}, "fixture"),
            CPTSpec("c", ("a",), {
                "t": {"t": 0.9, "f": 0.1}, "f": {"t": 0.2, "f": 0.8}}, "fixture"),
            CPTSpec("b", ("c",), {
                "t": {"t": 0.7, "f": 0.3}, "f": {"t": 0.1, "f": 0.9}}, "fixture"),
        ),
    )
    joint_ab = conditional_joint(network, {"a": "t", "b": "t"})
    marginal_a = conditional_joint(network, {"a": "t"})
    marginal_b = conditional_joint(network, {"b": "t"})
    assert abs(joint_ab - marginal_a * marginal_b) > 1e-4  # dependent marginally
    # Given c=t: P(a=t|c) * P(b=t|c) == P(a=t, b=t | c)
    p_a_given_c = run_posterior_inference(network, "a", {"c": "t"}, {})["posterior"]["t"]
    p_b_given_c = run_posterior_inference(network, "b", {"c": "t"}, {})["posterior"]["t"]
    joint_ab_given_c = (conditional_joint(network, {"a": "t", "b": "t", "c": "t"})
                        / conditional_joint(network, {"c": "t"}))
    assert joint_ab_given_c == pytest.approx(p_a_given_c * p_b_given_c, abs=1e-12)


# ---------------------------------------------------------------------------
# Acceptance 3: invalid models are rejected
# ---------------------------------------------------------------------------


def test_invalid_probability_tables_are_rejected() -> None:
    """Unnormalized, out-of-range or incomplete CPT rows fail validation."""
    bad = CPTSpec("x", (), {"": {"t": 0.6, "f": 0.6}}, "fixture")
    network = BayesianNetworkSpec(
        "bad", (VariableSpec("x", ("t", "f")),), (bad,))
    report = validate_network(network)
    assert not report["valid"]
    assert any("sum to 1" in error for error in report["errors"])

    bad_range = CPTSpec("x", (), {"": {"t": 1.4, "f": -0.4}}, "fixture")
    network2 = BayesianNetworkSpec("bad2", (VariableSpec("x", ("t", "f")),), (bad_range,))
    assert not validate_network(network2)["valid"]


def test_cycles_are_rejected_in_dags() -> None:
    """A cyclic parent structure raises during structural validation."""
    x = VariableSpec("x", ("t", "f"))
    y = VariableSpec("y", ("t", "f"))
    with pytest.raises(ValueError, match="cycle"):
        topological_order(("x", "y"), {"x": ("y",), "y": ("x",)})
    cyclic = BayesianNetworkSpec(
        "cyclic", (x, y), (
            CPTSpec("x", ("y",), {"t": {"t": 0.5, "f": 0.5}, "f": {"t": 0.5, "f": 0.5}}),
            CPTSpec("y", ("x",), {"t": {"t": 0.5, "f": 0.5}, "f": {"t": 0.5, "f": 0.5}}),
        ))
    assert not validate_network(cyclic)["valid"]
    with pytest.raises(ValueError, match="cycle"):
        BayesianCausalProcessor("runner").process(BayesianCausalRequest(
            question="cyclic?", network=cyclic))


def test_malformed_domains_and_missing_priors_are_rejected() -> None:
    """Single-state variables and variables without CPTs are rejected."""
    one_state = BayesianNetworkSpec(
        "one-state", (VariableSpec("x", ("t",)),),
        (CPTSpec("x", (), {"": {"t": 1.0}}, "fixture"),))
    with pytest.raises(ValueError, match="at least two states"):
        BayesianCausalProcessor("runner").process(
            BayesianCausalRequest(question="q?", network=one_state))
    missing = BayesianNetworkSpec(
        "missing-prior", (VariableSpec("x", ("t", "f")), VariableSpec("y", ("t", "f")),),
        (CPTSpec("y", ("x",), {"t": {"t": 0.5, "f": 0.5}, "f": {"t": 0.5, "f": 0.5}}),))
    report = validate_network(missing)
    assert not report["valid"]
    assert any("without CPT" in error for error in report["errors"])


# ---------------------------------------------------------------------------
# Acceptance 4: missing evidence is not negative evidence
# ---------------------------------------------------------------------------


def test_missing_evidence_is_not_negative_evidence() -> None:
    """With no evidence the posterior equals the prior; missing records never bind."""
    network = grass_network()
    posterior = run_posterior_inference(network, "rain", {}, {})["posterior"]
    assert posterior == pytest.approx({"yes": 0.2, "no": 0.8}, abs=1e-12)

    request = base_request(
        evidence=(EvidenceRecord("ev-missing", "rain", "observation",
                                 missing_value_code="sensor_timeout",
                                 source=SourceProfile("sensor-a", 0.9, "net")),),
        bindings=(), hypotheses=(HypothesisRecord("h1", "it rained", "rain", "yes", 0.2),),
    )
    result = BayesianCausalProcessor("runner-m").process(request)
    trace14 = result.operation_trace[13]["functional_output"]["evidence_bindings"]
    assert trace14["missing_evidence_is_not_negative"] is True
    assert trace14["explicitly_missing_evidence"] == ["ev-missing"]
    posterior = result.posterior_distributions["posteriors"]["rain"]
    assert posterior == pytest.approx({"yes": 0.2, "no": 0.8}, abs=1e-9)


# ---------------------------------------------------------------------------
# Acceptance 5: correlated duplicate evidence is not independent
# ---------------------------------------------------------------------------


def test_duplicate_sources_collapse_into_one_effective_group() -> None:
    """Two identical reports from one source count once, not twice."""
    source = SourceProfile("news-wire", 0.9, "wire service")
    dependent = SourceProfile("news-site", 0.7, "aggregator", depends_on=("news-wire",))
    first = EvidenceRecord("ev-1", "sprinkler", "source_assertion", hard_value="on",
                           source=source)
    second = EvidenceRecord("ev-2", "sprinkler", "source_assertion", hard_value="on",
                            source=dependent)
    report = detect_dependent_evidence((first, second))
    assert report["independent_group_count"] == 1
    assert report["dependent_duplicate_count"] == 1
    assert report["groups"][0]["representative"] == "ev-1"


def test_pipeline_deduplicates_dependent_evidence_in_posterior() -> None:
    """The posterior after duplicate dependent evidence equals the single-evidence one."""
    source = SourceProfile("news-wire", 0.9, "wire service")
    request_single = base_request(
        evidence=(EvidenceRecord("ev-1", "sprinkler", "source_assertion", hard_value="on",
                                 source=source),),
        bindings=(EvidenceBinding("sprinkler", "ev-1", hard_value="on"),),
    )
    request_double = base_request(
        evidence=(
            EvidenceRecord("ev-1", "sprinkler", "source_assertion", hard_value="on",
                           source=source),
            EvidenceRecord("ev-2", "sprinkler", "source_assertion", hard_value="on",
                           source=SourceProfile("news-site", 0.6, "aggregator",
                                                depends_on=("news-wire",))),
        ),
        bindings=(EvidenceBinding("sprinkler", "ev-1", hard_value="on"),
                  EvidenceBinding("sprinkler", "ev-2", hard_value="on")),
    )
    processor = BayesianCausalProcessor("runner-d")
    single = processor.process(request_single)
    double = processor.process(request_double)
    assert (single.posterior_distributions["posteriors"]["rain"]
            == pytest.approx(double.posterior_distributions["posteriors"]["rain"], abs=1e-12))


# ---------------------------------------------------------------------------
# Acceptance 6: reproducibility and sampling uncertainty
# ---------------------------------------------------------------------------


def test_enumeration_and_variable_elimination_agree() -> None:
    """Exact inference by enumeration and by variable elimination produce one posterior."""
    network = grass_network()
    enumeration = run_posterior_inference(network, "rain", {"sprinkler": "on"}, {})
    elimination = variable_elimination(network, "rain", {"sprinkler": "on"}, {})
    assert enumeration["posterior"] == pytest.approx(elimination["posterior"], abs=1e-9)
    assert elimination["method"] == "variable_elimination"


def test_likelihood_weighting_is_reproducible_and_reports_uncertainty() -> None:
    """Fixed seeds reproduce sampled posteriors; uncertainty is reported explicitly."""
    network = grass_network()
    first = likelihood_weighting(network, "rain", {"sprinkler": "on"}, {}, samples=800, seed=11)
    second = likelihood_weighting(network, "rain", {"sprinkler": "on"}, {}, samples=800, seed=11)
    third = likelihood_weighting(network, "rain", {"sprinkler": "on"}, {}, samples=800, seed=12)
    assert first["posterior"] == second["posterior"]
    assert first["approximate"] and "standard_errors" in first
    assert first["effective_sample_size"] > 10
    # The analytic value from enumeration must fall within the sampling error bars.
    exact = run_posterior_inference(network, "rain", {"sprinkler": "on"}, {})["posterior"]["yes"]
    error = first["standard_errors"]["yes"]
    assert abs(first["posterior"]["yes"] - exact) < max(4.0 * error, 0.05)
    assert third["seed"] == 12


# ---------------------------------------------------------------------------
# Acceptance 7: sensitivity analysis flags assumption dependence
# ---------------------------------------------------------------------------


def test_sensitivity_analysis_flags_reversal_under_perturbation() -> None:
    """Boosting a losing state's prior by 20% can flip the winning state."""
    coin = VariableSpec("coin", ("heads", "tails"))
    network = BayesianNetworkSpec(
        "coin", (coin,), (CPTSpec("coin", (), {"": {"heads": 0.55, "tails": 0.45}},
                                  " mint_specification"),))
    report = prior_sensitivity_analysis(network, "coin", {}, {}, perturbation=0.2,
                                        decision_threshold=0.75)
    assert report["baseline_winner"] == "heads"
    tails_scenario = next(item for item in report["scenarios"]
                          if item["perturbed_prior"] == "coin->tails")
    assert tails_scenario["perturbed_winner"] == "tails"
    assert tails_scenario["reversal"] is True
    assert report["any_reversal"] is True


# ---------------------------------------------------------------------------
# Acceptance 8: information-gain ranking matches the toy example
# ---------------------------------------------------------------------------


def test_information_gain_ranking_matches_toy_example() -> None:
    """For the garden, sprinkler is more informative than grass about rain at equal cost."""
    network = grass_network()
    equal_cost = expected_information_gain(network, "rain", {}, {},
                                           ("grass_wet", "sprinkler"))
    assert [row["variable"] for row in equal_cost] == ["sprinkler", "grass_wet"]
    assert equal_cost[0]["expected_information_gain_bits"] == pytest.approx(0.1136, abs=1e-3)
    assert equal_cost[1]["expected_information_gain_bits"] == pytest.approx(0.0945, abs=1e-3)
    # Acquisition cost changes the ranking: a cheap wetness check wins per unit cost.
    costed = expected_information_gain(network, "rain", {}, {},
                                       ("grass_wet", "sprinkler"),
                                       {"grass_wet": 0.5, "sprinkler": 1.0})
    assert [row["variable"] for row in costed] == ["grass_wet", "sprinkler"]


# ---------------------------------------------------------------------------
# Acceptance 9: HMM filtering, smoothing and Viterbi
# ---------------------------------------------------------------------------


def test_hmm_filtering_matches_reference_sequence() -> None:
    """Forward filtering matches the hand-computed posterior sequence."""
    hmm = hmm_fixture()
    result = forward_filter(hmm.hidden_states, hmm.transition_matrix, hmm.emission_matrix,
                            hmm.initial_distribution, ("x", "x", "y", "y"))
    assert result["filtered"][1] == pytest.approx({"A": 0.979526, "B": 0.020474}, abs=1e-5)
    assert result["final"] == pytest.approx({"A": 0.108908, "B": 0.891092}, abs=1e-5)


def test_hmm_viterbi_matches_reference_path() -> None:
    """Viterbi recovers the state path A, A, B, B for the reference sequence."""
    hmm = hmm_fixture()
    result = viterbi_decode(hmm.hidden_states, hmm.transition_matrix, hmm.emission_matrix,
                            hmm.initial_distribution, ("x", "x", "y", "y"))
    assert result["path"] == ["A", "A", "B", "B"]


def test_hmm_missing_observations_are_explicit() -> None:
    """Missing observations skip emission updates and are flagged, never guessed."""
    hmm = hmm_fixture()
    result = forward_filter(hmm.hidden_states, hmm.transition_matrix, hmm.emission_matrix,
                            hmm.initial_distribution, ("x", None, "y"))
    assert result["missing_steps"] == [2]
    # With a missing middle observation the filtered step equals the prediction step.
    prediction = {"A": 0.9 * 0.916667 + 0.2 * 0.083333,
                  "B": 0.1 * 0.916667 + 0.8 * 0.083333}
    assert result["filtered"][1]["A"] == pytest.approx(prediction["A"], abs=1e-4)


def test_distribution_shift_is_detected_and_exposed() -> None:
    """A regime change in the filtered sequence is flagged with its divergence."""
    drift = [{"A": 0.95, "B": 0.05}] * 4 + [{"A": 0.05, "B": 0.95}] * 2
    report = detect_distribution_shift(drift, threshold=0.25)
    assert report["flagged_steps"]
    assert report["flagged_steps"][0]["step"] == 5


# ---------------------------------------------------------------------------
# Acceptance 10-11: causal association vs intervention; confounding
# ---------------------------------------------------------------------------


def test_association_is_not_reported_as_intervention_effect() -> None:
    """P(Y|X) differs from P(Y|do(X)) in the confounded garden model."""
    result = BayesianCausalProcessor("runner-c").process(base_request())
    observational = next(item for item in result.causal_findings
                         if item["query_id"] == "q-obs")["posterior"]
    intervention = next(item for item in result.causal_findings
                        if item["query_id"] == "q-do")["mutilated_graph"]["posterior"]
    assert observational["yes"] == pytest.approx(0.9006, abs=1e-3)
    assert intervention["yes"] == pytest.approx(0.918, abs=1e-3)
    assert observational["yes"] != intervention["yes"]
    note = next(item for item in result.causal_findings if item["query_id"] == "q-do")["note"]
    assert "graph surgery" in note


def test_confounding_and_identifiability_limitations_are_surfaced() -> None:
    """An unobserved confounder blocks adjustment and surfaces identifiability warnings."""
    from ore_bayesian_causal import classify_causal_roles, find_backdoor_adjustment
    dag = CausalDAGSpec("latent-confounding", ("u", "x", "y"), (
        CausalEdge("u", "x", "theoretical", (), ("unmeasured common cause",)),
        CausalEdge("u", "y", "theoretical", (), ("unmeasured common cause",)),
        CausalEdge("x", "y", "interventional", ("ev-1",), ("experiment",)),
    ))
    roles = classify_causal_roles(dag, "x", "y", observed=("x", "y"))
    assert "u" in roles["confounders"]
    assert roles["unobserved_required_variables"] == ["u"]
    assert roles["warnings"]
    adjustment = find_backdoor_adjustment(dag, "x", "y", observed=("x", "y"))
    assert adjustment["identifiable"] is False
    assert "no observed backdoor adjustment set" in adjustment["reason"]

    observed_case = find_backdoor_adjustment(dag, "x", "y", observed=("u", "x", "y"))
    assert observed_case["identifiable"] is True
    assert observed_case["adjustment_set"] == ["u"]


def test_d_separation_reference_cases() -> None:
    """The collider and chain cases of d-separation are handled correctly."""
    children = {"a": {"c"}, "c": {"b"}, "b": set()}
    parents = {"a": set(), "c": {"a"}, "b": {"c"}}
    assert d_separated(children, parents, "a", "b", ["c"]) is True
    assert d_separated(children, parents, "a", "b", []) is False
    collider_children = {"a": {"c"}, "b": {"c"}, "c": set()}
    collider_parents = {"a": set(), "b": set(), "c": {"a", "b"}}
    assert d_separated(collider_children, collider_parents, "a", "b", []) is True
    assert d_separated(collider_children, collider_parents, "a", "b", ["c"]) is False


# ---------------------------------------------------------------------------
# Acceptance 12: counterfactual abduction-action-prediction
# ---------------------------------------------------------------------------


def test_counterfactual_matches_known_scm_fixture() -> None:
    """Abduction, action and prediction reproduce the hand-computed garden values."""
    result = counterfactual_query(
        grass_scm(), {"sprinkler": "on", "grass_wet": "yes"},
        {"sprinkler": "off"}, "grass_wet")
    abduction = result["abduction"]["exogenous_posterior"]
    assert abduction['{"rain_u": "yes"}'] == pytest.approx(0.09 / 0.279, abs=1e-9)
    assert abduction['{"rain_u": "no"}'] == pytest.approx(0.189 / 0.279, abs=1e-9)
    assert result["prediction"]["yes"] == pytest.approx(0.3226, abs=1e-3)
    assert result["factual_prediction"]["yes"] == pytest.approx(0.5055, abs=1e-3)
    assert result["status"] == "hypothetical"
    assert result["differences"]["yes"] == pytest.approx(
        result["prediction"]["yes"] - result["factual_prediction"]["yes"], abs=1e-9)


def test_pipeline_counterfactual_is_published_as_hypothetical() -> None:
    """The processor reports counterfactuals with hypothetical status and assumptions."""
    request = base_request(
        queries=(QuerySpec("q-cf", "counterfactual", "grass_wet",
                           evidence={"sprinkler": "on", "grass_wet": "yes"},
                           interventions={"sprinkler": "off"}),),
        scm=grass_scm(), evidence=(), bindings=(), hypotheses=())
    result = BayesianCausalProcessor("runner-cf").process(request)
    counterfactual = result.counterfactual_results[0]
    assert counterfactual["status"] == "hypothetical"
    assert counterfactual["assumptions"]
    assert result.confidence_and_uncertainty["probability_is_not_truth"] is True


# ---------------------------------------------------------------------------
# Acceptance 13: Dempster-Shafer conflict and bounds
# ---------------------------------------------------------------------------


def test_dempster_shafer_conflict_and_bounds() -> None:
    """Conflicting masses produce the documented conflict and Bel <= Pl bounds."""
    frame = ("a", "b")
    first = evidence_to_mass(frame, "a", 0.6)
    second = evidence_to_mass(frame, "b", 0.7)
    result = combine_mass_functions(first, second)
    assert result.conflict_mass == pytest.approx(0.42, abs=1e-12)
    combined_total = sum(result.combined.values())
    assert combined_total == pytest.approx(1.0, abs=1e-12)
    assert result.combined[frozenset({"a"})] == pytest.approx(0.18 / 0.58, abs=1e-12)
    assert result.combined[frozenset({"b"})] == pytest.approx(0.28 / 0.58, abs=1e-12)
    dictionary = result.to_dict()
    assert dictionary["belief"]["{a}"] <= dictionary["plausibility"]["{a}"]
    assert result.belief[frozenset({"a"})] <= result.plausibility[frozenset({"a"})] + 1e-12


def test_total_conflict_is_reported_not_hidden() -> None:
    """Completely contradictory evidence yields conflict mass one and an empty combination."""
    first = {frozenset({"a"}): 1.0}
    second = {frozenset({"b"}): 1.0}
    result = combine_mass_functions(first, second)
    assert result.conflict_mass == 1.0
    assert not result.combined


# ---------------------------------------------------------------------------
# Acceptance 14: provenance is complete and content-addressed
# ---------------------------------------------------------------------------


def test_provenance_is_complete_and_content_addressed() -> None:
    """Model and evidence digests are stable and change when content changes."""
    first = BayesianCausalProcessor("runner-p").process(base_request())
    second = BayesianCausalProcessor("runner-p").process(base_request())
    digest1 = first.diagnostics["model_registry_entry"]["model_digest"]
    digest2 = second.diagnostics["model_registry_entry"]["model_digest"]
    assert len(digest1) == 64 and all(c in "0123456789abcdef" for c in digest1)
    assert digest1 == digest2  # identical content, identical address
    assert first.diagnostics["model_registry_entry"]["content_addressed"] is True
    assert first.diagnostics["model_registry_entry"]["prior_sources"]["rain"] == "weather_history"
    assert first.diagnostics["model_registry_entry"]["data_snapshot_digest"] != content_digest({})

    changed = base_request(evidence=(EvidenceRecord(
        "ev-9", "grass_wet", "observation", hard_value="no",
        source=SourceProfile("sensor-b", 0.8, "net")),),
        bindings=(EvidenceBinding("grass_wet", "ev-9", hard_value="no"),))
    changed_result = BayesianCausalProcessor("runner-p").process(changed)
    assert (changed_result.posterior_distributions["reproducibility"]["evidence_digest"]
            != first.posterior_distributions["reproducibility"]["evidence_digest"])


# ---------------------------------------------------------------------------
# Acceptance 15: workspace publication through the existing API
# ---------------------------------------------------------------------------


def test_workspace_publication_uses_revision_safe_envelope(tmp_path) -> None:
    """Findings publish through the real WorkspaceManager and report the decision."""
    from ore_workspace import WorkspaceManager
    manager = WorkspaceManager(repository_root=tmp_path, enable_logging=False)
    try:
        processor = BayesianCausalProcessor("runner-w", workspace=manager)
        result = processor.process(base_request())
        assert result.publication_report["published"] is True
        assert result.publication_report["workspace"] in {"accepted", "merged", "duplicate",
                                                          "conflict", "deferred", "rejected"}
        assert result.intake_report["run_record"]["bound_revision"] == 0
        second = processor.process(base_request())
        assert second.intake_report["run_record"]["bound_revision"] >= 0
    finally:
        manager.close()


def test_publication_gates_block_conflicted_or_invalid_findings() -> None:
    """Hard evidence conflicts block publication and report the withheld reasons."""
    conflicted = base_request(
        evidence=(
            EvidenceRecord("ev-1", "grass_wet", "observation", hard_value="yes",
                           source=SourceProfile("sensor-a", 0.9, "net")),
            EvidenceRecord("ev-2", "grass_wet", "observation", hard_value="no",
                           source=SourceProfile("sensor-b", 0.9, "net")),
        ),
        bindings=(EvidenceBinding("grass_wet", "ev-1", hard_value="yes"),),
    )
    result = BayesianCausalProcessor("runner-g").process(conflicted)
    assert result.conflict_report["any_hard_conflict"] is True
    assert result.publication_report["decision"] == "blocked"
    assert "unresolved_conflicts" in result.publication_report["withheld_reasons"]
    assert result.publication_report["published"] is False


# ---------------------------------------------------------------------------
# Acceptance 16: logger correlation
# ---------------------------------------------------------------------------


def test_sqlite_logging_correlates_all_forty_operations(tmp_path) -> None:
    """All 40 operations, annotations and the run record persist to the ORE logger."""
    result = BayesianCausalProcessor("runner-l").process(base_request())
    database_path = tmp_path / "p11-logger.sqlite"
    report = record_result_to_sqlite(result, database_path)
    assert report["run_id"]
    summary = report["database_summary"]
    assert summary["runs"] == 1
    assert summary["events"] == 40
    assert summary["symbols"] > 0
    assert report["annotation_catalogue"]["files_indexed"] >= 7
    assert report["annotation_catalogue"]["parse_errors"] == 0
    assert report["annotation_catalogue"]["symbols_written"] > 50


def test_annotations_register_through_the_package_entry_point(tmp_path) -> None:
    """The annotation entry point indexes the package in the shared catalogue."""
    from ore_bayesian_causal import register_annotations
    report = register_annotations(tmp_path / "p11-annotations.sqlite")
    assert report["files_indexed"] >= 7
    assert report["parse_errors"] == 0
    assert report["symbols_written"] > 50


# ---------------------------------------------------------------------------
# Acceptance 17: resource limits, cancellation and partial outcomes
# ---------------------------------------------------------------------------


def wide_network() -> BayesianNetworkSpec:
    """Build a 12-binary-variable network whose joint space is 4096 states."""
    variables = tuple(VariableSpec(f"v{index}", ("t", "f")) for index in range(12))
    cpts = [CPTSpec("v0", (), {"": {"t": 0.5, "f": 0.5}}, "uniform_fixture")]
    for index in range(1, 12):
        parent = f"v{index - 1}"
        cpts.append(CPTSpec(f"v{index}", (parent,), {
            "t": {"t": 0.6, "f": 0.4}, "f": {"t": 0.3, "f": 0.7}}, "chain_fixture"))
    return BayesianNetworkSpec("wide-chain", variables, tuple(cpts))


def test_resource_budget_exhaustion_produces_partial_outcome() -> None:
    """A budget below the joint space skips inference and blocks publication."""
    request = base_request(
        network=wide_network(),
        variables=tuple(VariableSpec(f"v{index}", ("t", "f")) for index in range(12)),
        hypotheses=(HypothesisRecord("h1", "v11 is true", "v11", "t", 0.5, "uniform"),),
        evidence=(), bindings=(), queries=(), causal_dag=None,
        resource_budget=100,
    )
    result = BayesianCausalProcessor("runner-b").process(request)
    assert result.output["run_status"] == "partial_budget_exhausted"
    assert result.publication_report["gates"]["budget_completion"] is False
    assert result.publication_report["decision"] == "blocked"
    trace16 = result.operation_trace[15]
    assert trace16["status"] == "skipped_budget_exhausted"


def test_cancellation_returns_partial_outcome_with_full_trace() -> None:
    """Cooperative cancellation keeps the 40-step trace with a cancelled status."""
    request = base_request(cancel_check=lambda: True)
    result = BayesianCausalProcessor("runner-x").process(request)
    assert result.output["run_status"] == "cancelled_partial"
    assert len(result.operation_trace) == 40


# ---------------------------------------------------------------------------
# Acceptance 19: calibration on held-out labelled data
# ---------------------------------------------------------------------------


def test_calibration_metrics_measured_on_held_out_labels() -> None:
    """Calibration metrics are computed only when held-out labels are supplied."""
    labels = ((0.9, 1), (0.8, 1), (0.1, 0), (0.2, 0), (0.7, 1), (0.3, 0))
    predictions = [p for p, _ in labels]
    outcomes = [y for _, y in labels]
    assert brier_score(predictions, outcomes) == pytest.approx(0.28 / 6, abs=1e-9)
    assert log_loss(predictions, outcomes) < 0.5
    # Hand-computed ECE: |0.10|*1/6 + |0.25|*2/6 + |0.30|*1/6 + |0.15|*2/6 = 0.20.
    assert expected_calibration_error(predictions, outcomes) == pytest.approx(0.20, abs=1e-9)
    perfect = ((1.0, 1), (0.0, 0), (1.0, 1), (0.0, 0))
    assert expected_calibration_error([p for p, _ in perfect], [y for _, y in perfect]) == 0.0

    request = base_request(calibration_labels=labels)
    result = BayesianCausalProcessor("runner-cal").process(request)
    assert result.diagnostics["calibration"]["status"] == "calibrated_measured"
    assert result.diagnostics["calibration"]["held_out_labelled_data"] is True
    assert "brier_score" in result.diagnostics["calibration"]

    no_labels = BayesianCausalProcessor("runner-cal2").process(base_request())
    assert no_labels.diagnostics["calibration"]["status"] == "uncalibrated"


# ---------------------------------------------------------------------------
# Acceptance 20: software correctness is separate from scientific validity
# ---------------------------------------------------------------------------


def test_software_correctness_reported_separately_from_validity() -> None:
    """The result separates software status from scientific/predictive validity."""
    result = BayesianCausalProcessor("runner-h").process(base_request())
    assert result.output["software_correctness_is_separate_from_scientific_validity"] is True
    assert result.confidence_and_uncertainty["probability_is_not_truth"] is True
    assert result.biological["biological_fidelity"] == "computational surrogate"
    assert result.biological["literal_neuron_simulation"] is False


# ---------------------------------------------------------------------------
# Integration: USG identity resolution, symbolic constraints, verifier hook
# ---------------------------------------------------------------------------


def test_usg_identity_resolution_and_unresolved_alternatives() -> None:
    """Variables resolve to USG record IDs; unresolved names keep their alternatives."""
    from test_usg_structure_adapter import _structure_bundle

    from ore_usg import USGKnowledgeAdapter
    resource = USGKnowledgeAdapter().convert(_structure_bundle())
    rain = VariableSpec("bank", ("yes", "no"))
    unknown = VariableSpec("thrombosis", ("yes", "no"))
    request = base_request(
        variables=(rain, unknown), usg_resource=resource, evidence=(), bindings=(),
        hypotheses=(), queries=(), causal_dag=None)
    result = BayesianCausalProcessor("runner-u").process(request)
    resolution = result.intake_report["identity_resolution"]
    assert resolution["resolved"]["bank"]
    assert resolution["unresolved"]["thrombosis"] == ["no_exact_match_for_thrombosis"]
    assert resolution["alternatives_preserved"] is True


def test_symbolic_constraints_preserve_probabilistic_semantics() -> None:
    """Point 3 symbolic rules forbid hypotheses without erasing their probabilities."""
    constraints = ({"hypothesis_id": "h1", "verdict": "forbidden", "rule": "r1"},)
    result = BayesianCausalProcessor("runner-s").process(
        base_request(symbolic_constraints=constraints))
    row = result.hypothesis_evaluations[0]
    assert row["constraint_verdict"] == "forbidden"
    assert "posterior" in row  # the probabilistic semantics survive the constraint


def test_external_verifier_hook_can_reject_the_model() -> None:
    """A Point 5 formal-verification hook returning invalid blocks processing."""
    def reject(payload):
        return {"valid": False, "violations": ["prior provenance incomplete"]}
    with pytest.raises(ValueError, match="external verifier rejected the model"):
        BayesianCausalProcessor("runner-v").process(base_request(verifier_hook=reject))


# ---------------------------------------------------------------------------
# Model validation: sandboxed schemas and predictive checks
# ---------------------------------------------------------------------------


def test_model_schema_validation_rejects_untrusted_content() -> None:
    """Schemas with forbidden keys or exotic types are rejected without execution."""
    valid = {"variables": {"x": {"family": "categorical", "states": ["a", "b"]}}}
    assert validate_model_schema(valid)["valid"] is True
    dangerous = {"variables": {"x": {"family": "categorical"}},
                 "__import__": "os.system"}
    assert validate_model_schema(dangerous)["valid"] is False
    exotic = {"variables": {"x": object()}}
    assert validate_model_schema(exotic)["valid"] is False
    unknown_family = {"variables": {"x": {"family": "arbitrary_code_execution"}}}
    assert validate_model_schema(unknown_family)["valid"] is False


def test_prior_and_posterior_predictive_checks_run() -> None:
    """Prior predictive checks recover exact marginals; PPC flags surprises."""
    network = grass_network()
    prior_check = prior_predictive_check(network, seed=11, samples=400)
    assert prior_check["max_discrepancy"] < 0.25
    assert prior_check["passed"] is True
    ppc = posterior_predictive_check(network, {"sprinkler": "on"}, {})
    assert "variables" in ppc
    assert "sprinkler" in ppc["variables"]


# ---------------------------------------------------------------------------
# Biological surrogates
# ---------------------------------------------------------------------------


def test_biological_network_surrogates_update_and_snapshot() -> None:
    """The surrogate network advances with bounded activity and honest diagnostics."""
    network = BiologicalBayesianCausalNetwork(dimensions=32)
    report = network.process_step(3, ["rain", "sprinkler"], 0.2, 0.8, 0.1)
    assert report["interpretation"].startswith("functional surrogate")
    network.bind_evidence("ev-1", ["sprinkler"], {"source": "sensor"})
    competition = network.compete_hypotheses({"h1": 0.7, "h2": 0.3})
    assert competition["winner"] == "h1"
    network.observe_association(["rain"], ["wet"])
    snapshot = network.snapshot()
    assert snapshot["literal_neuron_simulation"] is False
    assert snapshot["hippocampus"]["episode_count"] == 1
    assert snapshot["causal_map"]["association_is_not_causation"] is True


# ---------------------------------------------------------------------------
# End-to-end trace integrity
# ---------------------------------------------------------------------------


def test_full_pipeline_emits_forty_ordered_operations() -> None:
    """A complete request produces the exact ordered 40-step operation trace."""
    result = BayesianCausalProcessor("runner-e2e").process(base_request())
    assert [item["operation_id"] for item in result.operation_trace] == [
        f"P11-{index:02d}" for index in range(1, 41)]
    assert all(item["status"] == "completed" for item in result.operation_trace)
    assert result.output["run_status"] == "completed"
    assert result.output["hypothesis_count"] == 1
    assert result.hypothesis_evaluations[0]["posterior"] == pytest.approx(0.0062, abs=1e-3)
    assert result.intake_report["classification"]["classifications"]["observation"] == 1
    assert result.intake_report["source_dependence"]["independent_group_count"] == 1
    # Epistemic separation is preserved end to end.
    assert result.confidence_and_uncertainty["calibration_status"] == "uncalibrated"
    assert result.confidence_and_uncertainty["uncertainty_sources"]
    serialized = result.to_dict()
    assert serialized["output"]["run_status"] == "completed"
