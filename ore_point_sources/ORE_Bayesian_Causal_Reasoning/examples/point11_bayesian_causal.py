"""Run a small Point 11 Bayesian, probabilistic and causal reasoning example.

The example asks one question about a garden, evaluates a hypothesis with Bayes'
rule over a discrete network, separates the associational quantity P(Y|X) from the
interventional quantity P(Y|do(X)), and answers a counterfactual with
abduction-action-prediction. Posterior confidence is never reported as truth and
association is never reported as causation.
"""
from __future__ import annotations

from ore_bayesian_causal import (
    BayesianCausalProcessor,
    BayesianCausalRequest,
    BayesianNetworkSpec,
    CausalDAGSpec,
    CausalEdge,
    CPTSpec,
    EvidenceBinding,
    EvidenceRecord,
    HypothesisRecord,
    QuerySpec,
    SCMSpec,
    SourceProfile,
    StructuralEquation,
    VariableSpec,
    registry_as_dicts,
)


def build_request() -> BayesianCausalRequest:
    """Create the classic rain/sprinkler/grass-wet reasoning scenario."""
    rain = VariableSpec("rain", ("yes", "no"))
    sprinkler = VariableSpec("sprinkler", ("on", "off"))
    grass = VariableSpec("grass_wet", ("yes", "no"))

    network = BayesianNetworkSpec(
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

    sensor = SourceProfile("campus-sensor", 0.9, "ground station network")
    evidence = (EvidenceRecord(
        "ev-sprinkler", "sprinkler", "observation", hard_value="on",
        source=sensor, timestamp="2026-11-24T08:00:00Z", unit="state",
        measurement_uncertainty=0.05,
    ),)
    bindings = (EvidenceBinding("sprinkler", "ev-sprinkler", hard_value="on"),)
    hypotheses = (HypothesisRecord(
        "h-rain", "it rained last night", "rain", "yes", 0.2, "weather_history"),)

    causal_dag = CausalDAGSpec("grass-causal", ("rain", "sprinkler", "grass_wet"), (
        CausalEdge("rain", "sprinkler", "observational_associational",
                   ("ev-schedule",), ("owner behaviour observed associationally",)),
        CausalEdge("rain", "grass_wet", "interventional", ("ev-rain-exp",),
                   ("randomized watering experiment",)),
        CausalEdge("sprinkler", "grass_wet", "interventional", ("ev-spr-exp",),
                   ("randomized watering experiment",)),
    ))

    scm = SCMSpec(
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

    queries = (
        QuerySpec("q-wet-obs", "observational", "grass_wet", conditions={"sprinkler": "on"}),
        QuerySpec("q-wet-do", "intervention", "grass_wet", interventions={"sprinkler": "on"}),
        QuerySpec("q-wet-cf", "counterfactual", "grass_wet",
                  evidence={"sprinkler": "on", "grass_wet": "yes"},
                  interventions={"sprinkler": "off"}),
    )

    return BayesianCausalRequest(
        question="did it rain, and would the sprinkler alone have wet the grass?",
        variables=(rain, sprinkler, grass),
        network=network,
        evidence=evidence,
        bindings=bindings,
        hypotheses=hypotheses,
        queries=queries,
        causal_dag=causal_dag,
        scm=scm,
        observation_candidates=("grass_wet", "rain"),
        acquisition_costs={"grass_wet": 0.5, "rain": 1.0},
    )


def main() -> None:
    """Run the 40-step Point 11 pipeline and print an honest summary."""
    request = build_request()
    processor = BayesianCausalProcessor("demo-runner")
    result = processor.process(request)

    print("Point 11 registry:", len(registry_as_dicts()), "operations")
    print("Run status:", result.output["run_status"])
    print("Steps completed:", sum(1 for item in result.operation_trace
                                  if item["status"] == "completed"), "/ 40")
    print()
    print("Hypothesis evaluation (Bayes over the network):")
    for row in result.hypothesis_evaluations:
        print(f"  {row['hypothesis_id']}: {row['statement']}")
        print(f"    prior={row['prior']:.2f} posterior={row['posterior']:.4f} "
              f"likelihood={row['marginal_likelihood']:.4f}")
    print()
    print("Causal findings (association is not causation):")
    for finding in result.causal_findings:
        posterior = finding.get("posterior") or finding.get(
            "mutilated_graph", {}).get("posterior", {})
        print(f"  {finding['query_id']} ({finding['kind']}): "
              f"{ {k: round(v, 3) for k, v in posterior.items()} }")
        if "identifiability" in finding:
            print(f"    identifiability: {finding['identifiability']['reason']}")
    print()
    print("Counterfactual (hypothetical, never a fact):")
    for item in result.counterfactual_results:
        print(f"  {item['query_id']}: P(wet | do(sprinkler=off), evidence) = "
              f"{ {k: round(v, 3) for k, v in item['prediction'].items()} }")
        print(f"    status={item['status']} assumptions={item['assumptions']}")
    print()
    print("Epistemic honesty:")
    print("  probability is not truth:", result.confidence_and_uncertainty["probability_is_not_truth"])
    print("  calibration status:", result.confidence_and_uncertainty["calibration_status"])
    print("  publication decision:", result.publication_report["decision"])
    print("  software correctness is separate from scientific validity:",
          result.output["software_correctness_is_separate_from_scientific_validity"])


if __name__ == "__main__":
    main()
