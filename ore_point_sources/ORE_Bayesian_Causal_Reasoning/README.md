# ORE-Neurocomputational — Point 11: Bayesian and Causal Reasoning Algorithm

This archive contains ONLY the files of the Point 11 algorithm (Bayesian,
probabilistic and causal reasoning). It is not a copy of the whole repository.

## Contents (repo-relative paths)

| Path | Purpose |
|---|---|
| `ore_bayesian_causal/` | The Point 11 package: contracts, math engine, functional operations, biological surrogates, 40-operation registry, 40-step pipeline, logger annotations |
| `tests/test_bayesian_causal_point11.py` | Acceptance tests covering every criterion of the Point 11 specification |
| `examples/point11_bayesian_causal.py` | Runnable rain/sprinkler/grass scenario (Bayes, do-intervention, counterfactual) |
| `docs/BAYESIAN_CAUSAL_POINT11.md` | Implementation documentation (architecture, 40 operations, integration map) |

## Installation into an ORE-Neurocomputational checkout

1. Extract this archive over the repository root (paths are repo-relative):

       unzip ORE_Point11_Bayesian_Causal_Reasoning.zip -d /path/to/ORE-Neurocomputational

2. Register the package in `pyproject.toml` by appending `"ore_bayesian_causal*"`
   to `[tool.setuptools.packages.find] include` (one-line change; the file is not
   included here because it is repository infrastructure, not algorithm code).

3. Run the acceptance tests from the repository root:

       python -m pytest tests/test_bayesian_causal_point11.py -q

4. Optional demo:

       python -m examples.point11_bayesian_causal

## Package modules

- `contracts.py` — typed evidence, hypothesis, network, HMM, causal-DAG, SCM and
  query records; provenance and epistemic labels are validated at construction.
- `math_ops.py` — distributions, Bayes' rule, stochastic-matrix validation, HMM
  filtering/smoothing/Viterbi, Dempster-Shafer combination, calibration metrics.
- `functional.py` — DAG validation, factor algebra, enumeration and variable
  elimination, likelihood weighting, sensitivity analysis, expected information
  gain, causal roles, d-separation, backdoor adjustment, graph surgery, and
  abduction-action-prediction counterfactuals.
- `biology.py` — predictive-coding, hypothesis-competition, hippocampal-evidence,
  cortical-causal-map and neuromodulatory-uncertainty surrogate networks.
- `registry.py` — the canonical 40-operation registry (P11-01 … P11-40).
- `pipeline.py` — the ordered 40-step processor with workspace publication gates,
  USG identity resolution, symbolic constraints, verifier hooks, resource budgets,
  cooperative cancellation and governed learning.
- `annotations.py` — entry point for the shared ORE SQLite annotation catalogue.

## Guarantees encoded in the implementation

- Probability, source reliability, measurement uncertainty, logical truth and
  causal confidence are kept as separate dimensions end to end.
- Missing evidence is never treated as negative evidence.
- Correlated or duplicate sources are counted once as effective evidence.
- Association P(Y|X) is reported separately from intervention P(Y|do(X));
  adjustment is applied only when the backdoor criterion is satisfied.
- Counterfactuals carry an explicit hypothetical status and their assumptions.
- Posteriors are reproducible; sampling reports effective sample size and
  standard errors; calibration is claimed only from held-out labelled data.
- Publication is gated on validity, convergence, lineage, identifiability,
  sensitivity, budget and unresolved conflicts.
- Learning from predictions happens only through governed, versioned updates and
  never treats unverified ORE outputs as labels.
