# ORE — Bayesian, probabilistic and causal reasoning (Point 11)

## Purpose

Point 11 implements the source algorithm in the file
"ORE Neurocomputational — Point 11 - Bayesian and Causal Reasoning Algorithm.txt"
(original source point 7, overall integration rank 11). It gives ORE a principled way to:

1. represent uncertainty over competing hypotheses with explicit, documented priors;
2. compare hypotheses by posterior, likelihood, evidence coverage and stability;
3. model temporal hidden states with Markov/HMM filtering, smoothing and decoding;
4. distinguish association from causation, and observational conditioning
   P(Y|X) from interventions P(Y|do(X)); and
5. answer counterfactuals with abduction, action and prediction over a structural
   causal model.

The design rule throughout: probability, source reliability, measurement uncertainty,
logical truth and causal confidence are never collapsed into one score. A 0.95
posterior is not a proven fact, `associated_with` is not a causal edge, and a
counterfactual is always a hypothetical conditional on the structural model.

The pipeline is a functional and neurocomputational engineering model. The biological
surrogates (predictive coding, hypothesis competition, hippocampal evidence binding,
cortical causal maps, neuromodulatory uncertainty gates) are computational analogies,
not literal neural simulations.

## Forty source-ordered operations

| Section | Steps | Operation IDs |
|---|---|---|
| A — Intake and evidence model | 1–8 | P11-01 … P11-08 |
| B — Bayesian hypothesis evaluation | 9–20 | P11-09 … P11-20 |
| C — Probabilistic programming and model validation | 21–25 | P11-21 … P11-25 |
| D — Sequential hidden-state reasoning | 26–30 | P11-26 … P11-30 |
| E — Causal and counterfactual reasoning | 31–36 | P11-31 … P11-36 |
| F — Evidence conflict, publishing and learning | 37–40 | P11-37 … P11-40 |

Highlights per step:

| ID | Operation |
|---|---|
| P11-01 | Receive question, variables, horizon, scope, decision threshold, resource budget |
| P11-02 | Bind to a workspace revision and create an inference-run record |
| P11-03 | Resolve variables to typed USG identifiers; preserve unresolved alternatives |
| P11-05 | Normalize units, timestamps, missing-value codes, intervals, entity identity |
| P11-06 | Classify inputs as observation/intervention/prior/derived/assertion/unknown |
| P11-07 | Record source reliability separately from measurement uncertainty |
| P11-08 | Detect duplicate/correlated sources; count evidence groups once |
| P11-09 | Define variables and domains; reject ambiguous semantics |
| P11-10 | Construct/retrieve the DAG; reject cycles unless explicitly dynamic |
| P11-12 | Validate probabilities: finite, in range, normalized, complete coverage |
| P11-13 | Documented priors (a root CPT without provenance is flagged) |
| P11-14 | Bind hard vs soft (likelihood) evidence |
| P11-15 | Select inference method justified by model size |
| P11-16 | Posteriors with method, seed and convergence diagnostics |
| P11-17 | Compare hypotheses (posterior, likelihood, coverage, stability) |
| P11-18 | Sensitivity analysis; flag conclusions that reverse under perturbation |
| P11-19 | Expected information gain per acquisition cost |
| P11-20 | Calibrated distributions or an explicit uncalibrated/approximate status |
| P11-21 | Route to optional PPL adapter only when simpler models are inadequate |
| P11-22 | Validate model schema in a restricted environment; never execute model content |
| P11-23 | Prior predictive checks before, posterior predictive checks after |
| P11-24 | Convergence/ESS gates; unreliable inference is inconclusive |
| P11-25 | Content-addressed model/data/provenance registry entry |
| P11-26 | Define HMM states, transitions, emissions, initial distribution |
| P11-27 | Validate stochastic matrices and observation mapping |
| P11-28 | Filter sequentially; smooth only with complete future observations |
| P11-29 | Viterbi decoding when that is the actual goal |
| P11-30 | Distribution-shift detection; missing observations stay explicit |
| P11-31 | Causal DAG separate from USG semantic relations, edges annotated |
| P11-32 | Confounders, mediators, colliders; unobserved-variable warnings |
| P11-33 | Structural equations X_i = f_i(PA_i, U_i) with exogenous distributions |
| P11-34 | P(Y|X) vs P(Y|do(X)); backdoor adjustment only when justified |
| P11-35 | Counterfactual abduction → action → prediction |
| P11-36 | Factual/counterfactual comparison with assumptions; hypothetical status |
| P11-37 | Conflict preservation; Dempster–Shafer belief/plausibility where appropriate |
| P11-38 | Publication gates (validity, calibration, lineage, identifiability, sensitivity, budget, conflicts) |
| P11-39 | Publish typed findings via the existing workspace; request review |
| P11-40 | Log events; governed versioned learning only from verified outcomes |

## Mathematical operations

1. Bayes' rule with explicit evidence normalization:
   `P(H|E) = P(E|H)P(H)/P(E)`; multiple observations require a justified
   conditional-independence claim.
2. Bayesian network factorization:
   `P(X_1..X_n) = prod_i P(X_i | Pa(X_i))`, with exact inference by enumeration or
   variable elimination, and reproducible likelihood-weighting sampling with
   effective-sample-size and standard-error reporting.
3. Information gain:
   `IG(E) = H(H) - E_e[H(H|e)]`, ranked against acquisition cost.
4. Markov filtering:
   `P(z_t|x_1:t) ∝ P(x_t|z_t) sum_{z_{t-1}} P(z_t|z_{t-1}) P(z_{t-1}|x_{1:t-1})`,
   plus forward-backward smoothing and Viterbi decoding.
5. Structural causal models:
   `X_i = f_i(PA_i, U_i)`; interventions replace mechanisms (graph surgery) and the
   backdoor criterion gates adjustment claims; unidentifiable effects are surfaced,
   not estimated.
6. Counterfactuals: abduction (posterior over exogenous state given evidence),
   action (modify the intervened mechanism), prediction (alternative outcome).
7. Dempster–Shafer: `Bel(A) ≤ Pl(A)` with reported conflict mass K; strongly
   conflicted evidence is preserved, not averaged away.
8. Calibration: Brier score, log loss and expected calibration error on held-out
   labelled data; calibration is never claimed from training fit.

## Package layout

- `ore_bayesian_causal/contracts.py` — typed evidence, hypothesis, network, HMM,
  causal DAG, SCM and query records with provenance and epistemic labels
- `ore_bayesian_causal/math_ops.py` — distributions, Bayes' rule, HMM algorithms,
  Dempster–Shafer, calibration metrics, sampling primitives
- `ore_bayesian_causal/functional.py` — DAG validation, factor algebra, enumeration
  and variable elimination, likelihood weighting, sensitivity, information gain,
  causal structure analysis, backdoor adjustment, mutilation, SCM counterfactuals
- `ore_bayesian_causal/biology.py` — predictive-coding, hypothesis-competition,
  hippocampal-evidence, cortical-causal-map and uncertainty-gate surrogates
- `ore_bayesian_causal/registry.py` — the canonical 40-operation registry
- `ore_bayesian_causal/pipeline.py` — the 40-step processor with workspace, USG,
  logger, coordinator and publication-gate integration
- `ore_bayesian_causal/annotations.py` — SQLite annotation catalogue entry point
- `tests/test_bayesian_causal_point11.py` — acceptance tests
- `examples/point11_bayesian_causal.py` — runnable garden scenario

## ORE integration

| Component | Integration |
|---|---|
| `ore_usg` | Variable/entity identity resolution via `USGKnowledgeIndex`; unresolved names keep their alternatives |
| `ore_workspace` | Typed findings published through `FindingPublisher` against a bound revision; the processor accepts a `SharedWorkspace` or a `WorkspaceManager` |
| `ore_contracts` | Resource budget accounting; optional coordinator handoff callback follows the `ore_text_localization` pattern |
| Point 3 neuro-symbolic layer | `symbolic_constraints` forbid hypotheses without erasing their probabilistic semantics |
| Point 5 formal reasoning | `verifier_hook` validates model invariants and can reject the model |
| `ore_logger` | `register_annotations` / `record_result_to_sqlite` index the package and persist all 40 operations |
| CognitiveTaskCoordinator | Findings are offered through the `coordinator_handoff` callback; no competing scheduler or workspace is created |

## Acceptance tests

The test suite covers: analytically solvable Bayes' rule cases; factorization and
conditional independence; rejection of invalid CPTs, cycles and malformed domains;
missing evidence never counts as negative evidence; dependent duplicate evidence
never counts twice; reproducible inference with reported sampling uncertainty;
sensitivity reversal flags; information-gain ranking against a hand-computed toy
example; HMM filtering/Viterbi against reference sequences with explicit missing
observations; P(Y|X) reported separately from P(Y|do(X)); confounding and
identifiability warnings; counterfactual abduction-action-prediction against a
hand-computed SCM fixture; Dempster–Shafer conflict and bounds; content-addressed
provenance; workspace publication through the real envelope; logger correlation of
all forty operations; resource budgets, cancellation and partial outcomes;
calibration on held-out labels; and software correctness reported separately from
scientific validity.
