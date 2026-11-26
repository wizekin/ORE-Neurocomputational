# Cognitive World Model (Point 04, Ranked 13)

## Scope

`ore_cwm` implements the Cognitive World Model and Predictive Control algorithm
(source: "Most Intelligent Systems 01-24", Point 4, sections 4.1-4.8; spec:
`Algorithms_Chats/ORE Neurocomputational - Point 4 - Cognitive World Model Algorithm.txt`).
CWM predicts the consequences of ORE's own cognitive operations **before** they execute:
given a versioned workspace snapshot, an active goal, feasible candidate operations, and
resource state, it estimates each operation's likely effect on goal progress, evidence,
uncertainty, conflicts, duration, resource consumption, and risk.

CWM is a **predictive advisor, not a new controller**:

- The shared workspace (`ore_workspace`) remains authoritative for current state and revisions.
- `CognitiveTaskCoordinator` remains responsible for task decomposition and dispatch.
- `AdaptiveResourceCoordinator` remains responsible for resource-aware safeguards.
- Modules execute operations; validators and workspace integration accept artifacts.
- USG remains the semantic knowledge source; CWM has no authority to edit it.
- The existing `ore_logger` SQLite infrastructure records predictions, actions, outcomes,
  and model versions.

CWM predicts; coordinators decide and dispatch; modules execute; validators accept or
reject; the logger preserves provenance. No parallel workspace, scheduler, resource
manager, module registry, or authoritative knowledge base is created.

## Package layout

- `ore_cwm/contracts.py` — typed records: features, operations, predictions, outcomes,
  lineage, budgets, and the `MISSING` measurement sentinel.
- `ore_cwm/registry.py` — the 52 registered algorithmic steps (`CWM-01` … `CWM-52`).
- `ore_cwm/features.py` — versioned, provenance-labeled feature extraction.
- `ore_cwm/baselines.py` — Beta-Binomial, EMA, and robust (median/MAD) estimators with
  conservative per-operation-class defaults.
- `ore_cwm/predictor.py` — the statistical transition model f_theta(state, action, context).
- `ore_cwm/planner.py` — bounded-horizon MPC rollout, hard-constraint filtering, and
  documented objective scoring.
- `ore_cwm/learning.py` — prediction/outcome alignment, calibration, drift detection,
  and model versioning with rollback.
- `ore_cwm/policy.py` — confidence gates, hard constraints, and deterministic fallback.
- `ore_cwm/observability.py` — SQLite observer and prediction-outcome store following the
  repository's existing logger conventions.
- `ore_cwm/service.py` — the `CognitiveWorldModel` facade: `predict_outcomes`, `plan`,
  the receding-horizon loop, shadow evaluation, and advisory bridges.
- `ore_cwm/annotations.py` — `register_annotations()` indexing `ore_cwm` into ore_logger.
- `ore_cwm/__main__.py` — a self-contained demonstration.

## The 52 operations

The registry enumerates the algorithm's steps in source order; each identifier cites the
specification section (A. State and operation contracts through G. ORE integration and
safeguards). Representative rows:

| ID | Operation | Computation |
|---:|---|---|
| CWM-01 | Acquire a consistent snapshot | Read one workspace revision atomically |
| CWM-02 | Extract state features | Derive the versioned compact feature vector |
| CWM-15 | Estimate success probability | p = (s+alpha)/(s+f+alpha+beta) with credible interval |
| CWM-16 | Estimate resource cost | EMA mean and variance with robust dispersion |
| CWM-17 | Estimate information gain | Prior entropy minus expected posterior entropy, or labeled proxies |
| CWM-25 | Simulate in isolation | Provisional state deltas; never mutate the workspace |
| CWM-29 | Roll the model forward | predicted_state_(k+1) = f_theta(predicted_state_k, action_k) |
| CWM-31 | Enforce hard constraints | Budgets, permissions, deadlines, verification — never penalty weights |
| CWM-34 | Execute only the first action | Dispatch exactly one operation through the coordinator |
| CWM-35 | Observe and replan | Read the new revision, regenerate candidates, plan again |
| CWM-40 | Prevent feedback contamination | Label aborted/duplicated/cancelled executions; exclude them |
| CWM-51 | Run shadow mode first | Predict without changing execution decisions |

## Mathematical models

- **State transition.** Actual state evolves under an unknown T(s, a, noise); the predictor
  approximates observable consequences with f_theta(state, action, context). The predictor
  is not the world itself.
- **Success (Beta-Binomial).** p = (s + alpha) / (s + f + alpha + beta), with equal-tailed
  credible intervals from the Beta posterior. Sparse data is not over-segmented: module-level
  estimates are used only after they clear the minimum support, otherwise the class-level
  bundle (shared across modules) provides the estimate.
- **Cost (EMA).** mean_t = rho * observed_t + (1 - rho) * mean_(t-1);
  variance_t = (1 - rho)(variance_(t-1) + rho (observed_t - mean_t)^2). Dispersion also
  uses a bounded median/MAD window (MAD scaled by 1.4826).
- **Information gain.** H = -sum(p_i log p_i); expected gain = prior entropy minus the
  outcome-weighted expected posterior entropy when a defensible hypothesis distribution
  exists; otherwise clearly labeled empirical proxies (expected evidence-coverage increase).
- **MPC objective.** Sequences are scored with documented weights balancing resource cost,
  expected information/goal benefit, risk, uncertainty, delay, and terminal goal shortfall.
  A conservative uncertainty margin is subtracted from benefits and added to costs. Hard
  constraints are filtered separately and can never be traded away as penalty weights.
- **Evaluation.** Continuous outcomes: MAE, RMSE, interval coverage. Probabilities: Brier
  score, log loss, and bucketed reliability on held-out chronological segments.
- **Drift.** Rolling error inflation beyond a ratio of the validated baseline lowers the
  confidence multiplier and triggers rollback to a validated model version.

## API example

```python
from ore_cwm import CognitiveWorldModel, CWMConfig
from ore_workspace import WorkspaceManager

model = CognitiveWorldModel(config=CWMConfig(min_support=2))
with WorkspaceManager(enable_logging=False) as manager:
    workspace = manager.create_workspace(
        current_input="The bilingual student translated the sentence.",
        current_goal="understand_sentence",
    )
    candidates = ...  # from CognitiveWorldModel.candidates_from_registry(registry, task)

    # Step 43: the narrow read-only prediction API
    records = model.predict_outcomes(workspace.snapshot(), candidates)

    # Bounded MPC plan; dispatch only the first action through the coordinator
    plan = model.plan(workspace.snapshot(), candidates)

    # Shadow evaluation first, then controlled advisory influence
    report = model.run_shadow(workspace, coordinator, task,
                              candidates_provider=lambda snapshot: candidates)
```

Advisory bridges keep existing coordinators authoritative:

- `advise_task_coordinator` injects only the documented
  `task.metadata["resource_module_preferences"]` channel that
  `CognitiveTaskCoordinator` already reads; it never dispatches.
- `advise_resource_coordinator` returns `ModuleResourceEstimate` records with
  `estimate_source="cwm_prediction"` for the `AdaptiveResourceCoordinator`.
- On timeout, stale revision, invalid schema, missing measurement, model error, or low
  confidence, the planner returns a deterministic `FallbackDecision` deferring to the
  existing coordinator policies: prediction failure never becomes system failure.

## Safeguards and failure modes

| Failure mode | Safeguard |
|---|---|
| Stale snapshot | Revision validation; stale predictions discarded |
| Sparse history | Bayesian smoothing, support thresholds, conservative fallback |
| Correlation mistaken for causation | Estimates are labeled predictive; no causal claims |
| Simulated artifact treated as fact | Provisional planning objects only; workspace untouched |
| Cheap-but-useless actions rewarded | Canonical goal criteria and accepted-outcome metrics |
| Mandatory requirements traded away | Hard constraints outside the objective |
| Module drift | Feature/model versioning and calibration monitoring |
| Search consuming excess resources | Candidate, horizon, runtime budgets with cancellation |
| Feedback reinforcing bad forecasts | Learning only from validated observations; shadow mode |
| Duplicate control logic | CWM stays advisory; coordinators retain authority |

## Tests

`tests/test_cognitive_world_model.py` covers the specification's section 8 acceptance
criteria: prediction/revision/model-version binding, provisional predictions that never
become workspace findings, missing measurements that never read as zero, hard constraints
overriding predicted utility, coordinator-only dispatch, deterministic fallbacks,
alignment across retries/cancellations/duplicates, hand-calculated estimator fixtures,
held-out chronological calibration, MPC dependency/deadline/permission enforcement,
first-action-only execution with replanning, shadow-mode non-interference, drift rollback,
SQLite observability with symbol-linked annotations, a deterministic
progress-per-resource-unit benchmark against a naive ordering, and an end-to-end run
through the real `CognitiveTaskCoordinator`.

Run with `python -m pytest tests/test_cognitive_world_model.py`.
