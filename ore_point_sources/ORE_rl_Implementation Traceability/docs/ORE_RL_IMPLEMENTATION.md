# ORE Point 10 — Implementation Traceability

This document maps every step of the **"ORE Integrated Algorithm — Point 10:
Reinforcement Learning, Hierarchical Learning and Adaptive Policy
Optimization"** specification (52 steps, Phases A–J) to its implementation in
the `ore_rl` package. Status column: **implemented** (code + tests) or
**implemented (thin)** (a documented, minimal-but-faithful realization, e.g.
pure-NumPy neural networks instead of torch).

The package follows the repository's conventions: numpy/stdlib only, typed
frozen dataclasses, SQLite persistence, absolute imports, flat pytest suites
run with `python -m pytest`.

## Module map

| Module | Responsibility (spec section 5) |
| --- | --- |
| `ore_rl/contracts.py` | Typed state, action, transition, reward, decision schemas |
| `ore_rl/state_encoder.py` | Deterministic state construction and versioning |
| `ore_rl/action_catalog.py` | Action definitions, parameter validation, eligibility masks |
| `ore_rl/environment.py` | Gymnasium-compatible training/simulation environment |
| `ore_rl/modules.py` | Example cognitive modules registered through real ORE contracts |
| `ore_rl/reward.py` | Evaluator-backed reward components |
| `ore_rl/trajectory_store.py` | SQLite trajectory persistence and validation |
| `ore_rl/policies/` | Baseline, bandit, DQN/PPO, options, offline policies |
| `ore_rl/safety.py` | Constraint checks and policy authority boundaries |
| `ore_rl/evaluation.py` | Baseline comparison, offline evaluation, reports |
| `ore_rl/registry.py` | Model metadata, versioning, promotion, rollback |
| `ore_rl/controller_adapter.py` | Read-only proposal integration with the coordinators |
| `ore_rl/constrained.py` | Lagrangian constrained policy optimization (added) |
| `ore_rl/world_model.py` | Cognitive world model with simulated/observed separation (added) |

## Phase A — Establish the learning boundary and contracts

| Step | Requirement | Implementation | Status |
| --- | --- | --- | --- |
| 1 | Replaceable policy service behind `propose_actions(state, eligible-actions, budget)`; coordinator remains authoritative; returns ranked proposals, uncertainty, version | `policies/base.PolicyService` protocol; `controller_adapter.RLPolicyService` (subordinate wrapper, never receives workspace objects); `contracts.PolicyDecision` | implemented |
| 2 | Define the decision boundary | `controller_adapter.DecisionBoundary` enum + `DecisionBoundaryPolicy.should_consult` (after decomposition, after module completion, evidence insufficient, dependency ready, stop/continue); `policies/base.ProgressBoundaryPolicy` (serving-time boundary) | implemented |
| 3 | Versioned schemas for OREState, OREAction, ORETransition, OREReward, PolicyDecision; reject unknown versions | `contracts.py` — all five schemas with `RL_SCHEMA_VERSION`, `require_schema_version` raises `UnknownSchemaVersionError` (tested) | implemented |
| 4 | Hard constraints as deterministic gates, not reward penalties | `safety.HardConstraintGate` (dependency, budget, authorization, verification, USG protection); penalties remain separate in `reward.RewardWeights` | implemented |
| 5 | Policy registry with identity, version, family, schema versions, dataset fingerprint, evaluation, resources, uncertainty behavior, authority | `contracts.PolicyMetadata` + `registry.PolicyRegistry` (SQLite, audit table); training success alone never promotes | implemented |
| 6 | Deterministic baseline capturing the coordinator's decisions | `policies/base.DeterministicBaselinePolicy` (chain order + difficulty-scaled depth); used as the comparison target by `evaluation.evaluate_policy` | implemented |

## Phase B — Build state representation

| Step | Requirement | Implementation | Status |
| --- | --- | --- | --- |
| 7 | Task-context vector; one-hot / embeddings; normalization with persisted (mu, sigma) | `state_encoder.TaskContextSignal` one-hots; `NormalizationStats.fit/transform/to_dict/from_dict` persisted with models | implemented |
| 8 | Goal progress as explicit counters, not one opaque score | `state_encoder.GoalProgressSignal` (completed/remaining subgoals, unresolved requirements, completion confidence) | implemented |
| 9 | Workspace and evidence state; quantity vs quality; bound to workspace revision | `state_encoder.WorkspaceEvidenceSignal` (finding count, verified fraction, contradictions, source reliability, evidence quality vs quantity) computed from a real `SharedWorkspace` in `environment.py` | implemented |
| 10 | Capability mask with readiness, health, queue, estimated cost | `state_encoder.CapabilityStateSignal` fed from `ModuleRegistry.find_by_capability` | implemented |
| 11 | Dependency mask m_a(s) | `safety.HardConstraintGate.compute_feasible_actions` (prerequisite violations); `DependencyStateSignal` features | implemented |
| 12 | Resource state; measurements separate from forecasts; CROS is authority | `state_encoder.ResourceStateSignal` (remaining budget fractions, reservations); budgets enforced by gates and `EnvironmentBudget` | implemented |
| 13 | Uncertainty and history; missingness explicit | `state_encoder.UncertaintyHistorySignal` (sample counts, mean outcomes, similar-task success with explicit availability flag) | implemented |
| 14 | Versioned state encoder; unit-test stable outputs | `state_encoder.StateEncoder` (version fingerprint, persisted normalization); determinism tests | implemented |

## Phase C — Define actions and safe execution

| Step | Requirement | Implementation | Status |
| --- | --- | --- | --- |
| 15 | Primitive action catalogue incl. retrieve, lexical, hypotheses, causal, verification, depth, concurrency, stop | `action_catalog.ActionCatalog` + `environment._build_catalog` (five module types at five depth variants, concurrent batch, adjust depth) | implemented |
| 16 | Validated parameters, budgets; reject bad parameters | `contracts.ParameterSpec.validate`; `CatalogAction.validate_parameters` (missing, out-of-range, contradictory, unknown) | implemented |
| 17 | A_valid(s) = {a : m_a(s)=1 and g_k(s,a)<=0}; empty set → deterministic escalation | `safety.compute_feasible_actions` with per-action rejection reasons; escalation fallback actions when empty (tested) | implemented |
| 18 | STOP_SUCCESS, STOP_INSUFFICIENT_EVIDENCE, DEFER_RESOURCE_LIMIT, REQUEST_HUMAN_REVIEW | `ActionType` enum + default catalogue; verification gate distinguishes successful completion from safe termination (tested) | implemented |
| 19 | Execute through the existing coordinator | `environment._run_module` dispatches every module action through the real `CognitiveTaskCoordinator.coordinate` with validated `ContractArtifact` inputs; `controller_adapter.ModulePreferenceBridge` feeds real `resource_module_preferences` metadata | implemented |
| 20 | Capture outcomes with distinct statuses and measured costs | `contracts.ActionOutcome` (SUCCESS/PARTIAL/INVALID_OUTPUT/TIMEOUT/EXCEPTION/CANCELLED/RESOURCE_EXHAUSTED) with measured latency/cpu/memory | implemented |

## Phase D — Build reward and evaluation

| Step | Requirement | Implementation | Status |
| --- | --- | --- | --- |
| 21 | Independent result-quality evaluation; evaluator identity/version | `reward.TaskQualityEvaluator` (correctness/completeness/relevance/compliance; evaluator id + version on every score) | implemented |
| 22 | Evidence-quality evaluation; unsupported output gets no positive evidence reward | `reward.EvidenceQualityEvaluator` (traceability, relevance, reliability, support, contradictions) | implemented |
| 23 | Resource and latency costs normalized against budgets | `reward.CostModel` (latency/resource normalization against explicit budgets) | implemented |
| 24 | Composite r_t = a·ΔQ + b·ΔE − g·C − d·L − e·F with versioned weights | `reward.compute_step_reward` + `RewardWeights` (versioned config; recomputation tested) | implemented |
| 25 | Terminal reward r_H for goals, constraints, verification | `reward.compute_terminal_reward` + `TerminalOutcome` | implemented |
| 26 | Prevent reward hacking; no counts in the reward | `reward.RewardHackingGuard` (duplicated findings, unsupported claims, unnecessary ops, premature stop); counts are audit-only fields | implemented |
| 27 | Delayed/shared credit with attribution method + uncertainty | `reward.attribute_episode_rewards` (STEP, TERMINAL_ONLY, UNIFORM, LAST_ACTION) | implemented |
| 28 | Calibrate scales and weights on dev data; freeze spec | `reward.calibrate_reward_scale`, `sensitivity_analysis`, `RewardSpec.fingerprint` | implemented |

## Phase E — Record trustworthy learning trajectories

| Step | Requirement | Implementation | Status |
| --- | --- | --- | --- |
| 29 | SQLite trajectory logging: episodes, steps, rewards, decisions, evaluations | `trajectory_store.TrajectoryStore` (rl_episodes, rl_steps, rl_rewards, rl_policy_decisions, rl_evaluations, rl_state_refs) | implemented |
| 30 | Provenance and immutability; append-only/integrity checks | per-episode hash chain (`integrity_hash`, `prev_hash`), immutable state refs, provenance columns | implemented |
| 31 | Validate completeness; quarantine, never silently repair | `validate_episode` (linkage, mask, reward recomputation, measured-column cross-checks) → `rl_quarantine` table; unverified rewards rejected on insert | implemented |
| 32 | Leakage-safe splits by task family / source / session / time | `SplitRule` + `build_dataset` (group-level splits, related episodes kept together; leakage tested) | implemented |
| 33 | Replay and bandit datasets with masks, terminals, propensities | `load_replay_dataset` (s, a, r, s', done, mask), `load_bandit_dataset` (context, action, propensity, reward) | implemented |

## Phase F — Start with contextual bandits

| Step | Requirement | Implementation | Status |
| --- | --- | --- | --- |
| 34 | Contextual bandit baseline; simple empirical start | `policies/bandit.EmpiricalBandit` | implemented |
| 35 | UCB(a) = mu + c·sqrt(ln(1+t)/(1+n)) | `policies/bandit.UCBPolicy` (formula unit-tested; eligible-only; serving mode) | implemented |
| 36 | LinUCB with ridge and conditioning monitoring | `policies/bandit.LinUCBPolicy` (disjoint model, A/b updates only with verified feedback, condition warnings) | implemented |
| 37 | Thompson Sampling with seeds and posterior version | `policies/bandit.ThompsonSamplingPolicy` (seeded draws; posterior-mean serving mode) | implemented |
| 38 | Attribution and exploration safety; never explore violating actions | bandits restricted to eligible actions by construction; `ProgressBoundaryPolicy` isolates decisions with attributable rewards | implemented |

## Phase G — Extend to sequential deep RL

| Step | Requirement | Implementation | Status |
| --- | --- | --- | --- |
| 39 | DQN only when justified; compare against simpler alternatives | `policies/dqn.DQNPolicy` (masked Q over the catalogue); demo compares bandit vs DQN vs PPO vs baseline under identical suites | implemented (pure NumPy MLP) |
| 40 | Target network, replay, Huber loss, next-state mask in the target | `DQNPolicy._train_step` (y = r + γ(1−d)·max over A_valid(s'), masked; target sync; Huber gradient) | implemented |
| 41 | Double DQN and prioritized replay with importance weights | `double_dqn` config selects with the online net and evaluates with the target net (tested); `PrioritizedReplayBuffer` ((|δ|+ε)^α, IS weights) | implemented |
| 42 | PPO/A2C with clipped surrogate and GAE; masks in sampling and evaluation | `policies/ppo.PPOPolicy` (masked softmax; clipped surrogate; GAE(λ); entropy bonus; A2C mode; gradient verified against finite differences) | implemented |
| 43 | Compare algorithms under identical splits, budgets, evaluators | `evaluation.BenchmarkSuite` + `compare_policies` + `evaluate_policy` (same tasks, budgets, evaluator and reward versions for every policy) | implemented |

## Phase H — Add hierarchical cognitive options

| Step | Requirement | Implementation | Status |
| --- | --- | --- | --- |
| 44 | Option library: UNDERSTAND_INPUT, INVESTIGATE_HYPOTHESIS, RETRIEVE_SUPPORTING_EVIDENCE, VERIFY_CONCLUSION, RESOLVE_CONTRADICTION, PRODUCE_FINAL_RESULT | `policies/options.OptionLibrary` (all six, o = (I_o, pi_o, beta_o)) | implemented |
| 45 | Option contracts with prerequisites, modules, ceilings, failure paths; cannot bypass the coordinator | `OptionContract` (permitted modules, resource ceilings, failure paths); options execute through environment/coordinator validation | implemented |
| 46 | Learn option selection with deterministic internal procedures | `HierarchicalOptionPolicy` (option-level UCB) + `OptionRuntime` (deterministic procedures) | implemented |
| 47 | Loop and nontermination detection | `OptionLoopDetector` (invocation ceilings, repeated state signatures, progress deltas, wall-clock budget → escalation) | implemented |

## Phase I — Offline learning and world-model cooperation

| Step | Requirement | Implementation | Status |
| --- | --- | --- | --- |
| 48 | Offline dataset from validated history; BC baseline; CQL/IQL; no extrapolation to unseen actions | `policies/offline` — `BehaviorCloningPolicy`, `CQLPolicy` (logsumexp penalty over dataset-supported actions), `IQLPolicy` (expectile value regression + advantage extraction); serving masks to dataset support | implemented |
| 49 | Keep predicted transitions separate from observed | `world_model.EmpiricalWorldModel` (SIMULATED-tagged predictions); trajectory store `origin` column excluded from all default loaders (tested) | implemented |

## Phase J — Constrained learning, rollout and governance

| Step | Requirement | Implementation | Status |
| --- | --- | --- | --- |
| 50 | max J_R subject to J_Ci <= B_i; Lagrangian for soft constraints; hard gates stay deterministic | `constrained.LagrangianOptimizer` (dual ascent, shaped rewards), `ResourceConstraintSet` (violations separate from the learner), `ConstrainedPolicyWrapper`; hard gates remain in `safety.py` | implemented |
| 51 | Shadow mode first; gradual promotion; rollback; fallback | `controller_adapter.ShadowModeController` (records proposals without authority), `LimitedAuthorityController` (risk/confidence-gated adoption), `RLPolicyService` (crash/timeout fallback to the deterministic coordinator), registry promotion/rollback | implemented |
| 52 | Monitor reward drift, task mix, constraint rejections, OOD; freeze or retire | `registry.record_monitoring_sample` + `monitoring_summary` (drift, constraint-rejection, OOD triggers), `freeze`/`retire` with audit trail | implemented |

## Section 5.1 — interface contracts

| Interface | Implementation |
| --- | --- |
| `encode_state(...)` | `StateEncoder.encode(RawStateSignals) -> OREState` |
| `get_eligible_actions(...)` | `HardConstraintGate.compute_feasible_actions -> FeasibleActionSet` |
| `propose(...)` | `PolicyService.propose_actions -> PolicyDecision` |
| `validate_decision(...)` | `HardConstraintGate.validate_decision -> DecisionValidation` |
| `execute_validated_action(...)` | `OREEnvironment.step` (dispatch via `CognitiveTaskCoordinator`) |
| `evaluate_outcome(...)` | `compute_step_reward` / `compute_terminal_reward` with `EvaluatorIdentity` |
| `record_transition(...)` | `TrajectoryStore.record_step` |
| `train_policy(...)` | policy `fit`/`observe`/`update` methods |
| `evaluate_candidate(...)` | `evaluation.evaluate_policy -> EvaluationReport` |
| `promote_policy(...)` | `PolicyRegistry.promote(candidate, evaluation, approval)` |

## Section 5.2 — SQLite trajectory schema

Implemented as specified (`rl_episodes`, `rl_steps`, `rl_rewards`,
`rl_policy_decisions`, `rl_evaluations`, `rl_dataset_manifests`) plus
`rl_state_refs` (immutable state artifacts by reference), `rl_quarantine`,
foreign keys, transactions, and episode/step indexes.

## Section 6 — ORE integration mapping

| ORE subsystem | RL responsibility | Boundary honored by |
| --- | --- | --- |
| Main Brain Coordinator (`CognitiveTaskCoordinator`) | proposals only | `RLPolicyService`/`ShadowModeController`; execution always via `coordinate()` |
| AdaptiveResourceCoordinator / CROS | budgets and measured state | `ResourceConstraintSet` + gates; `ModulePreferenceBridge` uses the documented `resource_module_preferences` hook |
| USG | read-only semantic context | gate rejects non-consult USG actions and mutation parameters |
| Shared workspace | versioned observations | environment reads `workspace.snapshot()`; publishes via `FindingPublisher` transactions |
| Specialist modules | execute operations | real manifests/ports through `ModuleRegistry` |
| Formal verification | evaluates correctness | STOP_SUCCESS gated on verified conclusions |
| SQLite logger | trajectories and provenance | dedicated `ore_rl` store with hash chains |
| Registry/monitoring | version, promote, roll back | `PolicyRegistry` with predeclared thresholds |

## Section 8 — acceptance tests

All acceptance criteria are covered by the test suite:

* **8.1 contract and safety tests** — `tests/test_rl_safety.py`,
  `tests/test_rl_contracts.py`, `tests/test_rl_trajectory_store.py`
  (schema rejection, ineligible selection, dependency blocking, budget
  enforcement, USG write protection, unverified rewards, simulated tagging,
  distinguishable outcomes, crash fallback, rollback).
* **8.2 data and reproducibility tests** — `tests/test_rl_trajectory_store.py`
  (complete linkage, checksum reproduction, leakage-safe splits, reward
  recomputation, tamper detection, state dedup).
* **8.3 learning and outcome tests** — `tests/test_rl_bandit.py`,
  `tests/test_rl_deep.py`, `tests/test_rl_options_offline.py`,
  `tests/test_rl_governance.py` (bandit vs random and baseline, identical
  benchmark conditions, promotion blocking, per-category reporting, safe
  fallback).

## Running

```bash
python -m pytest tests/test_rl_*.py -q     # ore_rl acceptance tests
python -m ore_rl                            # deterministic baseline smoke run
python examples/ore_rl_demo.py             # full roadmap demonstration
```

The demo writes `ore_logs/ore_rl_demo_report.json` with every stage's
metrics, the policy ranking, and the governance audit trail.
