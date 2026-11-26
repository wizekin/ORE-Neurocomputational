# Blackboard Cognitive Coordination — Implementation of Algorithm 1 (Priority 1 / Source Point 1)

**Source specification:** `ORE_Integrated_Algorithm_Point_01.md` — "Algorithm 1 —
Blackboard-based cognitive coordination" (Steps 1–32, acceptance criteria 1–14).
**Implementation package:** `ore_blackboard` (additive; no existing package was modified
except the additive `pyproject.toml` package list and an appended README section).
**Repository base:** `wizekin/ORE-Neurocomputational`, branch `main`, HEAD `e23e012`.
**Operation registry:** stable IDs `BBA-001..BBA-032`, one per source step.
**Status of this document:** implementation map and software-test evidence. It is
**not** a claim of task-validated or biologically supported status for any component.

---

## 1. Design summary

`ore_blackboard` connects ORE's existing shared infrastructure into one bounded,
inspectable blackboard loop. It reuses — and does not duplicate — the following
repository components:

| Algorithm responsibility (spec section D) | Reused component | Where used |
|---|---|---|
| Shared versioned state | `ore_workspace.SharedWorkspace` / `WorkspaceManager` | Step 2 |
| Controlled finding publication | `FindingPublisher`, `WorkspaceStateManager` | Steps 17–22 |
| Provenance/dependencies | `Finding`, `Provenance`, dependency validation | Steps 17–19, 27 |
| Conflict preservation | `ConflictRegistry`, merge policies, `HypothesisScorer` | Steps 20–21 |
| Selective event notification | `SubscriptionRegistry`, `WorkspaceEventBus` | Steps 8, 12–13 |
| Bounded specialist scheduling | `ModuleScheduler` (`run_round`) | Steps 15–16 |
| Capability/dependency planning | `CognitiveTask`+`DependencyGraphPlanner`, `ModuleRegistry` | Steps 6–7 |
| Adaptive module/resource choice | `SQLiteResourceHistory`, `ModuleReliabilityTracker`, `ModuleResourceEstimator`, `ExecutionPlanBuilder` | Steps 14, 26, 31 |
| USG knowledge | `ore_usg.load_knowledge_bundle`, `USGKnowledgeAdapter`, `USGKnowledgeIndex` | Step 5 |
| Goal completion | `GoalEvaluator` | Steps 9, 25 |
| Runtime annotations and trace | `SQLiteWorkspaceObserver`, workspace events/revision history | Steps 27, 32 |

The coordinator never creates a competing workspace, never writes shared state
directly, and never bypasses typed publication. Specialists are manifest-backed
module backends (`describe()` / `execute()` protocol) whose typed artifacts are
validated against `ContractCatalog` contracts before bridging into findings.

## 2. Step-by-step implementation map

| Step (BBA ID) | Implementation |
|---|---|
| 1 `BBA-001` | `contracts.BlackboardTaskRequest` (typed validation, finite scalars, normalized priority, capability identifiers) + `request_fingerprint()` SHA-256 + task-id idempotency cache in `coordinator.run()`; reused identities with new content raise `ValueError`. |
| 2 `BBA-002` | `WorkspaceManager.create_workspace` with a `PermissionPolicy` built from the authority matrix and an empty-default `SubscriptionRegistry`; run identity (`run_id`) kept distinct from workspace identity; explicit `resume_workspace_id` continuation supported. |
| 3 `BBA-003` | `contracts.BlackboardLayer` (8 layers) + `LAYER_FIELDS` + `one_hot_layer()` (Σz=1) + `LifecycleState` partial order; the bridge stamps `layer` and `lifecycle_state` onto every published record and validates field-layer membership. |
| 4 `BBA-004` | `contracts.build_input_root` / `InputRoot`: SHA-256 digest over all sources, `\w+` token spans validated `0 ≤ s ≤ e ≤ N`, per-source identity + source-group label `g(q)`; published once as an `input_root_observation` finding in `artifact_references`. |
| 5 `BBA-005` | `usg_integration.BlackboardUSGSnapshot`: `load_knowledge_bundle` digest verification → `USGKnowledgeAdapter.convert` → `USGKnowledgeIndex`; only the compact `USGSnapshotReference` (bundle id/digest/namespace/adapter version/categories) is published; specialists receive `ORE.USG.LEXICON@1.0.0` artifacts read-only; no write-back. |
| 6 `BBA-006` | `planning.register_specialist_modules`: manifest registration + backend binding in `ModuleRegistry`; write authority matrix `W_if` built from each specialist's `write_fields` and enforced by `build_permission_policy` (plus the established `cognitive_control` producer); capability coverage computed with mandatory omissions reported separately. |
| 7 `BBA-007` | `planning.build_dependency_plan` → `DependencyGraphPlanner.build` (exact contract versions, DAG layers, compatibility matrix, missing capabilities/dependencies/inputs, cycle nodes). A non-executable plan returns a **blocked** result with the exact missing work — never a partial pipeline. |
| 8 `BBA-008` | `ModuleScheduler.register_module` with task-specific fields, sensitivity weights `w_mf`, and per-module thresholds `θ_m` (specialist class attributes). Broadcast ablation available for Step 30. |
| 9 `BBA-009` | `GoalEvaluator(weights, threshold, critical_requirements)` — weighted coverage `C_G` plus critical-requirement hard gates; `planning.build_task_profile` derives the `TaskProfile` used by resource planning. |
| 10 `BBA-010` | `execution.build_snapshot_artifact` (detached typed snapshot per round) + `ComputationRegistry.invocation_id` = `H(task_id ∥ module_id ∥ version ∥ input_hashes ∥ r_t)`. |
| 11 `BBA-011` | `relevance.measure_workspace_changes`: boolean indicator for scalar fields, symmetric-difference ratio `|A△B|/max(1,\|A∪B\|)` for collections, changed-key ratio for keyed records; the metric used is recorded with every event trace. |
| 12 `BBA-012` | `SubscriptionRegistry.relevance_score` implements `R_m = Σ_f A_mf·w_mf·Δ_f`; `relevance_report` orders eligible modules by descending score with a module-ID tie-breaker. |
| 13 `BBA-013` | Committed events are dispatched by `WorkspaceStateManager.notify_subscribers` → `WorkspaceEventBus.publish`: eligibility by field intersection, per-`(module,event)` dedup, bounded worker pool, delivery timeout, per-module `DeliveryResult` isolation (callback failures never abort other subscribers). |
| 14 `BBA-014` | `resources.BlackboardResourceCoordinator` wraps `SQLiteResourceHistory` + `ModuleReliabilityTracker` (Beta(2,2)-prior posterior mean) + `ModuleResourceEstimator` (EMA costs) + `ExecutionPlanBuilder`; `compute_comparison_score` implements the proposed `U_m = αR+βIG+γG+δQ−λL−μM−νK` heuristic with its coefficients and "not a calibrated probability" disclosure; circuit-open gating is applied before every round. |
| 15 `BBA-015` | `ModuleScheduler` bounds (workers/rounds/activations/timeouts taken from `CycleBudgets`, which reuse the scheduler's defaults) + coordinator-level `max_module_calls`, `max_replans`, wall-clock and memory budgets; per-round gating of circuit-open modules. |
| 16 `BBA-016` | `execution.SpecialistInvocation` assembles manifest-declared typed inputs, invokes the registered backend, and `bridge.validate_module_result` enforces `V_c` (identity/version/producer/run/contract/lineage). Exceptions and timeouts are isolated by the scheduler without fabricating results. |
| 17 `BBA-017` | `bridge.ArtifactFindingBridge` converts validated finding-batch artifacts into `Finding` envelopes with producer/version/target/span/rules/evidence/USG references/dependencies/confidence; module content fingerprints come from the existing `Finding.fingerprint()`. |
| 18 `BBA-018` | `WorkspaceStateManager.submit` enforces the nine checks (workspace identity, writable field, producer authority, finite JSON payload, no future base revision, span validity, duplicate id/fingerprint, dependency readiness, stale-mutable regeneration) and returns the exact decision to the module and trace. |
| 19 `BBA-019` | Well-formed findings with missing prerequisites are stored in `pending_findings` and automatically reconsidered when a prerequisite is accepted (verified in tests); absence is never converted into a false/empty value. |
| 20 `BBA-020` | `MergePolicyRegistry` field-specific operators (append-only lists, keyed lexical records, keyed working memory with deactivation, memory dedup by id, discourse history, module status) — never a generic last-write-wins. |
| 21 `BBA-021` | `ConflictRegistry` retains both alternatives (`ConflictRecord`); `hypotheses.aggregate_evidence`/`HypothesisScorer` implement `E(h)=Σ_g ρ_g Aggregate_i(w_i e_i)` with conservative max-aggregation per correlation group; `BayesianHypothesisUpdater` performs log-odds updates only under declared independence with unique evidence ids; `resolve_conflict_with_evidence` is the only resolution path. |
| 22 `BBA-022` | Commits happen under the workspace lock inside the state manager; failed validation/merge leaves the previous accepted state intact (verified); the revision chain in `RevisionStore` is reconstructable; events reference the committed revision only. |
| 23 `BBA-023` | Routing = event delivery + relevance thresholds + `ComputationRegistry` suppression of `(module_version, input_fingerprints, relevant-revision)` repeats; only relevant subscribers activate. |
| 24 `BBA-024` | Bounded rounds/activations/calls/time budgets with explicit exhaustion reasons; the proposed stagnation test (`ΔC_G ≤ ε` for `patience` rounds) is implemented in `StagnationPolicy` and documented as a heuristic that never overrides hard gates. |
| 25 `BBA-025` | `GoalEvaluator.evaluate` after every meaningful revision: `C_G = Σw_kc_k/Σw_k` with `Complete = 1[C_G≥τ]·Π_j 1[r_j]` hard gates; uncertainty is reported separately from missing structure. |
| 26 `BBA-026` | `coordinator._recover`: failure classification (`FailureClassification`), shared retry budget `Σretries ≤ B_retry`, bounded exponential backoff `t_n = min(t_cap, t_0·2^n)+jitter`, circuit-breaker recording through `ModuleReliabilityTracker`, alternative-provider binding via `registry.find_by_capability` with permission grants, module-status publication for unrecovered failures; failed computations never become successful empty findings. |
| 27 `BBA-027` | `trace.ProvenanceTraceBuilder` builds `G_T` from findings (accepted/pending/rejected), invocations, events, conflicts, USG bundle, and goal evaluations; `completeness()` computes `T_c = |C_traceable|/max(1,|C_final|)`; `verify_input_integrity()` re-verifies the stored input digest before reproducibility claims. |
| 28 `BBA-028` | `contracts.BlackboardTaskResult` keeps separate sections for observations, imported knowledge references, supported hypotheses (ranked, with disclosed `E(h)` definitions and `calibrated_probability=False`), unresolved alternatives, verified conclusions, and recommended next actions. |
| 29 `BBA-029` | `evaluation.run_benchmark` + `SequentialBaselineRunner` (same specialists, same typed inputs, same publication path, fixed dependency order): SR, UAR, mean/median/p95 latency, conflicts preserved, recovery rate, provenance completeness, USG contribution; `paired_bootstrap` reports mean differences with bootstrap CIs and explicit no-single-run-claims notes. |
| 30 `BBA-030` | `evaluation.run_ablation` disables one mechanism at a time (`selective_subscriptions`, `resource_aware_choice`, `usg_retrieval`, `hypothesis_conflict_handling`, `dependency_deferral`, `provenance_checks`, `bounded_parallelism`) and reports `Δ_j = M_full − M_ablated(j)` contribution tables. |
| 31 `BBA-031` | `policy_learning.RunMeasurementRecorder` records EMA cost measurements and reliability outcomes as explicit post-run transactions; `PolicyProposalBuilder`/`PromotionGateEvaluator` gate proposals with `Promote = ∧_j Gate_j` (regression, safety, provenance); nothing is applied automatically and the accepted answer is unchanged by feedback statistics (tested). |
| 32 `BBA-032` | `coordinator.run` closes the scheduler, resource history, and manager, distinguishes completed / needs-more-processing / blocked / failed termination, retains all workspace state and revisions (nothing is deleted), and returns the reproducibility tuple `(task_fingerprint, workspace_id, revision, artifact digests, module versions, USG bundle digest, policy version)`. |

## 3. Acceptance-criteria coverage

All fourteen acceptance criteria are exercised by focused tests:

1. One workspace + recorded run identity — `TestAcceptanceOneWorkspaceAndRunIdentity`.
2. Detached snapshots, typed/authorized publication only — `TestAcceptanceDetachedSnapshotsAndTypedPublication`.
3. Committed findings activate only relevant subscribers — `TestAcceptanceSelectiveSubscriberActivation`.
4. Dependency-free modules run concurrently (same round, worker pool) — `TestAcceptanceConcurrentIndependentModules`.
5. Deferred findings reconsidered when prerequisites arrive — `TestAcceptanceDependencyDeferral`.
6. Duplicate submissions idempotent (content fingerprint + task-id replay) — `TestAcceptanceIdempotentDuplicates`.
7. Contradictions preserved until an explicit resolution action — `TestAcceptanceConflictPreservation`.
8. USG read-only with attached bundle/record provenance — `TestAcceptanceUSGReadOnly`.
9. Every commit advances a reconstructable revision; failed publications leave state intact — `TestAcceptanceRevisionIntegrity`.
10. Failures isolated, reported, bounded by retry/round/time budgets (transient retry, permanent failure, timeout classification, delivery isolation) — `TestAcceptanceBoundedFailures`.
11. Goal completion checks weighted coverage and critical requirements — `TestAcceptanceGoalCompletion`.
12. Results trace to inputs, modules, USG digest, and rules (reproducibility tuple + replay determinism) — `TestAcceptanceTraceability`.
13. Representative evaluation versus a sequential baseline — `TestBenchmarkHarness` (acceptance 13's metrics) plus result-shape tests in `TestAcceptanceResultSections`.
14. Ablations demonstrate which mechanisms provide measurable value — `TestAblationHarness`.

Operation coverage: a successful run records `BBA-001..BBA-028`, `BBA-031`,
`BBA-032` (29 of 32); a run containing a recovered failure adds `BBA-026`; the
evaluation harness supplies `BBA-029` and `BBA-030`. The full 32-step coverage
is asserted in `TestAcceptanceOperationCoverage`.

## 4. Running

```bash
python -m pytest tests/test_blackboard_contracts.py -q
python -m pytest tests/test_blackboard_coordination.py -q
python -m pytest tests/test_blackboard_evaluation.py -q
python examples/blackboard_coordination.py
```

## 5. Honest limitations

- The bundled specialists in `ore_blackboard/specialists.py` are deterministic
  rule-based reference processors (with a small built-in fallback lexicon) that
  exist to exercise and validate the coordination algorithm. They are not
  validated lexical, syntactic, or semantic models.
- Benchmark success rates measure goal completion of the coordination harness,
  not real-world language-understanding quality; the benchmark notes state this
  explicitly.
- Evidence scores `E(h)`, comparison scores `U_m`, and module confidences are
  uncalibrated heuristic quantities and are labelled as such everywhere.
- The Bayesian updater is provided for callers that can defensible declare
  conditional independence; the default path never treats scores as calibrated
  probabilities.
- This implementation is software-tested only. No claim of task validation,
  biological support, or empirical cognitive fidelity is made or implied.
