# Cognitive Control Layer — Implementation Map (Point 02)

**Package:** `ore_control` (version 0.1.0) · **Algorithm:** ORE Integrated
Algorithm Point 02 — cognitive architectures (Soar, ACT-R, CLARION)
· **Source spec:** `docs/COGNITIVE_ARCHITECTURES_POINT02_ALGORITHM.md`
(40 ordered steps, phases A–H) · **Operation registry:** `CCA-001..CCA-040`.

The control layer is executive-only: it wraps — never replaces — the
repository's `ore_workspace`, `CognitiveTaskCoordinator`,
`AdaptiveResourceCoordinator` machinery (via `ore_contracts.resource_coordination`),
`ore_usg`, `ore_logger` (via the workspace observer), and the Point 01
`ore_blackboard` coordination layer, to which whole subtasks are delegated.

## 1. Module map

| Module | Responsibility | Source steps |
|---|---|---|
| `contracts.py` | Typed, serializable contracts: `TriggerEnvelope`, `CognitiveContext` (+ budget, open questions), `OperatorSpec` (+ risk/retry policy), `RetrievalCandidate`, `DecisionTrace`, run result | §3, Steps 1, 4 |
| `config.py` | `ControlPolicy`: retrieval coefficients (α β γ δ λ), utility weights (w_g…w_f), cycle/child/retrieval budgets, ambiguity margin, procedure promotion gates, stagnation window | §5, §8 |
| `registry.py` | Stable operation registry `CCA-001..CCA-040`, one per source step | all |
| `activation.py` | ACT-R-inspired retrieval mathematics: `B_i`, `S_i`, `R_i`, competition/ambiguity shortlists, marginal-gain depth ladder | Steps 13–17 |
| `operators.py` | Operator registry with eligibility + preconditions, candidate generation and deduplication, deterministic reference operators (retrieve / hypothesize / verify / resolve / request-evidence / delegate-to-blackboard) | Steps 6–9, 8 |
| `utility.py` | Outcome estimates (measured vs labelled priors), `EU(o)` and `U(o)`, hard-constraint filtering, deterministic selection with safe-parallel sets | Steps 18–21 |
| `subproblems.py` | Child goal specs, contracts, boundary guards (depth, child count, allowed fields/operators), reconciliation with explicit incompleteness | Steps 22–25 |
| `routing.py` | Processing-path classification, learned-component registry + suitability checks, proposal-only learned outputs | Steps 26–28 |
| `metacognition.py` | Uncertainty/disagreement/failure/stale-evidence monitoring and escalation triggers | Step 29 |
| `resources.py` | Reservation gate over the upstream reliability/cost machinery (Beta-prior reliability, EMA costs, circuit breakers) | Step 30 |
| `trace.py` | Decision-trace builder + `ore_logger`-backed persistence with failure tolerance | §3.4 |
| `procedures.py` | Candidate extraction from verified runs, held-out/counterexample generalization, promotion gates, versioned non-authoritative registry, cautious reuse + monitoring + rollback/quarantine | Steps 37–40 |
| `cycle.py` | `CognitiveControlCoordinator`: the 40-step orchestrator over cycles with idempotent trigger intake | Steps 1–40 |
| `evaluation.py` | Paired benchmark against the sequential module-coordination baseline with bootstrap CI | §7 |

## 2. Step-by-step coverage (CCA-001..CCA-040)

| Step | Implemented by | Notes |
|---|---|---|
| 1 | `contracts.TriggerEnvelope` + `cycle.process_trigger` intake | idempotency key, malformed triggers rejected before work |
| 2 | `WorkspaceManager.create_workspace` + per-cycle `snapshot()` | stale revisions reconciled and recorded (CCA-002 trace) |
| 3 | goal id + `GoalEvaluator` with critical requirements | no completion from module success alone |
| 4 | `CognitiveContext` build/resume in `cycle` | version-checked resume; stale reconciliation |
| 5 | `_state_summary` + `_cycle_fingerprint` | goal/questions/evidence/revision fingerprint |
| 6 | `OperatorRegistry.query` + procedure operators | disabled/schema/handler/capability reasons recorded |
| 7 | `PRECONDITION_LIBRARY` + `evaluate_preconditions` | machine-readable failure names |
| 8 | `candidate_actions_for_state` | typed candidates incl. registered procedures |
| 9 | `deduplicate_actions` + `_already_satisfied` + hard constraints | operator identity + arguments + revision keys |
| 10–17 | `_retrieval_phase` in `cycle` + `activation.py` | cues traceable; cache/index/resource domains; `B_i`/`S_i`/`R_i`; ambiguity records; marginal-gain depth ladder |
| 18 | `estimate_outcomes` | measured history vs labelled priors |
| 19 | `compute_utilities` | `EU(o)` and `U(o)` with formula labels |
| 20 | `apply_hard_constraints` | budgets, permissions, preconditions, validation standard |
| 21 | `select_action` | deterministic tie-break; disjoint-write safe parallel set |
| 22 | `decomposition_justified` + operator `decompose` flag | dependencies and benefit justify splitting |
| 23 | `create_child_contexts` | narrow goal, references, allowed operators, budget, deadline, success predicate |
| 24 | `ChildBoundaryGuard` + policy budgets | depth/child-count/operation limits; no parent-goal change; contract-scoped publication |
| 25 | `reconcile_child_outcomes` | merge compatible, preserve conflicts, explicit incompleteness |
| 26 | `classify_processing_path` | explicit/learned/hybrid with degradation |
| 27 | `LearnedComponentRegistry.check` | identity/version/schemas/task/resources/provenance/calibration/restrictions |
| 28 | proposal-only learned outputs (`LEARNED_PROPOSAL_NOTE`) | validation precedes publication; USG never mutated |
| 29 | `metacognition.monitor` | escalations: targeted retrieval, independent verification, human review |
| 30 | `ControlResourceCoordinator.reserve` | denial ⇒ defer/re-rank, never bypass |
| 31 | handler dispatch with checkpoint events and bounded retries | idempotent actions |
| 32 | `_validate_outcome` | schema/provenance/postconditions; exit ≠ correctness |
| 33 | `_publish_finding` through `FindingPublisher` → `workspace.publish` | exact decision recorded |
| 34 | `GoalEvaluator.evaluate` | weighted coverage + critical hard gates |
| 35 | fingerprint history + stagnation policy | bounded retries; switch/stop on repetition |
| 36 | continue/revise/wait/stop decision | structured stop reasons only |
| 37 | `extract_candidate_procedure` | ≥ 2 verified runs; operator-id sequence stability |
| 38 | `evaluate_generalization` | held-out set + counterexamples; memorization and resource regression checks |
| 39 | `evaluate_promotion_gates` + `ProcedureRegistry.register` | conjunctive gates; versioned, non-authoritative, rollback info |
| 40 | `ProcedureOperatorHandler` + monitoring | precondition re-checks; disable/rollback/quarantine |

## 3. Acceptance-criteria coverage (§7)

- **Contract and state tests** (`tests/test_control_contracts.py`):
  serialization round-trips and schema rejection; stale-revision reconciliation
  (CCA-002 trace); child publication/operator/goal boundaries; invalid and
  unavailable operators never dispatched; duplicate triggers without duplicate
  side effects.
- **Cognitive-cycle tests** (`tests/test_control_cycle.py`):
  deterministic selection under same snapshot/config/history; eligible and
  rejected operators with reasons per cycle; completed-goal stop decisions;
  critical unmet requirements block success; repeated fingerprints trigger
  stagnation handling; retrieval monotonicity, missing-history semantics,
  ambiguity preservation, depth stopping, per-item provenance/version, and a
  labelled ranking comparison; subproblem inheritance, limits, conflict
  preservation, explicit timeouts; learned-component schema mismatch,
  read-only USG, escalation on low confidence; bounded retries, reservation
  deferral, logger-failure provenance retention, benchmark harness.
- **Procedure-learning tests** (`tests/test_control_procedures.py`):
  single-run insufficiency; held-out/counterexample gates; versioning and
  rollback; degradation disables without corrupting authoritative knowledge.
- **Reliability/performance** (`tests/test_control_cycle.py` +
  `examples/cognitive_control.py`): the paired benchmark measures goal
  completion, operator applications, latency, retrieval cost, and conflict
  preservation; no performance improvement is claimed.

## 4. Honesty boundaries

- The reference operators and the toy sense ranker are deterministic rule-based
  implementations: they validate the control algorithm and its traceability,
  not real language-understanding quality.
- `B_i`, `S_i`, `R_i`, `EU(o)`, `U(o)`, and the procedure promotion gates are
  labelled engineering heuristics; nothing is presented as a calibrated
  probability, a Soar/ACT-R prescribed equation, or a validated benchmark.
- Activation-based retrieval is an ACT-R-inspired approximation with explicit
  missing-feature handling, not the full ACT-R memory model.
- Procedure replay evaluation uses the deterministic reference replay path;
  it is a generalization harness, not an external benchmark claim.
- The package is software-tested only: no task-validated or biologically
  supported status is claimed for any component.
