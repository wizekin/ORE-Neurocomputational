# ORE Integrated Algorithm — Point 02
## Cognitive architectures: Soar, ACT-R and CLARION

**Repository:** `wizekin/ORE-Neurocomputational`  
**Source:** `Most Intelligent Systems 01-24.txt`, Point 2  
**Priority rank across the 24-point assessment:** 3  
**Document type:** Detailed algorithmic design specification  
**Status:** Proposed architecture; this document does not claim the mechanisms are already implemented.

---

## 1. Objective and design decision

Build one ORE Cognitive Control Layer that organizes the system's existing knowledge, workspace, task execution, memory retrieval and specialist modules into a continuous, bounded decision process.

The design selectively combines three mature cognitive-architecture ideas:

- **Soar-inspired executive control:** a decision cycle, explicit operators, temporary problem-solving contexts, and cautious learning from solved tasks.
- **ACT-R-inspired memory control:** activation-based retrieval, competition among candidate memories, accessibility states, and explicit retrieval-cost estimates.
- **CLARION-inspired processing coordination:** controlled cooperation between explicit symbolic operations and replaceable learned/statistical components, with meta-cognitive monitoring.

These are engineering inspirations, not a proposal to reproduce three complete cognitive architectures. ORE must not create duplicate workspaces, schedulers, memory stores or task coordinators.

## 2. Integration boundary and reuse rules

The cognitive control layer should sit above existing ORE infrastructure and call it through stable contracts.

| Existing ORE component | Reuse responsibility | New layer must not do |
|---|---|---|
| `ore_workspace` | Authoritative shared state, revisions, findings, evidence, goals, conflicts and publication | Maintain a competing copy of global cognitive state |
| `CognitiveTaskCoordinator` | Dependency-aware task planning and execution | Reimplement the lower-level task scheduler |
| `AdaptiveResourceCoordinator` | Resource estimates, allocation, workload and execution feedback | Bypass resource limits or create a second allocator |
| `ore_usg` | Structured concepts, semantic relations, rules and knowledge retrieval | Treat unverified model output as authoritative USG knowledge |
| `ore_logger` | Durable events, annotations, decisions, measurements and provenance | Store the only copy of operational state in logs |

Before implementation, inspect the current public APIs and tests. The names above identify integration targets; exact import paths and method signatures must be taken from the current repository rather than guessed.

## 3. Core data contracts

The first implementation should define typed, serializable contracts. Keep the initial schema small and versioned.

### 3.1 CognitiveContext

Required fields:

- `context_id`: stable unique identifier.
- `parent_context_id`: optional parent for a bounded subproblem.
- `goal_id`: authoritative workspace goal reference.
- `workspace_revision`: revision used to make the current decision.
- `status`: one of `READY`, `RUNNING`, `WAITING`, `SUCCEEDED`, `FAILED`, `CANCELLED`.
- `active_hypotheses`: references to hypotheses, not untyped free-form state.
- `open_questions`: unresolved questions with priority and evidence requirements.
- `budget`: remaining time, operation, memory and concurrency limits.
- `depth`: current subproblem depth.
- `cycle_fingerprint`: fingerprint used to detect repeated states.
- `created_at`, `updated_at`: timestamps.
- `schema_version`: explicit version.

### 3.2 OperatorSpec

Each cognitive operator must declare:

- Stable `operator_id` and version.
- Human-readable purpose.
- Preconditions and applicability predicate.
- Input and output schemas.
- Required capabilities/modules.
- Estimated time, memory and compute cost.
- Risk class and reversibility.
- Expected progress or information gain.
- Whether it may write to workspace state.
- Required validation policy.
- Retry and timeout policy.
- Execution handler reference.
- Provenance and annotation requirements.

An operator is a decision-level cognitive action. It may delegate actual work to the existing task coordinator.

### 3.3 RetrievalCandidate

Each candidate memory/knowledge item must include:

- Stable record reference and source type.
- Cue-match features.
- Recency/frequency evidence where available.
- Explicit semantic relevance.
- Confidence and provenance.
- Retrieval cost estimate.
- Access state.
- Version or revision.
- Reasons for inclusion and exclusion.

### 3.4 DecisionTrace

Every cycle must record:

- Context and workspace revision.
- Open goals/questions.
- Candidate operators and rejected candidates.
- Candidate retrievals and scores.
- Resource estimates and constraints.
- Selected action and selection rationale.
- Execution result and validation result.
- State transition and next-cycle reason.
- Timing and cost measurements.

Persist trace events through `ore_logger`; do not log secrets or unnecessarily duplicate large payloads.

---

## 4. Integrated algorithm: 40 detailed steps

### Phase A — Establish a stable cognitive situation

**Step 1 — Receive a cognitive trigger.**  
Accept a new user task, a workspace event, a completed subtask, a conflict notification, a resource event or a scheduled continuation. Normalize it into a trigger envelope containing the source, event ID, timestamp, relevant entity references and idempotency key. Reject malformed triggers before creating work.

**Step 2 — Read an authoritative workspace snapshot.**  
Request a consistent snapshot from `ore_workspace`. Record its revision. All decisions in this cycle are based on that revision. If the workspace changes during execution, later publication must use the workspace's conflict/revision protocol rather than silently overwriting newer state.

**Step 3 — Resolve the governing goal.**  
Map the trigger to an existing goal or create a proposed goal through the established goal API. Preserve explicit success conditions, constraints, priority, critical requirements and cancellation rules. Do not infer that a task is complete merely because a module returned successfully.

**Step 4 — Build or resume the CognitiveContext.**  
Create a context for a new task or resume a compatible nonterminal context. Attach references to the authoritative goal and workspace revision. Restore only validated, version-compatible state. If the stored context is stale, reconcile it against the current workspace before continuing.

**Step 5 — Derive the current cognitive state.**  
Compute a compact state summary: satisfied conditions, unsatisfied conditions, open questions, active hypotheses, evidence gaps, conflicts, available capabilities and remaining budget. Keep the full evidence in its authoritative store and use references in the context. Calculate a stable fingerprint from normalized goal, open questions, relevant evidence IDs and context state.

### Phase B — Generate explicit cognitive operators (Soar-inspired)

**Step 6 — Query the operator registry.**  
Load operators compatible with the task type and current system capabilities. Validate each operator's schema version, handler availability, permissions and execution restrictions. An unavailable operator is not eligible, even if its textual description appears relevant.

**Step 7 — Evaluate operator preconditions.**  
For each operator, evaluate its applicability predicate against the current context and workspace snapshot. Record every failed precondition with a machine-readable reason. Preconditions must be deterministic where possible; uncertain preconditions should be represented as an explicit evidence-gathering action.

**Step 8 — Generate candidate cognitive actions.**  
Create candidate actions for operations such as retrieving lexical/semantic knowledge, generating hypotheses, testing hypotheses, resolving conflicts, building causal models, planning, verifying and refining representations. Candidate generation may include domain-specific operators registered by specialist modules. Do not hard-code the list as the complete future vocabulary.

**Step 9 — Deduplicate and constrain candidates.**  
Deduplicate candidates by operator identity, normalized arguments and relevant context revision. Remove actions that violate permissions, resource budgets, depth limits, safety restrictions or already-satisfied goal conditions. Preserve alternative candidates when they represent meaningfully different hypotheses or methods.

### Phase C — Retrieve relevant knowledge and experience (ACT-R-inspired)

**Step 10 — Construct retrieval cues.**  
Build cues from the goal, unresolved questions, semantic entities, explicit relations, current hypotheses, recent evidence and operator requirements. Keep task-critical terms distinct from generic terms. Every cue must be traceable to a source field or rule.

**Step 11 — Identify retrieval domains.**  
Select only relevant stores: USG semantic knowledge, lexical records, episodic execution history, verified procedures, module capability metadata, workspace findings or external resources. Do not query every store for every task. Use a cheap index/cache first when its coverage and freshness are adequate.

**Step 12 — Generate candidate memories.**  
Retrieve a bounded candidate set from the selected stores. Apply access permissions, version filters, freshness requirements and provenance checks before scoring. Keep ambiguous candidates separate; do not collapse distinct senses or hypotheses prematurely.

**Step 13 — Estimate base-level activation.**  
For records with usage history, calculate a simplified recency/frequency activation:

\[
B_i = \ln\left(\varepsilon + \sum_{j=1}^{n_i} t_{ij}^{-d}\right)
\]

Here (t_{ij}) is elapsed time since a recorded use/encounter, (d>0) is a configurable decay parameter and \(\varepsilon\) prevents taking the logarithm of zero. This is an ACT-R-inspired engineering approximation, not a complete ACT-R equation. If usage timestamps are absent or unreliable, mark the feature unavailable instead of fabricating it.

**Step 14 — Estimate cue association.**  
For candidate (i), combine weighted cue associations:

\[
S_i = \sum_{k=1}^{m} W_k A_{ik}
\]

(W_k) is the importance of cue (k), and (A_{ik}) is the measured association between cue (k) and candidate (i). Normalize feature scales and preserve the source of each association. Do not treat string overlap alone as semantic proof.

**Step 15 — Combine relevance, confidence and cost.**  
Use a configurable retrieval score:

\[
R_i = \alpha B_i + \beta S_i + \gamma Q_i + \delta P_i - \lambda C_i
\]

where (Q_i) is task-specific relevance, (P_i) is provenance/quality support, and (C_i) is estimated retrieval cost. The coefficients are calibrated configuration, not universal constants. Missing features must be handled explicitly; do not silently substitute zero if that would distort ranking.

**Step 16 — Preserve retrieval competition and ambiguity.**  
Rank candidates, but retain a bounded shortlist when top scores are close or when candidates represent different plausible interpretations. A winner may be selected only if it passes the task's confidence/separation policy. Otherwise create an ambiguity record and schedule a discriminating retrieval or hypothesis test.

**Step 17 — Choose a retrieval depth under budget.**  
Estimate the marginal expected value of searching deeper. Stop when the next retrieval's expected information gain is lower than its cost, when the retrieval budget is exhausted, or when a sufficient evidence threshold is met. Escalate from local cache to structured index to larger knowledge resource only when justified. Record skipped sources and why they were skipped.

### Phase D — Select an operator using expected progress and cost

**Step 18 — Estimate operator outcomes.**  
For each eligible operator, estimate probability of useful progress, expected value, time, resource demand, risk, reversibility and likelihood of producing independently verifiable output. Use measured historical execution data when sufficient; otherwise use declared conservative estimates and mark them as priors.

**Step 19 — Calculate expected utility.**  
A basic selection score may be:

\[
EU(o)=\frac{\Pr(\text{useful progress}\mid o)\cdot V(o)}
{\epsilon + C(o)}
\]

A richer normalized score can be:

\[
U(o)=w_gG(o)+w_iI(o)+w_cK(o)-w_tT(o)-w_rR(o)-w_fF(o)
\]

where (G) is expected goal progress, (I) is information gain, (K) is confidence/quality, (T) is time cost, (R) is risk and (F) is resource footprint. These are ORE design scores, not equations prescribed by Soar or ACT-R.

**Step 20 — Apply hard constraints before ranking.**  
Reject operators that exceed remaining budgets, violate required preconditions, lack permissions, conflict with a critical goal condition or cannot be validated to the required standard. Hard constraints must not be overridden by a high utility score.

**Step 21 — Select one action or a safe parallel set.**  
Choose the highest-utility eligible action, or a set of mutually independent actions when their read/write sets and resource reservations prove safe concurrency. A softmax policy may be used for controlled exploration in experiments, but production selection should default to deterministic ranking and stable tie-breaking. Log the alternatives and rationale.

### Phase E — Create bounded problem-solving contexts (Soar-inspired)

**Step 22 — Detect whether the action requires decomposition.**  
If an operator needs unresolved evidence or multiple independent hypotheses, create child contexts. Otherwise delegate the operation directly. Decomposition must be justified by dependencies and expected benefit, not by arbitrary splitting.

**Step 23 — Define each child goal and contract.**  
Every child receives a narrow goal, input references, required evidence, allowed operators, output schema, budget, deadline and success predicate. Include its parent context ID and the workspace revision from which it was derived.

**Step 24 — Enforce subproblem boundaries.**  
Set maximum depth, maximum child count, time and operation budgets. Prevent a child from changing its parent's goal or publishing authoritative state outside its contract. Child contexts return typed findings, evidence references, unresolved issues and validation status.

**Step 25 — Reconcile child outcomes with the parent.**  
Collect child results only after validating their contracts. Merge compatible findings through workspace integration rules; preserve conflicting findings as conflicts rather than selecting one by arrival order. Mark incomplete or timed-out children explicitly and decide whether to retry, narrow, defer or abandon them.

### Phase F — Coordinate explicit and learned processing (CLARION-inspired)

**Step 26 — Classify the processing path.**  
Route the operation to explicit symbolic processing, a learned/statistical specialist, or a hybrid path. USG rules, typed predicates and deterministic constraints belong in explicit processing. Learned components may rank, classify, predict or propose candidates within declared schemas.

**Step 27 — Check learned-component suitability.**  
Before invoking a learned component, verify model identity/version, input/output schemas, supported task, resource requirements, evaluation provenance, confidence calibration, known limitations and execution restrictions. Select it through the replaceable model/specialist registry, not by hard-coded vendor identity.

**Step 28 — Keep proposals separate from authoritative knowledge.**  
Treat learned outputs as proposals with confidence and provenance. Require schema validation, evidence checks and the relevant deterministic verification before publication. A learned component must not directly mutate authoritative USG facts or rules.

**Step 29 — Apply meta-cognitive monitoring.**  
Track uncertainty, disagreement, resource consumption, repeated failures, stale evidence and confidence calibration. If confidence is low or specialists disagree, trigger a targeted retrieval, independent verification or human-review path. Never convert uncertainty into false certainty merely to force completion.

### Phase G — Execute, validate and continue the cognitive cycle

**Step 30 — Reserve resources and dispatch.**  
Ask `AdaptiveResourceCoordinator` to estimate/reserve resources and `CognitiveTaskCoordinator` to plan/execute the task. Pass idempotency key, deadlines, cancellation token, context references and declared resource bounds. If reservation fails, re-rank, defer or reduce scope rather than bypassing the coordinator.

**Step 31 — Execute with checkpoints and event capture.**  
Run the selected operator through its registered handler. Capture start, progress, completion, timeout, cancellation and failure events. Long-running tasks must expose checkpoints or resumable state where supported. Retries must be idempotent or guarded against duplicate side effects.

**Step 32 — Validate outputs against contracts.**  
Check output schema, provenance, evidence references, version compatibility, preconditions, postconditions and operator-specific validation. Use deterministic tests or formal checks whenever available. A successful process exit is not equivalent to a correct result.

**Step 33 — Integrate validated findings atomically.**  
Submit validated findings to the authoritative workspace integration path. Respect revisions, duplicate detection, conflicts and deferred/rejected states. Do not write directly around workspace APIs. Record the exact accepted/rejected/deferred result.

**Step 34 — Recompute goal progress.**  
Evaluate goal conditions from the current authoritative state. A weighted progress estimate may be:

\[
P = \frac{\sum_{k=1}^{n} w_k s_k}{\sum_{k=1}^{n} w_k}
\]

where (s_k\in[0,1]) is the satisfaction estimate for condition (k). Critical requirements remain hard gates: a high aggregate score cannot compensate for an unsatisfied critical condition.

**Step 35 — Detect stagnation and cycles.**  
Compare the current state fingerprint with recent fingerprints and inspect repeated operator/context patterns. If the same state recurs without measurable progress, stop repeating the same action. Select a materially different operator, obtain new evidence, revise the subgoal or escalate. Apply retry limits and backoff.

**Step 36 — Decide whether to continue, revise, wait or stop.**  
Continue when a useful eligible action remains and budget permits. Revise when evidence invalidates a hypothesis or the plan. Wait when an external dependency or scheduled event is required. Stop successfully only when all critical goal gates pass and the goal evaluator confirms completion. Stop unsuccessfully with a structured reason when constraints make further progress unjustified.

### Phase H — Learn reusable procedures only after verification

**Step 37 — Identify a candidate reusable procedure.**  
After a verified task, inspect its decision trace for a stable sequence of operators, conditions and outputs. Extract a candidate procedure only when the sequence is sufficiently repeatable and its assumptions are explicit. Do not generalize from a single successful execution by default.

**Step 38 — Evaluate generalization and safety.**  
Test the candidate against held-out examples, edge cases, changed contexts, failure cases and counterexamples. Measure success rate, calibration, resource cost and regression impact. Reject procedures that only memorize a specific instance or rely on unverified correlations.

**Step 39 — Version and register the procedure as non-authoritative.**  
Store procedure identity, version, preconditions, parameters, training/evaluation provenance, test results, limitations and rollback information. Require review/approval policy appropriate to the procedure's risk. The procedure registry is not a mechanism for silently rewriting USG knowledge.

**Step 40 — Reuse cautiously and monitor outcomes.**  
Retrieve the procedure as a candidate operator, re-check its preconditions and version, and validate each execution. Track success/failure and resource outcomes. Disable, roll back or quarantine a procedure when its performance degrades or a safety condition is violated.

---

## 5. Mathematical operations and calibration

### 5.1 Retrieval activation

\[
B_i=\ln\left(\varepsilon+\sum_j t_{ij}^{-d}\right)
\]

Use positive elapsed times, a bounded decay parameter and explicit handling for missing history. The formula captures recency/frequency effects but must not be presented as the full ACT-R model.

### 5.2 Cue association

\[
S_i=\sum_k W_kA_{ik}
\]

Normalize cue weights, avoid double-counting correlated cues and retain per-cue contribution traces.

### 5.3 Retrieval ranking

\[
R_i=\alpha B_i+\beta S_i+\gamma Q_i+\delta P_i-\lambda C_i
\]

Calibrate on labeled retrieval tasks. Evaluate precision@k, recall@k, mean reciprocal rank, ambiguity preservation and latency/cost. Compare against the current retrieval baseline.

### 5.4 Expected utility and operator selection

\[
EU(o)=\frac{p_oV_o}{\epsilon+C_o}
\]

\[
U(o)=w_gG_o+w_iI_o+w_cK_o-w_tT_o-w_rR_o-w_fF_o
\]

Normalize terms before combining them. Keep safety, permission and critical-goal checks as hard filters outside the score.

### 5.5 Goal progress

\[
P=\frac{\sum_kw_ks_k}{\sum_kw_k}
\]

Use this as a monitoring metric only. Critical requirements are Boolean gates and cannot be averaged away.

### 5.6 Online measurement smoothing

For noisy resource and latency observations, an exponential moving average can be used:

\[
\hat{x}_t=\rho x_t+(1-\rho)\hat{x}_{t-1},\quad 0<\rho\leq1
\]

Keep raw measurements as well as smoothed estimates. Do not use smoothed values to hide sudden resource-limit violations.

### 5.7 Cycle and progress checks

Use stable hashes/fingerprints of normalized state, but confirm semantic equality before treating a hash collision or superficial similarity as a true cycle. Progress should be measured by new validated evidence, satisfied conditions, reduced uncertainty or completed dependencies—not by number of tool calls.

---

## 6. Implementation sequence

Implement incrementally in this order:

1. **Repository/API audit:** map actual workspace, task coordinator, resource coordinator, USG and logger interfaces; record exact contracts and existing tests.
2. **Typed context and trace contracts:** add schema-versioned context, operator and decision-trace models without changing execution behavior.
3. **Deterministic cognitive cycle:** implement trigger → snapshot → goal/state evaluation → operator eligibility → dispatch → validate → progress → continue/stop.
4. **Operator registry:** add metadata, preconditions, capabilities, costs and validation policies; initially register wrappers around existing operations.
5. **Decision trace and annotations:** persist each decision and function/class annotation using the existing logger conventions.
6. **Bounded child contexts:** add parent/child linkage, contracts, depth limits, budgets and reconciliation.
7. **Activation-based retrieval:** integrate with existing memory/USG retrieval rather than adding a competing store.
8. **Retrieval cost model:** use measured lookup, processing and integration costs; feed estimates to the resource coordinator.
9. **Explicit/learned routing:** add replaceable learned specialists only when stable schemas and evaluation data exist.
10. **Procedure learning:** implement last, behind verification, generalization tests, versioning and rollback.

Do not begin by implementing all 40 steps at once. The first deployable milestone is the deterministic cycle and its traceability.

## 7. Acceptance criteria and tests

### Contract and state tests
- Context serialization round-trips across supported schema versions.
- Stale workspace revisions are detected and reconciled.
- A child cannot publish outside its declared contract.
- Invalid or unavailable operators are never dispatched.
- Duplicate triggers do not create duplicate side effects.

### Cognitive-cycle tests
- Given the same snapshot, configuration and history, deterministic mode selects the same operator.
- Each cycle records eligible and rejected operators with reasons.
- A completed goal produces a stop decision; an incomplete goal does not.
- A critical unmet requirement blocks success regardless of aggregate progress.
- Repeated state fingerprints trigger cycle handling rather than endless repetition.

### Retrieval tests
- Recency/frequency features behave monotonically under controlled inputs.
- Missing usage history is represented as missing, not fabricated.
- Retrieval ranking improves or at least does not regress against the current baseline on a labeled set.
- Ambiguous candidates remain available when scores are close.
- Cost-aware retrieval respects budgets and can stop before exhausting all stores.
- Every retrieved item carries provenance and a version/reference.

### Subproblem tests
- Child contexts inherit only intended references and bounded budgets.
- Depth, child-count, deadline and operation limits are enforced.
- Parent reconciliation preserves conflicting findings.
- Timed-out or failed children are represented explicitly and do not imply success.

### Hybrid-processing tests
- Learned model schema/version mismatches block invocation.
- Learned outputs cannot directly mutate authoritative USG knowledge.
- Low confidence or specialist disagreement triggers verification/escalation.
- Explicit symbolic checks remain independently testable.

### Reliability and performance tests
- Cancellation and retry behavior are idempotent.
- Resource reservation failure leads to defer/re-plan rather than bypass.
- Logger failures follow the repository's declared reliability policy and do not silently erase decision provenance.
- Measure latency, CPU/memory use, retrieval cost and goal-completion rate against the existing sequential/module-coordination baseline.
- No performance improvement is claimed until benchmark results are recorded.

### Procedure-learning tests
- A single successful run is insufficient to promote a reusable procedure.
- Held-out and counterexample tests are required.
- Every promoted procedure is versioned and reversible.
- A degraded procedure can be disabled without corrupting authoritative knowledge.

---

## 8. Key design risks and safeguards

| Risk | Safeguard |
|---|---|
| Duplicating existing coordinators/workspace | Keep the new layer executive-only; delegate execution and state integration |
| Unbounded subgoal creation | Enforce depth, child-count, time and operation budgets |
| Retrieval-score overconfidence | Preserve feature provenance, calibration and ambiguity |
| Cost estimates becoming stale | Update from measured outcomes and retain raw metrics |
| Learned model contaminating symbolic knowledge | Proposal-only outputs plus independent validation |
| Endless operator loops | State fingerprints, progress checks, retry limits and alternative actions |
| False task completion | Critical gates and authoritative goal evaluation |
| Unsafe learned procedures | Held-out testing, versioning, review policy and rollback |

## 9. Final integrated control loop

```text
TRIGGER
  -> READ AUTHORITATIVE WORKSPACE SNAPSHOT
  -> RESOLVE GOAL AND BUILD COGNITIVE CONTEXT
  -> SUMMARIZE STATE / OPEN QUESTIONS
  -> GENERATE AND FILTER COGNITIVE OPERATORS
  -> BUILD RETRIEVAL CUES
  -> RETRIEVE / SCORE / PRESERVE AMBIGUITY
  -> ESTIMATE OPERATOR UTILITY, COST AND RISK
  -> SELECT ACTION OR SAFE PARALLEL SET
  -> CREATE BOUNDED SUBPROBLEMS IF NEEDED
  -> ROUTE SYMBOLIC / LEARNED / HYBRID PROCESSING
  -> RESERVE RESOURCES AND DELEGATE TO EXISTING COORDINATORS
  -> EXECUTE WITH CHECKPOINTS
  -> VALIDATE CONTRACTS AND EVIDENCE
  -> INTEGRATE THROUGH AUTHORITATIVE WORKSPACE
  -> RECOMPUTE GOAL PROGRESS AND CRITICAL GATES
  -> DETECT CYCLES / STAGNATION
  -> CONTINUE | REVISE | WAIT | STOP
  -> AFTER VERIFIED SUCCESS: TEST AND REGISTER REUSABLE PROCEDURE
```

## 10. Recommendation

The highest-value first deliverable is a **deterministic ORE Cognitive Control Layer** that wraps—not replaces—the existing workspace, task coordinator and resource coordinator. Soar contributes the executive cycle and explicit operators; ACT-R contributes principled retrieval competition and cost awareness; CLARION contributes a future-proof boundary between symbolic and learned processing. Procedure learning should remain the final, most conservative stage.

This specification is an architectural algorithm proposal. It is not evidence that these mechanisms have been implemented or that the repository currently passes the proposed acceptance tests.
