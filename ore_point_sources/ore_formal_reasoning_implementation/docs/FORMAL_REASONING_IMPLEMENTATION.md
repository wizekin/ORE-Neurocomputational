# FRCE — Formal Reasoning and Constraint Engine (Point 5) Implementation

This document maps the 52-step integrated algorithm of **"ORE Integrated
Algorithm — Point 5: Automated Reasoning, Theorem Proving, SAT/SMT Solvers and
Constraint-Based Intelligence"** onto the `ore_formal_reasoning` package and
lists the acceptance-test coverage. The engine is the *specialist reasoning
service* of the specification (Section 2): it consumes read-only typed USG
knowledge, produces typed results with distinct epistemic categories, and
publishes them into the existing ORE shared cognitive workspace through the
controlled publication path.

## 1. Package layout

| Module | Specification role | Steps |
| --- | --- | --- |
| `contracts.py` | Request/result contracts, `ResultStatus` vocabulary, evidence tags, six epistemic result kinds, budgets, named assumptions, temporal scopes | 1, 3, 5, 6, 42 |
| `ir.py` | Typed logical IR: sorts, terms, formulas, quantifiers, cardinality, rules, units and conversions, per-domain semantic profiles, normalizer/type-checker | 8–15 |
| `usg_adapter.py` | Read-only USG mapping: typed records, entity resolution with preserved ambiguity, smallest-sufficient retrieval, source-to-IR traceability, write guard | 3, 4, 8 |
| `intake.py` | Request intake: operationalizability, task classification, fact/assumption separation, provenance ledger, completeness/ambiguity gate | 1–2, 5–7 |
| `datalog.py` | Dependency graph, stratification (Tarjan SCC), indexed semi-naive fixed point with delta iteration, naive reference evaluator, explanation graph, content-addressed cache | 16–20 |
| `consistency.py` | Conflict detection and classification (logical / source / temporal / scope / ambiguity), non-destructive policy | 21–22 |
| `smt.py` | Theory selection and gating, Z3 encoding with bidirectional provenance map, named assumptions, deterministic configuration, model decoding with an independent evaluator, unsat cores with optional greedy minimization, entailment by counterexample, numeric-semantics guards | 23–30 |
| `cpsat.py` | Module-selection contracts, OR-Tools CP-SAT adapter, exact pure-Python reference optimizer, independent feasibility validation, honest optimality reporting | 31–36 |
| `proofs.py` | Proof obligations, proof terms, trusted propositional natural-deduction kernel (Curry-Howard checker), untrusted proof search, Lean export/check interface | 37–41 |
| `envelope.py` | Unified result envelope validation and certificate replay | 42–43 |
| `governance.py` | Workspace publication of typed entries, read-only USG boundary, promotion workflow, SQLite audit records with secret redaction | 44–46 |
| `policy.py` | Resource budgets, bounded fork workers (RLIMIT_CPU/RLIMIT_AS + wall clock + output cap), failure taxonomy, bounded-backoff recovery, capability flags | 47–48, 52 |
| `evaluation.py` | Versioned benchmark suite, separate correctness/operational metrics, failure classification that never loosens guarantees, UNKNOWN-coercion regression guard | 49–51 |
| `coordinator.py` | The 52-step orchestration, run records with per-step traces (`FRC-001..FRC-052`), idempotent replay, service facade | all |
| `bridge.py` | Blackboard-specialist adapter (Point 1) and FRCE contract registration | §7 |
| `registry.py` | Stable operation IDs `FRC-001..FRC-052` with phases A–H | all |

## 2. The 52 steps in brief

- **Phase A (1–7) intake.** Requests are fingerprinted (idempotency key), routed
  into task classes without calling everything theorem proving, USG entities
  are resolved with ambiguity preserved (multiple matches block execution until
  a discriminator arrives), the smallest sufficient fact/rule set is retrieved
  with recorded query parameters, evidence is tagged fact/assumption/
  hypothesis/policy, provenance and temporal scopes are preserved, and
  incomplete or ambiguous requests return a structured "insufficient
  formalization" result — no solver is invoked and nothing is reported false.
- **Phase B (8–15) typed formalization.** USG records map to constants,
  predicates, typed statements and named rules. Sorts declare finiteness;
  predicates declare arity; quantifiers are explicit; units convert only
  through validated rules; every statement belongs to a declared per-domain
  semantic profile (open/closed world is never global by accident). The
  normalizer alpha-renames, folds only provably safe constants, type-checks
  every term (rejecting identifier-as-number coercion and mixed Int/Real
  arithmetic without explicit `ToReal`), and keeps the statement→source map.
- **Phase C (16–22) rule inference and conflicts.** Range-restricted,
  stratified Datalog evaluates non-recursive rules and recursive fixed points
  with indexed semi-naive delta iteration, cross-checked against a naive
  reference evaluator. Every derived fact carries rule ID, substitution, and
  parent-fact provenance; multiple independent derivations are preserved.
  Conflicts are classified and both sides are kept — destructive repairs are
  structurally zero.
- **Phase D (23–30) SAT/SMT.** Theory needs are detected and gated against the
  permitted-theory allowlist before any solve. Z3 runs with pinned random seed,
  timeout, and memory budget. Models are decoded and then **independently
  re-evaluated** against the IR with exact rational arithmetic and wrapping
  bit-vector semantics. Unsatisfiable cores map back to named inputs and are
  labeled "sufficient, not minimum" unless minimization actually ran.
  Entailment checks Γ ∧ ¬φ: UNSAT establishes entailment, SAT yields an
  independently validated countermodel, UNKNOWN/timeout yield "not
  established".
- **Phase E (31–36) optimization.** Module selection encodes coverage
  (Σ x_i ≥ 1 per capability), budgets (Σ r_ik x_i ≤ R_k), dependencies
  (x_a ≤ x_b), exclusivity (x_a + x_b ≤ 1), optional deadlines, and a
  transparent weighted objective with recorded weights and units. CP-SAT
  statuses are preserved exactly; the reference optimizer proves optimality by
  exhaustive search for small problems; every selection is re-validated
  against the original constraints before reporting.
- **Phase F (37–41) proofs.** Obligations are generated only for
  high-assurance claims and must be grounded in declared premises. Proof
  candidates from the bounded searcher are untrusted until the bundled
  propositional natural-deduction kernel accepts them; with a pinned Lean
  toolchain the same artifact is exported and checked externally. Artifacts
  record the checker identity/version, artifact hash, dependencies, and the
  premises they are conditional on. Proof-search failure is "not proved",
  never refutation.
- **Phase G (42–48) governance.** All outputs normalize into typed envelopes
  that are validated against the versioned schema (proof statuses, model
  validation flags, non-minimized core labeling, solver identity, provenance,
  reproducibility). Publications use distinct finding types per epistemic
  category and pass through the workspace permission policy; derived facts are
  proposals requiring configured approval; the USG adapter has no write path.
  Every run persists a replayable SQLite audit record with secret redaction.
  Solvers run under budgets; with process isolation they get RLIMIT
  CPU/memory, wall-clock termination, and output caps; failures are classified
  and retried only for transient classes with bounded backoff.
- **Phase H (49–52) evaluation.** The versioned benchmark suite (`FRCE-BENCH/1.0.0`)
  covers Boolean SAT/UNSAT, arithmetic, recursive Datalog, contradictions,
  proof obligations, CP-SAT selection, ambiguous formalizations, and invalid
  encodings; metrics report formalization validity, proof-check rate, model
  validity, detection vs false positives, optimization feasibility/gap, and
  reproducibility separately. Failures are classified for formalization
  improvements; a regression guard proves UNKNOWN/timeout/unsupported never
  coerce to TRUE/FALSE. Capabilities are gated behind acceptance-suite flags.

## 3. Epistemic discipline (Section 1 of the specification)

`FormalResult.kind` keeps six categories separate — retrieved source fact,
rule-derived inference, satisfiability result, optimization result,
machine-checked proof, empirical observation — and `FormalResult.status` uses
the exhaustive vocabulary `proved`, `refuted-under-model`, `satisfiable`,
`unsatisfiable`, `optimal`, `feasible`, `infeasible`, `unknown`, `timeout`,
`unsupported`, `invalid-input`, `error`. The structural invariants make the
distinctions non-optional: a PROVED result must carry `proof_check_status =
"kernel_checked"`, OPTIMAL is reserved for optimization results, a decoded
model must record whether it was independently validated, and the five
non-success statuses can never establish falsity.

## 4. Solver-selection policy (Section 6)

| Need | Mechanism in this implementation | Output preserved |
| --- | --- | --- |
| Recursive relational inference over USG | `datalog.DatalogEngine` | Derived facts with rule/parent provenance, strata |
| Boolean consistency | `smt.SmtAdapter` (Z3) | SAT/UNSAT, model or core |
| Arithmetic, bit-vectors, strings | `smt.SmtAdapter` with theory gating | Status, model/core, theory profile |
| Scheduling and discrete allocation | `cpsat.CpsatOptimizer` (OR-Tools) with `ReferenceOptimizer` fallback | Feasible/optimal status, bound, gap |
| High-assurance proof | `proofs.TrustedKernel` (+ `LeanChecker` when a toolchain is pinned) | Kernel-checked artifact, hashes, dependencies |

cvc5, dedicated SAT backends, and full first-order theorem provers are
deliberately *not* integrated: the specification's roadmap starts with one
SMT backend and adds others only when benchmarks demonstrate a concrete
advantage.

## 5. ORE integration contracts (Section 7)

- **USG adapter** — read-only over `TypedKnowledgeRecord` snapshots (or the
  `ore_usg` resources when a bundle is available); untyped or ambiguous
  mappings are rejected; every IR statement carries its record reference.
- **Coordinator** — `FormalReasoningCoordinator` decomposes mixed tasks into
  inference, consistency, optimization, and proof obligations, and returns a
  `FormalReasoningRun` whose step trace gives the coordinator decision-ready
  diagnostics.
- **Shared workspace** — publications flow through `ore_workspace`'s
  `FindingPublisher` and `PermissionPolicy` with distinct finding types
  (`frce_inferred_fact`, `frce_satisfiability_result`,
  `frce_optimization_proposal`, `frce_proof_artifact`, `frce_conflict`,
  `frce_counterexample`, `frce_assumption_set`).
- **AdaptiveResourceCoordinator / CROS** — optimization proposals are returned
  as *proposals* with bound/gap; feasibility and authorization stay with the
  resource coordinator.
- **SQLite logger** — `governance.SQLiteAuditLog` stores replayable execution
  records (request ID, IR hash, inputs, assumptions, solver versions,
  configurations, resource use, status, artifact hashes, trace) with secret
  redaction; the `ore_logger` runtime continues to trace Python-level calls.

The blackboard bridge (`bridge.FormalReasoningSpecialist`) registers the FRCE
contracts with the shared catalog and exposes the engine as a Point-1
specialist returning a validated finding-batch artifact.

## 6. Acceptance-test coverage (Section 9)

`tests/test_formal_reasoning_contracts.py` (35 tests), 
`tests/test_formal_reasoning_engines.py` (46 tests), and
`tests/test_formal_reasoning_evaluation.py` (29 tests) — **110 tests total** —
cover every acceptance test in the specification's Section 9, including:

- SAT with an independently validated model; UNSAT with named cores;
  timeout/UNKNOWN never UNSAT; unsupported theories gated before solving;
  entailment via Γ ∧ ¬φ with decoded, validated countermodels;
- non-minimized cores labeled sufficient-not-minimal; optional greedy
  minimization only when actually performed;
- strict Int/Real separation; literal-zero division rejected pre-solve;
  bit-vector wrapping semantics; string operators routed only to a permitted
  string theory; USG IDs traceable after normalization; ambiguous resolution
  blocks execution; missing quantifier scope detected; unit mismatch rejected
  before solver invocation; open-world absence never converted to false;
  closed-world only where configured;
- transitive Datalog closure, finite fixed points, semi-naive ≡ reference,
  rule+parent provenance, preserved multiple derivations, temporal
  non-conflict, stratification enforcement, unsafe-variable and
  function-symbol rejection, rule-change cache invalidation;
- CP-SAT coverage/budgets/dependencies/exclusivity, INFEASIBLE for impossible
  problems, FEASIBLE incumbents never labeled OPTIMAL, weights and units
  recorded;
- valid proofs accepted and invalid proofs rejected by the trusted kernel,
  full artifact metadata, conditional-on disclosure for unvalidated premises,
  search failure ≠ falsity;
- envelope schema validation, solver identity requirements, workspace
  publication type separation, read-only USG enforcement, promotion approval
  workflow, complete SQLite audit records, crash containment, wall-clock
  enforcement, output caps, file/network blocking, bounded retries,
  capability flags promoted only after their suites pass, UNKNOWN never
  coerced, deterministic replay equivalence, and the blackboard bridge
  round trip.

Run them with:

```bash
python -m pytest tests/test_formal_reasoning_contracts.py \
                  tests/test_formal_reasoning_engines.py \
                  tests/test_formal_reasoning_evaluation.py
```

The end-to-end demonstration is `examples/formal_reasoning.py`:
`python examples/formal_reasoning.py`.

## 7. Environment notes and honest limitations

- **Z3** (`z3-solver`) and **OR-Tools** (`ortools`) are used through adapters
  and are declared as optional extras (`frce` extra in `pyproject.toml`); the
  reference optimizer and capability flags keep the package functional without
  them, and missing backends produce `unsupported`/`unknown` outcomes, never
  guesses.
- **Lean** is not pinned in this environment. The trusted boundary is the
  bundled propositional natural-deduction kernel (a deterministic Curry-Howard
  checker, ~150 rules of implication/conjunction/disjunction/bottom
  elimination). When a `lean` binary is on the PATH and allowlisted, artifacts
  are exported and checked externally; otherwise the Lean path reports
  `unsupported` and the kernel fallback is disclosed in every artifact. The
  kernel establishes derivability within propositional ND only — it is not a
  substitute for Lean's dependent-type kernel over full mathematics.
- **Quantification** in SMT encodings requires finite sorts (enumerations or
  bounded finite integer ranges); infinite quantification is reported as
  unsupported rather than guessed.
- **Isolated workers** exchange JSON-safe payloads; non-serializable solver
  outputs are recorded via `repr` with an explicit note. Process isolation is
  opt-in through the `process_isolation` capability flag.
- The pre-existing upstream failure
  `tests/test_ore_logger_runtime.py::test_sitecustomize_automatically_logs_a_launched_repository_script`
  fails identically on the clean upstream tree in this environment and is
  unrelated to this package.
