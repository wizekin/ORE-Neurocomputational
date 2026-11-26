# Neuro-Symbolic Reasoning Layer — Implementation of Point 3 (45-step NSRL)

Status: **implemented and tested** against the proposed specification in
`ORE —Neurocomputational Point 03 Neuro-Symbolic Reasoning Algorithm.txt`.
Software correctness is covered by the test suite listed below; scientific
reasoning quality is **not** claimed — that requires independent labeled
evaluation on real corpora (Section 10 of the specification).

## 1. Design summary

The additive `ore_neurosymbolic` package turns structured USG knowledge into
explainable inference through one cooperating unit. It combines:

1. deterministic symbolic inference over typed concepts, objects, predicates,
   relations and rules (`symbolic_engine.py`, `unification.py`,
   `rule_validation.py`, `inference_queue.py`);
2. proof construction and independent verification (`proof_model.py`,
   `proof_builder.py`, `proof_verifier.py`);
3. hard/soft constraint reasoning for ambiguity resolution
   (`constraint_engine.py`, `ambiguity_resolver.py`);
4. probabilistic evidence handling that keeps uncertainty distinct from facts
   (`probabilistic_adapter.py`);
5. replaceable learned components for similarity, candidate retrieval and
   ranking (`learned_candidate_adapter.py`).

Non-negotiable invariants are enforced structurally:

- observations, hypotheses, model scores, probabilistic beliefs, strict facts
  and derived conclusions are separate `EpistemicStatus` values (`contracts.py`);
- learned models propose candidates through `ModelProposal`; they cannot write
  authoritative facts and cannot mutate USG (read-only adapter);
- every conclusion carries premises, rule ID/version, source lineage, the
  workspace revision and the inference-run ID (`ReasoningFinding`,
  `LineageRecord`, `trace.py`);
- unknown is a first-class status distinct from negation; missing evidence is
  never stored as negative evidence (`FactTable.has_positive` vs
  `has_negated`, negated premises require explicit negative facts);
- contradictions are recorded as `NSRLConflictRecord`s, never silently
  overwritten (`conflict_manager.py`);
- defeasible rules carry scope, priority and exceptions; specificity resolves
  before priority and incomparable pairs produce conflict records
  (`SymbolicEngine._resolve_defeasible_competition`);
- rule fires, recursion, depth, candidate count and wall-clock time are hard
  budgets (`EngineBudgets`, `ResourceBudgets`, `ResourcePolicy`);
- workspace, coordinator, publisher and logger facilities are reused, not
  duplicated: publication flows through `FindingPublisher` →
  `WorkspaceStateManager` → atomic commit → post-commit events; events flow
  through `ore_logger.LoggerDatabase`;
- no external ML/probabilistic package is required: the deterministic core
  runs on the standard library alone.

## 2. Step-by-step implementation map

Operation IDs `NSR-001..NSR-045` (`registry.py`) map one-to-one to the 45
ordered steps; the coordinator records one trace record per step with
`operation_id`, `step`, `phase`, `title` and step-specific details, and the
result exposes `operation_coverage`.

| Steps | Phase | Where implemented |
| --- | --- | --- |
| 1-2 | A request/snapshot | `coordinator.run` (validation, idempotent replay, one `SharedWorkspace`, one bound revision) |
| 3-6 | A resolution/normalization | `semantic_adapter.py` (`resolve_references`, `retrieve_facts`, `normalize_fact`, `assign_epistemic_status`) |
| 7 | A bounded context | `context.py` (`build_bounded_context`) |
| 8 | B rule validation | `rule_validation.py` (`validate_ruleset`, safe `RuleProfile`) |
| 9-16 | B inference | `symbolic_engine.py` (`select_relevant_rules`, typed joins, `_evaluate_conjunction`, `_try_fire`, `_resolve_defeasible_competition`, `forward_chain`, `backward_chain`, cycle events) |
| 17-21 | C proofs | `proof_builder.py` + `proof_model.py` (`ProofDAG.add_node`, `validate`, `verify_integrity`) + `proof_verifier.py` (`IndependentProofVerifier.verify_graph`) |
| 22-25 | D constraints/ambiguity | `constraint_engine.py` (`apply_hard`, `apply_soft`) + `ambiguity_resolver.py` (`generate_candidates`, `rank`, `preserve_ambiguity`) |
| 26-30 | E probabilistic | `probabilistic_adapter.py` (`wrap_model_evidence`, `separate_uncertain_from_crisp`, `combine_evidence`, `build_confidence_vector`) |
| 31-35 | F learned | `learned_candidate_adapter.py` (`invoke_when_needed`, `_build_proposal`, `to_hypotheses`, `filter_symbolically`, `record_verified_feedback`) |
| 36-37 | G conflicts | `conflict_manager.py` (`detect_contradictions`, `record_conflict`) |
| 38-40 | G publication | `workspace_publisher.py` (`apply_gates`, `publish`, `request_independent_review`) |
| 41-45 | H observability | `observability.py` (`record_event`, `measure`, `compare_with_verified_outcomes`, `record_parameter_transaction`, `checkpoint`) |

### Mathematical operations

| Spec | Implementation | Test |
| --- | --- | --- |
| 6.1 strict firing gate | `SymbolicEngine._evaluate_conjunction` (logical gate, never a probability) | `test_neurosymbolic_symbolic.py` |
| 6.2 fixed point `F_{k+1}=F_k∪T_R(F_k)` | `forward_chain` with deduplicating queue, stop on fixed point/goal/budget | `test_neurosymbolic_symbolic.py` |
| 6.3 candidate ranking `S(c)` | `AmbiguityResolver.score`/`rank` with versioned `RankingWeights` | `test_neurosymbolic_reasoning.py` |
| 6.4 `P(H|e_{1:n}) ∝ P(H)∏P(e_i|H)` | `combine_evidence` — multiplies only under declared conditional independence; dependent groups use declared joint likelihood; undeclared dependencies stay qualitative | `test_neurosymbolic_reasoning.py` |
| 6.5 confidence decomposition | `ConfidenceVector` — five dimensions, never collapsed | `test_neurosymbolic_reasoning.py` |
| 6.6 LTN-like consistency (optional) | `differentiable_consistency` — soft optimization signal, never a proof, never gates publication | `test_neurosymbolic_reasoning.py` |
| 6.7 `U_continue` | `ResourcePolicy.continuation_utility`/`evaluate_continuation` (hard budgets checked first) | `test_neurosymbolic_reasoning.py` |
| 6.8 proof-node integrity | `ProofNode.compute_content_hash`, `ProofDAG.verify_integrity` | `test_neurosymbolic_proofs.py` |

## 3. Acceptance-test coverage (Section 10)

| Acceptance area | Tests |
| --- | --- |
| Contract serialization/versioning, invalid-input rejection | `test_neurosymbolic_contracts.py::TestContractValidation`, `::TestContractSerialization` |
| Rule arity, typing, unbound variables, unsafe recursion, exceptions | `test_neurosymbolic_contracts.py::TestRuleValidation` |
| Unification exact matches, substitutions, repeated variables, type mismatch | `test_neurosymbolic_contracts.py::TestUnification` |
| Multi-hop/conjunctive inference, fixed point, no unsupported conclusion, determinism | `test_neurosymbolic_symbolic.py::TestForwardChaining` |
| Defeasible exception precedence, unresolved incomparable conflicts | `test_neurosymbolic_symbolic.py::TestDefeasibleSemantics` |
| Explicit negation vs absence/unknown | `test_neurosymbolic_symbolic.py::TestNegationVsUnknown` |
| Cycles, depth, rule-fire, queue and wall-time limits | `test_neurosymbolic_symbolic.py::TestBoundedInference` |
| Proof graph integrity; verifier catches tampered rule/substitution/premise/conclusion | `test_neurosymbolic_proofs.py` |
| Full source/rule-version lineage | `test_neurosymbolic_proofs.py::TestProofBuilder::test_lineage_traces_to_source_facts` |
| Ambiguity preserved; hard constraints filter only proven-invalid; soft ranking reproducible | `test_neurosymbolic_reasoning.py::TestConstraintEngine`, `::TestAmbiguityResolver` |
| Probabilistic toy distributions, correlated evidence, calibration, uncertainty propagation | `test_neurosymbolic_reasoning.py::TestProbabilisticAdapter` |
| Malformed/unavailable model rejection; no direct model mutation of USG | `test_neurosymbolic_reasoning.py::TestLearnedAdapter` |
| Workspace snapshot consistency, atomic publication, conflict/revision handling | `test_neurosymbolic_reasoning.py::TestCoordinatorEndToEnd` |
| Logger annotations/events correlate run, proof and finding | `test_neurosymbolic_reasoning.py::TestObservability::test_logger_events_correlate_run` |
| Timeout/partial-run recovery without duplicate publication | `test_neurosymbolic_reasoning.py::test_partial_status_on_tight_budget` |
| Benchmarks on synthetic corpora with precision/recall, proof validity, latency, memory, overhead | `test_neurosymbolic_reasoning.py::TestEvaluationHarness` (`evaluation.py`) |
| Existing ORE/USG/workspace/coordinator regression suite remains green | full-suite run (1490+ tests) |
| Software correctness separated from scientific accuracy | no quality-gain claims anywhere; `evaluate_corpus` notes state the separation |

## 4. Running

```bash
# focused NSRL tests
python -m pytest tests/test_neurosymbolic_contracts.py \
                  tests/test_neurosymbolic_symbolic.py \
                  tests/test_neurosymbolic_proofs.py \
                  tests/test_neurosymbolic_reasoning.py -q

# runnable demonstration (no GPU, no learned model required)
python examples/neurosymbolic_reasoning.py
```

Minimal usage:

```python
from ore_neurosymbolic import (
    NeuroSymbolicReasoningCoordinator, ReasoningRequest, SemanticFact,
    RuleDefinition, EpistemicStatus,
)

request = ReasoningRequest(
    goal_predicate="indirect_command",
    goal_args=("captain", "fleet"),
    facts=(
        SemanticFact(fact_id="f1", subject="captain", predicate="commanded_by",
                     object_value="admiral", epistemic_status=EpistemicStatus.ASSERTED),
        SemanticFact(fact_id="f2", subject="admiral", predicate="commands",
                     object_value="fleet", epistemic_status=EpistemicStatus.ASSERTED),
    ),
    rules=(my_ruleset,),          # tuple[RuleDefinition], versioned
)
result = NeuroSymbolicReasoningCoordinator().run(request)
print(result.summary_line())      # status, counts, operation coverage
```

## 5. Honest limitations

- The engine's fact store is triple-based; the safe rule profile therefore
  accepts binary literals only. Ternary relations must be reified.
- Backward chaining is SLD-style with depth limits and grounded-goal
  memoization; it is complete for the bounded safe profile, not a general
  theorem prover.
- Probabilistic combination requires explicitly declared priors and
  independence declarations; without them the adapter reports qualitative
  support instead of numbers (by design).
- The learned-component registry ships no model; registered backends are
  caller-supplied and validated at the boundary. The demo linker is a toy.
- Benchmarks run on synthetic corpora generated inside this repository; they
  measure the reasoning mechanism, not scientific reasoning quality.
- The Point 3 document is a proposed specification; this implementation is
  the first concrete realization and may deviate where the repository's
  native conventions require it (documented above).
- The one pre-existing environmental failure in the wider suite
  (`tests/test_ore_logger_runtime.py::test_sitecustomize_automatically_logs_a_launched_repository_script`,
  a subprocess/sitecustomize instrumentation probe) is unrelated to this
  package and fails identically without it.
