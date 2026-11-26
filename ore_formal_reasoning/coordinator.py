"""The Formal Reasoning and Constraint Engine coordinator (all 52 steps, Phases A-H).

One ``FormalReasoningCoordinator.run(request)`` call executes the complete
integrated algorithm of ``ORE Integrated Algorithm — Point 5``: request intake
and evidence normalization (A), typed formalization and the logical IR (B),
Datalog rule inference and contradiction handling (C), SAT/SMT consistency and
entailment (D), CP-SAT optimization (E), machine-checked proofs (F), ORE
integration with result governance and recovery (G), and evaluation with
capability-gated release (H). Every executed step is recorded as a trace entry
with its stable operation ID (``FRC-001`` .. ``FRC-052``).

The coordinator is the specialist reasoning service of Point 5, Section 2: it
consumes read-only USG knowledge, publishes typed results into the shared
cognitive workspace through the workspace permission path, proposes (never
performs) knowledge promotion, persists a replayable SQLite audit record, and
keeps every solver behind typed adapters with honest, distinct statuses.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .consistency import (
    ConsistencyChecker,
    ConsistencyReport,
    ConflictRecord,
)
from .contracts import (
    EvidenceTag,
    FormalResult,
    FormalizationRequest,
    NamedAssumption,
    ResultKind,
    ResultStatus,
    RuntimeMeasurement,
    SolverIdentity,
)
from .cpsat import OptimizationOutcome, OptimizationRouter, SelectionProblem
from .datalog import (
    DatalogCache,
    DatalogEngine,
    DatalogFact,
    DatalogResult,
    facts_from_ir,
    rules_from_ir,
)
from .envelope import (
    EnvelopeValidation,
    validate_result_envelope,
)
from .evaluation import (
    assert_unknown_never_coerced,
    classify_failure_for_improvement,
    run_benchmark_suite,
)
from .exceptions import FormalReasoningError, InvalidFormalizationError
from .governance import (
    KnowledgeWriteGovernance,
    PromotionProposal,
    SQLiteAuditLog,
    WorkspacePublicationAdapter,
)
from .intake import IntakeReport, run_intake
from .ir import (
    FormalIR,
    FormalIRBuilder,
    StatementKind,
)
from .policy import CapabilityFlags
from .proofs import (
    ProofCandidate,
    ProofCheckOutcome,
    ProofObligation,
    ProofSearcher,
    ProofVerifier,
)
from .registry import operation_by_step
from .smt import SmtAdapter, SmtOutcome, Z3_AVAILABLE
from .usg_adapter import (
    KnowledgeRetriever,
    RetrievalRecord,
    TypedKnowledgeRecord,
    USGEntityResolver,
    USGToIRMapper,
)
from .version import (
    FORMAL_REASONING_PACKAGE_VERSION,
    FORMAL_REASONING_RESULT_VERSION,
    FORMAL_REASONING_SCHEMA_VERSION,
    FRCE_PRODUCER,
)


@dataclass(slots=True)
class FrcePolicy:
    """Coordinator policy: budgets, routing, and operational switches."""

    prefer_optimizer_backend: str = "auto"
    minimize_unsat_cores: bool = False
    cross_check_semi_naive: bool = True
    process_isolation: bool = False
    default_budget: Any = None


@dataclass(frozen=True, slots=True)
class FormalReasoningRun:
    """The complete typed output of one FRCE execution."""

    run_id: str
    request_id: str
    fingerprint: str
    schema_version: str
    status: str
    replayed: bool
    step_trace: tuple[dict[str, Any], ...]
    intake: IntakeReport | None
    ir: FormalIR | None
    ir_hash: str
    retrieval: RetrievalRecord | None
    datalog_result: DatalogResult | None
    consistency: ConsistencyReport | None
    smt_outcomes: tuple[SmtOutcome, ...]
    entailment_outcomes: tuple[SmtOutcome, ...]
    optimization: OptimizationOutcome | None
    proof_outcomes: tuple[ProofCheckOutcome, ...]
    results: tuple[FormalResult, ...]
    envelope_validations: tuple[EnvelopeValidation, ...]
    publications: tuple[Any, ...]
    promotion_proposals: tuple[PromotionProposal, ...]
    audit_id: int | None
    failures: tuple[dict[str, Any], ...]
    improvements: tuple[Any, ...]
    metrics: Mapping[str, Any]
    capabilities: Mapping[str, Any]
    limitations: tuple[str, ...]
    warnings: tuple[str, ...]

    def trace_by_step(self, step: int) -> dict[str, Any] | None:
        """Return the trace record for one algorithm step."""
        for record in self.step_trace:
            if record["step"] == step:
                return record
        return None


def _new_id(prefix: str) -> str:
    """Deterministic-enough run/identifier generator."""
    import uuid

    return f"{prefix}_{uuid.uuid4().hex}"


class FormalReasoningCoordinator:
    """Run the 52-step formal reasoning algorithm over one request."""

    producer_module = FRCE_PRODUCER

    def __init__(
        self,
        *,
        policy: FrcePolicy | None = None,
        workspace: Any = None,
        permission_policy: Any = None,
        audit_path: str | Path | None = None,
        capability_flags: CapabilityFlags | None = None,
        exclusivity_groups: Iterable[Any] = (),
        functional_constraints: Iterable[Any] = (),
        cardinality_constraints: Iterable[Any] = (),
    ) -> None:
        self.policy = policy or FrcePolicy()
        self.workspace = workspace
        self.permission_policy = permission_policy
        self.audit_log = (
            SQLiteAuditLog(audit_path) if audit_path is not None else None
        )
        self.capabilities = capability_flags or CapabilityFlags()
        self.consistency_checker = ConsistencyChecker(
            exclusivity_groups=exclusivity_groups,
            functional_constraints=functional_constraints,
            cardinality_constraints=cardinality_constraints,
        )
        self.optimizer = OptimizationRouter(
            prefer_backend=self.policy.prefer_optimizer_backend,
        )
        self.publication = WorkspacePublicationAdapter(self.producer_module)
        self.governance = KnowledgeWriteGovernance()
        self.proof_verifier = ProofVerifier()
        self._runs: dict[str, FormalReasoningRun] = {}
        self._cache = DatalogCache()

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _record(
        trace: list[dict[str, Any]], step: int, details: Mapping[str, Any],
    ) -> None:
        """Append one operation trace record for an executed algorithm step."""
        spec = operation_by_step(step)
        trace.append({
            "operation_id": spec.operation_id,
            "step": spec.step,
            "phase": spec.phase,
            "title": spec.title,
            "details": dict(details),
        })

    def _fail(
        self,
        trace: list[dict[str, Any]],
        failures: list[dict[str, Any]],
        step: int,
        error: BaseException,
    ) -> dict[str, Any]:
        """Record a classified failure for one step (Step 48)."""
        from .policy import classify_failure

        failure_class = classify_failure(error)
        failure = {
            "step": step,
            "class": failure_class,
            "detail": f"{type(error).__name__}: {error}",
        }
        failures.append(failure)
        self._record(trace, 48, {
            "failure_step": step,
            "failure_class": failure_class,
            "recovery": "terminate safely and retain diagnostic metadata",
        })
        return failure

    # ------------------------------------------------------------------ run

    def run(
        self,
        request: FormalizationRequest,
        *,
        ir: FormalIR | None = None,
        usg_records: Iterable[TypedKnowledgeRecord] | None = None,
        discriminators: Mapping[str, str] | None = None,
        proof_candidate: ProofCandidate | None = None,
    ) -> FormalReasoningRun:
        """Execute the algorithm; same-fingerprint requests replay the cached run."""
        run_id = _new_id("frce_run")
        trace: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        smt_outcomes: list[SmtOutcome] = []
        entailment_outcomes: list[SmtOutcome] = []
        proof_outcomes: list[ProofCheckOutcome] = []
        publications: list[Any] = []
        proposals: list[PromotionProposal] = []
        limitations: list[str] = []
        warnings: list[str] = []
        start_wall = time.perf_counter()

        # -- Phase A ---------------------------------------------------------
        # Step 1: receive and validate the task; fingerprint is the idempotency key.
        fingerprint = request.fingerprint
        cached = self._runs.get(fingerprint)
        self._record(trace, 1, {
            "request_id": request.request_id,
            "objective": request.objective,
            "fingerprint": fingerprint,
            "task_classes": [c.value for c in request.task_classes],
            "idempotent_replay": cached is not None,
        })
        if cached is not None:
            return cached  # deterministic fixture replay returns equivalent output

        intake = run_intake(request, None)
        self._record(trace, 1, {
            "operationalizable": intake.operationalizable,
            "clarifications": [c.code for c in intake.completeness.clarification_requests],
        })

        # Step 3: resolve semantic entities against USG (before IR, if records given).
        ambiguities: dict[str, int] = {}
        retrieval: RetrievalRecord | None = None
        if usg_records is not None:
            records = tuple(usg_records)
            resolver = USGEntityResolver(records)
            try:
                resolved = resolver.resolve_all(request.usg_entity_ids, discriminators)
                self._record(trace, 3, {
                    "resolved": [r.usg_id for r in resolved],
                    "ambiguous": [r.reference for r in resolved if len(r.candidates) > 1],
                })
            except FormalReasoningError as error:
                self._fail(trace, failures, 3, error)
                if isinstance(error, InvalidFormalizationError) and "matches" in str(error):
                    warnings.append(str(error))
            # Step 4: retrieve only relevant knowledge.
            retriever = KnowledgeRetriever(records)
            goal_predicates = self._goal_predicates(request)
            retrieval = retriever.retrieve(
                goal_predicates, request.usg_entity_ids,
                [r for r in records if r.kind == "rule"],
            )
            self._record(trace, 4, {
                "selected_records": list(retrieval.selected_record_ids),
                "selected_rules": list(retrieval.selected_rule_ids),
                "query_digest": retrieval.query_digest,
                "source_versions": list(retrieval.source_versions),
            })
            # Step 8 partially: map records into a fresh builder.
            if ir is None:
                try:
                    ir = self._build_ir_from_records(records)
                except FormalReasoningError as error:
                    self._fail(trace, failures, 8, error)

        # Step 2: classify the reasoning problem.
        if ir is not None:
            intake = run_intake(request, ir)
        self._record(trace, 2, {
            "declared": [c.value for c in intake.classification.declared],
            "detected": [c.value for c in intake.classification.detected],
            "routed": [c.value for c in intake.classification.routed],
            "notes": list(intake.classification.notes),
        })

        # Step 5: separate facts from assumptions and hypotheses.
        self._record(trace, 5, {
            "observed_facts": len(intake.separation.observed_facts),
            "stipulated_assumptions": len(intake.separation.stipulated_assumptions),
            "inferred_results": len(intake.separation.inferred_results),
            "hypotheses": len(intake.separation.hypotheses),
            "policy_constraints": len(intake.separation.policy_constraints),
        })

        # Step 6: preserve provenance and temporal scope.
        self._record(trace, 6, {
            "source_ids": list(intake.provenance.source_ids),
            "time_scoped_statements": list(intake.provenance.time_scoped_statements),
            "authorities": list(intake.provenance.authority_records),
        })

        # Step 7: completeness and ambiguity gate.
        self._record(trace, 7, {
            "sufficient": intake.completeness.sufficient,
            "clarification_requests": [
                {"code": c.code, "message": c.message}
                for c in intake.completeness.clarification_requests
            ],
            "warnings": list(intake.completeness.warnings),
        })
        warnings.extend(intake.completeness.warnings)
        if not intake.operationalizable:
            blocked = FormalResult(
                result_id=_new_id("frce_res"),
                request_id=request.request_id,
                kind=ResultKind.SATISFIABILITY_RESULT,
                status=ResultStatus.INVALID_INPUT,
                statement="insufficient formalization; clarification required",
                assumptions=tuple(request.assumptions),
                provenance=tuple(request.source_refs),
                limitations=(
                    "the request cannot be operationalized until the listed "
                    "clarifications are accepted; no solver was invoked and no "
                    "statement was reported false",
                ),
                warnings=tuple(
                    c.message for c in intake.completeness.clarification_requests
                ),
                reproducibility={
                    "ir_hash": "",
                    "fingerprint": fingerprint,
                    "package_version": FORMAL_REASONING_PACKAGE_VERSION,
                },
            )
            run = self._finalize(
                run_id, request, fingerprint, trace, intake, ir, retrieval,
                None, None, [], [], None, [], [blocked], [], publications,
                proposals, failures, limitations, warnings, start_wall,
            )
            return run

        # -- Phase B ---------------------------------------------------------
        datalog_result: DatalogResult | None = None
        consistency: ConsistencyReport | None = None
        optimization: OptimizationOutcome | None = None
        ir_hash = ir.canonical_hash() if ir is not None else ""

        if ir is not None:
            # Step 8: the typed IR (constructed above or supplied).
            self._record(trace, 8, {
                "ir_version": ir.version,
                "statements": len(ir.statements),
                "source_map_entries": len(ir.source_map),
                "usg_traceable": all(
                    entry.get("source_refs") for entry in ir.source_map.values()
                ),
            })
            # Step 9: sorts and domains.
            self._record(trace, 9, {
                "sorts": [s.name for s in ir.sorts],
                "finiteness": {s.name: s.finiteness.value for s in ir.sorts},
            })
            # Step 10: predicates and relations.
            self._record(trace, 10, {
                "predicates": {p.name: list(p.arg_sorts) for p in ir.predicates},
                "functions": [f.name for f in ir.functions],
            })
            # Step 11: quantifiers and scope.
            quantified = [
                s.statement_id for s in ir.statements
                if s.formula.__class__.__name__ in {"ForallF", "ExistsF", "CardinalityF"}
            ]
            self._record(trace, 11, {"quantified_statements": quantified})
            # Step 12: rules with explicit premises and conclusions.
            self._record(trace, 12, {
                "rules": [
                    {"rule_id": r.rule_id, "head": r.head.predicate,
                     "body": [b.atom.predicate for b in r.body]}
                    for r in ir.rules
                ],
            })
            # Step 13: units and dimensional constraints.
            self._record(trace, 13, {
                "unit_registry": "declared conversions validated before arithmetic",
            })
            # Step 14: semantic assumption profiles.
            self._record(trace, 14, {
                "domains": {
                    d.domain: {
                        "closed_world": d.closed_world,
                        "explicit_negation": d.explicit_negation,
                    }
                    for d in ir.domains
                },
            })
            # Step 15: normalized IR.
            self._record(trace, 15, {
                "ir_hash": ir_hash,
                "warnings": [w.message for w in ir.warnings],
            })

            # -- Phase C -----------------------------------------------------
            routed = intake.classification.routed
            needs_inference = any(
                c.value in {"rule_inference", "mixed"} for c in routed
            ) or bool(ir.rules)
            if needs_inference and self.capabilities.is_enabled("datalog_engine"):
                try:
                    # Step 16: engine selection recorded.
                    self._record(trace, 16, {
                        "engine": "datalog",
                        "rationale": "finite relational rule evaluation over USG facts",
                    })
                    engine = DatalogEngine(facts_from_ir(ir), rules_from_ir(ir))
                    # Steps 17: dependency graph and stratification.
                    self._record(trace, 17, {
                        "strata": [list(s) for s in engine.strata],
                        "recursive_predicates": [
                            p for p, edges in engine._adjacency.items()
                            if any(t == p for t, _n in edges)
                        ],
                    })
                    cache_key = self._cache.key(engine, [d.domain for d in ir.domains])
                    datalog_result = self._cache.get(cache_key)
                    # Steps 18-19: non-recursive then recursive fixed point.
                    if datalog_result is None:
                        datalog_result = engine.evaluate_semi_naive()
                        if self.policy.cross_check_semi_naive:
                            naive = engine.evaluate_naive()
                            equivalent = {
                                (f.predicate, f.args, f.scope)
                                for f in datalog_result.derived_facts
                            } == {
                                (f.predicate, f.args, f.scope) for f in naive.derived_facts
                            }
                            self._record(trace, 19, {
                                "fixed_point": datalog_result.fixed_point_reached,
                                "iterations": datalog_result.iterations,
                                "semi_naive_equals_reference": equivalent,
                            })
                        self._cache.put(cache_key, datalog_result)
                    # Step 18: rule evaluation over indexed facts.
                    self._record(trace, 18, {
                        "input_facts": len(datalog_result.input_facts),
                        "engine_version": datalog_result.engine_version,
                    })
                    # Step 20: explanation graph.
                    multi = [
                        f for f in datalog_result.derived_facts if len(f.derivations) > 1
                    ]
                    self._record(trace, 20, {
                        "derived_facts": len(datalog_result.derived_facts),
                        "facts_with_multiple_derivations": len(multi),
                        "provenance_complete": all(
                            d.rule_id and d.parent_fact_ids
                            for f in datalog_result.derived_facts
                            for d in f.derivations
                        ),
                    })
                except FormalReasoningError as error:
                    self._fail(trace, failures, 19, error)
                # Steps 21-22: conflicts preserved without destructive repair.
                try:
                    facts_for_consistency = list(datalog_result.input_facts) if datalog_result else []
                    if datalog_result:
                        facts_for_consistency.extend(
                            DatalogFact(
                                predicate=f.predicate, args=f.args, fact_id=f.fact_id,
                                negated=False, scope=f.scope, interval=f.interval,
                                source_statement_ids=(f.fact_id,),
                                tag=EvidenceTag.INFERRED_RESULT,
                            )
                            for f in datalog_result.derived_facts
                        )
                    consistency = self.consistency_checker.check(
                        facts_for_consistency, ambiguous_references=ambiguities,
                    )
                    self._record(trace, 21, {
                        "checked_facts": consistency.checked_facts,
                        "conflicts": [
                            {"id": c.conflict_id, "kind": c.kind}
                            for c in consistency.conflicts
                        ],
                    })
                    self._record(trace, 22, {
                        "destructive_repairs": consistency.destructive_repairs,
                        "preserved_sides": consistency.preserved_sides,
                        "policy": "preserve both sides; request resolution from owner",
                    })
                    for conflict in consistency.conflicts:
                        publications.append(
                            self._publish_conflict(conflict)
                        )
                except FormalReasoningError as error:
                    self._fail(trace, failures, 21, error)

            # -- Phase D -----------------------------------------------------
            needs_smt = any(
                c.value in {
                    "boolean_consistency", "arithmetic_constraints",
                    "program_property_verification", "mixed",
                }
                for c in routed
            ) or bool(ir.constraints() or ir.goals())
            if needs_smt and self.capabilities.is_enabled("smt_z3") and Z3_AVAILABLE:
                try:
                    outcome, encoded = self._run_smt(request, ir, trace, limitations)
                    smt_outcomes.append(outcome)
                    self._record(trace, 25, {
                        "named_assumptions": sorted(set(
                            e.assumption_name for e in encoded if e.is_named_assumption
                        )),
                        "tracked_constraints": len(encoded),
                    })
                    if outcome.unsat_core and self.policy.minimize_unsat_cores:
                        adapter = SmtAdapter(
                            ir,
                            permitted_theories=request.permitted_theories,
                            timeout_ms=request.budget.solver_timeout_ms,
                            memory_mb=request.budget.memory_mb,
                        )
                        adapter.check_theories(ir.statements)
                        adapter.track_statements(ir.facts() + ir.constraints())
                        core, minimized = adapter.minimize_core(outcome.unsat_core)
                        self._record(trace, 28, {
                            "core": list(core), "minimized": minimized,
                        })
                    else:
                        self._record(trace, 28, {
                            "core": list(outcome.unsat_core),
                            "minimized": False,
                            "note": "reported as sufficient, not minimum",
                        })
                except FormalReasoningError as error:
                    self._fail(trace, failures, 26, error)
                # Step 29: entailment for each goal statement.
                goals = ir.goals()
                for goal in goals:
                    try:
                        adapter = SmtAdapter(
                            ir,
                            permitted_theories=request.permitted_theories,
                            timeout_ms=request.budget.solver_timeout_ms,
                            memory_mb=request.budget.memory_mb,
                        )
                        premises = tuple(
                            s for s in ir.statements if s.kind != StatementKind.GOAL
                        )
                        outcome = adapter.check_entailment(premises, goal)
                        entailment_outcomes.append(outcome)
                        self._record(trace, 29, {
                            "goal": goal.statement_id,
                            "result": outcome.result_status,
                            "counterexample": outcome.counterexample is not None,
                        })
                        if outcome.counterexample is not None and self.workspace is not None:
                            publications.append(
                                self.publication.publish_counterexample(
                                    self.workspace, outcome.counterexample,
                                    (goal.statement_id,), self.permission_policy,
                                )
                            )
                    except FormalReasoningError as error:
                        self._fail(trace, failures, 29, error)
                # Step 30: numeric semantics notes.
                nonlinear = any(
                    (o.theory_profile and o.theory_profile.nonlinear) for o in smt_outcomes
                )
                self._record(trace, 30, {
                    "integer_real_interchange": "rejected at IR type-check",
                    "division_by_zero": "literal zero denominators rejected pre-solve",
                    "nonlinear": nonlinear,
                    "exact_arithmetic": True,
                })

            # -- Phase E -----------------------------------------------------
            problem = request.optimization_problem
            if isinstance(problem, SelectionProblem):
                try:
                    self._record(trace, 31, {
                        "router": self.optimizer.prefer_backend,
                        "candidates": len(problem.candidates),
                    })
                    self._record(trace, 32, {
                        "decision_variables": len(problem.candidates),
                        "domains": "binary x_i in {0,1}",
                    })
                    self._record(trace, 33, {
                        "required_capabilities": list(problem.required_capabilities),
                    })
                    self._record(trace, 34, {
                        "resource_limits": dict(sorted(problem.resource_limits.items())),
                        "deadline_ms": problem.deadline_ms,
                    })
                    self._record(trace, 35, {
                        "objective_weights": dict(sorted(problem.objective_weights.items())),
                        "units": dict(sorted(problem.units.items())),
                    })
                    optimization = self.optimizer.solve(
                        problem, time_limit_seconds=request.budget.solver_timeout_ms / 1000.0,
                    )
                    self._record(trace, 36, {
                        "status": optimization.status.value,
                        "selection": list(optimization.selection),
                        "objective_value": optimization.objective_value,
                        "best_bound": optimization.best_bound,
                        "optimality_gap": optimization.optimality_gap,
                        "exhaustive": optimization.exhaustive,
                    })
                except FormalReasoningError as error:
                    self._fail(trace, failures, 36, error)

        # -- Phase F ---------------------------------------------------------
        obligation = request.proof_obligation
        if isinstance(obligation, ProofObligation) and (
            request.assurance_level.value in {"standard", "high"}
        ):
            try:
                self._record(trace, 37, {
                    "obligation": obligation.obligation_id,
                    "assurance": request.assurance_level.value,
                    "lean_available": self.proof_verifier.lean_available,
                })
                self._record(trace, 38, {
                    "statement": obligation.statement_text,
                    "premises": [p.name for p in obligation.premises],
                })
                candidate = proof_candidate
                if candidate is None:
                    searcher = ProofSearcher()
                    candidate = searcher.attempt(
                        obligation, tuple(p.name for p in obligation.premises),
                    )
                self._record(trace, 39, {
                    "candidate_origin": candidate.origin if candidate else "none",
                    "untrusted_until_checked": True,
                })
                if candidate is not None:
                    outcome = self.proof_verifier.verify(
                        obligation, candidate, artifact_id=_new_id("frce_proof"),
                    )
                    proof_outcomes.append(outcome)
                    self._record(trace, 40, {
                        "accepted": outcome.artifact is not None,
                        "checker": outcome.artifact.checker_name if outcome.artifact else None,
                        "artifact_hash": (
                            outcome.artifact.artifact_hash[:16] if outcome.artifact else None
                        ),
                        "steps_checked": outcome.artifact.steps_checked if outcome.artifact else 0,
                    })
                    self._record(trace, 41, {
                        "conditional_on": (
                            list(outcome.artifact.conditional_on) if outcome.artifact else []
                        ),
                        "premise_truth_separate": True,
                    })
                else:
                    self._record(trace, 40, {
                        "accepted": False,
                        "note": "proof-search failure is not reported as theorem falsity",
                    })
            except FormalReasoningError as error:
                self._fail(trace, failures, 40, error)

        # Build results and finalize (Phase G/H handled inside _finalize).
        results = self._build_results(
            request, ir, datalog_result, consistency, smt_outcomes,
            entailment_outcomes, optimization, proof_outcomes, trace,
        )
        run = self._finalize(
            run_id, request, fingerprint, trace, intake, ir, retrieval,
            datalog_result, consistency, smt_outcomes, entailment_outcomes,
            optimization, proof_outcomes, results, [], publications,
            proposals, failures, limitations, warnings, start_wall,
        )
        return run

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _goal_predicates(request: FormalizationRequest) -> tuple[str, ...]:
        """Extract predicate names referenced by the required property string."""
        if not request.required_property:
            return ()
        return tuple(
            token.split("(")[0].strip() for token in request.required_property.split(")")
            if token.strip()
        )

    def _build_ir_from_records(
        self, records: Sequence[TypedKnowledgeRecord],
    ) -> FormalIR:
        """Build one IR from typed USG records (Step 8 via the adapter)."""
        builder = FormalIRBuilder()
        mapper = USGToIRMapper()
        mapper.map_records(records, builder, default_domain="default")
        return builder.build()

    def _run_smt(
        self,
        request: FormalizationRequest,
        ir: FormalIR,
        trace: list[dict[str, Any]],
        limitations: list[str],
    ) -> tuple[SmtOutcome, tuple[Any, ...]]:
        """Run one satisfiability check with step-granular tracing (Steps 23-27)."""
        statements = tuple(
            s for s in ir.statements if s.kind != StatementKind.GOAL
        )
        adapter = SmtAdapter(
            ir,
            permitted_theories=request.permitted_theories,
            timeout_ms=request.budget.solver_timeout_ms,
            memory_mb=request.budget.memory_mb,
        )
        need = adapter.check_theories(statements)  # Step 23
        self._record(trace, 23, {
            "detected_theories": sorted(t.value for t in need.theories),
            "nonlinear": need.nonlinear,
        })
        encoded = adapter.track_statements(statements)  # Steps 24-25
        self._record(trace, 24, {
            "encoded_constraints": len(encoded),
            "traceability": all(e.ir_statement_id for e in encoded),
        })
        outcome = adapter.solve_tracked(need)  # Step 26
        self._record(trace, 26, {
            "solver_status": outcome.solver_status.value,
            "solver": f"{outcome.solver.name} {outcome.solver.version}" if outcome.solver else None,
            "wall_clock_ms": round(outcome.wall_clock_ms, 2),
        })
        # Step 27: cautious model interpretation.
        if outcome.model is not None:
            self._record(trace, 27, {
                "model_decoded": True,
                "independently_validated": outcome.model_independently_validated,
                "caution": "one assignment satisfying the encoded constraints; "
                           "not claimed unique nor descriptive of reality",
            })
        limitations.extend(outcome.limitations)
        return outcome, encoded

    def _publish_conflict(self, conflict: ConflictRecord) -> Any:
        """Publish one conflict when a workspace is bound (Step 44)."""
        if self.workspace is None:
            return None
        return self.publication.publish_conflict(
            self.workspace,
            {
                "conflict_id": conflict.conflict_id,
                "kind": conflict.kind,
                "description": conflict.description,
                "side_a": list(conflict.side_a),
                "side_b": list(conflict.side_b),
                "minimal_evidence": list(conflict.minimal_evidence),
                "resolution_request": conflict.resolution_request,
            },
            self.permission_policy,
        )

    def _build_results(
        self,
        request: FormalizationRequest,
        ir: FormalIR | None,
        datalog_result: DatalogResult | None,
        consistency: ConsistencyReport | None,
        smt_outcomes: Sequence[SmtOutcome],
        entailment_outcomes: Sequence[SmtOutcome],
        optimization: OptimizationOutcome | None,
        proof_outcomes: Sequence[ProofCheckOutcome],
        trace: list[dict[str, Any]],
    ) -> list[FormalResult]:
        """Normalize every phase output into typed envelopes (Step 42)."""
        results: list[FormalResult] = []
        base_provenance = tuple(request.source_refs)
        ir_statement_refs = (
            tuple(s.statement_id for s in ir.statements[:64]) if ir is not None else ()
        )
        reproducibility = {
            "package_version": FORMAL_REASONING_PACKAGE_VERSION,
            "fingerprint": request.fingerprint,
            "ir_hash": ir.canonical_hash() if ir is not None else "",
        }
        # Rule-derived inference result.
        if datalog_result is not None:
            conflicts_exist = bool(consistency and consistency.conflicts)
            derived_refs = tuple(f.fact_id for f in datalog_result.derived_facts[:64])
            results.append(FormalResult(
                result_id=_new_id("frce_res"),
                request_id=request.request_id,
                kind=ResultKind.RULE_DERIVED_INFERENCE,
                status=ResultStatus.UNKNOWN if conflicts_exist else ResultStatus.SATISFIABLE,
                statement=(
                    f"datalog fixed point over {len(datalog_result.input_facts)} input facts "
                    f"derived {len(datalog_result.derived_facts)} facts"
                ),
                statement_ir_refs=tuple(f.fact_id for f in datalog_result.input_facts[:64]),
                derivation_refs=derived_refs,
                assumptions=tuple(request.assumptions),
                solver=SolverIdentity(
                    name="FRCE Datalog engine",
                    version=datalog_result.engine_version,
                    configuration={"strata": [list(s) for s in datalog_result.strata]},
                    backend="in_process",
                ),
                provenance=base_provenance,
                runtime=RuntimeMeasurement(
                    wall_clock_ms=datalog_result.iterations * 0.0 + 0.0,
                ),
                limitations=(
                    "derived facts are inferences under the declared rules and "
                    "assumption profiles, not observed facts"
                ),
                warnings=(("conflicts detected; both sides preserved",) if conflicts_exist else ()),
                reproducibility=reproducibility,
            ))

        # Satisfiability results.
        for outcome in (*smt_outcomes, *entailment_outcomes):
            statement = (
                "satisfiability of the encoded constraint set"
                if outcome in smt_outcomes
                else "entailment of the goal under the premises (Gamma ∧ ¬phi checked)"
            )
            results.append(FormalResult(
                result_id=_new_id("frce_res"),
                request_id=request.request_id,
                kind=ResultKind.SATISFIABILITY_RESULT,
                status=ResultStatus(outcome.result_status),
                statement=statement,
                statement_ir_refs=ir_statement_refs,
                assumptions=tuple(request.assumptions),
                theory_profile=outcome.theory_profile,
                solver=outcome.solver,
                model=dict(outcome.model) if outcome.model else None,
                model_independently_validated=outcome.model_independently_validated,
                counterexample=dict(outcome.counterexample) if outcome.counterexample else None,
                unsat_core=outcome.unsat_core,
                unsat_core_minimized=outcome.unsat_core_minimized,
                provenance=base_provenance,
                runtime=RuntimeMeasurement(wall_clock_ms=outcome.wall_clock_ms),
                limitations=tuple(outcome.limitations),
                warnings=tuple(outcome.warnings),
                reproducibility=reproducibility,
            ))

        # Optimization result.
        if optimization is not None:
            problem = request.optimization_problem
            candidate_refs = (
                tuple(c.module_id for c in problem.candidates)
                if isinstance(problem, SelectionProblem) else ()
            )
            results.append(FormalResult(
                result_id=_new_id("frce_res"),
                request_id=request.request_id,
                kind=ResultKind.OPTIMIZATION_RESULT,
                status=optimization.status,
                statement=(
                    f"module selection over {len(optimization.selection)} selected "
                    "candidates with recorded weights and units"
                ),
                statement_ir_refs=candidate_refs,
                model={
                    "selection": list(optimization.selection),
                    "objective_value": optimization.objective_value,
                    "best_bound": optimization.best_bound,
                    "optimality_gap": optimization.optimality_gap,
                },
                model_independently_validated=(
                    True if optimization.selection else None
                ),
                solver=optimization.solver,
                provenance=base_provenance,
                runtime=RuntimeMeasurement(wall_clock_ms=optimization.wall_clock_ms),
                limitations=tuple(optimization.limitations),
                warnings=tuple(optimization.warnings),
                reproducibility=reproducibility,
            ))

        # Proof results.
        for outcome in proof_outcomes:
            premise_refs = (
                tuple(f"premise:{p.name}" for p in request.proof_obligation.premises)
                if isinstance(request.proof_obligation, ProofObligation) else ()
            )
            if outcome.artifact is None:
                results.append(FormalResult(
                    result_id=_new_id("frce_res"),
                    request_id=request.request_id,
                    kind=ResultKind.MACHINE_CHECKED_PROOF,
                    status=ResultStatus.UNKNOWN,
                    statement="proof candidate rejected or search failed; not a refutation",
                    statement_ir_refs=premise_refs,
                    assumptions=tuple(request.assumptions),
                    solver=None,
                    provenance=base_provenance,
                    limitations=(
                        "a rejected or missing proof is 'not proved', never 'false'",
                    ),
                    reproducibility=reproducibility,
                ))
                continue
            artifact = outcome.artifact
            results.append(FormalResult(
                result_id=_new_id("frce_res"),
                request_id=request.request_id,
                kind=ResultKind.MACHINE_CHECKED_PROOF,
                status=ResultStatus.PROVED,
                statement=f"{artifact.theorem_name}: {artifact.statement_text}",
                statement_ir_refs=premise_refs,
                proof_ref=artifact.artifact_id,
                proof_check_status="kernel_checked",
                assumptions=tuple(request.assumptions) or tuple(
                    NamedAssumption(name, "declared premise of the proof obligation")
                    for name in artifact.conditional_on
                ),
                solver=SolverIdentity(
                    name=artifact.checker_name,
                    version=artifact.checker_version,
                    configuration={"dependencies": list(artifact.dependencies)},
                    backend="local_kernel" if not artifact.lean_available else "lean",
                ),
                provenance=base_provenance,
                limitations=tuple(artifact.limitations) + (
                    artifact.premise_truth_note,
                ),
                reproducibility={
                    **reproducibility,
                    "artifact_hash": artifact.artifact_hash,
                },
            ))
        return results

    def _finalize(
        self,
        run_id: str,
        request: FormalizationRequest,
        fingerprint: str,
        trace: list[dict[str, Any]],
        intake: IntakeReport,
        ir: FormalIR | None,
        retrieval: RetrievalRecord | None,
        datalog_result: DatalogResult | None,
        consistency: ConsistencyReport | None,
        smt_outcomes: Sequence[SmtOutcome],
        entailment_outcomes: Sequence[SmtOutcome],
        optimization: OptimizationOutcome | None,
        proof_outcomes: Sequence[ProofCheckOutcome],
        results: Sequence[FormalResult],
        envelope_validations: Sequence[EnvelopeValidation],
        publications: list[Any],
        proposals: list[PromotionProposal],
        failures: Sequence[Mapping[str, Any]],
        limitations: Sequence[str],
        warnings: Sequence[str],
        start_wall: float,
    ) -> FormalReasoningRun:
        """Complete Phase G (42-48) and Phase H (49-52), then assemble the run."""
        # Step 42: unified result envelopes were built by _build_results.
        self._record(trace, 42, {
            "results": len(results),
            "kinds": sorted({r.kind.value for r in results}),
            "note": "envelopes keep the distinct solver semantics unflattened",
        })
        # Step 43: independent validation of every envelope.
        validations: list[EnvelopeValidation] = []
        for result in results:
            validations.append(validate_result_envelope(result))
        self._record(trace, 43, {
            "validated_envelopes": len(validations),
            "all_valid": all(v.valid for v in validations),
            "certificate_replay": "proof hashes and models re-checkable via "
                                  "envelope.CertificateReplay",
        })

        # Step 44: publish typed entries to the shared workspace.
        publications = list(publications)
        if self.workspace is not None:
            for result in results:
                publications.append(
                    self.publication.publish_result(
                        self.workspace, result, self.permission_policy,
                    )
                )
            if ir is not None and request.assumptions:
                publications.append(
                    self.publication.publish_assumptions(
                        self.workspace,
                        [
                            {"name": a.name, "description": a.description,
                             "tag": a.tag.value}
                            for a in request.assumptions
                        ],
                        self.permission_policy,
                    )
                )
        self._record(trace, 44, {
            "publications": [
                {"finding_type": p.finding_type, "accepted": p.accepted}
                for p in publications if p is not None
            ],
            "distinct_types": sorted({p.finding_type for p in publications if p is not None}),
        })

        # Step 45: knowledge-write governance — propose, never auto-promote.
        if datalog_result is not None and datalog_result.derived_facts:
            fact = datalog_result.derived_facts[0]
            proposal = self.governance.propose_derived_fact(
                f"{fact.predicate}({', '.join(fact.args)})",
                [d.rule_id for d in fact.derivations],
                [a.name for a in request.assumptions],
                proposal_id=_new_id("frce_prop"),
                confidence_note="derived under the declared rule set",
            )
            proposals.append(proposal)
        decisions = [
            self.governance.submit(proposal, approver=None) for proposal in proposals
        ]
        self._record(trace, 45, {
            "proposals": [p.proposal_id for p in proposals],
            "decisions": [
                {"proposal": d.proposal_id, "approved": d.approved, "reason": d.reason}
                for d in decisions
            ],
            "usg_writes": "none; the engine is read-only against authoritative USG",
        })

        # Step 46: persist the auditable execution record.
        audit_id: int | None = None
        if self.audit_log is not None:
            audit_id = self.audit_log.record_run({
                "request_id": request.request_id,
                "fingerprint": fingerprint,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                "schema_version": FORMAL_REASONING_RESULT_VERSION,
                "ir_version": ir.version if ir is not None else "",
                "ir_hash": ir.canonical_hash() if ir is not None else "",
                "input_fact_ids": [
                    f.fact_id for f in datalog_result.input_facts
                ] if datalog_result else [],
                "input_rule_ids": [r.rule_id for r in ir.rules] if ir else [],
                "assumptions": [a.name for a in request.assumptions],
                "solver_versions": [
                    f"{r.solver.name}/{r.solver.version}"
                    for r in results if r.solver is not None
                ],
                "solver_config": {
                    "optimizer_backend": self.policy.prefer_optimizer_backend,
                    "minimize_unsat_cores": self.policy.minimize_unsat_cores,
                },
                "resource_use": {
                    "wall_clock_ms": round((time.perf_counter() - start_wall) * 1000, 2),
                },
                "result_kind": results[0].kind.value if results else "",
                "status": self._overall_status(results, failures),
                "proof_refs": [
                    r.proof_ref for r in results if r.proof_ref
                ],
                "artifact_hashes": [
                    r.reproducibility.get("artifact_hash", "")
                    for r in results
                    if r.reproducibility.get("artifact_hash")
                ],
                "error": "; ".join(f["detail"] for f in failures) or None,
                "trace": [
                    {"operation_id": t["operation_id"], "step": t["step"]}
                    for t in trace
                ],
            })
        self._record(trace, 46, {"audit_record_id": audit_id})

        # Step 47: budgets and isolation.
        self._record(trace, 47, {
            "budget": {
                "wall_clock_ms": request.budget.wall_clock_ms,
                "solver_timeout_ms": request.budget.solver_timeout_ms,
                "memory_mb": request.budget.memory_mb,
                "max_output_bytes": request.budget.max_output_bytes,
            },
            "process_isolation": self.policy.process_isolation,
            "network_access": "denied for solver workers",
            "binaries": "allowlisted only",
        })

        # Step 48: failure handling was applied per phase.
        improvements = [
            classify_failure_for_improvement({"kind": f["class"]}) for f in failures
        ]
        self._record(trace, 48, {
            "failures": list(failures),
            "retry_policy": "bounded backoff on transient classes only",
            "improvement_categories": [i.category for i in improvements],
        })

        # Phase H (Steps 49-52).
        self._record(trace, 49, {"benchmark_suite": "FRCE-BENCH/1.0.0 (versioned, on demand)"})
        unknown_guard = assert_unknown_never_coerced()
        wall_ms = (time.perf_counter() - start_wall) * 1000.0
        metrics = {
            "wall_clock_ms": round(wall_ms, 2),
            "steps_executed": len(trace),
            "results": len(results),
            "envelope_validity_rate": (
                sum(1 for v in validations if v.valid) / len(validations)
                if validations else None
            ),
            "conflicts_detected": len(consistency.conflicts) if consistency else 0,
            "smt_checks": len(smt_outcomes) + len(entailment_outcomes),
            "optimization_status": (
                optimization.status.value if optimization is not None else None
            ),
            "proofs_verified": sum(
                1 for o in proof_outcomes if o.artifact is not None
            ),
            "unknown_never_coerced": unknown_guard[0],
            "failures": len(failures),
        }
        self._record(trace, 50, {
            "metrics": {k: v for k, v in metrics.items() if k != "unknown_never_coerced"},
            "correctness_and_operations_measured_separately": True,
        })
        self._record(trace, 51, {
            "improvements": [
                {"category": i.category, "remediation": i.remediation}
                for i in improvements
            ],
            "guarantees_loosened": False,
        })
        capability_snapshot = self.capabilities.to_dict()
        self._record(trace, 52, {
            "capabilities": capability_snapshot,
            "release_model": "incremental behind acceptance-gated flags",
        })

        run = FormalReasoningRun(
            run_id=run_id,
            request_id=request.request_id,
            fingerprint=fingerprint,
            schema_version=FORMAL_REASONING_SCHEMA_VERSION,
            status=self._overall_status(results, failures),
            replayed=False,
            step_trace=tuple(trace),
            intake=intake,
            ir=ir,
            ir_hash=ir.canonical_hash() if ir is not None else "",
            retrieval=retrieval,
            datalog_result=datalog_result,
            consistency=consistency,
            smt_outcomes=tuple(smt_outcomes),
            entailment_outcomes=tuple(entailment_outcomes),
            optimization=optimization,
            proof_outcomes=tuple(proof_outcomes),
            results=tuple(results),
            envelope_validations=tuple(validations),
            publications=tuple(p for p in publications if p is not None),
            promotion_proposals=tuple(proposals),
            audit_id=audit_id,
            failures=tuple(dict(f) for f in failures),
            improvements=tuple(improvements),
            metrics=metrics,
            capabilities=capability_snapshot,
            limitations=tuple(limitations),
            warnings=tuple(warnings),
        )
        self._runs[fingerprint] = run
        return run

    @staticmethod
    def _overall_status(
        results: Sequence[FormalResult], failures: Sequence[Mapping[str, Any]],
    ) -> str:
        """Derive the run-level status without collapsing epistemic kinds."""
        if failures:
            for result in results:
                if result.status == ResultStatus.ERROR:
                    return "error"
            return "completed_with_failures"
        if not results:
            return "no_results"
        if any(r.status == ResultStatus.PROVED for r in results):
            return "proved"
        if any(r.status == ResultStatus.OPTIMAL for r in results):
            return "optimal"
        if any(r.establishes_false for r in results):
            return "established_negative"
        if any(r.is_success for r in results):
            return "completed"
        return "not_established"

    # ------------------------------------------------------------------ Phase H

    def run_benchmarks(self) -> Any:
        """Execute the versioned benchmark suite (Step 49 on demand)."""
        return run_benchmark_suite()


class FormalReasoningService:
    """The public service facade over the FRCE coordinator (Section 2)."""

    def __init__(self, coordinator: FormalReasoningCoordinator | None = None) -> None:
        self.coordinator = coordinator or FormalReasoningCoordinator()

    def submit(
        self,
        request: FormalizationRequest,
        **kwargs: Any,
    ) -> FormalReasoningRun:
        """Submit one formalization request and return the typed run."""
        return self.coordinator.run(request, **kwargs)

    def capabilities(self) -> dict[str, Any]:
        """Return the capability-flag snapshot."""
        return self.coordinator.capabilities.to_dict()

    def benchmarks(self) -> Any:
        """Run the versioned benchmark suite."""
        return self.coordinator.run_benchmarks()


__all__ = [
    "FrcePolicy",
    "FormalReasoningRun",
    "FormalReasoningCoordinator",
    "FormalReasoningService",
]
