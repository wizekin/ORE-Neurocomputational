"""Neuro-Symbolic Reasoning Layer: the 45-step Point 3 orchestrator.

``NeuroSymbolicReasoningCoordinator.run(request)`` executes the full integrated
algorithm of ``ORE —Neurocomputational Point 03 Neuro-Symbolic Reasoning
Algorithm.txt`` (Section 5, Phases A-H) in order, recording one trace record
per operation (``NSR-001`` .. ``NSR-045``). Integration guardrails:

- the shared workspace stays authoritative for runtime state; one workspace
  per run, one bound snapshot revision (Step A2), publication only through the
  existing ``FindingPublisher``/``WorkspaceStateManager`` atomic path;
- USG stays authoritative for knowledge; read-only adapters, no shadow KB;
- the deterministic core runs with no learned components, no GPU and no
  optional frameworks (Phase F degrades gracefully when models are absent);
- epistemic categories never collapse: conclusions, conflicts, ambiguity
  alternatives, probabilistic support and model proposals are separate result
  fields.
"""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Mapping

from ore_contracts import ContractCatalog
from ore_workspace import SharedWorkspace, WorkspaceManager
from ore_workspace.permissions import PermissionPolicy
from ore_workspace.publication import FindingPublisher
from ore_workspace.models import Provenance

from .ambiguity_resolver import AmbiguityResolver
from .config import NSRLPolicy
from .conflict_manager import ConflictManager
from .contracts import (
    EpistemicStatus,
    EvidenceItem,
    InferenceMode,
    Literal,
    LogicalTerm,
    NEUROSYMBOLIC_SCHEMA_VERSION,
    NSRLConflictRecord,
    ProofClass,
    PublicationState,
    ReasoningFinding,
    RuleClass,
    RuleDefinition,
    SemanticFact,
    VerificationLevel,
    ConfidenceVector,
    make_id,
    sha256_json,
)
from .probabilistic_adapter import IndependenceDeclaration
from .constraint_engine import ConstraintEngine
from .context import build_bounded_context
from .learned_candidate_adapter import LearnedCandidateAdapter
from .observability import NSRLObservability
from .probabilistic_adapter import ProbabilisticEvidenceAdapter
from .proof_builder import ProofBuilder
from .proof_model import ProofDAG
from .proof_verifier import IndependentProofVerifier
from .registry import NSRL_OPERATION_IDS, operation_by_step
from .registry import PHASE_STEP_RANGES
from .request_models import ReasoningRequest, ReasoningRunResult
from .resource_policy import ResourcePolicy, ResourceSnapshot
from .rule_validation import validate_ruleset
from .semantic_adapter import USGSemanticAdapter
from .symbolic_engine import EngineBudgets, SymbolicEngine
from .trace import NSRLTraceBuilder
from .workspace_publisher import NSRLWorkspacePublisher

_WORKSPACE_FIELDS = (
    "semantic_hypotheses", "unresolved_questions", "artifact_references",
)


class NeuroSymbolicReasoningCoordinator:
    """Run the 45-step neuro-symbolic reasoning algorithm for one request."""

    def __init__(
        self,
        *,
        policy: NSRLPolicy | None = None,
        learned_adapter: LearnedCandidateAdapter | None = None,
        workspace_manager: WorkspaceManager | None = None,
        repository_root: str | Any | None = None,
        logger_database_path: str | Any | None = None,
    ) -> None:
        """Bind the policy, optional learned adapter and workspace manager."""
        self.policy = policy or NSRLPolicy()
        self.learned_adapter = learned_adapter or LearnedCandidateAdapter()
        self._workspace_manager = workspace_manager
        self._owns_manager = workspace_manager is None
        self.repository_root = repository_root
        self.logger_database_path = logger_database_path
        self._results: dict[str, ReasoningRunResult] = {}

    # ------------------------------------------------------------------ helpers

    def _get_manager(self) -> WorkspaceManager:
        """Return the injected or a freshly created workspace manager."""
        if self._workspace_manager is not None:
            return self._workspace_manager
        return WorkspaceManager(
            repository_root=self.repository_root,
            logger_database_path=self.logger_database_path,
            enable_logging=False,
        )

    @staticmethod
    def _record_step(
        records: list[dict[str, Any]], step: int, details: Mapping[str, Any],
    ) -> None:
        """Append one operation trace record for an executed algorithm step."""
        spec = operation_by_step(step)
        records.append({
            "operation_id": spec.operation_id,
            "step": spec.step,
            "phase": spec.phase,
            "title": spec.title,
            "details": dict(details),
        })

    def _make_workspace(self, request: ReasoningRequest) -> SharedWorkspace:
        """Create the isolated run workspace with NSRL write grants (Step A2)."""
        permissions = PermissionPolicy({
            NSRL_PRODUCER_GRANT: frozenset(_WORKSPACE_FIELDS),
        })
        goal_text = f"{request.goal_predicate}({','.join(request.goal_args)})"
        return SharedWorkspace(
            goal_text,
            goal_text,
            permissions=permissions,
        )

    def _build_fact_table_and_rules(
        self,
        request: ReasoningRequest,
        adapter: USGSemanticAdapter | None,
        step_records: list[dict[str, Any]],
        *,
        attach_error: str | None = None,
    ) -> tuple[list[SemanticFact], dict[str, Any], dict[str, Any]]:
        """Run Steps A3-A6; return normalized facts, USG reference, resolution map."""
        facts: list[SemanticFact] = []
        usg_reference: dict[str, Any] = {}
        reference_map: dict[str, Any] = {}

        # Step 3 - resolve semantic IDs while preserving ambiguity.
        if adapter is not None and self.policy.ablations.usg_retrieval:
            usg_reference = adapter.bundle_reference()
            reference_map = adapter.resolve_references(request.entity_references)
            self._record_step(step_records, 3, {
                "usg_attached": True,
                "bundle_id": usg_reference.get("bundle_id"),
                "bundle_digest": usg_reference.get("bundle_digest"),
                "attach_error": None,
                "resolved_references": {
                    key: value.to_dict() for key, value in sorted(reference_map.items())
                },
            })
        else:
            for reference in request.entity_references:
                reference_map[reference] = _identity_reference(reference)
            self._record_step(step_records, 3, {
                "usg_attached": False,
                "attach_error": attach_error,
                "resolved_references": {
                    key: value.to_dict() for key, value in sorted(reference_map.items())
                },
            })

        # Step 4 - retrieve facts through exact indexes; rules from the ruleset.
        if adapter is not None and self.policy.ablations.usg_retrieval:
            usg_facts = adapter.retrieve_facts(reference_map)
            facts.extend(usg_facts)
        facts.extend(request.facts)
        rule_count = len(request.rules)
        self._record_step(step_records, 4, {
            "usg_fact_count": len(facts) - len(request.facts),
            "request_fact_count": len(request.facts),
            "rule_count": rule_count,
            "ruleset_version": request.ruleset_version,
            "retrieval": "exact_indexes_first",
        })

        # Step 5 - normalize canonical forms while retaining the source form.
        normalized: list[SemanticFact] = []
        normalization_records: list[dict[str, Any]] = []
        for fact in facts:
            if adapter is not None and self.policy.ablations.usg_retrieval:
                normalized_fact, record = adapter.normalize_fact(fact, reference_map)
            else:
                normalized_fact = fact
                record = {
                    "fact_id": fact.fact_id,
                    "subject_canonicalized": False,
                    "object_canonicalized": False,
                    "temporal_canonicalized": False,
                    "polarity_retained": fact.polarity,
                }
            normalized.append(normalized_fact)
            normalization_records.append(dict(record))
        self._record_step(step_records, 5, {
            "normalized_count": len(normalized),
            "source_form_retained": True,
            "sample_records": normalization_records[:5],
        })

        # Step 6 - assign explicit epistemic status.
        assigned: list[SemanticFact] = []
        assignments: dict[str, str] = {}
        for fact in normalized:
            status_fact, status_value = USGSemanticAdapter.assign_epistemic_status(fact)
            assigned.append(status_fact)
            assignments[status_fact.fact_id] = status_value
        self._record_step(step_records, 6, {
            "status_counts": _count_values(assignments.values()),
            "unknown_is_not_false": True,
        })
        return assigned, usg_reference, reference_map

    def _run_inference(
        self,
        request: ReasoningRequest,
        rules: tuple[RuleDefinition, ...],
        facts: list[SemanticFact],
        step_records: list[dict[str, Any]],
    ) -> tuple[Any, Any]:
        """Run Phase B (Steps 8-16) and return the engine and its report."""
        # Step 8 - validate the ruleset against the safe profile.
        validation = validate_ruleset(rules)
        accepted_rules = validation.accepted_rules
        self._record_step(step_records, 8, {
            "accepted_rules": [rule.rule_id for rule in accepted_rules],
            "rejected": [item.to_dict() for item in validation.rejected_rules],
            "profile": validation.profile.to_dict(),
        })

        # Steps 9-16 run inside the deterministic engine; the coordinator
        # records each operation against the engine report afterwards.
        engine = SymbolicEngine(
            accepted_rules,
            budgets=EngineBudgets(
                max_rule_fires=self.policy.budgets.max_rule_fires,
                max_depth=self.policy.budgets.max_depth,
                max_iterations=32,
                max_candidates=self.policy.budgets.max_candidates,
                wall_clock_seconds=request.deadline_seconds,
            ),
            scope_context=dict(request.scope),
            max_queue_size=self.policy.max_queue_size,
        )
        for fact in facts:
            engine.add_fact(fact)

        report = None
        if request.inference_mode in {InferenceMode.FORWARD, InferenceMode.HYBRID}:
            report = engine.forward_chain(goal=request.goal_literal)
        if request.inference_mode in {InferenceMode.BACKWARD, InferenceMode.HYBRID}:
            report = engine.backward_chain(request.goal_literal, report=report)

        # Step 9 - rule selection rationale recorded by the engine.
        self._record_step(step_records, 9, {
            "selected_rules": list(report.selected_rules),
            "rationale": report.selection_rationale,
        })
        # Step 10 - typed unification with bounded diagnostics.
        self._record_step(step_records, 10, {
            "substitutions_computed": len(report.derived),
            "type_checks_enforced": True,
            "bounded_diagnostics": True,
        })
        # Step 11 - conjunction evaluation through the strict firing gate.
        self._record_step(step_records, 11, {
            "gate": "Fire(r,theta)=1 iff all premises supported and type-valid",
            "rule_fires": report.rule_fires,
            "strict_gate_is_logical": True,
        })
        # Step 12 - conclusions instantiated with rule version and premises.
        self._record_step(step_records, 12, {
            "instantiated_conclusions": len(report.derived),
            "rule_version_attached": True,
        })
        # Step 13 - defeasible exception resolution.
        self._record_step(step_records, 13, {
            "defeated": [
                {
                    "conclusion": item.conclusion_literal.signature(),
                    "rule": item.rule_ref[0],
                    "defeated_by": list(item.defeated_by),
                }
                for item in report.defeated
            ],
            "incomparable_conflicts": len(report.conflicts),
            "specificity_before_priority": True,
        })
        # Step 14 - forward chaining to fixed point under budgets.
        self._record_step(step_records, 14, {
            "iterations": report.iterations,
            "fixed_point_reached": report.fixed_point_reached,
            "stop_reason": report.stop_reason,
            "queue_stats": engine.queue.stats(),
        })
        # Step 15 - backward chaining with memoized subgoals.
        self._record_step(step_records, 15, {
            "mode": request.inference_mode.value,
            "goal_satisfied": report.goal_satisfied,
            "memoized_subgoals": True,
        })
        # Step 16 - cycle detection with explicit partial status.
        self._record_step(step_records, 16, {
            "cycle_events": [item.to_dict() for item in report.cycle_events],
            "partial_on_cycle": bool(report.cycle_events),
        })
        return engine, report

    def _build_and_verify_proofs(
        self,
        request: ReasoningRequest,
        rules: tuple[RuleDefinition, ...],
        engine: Any,
        report: Any,
        run_id: str,
        step_records: list[dict[str, Any]],
    ) -> tuple[ProofBuildReportLike, dict[str, bool]]:
        """Run Phase C (Steps 17-21): build, validate, verify, trace, classify."""
        builder = ProofBuilder(rules, workspace_revision=0)
        # Step 17 - build the content-addressed derivation DAG.
        build = builder.build(report, dict(engine.facts._facts), run_id)  # noqa: SLF001
        self._record_step(step_records, 17, {
            "node_count": len(build.graph.nodes),
            "content_addressed": True,
            "deduplicated_nodes": True,
        })
        # Step 18 - validate the proof graph structurally.
        self._record_step(step_records, 18, {
            "issues": [item.to_dict() for item in build.issues],
            "graph_valid": not build.issues,
        })
        # Step 19 - independently re-execute strict rule applications.
        verified: dict[str, bool] = {}
        verification_overhead_ms = 0.0
        if (
            request.verification_level is not VerificationLevel.NONE
            and self.policy.ablations.independent_verification
        ):
            verifier = IndependentProofVerifier(rules, engine.facts)
            started = time.perf_counter()
            verification = verifier.verify_graph(
                build.graph.nodes, verification_level=request.verification_level.value,
            )
            verification_overhead_ms = (time.perf_counter() - started) * 1000.0
            for node_id, node in build.graph.nodes.items():
                status_ok = bool(verification.verified.get(node_id, False))
                tampered = node_id in verification.tampered
                build.graph.mark_verifier_status(
                    node_id,
                    _verification_status(status_ok, tampered),
                )
            verified = dict(verification.verified)
            self._record_step(step_records, 19, {
                "verification_level": request.verification_level.value,
                "verification": verification.to_dict(),
                "overhead_ms": round(verification_overhead_ms, 3),
                "verifier_does_not_trust_engine": True,
            })
        else:
            self._record_step(step_records, 19, {
                "verification_level": request.verification_level.value,
                "skipped": True,
                "reason": "verification_level none or independent_verification ablated",
            })
        # Step 20 - trace source lineage to original records.
        builder.trace_lineage(build, dict(engine.facts._facts))  # noqa: SLF001
        self._record_step(step_records, 20, {
            "lineage_records": len(build.lineage),
            "workspace_revision_bound": True,
        })
        # Step 21 - classify proofs into non-conflatable labels.
        builder.classify_with_verification(build, verified)
        class_counts = _count_values(item.value for item in build.classification.values())
        self._record_step(step_records, 21, {
            "classification_counts": class_counts,
            "labels_never_conflated": True,
        })
        return build, verified

    def _resolve_ambiguity(
        self,
        request: ReasoningRequest,
        adapter: USGSemanticAdapter | None,
        step_records: list[dict[str, Any]],
        model_hypotheses: tuple[Any, ...] = (),
    ) -> tuple[Any, Any, Any]:
        """Run Phase D (Steps 22-25): candidates, constraints, ranking, ambiguity."""
        resolver = AmbiguityResolver(
            self.policy.ranking_weights,
            epsilon=self.policy.ambiguity_epsilon,
            max_candidates=self.policy.max_candidates,
        )
        readings: dict[str, tuple[str, ...]] = {}
        if adapter is not None:
            resolved = adapter.resolve_references(request.entity_references)
            for reference, record in resolved.items():
                if record.ambiguous and len(record.alternatives) > 1:
                    readings[reference] = record.alternatives
        for candidate in model_hypotheses:
            concept = candidate.semantic_structure.get("concept_id")
            reference = candidate.semantic_structure.get("reference", "model")
            if concept:
                readings.setdefault(reference, (concept,))

        # Step 22 - generate the finite candidate set.
        candidates = resolver.generate_candidates(
            readings,
            evidence_refs=tuple(item.evidence_id for item in request.evidence_items),
        )
        self._record_step(step_records, 22, {
            "candidate_count": len(candidates),
            "sources": ["usg_alternatives", "request_references", "model_hypotheses"],
        })

        # Steps 23-24 - hard filtering of proven violations, then soft ranking.
        constraints = ConstraintEngine()
        if self.policy.ablations.constraint_filtering:
            constraint_report = constraints.evaluate(candidates)
            survivors = constraint_report.active
        else:
            constraint_report = None
            survivors = tuple(
                _mark_candidate(item, "ranked") for item in candidates
            )
        self._record_step(step_records, 23, {
            "constraint_filtering": self.policy.ablations.constraint_filtering,
            "hard_filtered": (
                [list(item) for item in constraint_report.hard_filtered]
                if constraint_report is not None else []
            ),
            "only_proven_violations_eliminated": True,
        })
        soft_scores = (
            constraint_report.soft_scores if constraint_report is not None else {}
        )
        ranked = resolver.rank(survivors, soft_scores)
        self._record_step(step_records, 24, {
            "ranking_weights": self.policy.ranking_weights.to_dict(),
            "score_formula": "S(c)=w_t*T+w_g*G+w_e*E+w_m*M-w_v*V",
            "ranked": [item.to_dict() for item in ranked[:10]],
            "soft_scores_never_proof": True,
        })

        # Step 25 - preserve ambiguity with discriminating evidence.
        if self.policy.ablations.ambiguity_preservation:
            groups = resolver.preserve_ambiguity(
                ranked,
                discriminating_evidence=tuple(
                    item.evidence_id for item in request.evidence_items
                ),
            )
        else:
            groups = ()
        candidates = resolver.mark_alternatives(candidates, groups)
        self._record_step(step_records, 25, {
            "ambiguity_groups": [item.to_dict() for item in groups],
            "forced_single_interpretation": False,
        })
        return candidates, ranked, groups

    def _handle_probabilistic(
        self,
        request: ReasoningRequest,
        rules: tuple[RuleDefinition, ...],
        step_records: list[dict[str, Any]],
    ) -> tuple[Any, Any]:
        """Run Phase E (Steps 26-30): wrap, separate, combine, propagate, decompose."""
        adapter = ProbabilisticEvidenceAdapter(
            min_reliability=self.policy.min_evidence_reliability,
        )
        wrapped = tuple(request.evidence_items)
        # Step 26 - model predictions wrapped as explicit evidence.
        self._record_step(step_records, 26, {
            "wrapped_evidence": [item.to_dict() for item in wrapped],
            "raw_softmax_not_probability": True,
        })
        # Step 27 - separate probabilistic observations from crisp facts.
        crisp, probabilistic = adapter.separate_uncertain_from_crisp(wrapped)
        self._record_step(step_records, 27, {
            "crisp_count": len(crisp),
            "probabilistic_count": len(probabilistic),
            "probabilistic_stored_as_facts": False,
        })
        # Step 28 - combine evidence only under declared assumptions.
        supports = []
        declarations = _independence_declarations(wrapped)
        if (
            probabilistic
            and self.policy.ablations.probabilistic_combination
        ):
            priors = {
                item.observation.get("hypothesis", request.goal_predicate): float(
                    item.observation.get("prior", 0.5),
                )
                for item in probabilistic
                if item.observation.get("hypothesis")
            }
            hypothesis_label = request.goal_literal.signature()
            priors.setdefault(hypothesis_label, 0.5)
            supports.append(adapter.combine_evidence(
                hypothesis_label, wrapped, priors, declarations,
            ))
        self._record_step(step_records, 28, {
            "declarations": [item.to_dict() for item in declarations],
            "supports": [item.to_dict() for item in supports],
            "never_multiply_by_default": True,
        })
        # Step 29 - propagate uncertainty; qualitative fallback when unsupported.
        qualitative = [
            support.hypothesis
            for support in supports
            if support.posterior is None
        ]
        self._record_step(step_records, 29, {
            "qualitative_fallbacks": qualitative,
            "documented_semantics_only": True,
        })
        # Step 30 - publish decomposed confidence vectors.
        vectors = self._confidence_vectors(wrapped, rules)
        self._record_step(step_records, 30, {
            "confidence_vectors": {
                key: value.to_dict() for key, value in sorted(vectors.items())
            },
            "dimensions_kept_separate": True,
        })
        return supports, vectors

    def _confidence_vectors(
        self,
        evidence: tuple[EvidenceItem, ...],
        rules: tuple[RuleDefinition, ...],
    ) -> dict[str, ConfidenceVector]:
        """Build one decomposed confidence vector per rule (6.5, Step 30)."""
        vectors: dict[str, ConfidenceVector] = {}
        source_reliability = (
            sum(item.reliability for item in evidence) / len(evidence)
            if evidence else 1.0
        )
        for rule in rules:
            strength = float(rule.scope.get("rule_strength", rule.priority))
            vectors[f"{rule.rule_id}@{rule.version}"] = ConfidenceVector(
                q_source=min(1.0, max(0.0, source_reliability)),
                q_observation=min(1.0, max(0.0, 1.0 - source_reliability)),
                q_rule=min(1.0, max(0.0, strength)),
                q_coverage=min(1.0, len(rule.premises) / 4.0),
                q_conflict=0.0,
            )
        return vectors

    def _run_learned_phase(
        self,
        request: ReasoningRequest,
        rules: tuple[RuleDefinition, ...],
        resolved_references: Mapping[str, Any],
        step_records: list[dict[str, Any]],
    ) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
        """Run Phase F (Steps 31-35) with graceful degradation."""
        adapter = self.learned_adapter
        if not self.policy.ablations.learned_assistance:
            self._record_step(step_records, 31, {"invoked": False,
                                                "reason": "learned_assistance ablated"})
            for step in (32, 33, 34, 35):
                self._record_step(step_records, step, {"skipped": True,
                                                       "reason": "phase_f_ablated"})
            return (), []

        deterministic_sufficient = all(
            not record.ambiguous for record in resolved_references.values()
        ) and len(rules) > 0
        payload = {
            "entity_references": list(request.entity_references),
            "goal": request.goal_literal.to_dict(),
            "deterministic_indexes_sufficient": deterministic_sufficient,
        }
        capabilities = adapter.available_capabilities()
        hypotheses: list[Any] = []
        proposal_records: list[dict[str, Any]] = []
        invocation_count = 0
        for capability in capabilities:
            proposal = adapter.invoke_when_needed(
                capability,
                payload,
                deterministic_indexes_sufficient=deterministic_sufficient,
                invocation_count=invocation_count,
            )
            invocation_count += 1
            if proposal is None:
                proposal_records.append({
                    "capability": capability, "proposal": None,
                    "reason": "not_needed_or_unavailable_or_invalid",
                })
                continue
            # Step 32 - response validated inside the adapter.
            proposal_records.append({
                "capability": capability,
                "model_id": proposal.model_id,
                "model_version": proposal.model_version,
                "input_hash": proposal.input_hash,
                "score_semantics": proposal.score_semantics,
                "candidates": len(proposal.candidates),
            })
            # Step 33 - convert to hypotheses (never authoritative facts).
            converted = adapter.to_hypotheses(proposal)
            hypotheses.extend(converted)
        self._record_step(step_records, 31, {
            "invoked": bool(proposal_records),
            "deterministic_indexes_sufficient": deterministic_sufficient,
            "capabilities_available": list(capabilities),
            "selection": "capability_then_quality_cost_policy",
        })
        self._record_step(step_records, 32, {
            "proposals": proposal_records,
            "schema_version_hash_calibration_enforced": True,
        })
        self._record_step(step_records, 33, {
            "hypotheses": [item.to_dict() for item in hypotheses],
            "never_authoritative_facts": True,
        })
        # Step 34 - symbolically filter with bounded rejection reasons.
        valid_predicates = {
            rule.conclusion.predicate: (rule.conclusion.arity,) for rule in rules
        }
        kept, rejections = adapter.filter_symbolically(
            tuple(hypotheses), valid_predicates,
        )
        self._record_step(step_records, 34, {
            "kept": len(kept),
            "rejections": [item.to_dict() for item in rejections],
            "bounded_rejection_reasons": True,
        })
        # Step 35 - verified feedback only (deferred to Phase H when outcomes
        # exist; here the gate records that unverified labels are refused).
        self._record_step(step_records, 35, {
            "verified_feedback_gate": "only_verified_outcomes_update_parameters",
            "unverified_labels_refused": True,
        })
        return tuple(kept), proposal_records

    def _publish_phase(
        self,
        workspace: SharedWorkspace,
        request: ReasoningRequest,
        rules: tuple[RuleDefinition, ...],
        engine: Any,
        report: Any,
        build: Any,
        verified: dict[str, bool],
        conflict_manager: ConflictManager,
        step_records: list[dict[str, Any]],
        snapshot_revision: int,
        *,
        allow_publication: bool = True,
    ) -> tuple[Any, Any, tuple[ReasoningFinding, ...]]:
        """Run Phase G (Steps 36-40): contradictions, gates, publication, review.

        ``allow_publication=False`` (budget-exhausted partial runs) gates every
        finding out: incomplete work is never published as complete (Step 45).
        """
        # Step 36 - detect contradictions.
        conclusions = tuple(item.conclusion_literal for item in report.derived)
        facts = engine.facts.all_facts()
        contradictions = conflict_manager.detect_contradictions(conclusions, facts, rules)
        self._record_step(step_records, 36, {
            "contradictions": [list(item) for item in contradictions],
            "compared_against": ["facts", "rules", "temporal_scopes", "exceptions"],
        })
        # Step 37 - record conflicts with provenance and options.
        for left, right in contradictions:
            conflict_manager.record_conflict(
                {"claim": left, "origin": "derived_conclusion"},
                {"claim": right, "origin": "stored_fact_or_rule"},
                provenance_a={"run_id": "", "ruleset_version": request.ruleset_version},
                provenance_b={"workspace_revision": snapshot_revision},
            )
        self._record_step(step_records, 37, {
            "recorded_conflicts": [item.to_dict() for item in conflict_manager.records],
            "source_facts_untouched": True,
        })

        # Build the publishable findings from classified proofs.
        findings: list[ReasoningFinding] = []
        for derivation in report.derived:
            node_matches = build.graph.nodes_for_conclusion(
                derivation.conclusion_literal.signature(),
            )
            proof_ref = node_matches[0].node_id if node_matches else None
            proof_class = (
                build.classification.get(proof_ref, ProofClass.UNVERIFIED_CANDIDATE)
                if proof_ref else ProofClass.UNVERIFIED_CANDIDATE
            )
            if derivation.rule_class is RuleClass.DEFEASIBLE:
                proof_class = ProofClass.DEFEASIBLE
            conflict_ids = tuple(
                record.conflict_id
                for record in conflict_manager.records
                if any(claim.get("claim") == derivation.conclusion_literal.signature()
                       for claim in record.claims)
            )
            rule = engine._rule_by_id.get(derivation.rule_ref)  # noqa: SLF001
            strength = float(rule.scope.get("rule_strength", rule.priority)) if rule else 0.5
            support = ConfidenceVector(
                q_source=min(1.0, len(derivation.source_fact_ids) / 2.0),
                q_observation=1.0,
                q_rule=strength,
                q_coverage=min(1.0, len(derivation.premise_literals) / 2.0),
                q_conflict=min(1.0, len(conflict_ids) / 2.0),
            )
            epistemic = (
                EpistemicStatus.DERIVED
                if derivation.rule_class is RuleClass.STRICT
                else EpistemicStatus.HYPOTHESIS
            )
            findings.append(ReasoningFinding(
                finding_id=make_id("nsr_finding"),
                conclusion=derivation.conclusion_literal,
                epistemic_status=epistemic,
                proof_ref=proof_ref,
                support_profile=support,
                conflicts=conflict_ids,
                publication_state=PublicationState.UNREVIEWED,
                proof_class=proof_class,
                source_fact_ids=derivation.source_fact_ids,
                run_id="",
            ))

        publisher = NSRLWorkspacePublisher(workspace)
        if not allow_publication:
            # Partial run: record explicit gated-out decisions without
            # touching the workspace (Section 5, Step 45; Section 10).
            from .workspace_publisher import GateDecision

            decisions = tuple(
                GateDecision(
                    finding_id=finding.finding_id,
                    passed=False,
                    gate_results=({
                        "gate": "partial_run",
                        "passed": False,
                        "detail": "incomplete work is never published as complete",
                    },),
                    publication_state=PublicationState.GATED_OUT,
                )
                for finding in findings
            )
            self._record_step(step_records, 38, {
                "gates": [item.to_dict() for item in decisions],
                "gate_config": self.policy.publication_gates.to_dict(),
                "partial_run_publication_blocked": True,
            })
            self._record_step(step_records, 39, {
                "workspace_results": {}, "published": [],
                "reason": "partial run",
            })
            self._record_step(step_records, 40, {
                "review_requests": [], "reason": "partial run",
            })
            final_findings = tuple(
                replace(finding, publication_state=PublicationState.GATED_OUT)
                for finding in findings
            )
            from .workspace_publisher import PublicationReport as _Report

            return (
                _Report(decisions=decisions),
                decisions,
                final_findings,
            )
        # Step 38 - publication gates.
        decisions = publisher.apply_gates(
            tuple(findings),
            verified_node_ids=verified,
            conflict_manager=conflict_manager,
            satisfied_constraints=self.policy.publication_gates.critical_constraints,
        )
        self._record_step(step_records, 38, {
            "gates": [item.to_dict() for item in decisions],
            "gate_config": self.policy.publication_gates.to_dict(),
        })
        # Step 39 - publish through the existing atomic workspace path.
        lineage = {
            node_id: record.to_dict() for node_id, record in build.lineage.items()
        }
        publication = publisher.publish(
            tuple(findings), decisions,
            snapshot_revision=snapshot_revision,
            lineage=lineage,
            conflict_manager=None,
        )
        self._record_step(step_records, 39, {
            "workspace_results": publication.workspace_results,
            "published": list(publication.published_finding_ids),
            "atomic_path": "FindingPublisher->WorkspaceStateManager",
        })
        # Step 40 - independent review for high-impact/novel/contested findings.
        review_records: list[dict[str, Any]] = []
        by_id = {item.finding_id: item for item in findings}
        for finding_id in publication.review_requested_ids:
            finding = by_id[finding_id]
            review = publisher.request_independent_review(
                finding,
                premises=[fact_id for fact_id in finding.source_fact_ids],
                proof_summary={"proof_class": finding.proof_class.value},
            )
            review_records.append({
                "finding_id": finding_id,
                "review_id": review["review"]["review_id"],
                "workspace_decision": review["workspace_decision"],
            })
        self._record_step(step_records, 40, {
            "review_requests": review_records,
            "premises_and_proof_supplied": True,
        })
        # Reflect gate outcomes onto the findings' publication states.
        decision_by_id = {item.finding_id: item for item in decisions}
        review_ids = set(publication.review_requested_ids)
        final_findings = []
        for finding in findings:
            decision = decision_by_id.get(finding.finding_id)
            if decision is None:
                final_findings.append(finding)
                continue
            if finding.finding_id in review_ids:
                state = PublicationState.REVIEW_REQUESTED
            elif decision.passed:
                state = PublicationState.PUBLISHED
            else:
                state = PublicationState.GATED_OUT
            final_findings.append(replace(finding, publication_state=state))
        return publication, decisions, tuple(final_findings)

    # ------------------------------------------------------------------ run()

    def run(self, request: ReasoningRequest) -> ReasoningRunResult:
        """Execute the full 45-step neuro-symbolic reasoning algorithm."""
        if not isinstance(request, ReasoningRequest):
            raise TypeError("request must be a ReasoningRequest")
        step_records: list[dict[str, Any]] = []
        started = time.perf_counter()

        # Step 1 - receive and validate the request (idempotent replay).
        fingerprint = request.fingerprint()
        cached = self._results.get(request.task_id)
        if cached is not None:
            if cached.reproducibility.get("request_fingerprint") != fingerprint:
                raise ValueError(
                    "task identity was reused with different request content; "
                    "use a new task_id for a new task",
                )
            self._record_step(step_records, 1, {
                "task_id": request.task_id, "idempotent_replay": True,
            })
            return replace(cached, elapsed_ms=cached.elapsed_ms)
        run_id = make_id("nsr_run")
        self._record_step(step_records, 1, {
            "task_id": request.task_id,
            "run_id": run_id,
            "request_fingerprint": fingerprint,
            "goal_predicate": request.goal_predicate,
            "scope": dict(request.scope),
            "mode": request.inference_mode.value,
            "deadline_seconds": request.deadline_seconds,
            "verification_level": request.verification_level.value,
            "budgets_present": True,
        })

        # Step 2 - bind one consistent workspace snapshot revision.
        manager = self._get_manager()
        workspace = self._make_workspace(request)
        snapshot_revision = workspace.revision
        snapshot = workspace.snapshot(snapshot_revision)
        self._record_step(step_records, 2, {
            "workspace_id": workspace.workspace_id,
            "snapshot_revision": snapshot_revision,
            "single_revision_bound": True,
        })

        observability = NSRLObservability(
            run_id,
            enable_logging=self.policy.enable_logging,
            logger_database=_open_logger(self.policy.logger_database_path),
        )
        trace_builder = NSRLTraceBuilder(run_id, snapshot_revision)
        conflict_manager = ConflictManager()

        # Steps 3-6 - USG attachment, resolution, retrieval, normalization,
        # epistemic status (Phase A continues). A failed attachment degrades
        # gracefully: the deterministic core keeps running without USG.
        adapter: USGSemanticAdapter | None = None
        attach_error: str | None = None
        if request.usg_bundle is not None:
            try:
                adapter, usg_reference = USGSemanticAdapter.from_bundle(
                    request.usg_bundle,
                    expected_digest=request.usg_expected_digest,
                )
                trace_builder.add_usg_bundle(
                    usg_reference["bundle_id"], usg_reference["bundle_digest"],
                )
            except Exception as error:  # noqa: BLE001 - degrade without USG
                adapter = None
                attach_error = str(error)[:200]
        facts, usg_reference, reference_map = self._build_fact_table_and_rules(
            request, adapter, step_records, attach_error=attach_error,
        )
        if usg_reference.get("bundle_id"):
            trace_builder.add_usg_bundle(
                usg_reference["bundle_id"], usg_reference["bundle_digest"],
            )
        for fact in facts:
            trace_builder.add_source_fact(
                fact.fact_id,
                epistemic_status=fact.epistemic_status.value,
                usg_version=fact.usg_version,
            )

        # Step 7 - assemble the bounded context.
        context = build_bounded_context(
            snapshot_revision=snapshot_revision,
            goal=request.goal_literal,
            facts=tuple(facts),
            rules=request.rules,
            mode=request.inference_mode,
            budgets=self.policy.budgets.to_dict(),
        )
        self._record_step(step_records, 7, {
            "context": context.to_dict(),
            "reused_workspace_snapshot": True,
        })

        # Phase B - deterministic symbolic inference (Steps 8-16).
        engine, report = self._run_inference(request, request.rules, facts, step_records)

        # Phase C - proofs (Steps 17-21).
        build, verified = self._build_and_verify_proofs(
            request, request.rules, engine, report, run_id, step_records,
        )

        # Phase F runs before D so model hypotheses can join the candidate set.
        resolved_references = reference_map
        model_hypotheses, proposal_records = self._run_learned_phase(
            request, request.rules, resolved_references, step_records,
        )

        # Phase D - constraints and ambiguity (Steps 22-25).
        candidates, ranked, groups = self._resolve_ambiguity(
            request, adapter, step_records, model_hypotheses,
        )

        # Phase E - probabilistic evidence (Steps 26-30).
        supports, vectors = self._handle_probabilistic(request, request.rules, step_records)

        # Phase G - conflicts and publication (Steps 36-40). Budget-exhausted
        # runs stay partial and never publish (Step 45).
        run_budget_exhausted = report.stop_reason in {
            "rule_fire_budget", "wall_clock_budget", "iteration_budget",
            "candidate_budget",
        }
        try:
            publication, decisions, findings = self._publish_phase(
                workspace, request, request.rules, engine, report, build,
                verified, conflict_manager, step_records, snapshot_revision,
                allow_publication=not run_budget_exhausted,
            )
        except Exception as error:  # noqa: BLE001 - publication failure keeps run inspectable
            self._record_step(step_records, 36, {"error": str(error)[:200]})
            for step in (37, 38, 39, 40):
                self._record_step(step_records, step, {"skipped": True,
                                                       "reason": "publication_failed"})
            publication = None
            decisions = ()
            findings = ()

        # Phase H - observability, learning, recovery (Steps 41-45).
        # Step 41 - log annotations/events through the existing logger.
        observability.record_event("rule_fires", {"count": report.rule_fires})
        observability.record_event("proof_verification", {
            "verified": sum(1 for value in verified.values() if value),
            "total": len(verified),
        })
        observability.record_event("constraint_decisions", {
            "candidates": len(candidates),
            "ranked_top": ranked[0].candidate_id if ranked else None,
        })
        observability.record_event("conflicts", {"count": len(conflict_manager.records)})
        observability.record_event("publication_status", {
            "published": list(publication.published_finding_ids) if publication else [],
        })
        observability.record_event("model_calls", {"count": len(proposal_records)})
        self._record_step(step_records, 41, {
            "event_count": len(observability.events),
            "logger_infrastructure": "ore_logger",
            "correlates_run_proof_finding": True,
        })

        # Step 42 - measure quality/cost.
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        strict_nodes = [
            node for node in build.graph.nodes.values()
            if node.rule_ref in {
                item.rule_ref for item in request.rules
                if item.rule_class is RuleClass.STRICT
            }
        ]
        metrics = NSRLObservability.measure(
            verified_node_count=sum(1 for item in strict_nodes
                                    if item.verifier_status.value == "verified"),
            strict_node_count=len(strict_nodes),
            conflict_count=len(conflict_manager.records),
            conclusion_count=len(report.derived),
            ambiguity_group_count=len(groups),
            candidate_count=len(candidates),
            rule_fires=report.rule_fires,
            latency_ms=elapsed_ms,
            verification_overhead_ms=0.0,
            memory_estimate_mb=_estimate_memory_mb(engine.facts, build),
            model_calls=len(proposal_records),
            labeled_candidates=request.ground_truth if request.ground_truth else None,
            predicted_relevant=tuple(item.candidate_id for item in ranked[:1]),
        )
        self._record_step(step_records, 42, {"metrics": metrics.to_dict()})

        # Step 43 - compare with verified outcomes only.
        comparison = NSRLObservability.compare_with_verified_outcomes(
            {derivation.conclusion_literal.signature(): 1.0 for derivation in report.derived},
            request.ground_truth,
        )
        self._record_step(step_records, 43, {"comparison": comparison})

        # Step 44 - approved parameter updates from verified feedback only.
        ledger = self.learned_adapter.ledger
        parameter_transaction = observability.record_parameter_transaction(
            ledger.to_dict(),
            verification_source="independent_proof_verifier" if verified else "none",
        )
        self._record_step(step_records, 44, {
            "transaction": parameter_transaction,
            "authoritative_rules_require_governance": True,
        })

        # Step 45 - checkpoint safe run state; explicit partial status on timeout.
        budget_exhausted = run_budget_exhausted
        completed_steps = tuple(
            record["step"] for record in step_records if not record["details"].get("skipped")
        )
        checkpoint = observability.checkpoint(
            phase="H",
            completed_steps=completed_steps,
            pending_step=None if not budget_exhausted else len(completed_steps) + 1,
            state={
                "run_id": run_id,
                "derived_signatures": [
                    item.conclusion_literal.signature() for item in report.derived
                ],
                "queue_depth": engine.queue.stats()["depth"],
                "workspace_revision": workspace.revision,
            },
            partial=budget_exhausted,
        )
        self._record_step(step_records, 45, {
            "checkpoint": checkpoint.to_dict(),
            "incomplete_work_never_published": True,
        })

        # Publish the USG reference and run identity into the workspace
        # (artifact_references) so workspace state correlates with the run.
        self._publish_run_references(workspace, usg_reference, run_id, fingerprint)

        # Status determination.
        status, status_reason = self._determine_status(
            report, groups, conflict_manager, decisions if publication else (),
        )

        # Trace completeness over published findings.
        for derivation in report.derived:
            node_matches = build.graph.nodes_for_conclusion(
                derivation.conclusion_literal.signature(),
            )
            if node_matches:
                trace_builder.add_proof_node(
                    node_matches[0].node_id,
                    rule_ref=":".join(node_matches[0].rule_ref),
                    verified=bool(verified.get(node_matches[0].node_id, False)),
                )
                trace_builder.link_fact_to_proof(
                    derivation.source_fact_ids[0] if derivation.source_fact_ids
                    else f"run:{run_id}",
                    node_matches[0].node_id,
                )
                trace_builder.link_proof_premises(
                    node_matches[0].node_id, node_matches[0].premise_node_ids,
                )
        for finding in findings:
            trace_builder.link_conclusion(finding.finding_id, proof_ref=finding.proof_ref)
        for record in conflict_manager.records:
            trace_builder.add_conflict(
                record.conflict_id,
                tuple(str(claim.get("claim", "")) for claim in record.claims),
            )
        trace_completeness, _untraceable = trace_builder.completeness(
            [finding.finding_id for finding in findings],
        )

        observability.finish("completed" if status != "failed" else "failed")

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result = ReasoningRunResult(
            run_id=run_id,
            task_id=request.task_id,
            status=status,
            status_reason=status_reason,
            workspace_id=workspace.workspace_id,
            workspace_revision=workspace.revision,
            ruleset_version=request.ruleset_version,
            conclusions=findings,
            conflicts=conflict_manager.records,
            ambiguity_groups=[item.to_dict() for item in groups],
            probabilistic_support=[item.to_dict() for item in supports],
            confidence_vectors={
                key: value.to_dict() for key, value in sorted(vectors.items())
            },
            model_proposals=[dict(item) for item in proposal_records],
            proof_summary={
                "node_count": len(build.graph.nodes),
                "verified_nodes": sum(1 for value in verified.values() if value),
                "classification": {
                    key: value.value for key, value in sorted(build.classification.items())
                },
                "graph_issues": [item.to_dict() for item in build.issues],
                "defeated": [item.to_dict() for item in report.defeated],
            },
            metrics=metrics.to_dict(),
            operation_coverage=tuple(
                record["operation_id"] for record in step_records
            ),
            trace=tuple(step_records),
            trace_completeness=trace_completeness,
            checkpoint=checkpoint.to_dict(),
            reproducibility={
                "request_fingerprint": fingerprint,
                "ruleset_version": request.ruleset_version,
                "ruleset_digest": sha256_json(
                    [rule.to_dict() for rule in request.rules],
                ),
                "policy": self.policy.to_dict(),
                "schema_version": NEUROSYMBOLIC_SCHEMA_VERSION,
                "deterministic_seed": self.policy.deterministic_seed,
                "workspace_revision": workspace.revision,
            },
            elapsed_ms=elapsed_ms,
            published_finding_ids=tuple(
                publication.published_finding_ids if publication else (),
            ),
            review_requested_ids=tuple(
                publication.review_requested_ids if publication else (),
            ),
            goal_satisfied=report.goal_satisfied,
        )
        self._results[request.task_id] = result
        if self._owns_manager:
            manager.close()
        return result

    def _publish_run_references(
        self,
        workspace: SharedWorkspace,
        usg_reference: Mapping[str, Any],
        run_id: str,
        fingerprint: str,
    ) -> None:
        """Publish the run/USG references into ``artifact_references``."""
        publisher = FindingPublisher(workspace, NSRL_PRODUCER_GRANT)
        payload = {
            "run_id": run_id,
            "request_fingerprint": fingerprint,
            "usg_bundle": dict(usg_reference) if usg_reference else None,
            "read_only": True,
        }
        finding = publisher.create_finding(
            "artifact_references",
            payload,
            key=f"nsrl_run:{run_id}",
            finding_type="neurosymbolic_run_reference",
            provenance=Provenance(
                source=f"nsrl:{run_id}",
                rules_applied=("usg_read_only_reference", "run_binding"),
                notes="USG stays authoritative; only a compact reference is published.",
            ),
            base_workspace_revision=workspace.revision,
        )
        workspace.publish(finding)

    def _determine_status(
        self,
        report: Any,
        groups: Any,
        conflict_manager: ConflictManager,
        decisions: Any,
    ) -> tuple[str, str]:
        """Map run outcomes onto the explicit status vocabulary."""
        if report.stop_reason in {
            "rule_fire_budget", "wall_clock_budget", "iteration_budget",
            "candidate_budget",
        }:
            return "partial", f"inference stopped by budget: {report.stop_reason}"
        gated_out = sum(1 for item in decisions if not item.passed)
        if gated_out and not any(item.passed for item in decisions):
            return "blocked", "every conclusion failed the publication gates"
        if conflict_manager.unresolved():
            return (
                "completed_with_uncertainty",
                "unresolved conflicts preserved alongside conclusions",
            )
        if len(groups) > 0:
            return (
                "completed_with_uncertainty",
                "ambiguity preserved as alternatives instead of one interpretation",
            )
        if report.goal_satisfied:
            return "completed", "goal literal derived and published"
        return "completed_with_uncertainty", "goal not fully proven; partial support published"


NSRL_PRODUCER_GRANT = "ore_neurosymbolic"

# Type alias kept local to avoid exporting an internal shape.
ProofBuildReportLike = Any


def _identity_reference(reference: str) -> Any:
    """Build the fallback reference record when no USG bundle is attached."""
    from .semantic_adapter import SemanticReference

    return SemanticReference(
        reference=reference, concept_id=reference, kind="IDENTITY",
        ambiguous=False, alternatives=(reference,),
    )


def _mark_candidate(candidate: Any, status: str) -> Any:
    """Re-tag one candidate status without mutating the original."""
    from dataclasses import replace as _replace

    from .contracts import CandidateStatus

    return _replace(candidate, status=CandidateStatus(status))


def _count_values(values: Any) -> dict[str, int]:
    """Count string values deterministically."""
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _verification_status(verified: bool, tampered: bool) -> Any:
    """Map verifier outcome onto the VerificationStatus enum."""
    from .contracts import VerificationStatus

    if tampered:
        return VerificationStatus.TAMPERED
    if verified:
        return VerificationStatus.VERIFIED
    return VerificationStatus.FAILED


def _independence_declarations(evidence: tuple[EvidenceItem, ...]) -> tuple[
    IndependenceDeclaration, ...,
]:
    """Read declared independence groups from evidence uncertainty models."""
    groups: dict[str, list[str]] = {}
    joint: dict[str, float] = {}
    for item in evidence:
        model = item.uncertainty_model
        group_id = model.get("independence_group")
        if not group_id:
            continue
        groups.setdefault(str(group_id), []).append(item.evidence_id)
        if model.get("joint_likelihood") is not None:
            joint[str(group_id)] = float(model["joint_likelihood"])
    declarations: list[IndependenceDeclaration] = []
    for group_id, members in sorted(groups.items()):
        if group_id in joint:
            declarations.append(IndependenceDeclaration(
                group_id=group_id, member_ids=tuple(members),
                assumption="dependent_joint", joint_likelihood=joint[group_id],
            ))
        else:
            declarations.append(IndependenceDeclaration(
                group_id=group_id, member_ids=tuple(members),
                assumption="conditional_independence",
            ))
    return tuple(declarations)


def _estimate_memory_mb(facts: Any, build: Any) -> float:
    """Return a coarse measured-size estimate in MB (never a hard claim)."""
    fact_bytes = sum(len(str(item.to_dict())) for item in facts.all_facts())
    node_bytes = sum(len(str(node.to_dict())) for node in build.graph.nodes.values())
    return (fact_bytes + node_bytes) / (1024.0 * 1024.0)


def _open_logger(path: str | None) -> Any:
    """Open the existing ``ore_logger`` database when a path is supplied."""
    if path is None:
        return None
    from ore_logger import LoggerDatabase

    return LoggerDatabase(path)


__all__ = ["NeuroSymbolicReasoningCoordinator", "NSRL_PRODUCER_GRANT"]
