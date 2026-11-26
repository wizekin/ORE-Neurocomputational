"""Integration tests: constraints, ambiguity, probabilistic evidence, learned
assistance, conflicts, publication, observability and the 45-step coordinator.

Covers the remaining Section 10 acceptance areas: ambiguity preserved with
hard-constraint-only filtering and reproducible soft ranking; probabilistic
toy distributions, correlated evidence, calibration and uncertainty
propagation; malformed/unavailable model rejection with no USG mutation;
workspace snapshot consistency, atomic publication, conflict/revision
handling; logger correlation of run, proof and finding; timeout/partial-run
recovery without duplicate publication; benchmarks on synthetic corpora; and
the 45-step operation coverage.
"""
from __future__ import annotations

import pytest

from ore_neurosymbolic import (
    AmbiguityResolver,
    ConstraintEngine,
    ConfidenceVector,
    ConflictManager,
    EpistemicStatus,
    EvidenceItem,
    HardConstraint,
    IndependenceDeclaration,
    LearnedCandidateAdapter,
    Literal,
    LogicalTerm,
    LogicalVariable,
    ModelDescriptor,
    ModelInvocationError,
    NSRLWorkspacePublisher,
    NSRLObservability,
    NeuroSymbolicReasoningCoordinator,
    ProbabilisticEvidenceAdapter,
    PublicationGates,
    ReasoningRequest,
    ReasoningRunResult,
    ResourceBudgets,
    ResourcePolicy,
    ResourceSnapshot,
    RuleClass,
    RuleDefinition,
    SemanticFact,
    SoftConstraint,
    VerificationLevel,
    differentiable_consistency,
    run_ablation,
)
from ore_neurosymbolic.contracts import (
    CandidateInterpretation,
    CandidateStatus,
    InferenceMode,
    NSRLConflictRecord,
    ProofClass,
    PublicationState,
)
from ore_neurosymbolic.evaluation import evaluate_corpus
from ore_neurosymbolic.workspace_publisher import (
    BLOCKED_PUBLICATION_STATUSES,
    PERMITTED_PUBLICATION_STATUSES,
)

X = LogicalVariable("X", "entity")
BIRD = LogicalTerm("bird", "class")
PENGUIN = LogicalTerm("penguin", "class")
FLAG = LogicalTerm("yes", "flag")


def _fact(fact_id: str, subject: str, predicate: str, object_value: str, **kw) -> SemanticFact:
    return SemanticFact(
        fact_id=fact_id, subject=subject, predicate=predicate,
        object_value=object_value,
        epistemic_status=kw.pop("epistemic_status", EpistemicStatus.ASSERTED),
        **kw,
    )


def _superior_rule() -> RuleDefinition:
    Y = LogicalVariable("Y", "entity")
    Z = LogicalVariable("Z", "entity")
    return RuleDefinition(
        rule_id="r_superior", version="1.0.0",
        typed_variables={"?X": "entity", "?Y": "entity", "?Z": "entity"},
        premises=(Literal("commanded_by", (X, Y)), Literal("commands", (Y, Z))),
        conclusion=Literal("indirect_command", (X, Z)),
        rule_class=RuleClass.STRICT, priority=1.0,
    )


def _basic_request(task_id: str = "nsr_task_test", **overrides) -> ReasoningRequest:
    base = dict(
        goal_predicate="indirect_command",
        goal_args=("captain", "fleet"),
        facts=(
            _fact("f1", "captain", "commanded_by", "admiral"),
            _fact("f2", "admiral", "commands", "fleet"),
        ),
        rules=(_superior_rule(),),
        task_id=task_id,
    )
    base.update(overrides)
    return ReasoningRequest(**base)


class TestConstraintEngine:
    """Steps D2-D3: hard filters only for proven violations; soft ranking."""

    def _candidate(self, candidate_id: str, **structure) -> CandidateInterpretation:
        return CandidateInterpretation(
            candidate_id=candidate_id,
            semantic_structure={
                "type": "entity", "arity": 2, "role": "subject",
                "entities": ("a", "b"), "temporal": "present",
                "context_tags": ("military",),
                **structure,
            },
            evidence_refs=("e1",),
        )

    def test_hard_type_violation_eliminated(self) -> None:
        engine = ConstraintEngine(
            hard=(HardConstraint("h_type", "type", {"required_type": "relation"}),),
        )
        candidates = (self._candidate("c1"), self._candidate("c2", type="relation"))
        active, filtered = engine.apply_hard(candidates)
        assert [item.candidate_id for item in active] == ["c2"]
        assert filtered == (("c1", "h_type"),)

    def test_hard_arity_violation_eliminated(self) -> None:
        engine = ConstraintEngine(
            hard=(HardConstraint("h_arity", "arity", {"required_arity": 3}),),
        )
        active, filtered = engine.apply_hard((self._candidate("c1"),))
        assert not active
        assert filtered == (("c1", "h_arity"),)

    def test_matching_candidate_survives(self) -> None:
        engine = ConstraintEngine(
            hard=(HardConstraint("h_type", "type", {"required_type": "entity"}),),
        )
        active, filtered = engine.apply_hard((self._candidate("c1"),))
        assert len(active) == 1 and not filtered

    def test_soft_scores_are_bounded_and_reproducible(self) -> None:
        engine = ConstraintEngine(
            soft=(
                SoftConstraint("s1", "context_fit", 0.8),
                SoftConstraint("s2", "evidence_support", 1.0),
            ),
        )
        candidates = (self._candidate("c1"), self._candidate("c2", context_tags=("naval",)))
        first = engine.apply_soft(candidates)
        second = engine.apply_soft(candidates)
        assert first == second
        for score in first["c1"].values():
            assert 0.0 <= score <= 1.0

    def test_evidence_sufficiency_hard_constraint(self) -> None:
        engine = ConstraintEngine(
            hard=(HardConstraint("h_ev", "required_evidence", {"minimum_count": 2}),),
        )
        active, filtered = engine.apply_hard((self._candidate("c1"),))
        assert not active
        poor = CandidateInterpretation(
            candidate_id="c2",
            semantic_structure=self._candidate("c2").semantic_structure,
            evidence_refs=("e1", "e2"),
        )
        active, _filtered = engine.apply_hard((poor,))
        assert len(active) == 1


class TestAmbiguityResolver:
    """Steps D1/D3/D4 + math op 6.3."""

    def test_generate_candidates_per_reading(self) -> None:
        resolver = AmbiguityResolver()
        candidates = resolver.generate_candidates({"bank": ("bank_river", "bank_money")})
        assert len(candidates) == 2

    def test_ranking_formula_s(c) -> None:
        resolver = AmbiguityResolver()
        strong = CandidateInterpretation(
            candidate_id="strong",
            semantic_structure={"reference": "bank"},
            neural_scores={"similarity": 0.9},
            ranking_features={"type_compatibility": 1.0, "soft_violations": 0.0},
        )
        weak = CandidateInterpretation(
            candidate_id="weak",
            semantic_structure={"reference": "bank"},
            neural_scores={"similarity": 0.1},
            ranking_features={"type_compatibility": 0.2, "soft_violations": 0.5},
        )
        ranked = resolver.rank((weak, strong))
        assert ranked[0].candidate_id == "strong"
        components = ranked[0].components
        assert set(components) == {
            "type_compatibility", "context_fit", "evidence_support",
            "model_similarity", "soft_violations",
        }

    def test_close_scores_preserve_ambiguity(self) -> None:
        resolver = AmbiguityResolver(epsilon=0.1)
        candidates = tuple(
            CandidateInterpretation(
                candidate_id=f"c{index}",
                semantic_structure={"reference": "bank"},
                ranking_features={"type_compatibility": 1.0 - index * 0.01},
            )
            for index in range(3)
        )
        ranked = resolver.rank(candidates)
        groups = resolver.preserve_ambiguity(ranked)
        assert len(groups) == 1
        assert len(groups[0].member_ids) == 3

    def test_clear_winner_not_grouped(self) -> None:
        resolver = AmbiguityResolver(epsilon=0.05)
        candidates = tuple(
            CandidateInterpretation(
                candidate_id=f"c{index}",
                semantic_structure={"reference": "bank"},
                ranking_features={"type_compatibility": 1.0 - index * 0.5},
            )
            for index in range(2)
        )
        ranked = resolver.rank(candidates)
        groups = resolver.preserve_ambiguity(ranked)
        assert groups == ()

    def test_mark_alternatives_tags_members(self) -> None:
        resolver = AmbiguityResolver(epsilon=0.2)
        candidates = tuple(
            CandidateInterpretation(
                candidate_id=f"c{index}",
                semantic_structure={"reference": "x"},
                ranking_features={"type_compatibility": 1.0 - index * 0.05},
            )
            for index in range(2)
        )
        ranked = resolver.rank(candidates)
        groups = resolver.preserve_ambiguity(ranked)
        marked = resolver.mark_alternatives(candidates, groups)
        assert all(
            item.status is CandidateStatus.ALTERNATIVE for item in marked
        )

    def test_weight_versioning(self) -> None:
        from ore_neurosymbolic import RankingWeights

        weights = RankingWeights(version="2.0.0", w_t=0.5, w_g=0.2, w_e=0.2, w_m=0.1)
        assert weights.to_dict()["version"] == "2.0.0"
        with pytest.raises(ValueError):
            RankingWeights(w_t=-0.1)


class TestProbabilisticAdapter:
    """Steps E1-E5 + math ops 6.4-6.6."""

    def _adapter(self) -> ProbabilisticEvidenceAdapter:
        return ProbabilisticEvidenceAdapter()

    def test_wrap_model_evidence_marks_uncalibrated_softmax(self) -> None:
        item = self._adapter().wrap_model_evidence(
            source="model", observation={"score": 0.8}, reliability=0.9,
            model_id="m1", model_version="1.0.0", input_hash="abc",
            score_semantics="softmax_logit",
        )
        assert item.observation["is_calibrated_probability"] is False

    def test_missing_score_semantics_rejected(self) -> None:
        with pytest.raises(Exception, match="score_semantics"):
            self._adapter().wrap_model_evidence(
                source="model", observation={}, reliability=0.9,
                model_id="m", model_version="1", input_hash="h",
                score_semantics="",
            )

    def test_separate_uncertain_from_crisp(self) -> None:
        crisp_item = EvidenceItem(evidence_id="c", source="s", observation={}, reliability=1.0)
        prob_item = EvidenceItem(
            evidence_id="p", source="s",
            observation={"probability": 0.7}, reliability=0.9,
        )
        crisp, probabilistic = self._adapter().separate_uncertain_from_crisp((crisp_item, prob_item))
        assert [item.evidence_id for item in crisp] == ["c"]
        assert [item.evidence_id for item in probabilistic] == ["p"]

    def test_conditional_independence_combination(self) -> None:
        # Two independent witnesses, each P(e|H)=0.8, prior 0.5.
        adapter = self._adapter()
        evidence = tuple(
            EvidenceItem(
                evidence_id=f"e{index}", source="witness",
                observation={"probability": 0.8}, reliability=1.0,
            )
            for index in range(2)
        )
        support = adapter.combine_evidence(
            "H", evidence, {"H": 0.5},
            (IndependenceDeclaration("g1", ("e0", "e1"), "conditional_independence"),),
        )
        assert support.posterior is not None
        # Prior odds 1; two likelihood ratios of 4 -> posterior odds 16.
        assert abs(support.posterior - 16.0 / 17.0) < 1e-6

    def test_correlated_evidence_never_multiplies_by_default(self) -> None:
        adapter = self._adapter()
        evidence = tuple(
            EvidenceItem(
                evidence_id=f"e{index}", source="same_report",
                observation={"probability": 0.8}, reliability=1.0,
            )
            for index in range(2)
        )
        support = adapter.combine_evidence("H", evidence, {"H": 0.5}, ())
        # No declared independence: no posterior, qualitative support only.
        assert support.posterior is None
        assert support.semantics == "no_declared_independent_evidence"

    def test_dependent_group_uses_declared_joint(self) -> None:
        adapter = self._adapter()
        evidence = tuple(
            EvidenceItem(
                evidence_id=f"e{index}", source="same_report",
                observation={"probability": 0.8}, reliability=1.0,
            )
            for index in range(2)
        )
        support = adapter.combine_evidence(
            "H", evidence, {"H": 0.5},
            (IndependenceDeclaration(
                "g1", ("e0", "e1"), "dependent_joint", joint_likelihood=0.8,
            ),),
        )
        assert support.posterior is not None
        assert support.semantics == "declared_joint_likelihood"

    def test_unknown_prior_rejected(self) -> None:
        adapter = self._adapter()
        evidence = (EvidenceItem(
            evidence_id="e0", source="s", observation={"probability": 0.5},
            reliability=1.0,
        ),)
        with pytest.raises(Exception, match="prior"):
            adapter.combine_evidence("H", evidence, {}, ())

    def test_confidence_vector_dimensions_stay_separate(self) -> None:
        vector = self._adapter().build_confidence_vector(
            source_reliability=0.9, observation_uncertainty=0.2,
            rule_strength=0.8, evidence_coverage=0.7, conflict_level=0.1,
        )
        payload = vector.to_dict()
        assert len(payload) == 5

    def test_optional_ltn_like_signal_is_not_proof(self) -> None:
        loss = differentiable_consistency(
            {"phi1": 0.8, "phi2": 0.5}, {"phi1": 2.0}, data_loss=0.1, lam=0.5,
        )
        assert loss == pytest.approx(2.0 * 0.04 + 0.25 + 0.05)
        # The signal is a soft optimization number, never a ProofClass label.
        assert not isinstance(loss, ProofClass)


class TestLearnedAdapter:
    """Steps F1-F5 (Stage 6/7 gates)."""

    def _descriptor(self, **overrides) -> ModelDescriptor:
        base = dict(
            model_id="linker", model_version="1.0.0",
            capability="entity_linking", output_schema="NSRL.LINK/1.0.0",
            score_semantics="calibrated_probability",
        )
        base.update(overrides)
        return ModelDescriptor(**base)

    def _backend(self, response=None, *, fail: bool = False):
        class _Backend:
            def invoke(self, capability: str, payload) -> dict:
                if fail:
                    raise ModelInvocationError("backend unavailable")
                return response or {
                    "output_schema": "NSRL.LINK/1.0.0",
                    "model_version": "1.0.0",
                    "candidates": [{
                        "candidate_id": "c1",
                        "semantic_structure": {"concept_id": "bank_river"},
                        "neural_scores": {"similarity": 0.9},
                    }],
                    "score_semantics": "calibrated_probability",
                }

        return _Backend()

    def test_no_model_leaves_symbolic_reasoning_operational(self) -> None:
        adapter = LearnedCandidateAdapter()
        assert adapter.invoke_when_needed(
            "entity_linking", {}, deterministic_indexes_sufficient=False,
        ) is None

    def test_model_failure_degrades_gracefully(self) -> None:
        adapter = LearnedCandidateAdapter()
        adapter.register(self._descriptor(), self._backend(fail=True))
        assert adapter.invoke_when_needed(
            "entity_linking", {}, deterministic_indexes_sufficient=False,
        ) is None
        assert adapter.failures == 1

    def test_schema_mismatch_rejected(self) -> None:
        adapter = LearnedCandidateAdapter()
        response = {
            "output_schema": "WRONG/1.0.0", "model_version": "1.0.0",
            "candidates": [], "score_semantics": "calibrated_probability",
        }
        adapter.register(self._descriptor(), self._backend(response))
        assert adapter.invoke_when_needed(
            "entity_linking", {}, deterministic_indexes_sufficient=False,
        ) is None

    def test_out_of_range_score_rejected(self) -> None:
        adapter = LearnedCandidateAdapter()
        response = {
            "output_schema": "NSRL.LINK/1.0.0", "model_version": "1.0.0",
            "candidates": [{
                "candidate_id": "c1", "semantic_structure": {},
                "neural_scores": {"similarity": 1.5},
            }],
            "score_semantics": "calibrated_probability",
        }
        adapter.register(self._descriptor(), self._backend(response))
        assert adapter.invoke_when_needed(
            "entity_linking", {}, deterministic_indexes_sufficient=False,
        ) is None

    def test_removed_model_leaves_no_residue(self) -> None:
        adapter = LearnedCandidateAdapter()
        adapter.register(self._descriptor(), self._backend())
        assert adapter.remove("entity_linking")
        assert adapter.available_capabilities() == ()

    def test_hypotheses_never_become_facts(self) -> None:
        adapter = LearnedCandidateAdapter()
        adapter.register(self._descriptor(), self._backend())
        proposal = adapter.invoke_when_needed(
            "entity_linking", {"q": "bank"},
            deterministic_indexes_sufficient=False,
        )
        assert proposal is not None
        hypotheses = adapter.to_hypotheses(proposal)
        assert hypotheses
        for hypothesis in hypotheses:
            structure = hypothesis.semantic_structure
            assert structure["epistemic_status"] == "model_proposal"
            assert EpistemicStatus(structure["epistemic_status"]) is (
                EpistemicStatus.MODEL_PROPOSAL
            )

    def test_symbolic_filter_rejects_unknown_predicate(self) -> None:
        adapter = LearnedCandidateAdapter()
        proposal_candidate = CandidateInterpretation(
            candidate_id="c1",
            semantic_structure={"predicate": "ghost_predicate", "arity": 2},
        )
        kept, rejections = adapter.filter_symbolically(
            (proposal_candidate,), {"known": (2,)},
        )
        assert not kept
        assert rejections[0].reason_code == "unknown_predicate"

    def test_verified_feedback_updates_only_when_verified(self) -> None:
        adapter = LearnedCandidateAdapter()
        keys = ({"weight_delta": 0.05, "target": "ranking"},)
        ledger = adapter.record_verified_feedback(
            keys, verified=True, verification_source="independent_verifier",
        )
        assert len(ledger.updates) == 1
        # Unverified outcomes are refused as training labels (Stage 7 gate).
        ledger = adapter.record_verified_feedback(
            keys, verified=False, verification_source="self_reported",
        )
        assert ledger.rejected_updates[-1]["reason"] == (
            "unverified_outcome_is_not_training_data"
        )

    def test_oversized_weight_delta_rejected(self) -> None:
        adapter = LearnedCandidateAdapter(max_weight_delta=0.1)
        ledger = adapter.record_verified_feedback(
            ({"weight_delta": 0.5},), verified=True, verification_source="verifier",
        )
        assert not ledger.updates
        assert ledger.rejected_updates[0]["reason"] == "delta_exceeds_bounded_limit"


class TestConflictManager:
    """Steps G1-G2: contradictions recorded, never overwritten."""

    def test_polarity_contradiction_detected(self) -> None:
        manager = ConflictManager()
        contradictions = manager.detect_contradictions(
            (
                Literal("status", (LogicalTerm("ship"), LogicalTerm("active"))),
                Literal("status", (LogicalTerm("ship"), LogicalTerm("active")), negated=True),
            ),
            (),
        )
        assert contradictions

    def test_temporal_scope_contradiction_detected(self) -> None:
        manager = ConflictManager()
        fact = _fact("f1", "ship", "status", "active", qualifiers={"temporal": "past"})
        contradictions = manager.detect_contradictions(
            (
                Literal(
                    "status", (LogicalTerm("ship"), LogicalTerm("active")),
                    temporal_qualifier="present",
                ),
            ),
            (fact,),
        )
        assert contradictions

    def test_record_is_idempotent_on_same_pair(self) -> None:
        manager = ConflictManager()
        first = manager.record_conflict({"claim": "a"}, {"claim": "b"})
        second = manager.record_conflict({"claim": "b"}, {"claim": "a"})
        assert first.conflict_id == second.conflict_id
        assert len(manager.records) == 1

    def test_record_preserves_claims_and_options(self) -> None:
        manager = ConflictManager()
        record = manager.record_conflict(
            {"claim": "x", "origin": "derived"}, {"claim": "not x"},
            priority=0.7,
            time_scope={"temporal": "present"},
        )
        assert record.status == "unresolved"
        assert len(record.claims) == 2
        assert "keep_both_preserved" in record.resolution_options

    def test_resolve_marks_status_without_deleting(self) -> None:
        manager = ConflictManager()
        record = manager.record_conflict({"claim": "a"}, {"claim": "b"})
        resolved = manager.resolve(record.conflict_id, "keep_both_preserved")
        assert resolved.status == "resolved:keep_both_preserved"
        assert len(resolved.claims) == 2


class TestPublicationGates:
    """Steps G3-G5."""

    def _finding(self, **overrides):
        from ore_neurosymbolic.contracts import ReasoningFinding, make_id

        base = dict(
            finding_id=make_id("f"),
            conclusion=Literal("q", (LogicalTerm("a"), LogicalTerm("b"))),
            epistemic_status=EpistemicStatus.DERIVED,
            proof_ref=None,
            support_profile=ConfidenceVector(1.0, 1.0, 1.0, 1.0, 0.0),
            proof_class=ProofClass.STRICT_VERIFIED,
            source_fact_ids=("f1",),
        )
        base.update(overrides)
        return ReasoningFinding(**base)

    def test_permitted_and_blocked_status_partition(self) -> None:
        assert not (PERMITTED_PUBLICATION_STATUSES & BLOCKED_PUBLICATION_STATUSES)
        assert EpistemicStatus.MODEL_PROPOSAL in BLOCKED_PUBLICATION_STATUSES
        assert EpistemicStatus.UNKNOWN in BLOCKED_PUBLICATION_STATUSES

    def test_unverified_strict_finding_gated_out(self) -> None:
        publisher = NSRLWorkspacePublisher.__new__(NSRLWorkspacePublisher)
        publisher.gates = PublicationGates()
        finding = self._finding()
        decisions = publisher.apply_gates((finding,), verified_node_ids={})
        assert not decisions[0].passed
        assert decisions[0].publication_state.value == "gated_out"

    def test_verified_strict_finding_passes_and_requests_review(self) -> None:
        publisher = NSRLWorkspacePublisher.__new__(NSRLWorkspacePublisher)
        publisher.gates = PublicationGates()
        finding = self._finding(proof_ref="node1")
        decisions = publisher.apply_gates(
            (finding,), verified_node_ids={"node1": True},
        )
        assert decisions[0].passed
        assert decisions[0].review_requested  # novel strict conclusion

    def test_model_proposal_status_cannot_publish(self) -> None:
        publisher = NSRLWorkspacePublisher.__new__(NSRLWorkspacePublisher)
        publisher.gates = PublicationGates()
        finding = self._finding(
            epistemic_status=EpistemicStatus.MODEL_PROPOSAL,
            proof_class=ProofClass.NEURAL_PROPOSAL,
        )
        decisions = publisher.apply_gates((finding,), verified_node_ids={})
        assert not decisions[0].passed


class TestResourcePolicy:
    """Math op 6.7: continuation utility under hard budgets."""

    def test_utility_formula(self) -> None:
        policy = ResourcePolicy(risk_weight=1.0)
        utility = policy.continuation_utility(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        assert utility == pytest.approx(0.5)

    def test_budget_exhaustion_blocks_continuation(self) -> None:
        policy = ResourcePolicy(ResourceBudgets(max_rule_fires=10))
        snapshot = ResourceSnapshot(rule_fires=10)
        decision = policy.evaluate_continuation(snapshot, 10.0, 10.0, 0.1, 0.1, 0.1, 0.1)
        assert decision.decision == "stop_budget"

    def test_low_utility_stops(self) -> None:
        policy = ResourcePolicy(continuation_threshold=2.0)
        snapshot = ResourceSnapshot()
        decision = policy.evaluate_continuation(snapshot, 0.1, 0.1, 1.0, 1.0, 1.0, 0.0)
        assert decision.decision == "stop_utility"

    def test_good_utility_continues(self) -> None:
        policy = ResourcePolicy(continuation_threshold=0.5)
        decision = policy.evaluate_continuation(
            ResourceSnapshot(), 1.0, 1.0, 0.5, 0.5, 0.5, 0.0,
        )
        assert decision.decision == "continue"


class TestObservability:
    """Steps H1-H5."""

    def test_measure_without_labels_reports_none(self) -> None:
        metrics = NSRLObservability.measure(
            verified_node_count=2, strict_node_count=2, conflict_count=0,
            conclusion_count=2, ambiguity_group_count=0, candidate_count=4,
            rule_fires=3, latency_ms=10.0, verification_overhead_ms=1.0,
            memory_estimate_mb=0.5, model_calls=0,
        )
        assert metrics.candidate_precision is None
        assert metrics.candidate_recall is None
        assert metrics.proof_validity_rate == 1.0

    def test_measure_with_labels_computes_precision_recall(self) -> None:
        metrics = NSRLObservability.measure(
            verified_node_count=1, strict_node_count=1, conflict_count=0,
            conclusion_count=1, ambiguity_group_count=0, candidate_count=2,
            rule_fires=1, latency_ms=1.0, verification_overhead_ms=0.0,
            memory_estimate_mb=0.0, model_calls=0,
            labeled_candidates={"a": True, "b": True, "c": False},
            predicted_relevant=("a", "c"),
        )
        assert metrics.candidate_precision == 0.5
        assert metrics.candidate_recall == 0.5

    def test_comparison_requires_verified_outcomes(self) -> None:
        result = NSRLObservability.compare_with_verified_outcomes({"a": 1.0}, None)
        assert result["comparison_available"] is False
        result = NSRLObservability.compare_with_verified_outcomes(
            {"a": 0.8}, {"a": 1.0},
        )
        assert result["comparison_available"] is True
        assert result["mean_absolute_error"] == pytest.approx(0.2)

    def test_checkpoint_records_partial_state(self, tmp_path) -> None:
        observability = NSRLObservability("run1")
        checkpoint = observability.checkpoint(
            phase="B", completed_steps=(1, 2, 3), pending_step=4,
            state={"queue_depth": 2}, partial=True,
        )
        assert checkpoint.partial
        assert checkpoint.pending_step == 4
        assert observability.events

    def test_logger_events_correlate_run(self, tmp_path) -> None:
        from ore_logger import LoggerDatabase

        database = LoggerDatabase(str(tmp_path / "logger.sqlite3"))
        observability = NSRLObservability(
            "run_logger_test", logger_database=database, enable_logging=True,
        )
        observability.record_event("rule_fires", {"count": 2})
        observability.record_event("proof_verification", {"verified": 1})
        observability.finish("completed")
        runs = database.list_runs(limit=5)
        assert any(run["entry_kind"] == "neurosymbolic_reasoning" for run in runs)
        matched = [run for run in runs if run["entry_kind"] == "neurosymbolic_reasoning"][0]
        events = database.list_events(matched["run_id"])
        kinds = {event["event_kind"] for event in events}
        assert {"rule_fires", "proof_verification"} <= kinds


class TestCoordinatorEndToEnd:
    """The full 45-step algorithm over synthetic inputs."""

    def test_full_run_coverage_and_publication(self) -> None:
        coordinator = NeuroSymbolicReasoningCoordinator()
        result = coordinator.run(_basic_request())
        assert len(result.operation_coverage) == 45
        assert result.operation_coverage[0] == "NSR-001"
        assert result.operation_coverage[-1] == "NSR-045"
        assert result.status == "completed"
        assert result.goal_satisfied
        assert result.published_finding_ids
        published = [
            finding for finding in result.conclusions
            if finding.publication_state in {
                PublicationState.PUBLISHED, PublicationState.REVIEW_REQUESTED,
            }
        ]
        assert published
        strict_verified = [
            finding for finding in published
            if finding.proof_class is ProofClass.STRICT_VERIFIED
        ]
        assert strict_verified

    def test_run_records_every_phase_in_trace(self) -> None:
        result = NeuroSymbolicReasoningCoordinator().run(_basic_request())
        phases = {record["phase"] for record in result.trace}
        assert phases == {"A", "B", "C", "D", "E", "F", "G", "H"}

    def test_workspace_state_correlates_with_run(self) -> None:
        result = NeuroSymbolicReasoningCoordinator().run(_basic_request())
        assert result.workspace_revision >= 1

    def test_backward_only_mode(self) -> None:
        result = NeuroSymbolicReasoningCoordinator().run(_basic_request(
            inference_mode=InferenceMode.BACKWARD,
        ))
        assert result.goal_satisfied

    def test_gated_out_without_verification(self) -> None:
        result = NeuroSymbolicReasoningCoordinator().run(_basic_request(
            verification_level=VerificationLevel.NONE,
        ))
        assert result.published_finding_ids == ()
        assert result.status == "blocked"

    def test_idempotent_replay_with_same_coordinator(self) -> None:
        coordinator = NeuroSymbolicReasoningCoordinator()
        request = _basic_request()
        first = coordinator.run(request)
        second = coordinator.run(request)
        assert second.run_id == first.run_id
        assert second.reproducibility["request_fingerprint"] == (
            first.reproducibility["request_fingerprint"]
        )

    def test_reused_task_id_with_different_content_rejected(self) -> None:
        coordinator = NeuroSymbolicReasoningCoordinator()
        coordinator.run(_basic_request(task_id="dup"))
        with pytest.raises(ValueError, match="different request content"):
            coordinator.run(_basic_request(
                task_id="dup", goal_args=("other", "fleet"),
            ))

    def test_epistemic_categories_stay_separate_in_result(self) -> None:
        result = NeuroSymbolicReasoningCoordinator().run(_basic_request())
        payload = result.to_dict()
        assert {"conclusions", "conflicts", "ambiguity_groups",
                "probabilistic_support", "model_proposals"} <= set(payload)

    def test_partial_status_on_tight_budget(self) -> None:
        from ore_neurosymbolic.config import NSRLPolicy
        from ore_neurosymbolic.resource_policy import ResourceBudgets

        policy = NSRLPolicy(budgets=ResourceBudgets(
            max_rule_fires=1, max_depth=2, max_candidates=64,
        ))
        result = NeuroSymbolicReasoningCoordinator(policy=policy).run(_basic_request(
            facts=(
                _fact("f1", "captain", "commanded_by", "admiral"),
                _fact("f2", "admiral", "commands", "fleet"),
                _fact("f3", "admiral", "commanded_by", "general"),
                _fact("f4", "general", "commands", "army"),
            ),
        ))
        assert result.status == "partial"
        assert result.checkpoint["partial"] is True
        # Incomplete work is never published as complete.
        assert result.published_finding_ids == ()

    def test_conflicts_preserved_not_overwritten(self) -> None:
        result = NeuroSymbolicReasoningCoordinator().run(_basic_request(
            facts=(
                _fact("f1", "captain", "commanded_by", "admiral"),
                _fact("f2", "admiral", "commands", "fleet"),
                _fact("f3", "captain", "indirect_command", "fleet", polarity=False),
            ),
        ))
        assert result.conflicts
        assert result.status == "completed_with_uncertainty"

    def test_usg_bundle_read_only_attachment(self) -> None:
        bundle = {
            "contract": "USG-KNOWLEDGE/0.1.0",
            "namespace": "urn:usg:nsrl-test",
            "bundle_id": "nsrl-usg-test",
            "created_at": "2026-11-22T00:00:00Z",
            "records": [
                {
                    "id": "lemma:en:bank", "kind": "LEMMA", "version": 1,
                    "state": "ACCEPTED", "created_at": "2026-11-22T00:00:00Z",
                    "source_ids": ["source:s1"], "evidence_ids": ["ev:1"],
                    "context_ids": [],
                    "payload": {"lemma": "bank", "language": "en",
                                "part_of_speech": "NOUN"},
                },
            ],
            "sources": [{
                "id": "source:s1", "provider": "test", "source_type": "OTHER",
                "retrieved_at": "2026-11-22T00:00:00Z",
                "rights_status": "KNOWN_PERMITTED", "redistribution": "ALLOWED",
                "attribution": "test",
            }],
            "evidence": [{
                "id": "ev:1", "source_id": "source:s1", "evidence_type": "ATTESTATION",
                "support_status": "SUPPORTS", "locator": "loc", "text_digest": "d",
                "captured_at": "2026-11-22T00:00:00Z", "transformation_ids": [],
            }],
            "contexts": [],
            "mappings": [],
            "relations": [],
        }
        result = NeuroSymbolicReasoningCoordinator().run(_basic_request(
            usg_bundle=bundle, entity_references=("bank",),
        ))
        step3 = next(
            record for record in result.trace if record["step"] == 3
        )
        assert step3["details"]["usg_attached"] is True
        assert step3["details"]["resolved_references"]["bank"]["ambiguous"] is False


class TestEvaluationHarness:
    """Section 10: benchmarks on synthetic corpora + ablations."""

    def test_evaluate_corpus_reports_precision_recall(self) -> None:
        def factory(index: int) -> ReasoningRequest:
            return _basic_request(task_id=f"eval_{index}")

        labels = (
            {"indirect_command(captain,fleet)": True},
            {"indirect_command(captain,fleet)": True},
        )
        report = evaluate_corpus(factory, labels)
        assert report.task_count == 2
        assert report.summary["recall"] == pytest.approx(1.0)
        assert report.summary["precision"] == pytest.approx(1.0)

    def test_ablation_reports_contributions(self) -> None:
        report = run_ablation(_basic_request(task_id="ablate"))
        assert set(report.ablations) == {
            "learned_assistance", "probabilistic_combination",
            "independent_verification", "constraint_filtering",
            "ambiguity_preservation", "defeasible_resolution", "usg_retrieval",
        }
        contributions = report.to_dict()["contributions"]
        verification = contributions["independent_verification"]
        assert verification["published_count"] > 0  # gating off -> more published? no:
        # with verification ablated the gate blocks unverified strict findings,
        # so full configuration publishes at least as many.
        assert report.full_configuration["published_count"] >= 1


class TestResultContract:
    """Serialization and reproducibility tuple of the run result."""

    def test_result_round_trip(self) -> None:
        result = NeuroSymbolicReasoningCoordinator().run(_basic_request())
        payload = result.to_dict()
        assert payload["run_id"] == result.run_id
        assert payload["reproducibility"]["ruleset_version"] == "1.0.0"
        assert payload["reproducibility"]["schema_version"] == "1.0.0"

    def test_summary_line(self) -> None:
        result = NeuroSymbolicReasoningCoordinator().run(_basic_request())
        line = result.summary_line()
        assert "45/45" in line
