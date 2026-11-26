"""Unit tests for the ORE blackboard coordination contracts and math operations."""
from __future__ import annotations

import pytest

from ore_blackboard.config import (
    AblationSwitches,
    BlackboardPolicy,
    ComparisonCoefficients,
    CycleBudgets,
)
from ore_blackboard.contracts import (
    BlackboardLayer,
    BlackboardTaskRequest,
    FailureClassification,
    InputRoot,
    InputSource,
    LifecycleState,
    TokenSpan,
    build_input_root,
    layer_for_field,
    one_hot_layer,
)
from ore_blackboard.hypotheses import BayesianHypothesisUpdater, SCORE_DEFINITION
from ore_blackboard.relevance import (
    FieldChange,
    measure_workspace_changes,
    relevance_report,
)
from ore_blackboard.registry import (
    BLACKBOARD_OPERATIONS,
    BLACKBOARD_OPERATION_IDS,
    operation_by_id,
    operation_by_step,
)
from ore_blackboard.resources import (
    BlackboardResourceCoordinator,
    compute_comparison_score,
    exponential_backoff,
)
from ore_workspace.evidence import EvidenceContribution
from ore_workspace.evidence import HypothesisScorer


def _state(**overrides):
    """Create a minimal workspace-like state mapping for change measurements."""
    base = {
        "current_input": "The boy opened the door.",
        "current_goal": "understand_sentence",
        "lexical_hypotheses": {},
        "syntactic_hypotheses": [],
        "semantic_hypotheses": [],
        "discourse_context": {},
        "working_memory": {},
        "retrieved_memories": [],
        "unresolved_questions": [],
        "module_status": {},
        "artifact_references": [],
    }
    base.update(overrides)
    return base


class TestOperationRegistry:
    """The 32-step algorithm must be covered by stable operation IDs."""

    def test_registry_defines_all_thirty_two_steps(self):
        """All 32 steps of Algorithm 1 have stable operation IDs and purposes."""
        assert len(BLACKBOARD_OPERATIONS) == 32
        assert len(set(BLACKBOARD_OPERATION_IDS)) == 32
        assert BLACKBOARD_OPERATION_IDS[0] == "BBA-001"
        assert BLACKBOARD_OPERATION_IDS[-1] == "BBA-032"

    def test_operation_lookup_round_trips(self):
        """Step numbers and operation IDs resolve to the same specification."""
        for step in range(1, 33):
            spec = operation_by_step(step)
            assert spec.step == step
            assert operation_by_id(spec.operation_id) is spec
            assert spec.title.strip() and spec.purpose.strip()

    def test_operation_lookup_rejects_unknown_ids(self):
        """Unknown step numbers and operation IDs are rejected."""
        with pytest.raises(ValueError):
            operation_by_step(0)
        with pytest.raises(ValueError):
            operation_by_step(33)
        with pytest.raises(ValueError):
            operation_by_id("BBA-999")


class TestTaskRequestValidation:
    """Step 1 requires typed validation, finite scalars, and an idempotency key."""

    def test_request_rejects_empty_goal_and_task_id(self):
        """Empty goals and task identities are rejected before any execution."""
        with pytest.raises(ValueError):
            BlackboardTaskRequest(input="text", goal="  ")
        with pytest.raises(ValueError):
            BlackboardTaskRequest(input="text", goal="g", task_id=" ")

    def test_request_rejects_unbounded_priority(self):
        """Priorities are normalized to [0, 1]."""
        with pytest.raises(ValueError):
            BlackboardTaskRequest(input="text", goal="g", priority=1.5)
        with pytest.raises(ValueError):
            BlackboardTaskRequest(input="text", goal="g", priority=float("inf"))

    def test_request_rejects_capability_overlap_and_invalid_mode(self):
        """A capability cannot be both required and optional; modes are checked."""
        with pytest.raises(ValueError):
            BlackboardTaskRequest(
                input="text", goal="g",
                required_capabilities=("a.b",), optional_capabilities=("a.b",),
            )
        with pytest.raises(ValueError):
            BlackboardTaskRequest(input="text", goal="g", execution_mode="UNKNOWN")

    def test_request_fingerprint_is_deterministic_and_content_sensitive(self):
        """The request fingerprint is stable and changes with input content."""
        first = BlackboardTaskRequest(input="The boy opened the door.", goal="g", task_id="t")
        second = BlackboardTaskRequest(input="The boy opened the door.", goal="g", task_id="t")
        third = BlackboardTaskRequest(input="A different input.", goal="g", task_id="t")
        assert first.request_fingerprint() == second.request_fingerprint()
        assert first.request_fingerprint() != third.request_fingerprint()


class TestInputRoot:
    """Step 4 requires an immutable, digest-verified evidence root."""

    def test_input_root_digests_and_preserves_spans(self):
        """The input digest is stable and token spans stay within the source text."""
        root = build_input_root("The boy opened the door.")
        again = build_input_root("The boy opened the door.")
        assert root.input_digest == again.input_digest
        assert all(0 <= span.start <= span.end <= 27 for span in root.tokens)
        assert [span.token for span in root.tokens] == [
            "The", "boy", "opened", "the", "door",
        ]

    def test_input_root_keeps_multiple_sources_independent(self):
        """Multiple inputs keep separate identities and source-group labels."""
        root = build_input_root({
            "a": "The fisherman saw the bank.",
            "b": "The captain opened the door.",
        })
        assert {source.source_id for source in root.sources} == {"a", "b"}
        assert root.source_group_of("a") == "default"
        assert root.source_group_of("b") == "default"
        assert root.primary_text() == "The fisherman saw the bank."

    def test_input_root_rejects_invalid_spans(self):
        """Span bounds are validated against the source length."""
        with pytest.raises(ValueError):
            InputRoot(
                sources=(InputSource("s", "short"),),
                tokens=(TokenSpan("toolong", "s", 0, 100),),
                input_digest="0" * 64,
            )


class TestLayerTaxonomy:
    """Step 3 requires eight one-hot layers and an explicit lifecycle order."""

    def test_one_hot_vectors_sum_to_one(self):
        """Every layer has exactly one primary assignment."""
        for layer in BlackboardLayer:
            vector = one_hot_layer(layer)
            assert sum(vector) == 1
            assert vector[int(layer) - 1] == 1

    def test_every_workspace_field_has_a_layer(self):
        """All workspace fields are covered by the layer taxonomy."""
        from ore_workspace.models import WORKSPACE_FIELDS

        for field in WORKSPACE_FIELDS:
            assert layer_for_field(field) in BlackboardLayer

    def test_lifecycle_partial_order(self):
        """Lifecycle states only promote through explicit transitions."""
        assert LifecycleState.CANDIDATE.at_least(LifecycleState.CANDIDATE)
        assert LifecycleState.ACCEPTED.at_least(LifecycleState.CANDIDATE)
        assert not LifecycleState.CANDIDATE.at_least(LifecycleState.ACCEPTED)
        assert LifecycleState.EXTERNALLY_SUPPLIED.at_least(LifecycleState.EXTERNALLY_SUPPLIED)


class TestChangeMeasurement:
    """Step 11 requires field-specific, non-negative change magnitudes."""

    def test_boolean_change_uses_indicator_metric(self):
        """Goal and input changes use a 0/1 boolean indicator."""
        old = _state(current_goal="understand_sentence")
        new = _state(current_goal="answer_question")
        changes = measure_workspace_changes(old, new)
        assert FieldChange("current_goal", 1.0, "boolean_indicator") in changes

    def test_collection_change_uses_symmetric_difference_ratio(self):
        """Collection fields use |A triangle B| / max(1, |A union B|)."""
        old = _state(lexical_hypotheses={"bank": [{"finding_id": "f1"}]})
        new = _state(lexical_hypotheses={
            "bank": [{"finding_id": "f1"}, {"finding_id": "f2"}],
        })
        changes = measure_workspace_changes(old, new)
        lexical = next(change for change in changes if change.field == "lexical_hypotheses")
        assert lexical.delta == pytest.approx(1 / 2)
        assert lexical.metric == "collection_symmetric_difference_ratio"

    def test_unchanged_state_produces_no_deltas(self):
        """Identical states produce no change magnitudes."""
        state = _state(lexical_hypotheses={"bank": [{"finding_id": "f1"}]})
        assert measure_workspace_changes(state, dict(state)) == ()

    def test_deltas_are_bounded_to_unit_interval(self):
        """Change magnitudes never exceed 1."""
        old = _state()
        new = _state(
            lexical_hypotheses={str(i): [{"finding_id": f"f{i}"}] for i in range(64)},
            syntactic_hypotheses=[{"finding_id": f"s{i}"} for i in range(64)],
        )
        for change in measure_workspace_changes(old, new):
            assert 0.0 < change.delta <= 1.0


class TestRelevance:
    """Step 12 requires R_m = sum_f A_mf * w_mf * Delta_f with deterministic order."""

    def test_relevance_report_orders_by_score_then_module(self):
        """Eligible modules are ordered by descending score with a stable tie-break."""
        from ore_workspace import SharedWorkspace, SubscriptionRegistry

        workspace = SharedWorkspace(
            "text", "goal", subscriptions=SubscriptionRegistry(defaults={}),
        )
        try:
            workspace.subscriptions.register(
                "module_a", ("lexical_hypotheses",),
                sensitivity={"lexical_hypotheses": 1.0}, threshold=0.5,
            )
            workspace.subscriptions.register(
                "module_b", ("lexical_hypotheses",),
                sensitivity={"lexical_hypotheses": 0.4}, threshold=0.3,
            )
            report = relevance_report(workspace, {"lexical_hypotheses": 1.0})
            assert [item[0] for item in report] == ["module_a", "module_b"]
            assert report[0][1] == pytest.approx(1.0)
            assert report[1][1] == pytest.approx(0.4)
        finally:
            workspace.close()


class TestComparisonScore:
    """Step 14 provides the proposed U_m comparison score, labelled heuristic."""

    def test_comparison_score_formula_matches_the_specification(self):
        """U_m combines relevance, gain, goal, quality minus latency, memory, risk."""
        score = compute_comparison_score(
            "module",
            relevance=0.8, heuristic_information_gain=0.5, goal_contribution=0.6,
            reliability=0.7, latency_cost=0.2, memory_cost=0.1, risk_cost=0.3,
            coefficients=ComparisonCoefficients(
                alpha=0.4, beta=0.2, gamma=0.2, delta=0.1,
                lam=0.05, mu=0.025, nu=0.05,
            ),
        )
        expected = (
            0.4 * 0.8 + 0.2 * 0.5 + 0.2 * 0.6 + 0.1 * 0.7
            - 0.05 * 0.2 - 0.025 * 0.1 - 0.05 * 0.3
        )
        assert score.score == pytest.approx(expected)
        assert score.heuristic is True
        assert "not a calibrated probability" in score.definition

    def test_comparison_score_rejects_non_finite_inputs(self):
        """Non-finite inputs are rejected rather than silently scored."""
        with pytest.raises(ValueError):
            compute_comparison_score(
                "module", relevance=float("nan"),
                heuristic_information_gain=0.5, goal_contribution=0.5,
                reliability=0.5, latency_cost=0.5, memory_cost=0.5, risk_cost=0.5,
                coefficients=ComparisonCoefficients(),
            )


class TestBackoffAndCircuitBreaker:
    """Step 26 requires bounded backoff and circuit-breaker gating."""

    def test_exponential_backoff_is_bounded_and_monotonic(self):
        """t_n = min(t_cap, t_0 * 2^n) stays bounded by the cap."""
        values = [
            exponential_backoff(
                attempt, initial_seconds=0.1, cap_seconds=1.0, jitter_seconds=0.0,
            )
            for attempt in range(10)
        ]
        assert values[0] == pytest.approx(0.1)
        assert all(value <= 1.0 for value in values)
        assert values[3] == pytest.approx(0.8)
        assert values[4] == pytest.approx(1.0)
        assert values[9] == pytest.approx(1.0)

    def test_circuit_breaker_opens_after_configured_failures(self, tmp_path):
        """Repeated failures open the circuit; success closes it again."""
        policy = BlackboardPolicy(
            circuit_breaker_threshold=3, circuit_breaker_cooldown_seconds=60.0,
        )
        coordinator = BlackboardResourceCoordinator(
            policy, task_family="test_family",
        )
        try:
            assert not coordinator.circuit_open("module")
            coordinator.record_outcome("module", False)
            coordinator.record_outcome("module", False)
            assert not coordinator.circuit_open("module")
            coordinator.record_outcome("module", False)
            assert coordinator.circuit_open("module")
            coordinator.record_outcome("module", True)
            assert not coordinator.circuit_open("module")
        finally:
            coordinator.close()

    def test_reliability_uses_beta_prior_mean(self, tmp_path):
        """Beta(2, 2) priors start at 0.5 and move with observed outcomes."""
        policy = BlackboardPolicy(
            reliability_prior_alpha=2.0, reliability_prior_beta=2.0,
        )
        resource = BlackboardResourceCoordinator(policy, task_family="test_family")
        try:
            assert resource.reliability_probability("module") == pytest.approx(0.5)
            resource.record_outcome("module", True)
            resource.record_outcome("module", True)
            assert resource.reliability_probability("module") == pytest.approx(4 / 6)
        finally:
            resource.close()


class TestEvidenceAggregation:
    """Step 21 requires conservative, correlation-aware evidence aggregation."""

    def test_correlated_sources_are_suppressed_to_the_strongest(self):
        """Dependent sources in one group contribute only their maximum."""
        scorer = HypothesisScorer()
        score = scorer.score((
            EvidenceContribution("s1", 0.4, 1.0, correlation_group="memory"),
            EvidenceContribution("s2", 0.9, 1.0, correlation_group="memory"),
            EvidenceContribution("s3", 0.3, 1.0, correlation_group=None),
        ))
        assert score.score == pytest.approx(1.2)
        assert "s2" in [item.source_id for item in score.used_evidence]
        assert score.suppressed_correlated_sources == ("s1",)
        assert score.calibrated_probability is False

    def test_score_definition_is_disclosed(self):
        """The evidence-score definition is exposed and warns it is not a probability."""
        assert "Aggregate" in SCORE_DEFINITION
        assert "not automatically a calibrated probability" in SCORE_DEFINITION


class TestBayesianUpdater:
    """Step 21 allows Bayesian updates only under defensible independence."""

    def setUp(self):
        """Create the updater under test."""
        self.updater = BayesianHypothesisUpdater()

    def test_log_odds_update_requires_independence_declaration(self):
        """Missing independence declarations are refused."""
        with pytest.raises(ValueError):
            BayesianHypothesisUpdater().posterior_probability(
                0.5, (2.0,), evidence_ids=("e1",), independence_declared=False,
            )

    def test_log_odds_update_refuses_duplicate_evidence(self):
        """Duplicated evidence identifiers are never counted as independent."""
        with pytest.raises(ValueError):
            BayesianHypothesisUpdater().posterior_probability(
                0.5, (2.0, 2.0), evidence_ids=("e1", "e1"),
                independence_declared=True,
            )

    def test_log_odds_update_matches_closed_form(self):
        """Posterior odds equal prior odds times likelihood ratios."""
        posterior = BayesianHypothesisUpdater().posterior_probability(
            0.5, (3.0,), evidence_ids=("e1",), independence_declared=True,
        )
        assert posterior == pytest.approx(0.75)

    def test_extreme_priors_stay_in_unit_interval(self):
        """Zero and one priors are preserved exactly."""
        updater = BayesianHypothesisUpdater()
        assert updater.posterior_probability(
            0.0, (), evidence_ids=(), independence_declared=True,
        ) == 0.0
        assert updater.posterior_probability(
            1.0, (), evidence_ids=(), independence_declared=True,
        ) == 1.0


class TestPolicyValidation:
    """Policy objects must reject invalid budgets and coefficients."""

    def test_budgets_reject_non_positive_limits(self):
        """Non-positive budgets are rejected."""
        with pytest.raises(ValueError):
            CycleBudgets(max_rounds=0)
        with pytest.raises(ValueError):
            CycleBudgets(default_timeout_seconds=0)

    def test_coefficients_reject_negative_weights(self):
        """Negative comparison coefficients are rejected."""
        with pytest.raises(ValueError):
            ComparisonCoefficients(alpha=-0.1)

    def test_policy_rejects_invalid_goal_threshold(self):
        """Goal thresholds must stay inside [0, 1]."""
        with pytest.raises(ValueError):
            BlackboardPolicy(goal_threshold=1.5)

    def test_ablation_switches_report_disabled_mechanisms(self):
        """Disabled mechanism names are reported for ablation harnesses."""
        switches = AblationSwitches(selective_subscriptions=False, usg_retrieval=False)
        assert switches.disabled_names() == ("selective_subscriptions", "usg_retrieval")


class TestFailureClassification:
    """Step 26 requires explicit failure classification labels."""

    def test_failure_classifications_are_distinct(self):
        """Every failure mode has its own label; none is a success state."""
        labels = {item.value for item in FailureClassification}
        assert len(labels) == 9
        assert "success" not in labels
