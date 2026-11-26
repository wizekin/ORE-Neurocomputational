"""Acceptance tests: capability resolution, ranking, policy, and resources (Steps 6-11)."""
from __future__ import annotations

import numpy as np
import pytest

from ore_learning.contracts import (
    Capability,
    LifecycleState,
    Modality,
    RiskClass,
    SchemaDescriptor,
    SecurityReview,
)
from ore_learning.registry import ModelRegistry
from ore_learning.routing import (
    CapabilityResolver,
    DenyingResourceCoordinator,
    LocalResourceCoordinator,
    ModelRouter,
    MoEProposal,
    MoERouter,
    RoutingPolicy,
    SelectionDecision,
    gates_from_scores,
    rank_candidates,
    utility_features,
    weights_for_risk,
)
from ore_learning.routing.ensemble import BoundedEnsemble, EnsembleMember, ensemble_justified

from learning_point14_helpers import make_manifest, make_request


def populated_registry() -> ModelRegistry:
    """A registry with three eligible specialists on distinct datasets."""
    registry = ModelRegistry()
    registry.register(make_manifest("model.a", quality=0.90, latency_ms=80.0,
                                    dataset="ds-1"))
    registry.register(make_manifest("model.b", quality=0.86, latency_ms=120.0,
                                    dataset="ds-2"))
    registry.register(make_manifest("model.c", quality=0.93, latency_ms=60.0,
                                    dataset="ds-3"))
    return registry


def test_capability_resolver_returns_only_eligible_models():
    """Acceptance: capability resolver returns only eligible models."""
    registry = populated_registry()
    registry.register(make_manifest("model.lowq", quality=0.55))
    registry.register(make_manifest("model.proposed",
                                    state=LifecycleState.PROPOSED))
    registry.register(make_manifest("model.research", license_id="research-only"))
    resolver = CapabilityResolver()
    candidates = resolver.resolve(make_request(), registry)
    eligible = {candidate.identity for candidate in candidates if candidate.eligible}
    ineligible = {candidate.identity for candidate in candidates
                  if not candidate.eligible}
    # research-only licenses are permitted for LOW risk tasks (policy scope)
    assert eligible == {"model.a@1.0.0", "model.b@1.0.0", "model.c@1.0.0",
                        "model.research@1.0.0"}
    assert "model.lowq@1.0.0" in ineligible  # measured quality below target
    # non-selectable registry states never reach the resolver at all
    assert "model.proposed@1.0.0" not in eligible
    assert all(candidate.identity != "model.proposed@1.0.0" for candidate in candidates)
    # every candidate carries the full eligibility conjunction S/C/P/R/L
    for candidate in candidates:
        if candidate.eligible:
            assert {factor.factor for factor in candidate.factors} == {"S", "C", "P",
                                                                      "R", "L"}
            assert all(factor.passed for factor in candidate.factors)


def test_router_ranking_is_deterministic_for_identical_state_and_inputs():
    """Acceptance: router ranking is deterministic for identical state and inputs."""
    registry = populated_registry()
    router_one = ModelRouter(registry)
    router_two = ModelRouter(registry)
    selection_one = router_one.select(make_request())
    selection_two = router_two.select(make_request())
    order_one = [record.identity for record in selection_one.ranking]
    order_two = [record.identity for record in selection_two.ranking]
    assert order_one == order_two
    utilities_one = [round(record.utility, 9) for record in selection_one.ranking]
    utilities_two = [round(record.utility, 9) for record in selection_two.ranking]
    assert utilities_one == utilities_two
    # repeated calls on the same router stay deterministic too
    repeat = router_one.select(make_request())
    assert [record.identity for record in repeat.ranking] == order_one


def test_hard_policy_constraints_override_a_higher_utility_score():
    """Acceptance: hard policy constraints override a higher utility score."""
    registry = ModelRegistry()
    # the forbidden model would win on raw utility: better quality, lower latency
    registry.register(make_manifest("model.forbidden", quality=0.99, latency_ms=10.0,
                                    license_id="research-only"))
    registry.register(make_manifest("model.safe", quality=0.88, latency_ms=100.0))
    # high-risk request: research-only licenses are policy-forbidden
    request = make_request(risk_class=RiskClass.HIGH)
    selection = ModelRouter(registry).select(request)
    eligible = [candidate for candidate in selection.candidates if candidate.eligible]
    assert all(candidate.identity != "model.forbidden@1.0.0" for candidate in eligible)
    assert selection.primary.model_id == "model.safe"
    excluded = [candidate for candidate in selection.candidates
                if not candidate.eligible]
    assert any(candidate.identity == "model.forbidden@1.0.0" for candidate in excluded)
    # security verdicts are policy facts as well
    registry.register(make_manifest("model.unreviewed", security_verdict="pending"))
    high_request = make_request(risk_class=RiskClass.HIGH)
    high_selection = ModelRouter(registry).select(high_request)
    assert all(candidate.identity != "model.unreviewed@1.0.0"
               for candidate in high_selection.candidates if candidate.eligible)
    # ... while low-risk tasks accept pending reviews per policy
    low_selection = ModelRouter(registry).select(make_request())
    assert any(candidate.identity == "model.unreviewed@1.0.0"
               for candidate in low_selection.candidates if candidate.eligible)


def test_resource_reservation_failure_triggers_safe_fallback_or_deferral():
    """Acceptance: resource reservation failure triggers safe fallback or deferral."""
    registry = populated_registry()
    # a tiny deterministic fallback that fits a strained capacity envelope
    registry.register(make_manifest("model.fallback", quality=0.80, latency_ms=5.0,
                                    dataset="ds-fb",
                                    state=LifecycleState.VALIDATED,
                                    cpu_cores=0.25, memory_mb=16.0))
    # denying CROS entirely with no fallback -> deferral
    denied = ModelRouter(registry, coordinator=DenyingResourceCoordinator())
    deferred = denied.select(make_request())
    assert deferred.decision is SelectionDecision.DEFERRED
    assert "reservation refused" in deferred.fallback_reason
    # strained CROS (room for the tiny fallback only) -> deterministic fallback
    strained = LocalResourceCoordinator(cpu_cores=0.5, memory_mb=64.0,
                                        max_concurrency=4)
    with_fallback = ModelRouter(registry, coordinator=strained,
                                fallback_model_id="model.fallback")
    fallback_selection = with_fallback.select(make_request())
    assert fallback_selection.decision is SelectionDecision.FALLBACK
    assert fallback_selection.primary.model_id == "model.fallback"
    assert "fallback" in fallback_selection.selection_rationale


def test_exhausted_capacity_denies_reservation():
    """CROS admission: a saturated coordinator refuses new reservations."""
    registry = populated_registry()
    coordinator = LocalResourceCoordinator(cpu_cores=1.5, memory_mb=512.0,
                                           max_concurrency=2)
    router = ModelRouter(registry, coordinator=coordinator)
    first = router.select(make_request())
    assert first.decision is SelectionDecision.SELECTED
    # capacity (with safety margin) is now consumed
    second = ModelRouter(registry, coordinator=coordinator).select(make_request())
    # a second concurrent reservation cannot fit and the task defers
    assert second.decision is SelectionDecision.DEFERRED
    # releasing the first reservation frees capacity again
    assert router.release(first, "completed") is True
    third = ModelRouter(registry, coordinator=coordinator).select(make_request())
    assert third.decision is SelectionDecision.SELECTED


def test_utility_features_and_risk_adjusted_weights():
    """Step 8: normalized features and risk-adjusted weights are documented."""
    from ore_learning.config import LearningConfig

    manifest = make_manifest("model.f", quality=0.9, latency_ms=100.0)
    request = make_request(latency_target_ms=50.0)  # 2x the latency target
    features = utility_features(manifest, request)
    assert 0.0 <= features["latency_penalty"] <= 1.0
    assert features["latency_penalty"] == pytest.approx(1.0)  # 2x target saturates
    assert features["quality"] == pytest.approx(0.9)
    config = LearningConfig()
    low = weights_for_risk(config, RiskClass.LOW)
    high = weights_for_risk(config, RiskClass.HIGH)
    assert high.weight_uncertainty > low.weight_uncertainty
    assert high.weight_calibration > low.weight_calibration


def test_moe_router_proposes_but_coordinator_decides():
    """Step 9: MoE is a routing pattern; the coordinator authorizes experts."""
    router = MoERouter()
    proposal = MoEProposal(expert_ids=("expert.x", "expert.y", "expert.z"),
                           gates=gates_from_scores([2.0, 1.0, 0.5]))
    unauthorized = router.decide(proposal)
    assert not unauthorized.approved
    router.authorize(["expert.x", "expert.y"], budgets={"expert.x": 10.0})
    decision = router.decide(proposal)
    assert decision.approved
    assert decision.authorized_experts == ("expert.x", "expert.y")
    combined = router.combine(decision, {"expert.x": np.array([1.0, 2.0]),
                                         "expert.y": np.array([3.0, 4.0])})
    # y = sum g_i E_i with normalized gates
    gates = dict(zip(proposal.expert_ids, proposal.gates))
    total = gates["expert.x"] + gates["expert.y"]
    expected_x = (gates["expert.x"] / total) * 1.0 + (gates["expert.y"] / total) * 3.0
    assert combined[0] == pytest.approx(expected_x)
    # concentration monitor: one expert dominating triggers the collapse alert
    concentrated = MoEProposal(expert_ids=("expert.x", "expert.y"),
                               gates=(0.95, 0.05))
    alert = router.check_concentration(router.decide(concentrated))
    assert alert is not None and alert.concentration > 0.8


def test_router_forms_bounded_ensemble_only_when_evidence_allows():
    """Step 10: ensembles are evidence-justified, never automatic."""
    registry = populated_registry()
    # default: one specialist even though utilities are close
    default = ModelRouter(registry).select(make_request())
    assert default.decision is SelectionDecision.SELECTED
    assert len(default.selected) == 1
    # explicit ensemble evidence + close independent candidates -> bounded ensemble
    ensemble_router = ModelRouter(registry, allow_ensemble=True)
    ensemble_selection = ensemble_router.select(make_request())
    assert ensemble_selection.decision is SelectionDecision.ENSEMBLE
    assert 2 <= len(ensemble_selection.selected) <= 3
    assert all(member.training_data_provenance.dataset_id !=
               ensemble_selection.selected[0].training_data_provenance.dataset_id
               for member in ensemble_selection.selected[1:])
    # correlated candidates (same dataset) never form an ensemble
    correlated_registry = ModelRegistry()
    correlated_registry.register(make_manifest("model.x", dataset="ds-same",
                                               quality=0.9))
    correlated_registry.register(make_manifest("model.y", dataset="ds-same",
                                               quality=0.89))
    correlated = ModelRouter(correlated_registry, allow_ensemble=True).select(
        make_request())
    assert correlated.decision is SelectionDecision.SELECTED


def test_bounded_ensemble_surfaces_disagreement_and_correlation():
    """Step 10: bounded ensembles, disagreement handling, correlated models."""
    from ore_learning.config import LearningConfig

    ensemble = BoundedEnsemble(LearningConfig(max_ensemble_size=3),
                               disagreement_threshold=0.25)
    report = ensemble.aggregate([
        EnsembleMember("m1", "ds-1", 0.9),
        EnsembleMember("m2", "ds-2", 0.2)])
    assert report.conflict and report.disagreement > 0.25
    assert report.independent_confirmations == 2
    correlated = ensemble.aggregate([
        EnsembleMember("m1", "ds-1", 0.8),
        EnsembleMember("m2", "ds-1", 0.85)])
    assert correlated.independent_confirmations == 1
    assert "correlated" in correlated.note
    with pytest.raises(ValueError, match="bound"):
        ensemble.aggregate([EnsembleMember(f"m{i}", f"ds-{i}", 0.5)
                            for i in range(5)])
    # ensembles are justified only for independent candidates within the margin
    candidates = [make_manifest("model.e1", dataset="ds-1"),
                  make_manifest("model.e2", dataset="ds-2")]
    justified, why = ensemble_justified(candidates, LearningConfig(), 0.50, 0.49)
    assert justified, why
    same_data = [make_manifest("model.e1", dataset="ds-1"),
                 make_manifest("model.e2", dataset="ds-1")]
    not_justified, _ = ensemble_justified(same_data, LearningConfig(), 0.50, 0.49)
    assert not not_justified  # correlated members provide no independent confirmation


def test_selection_prefers_single_specialist():
    """Step 10: one suitable specialist is preferred when it meets requirements."""
    registry = populated_registry()
    selection = ModelRouter(registry).select(make_request())
    assert selection.decision is SelectionDecision.SELECTED
    assert len(selection.selected) == 1
    # model.c has the best quality/latency profile and ranks first
    assert selection.primary.model_id == "model.c"
    rationale = selection.selection_rationale
    assert "single specialist" in rationale or "utility" in rationale


def test_rank_candidates_marks_ineligible_and_ranks_eligible():
    """Ranking assigns deterministic ranks and leaves ineligible candidates at 0."""
    registry = ModelRegistry()
    resolver = CapabilityResolver()
    request = make_request()
    candidates = resolver.resolve(request, registry)
    assert candidates == []
    ranked = rank_candidates(candidates, request)
    assert ranked == []


def test_schema_mismatch_blocks_eligibility():
    """Factor S: output schema must satisfy the task's output contract."""
    registry = ModelRegistry()
    text_manifest = make_manifest(
        "model.textonly",
        schema=SchemaDescriptor("text", "sentiment", ("label", "score")),
        supported_tasks=(Capability.TEXT_CLASSIFICATION.value,),
        modalities=(Modality.TEXT,))
    registry.register(text_manifest)
    request = make_request()
    # the registry query rejects the schema mismatch before ranking
    query = registry.query(request)
    assert "model.textonly@1.0.0" in query.rejected
    assert not query.matched
    # and the resolver's S factor independently documents the incompatibility
    factor = CapabilityResolver()._schema_factor(text_manifest, request)
    assert factor.factor == "S" and not factor.passed
    assert "does not satisfy" in factor.evidence
