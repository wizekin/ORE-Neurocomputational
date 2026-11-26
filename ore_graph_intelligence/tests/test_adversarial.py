"""Adversarial safety tests: the core invariant under hostile conditions.

- An unobserved edge is unknown, never false (missing-edge failure mode).
- Uncalibrated averages are never called probabilities.
- A high learned score is not a proof.
- Confirmation-only verification is impossible (counter-evidence search).
- Learned components cannot block the deterministic path.
"""
from __future__ import annotations

from ore_graph_intelligence.discovery import (
    ComplementaryScorer,
    GraphCandidateService,
    LinkPredictor,
    UtilityRanker,
)
from ore_graph_intelligence.observability import (
    expected_calibration_error,
    false_positive_rate,
    relation_type_subgroup_metrics,
    SemanticSafetyMetrics,
)
from ore_graph_intelligence.projection import USGGraphProjector
from ore_graph_intelligence.schemas.candidate_schema import (
    CandidateDisposition,
    GraphCandidate,
    GraphQuery,
    GraphQueryAnchor,
)
from ore_graph_intelligence.verification import CandidateVerifier

from .fixtures import contradiction_bundle, graph_bundle


def _snapshot(bundle=None):
    return USGGraphProjector().project(bundle or graph_bundle())


def _river_query():
    return GraphQuery(
        anchor=GraphQueryAnchor(label="The land alongside a river."),
        relation_types=("DENOTES",), budget_candidates=32,
    )


def test_unobserved_edge_is_unknown_not_false():
    """A novel candidate without evidence is UNVERIFIABLE, never REJECTED as false."""
    snapshot = _snapshot()
    service = GraphCandidateService(snapshot)
    kept, _rejected = service.propose(_river_query())
    assert kept
    results = CandidateVerifier(snapshot).verify_all(kept)
    for result in results:
        if result.disposition == CandidateDisposition.REJECTED:
            assert result.type_violations or result.rule_violations, (
                "rejection requires a rule or type violation, never absence of evidence"
            )
        elif result.disposition == CandidateDisposition.UNVERIFIABLE:
            assert "model-only" in " ".join(result.notes).lower() or result.notes


def test_uncalibrated_average_is_not_a_probability():
    """Component scores stay separate; calibrated flags stay False without calibration."""
    snapshot = _snapshot()
    kept, _ = GraphCandidateService(snapshot).propose(_river_query())
    scored = LinkPredictor(ComplementaryScorer(snapshot)).score_all(kept)
    for candidate in scored:
        assert not candidate.calibrated
        assert candidate.calibrated_probability is None
        # Component scores are preserved separately, never merged into one truth value.
        assert len(candidate.component_scores.available()) >= 1
    ranked = UtilityRanker(snapshot).rank(scored)
    for candidate in ranked:
        assert candidate.utility_components is not None
        assert candidate.utility is not None
        assert -candidate.utility_components.cost <= candidate.utility + 1e-9


def test_high_score_is_not_a_proof():
    """A top-ranked candidate still ends UNVERIFIABLE without evidence."""
    snapshot = _snapshot()
    kept, _ = GraphCandidateService(snapshot).propose(_river_query())
    scored = LinkPredictor(ComplementaryScorer(snapshot)).score_all(kept)
    ranked = UtilityRanker(snapshot).rank(scored)
    results = CandidateVerifier(snapshot).verify_all(ranked)
    top = ranked[0]
    matching = [result for result in results if result.candidate_id == top.candidate_id]
    assert matching
    assert matching[0].disposition != CandidateDisposition.SUPPORTED, (
        "no learned score, however high, can establish truth"
    )


def test_verification_searches_counter_evidence():
    """Counter-evidence flips the disposition to AMBIGUOUS (no confirmation bias)."""
    snapshot = _snapshot(contradiction_bundle())
    service = GraphCandidateService(snapshot)
    kept, _rejected = service.propose(GraphQuery(
        anchor=GraphQueryAnchor(label="A financial institution."),
        relation_types=("DENOTES",), budget_candidates=32,
    ))
    results = CandidateVerifier(snapshot).verify_all(kept)
    contradicted = [result for result in results if result.counter_edge_ids]
    assert contradicted, "the contradiction fixture must trigger counter-evidence"
    assert all(
        result.disposition == CandidateDisposition.AMBIGUOUS for result in contradicted
    )


def test_candidate_without_anchor_abstains():
    """Ambiguous anchors abstain rather than guessing by nearest embedding."""
    snapshot = _snapshot()
    service = GraphCandidateService(snapshot)
    kept, rejected = service.propose(GraphQuery(
        anchor=GraphQueryAnchor(label="unknown label entirely"),
        relation_types=("DENOTES",), budget_candidates=32,
    ))
    assert not kept and not rejected, "an unresolvable anchor abstains silently"


def test_semantic_safety_metrics_detect_false_positives():
    """FPR among promoted candidates is computed against adjudicated ground truth."""
    snapshot = _snapshot()
    kept, _ = GraphCandidateService(snapshot).propose(_river_query())
    results = CandidateVerifier(snapshot).verify_all(kept)
    metrics = SemanticSafetyMetrics.from_results(results)
    assert metrics.n_candidates == len(results)
    assert metrics.n_supported + metrics.n_rejected + metrics.n_ambiguous \
        + metrics.n_unverifiable == metrics.n_candidates
    rate = false_positive_rate(
        results, {result.triple: False for result in results},
    )
    assert rate is None, "no promoted candidates here, so FPR is undefined, not zero"

    existing = next(e for e in snapshot.edges if e.relation_type == "INSTANCE_OF")
    confirmed = GraphCandidate(
        subject_id=existing.source_id, relation_type="INSTANCE_OF",
        object_id=existing.target_id, snapshot_hash=snapshot.graph_hash,
    )
    confirmed_results = CandidateVerifier(snapshot).verify_all([confirmed])
    adjudicated_false = {confirmed_results[0].triple: False}
    rate = false_positive_rate(confirmed_results, adjudicated_false)
    assert rate == 1.0, "a promoted candidate the workflow refuted is a false positive"


def test_calibration_error_measures_miscalibration():
    """ECE grows with deliberately miscalibrated probabilities."""
    labels = [1, 1, 0, 0, 1, 1, 0, 0]
    calibrated = [0.95, 0.95, 0.05, 0.05, 0.95, 0.95, 0.05, 0.05]
    overconfident_wrong = [0.95, 0.95, 0.95, 0.95, 0.05, 0.05, 0.05, 0.05]
    assert expected_calibration_error(calibrated, labels) \
        < expected_calibration_error(overconfident_wrong, labels)


def test_subgroup_metrics_keep_poor_subgroups_visible():
    """Relation-type subgroups are reported separately, never collapsed."""
    snapshot = _snapshot()
    kept, _ = GraphCandidateService(snapshot).propose(_river_query())
    results = CandidateVerifier(snapshot).verify_all(kept)
    subgroups = relation_type_subgroup_metrics(results)
    assert set(subgroups) == {"DENOTES"}
    assert subgroups["DENOTES"]["n_candidates"] == float(len(results))
