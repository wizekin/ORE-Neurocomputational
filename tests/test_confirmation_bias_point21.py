"""Acceptance tests for the fourteen Point 21 confirmation-bias operations."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ore_confirmation_bias import (
    EvidenceItem,
    EvidenceSource,
    Hypothesis,
    Point21Config,
    Point21ConfirmationBiasProcessor,
    Point21Request,
    compare_evidence_symmetrically,
    assess_evidence_quality,
    allocate_attention,
    select_information_sources,
    record_result_to_sqlite,
)


def make_request(
    *,
    sources: tuple[EvidenceSource, ...] | None = None,
    evidence: tuple[EvidenceItem, ...] | None = None,
    context: dict | None = None,
) -> Point21Request:
    """Build a two-hypothesis case with both confirming and disconfirming observations."""
    hypotheses = (
        Hypothesis(
            "rain",
            "Rain caused the street to become wet.",
            0.70,
            {"wet_street": 0.9, "rain_gauge": 0.9, "hose_activity": 0.1},
        ),
        Hypothesis(
            "cleaning",
            "Street cleaning caused the street to become wet.",
            0.30,
            {"wet_street": 0.7, "rain_gauge": 0.1, "hose_activity": 0.9},
        ),
    )
    if sources is None:
        sources = (
        EvidenceSource(
            "camera", "direct_record", reliability=0.95, diagnostic_value=0.8,
            accessibility=0.9, social_trust=0.8, independence_group="camera-system",
        ),
        EvidenceSource(
            "gauge", "instrument_record", reliability=0.92, diagnostic_value=0.9,
            accessibility=0.8, social_trust=0.7, independence_group="weather-station",
        ),
        EvidenceSource(
            "witness", "testimony", reliability=0.65, diagnostic_value=0.7,
            accessibility=0.8, social_trust=0.6, independence_group="witness-1",
        ),
        )
    if evidence is None:
        evidence = (
        EvidenceItem(
            "wet-street", "camera", "The street surface appears wet.",
            feature="wet_street", likelihoods={"rain": 0.85, "cleaning": 0.65},
            salience=0.6, relevance=0.9, novelty=0.4,
            uncertainty_reduction=0.6, measurement_quality=0.9,
            direct_observation=True,
        ),
        EvidenceItem(
            "no-rain", "gauge", "The rainfall gauge recorded no rain.",
            feature="rain_gauge", likelihoods={"rain": 0.10, "cleaning": 0.90},
            salience=0.7, relevance=0.95, novelty=0.7,
            uncertainty_reduction=0.9, measurement_quality=0.95,
            direct_observation=True,
        ),
        EvidenceItem(
            "hose-witness", "witness", "A witness reports seeing a cleaning hose.",
            feature="hose_activity", likelihoods={"rain": 0.15, "cleaning": 0.80},
            salience=0.8, relevance=0.8, novelty=0.6,
            uncertainty_reduction=0.8, measurement_quality=0.7,
            direct_observation=False,
        ),
        )
    return Point21Request(
        claim_id="wet-street-event",
        claim="Why was the street wet?",
        hypotheses=hypotheses,
        sources=sources,
        evidence=evidence,
        focus_hypothesis_id="rain",
        context=context if context is not None else {
            "memory_cues": ["wet_street", "rain_gauge"], "emotional_salience": 0.2
        },
    )


def test_pipeline_executes_fourteen_ordered_operations_with_two_output_tracks() -> None:
    """All source steps must produce ordered functional and biological trace records."""
    result = Point21ConfirmationBiasProcessor().process(make_request())
    assert len(result.operation_trace) == 14
    assert [row["operation_id"] for row in result.operation_trace] == [
        f"P21-{index:02d}" for index in range(1, 15)
    ]
    assert all(row["status"] == "completed" for row in result.operation_trace)
    assert all("functional_output" in row and "biological_output" in row
               for row in result.operation_trace)
    assert result.validation["operation_order_verified"] is True


def test_accuracy_first_mode_removes_congruence_gates_but_keeps_prior_and_reliability() -> None:
    """Accuracy-first control removes modelled agreement preferences, not evidence-quality checks."""
    request = make_request()
    result = Point21ConfirmationBiasProcessor(
        Point21Config(accuracy_first=True, simulate_biology=False)
    ).process(request)
    assert result.source_selection["selection_congruence_bias_enabled"] is False
    assert result.attention["attention_congruence_bias_enabled"] is False
    assert result.evidence_readout["readout_congruence_bias_enabled"] is False
    assert all(row["readout_gate"] == pytest.approx(1.0)
               for row in result.evidence_readout["item_contributions"])
    assert result.confidence >= 0.0
    assert result.evidence_quality["reliability_independent_of_agreement"] is True


def test_source_selection_can_expose_congruence_preference_separately_from_accuracy_mode() -> None:
    """With otherwise matched sources, a high congruence setting favours the congenial source."""
    sources = (
        EvidenceSource("agree", "report", 0.8, 0.8, 0.8, 0.8, 0.0, "group-a"),
        EvidenceSource("challenge", "report", 0.8, 0.8, 0.8, 0.8, 0.0, "group-b"),
    )
    evidence = (
        EvidenceItem("e-agree", "agree", "Observation compatible with rain.",
                     likelihoods={"rain": 0.95, "cleaning": 0.05}),
        EvidenceItem("e-challenge", "challenge", "Observation compatible with cleaning.",
                     likelihoods={"rain": 0.05, "cleaning": 0.95}),
    )
    request = make_request(sources=sources, evidence=evidence)
    biased = select_information_sources(
        request, {"rain": 0.7, "cleaning": 0.3},
        Point21Config(max_sources=1, selection_congruence_bias=0.9),
    )
    accuracy_first = select_information_sources(
        request, {"rain": 0.7, "cleaning": 0.3},
        Point21Config(max_sources=1, accuracy_first=True),
    )
    assert biased["selected_source_ids"] == ["agree"]
    assert biased["selection_congruence_bias_enabled"] is True
    assert accuracy_first["selection_congruence_bias_enabled"] is False


def test_unselected_source_evidence_is_retained_but_not_deeply_attended() -> None:
    """Unselected information stays in the audit trail and is not mistaken for absent evidence."""
    request = make_request()
    config = Point21Config(max_sources=1, simulate_biology=False)
    selection = select_information_sources(request, {"rain": 0.7, "cleaning": 0.3}, config)
    attention = allocate_attention(request, selection["selected_source_ids"],
                                   {"rain": 0.7, "cleaning": 0.3}, config)
    assert attention["unselected_evidence_ids"]
    assert set(attention["unselected_evidence_ids"]).issubset(
        {item.evidence_id for item in request.evidence}
    )


def test_evidence_comparison_uses_a_common_symmetric_log_likelihood_scale() -> None:
    """Equal-and-opposite likelihood patterns produce equal-and-opposite centered evidence."""
    request = make_request()
    quality = {
        "weighted_evidence": [
            {"evidence_id": item.evidence_id, "weight": 1.0} for item in request.evidence
        ]
    }
    comparison = compare_evidence_symmetrically(request, quality)
    rows = {row["evidence_id"]: row for row in comparison["comparisons"]}
    positive = rows["wet-street"]["centered_log_evidence"]
    negative = rows["no-rain"]["centered_log_evidence"]
    assert positive["rain"] > 0 > positive["cleaning"]
    assert negative["rain"] < 0 < negative["cleaning"]


def test_repeated_sources_sharing_one_origin_are_downweighted() -> None:
    """Multiple observations within one dependence group do not multiply confidence freely."""
    sources = (
        EvidenceSource("a", "report", 0.9, 0.8, 0.8, 0.8, 0.0, "same-origin"),
        EvidenceSource("b", "report", 0.9, 0.8, 0.8, 0.8, 0.0, "same-origin"),
        EvidenceSource("c", "report", 0.9, 0.8, 0.8, 0.8, 0.0, "independent-origin"),
    )
    evidence = (
        EvidenceItem("a1", "a", "Report one", likelihoods={"rain": 0.8, "cleaning": 0.2}),
        EvidenceItem("b1", "b", "Report two copied from report one",
                     likelihoods={"rain": 0.8, "cleaning": 0.2}),
        EvidenceItem("c1", "c", "Independent instrument result",
                     likelihoods={"rain": 0.8, "cleaning": 0.2}),
    )
    request = make_request(sources=sources, evidence=evidence)
    config = Point21Config(max_sources=3, simulate_biology=False)
    selection = select_information_sources(request, {"rain": 0.7, "cleaning": 0.3}, config)
    attention = allocate_attention(request, selection["selected_source_ids"],
                                   {"rain": 0.7, "cleaning": 0.3}, config)
    quality = assess_evidence_quality(request, attention, selection, config)
    rows = {row["evidence_id"]: row for row in quality["weighted_evidence"]}
    assert rows["a1"]["dependence_correction"] == pytest.approx(1 / (2 ** 0.5))
    assert rows["c1"]["dependence_correction"] == pytest.approx(1.0)


def test_memory_retrieval_keeps_original_content_and_source_attribution() -> None:
    """Current beliefs may affect retrieval accessibility but never rewrite stored evidence."""
    result = Point21ConfirmationBiasProcessor(
        Point21Config(simulate_biology=False)
    ).process(make_request())
    assert result.memory_retrieval["memory_count"] == 3
    assert all(item["original_content_preserved"]
               for item in result.memory_retrieval["retrieved_memories"])
    assert all(item["source_id"] for item in result.memory_retrieval["retrieved_memories"])


def test_corrective_control_pre_registers_alternatives_and_ranks_disconfirming_tests() -> None:
    """The correction policy compares hypotheses and ranks sources by expected information gain."""
    result = Point21ConfirmationBiasProcessor(
        Point21Config(simulate_biology=False)
    ).process(make_request())
    correction = result.corrective_control
    assert correction["pre_registered_focal_predictions"]
    assert correction["alternative_hypotheses"][0]["hypothesis_id"] == "cleaning"
    assert correction["ranked_discriminating_sources"]
    assert correction["symmetry_checks"]


def test_no_evidence_returns_prior_without_fabricating_a_posterior_shift() -> None:
    """An empty evidence set leaves the belief distribution unchanged and explicitly auditable."""
    request = make_request(sources=(), evidence=())
    result = Point21ConfirmationBiasProcessor(
        Point21Config(simulate_biology=False)
    ).process(request)
    assert result.beliefs == pytest.approx({"rain": 0.7, "cleaning": 0.3})
    assert result.evidence_audit == ()
    assert result.metadata["evidence_item_count"] == 0


def test_biological_track_can_be_disabled_without_changing_functional_trace_length() -> None:
    """Disabling the rate-coded reference does not disable the functional algorithm."""
    result = Point21ConfirmationBiasProcessor(
        Point21Config(simulate_biology=False)
    ).process(make_request())
    assert result.biological["model"] == "disabled"
    assert len(result.operation_trace) == 14
    assert all(row["biological_output"]["model"] == "disabled"
               for row in result.operation_trace)


def test_request_rejects_inconsistent_priors_and_unknown_evidence_hypotheses() -> None:
    """The request contract rejects invalid probability distributions and dangling likelihood keys."""
    with pytest.raises(ValueError, match="sum to 1"):
        Point21Request(
            "bad", "test",
            (Hypothesis("a", "A", 0.7), Hypothesis("b", "B", 0.4)),
            (), (),
        )
    with pytest.raises(ValueError, match="unknown hypotheses"):
        Point21Request(
            "bad", "test",
            (Hypothesis("a", "A", 0.7), Hypothesis("b", "B", 0.3)),
            (EvidenceSource("s", "test"),),
            (EvidenceItem("e", "s", "x", likelihoods={"a": 0.5, "unknown": 0.5}),),
        )


def test_sqlite_registers_package_symbols_and_logs_all_fourteen_operations(tmp_path: Path) -> None:
    """The shared annotation catalogue and event logger persist every Point 21 operation."""
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "ore_logger.sqlite3"
    result = Point21ConfirmationBiasProcessor(
        Point21Config(simulate_biology=False)
    ).process(make_request())
    report = record_result_to_sqlite(result, database_path, repository_root=root)
    assert report["database_summary"]["events"] == 14
    with sqlite3.connect(database_path) as connection:
        annotations = connection.execute(
            "SELECT COUNT(*) FROM annotations "
            "WHERE module_name LIKE 'ore_confirmation_bias%'"
        ).fetchone()[0]
        events = connection.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=?", (report["run_id"],)
        ).fetchone()[0]
        status = connection.execute(
            "SELECT status FROM runs WHERE run_id=?", (report["run_id"],)
        ).fetchone()[0]
    assert annotations >= 20
    assert events == 14
    assert status == "completed"
