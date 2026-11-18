"""Focused tests for Point 12 context-dependent interpretation and SQLite logging."""

from __future__ import annotations

import sqlite3

import pytest

from ore_logger.database import LoggerDatabase
from ore_context_dependent_interpretation import (
    CandidateHypothesis,
    ContextDependentInterpretationProcessor,
    ContextEvidence,
    ContextInterpretationConfig,
    ContextState,
    InputEvent,
    OPERATION_REGISTRY,
    bayesian_update,
    estimate_context_priors,
    register_annotations,
    retrieve_context,
    softmax,
)


def make_candidates():
    """Create finance and river interpretations with distinct semantic cues and fact bindings."""
    return (
        CandidateHypothesis(
            "finance", "financial institution", cue_terms=("money",),
            facts={"sense": "finance"},
        ),
        CandidateHypothesis(
            "river", "river bank", cue_terms=("river",),
            facts={"sense": "river"},
        ),
    )


def make_config(**overrides):
    """Use permissive thresholds for deterministic small illustrative test cases."""
    values = {
        "selection_threshold": 0.45,
        "minimum_posterior_margin": 0.05,
        "maximum_normalized_entropy": 0.99,
    }
    values.update(overrides)
    return ContextInterpretationConfig(**values)


def finance_state():
    """Build a discourse state with explicit financial-service context."""
    return ContextState(
        discourse_history=[
            ContextEvidence(
                evidence_id="finance-turn",
                source_type="discourse",
                content=(
                    "The bank is a financial institution handling money, account deposits, "
                    "cash withdrawals, and customer balances."
                ),
                concepts=("bank", "financial", "institution", "money", "account"),
                goal_tags=("financial_task",),
                provenance="previous_discourse",
            )
        ],
        current_goal="financial_task",
    )


def river_state():
    """Build a discourse state with explicit river and shoreline context."""
    return ContextState(
        discourse_history=[
            ContextEvidence(
                evidence_id="river-turn",
                source_type="discourse",
                content=(
                    "We followed the river bank along the shore, watching water and fishing boats."
                ),
                concepts=("bank", "river", "shore", "water", "fishing"),
                goal_tags=("outdoor_navigation",),
                provenance="previous_discourse",
            )
        ],
        current_goal="outdoor_navigation",
    )


def ambiguous_event(**overrides):
    """Create the same ambiguous phrase with both alternative meanings available."""
    values = {
        "event_id": "bank-event",
        "content": "The bank is closed.",
        "candidates": make_candidates(),
    }
    values.update(overrides)
    return InputEvent(**values)


def test_registry_has_all_16_operations_in_source_order():
    """Keep the software pipeline aligned to the complete Point 12 source sequence."""
    assert len(OPERATION_REGISTRY) == 16
    assert [item.operation_id for item in OPERATION_REGISTRY] == [
        f"P12-{index:02d}" for index in range(1, 17)
    ]


def test_softmax_and_bayesian_update_are_normalized_and_input_sensitive():
    """Verify stable score normalization and stronger likelihood changing the posterior."""
    prior = softmax({"finance": 1.2, "river": 0.4, "other": 0.0})
    posterior = bayesian_update(
        prior, {"finance": 0.05, "river": 1.0, "other": 0.5}
    )
    assert sum(prior.values()) == pytest.approx(1.0)
    assert sum(posterior.values()) == pytest.approx(1.0)
    assert posterior["river"] > posterior["finance"]


def test_same_ambiguous_phrase_changes_with_discourse_context():
    """Show that a stable input can receive different meanings from different preceding contexts."""
    event = ambiguous_event()
    processor = ContextDependentInterpretationProcessor(config=make_config())

    finance_result = processor.process(event, context=finance_state())
    river_result = processor.process(
        InputEvent(
            event_id="bank-event-river",
            content=event.content,
            candidates=event.candidates,
        ),
        context=river_state(),
    )

    finance_p = next(x.posterior for x in finance_result.candidates
                     if x.candidate.candidate_id == "finance")
    finance_river_p = next(x.posterior for x in river_result.candidates
                           if x.candidate.candidate_id == "river")
    assert finance_p > next(
        x.posterior for x in finance_result.candidates
        if x.candidate.candidate_id == "river"
    )
    assert finance_river_p > next(
        x.posterior for x in river_result.candidates
        if x.candidate.candidate_id == "finance"
    )


def test_explicit_input_evidence_is_scored_separately_from_contextual_prior():
    """Ensure direct candidate evidence can overcome a contextual expectation."""
    state = finance_state()
    event = ambiguous_event(
        event_id="bank-direct-evidence",
        candidate_evidence={"finance": 0.0, "river": 1.0},
    )
    result = ContextDependentInterpretationProcessor(config=make_config()).process(
        event, context=state
    )
    finance = next(x for x in result.candidates if x.candidate.candidate_id == "finance")
    river = next(x for x in result.candidates if x.candidate.candidate_id == "river")
    assert finance.context_prior > river.context_prior
    assert river.input_likelihood > finance.input_likelihood
    assert river.posterior > finance.posterior


def test_weak_context_preserves_ambiguity_and_open_set_alternative():
    """Prevent forced commitment when the input does not distinguish candidate meanings."""
    result = ContextDependentInterpretationProcessor().process(ambiguous_event())
    assert result.status == "ambiguous"
    assert result.selected is None
    assert result.diagnostics["open_set_candidate_retained"] is True
    assert result.output["clarification_needed"] is True
    assert len(result.candidates) == 3


def test_explicit_contradiction_can_revise_a_provisional_meaning():
    """Keep the contradiction trace and let new explicit evidence change candidate ranking."""
    state = ContextState()
    state.situation.facts["sense"] = "finance"
    state.situation.fact_provenance["sense"] = {
        "status": "retrieved", "provenance": "older_episode"
    }
    event = ambiguous_event(
        event_id="bank-revision",
        explicit_facts={"sense": "river"},
        candidate_evidence={"finance": 1.0, "river": 0.9},
    )
    result = ContextDependentInterpretationProcessor(config=make_config()).process(
        event, context=state
    )
    assert result.conflicts
    assert result.conflicts[0]["previous_value"] == "finance"
    assert result.conflicts[0]["incoming_value"] == "river"
    assert result.diagnostics["revision_applied"] is True
    assert result.status == "revised"
    assert result.selected is not None
    assert result.selected.candidate.candidate_id == "river"
    assert state.situation.revision_count == 1


def test_provenance_distinguishes_input_retrieval_and_hypothesis():
    """Ensure contextual hypotheses are never silently relabelled as observed facts."""
    result = ContextDependentInterpretationProcessor(config=make_config()).process(
        ambiguous_event(), context=finance_state()
    )
    statuses = {row["status"] for row in result.evidence_ledger}
    assert {"explicit_input", "retrieved_context", "hypothesis"} <= statuses
    assert result.diagnostics["source_provenance_preserved"] is True
    if result.selected is not None:
        for value in result.situation["fact_provenance"].values():
            assert value["status"] in {"stated_or_observed", "inferred"}


def test_goal_conditioning_and_context_retrieval_keep_source_ids():
    """Verify task-goal information influences retrieval without losing source identity."""
    state = finance_state()
    event = ambiguous_event(goal="financial_task")
    retrieved = retrieve_context(event, state, ContextInterpretationConfig())
    assert retrieved["discourse"]
    assert retrieved["discourse"][0]["evidence_id"] == "finance-turn"
    assert retrieved["discourse"][0]["goal_fit"] == pytest.approx(1.0)


def test_biological_network_reports_distributed_functional_surrogates():
    """Expose network state for language, memory, goal, scene, situation, and selection components."""
    result = ContextDependentInterpretationProcessor(config=make_config()).process(
        ambiguous_event(), context=finance_state()
    )
    assert "hippocampal_discourse" in result.biological
    assert "temporal_semantic" in result.biological
    assert "medial_prefrontal" in result.biological
    assert "posterior_midline" in result.biological
    assert "prefrontal_control" in result.biological
    assert "not anatomical replicas" in result.biological["scientific_scope"]


def test_every_result_has_exactly_16_trace_records():
    """Require one completed operation record for every registry stage."""
    result = ContextDependentInterpretationProcessor(config=make_config()).process(
        ambiguous_event(), context=finance_state()
    )
    assert len(result.operation_trace) == 16
    assert [item["operation_id"] for item in result.operation_trace] == [
        item.operation_id for item in OPERATION_REGISTRY
    ]
    assert all(item["status"] == "completed" for item in result.operation_trace)


def test_sqlite_annotation_registration_and_operation_logging(tmp_path):
    """Persist callable annotations and all 16 execution-stage records in the ORE logger."""
    database_path = tmp_path / "point12.sqlite3"
    result = ContextDependentInterpretationProcessor(config=make_config()).process(
        ambiguous_event(event_id="sqlite-event"), context=finance_state()
    )
    catalogue = register_annotations(database_path)
    assert catalogue["files_seen"] >= 5
    database = LoggerDatabase(database_path)
    summary = database.summary()
    assert summary["symbols"] > 0

    # Import here to exercise the public package export as a caller would use it.
    from ore_context_dependent_interpretation import record_result_to_sqlite

    logged = record_result_to_sqlite(result, database_path)
    assert logged["run_id"]
    with sqlite3.connect(database_path) as connection:
        run = connection.execute(
            "SELECT status, event_count FROM runs WHERE run_id = ?",
            (logged["run_id"],),
        ).fetchone()
        events = connection.execute(
            "SELECT COUNT(*) FROM events WHERE run_id = ?",
            (logged["run_id"],),
        ).fetchone()[0]
    assert run == ("completed", 16)
    assert events == 16


def test_invalid_contract_values_are_rejected():
    """Reject empty identifiers and non-probability candidate evidence at the boundary."""
    with pytest.raises(ValueError):
        InputEvent(event_id="", content="text")
    with pytest.raises(ValueError):
        InputEvent(
            event_id="bad-evidence", content="text",
            candidate_evidence={"candidate": 1.5},
        )
