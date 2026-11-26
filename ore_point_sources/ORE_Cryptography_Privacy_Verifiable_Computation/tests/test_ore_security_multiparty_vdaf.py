"""Acceptance tests: secure aggregation, MPC and VDAF (Phase H)."""
from __future__ import annotations

import numpy as np
import pytest

from ore_security.contracts import MPCSession
from ore_security.exceptions import AggregationError, MPCError, VDAFError
from ore_security.multiparty import (
    MPCProtocolEngine,
    PrimeField,
    SecureAggregationServer,
    AggregationParticipant,
    run_secure_aggregation_round,
)
from ore_security.vdaf import PrioAggregationRound, PrioClient


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


def _participants(count=4):
    return [AggregationParticipant(f"p{index + 1}") for index in range(count)]


def test_secure_aggregation_reveals_no_individual_contribution():
    """Acceptance: secure aggregation reveals no individual contribution."""
    participants = _participants(4)
    values = {"p1": [10, 1], "p2": [20, 2], "p3": [30, 3], "p4": [40, 4]}
    server = SecureAggregationServer(min_cohort_size=3)
    result = run_secure_aggregation_round(participants, values, server, "round-1")
    assert result.output == [100, 10]
    visible = server.server_visible_data("round-1")
    for participant, value in values.items():
        assert visible[participant] != value              # server sees masked data only
        assert visible[participant] != [0, 0]             # masks are nonzero in practice
    # the server cannot invert masked values without the pairwise seeds
    assert all(v not in values.values() for v in visible.values())
    assert result.validity_report["collusion_bound"]


def test_duplicate_and_replayed_submissions_are_rejected():
    """Acceptance: duplicate or replayed participant submissions are rejected."""
    participants = _participants(3)
    values = {"p1": [1, 1], "p2": [2, 2], "p3": [3, 3]}
    server = SecureAggregationServer(min_cohort_size=2)
    run_secure_aggregation_round(participants, values, server, "round-1")
    with pytest.raises(AggregationError, match="duplicate"):
        server.submit("round-1", "p1", [99, 99])
    with pytest.raises(AggregationError, match="not in the round roster"):
        server.submit("round-1", "p-outside", [1, 1])
    with pytest.raises(AggregationError, match="unknown round"):
        server.submit("round-missing", "p1", [1, 1])


def test_round_binding_prevents_cross_round_replay():
    """Masks are derived per round, so replaying a submission into another round corrupts nothing."""
    participants = _participants(3)
    values = {"p1": [5, 5], "p2": [6, 6], "p3": [7, 7]}
    server = SecureAggregationServer(min_cohort_size=2)
    first = run_secure_aggregation_round(participants, values, server, "round-A")
    second = run_secure_aggregation_round(participants, values, server, "round-B")
    assert first.output == [18, 18] and second.output == [18, 18]
    # an old masked vector submitted to the new round does not equal the honest mask
    peer_publics = {p.participant_id: p.public for p in participants}
    stale = participants[0].mask_for(["p1", "p2", "p3"], "round-A", values["p1"],
                                     PrimeField(), peer_publics)
    fresh = participants[0].mask_for(["p1", "p2", "p3"], "round-B", values["p1"],
                                     PrimeField(), peer_publics)
    assert stale != fresh


def test_dropout_recovery_restores_exact_aggregate():
    """Dropout handling: survivors reveal shared seeds; the aggregate stays exact."""
    participants = _participants(4)
    values = {"p1": [10, 1], "p2": [20, 2], "p3": [30, 3], "p4": [40, 4]}
    server = SecureAggregationServer(min_cohort_size=2)
    result = run_secure_aggregation_round(participants, values, server, "round-drop",
                                          dropped=("p3",))
    assert result.output == [70, 7]  # sum of the three submitted contributions


def test_mpc_result_matches_trusted_reference(rng):
    """Acceptance: MPC result matches trusted reference computation on test vectors."""
    engine = MPCProtocolEngine(rng=rng)
    session = engine.create_session(
        "sess-1", ["alice", "bob", "carol"], permitted_outputs=("sum", "inner_product"),
    )
    inputs = {"alice": [1, 2, 3], "bob": [4, 5, 6], "carol": [7, 8, 9]}
    result = engine.secure_sum(session, inputs)
    assert result["output"] == [12, 15, 18]
    inner = engine.secure_inner_product(session, "alice", [2, 3, 4], "bob", [5, 6, 7])
    assert inner["output"] == 2 * 5 + 3 * 6 + 4 * 7
    assert inner["beaver_triples_used"] == 3


def test_mpc_session_documents_collusion_and_dropout_assumptions(rng):
    """Acceptance: MPC protocol enforces documented collusion and dropout assumptions."""
    engine = MPCProtocolEngine(rng=rng)
    session = engine.create_session("sess-2", ["alice", "bob", "carol"])
    assert session.collusion_threshold == 2  # n - 1
    assert session.dropout_policy == "abort"
    assert "semi-honest" in session.threat_model
    with pytest.raises(ValueError):
        engine.create_session("sess-bad", ["solo"])          # needs >= 2 parties
    with pytest.raises(ValueError):
        engine.create_session("sess-bad2", ["a", "b", "c"], collusion_threshold=3)
    # non-participant inputs are refused
    with pytest.raises(MPCError):
        engine.secure_sum(session, {"mallory": [1, 1]})
    # outputs outside the permitted set are refused
    with pytest.raises(MPCError):
        engine.secure_inner_product(session, "alice", [1], "carol", [2])


def test_mpc_permitted_outputs_are_enforced(rng):
    """Only outputs listed in the session are computable."""
    engine = MPCProtocolEngine(rng=rng)
    session = engine.create_session("sess-3", ["a", "b"], permitted_outputs=("sum",))
    with pytest.raises(MPCError):
        engine.secure_inner_product(session, "a", [1, 2], "b", [3, 4])


def test_vdaf_rejects_malformed_measurement_shares(rng):
    """Acceptance: VDAF rejects malformed or invalid measurement shares."""
    client = PrioClient(rng=rng)
    round_state = PrioAggregationRound("vdaf-1", min_cohort_size=2, rng=rng)
    with pytest.raises(VDAFError):
        client.encode([0, 1, 2])                     # non-binary measurement
    with pytest.raises(VDAFError):
        client.encode([])
    with pytest.raises(VDAFError):
        round_state.submit("c1", {"protocol": "not-prio"})   # wrong protocol
    with pytest.raises(VDAFError):
        round_state.submit("c1", {"protocol": "prio-affine-agree-v1",
                                  "length": 3, "share_a": [1, 2], "share_b": [1, 2, 3]})
    with pytest.raises(VDAFError):
        round_state.submit("c1", {"protocol": "prio-affine-agree-v1",
                                  "length": 2, "share_a": ["x", "y"], "share_b": [1, 2]})


def test_vdaf_rejects_invalid_validity_check(rng):
    """Shares reconstructing a non-binary vector fail the AGREE validity check."""
    round_state = PrioAggregationRound("vdaf-2", min_cohort_size=1, rng=rng)
    field = round_state.field
    # shares of x = [1, 2] (invalid: 2 is not binary)
    share_a = [field.random(rng), field.random(rng)]
    share_b = [field.sub(x, a) for x, a in zip((1, 2), share_a)]
    verdict = round_state.submit("c-bad", {
        "protocol": "prio-affine-agree-v1", "length": 2,
        "share_a": share_a, "share_b": share_b,
    })
    assert not verdict["valid"]
    assert verdict["opened_check_value"] != 0


def test_vdaf_duplicate_participant_contributions_are_rejected(rng):
    """Duplicate participant submissions in one round are rejected."""
    client = PrioClient(rng=rng)
    round_state = PrioAggregationRound("vdaf-3", min_cohort_size=1, rng=rng)
    encoded = client.encode([1, 0, 1])
    assert round_state.submit("c1", encoded)["valid"]
    with pytest.raises(VDAFError, match="duplicate"):
        round_state.submit("c1", client.encode([0, 1, 0]))


def test_vdaf_aggregate_records_protocol_version_and_cohort(rng):
    """Acceptance: VDAF aggregate records protocol version and cohort metadata."""
    client = PrioClient(rng=rng)
    round_state = PrioAggregationRound("vdaf-4", min_cohort_size=2, rng=rng)
    for index in range(4):
        assert round_state.submit(f"c{index}", client.encode([1, 0, 1, 1]))["valid"]
    aggregate = round_state.aggregate()
    assert aggregate.protocol_ref == "prio-affine-agree-v1"
    assert aggregate.cohort_size == 4
    assert aggregate.output == [4, 0, 4, 4]
    report = aggregate.validity_report
    assert report["protocol_version"] == "prio-affine-agree-v1"
    assert report["valid_contributions"] == 4


def test_small_cohort_suppression_prevents_disclosure(rng):
    """Acceptance: small-cohort suppression prevents configured disclosure risks."""
    client = PrioClient(rng=rng)
    round_state = PrioAggregationRound("vdaf-5", min_cohort_size=3, rng=rng)
    round_state.submit("c1", client.encode([1, 1]))
    aggregate = round_state.aggregate()
    assert aggregate.suppressed
    assert aggregate.output is None
    assert "below minimum" in aggregate.suppression_reason


def test_field_arithmetic_is_consistent():
    """Prime-field helpers used by all multiparty protocols behave modularly."""
    field = PrimeField()
    assert field.add(field.modulus - 1, 2) == 1
    assert field.mul(3, field.sub(0, 3)) == field.modulus - 9
    assert field.mul(field.inv(7), 7) == 1
