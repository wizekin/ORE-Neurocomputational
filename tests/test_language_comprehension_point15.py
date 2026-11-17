"""Tests for Point 15 temporal and causal research validation."""
from __future__ import annotations

import inspect
import pytest

import ore_language_comprehension.point15 as point15
from ore_language_comprehension.point15 import (
    POINT15_STAGES, Point15MechanisticResearchProcessor, Point15Request,
)


def make_request(**kwargs):
    """Build a compact research request with controlled temporal and causal evidence."""
    defaults = {
        "research_question": "Does context alter temporal coupling during sentence comprehension?",
        "conditions": [
            {"condition_id":"literal","factors":{"context":"neutral"},"lexical_frequency":5},
            {"condition_id":"pragmatic","factors":{"context":"supportive"},"lexical_frequency":5},
        ],
        "event_times":[{"event":"critical_word","time":120}],
        "behavioural_trials":[
            {"condition":"literal","context":"neutral","outcome":1.0,"response_time":800,"split":"train"},
            {"condition":"pragmatic","context":"supportive","outcome":0.8,"response_time":950,"split":"train"},
            {"condition":"pragmatic","context":"neutral","outcome":0.6,"response_time":980,"split":"train"},
            {"condition":"test","context":"supportive","outcome":1.0,"predicted_probability":0.9,"split":"held_out"},
        ],
        "neural_records":[
            {"participant_id":"p1","region":"temporal","signal":"temporal","time":130,"event_time":120,"value":1.0,"modality":"EEG"},
            {"participant_id":"p1","region":"frontal","signal":"frontal","time":130,"event_time":120,"value":0.8,"modality":"EEG"},
            {"participant_id":"p1","region":"temporal","signal":"temporal","time":150,"event_time":120,"value":0.7,"modality":"EEG"},
            {"participant_id":"p1","region":"frontal","signal":"frontal","time":150,"event_time":120,"value":0.6,"modality":"EEG"},
            {"participant_id":"p1","region":"temporal","signal":"temporal","time":170,"event_time":120,"value":0.4,"modality":"EEG"},
            {"participant_id":"p1","region":"frontal","signal":"frontal","time":170,"event_time":120,"value":0.3,"modality":"EEG"},
        ],
        "network_edges":[{"source":"temporal","target":"frontal","weight":0.4,"time":130,"kind":"effective"}],
        "interventions":[{"method":"TMS","target":"frontal","outcome":0.6,"control_outcome":0.9,"approved":True}],
        "population_groups":[{"group_id":"g1","language":"L1","cultural_context":"A","measurement_invariance_tested":True}],
        "hypotheses":[{"hypothesis_id":"recurrent","family":"recurrent_interaction",
                       "prediction":{"context":"increases coupling"}}],
    }
    defaults.update(kwargs)
    return Point15Request(**defaults)


def test_source_ordered_trace_has_all_25_steps_and_biological_caveats():
    result = Point15MechanisticResearchProcessor().process(make_request())
    assert len(result.trace) == len(POINT15_STAGES) == 25
    assert [x["step"] for x in result.trace] == list(range(1,26))
    assert all(x["biological_reference"] and x["scientific_caveat"] for x in result.trace)


def test_event_alignment_and_network_are_explicit():
    result = Point15MechanisticResearchProcessor().process(make_request())
    assert result.trace[6]["output"]["ordered_events"][0]["time"] == 120
    assert result.trace[11]["output"]["aligned_signal_count"] == 4
    assert result.network["nodes"] == ["frontal","temporal"]
    assert result.network["adjacency_matrix"][1][0] == pytest.approx(0.4)


def test_temporal_association_does_not_claim_causality():
    result = Point15MechanisticResearchProcessor().process(make_request())
    assert result.network["temporal_associations"]["temporal|frontal"]["paired_samples"] == 2
    assert result.network["temporal_associations"]["temporal|frontal"]["zero_lag_correlation"] is not None
    assert result.causal["causal_claim_must_not_exceed_evidence"] is True


def test_intervention_evidence_requires_approval_and_complete_pair():
    result = Point15MechanisticResearchProcessor().process(make_request(
        interventions=[
            {"method":"TMS","outcome":0.4,"control_outcome":0.8,"approved":False},
            {"method":"lesion","outcome":0.7,"approved":True},
        ]))
    assert result.causal["intervention_effects"] == []
    assert result.trace[20]["output"]["approved_intervention_count"] == 0


def test_heldout_validation_does_not_reuse_training_rows():
    result = Point15MechanisticResearchProcessor().process(make_request())
    assert result.validation["held_out_n"] == 1
    assert result.validation["brier_score"] == pytest.approx(0.01)
    assert result.validation["training_data_reused_as_test"] is False


def test_population_comparability_is_not_assumed():
    result = Point15MechanisticResearchProcessor().process(make_request(
        population_groups=[{"group_id":"g1","language":"L1","measurement_invariance_tested":False}]))
    assert result.trace[22]["output"]["cross_group_comparability_established"] is False


def test_missing_neural_data_stays_missing_and_reported():
    result = Point15MechanisticResearchProcessor().process(make_request(
        neural_records=[], network_edges=[], interventions=[]))
    assert result.temporal == {}
    assert result.report["missing_evidence"] == [
        "neural_association","temporal_relationship","model_based_direction",
        "perturbational_evidence"]
    assert result.diagnostics["neural_values_fabricated"] is False


def test_context_effects_require_multiple_levels_and_are_reported():
    result = Point15MechanisticResearchProcessor().process(make_request())
    assert result.network["context_effects"]["pragmatic"]["level_means"]["supportive"] == pytest.approx(0.8)


def test_request_deep_copies_inputs_and_validates_question():
    rows=[{"region":"temporal"}]
    request=make_request(neural_records=rows)
    rows[0]["region"]="changed"
    assert request.neural_records[0]["region"]=="temporal"
    with pytest.raises(ValueError):
        Point15Request(research_question=" ")


def test_public_point15_definitions_have_docstrings():
    public=[v for n,v in vars(point15).items() if not n.startswith("_")
            and (inspect.isclass(v) or inspect.isfunction(v))]
    assert public and all(inspect.getdoc(v) for v in public)
