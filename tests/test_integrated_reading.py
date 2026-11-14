from __future__ import annotations

import numpy as np
import pytest

from ore_reading import (
    G2PRule,
    IntegratedReadingProcessor,
    LIFConfig,
    LIFPopulation,
    ReadingConfig,
    ReadingLexiconEntry,
    ReadingRequest,
    OPERATION_REGISTRY,
    get_operation,
    reading_request_from_n6_result,
)


@pytest.fixture
def lexicon():
    return [
        ReadingLexiconEntry("read-v", "read", ("R", "IY", "D"), 40,
                           {"READING": 1.0, "KNOWLEDGE": 0.6}, "verb",
                           verified_by="reviewer", source_reference="dataset://read"),
        ReadingLexiconEntry("reader-n", "reader", ("R", "IY", "D", "ER"), 20,
                           {"PERSON": 1.0, "READING": 0.7}, "noun",
                           verified_by="reviewer", source_reference="dataset://reader"),
        ReadingLexiconEntry("red-a", "red", ("R", "EH", "D"), 30,
                           {"COLOR": 1.0}, "adjective",
                           verified_by="reviewer", source_reference="dataset://red"),
        ReadingLexiconEntry("bread-n", "bread", ("B", "R", "EH", "D"), 15,
                           {"FOOD": 1.0}, "noun",
                           verified_by="reviewer", source_reference="dataset://bread"),
    ]


def test_registry_has_72_unique_ordered_operations():
    assert len(OPERATION_REGISTRY) == 72
    assert [item.operation_id for item in OPERATION_REGISTRY] == [f"IR-{i:03d}" for i in range(1, 73)]
    assert get_operation("IR-016").name == "Recurrent excitation and competition"


def test_end_to_end_functional_and_biological_pipeline(lexicon):
    processor = IntegratedReadingProcessor(lexicon)
    result = processor.process(ReadingRequest("s1", text="The reader can read."))
    assert result.diagnostics["valid"] is True
    assert result.diagnostics["operation_trace_complete"] is True
    assert len(result.operation_trace) == 72
    assert result.operation_trace[-1]["operation_id"] == "IR-072"
    assert len(result.tokens) == 4
    assert all(token["pronunciation"] for token in result.tokens)
    assert result.biological["model"] == "deterministic_current_based_LIF_population_reference"
    assert set(result.biological["layers"]) == {
        "orthographic", "lexical", "phonological", "auditory_wordform", "semantic", "control"
    }
    assert result.biological["layers"]["orthographic"]["units"]["voltage"] == "mV"
    assert result.biological["layers"]["orthographic"]["units"]["firing_rate"] == "Hz"


def test_exact_lexical_route_and_unknown_sublexical_fallback(lexicon):
    processor = IntegratedReadingProcessor(lexicon)
    result = processor.process(ReadingRequest("s2", tokens=["read", "λx"]))
    assert result.tokens[0]["pronunciation_source"] == "lexical"
    assert result.tokens[0]["pronunciation"] == ["R", "IY", "D"]
    assert "<UNK_PHONEME>" in result.tokens[1]["sublexical_pronunciation"]
    assert result.tokens[1]["status"] == "ABSTAIN"


def test_bayesian_posterior_rows_sum_to_one(lexicon):
    result = IntegratedReadingProcessor(lexicon).process(ReadingRequest("s3", text="red bread"))
    for token in result.tokens:
        values = [item["posterior"] for item in token["shortlist"]]
        assert not values or np.isclose(sum(values), 1.0)


def test_biology_can_be_disabled_but_registry_trace_remains_complete(lexicon):
    config = ReadingConfig(simulate_biology=False)
    result = IntegratedReadingProcessor(lexicon, config=config).process(
        ReadingRequest("s4", tokens=["read"])
    )
    assert len(result.operation_trace) == 72
    assert result.operation_trace[65]["status"] == "SKIPPED"
    assert result.operation_trace[70]["operation_id"] == "IR-071"
    assert result.biological["status"] == "DISABLED"


def test_lif_population_generates_reproducible_physiological_units():
    config = LIFConfig(steps=40, dt_ms=1.0)
    population = LIFPopulation(3, config)
    first = population.simulate(np.array([1.0, 0.5, 0.0]))
    second = population.simulate(np.array([1.0, 0.5, 0.0]))
    assert first == second
    assert first["spike_counts"][0] >= first["spike_counts"][2]
    assert first["units"] == {"voltage": "mV", "current": "pA", "firing_rate": "Hz", "time": "ms"}


def test_request_contract_rejects_ambiguous_input():
    with pytest.raises(ValueError, match="exactly one"):
        ReadingRequest("bad", text="hello", tokens=["hello"])


def test_lexicon_provenance_and_frequency_are_validated():
    with pytest.raises(ValueError, match="positive"):
        ReadingLexiconEntry("bad", "word", ("W",), 0)


def test_custom_g2p_rule_is_applied_by_longest_match():
    rules = [G2PRule("sh", ("SH",), 1.0), G2PRule("s", ("S",), 0.9), G2PRule("h", ("HH",), 0.9)]
    processor = IntegratedReadingProcessor((), g2p_rules=rules)
    result = processor.process(ReadingRequest("g2p", tokens=["sh"]))
    assert result.tokens[0]["sublexical_pronunciation"] == ["SH"]


def test_to_dict_is_json_serializable(lexicon):
    import json

    result = IntegratedReadingProcessor(lexicon).process(ReadingRequest("s5", text="read"))
    serialized = json.dumps(result.to_dict())
    assert '"schema": "ore.integrated_reading.v1"' in serialized


def test_n6_adapter_preserves_orthographic_token_order():
    request = reading_request_from_n6_result({
        "sequence_id": "n6-seq",
        "tokens": [
            {"orthography": "read", "graphemes": ["r", "e", "a", "d"]},
            {"orthography": "brain", "graphemes": ["b", "r", "a", "i", "n"]},
        ],
        "metadata": {"schema": "ore.n6.v1"},
    })
    assert request.sequence_id == "n6-seq"
    assert tuple(request.tokens) == ("read", "brain")
    assert request.source_reference == "ore.n6.v1"
