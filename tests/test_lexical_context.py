from ore_lexical_context import (
    G2PRule, LexicalContextConfig, LexicalContextProcessor, LexicalContextRequest,
    LexicalEntry, LexicalMeaning, MorphemeEntry, OPERATION_REGISTRY,
    lexical_context_request_from_reading_result,
)

def _bank_processor(**kwargs):
    bank = LexicalEntry("bank", "bank", frequency=30, meanings=(
        LexicalMeaning("river-bank", {"RIVER": 1.0, "LAND": 0.7}, prior=1.0),
        LexicalMeaning("financial-bank", {"MONEY": 1.0, "INSTITUTION": 0.7}, prior=1.0),
    ))
    return LexicalContextProcessor((bank,), **kwargs)

def test_registry_is_complete_and_ordered():
    assert len(OPERATION_REGISTRY) == 40
    assert [x.operation_id for x in OPERATION_REGISTRY] == [f"LC-{i:03d}" for i in range(1, 41)]
    assert all(x.mathematical_form and x.biological_reference for x in OPERATION_REGISTRY)

def test_all_steps_execute_and_emit_biological_trace():
    processor = _bank_processor()
    result = processor.process(LexicalContextRequest("bank-1", text="bank", context_features={"RIVER": 1.0}))
    assert len(result.operation_trace) == 40
    assert result.diagnostics["valid"] is True
    assert len(result.biological["operation_trace"]) == 40
    assert result.biological["numerically_valid"] is True
    assert result.tokens[0]["selected_meaning"]["meaning_id"] == "river-bank"

def test_ambiguous_word_retains_competing_meanings_without_context():
    processor = _bank_processor()
    result = processor.process(LexicalContextRequest("bank-neutral", text="bank"))
    token = result.tokens[0]
    assert len(token["meaning_candidates"]) == 2
    assert token["abstained"] is True
    assert token["selected_meaning"] is None

def test_supplied_morpheme_inventory_supports_suffix_decomposition():
    cat = LexicalEntry("cat", "cat", frequency=20, meanings=(
        LexicalMeaning("feline", {"ANIMAL": 1.0, "FELINE": 1.0}),
    ))
    plural = MorphemeEntry("plural-s", "s", "suffix", {"PLURAL": 1.0}, productivity=0.9)
    processor = LexicalContextProcessor((cat,), morphemes=(plural,))
    result = processor.process(LexicalContextRequest("cats", text="cats"))
    candidates = result.tokens[0]["morphological_candidates"]
    assert candidates
    assert candidates[0]["split"]["stem"] == "cat"
    assert candidates[0]["features"]["PLURAL"] == 1.0

def test_g2p_rule_competition_is_reported_with_weights():
    rules = (G2PRule("c", ("K",), .95), G2PRule("a", ("AE",), .9),
             G2PRule("t", ("T",), .95))
    cat = LexicalEntry("cat", "cat", frequency=20)
    processor = LexicalContextProcessor((cat,), g2p_rules=rules)
    result = processor.process(LexicalContextRequest("pronunciation", text="cat"))
    alternatives = result.tokens[0]["pronunciation_alternatives"]["cat"]
    assert alternatives
    assert sum(x["weight"] for x in alternatives) == pytest.approx(1.0)

def test_repetition_history_and_exposure_updates_are_stateful():
    processor = _bank_processor()
    first = processor.process(LexicalContextRequest("first", text="bank", timestamp_ms=0))
    second = processor.process(LexicalContextRequest("second", text="bank", timestamp_ms=100))
    assert second.tokens[0]["operation_count"] == 40
    assert second.tokens[0]["candidate_words"][0]["activation"] >= 0
    assert second.metadata["exposure_state"] == "updated"

def test_reading_result_adapter_preserves_token_sequence():
    request = lexical_context_request_from_reading_result({
        "sequence_id": "read-01",
        "tokens": [{"orthography": "bank"}, {"orthography": "cat"}],
    }, language="en", context_features={"RIVER": 0.4})
    assert request.tokens == ["bank", "cat"]
    assert request.language == "en"
    assert request.context_features["RIVER"] == 0.4

def test_exposure_recording_can_be_disabled():
    config = LexicalContextConfig(record_exposures=False, simulate_biology=False)
    result = _bank_processor(config=config).process(LexicalContextRequest("readonly", text="bank"))
    assert result.metadata["exposure_state"] == "read_only"
    assert result.biological["enabled"] is False
    assert result.diagnostics["operation_count"] == 40

# pytest is intentionally imported here to keep the runtime package dependency-free.
import pytest
