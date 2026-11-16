import pytest

from ore_lexical_context import LexicalEntry, LexicalMeaning, MorphemeEntry
from ore_lexical_access import (
    LexicalAccessConfig, LexicalAccessProcessor, LexicalAccessRequest,
    OPERATION_REGISTRY, POPULATIONS, STAGE_REGISTRY,
)


def test_registry_contains_seven_stages_and_36_numbered_operations():
    assert len(STAGE_REGISTRY) == 7
    assert [item["operation_count"] for item in STAGE_REGISTRY] == [5, 5, 4, 5, 4, 5, 8]
    assert len(OPERATION_REGISTRY) == 36
    assert [item.operation_id for item in OPERATION_REGISTRY] == [f"BLA-{i:03d}" for i in range(1, 37)]
    assert all(item.mathematical_form and item.biological_reference for item in OPERATION_REGISTRY)


def test_identity_and_meaning_retrieval_preserve_homographic_senses():
    bank_river = LexicalEntry(
        "bank-river", "bank", frequency=20,
        meanings=(LexicalMeaning("river-bank", {"RIVER": 1.0, "LAND": 0.7}),),
        lexical_category="noun", language="en",
    )
    bank_finance = LexicalEntry(
        "bank-finance", "bank", frequency=20,
        meanings=(LexicalMeaning("financial-bank", {"MONEY": 1.0, "INSTITUTION": 0.8}),),
        lexical_category="noun", language="en",
    )
    processor = LexicalAccessProcessor([bank_river, bank_finance], config=LexicalAccessConfig(simulate_biology=False))
    result = processor.process(LexicalAccessRequest("bank-01", "bank", context_features={"RIVER": 1.0}))
    assert len(result.lexical_identity_candidates) == 2
    assert len(result.meaning_candidates) == 2
    assert result.meaning_candidates[0]["meaning_id"] == "river-bank"
    assert result.diagnostics["valid"] is True
    assert len(result.operation_trace) == 36


def test_grammatical_category_uses_lexical_evidence_and_context_prior():
    noun = LexicalEntry("lead-noun", "lead", frequency=10, lexical_category="noun")
    verb = LexicalEntry("lead-verb", "lead", frequency=10, lexical_category="verb")
    processor = LexicalAccessProcessor([noun, verb], config=LexicalAccessConfig(simulate_biology=False))
    result = processor.process(LexicalAccessRequest(
        "lead-verb", "lead", contextual_category_priors={"verb": 1.0, "noun": 0.05}
    ))
    assert result.grammatical_categories[0]["category"] == "verb"
    assert result.grammatical_categories[0]["posterior"] > result.grammatical_categories[1]["posterior"]


def test_morphological_decomposition_retrieves_stem_affix_and_composed_features():
    cat = LexicalEntry(
        "cat", "cat", frequency=24,
        meanings=(LexicalMeaning("feline", {"ANIMAL": 1.0, "FELINE": 1.0}),),
        lexical_category="noun", language="en",
    )
    plural = MorphemeEntry(
        "plural-s", "s", "suffix", {"PLURAL": 1.0},
        frequency=20, productivity=0.95,
    )
    processor = LexicalAccessProcessor(
        [cat], morphemes=[plural], morpheme_category_cues={"plural-s": {"noun": 0.9}},
        config=LexicalAccessConfig(simulate_biology=False),
    )
    result = processor.process(LexicalAccessRequest("cats-01", "cats", language="en"))
    assert result.preferred_morphology is not None
    assert result.preferred_morphology["stem"] == "cat"
    assert result.preferred_morphology["suffix_id"] == "plural-s"
    retrieved = result.stem_affix_retrieval[0]
    assert retrieved["composed_features"]["PLURAL"] == pytest.approx(0.475)
    assert retrieved["composed_features"]["ANIMAL"] == 1.0
    assert result.operation_trace[22]["operation_id"] == "BLA-023"


def test_competing_morphological_alternatives_are_preserved_and_can_abstain():
    cat = LexicalEntry("cat", "cat", frequency=10, meanings=(LexicalMeaning("cat", {"ANIMAL": 1.0}),), lexical_category="noun")
    ca = LexicalEntry("ca", "ca", frequency=10, meanings=(LexicalMeaning("ca", {"ANIMAL": 1.0}),), lexical_category="noun")
    suffix_s = MorphemeEntry("suffix-s", "s", "suffix", {"PLURAL": 1.0}, frequency=10, productivity=0.8)
    suffix_ts = MorphemeEntry("suffix-ts", "ts", "suffix", {"PLURAL": 1.0}, frequency=10, productivity=0.8)
    processor = LexicalAccessProcessor(
        [cat, ca], morphemes=[suffix_s, suffix_ts],
        config=LexicalAccessConfig(simulate_biology=False, morphology_temperature=1.0),
    )
    result = processor.process(LexicalAccessRequest("ambiguous-morph", "cats"))
    segmented = [item for item in result.morphological_alternatives if item["kind"] == "segmented"]
    assert len(segmented) >= 2
    assert result.diagnostics["morphology_abstained"] is True
    assert result.preferred_morphology is None



def test_derivational_and_inflectional_affixes_are_explicitly_separated():
    """Keep derivation and inflection as distinct evidence channels when inventory labels them."""
    work = LexicalEntry(
        "work", "work", frequency=30,
        meanings=(LexicalMeaning("work-activity", {"ACTIVITY": 1.0, "WORK": 1.0}),),
        lexical_category="verb", language="en",
    )
    re_prefix = MorphemeEntry(
        "re-prefix", "re", "prefix", {"REPETITION": 1.0},
        frequency=12, productivity=0.8,
    )
    plural_s = MorphemeEntry(
        "s-inflection", "s", "suffix", {"PRESENT_THIRD_PERSON": 1.0},
        frequency=25, productivity=0.95,
    )
    processor = LexicalAccessProcessor(
        [work],
        morphemes=[re_prefix, plural_s],
        morpheme_affix_categories={
            "re-prefix": "derivational",
            "s-inflection": "inflectional",
        },
        config=LexicalAccessConfig(simulate_biology=False),
    )
    result = processor.process(LexicalAccessRequest("reworks", "reworks", language="en"))
    segmented = next(item for item in result.stem_affix_retrieval if item["stem"] == "work")

    assert [item["morpheme_id"] for item in segmented["derivational_affixes"]] == ["re-prefix"]
    assert [item["morpheme_id"] for item in segmented["inflectional_affixes"]] == ["s-inflection"]
    ranked = next(item for item in result.morphological_alternatives
                  if item["hypothesis_id"] == segmented["hypothesis_id"])
    assert ranked["derivational_affix_ids"] == ["re-prefix"]
    assert ranked["inflectional_affix_ids"] == ["s-inflection"]
    assert ranked["derivational_affix_support"] > 0.0
    assert ranked["inflectional_affix_support"] > 0.0
    assert ranked["affix_classification_complete"] is True
    assert result.operation_trace[20]["operation_id"] == "BLA-021"
    assert result.operation_trace[20]["output"]["affix_records"][0]["classification_source"] == "explicit_caller_inventory"


def test_unclassified_affixes_are_not_guessed_to_be_derivational_or_inflectional():
    """Preserve legacy morpheme inventories and avoid fabricated morphological classes."""
    root = LexicalEntry("cat", "cat", frequency=10, lexical_category="noun")
    suffix = MorphemeEntry("plural-s", "s", "suffix", {"PLURAL": 1.0})
    result = LexicalAccessProcessor(
        [root], morphemes=[suffix],
        config=LexicalAccessConfig(simulate_biology=False),
    ).process(LexicalAccessRequest("cats", "cats"))

    segmented = next(item for item in result.stem_affix_retrieval if item["stem"] == "cat")
    assert segmented["derivational_affixes"] == []
    assert segmented["inflectional_affixes"] == []
    assert [item["morpheme_id"] for item in segmented["unspecified_affixes"]] == ["plural-s"]

def test_incremental_activation_has_one_frame_per_character_and_bounded_activities():
    cat = LexicalEntry("cat", "cat", frequency=24, lexical_category="noun")
    cats = LexicalEntry("cats", "cats", frequency=16, lexical_category="noun")
    processor = LexicalAccessProcessor([cat, cats], config=LexicalAccessConfig(simulate_biology=False))
    result = processor.process(LexicalAccessRequest("incremental", "cats"))
    assert len(result.incremental_activation) == 4
    assert [item["observed_prefix"] for item in result.incremental_activation] == ["c", "ca", "cat", "cats"]
    assert all(0.0 <= candidate["activation"] <= 1.0
               for frame in result.incremental_activation for candidate in frame["candidates"])
    assert result.incremental_activation[-1]["candidates"][0]["entry_id"] == "cats"


def test_biological_reference_advances_seven_coupled_populations():
    word = LexicalEntry("word", "word", frequency=10, lexical_category="noun")
    result = LexicalAccessProcessor([word]).process(LexicalAccessRequest("biology", "word"))
    assert result.biological["enabled"] is True
    assert result.biological["numerically_valid"] is True
    assert result.biological["layer_order"] == list(POPULATIONS)
    assert len(result.biological["layers"]) == 7
    assert result.biological["simulated_time_ms"] > 0
    assert all("mean_firing_rate_hz" in result.biological["layers"][name] for name in POPULATIONS)


def test_unknown_word_still_emits_complete_trace_without_fabricating_lexical_candidates():
    known = LexicalEntry("known", "known", frequency=10, lexical_category="noun")
    result = LexicalAccessProcessor([known], config=LexicalAccessConfig(simulate_biology=False)).process(
        LexicalAccessRequest("unknown", "xyz")
    )
    assert result.lexical_identity_candidates == []
    assert result.meaning_candidates == []
    assert len(result.operation_trace) == 36
    assert result.diagnostics["valid"] is True


def test_input_and_configuration_validation():
    with pytest.raises(ValueError):
        LexicalAccessRequest("bad", "two words")
    with pytest.raises(ValueError):
        LexicalAccessConfig(activation_temperature=0.0)
    with pytest.raises(ValueError):
        LexicalAccessProcessor([])


def test_minimum_valid_frequency_inventory_keeps_identity_scores_finite():
    entry = LexicalEntry("minimum", "minimum", frequency=1, lexical_category="noun")
    processor = LexicalAccessProcessor([entry], config=LexicalAccessConfig(simulate_biology=False))
    result = processor.process(LexicalAccessRequest("minimum-frequency", "minimum"))
    assert result.lexical_identity_candidates[0]["frequency_strength"] == pytest.approx(1.0)
    assert result.diagnostics["valid"] is True
