"""End-to-end tests for the N9 language adapter stack."""
from __future__ import annotations

import re

from ore_discourse import DiscourseConfig, DiscourseProcessor
from ore_lexical_access import LexicalAccessConfig, LexicalAccessProcessor
from ore_lexical_context import LexicalEntry, LexicalMeaning
from ore_n9 import N9Patch, N9Processor, N9Request, build_n9_language_adapters
from ore_semantic import SemanticConcept
from ore_semantic_composition import (
    PredicateFrame,
    SemanticCompositionConfig,
    SemanticCompositionProcessor,
)
from ore_syntax import SyntaxConfig, SyntaxProcessor


def _language_stack():
    """Build a tiny explicit lexicon and world-knowledge frame for the test domain."""
    lexicon = [
        LexicalEntry(
            "musician", "musician", frequency=100,
            meanings=(LexicalMeaning(
                "musician-person",
                {"HUMAN": 1.0, "PERSON": 1.0, "MUSICIAN": 1.0, "ANIMATE": 1.0},
            ),),
            lexical_category="noun", language="en",
        ),
        LexicalEntry(
            "tunes-verb", "tunes", frequency=80,
            meanings=(LexicalMeaning(
                "tune-event", {"EVENT": 1.0, "ACTION": 1.0, "ADJUSTMENT": 1.0},
            ),),
            lexical_category="verb", language="en",
        ),
        LexicalEntry(
            "instrument", "instrument", frequency=90,
            meanings=(LexicalMeaning(
                "instrument-object",
                {"MUSICAL_INSTRUMENT": 1.0, "OBJECT": 1.0, "ARTIFACT": 1.0},
            ),),
            lexical_category="noun", language="en",
        ),
        LexicalEntry(
            "microwave", "microwave", frequency=70,
            meanings=(LexicalMeaning(
                "microwave-appliance",
                {"APPLIANCE": 1.0, "OBJECT": 1.0, "HOUSEHOLD": 1.0},
            ),),
            lexical_category="noun", language="en",
        ),
    ]
    concepts = [
        SemanticConcept(
            "musician-person", "musician", semantic_features={
                "HUMAN": 1.0, "PERSON": 1.0, "MUSICIAN": 1.0, "ANIMATE": 1.0,
            }, lexical_category="noun", language="en",
        ),
        SemanticConcept(
            "tune-event", "tune", aliases=("tunes",), semantic_features={
                "EVENT": 1.0, "ACTION": 1.0, "ADJUSTMENT": 1.0,
            }, lexical_category="verb", language="en",
        ),
        SemanticConcept(
            "instrument-object", "instrument", semantic_features={
                "MUSICAL_INSTRUMENT": 1.0, "OBJECT": 1.0, "ARTIFACT": 1.0,
            }, lexical_category="noun", language="en",
        ),
        SemanticConcept(
            "microwave-appliance", "microwave", semantic_features={
                "APPLIANCE": 1.0, "OBJECT": 1.0, "HOUSEHOLD": 1.0,
            }, lexical_category="noun", language="en",
        ),
    ]
    tune_frame = PredicateFrame(
        predicate="tunes",
        roles=("Agent", "Patient"),
        required_roles=("Agent", "Patient"),
        selectional_preferences={
            "Agent": {"HUMAN": 1.0, "PERSON": 1.0},
            "Patient": {"MUSICAL_INSTRUMENT": 1.0},
        },
        predicate_concept_id="tune-event",
        source_reference="test:musician-tunes-instrument-frame",
    )
    lexical = LexicalAccessProcessor(
        lexicon, config=LexicalAccessConfig(simulate_biology=False)
    )
    syntax = SyntaxProcessor(config=SyntaxConfig(simulate_biology=False))
    semantic = SemanticCompositionProcessor(
        concepts,
        predicate_frames=(tune_frame,),
        config=SemanticCompositionConfig(simulate_biology=False),
    )
    discourse = DiscourseProcessor(config=DiscourseConfig(simulate_biology=False))
    adapters = build_n9_language_adapters(
        lexical_access=lexical,
        syntax=syntax,
        semantic_composition=semantic,
        discourse=discourse,
        language="en",
    )
    return adapters


def _n9_request(sequence_id: str, text: str) -> N9Request:
    """Encode text as one one-hot grapheme posterior per already-localized patch."""
    patches = [
        N9Patch(
            patch_id=f"{sequence_id}-g{index}",
            grapheme_probabilities={character: 1.0},
        )
        for index, character in enumerate(text)
    ]
    spans = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
    return N9Request(
        sequence_id=sequence_id,
        patches=patches,
        word_spans=spans,
        task="comprehension",
        script_profile={"script": "Latin", "reading_direction": "ltr"},
    )


def test_n9_runs_lexical_syntax_semantic_and_discourse_for_instrument_vs_microwave():
    """Verify all four processes run through N9 and selectional evidence distinguishes objects."""
    processor = N9Processor(adapters=_language_stack())

    instrument_result = processor.process(
        _n9_request("instrument-case", "musician tunes instrument")
    )
    microwave_result = processor.process(
        _n9_request("microwave-case", "musician tunes microwave")
    )

    for result, expected_object in (
        (instrument_result, "instrument"),
        (microwave_result, "microwave"),
    ):
        assert result.diagnostics["valid"] is True
        assert result.diagnostics["executed_operation_count"] == 120
        assert [result.operation_trace[index]["status"] for index in (52, 61, 69, 75)] == [
            "ok", "ok", "ok", "ok",
        ]
        assert result.syntax["tokens"] == ["musician", "tunes", expected_object]
        assert len(result.propositions) == 1
        assert result.propositions[0]["predicate"] == "tune"
        assert result.propositions[0]["arguments"]["Agent"] == "token:0"
        assert result.propositions[0]["arguments"]["Patient"] == "token:2"
        assert result.context_model["situation_model"]["events"]
        assert result.semantic_graph["nodes"]
        assert result.provenance["upstream_stages"]["n6"] == "available"
        assert result.provenance["upstream_stages"]["syntax"] == "available"
        assert result.provenance["upstream_stages"]["n7"] == "available"
        assert result.provenance["upstream_stages"]["n8"] == "available"

    def patient_compatibility(result, surface: str) -> float:
        argument = next(
            item for item in result.propositions[0]["argument_details"]
            if item["surface"] == surface
        )
        assert argument["role"] == "Patient"
        return float(argument["selectional_compatibility"])

    instrument_fit = patient_compatibility(instrument_result, "instrument")
    microwave_fit = patient_compatibility(microwave_result, "microwave")
    assert instrument_fit > microwave_fit
    assert instrument_fit > 0.5
    assert microwave_fit < 0.1
