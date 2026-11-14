"""Run a deterministic lexical-access example."""
from ore_lexical_context import LexicalEntry, LexicalMeaning, MorphemeEntry
from . import LexicalAccessProcessor, LexicalAccessRequest


def main() -> None:
    lexicon = [
        LexicalEntry("cat", "cat", frequency=24, meanings=(
            LexicalMeaning("feline", {"ANIMAL": 1.0, "FELINE": 1.0}),
        ), lexical_category="noun", language="en", verified_by="demo",
            source_reference="dataset://ore-demo/cat"),
        LexicalEntry("bank-river", "bank", frequency=18, meanings=(
            LexicalMeaning("river-bank", {"RIVER": 1.0, "LAND": 0.7}),
        ), lexical_category="noun", language="en"),
        LexicalEntry("bank-finance", "bank", frequency=20, meanings=(
            LexicalMeaning("financial-bank", {"MONEY": 1.0, "INSTITUTION": 0.8}),
        ), lexical_category="noun", language="en"),
    ]
    morphemes = [MorphemeEntry(
        "plural-s", "s", "suffix", {"PLURAL": 1.0}, frequency=20,
        productivity=0.95, verified_by="demo", source_reference="dataset://ore-demo/plural-s",
    )]
    processor = LexicalAccessProcessor(lexicon, morphemes=morphemes)
    result = processor.process(LexicalAccessRequest("demo-001", "cats", language="en"))
    print("Identity candidates:", [item["entry_id"] for item in result.lexical_identity_candidates])
    print("Morphology:", result.preferred_morphology)
    print("Operation count:", len(result.operation_trace))
    print("Biology numerically valid:", result.biological["numerically_valid"])


if __name__ == "__main__":
    main()
