"""Minimal functional + biological Integrated Reading Algorithm example."""
from ore_reading import IntegratedReadingProcessor, ReadingLexiconEntry, ReadingRequest


lexicon = [
    ReadingLexiconEntry(
        "word-read", "read", ("R", "IY", "D"), 40.0,
        {"READING": 1.0, "KNOWLEDGE": 0.7}, "verb",
        verified_by="demo-review", source_reference="demo://lexicon/read",
    ),
    ReadingLexiconEntry(
        "word-reader", "reader", ("R", "IY", "D", "ER"), 18.0,
        {"PERSON": 1.0, "READING": 0.8}, "noun",
        verified_by="demo-review", source_reference="demo://lexicon/reader",
    ),
]

result = IntegratedReadingProcessor(lexicon).process(
    ReadingRequest("example-001", text="The reader can read.")
)
print("Decoded:", result.decoded_text)
print("Pronunciation:", result.tokens[-1]["pronunciation"])
print("Operation count:", len(result.operation_trace))
print("Biological layers:", list(result.biological.get("layers", {})))
