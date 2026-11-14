"""Run a small reproducible Integrated Reading Algorithm example."""
from __future__ import annotations

import json

from . import IntegratedReadingProcessor, ReadingLexiconEntry, ReadingRequest


def main() -> None:
    lexicon = [
        ReadingLexiconEntry("w-001", "read", ("R", "IY", "D"), 40,
                           {"READING": 1.0, "KNOWLEDGE": 0.7}, "verb", verified_by="demo",
                           source_reference="demo://lexicon/read"),
        ReadingLexiconEntry("w-002", "reader", ("R", "IY", "D", "ER"), 18,
                           {"PERSON": 1.0, "READING": 0.8}, "noun", verified_by="demo",
                           source_reference="demo://lexicon/reader"),
        ReadingLexiconEntry("w-003", "red", ("R", "EH", "D"), 24,
                           {"COLOR": 1.0}, "adjective", verified_by="demo",
                           source_reference="demo://lexicon/red"),
        ReadingLexiconEntry("w-004", "brain", ("B", "R", "EY", "N"), 12,
                           {"BIOLOGY": 1.0, "ORGAN": 0.9}, "noun", verified_by="demo",
                           source_reference="demo://lexicon/brain"),
    ]
    processor = IntegratedReadingProcessor(lexicon)
    result = processor.process(ReadingRequest("demo-001", text="The reader can read."))
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
