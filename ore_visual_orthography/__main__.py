"""Run a small deterministic demonstration of the visual-orthography stages A-E."""
from __future__ import annotations

import json

from .contracts import LexicalCandidate, VisualOrthographyConfig, VisualOrthographyRequest
from .pipeline import VisualOrthographyProcessor


def main() -> None:
    """Execute an explicit-sequence demonstration and print its JSON-safe result."""
    processor = VisualOrthographyProcessor(
        VisualOrthographyConfig(simulate_biology=True),
    )
    request = VisualOrthographyRequest(
        sample_id="visual-orthography-demo",
        orthographic_sequence=("j", "u", "d", "g", "e"),
        comparison_strings=("jugde", "jupte"),
        lexicon=(
            LexicalCandidate("judge", frequency=30.0),
            LexicalCandidate("jugde", frequency=1.0),
            LexicalCandidate("jungle", frequency=20.0),
        ),
        language="en",
    )
    result = processor.process(request)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
