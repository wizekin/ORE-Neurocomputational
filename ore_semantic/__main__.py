"""Run a small synthetic demonstration of ORE semantic cognition."""
from __future__ import annotations

import argparse
import json

from . import (
    SemanticConcept,
    SemanticCognitionProcessor,
    SemanticRequest,
    WordFormEpisode,
)


def build_demo() -> dict[str, object]:
    concepts = (
        SemanticConcept(
            concept_id="concept-apple",
            label="apple",
            aliases=("fruit",),
            semantic_features={"FOOD": 1.0, "PLANT": 0.8, "ROUND": 0.7, "SWEET": 0.7},
            modality_features={
                "visual": {"ROUND": 0.9, "RED_OR_GREEN": 0.9},
                "gustatory": {"SWEET": 0.8, "TART": 0.4},
            },
            lexical_category="noun",
            frequency=30.0,
            prototype_id="food",
            verified_by="synthetic-demo",
            source_reference="synthetic://semantic-demo/apple",
        ),
        SemanticConcept(
            concept_id="concept-orange",
            label="orange",
            aliases=("citrus",),
            semantic_features={"FOOD": 1.0, "PLANT": 0.8, "ROUND": 0.7, "CITRUS": 1.0},
            modality_features={
                "visual": {"ROUND": 0.9, "ORANGE_COLOR": 1.0},
                "gustatory": {"SWEET": 0.6, "TART": 0.8},
            },
            lexical_category="noun",
            frequency=24.0,
            prototype_id="food",
            verified_by="synthetic-demo",
            source_reference="synthetic://semantic-demo/orange",
        ),
        SemanticConcept(
            concept_id="concept-run",
            label="run",
            aliases=("running",),
            semantic_features={"ACTION": 1.0, "LOCOMOTION": 0.9, "LEGS": 0.7},
            modality_features={"action": {"LOCOMOTION": 1.0, "LEGS": 0.8}},
            lexical_category="verb",
            frequency=18.0,
            prototype_id="action",
            verified_by="synthetic-demo",
            source_reference="synthetic://semantic-demo/run",
        ),
    )
    episodes = (
        WordFormEpisode(
            episode_id="episode-apple-001",
            orthography="apple",
            concept_id="concept-apple",
            phonological_form=("AE", "P", "AH", "L"),
            semantic_features={"FOOD": 1.0, "PLANT": 0.8},
            context_features={"FOOD": 1.0},
            strength=0.95,
            recency=0.90,
            source_reference="synthetic://episode/apple",
        ),
    )
    processor = SemanticCognitionProcessor(concepts)
    result = processor.process(SemanticRequest(
        sequence_id="semantic-demo-001",
        text="apple run",
        language="en",
        task="comprehension",
        context={"FOOD": 0.8, "ACTION": 0.2},
        modality_evidence={
            "visual": {"ROUND": 0.8, "RED_OR_GREEN": 0.7},
            "gustatory": {"SWEET": 0.7},
            "action": {"LOCOMOTION": 0.7},
        },
        episodes=episodes,
        source_reference="synthetic://ore-semantic-demo",
    ))
    return result.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="optional JSON output path")
    args = parser.parse_args()
    payload = build_demo()
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
