"""Run a small multimodal amodal ATL demonstration from the command line."""

from __future__ import annotations

import argparse
import json

from .contracts import AmodalATLConfig, AmodalATLRequest, AmodalConcept
from .pipeline import AmodalATLProcessor


def main() -> int:
    """Run the documented dog-concept example and print a JSON processing trace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-biology", action="store_true", help="Skip the LIF reference simulation.")
    parser.add_argument("--phrase", default="", help="Optional adjective-noun phrase, such as 'fake diamond'.")
    args = parser.parse_args()

    concept = AmodalConcept(
        concept_id="dog",
        label="dog",
        aliases=("canine",),
        semantic_features={
            "ANIMAL": 1.0,
            "MAMMAL": 0.95,
            "FOUR_LEGS": 0.85,
            "BARKS": 0.8,
            "DOMESTIC": 0.75,
        },
        modality_features={
            "visual": {"FOUR_LEGS": 0.9, "FUR": 0.8, "TAIL": 0.7},
            "auditory": {"BARKS": 1.0, "WHINES": 0.6},
            "spatial_motor": {"RUNS": 0.8, "WAGS_TAIL": 0.7},
            "tactile": {"SOFT_FUR": 0.8},
        },
        category="animal",
        source_reference="demo://dog",
    )
    config = AmodalATLConfig(simulate_biology=not args.no_biology)
    processor = AmodalATLProcessor([concept], config=config)

    if args.phrase.strip():
        pieces = args.phrase.strip().split(maxsplit=1)
        adjective = pieces[0] if len(pieces) == 2 else None
        noun = pieces[-1]
        request = AmodalATLRequest(
            sequence_id="cli-demo-phrase",
            text=args.phrase,
            noun=noun,
            adjective=adjective,
            adjective_type="privative" if adjective and adjective.casefold() in {"fake", "counterfeit", "imitation"} else "subsective",
            modality_inputs={"linguistic": {"PHRASE_INPUT": 1.0}},
        )
    else:
        request = AmodalATLRequest(
            sequence_id="cli-demo-dog",
            text="dog",
            modality_inputs={
                "visual": {"FOUR_LEGS": 0.9, "FUR": 0.8},
                "auditory": {"BARKS": 0.95},
            },
        )
    result = processor.process(request)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
