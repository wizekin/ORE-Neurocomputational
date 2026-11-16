"""Run a small deterministic Hub-and-Spoke integration demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ore_semantic.contracts import SemanticConcept

from .contracts import HubAndSpokeConfig, HubAndSpokeRequest
from .pipeline import HubAndSpokeProcessor


def main() -> None:
    """Run the sample concept through all 21 stages and optionally write its result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    parser.add_argument("--no-biology", action="store_true", help="disable the LIF reference track")
    args = parser.parse_args()
    concepts = [
        SemanticConcept(
            concept_id="animal-giraffe",
            label="giraffe",
            aliases=("giraffe",),
            semantic_features={"ANIMAL": 1.0, "MAMMAL": 0.9, "LONG_NECK": 0.9, "SPOTTED": 0.8},
            modality_features={
                "visual": {"LONG_NECK": 1.0, "SPOTTED": 0.9, "TALL": 1.0},
                "auditory": {"LOW_FREQUENCY_CALL": 0.6},
                "spatial_motor": {"BROWSE_HIGH_LEAVES": 0.9},
            },
            lexical_category="noun",
            frequency=10.0,
            prototype_id="mammal",
            verified_by="synthetic_demo",
            source_reference="demo://giraffe",
        ),
    ]
    processor = HubAndSpokeProcessor(
        concepts,
        config=HubAndSpokeConfig(simulate_biology=not args.no_biology),
    )
    result = processor.process(HubAndSpokeRequest(
        sequence_id="hub-spoke-demo",
        text="giraffe",
        modality_inputs={
            "visual": {"LONG_NECK": 0.95, "SPOTTED": 0.8},
        },
        task="describe",
    ))
    payload = result.to_dict()
    print(json.dumps({
        "output": payload["output"],
        "stage_count": payload["diagnostics"]["stage_count"],
        "operation_count": payload["diagnostics"]["operation_count"],
        "biological_units": payload["biological"].get("units", {}),
        "valid": payload["diagnostics"]["valid"],
    }, indent=2))
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
