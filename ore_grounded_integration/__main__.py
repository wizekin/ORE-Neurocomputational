"""Command-line demonstration and SQLite registration for grounded semantic integration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from . import (
    GroundedSemanticConfig,
    GroundedSemanticIntegrationProcessor,
    GroundedSemanticRequest,
    record_grounded_result,
    register_grounded_annotations,
)


def _example_request() -> GroundedSemanticRequest:
    """Build a reproducible multimodal bird example that exercises the 22-stage pipeline."""
    return GroundedSemanticRequest(
        sample_id="grounded-semantic-demo-bird",
        text="The bird builds a nest",
        modality_inputs={
            "visual": {"wings": 0.95, "feathers": 0.9, "beak": 0.85},
            "auditory": {"chirp": 0.8, "tweet": 0.7},
            "motor": {"fly": 0.8, "build": 0.6},
        },
        modality_timestamps={"linguistic": 0.0, "visual": 0.2, "auditory": 0.35, "motor": 0.5},
        context={"forest": 0.8, "tree": 0.7, "animal": 0.6},
        task="describe the concept",
        candidate_senses={
            "bird_animal": {"bird": 1.0, "feathers": 0.8, "wings": 0.6},
            "sports_equipment": {"bat": 1.0, "ball": 0.7},
        },
        usage_examples=("A bird builds a nest in a tree.", "The bird chirps at dawn."),
        learner_response={
            "recall": 0.8,
            "identification": 0.9,
            "action": 0.6,
            "contextual": 0.7,
            "generalization": 0.5,
        },
        source_reference="built_in_demo",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the demonstration, register source annotations and persist its stage-operation trace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=None, help="SQLite database path")
    parser.add_argument("--no-biology", action="store_true", help="Disable the optional LIF reference")
    parser.add_argument("--index-only", action="store_true", help="Only register class/function annotations")
    parser.add_argument("--json", action="store_true", help="Print the full structured pipeline result")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    registration = register_grounded_annotations(root, args.database)
    print(json.dumps({"annotation_registration": registration}, indent=2, default=str))
    if args.index_only:
        return 0

    processor = GroundedSemanticIntegrationProcessor(
        config=GroundedSemanticConfig(simulate_biology=not args.no_biology)
    )
    result = processor.process(_example_request())
    log_report = record_grounded_result(result, root, args.database)
    print(json.dumps({
        "output": result.output,
        "diagnostics": result.diagnostics,
        "sqlite_run": log_report,
    }, indent=2, default=str))
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0 if result.diagnostics.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
