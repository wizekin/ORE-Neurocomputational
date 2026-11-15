"""Runnable example for the nine-stage semantic-composition package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ore_semantic import SemanticConcept

from .contracts import (
    PredicateFrame,
    SemanticCompositionConfig,
    SemanticCompositionRequest,
)
from .registry import STAGE_REGISTRY
from .pipeline import SemanticCompositionProcessor


def build_demo_processor(simulate_biology: bool = True) -> SemanticCompositionProcessor:
    """Create a small, provenance-labelled inventory for the package demonstration."""
    concepts = [
        SemanticConcept(
            concept_id="concept-scientist",
            label="scientist",
            aliases=("researcher",),
            semantic_features={"HUMAN": 1.0, "PROFESSION": 0.9, "AGENTIVE": 0.8},
            lexical_category="noun",
            frequency=18.0,
            verified_by="demo",
            source_reference="demo://concepts/scientist",
        ),
        SemanticConcept(
            concept_id="concept-study",
            label="studies",
            aliases=("study",),
            semantic_features={"COGNITION": 0.9, "INVESTIGATION": 1.0, "EVENT": 0.8},
            lexical_category="verb",
            frequency=12.0,
            verified_by="demo",
            source_reference="demo://concepts/study",
        ),
        SemanticConcept(
            concept_id="concept-neuron",
            label="neurons",
            aliases=("neuron",),
            semantic_features={"BIOLOGICAL_ENTITY": 1.0, "CELL": 0.9, "NERVOUS_SYSTEM": 0.9},
            lexical_category="noun",
            frequency=16.0,
            verified_by="demo",
            source_reference="demo://concepts/neuron",
        ),
    ]
    frames = [
        PredicateFrame(
            predicate="studies",
            predicate_concept_id="concept-study",
            roles=("Agent", "Patient"),
            required_roles=("Agent", "Patient"),
            selectional_preferences={
                "Agent": {"HUMAN": 1.0, "AGENTIVE": 0.8},
                "Patient": {"BIOLOGICAL_ENTITY": 0.8, "OBJECT_OF_STUDY": 1.0},
            },
            source_reference="demo://frames/study",
        ),
    ]
    return SemanticCompositionProcessor(
        concepts,
        frames,
        SemanticCompositionConfig(simulate_biology=simulate_biology),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the example sentence through the pipeline and optionally save its JSON result."""
    parser = argparse.ArgumentParser(description="Run ORE semantic composition demo.")
    parser.add_argument("--output", type=Path, help="Optional path for the JSON result.")
    parser.add_argument("--no-biology", action="store_true", help="Disable the LIF reference track.")
    args = parser.parse_args(argv)
    processor = build_demo_processor(simulate_biology=not args.no_biology)
    request = SemanticCompositionRequest(
        sequence_id="semantic-composition-demo",
        tokens=("scientist", "studies", "neurons"),
    )
    result = processor.process(request)
    payload = result.to_dict()
    encoded = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps({
        "sequence_id": result.sequence_id,
        "operation_count": len(result.operation_trace),
        "stages": len(STAGE_REGISTRY),
        "propositions": len(result.propositions),
        "valid": result.diagnostics["valid"],
        "output": str(args.output) if args.output else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
