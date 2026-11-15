"""Run a small reproducible ORE discourse-processing demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import DiscourseConfig, DiscourseRequest
from .pipeline import DiscourseProcessor


def main() -> None:
    """Execute a short discourse example and optionally save its JSON result."""
    parser = argparse.ArgumentParser(description="Run the ORE eight-stage discourse demonstration.")
    parser.add_argument("--output", type=Path, help="Optional path for a JSON result.")
    parser.add_argument("--no-biology", action="store_true", help="Disable the LIF reference track.")
    args = parser.parse_args()
    request = DiscourseRequest(
        sequence_id="discourse-demo-001",
        sentences=[
            "The software deployment failed yesterday.",
            "Great work on the release!",
            "Could you check the service logs?",
            "The engineer checked the logs, then restarted the service.",
        ],
        entities=[
            {"entity_id": "software", "label": "software", "entity_type": "object"},
            {"entity_id": "engineer", "label": "engineer", "entity_type": "person"},
            {"entity_id": "service", "label": "service", "entity_type": "object"},
        ],
        common_ground={"facts": ["The team is discussing a software release."]},
        speaker_model={"goals": ["restore service availability"]},
        prosody={1: {"intonation_mismatch": 0.7, "exaggeration": 0.5}},
        context={"prior_turns": ["The team reported a failed release."]},
    )
    result = DiscourseProcessor(
        DiscourseConfig(simulate_biology=not args.no_biology)
    ).process(request)
    payload = result.to_dict()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(json.dumps({
            "sequence_id": result.sequence_id,
            "coherence": result.coherence["score"],
            "intents": result.speaker_intentions,
            "indirect_requests": result.indirect_requests,
            "irony": result.irony,
            "operation_count": len(result.operation_trace),
            "biology": result.biological.get("model"),
            "diagnostics": result.diagnostics,
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
