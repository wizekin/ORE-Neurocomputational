"""Run a small working-memory and cognitive-control demonstration from the command line."""
from __future__ import annotations

import json

from .contracts import WorkingMemoryConfig, WorkingMemoryRequest
from .pipeline import WorkingMemoryProcessor


def main() -> None:
    """Execute a sample sentence through buffers, syntax maintenance, control, and task switching."""
    request = WorkingMemoryRequest(
        sequence_id="working-memory-demo",
        text="The researcher said that the model, which the team had tested, was reliable.",
        partial_structures=[
            {
                "structure_id": "relative-clause-1",
                "label": "relative clause",
                "depth": 2,
                "expected_slots": ["head_noun_binding"],
            }
        ],
        interpretations=[
            {
                "hypothesis_id": "H-literal",
                "interpretation": "the model passed the stated reliability criterion",
                "lexical_score": 0.8, "semantic_score": 0.82, "syntactic_score": 0.9,
                "discourse_score": 0.72, "world_score": 0.75, "prior": 0.8,
            },
            {
                "hypothesis_id": "H-alternative",
                "interpretation": "the speaker is expressing a tentative assessment",
                "lexical_score": 0.68, "semantic_score": 0.65, "syntactic_score": 0.8,
                "discourse_score": 0.7, "world_score": 0.62, "prior": 0.6,
            },
        ],
        task_sequence=["reading", "parsing", "semantic_resolution", "reading"],
        context={"task_cue_strength": 0.7},
    )
    result = WorkingMemoryProcessor(WorkingMemoryConfig(simulate_biology=True)).process(request)
    print(json.dumps({
        "schema": result.to_dict()["schema"],
        "tokens": result.tokens,
        "buffers": result.diagnostics["buffer_sizes"],
        "incomplete_structures": result.incomplete_structures,
        "semantic_resolution": result.semantic_resolution,
        "task_switching": result.task_switching,
        "operation_count": result.diagnostics["operation_count"],
        "biological_model": result.biological["model"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
