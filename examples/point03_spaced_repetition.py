"""Run a small Point 03 spaced-repetition cycle with an explicit simulated review."""
from __future__ import annotations

from ore_spaced_repetition import (
    LearningItem,
    ReviewObservation,
    SpacedRepetitionProcessor,
    SpacedRepetitionRequest,
)


def build_item() -> LearningItem:
    """Describe a science definition with separate components and relational structure."""
    return LearningItem(
        item_id="synapse-definition",
        content="A synapse is a specialized junction through which one neuron communicates with another cell.",
        components=("specialized junction", "communication", "neuron", "another cell"),
        relations=("neuron communicates across junction with another cell",),
        context={"topic": "neuroscience", "lesson": "cell communication"},
        difficulty=0.45,
        target_retention_hours=24.0 * 30.0,
        encoding_measurements={
            "attention": 0.9, "comprehension": 0.85,
            "relational_organization": 0.8, "meaningful_practice": 0.75,
        },
        source="biology-lesson",
        created_at_utc="2026-01-01T00:00:00Z",
    )


def main() -> None:
    """Register the item, inspect its due date, then record one delayed retrieval."""
    processor = SpacedRepetitionProcessor("demo-learner")
    item = build_item()
    first = processor.process(SpacedRepetitionRequest(
        "demo-learner", item, "2026-01-01T00:00:00Z"
    ))
    due = first.memory_record["next_due_at_utc"]
    print("First review due:", due)
    review = ReviewObservation(
        occurred_at_utc=due,
        outcome="partial",
        accuracy=0.8,
        completeness=0.65,
        relation_accuracy=0.7,
        fluency=0.6,
        effort=0.75,
        response_mode="free_recall",
        learner_response="A synapse is a junction where one neuron communicates.",
        feedback_provided=True,
        corrected_response="A synapse is a specialized junction through which a neuron communicates with another cell.",
    )
    second = processor.process(SpacedRepetitionRequest(
        "demo-learner", item, due, review_observation=review
    ))
    print("Recorded review:", second.review_event["event_type"])
    print("Next interval (hours):", round(second.output["next_interval_hours"], 2))
    print("Operations completed:", len(second.operation_trace))


if __name__ == "__main__":
    main()
