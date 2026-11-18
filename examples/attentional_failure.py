"""Runnable Point 18 example: detect missed information and recover it from a source."""
from __future__ import annotations

from ore_attentional_failure import (
    AttentionalExperience, AttentionalFailureProcessor, InformationRequirement,
)


def main() -> None:
    """Run a high-quality source-backed recovery example with explicit provenance."""
    processor = AttentionalFailureProcessor()
    event = AttentionalExperience(
        experience_id="attention-example-001",
        task_goal="Understand the sequence and causal relation in the instruction",
        required_information=(
            InformationRequirement("agent", importance=0.8, critical=True),
            InformationRequirement("action", importance=1.0, critical=True),
            InformationRequirement("cause", importance=0.9, depends_on=("action",)),
        ),
        incoming_items=(
            {"key": "agent", "content": "The technician", "goal_relevance": 1.0,
             "signal_quality": 0.95, "integration_support": 0.95, "processing_cost": 0.2},
            {"key": "action", "content": "shut down the machine", "goal_relevance": 1.0,
             "signal_quality": 0.2, "integration_support": 0.2, "processing_cost": 0.3},
            {"key": "cause", "content": "because the sensor overheated", "goal_relevance": 0.9,
             "signal_quality": 0.8, "integration_support": 0.8, "processing_cost": 0.3},
        ),
        source_items=(
            {"key": "agent", "content": "The technician", "confidence": 0.98,
             "source_id": "transcript:line-1"},
            {"key": "action", "content": "shut down the machine", "confidence": 0.96,
             "source_id": "transcript:line-1"},
            {"key": "cause", "content": "because the sensor overheated", "confidence": 0.94,
             "source_id": "transcript:line-1"},
        ),
        metrics={"alertness": 0.8, "fatigue": 0.15, "distraction_load": 0.3,
                 "response_slowing": 0.1, "omission_rate": 0.2, "engagement": 0.8},
        available_recovery={"source": True},
        context_key="instructional-listening",
    )
    result = processor.process(event)
    print("Operations:", len(result.operation_trace))
    print("Missing before repair:", [x["key"] for x in result.missing_information["missing"]])
    print("Recovered:", [x["key"] for x in result.reencoded_representation["recovered_items"]])
    print("Coverage:", round(result.repair_verification["coverage"], 3))
    print("Release:", result.release_decision["status"])
    print("Biological scope:", result.biological["scientific_scope"])


if __name__ == "__main__":
    main()
