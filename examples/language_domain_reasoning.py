"""Runnable Point 15 example showing separate language, knowledge, and reasoning outputs."""
from __future__ import annotations

from ore_language_domain_reasoning import (
    LanguageDomainReasoningProcessor, LanguageEvent, LanguageLearnerState,
)


def main() -> None:
    """Run a language task with conceptual retrieval and a separately recorded inference."""
    state = LanguageLearnerState(learner_id="example-learner")
    processor = LanguageDomainReasoningProcessor(state=state)
    event = LanguageEvent(
        event_id="point15-example-001",
        sensory_stream="Every metal rod in the sample expanded after heating.",
        modality="text",
        conceptual_memory={
            "thermal_expansion": {
                "description": "many metals expand as temperature rises",
                "reliability": 0.95,
            }
        },
        task_goal="reason",
        reasoning_rules=(
            {"rule_id": "thermal-generalization",
             "if_all": ("metal rod", "expanded after heating"),
             "then": "the sample contains an observed expansion event"},
        ),
        cognitive_state={"control_capacity": 0.55, "fatigue": 0.1, "requires_reasoning": True},
    )
    result = processor.process(event)
    print("Task output:", result.task_result)
    print("Explicit sentence content:", result.linguistic_representation["said_or_encoded"])
    print("Retrieved concepts:", result.conceptual_interpretation["activated_concepts"])
    print("Separate reasoning conclusions:", result.reasoning_result["inferred"])
    print("Controlled attention recruited:", result.control_request["control_recruited"])
    print("Ordered stages:", len(result.operation_trace))
    print("Biological functional systems:", sorted(result.biological))


if __name__ == "__main__":
    main()
