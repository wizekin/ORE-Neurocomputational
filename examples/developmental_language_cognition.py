"""Runnable Point 16 example: language, concepts, memory, and executive development."""
from __future__ import annotations

from ore_developmental_language_cognition import (
    DevelopmentalLanguageCognitionProcessor, DevelopmentalExperience,
    DevelopmentalLearnerState,
)


def main() -> None:
    """Run two learning episodes to demonstrate gradual vocabulary and memory development."""
    state = DevelopmentalLearnerState(learner_id="development-example")
    processor = DevelopmentalLanguageCognitionProcessor(state=state)
    experiences = (
        DevelopmentalExperience(
            experience_id="word-exposure-1", age_months=24,
            acoustic_frames=((0.1, 0.8, 0.2), (0.2, 0.75, 0.25)),
            syllables=("ba", "na"),
            boundary_cues=({"pause": 0.8, "stress": 0.7},),
            phonetic_label="ba-like", native_language="environment-A",
            target_word="banana",
            scene_referents=({"id": "fruit-1", "label": "banana", "salience": 0.9,
                              "features": (1.0, 0.2, 0.1)},),
            gaze_target="fruit-1", gesture_target="fruit-1", social_contingency=0.9,
            context_tokens=("fruit", "yellow", "eat"),
            concept_examples=({"label": "banana", "parent": "fruit", "features": (1.0, 0.2, 0.1),
                               "attributes": {"yellow": True, "edible": True}},),
            current_goal="identify the fruit", working_memory_items=("banana", "fruit"),
            action_candidates=({"action_id": "choose-fruit", "goal_match": 1.0,
                                "relevance": 0.9, "distraction": 0.0,
                                "prepotency": 0.1, "conflict": 0.0},),
            performance_feedback={"domains": {"vocabulary": {"success": 0.65}}},
        ),
        DevelopmentalExperience(
            experience_id="word-exposure-2", age_months=28,
            acoustic_frames=((0.12, 0.78, 0.22), (0.18, 0.77, 0.26)),
            syllables=("ba", "na"), target_word="banana",
            scene_referents=({"id": "fruit-1", "label": "banana", "salience": 0.9,
                              "features": (1.0, 0.2, 0.1)},),
            gaze_target="fruit-1", gesture_target="fruit-1", social_contingency=0.85,
            context_tokens=("fruit", "snack"),
            concept_examples=({"label": "banana", "parent": "fruit", "features": (0.95, 0.2, 0.1),
                               "attributes": {"yellow": True, "edible": True}},),
            current_goal="remember the fruit label",
            memory_cues=("banana", "fruit"), working_memory_items=("banana", "fruit", "snack"),
            performance_feedback={"domains": {"vocabulary": {"success": 0.82}}},
        ),
    )
    for event in experiences:
        result = processor.process(event)
        print(event.experience_id, "stages:", len(result.operation_trace))
        print("vocabulary:", result.vocabulary_growth)
        print("concepts:", result.concept_formation["updated_concepts"])
        print("working-memory load:", result.working_memory["load"])
    print("learner experiences:", state.experience_count)
    print("retained episodes:", len(state.episodic_memory))


if __name__ == "__main__":
    main()
