"""Runnable example for Point 13 emotion, motivation, attention, and memory encoding."""

from __future__ import annotations

from ore_emotion_motivation import (
    EmotionMotivationMemoryProcessor,
    EventFeature,
    LearningExperience,
    LearningState,
    RewardFeedback,
)


def main() -> None:
    """Process one learning event with a salient central cue and explicit learning goal."""
    state = LearningState()
    processor = EmotionMotivationMemoryProcessor(state=state)
    experience = LearningExperience(
        event_id="learning-event-001",
        goal="understand_the_safety_procedure",
        task_demands=0.55,
        novelty=0.35,
        uncertainty=0.20,
        threat=0.25,
        reward_relevance=0.80,
        valence_hint=0.20,
        arousal_hint=0.40,
        sleep_replay_support=0.5,
        features=(
            EventFeature(
                "main_instruction", value=1.0, salience=0.95,
                goal_relevance=1.0, centrality=1.0,
                valence=0.1, arousal=0.4,
                tags=("safety", "instruction"), role="central",
            ),
            EventFeature(
                "warning_signal", value=0.9, salience=0.85,
                goal_relevance=0.75, centrality=0.8,
                valence=-0.4, arousal=0.8,
                tags=("safety", "warning"), role="central",
            ),
            EventFeature(
                "room_background", value=0.7, salience=0.2,
                goal_relevance=0.15, centrality=0.1,
                tags=("environment",), role="peripheral",
            ),
            EventFeature(
                "time_and_location", value=0.8, salience=0.35,
                goal_relevance=0.65, centrality=0.3,
                tags=("context", "environment"), role="context",
            ),
        ),
        context={"feature_relevance": {"time_and_location": 0.8}},
        retrieval_cues=("safety", "instruction"),
        metadata={"elapsed_hours": 2.0},
    )
    result = processor.process(
        experience,
        reward_feedback=RewardFeedback(
            state_id="safety_learning", action_id="study_procedure",
            reward=0.8, next_state_id="procedure_practiced",
        ),
    )
    print(result.output)
    print("Trace stages:", len(result.operation_trace))
    print("Biological models:", sorted(result.biological))


if __name__ == "__main__":
    main()
