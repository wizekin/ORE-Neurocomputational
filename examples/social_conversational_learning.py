"""Runnable Point 14 example: joint attention, social feedback, and word learning."""

from __future__ import annotations

from ore_social_conversational_learning import (
    LearnerSocialState,
    ReferentCandidate,
    SocialFeedback,
    SocialInteractionEvent,
    SocialInteractionLearningProcessor,
    SocialSignal,
)


def main() -> None:
    """Run an example interaction in which a partner directs attention to a referent."""
    state = LearnerSocialState(learner_id="learner-1")
    processor = SocialInteractionLearningProcessor("learner-1", state=state)
    event = SocialInteractionEvent(
        event_id="interaction-001",
        learner_id="learner-1",
        partner_id="teacher-1",
        utterance="Look at the prism. What shape is it?",
        signals=(
            SocialSignal(
                "speech-1", "speech", "Look at the prism. What shape is it?",
                reliability=0.95, salience=0.8, directed_to_learner=0.95,
                timestamp=1.0,
            ),
            SocialSignal(
                "gaze-1", "gaze", "teacher looks toward prism",
                reliability=0.9, salience=0.8, directed_to_learner=0.9,
                target_id="prism", timestamp=1.1,
            ),
            SocialSignal(
                "point-1", "gesture", "teacher points to prism",
                reliability=0.95, salience=0.95, directed_to_learner=0.95,
                target_id="prism", timestamp=1.2,
            ),
        ),
        referents=(
            ReferentCandidate(
                "prism", "a transparent geometric prism",
                features=("solid", "geometry", "prism"),
                properties={"shape": "three-dimensional"},
                salience=0.9, context_fit=0.95,
            ),
            ReferentCandidate(
                "book", "a textbook",
                features=("pages", "reading"),
                salience=0.4, context_fit=0.35,
            ),
        ),
        learner_focus="prism",
        learning_goal="learn_geometry_vocabulary",
        target_word="prism",
        candidate_mappings={"prism": 0.95, "book": 0.05},
        partner_response="Yes, that is the prism.",
        predicted_partner_response="Yes, that is the prism.",
        feedback=(
            SocialFeedback(
                "confirmation-1", "confirmation",
                content="Yes, that is the prism.",
                target_key="prism", factual_value="prism",
                reliability=0.95, confirms=True, source_id="teacher-1",
            ),
            SocialFeedback(
                "praise-1", "social_affective", content="Good observation.",
                reward=0.7, reliability=0.8, source_id="teacher-1",
            ),
        ),
        target_concept="prism",
        required_explanation_terms=("solid", "geometry", "three-dimensional"),
        required_relations=("prism has shape",),
        learner_attempt="A prism is a three-dimensional solid in geometry.",
        competence_observation=0.55,
        difficulty=0.6,
        metadata={"timestamp": "lesson-001"},
    )
    result = processor.process(event)
    print(result.output)
    print("Operation stages:", len(result.operation_trace))
    print("Joint attention:", result.joint_attention)
    print("Biological subsystems:", sorted(result.biological))


if __name__ == "__main__":
    main()
