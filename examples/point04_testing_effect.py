"""Run a reproducible Point 04 retrieval, feedback, retest, and delayed-practice cycle."""
from __future__ import annotations

from pathlib import Path

from ore_testing_effect import (
    Feedback,
    LearningTarget,
    RetrievalAttempt,
    TestingEffectProcessor,
    TestingEffectRequest,
    record_result_to_sqlite,
)


def build_target() -> LearningTarget:
    """Create an explicit rubric for explaining the testing effect."""
    return LearningTarget(
        target_id="testing-effect-definition",
        prompt="Explain why retrieval practice can improve later retention.",
        components=(
            "active generation from memory",
            "improved later accessibility",
            "retrieval reveals knowledge gaps",
            "benefits depend on learning conditions",
        ),
        relations=(
            "retrieval practice requires generating information without direct exposure",
            "retrieval attempts can improve later access under suitable conditions",
        ),
        component_weights={
            "active generation from memory": 1.0,
            "improved later accessibility": 1.0,
            "retrieval reveals knowledge gaps": 0.8,
            "benefits depend on learning conditions": 1.0,
        },
        relation_weights={
            "retrieval practice requires generating information without direct exposure": 1.0,
            "retrieval attempts can improve later access under suitable conditions": 1.0,
        },
        criterion_source="validated_lesson_answer",
        criterion_reliability=0.95,
        initial_encoding_quality=0.75,
    )


def main() -> None:
    """Run one target through initial recall, targeted correction, retesting, and SQLite logging."""
    target = build_target()
    initial = RetrievalAttempt(
        cue="Explain why actively trying to recall a learned idea may improve later memory.",
        response="It makes the learner generate the information and can improve later access.",
        confidence=0.72,
        cue_type="causal",
        recalled_components=(
            "active generation from memory",
            "improved later accessibility",
        ),
        recalled_relations=(
            "retrieval practice requires generating information without direct exposure",
        ),
        component_status={
            "active generation from memory": "correct",
            "improved later accessibility": "correct",
            "retrieval reveals knowledge gaps": "missing",
            "benefits depend on learning conditions": "missing",
        },
        relation_status={
            "retrieval practice requires generating information without direct exposure": "correct",
            "retrieval attempts can improve later access under suitable conditions": "missing",
        },
    )
    feedback = Feedback(
        correct_representation=(
            "Retrieval practice makes learners generate information from memory; "
            "it can improve later access and expose gaps, but benefits depend on conditions."
        ),
        explanation=(
            "A failed attempt is not guaranteed to strengthen memory. Correct errors, "
            "then try to retrieve the corrected explanation without viewing it."
        ),
        source="validated_lesson_answer",
        source_reliability=0.95,
        verified=True,
    )
    retest = RetrievalAttempt(
        cue="Explain again, without the notes, why self-testing may improve delayed retention.",
        response=(
            "Self-testing requires generating information from memory, reveals missing "
            "knowledge, and can improve later access. Its benefit depends on the conditions."
        ),
        confidence=0.88,
        cue_type="causal",
        recalled_components=target.components,
        recalled_relations=target.relations,
        component_status={item: "correct" for item in target.components},
        relation_status={item: "correct" for item in target.relations},
    )
    transfer = RetrievalAttempt(
        cue="Apply the distinction to explain why rereading can create familiarity without strong later recall.",
        response=(
            "Rereading exposes the learner to the answer and can increase familiarity, "
            "whereas retrieval requires generating it without seeing it."
        ),
        confidence=0.82,
        cue_type="application",
        recalled_components=(
            "active generation from memory",
            "improved later accessibility",
        ),
        component_status={
            "active generation from memory": "correct",
            "improved later accessibility": "correct",
            "retrieval reveals knowledge gaps": "missing",
            "benefits depend on learning conditions": "missing",
        },
        relation_status={
            "retrieval practice requires generating information without direct exposure": "correct",
            "retrieval attempts can improve later access under suitable conditions": "uncertain",
        },
    )
    delayed = RetrievalAttempt(
        cue="Several days later, explain the testing effect without looking at the lesson.",
        response=(
            "Retrieval requires reconstructing information from memory and can support "
            "later retention while exposing gaps; it is not guaranteed to help after every failure."
        ),
        confidence=0.80,
        cue_type="relational",
        recalled_components=target.components,
        recalled_relations=target.relations,
        component_status={item: "correct" for item in target.components},
        relation_status={item: "correct" for item in target.relations},
    )
    processor = TestingEffectProcessor()
    result = processor.process(TestingEffectRequest(
        target=target,
        initial_attempt=initial,
        feedback=feedback,
        retest_attempt=retest,
        alternative_attempts=(transfer,),
        delayed_attempts=(delayed,),
        current_time_utc="2026-01-01T00:00:00Z",
        delay_after_hours=72.0,
    ))

    print("Point 04 output:", result.output)
    print("Operation count:", len(result.operation_trace))
    database_path = Path("ore_logs") / "point04.sqlite3"
    database_path.parent.mkdir(parents=True, exist_ok=True)
    report = record_result_to_sqlite(result, database_path)
    print("SQLite report:", report)


if __name__ == "__main__":
    main()
