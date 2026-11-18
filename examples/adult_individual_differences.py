"""Runnable Point 17 example: adaptive adult comprehension and memory support."""
from __future__ import annotations

from ore_adult_individual_differences import (
    AdultComprehensionExperience,
    AdultIndividualDifferencesProcessor,
    AdultLearnerState,
)


def main() -> None:
    """Process a domain-specific explanation using the learner's relevant prior schema."""
    learner = AdultLearnerState(
        learner_id="adult-learning-example",
        prior_knowledge={
            "physics": [{
                "schema_id": "thermal-expansion",
                "schema": "thermal expansion relation",
                "features": ("metal", "heated", "temperature", "expansion"),
                "reliability": 0.95,
                "goal_fit": 0.95,
                "recency": 0.9,
                "chunks": ({
                    "chunk_id": "thermal-relation",
                    "features": ("thermal", "expansion"),
                    "schema_cost": 0.5,
                    "exception_cost": 0.1,
                },),
            }]
        },
        expertise_by_domain={"physics": 0.8, "literature": 0.1},
        language_proficiency={"english": 0.95},
        language_cue_reliability={"english": {"word_order": 0.9, "connectives": 0.95}},
    )
    processor = AdultIndividualDifferencesProcessor(state=learner)
    event = AdultComprehensionExperience(
        experience_id="physics-example-001",
        content_units=("Metal", "expands", "when", "heated"),
        domain="physics",
        task_goal={"detail": 0.3, "gist": 0.2, "relation": 0.4, "source": 0.1},
        desired_output="explain_causal_relation",
        language="english",
        language_cues={"word_order": 0.9, "connectives": 1.0},
        interpretation_candidates=(
            {"hypothesis_id": "expansion", "label": "temperature increase produces expansion",
             "prior": 0.6, "cue_likelihoods": {"word_order": 0.9, "connectives": 0.95}},
            {"hypothesis_id": "contraction", "label": "temperature increase produces contraction",
             "prior": 0.4, "cue_likelihoods": {"word_order": 0.2, "connectives": 0.2}},
        ),
        dependency_pairs=((0, 1), (1, 3)),
        external_support=0.2,
        memory_request="balanced",
        memory_evidence={
            "gist_cue_quality": 0.9,
            "detail_cue_quality": 0.8,
            "source_cue_quality": 0.7,
            "encoding_time_ms": 900,
            "distinctiveness": 0.8,
        },
        evidence_quality=0.9,
        provenance_completeness=0.95,
    )
    result = processor.process(event)
    print("Operations:", len(result.operation_trace))
    print("Schema:", result.retrieved_knowledge["selected_schema"]["schema_id"])
    print("Load pressure:", round(result.working_memory_load["load_pressure"], 3))
    print("Support strategy:", result.adaptive_support["strategy"])
    print("Next action:", result.calibrated_decision["next_action"])
    print("Gist/detail/source:", (
        round(result.aging_memory["gist_probability"], 3),
        round(result.aging_memory["episodic_detail_probability"], 3),
        round(result.aging_memory["source_context_probability"], 3),
    ))
    print("Learner experiences:", learner.experience_count)


if __name__ == "__main__":
    main()
