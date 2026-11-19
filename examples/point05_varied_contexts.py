"""Run a varied-context and interleaved-learning demonstration on related math problems."""
from __future__ import annotations

from ore_varied_contexts import (
    ConceptDefinition,
    LearningExample,
    TrialObservation,
    VariedContextsProcessor,
    VariedContextsRequest,
    record_result_to_sqlite,
)


def build_concepts() -> tuple[ConceptDefinition, ...]:
    """Define direct, inverse, and additive relationships with explicit diagnostic features."""
    return (
        ConceptDefinition(
            "direct", "mathematics",
            {"ratio_constant": 1.0, "product_constant": 0.0, "difference_constant": 0.0},
            discriminative_features={"ratio_constant": 1.0},
            prerequisite_concepts=("arithmetic",),
            relations=("the ratio remains constant",),
            importance=1.0,
            initial_familiarity=0.7,
        ),
        ConceptDefinition(
            "inverse", "mathematics",
            {"ratio_constant": 0.0, "product_constant": 1.0, "difference_constant": 0.0},
            discriminative_features={"product_constant": 1.0},
            prerequisite_concepts=("arithmetic",),
            relations=("the product remains constant",),
            importance=1.0,
            initial_familiarity=0.7,
        ),
        ConceptDefinition(
            "additive", "mathematics",
            {"ratio_constant": 0.0, "product_constant": 0.0, "difference_constant": 1.0},
            discriminative_features={"difference_constant": 1.0},
            prerequisite_concepts=("arithmetic",),
            relations=("the difference remains constant",),
            importance=1.0,
            initial_familiarity=0.7,
        ),
    )


def build_examples() -> tuple[LearningExample, ...]:
    """Create training, near-transfer, far-transfer, and delayed examples with varied contexts."""
    def example(
        example_id: str,
        concept: str,
        feature: str,
        setting: str,
        format_name: str,
        modality: str,
        level: str = "training",
        anchor: bool = False,
    ) -> LearningExample:
        """Create a single labeled mathematics example while preserving its context separately."""
        features = {
            "ratio_constant": float(feature == "ratio_constant"),
            "product_constant": float(feature == "product_constant"),
            "difference_constant": float(feature == "difference_constant"),
        }
        relation = {
            "ratio_constant": "the ratio remains constant",
            "product_constant": "the product remains constant",
            "difference_constant": "the difference remains constant",
        }[feature]
        return LearningExample(
            example_id=example_id,
            concept_id=concept,
            concept_features=features,
            context_features={"setting": setting, "format": format_name},
            modality=modality,
            representation=format_name,
            relations=(relation,),
            is_anchor=anchor,
            is_novel=level != "training",
            transfer_level=level,
            session_id="point05-demo",
        )

    return (
        example("direct-recipe", "direct", "ratio_constant", "kitchen", "story", "text", anchor=True),
        example("direct-market", "direct", "ratio_constant", "market", "table", "table"),
        example("inverse-trip", "inverse", "product_constant", "road", "table", "table", anchor=True),
        example("inverse-workers", "inverse", "product_constant", "factory", "diagram", "diagram"),
        example("additive-fee", "additive", "difference_constant", "shop", "graph", "graph", anchor=True),
        example("additive-salary", "additive", "difference_constant", "workplace", "story", "text"),
        example("direct-garden-near", "direct", "ratio_constant", "garden", "diagram", "diagram", "near"),
        example("inverse-pipes-near", "inverse", "product_constant", "plumbing", "story", "text", "near"),
        example("additive-membership-near", "additive", "difference_constant", "club", "table", "table", "near"),
        example("direct-map-far", "direct", "ratio_constant", "navigation", "map", "diagram", "far"),
        example("inverse-workflow-far", "inverse", "product_constant", "production", "flowchart", "diagram", "far"),
        example("additive-offset-far", "additive", "difference_constant", "measurement", "equation", "text", "far"),
        example("direct-sports-delayed", "direct", "ratio_constant", "sports", "table", "table", "delayed"),
        example("inverse-distance-delayed", "inverse", "product_constant", "travel", "graph", "graph", "delayed"),
        example("additive-budget-delayed", "additive", "difference_constant", "budget", "story", "text", "delayed"),
    )


def build_observations(examples: tuple[LearningExample, ...]) -> tuple[TrialObservation, ...]:
    """Supply explicit correct selections and relation mappings for this reproducible demonstration."""
    relation_by_concept = {
        "direct": "the ratio remains constant",
        "inverse": "the product remains constant",
        "additive": "the difference remains constant",
    }
    return tuple(
        TrialObservation(
            example_id=item.example_id,
            selected_concept_id=item.concept_id,
            confidence=0.86,
            effort=0.42,
            context_judgment=dict(item.context_features),
            cue_type=item.modality,
            explanation="The selected relation matches the diagnostic feature.",
            preserved_relations=(relation_by_concept[item.concept_id],),
            feedback_verified=True,
            response_time_ms=3200.0,
        )
        for item in examples
    )


def main() -> None:
    """Execute all Point 05 stages, print evaluation evidence, and write SQLite annotations/events."""
    concepts = build_concepts()
    examples = build_examples()
    observations = build_observations(examples)
    request = VariedContextsRequest(
        concepts=concepts,
        examples=examples,
        observations=observations,
        prerequisite_readiness={"arithmetic": 0.92},
        current_context={"setting": "kitchen", "format": "story"},
        fatigue=0.15,
        available_time_minutes=45.0,
        original_context_id='{"format":"story","setting":"kitchen"}',
        current_time_utc="2026-01-01T00:00:00Z",
    )
    result = VariedContextsProcessor().process(request)
    print("Point 05 output:", result.output)
    print("Operation count:", len(result.operation_trace))
    report = record_result_to_sqlite(
        result, "ore_logs/point05.sqlite3"
    )
    print("SQLite report:", report)


if __name__ == "__main__":
    main()
