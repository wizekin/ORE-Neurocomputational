"""Run a small Point 02 prior-knowledge integration example."""
from __future__ import annotations

from ore_prior_knowledge import (
    KnowledgeNode,
    KnowledgeRelation,
    KnowledgeSchema,
    LearningUnit,
    PriorKnowledgeProcessor,
    PriorKnowledgeRequest,
    RelationInput,
)


def build_request() -> PriorKnowledgeRequest:
    """Create a transparent cell-membrane example with a familiar filter analogy."""
    prior_nodes = (
        KnowledgeNode(
            "cell", "cell", features={"living": 0.9, "cell": 1.0},
            context={"topic": "biology"}, reliability=0.98,
            provenance="previous_biology_lesson",
        ),
        KnowledgeNode(
            "filter", "water filter", features={"selective": 0.9, "boundary": 0.8,
                                                  "passage": 0.9},
            context={"topic": "biology"}, reliability=0.85,
            provenance="known_analogy",
        ),
        KnowledgeNode(
            "substances", "substances", features={"molecules": 0.8},
            context={"topic": "biology"}, reliability=0.9,
            provenance="previous_biology_lesson",
        ),
    )
    prior_relations = (
        KnowledgeRelation("filter", "regulates", "substances", 0.7, 0.8, "inferred"),
    )
    schema = KnowledgeSchema(
        "selective-boundary", "Selective boundary",
        roles=("cell membrane", "substances", "selective", "boundary"),
        relation_patterns=(("cell membrane", "regulates", "substances"),),
        expectations={"selective": 0.9, "boundary": 0.8},
        context={"topic": "biology"}, confidence=0.8,
    )
    new_unit = LearningUnit(
        "membrane-lesson-01",
        "A cell membrane selectively regulates the movement of substances across a boundary.",
        entities=("cell membrane", "substances"),
        features={"cell": 0.9, "membrane": 1.0, "selective": 1.0,
                  "boundary": 0.95, "transport": 0.8, "substances": 0.7},
        relations=(RelationInput("cell membrane", "regulates", "substances", 0.95),),
        context={"topic": "biology", "lesson": "cell structure"},
        reliability=0.95,
    )
    return PriorKnowledgeRequest(
        "demo-learner", new_unit,
        current_context={"topic": "biology", "goal": "understand cell membranes"},
        knowledge_nodes=prior_nodes, knowledge_relations=prior_relations,
        schemas=(schema,), learner_state={"attention": 0.9, "working_memory": 0.85},
    )


def main() -> None:
    """Run Point 02 and print strategy, diagnostics, and numbered operation IDs."""
    request = build_request()
    result = PriorKnowledgeProcessor(request.learner_id).process(request)
    print(result.output)
    for operation in result.operation_trace:
        print(operation["operation_id"], operation["operation"])


if __name__ == "__main__":
    main()
