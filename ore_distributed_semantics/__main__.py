"""Run a deterministic 36-stage semantic demonstration and register ORE annotations."""
from __future__ import annotations

from pathlib import Path

from ore_semantic.contracts import SemanticConcept

from .annotations import index_distributed_semantic_annotations, record_distributed_semantic_result
from .contracts import DistributedSemanticConfig, DistributedSemanticRequest
from .pipeline import DistributedSemanticProcessor


def _demo_concepts() -> list[SemanticConcept]:
    """Create a small provenance-labeled concept inventory for the runnable example."""
    return [
        SemanticConcept(
            concept_id="person-musician",
            label="musician",
            aliases=("player",),
            semantic_features={"PERSON": 1.0, "HUMAN": 1.0, "MUSICIAN": 1.0},
            modality_features={"visual": {"HUMAN_FORM": 0.8}, "spatial_motor": {"PERFORMS_MUSIC": 0.9}},
            lexical_category="noun",
            prototype_id="person",
            verified_by="demo",
            source_reference="demo://musician",
        ),
        SemanticConcept(
            concept_id="instrument-violin",
            label="violin",
            aliases=("fiddle",),
            semantic_features={"ARTIFACT": 1.0, "MUSICAL_INSTRUMENT": 1.0, "STRING_INSTRUMENT": 1.0},
            modality_features={"visual": {"WOODEN_BODY": 0.9, "STRINGS": 1.0}, "auditory": {"BOWED_TONE": 1.0}},
            lexical_category="noun",
            prototype_id="musical-instrument",
            verified_by="demo",
            source_reference="demo://violin",
        ),
        SemanticConcept(
            concept_id="food-sandwich",
            label="sandwich",
            semantic_features={"FOOD": 1.0, "EDIBLE": 1.0, "ARTIFACT": 0.1},
            modality_features={"visual": {"BREAD": 1.0}, "tactile": {"SOFT_TEXTURE": 0.7}},
            lexical_category="noun",
            prototype_id="food",
            verified_by="demo",
            source_reference="demo://sandwich",
        ),
    ]


def main() -> None:
    """Execute the demo, persist symbol annotations, and record its 36-stage run."""
    root = Path(__file__).resolve().parents[1]
    processor = DistributedSemanticProcessor(
        _demo_concepts(),
        config=DistributedSemanticConfig(
            simulate_biology=True,
            random_seed=23,
            steps_per_stage=2,
            population_size=6,
        ),
        associations={
            "person-musician": {"instrument-violin": 0.85},
            "instrument-violin": {"person-musician": 0.85, "food-sandwich": 0.02},
        },
    )
    request = DistributedSemanticRequest(
        sequence_id="distributed-semantics-demo",
        text="musician tuned violin",
        task="comprehension",
        modality_inputs={
            "visual": {"HUMAN_FORM": 0.7, "WOODEN_BODY": 0.8, "STRINGS": 0.9},
            "auditory": {"BOWED_TONE": 0.8},
        },
        syntactic_relations=(
            {"head": "tuned", "dependent": "musician", "relation": "subject"},
            {"head": "tuned", "dependent": "violin", "relation": "object"},
        ),
        event_roles={"Agent": "musician", "Action": "tuned", "Theme": "violin"},
    )
    result = processor.process(request)
    annotations = index_distributed_semantic_annotations(root)
    run = record_distributed_semantic_result(result, root)
    print("Stages:", len(result.stage_trace))
    print("Mathematical operations:", len(result.operation_trace))
    print("Biological reference stages:", len(result.biological_trace))
    print("Selected concepts:", result.output["selected_concepts"])
    print("Annotation catalogue:", annotations["database_path"])
    print("SQLite run:", run["run_id"])


if __name__ == "__main__":
    main()
