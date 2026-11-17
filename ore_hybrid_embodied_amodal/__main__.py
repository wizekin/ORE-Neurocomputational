"""Run a deterministic hybrid embodied–amodal concept-processing demonstration."""
from __future__ import annotations

from pathlib import Path

from . import (
    HybridConcept,
    HybridEmbodiedAmodalProcessor,
    HybridRequest,
    index_hybrid_annotations,
    record_hybrid_result,
)


def main() -> None:
    """Run the guitar example and persist source annotations and stage traces to SQLite."""
    root = Path(__file__).resolve().parents[1]
    concept = HybridConcept(
        concept_id="concept-guitar",
        label="guitar",
        aliases=("guitar",),
        category="musical instrument",
        features={"instrument": 1.0, "strings": 0.9, "music": 0.9, "neck": 0.8},
        modality_features={
            "visual": {"strings": 0.9, "neck": 0.8},
            "auditory": {"music": 0.9, "sound": 0.8},
            "motor": {"strumming": 0.9, "finger_position": 0.8},
        },
        verified_by="demo",
        source_reference="synthetic://hybrid-guitar-demo",
    )
    processor = HybridEmbodiedAmodalProcessor([concept])
    request = HybridRequest(
        sequence_id="hybrid-guitar-demo",
        modality_inputs={
            "visual": {"strings": 0.9, "neck": 0.8},
            "auditory": {"music": 0.9, "sound": 0.8},
            "motor": {"strumming": 0.9, "finger_position": 0.8},
        },
        task="recognition",
        context_features={"instrument": 0.9, "music": 0.8},
        task_features={"instrument": 1.0},
        modality_times={"visual": 0.0, "auditory": 0.05, "motor": 0.08},
        response_mode="recognition",
    )
    result = processor.process(request)
    annotations = index_hybrid_annotations(root)
    run = record_hybrid_result(result, root)
    print("Selected:", result.selected_concept)
    print("Stages:", result.diagnostics["stage_count"])
    print("Sub-operations:", result.diagnostics["operation_count"])
    print("Valid:", result.diagnostics["valid"])
    print("Indexed symbols:", annotations["hybrid_symbol_count"])
    print("SQLite run:", run["run_id"])
    print("SQLite database:", annotations["database_path"])


if __name__ == "__main__":
    main()
