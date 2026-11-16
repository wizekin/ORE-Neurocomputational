"""Run a deterministic six-stage VWFA smoke demonstration from the command line."""
from __future__ import annotations

import json

import numpy as np

from .contracts import LexicalEntry, MorphemeEntry, VWFAConfig, VWFARequest
from .pipeline import VWFAProcessor


def main() -> None:
    """Run a synthetic printed-form example with caller-supplied word and morpheme evidence."""
    image = np.zeros((72, 144), dtype=np.float64)
    image[12:60, 15:20] = 1.0
    image[12:17, 15:45] = 1.0
    image[34:39, 15:45] = 1.0
    image[12:60, 62:67] = 1.0
    image[12:17, 62:92] = 1.0
    image[55:60, 62:92] = 1.0
    request = VWFARequest(
        sample_id="vwfa-demo",
        visual_image=image,
        orthographic_sequence=("c", "a", "t", "s"),
        lexicon=(LexicalEntry("cats", 22.0, {"ANIMAL": 1.0, "PLURAL": 0.5}),),
        morphology_inventory=(
            MorphemeEntry("cat", "stem", 15.0, {"ANIMAL": 1.0}),
            MorphemeEntry("s", "inflection", 20.0, {"PLURAL": 1.0}, {"NUMBER_PLURAL": 1.0}),
        ),
        context_features={"ANIMAL": 0.8},
    )
    result = VWFAProcessor(VWFAConfig(simulate_biology=True), register_annotations=True).process(request)
    payload = result.to_dict()
    print(json.dumps({
        "schema": payload["schema"],
        "sample_id": result.sample_id,
        "stage_count": result.diagnostics["stage_count"],
        "operation_count": result.diagnostics["operation_count"],
        "operation_ids_complete": result.diagnostics["operation_ids_complete"],
        "selected_candidate": result.functional["selected_candidate"],
        "morphology_candidate_count": len(result.functional["morphology_candidates"]),
        "biological_model": result.biological["final_snapshot"]["model"] if result.biological["enabled"] else "disabled",
        "annotation_database": result.diagnostics["annotation_registration"].get("database_path"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
