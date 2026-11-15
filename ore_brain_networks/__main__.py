"""Run a deterministic BNBEA smoke demonstration from the command line."""
from __future__ import annotations

import json

import numpy as np

from .contracts import BNBEAConfig, BNBEARequest, LexicalEntry
from .pipeline import BNBEAProcessor


def main() -> None:
    """Run synthetic visual and acoustic inputs and print a compact model-status report."""
    image = np.zeros((64, 96), dtype=np.float64)
    image[12:52, 12:17] = 1.0
    image[12:17, 12:42] = 1.0
    image[32:37, 12:42] = 1.0
    image[12:52, 52:57] = 1.0
    image[12:17, 52:82] = 1.0
    image[47:52, 52:82] = 1.0
    times = np.arange(4096, dtype=np.float64) / 16000.0
    audio = 0.35 * np.sin(2.0 * np.pi * 220.0 * times) + 0.18 * np.sin(2.0 * np.pi * 440.0 * times)
    request = BNBEARequest(
        sample_id="bnbea-demo",
        visual_image=image,
        auditory_waveform=audio,
        orthographic_sequence=("o", "r", "e"),
        lexicon=(LexicalEntry("ore", frequency=12.0), LexicalEntry("orex", frequency=1.0)),
        semantic_features={"MATERIAL": 0.7, "OBJECT": 0.4},
        context_features={"READING_DEMO": 1.0},
    )
    result = BNBEAProcessor(BNBEAConfig(), register_annotations=True).process(request)
    print(json.dumps({
        "schema": result.to_dict()["schema"],
        "sample_id": result.sample_id,
        "operations": result.diagnostics["operation_count"],
        "operation_statuses": [
            {"id": item["operation_id"], "status": item["status"]}
            for item in result.operation_trace
        ],
        "best_lexical_candidate": result.functional["vwfa_hierarchy"].get("best_lexical_candidate"),
        "biological_model": result.biological["lif"]["model"] if result.biological["lif"] else "disabled",
        "human_validation": result.experimental["human_reading_validation"]["status"],
        "annotation_database": result.diagnostics["annotation_registration"].get("database_path"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
