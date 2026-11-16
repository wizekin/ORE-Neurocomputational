"""Run a deterministic six-stage sensory-perception smoke demonstration."""
from __future__ import annotations

import numpy as np

from .contracts import SensoryConfig, SensoryRequest
from .pipeline import SensoryPerceptionProcessor


def main() -> None:
    """Run a visual-and-auditory synthetic sample and print compact stage outputs.

    The synthetic waveform and page are only software fixtures. Their simulated neural activity is
    not human measurement data and the fallback phoneme set is English-oriented.
    """
    visual = np.full((96, 320), 0.94, dtype=np.float64)
    for line, y in enumerate((18, 42, 66)):
        visual[y:y + 5, 14:300 - line * 18:11] = 0.08
    sample_rate = 16000.0
    time = np.arange(int(0.32 * sample_rate)) / sample_rate
    waveform = 0.45 * np.sin(2.0 * np.pi * 180.0 * time) + 0.12 * np.sin(2.0 * np.pi * 720.0 * time)
    request = SensoryRequest(
        sample_id="sensory-demo-001",
        visual_image=visual,
        auditory_waveform=waveform,
        sample_rate_hz=sample_rate,
    )
    result = SensoryPerceptionProcessor(
        config=SensoryConfig(simulate_biology=True), register_annotations=False
    ).process(request)
    print({
        "sample_id": result.sample_id,
        "registered_operations": len(result.operation_trace),
        "operation_ids": [row["operation_id"] for row in result.operation_trace],
        "text_regions": result.integration["text_region_count"],
        "character_hypotheses": result.integration["character_hypothesis_count"],
        "phoneme_segments": result.integration["phoneme_segment_count"],
        "biological_updates": len(result.biological_trace),
        "scope": result.integration["biological_scope"],
    })


if __name__ == "__main__":
    main()
