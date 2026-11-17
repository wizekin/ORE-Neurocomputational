"""Run a deterministic audio-only demonstration of source Point 1."""
from __future__ import annotations

import numpy as np

from ore_sensory_perception import PhonemeSpec, SensoryConfig, SensoryRequest
from .point01 import Point1PerceptualProcessor


def main() -> None:
    """Create a synthetic harmonic signal, process it and print compact source-level results."""
    sample_rate = 16000.0
    time = np.arange(4800, dtype=np.float64) / sample_rate
    envelope = np.clip(time / 0.025, 0.0, 1.0) * np.clip((0.3 - time) / 0.025, 0.0, 1.0)
    waveform = envelope * (0.4 * np.sin(2 * np.pi * 180 * time) + 0.12 * np.sin(2 * np.pi * 540 * time))
    request = SensoryRequest(
        sample_id="language-comprehension-point-01-demo",
        auditory_waveform=waveform,
        sample_rate_hz=sample_rate,
        phoneme_inventory=(
            PhonemeSpec("a", "vowel", voiced=True, vowel_height="open", vowel_backness="front"),
            PhonemeSpec("m", "nasal", place="bilabial", voiced=True),
            PhonemeSpec("s", "fricative", place="alveolar", voiced=False),
        ),
    )
    processor = Point1PerceptualProcessor(
        config=SensoryConfig(lif_population_size=4, lif_steps_per_operation=1),
    )
    result = processor.process(request)
    print({
        "sample_id": result.sample_id,
        "functional_steps": len(result.functional_trace),
        "biological_steps": len(result.biological_trace),
        "phonetic_segments": len(result.form_hypotheses["auditory"]["form_hypotheses"]),
        "semantic_meaning_retrieved": False,
    })


if __name__ == "__main__":
    main()
