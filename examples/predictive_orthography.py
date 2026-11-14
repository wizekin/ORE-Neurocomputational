"""Runnable example for ORE's seven-stage predictive orthography master pipeline."""
from __future__ import annotations
import numpy as np
from ore_n3 import LabeledN2Patch, N3Patch, N3Processor, N3Sequence
from ore_predictive_orthography import LIFConfig, PredictiveOrthographyProcessor

# Demonstration vectors only; real studies should use labelled, segmented N2 patches.
references = [
    LabeledN2Patch("a", features=np.array([1.0, 0.0, 0.0, 0.0])),
    LabeledN2Patch("a", features=np.array([0.9, 0.1, 0.0, 0.0])),
    LabeledN2Patch("b", features=np.array([0.0, 1.0, 0.0, 0.0])),
    LabeledN2Patch("b", features=np.array([0.1, 0.9, 0.0, 0.0])),
    LabeledN2Patch("c", features=np.array([0.0, 0.0, 1.0, 0.0])),
    LabeledN2Patch("c", features=np.array([0.0, 0.1, 0.9, 0.0])),
]
n3 = N3Processor().fit(
    references, grapheme_sequences=[("a", "b", "c"), ("a", "a", "b"), ("b", "c", "a")]
)
processor = PredictiveOrthographyProcessor(
    n3, lif_config=LIFConfig(neuron_count=16, simulation_steps=120)
)
request = N3Sequence("example-001", [
    N3Patch("g1", features=np.array([0.95, 0.05, 0.0, 0.0]), segmentation_confidence=0.95),
    N3Patch("g2", features=np.array([0.05, 0.95, 0.0, 0.0]), segmentation_confidence=0.90),
    N3Patch("g3", features=np.array([0.0, 0.05, 0.95, 0.0]), segmentation_confidence=0.85),
])
result = processor.process(request, task_relevance=np.array([0.2, 1.0, 0.2]))
print("Stages:", len(result.stage_trace))
print("Master operations:", len(result.operation_trace))
print("Candidates:", result.candidates)
print("Mean attended feature residual:", result.diagnostics["mean_attended_feature_residual_norm"])
print("LIF mean firing rates:", {
    name: population["mean_firing_rate_hz"]
    for name, population in result.biological["populations"].items()
})
