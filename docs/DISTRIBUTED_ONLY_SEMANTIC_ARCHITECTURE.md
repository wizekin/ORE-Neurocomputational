# Distributed-Only Semantic Representation and Embodied Concept Processing

This package implements the 21-stage algorithm in the Google Drive source
**Master Algorithm - Distributed-Only Semantic Representation and Embodied
Concept Processing.txt**.

## Architectural rule

A concept is represented as a **collection of separate property-specific
population states**, not as a vector stored in a central semantic hub. The
default populations are visual, auditory, action, tactile, affective,
linguistic, contextual, and spatial. Inter-population connectivity is a
distributed graph. Candidate scores are an analytical readout over the
separate patterns and are not an internal amodal storage code.

## Quick start

~~~python
from ore_distributed_only import (
    ConceptPattern,
    DistributedOnlyProcessor,
    DistributedSemanticRequest,
)

concepts = [
    ConceptPattern(
        label="violin",
        category="musical_instrument",
        description="A stringed musical instrument played with a bow.",
        typical_action="Hold it and move the bow across its strings.",
        properties={
            "linguistic": {"token:violin": 1.0},
            "visual": {"shape:curved_body": 0.9, "parts:strings": 1.0},
            "auditory": {"sound:bowed_string": 0.95},
            "action": {"action:hold_and_bow": 1.0},
            "tactile": {"texture:wood": 0.7},
            "spatial": {"posture:under_chin": 0.8},
            "affective": {"affect:musical_enjoyment": 0.5},
        },
    )
]

processor = DistributedOnlyProcessor(concepts)
result = processor.process(DistributedSemanticRequest(
    linguistic_input="violin",
    modality_inputs={"auditory": {"sound:bowed_string": 0.8}},
    task="describe",
))

print(result.interpretation, result.confidence, result.response)
print(result.population_means)
print(result.diagnostics)
print(result.logging)
~~~

The feature inventory must be supplied by the application or learned from
authorized examples. The package does not contain a hidden world-knowledge
database or pretrained image/audio encoder. Text is tokenized into lexical
cues; audio-like time-series can be supplied as numeric arrays and summarized
with sampling, normalization, temporal filtering, and an FFT-based engineering
readout.

## Implemented sequence

| Phase | Stages | Functional scope |
|---|---:|---|
| Input encoding | 1–4 | Receive signals, extract property features, initialize populations, spread learned associations |
| Distributed reactivation and interaction | 5–14 | Perceptual, action, affective, episodic reactivation; recurrence, inhibition, reliability, timing, assemblies, completion |
| Interpretation and output | 15–18 | Category circuits, context/task modulation, ambiguity resolution, response readout |
| Learning and resilience | 19–21 | Hebbian/STDP-style updates, redundancy metrics, population/connection lesions and degradation analysis |

Every registered sub-operation has a mathematical expression, a functional
responsibility, a biological reference, a runtime status, and an output summary.
The runtime trace is checked against the stage registry so omitted operation
outputs cause an explicit error.

## Mathematical and biological interpretation

The functional track uses vector activation, bounded nonlinearities, cosine
similarity, population connectivity, recurrent updates, softmax normalization,
phase-coherence estimates, correlations, Hebbian co-activation, an optional
spike-timing-dependent update, and lesion masks. A separate current-based
leaky-integrate-and-fire track simulates each population group with membrane
voltage (mV), synaptic current (pA), refractory periods (ms), spike counts, and
firing-rate estimates (Hz) at every semantic stage. These are numerical models or
analysis readouts. The LIF reference is kept separate from semantic feature
inference so a spike in the reference simulator is not misrepresented as an
observed biological neural response. The brain does not literally execute Python, matrix
multiplication, softmax, cosine similarity, or symbolic formulae. No default
parameter should be interpreted as a measured human physiological constant.

Temporal phase and cross-correlation values are model diagnostics, not evidence
that oscillatory synchrony is the sole or established mechanism of semantic
binding. Lesion experiments are counterfactual network perturbations, not
clinical or disease-progression predictions.

Learning is **opt-in**: model connection weights are updated only when
DistributedSemanticRequest(learning=True, learning_target=...) explicitly
authorizes an update to a registered concept. spike_times_ms optionally
supplies population event times for the STDP-style approximation.

## SQLite annotations and run traces

By default, processing writes to:

ore_logs/ore_logger.sqlite3

The existing ore_logger AST catalogue indexes modules, classes, and
functions using their Python docstrings. The distributed package additionally
registers 21 stage annotations and one annotation for every sub-operation,
including its equation and biological interpretation. Each processing run
persists operation and stage events through SQLiteRunRecorder.

To explicitly index annotations without running inference:

~~~python
from ore_distributed_only import index_distributed_annotations

report = index_distributed_annotations(
    repository_root=".",
    database_path="ore_logs/ore_logger.sqlite3",
)
print(report)
~~~

Use auto_log=False in the processor constructor to disable automatic run
recording, or supply repository_root and database_path to select the database
location.

## Scientific and engineering limitations

- Property feature maps and concept descriptions are supplied data; the model
  does not infer reliable real-world properties from raw pixels or waveforms.
- Fixed-size deterministic feature hashing can create collisions; it is a
  compact demonstration representation, not a substitute for learned encoders.
- Population-level activation is not a neuron-by-neuron human brain simulation.
- Recurrent weights and learning rates require calibration against empirical
  behavioral and neural data before quantitative neuroscience claims can be made.
- Unit tests establish software behavior and trace coverage, not biological
  validity or human-level semantic understanding.
