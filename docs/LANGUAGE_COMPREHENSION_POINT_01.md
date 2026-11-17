# Language Comprehension — Point 1: Perceptual Processing

## Scope

This package implements the first source point: transform a continuous auditory and/or visual sensory stream into ordered, uncertainty-aware phonetic or orthographic form hypotheses. It ends at the interface to lexical access. It does not retrieve word meanings, parse syntax, compose a proposition, or infer speaker intent.

The implementation is source-aligned at two levels:

- `Point1PerceptualProcessor` exposes the source's 12 ordered functional operations (`LC-P01-01`–`LC-P01-12`).
- The established `ore_sensory_perception` dependency performs its 39 lower-level auditory/visual operations (`ISP-001`–`ISP-039`) and, when enabled, advances its separate 16-population LIF reference network.

The Point 1 biological trace groups the actual low-level LIF updates under their corresponding source-level operations. Both traces remain available in the result object.

## Source-level processing sequence

| ID | Functional operation | Mathematical operations represented | Biological reference |
|---|---|---|---|
| LC-P01-01 | Capture sensory samples and preserve time/space | Sampling, time/space indexing, finite-value and signal/noise summaries, leaky integration as an abstraction | Cochlear/retinal transduction and persistent sensory population activity |
| LC-P01-02 | Route auditory and visual evidence | Modality availability gates and parallel pathway bookkeeping | Auditory nerve/brainstem/thalamus/cortex; retina/LGN/V1/ventral visual pathway |
| LC-P01-03 | Extract elementary modality-specific features | Filter banks, convolution, rectification, log compression, STFT, Gabor filtering, feature vectors | Cochlear tonotopy, hair-cell/auditory-nerve activity, retina/LGN and V1-like feature populations |
| LC-P01-04 | Normalize nuisance variation | Speaker-relative z-scores, robust normalization, contrast and scale normalization | Adaptive gain and learned invariance across auditory and ventral visual processing |
| LC-P01-05 | Encode distributed population evidence | Graded feature vectors, nonlinear activation, divisive normalization and recurrent updates | Distributed STG phonetic and ventral occipitotemporal orthographic populations |
| LC-P01-06 | Integrate evidence over time/space | Temporal smoothing, coherence, spatial pooling and time-indexed matrices | Auditory temporal integration and recurrent visual grouping |
| LC-P01-07 | Detect and order sub-word candidates | Temporal segmentation, onset/offset estimates, ordinal position codes and sequence assembly | Dynamic STG phonetic coding and retinotopic-to-ordinal ventral visual transformations |
| LC-P01-08 | Combine cues according to confidence | Posterior probabilities, cue-weighted sums, margins and prior-weighted log evidence | Distributed nonlinear cue integration; probabilities are model estimates, not neural firing probabilities |
| LC-P01-09 | Maintain alternative candidates | Softmax competition, top-k retention and entropy | Competition and recurrent interactions across sensory populations |
| LC-P01-10 | Estimate likely boundaries | Change detection, frame-to-time conversion, bounding boxes and boundary uncertainty | Temporal event coding, visual grouping and attention-dependent sampling |
| LC-P01-11 | Apply attentional/predictive modulation | Attention logits, softmax, multiplicative gain and prior-weighted evidence | Auditory/visual attentional gain and recurrent language-related feedback |
| LC-P01-12 | Package forms for lexical access | Ranking, modality-local normalization and uncertainty propagation | Distributed handoff to lexical systems; no single anatomical handoff point is assumed |

## Python API

```python
import numpy as np
from ore_sensory_perception import SensoryRequest, SensoryConfig, PhonemeSpec
from ore_language_comprehension import Point1PerceptualProcessor

sample_rate = 16000.0
time = np.arange(4800) / sample_rate
waveform = 0.3 * np.sin(2 * np.pi * 180 * time)
request = SensoryRequest(
    sample_id="example-001",
    auditory_waveform=waveform,
    sample_rate_hz=sample_rate,
    phoneme_inventory=(
        PhonemeSpec("a", "vowel", voiced=True, vowel_height="open"),
        PhonemeSpec("m", "nasal", place="bilabial", voiced=True),
    ),
)
processor = Point1PerceptualProcessor(config=SensoryConfig(simulate_biology=True))
result = processor.process(request)
print(len(result.functional_trace))          # 12
print(len(result.sensory_operation_trace))    # 39
print(result.form_hypotheses["auditory"])
```

For visual input, supply `visual_image` and, when letter labels are required, caller-provided `grapheme_prototypes`. Without prototypes, unlabelled visual shapes are retained rather than assigned invented graphemes. Audio decoding likewise depends on the supplied language-specific `phoneme_inventory`; an empty or unsuitable inventory cannot provide reliable phoneme labels.

## SQLite annotations

On processor initialization, `register_point01_annotations` synchronizes module, class, function and method docstrings from `ore_language_comprehension` and its `ore_sensory_perception` dependency into the existing `ore_logger.annotations` table. The default database is `ore_logs/ore_logger.sqlite3`; `ORE_LOGGER_DB` or the `database_path` argument selects another file. Only these two packages are refreshed by this helper, so unrelated package annotations are not marked stale.

Execution records created by the existing ORE runtime logger continue to use its shared `runs` and `events` tables. Large raw images and waveforms are summarized in result serialization and are not copied into event metadata.

## Biological and mathematical qualifications

The LIF track is a computational reference with configurable membrane voltage, input/recurrent current, firing rate, spike count and simulated time. Its 16 named populations and projections are explicit hypotheses. They are not a complete cochlea, retina, STG, VWFA or whole-brain model; their values are not measurements from a human participant.

Convolution, STFT, softmax, entropy, Bayesian log-evidence, z-scores and leaky integration are useful mathematical descriptions or engineering operations. The code does not claim that neurons literally execute these exact software operations. Its functional track is an algorithmic approximation, while its LIF track is a separate biological analogy.

The implementation deliberately keeps auditory and visual candidate spaces separate. It does not infer cross-modal alignment unless the caller supplies `metadata["orthography_phonology_alignment"]`. It never turns a phoneme sequence or a glyph sequence into a word meaning during Point 1.

## Validation

Run the focused tests:

```bash
python -m pytest tests/test_language_comprehension_point01.py -q
```

Run the package demonstration:

```bash
python -m ore_language_comprehension
```

Software tests validate implementation contracts and trace integrity. They do not establish biological equivalence or clinical validity.
