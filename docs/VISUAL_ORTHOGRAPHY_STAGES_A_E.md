# Visual Processing and Orthographic Representations — Stages A-E

The additive \`ore_visual_orthography\` package implements the sixteen steps in the
Visual Processing and OR Algorithm library specification. It keeps functional
orthographic evidence separate from a configurable population-level biological
reference simulation.

## Stage and operation registry

| Stage | Operations | Purpose |
|---|---|---|
| A | OVO-001–003 | Visual sampling, Gabor feature extraction, candidate glyph-shape integration |
| B | OVO-004–005 | Abstract letter identity, supervised visual-variant learning |
| C | OVO-006–009 | Relative position, uncertainty, position-specific conjunctions, open bigrams |
| D | OVO-010–012 | Parallel letter populations, concurrent conjunctions, recurrence and competition |
| E | OVO-013–016 | Transposition similarity, channel integration, lexical activation, stabilization |

Each operation has a stable ID, mathematical-operation list, objective,
input/output descriptions and a qualified biological-reference statement in
\`ore_visual_orthography.registry.OPERATION_REGISTRY\`.

## Functional and mathematical implementation

- **A:** Local luminance contrast, normalization, quadrature Gabor energy,
  rectification, divisive normalization, feature pooling and candidate region
  representations.
- **B:** Diagonal Mahalanobis prototype distances, radial-basis-like tuning,
  softmax identity populations, exponential-moving-average learning and
  within-identity cosine similarity.
- **C:** Normalized horizontal coordinates, Gaussian position populations,
  pairwise activation multiplication, order-constrained conjunctions and
  distance-decayed open-bigram generation.
- **D:** Vectorized letter activation matrices, joint letter/conjunction
  population vectors, neighbor message passing, lateral inhibition,
  softmax normalization and damped recurrent updates.
- **E:** Letter multiset overlap, ordered edit similarity, open-bigram Jaccard
  similarity, weighted channel integration, frequency-sensitive lexical
  scoring, softmax competition, and iterative settling with a state-delta
  convergence measure.

For the simplified set of all forward ordered pairs, JUDGE has ten open
bigrams. JUGDE shares nine of the ten with JUDGE; their set union contains
eleven items, giving Jaccard similarity \(9/11 \approx 0.818\). A replacement
string such as JUPTE loses more pair evidence. This is an illustrative model
score, not an experimentally measured human-recognition probability.

## Biological-reference track

\`BiologicalReferenceSimulator\` advances ten interacting population groups:
retina/LGN, V1, V2/V4, posterior VWFA, letter identity, letter position,
letter conjunction, open bigram, integrated orthography and lexical candidates.
The model uses current-based leaky-integrate-and-fire updates:

\[
I_i(t)=I_i^{external}(t)+\sum_j w_{ij}r_j(t)-I_i^{inhibitory}(t)
\]

\[
\tau_m\frac{dV_i}{dt}=V_{rest}-V_i+R_m I_i(t)
\]

The numerical implementation uses forward Euler integration, refractory-state
tracking, spike threshold/reset and smoothed population firing rates. Its
voltage, current, time and firing-rate outputs are labelled with units.

This is an explicit, configurable computational hypothesis. It is not an
anatomically fitted or empirically validated simulation of human reading.
Population labels are broad functional analogies, not exclusive anatomical
assignments. Gaussian position distributions and open bigrams are model
representations, not claims that the brain explicitly stores those data
structures.

## Quick start

\`\`\`python
from ore_visual_orthography import (
    LexicalCandidate,
    VisualOrthographyConfig,
    VisualOrthographyProcessor,
    VisualOrthographyRequest,
)

processor = VisualOrthographyProcessor(VisualOrthographyConfig())
request = VisualOrthographyRequest(
    sample_id="reading-001",
    orthographic_sequence=("j", "u", "d", "g", "e"),
    comparison_strings=("jugde", "jupte"),
    lexicon=(
        LexicalCandidate("judge", frequency=30),
        LexicalCandidate("jugde", frequency=1),
        LexicalCandidate("jungle", frequency=20),
    ),
    language="en",
)
result = processor.process(request)
print(len(result.operation_trace))  # 16
print(result.functional["stage_e"]["stabilization"])
\`\`\`

For image-to-letter experiments, provide labelled visual-form prototypes.
The convenience method \`processor.make_letter_prototype("a", glyph_crop)\`
extracts a feature vector from one caller-supplied glyph crop. Prototype
labels are supplied by the experimenter; the package does not pretend that
connected-component segmentation is a complete OCR system.

For supervised visual invariance learning, pass labelled feature vectors in
\`VisualOrthographyRequest.labeled_variants\`. These can be produced by
\`make_letter_prototype\` from glyph crops with known labels.

## SQLite annotations and execution logging

The package calls \`register_visual_orthography_annotations\` by default when a
processor is initialized. It uses the repository's existing AST-backed
\`ore_logger\` and stores class, method, function and module docstrings in the
shared \`ore_logs/ore_logger.sqlite3\` annotation catalogue. Set
\`ORE_LOGGER_DB\` to choose another catalogue path. Set \`ORE_AUTO_LOG=0\` to
disable runtime tracing when launching a script.

Every new Python class and callable has a docstring; the focused tests verify
that the annotation catalogue contains the processor class, its process
method, and numerical helper functions.

## Run

\`\`\`bash
python -m ore_visual_orthography
python -m pytest tests/test_visual_orthography.py -q
python -m pytest
\`\`\`

The package is additive: it does not replace or alter the existing
\`ore_brain_networks\` pipeline. Its image segmentation is a thresholded
candidate-region proposal; reliable visual letter recognition requires an
appropriately constructed labelled prototype inventory or an explicit
orthographic sequence. Independent human behavioral and neural recordings
are required before making biological-validity claims.
