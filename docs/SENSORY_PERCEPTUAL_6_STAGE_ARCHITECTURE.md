# Integrated Brain Sensory and Perceptual Processing — six-stage implementation

## Scope and operation map

This document records the implementation of the six stages in the source algorithm. The runtime registry in
`ore_sensory_perception.registry` is canonical: each operation has a stable ID, objective, mathematical
description, biological analogue and execution trace.

| Stage | Name | Operations |
|---|---|---:|
| 1 | Automatic text-region detection | ISP-001–ISP-006 |
| 2 | Automatic character segmentation from raw pages | ISP-007–ISP-012 |
| 3 | Auditory signal processing | ISP-013–ISP-018 |
| 4 | Acoustic feature extraction | ISP-019–ISP-024 |
| 5 | Phonetic feature detection | ISP-025–ISP-031 |
| 6 | Auditory phoneme identification | ISP-032–ISP-039 |
| **Total** | | **39** |

The visual and auditory inputs are independent. A request may contain either or both. The visual path produces
candidate regions, connected stroke groups and optional grapheme hypotheses; it is deliberately not described
as general OCR. It assigns no letter label unless compatible caller-supplied prototypes are provided. The audio
path produces acoustic, phonetic and phoneme evidence. A caller can provide a language/dialect-specific
`PhonemeSpec` inventory, speaker baseline and phonetic context. The default inventory is an English-oriented
demonstration and is not universal.

## Stage 1 — Automatic text-region detection

**Implementation:** `TextRegionDetector.step_01` through `step_06`.

1. **Receive visual input.** Convert grayscale/RGB(A) input to luminance, then estimate local mean and
   variance. The normalized contrast field is
   `C(x,y) = (I(x,y) - μlocal(x,y)) / (σlocal(x,y) + ε)`. This approximates contrast-sensitive retinal/LGN
   processing; Gaussian neighborhoods are an engineering approximation to center-surround interactions.
2. **Extract primitive visual features.** Convolve the contrast field with even/odd Gabor kernels at several
   orientations and spatial frequencies. Their phase-insensitive energy is
   `E = sqrt(response_even² + response_odd²)`. Local divisive normalization controls gain. These filters are
   V1-inspired orientation/spatial-frequency populations, not literal cortical receptive fields.
3. **Detect text-like spatial organization.** Partition feature maps into tiles. Extract edge density,
   orientation entropy, spatial-frequency preference, element density and contrast variability. A linear
   score and logistic function `p = sigmoid(wᵀz + b)` estimate text-likeness. The coefficients are transparent
   model settings, not fitted biological parameters.
4. **Group neighboring candidates.** Compute spatial proximity and vertical alignment:
   `dij = exp(-||ci-cj||² / 2s²)`, `aij = exp(-|yi-yj|/s)`. Connect tiles when their weighted grouping score
   passes a threshold. This is an explicit graph/connected-group approximation to perceptual grouping.
5. **Estimate region boundaries.** Take the bounding-box union of grouped tiles and average their text-likelihood
   values. The output is a set of candidate regions, not a proof that the region contains readable text.
6. **Allocate visual attention.** Add caller-provided region bias to confidence logits and normalize with
   softmax. This models relative attentional gain/competition. It does not model eye movements or the full
   oculomotor system.

## Stage 2 — Automatic character segmentation from raw pages

**Implementation:** `CharacterSegmentationProcessor.step_07` through `step_12`.

7. **Detect local character-like structures.** Apply Otsu intensity thresholding together with local contrast,
   then connected-component labeling within attended regions. Component area, bounding box, density, width,
   height and center form local stroke/shape measurements. Connected components are not assumed to equal letters.
8. **Combine local features into glyph candidates.** Group spatially adjacent stroke fragments with compatible
   vertical overlap and region membership. Resize each grouped mask to a normalized grid and compute a
   16-dimensional shape descriptor. Optional caller prototypes are compared using mean-squared distance and
   converted into candidate probabilities with softmax. A prototype is a software reference, not a known
   biological storage format.
9. **Separate overlapping neighboring letters.** Compute a vertical ink projection and identify low-projection
   valleys. Candidate left/right cuts are retained beside the uncut hypothesis; the model does not force a split
   when the evidence is ambiguous.
10. **Encode letter positions and order.** Attach normalized x/y coordinates and ordinal reading position to
    every glyph candidate. Spatial/ordinal codes retain order; they are not permutation-invariant.
11. **Increase tolerance to visual variation.** Normalize feature vectors and retain aspect/view consistency.
    This gives limited tolerance to scale and translation, while preserving orientation and sequence information.
    It does not claim invariance to all fonts, scripts, rotations, or distortions.
12. **Use linguistic feedback.** Combine prototype likelihood with caller-supplied orthographic prior:
    `log P(letter|image,context) ∝ log P(image|letter) + log P(letter|context)`. Softmax yields graded
    candidate probabilities and entropy reports ambiguity. No missing context or labels are invented.

## Stage 3 — Auditory signal processing

**Implementation:** `AuditorySignalProcessor.step_13` through `step_18`.

13. **Receive the acoustic waveform.** Validate the finite mono waveform, subtract its DC mean and preserve
    the sample rate and time coordinate `t[n] = n/fs`. This is a digital representation of sound pressure
    after peripheral transmission; it does not simulate the pinna or middle-ear ossicles.
14. **Perform cochlear frequency separation.** Create bandpass filters with center frequencies spaced on an
    ERB-like scale, `ERB(f)=21.4 log10(1+0.00437f)`. Channel outputs provide a tonotopic filter-bank analogue
    of basilar-membrane/hair-cell frequency selectivity.
15. **Convert vibration to neural-signal proxies.** Compute analytic envelopes, rectify/compress them with
    `log(1 + α|Hilbert(x)|)`, threshold and normalize into bounded channel-rate proxies. This approximates
    transduction and auditory-nerve response nonlinearity; actual hair-cell mechanics and synaptic dynamics
    are much richer.
16. **Preserve temporal information.** Smooth channel envelopes, compute first differences, pool across channels
    and estimate onset peaks and inter-onset timing. The representation keeps timing rather than collapsing
    speech to average spectrum.
17. **Organize perceptual streams.** Build a channel adjacency graph from envelope correlation and approximate
    harmonic relationships, then find connected groups. This captures selected cues used in auditory scene
    analysis but does not perform blind source separation or solve the cocktail-party problem.
18. **Prioritize relevant input.** Add caller-specified attention to stream salience logits and normalize with
    softmax. The result is a relative gain/priority distribution, not a claim of measured attention.

## Stage 4 — Acoustic feature extraction

**Implementation:** `AcousticFeatureExtractor.step_19` through `step_24`.

19. **Construct a time-frequency representation.** Apply a windowed short-time Fourier transform:
    `X(τ,f)=Σn x[n]w[n-τ]exp(-j2πfn/fs)`. It gives a convenient engineering representation of time-varying
    spectral information. Cochlear/cortical processing need not compute an explicit Fourier transform.
20. **Extract energy and spectral distribution.** Calculate spectral power `P=|X|²`, frame energy, RMS,
    low/high-band energy and centroid `C=ΣfP/ΣP`. These are acoustic measurements, not phoneme labels.
21. **Detect harmonic structure and pitch.** Compute normalized short-window autocorrelation
    `R[k]=Σn x[n]x[n-k]`; when a non-zero lag `k0` has a sufficiently strong peak, estimate
    `F0≈fs/k0`. The analysis frames are centered on the STFT time grid to keep the frame arrays aligned.
22. **Extract formants.** Use linear predictive coding, polynomial roots and resonance bandwidth constraints
    to estimate up to F1–F3. Formant estimates with unstable roots remain unavailable/NaN in internal arrays;
    zero-filled summaries are clearly marked as summaries rather than measured zero formants.
23. **Extract acoustic transitions.** Compute first differences of formant and centroid trajectories, and
    estimate a cautious onset/VOT proxy. True VOT is tied to phonetic landmarks and cannot be robustly
    identified from every arbitrary waveform.
24. **Build the acoustic feature vector.** Assemble an ordered frame vector
    `[F0, F1, F2, F3, RMS, centroid, periodicity]`. It is time-indexed acoustic evidence and is explicitly
    passed to Stage 5 without preassigning phoneme identity.

## Stage 5 — Phonetic feature detection

**Implementation:** `PhoneticFeatureDetector.step_25` through `step_31`.

25. **Detect voicing-related characteristics.** Combine periodicity, low-frequency energy and VOT-related
    evidence using a logistic response. The output is a frame-wise probability/evidence trace, not a direct
    observation of vocal-fold motion.
26. **Detect manner of articulation.** Construct burst, frication, nasal-band, periodicity and closure cues.
    Normalize the feature dimensions and compare them with broad stop/fricative/nasal/approximant/affricate/
    vowel population prototypes using radial-basis similarity and softmax competition.
27. **Detect place of articulation.** Combine F1/F2 transitions, burst/spectral centroid and periodicity
    changes. Compare with broad place prototypes. The result is an uncertain inferred class; the articulators
    are not directly observed by the listener.
28. **Detect vowel characteristics.** Produce coarse height/backness/rounding tendencies from F1/F2.
    These mappings vary by speaker, language and vocal-tract characteristics and must not be treated as
    universal articulatory conversion formulae.
29. **Normalize across speakers.** Use caller-supplied baseline center/scale when available; otherwise use
    robust median and median absolute deviation (MAD). This reduces scale differences while acknowledging
    that human speaker normalization is nonlinear and context-sensitive.
30. **Detect coarticulation.** Concatenate previous, current and following acoustic-feature states and temporal
    differences. The trajectory retains contextual transitions because speech sounds overlap in production.
31. **Build a distributed phonetic population code.** Concatenate voicing, manner, place, vowel and trajectory
    representations, then L2-normalize each frame. This is a population-code analogue, not a one-feature/
    one-neuron anatomical statement.

## Stage 6 — Auditory phoneme identification

**Implementation:** `PhonemeIdentifier.step_32` through `step_39`.

32. **Activate candidate phonemes.** For each supplied/default category, combine manner, place and voicing
    evidence into a candidate activation score. All categories remain active as a normalized distribution.
33. **Compare with learned categories.** Convert category evidence to likelihood-like scores and combine with
    inventory priors using Bayes-style log evidence. The category inventory and priors are explicit caller/model
    data, not assumed universal templates.
34. **Integrate evidence over time.** Accumulate log posteriors across frames and normalize the cumulative
    score. This is a sequential evidence integration analogy, not a claim that the brain carries out symbolic
    logarithms.
35. **Apply categorical perception.** A sigmoid boundary over continuous voicing/VOT evidence yields graded
    category probabilities. Category boundaries depend on language, context and listener experience.
36. **Use context.** Add bounded caller-supplied phonetic/linguistic log priors to sensory evidence. Context
    biases ambiguous candidates but cannot replace missing acoustic input.
37. **Update through recurrent feedback.** Iteratively combine evidence with neighboring-frame predictions and
    apply a damped recurrent update. A convergence trace records the magnitude of successive probability
    changes.
38. **Select or retain candidates.** Choose a category only when both confidence and top-two margin meet
    configured thresholds. Otherwise return an abstention (`None`) and retain the full distribution.
    Shannon entropy `H=-Σk pk log(pk)` quantifies uncertainty.
39. **Construct a phoneme sequence.** Merge adjacent frame decisions into time-ordered segments with onset,
    offset, duration, mean confidence and entropy. Boundaries are frame-derived estimates; coarticulated speech
    does not always contain sharp segment boundaries.

## Biological reference track

`SensoryBiologicalSimulator` maintains 16 named populations spanning early visual, ventral visual,
language-feedback, cochlear/nerve, auditory-scene, auditory cortical, STG-phonetic and phoneme populations.
Directed excitatory projections implement forward and recurrent routes. Each population uses a current-based
leaky-integrate-and-fire update:

```text
τm dV/dt = (Vrest - V) + Rm I
V(t+Δt) = V(t) + (Δt/τm) * ((Vrest - V(t)) + Rm I)
if V >= Vthreshold: emit spike; V = Vreset; enter refractory state
```

Voltage is represented in mV, current in pA, resistance in MΩ, and the conversion factor 0.001 converts
pA·MΩ to mV. Each of the 39 algorithm operations advances the target population and the recurrent network,
recording membrane and spike summaries. The configured values are engineering defaults for reproducibility,
not physiological constants fitted to an individual. The network is a transparent, limited reference model;
it does not reproduce full anatomy, ion-channel conductances, synaptic kinetics, or measured cortical dynamics.

## SQLite annotation and execution logging

At processor construction, `register_sensory_annotations()` calls ORE's existing
`ore_logger.LoggerDatabase` and `sync_repository` APIs. The same repository-wide AST catalogue stores
module/class/function/method docstrings, qualified names, signatures, line locations and source hashes in
`ore_logger.annotations`. The default database path is `ore_logs/ore_logger.sqlite3`; the environment
variable `ORE_LOGGER_DB` or an explicit `database_path` overrides it. Annotation registration can be disabled
for isolated tests with `register_annotations=False`.

The execution result includes:
- `operation_trace`: exactly 39 entries in `ISP-001`–`ISP-039` order, each containing the registry
  annotation, status and bounded functional summary.
- `biological_trace`: 39 distinct LIF network updates when biological simulation is enabled.
- Separate visual, auditory, phonetic and phoneme outputs and a compact integration report.

Large arrays are summarized in output/log metadata. Raw images and raw audio are not written to execution
events by the result serializer. The annotation catalogue stores source descriptions, not sample content.

## API

```python
import numpy as np
from ore_sensory_perception import (
    SensoryConfig, SensoryPerceptionProcessor, SensoryRequest, PhonemeSpec
)

request = SensoryRequest(
    sample_id="sample-001",
    visual_image=np.zeros((64, 192), dtype=float),  # optional
    auditory_waveform=np.zeros(8000, dtype=float),  # optional
    sample_rate_hz=16000.0,
    language="en-US",
    phoneme_inventory=(),  # pass the correct inventory for the target language/dialect
)
processor = SensoryPerceptionProcessor(config=SensoryConfig(simulate_biology=True))
result = processor.process(request)
assert len(result.operation_trace) == 39
assert len(result.biological_trace) == 39
```

## Validation and limitations

The focused suite checks registry integrity, combined and single-modality execution, time alignment for short
audio, invalid input rejection, annotation docstrings and the SQLite catalogue integration. Passing those
tests demonstrates software behavior only. It does not establish biological equivalence or phoneme-recognition
accuracy on natural speech. Human validation requires independently collected, consented, appropriately
annotated image/audio/behavior data and a declared evaluation protocol. The fallback feature weights and
phoneme inventory should be treated as transparent baselines that require empirical fitting before use in
scientific inference.
