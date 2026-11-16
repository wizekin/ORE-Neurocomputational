"""Canonical registry of the six sensory-perception stages and their 39 ordered operations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one operation's purpose, mathematical operations and qualified biological analogue."""

    operation_id: str
    stage: int
    stage_name: str
    step: int
    operation_name: str
    objective: str
    mathematical_operations: tuple[str, ...]
    biological_reference: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready registry record for documentation and per-run execution traces."""
        return asdict(self)


STAGE_STEP_COUNTS = {1: 6, 2: 6, 3: 6, 4: 6, 5: 7, 6: 8}
STAGE_NAMES = {
    1: "Automatic text-region detection",
    2: "Automatic character segmentation from raw pages",
    3: "Auditory signal processing",
    4: "Acoustic feature extraction",
    5: "Phonetic feature detection",
    6: "Auditory phoneme identification",
}

_ROWS = (
    (1, "Receive visual input", "Represent pixels as luminance and local contrast.", "RGB luminance; local Gaussian mean/variance; C=(L-μ)/(σ+ε).", "Retinal photoreceptors and center-surround retinal/LGN contrast processing."),
    (1, "Extract primitive visual features", "Compute oriented, multiscale early visual responses.", "Gabor convolution; quadrature energy sqrt(Re²+Ro²); rectification; divisive normalization.", "V1-like orientation and spatial-frequency selective populations."),
    (1, "Detect text-like spatial organization", "Estimate whether local visual tiles contain writing-like structure.", "Edge density; orientation entropy; spatial-frequency statistics; sigmoid(wᵀx+b).", "Distributed visual populations sensitive to repeated contours and regular spatial patterns."),
    (1, "Group neighboring text-like elements", "Link adjacent candidates using distance, height, alignment and baseline cues.", "Pairwise distance; similarity scores; weighted grouping graph; connected components.", "Perceptual grouping and recurrent interactions among spatially related visual populations."),
    (1, "Identify probable text-region boundaries", "Aggregate groups into candidate text-region boxes and confidence values.", "Bounding-box union; pooled evidence; regional confidence.", "Figure-ground organization and spatial attention; cortical regions need not be rectangular boxes."),
    (1, "Direct visual attention toward the region", "Prioritize candidate regions using caller attention and salience.", "Logit(confidence)+bias; softmax weights; multiplicative gain.", "Attentional gain and competitive selection across visual representations."),
    (2, "Detect local character-like structures", "Detect strokes, curves, junctions and connected ink components.", "Adaptive threshold; connected-component labels; area/aspect/centroid/moments.", "Intermediate visual populations encode contour, endpoint, curvature and junction evidence."),
    (2, "Combine local features into candidate letters", "Combine neighboring stroke parts into glyph hypotheses and compare supplied prototypes.", "Feature conjunction; Mahalanobis/RBF distance; softmax over supplied labels.", "Distributed ventral visual populations combine stroke and curve configurations."),
    (2, "Separate overlapping neighboring letters", "Retain alternate cuts for touching or crowded glyph candidates.", "Vertical ink projection; valley detection/prominence; segmentation posterior.", "Crowding-sensitive grouping and separation; multiple hypotheses preserve perceptual ambiguity."),
    (2, "Encode letter positions and order", "Encode normalized location and ordinal reading position.", "Relative x/y coordinates; ordinal rank; position-sensitive feature code.", "Retinotopic-to-abstract position coding along ventral visual pathways."),
    (2, "Increase tolerance to visual variation", "Normalize glyph shape for font/size changes without erasing identity or order.", "Scale normalization; shape moments; cosine similarity; view-consistency distance.", "Learned visual invariance supports tolerance to font, size and rendering variation."),
    (2, "Use linguistic feedback to resolve ambiguity", "Combine visual likelihood with explicit orthographic priors and retain alternatives.", "log P(image|letter)+log P(letter|context); softmax; entropy.", "Recurrent interactions between visual word-form and language representations."),
    (3, "Receive the acoustic waveform", "Validate and preserve the sampled sound pressure waveform and timing.", "x[n]; DC removal x-mean(x); time=n/fs; amplitude statistics.", "Outer/middle-ear transmission and cochlear input."),
    (3, "Perform cochlear frequency separation", "Separate the waveform into ERB-spaced tonotopic channels.", "ERB(f)=21.4log10(1+0.00437f); IIR bandpass filtering; convolution.", "Basilar-membrane tonotopy and frequency-selective hair-cell/auditory-nerve populations."),
    (3, "Convert mechanical vibration into neural signals", "Transform channel envelopes into bounded auditory-nerve rate proxies.", "Hilbert envelope; half-wave-like rectification; log compression; threshold/saturation.", "Hair-cell mechanoelectrical transduction, synaptic release and auditory-nerve firing."),
    (3, "Preserve temporal information", "Track envelopes, derivatives, onset events, durations and synchrony.", "Temporal smoothing; ΔE; onset peaks; inter-onset intervals; cross-channel coherence.", "Auditory neurons have diverse temporal integration and phase-locking properties."),
    (3, "Organize sound into perceptual streams", "Group components by envelope coherence, synchronous onset and harmonicity.", "Envelope correlation; onset synchrony; harmonic relation; graph grouping.", "Auditory scene analysis uses common onset, harmonicity and temporal continuity."),
    (3, "Increase priority of relevant auditory input", "Apply caller-directed attention and gain to candidate streams.", "Salience+attention bias; softmax; multiplicative gain.", "Attention modulates auditory cortical responses during selective listening."),
    (4, "Construct a time-frequency representation", "Represent changing spectral content using overlapping analysis windows.", "STFT X(τ,f); windowing; complex magnitude and phase.", "Cochlear and auditory-cortical time/frequency sensitivity; STFT is not a claimed cortical FFT."),
    (4, "Extract energy and spectral distribution", "Measure frame power, energy, low/high bands and spectral centroid.", "P=|X|²; band sums; RMS; C=ΣfP/ΣP.", "Auditory populations encode intensity and spectral distribution."),
    (4, "Detect harmonic structure and pitch", "Estimate periodicity and candidate fundamental frequency.", "Autocorrelation R(τ); normalized peak; F0≈fs/τ.", "Temporal periodicity and harmonic population coding in auditory pathways."),
    (4, "Extract formant structure", "Estimate vocal-tract resonances F1–F3 from spectral structure.", "LPC/autocorrelation; polynomial roots; resonance frequency and bandwidth.", "Vocal-tract resonances shape speech spectra and are represented by auditory populations."),
    (4, "Extract acoustic transitions", "Estimate spectral/formant change, onset timing and candidate VOT.", "First/second temporal derivatives; onset peaks; VOT interval.", "Auditory coding tracks changing acoustic cues and coarticulatory trajectories."),
    (4, "Build a combined acoustic feature vector", "Assemble a time-indexed vector without deciding phoneme identity.", "Concatenate F0, F1–F3, energy, centroid, periodicity, VOT and Δfeatures.", "Superior temporal auditory representations preserve multiple acoustic dimensions."),
    (5, "Detect voicing-related characteristics", "Combine periodicity, low-frequency energy and onset-to-voicing timing.", "Local autocorrelation; low-band ratio; sigmoid(wᵀx+b).", "Voicing-sensitive auditory populations respond to periodicity and temporal cues."),
    (5, "Detect manner of articulation", "Estimate stop, fricative, nasal, approximant, affricate and vowel evidence.", "Burst/frication/nasal/periodicity/closure vector; class logits; softmax.", "Heterogeneous STG populations encode acoustic-phonetic patterns in a distributed manner."),
    (5, "Detect place of articulation", "Infer broad place classes from spectral burst and formant-transition evidence.", "ΔF1, ΔF2, burst centroid; prototype distances; Bayesian posterior.", "Place is inferred from acoustic consequences rather than directly observed."),
    (5, "Detect vowel-related characteristics", "Estimate vowel height, backness and cautious rounding tendencies.", "F1/F2 formant-space coordinates; learned monotonic mappings.", "Vowel-sensitive representations use speaker- and language-dependent formant relationships."),
    (5, "Normalize phonetic features across speakers", "Normalize using caller baselines or robust within-sample statistics.", "(F-μspeaker)/(σspeaker+ε); median/MAD; bounded z-score.", "Context-sensitive speaker normalization is associated with auditory cortical processing including STG."),
    (5, "Detect temporal relationships and coarticulation", "Represent current, preceding, following and derivative feature states.", "qt=[at,at-1,Δat,at+1]; recurrent state h=σ(Wa+Wh+Wc).", "Speech gestures overlap; auditory populations integrate contextual trajectories."),
    (5, "Represent phonetic features as a distributed population pattern", "Combine voicing, manner, place, vowel and trajectory evidence.", "RBF population responses; feature concatenation; L2 normalization.", "Heterogeneous auditory/STG populations jointly encode acoustic-phonetic characteristics."),
    (6, "Activate candidate phoneme categories", "Activate multiple candidates from a language-specific inventory.", "zk=wkᵀr+bk; candidate softmax; parallel activation.", "Learned phoneme-sensitive populations can co-activate; inventory depends on language and listener."),
    (6, "Compare evidence with learned phoneme categories", "Combine acoustic likelihoods with learned category priors.", "P(φ|r)∝P(r|φ)P(φ); RBF/Gaussian similarity; log prior.", "Experience-dependent phonetic categories combine variable sensory realizations and expectations."),
    (6, "Integrate evidence over time", "Accumulate evidence while retaining competing phoneme hypotheses.", "Cumulative log posterior/log odds; temporal sum; normalization.", "Temporal integration stabilizes decisions from continuously evolving acoustic evidence."),
    (6, "Apply categorical perception", "Map continuous cues through learned category boundaries.", "P(φ1|x)=sigmoid(k(x-b)); complementary probability; temperature scaling.", "Category boundaries vary by language, task and listener; sigmoid is a psychophysical approximation."),
    (6, "Use phonetic and linguistic context", "Combine bottom-up evidence with explicit caller-supplied context.", "P(φ|A,C)∝P(A|φ,C)P(φ|C); log-prior addition; softmax.", "Top-down linguistic/phonetic context can bias auditory interpretation without replacing input."),
    (6, "Update interpretation through recurrent feedback", "Iteratively combine acoustic evidence, previous state and predictions.", "h(n+1)=σ(Waa+Wpp+Wrr); damped update; convergence norm.", "Recurrent auditory-language interactions support perceptual updating."),
    (6, "Select or retain the likely phoneme", "Select only above confidence/margin thresholds and preserve uncertainty.", "Argmax; entropy H=-ΣPlogP; top-two margin; abstention.", "Neural competition can settle or remain ambiguous; distributed activity need not force a hard choice."),
    (6, "Construct the phoneme sequence", "Merge adjacent frame estimates into ordered segments with timing and confidence.", "Frame-to-segment aggregation; onset/offset; duration; uncertainty.", "Speech is continuous and coarticulated; inferred boundaries are not exact biological cuts."),
)

OPERATION_REGISTRY = tuple(
    OperationSpec(
        operation_id=f"ISP-{index:03d}", stage=row[0], stage_name=STAGE_NAMES[row[0]],
        step=1 + sum(1 for previous in _ROWS[:index - 1] if previous[0] == row[0]),
        operation_name=row[1], objective=row[2],
        mathematical_operations=tuple(row[3].split("; ")),
        biological_reference=row[4],
    )
    for index, row in enumerate(_ROWS, start=1)
)

if len(OPERATION_REGISTRY) != 39:
    raise RuntimeError("sensory-perception registry must contain exactly 39 operations")
if tuple(item.operation_id for item in OPERATION_REGISTRY) != tuple(f"ISP-{i:03d}" for i in range(1, 40)):
    raise RuntimeError("operation IDs must be consecutive ISP-001 through ISP-039")


def registry_as_dicts() -> dict[str, Any]:
    """Return the six stage definitions and all 39 operation annotations as JSON-ready records."""
    return {
        "stage_step_counts": dict(STAGE_STEP_COUNTS),
        "stage_names": dict(STAGE_NAMES),
        "operations": [operation.to_dict() for operation in OPERATION_REGISTRY],
    }
