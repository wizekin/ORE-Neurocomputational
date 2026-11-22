# Language Comprehension in the Human Brain — Algorithmic Reconstruction

## Working document: Parts 1–13 of a sequential series

**Source document:** `Language Comprehension_ From Words to Contextual Meaning.txt` (Library file supplied by the user)  
**Coverage in this cumulative file:** Parts 1–8 develop the moment-to-moment comprehension pipeline, from sensory word-form processing through lexical access, parsing, semantic composition, contextual/pragmatic integration, working memory, memory/decision use, and recurrent cross-stage coordination. Parts 9–11 address neural architecture, evidence-based validation, and the functional distinction and cooperation between the core language network and other brain systems. Part 12 covers methodological considerations, scientific debates, and model selection. Part 13 models individual differences and developmental changes in comprehension.  
**Status:** Parts 1–13 complete in this version. Further source topics will be added only when the user requests continuation.

---

## How to read this reconstruction

This document translates the source's first brain-related processing point into a computationally explicit algorithm. It covers two alternative sensory routes:

1. **Auditory route:** acoustic speech → auditory features → candidate phonetic/phonological sequences and word-form hypotheses.
2. **Visual route:** printed or displayed text → visual features → ordered letter/orthographic sequences and word-form hypotheses.

The two routes are not identical, and they should not be forced into one artificial sensory pathway. They produce modality-specific representations that can be passed to later language operations.

### Important scientific distinction

The equations below are **candidate computational descriptions** of useful operations, not claims that individual neurons literally calculate the displayed formulas. Neuroscience supports many of the underlying functions — frequency-selective processing, nonlinear feature coding, temporal integration, categorization, normalization, distributed population activity, and transformations from retinal to more abstract letter-position codes. In many cases, however, the exact neural implementation remains under investigation.

Accordingly, every mathematical proposal should be interpreted at one of three levels:

- **Biophysical/peripheral operation:** a process with a relatively direct physical or physiological basis, such as cochlear filtering, membrane integration, synaptic excitation/inhibition, or spike generation.
- **Neural-computation model:** a mathematical abstraction of a plausible population-level operation, such as gain control, competition, temporal integration, or population coding.
- **Normative/algorithmic analogy:** a useful description of what the system must accomplish, such as Bayesian cue combination or dynamic-programming segmentation. It does not establish that the brain explicitly represents probabilities or runs a conventional computer algorithm.

The stages below are written in an understandable order, but real perception is **incremental, parallel, and recurrent**. Later information can influence the interpretation of earlier sensory material. The sequence is therefore a functional decomposition, not a claim of strictly serial processing.

---

# Point 1 — Perceptual processing: transform auditory or visual input into preliminary word forms

## 1. Objective and input/output contract

### Objective

Transform a continuous sensory stream into a time-ordered, uncertainty-aware representation of the linguistic forms that may have been heard or seen. The system should preserve enough information for subsequent lexical access to identify candidate words and retrieve their meanings.

### Input

- **Auditory input:** a time-varying acoustic pressure waveform containing speech, background sounds, and possibly multiple speakers.
- **Visual input:** a changing retinal image containing letters, punctuation, spaces, and visual noise, potentially affected by font, size, contrast, illumination, and position.
- **Modulatory input:** attention, expectations, current task, and recent sensory history. These can alter perceptual gain and interpretation before a complete sentence meaning has been assembled.

### Output

A set or sequence of **candidate linguistic-form representations**, each with:

- the modality from which it was derived;
- an estimated temporal or spatial position;
- acoustic-phonetic or visual-orthographic features;
- a confidence or activation measure;
- alternative candidates where the signal is ambiguous;
- links to the preceding and following elements in the sensory stream.

The output is not yet a complete semantic interpretation. In particular, this point does **not** perform the later operations of retrieving a word's meaning, building syntax, composing a proposition, or integrating that proposition with discourse context.

## 2. High-level algorithm

```text
ALGORITHM PERCEPTUAL_TO_WORD_FORM(input_stream, modality, attention_state):

    1. Detect and maintain the incoming sensory stream.
    2. Select the appropriate sensory pathway: auditory or visual.
    3. Extract elementary features using modality-specific receptive fields.
    4. Normalize nuisance variation while preserving linguistically useful differences.
    5. Encode local features in distributed neural populations.
    6. Integrate evidence over time (speech) or across visual locations (text).
    7. Detect candidate sub-word units and their order.
    8. Combine multiple cues according to their reliability.
    9. Generate and maintain competing phonetic or orthographic hypotheses.
   10. Estimate likely boundaries and preserve uncertainty about segmentation.
   11. Apply attention- and prediction-related modulation where relevant.
   12. Package the resulting form hypotheses for downstream lexical access.

    RETURN time_ordered_form_hypotheses
```

This is the shared outer algorithm. Steps 2–10 have different implementations in the auditory and visual branches.

---

## 3. Shared processing operations

### Step 1 — Capture the sensory stream and establish a running representation

**What the system must do**

The brain receives a continuous stream, not a pre-divided list of words. Speech unfolds over time; written text is sampled through eye movements and visual attention. The first operation is therefore to maintain a usable representation of incoming sensory evidence while new evidence continues to arrive.

**Possible neural implementation**

- Sensory receptors convert physical energy into neural signals.
- Early sensory pathways preserve spatial or temporal organization.
- Neural populations respond to overlapping portions of the stream; the representation of a recent segment can remain active after the external stimulus has changed.
- Attention can increase the effective influence of selected inputs and reduce the influence of competing inputs.

**Possible mathematical operations**

1. **Sampling:** represent the incoming signal as a sequence of measurements, for example `x[t]` for sound or `I[u,v,t]` for a visual image.
2. **Temporal/spatial indexing:** associate each measurement with a time or visual position so that order is not lost.
3. **Leaky integration:** retain recent evidence while allowing older evidence to decay:

   `m[t] = λ m[t−1] + (1−λ) x[t]`, where `0 < λ < 1`.

   This is a simplified model of persistence and integration, not a literal universal neural equation.
4. **Signal-to-noise estimation:** estimate whether the current input is reliable enough to influence later processing.

**Why it matters**

Without persistence and order information, a listener could not combine successive portions of a speech sound, and a reader could not determine the order of letters. Recent work describes speech representations in terms of persistent, parallel, and time-stamped coding properties; these are useful principles for modelling how transient input can support later structure building.

### Step 2 — Route input through the appropriate sensory pathway

The algorithm selects a pathway based on the available modality. If both modalities are present, both can operate in parallel and their outputs may later support one another.

- **Auditory:** cochlea → auditory nerve and brainstem pathways → thalamic auditory relay (medial geniculate body) → primary and nonprimary auditory cortex, including superior temporal regions.
- **Visual:** retina → lateral geniculate nucleus → primary visual cortex (V1) → extrastriate visual cortex → ventral occipitotemporal regions involved in visual object and orthographic processing.

This routing is not a single isolated chain: the pathways contain parallel projections, recurrent connections, and attention-related modulation.

**Possible mathematical operation**

A simple modality gate can be written as:

`z = g_A z_A + g_V z_V`,

where `z_A` and `z_V` are auditory and visual feature representations and `g_A`, `g_V` are task- and availability-dependent gains. This equation is only a convenient description of parallel evidence weighting; it does not imply that the brain first converts both modalities into identical signals.

---

# 4. Auditory branch — from speech acoustics to phonetic/phonological form hypotheses

## A1. Cochlear transduction and frequency-selective decomposition

**Purpose**

Convert air-pressure changes into neural activity and make the different frequency components of speech separately available to the auditory system.

**Biological operations**

1. The outer and middle ear transmit and mechanically transform sound.
2. The cochlea's basilar membrane responds differently along its length to different frequency ranges.
3. Hair cells convert mechanical movement into changes in electrical signalling.
4. Auditory nerve fibres transmit activity patterns to brainstem auditory structures and onward to the auditory thalamus and cortex.

The cochlea therefore performs an important frequency-analysis function before cortical speech processing. It is misleading to imagine that the cortex receives a single undifferentiated sound-pressure value.

**Possible mathematical model**

Represent the sound as `x(t)`. A bank of frequency-selective filters produces channel signals:

`a_k(t) = (h_k * x)(t)`,

where `h_k` is a filter tuned to frequency region `k`, and `*` denotes convolution.

An approximate compressive transform can be used to model the auditory system's broad dynamic-range handling:

`c_k(t) = log(ε + |a_k(t)|)`.

The exact filter shapes and compression law vary across levels of the auditory pathway. The filter-bank equation is a computational abstraction of frequency-selective mechanics and neural tuning, not proof that the brain calculates a digital convolution.

**Possible neural operations**

- frequency-selective mechanical resonance;
- receptor transduction;
- spike-rate changes with stimulus intensity;
- nonlinear compression;
- adaptation to ongoing stimulation.

**Output**

A time-varying set of activity patterns indexed by frequency channel.

## A2. Extract temporal envelopes, onsets, offsets, and acoustic change

**Purpose**

Speech identity depends not only on which frequencies are present but also on how energy changes over time. Stops, fricatives, vowels, voicing, syllable rhythm, and transitions all contain temporal evidence.

**Algorithmic operations**

1. Estimate the amplitude envelope within each frequency channel.
2. Detect rapid increases or decreases in channel activity.
3. Track the timing and duration of acoustic events.
4. Preserve relationships between frequency channels, since a speech cue may consist of a coordinated pattern rather than one isolated frequency.

**Possible mathematical model**

An envelope estimate may be represented as:

`e_k(t) = LPF(|a_k(t)|)`,

where `LPF` denotes low-pass smoothing. A simple onset signal is:

`o_k(t) = max(0, e_k(t) − e_k(t−Δt))`.

A corresponding offset detector can track sufficiently large negative changes. A spectrotemporal representation may then be assembled as a matrix whose rows represent frequency channels and columns represent successive time points.

**Neural interpretation**

Auditory neurons and cortical populations are sensitive to combinations of frequency and time. The superior temporal gyrus (STG) contains nonprimary auditory cortex and is important for phonological processing. Research describes its speech representations as nonlinear and dynamic, involving temporal structure, categorization, normalization, and contextual restoration.

**Output**

A spectrotemporal feature representation that retains both sound content and its evolution.

## A3. Extract acoustic-phonetic cues in parallel

**Purpose**

Estimate the distinctive properties that can support speech-sound categorization. Examples include:

- formant frequencies and their transitions;
- voice-onset time;
- voicing and periodicity;
- frication and spectral shape;
- burst timing;
- duration and amplitude;
- stress and prosodic contour.

These cues are not necessarily processed one at a time. Several feature populations can be active simultaneously, and different cues can support the same phonetic distinction.

**Algorithmic operations**

1. Construct a feature vector for a short speech interval:

   `f_t = [F1, F2, F3, VOT, voicing, frication, duration, ...]`.

2. Compare the observed feature pattern with learned phonetic patterns.
3. Maintain multiple plausible interpretations when cues are incomplete or noisy.
4. Preserve the temporal relationship between adjacent feature patterns; coarticulation means a sound is affected by its neighbours.

**Possible mathematical operations**

- feature extraction and dimensionality reduction;
- local temporal derivatives, such as `ΔF2(t) = F2(t) − F2(t−1)`;
- nonlinear feature combinations, such as `φ(W f_t + b)`;
- population coding, where the joint activity of many neurons represents a feature pattern;
- normalization against speaker- or context-dependent variation.

Here `W`, `b`, and `φ` describe a generic neural-computation model. They are not measured synaptic weights for a specific person.

**Output**

A sequence of acoustic-phonetic evidence vectors, each with uncertainty.

## A4. Normalize speaker, intensity, and channel variation

**Purpose**

The same linguistic sound can be produced by different speakers, at different pitches and volumes, with different accents, speaking rates, emotional states, microphones, and background noise. The perceptual system must reduce irrelevant variation without discarding meaningful distinctions.

**Algorithmic operations**

1. Estimate slowly changing properties of the current speaker or acoustic channel.
2. Compare local cues relative to these estimated properties.
3. Adapt sensitivity to the current signal range.
4. Preserve differences that distinguish one speech category from another.

**Possible mathematical operations**

A simplified normalization operation is:

`f'_i = (f_i − μ_i) / (σ_i + ε)`,

where `μ_i` and `σ_i` estimate a feature's recent mean and scale. A population-level gain-control model is:

`r'_i = r_i / (σ_0 + Σ_j w_j r_j)`,

where `r_i` is a neuron's or population's activity and the denominator represents divisive normalization by pooled activity.

**Biological interpretation**

Adaptation, inhibitory interactions, gain modulation, and normalization-like effects are plausible biological contributors. The equations are compact models of those effects, not an assertion that one cortical neuron performs explicit division.

**Output**

Acoustic features that are less sensitive to irrelevant speaker and channel differences.

## A5. Integrate evidence over time and preserve temporal order

**Purpose**

Speech sounds overlap. The beginning of a phoneme may be influenced by the preceding sound, and the identity of a sound may become clearer only after later acoustic information arrives. The system therefore needs to combine evidence across short time windows without losing the order of events.

**Algorithmic operations**

1. Maintain representations for multiple recent time points.
2. Link acoustic transitions that are likely to belong to the same evolving speech event.
3. Keep separate but interacting representations of acoustic, phonetic, syllabic, and prosodic structure.
4. Update an earlier hypothesis when later evidence clarifies it.

**Possible mathematical operations**

A leaky temporal accumulator for feature `i` can be written as:

`q_i(t) = α q_i(t−1) + β r_i(t)`,

where `r_i(t)` is current evidence, `α` controls persistence, and `β` controls the influence of new evidence.

A more explicit history representation is:

`H_t = [f_t, f_(t−1), ..., f_(t−K)]`.

A recurrent update can be represented generically as:

`h_t = φ(W_f f_t + W_h h_(t−1) + b)`.

These forms illustrate persistence and recurrence. The actual brain uses distributed neural activity, synaptic dynamics, and feedback pathways rather than a single software-style recurrent unit.

**Output**

A temporally extended representation that can support phonetic categorization and syllabic organization.

## A6. Categorize speech sounds while retaining alternatives

**Purpose**

Map variable acoustic patterns to linguistically relevant categories, such as candidate consonants, vowels, or syllabic units. Speech perception is not merely passive copying of sound: it requires categorization under considerable variability.

**Algorithmic operations**

1. Compare the current evidence with learned category patterns.
2. Compute or approximate the relative support for each candidate.
3. Preserve more than one candidate when evidence is ambiguous.
4. Update candidate support as additional acoustic information arrives.

**Possible mathematical model**

A Bayesian description of categorization is:

`P(c | f) ∝ P(f | c) P(c)`,

where `c` is a candidate speech category, `f` is the observed feature pattern, `P(f | c)` is the likelihood of that pattern under the category, and `P(c)` is its prior probability.

For a population-code interpretation, candidate support could be represented by a normalized activity pattern:

`p_i = exp(s_i) / Σ_j exp(s_j)`,

where `s_i` is a candidate's support score. This softmax is a convenient mathematical representation of graded competition; it does not show that neurons literally calculate exponentials or store explicit probability values.

**Neural interpretation**

The STG contains a spatial mosaic of neural populations that respond to different acoustic-phonetic and prosodic features. Their distributed activity can support increasingly abstract phonemic and syllabic representations. Categorization is nonlinear and dynamic rather than a simple one-to-one conversion from one acoustic feature to one phoneme.

**Output**

A time-ordered set of phonetic/phonological candidates with graded confidence.

## A7. Estimate syllable and word-boundary candidates

**Purpose**

Natural speech generally does not contain silent gaps between every word. The system must estimate where meaningful units begin and end using timing, stress, phonotactics, transitions, and other cues. Because the source point concerns conversion to word forms, this step produces candidate boundaries; it does not yet retrieve the words' meanings.

**Algorithmic operations**

1. Detect likely acoustic landmarks, including bursts, energy changes, and syllabic timing cues.
2. Track stress and prosodic grouping.
3. Evaluate whether adjacent phonetic candidates form plausible sequences.
4. Maintain competing segmentations if boundaries are uncertain.
5. revise earlier segmentation when later evidence changes the best interpretation.

**Possible mathematical operations**

A boundary detector may estimate:

`P(B_t = 1 | F_1:t)`,

where `B_t = 1` means a boundary is hypothesized at time `t` and `F_1:t` is the available feature history.

A candidate segmentation `π` can be scored as:

`Score(π) = Σ_segments log P(features_segment | unit) + Σ_transitions log P(next_unit | previous_unit)`.

A dynamic-programming or Viterbi procedure can identify a high-scoring sequence of candidate units. This is an **algorithmic analogy** for sequential constraint satisfaction; it is not evidence that the brain literally runs the Viterbi algorithm. Biological implementations may instead use recurrent competition, prediction, temporal integration, and distributed constraint satisfaction.

**Output**

A lattice or ranked list of possible phonetic/syllabic/word-form segmentations, with boundary uncertainty retained.

## A8. Apply attention and prediction-related modulation

**Purpose**

The sensory system does not give every signal equal influence. Attention can prioritize a speaker, a location, or a task-relevant stream. Expectations can also influence the processing of ambiguous input. This modulation can occur while sensory analysis is still underway.

**Algorithmic operations**

1. Estimate which stream or feature is relevant to the current task.
2. Increase the effective gain of task-relevant evidence.
3. reduce the influence of competing or irrelevant activity where possible.
4. compare incoming features with predictions, using mismatch to update the current representation.
5. avoid allowing a strong expectation to erase contradictory sensory evidence.

**Possible mathematical operations**

A simple gain model is:

`r_i^* = g_i r_i`,

where `g_i` is an attentional gain. A prediction-error model is:

`δ_t = observed_t − predicted_t`,

followed by an update such as:

`state_(t+1) = state_t + η δ_t`.

The equations express the computational ideas of selective gain and error-driven updating. They should not be taken as a complete or uniquely established account of predictive processing in language.

**Neural interpretation**

Attention-related modulation can affect activity in sensory cortex and interact with broader control systems. The detailed contribution of each control region depends on task demands; this first point does not require the full executive or social-reasoning networks discussed later in the source document.

**Output**

A revised set of form hypotheses weighted by current task relevance and sensory evidence.

---

# 5. Visual branch — from retinal image to ordered orthographic form hypotheses

## V1. Retinal encoding and early visual feature extraction

**Purpose**

Convert light reflected from text into neural signals that preserve the visual details needed to distinguish letters and their positions.

**Biological operations**

1. Photoreceptors in the retina transduce light into neural signals.
2. Retinal circuits adjust sensitivity to local contrast and illumination.
3. Retinal ganglion cells transmit spatially structured activity through the optic pathway and lateral geniculate nucleus.
4. V1 and subsequent visual areas respond to local properties such as contrast boundaries, orientation, spatial frequency, and position.

**Possible mathematical operations**

For image `I(u,v)`, a simple feature detector can be represented as:

`z_k(u,v) = φ((I * G_k)(u,v))`,

where `G_k` is a filter sensitive to a particular orientation or spatial pattern and `φ` is a nonlinear response function. Local contrast normalization can be represented as:

`z'_k = z_k / (ε + local_activity)`.

These operations approximate receptive-field filtering, nonlinear response, and normalization. The brain is not a conventional convolutional neural network, even though some visual computations can be modelled with related mathematics.

**Output**

A retinotopically organized set of visual features. At early visual stages, the exact retinal location of a feature remains important.

## V2. Detect candidate letter features

**Purpose**

Extract the visual components that distinguish letters: lines, curves, junctions, endpoints, enclosed regions, and their spatial relationships.

**Algorithmic operations**

1. Detect local edges and orientations.
2. combine local components into candidate strokes and shapes.
3. compare the resulting patterns with learned visual letter features.
4. preserve uncertainty for visually similar letters or degraded images.

**Possible mathematical operations**

- oriented filtering;
- thresholding and nonlinear feature activation;
- pooling over small local regions;
- similarity scoring, for example `s(ℓ) = −||z − μ_ℓ||²`, where `μ_ℓ` is a learned feature prototype for candidate letter `ℓ`;
- competition between candidate letter populations.

**Output**

A set of candidate letter identities at approximate visual locations.

## V3. Transform retinal positions into ordered, more abstract letter positions

**Purpose**

A reader must recognize a letter string despite changes in font, size, spacing, or where the string appears on the retina. At the same time, the order of letters must be preserved: changing the order can change the word.

**Biological interpretation**

Early visual areas predominantly represent features in relation to retinal location. Along the ventral visual pathway, representations become more appropriate for identifying letters and their order as part of a written string. The left ventral occipitotemporal cortex, including regions commonly discussed as the visual word form area (VWFA), is strongly implicated in orthographic processing. Recent work indicates a shift from retinotopic coding toward more abstract ordinal letter-position coding in higher ventral visual regions. The transformation is graded, not an instantaneous switch from one code to another.

**Algorithmic operations**

1. Estimate the location and identity of each candidate letter.
2. determine the relative order of letters within the attended string.
3. encode position in a way that is less dependent on the exact retinal coordinate.
4. retain order-sensitive distinctions so that strings such as `FORM` and `FROM` are not treated as identical.
5. preserve a residual link to the original visual locations so the system can return to the input if uncertainty remains.

**Possible mathematical operations**

Let `ℓ_i` be a letter candidate and `p_i` its relative position in the string. Form an ordered representation:

`O = [(ℓ_1,p_1), (ℓ_2,p_2), ..., (ℓ_n,p_n)]`.

To represent tolerance to small changes in retinal position, one can pool evidence across positions with a weighted function:

`R(ℓ,p) = Σ_(u,v) K_p(u,v) A_ℓ(u,v)`,

where `A_ℓ(u,v)` is evidence for letter `ℓ` at retinal location `(u,v)` and `K_p` maps retinal locations to a relative string position. The weights must preserve ordinal order while reducing sensitivity to irrelevant shifts.

A candidate string can be scored as:

`Score(S) = Σ_i s(ℓ_i,p_i) + Σ_i T(ℓ_i,ℓ_(i+1))`,

where `T` captures learned compatibility between neighbouring letters. This is a computational model of ordered orthographic coding, not an identified neural formula.

**Output**

A position-sensitive but increasingly location-tolerant ordered letter representation.

## V4. Combine letters into orthographic patterns

**Purpose**

Represent common letter combinations and whole written forms. A familiar string may be supported by both individual-letter evidence and learned regularities in how letters occur together.

**Algorithmic operations**

1. Combine letter identity and ordinal position.
2. detect recurring multi-letter patterns.
3. compare the observed string with stored orthographic regularities.
4. keep close alternatives alive when the visual input is incomplete or ambiguous.
5. preserve the exact sequence, rather than reducing the string to an unordered bag of letters.

**Possible mathematical operations**

- local n-gram feature extraction;
- sequence scoring;
- recurrent combination of neighbouring letter features;
- graded activation of orthographic candidates;
- competition and normalization among candidate strings.

For example, a candidate sequence `S = (ℓ_1,...,ℓ_n)` can receive a score:

`Score(S) = Σ_i log P(ℓ_i | visual_i) + Σ_i log P(ℓ_i | preceding_letters)`.

The second term expresses sensitivity to learned orthographic regularities. It should not be confused with semantic knowledge or full word-meaning retrieval.

**Output**

Candidate ordered orthographic forms, with confidence and alternatives.

## V5. Control visual sampling and update uncertain regions

**Purpose**

Reading involves eye movements, shifts of attention, and unequal visual resolution across the field. The system may need to sample a region again when a letter or boundary is uncertain.

**Algorithmic operations**

1. Estimate the confidence of the current orthographic representation.
2. detect locations or letters with high uncertainty.
3. allocate attention or a new fixation when useful.
4. combine new evidence with the existing representation.

**Possible mathematical operations**

Uncertainty can be represented by entropy:

`H(L) = −Σ_i P(ℓ_i) log P(ℓ_i)`.

A high value indicates that several letter candidates remain plausible. A simplified sampling policy could prioritize a location with high uncertainty and high expected information gain. This is a normative description of an active-perception strategy, not a claim that the brain explicitly calculates entropy before every eye movement.

**Output**

A refined visual word-form hypothesis, or an explicit uncertainty signal requesting further sampling.

---

# 6. Convergence and handoff to the next language operation

## Step 13 — Package modality-specific form hypotheses

**Purpose**

Create a clean interface between sensory-form analysis and the next stage of language comprehension. Auditory and visual processing should remain distinguishable while making their results available to a shared downstream language system.

**Algorithmic operations**

1. Collect the current auditory phonetic/phonological hypotheses, if present.
2. collect the current visual orthographic hypotheses, if present.
3. preserve order, timing, confidence, and alternative candidates.
4. preserve links back to the supporting sensory evidence.
5. pass the representations forward without adding a word meaning that has not yet been retrieved.

**Possible data structure**

```text
FORM_HYPOTHESIS:
    modality: AUDITORY | VISUAL
    sequence: ordered list of candidate units
    onset_or_position: sensory location / time
    confidence: graded support value
    alternatives: competing form hypotheses
    feature_trace: acoustic or visual evidence summary
    uncertainty: unresolved ambiguity
```

**Possible mathematical operations**

- ranking of candidate hypotheses;
- normalization of candidate support within a local competition set;
- alignment of auditory and visual candidates when both are present;
- uncertainty propagation to downstream processing.

A candidate set can be represented abstractly as:

`H = {(s_i, c_i, τ_i, u_i)}`, 

where `s_i` is a candidate form, `c_i` is its support, `τ_i` is its time or position, and `u_i` records uncertainty. The representation should not force a single answer when the evidence does not support one.

## 7. Completion criteria for Point 1

Point 1 is considered complete when the system can:

- transform sound into a structured auditory feature stream;
- transform visible text into an ordered visual feature stream;
- preserve temporal or letter order;
- build candidate phonetic or orthographic forms;
- handle noise and irrelevant variation to a useful degree;
- retain uncertainty and competing hypotheses;
- produce a representation that can be passed to lexical access;
- avoid confusing sensory-form identification with semantic retrieval.

## 8. What this point does not do

To preserve the source's sequence, this algorithm deliberately does **not** implement:

- retrieval of a word's stored meaning;
- syntactic parsing or grammatical-role assignment;
- combination of word meanings into a proposition;
- integration with prior discourse or the physical situation;
- inference about speaker intention, sarcasm, implicature, or unstated meaning;
- maintenance of a complete propositional representation in working memory;
- long-term storage or decision-making based on sentence meaning.

Those belong to later points in the source and will be treated separately.

## 9. Biological caution and open questions

1. The exact boundary between acoustic, phonetic, syllabic, and lexical processing is debated. Evidence suggests that representations can coexist and interact rather than appearing as isolated layers in a rigid hierarchy.
2. The superior temporal cortex contains distributed, nonlinear, and dynamic speech-sensitive populations. It is safer to model population activity than to assign one complete computation to one small anatomical location.
3. Visual word recognition depends on a ventral visual pathway and left ventral occipitotemporal regions, but the precise organization varies across individuals and tasks.
4. Mathematical tools such as Bayesian inference, convolution, softmax, and dynamic programming are useful ways to state computational problems. They should not be mistaken for directly observed neural instructions.
5. Attention and prediction may influence sensory processing early. The algorithm's staged layout is an explanatory decomposition; it does not imply that top-down effects wait until sensory analysis is finished.

---

## 10. References for Point 1

The following sources support the neural and computational framing used in this installment. They supplement — and do not replace — the source document supplied by the user.

1. **Speech Computations of the Human Superior Temporal Gyrus.** *Annual Review of Psychology* (2022 issue; published online 2021). Review of nonlinear and dynamic speech representations in STG, including categorization, normalization, temporal structure, and contextual restoration. [https://doi.org/10.1146/annurev-psych-022321-035256](https://doi.org/10.1146/annurev-psych-022321-035256)
2. **Computational Architecture of Speech Comprehension in the Human Brain.** *Annual Review of Linguistics*. Discusses representational levels and the operations used to transform acoustic input into increasingly abstract linguistic representations. [https://doi.org/10.1146/annurev-linguistics-031120-111245](https://doi.org/10.1146/annurev-linguistics-031120-111245)
3. **From retinotopic to ordinal coding: Dissecting the cortical stages of visual word recognition.** *Proceedings of the National Academy of Sciences* (2025). Reports a graded transformation from retinally anchored letter coding in early visual cortex toward more abstract letter-order representations in higher ventral visual regions, including the visual word form area. [https://doi.org/10.1073/pnas.2507291122](https://doi.org/10.1073/pnas.2507291122)
4. **Spatiotemporal Dynamics of Orthographic and Lexical Processing in the Ventral Visual Pathway.** *Nature Human Behaviour* (2021). Intracranial recordings describe the timing and distribution of orthographic and lexical sensitivity in left ventral occipitotemporal cortex. [https://doi.org/10.1038/s41562-020-00982-w](https://doi.org/10.1038/s41562-020-00982-w)
5. **Intracranial dissection of word reading mechanisms.** (2016). Reviews and examines the role of ventral occipitotemporal cortex in identifying letters and their order during reading. [https://pmc.ncbi.nlm.nih.gov/articles/PMC4961167/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4961167/)
6. **Time after time: neural codes of speech comprehension.** *Trends in Cognitive Sciences* (published online 5 October 2026). Reviews persistent, parallel, and time-stamped neural coding as properties supporting the transformation of transient speech into structured representations. [https://doi.org/10.1016/j.tics.2026.09.003](https://doi.org/10.1016/j.tics.2026.09.003)

---

**End of Part 1.** The next part should begin with the next distinct processing point in the source sequence: **lexical access — retrieving word meanings from the recognized word forms**. It has not been added to this file yet.


# Point 2 — Lexical access: retrieve word-level lexical entries and meanings from recognized word forms

## 1. Position of this point in the source sequence

The source describes the next operation after perceptual processing as retrieval of word meanings. The output of Point 1 is therefore the input to this algorithm: a spoken or written form hypothesis, possibly with more than one candidate and with a confidence value. Point 2 attempts to activate the corresponding lexical entry or entries and retrieve the word-level knowledge attached to them.

This is not yet full syntactic parsing, sentence-level semantic composition, or broad discourse/pragmatic interpretation. A lexical entry can make information available to those later operations — for example, a word's grammatical category, argument-taking properties, morphological features, and semantic features — without Point 2 deciding how the word functions in the complete sentence.

Lexical access is not necessarily a single lookup after a word has been fully recognized. It can begin before the complete word is heard or seen, proceed incrementally, activate several competitors, and be revised as new form information arrives. The auditory and visual routes have different input codes, but converge on overlapping lexical-semantic knowledge. The algorithm keeps those routes separate until their outputs can be compared or combined.

## 2. Biological and computational scope

### Main functions to implement

1. Receive a graded spoken or written word-form hypothesis from Point 1.
2. Search or activate candidate lexical representations compatible with that form.
3. Combine form evidence with lexical familiarity, frequency, and available local expectations.
4. Allow competing candidates to remain active while the sensory evidence is incomplete or ambiguous.
5. Retrieve word-level semantic features and associated lexical information.
6. Handle morphological forms, inflections, compounds, and multi-morpheme words where the language requires them.
7. Distinguish a word's identity from its possible senses when the word is ambiguous or polysemous.
8. Deliver a ranked, confidence-weighted lexical-semantic representation to syntactic parsing and later contextual integration.

### Likely neural systems

The evidence points to a distributed network rather than a single “dictionary area.” The functional emphasis differs by modality and processing demand:

- **Auditory route:** superior temporal regions analyse and represent speech sound patterns. Left middle superior temporal gyrus is consistently implicated in short-timescale phonetic processing, while left anterior superior temporal regions are associated with integration of phonemes into word-like patterns. The auditory ventral pathway links sound patterns to lexical-semantic knowledge. The anterior superior temporal gyrus has also been reported to contain word-selective auditory representations.
- **Visual route:** left ventral occipitotemporal cortex, including word-selective mid-fusiform/visual-word-form regions, represents familiar orthographic patterns. These regions are connected to perisylvian language areas involved in phonological conversion and lexical access. Recent intracranial work indicates that lexicality and word-frequency information can be detectable in mid-fusiform cortex and propagate through the ventral visual pathway; this makes the visual lexical system more dynamic than a simple one-way ladder of visual features.
- **Lexical-semantic access:** middle and anterior temporal regions, posterior temporal cortex, and temporoparietal areas participate in mapping word forms to lexical and conceptual information. The exact contribution of individual temporal and parietal subdivisions varies by task and theory.
- **Controlled retrieval and selection:** left inferior frontal gyrus can contribute when several candidates compete, when a weakly activated word must be retrieved, or when irrelevant alternatives must be suppressed. It should not be treated as a mandatory gate that must consciously approve every familiar word.
- **Phonological interface:** supramarginal and other inferior parietal regions contribute to phonological representations and their links with articulatory or short-term maintenance systems. Their exact contribution depends on whether the task involves listening, repetition, naming, or retention.
- **Cross-modal conceptual knowledge:** conceptual information is supported by distributed temporal and modality-specific association systems. Anterior temporal regions are often proposed to help integrate information across different sources, but the “hub” account and the precise anatomical boundaries remain active areas of research.

These are functional emphases, not exclusive assignments. A brain region may contribute to more than one operation, and the same lexical task can recruit different proportions of the network depending on language, word type, familiarity, attention, and difficulty.

## 3. Algorithmic data structures

The following structures are useful for making the algorithm explicit. They are proposed engineering abstractions; they are not claims that the brain stores literal software records.

```text
INPUT_FORM_HYPOTHESIS:
    modality                  # AUDITORY or VISUAL
    ordered_form_features     # phonological or orthographic evidence
    candidate_boundaries      # possible onset/offset or letter-span boundaries
    sensory_confidence        # graded quality of the form evidence
    alternative_forms         # competing word-form hypotheses
    timing_or_position        # when or where the form occurred
    local_context_summary     # limited preceding linguistic/situational cues, if available

LEXICAL_CANDIDATE:
    lexical_id
    canonical_word_form
    modality_specific_form    # spoken and/or written form representation
    frequency_prior
    familiarity_strength
    morphological_analysis
    lexical_category          # noun, verb, adjective, etc.; a stored property only
    grammatical_features      # e.g., number, gender, tense-related form features
    semantic_feature_pattern
    possible_senses
    activation
    evidence_score
    uncertainty

LEXICAL_OUTPUT:
    ranked_candidates
    selected_or_unresolved_entry
    word_level_semantic_features
    lexical_properties
    sense_candidates
    confidence
    unresolved_ambiguities
```

A lexical candidate is not merely a text string. It stands for a stored or learned relation among a word's form, its conventional lexical information, its possible senses, and its links to other knowledge. In a biological system, these relations are distributed across populations of connected neurons rather than necessarily held in one discrete cell or location.

## 4. Detailed algorithmic sequence

### Step 1 — Accept the incoming form evidence without forcing a final word decision

**Purpose:** receive Point 1's sensory representation while preserving the difference between “the form resembles a word” and “the word's meaning has been retrieved.”

1. Read the current auditory or visual form hypothesis.
2. Identify whether it contains an ordered phonological sequence, an ordered orthographic sequence, or multiple possible sequences.
3. Preserve the original evidence and its uncertainty. Do not convert a weakly heard consonant or an unclear letter into a certain one merely to simplify the lookup.
4. If the input is continuous speech, keep candidate word boundaries open when the acoustic signal does not establish them clearly. Lexical candidates can later provide feedback that helps revise segmentation.
5. If the input is text, retain letter order, script-specific position information, and possible segmentation uncertainty.

**Possible mathematical operations**

- **Feature-vector representation:** encode the input as a vector \(\mathbf{x}\), where dimensions represent relevant phonological or orthographic features. The vector may be sparse, distributed, or population-coded.
- **Confidence weighting:** scale each feature by its reliability, such as \(\tilde{x}_k = r_k x_k\), with \(r_k\) representing the reliability of feature \(k\). This is an abstract model of the fact that some parts of the signal are clearer than others.
- **Uncertainty preservation:** carry a covariance or uncertainty descriptor \(\Sigma_x\) alongside the feature vector, or simply retain a set of alternatives with confidence weights.

**Possible neural implementation:** graded activity in sensory and language-responsive populations, together with recurrent signalling from higher-level candidate representations to temporal sensory-language areas. The uncertainty representation is a modelling convenience; biological circuits need not encode a formal covariance matrix.

### Step 2 — Select the appropriate modality-specific lexical access route

**Purpose:** use the correct entry pathway while allowing spoken and written words to converge on shared lexical-semantic knowledge.

1. For auditory input, compare the phonological pattern with learned spoken-word representations. Relevant evidence includes phoneme order, syllabic structure, stress, duration, and other language-specific cues.
2. For visual input, compare the orthographic sequence with learned written-word patterns. Relevant evidence includes letter identity, letter order, position, familiar letter combinations, and script-specific conventions.
3. Do not assume that the same neural code is used for a spoken word and its written counterpart. The sensory and word-form representations differ even when the lexical item is the same.
4. Allow either route to activate the same conceptual or lexical-semantic knowledge once a candidate is sufficiently supported.
5. If both spoken and written evidence are available (for example, a person hears a word while seeing it), compare their candidate sets without assuming they are perfectly identical.

**Possible mathematical operations**

- **Modality-specific transformations:** \(\mathbf{z}_a=f_a(\mathbf{x}_a)\) for auditory evidence and \(\mathbf{z}_v=f_v(\mathbf{x}_v)\) for visual evidence. The transformations are different because their input features are different.
- **Convergence to a shared lexical space:** a learned mapping can be written as \(\mathbf{h}_a=W_a\mathbf{z}_a\) and \(\mathbf{h}_v=W_v\mathbf{z}_v\), with the two outputs activating overlapping lexical-semantic patterns. The matrices are an algorithmic description of learned connectivity, not literal anatomical matrices.
- **Cross-modal agreement:** when both modalities are present, a candidate's support may increase with agreement between the two routes. One abstract agreement measure is cosine similarity, \(\mathrm{sim}(\mathbf{u},\mathbf{v})=(\mathbf{u}\cdot\mathbf{v})/(\|\mathbf{u}\|\|\mathbf{v}\|)\). It should be used only when the representations have been placed in comparable spaces.

**Possible neural implementation:** modality-specific auditory and visual pathways project into interconnected language and conceptual systems. The visual-word-form region's anatomical connections with perisylvian areas provide a plausible route from orthography toward phonological and lexical processing. The auditory ventral stream provides a complementary route from speech sounds toward word identity and meaning.

### Step 3 — Generate an initial candidate set from form similarity

**Purpose:** activate multiple possible lexical entries whose stored forms resemble the incoming evidence.

1. Search the learned lexical network for entries with compatible phonological or orthographic patterns.
2. Generate candidates from partial matches as well as exact matches. This is essential when a spoken word is still unfolding, when speech is noisy, or when text is degraded.
3. Retain candidates that differ in a single uncertain feature if the sensory evidence does not yet distinguish them.
4. Avoid requiring a completed sequence before candidate activation begins. In natural speech, recognition can be incremental; in reading, lexical and sublexical information can interact while the eyes move through a word.
5. Update candidates as each new phoneme, syllabic cue, letter, or letter cluster becomes available.

**Possible mathematical operations**

- **Similarity scoring:** for candidate \(w_i\), compare the input vector \(\mathbf{x}\) with the candidate's stored form vector \(\boldsymbol{\phi}_i\). A simple score is \(m_i=\mathbf{x}^{\mathsf T}\boldsymbol{\phi}_i\). More realistic models use weighted feature matches and penalties for order violations.
- **Distance-based matching:** use a distance such as \(d_i=\|\mathbf{x}-\boldsymbol{\phi}_i\|\), or a sequence-edit distance for discrete form strings. Smaller distance means greater similarity, but no single distance metric is known to be the brain's literal operation.
- **Position-sensitive matching:** weight evidence by its sequence position, for example \(m_i=\sum_k r_k\,s(x_k,\phi_{ik})\), where \(s\) measures feature compatibility and \(r_k\) reflects reliability.
- **Candidate-set expansion:** add a candidate when its score exceeds a permissive activation threshold. A low threshold allows early activation but also creates more competitors.

**Possible neural implementation:** overlapping neural populations respond to shared phonological or orthographic features. A candidate can receive partial activation before all of its features have arrived. The progressive change in activation is more biologically plausible than a single exact-match query.

### Step 4 — Add lexical familiarity and word-frequency influences

**Purpose:** account for the fact that familiar and frequently encountered words are generally accessed more efficiently than rare or unfamiliar words, all else being equal.

1. For each form-compatible candidate, retrieve its learned baseline activation or familiarity strength.
2. Give frequently experienced lexical entries a higher initial readiness, without allowing frequency to override strong contradictory sensory evidence.
3. Allow personal experience to influence the strength of a candidate. A person's occupation, interests, dialect, reading history, and recent exposure can change how familiar a word is to that individual.
4. Keep frequency and contextual predictability conceptually separate. A common word can be unexpected in a particular sentence; a rare word can be highly predictable in a specialist context.
5. Update readiness through learning and repeated exposure over time, rather than treating frequency as an immutable property.

**Possible mathematical operations**

- **Log-frequency prior:** \(b_i=\alpha\log(1+f_i)\), where \(f_i\) is an exposure-frequency estimate and \(\alpha\) determines its influence. The logarithm represents diminishing gains: moving from very rare to moderately common can have a larger effect than an equal absolute increase among already frequent words.
- **Experience-dependent strength:** \(q_i(t+1)=q_i(t)+\eta\,\Delta_i\), where \(q_i\) is learned candidate readiness, \(\eta\) is a learning rate, and \(\Delta_i\) reflects experience or prediction error. This is a generic learning rule, not a claim that lexical frequency is stored in one scalar.
- **Decay or recency:** a short-term activation trace can be represented as \(r_i(t+1)=\lambda r_i(t)+u_i(t)\), with \(0<\lambda<1\). This models temporary facilitation from recent exposure, distinct from long-term familiarity.

**Possible neural implementation:** repeated co-activation can strengthen synaptic connections and make a learned pattern easier to reactivate. Frequency effects are observed in word recognition, but their neural implementation likely involves distributed changes in connectivity, tuning, and network dynamics rather than one frequency counter.

### Step 5 — Incorporate only the local expectations that are already available

**Purpose:** allow early predictions and immediate context to bias lexical access without prematurely performing the full contextual/pragmatic integration assigned to a later point.

1. Read a compact representation of already-heard or already-read nearby material, if available.
2. Estimate which word forms are compatible with the immediately preceding linguistic material, the current topic cue, or a strongly established perceptual situation.
3. Use this information as a bias, not as a substitute for the incoming form evidence.
4. Maintain the distinction between lexical likelihood and contextual expectation. The same word can be highly familiar but contextually improbable, or rare but contextually expected.
5. Leave wider discourse coherence, speaker intention, sarcasm, implicature, and causal bridging for the later contextual/pragmatic algorithm.

**Possible mathematical operations**

- **Bayesian-style combination:** \(P(w_i\mid x,c)\propto P(x\mid w_i)P(w_i\mid c)\), where \(x\) is the current form evidence and \(c\) is the limited context available at that moment. This is a normative model of cue combination; it does not establish that the brain explicitly computes Bayes' rule.
- **Weighted log-score:** \(S_i=\alpha M_i+\beta F_i+\gamma C_i\), where \(M_i\) is form match, \(F_i\) is frequency/familiarity, and \(C_i\) is local-context support. The coefficients vary with task, attention, and reliability.
- **Reliability-dependent weighting:** when the acoustic signal is noisy, context may receive more relative weight; when the sensory form is clear, form evidence can dominate. One compact abstraction is \(\gamma=\gamma_0(1-r_x)\), where \(r_x\) is sensory reliability. This relation is a modelling proposal, not a measured universal neural formula.

**Possible neural implementation:** top-down feedback from temporal and frontal language regions can bias sensory and lexical populations. Predictive effects may arise before full semantic integration. The model must avoid treating context as an all-powerful correction mechanism: a compelling context should not force recognition of a word that is incompatible with the actual signal.

### Step 6 — Activate morphological analyses when the word's structure supports them

**Purpose:** retrieve the lexical information for words built from meaningful parts, while preserving whole-word recognition where that is more efficient.

1. Test whether the input is compatible with a stored whole-word representation.
2. In parallel, consider plausible morpheme boundaries where the language uses productive roots, prefixes, suffixes, inflections, or compounds.
3. Retrieve candidate stems or roots and the possible contribution of affixes or other bound morphemes.
4. Combine the candidate parts to construct a word-level lexical interpretation.
5. Compare the composed analysis with stored whole-word knowledge. Irregular forms, highly familiar words, and opaque compounds may not be transparently decomposable.
6. Retain more than one analysis when the same string admits multiple morphological segmentations.

**Possible mathematical operations**

- **Segmentation search:** for an input sequence of length \(n\), generate plausible partitions \(s_1,\ldots,s_k\) whose concatenation or morphophonological transformation can produce the observed form.
- **Posterior scoring:** \(P(s_1,\ldots,s_k\mid x)\propto P(x\mid s_1,\ldots,s_k)P(s_1,\ldots,s_k)\). The prior favours possible morpheme sequences in the language; the likelihood measures how well they explain the observed form.
- **Compositional feature combination:** if a stem activates semantic vector \(\mathbf{v}_{stem}\) and an affix activates an operation or feature vector \(\mathbf{v}_{affix}\), a simplified model is \(\mathbf{v}_{word}=g(\mathbf{v}_{stem},\mathbf{v}_{affix})\). The function \(g\) need not be simple addition; inflection and derivation can modify different dimensions.
- **Whole-word versus decomposition competition:** compare a whole-form score with a decomposed-form score, while allowing both representations to contribute. A strict either/or switch is not required.

**Possible neural implementation:** recurring patterns can be learned as distributed form-to-meaning associations. Morphological families may share activation across related forms. The exact balance between whole-word storage and decomposition differs across languages, word types, proficiency, and learning history. Morphological operations should therefore be language-specific rather than imposed as one universal rule.

### Step 7 — Resolve form-level competition gradually, not by an immediate winner-take-all decision

**Purpose:** allow the most supported candidate to gain strength while weaker alternatives lose influence, with enough flexibility to recover from early ambiguity.

1. Collect bottom-up form-match evidence, lexical familiarity, local expectations, and morphological support for each candidate.
2. Let candidates that share features remain co-active. This is especially important early in a spoken word, when several words may begin with the same sounds.
3. Increase the activation of candidates that continue to match incoming evidence.
4. Reduce activation of candidates that are contradicted by later features or that are strongly outcompeted by a better-supported alternative.
5. When multiple candidates remain plausible, preserve ambiguity rather than forcing an arbitrary selection.
6. If a later phoneme or letter changes the balance, update the ranking and allow the previously leading candidate to lose.

**Possible mathematical operations**

- **Excitation and lateral inhibition:** \(a_i(t+1)=\rho a_i(t)+E_i(t)-\sum_{j\ne i}\kappa_{ij}a_j(t)+b_i\). Here \(a_i\) is candidate activation, \(\rho\) is persistence, \(E_i\) is current evidence, \(\kappa_{ij}\) is competition strength, and \(b_i\) includes learned bias. This is an abstract recurrent network equation.
- **Nonlinear activation:** \(a_i=\sigma(z_i)=1/(1+e^{-z_i})\), or another saturating function, prevents a candidate's activation from growing without bound.
- **Soft competition distribution:** \(p_i=\exp(S_i/T)/\sum_j\exp(S_j/T)\). Lower temperature \(T\) sharpens the distribution; higher temperature keeps several candidates active. This is useful for a computational model but is not evidence of a literal neuronal softmax calculation.
- **Evidence accumulation:** \(A_i(t)=A_i(t-1)+\delta_i(t)\), where \(\delta_i(t)\) is the new evidence for candidate \(i\). A candidate can accumulate support across successive time windows.
- **Entropy as uncertainty measure:** \(H=-\sum_i p_i\log p_i\). High entropy indicates that several candidates remain plausible; low entropy indicates that one or a few candidates dominate. This can be used by a model to decide whether to wait for more input.

**Possible neural implementation:** recurrent excitation, inhibitory interactions, adaptation, and changing sensory evidence can produce gradual competition. Left inferior frontal and temporal networks may contribute when selection is difficult. The competition model should not imply that every word requires effortful executive control; high-familiarity items may settle rapidly with little additional control demand.

### Step 8 — Retrieve the semantic features associated with each active lexical candidate

**Purpose:** move from a recognized form to the word's conventional word-level meaning, without yet combining it with other words into a proposition.

1. For each sufficiently active candidate, activate its associated semantic and conceptual knowledge.
2. Retrieve a distributed pattern of features rather than only a dictionary-style verbal definition.
3. Include properties relevant to the word's conventional meaning: category, typical attributes, actions, relations, functions, and associations.
4. Allow different knowledge sources to contribute. For a concrete noun, visual, tactile, functional, and action-related associations may all be relevant. For an abstract word, social, emotional, relational, or linguistic knowledge may be more important.
5. Keep word-level semantic features separate from the roles that the word will take in the upcoming sentence.
6. Preserve weaker or less typical features as graded activation rather than converting every association into a defining property.

**Possible mathematical operations**

- **Associative spread:** if lexical node \(i\) is active, connected semantic nodes can receive activation \(\mathbf{y}=W\mathbf{a}+\mathbf{b}\), where \(W\) represents learned association strengths. This is a graph or population-level abstraction.
- **Nonlinear feature activation:** \(y_k=\sigma(\sum_i W_{ki}a_i+b_k)\). The output is a graded activation pattern across semantic dimensions.
- **Distributed semantic code:** represent the candidate meaning as \(\mathbf{v}_w=(v_1,\ldots,v_d)\), where components correspond to semantic features or latent dimensions. Similar concepts can have overlapping patterns without being identical.
- **Sparse or thresholded coding:** weakly activated features may be suppressed or down-weighted to reduce noise: \(\hat{y}_k=y_k\,\mathbb{1}[y_k>\theta]\), or with a smooth threshold. Biological activity is not necessarily binary; this is a possible computational simplification.
- **Feature reliability weighting:** weight semantic features by how consistently they are associated with the concept, \(\tilde{y}_k=c_k y_k\), where \(c_k\) represents feature reliability or learned diagnosticity.

**Possible neural implementation:** semantic knowledge is distributed across temporal and modality-linked association systems. A word may activate auditory, visual, motor, affective, or social knowledge in differing proportions. Temporal regions help link lexical forms with conceptual representations, while distributed cortical networks contribute content-specific features.

### Step 9 — Retrieve lexical properties needed by subsequent syntactic operations

**Purpose:** make stored grammatical information available without carrying out the actual sentence parse yet.

1. Retrieve lexical category information where it is part of the entry (for example, noun-like, verb-like, adjective-like, or function-word properties).
2. Retrieve morphological and grammatical features that are expressed in the form, such as number, person, tense, aspect, gender, case, or agreement class where relevant to the language.
3. For verbs and relational words, make their conventional argument-taking or complement properties available as lexical information. Do not assign actual sentence participants to those roles at this stage.
4. Keep uncertainty where a form has more than one lexical category or grammatical use.
5. Pass these properties to the syntactic parser as candidate constraints, not as an already completed grammatical structure.

**Possible mathematical operations**

- **Attribute activation:** each candidate can activate a feature vector \(\mathbf{g}_i\) containing lexical-category and morphological properties.
- **Conditional feature probability:** \(P(g\mid w,x)\) represents how strongly a grammatical feature is supported by the lexical candidate and its form.
- **Constraint weighting:** later parsing can use the lexical features as soft constraints. If a later structural candidate conflicts with a strongly supported lexical property, its score may be reduced rather than automatically rejected.

**Possible neural implementation:** lexical and grammatical information is distributed within the language network. Temporal and frontal language-responsive populations exchange information as a word is accessed. The algorithm treats grammar-relevant features as information retrieved from the lexical item, while deferring their composition into sentence structure.

### Step 10 — Handle homonyms and polysemous words without confusing lexical identity with sense selection

**Purpose:** distinguish a word's form-level identity from the selection of a particular sense.

1. Determine whether the candidate form maps to one lexical entry or to several entries that share a form.
2. For a homonym, retain distinct lexical entries if the shared form has unrelated meanings.
3. For a polysemous word, activate the related senses or a shared semantic core, depending on the representation used by the model.
4. Use the limited local context to adjust the relative activation of senses, but do not perform a full discourse or speaker-intention analysis here.
5. If the current evidence does not distinguish the senses, pass the alternatives forward with their confidence values.
6. Allow later syntactic and contextual processing to revise the preferred sense.

**Possible mathematical operations**

- **Sense competition:** for possible senses \(s_j\), compute \(P(s_j\mid w,c)\propto P(c\mid s_j,w)P(s_j\mid w)\), using only available local context \(c\).
- **Shared-core representation:** for related senses, model a common feature vector \(\mathbf{v}_{core}\) plus sense-specific offsets \(\mathbf{d}_j\): \(\mathbf{v}_{s_j}=\mathbf{v}_{core}+\mathbf{d}_j\). This is one possible representation, not a universal theory of polysemy.
- **Ambiguity retention:** keep the top \(k\) sense candidates when the probability margin between the first and second candidate is too small.
- **Confidence margin:** \(\Delta=p_{best}-p_{second}\). A small \(\Delta\) signals that additional input is needed before the model should commit.

**Possible neural implementation:** overlapping semantic populations may support related meanings, while distinct lexical/conceptual patterns support unrelated meanings. Contextual activation can bias one interpretation, but ambiguity may persist until more of the sentence or discourse is processed.

### Step 11 — Reconcile the candidate with new sensory evidence and allow feedback to Point 1

**Purpose:** make lexical access interactive with perceptual analysis.

1. Receive newly available phonological or orthographic evidence.
2. Re-score the existing candidate set rather than starting over from scratch.
3. Add candidates if new evidence supports them; suppress candidates that are now inconsistent.
4. If no candidate matches adequately, request a less committed or revised form hypothesis from Point 1. For speech, the lexical system may help reconsider word boundaries; for print, it may help reconsider an uncertain letter or letter order.
5. Continue this loop until the form evidence and lexical candidate set reach a useful level of stability, the word ends, or the task requires a provisional interpretation.
6. Do not mistake stability for absolute certainty. Noise, rare words, names, foreign words, and novel words may remain uncertain.

**Possible mathematical operations**

- **Incremental update:** \(S_i(t+1)=S_i(t)+\Delta S_i(t)\), where the new evidence modifies the current candidate score.
- **Prediction error:** \(e(t)=x(t)-\hat{x}(t)\), where \(\hat{x}\) is the form predicted by an active candidate. Large mismatch reduces candidate support or triggers revision.
- **Recurrent correction:** \(\mathbf{h}_{t+1}=f(W_{in}\mathbf{x}_t+W_{rec}\mathbf{h}_t+W_{top}\mathbf{c}_t)\), an abstract description of bottom-up input, recurrent state, and top-down bias.
- **Stopping criterion:** stop or pass a provisional result when the best candidate exceeds a confidence threshold and its margin over alternatives is adequate, or when the processing window closes. A biologically realistic system may continue revising after this point.

**Possible neural implementation:** recurrent connectivity among auditory/visual form-responsive cortex, temporal language regions, and frontal control regions permits candidate feedback. Recent recordings from human speech cortex show dynamic word-related activity in the superior temporal gyrus, including activity changes associated with perceived word boundaries; this supports modelling word recognition as temporally evolving rather than as a one-time lookup.

### Step 12 — Package the lexical-semantic output for the next processing point

**Purpose:** deliver a structured result that downstream syntactic parsing can use while retaining uncertainty and alternatives.

1. Select the leading lexical candidate when the evidence supports one.
2. If the evidence remains ambiguous, return a ranked candidate list rather than a false single answer.
3. Attach the word-level semantic feature pattern, possible senses, lexical category, morphological information, and confidence.
4. Preserve the original form evidence and the alternatives that remain plausible.
5. Mark the output as a lexical item, not as a proposition or a complete sentence meaning.
6. Pass the result to the next point: syntactic parsing, where words and their lexical properties will be organized into a grammatical structure.

**Possible mathematical operations**

- **Normalization:** rescale candidate confidence values so they can be compared, for example \(\sum_i p_i=1\), while keeping in mind that real neural firing rates are not necessarily probabilities.
- **Confidence calibration:** compare confidence with observed correctness over experience, updating the model's confidence mapping when it is systematically over- or under-confident.
- **Structured output assembly:** combine lexical identity, feature activations, morphological analysis, and uncertainty into one data object. This is an engineering representation of what information must be available to the next stage.

**Completion condition:** Point 2 is complete when the system has activated a word-level lexical-semantic representation from a spoken or written form, has preserved unresolved candidates or senses, and has supplied relevant lexical properties to the syntactic parser without yet constructing the sentence's predicate-argument structure.

## 5. Integrated mathematical sketch

The following compact system collects the possible computations above. It is a computational model of the functional problem, not a claim that the brain runs this exact set of equations.

Let:

- \(x_t\) be the current auditory or visual evidence at time \(t\);
- \(z_t\) be the modality-specific form representation;
- \(w_i\) be a candidate lexical entry;
- \(M_i\) be its form-match score;
- \(F_i\) be its familiarity/frequency contribution;
- \(C_i\) be its limited local-context contribution;
- \(R_i\) be morphological compatibility;
- \(I_i\) be inhibition from competing candidates;
- \(V_i\) be its retrieved semantic feature pattern.

A high-level candidate score can be described as:

\[
S_i(t)=\alpha M_i(t)+\beta F_i+\gamma C_i(t)+\delta R_i(t)-\lambda I_i(t)
\]

Candidate activation evolves over time:

\[
a_i(t+1)=\rho a_i(t)+(1-\rho)\,\sigma(S_i(t))
\]

A model can convert the scores to a relative candidate distribution:

\[
p_i(t)=\frac{\exp(S_i(t)/T)}{\sum_j\exp(S_j(t)/T)}
\]

The selected candidate or candidates activate lexical-semantic features:

\[
\mathbf{V}(t)=\sigma\left(W_{sem}\sum_i p_i(t)\mathbf{e}_i+\mathbf{b}\right)
\]

where \(\mathbf{e}_i\) is a representation of candidate \(i\), and \(W_{sem}\) maps candidate activity into a semantic feature pattern. The system returns the ranked entries, semantic pattern, lexical properties, confidence, and ambiguity state.

A model should not blindly maximize one candidate. If the two strongest candidates are close, or if the sensory reliability is low, it should preserve multiple alternatives and request further evidence. This is important for homophones, homographs, short function words, noisy speech, unfamiliar names, and ambiguous morphology.

## 6. Example walkthroughs (illustrative, not extra source points)

### Example A — Hearing an incomplete spoken word

A listener hears the beginning of a word that is compatible with several familiar candidates. The auditory form system activates multiple candidates from the partial phonological sequence. Frequency and recent context bias their starting strengths, but none is yet final. As further speech arrives, incompatible candidates lose support. The remaining candidate activates its word-level semantic pattern and lexical properties. The syntactic parser then uses those properties in its own operation.

### Example B — Reading a familiar written word

Visual processing supplies an ordered orthographic pattern. Left ventral occipitotemporal populations respond to familiar letter and word patterns. A lexical entry is activated, and its associated semantic features become available. A familiar high-frequency word may be retrieved rapidly, while a rare word may produce weaker or slower activation and remain uncertain until additional context is available.

### Example C — A word with several senses

A form activates more than one sense. The most common sense may receive a stronger baseline activation, but the local phrase may favour another sense. Point 2 can retain both with different confidence values. The complete sentence and discourse context will later provide additional evidence; Point 2 should not prematurely resolve every ambiguity.

## 7. What this point does not do

To preserve the source's ordering, this algorithm deliberately does **not** implement:

- full syntactic parsing or assignment of actual grammatical roles;
- combining multiple word meanings into a sentence-level proposition;
- constructing predicate-argument structure for the entire clause;
- integrating the completed sentence with the full discourse history or physical situation;
- resolving irony, sarcasm, indirect speech acts, presupposition, or Gricean implicature;
- generating bridging inferences or broad causal explanations;
- building and maintaining the final propositional working-memory representation;
- long-term episodic storage or decision-making based on the sentence.

Local expectations may bias lexical activation, but the later contextual/pragmatic point remains responsible for evaluating the sentence-level interpretation against broader context and speaker intention.

## 8. Biological cautions and open questions

1. **Lexical access is distributed.** It is inaccurate to assign the whole dictionary, all word meanings, or all lexical selection to one cortical region. Form, lexical, semantic, and control information is represented across interconnected neural populations.
2. **The two input modalities are not identical.** Spoken and written words begin with different sensory codes. They can converge on overlapping language and conceptual systems, but there is no need to assume a single shared form code.
3. **The processing sequence is not strictly serial.** Partial word-form activation, lexical activation, semantic activation, and context-based prediction can overlap in time and influence each other.
4. **Candidate competition is a useful model, not a fully settled microscopic explanation.** Activation, inhibition, Bayesian scoring, softmax, and entropy make the computational problem explicit; they do not prove that biological neurons perform those exact mathematical operations.
5. **Frequency and context are not the same variable.** Frequency reflects long-term experience; context reflects what is expected in the current situation. They must be represented separately in both the algorithm and its tests.
6. **Meaning is not always a single fixed vector.** Concrete concepts, abstract concepts, function words, polysemous words, and morphologically complex words may require different representational treatments.
7. **The role of frontal control is conditional.** Left inferior frontal activity may be especially relevant when selection is difficult or alternatives compete; automatic access to a familiar word should not be modelled as requiring deliberate executive control on every occasion.
8. **Neural timing and region boundaries vary.** Findings from intracranial recording, fMRI, EEG/MEG, and lesion studies measure different aspects of the system. A region's activation does not by itself establish that it performs one exclusive operation.
9. **Language-specific structure matters.** Phonology, orthography, morphology, and word segmentation differ across languages and writing systems. A biologically inspired implementation must be adaptable rather than hard-coded for English.

## 9. References for Point 2

These sources supplement the user's Library document and support the neural framing in this installment:

1. **Blumstein, S. E. (2009). Auditory word recognition: evidence from aphasia and functional neuroimaging.** *Language and Linguistics Compass, 3*(4), 824–838. Reviews a distributed auditory word-recognition pathway involving acoustic-phonetic analysis, mapping to lexical candidates, lexical-semantic access, and frontal selection. [https://doi.org/10.1111/j.1749-818X.2009.00136.x](https://doi.org/10.1111/j.1749-818X.2009.00136.x)
2. **DeWitt, I., & Rauschecker, J. P. (2012). Phoneme and word recognition in the auditory ventral stream.** *Proceedings of the National Academy of Sciences, 109*(8), E505–E514. A meta-analysis of imaging studies reporting a hierarchical pattern from phoneme-sensitive left mid-STG toward word-level integration in left anterior STG. [https://doi.org/10.1073/pnas.1113427109](https://doi.org/10.1073/pnas.1113427109)
3. **Damera, S. R., et al. (2023). Evidence for a Spoken Word Lexicon in the Auditory Ventral Stream.** *Neurobiology of Language, 4*(3), 420–434. Reports evidence for word-selective representations in the left anterior superior temporal gyrus and changes in selectivity after familiarization with novel spoken words. [https://doi.org/10.1162/nol_a_00108](https://doi.org/10.1162/nol_a_00108)
4. **Zhang, Y., Leonard, M. K., Bhaya-Grossman, I., Gwilliams, L., & Chang, E. F. (2026). Human cortical dynamics of auditory word form encoding.** *Neuron, 114*(1), 167–180.e6. Intracranial recordings show that STG activity dynamically combines acoustic-phonetic, prosodic, and lexical information during continuous speech and tracks perceived word boundaries. [https://doi.org/10.1016/j.neuron.2025.10.011](https://doi.org/10.1016/j.neuron.2025.10.011)
5. **Glezer, L. S., et al. (2021). Spatiotemporal Dynamics of Orthographic and Lexical Processing in the Ventral Visual Pathway.** *Nature Human Behaviour, 5*, 389–398. Intracranial recordings characterize the timing of lexicality and word-frequency sensitivity in left ventral occipitotemporal cortex. [https://doi.org/10.1038/s41562-020-00982-w](https://doi.org/10.1038/s41562-020-00982-w)
6. **Bouhali, F., et al. (2014). Anatomical Connections of the Visual Word Form Area.** *Journal of Neuroscience, 34*(46), 15402–15414. Reports connectivity between the visual word form area and left perisylvian language regions implicated in phonological processing and lexical access. [https://doi.org/10.1523/JNEUROSCI.4918-13.2014](https://doi.org/10.1523/JNEUROSCI.4918-13.2014)
7. **Lerma-Usabiaga, G., et al. (2024). Demystifying visual word form area visual and nonvisual response properties with precision fMRI.** *iScience, 27*(12), 111481. Examines the specificity of the visual word form area and its relation to both visual and language-related processing. [https://doi.org/10.1016/j.isci.2024.111481](https://doi.org/10.1016/j.isci.2024.111481)

---

**End of Part 2.** The next distinct point in the source sequence is **syntactic parsing — organizing retrieved lexical items into a grammatical structure**. It has not been added yet.
# Part 3 — Syntactic Parsing: Building a Grammatical Structure from Retrieved Words

**Position in the source sequence:** Part 3, following perceptual word-form processing (Part 1) and lexical access (Part 2), and preceding sentence-level semantic composition.

**Input:** A stream of recognized lexical candidates and their available features. Each incoming item may include a lemma, possible part-of-speech categories, morphological features, phonological or orthographic form, and confidence values inherited from earlier processing.

**Output:** An incremental, possibly ambiguous syntactic representation: constituent groupings, grammatical dependency links, grammatical functions, unresolved structural slots, and confidence values. This output is passed to semantic composition. It is not yet a complete proposition or a fully interpreted event representation.

## 1. Goal and boundary of this algorithm

Syntactic parsing is the process of determining how words are grammatically organized with respect to one another. It answers questions such as: which words form a noun phrase; which noun phrase is attached to a verb; where a clause begins and ends; which word is the head of a phrase; whether a phrase is a complement or modifier; and how grammatical features such as number, person, gender, tense, or case constrain the structure.

The parser must work incrementally. It cannot wait for the whole sentence, because listeners and readers start organizing a sentence as soon as enough information becomes available. It therefore maintains a partial structure, predicts what kinds of material may come next, updates that structure when a new word arrives, and sometimes revises an earlier decision when later material makes another analysis more likely.

This algorithm deliberately separates **grammatical relations** from **semantic interpretation**. For example, the parser can identify that “the cat” is the direct-object noun phrase of “chased” in “The dog chased the cat.” The later semantic-composition process can use that syntactic relation, together with the lexical meanings of *dog*, *chase*, and *cat*, to construct an event meaning such as CHASE(dog, cat). The present algorithm does not itself complete that semantic representation.

The brain does not necessarily execute a neat, strictly serial list of operations. Word-category activation, structural prediction, attachment, agreement checking, and memory retrieval can overlap and interact. The numbered steps below are an explicit computational decomposition of the problem, not a claim that the brain has one anatomically isolated processor for each step.

## 2. Representations and state variables

Before processing a sentence, define the following computational objects. They are useful for making the algorithm testable; they should not be mistaken for literal data structures stored in neurons.

### 2.1 Incoming lexical item

For word position `i`, represent the incoming lexical item as:

`W_i = {form, lemma_candidates, category_candidates, morphology, boundary_cues, confidence}`

Where:

- `form` is the recognized spoken or written form.
- `lemma_candidates` are possible dictionary entries supplied by lexical access.
- `category_candidates` are possible grammatical categories, such as noun, verb, adjective, adverb, determiner, pronoun, preposition, complementizer, or conjunction.
- `morphology` contains features such as tense, aspect, number, person, gender, case, mood, and inflectional markers when relevant to the language.
- `boundary_cues` may include speech prosody, pauses, stress, punctuation, or orthographic layout.
- `confidence` records uncertainty inherited from perception and lexical access.

A word may have more than one category. For example, an English form such as *flies* can be a noun or a verb depending on the sentence. The parser should preserve competing categories until contextual and grammatical evidence distinguishes them.

### 2.2 Partial parse state

Maintain a state `S_i` after processing the first `i` words:

`S_i = {constituent_forest, dependency_graph, active_expectations, unresolved_slots, feature_constraints, memory_state, parse_scores}`

- `constituent_forest` stores one or more possible phrase-structure trees or partial trees.
- `dependency_graph` stores candidate head–dependent relations between words.
- `active_expectations` stores grammatical categories or phrase types that could plausibly occur next.
- `unresolved_slots` records incomplete structures, such as an opened clause that still requires a predicate or an expected complement.
- `feature_constraints` stores agreement and compatibility requirements.
- `memory_state` tracks incomplete phrases and dependencies that must remain available while later words are processed.
- `parse_scores` stores the current relative support for alternative analyses.

Using a **parse forest**, rather than forcing one tree immediately, is important for ambiguous sentences. The same word sequence can initially support multiple grammatically possible structures.

### 2.3 Grammar knowledge

The parser uses learned knowledge of the language's grammatical regularities. This may include:

- phrase-building patterns, such as determiner + noun forming a noun phrase;
- head–dependent patterns, such as a verb taking a subject and possibly an object or complement;
- word-order regularities;
- agreement patterns;
- case-marking patterns;
- the distribution of function words and inflectional morphemes;
- language-specific rules for clauses, questions, relative clauses, and coordination.

These regularities are learned from linguistic experience and are sensitive to the language and construction being processed. English word order is not a universal template; languages with richer case marking, different constituent orders, or different agreement systems require different parameterizations.

## 3. Detailed algorithmic steps

### Step 1 — Receive the next lexical item and align it with the current parse

**Purpose:** Add the next recognized word to the ongoing sentence-processing state without discarding the partially built structure.

**Operations:**

1. Receive `W_i` from the lexical-access stage.
2. Attach its word position and temporal or text-order index.
3. Retain the lexical candidates and their confidence values rather than assuming that the most likely lexical entry is certainly correct.
4. Retrieve the current parse state `S_(i-1)`.
5. Make the new item available to all active syntactic hypotheses.

**Possible mathematical description:**

Let `q_i(k)` be the lexical confidence that incoming item `i` corresponds to candidate lexical entry `k`. Let `P_(i-1)(T)` be the current support for partial parse `T`. The initial joint support for extending `T` with candidate `k` can be represented as:

`J_i(T,k) = P_(i-1)(T) × q_i(k)`

This multiplication is a simple factorization of two information sources. A more complete probabilistic model would account for dependence between lexical and syntactic evidence rather than assuming they are independent.

**Neural interpretation:** The incoming word representation is made available to the distributed language network. Posterior temporal language regions are implicated in processing and combining sequential linguistic information; frontal and temporal language areas interact as structure becomes more demanding. No single region should be treated as a literal input queue.

**Result:** The new word is available to the currently active parses, with its uncertainty preserved.

### Step 2 — Estimate the word's grammatical category

**Purpose:** Determine which grammatical roles the incoming lexical item could occupy in the current language and sentence position.

**Operations:**

1. Retrieve the candidate grammatical categories associated with each lexical entry.
2. Examine morphological evidence. For example, an inflectional ending may favour a verb form, plural noun, or agreement-bearing adjective, depending on the language.
3. Examine local grammatical context. A determiner may make a noun or adjective sequence more likely; an auxiliary may make a verb phrase continuation likely.
4. Use word order and available function words as additional evidence.
5. Preserve more than one category when the evidence is inconclusive.

**Possible mathematical description:**

For category `c`, lexical entry `k`, morphological features `m_i`, and previous parse context `C_i`, estimate:

`P(c | k, m_i, C_i) ∝ P(m_i | c,k) × P(c | k,C_i)`

A log-score implementation can use:

`Score_category(c) = log P(m_i | c,k) + log P(c | k,C_i)`

If the input is ambiguous, normalize the candidate scores:

`P(c | evidence) = exp(Score_category(c)) / Σ_c' exp(Score_category(c'))`

The exponential normalization is a convenient computational model. It is not evidence that cortical tissue literally applies a softmax function.

**Neural interpretation:** Category expectations can be informed by lexical knowledge and morphosyntactic regularities. Left inferior frontal and posterior temporal language regions are often implicated in syntactic processing, but the precise division of labour is debated. Studies of hierarchical syntax associate left inferior frontal regions, particularly parts of BA44, and posterior temporal regions with structure-building and the processing of syntactic relations.

**Result:** Each current parse receives one or more possible category assignments for the new word.

### Step 3 — Generate expectations for the next grammatical material

**Purpose:** Use the incomplete structure to estimate what types of word or phrase may follow.

**Operations:**

1. Inspect each active parse and identify its open grammatical positions.
2. Use the current head and construction to estimate which continuations are permitted or expected.
3. Combine grammatical expectations with language-specific ordering patterns.
4. Represent uncertainty as a distribution over possible upcoming categories or phrase types.
5. Update these expectations whenever new evidence changes the parse.

For example, after “The child will …”, the parser may assign high probability to a verb phrase. After “The child with …”, a noun-phrase continuation may be expected. The expectations are graded, not absolute: unusual but grammatical continuations remain possible.

**Possible mathematical description:**

For candidate next category `c`, estimate:

`P(c_next | S_i, grammar) = Σ_T P(c_next | T, grammar) P(T | S_i)`

The parser can represent the unexpectedness of an actual category `c_i` as syntactic surprisal:

`Surprisal_i = -log₂ P(c_i | S_(i-1), grammar)`

A lower probability produces a higher surprisal value. This gives a principled way to represent how strongly an incoming word conflicts with the parser's current expectations.

**Neural interpretation:** Predictive structure-building has been proposed in accounts of human sentence processing. MEG and fMRI research has reported neural responses related to syntactic expectations and syntactic surprisal in fronto-temporal language regions. These responses should not be interpreted as a direct readout of one exact probability calculation in the brain.

**Result:** Each partial parse has an updated set of expectations that can guide the interpretation of the next word.

### Step 4 — Open or extend phrase-structure constituents

**Purpose:** Group words into hierarchical constituents rather than treating the sentence as an unstructured list.

**Operations:**

1. Use the category of the incoming word and the active grammatical expectations to decide whether it can begin a new phrase, continue an open phrase, or close a constituent.
2. Add the word to a candidate constituent.
3. Create nested phrase groupings where the grammar requires them. For example, a determiner and noun may form a noun phrase; a verb and its dependents may form a verb phrase; phrases may combine into a clause.
4. Keep track of the head of each phrase and the words that depend on it.
5. Leave a constituent open when more words may still belong to it.
6. Permit alternative groupings when attachment or category is ambiguous.

For “The small dog chased the cat,” a simplified constituent structure develops as follows:

- `The small dog` is grouped as a noun phrase.
- `chased the cat` is grouped as a verb phrase.
- The noun phrase and verb phrase combine into a clause or sentence-level constituent.
- `the cat` is grouped as a second noun phrase inside the verb phrase.

**Possible mathematical description:**

A phrase-structure grammar can be represented by productions such as:

`S → NP VP`

`NP → Det Adj* N`

`VP → V NP`

These are illustrative rules for a simplified grammar, not a universal grammar for every language. A parser can use a chart or parse forest to store partial constituents. If `A[i,j]` denotes a constituent of type `A` spanning words `i` through `j`, a chart algorithm can combine compatible subspans according to grammatical production rules.

A candidate constituent can receive a score:

`Score_constituent = Score_children + Score_rule + Score_boundary`

where `Score_rule` represents support for the grammatical combination and `Score_boundary` represents supporting prosodic or punctuation evidence when available.

**Neural interpretation:** A study comparing neural activity with phrase-structure and dependency descriptors reported that left posterior superior temporal gyrus activity favoured phrase-structure descriptions, while left inferior frontal gyrus and left anterior temporal pole favoured dependency-structure descriptions. This is evidence for differentiated sensitivities within a network, not proof that each region stores a complete formal grammar.

**Result:** The parse contains nested phrase groupings and records which constituents remain incomplete.

### Step 5 — Identify heads and construct dependency relations

**Purpose:** Record which word grammatically organizes another word or phrase.

**Operations:**

1. Identify the likely head of the current phrase. In a verb phrase, the verb is commonly the head; in a noun phrase, the noun is commonly the head.
2. Identify candidate dependents, such as subjects, objects, modifiers, determiners, complements, and adjuncts.
3. Connect the dependent to its likely head while preserving alternative attachments when needed.
4. Record the grammatical relation type.
5. Maintain the direction of the relation and the word positions involved.

In “The dog chased the cat,” the parser can represent:

- `chased` as the head of the clause's predicate structure;
- `dog` as the subject of `chased`;
- `cat` as the object of `chased`;
- `The` as a determiner dependent of `dog`;
- `the` as a determiner dependent of `cat`.

The labels *subject* and *object* here are grammatical relations. The parser has not yet calculated the event roles or conceptual meaning of the chase.

**Possible mathematical description:**

Represent a dependency edge from head `h` to dependent `d` as:

`e = (h, d, relation, confidence)`

For a candidate edge, compute:

`Score_edge = w₁ × category_compatibility + w₂ × order_compatibility + w₃ × morphology_compatibility + w₄ × construction_prior`

The weights `w₁ ... w₄` are model parameters. They can be learned or estimated for a particular language and construction. A global parse score should also enforce structural constraints, such as avoiding incompatible head assignments in a basic dependency tree.

**Neural interpretation:** The 2021 study *Distinguishing Syntactic Operations in the Brain: Dependency and Phrase-Structure Parsing* found that left anterior temporal pole and left inferior frontal gyrus activity aligned more strongly with dependency-structure descriptions, while left posterior superior temporal gyrus aligned more strongly with phrase-structure descriptions. These results support investigating different syntactic computations across interacting regions; they do not settle the exact computational roles of all areas.

**Result:** A dependency graph records head–dependent relations alongside phrase groupings.

### Step 6 — Integrate grammatical features and check agreement

**Purpose:** Test whether words that are grammatically linked have compatible features.

**Operations:**

1. Retrieve relevant grammatical features for the head and dependent.
2. Determine which features must agree in the current construction. Depending on the language, these may include number, person, gender, case, or noun-class features.
3. Compare the feature values across the proposed relation.
4. Increase support for compatible structures and decrease support for incompatible ones.
5. Do not reject a structure solely because a feature is missing or uncertain; some languages omit overt marking, and perception may be noisy.
6. Keep track of unresolved agreement requirements until the relevant word appears.

For example, in English “These dogs run,” plural marking on *these* and *dogs* is compatible, and the plural subject is compatible with *run*. In a language with richer case marking, a case suffix may provide evidence about whether a noun phrase is a subject, object, or another grammatical dependent.

**Possible mathematical description:**

Let `F_h` and `F_d` be feature sets for head and dependent. A compatibility function can be defined as:

`Compat(F_h,F_d) = 1` if all required feature constraints are satisfied, otherwise a value between 0 and 1 reflecting uncertainty or the severity of mismatch.

An energy or penalty term can be added to the parse score:

`E_agreement = Σ_r λ_r × mismatch_r`

`Score_updated = Score_previous - E_agreement`

Here `mismatch_r` quantifies a violation of agreement constraint `r`, and `λ_r` controls its influence. A probabilistic implementation may instead use `log P(feature evidence | structure)`.

**Neural interpretation:** Morphosyntactic information is processed within the broader fronto-temporal language network. EEG studies report left-anterior negativities in some agreement and phrase-structure conditions, but the timing, scalp distribution, and precise functional meaning of LAN effects vary across studies. Agreement processing should not be assigned to one ERP component or one cortical location alone.

**Result:** Parse candidates are marked as morphologically compatible, uncertain, or mismatching.

### Step 7 — Combine word order, case, function words, and prosodic boundary evidence

**Purpose:** Use multiple grammatical cues rather than depending on a single word-order rule.

**Operations:**

1. Examine the position of the word relative to its likely head and other dependents.
2. Identify function words such as determiners, auxiliaries, prepositions, complementizers, and conjunctions.
3. Examine case markers or other inflectional cues where the language uses them.
4. For speech, evaluate prosodic cues such as stress, pitch movement, rhythm, and pause placement when relevant.
5. For reading, use punctuation and visual boundary cues as imperfect evidence of syntactic grouping.
6. Combine the cues according to the grammar of the language and the current construction.

For example, the English order in “The dog chased the cat” is a strong cue to the subject and object positions. In a language with case marking, noun endings may carry more information about grammatical function, so word order may be more flexible. In speech, a prosodic boundary may support a clause boundary, but pauses alone do not determine syntax.

**Possible mathematical description:**

Let the cues be `C = {order, morphology, function_words, prosody, punctuation}`. For a candidate parse `T`, combine their evidence as:

`Score_cues(T) = Σ_j α_j × log P(C_j | T)`

where `α_j` weights each cue. A full model should account for correlated cues and avoid counting the same information twice. Cue weights may vary across languages, speakers, speaking rates, and tasks.

**Neural interpretation:** Speech prosody involves auditory and prosodic processing beyond the narrow syntactic network; frontal-temporal regions can interact with this information during sentence comprehension. The importance of prosody differs by construction and language. Punctuation is a visual convention in reading, not a direct neural counterpart of spoken intonation.

**Result:** Each parse is evaluated using the grammatical cues available in the current modality and language.

### Step 8 — Maintain unresolved dependencies and incomplete structures

**Purpose:** Preserve earlier words that must be connected to later words, even when they are separated by intervening material.

**Operations:**

1. Identify grammatical dependencies that are not yet complete.
2. Keep the relevant head, dependent, or open phrase accessible.
3. Record which later word or phrase could complete the dependency.
4. Update the memory state as new words arrive.
5. When a matching word appears, attempt to connect it to the unresolved structure.
6. If several open dependencies compete, retain enough alternatives to avoid premature attachment.

Examples include a subject waiting for its verb, a verb awaiting a complement, or a noun phrase that is separated from a later clause position by a relative clause. Long-distance dependencies require information from earlier parts of the sentence to remain available.

**Possible mathematical description:**

For an unresolved dependency `d`, define its availability at time `t` as:

`A_d(t) = A₀ × exp(-λΔt) × CueMatch(d,t)`

where `A₀` is initial activation, `Δt` is elapsed processing time, `λ` is a decay parameter, and `CueMatch` measures the fit between the stored dependency and the current retrieval cues. A similarity-based interference term can be included:

`A'_d(t) = A_d(t) - γ × Σ_j Similarity(d,d_j)`

This is one possible model of memory accessibility and interference. The brain's actual mechanisms may involve distributed neural activity, recurrent interactions, oscillatory coordination, and several memory systems rather than a literal decaying stack.

**Neural interpretation:** Syntactic dependencies place demands on the language network and, when the material is complex or spans a long distance, can also increase demands on domain-general memory and control systems. Left inferior frontal and posterior temporal regions are repeatedly implicated in sentence-level syntactic processing; fronto-parietal support can become more important under high memory or task demands.

**Result:** Earlier grammatical information remains available for later integration, with uncertainty and interference represented explicitly.

### Step 9 — Rank competing structural analyses without forcing an early decision

**Purpose:** Select a coherent analysis when several parse structures are possible, while retaining alternatives if the evidence is insufficient.

**Operations:**

1. Collect the current candidate parses from category assignment, phrase construction, dependency attachment, feature checking, and cue integration.
2. Add their grammatical and contextual support scores.
3. Penalize contradictions, impossible attachments, and unmet constraints where the grammar requires them.
4. Keep the best-supported candidates in a bounded parse set (a computational beam), or maintain a probability distribution across the parse forest.
5. Avoid treating a small difference in scores as certainty.

A useful example is “I saw the man with the telescope.” The phrase “with the telescope” may attach to the verb phrase (the telescope was used for seeing) or to the noun phrase (the man had the telescope). Both analyses can remain active until more information arrives.

**Possible mathematical description:**

For parse `T`, combine evidence into:

`LogScore(T) = LogPrior(T) + LogLexical(T) + LogMorphology(T) + LogOrder(T) + LogProsody(T) - Penalties(T)`

Normalize scores into relative support:

`P(T | input) = exp(LogScore(T)) / Σ_T' exp(LogScore(T'))`

For a bounded beam, retain the top `B` candidates, where `B` is a model capacity parameter. The brain is not known to use a literal top-`B` algorithm; this is a practical approximation for a computational implementation.

**Neural interpretation:** Multiple syntactic analyses may compete during incremental comprehension. Neural responses to structural expectation and ambiguity are consistent with ongoing prediction and integration, but do not reveal one single ranking equation. Frontal-temporal interaction is a plausible broad substrate for evaluating structural alternatives.

**Result:** The parser maintains a ranked set of grammatical structures, including unresolved ambiguity when appropriate.

### Step 10 — Detect when new material contradicts the preferred parse

**Purpose:** Identify evidence that the current preferred grammatical analysis is no longer adequate.

**Operations:**

1. Compare the incoming item's category and features with the open positions in the current parse.
2. Check whether the preferred parse can legally incorporate the new item.
3. Detect conflicts such as an unexpected category, an agreement mismatch, an impossible attachment, or a phrase that cannot be integrated under the current structure.
4. Distinguish a genuine grammatical violation from a merely unexpected but grammatical continuation.
5. Mark the affected portion of the parse as requiring review.

**Possible mathematical description:**

A simple conflict value is:

`Conflict_i = -log P(W_i | preferred_parse, grammar)`

A discrepancy can be flagged when:

`Conflict_i > θ_conflict`

where `θ_conflict` is a decision threshold. A more nuanced model can compute the posterior probability of a violation versus a low-probability but grammatical continuation.

**Neural interpretation:** EEG/ERP research has identified early and later responses associated with syntactic expectancy violations and reanalysis. Early left-anterior negativities have been linked to some phrase-structure violations, while P600-like positivities have been associated with later re-evaluation and repair. Their interpretation is debated and they are not unique, one-to-one markers for a single parsing operation.

**Result:** The parser identifies whether the current structure can continue normally or whether one or more structural alternatives need to be reconsidered.

### Step 11 — Reanalyse and repair a garden-path or conflicting structure

**Purpose:** Revise an earlier parse when later words make another grammatical interpretation more coherent or necessary.

**Operations:**

1. Locate the earliest point in the current parse at which the preferred structure becomes incompatible with the new evidence.
2. Reopen alternative categories, constituent boundaries, or dependency attachments that were previously down-ranked.
3. Remove or weaken the unsupported structural links.
4. Reattach the affected words under the best-supported alternative.
5. Recompute agreement and unresolved dependency constraints.
6. Re-rank the surviving candidates and update confidence values.
7. Continue parsing from the revised state.

For example, “The horse raced past the barn fell” often encourages an initial analysis in which *raced* is the main verb. The later word *fell* forces the reader to reconsider *raced past the barn* as a reduced relative clause modifying *horse*. The example illustrates structural revision; the exact difficulty depends on familiarity, prosody, context, and individual processing.

**Possible mathematical description:**

Let `T_old` be the previous preferred parse and `T_new` a revised candidate. Reanalysis is warranted when:

`LogScore(T_new | W_1...W_i) - LogScore(T_old | W_1...W_i) > θ_revision`

A repair cost can be included:

`RevisedScore(T_new) = LogScore(T_new) - ρ × RevisionCost(T_old,T_new)`

where `ρ` represents the weight assigned to the effort of restructuring. The system should not preserve a bad parse merely because revision is costly; new evidence can outweigh the cost.

**Neural interpretation:** P600-like responses are often observed during syntactic reanalysis, but P600 is not exclusively syntactic and can appear in other forms of integration difficulty. Reviews emphasize that it may reflect re-evaluation, monitoring, or repair, and that the mapping between ERP components and cognitive operations remains contested. The neural implementation likely recruits interacting language and control processes rather than one dedicated “repair centre.”

**Result:** A revised parse replaces or competes with the earlier structure, and the parser resumes incremental processing.

### Step 12 — Close completed constituents and consolidate the sentence-level syntactic frame

**Purpose:** Determine when a phrase or clause has received enough grammatical material to be treated as structurally complete, while preserving any dependencies that remain open.

**Operations:**

1. Check whether the current constituent has satisfied the grammatical requirements of its head and construction.
2. Close the constituent when the relevant language-specific completion conditions are met.
3. Attach the completed phrase to its parent constituent.
4. Preserve open dependencies or unresolved ambiguities that still require later material.
5. At a clause or sentence boundary, consolidate the best-supported syntactic representation.
6. Pass the output to semantic composition together with confidence values and unresolved alternatives.

A sentence-final punctuation mark or prosodic boundary can support closure, but structural completion is not determined by punctuation alone. Some clauses contain dependencies that extend across apparent boundaries, and spoken language may contain repairs, fragments, or disfluencies.

**Possible mathematical description:**

Define a completion function for constituent `C`:

`Complete(C) = 1` when all obligatory syntactic requirements of `C` are satisfied and the current evidence supports closure; otherwise `Complete(C) = 0` or a graded probability between 0 and 1.

The final syntactic output can be represented as:

`ParseOutput = {best_parse, alternative_parses, category_labels, phrase_edges, dependency_edges, grammatical_features, unresolved_links, confidence}`

**Neural interpretation:** The syntactic representation is supported by distributed fronto-temporal language circuitry. The exact neural timing of constituent closure and the point at which a parse becomes stable vary with sentence type, language, task, and listener or reader. The subsequent semantic-composition stage uses this structural information to combine lexical meanings.

**Result:** A structured grammatical representation is available for semantic composition.

## 4. Integrated pseudocode for the complete syntactic-parsing algorithm

```text
FUNCTION SYNTACTIC_PARSE(lexical_stream, language_grammar):
    parse_state = INITIALIZE_PARSE_FOREST(language_grammar)

    FOR each lexical_item W_i IN lexical_stream:
        W_i = ALIGN_WITH_SEQUENCE(W_i, parse_state)

        category_hypotheses = ESTIMATE_GRAMMATICAL_CATEGORIES(
            W_i,
            parse_state,
            language_grammar
        )

        FOR each parse T IN parse_state.candidates:
            expectations = PREDICT_NEXT_CATEGORIES(T, language_grammar)

            FOR each category candidate c IN category_hypotheses:
                candidate = COPY(T)
                candidate = ASSIGN_CATEGORY(candidate, W_i, c)
                candidate = EXTEND_PHRASE_STRUCTURE(candidate, W_i, c)
                candidate = ADD_DEPENDENCY_RELATIONS(candidate, W_i, c)
                candidate = CHECK_MORPHOSYNTACTIC_FEATURES(candidate)
                candidate = INTEGRATE_ORDER_CASE_FUNCTION_WORDS_AND_PROSODY(
                    candidate,
                    W_i
                )
                candidate = UPDATE_UNRESOLVED_DEPENDENCIES(candidate)
                candidate = SCORE_PARSE(candidate, expectations)
                ADD candidate TO parse_state.parse_forest

        parse_state = MERGE_EQUIVALENT_PARTIAL_PARSES(parse_state)
        parse_state = RANK_AND_RETAIN_ALTERNATIVES(parse_state)

        IF STRUCTURAL_CONFLICT_DETECTED(parse_state, W_i):
            parse_state = REOPEN_RELEVANT_ALTERNATIVES(parse_state)
            parse_state = REANALYSE_AND_REPAIR(parse_state, language_grammar)
            parse_state = RANK_AND_RETAIN_ALTERNATIVES(parse_state)

        parse_state = CLOSE_COMPLETED_CONSTITUENTS(parse_state)
        parse_state = UPDATE_ACTIVE_EXPECTATIONS(parse_state)

    RETURN CONSOLIDATE_SYNTACTIC_OUTPUT(parse_state)
```

The pseudocode is a high-level functional description. It does not imply that all operations occur in this order in the brain or that the brain uses a conventional software parse forest. Its purpose is to make the hypothesized computations explicit enough to implement, test, and compare with behavioral or neural data.

## 5. Neural systems potentially contributing to the algorithm

### 5.1 Left posterior superior temporal gyrus and posterior temporal language regions

These regions are frequently implicated in sentence-level syntactic processing and the construction of hierarchical phrase structures. In the 2021 study comparing phrase-structure and dependency descriptions, left posterior superior temporal gyrus activity was better aligned with phrase-structure measures. This is compatible with a role in grouping sequential words into larger grammatical units, while leaving room for additional functions and interactions.

### 5.2 Left inferior frontal gyrus

Parts of the left inferior frontal gyrus, including regions commonly discussed around BA44, are implicated in syntactic processing, particularly when structural relationships are complex or require integration. The same 2021 comparison found left inferior frontal activity more aligned with dependency-structure measures. Other accounts emphasize selection, unification, or control contributions. It is therefore better to model IFG as an important interacting node than to assign it one exclusive operation.

### 5.3 Left anterior temporal region

The left anterior temporal pole and adjacent anterior temporal regions have been associated with syntactic structure, including dependency-structure descriptions in the comparative study. The anterior temporal lobe may contribute to combining relational structure across words and phrases. Its functions are not limited to syntax in every theoretical account.

### 5.4 Frontal-temporal interaction

Sentence parsing is unlikely to be implemented by isolated cortical modules. Communication between frontal and temporal language regions may support the combination of word-category information, structural expectations, and dependencies. Functional connectivity studies and neuroimaging meta-analyses support a distributed network view, though the exact direction and timing of information exchange remain active research topics.

### 5.5 Memory and domain-general control systems

Simple grammatical combinations may be processed largely by the core language network. Long-distance dependencies, ambiguity, or complex sentences can impose additional demands on temporary maintenance and control. Domain-general frontal and parietal systems may contribute when the task or sentence makes these demands substantial. Their participation should not be assumed to be necessary for every ordinary grammatical phrase.

## 6. Candidate mathematical operations and what they represent

| Mathematical operation | Role in the model | Important qualification |
|---|---|---|
| Bayesian or conditional probability | Estimate word category and parse alternatives from lexical, morphological, and structural evidence | A formal model of uncertainty, not a proven literal neural calculation |
| Log probability | Combine evidence and express the cost of unlikely continuations | Convenient for numerical stability and additive scoring |
| Syntactic surprisal, `-log P` | Quantify how unexpected the next category or word is under the current parse | Neural correlates do not imply the brain explicitly computes this equation |
| Weighted evidence integration | Combine word order, case, function words, morphology, and prosody | Cue weights are language- and construction-dependent |
| Feature unification / constraint checking | Test agreement and grammatical compatibility | Can be implemented as hard constraints or graded penalties |
| Graph construction | Represent head–dependent relations | Dependency graphs are computational descriptions of grammar |
| Hierarchical tree construction | Group words into phrases and clauses | Phrase-structure theories differ in their formal details |
| Beam search / parse forest | Preserve multiple plausible interpretations | A bounded beam is an engineering approximation, not an established neural mechanism |
| Decay and similarity-based interference | Model the accessibility of unresolved dependencies | The brain's memory mechanisms are distributed and more complex than a single decay equation |
| Prediction error / conflict score | Detect when a new word challenges a current structural analysis | Unexpectedness, ungrammaticality, and processing difficulty are not identical |
| Re-ranking and revision cost | Support repair after a garden-path interpretation | Reanalysis can occur in parallel with ongoing processing |
| Thresholding | Decide when evidence is strong enough to prefer, revise, or close a structure | Real comprehension may use graded confidence rather than a fixed threshold |

## 7. Worked example: “The dog chased the cat”

This example demonstrates the algorithm's structural operations without completing the later semantic-composition stage.

1. **“The”** arrives. The parser activates determiner-related category information and predicts that a noun phrase may be forming. The phrase is opened but remains incomplete.
2. **“dog”** arrives. Noun-category evidence is high. The parser combines `The` and `dog` into a noun phrase, with `dog` as its head and `The` as its determiner.
3. The noun phrase can now be considered a candidate subject phrase because of its position and the sentence's grammatical expectations. The parser still awaits the clause predicate.
4. **“chased”** arrives. Verb-category evidence is high. The parser opens a verb phrase and links the subject noun phrase to the verb using a grammatical subject relation.
5. The verb's grammatical frame and the current language pattern make an object phrase a plausible continuation. This is a syntactic expectation, not yet a complete conceptual event.
6. **“the”** arrives. A second determiner and noun phrase are expected; a new noun phrase is opened.
7. **“cat”** arrives. It completes the second noun phrase. The parser links this noun phrase to `chased` as its grammatical object.
8. At the sentence boundary, the parse can be consolidated as a clause containing a subject noun phrase and a verb phrase; the verb phrase contains the verb and object noun phrase.
9. The parser passes this syntactic representation to semantic composition. Semantic composition can then use the lexical meanings and grammatical relations to construct the sentence proposition.

## 8. Worked ambiguity example: “I saw the man with the telescope”

The prepositional phrase “with the telescope” has at least two possible grammatical attachments:

- **Verb-phrase attachment:** the telescope is associated with the seeing event.
- **Noun-phrase attachment:** the man is described as having the telescope.

The parser should build and retain both structures when the preceding words do not distinguish them. It can represent their current probability or score separately. Later context, prosody, or additional grammatical material may favour one attachment. The algorithm must not invent certainty merely because one analysis was initially ranked first.

This case illustrates why incremental parsing requires a parse forest, alternative dependency links, and the ability to revise or select a structure as more evidence becomes available.

## 9. What this point does not do

To preserve the sequence in the source document, this syntactic-parsing algorithm does **not** implement:

- retrieval of the detailed meanings of words beyond the lexical information delivered by Part 2;
- construction of a full proposition such as CHASE(dog, cat);
- assignment of conceptual event roles such as agent, patient, instrument, or experiencer, except where such labels are being used merely as grammatical placeholders;
- combining predicates and arguments into sentence-level semantic meaning;
- integrating the completed sentence with prior discourse, the physical scene, or speaker identity;
- resolving sarcasm, irony, indirect requests, conversational implicatures, or other pragmatic meanings;
- constructing the final enriched situation model;
- long-term storage or decision-making based on the proposition.

Syntactic expectations can interact with lexical and semantic information during real comprehension. The separation here is an organizational choice so that each source point can be converted into a distinct, testable algorithmic stage.

## 10. Biological cautions and open questions

1. **The brain is not a conventional parser.** A tree, dependency graph, chart, parse forest, Bayesian score, or beam search is a useful computational representation. It is not evidence that cortical tissue literally stores these structures in software-like form.
2. **Neural localization is distributed.** Left inferior frontal, posterior temporal, and anterior temporal regions are all implicated in syntactic processing, but their functions overlap and vary with the construction, task, language, and analysis method.
3. **Phrase structure and dependency structure are theoretical descriptions.** Neural findings that align different regions with different structural descriptors are informative, but they do not prove that the brain separately runs two complete formal grammars.
4. **The role of left IFG is debated.** Some accounts emphasize syntactic computation or unification; others assign more importance to controlled retrieval, selection, or domain-general demands. A biologically inspired model should not hard-code a single disputed interpretation as fact.
5. **ERP components are not one-to-one labels for mental steps.** ELAN, LAN, and P600 effects vary across studies and can be influenced by factors beyond syntax. The P600 is often associated with reanalysis or repair but is not exclusively syntactic.
6. **Prediction is probabilistic.** A highly expected continuation is not necessarily the only grammatical one. People can process rare, surprising, or temporarily difficult structures.
7. **Working memory is not merely a stack.** Long-distance dependencies can produce interference, but neural maintenance likely involves distributed activation and interactions among several systems.
8. **Language-specific grammar matters.** A parser for English cannot simply be reused unchanged for Arabic or other languages. Word order, case, agreement, gender, number, clitics, morphology, and constituent structure must be represented in language-specific ways.
9. **Speech and reading provide different cues.** Prosody is available in spoken language, whereas punctuation and visual layout support written-language parsing. The two modalities converge on many language processes but should not be treated as identical at the input stage.
10. **Parsing and interpretation overlap.** Semantic and contextual information can influence syntactic preferences before the final parse is complete. The staged algorithm is a practical decomposition, not a claim of strict neural seriality.

## 11. References for Part 3

These references support the broad neural framing and the distinction between phrase-structure, dependency, prediction, and reanalysis. They supplement the user's Library source file.

1. **Distinguishing Syntactic Operations in the Brain: Dependency and Phrase-Structure Parsing.** *Neurobiology of Language* (published online 2021; indexed in 2023). The fMRI study compared neural responses to phrase-structure and dependency-structure descriptors. It reported stronger phrase-structure alignment in left posterior superior temporal gyrus and stronger dependency-structure alignment in left inferior frontal gyrus and left anterior temporal pole. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/37213416/) · [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10174354/).
2. **Neural networks for sentence comprehension and production: An ALE-based meta-analysis of neuroimaging studies.** This meta-analysis reviews fronto-temporal networks involved in sentence processing and discusses models in which word-category information supports initial structure formation, thematic-role assignment, and subsequent repair. [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6597252/).
3. **The Neurobiology of Language Beyond Single Words.** *Annual Review of Linguistics* (2014). Reviews syntactic and semantic unification and reports a dorsal/ventral gradient in left inferior frontal and posterior temporal cortex, with dorsal regions more associated with syntactic processing in the reviewed studies. [Article](https://www.annualreviews.org/content/journals/10.1146/annurev-neuro-071013-013847).
4. **Time is of the Essence: A Review of Electroencephalography (EEG) and Event-Related Brain Potentials (ERPs) in Language Research.** Reviews ELAN, LAN, and P600 findings, including the variability and interpretive debates surrounding these components. [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC5693972/).
5. **Language-specific neural dynamics extend syntax into the time domain.** Reports MEG evidence consistent with early predictive structure building and later responses related to syntactic surprisal or revision of predicted structures. [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11750093/).
6. **Language structure in the brain: A fixation-related fMRI study of syntactic surprisal in reading.** Reports higher activity for words with greater syntactic surprisal in left inferior frontal gyrus and, less robustly, left anterior superior temporal regions. [Article](https://www.sciencedirect.com/science/article/pii/S1053811916001592).

---

**End of Part 3.** The next distinct point in the source sequence is **semantic composition — combining retrieved word meanings according to the syntactic structure to build a basic sentence-level proposition**. It has not been added yet.


# Point 4 — Semantic composition: combine lexical meanings into a sentence-level proposition

## 1. Objective and scope

### Objective

Construct a structured meaning from the lexical concepts delivered by lexical access (Part 2) and the grammatical relationships delivered by syntactic parsing (Part 3). The output should represent not merely which concepts occurred, but how those concepts relate: who or what participated in an event, what happened, to whom or what it happened, and which properties or modifiers constrain the event or entities.

For example, the words in **“The dog chased the cat”** should not remain as three unrelated concepts (`DOG`, `CHASE`, `CAT`). The system should build a relational representation such as:

```text
CHASE(
    agent = DOG,
    patient = CAT
)
```

This basic proposition is different from a bag of word meanings. Reversing the noun phrases changes the roles and therefore changes the proposition:

```text
CHASE(agent = DOG, patient = CAT)
CHASE(agent = CAT, patient = DOG)
```

The two representations contain the same lexical concepts but describe different events. The algorithm must preserve this relational information.

### Input contract

The semantic-composition system receives an incremental bundle from the previous stages:

1. **Lexical candidates:** one or more possible lemmas/senses for each recognized word, with confidence or activation information where available.
2. **Lexical semantic information:** concept features, semantic categories, predicate frames, possible arguments, and modifier properties associated with each candidate.
3. **Syntactic structure:** phrase constituents, dependency links, grammatical functions, agreement information, and alternative parses if ambiguity remains.
4. **Morphological information:** tense, aspect, number, person, case, voice, gender, and other language-specific markings that can affect interpretation.
5. **Current processing state:** the partial representation already built from earlier words in the same sentence.

The input may be incomplete because comprehension proceeds word by word. A verb may arrive before all its arguments; a noun phrase may be only partly recognized; or a later word may force the parser to reconsider a provisional attachment.

### Output contract

The output is a **basic compositional semantic representation** containing, as applicable:

- predicate or relation identity;
- entity/concept representations;
- argument slots and their fillers;
- thematic or semantic roles (agent, patient, experiencer, stimulus, recipient, instrument, location, source, goal, and others when licensed);
- modifier/property relations;
- event or state structure;
- negation, quantification, tense, aspect, modality, and reference operators when their meanings have been retrieved and compositionally licensed;
- unresolved slots or alternative readings when the sentence has not supplied enough information;
- confidence or uncertainty values for competing interpretations.

The output is passed to the next source point: **contextual and pragmatic integration**. This part does not independently infer a speaker's hidden intention, construct a rich causal story, resolve sarcasm, or invent facts that are absent from the sentence.

### Important separation of stages

Semantic composition is not simply another name for lexical access or syntactic parsing:

- **Lexical access** asks: *Which word form was recognized, and which lexical meanings are activated?*
- **Syntactic parsing** asks: *How are the words grammatically grouped and related?*
- **Semantic composition** asks: *Given those meanings and grammatical relations, what structured proposition do they express together?*
- **Contextual/pragmatic integration** asks: *How does that proposition fit the discourse, physical situation, background knowledge, and likely speaker intention?*

The stages interact in the brain, but this division keeps the algorithm's responsibilities testable and prevents Part 4 from absorbing the following point.

## 2. Biological architecture: networks that may contribute

No single brain region should be treated as a self-contained “semantic-composition module.” The evidence points to a distributed and interacting left-weighted fronto-temporo-parietal language/semantic network. Different experiments and tasks emphasize different nodes, and the functions assigned to each region remain debated.

### 2.1 Left temporal cortex, including middle and posterior temporal regions

Temporal cortical regions contribute to access to lexical-semantic information and to combining or integrating semantic information across words. The left posterior middle temporal gyrus (pMTG) is often discussed in relation to semantic unification, controlled retrieval, and the construction of sentence-level meaning. Temporal regions can represent information about event participants and their relations, rather than merely the identities of individual words.

A useful computational interpretation is that temporal populations provide semantic content and help maintain or transform the relational information required by a developing proposition. This does not mean that the pMTG alone stores a complete sentence representation.

### 2.2 Left anterior temporal lobe (ATL)

The left ATL has been implicated in combining lexical concepts into more complex meanings, including simple modifier–noun phrases. Some models treat anterior temporal cortex as a convergence or integration region for conceptual information across modalities. Other accounts emphasize its role in semantic representation more broadly. The exact contribution of ATL to composition, as distinct from semantic access or conceptual representation, is still actively investigated.

For this algorithm, ATL can be modeled as supporting the integration of constituent conceptual representations into a more coherent combined representation. The algorithm should not assume that all sentence-level composition is completed in ATL.

### 2.3 Left inferior frontal gyrus (LIFG)

The LIFG is implicated in several language and semantic operations, including semantic control, selection among competing meanings, and the unification of retrieved lexical information. It can help constrain how retrieved concepts are combined and support the resolution of competing or weakly compatible combinations.

A practical model assigns LIFG-related processing a control-and-combination role: it helps select task-relevant semantic information, coordinate combination with syntactic structure, and handle competition or increased demands. It is not justified to identify LIFG with one single mathematical operator or to claim that every composition operation happens there.

### 2.4 Angular gyrus (AG) and inferior parietal regions

The angular gyrus has been associated with aspects of semantic integration, relational meaning, event knowledge, and the coherence of conceptual combinations. Some studies report its involvement in basic phrase-level combination; others suggest that its contribution depends on the type of semantic information, task, or level of representation being tested.

In the algorithm, AG can be treated as one node in a distributed network that helps integrate conceptual and relational information into a coherent representation. The model does not assume that AG is a dedicated proposition-storage location.

### 2.5 Distributed fronto-temporo-parietal interaction

Compositional meaning can be represented across a broad network. A sentence-level representation may include lexical, relational, event-specific, and role-specific information. The frontal, temporal, and parietal regions can therefore be modeled as exchanging information recurrently rather than passing a completed message down a one-way chain.

The anatomical mapping above is functional and approximate. Neural activity in a region is not proof that the region performs only the operation assigned to it here. The same region may participate in several operations, and different people may show different activation patterns.

## 3. Core data structures used by the algorithm

The following are computational representations for making the proposed processing explicit. They are not claims that neurons contain literal programming-language records.

### 3.1 Lexical semantic record

Represent each activated lexical candidate as a semantic record:

```text
LexicalMeaning = {
    token_id,
    lemma,
    sense_id,
    semantic_features,
    semantic_type,
    predicate_frame,
    argument_slots,
    modifier_properties,
    grammatical_features,
    activation_or_confidence
}
```

Example records, simplified:

```text
DOG:
    semantic_type = ANIMAL
    features = {animate, canine, physical_entity}

CAT:
    semantic_type = ANIMAL
    features = {animate, feline, physical_entity}

CHASE:
    semantic_type = EVENT_PREDICATE
    frame = CHASE(agent, patient)
    constraints = {
        agent: animate_or_agentive_entity,
        patient: chaseable_entity_or_target
    }
```

A predicate frame is a structured description of the kinds of participants and relations associated with a predicate. The example is schematic; actual lexical frames are more detailed and can vary with a word's sense, language, and construction.

### 3.2 Syntactic-to-semantic interface record

For each syntactic constituent, preserve an interface record:

```text
Constituent = {
    syntactic_node_id,
    category,
    head_token,
    grammatical_function,
    dependency_relation,
    semantic_candidate_ids,
    morphological_markers,
    attachment_confidence
}
```

The interface is important because grammatical functions are not identical to semantic roles. A grammatical subject is not always an agent. For instance, in “The window was broken by the child,” the grammatical subject `the window` is semantically the patient, not the agent. In “Maya fears thunder,” the subject is an experiencer. The semantic algorithm must use the predicate's argument frame, voice, construction, and grammatical markings to map syntax onto semantic roles.

### 3.3 Proposition graph

Store the developing meaning as a graph:

```text
PropositionGraph = {
    entities: [...],
    events_or_states: [...],
    predicates: [...],
    role_links: [...],
    property_links: [...],
    operators: [...],
    unresolved_slots: [...],
    candidate_readings: [...]
}
```

Nodes represent entities, events, states, or concepts. Edges represent semantic relations such as `AGENT_OF`, `PATIENT_OF`, `MODIFIES`, `LOCATED_AT`, or `HAS_PROPERTY`. A graph preserves who-did-what-to-whom information that would be lost by merely adding word vectors together.

### 3.4 Partial proposition state

Because words arrive incrementally, the system must be able to hold incomplete representations:

```text
CHASE(
    agent = DOG,
    patient = UNRESOLVED
)
```

This is a legitimate partial state after “The dog chased” and before the object is known. The algorithm should not fabricate the patient or prematurely declare the proposition complete.

## 4. Detailed algorithmic sequence

### Step 1 — Receive the lexical-semantic candidates and the current syntactic structure

**Purpose:** Establish the exact materials available for composition at the current word or phrase boundary.

1. Receive the new lexical candidate or candidates from Part 2.
2. Receive the current syntactic parse or parse alternatives from Part 3.
3. Retrieve semantic records for the heads and relevant dependents already present in the sentence.
4. Keep candidate senses separate when lexical access has not yet resolved ambiguity.
5. Preserve morphological information that may influence the interpretation of argument roles or operators.
6. Link each candidate to its source token and syntactic node so later updates can revise the correct part of the representation.
7. If the word or phrase is not yet semantically identified, mark the corresponding slot as pending rather than inserting a guessed concept.

**Possible mathematical operations:**

- **Indexing:** map token identifiers to lexical and syntactic records.
- **Weighted candidate representation:** maintain a candidate set \(C_i = \{(s_k,p_k)\}\), where \(s_k\) is a candidate sense and \(p_k\) its current activation or confidence.
- **Normalization:** where candidate weights are meaningful probabilities, calculate \(p_k = a_k / \sum_j a_j\). This is a modeling convenience; neural activation magnitudes are not automatically probabilities.
- **State merge:** combine the new input with the previous partial state while preserving the distinction between old and new evidence.

**Biological interpretation:** Distributed neural populations associated with word meaning and grammatical structure are already active. New activity changes the current network state; the brain does not need to wait until the entire sentence has been heard before beginning composition.

### Step 2 — Activate the semantic features needed for the current combination

**Purpose:** Retrieve the meaning components that are relevant to this particular composition, rather than indiscriminately combining every association connected to a word.

1. For each candidate lexical sense, activate its semantic features and conceptual relations.
2. Distinguish relatively stable features from context-sensitive or weakly associated features.
3. For predicates, activate the relevant event or state frame: the kind of relation expressed and the participant slots that may be required.
4. For nouns, activate concept-category information and properties relevant to the current relation.
5. For adjectives, adverbs, and other modifiers, activate the property or event-modification information they contribute.
6. Preserve multiple feature patterns if multiple senses remain possible.

**Possible mathematical operations:**

- **Distributed feature vector:** represent a concept as \(\mathbf{x}=[x_1,x_2,\ldots,x_n]\), where dimensions represent learned semantic features or latent population patterns.
- **Feature gating:** use a relevance vector \(\mathbf{g}\) to select features for the current operation: \(\mathbf{x}_{rel}=\mathbf{g}\odot\mathbf{x}\), where \(\odot\) is elementwise multiplication.
- **Weighted activation:** \(a_j = \phi(\sum_i w_{ij}x_i - \theta_j)\), where \(w_{ij}\) models connection strength, \(\theta_j\) a threshold, and \(\phi\) a nonlinear activation function.
- **Similarity:** cosine similarity or a learned compatibility score can approximate the overlap between concept features and a predicate's requirements.

**Biological interpretation:** Semantic concepts are associated with distributed neural activity rather than necessarily being stored as one isolated symbol. Feature selection can be approximated by selective activation, inhibition, and recurrent interaction among populations. The vector notation is an engineering description of population states, not a literal neural data format.

### Step 3 — Identify the semantic head that determines the current relation or property

**Purpose:** Determine which lexical element supplies the principal semantic operation for the local phrase or clause.

1. Read the phrase or clause head from the syntactic structure.
2. Determine whether the head denotes an event predicate, state predicate, entity, property, relation, or operator.
3. If it is a verb, retrieve its event or state frame and its possible argument positions.
4. If it is an adjective, retrieve the property it contributes and the type of entity it can modify.
5. If it is a noun used relationally (for example, “friend of”), retrieve its relational frame where available.
6. If the structure is a noun compound, determine which constituent acts as the semantic head and which constituent modifies or classifies it; preserve alternative readings if the relation is genuinely ambiguous.

**Possible mathematical operations:**

- **Type classification:** estimate \(P(type\mid features, syntactic\_position)\).
- **Frame lookup:** retrieve a predicate schema \(F_p = (r_1,r_2,\ldots,r_m)\), where each \(r_i\) is a semantic argument role.
- **Arity check:** count the argument slots specified by the selected sense, while allowing optional arguments and construction-specific changes.
- **Head selection:** use the syntactic parse to choose a local semantic operator; this is a symbolic interface step, not a claim of explicit symbolic instructions in the cortex.

**Biological interpretation:** Temporal semantic representations and frontal/temporal language operations jointly make the relevant predicate or property information available. A computational model can represent this as retrieving a frame, but the brain may implement it through distributed activation and learned relational patterns.

### Step 4 — Create or update semantic variables for the entities and events

**Purpose:** Give the emerging representation distinct placeholders for the concepts participating in the sentence.

1. For each referential noun phrase, create or update an entity variable.
2. Attach the currently activated concept/sense to that variable.
3. If the noun phrase contains a determiner, numeral, possessive, or quantifier, retain its operator information for composition.
4. For an event predicate, create an event variable when the representation benefits from explicit event identity.
5. Preserve identity links when syntax indicates that two expressions refer to the same discourse entity (for example, a pronoun and its antecedent) only to the extent the preceding stages have supplied a candidate link. Full discourse-level reference resolution belongs to contextual integration.
6. If the constituent is incomplete, create a provisional variable and mark its status as unresolved.

**Possible mathematical operations:**

- **Variable allocation:** create a unique symbol \(x_i\) for each participant candidate and \(e_j\) for each event candidate.
- **Identity constraint:** represent a known identity relation as \(x_i=x_j\), while maintaining alternatives if identity is uncertain.
- **Feature assignment:** attach the entity's feature state \(\mathbf{f}(x_i)\) to its variable.
- **Set-valued representation:** for a plural or quantified noun phrase, represent a set or generalized quantifier rather than forcing it into a single individual variable.

**Biological interpretation:** The use of variables is a formal way to preserve distinct conceptual participants and the possibility of binding relations. Neural systems can maintain participant-specific information through distributed patterns, selective attention, and recurrent interactions. The brain is not known to allocate variables in the same explicit manner as a programming language.

### Step 5 — Map grammatical functions onto semantic argument roles

**Purpose:** Use the syntactic structure to determine which concept fills which semantic role in the predicate frame.

1. Identify the predicate's syntactic dependents.
2. Read their grammatical functions, case, agreement, prepositions, voice, and constructional markers.
3. Retrieve the predicate's lexical argument frame.
4. Map each syntactic constituent to a semantic role using the predicate frame and grammatical cues.
5. Do not assume `subject = agent` or `object = patient` in every sentence.
6. For passive constructions, map the grammatical subject to the patient/theme role when the verb's frame and voice indicate it.
7. For experiencer predicates, assign experiencer and stimulus roles where licensed.
8. For ditransitive predicates, distinguish recipient, goal, and theme roles as appropriate to the verb and construction.
9. If the mapping remains uncertain, store more than one role assignment with separate confidence values.

**Example:**

```text
Maya feared the storm.

Syntax:
    subject = Maya
    verb = feared
    object = the storm

Semantic roles:
    FEAR(
        experiencer = MAYA,
        stimulus = STORM
    )
```

The grammatical subject is the experiencer, not an agent performing an action.

**Possible mathematical operations:**

- **Role compatibility:** compute a score \(S(r_i,x_j)\) between a candidate filler and a role slot.
- **Constraint matching:** test whether morphological and syntactic cues license a role assignment.
- **Assignment optimization:** where several arguments and roles compete, choose a high-scoring compatible mapping, for example \(\arg\max_{\pi}\sum_i S(r_i,x_{\pi(i)})\), subject to predicate-frame constraints.
- **Soft assignment:** retain a distribution over role mappings rather than committing when evidence is incomplete.
- **Feature unification:** merge compatible grammatical and semantic features; mark incompatible features as conflicts or unresolved alternatives.

**Biological interpretation:** A left fronto-temporal network is implicated in mapping sentence structure and lexical information into relational meaning. Role assignment can be described as constraint satisfaction, but no evidence requires the brain to run a literal assignment optimizer. Population-level activity and learned dependencies can produce graded role preferences.

### Step 6 — Combine the predicate with its licensed arguments

**Purpose:** Build the core proposition by binding each participant to the correct role of the predicate.

1. Retrieve the semantic predicate and its open argument slots.
2. Locate the entity variables that the syntactic-to-semantic mapping assigned to each role.
3. Bind each available variable into the appropriate slot.
4. Preserve unfilled slots if the sentence has not yet supplied the corresponding participant.
5. Keep the role labels attached to the participant; do not collapse them into an unordered set of concepts.
6. Create the proposition node once the predicate and enough argument information are available.
7. If multiple predicate senses or role mappings remain active, build alternative proposition candidates rather than blending them into one incoherent meaning.

**Example:**

```text
The dog chased the cat.

Predicate frame:
    CHASE(agent, patient)

Argument binding:
    agent  <- DOG
    patient <- CAT

Composed proposition:
    CHASE(agent = DOG, patient = CAT)
```

**Possible mathematical operations:**

- **Function application (formal semantic model):** if a verb denotes a function over arguments, combine it with the meaning of its argument. For example, \(\llbracket chase\rrbracket(\text{dog},\text{cat})\) yields the proposition that the dog chased the cat. This notation describes the compositional rule, not a confirmed neural calculation.
- **Role–filler binding:** represent the result as \(P(e)=CHASE(e)\) together with \(AGENT(e,dog)\) and \(PATIENT(e,cat)\).
- **Tensor-product binding (one possible computational analogy):** \(\mathbf{z}=\sum_r \mathbf{u}_r\otimes\mathbf{f}_r\), where \(\mathbf{u}_r\) encodes a role and \(\mathbf{f}_r\) its filler. This preserves role/filler pairings in a structured vector space. It is a model family, not an established literal operation of cortical tissue.
- **Relational graph edge creation:** add directed links from the event node to each entity node under the corresponding semantic role.
- **Incremental binding update:** \(G_t=G_{t-1}\cup\{(e,r,x)\}\), where the new tuple records that entity \(x\) fills role \(r\) in event \(e\).

**Biological interpretation:** Semantic composition requires more than simultaneous activation of the words. Their meanings must become integrated in a way that preserves relational structure. A distributed neural code can represent event identity and participant roles through patterns of activity and connectivity. Whether this is best explained by binding, unification, feature intersection, structured relational coding, or a combination remains an empirical question.

### Step 7 — Compose modifiers and properties with the concepts they modify

**Purpose:** Incorporate adjectives, adverbs, possessives, and other modifiers into the correct semantic constituent.

1. Use the syntactic parse to identify the constituent a modifier attaches to.
2. Retrieve the modifier's semantic property or operator.
3. Determine whether it restricts an entity category, contributes a temporary property, changes the event manner, or modifies another relation.
4. Combine the modifier with the head concept while preserving the head's identity.
5. If the modifier has multiple interpretations, keep the relevant alternatives distinct.
6. Attach the result to the proposition graph at the correct node.

**Examples:**

```text
The red car stopped.

CAR(x)
RED(x)
STOP(e)
THEME(e, x)
```

```text
The dog chased the cat quickly.

CHASE(e)
AGENT(e, DOG)
PATIENT(e, CAT)
MANNER(e, QUICKLY)
```

**Possible mathematical operations:**

- **Feature intersection:** combine a noun's features with a modifier's restrictions: \(F_{combined}=F_{noun}\cap F_{modifier}\), where set intersection is a simplified model.
- **Graded feature combination:** \(\mathbf{h}=\phi(W_n\mathbf{x}_{noun}+W_m\mathbf{x}_{modifier}+\mathbf{b})\), a learned transformation that can encode the combined phrase.
- **Predicate modification:** apply a property function to an entity or event representation.
- **Constraint narrowing:** eliminate candidate referents that do not satisfy the modifier, or reduce their compatibility scores.
- **Composition vs. association test:** distinguish a modifier that changes the meaning of a head from a mere co-occurrence association between the two words.

**Biological interpretation:** Modifier–head combination has been investigated with MEG and fMRI paradigms. The left ATL, frontal language regions, temporal cortex, and parts of the parietal semantic network have all been implicated in different forms of composition. Neural responses vary with semantic plausibility, task demands, and whether a phrase is genuinely compositional or merely associative.

### Step 8 — Compose operators and grammatical meaning with lexical content

**Purpose:** Ensure the proposition reflects meaning contributed by function words and grammatical morphology, not only content words.

1. Retrieve the semantic contribution of negation, quantifiers, determiners, tense, aspect, modality, and relevant grammatical particles.
2. Use syntax and morphology to determine the scope of each operator.
3. Apply the operator to the appropriate predicate, phrase, or proposition.
4. Distinguish an operator's semantic contribution from pragmatic inferences that might later be drawn from it.
5. Retain ambiguity when scope or attachment is not uniquely determined.

**Examples:**

```text
The dog did not chase the cat.

NOT(CHASE(agent = DOG, patient = CAT))
```

```text
Every dog chased a cat.

FOR_EVERY(x, DOG(x), EXISTS(y, CAT(y) AND CHASE(x, y)))
```

The second example has scope issues and possible readings that depend on the language and context. The expression above is only one simplified formalization.

**Possible mathematical operations:**

- **Logical operator application:** apply \(\neg P\) for negation, or quantifier operators such as \(\forall\) and \(\exists\).
- **Scope composition:** construct operator nesting, such as \(\neg\forall x\,P(x)\) versus \(\forall x\,\neg P(x)\), when more than one reading is possible.
- **Temporal indexing:** represent tense/aspect using event time, reference time, and speech time variables.
- **Modal weighting or accessibility:** represent possibility/necessity as an operator over candidate situations, while recognizing that modal semantics is richer than a scalar probability.
- **Feature propagation:** pass person, number, tense, or aspect information to the relevant event or referent node.

**Biological interpretation:** The meanings of grammatical morphemes and function words are part of linguistic-semantic processing. Neural populations may encode these contributions in interaction with lexical and structural information. Formal logic is useful for specifying the output, but there is no basis for claiming that the brain literally executes symbolic quantifier instructions in the same way as a theorem prover.

### Step 9 — Evaluate semantic compatibility without prematurely performing pragmatic inference

**Purpose:** Detect whether the combined expression is compositionally compatible or whether it contains a semantic tension that should be passed onward for further processing.

1. Compare each argument's semantic type and features with the predicate's requirements.
2. Compare modifiers with the concepts they modify.
3. Detect strong incompatibilities, weak preferences, and cases that are unusual but still interpretable.
4. Distinguish grammatical well-formedness from semantic plausibility.
5. Do not reject a sentence merely because it describes an unlikely event; unusual content may be literally possible, metaphorical, or contextually licensed.
6. Do not silently invent a missing cause, intention, or event to make the sentence seem coherent.
7. Mark unresolved semantic conflict for the next contextual/pragmatic stage.

**Example:**

```text
The stone admired the painting.
```

The sentence may be grammatically well formed, but the predicate `ADMIRE` typically selects an experiencer capable of a mental attitude. The noun `STONE` creates a strong semantic-type conflict under a literal reading. The system should mark the mismatch, not automatically conclude that the sentence is impossible. Personification or a fictional context could make it interpretable.

**Possible mathematical operations:**

- **Type compatibility:** compute \(C(type(x),role_r)\), a graded measure of fit between a candidate filler and a role.
- **Selectional preference score:** \(S=\mathbf{v}_{filler}^{T}W_r\mathbf{v}_{role}\), or another learned compatibility function.
- **Constraint penalty:** \(E_{sem}=\sum_i \lambda_i(1-C_i)\), where larger values indicate greater mismatch.
- **Anomaly score:** compare a candidate composition with the distribution of learned semantic combinations, for example using negative log-likelihood \(-\log P(combination)\).
- **Uncertainty:** preserve alternatives and measure their entropy \(H=-\sum_k p_k\log p_k\) when candidate readings are probabilistically represented.

**Biological interpretation:** Semantic anomalies and difficult combinations can modulate neural responses and recruit control mechanisms. A compatibility score is a computational abstraction of graded semantic fit. It should not be confused with a proven neural probability calculation or with the full pragmatic explanation of an unusual utterance.

### Step 10 — Maintain competing compositional readings when necessary

**Purpose:** Avoid forcing a single interpretation when lexical ambiguity, structural ambiguity, or scope ambiguity remains unresolved.

1. Generate a separate proposition candidate for each viable combination of word senses, role assignments, modifier attachments, or operator scopes.
2. Keep the candidates' constituent links explicit so later evidence can revise one candidate without corrupting another.
3. Use current grammatical and lexical evidence to rank the candidates provisionally.
4. Avoid combining incompatible candidate meanings into one blended representation.
5. Retain a bounded number of alternatives only as an engineering approximation; the brain's actual representational dynamics need not resemble a software beam search.
6. Pass the candidate set and its uncertainty forward if the sentence or context has not yet disambiguated it.

**Example:**

For an ambiguous word such as `bank`, the lexical stage may provide `FINANCIAL_INSTITUTION` and `RIVER_EDGE`. Semantic composition can create distinct candidates if the surrounding predicate permits both. A later word or broader discourse may favour one candidate.

**Possible mathematical operations:**

- **Candidate scoring:** \(Score(M_k)=\alpha L_k+\beta S_k+\gamma C_k\), where terms can represent lexical, syntactic, and compositional compatibility. The weights are model parameters, not known neural constants.
- **Bayesian update:** \(P(M_k\mid E)\propto P(E\mid M_k)P(M_k)\), for a model that explicitly tracks probabilities.
- **Normalization:** convert candidate scores to a probability-like distribution when that is justified by the model.
- **Entropy:** quantify unresolved uncertainty and preserve multiple high-probability candidates.
- **Pruning:** remove candidates only when evidence makes them sufficiently implausible; do not prune solely because one is less frequent.

**Biological interpretation:** Competition and parallel activation are plausible descriptions of how alternatives can coexist and interact. The algorithm's candidate list is a transparent computational device; it should not be taken as evidence for a literal list of parse objects stored in the brain.

### Step 11 — Update the partial proposition incrementally as each new constituent arrives

**Purpose:** Make composition continuous across the sentence rather than postponing it until the final word.

1. When a new lexical item arrives, determine which open phrase, predicate, event, or modifier it can contribute to.
2. Update the relevant semantic feature state.
3. If a new predicate appears, create its event/state frame and open argument slots.
4. If an argument appears, bind it to a licensed role if enough information is available.
5. If a modifier appears, attach it to its syntactic-semantic target.
6. If a function word or inflection appears, update the operator or grammatical feature representation.
7. Preserve unresolved roles and candidate readings until further words provide evidence.
8. Recompute only the affected portions of the semantic graph where possible, while allowing broader revision if the parse changes.

**Example progression:**

```text
After "The dog":
    ENTITY(x1, DOG)
    NP(x1, determiner = THE)

After "chased":
    EVENT(e1, CHASE)
    AGENT(e1, x1)
    PATIENT(e1, unresolved)

After "the cat":
    ENTITY(x2, CAT)
    PATIENT(e1, x2)

Current proposition:
    CHASE(e1)
    AGENT(e1, x1)
    PATIENT(e1, x2)
```

**Possible mathematical operations:**

- **Recurrent state update:** \(M_t=U(M_{t-1},L_t,S_t)\), where \(M_t\) is the semantic state after time step \(t\), \(L_t\) the current lexical-semantic input, and \(S_t\) the current syntactic information.
- **Incremental graph update:** add, modify, or remove nodes and edges as evidence changes.
- **Temporal integration:** \(h_t=\alpha h_{t-1}+\beta x_t\), as a simple model of retaining previous information while integrating new input.
- **Prediction error:** \(\delta_t=x_t-\hat{x}_t\), where \(\hat{x}_t\) is a model's expected semantic contribution. This is a generic predictive-processing analogy, not proof that the brain computes a literal vector subtraction.
- **Selective revision:** update a local subgraph when only one role or attachment changes.

**Biological interpretation:** EEG/MEG and neuroimaging research supports the idea that semantic composition is dynamic and can begin before a sentence is complete. The system repeatedly updates its current interpretation as words arrive. A computer-style state-update equation makes the idea operational but does not identify the precise synaptic mechanism.

### Step 12 — Check whether the basic proposition is sufficiently complete

**Purpose:** Determine whether the sentence has supplied the components needed for a basic sentence-level meaning.

1. Check whether the main predicate or state has been identified.
2. Check which arguments are required by that predicate sense and construction.
3. Distinguish required arguments from optional adjuncts.
4. Verify that the arguments already supplied have role assignments.
5. Verify that any modifiers or operators that affect the main proposition have been attached or explicitly marked unresolved.
6. If required information is missing because the sentence is incomplete, retain the partial proposition.
7. If the sentence has ended and an argument is genuinely unexpressed but licensed by the language or construction, represent it as implicit/unspecified rather than inventing its identity.
8. Do not require every event frame to contain every conceivable role; predicates differ in their argument structure.

**Possible mathematical operations:**

- **Slot completion ratio:** \(R=\frac{n_{filled}}{n_{required}}\), a rough engineering diagnostic rather than a biological measure.
- **Constraint satisfaction:** test whether all obligatory role and operator constraints are met.
- **Confidence threshold:** declare a proposition provisionally complete only when the relevant role assignments and predicate identity exceed chosen model thresholds.
- **Missingness encoding:** represent an unknown argument as `UNSPECIFIED`, `UNRESOLVED`, or `NOT_EXPRESSED`, and do not conflate these states.

**Biological interpretation:** The brain can understand elliptical or incomplete utterances, so semantic completeness cannot be reduced to a rigid checklist. The algorithm should report unresolved structure while allowing the next context stage to interpret ellipsis or omission.

### Step 13 — Consolidate the compositional result into a structured proposition

**Purpose:** Produce an explicit, internally consistent representation that can be used by the next processing stage.

1. Consolidate the selected or still-competing predicate and argument structures.
2. Preserve semantic role labels and links between event and participants.
3. Include modifier, negation, quantifier, tense, aspect, and modality contributions that were compositionally licensed.
4. Preserve the provenance of each relation: which word, syntactic relation, and semantic frame supported it.
5. Keep uncertainty and ambiguity explicit.
6. Avoid including inferred intentions, causal explanations, or unstated facts unless they were part of the literal compositional content.
7. Export the proposition graph and its confidence information to contextual/pragmatic integration.

**Possible mathematical operations:**

- **Graph serialization:** convert the internal graph into a canonical node-and-edge representation.
- **Consistency check:** ensure that a role assignment does not contradict the selected predicate frame or the same token's current analysis.
- **Confidence propagation:** combine the confidence of the predicate, arguments, and role mapping. A simple product is one option under strong independence assumptions; a learned joint model is generally more realistic.
- **Representation compression:** remove duplicate links while retaining all meaning-bearing distinctions.

**Biological interpretation:** Sentence-level meaning is supported by interacting language and semantic networks and can be maintained in an active state as comprehension continues. The exact format of the brain's proposition representation is not settled; a graph or predicate–argument structure is an interpretable scientific model.

### Step 14 — Pass the result to contextual and pragmatic integration

**Purpose:** Hand off a clearly delimited basic proposition without prematurely performing the next stage's work.

1. Send the proposition graph and any alternative readings to the contextual/pragmatic system.
2. Include unresolved semantic conflicts and missing arguments.
3. Include the literal composition and its confidence separately from any later contextual interpretation.
4. Allow the next stage to compare the proposition with discourse history, physical context, speaker identity, social knowledge, and conversational expectations.
5. Do not mark a pragmatic interpretation as though it were a literal predicate–argument relation constructed by this stage.

**Output example:**

```text
Literal compositional proposition:
    CHASE(
        agent = DOG,
        patient = CAT
    )

Composition status:
    predicate identified = yes
    required roles filled = yes
    unresolved lexical alternatives = none
    semantic anomaly = none detected

Deferred to contextual/pragmatic integration:
    Does the event fit the preceding discourse?
    Is the speaker being ironic or figurative?
    Are causal or social inferences required?
```

**Possible mathematical operations:**

- **Interface packaging:** bundle graph, candidate readings, uncertainty, and provenance into a structured message.
- **Information-preserving transformation:** ensure the next stage receives the same role structure and alternatives without flattening them into a word list.
- **Stage boundary flag:** record which operations have been performed and which remain pending; this is a software architecture safeguard, not a neural computation.

## 5. Mathematical model of composition at multiple levels

The algorithm above uses a collection of possible mathematical descriptions. They answer different questions and should not be conflated into one universal equation.

### 5.1 Predicate–argument composition

A simple formal semantic account represents a transitive verb as a relation that takes two arguments:

\[
\llbracket chase \rrbracket = \lambda x.\lambda y.\,CHASE(x,y)
\]

Applying the arguments `dog` and `cat` yields:

\[
\llbracket chase\ the\ cat\rrbracket
=\lambda x.\,CHASE(x,cat)
\]

\[
\llbracket the\ dog\ chased\ the\ cat\rrbracket
=CHASE(dog,cat)
\]

This makes the compositional rule explicit: the whole meaning depends on the meanings of the parts and the structure that combines them. It is a formal semantic description. It does not establish that cortical neurons perform lambda calculus.

### 5.2 Role–filler representation

An event-centered representation can be written as:

\[
M=\{CHASE(e),\ AGENT(e,dog),\ PATIENT(e,cat)\}
\]

This representation preserves relational structure. It also allows a proposition to be enriched later without changing its core event identity:

\[
M'=M\cup\{MANNER(e,quickly)\}
\]

The graph form is useful for modeling thematic roles and for preventing a bag-of-words representation from confusing “dog chased cat” with “cat chased dog.”

### 5.3 Distributed-vector and feature-based models

A neural-network model may represent each lexical concept as a distributed vector:

\[
\mathbf{x}_{dog},\quad \mathbf{x}_{chase},\quad \mathbf{x}_{cat}
\]

A learned composition function combines them using structural information:

\[
\mathbf{h}_{sentence}=f(\mathbf{x}_{dog},\mathbf{x}_{chase},\mathbf{x}_{cat},\mathbf{s})
\]

where \(\mathbf{s}\) encodes the relevant syntactic/role structure. The function \(f\) could be a recurrent network, tensor operation, gated network, or another learned transformation.

A plain sum such as \(\mathbf{x}_{dog}+\mathbf{x}_{chase}+\mathbf{x}_{cat}\) is generally insufficient when it discards who fills which role. If the same three vectors are added for both “dog chased cat” and “cat chased dog,” the two sentences could become indistinguishable. The model must therefore encode argument order or role binding.

### 5.4 Compatibility and semantic selection

A simple model of role compatibility is:

\[
C(x,r)=\sigma(\mathbf{x}^{T}W_r\mathbf{q}_r+b_r)
\]

where:

- \(\mathbf{x}\) represents the candidate filler;
- \(\mathbf{q}_r\) represents the role's semantic constraints;
- \(W_r\) is a learned transformation;
- \(b_r\) is a bias;
- \(\sigma\) maps the score to a bounded range.

This can rank possible role assignments. In a biological model it should be regarded as an abstract measure of compatibility, not a direct recording of a neuron calculating a sigmoid.

### 5.5 Neural integration and recurrent activity

A simplified population model may describe activity as:

\[
\mathbf{a}_{t+1}=\phi(W\mathbf{a}_t+U\mathbf{x}_t-\boldsymbol{\theta})
\]

where:

- \(\mathbf{a}_t\) is the current neural-population activity pattern;
- \(W\) describes recurrent interactions;
- \(\mathbf{x}_t\) is the incoming lexical/structural information;
- \(U\) maps input to the population;
- \(\boldsymbol{\theta}\) represents thresholds or offsets;
- \(\phi\) is a nonlinear activation function.

This captures the general idea that new input changes an existing distributed state through recurrent interactions. It is not a detailed biophysical simulation; a more biologically constrained model would include spiking, synaptic time constants, inhibition, conduction delays, and local circuit architecture.

### 5.6 Prediction error and revision

If the current composition predicts a likely semantic continuation \(\hat{\mathbf{x}}_t\), one can define a mismatch:

\[
\boldsymbol{\delta}_t=\mathbf{x}_t-\hat{\mathbf{x}}_t
\]

Large mismatch can trigger reassessment of the candidate combination, but it does not automatically mean the sentence is false or ungrammatical. Predictive-processing equations are model proposals; the exact neural implementation and its relation to semantic composition remain open research questions.

## 6. End-to-end pseudocode

```text
FUNCTION SEMANTIC_COMPOSITION(lexical_candidates, syntactic_state, partial_state):

    state = COPY(partial_state)

    FOR each new lexical item or completed constituent:
        candidates = GET_LEXICAL_SEMANTIC_RECORDS(item)
        structure  = GET_RELEVANT_SYNTACTIC_RELATIONS(syntactic_state, item)

        IF candidates are unresolved:
            STORE_PENDING_SEMANTIC_CANDIDATES(state, item, candidates)
            CONTINUE

        FOR each candidate sense:
            semantic_features = ACTIVATE_RELEVANT_FEATURES(candidate)
            head_or_modifier = CLASSIFY_SEMANTIC_FUNCTION(candidate, structure)

            IF candidate introduces a predicate:
                frame = RETRIEVE_PREDICATE_FRAME(candidate)
                event = CREATE_OR_UPDATE_EVENT_NODE(state, candidate)
                CREATE_OPEN_ARGUMENT_SLOTS(event, frame)

            IF candidate introduces an entity:
                entity = CREATE_OR_UPDATE_ENTITY_NODE(state, candidate)

            IF candidate is a modifier:
                target = FIND_SYNTACTIC_AND_SEMANTIC_TARGET(structure)
                COMPOSE_MODIFIER(state, candidate, target)

            IF candidate is a function word or grammatical operator:
                target = DETERMINE_OPERATOR_SCOPE(structure)
                APPLY_OR_STAGE_OPERATOR(state, candidate, target)

        FOR each predicate with available arguments:
            role_candidates = MAP_SYNTAX_TO_SEMANTIC_ROLES(
                predicate, structure, lexical_candidates
            )
            role_candidates = CHECK_VOICE_CASE_AND_MORPHOLOGY(role_candidates)
            role_candidates = SCORE_ROLE_FILLER_COMPATIBILITY(role_candidates)
            BIND_AVAILABLE_ARGUMENTS(state, role_candidates)

        CHECK_TYPE_AND_SELECTIONAL_COMPATIBILITY(state)
        PRESERVE_UNRESOLVED_SLOTS(state)
        PRESERVE_COMPETING_READINGS_IF_NEEDED(state)
        UPDATE_ONLY_AFFECTED_PROPOSITION_SUBGRAPHS(state)

    IF sentence_boundary_reached:
        CHECK_PREDICATE_AND_ARGUMENT_COMPLETENESS(state)
        CHECK_MODIFIERS_AND_OPERATOR_ATTACHMENTS(state)
        CONSOLIDATE_PROPOSITION_GRAPH(state)

    OUTPUT = PACKAGE_LITERAL_PROPOSITION_AND_UNCERTAINTY(state)
    RETURN OUTPUT  // send to contextual/pragmatic integration
```

The pseudocode is a transparent software decomposition. Real neural activity is more continuous, overlapping, distributed, and recurrent than this sequence of named function calls.

## 7. Worked examples

### Example A — Basic transitive proposition: “The dog chased the cat”

**Available input:** lexical concepts `DOG`, `CHASE`, and `CAT`; syntactic parse identifies the subject NP, verb, and object NP.

1. Retrieve `DOG` and `CAT` as entity concepts.
2. Retrieve the `CHASE` predicate frame with agent and patient roles.
3. Create event variable \(e_1\), dog variable \(x_1\), and cat variable \(x_2\).
4. Map the subject NP to the agent role for this active construction and verb sense.
5. Map the object NP to the patient role.
6. Bind the role fillers and create the proposition graph.

Result:

```text
CHASE(e1)
AGENT(e1, x1)      where x1 = DOG
PATIENT(e1, x2)    where x2 = CAT
```

This is the literal compositional result. Whether the event is expected, surprising, important, or ironic belongs to later processing.

### Example B — Same words, different semantic roles: “The cat chased the dog”

The predicate is the same, but the syntactic-to-semantic mapping is different:

```text
CHASE(e1)
AGENT(e1, CAT)
PATIENT(e1, DOG)
```

The example tests whether the algorithm preserves role binding instead of simply combining the word representations.

### Example C — Passive voice: “The cat was chased by the dog”

The grammatical subject is `the cat`, but the semantic role is patient. The `by` phrase supplies the agent:

```text
CHASE(e1)
PATIENT(e1, CAT)
AGENT(e1, DOG)
VOICE(e1, PASSIVE)
```

This verifies that grammatical subject and semantic agent are not treated as identical.

### Example D — Adjective composition: “The red boat moved”

The adjective composes with the noun, while the noun phrase fills an argument role of the event predicate:

```text
BOAT(x1)
RED(x1)
MOVE(e1)
THEME(e1, x1)
```

The adjective narrows or enriches the boat concept. Its meaning is not merely associated with the word `boat` because the two words happened to co-occur.

### Example E — Semantic mismatch without pragmatic repair: “The idea drank water”

The verb `DRINK` normally expects an animate or drinking-capable agent and a liquid consumed as patient/theme. The subject `IDEA` produces a strong type mismatch under a literal reading.

The composition algorithm should output a candidate with a mismatch flag, for example:

```text
DRINK(e1)
AGENT(e1, IDEA)
PATIENT(e1, WATER)
SEMANTIC_COMPATIBILITY = LOW
```

It should not autonomously invent a metaphorical interpretation such as “the idea was absorbed” or infer a poetic intention. Such a repair belongs to a later interpretation process.

### Example F — A sentence with a partially implicit argument: “Maya ate”

The verb `EAT` may occur without an overt object in a grammatically complete sentence. The system should not mark the proposition as automatically malformed because one possible argument is not spoken. A reasonable output is:

```text
EAT(e1)
AGENT(e1, MAYA)
THEME(e1, NOT_EXPRESSED_OR_UNSPECIFIED)
```

This makes a distinction between a syntactically licensed unexpressed argument and a role that is still pending because the utterance has not finished.

## 8. Mathematical operations inventory

| Operation | What it contributes | Status of the mathematical description |
|---|---|---|
| Distributed feature activation | Makes relevant concept features available | Population-vector model; not a literal symbolic vector in tissue |
| Weighted synaptic integration | Combines excitatory and inhibitory influences | Grounded in neural physiology, although the network-level equation is simplified |
| Nonlinear activation | Models thresholds and saturation in population responses | Common neural-computation abstraction |
| Predicate–argument function application | Specifies how word meanings compose into a proposition | Formal semantic model, not evidence for literal lambda calculus in the brain |
| Role–filler binding | Preserves who did what to whom | Computational representation of relational structure |
| Tensor-product binding | Encodes role/filler pairings in a vector space | Candidate mathematical model; not established as the brain's unique mechanism |
| Feature intersection or gated combination | Combines modifiers and head concepts | One family of compositional models; the best operation depends on the construction |
| Type and selectional compatibility | Detects semantic fit and mismatch | Graded computational approximation |
| Bayesian or probabilistic updating | Maintains uncertainty over candidate readings | Normative/modeling tool; not proof of explicit neural probability calculation |
| Entropy | Quantifies uncertainty among competing readings | Mathematical diagnostic for a model |
| Graph operations | Create event, participant, and property relations | Explicit symbolic model of proposition structure |
| Recurrent state updating | Integrates incoming words with the developing meaning | Consistent with recurrent neural processing at a broad level; exact mechanism remains under study |
| Prediction error | Quantifies deviation from an expected continuation | Predictive-processing analogy; interpretation is debated |
| Constraint satisfaction | Tests whether predicate roles, modifiers, and operators are compatible | Computational formalization, not a single known cortical algorithm |

## 9. Biological cautions and unresolved issues

1. **There is no single agreed neural algorithm for semantic composition.** Research supports distributed semantic and language networks, but the exact division of labor among ATL, pMTG, IFG, and AG remains debated.
2. **Semantic composition is not the same as semantic association.** Two words can be strongly associated without one modifying or serving as an argument of the other. Neural models and experiments must distinguish relational composition from co-occurrence or association.
3. **The role of ATL is under active investigation.** Some MEG work finds early LATL involvement in phrase-level composition; other work emphasizes temporal, parietal, or frontal contributions depending on the operation and task. No single region should be treated as the sole composition center.
4. **LIFG is functionally heterogeneous.** It has been linked to semantic control, competition resolution, selection, unification, and other language processes. A model assigning it a single fixed role would overstate the evidence.
5. **Angular gyrus findings are task-sensitive.** Some studies identify its contribution to basic semantic combination or event-related meaning, while others distinguish it more clearly from compositional representations. Its role should remain parameterized rather than assumed to be identical in every sentence.
6. **Semantic role labels are model constructs.** Agent, patient, experiencer, and theme are useful linguistic/semantic categories. Neural representations may encode relational information in distributed ways that do not resemble a list of named slots.
7. **Grammatical functions and semantic roles differ.** Subject, object, agent, patient, experiencer, and theme are not interchangeable. Voice, case, construction, and predicate meaning influence the mapping.
8. **Composition is incremental and recurrent.** A strictly linear sequence is a convenient algorithmic explanation. Neural processing may activate several operations in overlapping time windows and revisit earlier interpretations.
9. **Literal plausibility and contextual interpretation are distinct.** A semantically anomalous sentence may be metaphorical, fictional, ironic, or contextually licensed. This part records the tension; later stages handle the broader interpretation.
10. **The language network is not the entire semantic system.** Word meanings draw on conceptual knowledge distributed across the brain. The distinction between language-selective processing and general conceptual knowledge is useful but does not imply complete isolation.
11. **Neuroimaging is often correlational.** Activation during a task does not by itself prove that a region is necessary for a particular computational operation. Lesion, stimulation, electrophysiology, and carefully controlled behavioral evidence are needed to strengthen causal claims.
12. **Language-specific structures matter.** Role mapping and operator scope differ across languages. Arabic case marking, agreement, clitics, word order variation, and morphology, for example, cannot be modeled by simply copying an English-only parser's assumptions.

## 10. References for Part 4

The following sources provide empirical and theoretical grounding for the neural discussion. They do not establish every mathematical operation above as a literal biological calculation; the equations are explicitly presented as candidate models.

1. **Lyu, B., Chuang, Y.-Y., & Yoshida, M. (2019). “Neural dynamics of semantic composition.”** *Proceedings of the National Academy of Sciences*. This study examines the dynamic combination of lexical meanings during spoken-language comprehension, including verb–direct-object composition, and discusses interactions among left inferior frontal, posterior middle temporal, and left anterior temporal regions. [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6800340/).
2. **“Differential contributions of left-hemispheric language regions to basic semantic composition” (2021).** *Brain Communications*. A controlled fMRI study separating aspects of conceptual combination and semantic plausibility, reporting contributions from left-hemisphere semantic regions including angular gyrus and a broader fronto-temporal network. [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC7910266/).
3. **“Dissociable contributions of frontal and temporal brain regions to basic semantic composition” (2021).** *Brain*. Lesion-behavior mapping found differential associations between anterior inferior frontal and anterior-to-mid temporal damage and performance on anomalous two-word phrases, supporting the idea that frontal control and temporal semantic processing make distinguishable contributions. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/34159319/) · [Article](https://academic.oup.com/braincomms/article/3/2/fcab090/6248084).
4. **“Neural encoding of semantic structures during sentence production” (2024).** *Cerebral Cortex*. Uses encoding models to investigate word-level and relational/event-specific sentence representations across a distributed left frontal–parietal–temporal network; reports event-specific relational information in left inferior frontal regions. The article also summarizes prior findings on LATL, ventromedial prefrontal cortex, angular gyrus, and temporal regions in different forms of composition. [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11666472/).
5. **“Compositionality in the semantic network: a model-driven representational similarity analysis” (2025).** *Cerebral Cortex*. Reanalyzes fMRI data from multiple studies and reports compositional representations in left inferior frontal gyrus and left middle superior temporal sulcus, with distinctions between semantic and compositional representations across regions. This is a recent model-driven result and should be interpreted within its methods and task range. [Article](https://academic.oup.com/cercor/article/35/8/bhaf246/8250582).
6. **Hagoort, P. (2014). “Nodes and networks in the neural architecture for language: Broca's region and beyond.”** *Current Opinion in Neurobiology*. Reviews the role of frontal and temporal language networks in unification and the interaction of retrieved lexical information with syntactic and semantic structure. [Review context and related work](https://www.annualreviews.org/content/journals/10.1146/annurev-neuro-071013-013847).
7. **Binder, J. R., et al. (2009). “Where Is the Semantic System? A Critical Review and Meta-Analysis of 120 Functional Neuroimaging Studies.”** *Cerebral Cortex*. A broad meta-analysis relevant to the distributed nature of semantic representation and processing. [PubMed](https://pubmed.ncbi.nlm.nih.gov/19008458/).

---

**End of Part 4.** The next distinct point in the source sequence is **contextual/pragmatic integration**: comparing the developing proposition with discourse history, the physical situation, speaker identity, world knowledge, and conversational expectations. That next point has not been implemented in this installment.


---

# Point 5 — Contextual and pragmatic integration: evaluate the developing proposition against discourse, situation, and speaker intentions

## 1. Objective and boundary of this point

### Objective

Take the developing propositional representation produced by semantic composition and interpret it in relation to information that is not contained in the sentence's literal word meanings alone. The brain must combine the current linguistic message with relevant discourse history, the physical or imagined scene, background knowledge, speaker identity and likely goals, conversational conventions, and the listener's common ground with the speaker.

The target is not simply to decide whether a sentence is grammatically well formed or whether its words fit together. It is to determine how the current sentence contributes to a coherent interpretation of what is happening, what the speaker means, and what the listener should infer or do next.

### Inputs

1. **Current propositional candidate:** a structured representation from Point 4, including predicates, participants, semantic roles, modifiers, polarity, tense/aspect where available, and unresolved elements.
2. **Discourse history:** propositions and events from earlier sentences or conversational turns, their referents, temporal order, causal links, and current topic.
3. **Situation model:** the perceived or imagined setting, objects, people, events, spatial relations, and currently relevant affordances.
4. **Speaker model:** the speaker's identity, knowledge, beliefs, goals, emotional state, conversational role, and characteristics inferred from voice, expression, or prior behavior.
5. **Common ground:** facts the listener estimates are mutually known or assumed by speaker and listener.
6. **World and event knowledge:** learned facts, scripts, causal expectations, social conventions, and knowledge about how objects and agents usually behave.
7. **Conversation state:** the question under discussion, the current task, turn-taking state, recent speech acts, and the listener's own goals.
8. **Uncertainty from earlier stages:** lexical, syntactic, referential, and compositional alternatives that have not yet been fully resolved.

### Output

An **enriched interpretation state** containing:

- the literal proposition, retained rather than erased;
- links to relevant earlier propositions and scene entities;
- resolved or still-ambiguous references;
- contextually licensed causal, temporal, social, and explanatory relations;
- possible implicatures, presuppositions, indirect speech acts, or figurative interpretations when warranted;
- a representation of likely speaker intention, with confidence and alternatives;
- unresolved conflicts or missing information that still require attention;
- an updated discourse/situation model that can be used by working memory, later sentences, memory, and decisions.

### Boundary: what this point does not do

This point does **not** replace the core language system or rebuild every word from scratch. It receives a developing linguistic representation and coordinates it with broader information. It also does not mean that every ordinary sentence triggers elaborate social reasoning. Many sentences are understood with little extra pragmatic work because their literal meaning fits the current context. A larger inference process is recruited when a reference is unclear, a causal or coherence link is missing, the message is underspecified, expectations conflict, or the speaker appears to communicate more than the literal sentence states.

The source document describes the left frontotemporal language network as central to linguistic retrieval and composition, with additional systems such as medial prefrontal cortex (mPFC), temporoparietal junction (TPJ), and domain-general frontoparietal control regions contributing when the task requires broader inference. This is a useful core/periphery distinction, but not an absolute anatomical wall: semantic knowledge is distributed, and the exact contributions of regions vary with task and individual.

---

## 2. High-level algorithm

```text
ALGORITHM CONTEXTUAL_PRAGMATIC_INTEGRATION(
    current_proposition,
    discourse_state,
    situation_state,
    speaker_model,
    common_ground,
    world_knowledge,
    conversation_goal,
    unresolved_alternatives
):

    1. Read the current proposition without discarding its literal structure.
    2. Retrieve the most relevant parts of recent discourse and long-term knowledge.
    3. Build a unified, time-sensitive context state.
    4. Identify the entities, events, relations, and speech acts currently in focus.
    5. Resolve references by matching linguistic candidates to discourse and scene entities.
    6. Generate expectations about likely continuations and relations.
    7. Compare the proposition with the discourse, situation, common ground, and world model.
    8. Estimate coherence, relevance, plausibility, and contextual fit.
    9. Detect missing links, contradictions, surprising details, or underspecification.
   10. Decide whether ordinary integration is sufficient or additional inference is needed.
   11. Generate candidate bridging, causal, temporal, explanatory, or social inferences.
   12. Generate candidate pragmatic interpretations when literal meaning may not express the speaker's goal.
   13. Evaluate each candidate against evidence, speaker goals, common ground, and conversational norms.
   14. Preserve multiple interpretations when evidence is insufficient.
   15. Select, combine, or defer candidate interpretations using a confidence-sensitive policy.
   16. Reconcile the selected interpretation with the original proposition.
   17. Reanalyse earlier assignments when new contextual evidence requires it.
   18. Update the situation model, discourse graph, and speaker model.
   19. Maintain the enriched representation in working memory.
   20. Pass forward the literal proposition, licensed inferences, uncertainty, and updated context.

    RETURN enriched_interpretation_state
```

The steps are an explanatory decomposition. In actual comprehension, retrieval, prediction, comparison, and updating can overlap in time and feed back into one another.

---

## 3. Detailed algorithmic steps and possible mathematical operations

### Step 1 — Receive the current proposition while preserving its literal content

**Purpose**

Take the output of semantic composition as a structured candidate, not as an unstructured sentence string. The representation might include `APOLOGIZE(e1)`, `AGENT(e1, LISA)`, plus a time and a possible but unexpressed recipient. The literal content must remain available even if a later pragmatic interpretation adds information or suggests irony.

**Algorithmic operations**

1. Copy or reference the current predicate–argument structure.
2. Mark which roles are expressed, omitted, underspecified, or ambiguous.
3. Preserve tense, aspect, negation, modality, quantification, and other operators supplied by earlier language processing.
4. Attach confidence values to unresolved lexical, syntactic, and semantic alternatives.
5. Create a new discourse-processing record that points to the original proposition rather than overwriting it.

**Possible mathematical operations**

- **Structured copying / pointer assignment:** preserve identity of the source proposition while creating a context-integration workspace.
- **Vector or tensor concatenation:** combine proposition features with a context-processing state, for example `z_t = [p_t ; c_t]`. This is a model-level representation, not a claim that the brain concatenates literal arrays.
- **Masking:** mark absent or unresolved arguments so the system does not accidentally treat missing information as known: `m_j = 1` for known fields and `m_j = 0` for unresolved fields.
- **Confidence propagation:** if a role assignment has confidence `q`, retain that confidence rather than converting it prematurely to a certain fact.

**Biological interpretation**

The left frontotemporal language network continues to support the linguistic-semantic representation. Working-memory systems keep the currently relevant content available. Contextual processing is added to this representation; it does not require the literal sentence meaning to disappear.

**Checkpoint**

The system can state what the sentence literally says and can distinguish that from what it might imply.

### Step 2 — Retrieve relevant discourse history and background knowledge

**Purpose**

Recover only the parts of the past that are likely to matter. Context is not an unlimited undifferentiated memory store. The listener prioritizes recent, active, semantically related, causally relevant, and task-relevant information.

**Algorithmic operations**

1. Search the recent discourse buffer for entities, events, topics, and unresolved questions related to the current proposition.
2. Retrieve older episodes or semantic knowledge if the current sentence contains cues that make them relevant.
3. Retrieve event schemas and social scripts that can support likely relations, such as apologizing, requesting, giving, travelling, or eating.
4. Search the physical scene representation for currently visible or recently attended entities.
5. Retrieve speaker-specific information when it is relevant, such as a known preference, prior commitment, or established conversational goal.
6. Rank retrieved records and retain a manageable subset for active processing.

**Possible mathematical operations**

- **Similarity-based retrieval:** compute a match score between a query derived from the current proposition and stored records: `r_i = sim(q, m_i)`.
- **Recency decay:** give recent events greater initial accessibility: `w_i = exp(−Δt_i / τ)`, where `Δt_i` is elapsed time and `τ` controls decay.
- **Relevance weighting:** combine recency, semantic overlap, causal relevance, and current-task relevance: `R_i = a r_i + b d_i + c k_i + d g_i`.
- **Top-k selection:** keep the best-ranked records, subject to capacity and task demands.
- **Associative spreading activation:** allow a currently active entity or concept to increase activation of linked memories: `A_j(t+1) = decay(A_j(t)) + Σ_i w_ij A_i(t)`.
- **Thresholded retrieval:** retrieve a record when its activation or match score exceeds a context-sensitive threshold.

These equations formalize retrieval and prioritization. They do not prove that neurons explicitly compute a cosine similarity, exponential clock, or top-k database operation. Neural memory retrieval is distributed and can involve multiple cortical and medial temporal structures.

**Potential neural contributors**

- Temporal semantic regions contribute stored conceptual and event knowledge.
- Angular gyrus and posterior cingulate/precuneus may participate in integrating information into broader event or situation representations, depending on task.
- Medial temporal lobe structures can contribute when episodic details or relations among events must be retrieved.
- Frontal systems may guide controlled retrieval when the relevant knowledge is weak, competing, or not automatically activated.

**Checkpoint**

The system has a small, ranked working context rather than attempting to compare the current sentence with every memory.

### Step 3 — Construct a unified, time-sensitive context state

**Purpose**

Represent the different sources of context in a form that can jointly influence interpretation while retaining their distinct origins. Discourse history, a visible scene, speaker beliefs, and general world knowledge are not interchangeable evidence.

**Algorithmic operations**

1. Create separate fields for discourse, situation, speaker, common ground, world knowledge, and conversation goal.
2. Attach a time index and source label to each item.
3. Represent which entities and events are currently active, which are backgrounded, and which are no longer relevant.
4. Preserve conflicting information rather than averaging it away.
5. Record the reliability and accessibility of each contextual source.
6. Link the current proposition to candidate context elements without forcing a final interpretation.

**Possible mathematical operations**

A useful model is a structured context state:

`C_t = {D_t, S_t, M_t, G_t, W_t, Q_t}`

where:

- `D_t` = discourse history and topic state;
- `S_t` = physical or imagined situation;
- `M_t` = speaker/mental-state model;
- `G_t` = estimated common ground;
- `W_t` = world and event knowledge;
- `Q_t` = question under discussion and current conversational goal.

Operations may include:

- **Feature integration:** construct a context vector from selected fields, while retaining provenance tags.
- **Gated weighting:** `c_t = Σ_j g_j c_j`, where each gate `g_j` depends on relevance, reliability, and task demands.
- **Temporal tagging:** attach timestamps or event-order variables to contextual propositions.
- **Graph construction:** represent people, objects, events, and relations as nodes and edges.
- **Conflict preservation:** store incompatible candidate beliefs as alternatives, rather than forcing a single average value.

**Biological interpretation**

A unified situation model need not reside in one anatomical location. It can be a coordinated pattern across language, memory, conceptual, social-cognitive, and control networks. The source emphasizes that left language areas and social/reasoning networks are distinct but can interact during rich discourse comprehension.

**Checkpoint**

The context is available as a structured, source-sensitive state with time and relevance information.

### Step 4 — Identify focus, discourse topic, entities, and event structure

**Purpose**

Determine what the conversation is currently about and which entities or events are most likely to be referred to or elaborated. The same sentence may have different interpretations depending on whether a person, object, problem, or event is currently in focus.

**Algorithmic operations**

1. Identify the current discourse topic and question under discussion.
2. Detect recently mentioned entities and assign discourse prominence.
3. Track whether each entity is the current topic, grammatical subject, recent object, or focus of contrast.
4. Identify event frames from earlier propositions, such as an apology, argument, journey, purchase, or accident.
5. Detect whether the new proposition continues, explains, contrasts with, or changes the topic.
6. Preserve multiple potential centers of attention when the discourse supports more than one.

**Possible mathematical operations**

- **Salience scoring:** `Salience(e) = a·recency + b·frequency + c·topic_status + d·syntactic_prominence + e·attentional_focus`.
- **Normalization:** convert raw scores to relative weights, for example `α_i = exp(s_i) / Σ_j exp(s_j)`.
- **State transition:** update a topic-state variable `T_t` using the current proposition and previous topic.
- **Graph centrality:** a model may estimate which event or entity has the strongest links to the active discourse graph.
- **Winner-take-most selection:** a highly salient entity can become the leading focus, while alternatives remain available.

The exact weighting of recency, grammatical prominence, and topic status is language- and discourse-dependent. No universal numeric formula is established for human reference tracking.

**Potential neural contributors**

Left temporal and parietal language regions support linguistic relations and discourse-level meaning. Posterior medial regions and angular gyrus may contribute to event/situation representation. Frontal and attentional systems may help maintain the active topic when several candidates compete.

**Checkpoint**

The processor has an estimate of the current focus and can use it in reference resolution and coherence evaluation.

### Step 5 — Resolve references by linking linguistic expressions to discourse or scene entities

**Purpose**

Determine which person, object, event, or proposition an expression refers to. Expressions may be pronouns (“he”, “it”), definite descriptions (“the dog”), names (“John”), demonstratives (“that”), or omitted arguments understood from context. Reference resolution is strongly context-sensitive and can remain uncertain.

**Algorithmic operations**

1. Extract each referential expression and its grammatical and semantic constraints.
2. Generate candidate antecedents from recent discourse and the current scene.
3. Remove candidates incompatible with number, person, gender features where relevant to the language, animacy, semantic type, or syntactic constraints.
4. Score remaining candidates by recency, topic prominence, grammatical role, semantic compatibility, event structure, and plausibility.
5. Consider whether the expression refers to an entity, an event, or an entire proposition.
6. Select a candidate only when evidence is sufficiently strong; otherwise preserve alternatives.
7. Update entity links and maintain the referential chain across sentences.

**Possible mathematical operations**

- **Candidate generation:** construct a finite set `A = {a_1, ..., a_n}`.
- **Constraint filtering:** set `score(a_i) = −∞` for candidates that violate hard constraints.
- **Weighted scoring:** `s_i = Σ_k w_k f_k(a_i, e)` for features such as recency, salience, agreement, semantic fit, and discourse role.
- **Softmax normalization:** `P(a_i | e,C) = exp(s_i) / Σ_j exp(s_j)`.
- **Bayesian updating:** `P(a_i | e,C) ∝ P(e | a_i,C) P(a_i | C)`.
- **Entropy:** `H(A) = −Σ_i P(a_i) log P(a_i)` measures how uncertain the current reference assignment remains.
- **Threshold / margin rule:** commit when the leading candidate exceeds a minimum confidence and its score is sufficiently separated from the second candidate.

**Biological interpretation**

Reference resolution is not necessarily handled by one dedicated “pronoun module.” It can recruit linguistic constraints, semantic knowledge, discourse memory, attention, and inferential processing. The left TPJ is discussed in the source as a region that can help integrate linguistic input with general world knowledge and local discourse information. mPFC and broader social-cognition regions may be relevant when resolving references requires reasoning about another person's beliefs or intentions.

**Checkpoint**

Every expression has either a high-confidence referent, a set of ranked possible referents, or an explicit unresolved status.

### Step 6 — Generate context-sensitive predictions and expectations

**Purpose**

Use the current context to anticipate which events, words, relations, or intentions are likely. Predictions can guide attention and facilitate processing when subsequent input matches them. They must remain revisable because unexpected but valid events occur.

**Algorithmic operations**

1. Use the current discourse topic and situation model to activate likely continuations.
2. Generate expectations about event participants, plausible causes, temporal order, and likely next actions.
3. Predict whether the current sentence is likely to be an explanation, a consequence, a contrast, a request, or a response.
4. If a new sentence is arriving word by word, let context influence lexical and structural expectations before the sentence is complete.
5. Keep a distribution over plausible continuations rather than a single compulsory prediction.

**Possible mathematical operations**

- **Conditional probability:** `P(x_next | C_t, p_1...p_t)` represents the probability of a possible continuation given the context.
- **Prediction distribution:** maintain `P(H_i | C_t)` across candidate events or interpretations.
- **Surprisal:** `I(x) = −log P(x | C_t)` quantifies how unexpected an observed element is under the model.
- **Prediction error:** `δ_t = observed_t − expected_t`, where subtraction is a model-specific difference between compatible representations.
- **Precision weighting:** `PE_t = π_t · δ_t`, where `π_t` represents confidence in the prediction. A strongly expected event can produce a larger mismatch when violated; an uncertain expectation should have less influence.
- **Dynamic updating:** adjust expectations as each new word or cue arrives.

Surprisal and prediction-error equations are useful computational descriptions. They do not imply that the brain literally computes logarithms or stores an explicit probability table. The relationship between predictive-processing theories and particular ERP effects remains debated.

**Potential neural contributors**

The core language network supports linguistic expectations. Temporal semantic regions contribute learned event and conceptual regularities. Attention-related systems may change processing priorities when a prediction is violated. Context-sensitive effects can be observed in electrophysiological responses such as the N400, but the N400 is not a direct meter of one explicit probability computation.

**Checkpoint**

The system has a graded set of expectations and can recognize when actual input is less expected than the current context predicts.

### Step 7 — Compare the proposition with the local discourse

**Purpose**

Check whether the new proposition connects coherently to what has already been said. A sentence may be grammatically correct and literally interpretable but still seem disconnected, contradictory, repetitive, or incomplete in the current discourse.

**Algorithmic operations**

1. Search for entities shared with earlier propositions.
2. Search for relations that connect the new event to prior events: continuation, elaboration, cause, result, explanation, contrast, sequence, condition, or correction.
3. Check whether the current sentence answers or advances the question under discussion.
4. Detect contradictions with earlier claims, while allowing correction, quotation, uncertainty, fiction, or speaker disagreement.
5. Determine whether a relation is explicitly stated or must be inferred.
6. Assign a coherence status: high, acceptable, weak, conflicting, or unresolved.

**Possible mathematical operations**

- **Semantic similarity:** compare current proposition and preceding propositions using a similarity function. Similarity alone is insufficient because a coherent contrast may have low similarity.
- **Relation classification:** estimate probabilities for discourse relations `P(r | p_t, p_{t-1}, C)`.
- **Graph connectivity:** measure whether the new event has plausible edges to active discourse nodes.
- **Constraint satisfaction:** check whether temporal, referential, causal, and logical constraints can jointly hold.
- **Contradiction score:** calculate conflict between incompatible propositions, while tagging their sources and times.
- **Coherence function:** `Coherence(p_t,D_t) = Σ_r P(r | p_t,D_t) · utility(r)`.

**Biological interpretation**

The source describes left temporal language regions as active in discourse integration and reports sustained activity in left temporal and other regions when readers must establish causal links across clauses. It also reports bilateral medial frontal engagement in some bridging-inference conditions. These results support a network account rather than a single coherence calculator.

**Checkpoint**

The system knows whether the current proposition connects directly to the prior discourse or whether a missing relation needs to be inferred.

### Step 8 — Compare the proposition with the physical or imagined situation

**Purpose**

Evaluate the sentence against what is visible, remembered, or imagined about the current scene. A sentence may be linguistically coherent but inconsistent with the scene, or may make sense only when the physical situation supplies missing information.

**Algorithmic operations**

1. Identify scene entities and events currently represented.
2. Map linguistic participants to scene objects or people where appropriate.
3. Compare described properties, actions, locations, and temporal relations with the situation model.
4. Check physical and causal compatibility using world knowledge, but do not assume that unusual events are impossible.
5. Detect whether the sentence describes the present scene, a remembered event, a hypothetical situation, a counterfactual, or a future plan.
6. Maintain the distinction between an actual observation and an imagined or asserted event.

**Possible mathematical operations**

- **Cross-modal matching:** `match(e_linguistic, e_scene)` scores how well a described entity aligns with a perceived or remembered object.
- **Spatial transformation:** convert between viewpoint-relative and scene-relative coordinates when the task requires it.
- **Temporal alignment:** compare event intervals using ordering or overlap relations.
- **Constraint checking:** test properties such as object identity, location, containment, movement, and physical affordance.
- **Reliability weighting:** combine scene evidence and linguistic assertion according to source reliability and task.
- **Conflict preservation:** store `ASSERTED(p)` separately from `OBSERVED(not p)` if the sentence conflicts with perception; do not silently erase one source.

**Potential neural contributors**

Visual and auditory association areas contribute scene information; temporal and parietal conceptual networks help align language with knowledge. Right TPJ is described in the source as important for attentional reorienting to unexpected information and for broader context updating. Its contribution should not be reduced to a literal “scene mismatch detector.”

**Checkpoint**

The system can distinguish a proposition that fits the current scene from one that conflicts with it, is hypothetical, or requires additional context.

### Step 9 — Evaluate world knowledge, event plausibility, and causal compatibility

**Purpose**

Use general knowledge about objects, people, social behavior, and events to understand relations that are not explicitly stated. World knowledge supports rapid comprehension, but plausibility is not the same as truth: unusual events, metaphors, jokes, and fictional descriptions can be meaningful.

**Algorithmic operations**

1. Retrieve relevant event schemas and causal knowledge.
2. Check whether the predicate's participants and roles are typical or atypical for the event.
3. Estimate what causes, enables, prevents, or normally follows the described event.
4. Identify a missing causal or explanatory edge if it would make the discourse coherent.
5. Generate a small number of plausible alternatives when the literal account conflicts with strong background knowledge.
6. Retain the literal proposition and distinguish inferred causes from directly asserted causes.

**Possible mathematical operations**

- **Compatibility score:** `K(p,W) = Σ_j w_j feature_match_j`, with separate positive and negative evidence.
- **Bayesian causal inference:** `P(cause | effect, W) ∝ P(effect | cause,W)P(cause | W)`.
- **Abductive inference:** choose a hypothesis `h` that explains observed facts while penalizing unsupported assumptions: `Score(h) = fit(h,data) − λ·complexity(h)`.
- **Graph completion:** add a candidate causal or enabling edge to an event graph, then estimate its confidence.
- **Counterfactual simulation:** compare predicted outcomes under alternative conditions, such as whether an event would have occurred if a suspected cause were absent.
- **Minimum-description-length criterion:** prefer an explanation that provides good coherence with relatively few unsupported assumptions.

**Important distinction**

These operations are algorithmic analogies for inference. The brain may rely on distributed learned associations, structured event knowledge, recurrent interactions, and memory retrieval; it is not established that it literally solves a formal Bayesian network or runs a minimum-description-length routine.

**Potential neural contributors**

Left temporal and inferior parietal language-related regions may participate in integrating event knowledge with discourse. mPFC can become active during higher-level causal or social inference, as summarized in the source's discussion of discourse studies. Other regions can contribute depending on whether the inference is physical, autobiographical, social, or task demanding.

**Checkpoint**

The system has a set of plausible world-knowledge links, each marked as asserted, strongly supported, or merely possible.

### Step 10 — Estimate whether additional inference is required

**Purpose**

Avoid overthinking every sentence while ensuring that important gaps and conflicts are not ignored. The listener must decide whether the literal proposition and ordinary contextual integration are enough, or whether a more elaborate inference loop should be initiated.

**Algorithmic operations**

1. Collect mismatch signals from discourse, scene, world knowledge, and speaker expectations.
2. Assess whether the mismatch is meaningful or can be explained by an ordinary exception, a change of topic, uncertainty, or a different viewpoint.
3. Estimate whether an omitted relation is required for coherence.
4. Estimate whether the speaker's likely goal makes a nonliteral or indirect reading relevant.
5. Compare expected benefit of additional inference with the processing cost and risk of inventing unsupported meaning.
6. Either continue with a relatively literal interpretation or launch targeted inference.

**Possible mathematical operations**

- **Weighted conflict score:** `M = w_D M_D + w_S M_S + w_W M_W + w_G M_G + w_M M_M`, where each component measures a different type of mismatch.
- **Uncertainty aggregation:** combine uncertainty about reference, event relations, and intention without treating them as independent unless justified.
- **Decision threshold:** trigger extra inference when `M > θ`, where threshold `θ` varies with task demands, attention, and expected cost.
- **Expected value of information:** `EVI = expected_loss_without_inference − expected_loss_after_inference − inference_cost`.
- **Sequential evidence accumulation:** add evidence over time until the system reaches a sufficient-confidence boundary or the utterance/context changes.

**Biological interpretation**

The source describes additional mPFC/TPJ and multiple-demand engagement when inference or difficult context resolution is required, while the core language network remains active. This should be read as a graded change in network participation, not an all-or-none switch that turns entire brain systems on or off.

**Checkpoint**

The system either accepts a sufficiently coherent interpretation or records exactly what is unresolved and why more inference is needed.

### Step 11 — Generate bridging and causal inferences to fill discourse gaps

**Purpose**

A sentence or pair of sentences may omit a relation that a listener can infer. A bridging inference connects a new statement to a prior event, entity, or causal sequence, making the discourse coherent without requiring the speaker to state every link.

**Algorithmic operations**

1. Identify the current gap: missing participant, event relation, causal link, temporal sequence, or explanatory step.
2. Search retrieved event knowledge for common relations that could fill the gap.
3. Generate candidate links, such as `CAUSES`, `ENABLES`, `PRECEDES`, `PART_OF`, `OCCURS_AT`, or `ABOUT`.
4. Verify that each candidate is compatible with both the preceding and current propositions.
5. Prefer candidates that explain the discourse with the fewest unsupported assumptions.
6. Mark inferred links as inferred rather than explicitly spoken.
7. Preserve alternatives if more than one bridge is plausible.

**Possible mathematical operations**

- **Abductive ranking:** rank candidate inferences by explanatory fit and prior plausibility.
- **Graph path completion:** find plausible paths between current and prior discourse nodes.
- **Constraint propagation:** reject links that violate known time, role, identity, or causal constraints.
- **Bayesian model comparison:** estimate relative support for competing bridge hypotheses.
- **Confidence thresholding:** add a bridge to the active model only when it has adequate support; otherwise keep it tentative.

**Example: “Lisa apologized. Peter shrugged.”**

A listener may infer that Lisa apologized to Peter, because the two events are adjacent and the apology normally has a recipient. But the sentence does not entail this. Peter might have shrugged in response to something else. A careful algorithm should therefore represent:

```text
APOLOGIZE(e1)
AGENT(e1, LISA)
SHRUG(e2)
AGENT(e2, PETER)

CANDIDATE_BRIDGE:
RECIPIENT(e1, PETER)     confidence = context-dependent
```

The candidate bridge can be promoted if earlier discourse or scene information supports it. Without such support, it remains uncertain.

**Potential neural contributors**

The source summarizes Kuperberg et al. (2006) as finding sustained activity in left temporal language areas and bilateral superior medial prefrontal regions in some causal-bridging conditions. The result suggests participation of both language and broader reasoning systems; it does not identify a single inference circuit.

**Checkpoint**

Any bridging relation is explicitly tagged as inferred and accompanied by a confidence estimate or retained alternative.

### Step 12 — Generate pragmatic interpretations beyond literal sentence meaning

**Purpose**

Infer what a speaker is trying to communicate when the literal sentence alone is not the complete message. Pragmatic interpretation can involve conversational implicature, presupposition, indirect speech acts, irony, sarcasm, metaphor, understatement, or a contextually implied request.

**Algorithmic operations**

1. Retain the literal proposition as the baseline interpretation.
2. Identify a possible mismatch between literal content and the speaker's apparent goal, tone, situation, or conversational role.
3. Generate a small set of pragmatic hypotheses, such as request, warning, refusal, criticism, reassurance, joke, ironic reversal, or implied evaluation.
4. Check whether a candidate explains why the speaker chose this utterance at this point in the conversation.
5. Use prosody, facial expression, gesture, speaker identity, shared knowledge, and prior behavior as evidence when available.
6. Determine whether the literal interpretation remains plausible, is supplemented by an implicature, or is pragmatically adjusted.
7. Avoid assuming that a nonliteral reading is mandatory merely because one is possible.

**Possible mathematical operations**

- **Hypothesis generation:** create a candidate set `H = {literal, request, warning, irony, criticism, ...}`.
- **Bayesian inference of intention:** `P(I | U,C) ∝ P(U | I,C)P(I | C)`, where `U` is the utterance and `I` is a possible intention.
- **Multimodal evidence combination:** integrate lexical content, prosody, scene, facial expression, and discourse evidence with source-specific reliability.
- **Utility calculation:** estimate how well each candidate interpretation explains the speaker's communicative goal and the listener's task.
- **Pragmatic rationality models:** a formal model may approximate a listener as reasoning about what a cooperative speaker would likely say to convey a goal. Such Rational Speech Act (RSA)-style equations are useful models of pragmatic reasoning, not proof that the brain literally executes a nested speaker-listener probability program.

An illustrative RSA-style model can be written as:

`P_L(I | U,C) ∝ P_S(U | I,C) · P(I | C)`

where a modelled listener `L` infers intention `I` from utterance `U`, context `C`, and a modelled speaker `S`. The speaker model may choose utterances using a softmax over communicative utility:

`P_S(U | I,C) = exp(α · Utility(U,I,C)) / Σ_{U'} exp(α · Utility(U',I,C))`.

The parameter `α` controls how strongly the model favors high-utility utterances. These equations help specify a computational hypothesis; they are not direct measurements of neural firing.

**Example: “It’s cold in here.”**

Literal content: the room is cold. In a context where a window is open, the speaker is near the listener, and closing the window is possible, the listener may infer an indirect request to close it.

Candidate representation:

```text
LITERAL: COLD(ROOM)
CANDIDATE_INTENTION: REQUEST(CLOSE(LISTENER, WINDOW))
SUPPORT: open_window + listener_can_close + conversational relevance
STATUS: inferred request, not literal entailment
```

If the speaker is simply reporting the temperature to a technician, the request inference may be inappropriate. The same sentence does not have one fixed pragmatic meaning independent of context.

**Checkpoint**

The system can represent both what was said and what the speaker may have intended, without confusing the two.

### Step 13 — Infer speaker goals, beliefs, and perspective when social reasoning is needed

**Purpose**

Some utterances require the listener to reason about another person's knowledge, beliefs, intentions, emotions, or perspective. This is especially relevant for irony, sarcasm, deception, politeness, indirect requests, false beliefs, and socially meaningful omissions.

**Algorithmic operations**

1. Construct a provisional model of what the speaker knows and believes.
2. Estimate what the speaker wants the listener to know, believe, feel, or do.
3. Compare the speaker's presumed knowledge with the listener's knowledge and common ground.
4. Determine whether the speaker is likely to be sincere, mistaken, joking, ironic, deceptive, or strategically indirect, while avoiding premature commitment.
5. Use the speaker's goals to rank possible interpretations.
6. Update the speaker model when subsequent actions or utterances provide new evidence.

**Possible mathematical operations**

- **Belief-state representation:** represent candidate beliefs as probability distributions rather than certain facts.
- **Perspective separation:** maintain distinct states for `Belief_listener(p)` and `Belief_speaker(p)`.
- **Nested belief estimation:** a computational model may represent what the listener thinks the speaker believes, but the depth should be limited to what the task requires.
- **Inverse planning:** infer a likely goal from an observed utterance or action by evaluating which goal would make that action useful.
- **Social prediction error:** compare the speaker's subsequent behavior with the listener's current model of the speaker.
- **Bayesian model update:** revise the speaker model as evidence accumulates.

**Potential neural contributors**

The source identifies mPFC and TPJ as major components of the mentalizing/theory-of-mind network. These regions are recruited when the task requires inferences about other people's mental states or unstated meanings. TPJ contributions can differ by hemisphere and task: left TPJ is discussed in relation to contextual semantic integration, while right TPJ is also associated with attentional reorienting and social inference. The precise boundaries and functions are not absolute.

**Checkpoint**

The system maintains a provisional speaker-perspective model and uses it only when mental-state reasoning is relevant.

### Step 14 — Interpret irony, sarcasm, and apparent literal-context conflict

**Purpose**

Detect cases in which the literal proposition appears inconsistent with the speaker's tone, situation, shared knowledge, or apparent evaluation. A nonliteral interpretation may reverse or modify the literal polarity, but that should occur only when enough cues support it.

**Algorithmic operations**

1. Compute the literal interpretation.
2. Compare it with prosodic, facial, situational, and discourse cues.
3. Detect evidence of mismatch between literal evaluation and the speaker's likely emotional or communicative stance.
4. Generate candidate readings: sincere literal, ironic, sarcastic, joking, quoted, or intentionally misleading.
5. Evaluate whether the speaker expects the listener to recognize the mismatch.
6. Use common ground and speaker beliefs to discriminate among candidates.
7. If a nonliteral reading is sufficiently supported, attach it as a pragmatic interpretation while retaining the literal form.
8. If cues are weak, preserve ambiguity instead of forcing sarcasm.

**Possible mathematical operations**

- **Valence comparison:** compare the polarity of literal content `v_lit` with estimated speaker stance `v_spk`.
- **Mismatch measure:** `D = |v_lit − v_spk|`, adjusted by prosodic and contextual reliability.
- **Multicue integration:** `Score(H_i) = Σ_j w_j evidence_j(H_i)`.
- **Bayes factor:** compare support for ironic versus sincere readings: `BF = P(E | irony,C) / P(E | sincere,C)`.
- **Decision threshold:** choose irony only if posterior evidence is sufficiently strong and the pragmatic cost of a false reading is acceptable.

**Example: “Great job!” said in a scolding tone**

The words express positive evaluation, but the prosody and situation may indicate criticism. The system should not simply invert all positive phrases following any loud or negative-sounding voice. It should combine the words, tone, preceding event, speaker stance, and shared expectations. The result may be `CRITICISM` or `NEGATIVE_EVALUATION` with an ironic/sarcastic flag, while the literal positive wording remains represented.

**Potential neural contributors**

The source reports that irony and sarcasm can recruit mPFC and TPJ, alongside language regions and sometimes broader frontal/temporal areas. It also describes electrophysiological late positivities such as P600 in some pragmatic-anomaly tasks. These effects are not unique signatures of sarcasm; their interpretation depends on task and analysis.

**Checkpoint**

The system has an evidence-based pragmatic interpretation, not a simple rule that flips literal meaning whenever there is a mismatch.

### Step 15 — Integrate presuppositions, conversational implicatures, and indirect speech acts

**Purpose**

Recover information that a speaker treats as already accepted, likely to be inferred, or indirectly communicated. These classes differ and should not be collapsed into one generic “hidden meaning.”

**Algorithmic operations**

1. Detect linguistic triggers that may carry presuppositions, such as factive verbs, change-of-state expressions, definite descriptions, and iterative terms.
2. Identify information that the utterance treats as given or backgrounded.
3. Check whether the presupposed information is already in common ground, can be accommodated, or conflicts with prior discourse.
4. Detect likely conversational implicatures where the literal message appears weaker, stronger, or less direct than the speaker's likely goal.
5. Infer an indirect speech act when the utterance can function as a request, offer, warning, or refusal in the current situation.
6. Check whether the inferred content is cancellable or defeasible; avoid storing a defeasible inference as a logical entailment.
7. Preserve the distinction among asserted, presupposed, implicated, and inferred content.

**Possible mathematical operations**

- **Operator scope:** represent negation, modality, quantifiers, and presupposition triggers with explicit scope relations.
- **Common-ground update:** add a proposition to the shared information state only if it is accepted or appropriately accommodated.
- **Cancellable-inference test:** evaluate whether adding a continuation that denies the candidate implication creates a contradiction or merely cancels an inference.
- **Utility/relevance score:** estimate whether an indirect interpretation explains why the speaker selected that utterance in the current turn.
- **Confidence tagging:** preserve different strengths for entailment, presupposition, and pragmatic inference.

**Biological interpretation**

The source treats these processes as context-sensitive operations that can recruit the core language network together with social and reasoning systems. It does not identify a separate brain region for each pragmatic category. The likely biological architecture is distributed and task dependent.

**Checkpoint**

The integrated representation labels the epistemic status of each added piece of content, preventing an implicature from being mistaken for something literally asserted.

### Step 16 — Evaluate candidate interpretations and select, combine, or defer

**Purpose**

When several interpretations are possible, compare them using evidence from language, discourse, scene, world knowledge, speaker goals, and conversational expectations. The system should avoid overconfident inference where cues are weak or conflicting.

**Algorithmic operations**

1. Collect candidate interpretations from reference, causal, pragmatic, and social inference stages.
2. Score each candidate for linguistic compatibility, discourse coherence, situational fit, speaker-goal fit, and background plausibility.
3. Penalize candidates that require many unsupported assumptions.
4. Allow independent candidates to be combined when they explain different aspects of the same utterance.
5. Retain multiple candidates if posterior confidence is close or evidence sources conflict.
6. Commit to a single interpretation only when confidence and task demands justify it.
7. Record the reasons for selection so later evidence can revise the result.

**Possible mathematical operations**

A candidate score can be defined as:

`S(H_i) = w_L L_i + w_D D_i + w_S S_i + w_W W_i + w_M M_i + w_G G_i − λ A_i`

where:

- `L_i` = compatibility with literal linguistic content;
- `D_i` = discourse coherence;
- `S_i` = fit with the physical situation;
- `W_i` = world-knowledge compatibility;
- `M_i` = fit with the speaker's inferred mental state;
- `G_i` = fit with common ground and the conversational goal;
- `A_i` = unsupported assumptions required by the candidate;
- `w` and `λ` = task-dependent weights and penalties.

Other operations include:

- posterior normalization across candidates;
- entropy-based ambiguity estimates;
- Pareto comparison when one candidate fits the discourse but another fits the scene better;
- thresholded commitment;
- beam maintenance of a small number of alternatives;
- decision-theoretic selection based on the cost of misunderstanding.

**Important caution**

A weighted sum is convenient but may incorrectly assume that all evidence sources are independent and commensurable. A Bayesian network, factor graph, recurrent constraint system, or non-probabilistic ranking model may be more suitable for a specific research hypothesis. The brain's exact computation is not established by writing down one formula.

**Checkpoint**

The selected interpretation, alternatives, confidence, and unresolved conflicts are available for updating.

### Step 17 — Reanalyse earlier material when contextual evidence changes the interpretation

**Purpose**

Later words or contextual cues can reveal that an earlier interpretation was incomplete or mistaken. A context-sensitive system must revise previous assignments rather than treating the first interpretation as permanent.

**Algorithmic operations**

1. Detect a conflict between the new contextual evidence and the current interpretation.
2. Identify which earlier decisions depended on the assumption that is now challenged.
3. Reopen only the relevant lexical, syntactic, reference, or semantic alternatives.
4. Re-score those alternatives using the new evidence.
5. Update the proposition and downstream inferences if a different candidate becomes stronger.
6. Retain a trace of the previous interpretation if needed to explain the revision.
7. Stop revising when a coherent interpretation reaches an adequate confidence level or the task ends.

**Possible mathematical operations**

- **Backtracking:** return to a prior state in a candidate interpretation tree.
- **Belief revision:** reduce confidence in a prior hypothesis when contradictory evidence arrives.
- **Incremental Bayesian update:** `P(H | E_1...E_t) ∝ P(E_t | H)P(H | E_1...E_{t−1})`.
- **Prediction-error accumulation:** use mismatch signals to determine which decisions need revision.
- **Constraint propagation:** rerun only constraints affected by the changed assignment.
- **Cost-sensitive stopping:** stop reanalysis when the expected benefit of further processing is lower than its cost or when no sufficiently supported alternative exists.

**Neural interpretation**

The source describes P600-like late positivities and sustained activity in some contexts requiring reanalysis or inference. These observations are consistent with additional processing after an unexpected or conflicting input, but P600 is not a unique marker of one specific operation. A late positivity can reflect different processes depending on the experiment.

**Checkpoint**

Earlier interpretations remain revisable, and revisions are targeted rather than indiscriminate.

### Step 18 — Update the discourse, event, and situation models

**Purpose**

After deciding which information is supported, add it to the active model so that the next sentence can be understood in relation to the current one. The update must distinguish what was stated from what was inferred and what remains uncertain.

**Algorithmic operations**

1. Add the current asserted proposition to the discourse history.
2. Add high-confidence referential links and event relations.
3. Add inferred information with explicit source and confidence labels.
4. Update temporal order, causal relations, event participants, and current topic.
5. Update the scene model if the utterance reports a change in the physical or imagined situation.
6. Update the speaker model if new evidence indicates a change in knowledge, goal, or attitude.
7. Update common ground only when the information is mutually accepted or reasonably accommodated.
8. Mark superseded, corrected, or contradicted information rather than silently deleting it.

**Possible mathematical operations**

- **Graph insertion:** add nodes and typed edges to the discourse/situation graph.
- **State transition:** `C_{t+1} = Update(C_t, p_t, I_t, O_t)` where `I_t` is inferred content and `O_t` is new observation.
- **Confidence propagation:** update confidence on linked nodes and edges.
- **Temporal ordering:** maintain relations such as before, after, overlaps, during, or simultaneous.
- **Belief revision:** replace or qualify earlier beliefs when a correction is explicitly recognized.
- **Memory consolidation gate:** decide which information remains active and which may be passed to longer-term storage.

**Potential neural contributors**

The source describes language areas, TPJ, mPFC, angular gyrus, posterior cingulate, and working-memory systems as possible contributors to integrating propositional content with wider situation models. Medial temporal memory circuits can support longer-term encoding, but encoding is a subsequent outcome and is not identical to the contextual-integration operation itself.

**Checkpoint**

The next sentence has access to an updated, temporally organized context containing the new proposition and any supported inferences.

### Step 19 — Maintain the enriched representation and control processing demands

**Purpose**

Keep the current interpretation active long enough to relate it to following words and sentences. When a sentence or discourse is long, ambiguous, or inference-heavy, executive support may be needed to maintain competing alternatives and direct attention.

**Algorithmic operations**

1. Keep the current proposition and its most relevant context links active.
2. Preserve unresolved referents and inference alternatives until enough evidence arrives.
3. Remove irrelevant or obsolete details from the active buffer while retaining access to the broader model.
4. Allocate more control resources when multiple candidate interpretations compete or the task requires explicit judgment.
5. Protect critical information from interference by unrelated input.
6. Release or consolidate information when the comprehension task no longer requires active maintenance.

**Possible mathematical operations**

- **Leaky state maintenance:** `h_t = λh_{t−1} + input_t − interference_t`.
- **Gating:** determine which information is maintained, updated, or allowed to decay.
- **Capacity allocation:** distribute limited attentional resources among active items according to task relevance.
- **Competition/inhibition:** suppress low-relevance candidates while preserving enough activation to reopen them.
- **Priority queue:** a computational analogy for maintaining the most task-relevant unresolved items.
- **Interference control:** reduce overlap between active representations when several propositions share similar features.

**Potential neural contributors**

The source associates bilateral dorsolateral prefrontal cortex and intraparietal sulcus with domain-general multiple-demand functions when comprehension is difficult or working-memory demands are high. These regions should not be treated as necessary for every ordinary sentence. Left IFG and temporal language areas continue to support linguistic-semantic content, while parietal/frontal working-memory systems can maintain relevant information as needed.

**Checkpoint**

The enriched proposition remains accessible without assuming unlimited working-memory capacity.

### Step 20 — Deliver an enriched interpretation to subsequent comprehension, memory, and action systems

**Purpose**

Make the current interpretation usable. Contextual comprehension is not complete merely because a score or label has been assigned; the resulting model must be available for understanding the next sentence, answering a question, deciding whether to act, or supporting later memory formation.

**Algorithmic operations**

1. Package the literal proposition.
2. Attach reference resolutions, discourse links, and scene relations.
3. Attach pragmatic inferences, each labeled as explicit, presupposed, implicated, or inferred.
4. Preserve confidence and unresolved alternatives.
5. Provide a summary of the current discourse and situation state to downstream operations.
6. Allow new information to re-enter the contextual-integration loop when the next word or sentence arrives.

**Possible mathematical operations**

- **Structured serialization:** encode the interpretation in a consistent graph or record format.
- **Confidence annotation:** carry probability or ordinal confidence values with inferred relations.
- **Compression:** retain high-value information in a compact active representation while preserving pointers to details.
- **Output gating:** route the enriched model to working memory, response selection, planning, or memory encoding depending on task.

**Checkpoint**

The outcome is a coherent but revisable interpretation that distinguishes literal linguistic content from contextual enrichment.

---

## 4. A unified computational model of the integration loop

The preceding steps can be summarized as a recurrent loop. The following equations define one possible computational model for simulation. They are not proposed as a literal account of every neuronal calculation.

### 4.1 Context representation

Let the current context be:

`C_t = {D_t, S_t, K_t, M_t, G_t, Q_t}`

where `D` is discourse, `S` is situation, `K` is world knowledge, `M` is the speaker/mental-state model, `G` is common ground, and `Q` is the conversational goal or question under discussion.

Let `P_t` be the current literal proposition and `H_t = {h_1,...,h_n}` be candidate enriched interpretations.

### 4.2 Candidate score

For each candidate `h_i`, calculate a model score:

`score(h_i) = L(h_i,P_t) + D(h_i,D_t) + S(h_i,S_t) + K(h_i,K_t) + M(h_i,M_t) + G(h_i,G_t) − λA(h_i)`

The terms evaluate linguistic compatibility, discourse coherence, situation fit, world plausibility, mental-state fit, common-ground relevance, and additional-assumption cost. In a probabilistic implementation, these may be represented as log likelihoods or factors rather than a raw weighted sum.

### 4.3 Candidate distribution

Convert scores into a normalized distribution:

`P(h_i | P_t,C_t) = exp(β score(h_i)) / Σ_j exp(β score(h_j))`

`β` controls how sharply the model favors the highest-scoring hypothesis. A high `β` makes choices more decisive; a low `β` keeps more alternatives active. This is a modeling convenience, not a known neural temperature parameter.

### 4.4 Inference trigger

Define a conflict/underspecification measure:

`M_t = a·gap_t + b·contradiction_t + c·surprisal_t + d·referential_uncertainty_t + e·intention_uncertainty_t`

Launch additional inference when:

`M_t > θ(task, attention, processing_cost)`

The threshold can vary with task demands. A casual conversation may tolerate a lower level of precision than a legal or safety-critical instruction.

### 4.5 Evidence accumulation and revision

When new information `E_{t+1}` arrives:

`P(h_i | E_{1:t+1}, C) ∝ P(E_{t+1} | h_i,C) P(h_i | E_{1:t},C)`

This formalizes revision: a new word, visual cue, facial expression, or later sentence can change the ranking of interpretations.

### 4.6 Context update

After integration:

`C_{t+1} = F(C_t, P_t, h_t^*, confidence(h_t^*), new_observations)`

The update function adds asserted and supported inferred information, updates event and reference links, and keeps source labels. It should not automatically add every low-confidence hypothesis to long-term knowledge.

### 4.7 Compact pseudocode for the full loop

```text
FUNCTION INTEGRATE_CONTEXT(proposition, context, alternatives):
    literal = preserve_literal(proposition)
    retrieved = retrieve_relevant_context(literal, context)
    focus = determine_discourse_focus(retrieved)
    references = resolve_references(literal, retrieved, focus)
    predictions = generate_contextual_expectations(retrieved, focus)

    coherence = compare_with_discourse(literal, retrieved, references)
    scene_fit = compare_with_situation(literal, context.situation)
    world_fit = evaluate_world_knowledge(literal, retrieved.world_knowledge)
    mismatch = estimate_gaps_conflicts_and_uncertainty(
        coherence, scene_fit, world_fit, references, predictions
    )

    IF mismatch is below inference threshold:
        enriched = literal + supported_referential_links(references)
    ELSE:
        bridges = generate_bridging_and_causal_candidates(literal, retrieved)
        pragmatic = generate_pragmatic_candidates(literal, context)
        social = infer_speaker_goals_if_relevant(literal, context.speaker_model)
        candidates = merge_candidates(bridges, pragmatic, social, alternatives)
        scored = evaluate_candidates(candidates, literal, context)
        enriched = select_or_preserve_alternatives(scored)
        enriched = reanalyse_if_new_evidence_changes_prior_decisions(enriched)

    context_next = update_discourse_and_situation_models(
        context, literal, enriched, confidence_and_source_labels(enriched)
    )
    maintain_in_working_memory(enriched, context_next)

    RETURN enriched, context_next
```

---

## 5. Worked examples

### Example A — Contextual facilitation without elaborate pragmatic inference

Context: “Mina entered the kitchen. She opened the refrigerator.”

The pronoun `she` can be linked to Mina using discourse recency and agreement. The action `opened the refrigerator` fits a familiar event schema. The system does not need to infer a hidden social intention or generate an elaborate causal story.

Possible representation:

```text
ENTER(e1)
AGENT(e1, MINA)
LOCATION(e1, KITCHEN)

OPEN(e2)
AGENT(e2, MINA)
PATIENT(e2, REFRIGERATOR)

COREFERENCE(SHE, MINA)
TEMPORAL_ORDER(e1, BEFORE, e2)
```

The core task is incremental reference and event integration. A small number of context links produce a coherent model.

### Example B — Bridging inference: “Lisa apologized. Peter shrugged.”

Literal propositions:

```text
APOLOGIZE(e1)
AGENT(e1, LISA)
SHRUG(e2)
AGENT(e2, PETER)
```

A likely bridge is that Peter is the recipient of Lisa's apology, perhaps explaining his shrug. The listener can infer this because apology events commonly involve a recipient and the two clauses are adjacent. However, the inference is defeasible: Peter might have shrugged for another reason.

A rigorous representation stores:

```text
CANDIDATE(RECIPIENT(e1, PETER))
CANDIDATE(RESPONSE_TO(e2, e1))
```

with confidence values that depend on earlier discourse and the situation.

### Example C — Indirect request: “It’s cold in here.”

Literal proposition:

```text
COLD(ROOM)
```

Context: a window is open; the listener is near it; the speaker is uncomfortable; closing the window is an available action.

Candidate pragmatic inference:

```text
REQUEST(SPEAKER, LISTENER, CLOSE(WINDOW))
```

The indirect request is supported by physical affordance, speaker need, and conversational relevance. It is not entailed by the literal proposition. If the speaker is answering a question about room temperature, the request inference should be weaker.

### Example D — Sarcasm: “Great job!” delivered in a scolding tone

Literal content expresses praise. The actual context may indicate that the listener has made a mistake and the speaker is criticizing it. The algorithm compares literal valence with prosody, preceding events, speaker stance, and common ground.

Possible result:

```text
LITERAL(PRAISE(SPEAKER, LISTENER))
PRAGMATIC(CRITICISM(SPEAKER, LISTENER))
MODE(SARCASM_OR_IRONY)
```

This is not a simple polarity inversion rule. A person can say “Great job!” sincerely in a loud voice. Tone must be interpreted with context.

### Example E — Causal/contextual inference: “The man couldn’t see, so he opened an umbrella.”

Under ordinary world knowledge, an umbrella may be associated with rain, but the sentence as stated does not logically entail that it was raining. The listener may generate `RAINING` as a candidate explanation, especially if the scene includes rain clouds or wet ground. Other explanations remain possible: he may be using the umbrella for shade, as a signal, or for another purpose.

A cautious output is:

```text
CANDIDATE(RAINING, confidence = context-dependent)
ENABLES_OR_MOTIVATES(WEATHER, OPEN_UMBRELLA)
```

The inference is promoted only when additional evidence supports it.

### Example F — Apparent incongruity and a wished-for event

The source gives the example: “John’s cat is dying of cancer, and his dog is chasing the cat,” with a possible inference such as “Fortunately, the dog didn’t actually catch it.” The sentence may evoke a disturbing or undesired possible outcome. A listener might infer a hope or wish that the dog does not catch the cat, especially if the narrative context makes the cat's vulnerability salient.

The model must separate:

- the asserted events (`CAT_ILL`, `DOG_CHASES_CAT`);
- the possible consequence (`DOG_CATCHES_CAT`);
- the listener's or speaker's evaluative stance (`WISH(NOT CATCH)`), if supported.

A possible inference should not be inserted as an asserted fact. This example illustrates how event knowledge and evaluation can interact, but the wish is context dependent, not an automatic consequence of the sentence.

### Example G — Presupposition and common ground

Sentence: “John stopped smoking.”

The verb `stopped` commonly triggers the presupposition that John previously smoked. The listener should distinguish the asserted change from the backgrounded prior state:

```text
ASSERTED: NOT_SMOKING(John, now)
PRESUPPOSED: SMOKING(John, earlier)
```

If the prior smoking history is not in common ground, the listener may accommodate it, question it, or flag it as unsupported depending on context. It should not be treated as though the speaker had separately asserted it.

---

## 6. Possible neural operations and their computational analogies

The table below maps functions in the algorithm to plausible network-level contributions and mathematical abstractions. The anatomical relationships are not one-to-one mappings; the same region can participate in several operations, and a single operation can be distributed across multiple regions.

| Functional operation | Neural systems discussed in the source | Possible mathematical/computational abstraction | Important qualification |
|---|---|---|---|
| Maintain the literal proposition | Left IFG and temporal language network; verbal working-memory support | Recurrent activation, gated maintenance, structured state | Not every sentence requires high executive demand |
| Retrieve discourse-relevant meaning | Left temporal regions, angular gyrus, posterior medial regions; memory systems as needed | Associative activation, similarity-based retrieval, recency weighting | Retrieval is distributed; explicit vector search is a model analogy |
| Integrate local discourse | Left temporal language areas and left TPJ | Graph linking, relation scoring, constraint satisfaction | The exact role of left TPJ is still debated and task dependent |
| Align with world knowledge | Left TPJ and distributed conceptual knowledge | Semantic compatibility, Bayesian/abductive inference | No single hub contains all world knowledge |
| Reorient to unexpected information | Right TPJ and attention-related networks | Surprise detection, gain adjustment, attentional reallocation | Right TPJ has multiple functions; not merely a context switch |
| Infer speaker intention or belief | mPFC and TPJ mentalizing network | Inverse planning, belief-state inference, nested perspective models | The mentalizing network is domain-general, not language specific |
| Handle difficult, competing interpretations | Dorsolateral PFC and intraparietal/superior parietal multiple-demand network | Working-memory control, competition resolution, resource allocation | These regions are recruited especially under high demand, not necessarily for basic composition |
| Maintain a broad event/situation model | Angular gyrus, posterior cingulate/precuneus, temporal and memory networks | Event graph, relational binding, temporal sequencing | These structures do not form one single “situation-model center” |
| Detect contextual facilitation or mismatch | Distributed language and context-sensitive networks; observable ERP effects such as N400 | Prediction distribution, surprisal, prediction-error model | N400 is not uniquely equivalent to surprisal or a single prediction operation |
| Reanalyse unexpected input | Language network plus frontal/parietal and, when required, social/reasoning systems; late ERP effects such as P600 in some paradigms | Belief revision, backtracking, constraint re-evaluation | P600 interpretations vary; it is not a dedicated pragmatic-repair marker |

### Candidate biological computations

1. **Synaptic integration:** neurons receive excitatory and inhibitory inputs and combine them through membrane dynamics. A simplified membrane model is `τ_m dV/dt = −(V−V_rest) + R I_syn(t)`. This provides a biophysical foundation for integrating evidence, although the model does not specify the full contextual computation.
2. **Population coding:** different neural populations respond to features, concepts, entities, or relations. A population state can be represented in a model as a vector of firing rates or other neural response measures.
3. **Recurrent connectivity:** feedback among temporal, frontal, parietal, and memory networks can support iterative refinement and context effects.
4. **Gain modulation and attention:** top-down signals can increase or reduce the influence of selected representations, approximated by multiplicative gain or gating equations.
5. **Temporal integration:** neural activity across time can accumulate evidence for a reference or interpretation, although the duration and mechanism depend on the task.
6. **Error-sensitive updating:** unexpected or conflicting input can change the activity pattern and recruit additional processing. This should not be assumed to use one universal error signal.
7. **Plasticity and longer-term learning:** repeated discourse and social experiences can gradually alter knowledge structures and future expectations. This is distinct from the rapid within-sentence contextual integration described here.

---

## 7. Mathematical operations inventory

| Mathematical operation | Function in the algorithm | Interpretive level |
|---|---|---|
| Exponential decay | Model declining accessibility of older discourse items | Neural/computational abstraction |
| Weighted relevance scoring | Rank retrieved context items | Algorithmic analogy |
| Associative spreading activation | Activate related concepts and events | Neural-network abstraction |
| Vector similarity | Estimate semantic or contextual match | Computational tool; not direct proof of neural vector arithmetic |
| Softmax normalization | Convert candidate scores into relative weights | Modeling device |
| Bayesian updating | Combine prior expectations with new contextual evidence | Normative/computational model |
| Entropy | Quantify uncertainty across interpretations | Mathematical diagnostic |
| Surprisal | Quantify how unexpected an input is under a model | Information-theoretic measure; relation to brain signals is not one-to-one |
| Prediction error | Represent mismatch between expected and observed information | Predictive-processing abstraction |
| Constraint satisfaction | Check whether discourse, temporal, semantic, and causal relations can coexist | Algorithmic formalization |
| Graph insertion and path completion | Add referential, causal, temporal, or discourse links | Symbolic/graph model |
| Abductive inference | Generate explanations for missing or surprising relations | Reasoning model |
| Inverse planning | Infer goals from a speaker's utterance or action | Computational social-inference model |
| Utility maximization | Compare possible communicative intentions | Normative pragmatic model |
| Recurrent state update | Revise interpretation as words and context arrive | Neural-network abstraction |
| Backtracking and belief revision | Reanalyse previously selected meanings | Algorithmic analogy |
| Thresholding and sequential evidence accumulation | Decide when to infer, commit, or continue processing | Decision model |
| Leaky integration and gating | Maintain relevant information while controlling interference | Neural-computation abstraction |
| Temporal interval relations | Represent event sequence and overlap | Formal representation of event time |

---

## 8. Failure modes and algorithmic safeguards

1. **Over-inference:** the system invents a hidden cause or intention not supported by evidence. Safeguard: penalize unsupported assumptions, label defeasible inferences, and preserve alternatives.
2. **Literalism:** the system ignores speaker intention, tone, or a strong situational cue. Safeguard: monitor for mismatch between literal content and discourse/speaker context.
3. **Context neglect:** the system interprets a sentence in isolation. Safeguard: retrieve a limited but relevant discourse and situation state before committing.
4. **Context over-dominance:** prior expectations override clear new evidence. Safeguard: distinguish prior probability from likelihood of the observed input and keep evidence source labels.
5. **Recency bias:** the latest entity is selected as the referent even when another entity is more salient or semantically appropriate. Safeguard: combine recency with grammatical, semantic, and discourse constraints.
6. **Stereotype-based inference:** generic assumptions about people or social groups are treated as certain facts about a speaker. Safeguard: require evidence, represent uncertainty, and avoid treating broad statistical associations as individual truth.
7. **Speaker-model overconfidence:** the system assumes it knows what the speaker believes or intends. Safeguard: maintain a distribution over mental-state hypotheses and update it with later evidence.
8. **False contradiction:** two statements that differ are treated as logically incompatible even though they refer to different times, perspectives, or hypothetical worlds. Safeguard: track time, source, modality, and perspective.
9. **Literal/inferred content conflation:** an implicature or bridging inference is stored as an explicitly stated fact. Safeguard: label each proposition by epistemic and discourse status.
10. **Premature resolution:** one candidate interpretation is selected before enough evidence arrives. Safeguard: preserve multiple candidates when posterior scores are close.
11. **Excessive reanalysis:** the model repeatedly revisits settled decisions. Safeguard: use confidence thresholds and a cost-sensitive stopping policy.
12. **Working-memory overload:** too many context items and interpretations are kept active. Safeguard: prioritize, compress, and preserve retrievable pointers to less active information.
13. **Cultural mismatch:** conversational norms learned in one cultural or linguistic setting are applied indiscriminately in another. Safeguard: condition expectations on language, culture, community, and speaker-specific context, and retain uncertainty.
14. **Language-specific overgeneralization:** an English-oriented model is applied directly to languages with different word order, morphology, politeness systems, or pragmatic conventions. Safeguard: parameterize grammatical and pragmatic constraints by language.
15. **Misreading ERP evidence:** an N400 or P600 is treated as a uniquely diagnostic marker for one computation. Safeguard: interpret electrophysiological components with task design, timing, behavior, and convergent evidence.

---

## 9. Biological and scientific cautions

1. **Contextual integration is not a single final stage that waits until a sentence ends.** The source explicitly describes incremental integration: earlier words and context influence expectations while the utterance is still unfolding.
2. **The language network and social/reasoning networks are distinct but interactive.** The source presents the left frontotemporal language network as relatively language selective, while mPFC and TPJ support domain-general social reasoning and mentalizing when those abilities are needed.
3. **The left TPJ is not a universally agreed single-purpose context hub.** The source associates it with contextual semantic integration and world-knowledge integration, but cortical boundaries and functions vary across studies and individuals.
4. **Right TPJ is not exclusively an attentional reorienting center.** It is also implicated in social inference and broader context updating. Function depends on task and network interaction.
5. **mPFC activation does not prove that every pragmatic task requires theory-of-mind reasoning.** Some inferences are causal or event-based; others require speaker-belief reasoning. The recruited networks depend on the inferential problem.
6. **Multiple-demand regions are not necessarily part of core semantic composition.** Dorsolateral PFC and parietal control regions are especially relevant when memory, conflict, or task demands are high.
7. **N400 is not a literal numerical probability signal.** Contextually congruent words often elicit smaller N400 responses, but the component has multiple theoretical interpretations and should not be equated with a single equation.
8. **P600 is not a dedicated sarcasm, pragmatics, or inference signal.** Late positivities have been associated with reanalysis and other processes; interpretation depends on paradigm and theory.
9. **Brain networks operate in parallel and recurrently.** The algorithm's serial numbering makes the explanation readable but does not imply a fixed sequence of isolated neural modules.
10. **fMRI correlations do not establish causal necessity.** The source highlights the importance of converging evidence from lesions, electrophysiology, behavior, and causal methods such as stimulation.
11. **Individual differences matter.** Language-network localization, contextual strategies, cognitive control, development, aging, and linguistic experience can alter the degree and timing of network recruitment.
12. **Social and cultural knowledge is learned.** A pragmatic model built from one population's conversational conventions may misinterpret speakers from other communities if it assumes universal norms.
13. **Literal plausibility is not truth.** A surprising sentence can be true, fictional, humorous, metaphorical, or a deliberate test of expectations. Context integration must not turn statistical typicality into a hard logical rule.
14. **The formulae are models of possible computations.** The brain's biological operations may be implemented through distributed synaptic dynamics and population interactions rather than explicit symbolic lists, hand-written probabilities, or conventional computer instructions.

---

## 10. References and source alignment for Part 5

This point is based primarily on the supplied Library file, **“Language Comprehension_ From Words to Contextual Meaning.txt.”** The following studies and reviews are mentioned or summarized in that source and provide the empirical background for the algorithmic reconstruction. The equations and pseudocode in this section are proposed computational models; they are not claims that the cited authors demonstrated those exact mathematical operations in neurons.

1. **Kuperberg et al. (2006).** The supplied source summarizes findings that sentences requiring causal bridging inferences can produce sustained activity in left temporal and inferior parietal/frontal regions and bilateral medial frontal regions, compared with highly related sentence pairs. This supports a distributed account of discourse-level inference.
2. **Fletcher et al. (1995).** The supplied source cites this work in connection with temporoparietal and medial prefrontal recruitment during stories requiring the reader to reason about characters' beliefs.
3. **Fedorenko et al. (2024).** The supplied source summarizes the distinction between a core language-selective network and broader systems of knowledge, reasoning, and social cognition. It emphasizes that these systems can cooperate while remaining functionally dissociable.
4. **Kuperberg (2021 review).** The supplied source refers to a review of pragmatic language processing in which left language areas and additional regions such as dorsolateral PFC, TPJ, and precuneus are reported across different pragmatic tasks.
5. **ERP/MEG work on N400 and P600.** The supplied source describes N400 modulation by contextual congruence and late positivities in some pragmatic anomaly/reanalysis paradigms. These components constrain timing but do not uniquely identify one computational operation.
6. **Patient and lesion evidence summarized in the supplied source.** The source describes dissociations in which some patients with left-hemisphere language damage have difficulties with basic linguistic combination while retaining aspects of pragmatic judgment, and some patients with right-hemisphere or medial prefrontal damage show pragmatic or discourse-level difficulties despite relatively preserved grammar. Such findings support a multi-network account but do not imply a strict one-region/one-function mapping.

### Mapping to the supplied source's sequence

| Source concept | Where it is implemented in this algorithm |
|---|---|
| Incremental integration and immediate contextual modulation | Steps 1–10 and the recurrent update model |
| Discourse history and preceding sentences | Steps 2–4 and 7 |
| Physical situation and scene information | Steps 3 and 8 |
| Speaker identity, tone, and intention | Steps 3, 12–14 |
| World knowledge and expectations | Steps 2, 6, and 9 |
| Pragmatic inference: implicature, presupposition, indirect speech acts, sarcasm | Steps 12–15 |
| mPFC and TPJ mentalizing/context roles | Steps 13–14 and the neural mapping table |
| Left TPJ contextual semantic integration | Steps 3, 5, 7, and 9 |
| Right TPJ attentional reorienting/context updating | Steps 6, 8, 10, and the neural mapping table |
| Dorsolateral PFC/parietal multiple-demand support | Steps 16, 17, and 19 |
| N400/P600 evidence | Steps 6, 14, 17, and the scientific cautions |
| Context-enriched proposition held in working memory | Steps 18–20 |

---

**End of Part 5.** The next source-related processing point is **propositional working-memory representation and maintenance**: maintaining the structured literal-plus-inferred meaning so it can be used for later comprehension, memory formation, question answering, or decisions. That point has not been implemented in this installment.

# Point 6 — Propositional working-memory representation and maintenance

## 1. Objective and boundary of this algorithm

This algorithm begins when the comprehension system has assembled a developing sentence interpretation from lexical meanings, grammatical structure, and any contextual or pragmatic enrichment currently justified. Its goal is to organize that interpretation into a structured representation and keep the relevant components available for the ongoing comprehension task.

The working-memory representation should preserve more than a flat sequence of words. It should retain, as far as the task requires:

- **Predicates and events:** what happened, what state holds, or what relation is being asserted.
- **Arguments and semantic roles:** who or what participated, and in what role (agent, patient, experiencer, recipient, instrument, location, and so on).
- **Referential identity:** which mention refers to which person, object, place, or event.
- **Relations among propositions:** temporal order, cause, consequence, contrast, condition, elaboration, and other discourse links.
- **Modifiers and operators:** negation, tense, aspect, modality, quantification, degree, and restrictive descriptions.
- **Contextual enrichment:** relevant inferences or situation-model details, kept distinguishable from what was explicitly said.
- **Uncertainty and provenance:** whether a detail was stated, inferred, presupposed, assumed provisionally, or remains unresolved.

The central output is a **temporarily available propositional/situation representation**, not a permanent memory trace. Long-term storage and decision-making are downstream uses and are not implemented as the main objective of this part. The system may prepare information for those operations, but it should not silently treat temporary maintenance as durable learning.

### Important scientific distinction

The supplied source describes proposition formation in left inferior frontal and middle temporal regions, possible contribution from angular gyrus and posterior cingulate in binding the proposition into a broader model, and temporary maintenance involving left inferior frontal and bilateral parietal regions. These are useful functional associations, not proof that a single region stores a complete symbolic proposition. Neural representations are distributed, dynamic, and task-dependent.

The algorithm uses explicit data structures and equations because they make the required operations understandable and testable. They are **computational models of possible functions**, not assertions that neurons literally manipulate Python-like objects, write predicate symbols, calculate a dot product, or execute a conventional buffer-management program.

The distinction among levels of explanation remains important:

1. **Biophysical operations:** synaptic currents, membrane integration, changes in excitability, and activity-dependent neural dynamics.
2. **Neural-population models:** distributed activation patterns, recurrent interactions, attention-dependent gain, and competition among representations.
3. **Cognitive/computational abstractions:** propositions, role-filler bindings, working-memory slots, graphs, confidence values, and update rules.

The stages below are separated for clarity. Actual comprehension is incremental: a new word or clause can modify, enrich, or destabilize a representation that is already being maintained.

---

## 2. Inputs and outputs

### Inputs

1. **Current linguistic interpretation:** lexical items, grammatical dependencies, and a developing predicate–argument structure from earlier processing.
2. **Context-enriched information:** discourse entities, situation information, causal links, speaker-related information, and any pragmatic inferences already generated.
3. **Task state:** what the listener is trying to understand or remember (ordinary comprehension, answering a question, following instructions, or judging coherence).
4. **Working-memory state:** currently active propositions, unresolved dependencies, referents, event representations, and their current activation/priority.
5. **Control signals:** attentional focus, relevance, conflict, and memory-demand signals that may recruit domain-general control systems.

### Outputs

- A structured, incrementally updated **propositional working-memory representation**.
- A linked **situation model** where event, entity, temporal, spatial, causal, and discourse relations are required.
- A record of unresolved or low-confidence elements rather than a forced guess.
- A compact active state that can support the next word, clause, or comprehension decision.
- Pointers and selected information that may later be passed to long-term memory or decision systems.

---

## 3. Data model: what the algorithm maintains

A convenient abstract representation is a set of proposition records linked into a graph. A proposition can be written as:

\[
P_i = (\text{predicate},\; \text{arguments},\; \text{operators},\; \text{modifiers},\; \text{time},\; \text{source},\; \text{status},\; \text{confidence})
\]

For example:

\[
\operatorname{APOLOGIZE}(\text{Lisa},\; \text{recipient}=\text{Peter})
\]

The recipient may have been inferred from the discourse rather than explicitly stated. The record must therefore retain a status label such as `inferred`, rather than converting the inference into an explicitly asserted fact.

A proposition network may be modelled as a labelled graph:

\[
G=(V,E,\lambda_V,\lambda_E)
\]

where:

- \(V\) contains entity, event, state, and proposition nodes;
- \(E\) contains role links and relations between nodes;
- \(\lambda_V\) stores node type and features;
- \(\lambda_E\) stores the meaning of each connection (for example, `agent-of`, `patient-of`, `before`, `causes`, `refers-to`, or `contrasts-with`).

This is a useful symbolic model. A biological implementation would more plausibly involve distributed neural population activity whose patterns carry information about predicates, participants, and their relationships.

### 3.1 Example proposition record

```text
Proposition ID: P17
Predicate/event: APOLOGIZE
Arguments:
  agent: Lisa [explicit]
  recipient: Peter [inferred from discourse]
Operators:
  tense: past
Relations:
  discourse-link: follows P16
Source:
  clause: "Lisa apologized."
Status:
  predicate and agent: explicitly stated
  recipient: inferred / defeasible
Confidence:
  predicate and agent: high
  recipient: moderate
Working-memory priority: task-relevant
```

The data model explicitly separates a proposition's content from its epistemic status. This prevents a later operation from confusing the speaker's words with a plausible but uncertain completion.

---

## 4. Detailed algorithmic stages

### Step 1 — Receive the developing meaning representation

**Purpose:** Accept the partial output of semantic composition and contextual integration without restarting earlier perceptual or lexical operations.

**Operation:** Gather the active predicate, argument candidates, grammatical features, referents, contextual relations, and inference records. Preserve word- and clause-level timing where it may matter for resolving a later dependency.

The system should not assume that the sentence is complete. A predicate may have an argument that has not yet appeared; a pronoun may not yet have a resolved antecedent; a modifier may still be pending. The representation therefore supports incomplete records.

**Possible mathematical model:** Let the input at time \(t\) be a feature bundle \(u_t\). The maintained state before the new input is \(x_{t-1}\). A basic update interface is:

\[
x_t = F(x_{t-1},u_t,c_t)
\]

where \(c_t\) represents currently relevant context and \(F\) is an update function. This formula is a high-level description, not a demonstrated literal neural equation.

**Biological interpretation:** The left temporal and frontal language network contributes linguistic-semantic content; parietal and other association regions can contribute to broader integration. The system receives activity from multiple interacting populations, not a single completed symbolic packet.

### Step 2 — Segment the developing interpretation into proposition candidates

**Purpose:** Determine which event, state, or relation each clause expresses, and where one proposition ends and another begins.

**Operation:** Use predicate structure, clause boundaries, coordination, subordination, and discourse markers to establish candidate proposition units. Do not assume that every clause maps one-to-one to an isolated proposition: embedded clauses, ellipsis, coordination, and complex predicates may require nested or linked records.

For example, “Maya said that Omar had left” contains a saying event and a proposition embedded as its content:

\[
\operatorname{SAY}(\text{Maya},\; P_{\text{content}})
\]

\[
P_{\text{content}}=\operatorname{LEAVE}(\text{Omar})
\]

**Possible mathematical operations:**

- Boundary detection as classification over a sequence of structural features.
- Hierarchical grouping of clause and phrase units.
- Construction of nested records or a graph with proposition nodes.

A sequence boundary score could be represented abstractly as:

\[
B_t = \sigma(w^\top f_t+b)
\]

where \(f_t\) is a feature vector and \(\sigma\) is a logistic function. The equation only illustrates a possible classifier; it is not a claim that the brain computes a logistic regression.

**Neural interpretation:** Left inferior frontal and temporal language regions contribute to syntactic-semantic structuring. The exact neural basis of clause segmentation is distributed and depends on the linguistic construction.

### Step 3 — Establish entity and event identities

**Purpose:** Ensure that repeated mentions of the same participant do not become unrelated entities in the working representation.

**Operation:** Compare each newly introduced noun phrase, pronoun, demonstrative, or event reference with active discourse entities. Reuse an existing identity when the linguistic and contextual evidence supports coreference; otherwise create a new entity candidate. Preserve ambiguity when more than one antecedent remains plausible.

For example, in “Nadia met Salma after she arrived,” `she` may refer to Nadia or Salma. The algorithm should retain competing links until available cues discriminate between them.

**Possible mathematical operations:**

1. Build a feature vector for the new mention and each candidate antecedent.
2. Compute a compatibility score using grammatical agreement, semantic fit, recency, discourse prominence, and contextual constraints.
3. Normalize candidate scores to produce a graded preference.
4. Keep more than one candidate when score differences are small.

One abstract scoring rule is:

\[
s_j = w_a A_j+w_g G_j+w_r R_j+w_d D_j+w_c C_j
\]

where the terms represent agreement, grammatical compatibility, recency, discourse prominence, and contextual fit. A softmax can turn scores into a candidate distribution:

\[
Pr(e_j\mid m)=\frac{e^{s_j/\tau}}{\sum_k e^{s_k/\tau}}
\]

Here \(\tau\) controls how sharply the highest-scoring candidate dominates. This is a modelling tool, not evidence that the brain stores explicit probability values.

**Biological interpretation:** Temporal and parietal language-related regions participate in reference and meaning integration; broader memory and attentional processes may contribute when antecedents are distant or ambiguous. Do not assign coreference to one brain region alone.

### Step 4 — Bind predicates to their arguments and roles

**Purpose:** Preserve who did what to whom, who experienced a state, and how each participant relates to the event.

**Operation:** Connect each argument to its predicate using the syntactic structure, lexical semantics, morphological cues, and discourse information. Distinguish grammatical function from semantic role: a grammatical subject is not always an agent, as in “The window broke” or “Mina was frightened by the noise.”

The representation should encode role labels rather than store participants as an unordered list:

\[
\operatorname{GIVE}(\text{agent} = \text{Omar},\;\text{theme}=\text{book},\;\text{recipient}=\text{Lina})
\]

**Possible mathematical operations:**

- Role assignment as constrained classification.
- Binding a role vector to a participant representation.
- Checking consistency between role expectations and syntactic evidence.

A simple role-binding abstraction is:

\[
b_r = \phi(r,e)
\]

where \(r\) is a role, \(e\) is an entity representation, and \(\phi\) is a binding operation. In vector-symbolic or tensor-product models, the binding may be represented as:

\[
B = \sum_r \mathbf{r}\otimes\mathbf{e}_r
\]

The tensor product is one possible formal device for preserving role–filler relationships. It is not established as the literal representation used by neurons.

**Biological interpretation:** The source associates basic proposition construction with left inferior frontal and middle temporal regions. These regions operate in a broader network; the algorithmic role-binding description should not be read as a precise one-to-one localization.

### Step 5 — Attach operators, modifiers, and scope

**Purpose:** Keep the proposition's truth conditions and event details intact.

**Operation:** Attach features such as negation, tense, aspect, modality, quantification, degree, and restrictive modifiers to the correct constituent or proposition. Scope matters: “Not every student passed” differs from “Every student did not pass” under many interpretations.

The algorithm should store an operator's target explicitly rather than append the operator as an unstructured feature.

**Possible mathematical operations:**

- Tree or graph attachment to the correct syntactic-semantic node.
- Scope ordering and composition.
- Boolean or graded transformations for negation and modality, depending on the semantic model.

For a simple binary representation, negation may be modelled as:

\[
\operatorname{NOT}(P)=1-P
\]

This is only a convenient formalization for truth-valued propositions. Natural-language negation interacts with scope, presupposition, uncertainty, and discourse and cannot always be reduced to a single binary flip.

**Biological interpretation:** Linguistic operators are integrated through the distributed language network. More complex scope relations may increase memory and control demands, bringing additional frontoparietal support into play.

### Step 6 — Preserve the distinction between stated, inferred, and assumed content

**Purpose:** Prevent plausible enrichment from becoming indistinguishable from the sentence's explicit content.

**Operation:** Assign provenance labels to each element:

- `explicit`: directly expressed by the linguistic input;
- `entailed`: follows from the interpreted content under the adopted semantic assumptions;
- `inferred`: supplied by causal, bridging, social, or pragmatic reasoning;
- `presupposed`: treated as background by the construction or utterance;
- `assumed`: temporarily adopted to continue interpretation but not yet confirmed;
- `unresolved`: more than one interpretation remains viable.

For example, “Lisa apologized. Peter shrugged.” may support an inferred recipient relation, but it does not explicitly state that Lisa apologized to Peter. The recipient should therefore be tagged as inferred and defeasible.

**Possible mathematical operations:** Store provenance as a categorical variable, and maintain a confidence value separately. If evidence is combined, use a graded update rule such as:

\[
q_{t+1}=q_t+\eta( e_t-q_t)
\]

where \(q_t\) is a confidence estimate, \(e_t\) is new evidence coded on a compatible scale, and \(\eta\) is an update rate. This is a generic smoothing model, not a neural fact.

**Biological interpretation:** The source suggests language and broader reasoning/social systems contribute different types of information. A provenance label is an algorithmic design requirement; there is no established evidence that the brain stores explicit tags with these exact names.

### Step 7 — Link propositions into a coherent event and discourse structure

**Purpose:** Represent relations across propositions rather than leaving sentence meanings as unrelated records.

**Operation:** Add links such as:

- temporal order (`before`, `after`, `during`, `overlaps`);
- causal relations (`causes`, `enables`, `prevents`);
- conditional relations (`if`, `unless`);
- contrast or concession (`but`, `although`);
- elaboration, explanation, and correction;
- coreference and shared-event references.

A discourse graph can be expanded whenever a new proposition arrives. A causal link should be inserted only when supported by the wording, situation, or a sufficiently strong inference; temporal succession alone does not prove causation.

**Possible mathematical operations:**

- Graph-edge creation and update.
- Temporal interval comparison.
- Constraint propagation across event relations.
- Detection of cycles or incompatible ordering constraints.

For a simple temporal model, each event \(E_i\) can have an interval \([t_i^{start},t_i^{end}]\). The system can test whether a proposed relation is compatible with known constraints. In natural language, these intervals may be partially ordered or uncertain rather than assigned exact clock times.

**Biological interpretation:** The source highlights angular gyrus and posterior cingulate as possible contributors to binding a proposition into a broader model. These regions should be treated as components of a network, not as a single graph-storage device.

### Step 8 — Construct or update the situation model

**Purpose:** Connect the propositional content to a representation of the described world or event, beyond the sentence's surface form.

**Operation:** Maintain an evolving model of relevant entities, states, events, time, place, causal relations, and goals. Update only those elements that the new sentence changes or adds. A situation model is not necessarily a detailed mental picture; it may be abstract, spatial, causal, social, or event-structured.

For example, “The glass was on the table. It fell.” requires the system to maintain a referent for `the glass` and link the falling event to that entity. A reader need not visualize the glass vividly to preserve the relation.

**Possible mathematical operations:**

- Incremental graph update: \(G_t=G_{t-1}\cup\Delta G_t\).
- State transition: \(S_{t+1}=T(S_t,E_t)\).
- Consistency checks between new event information and the current situation state.
- Selective revision when a new sentence contradicts or corrects an earlier assumption.

**Biological interpretation:** The source describes contributions from language regions and broader systems that support context, memory, and reasoning. Posterior cingulate and angular gyrus may contribute to integration into a broader model, while other cortical and memory systems supply relevant knowledge. The precise neural format of a situation model remains an open research question.

### Step 9 — Integrate linguistic-semantic and conceptual-event information

**Purpose:** Maintain both the proposition's linguistic structure and the event or concept information needed to understand it.

**Operation:** Preserve links between lexical-semantic features, predicate structure, participant roles, and event knowledge. Do not collapse all information into one undifferentiated representation. A word such as `apologized` activates knowledge about an apology event, but the current sentence specifies particular participants and temporal relations.

**Possible mathematical operations:**

- Feature concatenation or weighted integration:

\[
\mathbf{z}=W_l\mathbf{l}+W_s\mathbf{s}+W_e\mathbf{e}
\]

where \(\mathbf{l}\), \(\mathbf{s}\), and \(\mathbf{e}\) stand for linguistic, semantic, and event-context features.
- Nonlinear transformation to represent interactions:

\[
\mathbf{h}=f(\mathbf{z})
\]
- Similarity or compatibility tests between the current event representation and retrieved knowledge.

These equations describe candidate computational architectures. They do not show that the brain literally concatenates three separate vectors.

**Biological interpretation:** Left temporal and inferior frontal regions contribute to sentence-level linguistic-semantic content, while parietal and other association systems may contribute to linking it with broader knowledge. The boundary between linguistic semantics and conceptual knowledge is debated and likely not perfectly sharp.

### Step 10 — Allocate limited working-memory resources to relevant content

**Purpose:** Keep task-relevant propositions and unresolved information sufficiently accessible without assuming every word remains equally active.

**Operation:** Estimate the current relevance of each active item. Give priority to information needed for the next integration step, such as an unresolved subject, a pending argument, a recent referent, or a proposition relevant to the task. Less relevant details may receive lower activation while remaining recoverable through cues.

The algorithm should not impose a universal fixed number of proposition slots. Working-memory capacity varies with complexity, chunking, interference, attention, and individual differences.

**Possible mathematical operations:**

A priority score may combine relevance, recency, unresolved status, and task demand:

\[
\pi_i = aR_i+bU_i+cT_i+dD_i
\]

where \(R_i\) is current relevance, \(U_i\) unresolved importance, \(T_i\) task relevance, and \(D_i\) dependency importance. Normalize scores if a limited attention budget is modelled:

\[
\alpha_i=\frac{e^{\pi_i}}{\sum_j e^{\pi_j}}
\]

A more elaborate model can use a resource budget \(C\) and allocate activation \(m_i\) subject to \(\sum_i m_i\le C\). The values are abstract resources, not literal units of brain storage.

**Biological interpretation:** The source identifies left inferior frontal and bilateral parietal regions as part of the temporary maintenance network and notes that left IFG/intraparietal sulcus activity can covary with working-memory load. Dorsolateral prefrontal and parietal multiple-demand regions may contribute more when control or memory demands are high.

### Step 11 — Maintain active information over the time needed for comprehension

**Purpose:** Prevent a proposition or unresolved relation from disappearing before later information can be integrated.

**Operation:** Keep relevant representations available through ongoing neural activity, reactivation, or other maintenance mechanisms. The algorithm may model a maintained item as activation that decays unless refreshed, but this is only one possible account. Human working memory is not fully explained by a single persistent-firing mechanism: active maintenance, activity-silent mechanisms, attention, and long-term-memory interaction are all considered in the research literature.

**Possible mathematical operations:**

A simple leaky state model is:

\[
\mathbf{x}_{t+1}=\lambda\mathbf{x}_t+\mathbf{u}_t
\]

where \(\lambda\) is a retention factor between 0 and 1 and \(\mathbf{u}_t\) is new input or refresh. A continuous-time analogue is:

\[
\frac{d\mathbf{x}}{dt}=-\frac{1}{\tau}\mathbf{x}+\mathbf{u}(t)
\]

Here \(\tau\) is a time constant. These equations illustrate decay and refresh; they are not a complete account of biological working memory.

**Potential population-dynamics model:**

\[
\tau\frac{d\mathbf{x}}{dt}=-\mathbf{x}+W_{rec}f(\mathbf{x})+B\mathbf{u}(t)+\boldsymbol{\epsilon}(t)
\]

where recurrent network interactions can help sustain a pattern, \(f\) is a nonlinear activation function, and \(\boldsymbol{\epsilon}\) represents noise. The equation is a standard style of neural-network modelling, not proof that a specific language proposition is maintained by one recurrent circuit.

**Biological interpretation:** Temporarily maintaining sentence content likely involves distributed activity across language and working-memory-related networks. The source mentions left IFG and bilateral parietal regions; it does not establish one exclusive storage location.

### Step 12 — Update the representation as each new word or clause arrives

**Purpose:** Make working memory incremental rather than waiting until the end of the sentence.

**Operation:** For each new piece of interpreted input:

1. Identify which active proposition or situation-model component it belongs to.
2. Add a new predicate, argument, modifier, or relation when required.
3. Resolve an open role if the new input supplies its filler.
4. Revise a candidate interpretation when grammatical or contextual evidence changes.
5. Retain still-relevant information from earlier clauses.
6. Recompute priorities for the updated state.

For instance, in “The student that the teacher praised won,” the system must maintain the incomplete dependency introduced by “the student that…” until later material supplies the relevant structure. A working-memory model that stores only the latest word would fail.

**Possible mathematical operations:**

- State update: \(M_{t+1}=U(M_t,u_{t+1})\).
- Local graph modification rather than full reconstruction.
- Prediction-error calculation:

\[
\delta_t = u_t-\hat{u}_t
\]

where \(\hat{u}_t\) is an expected input representation. Large prediction error may prompt revision or increased attention, but not every unexpected word requires a complete reanalysis.

**Biological interpretation:** Incremental integration requires recurrent communication between language-related populations and working-memory support systems. The timing and exact direction of information flow vary across tasks and remain active research questions.

### Step 13 — Protect active representations from interference

**Purpose:** Reduce confusion between similar propositions, competing referents, and recently activated but irrelevant information.

**Operation:** Detect when two active items share features or compete for the same role. Use distinct entity identities, role labels, temporal context, and task relevance to keep them separate. When necessary, reactivate the target representation using a cue, rather than selecting the most recent item automatically.

**Possible mathematical operations:**

A similarity score between two representations may be:

\[
\operatorname{sim}(\mathbf{x}_i,\mathbf{x}_j)=\frac{\mathbf{x}_i^\top\mathbf{x}_j}{\|\mathbf{x}_i\|\|\mathbf{x}_j\|}
\]

High similarity can predict possible interference in a computational model. A retrieval score can combine similarity with role and context cues:

\[
r_i=\beta_1\operatorname{sim}(q,x_i)+\beta_2\operatorname{role}(q,x_i)+\beta_3\operatorname{context}(q,x_i)
\]

**Biological interpretation:** Interference control may involve selective attention, competitive neural dynamics, and control-related activity. Do not assume that one frontal region acts as a literal gate for all language representations.

### Step 14 — Retrieve maintained information when a later expression depends on it

**Purpose:** Reactivate a previous entity, event, or proposition when a pronoun, ellipsis, connective, or later clause requires it.

**Operation:** Use retrieval cues from the new input. Match them against active and accessible representations using grammatical constraints, semantic compatibility, recency, and discourse focus. If the best match is uncertain, maintain alternatives or request further evidence from subsequent input.

**Possible mathematical operations:**

\[
R_i = \gamma_1C_i+\gamma_2G_i+\gamma_3S_i+\gamma_4F_i
\]

where the components represent cue match, grammatical fit, semantic fit, and focus/prominence. A probability distribution can be formed with softmax, but an algorithm should not force a single retrieval if candidates remain close.

**Biological interpretation:** Retrieval is likely distributed across language and memory-related systems. The source discusses working-memory maintenance, not a single dedicated retrieval centre for every referential dependency.

### Step 15 — Monitor whether the proposition is complete enough for the current task

**Purpose:** Distinguish a fully specified proposition from one that still contains open arguments, ambiguous references, or unresolved relations.

**Operation:** Check for:

- obligatory arguments that have not yet been supplied;
- unresolved pronouns or entity references;
- pending modifiers or operator scope;
- missing links required to interpret the current clause;
- conflicts between the proposition and the current situation model;
- low-confidence inferences that should remain provisional.

A proposition may be complete enough for local interpretation while the overall discourse remains open. Completion is task-relative, not an all-or-nothing property.

**Possible mathematical operations:**

A completion score can summarize the fraction or weighted importance of required fields resolved:

\[
K(P)=\frac{\sum_j w_j c_j}{\sum_j w_j}
\]

where \(c_j\) indicates the degree to which component \(j\) is resolved. This is a bookkeeping metric for the model, not a proposed neural scalar.

### Step 16 — Compress and organize information without losing essential relations

**Purpose:** Keep a compact active representation when the full linguistic sequence is no longer necessary, while preserving information required for the next comprehension step.

**Operation:** Group tightly related information into meaningful chunks: a familiar event with its participants, a proposition with its modifiers, or a sequence of events with a causal/temporal relation. Retain pointers to details that may need to be revisited. Compression must preserve role bindings, negation, quantifier scope, temporal order, and explicit-versus-inferred status.

**Possible mathematical operations:**

- Grouping nodes into higher-level subgraphs.
- Feature selection based on task relevance.
- Loss-aware compression, minimizing discarded information that matters for future interpretation.

An abstract objective is:

\[
\min_{\tilde{M}} D(M,\tilde{M})+\lambda\,\operatorname{Cost}(\tilde{M})
\]

where \(M\) is the original state, \(\tilde{M}\) is a compressed representation, \(D\) measures task-relevant information loss, and `Cost` measures representational complexity. This is an engineering analogy, not a known neural optimization formula.

### Step 17 — Recruit executive control only when maintenance demands require it

**Purpose:** Provide additional control when a sentence is long, structurally complex, ambiguous, or requires several propositions to remain active simultaneously.

**Operation:** Estimate whether ordinary incremental maintenance is sufficient. When the task requires extra selection, manipulation, inhibition, or sustained attention, increase control support. This should not be treated as necessary for every simple sentence.

**Possible mathematical operations:**

- Demand estimation from the number of unresolved dependencies, interference, and task requirements.
- Threshold-based control recruitment.
- Adaptive gain or attention allocation.

For example:

\[
D_t = aN_{open}+bI+cA+dT
\]

where \(N_{open}\) is the number/importance of unresolved dependencies, \(I\) is interference, \(A\) is ambiguity, and \(T\) is task demand. Additional control is modelled as rising when \(D_t\) exceeds a context-dependent threshold. This is not a diagnostic formula for a brain region.

**Biological interpretation:** The source identifies dorsolateral prefrontal and parietal multiple-demand regions as important under high working-memory or task-control demands. Basic combinatorial composition can occur without strong recruitment of these domain-general systems.

### Step 18 — Keep a coherent active state and pass it to downstream systems

**Purpose:** Make the current proposition and situation model available for continued discourse comprehension, later memory operations, or an immediate response.

**Operation:** Consolidate the active representation without implying that it has become permanent. Provide downstream systems with:

- proposition and event structure;
- explicit semantic roles and referential links;
- relevant discourse and situation-model relations;
- inference/provenance labels;
- unresolved ambiguities and confidence information;
- pointers to details that can be reactivated if needed.

**Possible mathematical operation:** The final state can be represented as a structured output:

\[
O_t = \{P_{active},G_{discourse},S_{situation},U_{unresolved},Q_{confidence}\}
\]

This output is a functional interface for later processing, not a claim that the brain produces a literal data object.

**Biological interpretation:** The source describes temporary maintenance by language and parietal working-memory-related regions, with broader networks assisting according to the task. Long-term encoding and decision selection are downstream and will be treated separately.

---

## 5. Unified algorithm: end-to-end pseudocode

```text
ALGORITHM Maintain_Propositional_Working_Memory

INPUT:
    current_semantic_input
    current_parse
    contextual_and_pragmatic_information
    active_working_memory_state
    current_task_state

OUTPUT:
    updated_propositional_state
    updated_situation_model
    unresolved_items
    downstream_ready_representation

1.  state <- load(active_working_memory_state)
2.  input <- collect(current_semantic_input,
                     current_parse,
                     contextual_and_pragmatic_information)

3.  proposition_candidates <- segment_into_proposition_candidates(input)

4.  FOR each candidate proposition p:
5.      identify predicates, events, and states in p
6.      identify entity and event mentions
7.      compare new mentions with active discourse entities
8.      resolve only those references supported by sufficient evidence
9.      preserve competing antecedents when evidence is insufficient

10.     FOR each predicate or event e in p:
11.         identify its argument roles
12.         bind each available argument to its semantic role
13.         retain open roles when required arguments have not arrived
14.         attach tense, aspect, negation, modality, quantifiers,
            and modifiers to their correct scope

15.     label each proposition component as explicit, entailed,
        inferred, presupposed, assumed, or unresolved
16.     store confidence/provenance separately from proposition content
17.     connect p to related propositions using temporal, causal,
        conditional, contrastive, or referential links when justified
18.     update the situation model with supported event/state changes

19.     integrate linguistic-semantic and event/context features
20.     add or revise p and its links in the active representation

21. RECOMPUTE priority for active propositions, entities,
    unresolved dependencies, and task-relevant information
22. MAINTAIN high-priority items through available active/refresh mechanisms
23. REDUCE priority of less relevant items without deleting necessary pointers
24. CHECK for interference and distinguish competing entities/events
25. RETRIEVE earlier representations when new input depends on them
26. CHECK completeness, unresolved roles, ambiguity, and consistency
27. IF new evidence changes an earlier interpretation:
28.     revise the affected proposition and situation-model links
29.     preserve a record of what remains uncertain
30. END IF

31. IF working-memory or control demand is high:
32.     request additional selection, attention, or executive control
33. END IF

34. compress tightly related information while preserving essential relations
35. keep literal and inferred content distinguishable
36. package active propositions, situation links, and unresolved items
37. return the updated state for ongoing comprehension or downstream use

END ALGORITHM
```

---

## 6. Worked examples

### Example A — A simple predicate–argument proposition

Sentence: **“Omar opened the door.”**

1. The predicate is `OPEN`.
2. `Omar` is bound as the agent.
3. `the door` is bound as the patient/theme.
4. Past tense is attached to the event.
5. The proposition is stored in structured form:

\[
P_1=\operatorname{OPEN}(\text{agent}=\text{Omar},\;\text{patient}=\text{door},\;\text{time}=past)
\]

6. The participants are linked to the current discourse entity records.
7. The proposition remains active while subsequent sentence material is interpreted.

This is not simply a memory of the word sequence `Omar / opened / the / door`. It is a structured representation that preserves the event and role relations.

### Example B — Maintain a referent across sentences

Text: **“The glass was on the table. It fell.”**

After the first sentence, the active model contains an entity `glass`, a location relation `ON(glass, table)`, and a discourse referent for the glass. When `It` arrives, the system compares candidate entities and links the pronoun to the glass if the evidence supports that interpretation. It then adds a falling event:

\[
P_1=\operatorname{ON}(\text{glass},\text{table})
\]

\[
P_2=\operatorname{FALL}(\text{glass})
\]

The representation can also update the state of the glass: the earlier location may no longer describe its current state after the fall. The model must distinguish the prior state from the event's later consequences.

### Example C — Preserve an inferred relation separately

Text: **“Lisa apologized. Peter shrugged.”**

The explicit propositions are:

\[
P_1=\operatorname{APOLOGIZE}(\text{Lisa})
\]

\[
P_2=\operatorname{SHRUG}(\text{Peter})
\]

A discourse interpretation may infer that Lisa apologized to Peter. If the model adds this relation, it records:

\[
\operatorname{recipient}(P_1)=\text{Peter}\quad [\text{inferred}]
\]

The system therefore retains both what the sentences explicitly state and what it inferred to make the discourse coherent. This distinction is essential because the inference could be revised if later context shows that Lisa apologized to someone else.

### Example D — Maintain a long-distance dependency

Sentence: **“The report that the scientist who the committee interviewed wrote was published.”**

As the sentence unfolds, the parser and working-memory system must maintain dependencies across intervening material. The representation may include partially filled roles until the later verb supplies the missing relationship. A crude last-in-first-out buffer would not be sufficient; retrieval must use grammatical and semantic cues to access the appropriate pending representation.

This example illustrates why maintaining the current proposition involves both a structured representation and controlled retrieval of earlier material.

### Example E — Preserve negation and scope

Sentence: **“Not every visitor signed the form.”**

A correct representation must attach negation to the quantificational structure. It must not simply record `NOT` as a property of an otherwise complete proposition and lose the operator's scope. The relevant structure is approximately:

\[
\neg\forall x\,[VISITOR(x)\rightarrow SIGN(x,form)]
\]

This differs from a structure equivalent to:

\[
\forall x\,[VISITOR(x)\rightarrow\neg SIGN(x,form)]
\]

The working-memory representation must retain the operator and its scope so that later reasoning does not change the sentence's meaning.

---

## 7. Neural systems and candidate biological operations

The supplied source describes the following broad functional associations. They are not one-region/one-operation assignments.

| Brain region/network | Role described in the source | Possible computational contribution in this algorithm | Caution |
|---|---|---|---|
| Left inferior frontal gyrus (IFG) | Contributes to proposition construction and verbal working-memory-related maintenance | Structural-semantic integration, controlled retrieval, maintenance under verbal load | IFG is functionally heterogeneous; it is not a literal proposition box |
| Left middle temporal gyrus (MTG) | Contributes to semantic composition and initial propositional construction | Access to lexical/event-semantic information and integration with predicate structure | Semantic processing is distributed across temporal and other regions |
| Bilateral parietal regions | Participate in temporary maintenance; activation can relate to working-memory load | Attention to active representations, maintenance/retrieval support, and resource allocation | The source does not define a single parietal storage mechanism |
| Intraparietal sulcus (IPS) | Part of frontoparietal working-memory/control systems under load | Selection, prioritization, and attention to task-relevant information | Activity is not specific to language or proposition storage |
| Angular gyrus | May help bind propositions into a broader model | Integration of semantic and relational information | Its exact contribution and functional boundaries are debated |
| Posterior cingulate cortex | May participate in broader situation-model integration | Linking current meaning with internally maintained contextual information | Avoid treating it as a dedicated sentence-memory store |
| Dorsolateral prefrontal cortex (DLPFC) | Supports demanding working-memory and decision-related tasks | Control, manipulation, interference management, and goal-dependent selection | Not required to the same degree for every ordinary sentence |
| Medial temporal lobe (MTL) | Mentioned in the source as relevant to later episodic/semantic encoding | Potential downstream interaction when sentence information is learned or retained | Durable encoding is a later operation, not the central goal of this part |

### Candidate neural computations

1. **Synaptic integration:** postsynaptic potentials sum over time and across inputs. This has a direct biophysical basis, although it does not by itself explain propositional representation.
2. **Excitation and inhibition:** neural populations can increase or suppress one another's activity, supporting selective maintenance and competition.
3. **Recurrent population dynamics:** feedback connections can help stabilize or reactivate a task-relevant pattern. This is a model family, not a complete explanation of human working memory.
4. **Attention-dependent gain:** top-down signals may change the impact of selected inputs, making relevant representations more influential.
5. **Pattern separation and pattern completion:** memory-related systems can, in some contexts, support distinguishing similar experiences or retrieving a representation from partial cues. Their exact role in sentence-level working memory depends on the task and should not be overgeneralized.
6. **Oscillatory coordination:** neural rhythms may coordinate communication and sampling across regions, but no single oscillation is a universal working-memory clock or a direct code for a proposition.
7. **Activity-silent or synaptic-state proposals:** short-lived changes in synaptic efficacy may help retain information even when continuous elevated firing is reduced. These proposals complement, rather than simply replace, active-maintenance accounts.

---

## 8. Mathematical operations inventory

The following table collects the main mathematical operations used to describe the algorithm. Some have direct biological analogues; others are engineering abstractions.

| Operation | Example expression | Purpose in the algorithm | Interpretation level |
|---|---|---|---|
| State update | \(x_t=F(x_{t-1},u_t,c_t)\) | Incorporate new input into the active representation | Cognitive/computational abstraction |
| Segmentation/classification | \(B_t=\sigma(w^Tf_t+b)\) | Estimate proposition or clause boundaries | Algorithmic analogy |
| Weighted candidate scoring | \(s_j=\sum_k w_k f_{jk}\) | Compare antecedents or event interpretations | Algorithmic analogy |
| Softmax normalization | \(p_j=e^{s_j/\tau}/\sum_ke^{s_k/\tau}\) | Maintain graded candidate preferences | Normative/computational model |
| Role–filler binding | \(B=\sum_r\mathbf r\otimes\mathbf e_r\) | Preserve which participant fills which role | Representation model |
| Graph construction | \(G=(V,E)\) | Link propositions, entities, events, and relations | Symbolic abstraction |
| Temporal constraint checking | Interval relations over \([t_s,t_e]\) | Preserve event order and overlap | Computational abstraction |
| Feature integration | \(z=W_l l+W_s s+W_e e\) | Combine linguistic, semantic, and event information | Neural-network analogy |
| Leaky maintenance | \(x_{t+1}=\lambda x_t+u_t\) | Model gradual loss and refresh | Neural-population abstraction |
| Recurrent dynamics | \(\tau\dot{x}=-x+W_{rec}f(x)+Bu+\epsilon\) | Model possible stabilization of active patterns | Neural-computation model |
| Similarity | \(sim(x_i,x_j)=\frac{x_i^Tx_j}{\|x_i\|\|x_j\|}\) | Estimate overlap/interference between representations | Vector-model analogy |
| Cue-based retrieval | \(r_i=\sum_k\beta_k c_{ik}\) | Retrieve a prior representation using multiple cues | Computational abstraction |
| Priority allocation | \(\pi_i=aR_i+bU_i+cT_i+dD_i\) | Allocate limited active attention | Algorithmic analogy |
| Confidence updating | \(q'=q+\eta(e-q)\) | Revise confidence as evidence arrives | Statistical analogy |
| Compression objective | \(\min D(M,\tilde M)+\lambda Cost(\tilde M)\) | Reduce active complexity while preserving task-relevant meaning | Engineering analogy |
| Demand threshold | \(D_t>\theta\) | Model when extra control may be needed | Cognitive-control analogy |

### What should not be inferred from these equations

- A softmax distribution does not prove that neurons explicitly encode normalized probabilities.
- A graph representation does not prove that the brain stores literal graph data structures.
- A decay parameter does not imply that every memory trace decays at one fixed rate.
- A resource budget does not imply a universal fixed number of propositions that every person can hold.
- A neural-network recurrence does not show that one anatomical region maintains the whole proposition.
- A scalar confidence variable is an implementation convenience; subjective certainty and neural population codes are more complex.

---

## 9. Failure modes and safeguards

1. **Role reversal:** agent and patient are swapped. Safeguard: preserve explicit role labels and check them against grammatical, morphological, and semantic constraints.
2. **Pronoun misbinding:** a pronoun is attached to the most recent noun without considering agreement or discourse context. Safeguard: preserve candidate antecedents and score multiple cues.
3. **Inference presented as fact:** a plausible recipient, cause, or intention becomes indistinguishable from explicit content. Safeguard: maintain provenance labels.
4. **Loss of negation or scope:** the operator is retained but attached to the wrong proposition. Safeguard: store scope structurally.
5. **Temporal flattening:** event order is lost when propositions are compressed. Safeguard: retain temporal links and relevant interval constraints.
6. **Causal overreach:** two successive events are treated as cause and effect without evidence. Safeguard: distinguish temporal succession from causation.
7. **Premature closure:** an incomplete proposition is marked complete before an argument or dependency arrives. Safeguard: retain open roles and unresolved dependency records.
8. **Over-retention:** every detail remains equally active, producing overload. Safeguard: prioritize by current task and dependency importance, while retaining retrievable pointers.
9. **Excessive decay:** earlier information becomes unavailable before it can be integrated. Safeguard: use cue-triggered reactivation and maintain high-priority dependencies.
10. **Similarity interference:** two people or events with similar descriptions become confused. Safeguard: use identity, role, time, and discourse-context features.
11. **Overconfident resolution:** one antecedent or inference is selected despite nearly equal alternatives. Safeguard: preserve ambiguity or defer the choice.
12. **Compression destroys meaning:** role bindings, negation, quantifier scope, or provenance are lost. Safeguard: make compression sensitive to semantic relations, not just word count.
13. **Situation-model rigidity:** a later sentence corrects an earlier assumption but the model is not revised. Safeguard: permit local graph revision and reactivation of affected propositions.
14. **Executive over-recruitment:** every sentence is treated as requiring high-level control. Safeguard: make control demand-sensitive.
15. **False anatomical localization:** a computational operation is assigned exclusively to one region. Safeguard: describe distributed network contributions and acknowledge individual/task variability.

---

## 10. Biological and scientific cautions

1. **Working memory is not simply a passive storage container.** Maintaining sentence meaning requires selection, updating, retrieval, and coordination with ongoing comprehension.
2. **Propositions are cognitive descriptions, not necessarily literal neural objects.** A predicate–argument record is a useful model for explaining semantic structure; the brain may encode the relevant information in distributed patterns.
3. **Working-memory maintenance is not fully settled.** Persistent neural activity is one influential account, but activity-silent, synaptic, attentional, and long-term-memory interaction accounts are also considered.
4. **The source's regional descriptions are broad.** Left IFG, MTG, angular gyrus, posterior cingulate, and bilateral parietal regions have multiple functions and participate in larger networks.
5. **The language network and domain-general working-memory systems interact without becoming identical.** Linguistic composition can proceed with relatively little multiple-demand recruitment; more demanding sentences can recruit additional control systems.
6. **The algorithm does not assume a rigid seven-item or other universal capacity limit.** Effective capacity depends on material, task, chunking, similarity, interference, attention, and the individual.
7. **A situation model need not be a visual image.** It may encode relations, event structure, causal dependencies, and social information in abstract form.
8. **Contextual enrichment remains defeasible.** Inferred details must be available for revision when later information contradicts them.
9. **fMRI activation is not proof of necessity.** Lesion, stimulation, electrophysiological, behavioral, and computational evidence are needed to constrain causal claims.
10. **The equations are not measured neural constants.** Parameters such as decay rates, thresholds, weights, and confidence values are modelling choices unless independently estimated in a specified experiment.
11. **Individual differences matter.** Working-memory strategy, language experience, age, attention, and task demands can change which representations remain active and how they are retrieved.
12. **This section stops before long-term encoding and decision selection.** The fact that sentence meaning is available in working memory does not mean it has been consolidated into long-term memory or used to choose an action.

---

## 11. References and source alignment for Part 6

This part is based primarily on the supplied Library document, **“Language Comprehension_ From Words to Contextual Meaning.txt.”** The source describes propositions as structured predicate–argument meanings, refers to construction–integration approaches to text representation and situation models, and identifies neural contributions from left IFG, MTG, angular gyrus, posterior cingulate, bilateral parietal regions, and prefrontal control systems.

1. **Kintsch's construction–integration framework**, as summarized in the supplied source. It motivates distinguishing a textbase of propositions from a richer situation model. The present graph and update equations are explanatory reconstructions, not a claim that the framework uses these exact equations.
2. **The source's proposition-formation discussion.** It associates initial composition with left IFG and MTG and gives predicate–argument representations such as `KNOW(John, rain)` and `DRIVE(He, nails)` as examples of structured sentence meaning.
3. **The source's working-memory maintenance discussion.** It describes temporary buffering by a network including left IFG and bilateral parietal regions, with left IFG/IPS activation related to working-memory load in sentence-coherence tasks.
4. **The source's broader-model discussion.** It notes possible contributions of angular gyrus and posterior cingulate in binding propositions into a broader situation model.
5. **The source's executive-control discussion.** It identifies dorsolateral prefrontal and parietal multiple-demand regions as relevant when memory and task demands are high, and distinguishes these systems from the core language network.

### Mapping to the supplied source's sequence

| Source concept | Where it is implemented in this algorithm |
|---|---|
| Sentence meaning held in propositional form in working memory | Steps 1–5 and 18 |
| Predicate–argument structure | Steps 2 and 4; data model and worked examples |
| Composition in left IFG and MTG | Steps 2, 4, and 9; neural mapping table |
| Binding into a broader model through angular gyrus and posterior cingulate contributions | Steps 7–9; neural mapping table |
| Temporary buffering through left IFG and bilateral parietal regions | Steps 10–14; neural mapping table |
| Verbal working-memory load | Steps 10, 13, and 17 |
| Contextual inferences becoming part of the mental model | Steps 6–9 and Worked Example C |
| DLPFC coordination under demanding tasks | Step 17 |
| Information available for later memory or decision use | Step 18, as a handoff only; implementation deferred to the next point |

---

**End of Part 6.** The next source-related point is **using the maintained proposition for longer-term memory storage or immediate decision-making**. That includes memory encoding, retrieval for questions or judgments, option comparison, response selection, and the distinction between temporary sentence maintenance and durable memory. It has not been implemented in this installment.


# PART 7 — LONG-TERM MEMORY STORAGE AND DECISION USE OF THE PROPOSITION

**Source point:** “Long-term storage and decision use” in *Language Comprehension_ From Words to Contextual Meaning.txt*.

**Position in the sequence:** This stage receives the structured, context-enriched proposition maintained in working memory by Part 6. It branches into (A) encoding and later retrieval from long-term memory, and/or (B) immediate use for answering, judging, choosing, or planning. These branches can operate together: a person may answer a question now while also retaining some aspects of the sentence for later.

**Scope boundary:** This part does not re-implement perception, lexical retrieval, parsing, semantic composition, pragmatic integration, or short-term maintenance. It treats their output as an input record and describes what may happen next.

---

## 1. Purpose and central computational problem

The brain does not need to preserve every understood sentence as a permanent, word-for-word record. It must decide what information is useful to retain, integrate it with existing knowledge when appropriate, and make relevant information available when a later question or decision requires it. A proposition can therefore follow several paths:

1. **Transient use:** it supports the current conversation or task and then becomes less accessible.
2. **Episodic encoding:** it becomes part of a memory for a particular event or experience, retaining some combination of what happened, who was involved, when and where it happened, and the surrounding context.
3. **Semantic learning:** its content updates general knowledge, concepts, or relationships that can be used without recalling the original learning episode.
4. **Immediate decision use:** it is retrieved or kept active to answer a question, make an acceptability judgment, select a response, or plan an action.
5. **Combined path:** it supports an immediate response while some of its content is also encoded for later retrieval.

The algorithm therefore has two main branches, with a shared preparation stage:

```text
Context-enriched proposition in working memory
                    |
      Validate content and provenance
                    |
       Estimate task and retention needs
                    |
          +---------+----------+
          |                    |
   LONG-TERM MEMORY       IMMEDIATE USE
          |                    |
 episodic encoding       retrieve relevant facts
 semantic integration    compare possible answers
 consolidation           evaluate options
 later retrieval         select response/action
          |                    |
          +----------+---------+
                     |
          Update working model / task state
```

This is a computational description of possible functional organization. It is not a claim that the brain executes a literal serial program or that each box corresponds to one isolated anatomical region.

---

## 2. Input and output specification

### 2.1 Input record

The input is the Part 6 proposition or situation-model representation. A useful computational record is:

```text
PROPOSITION_RECORD = {
    proposition_id,
    predicate,
    arguments,
    semantic_roles,
    modifiers,
    negation_and_scope,
    quantifiers_and_scope,
    tense_and_aspect,
    temporal_relations,
    causal_relations,
    discourse_links,
    contextual_inferences,
    source_provenance,
    confidence_by_component,
    unresolved_variables,
    working_memory_activation,
    current_task_goal
}
```

The proposition may contain multiple linked events. Each component should retain its source status. For example:

```text
EXPLICIT: Lisa apologized.
EXPLICIT: Peter shrugged.
INFERRED: Lisa may have apologized to Peter.
UNCERTAIN: Peter may be the recipient of the apology.
```

The inferred recipient must not silently become an explicit fact. This matters when the proposition is later used to answer a question or stored as an episode.

### 2.2 Outputs

The procedure can produce one or more of the following:

- an episodic memory trace linked to time, place, people, and context;
- a semantic update to existing knowledge, if warranted;
- a memory-retrieval cue or index;
- a task answer, judgment, or selected action;
- a confidence estimate and provenance record for the output;
- an updated working-memory state;
- a decision to defer, request clarification, or preserve uncertainty rather than force a choice.

---

## 3. Brain systems potentially involved

The source identifies medial temporal lobe circuitry for longer-term memory and prefrontal coordination for retrieval and decision use. The following mapping expands those roles while avoiding a one-region/one-operation interpretation.

| System or region | Possible contribution to this stage | Important qualification |
|---|---|---|
| Hippocampal formation | Rapidly binds elements of an episode and their relations; supports later recollection and pattern completion | Not a literal database; hippocampal involvement varies with memory type and task |
| Entorhinal cortex | Interface between distributed cortical representations and hippocampal circuitry | Contains functionally differentiated pathways; not merely a passive relay |
| Perirhinal cortex | Contributes object/item representations and familiarity-related information | Its contributions overlap with wider medial temporal networks |
| Parahippocampal cortex | Contributes contextual and scene-related representations | The precise division of labor depends on task and content |
| Neocortical association regions | Represent distributed conceptual, perceptual, linguistic, and event knowledge; participate in learning and later reinstatement | A remembered proposition is not necessarily stored as one local cortical record |
| Medial prefrontal cortex | Supports schema-related processing, contextual organization, memory-guided inference, and coordination with hippocampal systems | Its contribution is not limited to memory and is sensitive to task demands |
| Lateral prefrontal cortex, including DLPFC | Maintains goals, controls retrieval, compares candidate answers, and coordinates task-relevant information | Executive control is not identical to the memory trace itself |
| Orbitofrontal / ventromedial prefrontal regions | May contribute to value, outcome expectation, and evaluation of options | Their precise role varies across decision paradigms; avoid assigning all valuation to one region |
| Anterior cingulate and related control circuitry | May contribute to conflict, error, uncertainty, and need-for-control monitoring | A conflict signal is not a complete decision by itself |
| Parietal and posterior medial regions | May contribute to attention to retrieved information, contextual representation, and memory-guided evaluation | Functions are distributed and not exclusive to language |
| Basal ganglia and motor-related circuits | Participate in action gating and implementation when a decision requires an overt response | These are downstream action systems, not the semantic-memory store |

A key architectural distinction is between **the content being remembered**, **the control process that chooses what to retrieve**, and **the action-selection process that uses the retrieved content**. These functions interact but should not be collapsed into one operation.

---

## 4. Algorithm A — Shared preparation of a proposition for memory or use

### Step A1 — Receive the stable-enough working-memory representation

**Purpose:** Receive the proposition from Part 6 without assuming that the earlier representation is perfectly complete.

**Operations:**

1. Read the predicate, arguments, role bindings, modifiers, scope, event relations, and discourse links.
2. Identify which components are explicit, inferred, predicted, or unresolved.
3. Preserve source and confidence metadata for each component.
4. Check whether any critical variable is still open, such as an unresolved pronoun, an uncertain event participant, or an incomplete temporal relation.
5. If the current task can proceed despite an unresolved detail, retain the uncertainty marker. If the detail is essential, request additional context or delay the decision.

**Possible computation:** Represent confidence for component `i` as `c_i ∈ [0,1]`, while keeping provenance as a separate categorical label. Confidence must not erase provenance: a highly plausible inference remains an inference.

**Output:** A validated proposition record.

### Step A2 — Determine the current task goal

**Purpose:** Decide what the proposition is currently needed for.

**Operations:**

1. Detect whether the person is answering a question, following an instruction, evaluating truth or acceptability, learning new information, planning, or simply following a conversation.
2. Extract the relevant target: for example, “Who apologized?”, “Was the sentence coherent?”, or “What should happen next?”
3. Identify the time horizon: immediate response, near-future task, or later retention.
4. Keep the goal active so later retrieval and comparison are directed by it.

**Possible computation:** Encode task goal as a vector `g`. Score each proposition component `x_i` for task relevance using `r_i = similarity(x_i, g)` or a learned relevance function. This is an algorithmic abstraction; no claim is made that the brain explicitly calculates cosine similarity.

**Output:** A goal-conditioned representation and a task context.

### Step A3 — Estimate the need for durable retention

**Purpose:** Decide whether the content should receive additional encoding priority.

**Operations:**

1. Assess novelty: is the information new relative to existing knowledge?
2. Assess relevance: does it matter for a current goal or future task?
3. Assess distinctiveness: does it differ from similar recent episodes?
4. Assess emotional or motivational significance where relevant.
5. Assess intentional learning demands: has the person been asked to remember it?
6. Estimate whether the information is likely to be useful later.
7. Avoid assuming that attention or emotional intensity guarantees accurate memory.

**Possible computational model:**

\[
E = w_n N + w_r R + w_d D + w_g G + w_s S
\]

where `N` is novelty, `R` is task relevance, `D` is distinctiveness, `G` is goal/intentional-learning relevance, and `S` is salience. The weights are modelling parameters, not fixed biological constants. Emotional salience can alter encoding, but it can also bias or distort what is retained.

**Output:** An encoding-priority estimate `E`.

### Step A4 — Choose a retention/use route

**Purpose:** Select whether to prioritize transient task use, episodic encoding, semantic integration, or more than one path.

**Operations:**

1. If there is no meaningful retention need, keep the proposition available for the immediate task and allow it to lose activation later.
2. If the proposition describes a particular event or learning episode, prepare an episodic representation.
3. If it expresses generalizable factual or conceptual knowledge, consider semantic integration.
4. If it is both a specific experience and a source of general knowledge, preserve the episodic source while allowing a separate semantic update.
5. Never force a semantic update from one weak or unreliable example.

**Possible computation:** Use route probabilities `P(route | E, goal, content, prior knowledge)` as a design abstraction. The output can select multiple routes rather than a mutually exclusive class.

**Output:** One or more route flags: `TRANSIENT`, `EPISODIC`, `SEMANTIC`, `IMMEDIATE_DECISION`.

---

## 5. Algorithm B — Episodic encoding of the proposition

Episodic encoding concerns memory for a particular occurrence or learning episode. It is not simply saving a sentence string. It can bind the proposition's content to contextual information such as the speaker, situation, temporal order, location, and other co-occurring features.

### Step B1 — Extract the event's central content

**Purpose:** Identify what the episode is about.

**Operations:**

1. Identify the main event or state represented by the proposition.
2. Extract the participants and their roles.
3. Preserve event features that distinguish this occurrence from similar events.
4. Identify the outcome, if it is known.
5. Retain uncertainty where the outcome or participant identity is unclear.

**Possible representation:**

```text
EVENT_CORE = {
    event_type,
    participants,
    participant_roles,
    object_or_topic,
    action_or_state,
    outcome,
    certainty
}
```

**Output:** An event-content feature set.

### Step B2 — Bind the event to its context

**Purpose:** Preserve the “what, where, when, and under what circumstances” relations that distinguish an episode.

**Operations:**

1. Gather contextual features that were actually available during the event: speaker, surrounding scene, temporal position, location, social setting, and preceding/following events.
2. Bind those features to the event representation.
3. Preserve temporal order and relevant causal or discourse links.
4. Distinguish directly perceived context from inferred context.
5. Avoid inventing missing contextual details merely to complete the record.

**Possible computation:** Let `e` be the event feature vector and `c` the context feature vector. A binding model can represent an episodic trace as `m = Bind(e,c)`. One mathematical analogy is a conjunctive or tensor-like representation `m = e ⊗ c`; the brain is not known to store literal tensor products in this form.

**Output:** An event-context binding candidate.

### Step B3 — Separate item and context components while preserving their links

**Purpose:** Permit later retrieval from an item cue, a context cue, or a combination, without confusing one event with another.

**Operations:**

1. Build a representation of the entities and objects involved.
2. Build a representation of spatial, temporal, and situational context.
3. Link the two representations through their shared episode.
4. Preserve distinguishing features where similar events occur close together.
5. Keep the links bidirectionally accessible where the system supports such retrieval.

**Neural interpretation:** Medial temporal lobe pathways include cortical systems that represent item-related and contextual features and hippocampal circuitry that helps bind them. The exact processing is distributed and more complex than a simple two-list join.

**Possible computation:**

\[
M_{episode} = f_{bind}(I,C,R)
\]

where `I` is item information, `C` is contextual information, and `R` is the set of relations that bind them.

**Output:** An integrated episodic trace candidate.

### Step B4 — Differentiate similar episodes

**Purpose:** Reduce confusion between events that share many features.

**Operations:**

1. Compare the candidate episode with nearby or previously encoded episodes.
2. Detect overlap in participants, event type, location, and wording.
3. Preserve the distinctive combination of contextual features.
4. If two events are highly similar, attach discriminating details such as time, order, outcome, or source.
5. Avoid merging two separate occurrences into one memory unless the task is explicitly to summarize a repeated pattern.

**Possible computation:** A similarity function `s(m_i,m_j)` estimates overlap. A pattern-separation model maps similar inputs to more distinguishable representations, for example by increasing separation between their codes. This is a computational model of hippocampal function, not proof that the brain uses a specific vector-distance formula.

**Output:** A differentiated episodic representation.

### Step B5 — Encode relations, not just isolated items

**Purpose:** Preserve who did what to whom and how events relate.

**Operations:**

1. Maintain participant identities and semantic roles.
2. Preserve event order: before, during, after, simultaneous.
3. Preserve causal relations only when supported.
4. Preserve social relations such as speaker, listener, intention, and recipient where available.
5. Maintain links between related propositions in the same discourse or episode.

**Possible representation:** An event graph `G=(V,E)`, with nodes for events/entities and edges for agent, patient, temporal, causal, and discourse relations. Again, this is a useful model, not a claim that neurons store a literal graph data structure.

**Output:** A relational episodic representation.

### Step B6 — Initiate synaptic and network-level encoding

**Purpose:** Convert currently active representations into changes that can support future reactivation.

**Operations:**

1. Co-active cortical patterns provide information about the event's components.
2. Medial temporal circuitry receives converging information through entorhinal and related pathways.
3. Activity-dependent plasticity changes the efficacy of connections among participating neural populations.
4. The resulting pattern can support later reactivation, association, and retrieval.
5. The strength and precision of encoding depend on attention, novelty, prior learning, interference, and individual state.

**Possible mathematical analogy:** A simplified Hebbian learning rule is:

\[
\Delta w_{ij}=\eta x_i x_j
\]

where `x_i` and `x_j` represent activity in two units, `w_ij` their connection strength, and `η` a learning-rate parameter. More biologically detailed models include timing, neuromodulation, receptor dynamics, and cell-specific plasticity. This simple equation is not a complete account of hippocampal memory formation.

**Output:** An initially encoded, potentially labile memory trace.

### Step B7 — Register source and confidence

**Purpose:** Retain information about how the proposition was learned and how reliable each part was.

**Operations:**

1. Record whether a detail was directly heard, read, seen, recalled, or inferred.
2. Keep source identity and context when relevant.
3. Preserve confidence separately from source category.
4. Maintain distinctions between an event itself and a later belief about that event.
5. Avoid converting repeated retellings into independent evidence when they originate from the same source.

**Possible computation:** Store a provenance field `p_i` and confidence `c_i` for each important component. During retrieval, use provenance as a constraint on what can be asserted. This is an algorithmic safeguard rather than a claim about a single neural confidence register.

**Output:** A trace with source and uncertainty metadata.

---

## 6. Algorithm C — Semantic integration and consolidation

Episodic memory and semantic memory are related but not identical. Episodic memory preserves information about a particular event. Semantic memory supports general knowledge that can be used without remembering the occasion on which it was learned. A sentence can contribute to both, but the transition is not automatic or guaranteed.

### Step C1 — Detect whether the proposition expresses generalizable knowledge

**Purpose:** Identify whether the content can update a concept, fact, rule, or relationship.

**Operations:**

1. Identify factual or conceptual claims that may generalize beyond the current event.
2. Separate stable knowledge from one-off details.
3. Check whether the proposition is merely a speaker's claim, a hypothetical, a joke, a question, or a negated statement.
4. Preserve conditionality and source attribution.
5. Avoid learning a general rule from an isolated example when the evidence is weak.

**Example:** “The museum closes at six today” is a time-bound claim about a specific day. It should not automatically update the general fact “The museum always closes at six.”

**Output:** Candidate semantic knowledge units.

### Step C2 — Compare candidate knowledge with existing knowledge

**Purpose:** Identify whether the new proposition confirms, extends, contradicts, or is unrelated to what is already known.

**Operations:**

1. Retrieve relevant concepts and relations.
2. Compare entity identity, category, time, scope, and source.
3. Determine whether the new information is compatible with existing knowledge.
4. Detect contradictions and scope mismatches.
5. Identify whether a difference can be explained by context, changed circumstances, or uncertainty.

**Possible computation:**

\[
C_{match}=sim(K_{new},K_{existing})
\]

A separate consistency function is required because semantic similarity does not imply truth or compatibility. Two statements can be highly similar yet contradictory.

**Output:** A relation label such as `CONFIRMS`, `EXTENDS`, `CONTRADICTS`, `CONTEXT_DEPENDENT`, or `UNRELATED`.

### Step C3 — Integrate rather than blindly overwrite

**Purpose:** Update knowledge while avoiding destructive replacement of older information.

**Operations:**

1. If the new proposition is consistent and more specific, add the detail under its appropriate conditions.
2. If it provides a valid exception, represent the exception and preserve the broader rule.
3. If it conflicts with prior knowledge, retain provenance and seek corroboration or context.
4. If the claim is unreliable, keep it as a reported claim rather than a verified fact.
5. If the claim is time-sensitive, associate it with its valid time interval.

**Possible computation:** A confidence update can be represented as:

\[
P(H\mid D)=\frac{P(D\mid H)P(H)}{P(D)}
\]

where `H` is a knowledge hypothesis and `D` is new evidence. Bayesian updating is a normative computational framework; it does not establish that the brain explicitly calculates Bayes' rule for every fact.

**Output:** A revised knowledge state with preserved source and temporal scope.

### Step C4 — Consolidate and reorganize representations over time

**Purpose:** Stabilize useful memory and integrate it with existing knowledge.

**Operations:**

1. Newly formed memory traces remain susceptible to interference and change.
2. Reactivation and offline processes can contribute to stabilization and reorganization.
3. Repeated or meaningful information may become more integrated with existing cortical knowledge.
4. Sleep and rest can support some forms of memory consolidation, but effects depend on task, memory type, and conditions.
5. Consolidation does not mean a memory becomes an exact, permanent copy of the original sentence.
6. Episodic detail and generalized knowledge may evolve differently over time.

**Possible computation:** A simplified strengthening model is:

\[
S_{t+1}=S_t+\alpha R_t-\beta I_t
\]

where `S_t` is a modelled trace-strength variable, `R_t` is reactivation or rehearsal, `I_t` is interference, and `α`, `β` are parameters. This is an abstraction of changing accessibility, not a direct measure of synaptic strength or a complete model of systems consolidation.

**Output:** A more stable or reorganized trace, with the understanding that memory remains reconstructive and fallible.

### Step C5 — Preserve the distinction between memory and truth

**Purpose:** Prevent stored information from being treated as objectively verified merely because it is familiar or easily retrieved.

**Operations:**

1. Preserve whether the content was perceived, reported, inferred, imagined, or hypothesized.
2. Keep source reliability separate from familiarity.
3. Record contradictions and later corrections where possible.
4. During later recall, retrieve source information alongside content.
5. If source information is unavailable, report uncertainty instead of inventing provenance.

**Output:** Memory content with calibrated source monitoring.

---

## 7. Algorithm D — Retrieval of stored information when a later task requires it

Long-term memory is useful only when relevant information can be accessed. Retrieval is cue-dependent and can reconstruct a memory rather than replay an exact recording.

### Step D1 — Translate the current question or goal into retrieval cues

**Purpose:** Define what information is needed.

**Operations:**

1. Extract target entities, events, relations, time constraints, and requested level of detail.
2. Identify whether the task requires a specific episode or general knowledge.
3. Build cues from wording, semantic relations, context, and task goals.
4. Avoid using a cue so broad that many unrelated memories compete.

**Possible computation:** Represent a query as cue vector `q` and candidate memories as `m_j`. Initial retrieval activation may be approximated as `a_j = sim(q,m_j)`.

**Output:** A retrieval query and cue set.

### Step D2 — Reactivate potentially relevant memory traces

**Purpose:** Bring candidate information into an accessible state.

**Operations:**

1. Use current cues to activate related representations.
2. Allow associated contextual and relational information to become available.
3. Generate a limited set of candidate memories.
4. Use the hippocampal system as one contributor to relational reinstatement and completion when episodic recall is required.
5. Allow prefrontal control to bias retrieval toward goal-relevant material.

**Possible computation:**

\[
a_j=\lambda_1 sim(q,m_j)+\lambda_2 contextMatch_j+\lambda_3 goalMatch_j
\]

This is a candidate-ranking model, not a literal neural formula.

**Output:** Activated candidate memories.

### Step D3 — Complete partial event information when appropriate

**Purpose:** Recover linked details from a partial cue.

**Operations:**

1. Detect whether the cue matches one component of a known episode.
2. Reactivate the associated participant, event, and context representations.
3. Reconstruct missing links when the memory system supports pattern completion.
4. Distinguish recovered details from details merely guessed from general knowledge.
5. Keep uncertainty where the cue is insufficient to identify one episode uniquely.

**Possible computation:** An autoassociative memory model may estimate a completed state `m* = F(q, W)`, where `W` represents learned associations. This is a useful computational analogy for pattern completion, not a complete biological model of hippocampal retrieval.

**Output:** A candidate reconstructed memory.

### Step D4 — Select among competing memories

**Purpose:** Avoid retrieving a similar but wrong episode.

**Operations:**

1. Compare candidate identity, time, location, participants, and relational structure.
2. Weight distinctive details more strongly than generic similarities.
3. Use source and context cues to reject inappropriate candidates.
4. If two candidates remain plausible, preserve ambiguity or seek another cue.
5. Do not treat retrieval fluency as proof of accuracy.

**Possible computation:** A softmax candidate distribution may be used in a model:

\[
P(m_j\mid q)=\frac{e^{a_j/\tau}}{\sum_k e^{a_k/\tau}}
\]

Here `τ` controls modelled choice stochasticity. This is a formal approximation, not evidence that neurons run a softmax routine.

**Output:** A selected candidate or an unresolved candidate set.

### Step D5 — Verify the retrieved information against the current proposition and task

**Purpose:** Ensure retrieval actually answers the question and does not contradict the currently understood sentence.

**Operations:**

1. Compare the retrieved facts with the current discourse and proposition.
2. Check entity identity and semantic-role consistency.
3. Check temporal order, negation, and quantifier scope.
4. Detect mismatches between remembered facts and new information.
5. Revise the candidate if a better-supported interpretation is available.

**Output:** A task-relevant memory representation with any discrepancies flagged.

---

## 8. Algorithm E — Immediate question answering and judgment

This branch uses the proposition directly, or retrieves additional information, to produce an answer or judgment. It does not require long-term encoding of every sentence.

### Step E1 — Parse the decision request

**Purpose:** Convert the external task into a target response.

**Operations:**

1. Identify the question type: who, what, when, where, why, how, true/false, coherent/incoherent, acceptable/unacceptable, or action choice.
2. Determine the evidence required to answer.
3. Identify whether the answer must be extracted from the proposition or requires additional memory retrieval.
4. Define the response format and constraints.

**Possible computation:** Map task class `T` and proposition `P` to an answer schema `A_schema = g(T,P)`.

**Output:** A decision specification.

### Step E2 — Identify the answer-bearing proposition components

**Purpose:** Select only the information relevant to the requested answer.

**Operations:**

1. Locate the predicate and argument positions relevant to the question.
2. Use semantic roles to distinguish agent, patient, recipient, experiencer, and other roles.
3. Retrieve temporal, causal, or discourse links only when needed.
4. Preserve negation, modality, and uncertainty.
5. Exclude irrelevant material without discarding information that changes the answer.

**Possible computation:** Relevance score for component `i`:

\[
r_i=\alpha sim(x_i,T)+\beta roleMatch_i+\gamma constraintMatch_i
\]

**Output:** A focused evidence set.

### Step E3 — Generate candidate answers or judgments

**Purpose:** Produce possible outputs before selecting one.

**Operations:**

1. Construct one or more candidate answers from explicit proposition content.
2. If the answer requires an inference, generate candidate inferences separately.
3. If the question is underdetermined, include “unknown,” “not stated,” or “cannot be determined” as a legitimate candidate.
4. If the task is an acceptability or coherence judgment, generate a judgment from the relevant linguistic and contextual evidence.
5. Keep hypothetical or counterfactual content distinct from asserted facts.

**Output:** A candidate response set.

### Step E4 — Evaluate evidence for each candidate

**Purpose:** Determine how well each candidate is supported.

**Operations:**

1. Check direct support in the proposition.
2. Check support from retrieved knowledge, if used.
3. Check consistency with contextual information.
4. Detect contradictions or missing premises.
5. Penalize unsupported assumptions.
6. Account for uncertainty and source reliability.

**Possible computation:**

\[
Score(a)=w_e Evidence(a)+w_c ContextFit(a)+w_g GoalFit(a)-w_u UnsupportedAssumptions(a)
\]

A more probabilistic model can estimate `P(a | evidence, context)`. Neither model is asserted to be the exact neural computation.

**Output:** Candidate scores and an evidence trace.

### Step E5 — Compare candidates and monitor conflict

**Purpose:** Identify whether one answer is adequately better supported than alternatives.

**Operations:**

1. Compare candidate support scores.
2. Detect close competition between answers.
3. Identify whether conflict arises from ambiguity, inconsistent facts, uncertain source, or insufficient context.
4. Increase retrieval or control effort if the task is important and the ambiguity is resolvable.
5. If no candidate is adequately supported, preserve uncertainty or request clarification.

**Possible computation:** Let `S_1` and `S_2` be the best and second-best candidate scores. A margin `M=S_1-S_2` can act as a modelled confidence cue. If `M` is too small, the procedure may defer rather than select prematurely.

**Neural caution:** Conflict monitoring and control recruitment involve distributed systems. Anterior cingulate and lateral prefrontal involvement is task-dependent; an algorithmic margin is not a direct measurement of a neural conflict signal.

**Output:** A selected candidate, a request for more evidence, or an explicit uncertain state.

### Step E6 — Calibrate the response

**Purpose:** Match the strength of the answer to the strength of the evidence.

**Operations:**

1. State explicit information as explicit.
2. State a reasonable inference as an inference.
3. Use qualified language when evidence is incomplete.
4. Avoid inventing details to make the answer sound complete.
5. Report “not enough information” when that is the correct result.

**Possible computation:** Estimate confidence `C` from evidence quantity, reliability, consistency, and candidate separation. A calibration function maps raw score to an empirically calibrated probability. The calibration should be learned or validated against task outcomes rather than assumed.

**Output:** A calibrated answer or judgment.

---

## 9. Algorithm F — Decision-making and action planning

When the proposition is used to guide a choice or plan, the task shifts from identifying what the sentence means to evaluating what should be done. This requires a goal, candidate options, consequences, constraints, and some way to compare outcomes.

### Step F1 — Define the decision state

**Purpose:** Construct a representation of the current situation relevant to the choice.

**Operations:**

1. Combine the proposition with the current goal and any relevant retrieved knowledge.
2. Identify the current state, constraints, and unresolved facts.
3. Separate what is known from what is assumed.
4. Identify the time horizon and whether the decision is reversible.
5. Determine whether action is actually required now.

**Output:** A state representation `s`.

### Step F2 — Generate feasible options

**Purpose:** Construct alternative responses or actions.

**Operations:**

1. Generate actions that are compatible with the current state.
2. Include a no-action or wait option where appropriate.
3. Exclude options that violate hard constraints.
4. Avoid generating an unrealistically large option set when time or cognitive resources are limited.
5. Retain an information-gathering option when uncertainty can be reduced.

**Output:** Candidate action set `A={a_1,...,a_n}`.

### Step F3 — Predict likely outcomes

**Purpose:** Estimate what may happen under each option.

**Operations:**

1. Retrieve prior knowledge and relevant episodes.
2. Infer likely consequences from the current context.
3. Generate more than one possible outcome when the future is uncertain.
4. Estimate the likelihood of each outcome, with caution about unsupported precision.
5. Identify consequences for the relevant people, goals, and constraints.

**Possible computation:**

\[
P(o\mid a,s,K)
\]

represents the modelled probability of outcome `o`, given action `a`, current state `s`, and relevant knowledge `K`.

**Output:** An outcome distribution or qualitative forecast for each option.

### Step F4 — Evaluate options against goals and costs

**Purpose:** Compare actions using relevant benefits, risks, effort, and constraints.

**Operations:**

1. Estimate how well each outcome advances the goal.
2. Consider costs, delay, effort, uncertainty, and possible harm.
3. Apply hard constraints before soft preferences.
4. Consider whether the action preserves future options.
5. Avoid treating every decision as a single scalar-value problem; ethical, social, and multi-objective choices can involve incommensurable goals.

**Possible mathematical model:**

\[
EU(a)=\sum_o P(o\mid a,s,K)U(o)-Cost(a)
\]

where `EU` is expected utility, `U(o)` is the modelled value of outcome `o`, and `Cost(a)` captures relevant costs. This is a normative decision model, not a claim that the brain explicitly multiplies probabilities by utilities in every choice.

**Output:** A value or preference profile for each feasible option.

### Step F5 — Accumulate evidence and compare alternatives

**Purpose:** Allow evidence to support or weaken candidate options over time.

**Operations:**

1. Accumulate information from the proposition, memory, current perception, and task constraints.
2. Update support for each option as new evidence arrives.
3. Keep the evidence source and confidence visible.
4. Stop accumulating when a decision is sufficiently supported, when a deadline is reached, or when additional evidence is no longer useful.
5. If evidence remains weak, choose an information-gathering or defer option when available.

**Possible model:** A simplified evidence accumulator can be written:

\[
X_{t+1}=X_t+\delta_t+\epsilon_t
\]

where `δ_t` is evidence favouring an option and `ε_t` is noise. In a drift-diffusion model, decision evidence evolves until it reaches a boundary. This family of models has empirical support in particular decision tasks, but it should not be treated as a universal mechanism for all language-based decisions.

### Step F6 — Apply a decision threshold appropriate to the task

**Purpose:** Balance speed, accuracy, and consequences.

**Operations:**

1. Estimate the cost of an incorrect answer or action.
2. Estimate the cost of delay.
3. Require stronger evidence for irreversible or high-consequence actions.
4. Permit faster decisions for low-risk, familiar, reversible choices.
5. Reassess the threshold when task demands or stakes change.

**Possible computation:** Choose when evidence `X_t` reaches a criterion `θ`, where `θ` depends on the cost structure and current goal. The criterion is a modelling variable, not a fixed neural threshold shared by all people.

### Step F7 — Select and prepare the response or action

**Purpose:** Convert the selected option into an output plan.

**Operations:**

1. Select the candidate with sufficient support or apply the best justified fallback.
2. Convert the decision into a response plan: spoken answer, written answer, acceptance/rejection, or intended action.
3. Preserve the rationale and uncertainty needed for the response.
4. Coordinate with language production or motor systems when execution is required.
5. Do not confuse selecting an action with successfully carrying it out.

**Neural interpretation:** Prefrontal systems can coordinate goal and response selection; basal-ganglia and motor networks may participate when an overt action must be gated and implemented. The exact circuits depend on the type of decision and output.

**Output:** A selected response/action plan.

### Step F8 — Monitor the result and update the model

**Purpose:** Compare the outcome with the prediction and learn from discrepancies.

**Operations:**

1. Observe whether the answer was accepted, corrected, or contradicted by later information.
2. For actions, compare the actual outcome with the predicted outcome.
3. Detect prediction errors or unexpected consequences.
4. Update relevant beliefs, option values, and memory records.
5. Preserve the original decision context so the person can later understand why the choice was made.

**Possible computation:**

\[
\delta_t = Outcome_t-PredictedOutcome_t
\]

This is a generic prediction-error notation. Different learning systems use different error signals; a single scalar error is not a universal description of the brain.

**Output:** Updated task state, predictions, and potentially memory.

---

## 10. Integrated end-to-end algorithm

The following pseudocode combines the shared preparation stage with the two main branches. It is a functional algorithmic model that could be implemented in software; it is not a literal transcription of neuronal code.

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Provenance(str, Enum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"
    REPORTED = "reported"
    HYPOTHETICAL = "hypothetical"
    UNRESOLVED = "unresolved"


@dataclass
class PropositionComponent:
    content: Any
    provenance: Provenance
    confidence: float
    relations: list[str] = field(default_factory=list)


@dataclass
class PropositionRecord:
    components: list[PropositionComponent]
    predicate_structure: Any
    context: dict[str, Any]
    current_goal: dict[str, Any]
    unresolved_variables: list[str] = field(default_factory=list)


@dataclass
class ProcessingResult:
    episodic_trace: Optional[dict[str, Any]] = None
    semantic_update: Optional[dict[str, Any]] = None
    answer: Optional[Any] = None
    action_plan: Optional[Any] = None
    confidence: Optional[float] = None
    deferred_reason: Optional[str] = None


def validate_proposition(record: PropositionRecord) -> bool:
    if not record.components:
        return False
    for component in record.components:
        if not 0.0 <= component.confidence <= 1.0:
            raise ValueError("Component confidence must be in [0, 1].")
    return True


def score_retention_priority(record: PropositionRecord) -> float:
    # The inputs below would be estimated from task and experience signals.
    novelty = float(record.current_goal.get("novelty", 0.0))
    relevance = float(record.current_goal.get("relevance", 0.0))
    distinctiveness = float(record.current_goal.get("distinctiveness", 0.0))
    intentional_learning = float(record.current_goal.get("intentional_learning", 0.0))
    salience = float(record.current_goal.get("salience", 0.0))

    values = [novelty, relevance, distinctiveness, intentional_learning, salience]
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("Retention features must be normalized to [0, 1].")

    weights = (0.20, 0.30, 0.15, 0.25, 0.10)
    return sum(w * x for w, x in zip(weights, values))


def episodic_encode(record: PropositionRecord) -> dict[str, Any]:
    # Preserve the source distinction; do not rewrite inference as fact.
    return {
        "event_content": record.predicate_structure,
        "participants_and_roles": record.context.get("participants_and_roles", {}),
        "time": record.context.get("time"),
        "place": record.context.get("place"),
        "situation": record.context.get("situation"),
        "discourse_context": record.context.get("discourse_context", []),
        "components": [
            {
                "content": c.content,
                "provenance": c.provenance.value,
                "confidence": c.confidence,
                "relations": list(c.relations),
            }
            for c in record.components
        ],
        "encoding_status": "candidate_trace_not_guaranteed_permanent",
    }


def semantic_candidate(record: PropositionRecord) -> Optional[dict[str, Any]]:
    generalizable = record.current_goal.get("generalizable_knowledge", False)
    if not generalizable:
        return None

    return {
        "claim": record.predicate_structure,
        "source": record.context.get("source"),
        "valid_time": record.context.get("valid_time"),
        "provenance": "retain_component_level_provenance",
        "status": "candidate_requires_consistency_and_reliability_check",
    }


def evaluate_candidates(
    candidates: list[Any],
    evidence: list[PropositionComponent],
    context: dict[str, Any],
) -> tuple[Optional[Any], float]:
    if not candidates:
        return None, 0.0

    # Placeholder scoring: a real system must define task-specific evidence rules.
    scored = []
    for candidate in candidates:
        score = 0.0
        for component in evidence:
            if component.content == candidate:
                score += component.confidence
        scored.append((score, candidate))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_candidate = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    margin = max(0.0, best_score - second_score)
    confidence = min(1.0, best_score / max(1, len(evidence)))

    # A real implementation must calibrate this margin on empirical data.
    if margin < 0.10 and len(scored) > 1:
        return None, confidence
    return best_candidate, confidence


def process_proposition(
    record: PropositionRecord,
    task: dict[str, Any],
) -> ProcessingResult:
    if not validate_proposition(record):
        return ProcessingResult(deferred_reason="Invalid or empty proposition record.")

    if record.unresolved_variables and task.get("requires_unresolved_fields", False):
        return ProcessingResult(
            deferred_reason="The task depends on unresolved proposition components."
        )

    result = ProcessingResult()
    retention_priority = score_retention_priority(record)

    # Memory branch: either route may be used, and both may run.
    if task.get("encode_episode", False) or retention_priority >= 0.60:
        result.episodic_trace = episodic_encode(record)

    candidate = semantic_candidate(record)
    if candidate is not None and task.get("allow_semantic_update", False):
        # Integration, verification, and corroboration are deliberately explicit
        # downstream checks; this function does not automatically assert the claim.
        candidate["requires_prior_knowledge_comparison"] = True
        candidate["requires_source_reliability_check"] = True
        result.semantic_update = candidate

    # Immediate-use branch.
    if task.get("requires_answer", False):
        answer, confidence = evaluate_candidates(
            task.get("answer_candidates", []),
            record.components,
            record.context,
        )
        if answer is None:
            result.deferred_reason = "Insufficient or competing evidence for a confident answer."
        else:
            result.answer = answer
            result.confidence = confidence

    if task.get("requires_action_plan", False):
        # The option generator and expected-outcome model are task-specific.
        options = task.get("feasible_options", [])
        if options:
            result.action_plan = {
                "options": options,
                "status": "requires_outcome_prediction_and_cost_evaluation",
            }
        else:
            result.deferred_reason = "No feasible action options were supplied."

    return result
```

### Pseudocode limitations

- The sample code is an architecture sketch, not a biologically realistic neural simulation.
- Its fixed retention weights are placeholders, not measured brain parameters.
- Exact string equality in the candidate scorer is intentionally simplistic and would need semantic-role-aware matching.
- A real memory system must handle interference, time, source reliability, context drift, forgetting, and reconstruction.
- A real decision system needs task-specific option generation, consequence models, calibrated uncertainty, and hard safety constraints.
- Episodic and semantic memory should not be merged into one undifferentiated store.

---

## 11. Worked example 1 — Episodic encoding from a spoken sentence

**Input sentence:** “Yesterday, Maya told me that the old bridge was closed because of flooding.”

### Shared preparation

1. The proposition contains a report event: `TELL(Maya, listener, proposition)`.
2. The embedded proposition is `CLOSED(bridge)` with a causal explanation `BECAUSE_OF(flooding)`.
3. The temporal context is “yesterday,” relative to the time of the conversation.
4. The source is Maya's report, not necessarily direct observation by the listener.
5. The content may matter later because the listener may need to choose a route.

### Episodic path

1. Encode the event of hearing Maya's report.
2. Bind the content to Maya as speaker, the listener as recipient, and the conversation's time and setting if available.
3. Preserve that the bridge closure was reported by Maya.
4. Do not encode “the bridge was definitely closed” as personally witnessed fact.
5. If later the listener must recall where the information came from, retrieve the speaker-context association.

**Possible episodic trace:**

```text
EPISODE:
  event: Maya informed the listener about a bridge closure
  time: yesterday
  speaker/source: Maya
  content reported: bridge closed due to flooding
  direct observation by listener: not established
  confidence in report: depends on trust and corroboration
```

### Semantic path

The sentence should not automatically create a permanent general fact that the bridge is closed. A time-sensitive report should carry a valid time and source. It may inform a temporary route decision, but its current accuracy should be checked before relying on it later.

---

## 12. Worked example 2 — Answering a question from a proposition

**Input sentence:** “Lisa apologized. Peter shrugged.”

**Question:** “Who apologized?”

1. Parse the goal as a `WHO` query over the event `APOLOGIZE`.
2. Locate the agent role of `APOLOGIZE`.
3. Retrieve the explicit role binding `agent = Lisa`.
4. Do not use Peter's presence in the second sentence as a reason to change the agent.
5. Answer “Lisa.”
6. Do not add “to Peter” unless it is clearly marked as an inference or supported by further context.

**Mathematical abstraction:** A role query selects `x` such that `APOLOGIZE(x, ...)` is explicitly represented. The mapping from `WHO` to `agent` is a task-specific semantic operation, not a probabilistic guess from word order alone.

---

## 13. Worked example 3 — Choosing whether to store or act

**Input proposition:** “The meeting starts at 3 p.m. in Room 12.”

**Current task:** Find the room and attend the meeting.

1. The immediate goal is action planning, not abstract knowledge learning.
2. Extract the time and place as task-relevant components.
3. Hold or retrieve the information until the meeting.
4. A reminder or episodic/task memory may be useful because the information must guide later behavior.
5. A general semantic memory update is unnecessary.
6. If a later message changes the room, update the active task representation and preserve the newer source/time information.
7. At the meeting time, retrieve the room cue and use it to guide navigation.

This example illustrates why a proposition can have high immediate and prospective relevance without needing to become general semantic knowledge.

---

## 14. Mathematical operations potentially useful in a computational model

| Operation | Equation or model | Purpose in the algorithm | Scientific status |
|---|---|---|---|
| Relevance weighting | `R = Σ w_i x_i` | Prioritize content for memory or task use | Useful model; weights are task-dependent |
| Associative plasticity | `Δw_ij = η x_i x_j` | Illustrate activity-dependent strengthening of co-active representations | Simplified learning rule, not a complete biological mechanism |
| Temporal plasticity | STDP-style timing window | Represent sensitivity to relative spike timing | Biologically motivated; exact forms vary by synapse and circuit |
| Conjunctive binding | `m = Bind(item, context, relations)` | Link an event to its context | Computational abstraction of relational memory |
| Similarity | cosine, dot product, or learned metric | Retrieve candidate memories from cues | Algorithmic approximation; neural similarity is population- and task-dependent |
| Pattern separation | `d(f(x), f(y)) > d(x,y)` for similar inputs | Reduce interference among similar episodes | Conceptual model of a hippocampal computation |
| Pattern completion | `m* = F(q,W)` | Reconstruct linked event features from partial cues | Autoassociative model; not a complete account of hippocampal recall |
| Bayesian updating | `P(H|D) ∝ P(D|H)P(H)` | Revise knowledge with new evidence | Normative computational framework, not proof of exact neural Bayesian calculation |
| Trace-strength dynamics | `S_(t+1)=S_t+αR_t−βI_t` | Illustrate effects of reactivation and interference | Simplified phenomenological model |
| Cue-based activation | `a_j = λ_1 sim(q,m_j)+λ_2 context_j+λ_3 goal_j` | Rank candidate memories | Algorithmic model |
| Softmax choice | `P_j = exp(a_j/τ)/Σ exp(a_k/τ)` | Convert candidate activation to choice probability | Common model; not evidence for a literal neural softmax |
| Evidence accumulation | `X_(t+1)=X_t+δ_t+ε_t` | Integrate information across time | Supported for some tasks; not universal |
| Drift-diffusion threshold | `dX=μdt+σdW`, stop at boundary | Model speed–accuracy trade-off | Empirically useful in some decisions; scope is task-specific |
| Expected utility | `EU(a)=ΣP(o|a)U(o)−Cost(a)` | Compare action consequences | Normative model, not a universal neural implementation |
| Prediction error | `δ = outcome − prediction` | Update expectations after an outcome | Broad computational principle; signals and circuits differ by learning domain |
| Confidence calibration | `C_cal = f(score, outcomes)` | Match reported certainty to empirical accuracy | Requires behavioral validation; no single neural confidence value |

### A. Temporal integration and plasticity

A detailed biological model could simulate pre- and postsynaptic activity, receptor-mediated plasticity, neuromodulatory influences, and local inhibitory circuits. A simple Hebbian rule captures only the broad principle that correlated activity can change connectivity. It does not model the full molecular and circuit biology of memory.

### B. Relational binding and retrieval

The hippocampal network is often modelled as enabling rapid association among distributed representations. Pattern separation and pattern completion are useful computational descriptions: the first helps distinguish overlapping episodes; the second allows partial cues to reactivate associated information. Neither should be reduced to a single scalar operation in every situation.

### C. Controlled retrieval and selection

Prefrontal systems can help maintain retrieval goals and guide selection among competing memories. The model's relevance scores, cue weights, and candidate thresholds are abstractions for this goal-sensitive control. They should be fitted to behavioral or neural data before being treated as quantitative predictions.

### D. Decision evidence accumulation

Accumulating evidence can explain some speeded decisions. It is especially important to avoid overgeneralizing this model: complex social, moral, linguistic, and long-horizon decisions may recruit multiple interacting processes rather than a single accumulator.

---

## 15. Failure modes and safeguards

1. **Treating every sentence as permanently stored:** Many sentences are used transiently. Safeguard: estimate retention need rather than forcing encoding.
2. **Confusing episodic and semantic memory:** A specific report becomes an unjustified universal fact. Safeguard: retain event context, source, and valid time.
3. **Losing source information:** A reported claim becomes remembered as direct observation. Safeguard: preserve provenance and source monitoring.
4. **Conflating confidence with truth:** A fluent or familiar memory feels certain. Safeguard: distinguish subjective confidence from objective verification.
5. **Overgeneralizing from one example:** A single event becomes a broad rule. Safeguard: require corroboration and scope checks before semantic updating.
6. **Merging similar episodes:** Two events with similar people or content are recalled as one. Safeguard: preserve discriminating temporal and contextual features.
7. **Retrieval by recency alone:** The most recent candidate is selected despite a better contextual match. Safeguard: score role, source, goal, and context cues.
8. **Pattern completion invents missing information:** A plausible detail is reconstructed as if remembered. Safeguard: label recovered, inferred, and unknown details separately.
9. **Ignoring contradictions:** New information is added without revising an incompatible prior belief. Safeguard: run explicit consistency checks and preserve competing sources.
10. **Premature decision:** An answer or action is selected when evidence is weak. Safeguard: use task-sensitive thresholds and permit deferral.
11. **Excessive decision delay:** The system keeps gathering evidence even when the decision is low-risk and time-sensitive. Safeguard: include the cost of delay.
12. **Expected-value oversimplification:** A complex social or ethical decision is reduced to one scalar score. Safeguard: preserve constraints, rights, and multiple objectives separately.
13. **Mixing answer extraction with inference:** A question asks what was stated, but the algorithm supplies what is merely plausible. Safeguard: distinguish direct extraction from inferential answering.
14. **Overwriting time-sensitive information:** A previously valid fact remains active after conditions change. Safeguard: attach temporal validity and update timestamps.
15. **Treating the pseudocode as a brain simulator:** Software data structures are mistaken for literal neural objects. Safeguard: separate functional algorithm, computational hypothesis, and biological mechanism.

---

## 16. What is supported, what is modelled, and what remains uncertain

### Relatively well-supported broad principles

- Episodic memory involves binding event content with contextual features, including temporal and spatial context.
- The hippocampal formation and connected medial temporal pathways play an important role in episodic memory formation and retrieval.
- Prefrontal and hippocampal systems interact during memory encoding and retrieval; prefrontal regions can support goal-directed control over memory.
- Memory is not a verbatim recording of every sentence. It can be selective, reconstructive, and affected by later context and retrieval.
- Some decisions can be described by evidence-accumulation models, with a speed–accuracy trade-off.

### Useful but abstract computational models

- Proposition records, event graphs, item-context tensors, and source labels are implementation devices for a computational system.
- Hebbian learning, pattern separation, pattern completion, Bayesian updating, softmax choice, utility functions, and drift-diffusion models are mathematical tools for describing selected aspects of learning, retrieval, or decision-making.
- These models do not imply that the brain explicitly stores data structures or performs the corresponding textbook equations step by step.

### Open or task-dependent questions

- How particular semantic propositions are encoded across distributed cortical networks.
- How episodic and semantic representations interact over short and long timescales.
- How much a particular memory remains dependent on the hippocampus as it ages; theories of systems consolidation and contextual binding remain debated.
- How sleep, replay, reactivation, and schema knowledge contribute to different types of memory.
- How decision computations are divided across prefrontal, parietal, medial temporal, striatal, and sensory systems for complex language-driven choices.
- How confidence, source monitoring, and uncertainty are represented in the brain.

---

## 17. References and source alignment for Part 7

### Primary source alignment

The supplied Library document, *Language Comprehension_ From Words to Contextual Meaning.txt*, directly motivates this section with the following points:

1. The proposition held in working memory can inform longer-term memory storage or drive decisions, including answering questions and planning actions.
2. Propositions may be encoded into episodic or semantic memory through medial temporal lobe circuits.
3. Answering questions about a sentence can involve prefrontal coordination of propositional retrieval and response selection.
4. Decision tasks that require holding sentence content and comparing alternatives can recruit dorsolateral prefrontal cortex.
5. Contextual inferences can become part of the mental model and therefore affect what is later remembered or used.

### External research grounding

1. **Danieli, K., Guyon, A., & Bethus, I. (2023).** “Episodic Memory formation: A review of complex Hippocampus input pathways.” *Progress in Neuro-Psychopharmacology and Biological Psychiatry*, 126, 110757. DOI: [10.1016/j.pnpbp.2023.110757](https://doi.org/10.1016/j.pnpbp.2023.110757). Reviews cortical inputs and medial temporal pathways involved in representing objects, contextual features, and hippocampal binding.
2. **Eichenbaum, H. (2017).** “Prefrontal–hippocampal interactions in episodic memory.” *Nature Reviews Neuroscience*, 18, 547–558. DOI: [10.1038/nrn.2017.74](https://doi.org/10.1038/nrn.2017.74). Describes complementary prefrontal and hippocampal functions and their interactions during episodic memory.
3. **Preston, A. R., & Eichenbaum, H. (2013).** “Interplay of hippocampus and prefrontal cortex in memory.” *Current Biology*, 23(17), R764–R773. DOI: [10.1016/j.cub.2013.05.041](https://doi.org/10.1016/j.cub.2013.05.041). Reviews interactions in memory encoding, retrieval, and consolidation.
4. **Yonelinas, A. P., Ranganath, C., Ekstrom, A. D., & Wiltgen, B. J. (2019).** “A contextual binding theory of episodic memory: systems consolidation reconsidered.” *Nature Reviews Neuroscience*, 20, 364–375. DOI: [10.1038/s41583-019-0150-4](https://doi.org/10.1038/s41583-019-0150-4). Discusses hippocampal contextual binding and debates about systems consolidation; this is included to avoid presenting one consolidation theory as settled fact.
5. **Lewis, R. L., Vasishth, S., & Van Dyke, J. A. (2006).** “Computational principles of working memory in sentence comprehension.” *Trends in Cognitive Sciences*, 10(10), 447–454. DOI: [10.1016/j.tics.2006.08.007](https://doi.org/10.1016/j.tics.2006.08.007). This is adjacent evidence about cue-based retrieval, interference, and activation dynamics during sentence comprehension; it is not direct evidence about long-term episodic encoding.
6. **Philiastides, M. G., Auksztulewicz, R., Heekeren, H. R., & Blankenburg, F. (2011).** “Causal role of dorsolateral prefrontal cortex in human perceptual decision making.” *Current Biology*, 21(11), 980–983. DOI: [10.1016/j.cub.2011.04.034](https://doi.org/10.1016/j.cub.2011.04.034). Provides causal evidence for a DLPFC contribution to evidence integration in a particular perceptual decision task; it does not establish that every linguistic or everyday decision uses the same mechanism.
7. **Hutcherson, C. A., & Tusche, A. (2022).** “Evidence accumulation, not ‘self-control’, explains dorsolateral prefrontal activation during normative choice.” *eLife*, 11, e65661. DOI: [10.7554/eLife.65661](https://doi.org/10.7554/eLife.65661). Illustrates the importance of interpreting DLPFC activity in relation to evidence accumulation and task structure.
8. **Eichenbaum, H. (2017).** “Memory: Organization and Control.” *Annual Review of Psychology*, 68, 19–45. DOI: [10.1146/annurev-psych-010416-044131](https://doi.org/10.1146/annurev-psych-010416-044131). Reviews hippocampal organization of associative, sequential, and schematic memories and prefrontal control over encoding and retrieval.

### Mapping to the source's sequence

| Source concept | Where it appears in Part 7 |
|---|---|
| Proposition can enter longer-term memory or support decisions | Sections 1 and 4 |
| Medial temporal lobe circuits contribute to episodic or semantic memory | Sections 3, 5, and 6 |
| Hippocampal and contextual binding processes | Sections 5 and 6 |
| Prefrontal coordination of question answering | Sections 7 and 8 |
| DLPFC involvement in holding content and comparing alternatives | Sections 8 and 9 |
| Contextual inference can influence the remembered mental model | Sections 5, 6, and 11 |
| Decisions can guide planning or action | Section 9 |

---

## 18. Handoff and completion status

**Input received:** A context-enriched proposition maintained in working memory.

**Operations completed in this algorithm:**

- validation and provenance preservation;
- retention-priority estimation;
- route selection between transient use, episodic encoding, semantic integration, and immediate decision use;
- event-context binding and differentiation of similar episodes;
- candidate semantic knowledge integration and consolidation;
- cue-based retrieval and source checking;
- question answering and judgment;
- option generation, outcome prediction, evidence comparison, response selection, and post-decision updating.

**Final conceptual output:** A proposition that has either been used transiently, contributed to an episodic trace, contributed to semantic knowledge, supported an immediate answer or action, or done more than one of these. No branch guarantees permanent storage or perfect recall.

**End of Part 7 — Long-Term Memory Storage and Decision Use.**

The seven explicitly sequenced processing stages in the supplied comprehension pipeline are now covered. The remaining material in the source—especially its *Integrative Model of Language Comprehension*, evidence synthesis, methodological debates, and clinical implications—does not add another simple stage after decision use. It can instead be turned into a separate **Part 8: Cross-Stage Integration Algorithm**, showing feedback loops, parallel processing, conflict-triggered reanalysis, working-memory coordination, and handoffs across the seven stages.


# Part 8 — Cross-Stage Integration: A Recurrent Algorithm for Whole-Sentence and Discourse Comprehension

## 1. Purpose and scope

The first seven parts described major operations that contribute to comprehension:

1. perceptual analysis and recognition of linguistic word forms;
2. lexical access and activation of word meanings;
3. syntactic parsing;
4. semantic composition;
5. contextual and pragmatic integration;
6. propositional working-memory representation and maintenance; and
7. long-term memory storage and decision use.

This part does **not** introduce another obligatory stage that happens after those seven. It converts the integrative model in the supplied source into a system-level algorithm. Its job is to explain how the processes can cooperate, exchange partial results, constrain one another, recover from errors, and produce a coherent interpretation while language is still arriving.

The most important design principle is that the seven operations should not be treated as seven sealed boxes that finish one at a time. The source itself says that the stages interact and that context and expectations affect interpretation incrementally. A word-form hypothesis can activate lexical candidates; lexical candidates can constrain a parse; the parse can determine which word sense is relevant; the evolving proposition can change the discourse model; and discourse expectations can in turn influence the processing of later words. The algorithm below therefore uses **recurrent, partially parallel processing** with explicit information exchange.

The brain does not literally execute the equations or pseudocode written here. The equations are computational descriptions of possible operations. A useful algorithmic model must be distinguished from a complete biological account: the exact neural implementation, timing, and contribution of individual regions remain active research questions.

## 2. Overall computational objective

At any moment, the system receives an incomplete stream of linguistic evidence and must estimate what the speaker or writer is communicating. It must do so while:

- the signal may be noisy, incomplete, or ambiguous;
- several word, syntactic, or semantic hypotheses may be plausible;
- context can support or contradict the literal interpretation;
- the speaker may rely on shared knowledge or unstated intentions;
- earlier interpretations may need revision when new material arrives;
- limited working-memory and attentional resources must be allocated; and
- an interpretation may be needed for immediate action before a complete discourse has arrived.

A compact objective is:

\[
\hat{H}_{1:t} = \arg\max_{H_{1:t}}
P(H_{1:t}\mid X_{1:t}, C_{t}, K, S)
\]

where:

- \(X_{1:t}\) is the sensory linguistic evidence received up to time \(t\);
- \(H_{1:t}\) is a candidate interpretation history, including word identities, structure, and proposition(s);
- \(C_t\) is the current discourse and physical context;
- \(K\) is relevant background knowledge; and
- \(S\) is the current speaker/listener model, including available information about identity, goals, beliefs, and communicative conventions.

This expression is a convenient formalization of inference under uncertainty. It does not imply that the brain calculates a single global posterior explicitly. Real neural processing is distributed, and different computations may use different information and timescales.

A second objective is to minimize the cost of comprehension while maintaining sufficient accuracy:

\[
J = L_{\text{interpretation}} + \lambda_1 C_{\text{memory}}
  + \lambda_2 C_{\text{control}} + \lambda_3 C_{\text{delay}}
\]

Here, \(L_{\text{interpretation}}\) represents the cost of an incorrect or incomplete interpretation; the other terms represent memory, executive-control, and delay costs. The weights \(\lambda_i\) are theoretical parameters, not known biological constants. This formulation captures why a person may accept a sufficiently clear interpretation in ordinary conversation but invest more effort when the stakes are high or ambiguity matters.

## 3. Integrated state representation

The algorithm needs a shared working state. It is useful to imagine several linked representations rather than one monolithic sentence record.

### 3.1 Sensory evidence state

Stores recent acoustic or visual evidence, timing, signal quality, and uncertainty. For speech it can include phonetic-feature hypotheses and temporal alignment; for reading it can include letter, word-boundary, and orthographic hypotheses. The representation is not yet a full meaning.

### 3.2 Lexical candidate state

Stores possible word forms and their activation or plausibility values. A position in the stream can temporarily have more than one candidate. Each candidate can carry lexical frequency, morphological information, syntactic category probabilities, and possible senses.

### 3.3 Syntactic hypothesis state

Stores one or more partially built grammatical structures. It tracks phrase boundaries, heads, dependencies, agreement constraints, unresolved arguments, and the confidence associated with each relationship. It should allow competing parses to coexist temporarily rather than forcing an early irreversible decision.

### 3.4 Semantic frame and proposition state

Stores the semantic content supported by the current lexical and syntactic hypotheses. A frame may contain a predicate, participant slots, semantic roles, modifiers, quantifiers, tense, aspect, negation, and modality. Unfilled roles remain explicitly unresolved rather than being silently guessed.

### 3.5 Context and situation-model state

Stores the discourse entities, events, relations, locations, time, causal links, and previously established propositions relevant to the current utterance. It also carries source information: what was directly stated, what was inferred, what was perceived in the environment, and what is merely assumed from general knowledge.

### 3.6 Speaker and communicative-goal state

Stores candidate beliefs about what the speaker knows, intends, presupposes, requests, or expects the listener to infer. These hypotheses should be activated when the task or evidence calls for them, not automatically for every ordinary sentence.

### 3.7 Working-memory and control state

Stores the currently relevant propositions, unresolved dependencies, retrieval cues, attention priorities, and the current task goal. It also tracks load, interference, and which unresolved issue is worth additional processing.

### 3.8 Output and memory-routing state

Stores candidate interpretations ready for use, plus a decision about whether the information is only temporarily useful, should be encoded as an event, should update general semantic knowledge, or should drive an immediate answer or action.

Each representation should preserve at least four labels:

1. **content** — what the candidate says;
2. **source** — whether it came from sensory evidence, a word meaning, context, memory, or inference;
3. **confidence** — how strongly the current evidence supports it;
4. **status** — confirmed, provisional, rejected, unresolved, or inferred.

These labels prevent a plausible inference from being mistaken for something the speaker explicitly said.

## 4. Main algorithm: recurrent integration of the seven stages

### Step 1 — Receive a new increment of sensory evidence

**Goal:** Update the input representation as new speech sounds or written material becomes available.

1. Obtain a time-bounded segment of acoustic or visual input.
2. Estimate signal quality and identify uncertain regions.
3. Extract relevant features and align them with the current stream position.
4. Preserve more than one sensory interpretation when the input does not uniquely determine a sound, letter, or boundary.
5. Attach a temporal index so the new evidence can be related to preceding and following material.
6. Pass the updated evidence to the word-form recognition process without waiting for the complete sentence.

**Possible mathematical operations**

A simplified feature representation may be written as:

\[
\mathbf{x}_t = f_{\theta}(\mathbf{s}_{t-k:t})
\]

where \(\mathbf{s}_{t-k:t}\) is a short window of sensory input and \(f_\theta\) is a feature-extraction function. Confidence can be represented as a probability distribution over candidate features rather than a single hard label.

For a candidate sensory feature \(z_i\):

\[
P(z_i\mid \mathbf{s}_{t-k:t})
\propto P(\mathbf{s}_{t-k:t}\mid z_i)P(z_i)
\]

The equation describes evidence-weighted inference. Neural sensory processing is not known to be a literal Bayesian calculator.

**Possible biological implementation:** distributed activity in auditory or visual pathways, local excitatory and inhibitory interactions, temporal integration, and recurrent communication with higher-level language-sensitive regions. The exact pathway differs between spoken and written input.

### Step 2 — Activate lexical candidates before the stream is complete

**Goal:** Generate plausible word-form identities from incomplete sensory evidence.

1. Use the sensory evidence to activate matching phonological or orthographic entries.
2. Include candidates that differ in one or more uncertain features.
3. Weight candidates using signal fit, learned lexical knowledge, morphological patterns, and the current linguistic context.
4. Maintain several candidates when evidence is insufficient.
5. Update candidate activation when another phoneme, syllable, letter, or word-boundary cue arrives.
6. Send candidate identities and their uncertainty to syntactic and semantic processes.

**Possible mathematical operations**

A simplified candidate score:

\[
A_i = w_s S_i + w_l L_i + w_c C_i + w_m M_i - w_i I_i
\]

where \(S_i\) is sensory match, \(L_i\) is lexical familiarity, \(C_i\) is contextual support, \(M_i\) is morphological compatibility, and \(I_i\) is an incompatibility or inhibition term. The weights are model parameters.

A normalized competition model could use:

\[
P(w_i\mid x,C) =
\frac{\exp(A_i/\tau)}{\sum_j \exp(A_j/\tau)}
\]

where \(\tau\) controls how sharply the strongest candidate dominates. This is a computational approximation, not a claim that a specific group of neurons performs a softmax operation.

### Step 3 — Update grammatical hypotheses in parallel

**Goal:** Use available word and morpheme information to build one or more structural interpretations.

1. Receive the current lexical candidate distribution.
2. Estimate possible grammatical categories for each candidate.
3. Test whether each candidate can attach to the active phrase or dependency structure.
4. Update phrase boundaries, heads, dependents, agreement, and argument expectations.
5. Keep unresolved attachments open if the incoming sequence has not disambiguated them.
6. Assign lower priority to structurally implausible hypotheses without necessarily deleting them immediately.
7. Make predicted grammatical requirements available to lexical and perceptual processes.

**Possible mathematical operations**

Let \(G_j\) represent a candidate grammatical structure. Its score can be updated as:

\[
Score(G_j) = Score_{t-1}(G_j) +
\log P(w_t, m_t\mid G_j, G_{<t})
\]

where \(w_t\) is a lexical candidate and \(m_t\) represents relevant morphological cues. A beam-search implementation may retain only the best \(B\) hypotheses:

\[
\mathcal{G}_t = TopB(\mathcal{G}_{t-1}\cup\text{new extensions})
\]

A bounded beam is a modelling convenience for limited resources. The brain is not established to maintain a literal list of exactly \(B\) parses.

**Possible biological implementation:** interaction between posterior temporal and frontal language-sensitive regions, with local recurrent activity representing candidate structure and constraints. The precise division of syntactic operations across regions is not one-to-one and varies with the task and analysis.

### Step 4 — Retrieve candidate meanings while structure is still developing

**Goal:** Activate semantic information associated with the current lexical candidates.

1. For each viable lexical candidate, retrieve possible senses and semantic features.
2. Retrieve argument structure, selectional tendencies, event knowledge, and common associations.
3. Use grammatical category and local structure to change the relative activation of senses.
4. Keep several senses active when the word is ambiguous.
5. Allow context to influence candidate activation without allowing contextual expectation to erase contrary sensory evidence.
6. Send semantic features to the composition process.

**Possible mathematical operations**

For candidate sense \(s\):

\[
P(s\mid w,C,G) \propto
P(w\mid s)P(s\mid C,G)
\]

Here \(G\) denotes the current grammatical hypothesis. A semantic feature vector can be updated by weighted activation:

\[
\mathbf{v}_{sense} = \sum_i p(s_i\mid w,C,G)\mathbf{v}_{s_i}
\]

This represents a graded candidate mixture. It should not be mistaken for the only possible neural format for conceptual knowledge.

### Step 5 — Compose partial semantic structures

**Goal:** Combine meanings into an incremental proposition as soon as enough information is available.

1. Read the current syntactic hypotheses and the semantic features associated with their words.
2. Identify predicates, arguments, modifiers, and relational operators.
3. Bind known participants to roles supported by the current structure.
4. Preserve unfilled argument slots until the sentence supplies their fillers or context justifies an explicitly labelled inference.
5. Incorporate negation, tense, aspect, quantification, modality, and other operators when their scope becomes sufficiently clear.
6. Produce a provisional proposition for each sufficiently plausible parse.
7. Keep semantic uncertainty attached to the proposition when the parse or word sense remains uncertain.

For example, after “The child opened ...”, the system may represent an event with an identified agent and an unresolved theme:

\[
OPEN(agent=CHILD, theme=UNKNOWN)
\]

When “the door” arrives, the representation can be completed:

\[
OPEN(agent=CHILD, theme=DOOR)
\]

**Possible mathematical operations**

Composition can be described as role binding and constraint satisfaction. A simple compatibility score for a candidate role filler \(e\) and role \(r\) is:

\[
Compat(e,r) = \mathbf{v}_e^T W_r\mathbf{v}_{predicate}
\]

where \(W_r\) is a learned role-specific transformation. A separate structural constraint determines whether the candidate is grammatically licensed. This equation is a possible model, not a direct measurement of a biological synapse.

### Step 6 — Update the discourse and situation model

**Goal:** Connect the evolving proposition to entities, events, and relations already represented in discourse.

1. Retrieve currently active discourse entities and events.
2. Resolve references such as pronouns and definite descriptions against candidate antecedents.
3. Compare the new proposition with prior propositions and the current situation model.
4. Update event order, temporal relations, locations, causal links, and entity properties when supported.
5. Preserve contradictions rather than silently overwriting one side.
6. Label direct statements separately from bridging or world-knowledge inferences.
7. Return the updated context to lexical, syntactic, semantic, and pragmatic processes.

**Possible mathematical operations**

A discourse entity candidate \(e_i\) can be ranked by a weighted score:

\[
R(e_i) = \alpha Recency_i + \beta Salience_i +
\gamma Agreement_i + \delta SemanticFit_i + \eta ContextFit_i
\]

The parameters depend on the language, task, and context. A relation graph may be updated by adding a node or edge only when its confidence exceeds a chosen criterion, or by storing it with a graded confidence if the system supports uncertain relations.

A simple graph update is:

\[
D_t = D_{t-1}\cup\{(u,r,v,source,confidence)\}
\]

where \(D_t\) is the discourse graph and \((u,r,v)\) records a relation between entities or events.

### Step 7 — Generate multi-level expectations

**Goal:** Use the developing interpretation to anticipate what kinds of linguistic material may come next.

1. Read the current discourse model, semantic proposition, grammatical structure, and task context.
2. Generate expectations at more than one level: likely meanings, grammatical constructions, lexical items, and possibly speech sounds or orthographic forms.
3. Keep these expectations graded rather than treating the most likely continuation as guaranteed.
4. Use expectations to allocate attention and facilitate processing, but preserve the ability to process unexpected input.
5. Revise expectations whenever new evidence arrives.

**Possible mathematical operations**

A next-event distribution can be represented as:

\[
P(X_{t+1}\mid X_{1:t}, C_t, G_t, M_t)
\]

where \(G_t\) is the current grammatical state and \(M_t\) is the evolving meaning representation. The information-theoretic surprisal of an observed item is:

\[
Surprisal(x_t) = -\log P(x_t\mid X_{1:t-1},C_{t-1})
\]

Higher surprisal means the observed item was less expected under the model. Surprisal is a useful computational measure; it is not identical to an N400 amplitude or any single neural response.

### Step 8 — Compare actual evidence with the current interpretation

**Goal:** Detect whether new input fits current word, structure, meaning, and context hypotheses.

1. Compare observed sensory evidence with predicted word-form features.
2. Compare the actual lexical candidate with the predicted lexical and grammatical requirements.
3. Compare the updated parse with syntactic constraints and expectations.
4. Compare the new proposition with world knowledge and discourse coherence.
5. Compute separate mismatch indicators for sensory, lexical, syntactic, semantic, referential, and pragmatic levels.
6. Identify whether the mismatch is small and resolvable locally or whether it threatens the current interpretation.
7. Pass mismatch information to the control and reanalysis process.

A generic prediction-error vector may be written as:

\[
\boldsymbol{\epsilon}_t =
\begin{bmatrix}
\epsilon_{sensory}\\
\epsilon_{lexical}\\
\epsilon_{syntactic}\\
\epsilon_{semantic}\\
\epsilon_{discourse}\\
\epsilon_{pragmatic}
\end{bmatrix}
\]

In a simple numerical model, each component could be the difference between observed and predicted feature values. For categorical hypotheses, it may instead be represented by a probability or log-likelihood ratio. Do not collapse all discrepancies into one undifferentiated scalar: a noisy acoustic segment and a socially implausible intention require different responses.

### Step 9 — Allocate attention and control resources

**Goal:** Decide which unresolved issue deserves additional processing.

1. Estimate the uncertainty and importance of each unresolved issue.
2. Estimate working-memory load and the number of active alternatives.
3. Identify whether more sensory evidence is likely to arrive naturally.
4. Determine whether an ambiguity matters to the interpretation or can safely remain unresolved.
5. Prioritize high-impact ambiguities, conflicts, and dependencies.
6. Recruit more domain-general executive support when task difficulty, memory load, or response demands warrant it.
7. Reduce processing of low-value alternatives when resources are limited.

**Possible mathematical operations**

A control priority can be approximated by:

\[
Priority_i = Uncertainty_i\times Impact_i\times Relevance_i
\]

A resource allocator can distribute a limited control budget \(B\):

\[
\sum_i r_i\leq B
\]

where \(r_i\) is the amount of modelled processing resource assigned to issue \(i\). This is a computational abstraction; neural attention and executive control involve distributed, task-dependent mechanisms rather than a known single central budget counter.

### Step 10 — Trigger reanalysis only when needed

**Goal:** Repair a provisional interpretation when new evidence invalidates it or leaves unacceptable uncertainty.

1. Identify the earliest point at which the active interpretation became incompatible with the input.
2. Determine whether the problem is a word identity, word sense, attachment, grammatical dependency, semantic role, referent, or contextual inference.
3. Reopen only the affected hypotheses where possible.
4. Generate alternatives using the new evidence and relevant context.
5. Recompute the downstream propositions associated with those alternatives.
6. Compare the alternatives on sensory fit, grammatical fit, semantic plausibility, discourse coherence, and processing cost.
7. Select the best-supported interpretation provisionally, or retain multiple interpretations if the evidence remains insufficient.
8. Update the situation model without deleting provenance for the abandoned interpretation.

**Possible mathematical operations**

A hypothesis score can combine evidence and costs:

\[
Score(H_i)=\log P(X\mid H_i)+\log P(H_i\mid C)
-\lambda Cost_{revision}(H_i)
\]

The revision cost is not a fixed biological quantity; it represents the fact that changing a prior interpretation may require rebuilding dependent representations. Reanalysis can use a bounded search over alternatives or a local update to only the affected nodes of the parse and situation graph.

### Step 11 — Perform pragmatic or social inference when the literal proposition is insufficient

**Goal:** Generate and assess candidate unstated meanings when ordinary linguistic composition does not fully explain the utterance in context.

1. Check whether the literal proposition is coherent with the discourse and situation.
2. Check whether the utterance appears relevant to the speaker's apparent communicative goal.
3. Identify missing causal, referential, social, or intention-related links.
4. Generate a small set of candidate inferences rather than inventing an unrestricted story.
5. Estimate whether each inference is supported by shared knowledge, preceding discourse, observed context, and likely speaker goals.
6. Compare the inferred interpretation with the literal proposition.
7. Keep literal content and inferred content in separate fields, even when the inference becomes the most plausible intended meaning.
8. Stop generating additional inferences when a sufficiently coherent explanation is available or the evidence remains too weak.

**Possible mathematical operations**

For inference hypothesis \(I_j\):

\[
P(I_j\mid U,C,S,K)\propto
P(U\mid I_j,S,C)P(I_j\mid C,K,S)
\]

where \(U\) is the utterance, \(S\) the speaker model, and \(K\) background knowledge. The posterior describes relative support; it does not prove that the speaker intended the inference.

A coherence score can combine independent dimensions:

\[
Coherence(I_j)=aL_j+bD_j+cW_j+dG_j-eContradiction_j
\]

where \(L_j\) is linguistic fit, \(D_j\) discourse fit, \(W_j\) world-knowledge fit, \(G_j\) goal or intention fit, and \(Contradiction_j\) the degree of unresolved conflict. The coefficients must be estimated or treated qualitatively.

### Step 12 — Maintain the evolving proposition and unresolved dependencies

**Goal:** Keep the current interpretation available while later input is processed.

1. Update the proposition record with newly supported predicates, arguments, and modifiers.
2. Keep unresolved roles, references, and attachments marked as open.
3. Preserve alternative propositions while meaningful ambiguity remains.
4. Update activation or accessibility of currently relevant material.
5. Prioritize entities and relations likely to be needed for the next clause or task.
6. Reduce the priority of irrelevant details without assuming that they have been erased from long-term memory.
7. Use the current task to determine which information must remain highly accessible.

**Possible mathematical operations**

One abstract activation model is:

\[
A_i(t+1)=\rho A_i(t)+Input_i(t)-\kappa Interference_i(t)
\]

where \(\rho\) represents persistence, \(Input_i(t)\) represents reactivation, and \(\kappa\) scales interference. This is a simplified computational model. It is not a settled equation for the firing of neurons or the literal decay of a proposition in the brain.

A more biologically grounded model would need to specify the neural population, recurrent connectivity, synaptic and cellular mechanisms, oscillatory coordination, and task-dependent modulation.

### Step 13 — Test whether the current interpretation is coherent enough to continue

**Goal:** Determine whether comprehension can proceed without additional repair.

1. Evaluate whether the current parse is sufficiently supported.
2. Evaluate whether the core predicate and arguments are sufficiently specified.
3. Check whether references have plausible antecedents or are explicitly unresolved.
4. Check whether the proposition can be integrated into the current situation model.
5. Determine whether any mismatch is central to the meaning or merely an optional detail.
6. If the interpretation is adequate, continue processing the next increment.
7. If the interpretation is inadequate, return to the relevant earlier step: sensory decoding, lexical candidate selection, parsing, semantic composition, contextual resolution, or pragmatic inference.
8. If the evidence cannot resolve the issue, preserve uncertainty rather than forcing a false certainty.

A threshold model can be used:

\[
Continue\;if\;Q_t\geq\theta_{task}
\]

where \(Q_t\) is a composite quality estimate and \(\theta_{task}\) depends on the task, stakes, and consequences of an error. This threshold is not necessarily implemented as one neural threshold.

### Step 14 — Finalize a sentence-level proposition when sufficient evidence is available

**Goal:** Create a stable enough representation of what the sentence explicitly states and what contextually inferred content has been added.

1. Select the best-supported word, parse, and semantic hypotheses, subject to any unresolved ambiguity.
2. Construct a predicate–argument representation or equivalent structured representation.
3. Attach tense, aspect, negation, modality, quantification, and referential information.
4. Add supported causal or pragmatic relations while marking their inferential status.
5. Preserve uncertainty and alternative readings where necessary.
6. Integrate the resulting proposition into the working situation model.
7. Mark which details are explicit, inferred, background assumptions, or unknown.

Example structure:

```text
PROPOSITION:
    predicate: APOLOGIZE
    agent: LISA
    recipient: PETER
    explicitness:
        agent: explicit
        predicate: explicit
        recipient: inferred_from_context
    confidence:
        predicate: high
        agent: high
        recipient: provisional
    source:
        agent: linguistic_input
        predicate: linguistic_input
        recipient: discourse_inference
```

This representation is a computational record. It is not a claim that the brain stores propositions in a programming-language structure.

### Step 15 — Route the result to the appropriate downstream process

**Goal:** Decide what to do with the sentence-level interpretation.

1. If the interpretation is needed only to understand the next sentence, keep it accessible in the current discourse model.
2. If the task requires answering a question, retrieve the relevant proposition and compare it with the question's requested information.
3. If an immediate decision is required, combine the proposition with task goals, available options, expected outcomes, and action constraints.
4. If the information is likely to be useful later, prioritize episodic encoding with source and context information.
5. If the content expresses a generalizable fact, consider whether it should update semantic knowledge; do not generalize from one weak or context-specific example.
6. Maintain separation between what the sentence said and what the listener concluded.
7. Continue monitoring new context because later discourse can revise earlier interpretation.

The branches can occur together. A listener may answer a question while also retaining the episode in memory, but neither outcome is mandatory for every sentence.

### Step 16 — Reopen earlier processing when later discourse changes the interpretation

**Goal:** Allow later words or sentences to revise the current model.

1. Compare new discourse information against active propositions and inferred relations.
2. Detect contradiction, correction, changed reference, irony, or a newly available causal explanation.
3. Identify which earlier representation is affected.
4. Reactivate the corresponding lexical, syntactic, semantic, referential, or pragmatic alternatives.
5. Revise the relevant proposition and downstream conclusions.
6. Preserve the history of the revision when it matters for understanding the discourse or remembering what was originally assumed.

A simple revision rule is:

\[
M_t = Update(M_{t-1},E_t,\Delta_t)
\]

where \(M_{t-1}\) is the current mental model, \(E_t\) is the new evidence, and \(\Delta_t\) records the affected part of the model. A local update is often computationally efficient, but some contradictions require rebuilding a larger portion of the interpretation.

### Step 17 — Consolidate only the information that merits longer-term retention

**Goal:** Link the integrated proposition to memory when retention is useful.

1. Estimate relevance, novelty, emotional or goal significance, and anticipated future usefulness.
2. Preserve the episode context and source when the information concerns a particular event.
3. Distinguish event-specific details from general knowledge.
4. Avoid turning an uncertain inference into an unquestioned memory.
5. Allow later retrieval, reactivation, and consolidation to modify accessibility and representation.

Possible models of memory strength, interference, and consolidation were addressed in Part 7. These operations are included here only as the final handoff from online comprehension, not as a claim that long-term storage immediately follows every sentence.

### Step 18 — Monitor comprehension quality and adapt the strategy

**Goal:** Use accumulated performance and feedback to adjust future comprehension.

1. Check whether the interpretation supports the current task.
2. Compare expected and observed outcomes when the interpretation is used.
3. Identify recurring error types: lexical confusion, misattachment, role reversal, reference failure, unsupported inference, or memory error.
4. Adjust attention or context use for the remainder of the conversation.
5. Learn stable linguistic regularities over time through experience-dependent changes.
6. Do not treat one correction as proof that a broad rule should be permanently changed.

This step is a functional learning loop. The underlying biological learning involves multiple mechanisms and timescales, including synaptic plasticity and changes in network-level accessibility.

## 5. Global feedback loops

The algorithm above contains several feedback loops. These are crucial because the stages do not operate as a strict one-way chain.

### Loop A — Context-to-perception and lexical access

The current linguistic and discourse context can help the system anticipate likely sounds, letters, word boundaries, and lexical items. When the sensory input contradicts the expectation, the interpretation must still be revised to fit the evidence. Context should bias processing, not hallucinate an input that was not present.

### Loop B — Syntax-to-lexicon and lexicon-to-syntax

A grammatical position can make some lexical categories more plausible. Conversely, a selected word can provide information about its likely arguments, grammatical behavior, or attachment. The two processes can therefore constrain each other.

### Loop C — Syntax-to-semantics and semantics-to-syntax

The grammatical structure determines which participant fills which semantic role. Semantic and world-knowledge constraints can reveal that an initially attractive parse is unlikely or that an alternative structure should be reconsidered. Semantic plausibility must not be used to override strong grammatical or sensory evidence without justification.

### Loop D — Proposition-to-context and context-to-proposition

A newly constructed proposition updates the situation model. The situation model then changes which meanings, references, causal links, and speaker intentions are plausible for later input.

### Loop E — Prediction-to-error-to-reanalysis

Expectations facilitate processing. Mismatches identify where the current interpretation may be wrong. Reanalysis produces revised predictions. This loop can repeat until a satisfactory interpretation is found or the system retains unresolved uncertainty.

### Loop F — Working-memory-to-control and control-to-working-memory

The quantity and complexity of material being maintained influence the need for control. Control processes can prioritize or refresh information in working memory. When load is high, the system may simplify the active representation, rely more on context, or request clarification.

### Loop G — Pragmatic inference-to-literal interpretation

An inferred intention can change the listener's understanding of why a sentence was said. But the inferred intention remains distinct from literal semantic content. If later evidence contradicts the inference, the system must be able to revise it.

### Loop H — Memory-to-current comprehension

Long-term semantic, episodic, and social knowledge can guide current interpretation. The current sentence can also modify what is later remembered or how a related memory is retrieved. This interaction creates a risk of memory distortion, so source and confidence information should be retained.

## 6. Mathematical integration models

The following equations summarize useful computational approximations. They are alternatives or complementary descriptions, not a claim that the brain performs all of them as explicit calculations.

### 6.1 Bayesian hypothesis revision

\[
P(H_i\mid X,C) =
\frac{P(X\mid H_i,C)P(H_i\mid C)}
{\sum_j P(X\mid H_j,C)P(H_j\mid C)}
\]

New evidence changes the relative plausibility of candidate interpretations. This is useful for describing ambiguity resolution, but the brain need not represent the full hypothesis space or compute this exact normalization.

### 6.2 Incremental log-evidence accumulation

\[
L_t(H_i)=L_{t-1}(H_i)+\log P(x_t\mid H_i,C_t)
\]

Evidence for or against a hypothesis can accumulate over successive increments. In practice, observations may not be conditionally independent, and the representation of evidence is likely distributed.

### 6.3 Prediction error and surprisal

\[
\epsilon_t = x_t-\hat{x}_t
\]

\[
Surprisal_t=-\log P(x_t\mid Context_{t-1})
\]

The first equation is appropriate for numeric feature representations; the second measures the unexpectedness of an observed event under a probability model. Neither should be equated with one ERP component. The meaning of N400 and P600 effects remains dependent on the task and theoretical account.

### 6.4 Context-sensitive candidate activation

\[
A_i(t)=\alpha A_i(t-1)+\beta SensoryFit_i+
\gamma ContextFit_i+\delta StructuralFit_i-\mu Conflict_i
\]

This equation summarizes how a candidate might be strengthened or weakened by new evidence, context, structure, and conflict. The coefficients are model parameters rather than established biological constants.

### 6.5 Semantic composition as constrained binding

\[
P(roles\mid words,syntax,context)
\propto P(words\mid roles)P(roles\mid syntax,context)
\]

This captures the fact that word meanings, grammatical structure, and context jointly constrain role assignment. The context term should be used carefully so stereotypical world knowledge does not replace what the sentence actually says.

### 6.6 Coherence and inference selection

\[
Q(H)=w_1 LinguisticFit+w_2 DiscourseFit+w_3 WorldKnowledgeFit
+w_4 GoalFit-w_5 UnresolvedConflict
\]

The resulting score can rank interpretations or determine whether additional inference is warranted. In real comprehension, relevance, culture, social relationships, task demands, and uncertainty may not reduce to one universal scalar.

### 6.7 Resource allocation

\[
\max_{r_1,\ldots,r_n}\sum_i Value_i(r_i)
\quad \text{subject to}\quad \sum_i r_i\leq B
\]

This is a general resource-allocation model. It can represent why high-impact ambiguities receive more attention than details that do not affect the current goal. It is not a literal map of a central executive that distributes a fixed numeric resource pool.

### 6.8 Working-memory accessibility

\[
A_i(t+1)=\rho A_i(t)+u_i(t)-\kappa I_i(t)
\]

This abstraction combines persistence, reactivation, and interference. Alternative working-memory theories may use activity-silent states, synaptic traces, oscillatory mechanisms, or combinations of active and latent representations. No single simple decay equation captures the complete biological system.

### 6.9 Decision and stopping rule

\[
Stop\;reanalysis\;when\;
ExpectedBenefit(another\;update)\leq Cost(another\;update)
\]

The expected benefit includes the likelihood that more processing will resolve an important uncertainty. The cost includes time, effort, and interference with other tasks. This is a rational-computation abstraction; actual stopping behavior is affected by attention, fatigue, urgency, and learned habits.

## 7. Neural systems: possible division of labour and interaction

No single region performs one entire step in isolation. The following mapping describes broad contributions and should not be interpreted as a rigid one-region/one-operation assignment.

### 7.1 Auditory and visual processing pathways

Auditory cortical pathways analyze speech-related acoustic information; visual pathways analyze written forms. They provide increasingly structured sensory evidence to language-related systems. The exact pathways differ by modality, and early perceptual operations can interact with higher-level expectations.

### 7.2 Core language network

The left-lateralized frontotemporal language network, including portions of inferior frontal and temporal cortex, is strongly implicated in language-selective operations such as accessing linguistic knowledge and combining linguistic representations. The source identifies left IFG, middle/superior temporal regions, and angular gyrus as key components. Their contributions are distributed and interact; a single anatomical site should not be assigned the entire lexical, syntactic, or semantic algorithm.

### 7.3 Semantic and conceptual knowledge systems

Temporal and temporoparietal regions contribute to the representation and use of conceptual and semantic information. Their role depends on the representational and control demands of the task. Left temporoparietal regions may be especially relevant when linguistic input must be integrated with broader knowledge, but anatomical boundaries and functional distinctions remain debated.

### 7.4 Semantic control and executive systems

Left inferior frontal and posterior middle temporal regions are often discussed in relation to semantic control: selecting context-relevant knowledge among competing semantic information. Bilateral frontoparietal multiple-demand systems, including dorsolateral prefrontal and intraparietal regions, can contribute when tasks require substantial executive control or working-memory support. These systems are not identical to the language-selective network.

### 7.5 Mentalizing and social-cognition systems

Medial prefrontal cortex and temporoparietal junction regions can contribute when understanding a sentence requires reasoning about beliefs, intentions, knowledge, or perspectives. They should not be assumed to activate maximally for every sentence. Their recruitment depends on whether the discourse creates a genuine social-inference demand.

### 7.6 Working-memory systems

Frontal and parietal networks help maintain task-relevant information and coordinate attention, especially when sentences are long, ambiguous, or difficult. Linguistic content and control demands are not necessarily stored in one anatomical buffer. Different forms of information may be maintained through interacting cortical populations and memory mechanisms.

### 7.7 Medial temporal and memory systems

Hippocampal and related medial temporal pathways can support relational and episodic memory processes. They help connect current information with remembered events and contextual details, but a sentence does not automatically become a permanent episodic memory. This is the downstream handoff discussed in Part 7.

### 7.8 Communication across systems

The system-level algorithm requires information exchange between perceptual, language, semantic, social, executive, and memory systems. Such communication may involve anatomical pathways, recurrent cortical interactions, and modulation of activity by attention and task demands. The model should not assume that every pair of regions exchanges every type of information on every sentence.

## 8. Possible biological operations underlying the integrated computation

The mathematical operations described above need plausible neural mechanisms. The following are candidate biological principles, not an exhaustive or definitive inventory.

1. **Synaptic integration:** incoming excitatory and inhibitory inputs are combined within neural populations, affecting whether and how strongly cells respond.
2. **Recurrent excitation:** activity can be sustained or reactivated through recurrent connections, supporting temporary representations and iterative processing.
3. **Lateral inhibition and competition:** interacting populations can suppress incompatible candidates, contributing to selection among word, parse, or semantic alternatives.
4. **Gain modulation:** attention and task demands can change the sensitivity of neural populations to incoming information.
5. **Top-down modulation:** higher-level context and expectations can bias lower-level processing without eliminating the influence of bottom-up evidence.
6. **Neural population coding:** information may be represented in patterns of activity across many neurons rather than in one neuron per concept or grammatical role.
7. **Temporal coordination:** oscillatory and phase-related mechanisms may help coordinate information across neural populations, but their exact role in each language operation remains an active research topic.
8. **Synaptic plasticity:** repeated co-activation and experience-dependent plasticity can change future accessibility of linguistic and conceptual information.
9. **Neuromodulation:** neurotransmitter systems can influence attention, learning, arousal, and the stability or flexibility of neural representations.
10. **Pathway-specific feedback:** recurrent cortico-cortical and cortico-thalamo-cortical pathways may contribute to maintaining, refining, and comparing linguistic and semantic representations.

The existence of these biological principles does not establish a unique mapping from a mathematical operation to a specific molecule, cell type, or brain region. A model that claims such a mapping would need direct experimental support.

## 9. Worked example A — Resolving a temporary syntactic ambiguity

Consider the sentence:

> The man saw the boy with the telescope.

The phrase “with the telescope” can be interpreted as describing the instrument used by the man to see, or as describing the boy who had the telescope. The sentence alone may not uniquely determine the intended attachment.

### Incremental processing trace

1. **“The”** — sensory processing and lexical access identify a likely determiner. The parser opens a noun phrase.
2. **“man”** — a noun candidate is selected; the noun phrase can be represented as a possible subject or discourse entity.
3. **“saw”** — a verb candidate is retrieved. The parser expects a subject–predicate relation and anticipates an object.
4. **“the boy”** — a new noun phrase is constructed and assigned as the likely object of “saw”. A provisional proposition is created: `SEE(agent=MAN, patient=BOY, instrument=UNRESOLVED)`.
5. **“with”** — the system activates an attachment ambiguity. One candidate attaches the prepositional phrase to the seeing event as an instrument; another attaches it to the noun phrase “boy”.
6. **“the telescope”** — the noun phrase fills the complement of “with”. Both candidate structures remain possible.
7. **Context check** — the system evaluates linguistic structure, the preceding discourse, and world knowledge. If the previous context says that the man was looking through a telescope, the instrumental reading gains support. If the previous context describes the boy's possessions, the noun-modifier reading gains support.
8. **Reanalysis or unresolved status** — if a later sentence clarifies the intended attachment, the model revises the proposition. Without clarifying information, it preserves the ambiguity rather than pretending that one reading is certain.

### Mathematical view

Let \(H_1\) be instrumental attachment and \(H_2\) be noun-modifier attachment. Then:

\[
P(H_1\mid X,C) \propto P(X\mid H_1,C)P(H_1\mid C)
\]

\[
P(H_2\mid X,C) \propto P(X\mid H_2,C)P(H_2\mid C)
\]

The context can change the prior support for each reading. The grammatical and lexical evidence then updates the relative support. If the two interpretations remain similarly plausible, the system should keep both active or mark the sentence as ambiguous.

### Neural interpretation

The example may involve interactions among temporal and frontal language-sensitive regions, lexical-semantic systems, and working-memory/control systems if the ambiguity requires extra effort. It is not appropriate to assign each candidate attachment to a unique anatomical region.

## 10. Worked example B — Inferring an unstated relation across sentences

Consider:

> Lisa apologized. Peter shrugged.

The two clauses explicitly state that Lisa apologized and Peter shrugged. The sentence does not explicitly say whom Lisa apologized to. Depending on discourse context, the listener may infer that Peter was the recipient or that Peter's reaction is relevant to the apology.

### Incremental processing trace

1. The first clause activates the predicate `APOLOGIZE`, its agent `LISA`, and an unresolved or context-dependent recipient role.
2. The proposition is added to the discourse model with its explicit content distinguished from any inferred recipient.
3. The second clause activates `SHRUG`, with `PETER` as its agent.
4. The discourse model notices that the two events occur in immediate succession and may be related.
5. Event knowledge supplies a plausible social relation: a person may shrug in response to an apology.
6. The system generates a candidate bridging inference: `RECIPIENT(APOLOGIZE, PETER)` or a looser relation linking Peter's reaction to the apology.
7. The system checks whether the wider context supports this relation. It must not treat adjacency alone as proof.
8. The enriched situation model stores the inferred link with an inferential label and confidence value.

### Mathematical view

Let \(I\) be the candidate relation that Peter is the recipient. A simplified inference score is:

\[
P(I\mid U,C,K)\propto P(U\mid I,C)P(I\mid C,K)
\]

A stronger context may raise the posterior probability; a context where Lisa apologized to someone else should lower it. The model should preserve the difference between the literal propositions and the inferred relation.

### Neural interpretation

The language network supports the processing and combination of the clauses. Temporal and conceptual systems can provide event knowledge; medial prefrontal and temporoparietal regions may contribute if understanding requires social or mental-state inference. Working-memory systems help maintain the first clause while the second is processed.

## 11. Worked example C — Literal sentence, speaker intention, and pragmatic reinterpretation

Consider a situation in which someone says:

> It is cold in here.

The literal proposition describes the temperature. In a context where a window is open and the listener is near it, the speaker may be indirectly requesting that the window be closed.

### Incremental processing trace

1. The sensory and lexical stages identify the words.
2. Syntactic parsing builds a declarative sentence.
3. Semantic composition creates the literal proposition `COLD(ROOM)`.
4. The situation model records the room, temperature, and visible window if those are known.
5. The system checks why this statement is relevant at this moment.
6. If the speaker is addressing someone able to close the window, a request interpretation becomes a plausible candidate.
7. The speaker model evaluates whether the speaker likely intends an action rather than merely reporting a temperature.
8. The listener keeps the literal content and inferred request distinct.
9. The inferred intention can guide a response, but the listener should remain open to the possibility that the speaker was only making an observation.

### Mathematical view

Let \(I_1\) be “the speaker is merely reporting the temperature” and \(I_2\) be “the speaker is indirectly requesting that the window be closed”. Their relative support depends on utterance content, context, speaker goals, and shared conventions:

\[
P(I_j\mid U,C,S,K)\propto P(U\mid I_j,S,C)P(I_j\mid C,S,K)
\]

The decision to close the window also depends on action costs, permissions, and possible alternative explanations. Understanding the request and deciding to act on it are related but distinct processes.

## 12. Integrated pseudocode

The pseudocode below expresses the functional workflow. It is intentionally more explicit and serial than neural processing itself so that the computational dependencies can be implemented and tested.

```python
from dataclasses import dataclass, field
from typing import Any, Literal

Status = Literal["provisional", "supported", "rejected", "unresolved", "inferred"]

@dataclass
class Candidate:
    content: Any
    confidence: float
    source: str
    status: Status = "provisional"
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ComprehensionState:
    sensory: list[Candidate] = field(default_factory=list)
    lexical: list[Candidate] = field(default_factory=list)
    parses: list[Candidate] = field(default_factory=list)
    propositions: list[Candidate] = field(default_factory=list)
    discourse_model: list[Candidate] = field(default_factory=list)
    speaker_model: list[Candidate] = field(default_factory=list)
    predictions: dict[str, Any] = field(default_factory=dict)
    unresolved: list[Candidate] = field(default_factory=list)
    working_memory: list[Candidate] = field(default_factory=list)
    revision_log: list[dict[str, Any]] = field(default_factory=list)


def integrate_increment(increment, state, context, task):
    """Functional model of recurrent language comprehension.

    Each helper represents a computational operation, not a literal
    one-region/one-operation mapping in the human brain.
    """
    # 1. Decode current sensory evidence without discarding uncertainty.
    sensory_hypotheses = analyze_perceptual_input(increment, state.sensory)
    state.sensory = update_candidates(state.sensory, sensory_hypotheses)

    # 2. Activate possible word forms and meanings.
    lexical_hypotheses = activate_lexical_candidates(
        state.sensory, context, state.predictions
    )
    state.lexical = update_candidates(state.lexical, lexical_hypotheses)

    # 3. Update grammatical hypotheses using lexical and morphological cues.
    parse_hypotheses = extend_parse_hypotheses(
        state.parses, state.lexical, context
    )
    state.parses = retain_viable_hypotheses(parse_hypotheses)

    # 4. Retrieve and compose semantic representations incrementally.
    sense_hypotheses = retrieve_context_sensitive_senses(
        state.lexical, state.parses, context
    )
    partial_propositions = compose_partial_propositions(
        sense_hypotheses, state.parses, state.propositions
    )
    state.propositions = update_propositions(
        state.propositions, partial_propositions
    )

    # 5. Link current content to the discourse and situation model.
    state.discourse_model = integrate_with_discourse(
        state.discourse_model, state.propositions, context
    )

    # 6. Produce graded expectations at linguistic and semantic levels.
    state.predictions = predict_upcoming_input(
        state.lexical,
        state.parses,
        state.propositions,
        state.discourse_model,
        context,
    )

    # 7. Detect mismatches separately at relevant processing levels.
    errors = compare_evidence_and_hypotheses(
        increment=increment,
        sensory=state.sensory,
        lexical=state.lexical,
        parses=state.parses,
        propositions=state.propositions,
        discourse=state.discourse_model,
        predictions=state.predictions,
    )

    # 8. Decide whether more control or reanalysis is warranted.
    priorities = prioritize_unresolved_issues(errors, state, task)
    if priorities.requires_reanalysis:
        old_snapshot = snapshot_interpretation(state)
        alternatives = generate_targeted_alternatives(
            priorities, state, context
        )
        state = revise_affected_hypotheses(state, alternatives, context)
        state.revision_log.append({
            "previous": old_snapshot,
            "reason": priorities.reason,
        })

    # 9. Generate pragmatic inferences only where needed.
    pragmatic_need = assess_need_for_pragmatic_inference(
        state.propositions, state.discourse_model, context, state.speaker_model
    )
    if pragmatic_need.is_relevant:
        inferences = generate_bounded_inferences(
            state.propositions,
            state.discourse_model,
            state.speaker_model,
            context,
        )
        state.propositions = attach_inferences_with_provenance(
            state.propositions, inferences
        )

    # 10. Maintain current propositions and unresolved dependencies.
    state.working_memory = update_working_memory(
        state.working_memory, state.propositions, state.unresolved, task
    )
    state.unresolved = collect_unresolved_items(state)

    # 11. Decide whether the present interpretation is adequate.
    quality = estimate_interpretation_quality(state, task)
    if quality.is_insufficient:
        return state, "continue_or_reanalyse"

    # 12. If a sentence or discourse unit is complete enough, finalize it.
    finalized = finalize_propositions_with_uncertainty(state, context)

    # 13. Use the result and route it to optional downstream processes.
    response = answer_or_act_if_requested(finalized, task)
    memory_action = select_memory_route(finalized, task, context)
    update_memory_if_appropriate(memory_action, finalized)

    return state, {
        "interpretation": finalized,
        "response": response,
        "memory_action": memory_action,
    }
```

The helper functions are intentionally abstract. An implementable software system would need concrete definitions for each representation, scoring function, confidence measure, revision policy, and memory interface. The code is not a literal simulation of brain tissue.

## 13. System-level failure modes and safeguards

1. **Strictly serial processing:** the system waits for one stage to finish before beginning another. Safeguard: allow partial outputs and recurrent updates.
2. **Premature commitment:** the first plausible parse or word sense is treated as final. Safeguard: retain alternatives while uncertainty is consequential.
3. **Context overpowering the signal:** expectation makes the system perceive a word that was not present. Safeguard: keep sensory evidence and contextual prior separately traceable.
4. **Syntax–semantics circularity without control:** semantic plausibility reinforces a parse that is structurally unsupported. Safeguard: score syntactic and semantic evidence separately, then combine transparently.
5. **Unbounded hypothesis growth:** every ambiguity creates an uncontrolled number of interpretations. Safeguard: prune low-value candidates while preserving alternatives with meaningful posterior support.
6. **Over-pruning:** a temporarily unlikely hypothesis is deleted even though later context would support it. Safeguard: keep a limited revision history or a small set of alternate structures.
7. **Inference presented as literal content:** a bridging inference or intention is remembered as if explicitly stated. Safeguard: attach source and inferential status to every relation.
8. **One global conflict score:** a sensory error is confused with a pragmatic mismatch. Safeguard: retain error categories and route each to an appropriate repair process.
9. **Unnecessary mentalizing:** the system infers elaborate beliefs and intentions when literal comprehension is sufficient. Safeguard: engage social-inference processing conditionally.
10. **Ignoring mentalizing when needed:** sarcasm, indirect requests, and false-belief narratives are interpreted literally. Safeguard: trigger pragmatic inference when literal content and communicative context create a relevant gap.
11. **Working-memory overload:** too many candidate parses and prior propositions remain equally active. Safeguard: use task relevance, recency, unresolved dependencies, and uncertainty to prioritize maintenance.
12. **Failure to revise earlier material:** later discourse contradicts an earlier assumption, but the model does not update. Safeguard: maintain links from downstream conclusions to the evidence on which they depend.
13. **False coherence:** the system creates an elegant causal story that is not supported by the discourse. Safeguard: require evidence, label speculation, and permit unresolved gaps.
14. **Confidence inflation:** repeated processing increases confidence without adding evidence. Safeguard: track evidence separately from the number of times a hypothesis was activated.
15. **Premature storage:** uncertain interpretations enter long-term memory as established facts. Safeguard: preserve provenance and uncertainty during memory encoding.
16. **Overgeneralized brain map:** each software function is assigned to one anatomical area. Safeguard: describe network-level contributions and note task dependence.
17. **Equation reification:** a model equation is described as a literal operation performed by neurons. Safeguard: label it a computational analogy unless directly tested biologically.

## 14. Scientific status and limitations

### Broadly supported principles

- Language comprehension involves interacting perceptual and language-sensitive systems.
- A left frontotemporal language network supports important language-selective computations, including accessing linguistic knowledge and combining linguistic representations.
- Higher-level systems for knowledge, reasoning, executive control, and social cognition can contribute when the task requires them, while remaining distinguishable from the core language network.
- Linguistic predictions can operate at several levels, including meaning, grammar, words, and speech sounds, although their extent and functional role are still debated.
- ERP effects such as N400 and P600 are useful measures of processing dynamics, but their interpretation is not reducible to one universally accepted operation.
- Recurrent and cortico-cortical interactions are plausible and supported by multiple lines of research, but the exact circuit-level computation for every comprehension stage is not fully settled.

### Computational hypotheses used in this part

- Bayesian candidate updating is a formal description of inference under uncertainty.
- Softmax competition can approximate graded selection among alternatives.
- Beam search can approximate bounded maintenance of competing parses.
- Prediction error and surprisal can quantify mismatch or unexpectedness.
- Graph updates can represent discourse entities and relations.
- Resource-allocation and stopping rules can describe attention and control demands.
- Recurrent activation equations can approximate maintenance and interference.

These mathematical descriptions are useful for building testable computational models. None should be presented as a directly observed neural equation without specific experimental evidence.

### Important open questions

- How do language-selective regions exchange information with semantic, executive, social, and memory systems at millisecond-to-second timescales?
- Which predictions are generated automatically, and which depend on task, attention, or strategic effort?
- How are uncertain interpretations maintained without overwhelming working memory?
- How do neural populations represent alternatives and revise them when later input arrives?
- Which parts of contextual inference are supported by language-selective regions and which by domain-general or social-cognition systems?
- How do individual differences, language experience, culture, age, and developmental changes modify the relative contribution of these networks?
- How can causal methods distinguish necessary operations from regions that merely co-activate during a demanding task?

## 15. References and source alignment for Part 8

### Primary source alignment

The supplied Library file, *Language Comprehension_ From Words to Contextual Meaning.txt*, motivates this system-level algorithm through its description of:

1. incremental integration, in which listeners update a mental model with each word;
2. contextual cues that immediately modulate interpretation;
3. world knowledge and expectations used during processing;
4. the interaction of a core language network with additional reasoning and social-cognition systems when needed;
5. ambiguity that can trigger semantic processing and theory-of-mind processes to test possible meanings;
6. a working-memory system that maintains partial representations while comprehension continues;
7. an inferencing loop that generates possible intentions and implications and integrates them with linguistic input and background knowledge; and
8. the final coherent mental model, which can guide immediate response or be encoded in memory.

### External research grounding

1. **Fedorenko, E., Ivanova, A. A., & Regev, T. I. (2024).** “The language network as a natural kind within the broader landscape of the human brain.” *Nature Reviews Neuroscience*, 25, 289–312. DOI: [10.1038/s41583-024-00802-4](https://doi.org/10.1038/s41583-024-00802-4). Describes the core language network and its distinction from perceptual/motor systems and higher-level systems of knowledge and reasoning.
2. **Tuckute, G., Kanwisher, N., & Fedorenko, E. (2024).** “Language in Brains, Minds, and Machines.” *Annual Review of Neuroscience*, 47, 277–301. DOI: [10.1146/annurev-neuro-120623-101142](https://doi.org/10.1146/annurev-neuro-120623-101142). Reviews language-related brain systems and implications for computational modelling.
3. **Heilbron, M., Armeni, K., Schoffelen, J.-M., Hagoort, P., & de Lange, F. P. (2022).** “A hierarchy of linguistic predictions during natural language comprehension.” *Proceedings of the National Academy of Sciences*, 119(32), e2201968119. DOI: [10.1073/pnas.2201968119](https://doi.org/10.1073/pnas.2201968119). Reports neural evidence for predictions at multiple levels of linguistic abstraction during naturalistic comprehension.
4. **Ryskin, R., & Nieuwland, M. S. (2023).** “Prediction during language comprehension: what is next?” *Trends in Cognitive Sciences*, 27(11), 1032–1052. DOI: [10.1016/j.tics.2023.08.003](https://doi.org/10.1016/j.tics.2023.08.003). Reviews contextual predictions during language comprehension and discusses unresolved questions about the mechanisms and conditions that support them.
5. **Delogu, F., et al. (2019).** “Event-related potentials index lexical retrieval (N400) and integration (P600) during language comprehension.” *Brain and Cognition*. DOI: [10.1016/j.bandc.2019.05.007](https://doi.org/10.1016/j.bandc.2019.05.007). Examines competing interpretations of N400 and P600 effects; useful evidence for avoiding overly simple one-component/one-operation mappings.
6. **Crosson, B. (2021).** “The Role of Cortico-Thalamo-Cortical Circuits in Language: Recurrent Circuits Revisited.” *Neuropsychology Review*, 31, 516–533. DOI: [10.1007/s11065-019-09421-8](https://doi.org/10.1007/s11065-019-09421-8). Reviews recurrent cortico-cortical and cortico-thalamo-cortical models relevant to language functions.
7. **Gwilliams, L., Bhaya-Grossman, I., Zhang, Y., Scott, T., Harper, S., & Levy, D. (2025).** “Computational Architecture of Speech Comprehension in the Human Brain.” *Annual Review of Linguistics*, 11, 209–226. DOI: [10.1146/annurev-linguistics-031120-111245](https://doi.org/10.1146/annurev-linguistics-031120-111245). Presents a computational framework focused on representations, operations, and information flow in spoken-language comprehension, emphasizing parallel and redundant processing.

### Mapping to the source's integrative model

| Source concept | Where it appears in Part 8 |
|---|---|
| Left auditory/visual regions decode linguistic form | Sections 4, 7.1, and 8 |
| Left IFG and temporal cortex retrieve words and build a parse | Sections 4, 7.2, and 7.4 |
| Semantic composition creates a sentence proposition | Steps 4–5 and 14 |
| Contextual check compares proposition with discourse and situation | Steps 6, 8, and 13 |
| mPFC/social-reasoning systems contribute to inference | Steps 11 and 7.5 |
| TPJ contributes to contextual/world-knowledge and social integration as task demands require | Steps 6 and 11; Section 7.5 |
| Inferencing loop may iterate until coherence is achieved | Steps 10–13; Section 5 |
| Frontal-parietal working-memory systems maintain partial representations | Step 12 and Section 7.6 |
| DLPFC provides top-down control when predictions fail or demands rise | Steps 8–10; Sections 7.4 and 9 |
| Coherent mental model guides response or memory | Steps 14–17 |

---

## 16. Handoff and completion status

**Input:** A continuous stream of speech or written language, together with available context, background knowledge, speaker information, and task goals.

**Integrated operations:**

- sensory evidence analysis;
- lexical candidate activation;
- syntactic hypothesis construction;
- context-sensitive semantic retrieval;
- incremental semantic composition;
- discourse and situation-model updating;
- multi-level prediction;
- mismatch detection;
- attention and control allocation;
- targeted reanalysis;
- conditional pragmatic inference;
- working-memory maintenance;
- coherence assessment;
- proposition finalization;
- response, action, and memory routing; and
- ongoing model revision as later discourse arrives.

**Final conceptual output:** A coherent but revisable representation of what was said, what it means in context, what is inferred rather than explicit, what remains uncertain, and what response or memory operation is appropriate.

**End of Part 8 — Cross-Stage Integration Algorithm.**

The seven processing stages and the source's integrative model have now been converted into algorithmic parts. The result is a functional computational framework inspired by human language comprehension, not a claim that the brain literally executes this pseudocode or that every listed mathematical operation has been directly observed in neural tissue.

# PART 9 — NEURAL ARCHITECTURE AND FUNCTIONAL DIVISION OF LABOUR IN LANGUAGE COMPREHENSION

## 1. Why this is a separate point

Parts 1–8 translated the major operations of language comprehension into algorithms: sensory-form analysis, lexical access, syntactic parsing, semantic composition, contextual/pragmatic integration, propositional working memory, downstream memory/decision use, and recurrent integration. The supplied source also describes a second, cross-cutting question: **which large-scale neural systems contribute to those operations, how their functions differ, and how they cooperate without becoming one undifferentiated language system**.

This part turns that neural-architecture account into an algorithm for coordinating specialized and domain-general neural systems. It does not introduce another linguistic stage after sentence comprehension. Instead, it explains the distributed biological architecture that can support all the earlier stages.

The working model distinguishes four broad classes of systems:

1. **Perceptual and motor systems** that encode the physical form of speech, writing, and language-related movement.
2. **The core language network**, typically left-dominant and distributed across lateral frontal and temporal cortex, which supports language-selective access to lexical/constructional knowledge and linguistic combination.
3. **Conceptual, semantic, and world-knowledge systems**, distributed across temporal, parietal, and other cortical areas, which supply knowledge about entities, events, properties, and relations.
4. **Additional domain-general and social-cognitive systems**, especially the multiple-demand/executive network and the mentalizing/theory-of-mind network, which contribute when comprehension requires control, difficult maintenance, perspective-taking, belief reasoning, or speaker-intention inference.

These systems are functionally distinguishable, but not isolated. Functional distinction means that they show different response profiles and make different contributions; it does not mean that they cannot communicate. During ordinary comprehension, a core language network can do much of the linguistic work without strong recruitment of every other system. During socially rich, ambiguous, or demanding discourse, additional networks can contribute information and control.

## 2. Central computational objective

Given a stream of linguistic input and the current task, estimate which neural-system classes are likely to be needed, maintain their specialized representations, exchange information when useful, and regulate additional recruitment according to the demands of the current interpretation.

The algorithm should avoid two opposite errors:

- **Over-unification:** treating all language, reasoning, memory, social cognition, and executive operations as if they were carried out by one generic language processor.
- **Over-separation:** treating the language network, semantic knowledge, working memory, and social reasoning as independent modules that never exchange information.

The target is a coordinated, recurrent, partially specialized architecture.

### 2.1 Inputs

- sensory evidence from speech or writing;
- currently active lexical, syntactic, and semantic hypotheses;
- the developing proposition and discourse/situation model;
- relevant background knowledge;
- speaker, listener, and task information;
- current uncertainty, ambiguity, and comprehension difficulty;
- current working-memory load and attentional state; and
- information about the individual's functional brain organization, where such information is available.

### 2.2 Outputs

- a set of active neural-system states and their current functional roles;
- estimates of which systems need to exchange information;
- controlled recruitment of conceptual, social-cognitive, or executive support when the task calls for it;
- a coherent linguistic and situation representation; and
- an updated estimate of system demands for the next word, clause, or discourse event.

## 3. Architecture: systems, representations, and roles

### 3.1 Perceptual and motor systems

**Input-side function:** auditory pathways analyze speech acoustics; visual pathways analyze the visual form of written language. These systems represent features such as frequency, timing, intensity, phonetic cues, edges, letter-like forms, and spatial configuration. They provide structured sensory evidence to language-related systems.

**Important boundary:** a perceptual representation is not automatically a complete linguistic meaning. A region can respond to speech-like or letter-like properties without itself implementing all the semantic and syntactic computations associated with comprehension.

**Possible mathematical description:** a perceptual state vector can be represented as \(p(t)\), with dimensions encoding currently relevant sensory features. A feature detector or population code can be approximated by weighted inputs followed by a nonlinear activation:

\[
 p_j(t)=\phi\!\left(\sum_i w_{ji}x_i(t)+b_j\right).
\]

Here \(x_i\) denotes sensory input, \(w_{ji}\) the effective influence of input feature \(i\) on response unit \(j\), and \(\phi\) a nonlinear response function. This is a computational approximation, not a claim that the brain explicitly evaluates this equation.

### 3.2 Core language network

The source describes a core language network that is usually left-lateralized and includes lateral frontal and temporal language-responsive regions. Across individuals, the precise location, size, and laterality of functional areas can vary. Commonly discussed components include:

- **Inferior frontal gyrus (IFG):** contributes to language-selective computations involving linguistic structure and combinations. The source associates it with semantic/syntactic integration and with controlled processing demands in difficult sentences. Its role should not be reduced to a single operation such as “grammar” or “working memory.”
- **Middle and superior temporal regions:** contribute to accessing and representing lexical and linguistic information and to combining incoming language with stored linguistic knowledge. Different temporal subregions may have different temporal receptive windows and sensitivities.
- **Anterior temporal regions:** often implicated in combining information into more integrated conceptual/semantic representations. Their exact contribution depends on the semantic task and theoretical model.
- **Posterior temporal regions:** implicated in mapping between sound/word forms and linguistic structure, and in aspects of syntactic and semantic processing.
- **Angular and adjacent inferior parietal regions:** may contribute to semantic integration, combinatorial processing, or working-memory interfaces depending on task and parcellation. Their exact membership in a narrowly defined language-selective network is debated.

A key principle from the source is that this network supports computations over linguistic knowledge: retrieving words and constructions from memory and combining them to interpret messages. It is not simply the sensory system for hearing or reading.

**Possible computational abstraction:** let \(h_L(t)\) represent the distributed state of the core language network. It can be updated from sensory-form evidence, lexical memory, grammatical hypotheses, and its own previous state:

\[
 h_L(t+1)=f_L\!\left(W_{LL}h_L(t)+W_{LP}p(t)+W_{LK}k_L(t)+b_L\right),
\]

where \(k_L(t)\) represents retrieved linguistic knowledge, \(W\) terms represent effective interactions, and \(f_L\) is a nonlinear update function. The equation summarizes recurrent interaction; it is not a literal neural circuit diagram.

### 3.3 Conceptual and world-knowledge systems

Linguistic expressions activate information that is not limited to word-form or grammar representations: knowledge about objects, people, actions, events, physical regularities, social scripts, and causal relationships. This information is distributed across the brain rather than stored in a single “world knowledge center.” Temporal semantic systems, including anterior temporal regions in some accounts, may contribute to integration across modalities; other cortical regions contribute features tied to sensory, motor, spatial, affective, and action knowledge.

The algorithm must therefore retrieve the relevant knowledge without assuming that every semantic feature is stored in the language network itself. For example, understanding “The glass fell and shattered” may require event and physical knowledge about fragile objects, gravity, and impact. Understanding “She apologized, but he remained angry” may additionally require interpersonal and emotional knowledge.

**Possible mathematical abstraction:** let \(h_K(t)\) be the active conceptual-knowledge state, and let \(q(t)\) be a cue extracted from the current proposition. Retrieval can be approximated as similarity-weighted activation:

\[
 a_i(t)=\operatorname{sim}(q(t),m_i), \qquad
 r_i(t)=\frac{\exp(\beta a_i(t))}{\sum_j\exp(\beta a_j(t))},
\]

where \(m_i\) is a stored conceptual pattern, \(a_i\) its cue match, and \(r_i\) a normalized activation weight. This is an algorithmic model for cue-dependent retrieval, not a claim that the brain calculates softmax probabilities.

### 3.4 Theory-of-mind / mentalizing network

The source highlights the **medial prefrontal cortex (mPFC)** and **temporoparietal junction (TPJ)** as important nodes when a comprehender needs to reason about another person's beliefs, intentions, desires, knowledge, or perspective. These regions are part of broader systems that support social cognition and are not language-selective in the same way as the core language network.

The mentalizing system may become relevant when a sentence requires the listener to infer what a speaker meant, what a character believes, why someone acted, or why a remark is ironic. It is not necessary to assume that mPFC and TPJ are recruited equally for every sentence or every inference. Task design, discourse, individual differences, and the type of mental-state inference all matter.

The source describes possible contributions from:

- **mPFC:** evaluating social context, perspectives, beliefs, goals, and inferred intentions;
- **TPJ:** representing or updating information about agents' beliefs and perspectives, and participating in social inference; and
- **precuneus/posterior cingulate and other default-mode regions:** potentially supporting the construction of coherent event and social situation models.

The left and right TPJ should not be treated as perfectly interchangeable or as having universally fixed roles. Some research links left TPJ more closely to language-related contextual integration and right TPJ to attentional reorienting, but TPJ is functionally heterogeneous and the distinction depends on task and subregion.

**Possible computational abstraction:** let \(h_M(t)\) represent the active mental-state model. Given an agent \(a\), the algorithm may maintain estimates of beliefs \(B_a\), goals \(G_a\), and knowledge \(K_a\):

\[
 h_M(t)=\{B_a(t),G_a(t),K_a(t),P_a(t)\}_{a\in\mathcal A},
\]

where \(P_a\) represents a candidate perspective and \(\mathcal A\) is the set of relevant agents. A candidate intention \(I\) can be scored against the utterance, situation, and estimated beliefs:

\[
 S(I)=w_u\,C_{utterance}(I)+w_s\,C_{situation}(I)+w_b\,C_{belief}(I)-w_x\,Conflict(I).
\]

The variables and weights are explanatory model components, not directly measured neuronal quantities.

### 3.5 Multiple-demand / executive-control network

The source describes a domain-general multiple-demand (MD) network, including bilateral dorsolateral prefrontal and intraparietal regions. These systems can contribute when comprehension involves high working-memory load, selection among alternatives, sustained task goals, conflict, or difficult decisions.

The key distinction is that **helping with a difficult language task does not automatically make a region part of the core language mechanism**. A demanding sentence may require the language network to construct its meaning while executive regions help maintain multiple candidate interpretations, protect task goals, allocate attention, or decide when to reanalyse.

Possible contributions include:

- maintaining a task goal or question while reading/listening;
- allocating attention to disambiguating cues;
- selecting among competing interpretations when demands are high;
- controlling interference from irrelevant information;
- monitoring whether comprehension is sufficiently complete; and
- coordinating response selection for a decision task.

**Possible mathematical abstraction:** define a demand signal

\[
 D(t)=\alpha U(t)+\beta L_{WM}(t)+\gamma C(t)+\delta T(t),
\]

where \(U\) is unresolved uncertainty, \(L_{WM}\) is working-memory load, \(C\) is conflict between active candidates, and \(T\) is task-control demand. Executive recruitment can be modelled as

\[
 g_{MD}(t)=\sigma\big(\kappa[D(t)-\theta_{MD}]\big),
\]

where \(\sigma\) is a logistic function and \(\theta_{MD}\) is a task-dependent threshold. This is a compact way to represent conditional recruitment; there is no evidence that the MD network literally computes this scalar formula.

### 3.6 Working-memory systems

Working memory is not assumed to be one anatomical storage box. The source describes contributions from left IFG and bilateral parietal areas, while broader models also distinguish maintenance, attention, rehearsal, and executive control. In the integrated architecture, working memory provides temporary availability for representations that have not yet been resolved or that must be compared across words, clauses, or task steps.

The algorithm should keep semantic content and control processes conceptually separate: maintaining a proposition is not identical to deciding which proposition to believe or which answer to give.

### 3.7 Medial temporal and long-term memory systems

When the task requires durable event memory, relational binding, or later retrieval, medial temporal lobe systems, including hippocampal circuitry, may participate. These systems should not be invoked as the default explanation for every moment of ordinary online comprehension. The memory-routing algorithm in Part 7 describes this downstream branch in more detail.

### 3.8 Network boundaries are probabilistic and individual-specific

Do not represent each named anatomical region as a fixed, one-operation module. Functional areas vary across individuals, and the borders between language-selective, semantic, mentalizing, and executive systems are not always sharp. In particular, group-averaged maps can blur individual functional topography. A scientifically stronger algorithm uses subject-specific localization where available and treats group maps as priors rather than exact coordinates.

## 4. Detailed network-coordination algorithm

### Step 1 — Initialize a multi-system state rather than a single monolithic processor

Create separate state records for perceptual form, linguistic knowledge, current parse, semantic/conceptual knowledge, discourse context, mental-state hypotheses, working-memory contents, executive/task demands, and memory-routing goals.

Each record should carry: content, activation/confidence, source, time of last update, dependencies, and uncertainty. Preserve these sources separately so that an inference does not become indistinguishable from a word that was explicitly heard or read.

**Possible mathematical operation:** concatenate or link system states into a global state vector \(H(t)=[h_P,h_L,h_K,h_M,h_{MD},h_{WM},h_{MTL}]\), while maintaining a typed graph so each element retains its system identity.

### Step 2 — Identify the individual's likely language-selective network

When individual neuroimaging data are available, identify candidate language-responsive regions using a contrast between meaningful linguistic stimuli and perceptually similar but linguistically degraded or incomprehensible stimuli. In a research or computational model without imaging, use a literature-based prior but explicitly mark the localization as approximate.

A functional localizer should be designed to distinguish linguistic processing from sensory complexity, task difficulty, and general engagement. Do not infer language selectivity from activation to a single language task alone.

**Possible mathematical operation:** for person \(s\) and region \(r\), calculate a contrast score

\[
 L_{s,r}=\overline{BOLD}_{s,r}(\text{language})-\overline{BOLD}_{s,r}(\text{matched control}).
\]

A standardized effect size, reliability estimate, and independent validation should accompany the score. A threshold is a methodological decision, not a biological boundary.

### Step 3 — Separate the sensory-form problem from the linguistic computation

Pass acoustic or orthographic evidence from perceptual systems to the language network. Keep the sensory representation available for feedback, but do not use sensory activation alone as evidence that meaning has been constructed.

**Possible operation:** compute a modality-specific handoff state \(p(t)\), then map it into a language-relevant representation \(u_L(t)=W_{LP}p(t)\). In a biological account this mapping is implemented by distributed neural pathways, not a single software-style interface.

### Step 4 — Let the core language network access words and constructions

Use the language network's stored linguistic knowledge to retrieve candidate word identities, lexical properties, grammatical constructions, and structural relations. Allow multiple candidates to remain active while sensory evidence is incomplete.

**Possible operation:** candidate activation can combine form evidence, familiarity, and current grammatical/discourse expectations. Avoid forcing the highest candidate to win before enough evidence arrives.

### Step 5 — Construct linguistic structure and literal compositional meaning

Within the core language system and its interactions with conceptual systems, incrementally build a syntactic structure and combine lexical meanings. Generate a literal proposition with explicit role bindings and operator scope. Keep this operation distinct from inferring hidden speaker intentions.

**Possible operation:** apply constrained graph binding. If a predicate requires arguments \(a_1,\ldots,a_n\), bind available constituents while preserving unresolved slots:

\[
 P=R(a_1,\ldots,a_k,\square,\ldots),\qquad k<n.
\]

The blank slots encode incomplete structure rather than a failed interpretation.

### Step 6 — Retrieve relevant conceptual knowledge in parallel

Use the evolving proposition to cue knowledge about entities, events, properties, physical regularities, social conventions, and causal relationships. Retrieve only what is relevant to the current content and task; avoid flooding the interpretation with remote associations.

**Possible operation:** cue-based activation and relevance weighting. One implementation is a similarity or associative score followed by normalization, with a threshold that limits how many knowledge items become active.

### Step 7 — Check whether the task requires reasoning about an agent's mind

Inspect the utterance and situation for cues that make mental states relevant: explicit belief predicates, disagreement, deception, irony, sarcasm, indirect requests, emotional reaction, false belief, or a question about what a speaker intended.

If none is present and the literal proposition is coherent, do not force an elaborate theory-of-mind inference. If such a cue appears, activate or update mental-state hypotheses.

**Possible mathematical operation:** a trigger score \(T_M\) can combine evidence for agent reference, perspective dependence, intention relevance, and literal-context mismatch:

\[
 T_M=w_aA+w_pP+w_iI+w_cC.
\]

The mentalizing route is considered when the score exceeds a task-sensitive threshold, or when the explicit task asks for a mental-state judgment.

### Step 8 — Estimate whether general executive support is needed

Estimate unresolved ambiguity, number of simultaneously active dependencies, working-memory load, conflict, time pressure, and task demands. Recruit more executive control when those demands rise, not simply because the input is linguistic.

**Possible mathematical operation:** compute a load estimate \(D(t)\) and a graded control signal \(g_{MD}(t)\). In a simulation this signal can increase the persistence of relevant items, bias attention toward discriminating evidence, or slow commitment when conflict remains high.

### Step 9 — Establish inter-system communication pathways

Create links between the systems that need to exchange information. Examples include:

- perceptual systems → language network: acoustic/orthographic form evidence;
- language network → conceptual systems: lexical and compositional cues;
- conceptual systems → language network: event plausibility and semantic constraints;
- language/discourse representation → mentalizing system: agent, utterance, and context information;
- mentalizing system → language/discourse representation: candidate intentions or belief-based constraints;
- executive system → other systems: task-goal and attention-control signals;
- working memory ↔ language network: maintenance of unresolved linguistic content; and
- proposition/context → medial temporal systems: information selected for longer-term encoding.

Do not assume each link is a direct monosynaptic connection. At the algorithmic level, an edge represents a functional information dependency. Neural communication can be implemented by multiple cortical, subcortical, and white-matter pathways.

**Possible mathematical operation:** represent systems as a directed weighted graph \(G=(V,E,W)\), where each node is a system state and each edge is an effective information-transfer weight. The graph is a modelling device, not a literal anatomical connectome.

### Step 10 — Coordinate parallel activity while preserving functional specialization

Update active systems in overlapping time windows. The language network can continue processing incoming structure while conceptual knowledge is retrieved and a social-cognitive hypothesis is developed. The network states should exchange selected information without collapsing their different representational roles.

**Possible mathematical operation:**

\[
 h_i(t+1)=f_i\!\left(W_{ii}h_i(t)+\sum_{j\ne i}g_{ji}(t)W_{ij}h_j(t)+B_i x(t)\right).
\]

Here \(g_{ji}(t)\) is a time-varying coupling or gating term. This equation is an abstract recurrent network model; it is not a directly observed synaptic equation for a named brain network.

### Step 11 — Preserve a functional distinction between language and mentalizing

When a person reads a story about beliefs, both language and theory-of-mind systems may be active because the story is linguistic and requires belief reasoning. Do not conclude from co-activation that both networks perform the same computation. Track what information each network contributes and compare its sensitivity to non-linguistic versions of the task.

**Possible operation:** maintain a feature-specific output for each network. Let \(z_L\) encode linguistic structure and \(z_M\) encode agent beliefs/goals. The final response can use both, but the intermediate records should remain distinguishable.

### Step 12 — Recruit multiple-demand resources conditionally

If a sentence contains multiple unresolved dependencies, competing structural readings, a complex question, or heavy maintenance demands, allow bilateral frontoparietal control to help allocate resources. If the sentence is straightforward and the task places little demand on maintenance or selection, the core language network may suffice for the linguistic computation.

Do not model the MD network as the primary repository of grammar or as a necessary component of all basic sentence composition.

### Step 13 — Synchronize information exchange without equating synchronization with identical function

Different networks may show correlated or synchronized activity at rest and during comprehension while retaining distinct response profiles. A model can represent coordination as changing coupling between systems, but synchronization alone does not prove that the systems share the same representation or causal role.

**Possible mathematical operation:** estimate pairwise time-series association, for example

\[
 FC_{ij}=\operatorname{corr}(x_i(t),x_j(t)),
\]

or a more robust measure such as partial correlation or coherence. Functional connectivity is statistical dependence; it does not by itself reveal direction of information flow or direct anatomical connectivity.

### Step 14 — Use prediction errors to update the correct subsystem

When new evidence contradicts the current interpretation, determine which representation is uncertain. A sensory mismatch may require perceptual re-evaluation; a word-form mismatch may require lexical reconsideration; a structural mismatch may require syntactic reanalysis; a semantic mismatch may require composition revision; a social-intention mismatch may require mental-state revision.

Do not respond to every error by activating all networks. Route the error to the system whose representation is most directly challenged, then share the updated evidence with dependent systems.

**Possible mathematical operation:**

\[
 e_k(t)=y_k(t)-\hat y_k(t),
\]

where \(y_k\) is new evidence relevant to subsystem \(k\) and \(\hat y_k\) its prediction. The magnitude and structure of the error can determine which hypothesis is revised.

### Step 15 — Integrate candidate outputs and resolve conflicts

Combine linguistic, conceptual, and social-cognitive evidence when the task requires it. Retain provenance: literal linguistic content, world-knowledge inference, and inferred speaker intention should remain distinguishable. If evidence is insufficient, preserve uncertainty rather than forcing a confident answer.

**Possible mathematical operation:** a candidate interpretation \(H\) may be evaluated using a weighted log-evidence model:

\[
 \log P(H\mid E)\propto \lambda_L\log P(E_L\mid H)
 +\lambda_K\log P(E_K\mid H)
 +\lambda_M\log P(E_M\mid H)
 +\log P(H),
\]

where the weights \(\lambda\) control the contribution of linguistic, conceptual, and mental-state evidence. This is a candidate computational model, not a direct neural measurement.

### Step 16 — Check whether additional control or inference is still necessary

Evaluate whether the current interpretation is coherent, sufficiently supported, and adequate for the task. If so, reduce unnecessary recruitment and proceed. If not, identify the unresolved issue and return to the relevant network/system update.

**Possible mathematical operation:** use a stopping condition based on confidence, remaining conflict, and task-specific cost:

\[
 Stop=1 \quad \text{if}\quad C(H)\geq\theta_C,\quad U(H)\leq\theta_U,
\]

with thresholds adjusted for the consequences of error. The brain is not assumed to use this explicit deterministic rule.

### Step 17 — Route the integrated representation to working memory, memory, or action

Maintain the proposition in working memory while it remains relevant. Send event/context information to long-term memory systems when retention is useful. Use prefrontal and other task-relevant systems to select a response or action when required. These are different destinations and need not all be engaged for every sentence.

### Step 18 — Update the system-demand profile as context evolves

A new speaker, change of topic, ambiguous pronoun, contradiction, or explicit question about a character's belief can alter which systems are useful. Recalculate the demand profile after meaningful changes. A low-demand literal sentence should not force the same network pattern as a sarcastic remark or a multi-clause story requiring causal and social inference.

### Step 19 — Maintain individual-specific variation

Adapt localization, thresholds, and estimated network contributions to the individual. A population-average left-lateralized pattern is a useful prior but not a guarantee of a particular person's exact functional topography. Development, bilingual experience, age, neuroplasticity, acquired injury, and other individual factors may affect the architecture and its recruitment.

### Step 20 — Return an auditable interpretation state

Output: the literal proposition; relevant contextual knowledge; any inferred mental state or communicative intention; uncertainty and unresolved alternatives; the systems hypothetically contributing; and the downstream route selected. Make the provenance of each inference explicit.

## 5. Network-level mathematical models

The equations in this section are computational abstractions useful for building simulations and reasoning models. They should not be presented as operations directly measured inside the living brain.

### 5.1 Functional selectivity index

For each candidate region or network, estimate the difference between response to language and response to matched non-language conditions:

\[
 S_{L}(r)=\mu_r(\text{language})-\mu_r(\text{control}).
\]

A more balanced selectivity score can normalize by variability:

\[
 d'_L(r)=\frac{\mu_r(L)-\mu_r(C)}{\sqrt{\tfrac12(\sigma_r^2(L)+\sigma_r^2(C))}}.
\]

Use several control conditions to test whether the response is truly language-selective rather than driven by general difficulty, visual/auditory complexity, or task engagement. This selectivity score is an analysis technique, not a signal that a brain region itself computes.

### 5.2 Network graph and coupling matrix

Let \(V=\{P,L,K,M,MD,WM,MTL\}\) denote perceptual, language, knowledge, mentalizing, multiple-demand, working-memory, and medial-temporal systems. Define a time-varying adjacency/coupling matrix \(W(t)\):

\[
 W(t)=\begin{bmatrix}
 w_{PP}&w_{PL}&\cdots\\
 w_{LP}&w_{LL}&\cdots\\
 \vdots&\vdots&\ddots
 \end{bmatrix}.
\]

The off-diagonal values describe effective modelled interactions. A sparse matrix is often more appropriate than assuming every system directly interacts with every other system at every moment.

### 5.3 Conditional routing / gating

For a support system \(j\), define a recruitment gate:

\[
 g_j(t)=\sigma\big(\mathbf{w}_j^\top \mathbf{d}(t)-\theta_j\big),
\]

where \(\mathbf{d}(t)\) contains uncertainty, social-inference cues, memory load, ambiguity, and task goals. Separate gates can control mentalizing and executive support. This makes the architecture conditional rather than all-on-all-the-time.

### 5.4 Dynamic network interaction

A recurrent update can be written:

\[
 \mathbf h(t+1)=\Phi\left(W(t)\mathbf h(t)+B\mathbf u(t)\right),
\]

where \(\mathbf h(t)\) contains system states, \(\mathbf u(t)\) is new sensory/context input, and \(\Phi\) is a vector of system-specific nonlinearities. The equation captures feedback and distributed computation, but it is not a literal reconstruction of neuronal dynamics.

### 5.5 Information bottleneck and selective exchange

A network should exchange information relevant to the current task without indiscriminately copying every active feature. A model can select a message \(m_{i\rightarrow j}\) by maximizing task-relevant information while penalizing message size or irrelevant content:

\[
 \max I(m;Y)-\beta I(m;X),
\]

where \(Y\) is the target-relevant variable and \(X\) is the full source state. This is an information-theoretic design analogy; there is no single known biological bottleneck implementing this exact objective.

### 5.6 Synchrony and functional connectivity

For recorded time series \(x_i(t)\) and \(x_j(t)\), one can estimate correlation, coherence, phase synchrony, or other dependence measures. These methods help describe coordinated activity. They do not, by themselves, establish direction, necessity, or direct anatomical wiring. Use causal perturbation, lesion evidence, and converging methods to test stronger claims.

### 5.7 System-demand / resource allocation

A control system may be modelled as allocating additional resources to the current bottleneck:

\[
 r_k(t)=\frac{\exp(\eta d_k(t))}{\sum_j \exp(\eta d_j(t))},
\]

where \(d_k(t)\) estimates the current demand associated with subsystem \(k\). In a simulation this could control attention, memory persistence, or candidate sampling. The biological interpretation is that attention and control can bias processing; the exact softmax computation is an engineering approximation.

## 6. Worked examples

### Example A — A straightforward literal sentence

**Input:** “The child opened the door.”

1. Auditory/visual systems encode the sensory form.
2. The core language network retrieves the words and builds a grammatical structure.
3. Semantic composition produces an event proposition with the child as agent and door as patient/theme.
4. Conceptual knowledge supports the action and object interpretation.
5. There is no strong cue that speaker belief or intention must be inferred; mentalizing support is not deliberately forced by the algorithm.
6. If the task is simply to understand the sentence, the proposition is maintained briefly and then used or allowed to decay according to task demands.

**Network-level interpretation:** language processing and conceptual knowledge are coordinated. There is no reason to assume that high-level executive control or mentalizing must dominate this uncomplicated case.

### Example B — A sentence requiring belief reasoning

**Input:** “Maya believed that the meeting had been cancelled, but it was still scheduled.”

1. The language network decodes the clause structure and the belief predicate.
2. Semantic composition constructs a proposition about Maya's belief and a contrasting proposition about the actual schedule.
3. The mentalizing system becomes relevant because the sentence explicitly represents an agent's belief that differs from reality.
4. The discourse representation must keep Maya's belief distinct from the narrator's asserted description of the meeting.
5. The final interpretation preserves the distinction between the agent's mental state and the actual state of affairs.

**Network-level interpretation:** language-selective processing and mental-state reasoning cooperate but do not perform identical computations. A purely linguistic representation is not enough to answer a question such as “What did Maya believe?” without tracking the belief holder and content.

### Example C — A difficult, ambiguous sentence

**Input:** “While the teacher watched the student wrote the answer.”

1. Early input supports more than one possible structural continuation.
2. The language network incrementally constructs candidate syntactic structures.
3. When “wrote” arrives, the initial analysis may need revision: “the teacher watched” is a subordinate clause, and “the student wrote the answer” is the main clause.
4. The revised interpretation requires resolving the earlier ambiguity and maintaining multiple elements over the sentence.
5. Executive/MD resources may assist if the task is demanding, if several alternatives remain active, or if the user must explain the ambiguity.
6. The algorithm should attribute the grammatical repair to language computations, with executive support for difficult selection/maintenance rather than assuming executive regions themselves contain the grammar.

**Network-level interpretation:** difficult language may recruit additional control while remaining fundamentally a language-comprehension task.

### Example D — Sarcasm and speaker intention

**Input:** “Great job!” spoken after a person has made an obvious mistake, with a scolding tone.

1. Perceptual systems encode the words and prosodic pattern.
2. The language network retrieves the literal positive expression.
3. The situation model supplies the preceding mistake and the speaker's reaction.
4. A conflict arises between the positive literal wording and the negative situation/prosody.
5. The mentalizing system may contribute a hypothesis that the speaker intends criticism rather than praise.
6. Candidate interpretations are compared; the sarcastic interpretation is selected only if the contextual and communicative evidence supports it.
7. The final representation preserves the literal wording and the inferred intention as related but distinct content.

**Network-level interpretation:** a language-selective network builds the linguistic message, while social-cognitive systems can help infer the speaker's intention. Prosody and context can contribute without making sarcasm a simple one-region function.

## 7. Integrated pseudocode

```text
FUNCTION coordinate_language_brain_networks(input_stream, context, task, person_profile):
    states = initialize_separate_states(
        perceptual_form,
        core_language,
        conceptual_knowledge,
        mentalizing,
        multiple_demand_control,
        working_memory,
        long_term_memory
    )

    localization = estimate_individual_network_topography(person_profile)
    mark_localization_uncertainty(localization)

    FOR each new input increment IN input_stream:
        sensory_state = update_perceptual_systems(input_increment)
        language_state = update_core_language_network(
            sensory_state,
            states.core_language,
            linguistic_memory
        )

        parse_state = update_lexical_and_syntactic_hypotheses(language_state)
        literal_proposition = compose_literal_meaning(parse_state)

        knowledge_state = retrieve_relevant_conceptual_knowledge(
            literal_proposition,
            context
        )

        context_state = update_situation_and_discourse_model(
            context,
            literal_proposition,
            knowledge_state
        )

        mentalizing_demand = estimate_need_for_perspective_or_intention_inference(
            literal_proposition,
            context_state,
            task
        )
        IF mentalizing_demand is sufficiently high:
            mental_state = update_agent_belief_goal_intention_hypotheses(
                literal_proposition,
                context_state
            )
        ELSE:
            mental_state = preserve_current_state_without_forced_inference()

        executive_demand = estimate_ambiguity_conflict_and_memory_load(
            parse_state,
            literal_proposition,
            context_state,
            task
        )
        IF executive_demand is high:
            control_state = recruit_attention_and_multiple_demand_support(
                unresolved_items,
                task_goal
            )
        ELSE:
            control_state = use_minimal_additional_control()

        exchange_relevant_information_between_systems(
            language_state,
            knowledge_state,
            context_state,
            mental_state,
            control_state
        )

        errors = detect_representation_specific_prediction_errors()
        FOR each error IN errors:
            revise_the_relevant_subsystem(error)
            notify_dependent_systems_of_revision(error)

        interpretation = integrate_outputs_without_losing_provenance(
            literal_proposition,
            knowledge_state,
            mental_state,
            context_state
        )

        IF interpretation_is_sufficient_for_current_task(interpretation, task):
            maintain_in_working_memory(interpretation)
            route_to_memory_or_response_as_needed(interpretation, task)
        ELSE:
            preserve_uncertainty_and_continue_processing()

        update_network_demand_estimates()

    RETURN auditable_interpretation_with_uncertainty_and_provenance
```

## 8. Evidence types and what they can establish

A network-coordination algorithm should not confuse different kinds of evidence. Each method answers a different question.

### 8.1 Functional MRI

fMRI can identify regions and networks whose blood-oxygen-level-dependent responses covary with language, mentalizing, or executive tasks. It is useful for mapping spatially distributed activity and functional connectivity. However, BOLD signals are indirect and temporally slower than neuronal firing; co-activation alone does not establish that a region is necessary or that it performs the exact computation assigned to it.

### 8.2 EEG and MEG

EEG/MEG provide more temporally precise measures of neural activity and can help investigate when linguistic, semantic, and later reanalysis-related effects emerge. Their spatial localization is generally less straightforward than fMRI and depends on modelling assumptions. Components such as N400 or P600 should not be treated as unique readouts of one isolated algorithmic operation.

### 8.3 Lesion evidence

Lesions can provide evidence about which structures are causally important for particular abilities, especially when a selective impairment or double dissociation is observed. However, lesions can affect connected regions and pathways, and chronic reorganization can alter the pattern. A lesion's effects should not be read as a perfect deletion of one computational module.

### 8.4 TMS, intracranial recording, and other causal/physiological methods

Perturbation methods can test whether a region's activity is necessary at a particular time or for a particular operation. Intracranial recordings can sample neural activity with high temporal precision in clinical contexts. Their coverage is limited, and findings require careful generalization.

### 8.5 Functional connectivity

Correlated activity between regions indicates statistical coordination. It does not directly show that information flows from region A to region B, that the link is anatomical, or that the two regions perform the same computation. Directional and causal claims require stronger analyses and converging methods.

## 9. Failure modes and safeguards

| Failure mode | Why it is wrong | Algorithmic safeguard |
|---|---|---|
| Treating the left hemisphere as the only possible language location | Typical left dominance is not universal; individual variation exists | Use subject-specific localization and allow atypical lateralization |
| Treating every IFG response as syntax | IFG activity can vary with linguistic and task demands | Describe the task and contrast; avoid one-region/one-function assumptions |
| Treating the whole temporal lobe as one semantic store | Temporal regions are functionally heterogeneous | Use distributed, subregion-sensitive representations |
| Treating mPFC/TPJ as mandatory for every sentence | Mentalizing demands are conditional | Trigger social reasoning only when the task or context calls for it |
| Treating MD activity as core grammar | Executive recruitment may reflect load, control, or conflict | Separate language computation from domain-general support |
| Equating synchronization with shared function | Distinct networks can coordinate while remaining functionally distinct | Preserve separate representations and task-selectivity tests |
| Equating fMRI correlation with causal information flow | Correlation does not prove direction or necessity | Use careful language and triangulate with perturbation and lesion evidence |
| Assuming all people have identical network boundaries | Functional topography varies between individuals | Use individualized functional localizers when feasible |
| Assuming each cognitive system is anatomically isolated | Distributed systems interact and overlap at boundaries | Represent graded connectivity and uncertainty |
| Turning mathematical abstractions into biological facts | Equations are models unless directly tested | Label every equation as an analogy or computational hypothesis |

## 10. Biological cautions and unresolved questions

1. **Network names are operational constructs.** “Language network,” “mentalizing network,” and “multiple-demand network” describe patterns of functional organization, not sealed anatomical organs.
2. **The language network is usually left-lateralized, not universally left-only.** Atypical organization and individual variation must remain possible.
3. **Regional functions are not one-to-one.** IFG, temporal cortex, TPJ, and parietal areas participate in multiple interacting operations; a label should not be mistaken for a complete causal explanation.
4. **The exact boundaries of semantic systems remain debated.** Some theories place greater emphasis on anterior temporal convergence; others stress distributed semantic representations and control systems.
5. **The language/ToM boundary is a functional distinction, not an absolute wall.** A story about beliefs uses linguistic processing and may simultaneously recruit mentalizing.
6. **The MD network is not synonymous with all working memory.** Working-memory maintenance, attention, conflict control, and task management overlap but are not identical.
7. **Functional connectivity is not anatomical connectivity.** Statistical coupling can change with task, state, and measurement choices.
8. **Mathematical models are explanatory approximations.** Bayesian inference, vector similarity, graph operations, gating, and information-theoretic objectives can help formalize hypotheses, but they are not directly observed as literal symbolic calculations in neurons.
9. **Development and learning matter.** Network specialization and inter-system interactions change with experience, age, and developmental history.
10. **The best architecture may be more distributed than any single diagram.** Different tasks and individuals can recruit partially overlapping systems.

## 11. References and source alignment for Part 9

### Primary source alignment

The supplied file, *Language Comprehension_ From Words to Contextual Meaning.txt*, motivates this part through its sections on:

- the core left-hemisphere language network, including IFG, middle/superior temporal cortex, and angular gyrus;
- the distinction between language-selective processing and extra-linguistic networks;
- mPFC and TPJ contributions to mentalizing, belief, intention, sarcasm, and implicit-meaning inference;
- left/right TPJ distinctions in contextual semantic integration and attentional reorienting, with the qualification that roles can vary by task;
- bilateral dorsolateral prefrontal and parietal multiple-demand systems during difficult comprehension and high working-memory demands;
- the claim that the language network works closely with, but is distinct from, systems of knowledge and reasoning;
- evidence from fMRI, ERP/MEG, lesion studies, and individual functional localization; and
- the importance of individual variability and the limits of group-averaged maps.

### Verified external research grounding

1. **Fedorenko, E., Ivanova, A. A., & Regev, T. I. (2024).** “The language network as a natural kind within the broader landscape of the human brain.” *Nature Reviews Neuroscience*, 25, 289–312. DOI: [10.1038/s41583-024-00802-4](https://doi.org/10.1038/s41583-024-00802-4). Reviews the core language network, its left-lateralized frontotemporal organization, language selectivity, and distinction from perceptual/motor and higher-level knowledge/reasoning systems.
2. **Tuckute, G., Kanwisher, N., & Fedorenko, E. (2024).** “Language in Brains, Minds, and Machines.” *Annual Review of Neuroscience*, 47, 277–301. DOI: [10.1146/annurev-neuro-120623-101142](https://doi.org/10.1146/annurev-neuro-120623-101142). Reviews relations between human neural language representations and computational language models, and how models can help test hypotheses about language processing.
3. **Paunov, A. M., Blank, I. A., & Fedorenko, E. (2019).** “Functionally distinct language and Theory of Mind networks are synchronized at rest and during language comprehension.” *Journal of Neurophysiology*, 121(4), 1244–1265. DOI: [10.1152/jn.00619.2018](https://doi.org/10.1152/jn.00619.2018). Provides evidence that language and ToM networks can be functionally distinct while showing coordinated activity.
4. **Diachek, E., Blank, I., Siegelman, M., Affourtit, J., & Fedorenko, E. (2020).** “The Domain-General Multiple Demand (MD) Network Does Not Support Core Aspects of Language Comprehension: A Large-Scale fMRI Investigation.” *Journal of Neuroscience*, 40(23), 4536–4550. DOI: [10.1523/JNEUROSCI.2036-19.2020](https://doi.org/10.1523/JNEUROSCI.2036-19.2020). Tests the distinction between core language processing and domain-general MD responses.
5. **Shain, C., & Fedorenko, E. (2026).** “A language network in the individualized functional connectomes of 1199 human brains doing arbitrary tasks.” *Nature Communications*, 17, Article 8180. DOI: [10.1038/s41467-026-75745-8](https://doi.org/10.1038/s41467-026-75745-8). Reports individualized functional-connectivity analyses across 1,199 brains and 1,957 fMRI sessions, finding a largely left-dominant frontotemporal network that was stable within individuals and selective for language. This finding reinforces the importance of individual-level network characterization; it does not imply that every person's network has identical boundaries.
6. **Blank, I. A., Kanwisher, N., & Fedorenko, E. (2014).** “A functional dissociation between language and multiple-demand systems revealed in patterns of BOLD signal fluctuations.” *Journal of Neurophysiology*, 112, 1105–1118. Relevant to the functional distinction between language-selective and domain-general systems.
7. **Fedorenko, E., & Shain, C. (2021).** “Similarity of computations across domains does not imply shared implementation: the case of language comprehension.” *Current Directions in Psychological Science*, 30, 526–534. Relevant to the caution that two tasks may use similar computational principles while depending on functionally distinct brain systems.

### Mapping to earlier parts

| Earlier algorithm | Main system-level contribution explained in Part 9 |
|---|---|
| Part 1 — Perceptual processing | Auditory/visual systems provide sensory-form evidence to language-related systems |
| Part 2 — Lexical access | Core language network accesses stored word and construction knowledge |
| Part 3 — Syntactic parsing | Language-selective frontotemporal circuitry builds and revises linguistic structure |
| Part 4 — Semantic composition | Language and conceptual systems contribute to a structured proposition |
| Part 5 — Contextual/pragmatic integration | Conceptual and social-cognitive systems provide discourse, world, and intention-related information |
| Part 6 — Propositional working memory | Language-related representations interact with temporary maintenance and control systems |
| Part 7 — Memory and decision use | Medial temporal memory systems and task-relevant decision/control systems support downstream use |
| Part 8 — Cross-stage integration | Recurrent exchange and feedback coordinate all systems over time |

---

## 12. Handoff and completion status

**Part 9 is a cross-cutting neural-architecture algorithm, not a new chronological sentence stage.** It explains how specialized perceptual, language, conceptual, mentalizing, executive, working-memory, and memory systems can cooperate across Parts 1–8.

**Final network-level output:** a coordinated interpretation state in which linguistic structure, conceptual knowledge, speaker/agent mental states, uncertainty, working-memory contents, and downstream task goals remain distinguishable but can exchange relevant information.

**End of Part 9 — Neural Architecture and Functional Division of Labour.**


# PART 10 — EVIDENCE-BASED IDENTIFICATION AND VALIDATION OF LANGUAGE-COMPREHENSION MECHANISMS

## 10.0 Purpose and scope

This part converts the source document's **Evidence from Neuroimaging and Patient Studies** section into a research-and-validation algorithm. Unlike Parts 1–8, this is not another chronological operation performed every time a person understands a sentence. It is an algorithm for determining which mechanisms and neural systems are supported by evidence, how strongly they are supported, and how to distinguish competing explanations.

The central principle is **converging evidence**. A brain region becoming active during a language task does not, by itself, prove that the region is necessary for that operation, that it performs the operation alone, or that a particular mathematical computation occurs there. Stronger conclusions emerge when behavioral experiments, fMRI, EEG/MEG, lesion studies, stimulation, and computational models constrain the same hypothesis from different directions.

The target output is a calibrated model containing:

1. a clearly defined cognitive operation;
2. the conditions under which that operation is needed;
3. candidate brain regions and connections;
4. the timing and sequence of the operation, where measurable;
5. evidence for association, necessity, or causal contribution;
6. competing explanations and uncertainty;
7. a computational description that can be tested against new data.

**Important distinction:** The equations in this part primarily describe how a scientist analyzes evidence. They are not assertions that the brain itself explicitly calculates regression coefficients, p-values, Bayes factors, or statistical maps while comprehending language. The brain may use biological processes that can be approximated by mathematical models, but research-statistical operations and neural operations must not be conflated.

## 10.1 Inputs, internal records, and outputs

### Inputs

- A processing hypothesis from Parts 1–9, such as lexical retrieval, syntactic structure building, semantic composition, context updating, pragmatic inference, or proposition maintenance.
- Carefully designed linguistic stimuli and non-linguistic control stimuli.
- Behavioral observations: accuracy, response time, confidence, eye movements, and error types.
- Neuroimaging measurements: fMRI, structural MRI, diffusion imaging, and, where relevant, PET.
- Electrophysiological measurements: EEG/ERP and MEG.
- Patient information: lesion location, lesion volume, white-matter disconnection, clinical language profile, and relevant non-language abilities.
- Perturbation information: TMS/tES target, timing, stimulation parameters, sham/control-site condition, and task performance.
- Participant-level variables: age, handedness, language history, education, hearing/vision, fatigue, and other factors that could alter performance or measurement.

### Internal records

For every candidate mechanism, maintain an evidence record:

```text
MechanismRecord:
    operation_name
    operational_definition
    processing_stage
    triggering_conditions
    predicted_behavior
    predicted_brain_regions
    predicted_time_window
    predicted_network_interactions
    competing_hypotheses
    confounds_and_nuisance_variables
    evidence_by_method
    individual_variability
    confidence_level
    unresolved_questions
```

### Outputs

- A ranked set of candidate mechanisms.
- A map of supporting and contradicting findings.
- A distinction between correlational, temporal, lesion-based, and perturbational evidence.
- A model of when a neural system is recruited and what it may contribute.
- Explicit uncertainty labels and experiments capable of reducing that uncertainty.

## 10.2 Core evidence principles

### Principle A — Define a cognitive operation before searching for its location

A vague question such as “Where does meaning happen?” is not experimentally precise. Replace it with a testable operation, for example:

- retrieving a lexical-semantic representation after a word form is recognized;
- assigning a thematic role to a noun phrase;
- combining a predicate with its arguments;
- detecting a mismatch between a new word and its discourse context;
- inferring an unstated causal relation;
- maintaining a proposition until a later question is answered.

Each operation must have observable predictions. A good operational definition specifies what is changed between conditions and what outcome should change if the operation is engaged.

### Principle B — Separate activation from functional contribution

An fMRI activation shows that measured activity differs between conditions under a particular analysis. It does not automatically show that the active region is indispensable. The region may be performing the target computation, supplying input, monitoring difficulty, maintaining information, shifting attention, or responding to an associated feature of the task.

Therefore, the algorithm must use distinct evidence labels:

| Evidence label | What it supports | What it does not establish alone |
|---|---|---|
| Association | A neural measure covaries with a task or behavior | Necessity or causal direction |
| Selectivity | A response is stronger for one task contrast than for controls | A single exclusive function for that region |
| Temporal precedence | A signal occurs earlier or later relative to an event | That the earlier signal causes the later one |
| Lesion association | Damage or disconnection is associated with impaired performance | A precise one-region/one-function map |
| Perturbational effect | Changing neural activity alters a task outcome | Exact local computation or complete circuit mechanism |
| Cross-method convergence | Different methods support compatible aspects of a hypothesis | That all details of the hypothesis are settled |

### Principle C — Use converging methods with different limitations

fMRI offers relatively good spatial coverage but a slow hemodynamic response. EEG and MEG capture fast temporal dynamics but source localization can be difficult. Lesion studies provide evidence about the consequences of damage, but lesion distributions and network disconnection complicate localization. TMS/tES can perturb some cortical targets, but stimulation is not perfectly focal and its effects depend on timing, intensity, anatomy, and task. A robust model should not depend on the strengths of only one measurement technique.

### Principle D — Model individual differences

Group averages can obscure differences in functional anatomy, language dominance, task strategy, and proficiency. Where feasible, collect individual functional localizers and estimate participant-level effects before reporting group-level summaries. A group-level result should not be presented as an exact map of every individual's language network.

## 10.3 Main algorithm: from a brain-language hypothesis to a validated mechanism

### Step 1 — Select one processing operation

**Input:** a mechanism proposed in the preceding parts.

1. Read the proposed algorithm and extract the smallest operation that can be tested independently.
2. Identify its inputs, outputs, and triggering conditions.
3. Identify which earlier or later operations could be confused with it.
4. Write at least one positive prediction and one disconfirming prediction.
5. Define what counts as success, failure, or an ambiguous result.

**Example:** If the hypothesis is that left temporal language regions contribute to lexical-semantic retrieval, the positive prediction may be that a manipulation of lexical-semantic access changes a neural or behavioral measure in a reproducible way. A disconfirming pattern may be that the effect disappears when low-level perceptual difficulty and response demands are controlled.

**Computational representation:**

Let the target operation be \(M\), task condition be \(C\), and measurable outcome be \(Y\). A minimal hypothesis states:

\[
P(Y\mid C_1,M) \neq P(Y\mid C_0,M)
\]

This expresses a predicted difference between experimental conditions. It is a scientist's formal statement, not a claim that neurons calculate this probability explicitly.

### Step 2 — Translate the operation into controlled experimental contrasts

1. Construct a target condition that requires the operation.
2. Construct a control condition that preserves as many other features as possible.
3. Match or statistically control sentence length, word frequency, plausibility, syntax, emotional content, imageability, predictability, and response requirements as appropriate.
4. Use more than one item per condition so that a result is not driven by a small number of unusual sentences.
5. Counterbalance item assignment and presentation order.
6. Include comprehension checks without making the task itself the main source of difficulty.

For instance, a pragmatic-inference experiment should not compare a short, familiar literal sentence to a long, emotionally charged sarcastic sentence and attribute every neural difference to pragmatics. Length, familiarity, emotional tone, inferential demand, and memory load would all be potential confounds.

**Possible statistical model:**

\[
Y_{ij}=\beta_0+\beta_1 C_{ij}+u_i+v_j+\epsilon_{ij}
\]

where \(Y_{ij}\) is an outcome for participant \(i\) and item \(j\), \(C_{ij}\) is the experimental condition, \(u_i\) captures participant variation, \(v_j\) captures item variation, and \(\epsilon_{ij}\) is residual variation. More complete mixed-effects models can include participant- and item-specific condition slopes.

The purpose is to avoid confusing a condition effect with a peculiarity of particular participants or particular sentences.

### Step 3 — Establish the behavioral signature

1. Measure whether participants understand the stimuli.
2. Record accuracy and response time where a response is appropriate.
3. Record the type of error, not only whether an answer is right or wrong.
4. Measure confidence when uncertainty is relevant.
5. Use eye tracking or reading-time measures when the time course of comprehension is important.
6. Check whether the target condition creates a general difficulty increase rather than a selective effect.
7. Confirm that the intended inference or interpretation was actually made.

**Accuracy:**

\[
Accuracy=\frac{N_{correct}}{N_{valid\ trials}}
\]

**Mean response time:**

\[
\overline{RT}=\frac{1}{N}\sum_{i=1}^{N}RT_i
\]

Response-time analysis should account for errors, outliers, speed–accuracy trade-offs, and the distribution of response times. A slower response does not uniquely identify a particular neural computation.

### Step 4 — Design the fMRI test for spatial/network predictions

1. State which regions are predicted to respond and which are expected to be relatively less engaged.
2. Use independent functional localizers where possible to identify language-selective regions in each participant.
3. Separate language-selective regions from regions associated with mentalizing, executive control, visual processing, auditory processing, and motor responses.
4. Define the regions of interest before inspecting the critical contrast, or use an independent dataset to avoid circular selection.
5. Acquire appropriate structural images and assess motion and data quality.
6. Estimate condition effects with a general linear model (GLM).
7. Examine both regional responses and network-level relationships, but do not treat correlated activity as proof of direct communication.

A simplified voxel-wise GLM can be expressed as:

\[
\mathbf y_v=\mathbf X\boldsymbol\beta_v+\boldsymbol\epsilon_v
\]

where \(\mathbf y_v\) is the measured time series at voxel \(v\), \(\mathbf X\) contains task and nuisance predictors, \(\boldsymbol\beta_v\) contains fitted coefficients, and \(\boldsymbol\epsilon_v\) is residual error.

The task predictor is commonly convolved with an assumed hemodynamic response function (HRF). The BOLD signal is an indirect vascular measure associated with neural activity; its temporal response is much slower than individual neural events.

A contrast such as:

\[
\Delta BOLD_v=\widehat\beta_{v,condition\ A}-\widehat\beta_{v,condition\ B}
\]

estimates the response difference between two modeled conditions. It does not directly identify the computation performed by the voxel.

### Step 5 — Use EEG/ERP or MEG to test temporal predictions

1. Define the event that should trigger the operation, such as the onset of a critical word.
2. Align EEG/MEG trials to that event.
3. Remove or model artifacts, including eye movements, muscle activity, line noise, and bad channels.
4. Apply pre-registered time windows and scalp/region-of-interest definitions where possible.
5. Compare the target and control conditions.
6. Inspect both average waveforms and the underlying distribution of effects.
7. Avoid interpreting a component as a unique label for a mental process.

A simplified event-related potential is:

\[
ERP_c(t)=\frac{1}{N_c}\sum_{k=1}^{N_c}EEG_{c,k}(t)
\]

where \(EEG_{c,k}(t)\) is the event-locked voltage for trial \(k\) in condition \(c\), and \(N_c\) is the number of accepted trials.

A condition difference is:

\[
\Delta ERP(t)=ERP_{target}(t)-ERP_{control}(t)
\]

This difference can be measured over a pre-specified time window and electrode/region set. The exact window and interpretation depend on the task, reference, preprocessing, and competing ERP theories.

#### N400-related evidence

The N400 is commonly observed as a negative-going ERP effect around 300–500 ms after a meaningful stimulus, often larger for words that are less expected or less supported by context. It is relevant to lexical-semantic access and context-sensitive processing. However, accounts differ on whether it primarily reflects retrieval, integration, or a combination of processes. Therefore:

- Do not translate “larger N400” into “the brain failed to understand the word.”
- Do not assume that N400 is a direct meter of semantic integration alone.
- Consider cloze probability, lexical frequency, semantic association, plausibility, and contextual constraint.
- Compare several kinds of semantic mismatch to determine which factor explains the effect.

#### P600-related evidence

A late positive effect often occurs from around 500–600 ms onward, with timing and distribution dependent on the task. It has been associated with syntactic reanalysis, repair, integration difficulty, conflict resolution, and some pragmatic or semantic phenomena. Its interpretation is debated. Therefore:

- Do not assume every P600 is syntactic.
- Do not assume it uniquely marks pragmatic inference.
- Use experimental manipulations that distinguish structural repair from semantic conflict and general task difficulty.
- Where possible, relate the electrophysiological effect to behavioral evidence and later interpretation.

The algorithm should record N400/P600 as measured temporal signatures, not as self-sufficient explanations.

### Step 6 — Test lesion and neurodegeneration evidence

1. Identify patients with relevant focal lesions or progressive changes.
2. Characterize language performance using multiple tasks rather than one total score.
3. Measure non-language abilities that could affect the same task, including attention, memory, executive control, hearing, and vision.
4. Map lesion location, volume, and potentially affected white-matter pathways.
5. Determine whether the deficit is selective, broad, or part of a wider cognitive profile.
6. Compare patient performance with suitable control groups.
7. Use lesion-symptom mapping only when the sample and lesion distribution permit meaningful inference.
8. Consider diaschisis, disconnection, compensation, reorganization, and time since injury.

A simplified lesion-symptom model could be:

\[
Performance_i=\beta_0+\beta_1 LesionFeature_i+\boldsymbol\gamma^T\mathbf Z_i+\epsilon_i
\]

where \(LesionFeature_i\) indicates damage to a voxel, region, tract, or network feature and \(\mathbf Z_i\) includes covariates such as lesion volume and demographic or cognitive variables.

For a binary outcome, a logistic model may be used:

\[
P(deficit_i=1)=\sigma(\beta_0+\beta_1 LesionFeature_i+\boldsymbol\gamma^T\mathbf Z_i)
\]

with \(\sigma(z)=1/(1+e^{-z})\).

**Interpretation safeguard:** A lesion-associated deficit can support the claim that intact tissue or connections in that system are important for normal performance. It does not automatically prove that the damaged voxel alone implements the operation. Lesions affect networks, may disconnect remote regions, and are constrained by vascular anatomy. Absence of damage in a region is not proof that the region is unnecessary.

### Step 7 — Use stimulation when a causal test is justified

Where appropriate, TMS or tES can test whether transiently perturbing a cortical target changes performance.

1. Select a target based on an explicit hypothesis and prior evidence.
2. Select stimulation timing based on a temporal prediction, or justify why offline stimulation is appropriate.
3. Include sham stimulation and/or a control site.
4. Randomize and counterbalance conditions where feasible.
5. Verify participant safety and follow accepted ethical and clinical protocols.
6. Measure accuracy, reaction time, and error type.
7. Check for non-specific effects such as discomfort, alertness, or motor interference.
8. Replicate and interpret null findings in light of stimulation strength, target accessibility, and statistical power.

A simple within-participant stimulation contrast is:

\[
\Delta RT=RT_{active}-RT_{sham}
\]

and:

\[
\Delta Accuracy=Accuracy_{active}-Accuracy_{sham}
\]

A selective effect on a target operation, relative to matched control operations, is more informative than a general slowing across every task. But TMS/tES does not provide perfectly focal control over a single neural computation; effects can propagate through connected networks.

### Step 8 — Compare the predictions of each method

Create a method-by-prediction matrix before drawing conclusions.

| Method | Primary contribution | Typical inference | Main limitation |
|---|---|---|---|
| Behavior | Whether and how well the operation is performed | Functional signature | Indirect about neural implementation |
| fMRI | Spatially distributed condition-related BOLD responses | Regional/network association | Slow indirect signal; correlational |
| EEG/ERP | Event-related voltage and temporal dynamics | Timing and condition sensitivity | Source localization and component interpretation are limited |
| MEG | Magnetic fields associated with neural currents | Temporal dynamics and source estimates | Inverse problem and coverage/sensitivity limitations |
| Lesion studies | Performance after damage or disconnection | Contribution/necessity under lesion conditions | Lesion distribution, network effects, compensation |
| TMS/tES | Performance after perturbing cortical activity | Causal contribution under stimulation conditions | Limited spatial specificity and variable effects |
| Computational model | Whether proposed operations can reproduce data | Explanatory adequacy and predictions | Model assumptions may not match biology |

The methods do not need to produce identical numerical results. They should, however, be compatible with a coherent account of the operation, its timing, and its neural dependencies.

### Step 9 — Evaluate competing explanations

For every apparent effect, generate plausible alternatives. Examples include:

- lexical difficulty rather than semantic composition;
- working-memory load rather than syntactic complexity;
- general task difficulty rather than pragmatic inference;
- attention reorienting rather than social mentalizing;
- response selection rather than comprehension itself;
- visual or auditory processing rather than linguistic computation;
- prediction violation rather than failed understanding;
- lesion-induced disconnection rather than local loss of a computation.

Then ask which additional manipulation would separate these explanations. A good experiment changes one theoretically important factor while keeping other candidate explanations as constant as possible.

### Step 10 — Estimate effect size and uncertainty

Do not decide that a hypothesis is supported merely because one p-value is below a conventional threshold. Estimate the size and precision of the effect, use confidence or credible intervals, and assess whether the result is practically meaningful.

A standardized mean difference can be summarized as:

\[
d=\frac{\overline X_1-\overline X_0}{S_{pooled}}
\]

where \(S_{pooled}\) is the pooled standard deviation. This is one possible measure, not a universal choice for every data type.

For brain-wide analyses, correct for multiple comparisons using an appropriate method, such as family-wise error control or false discovery rate. Report the correction method, threshold, region definitions, and whether analyses were exploratory or confirmatory.

### Step 11 — Test individual-level reliability

1. Estimate each participant's effect where data quality permits.
2. Determine whether the effect is reliable within the same participant across runs or sessions.
3. Compare group-average and individual-level localization.
4. Examine whether outliers reflect data quality, strategy differences, or genuine functional variation.
5. Avoid claiming universal anatomical boundaries from a group map alone.

Possible reliability measures include an intraclass correlation coefficient (ICC) or test–retest correlation, selected according to the data structure. Reliability is not equivalent to validity: a measure can be consistently wrong or can reliably measure a different process than intended.

### Step 12 — Integrate evidence without double-counting

A Bayesian framing can be useful:

\[
P(H\mid D)\propto P(D\mid H)P(H)
\]

where \(H\) is a mechanism hypothesis and \(D\) is the collected evidence.

However, evidence sources are not automatically independent. For example, an fMRI contrast and a region-of-interest analysis derived from the same dataset are not two independent replications. Similarly, several ERP measures from the same participants may share the same confounds. The integration step must record shared participants, shared stimuli, shared preprocessing, and correlated measures before treating evidence as cumulative.

A practical evidence ledger can use qualitative levels:

- **Preliminary:** one method or small sample; multiple alternatives remain.
- **Convergent association:** repeated behavioral and neural associations support the operation.
- **Temporally constrained:** EEG/MEG or intracranial timing narrows when the operation occurs.
- **Lesion-supported:** damage or disconnection is associated with the predicted impairment.
- **Perturbation-supported:** controlled stimulation produces a selective effect consistent with the hypothesis.
- **Replicated and predictive:** independent datasets and new predictions support the same mechanism.

These labels are descriptive, not a universal grading scale. They should always be accompanied by study quality and uncertainty.

### Step 13 — Update the mechanistic model

1. Retain findings that replicate across methods.
2. Modify the model when a finding supports the broad operation but contradicts a specific regional assignment.
3. Add conditional recruitment if a network is engaged only under high inference, ambiguity, or memory load.
4. Separate core language computations from supporting domain-general or social-cognitive contributions when the evidence supports that distinction.
5. Preserve overlap and interaction where the data show coordination.
6. Record unresolved alternatives rather than forcing a single explanation.
7. Generate a new prediction that was not used to fit the existing model.

### Step 14 — Validate out of sample

A mechanistic model should explain data beyond the observations used to construct it.

1. Fit model parameters on a training dataset or subset.
2. Freeze the model and its decision rules.
3. Test it on held-out participants, items, sessions, or an independent dataset.
4. Compare it with simpler baselines and plausible competing models.
5. Evaluate prediction error and calibration, not only classification accuracy.
6. Report failure cases and participant/item variation.
7. Revise the model only after the held-out evaluation is complete; use a new test set for subsequent confirmation.

A simple mean squared prediction error is:

\[
MSE=\frac{1}{N}\sum_{i=1}^{N}(y_i-\widehat y_i)^2
\]

where \(y_i\) is the observed outcome and \(\widehat y_i\) is the model prediction. Other outcomes may require log loss, likelihood, calibration error, or task-specific measures.

### Step 15 — Produce a calibrated mechanism report

For each operation, report:

- what the operation is;
- which task manipulation isolates it;
- what behavior demonstrates it;
- which regions/networks are associated with it;
- what the timing evidence suggests;
- what lesion evidence indicates;
- whether perturbation evidence exists;
- which alternative explanations remain;
- whether the result generalizes across participants and tasks;
- which components are directly measured and which are computational assumptions.

The end product is not a claim that every step has been discovered with certainty. It is a testable, revisable account of the evidence and the remaining unknowns.

## 10.4 Mapping the source's principal evidence claims into the algorithm

### A. Language selectivity of the core network

**Source claim:** A left frontotemporal language network, including inferior frontal and temporal regions, responds robustly to linguistic input and can be distinguished functionally from systems used for non-linguistic reasoning or executive control.

**Algorithmic translation:**

1. Independently localize each participant's language-responsive regions.
2. Compare linguistic conditions with carefully matched non-linguistic conditions.
3. Verify that the contrast is not explained by difficulty, response format, sensory input, or memory demands.
4. Test whether the response generalizes across multiple linguistic tasks.
5. Test whether the network's response is distinct from mentalizing and multiple-demand task responses.
6. Use lesion or stimulation evidence, when available, to assess the contribution of the network to language performance.

**Mathematical candidate:** a selectivity index can be defined as:

\[
SI=\frac{R_{language}-R_{control}}{|R_{language}|+|R_{control}|+\varepsilon}
\]

where \(R\) is a response estimate and \(\varepsilon\) prevents division by zero. This is an analyst-defined summary score, not a biological quantity. The precise contrast and baseline determine its interpretation.

### B. Distinct role of theory-of-mind and social networks

**Source claim:** mPFC and TPJ are recruited when understanding beliefs, intentions, sarcasm, irony, or unstated social meaning is required, but they are not necessarily engaged by every literal sentence.

**Algorithmic translation:**

1. Design literal and mental-state/inference conditions with similar linguistic complexity.
2. Manipulate whether speaker belief or intention is necessary for the intended interpretation.
3. Test whether mPFC/TPJ responses increase specifically when the mental-state variable changes.
4. Determine whether the effect persists when sentence length and working-memory demand are controlled.
5. Compare language-selective and mentalizing responses within the same individual.
6. Use lesion evidence to test whether social-pragmatic judgments can be impaired while literal sentence comprehension remains relatively preserved.

**Computational candidate:** represent a speaker's possible belief or intention as a latent state \(z\). Candidate interpretations \(m\) can be compared under that state:

\[
P(m\mid u,c)=\sum_z P(m\mid u,c,z)P(z\mid c)
\]

where \(u\) is the utterance and \(c\) is context. This is a formal model of inference, not proof that mPFC or TPJ explicitly stores a probability distribution.

### C. Pragmatic and causal inference across clauses

**Source claim:** Inference-rich discourse can sustain activity in language-related temporal/frontal regions and recruit medial frontal or social-reasoning regions.

**Algorithmic translation:**

1. Provide sentence pairs that vary in whether a causal or bridging relation is explicit, strongly implied, weakly implied, or absent.
2. Collect comprehension and inference judgments.
3. Measure neural responses during the clause where the missing relation becomes relevant.
4. Compare local lexical-semantic demands with discourse-level integration demands.
5. Test whether medial frontal responses track inferred causal or mental-state structure after controlling for general task difficulty.
6. Check whether the inferred relation is retained and used in later questions.

A simple coherence score can be modelled as:

\[
Coherence(m,c)=\sum_{r\in Relations(m,c)}w_r\,compatibility(r)
\]

where relations may include causal, temporal, referential, or social links. This is an analyst's computational approximation; the brain's exact representation and weighting remain open research questions.

### D. ERP evidence for incremental integration and reanalysis

**Source claim:** Contextual expectations can influence comprehension within a few hundred milliseconds; N400 and P600-like effects are observed under different forms of mismatch and reanalysis.

**Algorithmic translation:**

1. Construct contexts that vary in predictability and semantic fit.
2. Hold the critical word constant across conditions when possible, or counterbalance word identity.
3. Time-lock EEG to the critical word.
4. Compare pre-registered N400 and late-positive measures.
5. Test whether observed effects track lexical access, contextual fit, structural repair, pragmatic reanalysis, or more than one of these.
6. Avoid a rigid one-to-one mapping between each ERP component and a single processing stage.

A useful surprisal predictor is:

\[
Surprisal(w_t)=-\log P(w_t\mid context_{<t})
\]

Researchers can test whether surprisal predicts an ERP or behavioral measure. This does not establish that the brain literally computes logarithms; it tests whether an information-theoretic model captures variance in human responses.

### E. Patient dissociations

**Source claim:** Left-hemisphere language lesions can disrupt basic linguistic combination, while right-hemisphere or medial prefrontal damage may affect discourse-level or pragmatic interpretation with some literal language skills preserved.

**Algorithmic translation:**

1. Create a multidimensional assessment separating word recognition, syntax, semantic composition, literal comprehension, reference resolution, inference, sarcasm, and mental-state attribution.
2. Measure non-language capacities that could contribute to performance.
3. Relate error patterns to lesion and disconnection maps.
4. Test whether the pattern is selective or reflects broad cognitive load.
5. Compare with control participants and with patients whose lesions affect different networks.
6. Interpret dissociations as evidence for partially separable contributions rather than perfectly isolated modules.

For example, if a participant performs well on literal sentence comprehension but poorly on sarcasm, the result is consistent with a difficulty in pragmatic or mental-state interpretation. It does not by itself specify the damaged computation; the full cognitive profile and lesion distribution are necessary.

## 10.5 Research-analysis pseudocode

```python
class MechanismHypothesis:
    def __init__(self, name, operation, conditions, predictions, alternatives):
        self.name = name
        self.operation = operation
        self.conditions = conditions
        self.predictions = predictions
        self.alternatives = alternatives
        self.evidence = {}
        self.uncertainties = []


def validate_language_mechanism(hypothesis, study_plan, datasets):
    # 1. Confirm that the operation is operationally defined.
    assert hypothesis.operation is not None
    assert hypothesis.predictions
    assert hypothesis.alternatives

    # 2. Check experimental contrasts and confound controls.
    verify_conditions_match(study_plan.conditions)
    verify_counterbalancing(study_plan)
    verify_power_and_sample_plan(study_plan)

    # 3. Evaluate behavior before neural interpretation.
    behavioral = analyze_behavior(
        datasets.behavior,
        controls=study_plan.behavioral_covariates,
        participant_and_item_effects=True,
    )
    hypothesis.evidence["behavior"] = behavioral

    # 4. Evaluate spatial associations.
    if datasets.fmri is not None:
        fmri_result = analyze_fmri_glm(
            datasets.fmri,
            contrasts=study_plan.fmri_contrasts,
            nuisance_regressors=study_plan.nuisance_regressors,
            independent_rois=study_plan.independent_rois,
            multiple_comparison_control=True,
        )
        hypothesis.evidence["fmri_association"] = fmri_result

    # 5. Evaluate temporal predictions.
    if datasets.eeg_or_meg is not None:
        temporal_result = analyze_event_locked_signals(
            datasets.eeg_or_meg,
            event=study_plan.critical_event,
            pre_registered_windows=study_plan.time_windows,
            artifact_control=True,
        )
        hypothesis.evidence["temporal"] = temporal_result

    # 6. Evaluate lesion evidence separately from correlational evidence.
    if datasets.patient_data is not None:
        lesion_result = analyze_lesion_behavior_relationship(
            datasets.patient_data,
            include_lesion_volume=True,
            evaluate_disconnection=True,
            correct_for_spatial_multiple_testing=True,
        )
        hypothesis.evidence["lesion"] = lesion_result

    # 7. Evaluate perturbational evidence if available and ethical.
    if datasets.stimulation is not None:
        perturbation_result = compare_active_with_control_stimulation(
            datasets.stimulation,
            test_selective_effects=True,
        )
        hypothesis.evidence["perturbation"] = perturbation_result

    # 8. Explicitly test competing explanations.
    for alternative in hypothesis.alternatives:
        hypothesis.evidence.setdefault("alternatives", {})[alternative] = (
            evaluate_alternative_explanation(alternative, datasets, study_plan)
        )

    # 9. Integrate evidence while tracking dependence and uncertainty.
    evidence_summary = synthesize_converging_evidence(
        hypothesis.evidence,
        account_for_shared_samples=True,
        distinguish_association_from_causation=True,
    )

    # 10. Test predictions on data not used to build the model.
    if datasets.held_out is not None:
        evidence_summary["generalization"] = evaluate_held_out_predictions(
            hypothesis, datasets.held_out
        )

    # 11. Return a calibrated report, not an unsupported certainty claim.
    return build_mechanism_report(
        hypothesis=hypothesis,
        evidence=evidence_summary,
        list_unresolved_alternatives=True,
        list_next_discriminating_experiments=True,
    )
```

The pseudocode is a research workflow abstraction. Function names represent analysis tasks; they are not brain nuclei or literal biological modules.

## 10.6 Worked example — testing the role of left temporal language regions in semantic access

**Hypothesis:** Left temporal language regions contribute to accessing lexical-semantic information during comprehension.

**Experiment design:**

- Present words or sentences in contexts that vary semantic constraint.
- Control word frequency, length, visual/auditory features, and response demands.
- Include a non-linguistic control task matched for general difficulty.
- Collect behavioral data and EEG/ERP; use fMRI in a separate or combined experiment.

**Predictions:**

1. Behavioral responses should vary with semantic constraint or lexical access demand.
2. EEG may show a context-sensitive N400-related difference, with interpretation evaluated against competing retrieval/integration accounts.
3. fMRI may show condition-related response in independently identified language-selective temporal regions.
4. A lesion affecting relevant temporal language tissue or its connections may be associated with a lexical-semantic deficit, but the full lesion and cognitive profile must be examined.
5. If carefully targeted stimulation changes semantic performance relative to sham/control conditions, that would provide additional evidence for a causal contribution.

**Model comparison:**

- Model A: effect reflects lexical-semantic access.
- Model B: effect reflects general task difficulty.
- Model C: effect reflects contextual prediction only.
- Model D: effect reflects working-memory or response selection demands.

A good design includes contrasts that make these models predict different outcomes. The final conclusion should specify which elements are supported and which remain unresolved.

## 10.7 Worked example — testing pragmatic inference and mentalizing

**Hypothesis:** When an utterance requires inferring a speaker's unstated intention, language-processing regions cooperate with mentalizing systems such as mPFC and TPJ.

**Experiment design:**

- Construct literal utterances, indirect requests, and sarcasm/irony conditions.
- Provide speaker and situational context that makes the intended meaning recoverable.
- Match linguistic form and difficulty as closely as possible.
- Ask participants to identify both the literal content and the speaker's likely intention.
- Measure behavior and fMRI; use EEG/MEG if the timing of inference is central.

**Predictions:**

1. Literal comprehension may be accurate across all conditions.
2. Intention judgments should be more demanding when context or speaker belief is essential.
3. mPFC/TPJ responses may increase when mental-state inference is required, with exact activation depending on the task.
4. Language-selective regions may continue to process linguistic content across conditions.
5. Patient groups with relevant social-cognitive or frontal impairments may show disproportionate difficulties in intention judgments, but this must be established with broad control testing.

**Alternative explanations:** working-memory load, emotional salience, incongruity detection, attention, and response selection. Each must be considered before attributing the result specifically to mentalizing.

## 10.8 Worked example — distinguishing semantic composition from executive difficulty

**Hypothesis:** Basic semantic composition can be performed by the core language system without requiring strong recruitment of the multiple-demand network in every sentence.

**Experiment design:**

- Compare simple compositional sentences with semantically unusual or structurally complex sentences.
- Independently manipulate working-memory load and linguistic composition.
- Use independent localizers for language-selective and multiple-demand regions.
- Measure accuracy, response time, and fMRI activity.

**Predictions:**

1. Language-selective regions should respond to linguistic composition in both simple and more difficult conditions.
2. Multiple-demand regions may increase their response when executive or memory demands rise.
3. A linguistic effect should remain after accounting for general task demand if it reflects core language processing.
4. A broad frontal response should not be automatically classified as semantic composition; it may reflect task management or difficulty.

**Interpretation:** The result can support a division of labour between language-selective processing and domain-general assistance without implying that the systems never interact.

## 10.9 Quality-control and reproducibility loop

The algorithm must reject or flag an analysis when:

- the target cognitive operation was not operationally defined;
- control conditions differ on uncontrolled dimensions;
- too few valid trials remain for the planned ERP analysis;
- excessive motion or artifacts undermine imaging quality;
- the region of interest was selected from the same contrast used to test it without an independent procedure;
- the analysis changes after inspecting the results but is presented as pre-registered;
- multiple comparisons are ignored;
- only group averages are reported despite substantial individual differences;
- a null result is interpreted as proof that a region has no role;
- a correlation is described as causal;
- a lesion association is interpreted without considering lesion volume, disconnection, or co-occurring deficits;
- a stimulation effect is interpreted without sham/control conditions or checks for non-specific effects;
- evidence sources derived from the same sample are counted as independent replications.

**Reproducibility operations:**

1. Store stimulus lists, counterbalancing schedules, task code, and analysis code.
2. Record preprocessing settings and exclusion criteria.
3. Preserve raw data and quality-control summaries under appropriate data-governance rules.
4. Register primary hypotheses and analysis choices before testing where feasible.
5. Report deviations transparently.
6. Share de-identified data and code when consent, privacy, and institutional policy permit.
7. Replicate critical effects in a new participant sample or dataset.

## 10.10 What mathematical operations are relevant, and what do they mean?

| Mathematical operation | Use in the scientific validation algorithm | Does it imply the brain performs this exact operation? |
|---|---|---|
| Difference/contrast | Quantifies task-condition differences | No; it is an analysis operation |
| Regression/GLM | Estimates how measured signals relate to task predictors | No; it models measured data |
| Averaging | Estimates event-related signals across trials | No; it is a measurement-analysis procedure |
| Convolution | Models the delayed fMRI hemodynamic response to task events | No; it represents the measurement response |
| Mixed-effects modelling | Accounts for variation across participants and sentence items | No; it is a statistical model |
| Logistic function | Maps a model score to a probability in a statistical classifier | Not as a literal neural calculation without further evidence |
| Multiple-comparison correction | Controls false-positive risk when many tests are run | No; it is a research safeguard |
| Bayesian updating | Formalizes how evidence changes confidence in a hypothesis | It may inspire cognitive models, but research use does not prove biological implementation |
| Surprisal \(-\log P\) | Tests whether contextual predictability relates to neural/behavioral responses | It is a model quantity; literal logarithmic computation is not established |
| Cross-validation | Tests generalization to unseen data | No; it is a model-evaluation procedure |
| Graph/network analysis | Describes relationships among measured brain regions | It does not prove direct anatomical pathways or information flow |

## 10.11 Limitations and unresolved questions

1. **fMRI is correlational in ordinary task designs.** A region's response may reflect several computations or inputs.
2. **BOLD is indirect.** It depends on vascular and metabolic responses and has a slower time course than neuronal signaling.
3. **EEG/MEG source localization is an inverse problem.** Different source configurations can produce similar sensor patterns; temporal precision does not automatically provide exact anatomical localization.
4. **ERP components are not one-to-one process labels.** N400 and P600 interpretations are actively discussed and can be sensitive to multiple task factors.
5. **Lesion evidence is constrained by natural lesion patterns.** Vascular territories, lesion volume, disconnection, diaschisis, and recovery affect inference.
6. **Stimulation is not perfectly focal.** TMS/tES effects depend on individual anatomy, target accessibility, timing, intensity, and connected networks. Published meta-analyses report modest average effects and emphasize design limitations.
7. **Patient dissociations are informative but not absolute.** A person may show a selective pattern on one task and broader difficulties on another.
8. **Task demands can recruit extra systems.** Memory, attention, decision-making, and explicit judgment requirements can change the network profile.
9. **Individual variability matters.** Group-averaged localization can conceal subject-level differences.
10. **Computational fit is not biological proof.** A model may reproduce behavior without identifying the actual biological implementation.
11. **A null result is ambiguous.** It may reflect insufficient power, weak manipulation, measurement noise, task strategy, or true absence of the predicted effect.
12. **The best explanation may be distributed.** Many comprehension operations depend on interactions among regions rather than a single isolated center.

## 10.12 Source alignment and references for Part 10

### Alignment with the supplied source file

This part translates the file's **Evidence from Neuroimaging and Patient Studies** section, including:

- the language-selectivity of the core left frontotemporal network;
- the distinction between language-selective and theory-of-mind/social systems;
- fMRI findings involving discourse-level causal inference and mental-state reasoning;
- ERP/MEG findings associated with N400 and P600 effects;
- lesion-based dissociations between basic language and pragmatic interpretation;
- the need for converging evidence and caution about task confounds and individual variability.

### Research references used to refine methodological cautions

1. **Bajracharya, A., & Peelle, J. E. (2023).** “A systematic review of neuroimaging approaches to mapping language in individuals.” *Journal of Neurolinguistics*, 68, 101163. DOI: [10.1016/j.jneuroling.2023.101163](https://doi.org/10.1016/j.jneuroling.2023.101163). Reviews individual-level language mapping and cautions against relying solely on group averages.
2. **“Lesion mapping in neuropsychological research: A practical and conceptual guide.” (2024).** *Cortex*. DOI: [10.1016/j.cortex.2023.10.001](https://doi.org/10.1016/j.cortex.2023.10.001). Explains what lesion mapping can and cannot establish, including lesion coverage, sample size, and causal-inference limitations.
3. **“Retrieval (N400) and integration (P600) in expectation-based comprehension.”** Available at [PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC8478172/). Reviews competing interpretations of N400 and P600 and their use in expectation-based comprehension models.
4. **Delogu, F., Aurnhammer, C., Brouwer, H., & Crocker, M. W. (2025).** “On the biphasic nature of the N400-P600 complex underlying language comprehension.” *Brain and Cognition*, 186, 106293. DOI: [10.1016/j.bandc.2025.106293](https://doi.org/10.1016/j.bandc.2025.106293). Examines variability in N400/P600 patterns and the role of component overlap and comprehension depth.
5. **“Verb and sentence processing with TMS: A systematic review and meta-analysis.” (2023).** *Cortex*, 162, 38–55. DOI: [10.1016/j.cortex.2023.01.005](https://doi.org/10.1016/j.cortex.2023.01.005). Reviews TMS evidence for verb and sentence processing; reports a small overall effect size and highlights the need for broader and better controlled target comparisons.
6. **“Neuromodulatory effects of transcranial magnetic stimulation on language performance in healthy participants: Systematic review and meta-analysis.” (2022).** Available at [PubMed](https://pubmed.ncbi.nlm.nih.gov/36545349/). Reports modest average effects and notes limitations related to task, target, and experimental design.
7. **“A time and place for language comprehension: mapping the N400 and the P600 to a minimal cortical network.”** Available at [PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC3824103/). Reviews the history of N400/P600 interpretations and cautions against an overly simple one-component/one-process mapping.
8. **Kutas, M., & Hillyard, S. A. (1980).** “Reading senseless sentences: Brain potentials reflect semantic incongruity.” *Science*, 207(4427), 203–205. Foundational ERP study of semantic incongruity and the N400.

## 10.13 Handoff and completion status

**Part 10 is a validation algorithm, not a new stage in the listener's or reader's moment-to-moment comprehension pipeline.** It specifies how to test the mechanisms described in Parts 1–9 with behavioral, imaging, electrophysiological, lesion, perturbational, and computational evidence.

**Final output:** a calibrated and falsifiable account of language-comprehension mechanisms, including which functions are supported by which methods, how neural systems may contribute, which alternatives remain viable, and what future experiment can distinguish them.

**End of Part 10 — Evidence-Based Identification and Validation of Language-Comprehension Mechanisms.**

# PART 11 — FEDORENKO'S LANGUAGE NETWORK VS. OTHER BRAIN SYSTEMS: A FUNCTIONAL-SEPARATION AND COOPERATION ALGORITHM

## 11.0 Purpose and scope

The supplied source contains a distinct discussion of Fedorenko and colleagues' account of the core language network and its relationship to other brain systems. Parts 9 and 10 already described the broad neural architecture and the methods used to validate proposed mechanisms. This part narrows the question: **when a person understands a linguistic message, which computations belong to the language-selective system, which are supplied by other systems, and how can these systems cooperate without being treated as one undifferentiated processor?**

This is not an eighth or ninth moment in the sequence of sentence comprehension. It is a cross-cutting algorithm for assigning functional responsibilities throughout comprehension. It also is not an assertion that the brain consciously chooses a module or executes symbolic routing instructions. The algorithm is a computational reconstruction of functional distinctions inferred from neuroimaging, lesion, stimulation, and behavioural evidence.

The 2024 review by Fedorenko, Ivanova, and Regev characterizes a core language network in left frontal and temporal cortex as strongly interconnected, largely independent of input/output modality, causally important for language, and selective for linguistic processing. The review argues that this network works closely with lower-level perceptual/motor mechanisms and higher-level knowledge/reasoning systems while remaining functionally distinct from them. The algorithm below turns that proposal into explicit operations, tests, decision rules, and mathematical approximations.

### Key distinction

**Functional specialization is not the same as physical isolation.** Two systems can have distinguishable response profiles while exchanging information continuously. The language system can retrieve and combine linguistic knowledge; conceptual and world-knowledge systems can supply information about entities and events; social-cognitive systems can contribute representations of beliefs and intentions; and executive systems can support selection, monitoring, and demanding task control. Their interaction does not make them one system, just as their selectivity does not imply that they never interact.

### What this point does not assume

1. It does not assume that every sentence requires strong activation of every higher-level network.
2. It does not assume that every operation is localized to one cortical point.
3. It does not equate an fMRI activation with proof that a region is necessary for a computation.
4. It does not assume that all people have identical language-network topography or the same degree of left lateralization.
5. It does not assume that language and thought are completely independent. It asks which parts of cognition can be carried out without language and which tasks require linguistic representations.
6. It does not treat the 2024 framework as immune to revision. Later meta-analytic and individualized-connectivity findings are incorporated as boundary conditions and refinements.

---

## 11.1 Overall objective

Given a spoken or written message, a current task, the developing linguistic representation, and relevant situational information:

- identify the linguistic computations that must be performed;
- distinguish sensory-form processing from linguistic computation;
- maintain access to the core language system for word/construction retrieval and linguistic combination;
- estimate whether additional conceptual, social-cognitive, memory, or executive operations are required;
- permit the systems to exchange only the information needed by the ongoing task;
- preserve the identity of each system's contribution when producing an integrated interpretation; and
- revise the allocation as the sentence, discourse, and task evolve.

The result is a **coordinated multi-system comprehension state**, not a single undifferentiated neural state.

### Inputs

- acoustic or visual linguistic signals;
- candidate word forms and lexical entries;
- syntactic and semantic structures under construction;
- current proposition and situation-model hypotheses;
- preceding discourse and the physical environment;
- relevant world knowledge and episodic memories;
- speaker identity, communicative context, and possible beliefs/intentions;
- task instructions and response requirements;
- current uncertainty, ambiguity, conflict, and working-memory demand; and
- where available, an individual's measured functional organization rather than only a group-average atlas.

### Outputs

- a decomposition of the task into linguistic, sensory, conceptual, social, and executive demands;
- estimates of which functional systems contribute to each demand;
- a language-selective representation that can be accessed across input modalities;
- separately represented conceptual, social, and control contributions;
- a set of information-exchange relationships among the systems;
- an integrated but provenance-aware interpretation; and
- an updated account of the demands expected at the next word, clause, or discourse event.

---

## 11.2 Functional systems and their computational responsibilities

### 11.2.1 Perceptual and motor mechanisms: represent the surface form

**Auditory mechanisms** analyze the acoustic stream: timing, frequency structure, intensity, voice characteristics, phonetic cues, and speech rhythm. **Visual mechanisms** process the visual features of printed language: edges, spatial arrangement, letter-like patterns, and orthographic position. Speech production and articulation additionally involve motor planning and execution systems.

Their key contribution is to transform physical signals into structured sensory evidence that can support language processing. A sensory representation may be highly specialized and necessary for reading or listening without itself representing the full meanings of words or sentences.

**Algorithmic responsibility:** return a modality-specific representation plus confidence/quality information. Do not mark a sensory feature detector as a semantic processor merely because it responds strongly to a linguistic stimulus.

### 11.2.2 Core language-selective network: retrieve and combine linguistic knowledge

The source describes a predominantly left-hemisphere frontotemporal network, with individual variation in its exact anatomy. Its proposed core responsibilities include:

- accessing lexical and constructional knowledge stored in long-term linguistic representations;
- mapping recognized input onto words and grammatical constructions;
- building and interpreting hierarchical linguistic structures;
- combining words and constructions into larger linguistic representations;
- supporting comprehension and language production; and
- operating across auditory and visual input, rather than being reducible to one sensory modality.

Commonly discussed components include lateral frontal language-responsive areas, including parts of the inferior frontal gyrus, and temporal language-responsive regions spanning superior and middle temporal cortex. The exact contribution of individual subregions should not be over-specified: the language network is distributed and its internal division of labour remains an active research topic.

**Algorithmic responsibility:** preserve linguistic form and structure, access learned linguistic knowledge, and construct the linguistic message representation. It does not need to independently solve every social, logical, numerical, or world-knowledge problem encountered while reading a sentence.

### 11.2.3 Conceptual and world-knowledge systems: supply information about what the message is about

Understanding words and sentences depends on knowledge about objects, events, properties, actions, relations, and the regularities of the world. Such information is not necessarily stored in one isolated “meaning centre.” Semantic knowledge is distributed, and different theories emphasize different contributions from anterior temporal, posterior temporal, inferior parietal, angular gyrus, and other cortical areas. Default-mode regions can also contribute to building coherent event and situation models in some tasks.

**Algorithmic responsibility:** when the linguistic representation activates an entity, event, or relation, retrieve relevant conceptual features and associations. Supply those features to composition and contextual integration while maintaining a distinction between the linguistic cue and the knowledge retrieved from it.

### 11.2.4 Mentalizing / theory-of-mind systems: represent agents' perspectives and intentions

Medial prefrontal cortex and temporoparietal junction regions are frequently implicated when a task requires reasoning about another person's beliefs, knowledge, communicative intentions, or perspective. These are not simply language processors. They also participate in non-linguistic social-cognition tasks.

**Algorithmic responsibility:** when an interpretation depends on what a speaker or character knows, believes, wants, or intends, construct or update an agent-centred mental-state hypothesis. Send the inferred intention back to the language/context system as supporting information, not as a replacement for the literal linguistic content.

### 11.2.5 Multiple-demand / executive systems: provide flexible control under task demand

Bilateral frontal and parietal regions, including dorsolateral prefrontal and intraparietal areas, are associated with domain-general control and demanding cognition. Their contribution can increase when a task requires substantial working-memory maintenance, selection among competitors, multi-step reasoning, response inhibition, or explicit decision-making.

**Algorithmic responsibility:** support control when the current task demands it. Do not assume that every difficult sentence recruits these regions because its linguistic structure is inherently computed there. Distinguish intrinsic linguistic complexity from external task requirements such as remembering a list, making a speeded judgment, or comparing multiple alternatives.

### 11.2.6 Memory systems: preserve and retrieve information beyond the current linguistic operation

Working memory maintains active content over a short interval. Episodic memory can bind events to contexts, while semantic memory supports durable knowledge. Medial temporal lobe and hippocampal circuitry are particularly relevant to forming and retrieving episodic relationships; longer-term semantic learning depends on distributed cortical representations and consolidation processes.

**Algorithmic responsibility:** maintain the current proposition during comprehension and support encoding or retrieval when the task requires it. Do not confuse temporary maintenance with the language-selective computation that originally built the proposition.

---

## 11.3 Main algorithm: functional separation with coordinated exchange

The steps below describe a functional algorithm. They are ordered for clarity, but many processes operate in parallel and may feed information back to earlier steps.

### Step 1 — Decompose the current task before assigning neural responsibilities

**Purpose:** avoid attributing every operation to the language network merely because the task is delivered through words.

1. Receive the linguistic input and task instruction.
2. Separate the operations that the task requires: sensory decoding, lexical recognition, syntactic interpretation, semantic composition, conceptual retrieval, social inference, memory maintenance, executive control, and response selection.
3. Distinguish the message from the task imposed on the message. For example, reading a simple sentence while answering a difficult multi-choice question adds decision and comparison demands that are not identical to reading the sentence itself.
4. Record whether the input is spoken, written, signed, or mixed. Do not treat a modality difference as proof that the underlying linguistic computation is different.
5. Record the relevant context and the type of information needed to resolve the sentence.

**Possible mathematical description:** represent the task by a demand vector

\[
\mathbf d_t=[d_{sens},d_{lex},d_{syn},d_{sem},d_{world},d_{social},d_{WM},d_{exec},d_{resp}]^T.
\]

Each component is an estimated demand, not a directly observed neural variable. A task profile may be built from its stimulus structure, behavioural requirements, and experimentally measured response patterns.

**Biological interpretation:** the brain is exposed to a structured problem. Different systems can respond to the different properties of that problem; there is no evidence that a central homunculus explicitly writes this vector.

### Step 2 — Encode sensory form without prematurely assigning meaning

1. Route speech to auditory processing and text to visual processing, with convergence into language-related representations.
2. Extract the features needed to distinguish the current signal from alternatives.
3. Estimate signal quality, noise, and uncertainty.
4. Pass candidate phonetic/orthographic and temporal information toward the language network.
5. Retain modality-specific information where it remains useful, but do not force the core linguistic representation to remain tied to the sensory channel.

**Possible mathematical operations:** filtering, temporal integration, contrast normalization, feature population coding, nonlinear response functions, and reliability-weighted cue combination.

A generic sensory population model is:

\[
\mathbf p(t)=\phi(W_s\mathbf x(t)+\mathbf b_s),
\]

where \(\mathbf x(t)\) is an input feature vector, \(W_s\) is an effective connectivity matrix, \(\mathbf b_s\) is a bias vector, and \(\phi\) is a nonlinear response function. This is a model of population transformation, not proof that a cortical area performs a literal matrix multiplication.

### Step 3 — Identify the individual's language-selective network

1. Use evidence from an appropriate language localizer or individualized functional-connectivity method where research-grade mapping is available.
2. Compare linguistic conditions with carefully selected non-linguistic control conditions.
3. Identify regions that reliably respond more to linguistic stimuli than to matched controls.
4. Examine the distributed pattern and its connectivity, rather than selecting a region solely because it is anatomically labelled “Broca's area” or “Wernicke's area.”
5. Where possible, estimate the network separately for each individual; use group atlases as priors rather than as exact personal maps.
6. Record uncertainty and atypical lateralization rather than forcing every person into a standard left-hemisphere template.

**Possible mathematical operations:** within-subject contrast estimates, effect-size calculation, spatial overlap, functional-connectivity matrices, reliability estimates, and cross-validated classification.

A simplified selectivity score for region or network \(r\) is:

\[
SI_r=\frac{\mu_{r,lang}-\mu_{r,control}}{\sigma_{r,pooled}+\epsilon}.
\]

Here \(\mu\) is the average measured response, \(\sigma\) is a variability estimate, and \(\epsilon\) prevents division by zero. This is a research statistic; it is not a biological signal that neurons are known to compute.

### Step 4 — Route lexical and grammatical information through language-selective computations

1. Activate candidate lexical and constructional representations using the incoming sensory-derived word-form evidence.
2. Resolve word-form candidates and retrieve the linguistic properties needed for interpretation.
3. Construct grammatical relationships incrementally.
4. Combine lexical items and grammatical constructions into a literal sentence-level representation.
5. Maintain alternatives when the linguistic evidence is ambiguous.
6. Update the representation as new words arrive.

**Key boundary:** word meanings may connect to conceptual knowledge, but the fact that a sentence expresses world knowledge does not mean that all world knowledge is represented within the language-selective network.

**Possible mathematical operations:** distributed activation, competition among lexical candidates, recurrent message passing, structured binding, probability-weighted prediction, and incremental state updating.

### Step 5 — Separate linguistic composition from conceptual enrichment

1. Construct the linguistic representation licensed by the words and grammatical relations.
2. Identify the entities, events, properties, and relations activated by that representation.
3. Retrieve relevant conceptual features from distributed knowledge systems.
4. Distinguish information explicitly stated by the sentence from information retrieved by association or inferred from world knowledge.
5. Combine compatible conceptual features with the developing proposition.
6. Preserve unresolved or conflicting features for subsequent evaluation rather than silently forcing them into a single interpretation.

**Mathematical model:** let \(\mathbf l_t\) be a linguistic representation and \(\mathbf k_t\) a conceptual-knowledge state. A model may produce an enriched state

\[
\mathbf e_t=F(\mathbf l_t,\mathbf k_t,\mathbf c_t),
\]

where \(\mathbf c_t\) is contextual information. The function \(F\) expresses a hypothesized interaction. It does not establish that linguistic and conceptual information occupy the same neurons or representation space.

### Step 6 — Ask whether the interpretation requires reasoning about an agent's mind

1. Inspect the developing message for a person, speaker, character, or other intentional agent.
2. Determine whether the sentence can be understood from literal content and ordinary world knowledge alone.
3. Test whether a missing belief, knowledge state, intention, or perspective is required to resolve the meaning.
4. If no mental-state inference is needed, do not assume strong mentalizing-network recruitment.
5. If it is needed, generate agent-centred hypotheses and evaluate them against the words, discourse history, and available situation cues.
6. Return the best-supported mental-state information to the contextual interpretation while retaining its inferred status.

**Possible mathematical operations:** candidate generation, belief-state tracking, perspective transformation, consistency testing, and Bayesian-style evidence updating.

For an agent \(a\) and proposition \(q\), a model may track:

\[
B_a(q,t)=P(\text{agent }a\text{ believes }q\mid E_{1:t}),
\]

where \(E_{1:t}\) denotes the available evidence. This is a formal model of belief attribution, not a claim that the brain stores explicit numerical probabilities for every belief.

### Step 7 — Estimate executive demand independently of linguistic demand

1. Estimate how much information must be maintained simultaneously.
2. Estimate whether the task requires selecting between competing representations.
3. Determine whether the reader/listener must inhibit a misleading interpretation or compare the sentence with an explicit rule.
4. Account for speed pressure, dual-task load, response complexity, and explicit metalinguistic judgment.
5. Recruit domain-general control support when these demands are high enough to require it.
6. Keep a distinction between a sentence that is structurally complex and an experiment that imposes a difficult response task.

A simple abstract demand estimate is:

\[
D_{exec}=w_1L_{WM}+w_2N_{comp}+w_3I_{conflict}+w_4T_{pressure}+w_5R_{response},
\]

where \(L_{WM}\) is working-memory load, \(N_{comp}\) the number/strength of active competitors, \(I_{conflict}\) an interference estimate, \(T_{pressure}\) time pressure, and \(R_{response}\) response-selection complexity. The weights are model parameters. They must be estimated or justified, not invented as fixed facts about the brain.

### Step 8 — Maintain specialization while allowing communication

1. Preserve the linguistic representation in the language network.
2. Preserve relevant event/world knowledge in conceptual systems.
3. Preserve agent beliefs and intentions in social-cognitive representations when required.
4. Maintain task goals and selection constraints in control systems when required.
5. Exchange task-relevant information across systems.
6. Do not overwrite the source of an inferred feature: retain whether it came from literal language, world knowledge, a speaker model, or a decision rule.
7. Allow feedback when an external system identifies a mismatch that changes the likely linguistic interpretation.

**Possible mathematical description:** represent functional systems as nodes in a directed, weighted graph. Let \(C_{ij}\) denote the effective coupling from system \(j\) to system \(i\). A simplified state update is:

\[
\mathbf h_i(t+1)=\phi\left(A_i\mathbf h_i(t)+\sum_{j\ne i}g_{ij}(t)C_{ij}\mathbf h_j(t)+B_i\mathbf u_i(t)\right).
\]

\(\mathbf h_i\) is a system-level state, \(A_i\) its within-system dynamics, \(g_{ij}\) a time-varying coupling/gating term, and \(\mathbf u_i\) the local input. Functional MRI connectivity does not directly provide the directed causal coefficients in this equation; the equation is a modelling framework for interactions.

### Step 9 — Test for task-appropriate selectivity

1. Compare responses to language and non-language tasks matched for difficulty as closely as possible.
2. Test whether a candidate language region responds to linguistic content rather than merely to visual complexity, auditory energy, memory load, or response selection.
3. Test non-verbal social reasoning separately from verbal theory-of-mind tasks, because a verbal story includes language processing as well as mentalizing.
4. Test arithmetic, executive control, music, and high-level visual tasks where relevant.
5. Assess whether the candidate network's response pattern is selective, rather than relying only on the presence of activation in one condition.
6. Replicate the comparison at the individual level and across task variants.

**Important interpretation:** selectivity is a profile across a set of contrasts, not a claim that a system can never respond to anything outside its preferred domain.

### Step 10 — Distinguish a network's contribution from its co-activation

1. Identify which system is active during the target task.
2. Ask whether its response predicts a specific behavioural component after controlling relevant nuisance/task factors.
3. Compare conditions that require the proposed computation with conditions that do not.
4. Use lesion or stimulation evidence, where possible, to test whether perturbing the system changes the target ability.
5. Examine whether an observed deficit is specific to the proposed operation or reflects broader language, memory, attention, or task-control difficulty.
6. Report whether the evidence supports association, contribution, necessity, or causal sufficiency; these are not interchangeable conclusions.

### Step 11 — Preserve separate measures for language, thought, and task performance

1. Measure linguistic competence with appropriate comprehension and production tasks.
2. Measure non-linguistic reasoning using tasks that minimize verbal instructions and verbal response requirements.
3. Measure social cognition using both verbal and non-verbal tasks where possible.
4. Measure executive function and working memory independently.
5. Compare the profiles, taking account of sensory, motor, and general cognitive demands.
6. Use dissociations and converging evidence to determine whether a language impairment necessarily entails a reasoning impairment, or vice versa.

A double dissociation is especially informative: one group or patient profile may show impaired language with relatively preserved non-linguistic reasoning, while another shows impaired social reasoning or executive control with relatively preserved core language. Such patterns support functional separability, but they do not prove that the systems never interact.

### Step 12 — Add individual-specific network organization

1. Estimate each person's language-responsive network where reliable data are available.
2. Record the network's spatial extent, laterality, and connectivity.
3. Compare within-person stability across sessions with between-person variability.
4. Avoid assuming that a group peak coordinate is the correct target for every brain.
5. Consider developmental history, bilingual experience, literacy, handedness, neurological history, and modality-specific experience as possible contributors to variability.
6. Separate individual anatomical variation from differences in the underlying function.

### Step 13 — Incorporate right-hemisphere contributions without collapsing the model

1. Do not assume that all linguistic processing is exclusively left-lateralized.
2. Ask whether the task involves discourse, narrative, semantic control, social content, prosody, or broad context, each of which may change right-hemisphere recruitment.
3. Distinguish a right-hemisphere language-homologous response from a domain-general control or social-cognitive response.
4. Compare single-word and word-pair conditions with sentence and narrative conditions.
5. Attribute right-hemisphere activity to the process best supported by the condition contrasts, not merely by anatomical location.

A 2025 meta-analysis of 561 functional neuroimaging studies reported that right-hemisphere language-network regions were more engaged by sentence/narrative semantic stimuli than by single words/word pairs; right inferior frontal engagement increased with semantic-control demands, while multiple-demand network recruitment was associated with increased external task demands. This refines any simple rule that the right hemisphere or multiple-demand system is only recruited when language comprehension fails.

### Step 14 — Update the functional allocation when interpretation changes

1. Monitor the current sentence representation and uncertainty.
2. If a new word changes the grammatical structure, update the language-system state.
3. If new context changes which conceptual knowledge is relevant, update the conceptual state.
4. If the speaker's intention becomes important, increase the contribution of agent/perspective inference.
5. If the task introduces competing response options or a memory demand, increase control and maintenance support.
6. Re-evaluate interactions after every important update.
7. Do not treat this as a fixed routing decision made once at sentence onset.

### Step 15 — Integrate the outputs while retaining their provenance

1. Combine the literal linguistic proposition with contextually relevant knowledge.
2. Include mental-state inferences only where supported.
3. Distinguish directly stated facts from inferred facts and uncertain hypotheses.
4. Keep incompatible alternatives available when evidence is insufficient.
5. Produce a coherent representation suitable for working memory and the next comprehension step.
6. Store the system-specific contribution record for audit or model analysis.

### Step 16 — Stop recruiting additional systems when their contribution is no longer useful

1. Estimate whether the current interpretation is stable enough for the task.
2. Ask whether a further inference changes the answer or merely adds speculative detail.
3. Reduce the demand assigned to a supporting system when its additional contribution is no longer needed.
4. Preserve the information in working memory or longer-term memory if the task requires it.
5. Continue monitoring because new input can re-open an ambiguity or change the speaker model.

This step is not a claim that a single threshold neuron switches an entire network off. It is an algorithmic representation of conditional recruitment and changing task demand.

---

## 11.4 Mathematical operations that can describe the functional distinction

The equations in this section are useful for modelling and algorithm design. They should not be presented as literal formulas known to be evaluated by neurons.

### 11.4.1 Demand decomposition

Represent a task as a vector of demand features:

\[
\mathbf d=[d_{sens},d_{ling},d_{know},d_{social},d_{exec},d_{mem}]^T.
\]

This enables an algorithm or experiment to distinguish a sentence's language requirements from its social reasoning and task-control requirements. The vector is an analytical construct.

### 11.4.2 Functional selectivity

For a network \(r\), compare a target linguistic response with a matched control response:

\[
SI_r=\mu_{r,L}-\mu_{r,C}
\]

or use a standardized contrast:

\[
SI_r^{*}=\frac{\mu_{r,L}-\mu_{r,C}}{\sigma_{r,pooled}+\epsilon}.
\]

The contrast must be interpreted in relation to the control task. Poor controls can make a network look language-selective when it actually responds to generic difficulty, sensory stimulation, or memory load.

### 11.4.3 Multi-task response profile

A selectivity profile is more informative than a single contrast. For network \(r\), define:

\[
\mathbf v_r=[R_{lang},R_{math},R_{ToM,verbal},R_{ToM,nonverbal},R_{exec},R_{music},R_{visual}] .
\]

The relative pattern across tasks helps estimate the system's functional preference. This is a measurement/modeling operation, not an intrinsic neural data structure.

### 11.4.4 Conditional recruitment

A model can estimate recruitment of system \(k\) from current task demands:

\[
g_k=\sigma(\mathbf w_k^T\mathbf d-\theta_k),\qquad \sigma(z)=\frac{1}{1+e^{-z}}.
\]

Here \(g_k\) is a modelled recruitment weight, \(\mathbf w_k\) represents the relevance of each demand feature, and \(\theta_k\) is a threshold. The logistic function is an algorithmic abstraction; it does not imply a literal logistic calculation in cortex.

### 11.4.5 Distributed information exchange

For a system graph with nodes \(i\) and \(j\), a simplified exchange model is:

\[
\mathbf h_i(t+1)=\phi\left(A_i\mathbf h_i(t)+\sum_j g_{ij}(t)C_{ij}\mathbf h_j(t)+B_i\mathbf u_i(t)\right).
\]

This captures recurrence and selective interaction. The coupling matrix is not the same as an anatomical tract map. Functional connectivity is statistical dependence and cannot, by itself, determine direction or causality.

### 11.4.6 Representational provenance

A computational system can track the source of each feature:

\[
z_m=(value_m,source_m,confidence_m,time_m),
\]

where \(source_m\) may be literal-language, conceptual knowledge, speaker model, physical context, or task instruction. This is especially useful for separating what a sentence states from what the comprehender infers. It is a design principle for an algorithm, not evidence that the brain stores explicit provenance tags in this format.

### 11.4.7 Double dissociation and functional separability

Let \(L\) be a measure of language ability and \(T\) a measure of non-linguistic thought/reasoning. The analysis compares profiles across groups or patients:

\[
\Delta_L= L_{control}-L_{patient},\qquad
\Delta_T= T_{control}-T_{patient}.
\]

A selective language impairment may show a relatively large \(\Delta_L\) with a smaller \(\Delta_T\), whereas another profile can show the opposite. The inference must be based on reliable tests, matched general demands, and a full clinical profile, not one raw score.

### 11.4.8 Individual functional localization

Let \(A_{i,r}\) be the measured response of person \(i\) in region/network \(r\). Estimate each individual's language contrast and repeatability:

\[
SI_{i,r}=\mu_{i,r,L}-\mu_{i,r,C},\qquad
Rel_{i,r}=Corr(SI_{i,r}^{(session1)},SI_{i,r}^{(session2)}).
\]

The exact reliability statistic should fit the data and design. The main idea is to avoid assuming that a group-average map provides a precise personal localization.

---

## 11.5 Four worked examples

### Example A — Understanding an ordinary literal sentence

**Input:** “The child opened the door.”

1. Visual or auditory sensory systems encode the signal.
2. The language-selective network identifies the words and their grammatical relations.
3. Linguistic composition constructs an event representation: OPEN(child, door).
4. Conceptual knowledge activates relevant information about children, doors, and opening events.
5. No explicit speaker-belief inference is needed.
6. No unusual external control demand is present, so strong additional executive recruitment is not required by the model.
7. The integrated proposition is maintained in working memory.

**Functional conclusion:** the core linguistic operation is supported by the language network, while ordinary conceptual knowledge enriches the message. A full theory-of-mind inference is not automatically required merely because a human agent appears in the sentence.

### Example B — A sentence about a character's false belief

**Input:** “Maya thinks the keys are in the drawer, but they were moved to the table.”

1. The language network builds the sentence structure and literal propositions.
2. Conceptual knowledge supplies representations of keys, drawers, tables, and moving events.
3. The message explicitly contrasts Maya's belief with the actual location.
4. The task requires maintaining two representations: the agent's belief and the world state.
5. Mentalizing systems are recruited to represent the difference between what Maya believes and what is actually true.
6. Executive/working-memory support may be needed if the task requires tracking multiple agents, locations, or time points.
7. The final representation preserves both propositions rather than merging them into one.

**Functional conclusion:** language is necessary to decode the verbal message, but the reasoning about Maya's mental state is not reducible to a language-specific operation.

### Example C — A linguistic task with high external memory demand

**Task:** Read a moderately complex sentence and retain four unrelated digits while deciding whether the sentence is grammatical.

1. The language network processes the words and structure.
2. The digit-retention instruction adds a working-memory demand unrelated to the sentence's linguistic content.
3. Domain-general executive and frontoparietal control systems may be recruited to maintain the digits and coordinate the response.
4. If the experiment does not model the extra memory demand, activity in these systems may be incorrectly attributed to sentence parsing.
5. A matched control task should separate grammatical processing from digit retention and response selection.

**Functional conclusion:** activity in multiple-demand regions can reflect external task requirements rather than a necessary component of basic linguistic composition.

### Example D — Socially meaningful narrative with right-hemisphere and executive contributions

**Input:** A multi-sentence narrative describes a character's ambiguous remark, the listener's reaction, and a later revelation about the character's intention.

1. The language network processes each sentence and builds a linguistic discourse representation.
2. Conceptual systems retrieve event knowledge and maintain a coherent situation model.
3. Mentalizing systems may infer what the speaker knows or intends.
4. Right-hemisphere language-related regions may contribute to broader discourse/narrative demands, semantic control, or social-content processing depending on the task and contrast.
5. Multiple-demand resources may be engaged if the task adds explicit comparison, memory load, or difficult response selection.
6. The algorithm keeps these contributions distinct and integrates them only at the point where they inform the final interpretation.

**Functional conclusion:** the presence of right-hemisphere or executive activity is not automatically evidence that the core left language network has failed. Rich narrative tasks impose demands that are not present in single-word tasks.

---

## 11.6 Integrated pseudocode

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EvidenceSource(str, Enum):
    LITERAL_LANGUAGE = "literal_language"
    CONCEPTUAL_KNOWLEDGE = "conceptual_knowledge"
    PHYSICAL_CONTEXT = "physical_context"
    SPEAKER_MODEL = "speaker_model"
    TASK_INSTRUCTION = "task_instruction"


@dataclass
class SourcedFeature:
    value: Any
    source: EvidenceSource
    confidence: float
    time_index: int


@dataclass
class DemandProfile:
    sensory: float = 0.0
    linguistic: float = 0.0
    conceptual: float = 0.0
    social: float = 0.0
    working_memory: float = 0.0
    executive: float = 0.0
    response: float = 0.0


@dataclass
class ComprehensionState:
    sensory_form: Optional[Any] = None
    lexical_candidates: List[Any] = field(default_factory=list)
    syntactic_parse: Optional[Any] = None
    literal_proposition: Optional[Any] = None
    conceptual_features: List[SourcedFeature] = field(default_factory=list)
    agent_beliefs: List[SourcedFeature] = field(default_factory=list)
    inferred_features: List[SourcedFeature] = field(default_factory=list)
    alternatives: List[Any] = field(default_factory=list)
    demand: DemandProfile = field(default_factory=DemandProfile)
    language_network_state: Optional[Any] = None
    conceptual_system_state: Optional[Any] = None
    mentalizing_state: Optional[Any] = None
    executive_state: Optional[Any] = None
    integrated_representation: Optional[Any] = None
    uncertainty: float = 1.0


def estimate_demands(input_signal, task, context) -> DemandProfile:
    """Estimate task demands; this is an algorithmic model, not a neural report."""
    return DemandProfile(
        sensory=estimate_sensory_demand(input_signal),
        linguistic=estimate_linguistic_demand(input_signal),
        conceptual=estimate_conceptual_demand(input_signal, context),
        social=estimate_mental_state_demand(input_signal, task, context),
        working_memory=estimate_wm_demand(input_signal, task, context),
        executive=estimate_control_demand(input_signal, task, context),
        response=estimate_response_demand(task),
    )


def process_comprehension(input_signal, task, context, individual_profile):
    state = ComprehensionState()
    state.demand = estimate_demands(input_signal, task, context)

    # 1. Sensory systems encode the form of speech or text.
    state.sensory_form = sensory_encode(input_signal)

    # 2. The core language system retrieves linguistic knowledge and builds structure.
    state.language_network_state = identify_individual_language_network(
        individual_profile
    )
    state.lexical_candidates = retrieve_lexical_candidates(state.sensory_form)
    state.syntactic_parse = build_incremental_parse(
        state.lexical_candidates,
        context=context,
    )
    state.literal_proposition = compose_literal_proposition(
        state.lexical_candidates,
        state.syntactic_parse,
    )

    # 3. Conceptual knowledge is recruited for relevant entities and events.
    state.conceptual_system_state = retrieve_conceptual_knowledge(
        state.literal_proposition,
        context,
    )
    state.conceptual_features = annotate_features_with_source(
        state.conceptual_system_state,
        EvidenceSource.CONCEPTUAL_KNOWLEDGE,
    )

    # 4. Mental-state reasoning is conditional, not compulsory for every sentence.
    if state.demand.social >= SOCIAL_INFERENCE_THRESHOLD:
        state.mentalizing_state = infer_agent_perspectives(
            state.literal_proposition,
            context,
        )
        state.agent_beliefs = annotate_features_with_source(
            state.mentalizing_state,
            EvidenceSource.SPEAKER_MODEL,
        )

    # 5. Executive support depends on task/control requirements.
    if (
        state.demand.working_memory >= WM_SUPPORT_THRESHOLD
        or state.demand.executive >= EXECUTIVE_SUPPORT_THRESHOLD
        or state.demand.response >= RESPONSE_CONTROL_THRESHOLD
    ):
        state.executive_state = recruit_control_support(
            state.demand,
            current_alternatives=state.alternatives,
        )

    # 6. Exchange information without erasing which system supplied it.
    state.alternatives = generate_or_update_interpretations(
        literal=state.literal_proposition,
        conceptual=state.conceptual_features,
        social=state.agent_beliefs,
        context=context,
    )

    # 7. Integrate supported information and preserve uncertainty.
    state.integrated_representation, state.uncertainty = integrate_with_provenance(
        alternatives=state.alternatives,
        context=context,
        task=task,
    )

    # 8. Re-evaluate when later words or context change the task demands.
    while new_input_is_available():
        next_item = receive_next_input()
        state = update_incrementally(
            state=state,
            next_item=next_item,
            context=context,
            task=task,
        )

    return state
```

The functions above are deliberately abstract interfaces. They describe the operations that a computational model would need to implement; they are not claims that the brain executes Python-like procedures.

---

## 11.7 Evidence and limitations: what supports the separation and what remains debated

### 11.7.1 Evidence for a core language network

The 2024 review by Fedorenko, Ivanova, and Regev synthesizes evidence that a left-dominant frontotemporal network is strongly responsive to linguistic input, functionally interconnected, and dissociable from several non-linguistic systems. It describes this network as supporting access to words and constructions from memory and the combination of those representations into messages. The review also emphasizes that the language system interacts with sensory/motor systems and broader knowledge/reasoning systems without becoming identical to them.

### 11.7.2 Evidence for language/thought separability

Fedorenko and Varley (2016) reviewed neuroimaging and neurological patient evidence relevant to whether thought can occur without language. The preservation of some non-linguistic reasoning abilities in people with severe aphasia is important evidence that loss of language does not automatically eliminate all forms of thought. This does not establish that language never influences thought, learning, categorization, memory, or reasoning. It establishes a more limited and important point: many cognitive capacities are not reducible to the language system.

### 11.7.3 Distinct systems can still synchronize

Functional distinctions do not imply that networks are silent when not performing their preferred operation, nor that they are disconnected. Paunov, Blank, and Fedorenko (2019) reported that functionally distinct language and theory-of-mind networks can show synchronization at rest and during language comprehension. Synchronization is compatible with distinct functional specializations; it should not be interpreted as proof that the two networks implement the same computation.

### 11.7.4 Individualized functional connectivity in 2026

Shain and Fedorenko (2026) examined 1,957 fMRI sessions from 1,199 unique brains and reported that individualized functional-connectivity analyses could recover a network with properties similar to the task-localized language network. Their reported pattern was selective for language across the tested tasks, stable across task states, robust to parcellation choices, and variable across individuals while relatively stable within an individual. These findings strengthen the case for individualized network mapping, but they do not remove the need to test different populations, tasks, and methods.

### 11.7.5 Right-hemisphere and multiple-demand contributions are task-sensitive

A 2025 meta-analysis of 561 functional neuroimaging studies examined the role of right-hemisphere language-homologous regions and the multiple-demand network in verbal semantic processing. It reported stronger right-language-network involvement for sentence/narrative stimuli than for isolated words/word pairs, with right inferior frontal involvement related to semantic-control demands. It also reported multiple-demand recruitment when external task demands increased. This suggests that the division between core language, right-hemisphere contributions, and domain-general control must be evaluated against the specific task rather than asserted as an absolute rule.

### 11.7.6 Default-mode / conceptual integration remains an area of debate

A 2024 review on the default-mode network and semantic cognition discusses a possible role for this network in integrating distributed traces into richer situation models. This creates a useful caution: the distinction between a language-selective network and broader knowledge/reasoning systems does not mean that all semantic information is outside language-responsive cortex, or that one non-language system alone contains all conceptual knowledge. Semantic cognition is distributed, and the exact contribution of default-mode, temporal semantic, and language-selective regions depends on the task and theoretical framework.

### 11.7.7 The natural-kind framing is a theoretical interpretation

The phrase “natural kind” expresses the claim that the language network forms a coherent functional system with recurring properties across individuals and tasks. It should not be read as implying a perfectly uniform anatomical object or a set of immutable boundaries. Functional topography varies, network borders depend partly on measurement choices, and researchers continue to debate the extent of shared versus distinct resources for linguistic and conceptual processing.

---

## 11.8 Quality-control rules for implementing this architecture in a computational model

1. **Do not collapse all representations into one vector without provenance.** Preserve the distinction between word-derived content, conceptual knowledge, social inference, and task instructions.
2. **Do not route every sentence to every subsystem at full strength.** Use demand-sensitive recruitment.
3. **Do not treat task difficulty as synonymous with linguistic complexity.** Model memory load, response selection, speed pressure, and explicit judgment separately.
4. **Do not use one brain atlas as a universal individual map.** Support person-specific localization or mark the mapping as uncertain.
5. **Do not equate co-activation with identical function.** Distinct systems can be active together.
6. **Do not treat language/thought dissociation as proof of total independence.** Preserve the possibility of language influencing some forms of thought.
7. **Do not assume the right hemisphere is irrelevant.** Test narrative, discourse, social-content, prosodic, and semantic-control demands explicitly.
8. **Do not label all default-mode activation as theory of mind.** The default-mode network is functionally heterogeneous and can contribute to episodic, semantic, and situation-model processes.
9. **Do not claim that all semantic knowledge is stored in the language network.** Separate linguistic knowledge from general conceptual knowledge while allowing integration.
10. **Do not encode a single brain region as the sole implementation of a high-level process.** Use distributed population/network representations.
11. **Do not interpret an algorithm's mathematical convenience as evidence of biological reality.** Mark model equations as abstractions unless direct physiological evidence supports the operation.
12. **Do not accept a single task contrast as proof of selectivity.** Use multiple control tasks and converging measures.
13. **Do not compare verbal ToM with non-verbal reasoning without controlling language demands.** A written story inherently requires language comprehension.
14. **Do not discard uncertainty after integration.** Preserve alternative interpretations when the linguistic or contextual evidence remains ambiguous.
15. **Do not assume a fixed module schedule.** Allow parallel activation, recurrent feedback, and dynamic changes as new words and context arrive.

---

## 11.9 Source alignment and references

### Alignment with the supplied source file

This part formalizes the section titled **“Fedorenko et al. (2024): Language Network vs. Other Systems”**, including these claims and qualifications:

- a core left frontotemporal language network supports lexical and constructional knowledge and linguistic composition;
- the network is distinguished from lower-level perceptual/motor mechanisms;
- the network is also distinguished from higher-level systems of knowledge and reasoning, including social reasoning;
- language and reasoning/social systems can cooperate while remaining functionally separable;
- the network's exact anatomical boundaries and the degree of overlap with semantic/conceptual systems remain debated;
- the concept of a language network distinct from thought is supported by neuroimaging and patient dissociations but is not equivalent to a claim that language has no influence on cognition.

### Research references used to ground and refine the algorithm

1. **Fedorenko, E., Ivanova, A. A., & Regev, T. I. (2024).** The language network as a natural kind within the broader landscape of the human brain. *Nature Reviews Neuroscience, 25*, 289–312. https://doi.org/10.1038/s41583-024-00802-4. Core framework for the language network's selectivity, connectivity, and relation to perceptual and domain-general systems.
2. **Shain, C., & Fedorenko, E. (2026).** A language network in the individualized functional connectomes of 1199 human brains doing arbitrary tasks. *Nature Communications, 17*, 8180. https://doi.org/10.1038/s41467-026-75745-8. Individualized functional-connectivity evidence, using 1,957 sessions from 1,199 unique brains.
3. **Fedorenko, E., & Varley, R. (2016).** Language and thought are not the same thing: evidence from neuroimaging and neurological patients. *Annals of the New York Academy of Sciences, 1369*(1), 132–153. https://doi.org/10.1111/nyas.13046. Reviews evidence for dissociations between language and non-linguistic thought.
4. **Fedorenko, E., Piantadosi, S. T., & Gibson, E. A. F. (2024).** Language is primarily a tool for communication rather than thought. *Nature, 630*, 575–586. https://doi.org/10.1038/s41586-024-07522-w. A related perspective on language, communication, and cognition; its theoretical claims should be evaluated alongside competing views.
5. **Paunov, A. M., Blank, I. A., & Fedorenko, E. (2019).** Functionally distinct language and Theory of Mind networks are synchronized at rest and during language comprehension. *Journal of Neurophysiology, 121*, 1244–1265. https://doi.org/10.1152/jn.00619.2018. Relevant to the distinction between functional specialization and network interaction.
6. **Demirkan, E., & Branzi, F. M. (2025).** The role of the right language network and the multiple-demand network in verbal semantics: insights from an activation likelihood estimation meta-analysis of 561 functional neuroimaging studies. *Human Brain Mapping, 46*(18), e70415. https://doi.org/10.1002/hbm.70415. Refines the account of right-hemisphere and multiple-demand contributions across different verbal semantic task demands.
7. **“How Does the ‘Default Mode’ Network Contribute to Semantic Cognition?” (2024).** *Brain and Language, 252*, 105405. https://doi.org/10.1016/j.bandl.2024.105405. Reviews proposals concerning distributed semantic integration and situation-model construction.
8. **Mahowald, K., Ivanova, A. A., Blank, I., Kanwisher, N., Tenenbaum, J. B., & Fedorenko, E. (2024).** Dissociating language and thought in large language models. *Trends in Cognitive Sciences, 28*(6), 517–540. https://doi.org/10.1016/j.tics.2024.01.011. Useful as a conceptual comparison between formal linguistic competence and broader functional/cognitive competence; it is not direct evidence about neural implementation.

---

## 11.10 Completion and handoff

**Part 11 is a functional-separation and cooperation algorithm, not another serial stage in comprehension.** It specifies how linguistic operations can be distinguished from sensory encoding, conceptual knowledge, social reasoning, and executive control; how these systems can cooperate; and how experimental and modelling evidence can constrain the separation.

The central computational outcome is a coordinated representation in which:

- linguistic content is constructed by language-selective computations;
- conceptual features are supplied by broader knowledge systems;
- mental-state information is added when the communicative or narrative task requires it;
- executive control is recruited in proportion to task demands; and
- the source and uncertainty of each contribution remain traceable.

The central biological caution is equally important: this is a useful algorithmic reconstruction of a distributed, interactive brain architecture. It is not proof that neurons execute the equations or pseudocode literally, nor that functional boundaries are perfectly sharp.

**End of Part 11 — Fedorenko's Language Network vs. Other Brain Systems.**

# Part 12 — Methodological Considerations, Scientific Debates, and Model Selection

## 12.1 Scope and an important distinction

The source file next discusses methodological problems in the neuroscience of pragmatics and language comprehension, including task confounds, disagreement about ERP components, uncertainty about the boundaries between language and domain-general networks, individual variability, development, and cultural generalizability.

**This section is different from Parts 1–9.** It does not describe a further stage that the brain must execute each time it understands a sentence. It describes an algorithm for researchers to infer, test, and refine models of the brain's language-comprehension mechanisms. Therefore, the algorithm below is a *scientific model-evaluation procedure*, not a claim that the brain itself performs literature review, statistical regression, Bayesian model comparison, or experimental design during comprehension.

The distinction matters because the source asks for mathematical operations that may implement brain processes. In this methodological section, most equations are tools for evaluating candidate explanations, not equations known to be executed by neurons. A small final subsection identifies computational principles that could also have a biological analogue, while keeping that interpretation explicitly provisional.

## 12.2 Objective

Construct a defensible account of a candidate neural mechanism by testing whether it:

1. explains the relevant behavioral phenomenon;
2. predicts when and where neural activity should change;
3. survives controls for linguistic and non-linguistic confounds;
4. generalizes across participants, materials, tasks, and populations;
5. is consistent with different measurement methods;
6. distinguishes association from causal contribution; and
7. states uncertainty and competing explanations instead of forcing a single interpretation.

The algorithm accepts:

- a target phenomenon, such as resolving sarcasm, a causal gap, a pronoun, or an ambiguous sentence;
- candidate explanations, such as semantic integration, syntactic reanalysis, conflict monitoring, working-memory demand, or mental-state inference;
- behavioral data, such as accuracy, response times, confidence, and eye movements;
- neural data, such as fMRI, EEG/ERP, MEG, lesion measures, or perturbation results;
- participant-level variables, including language experience, age, cognitive abilities, and relevant clinical characteristics; and
- a preregistered analysis plan, where feasible.

The output is not simply a brain-region label. It is a ranked set of mechanistic explanations, the evidence supporting each one, the conditions under which each is supported, and the remaining uncertainty.

## 12.3 Core data structures

Represent a candidate mechanism as:

`H = (operation, predicted_information, predicted_timecourse, predicted_network, boundary_conditions)`

For example:

- `operation`: integrate a word with a discourse representation;
- `predicted_information`: semantic fit and discourse coherence;
- `predicted_timecourse`: an early context-sensitive response followed, under some conditions, by later revision;
- `predicted_network`: language-selective temporal/frontal regions, with additional systems only under specified inferential demands;
- `boundary_conditions`: the effect should depend on context and should not be reducible to difficulty, working-memory load, or emotional salience.

Represent each experimental condition as a feature vector:

`x = [linguistic_form, lexical_frequency, surprisal, syntactic_complexity, semantic_fit, pragmatic_fit, memory_load, social_inference, emotional_valence, familiarity, task_demand]`

These variables should be measured or manipulated independently whenever possible. If two variables always change together, their effects cannot be cleanly separated by statistical analysis alone.

For each participant `s`, preserve a separate record of:

- behavioral responses;
- neural measurements;
- anatomical and functional localization;
- demographic and language-background variables relevant to the research question;
- data-quality indicators; and
- missingness or exclusion reasons.

Do not replace participant-specific information with group averages before checking whether the group-level result is representative of individuals.

## 12.4 Detailed algorithmic steps

### Step 1 — Define the phenomenon as an observable target

**Purpose:** Prevent a broad label such as “pragmatics” or “semantic processing” from hiding several different computations.

1. Write the exact comprehension problem in operational terms. Examples include selecting the intended referent of a pronoun, detecting that a sentence conflicts with world knowledge, inferring an unstated causal link, or interpreting an utterance as an indirect request.
2. Specify the observable response that indicates successful comprehension. This may be an interpretation choice, a paraphrase, a response to a question, a reading-time pattern, or a behavioral judgment.
3. Specify what would count as an error, an unresolved interpretation, or a delayed decision.
4. Separate the target computation from the task used to measure it. Answering a multiple-choice question about sarcasm, for example, also involves reading the response options and selecting a response.

**Mathematical formalization:** Define a target variable `Y` and a set of explanatory variables `X`. The simplest predictive model is:

`P(Y | X, H)`

where `H` is a candidate mechanistic account. The model should predict the target outcome, not merely fit a label assigned after observing the data.

**Neuroscientific caution:** A behavioral response is evidence about comprehension, but does not by itself reveal which neural region or computation caused it.

### Step 2 — Generate multiple competing hypotheses

**Purpose:** Avoid explaining every neural response with the first plausible mechanism.

Create at least two or three candidate accounts that make distinguishable predictions. For an apparent pragmatic anomaly, possible candidates may include:

- `H1`: difficulty retrieving or integrating a lexical-semantic representation;
- `H2`: syntactic or referential reanalysis;
- `H3`: conflict between literal sentence content and discourse/world knowledge;
- `H4`: inference about the speaker's beliefs or intentions;
- `H5`: increased working-memory or general task demand.

For each candidate, state what result would support it and what result would weaken it. A useful hypothesis is falsifiable: it must be possible to observe data that count against it.

**Mathematical formalization:** Assign prior probabilities `P(H_i)` only when they are justified by prior evidence or a transparent modelling choice. After collecting data `D`, update them with Bayes' rule:

`P(H_i | D) = P(D | H_i) P(H_i) / sum_j[P(D | H_j) P(H_j)]`

The numerical posterior depends on the quality of the model and its assumptions. Bayesian notation does not make an unsupported hypothesis reliable.

### Step 3 — Build a causal diagram of confounds

**Purpose:** Identify variables that could produce the same behavioral or neural effect as the proposed computation.

Draw a directed acyclic graph or an equivalent causal diagram. Include:

- the intended experimental manipulation;
- linguistic variables such as lexical frequency, sentence length, and syntactic complexity;
- task variables such as response format and difficulty;
- participant variables such as age and language experience;
- behavioral outcomes; and
- neural measurements.

For example, a highly inferential sentence may also be longer, less frequent, emotionally stronger, and harder to remember. If all those features differ from the control sentence, increased frontal activity cannot be uniquely attributed to pragmatic inference.

Mark each variable as manipulated, measured, controlled, or unobserved. Distinguish confounders from mediators: controlling for a mediator can remove part of the very mechanism being studied.

**Mathematical formalization:** In a linear model, an omitted confounding variable `Z` can bias the estimated effect of manipulation `X` on outcome `Y` when `Z` is associated with both `X` and `Y`. A regression model can include measured covariates, but cannot repair unmeasured confounding automatically.

### Step 4 — Design contrasts that isolate one computation at a time

**Purpose:** Make the candidate explanations predict different outcomes.

1. Construct a baseline condition and one or more carefully matched comparison conditions.
2. Manipulate the target property while matching unrelated properties as closely as possible.
3. Use factorial designs when two factors are theoretically distinct. For example, independently manipulate discourse congruence and the need to infer a speaker's mental state.
4. Include control conditions for general difficulty, memory load, emotional content, and response selection where those could explain the effect.
5. Pilot the stimuli to ensure that the intended interpretations are understood as expected.
6. Randomize or counterbalance item order and condition assignment.

A factorial design can test whether a neural response is driven by one factor, another factor, or their interaction. If `A` is context congruence and `B` is mental-state inference, estimate:

`Y = beta_0 + beta_A A + beta_B B + beta_AB(A*B) + error`

The interaction coefficient `beta_AB` tests whether the effect of one manipulation changes depending on the other. It does not automatically identify a neural mechanism; interpretation still depends on design quality and model assumptions.

### Step 5 — Account for individual differences before aggregation

**Purpose:** Avoid treating a group-average brain map as if every participant has the same functional anatomy.

1. Localize functional language-responsive regions in each participant when the study design and data quality allow it.
2. Keep participant-level responses and network estimates available throughout the analysis.
3. Estimate whether the effect is consistent across participants or driven by a subset.
4. Examine relevant language experience, age, proficiency, and other prespecified moderators.
5. Report how localization uncertainty and missing data affect conclusions.

**Mathematical formalization:** Use a hierarchical model rather than assuming all participants share exactly the same coefficient:

`y_s = beta_0s + beta_1s*x_s + error_s`

`beta_1s ~ Normal(mu_beta, sigma_beta^2)`

Here, `mu_beta` estimates the population-level average and `sigma_beta` describes between-participant variability. Partial pooling can stabilize estimates while preserving individual differences. It does not eliminate the need to inspect the distribution of individual effects.

### Step 6 — Separate spatial, temporal, and functional claims

**Purpose:** Avoid confusing “where,” “when,” and “what computation” into a single inference.

- fMRI can help identify locations whose activity covaries with a condition, but its haemodynamic response is relatively slow.
- EEG/ERP and MEG can provide more precise timing information, but source localization and component interpretation have limitations.
- Functional connectivity describes statistical relationships between signals; it does not, by itself, demonstrate direct communication or causal direction.
- A region's activation does not prove that the region is necessary for a computation.

Analyse spatial localization, temporal dynamics, and functional interaction as related but distinct claims. When feasible, combine methods so that the strengths of one compensate for limitations of another.

**Mathematical formalization:**

- A general linear model for fMRI estimates condition-related signal changes from a design matrix and a haemodynamic response model.
- ERP analysis may estimate time-locked mean amplitude or latency in prespecified windows, with appropriate correction for multiple comparisons.
- MEG analyses may estimate time-resolved source activity or oscillatory features, with uncertainty in source reconstruction stated explicitly.
- Functional connectivity may use correlation, coherence, or model-based measures; each measure answers a different statistical question.

No one of these measurements uniquely identifies a biological computation.

### Step 7 — Interpret N400 and P600 patterns conservatively

**Purpose:** Prevent an ERP label from being treated as a direct readout of one mental operation.

For N400-like effects:

1. Test whether the effect depends on semantic expectation, lexical access, discourse context, or other controlled factors.
2. Inspect the timing, scalp distribution, and participant/item variability rather than using the component name alone.
3. Consider whether contextual support changes processing before the target word appears.
4. Do not equate every N400 modulation with a single isolated “semantic integration module.”

For P600 or late-positive effects:

1. Test whether the response is associated with syntactic revision, conflict, unexpectedness, task demands, pragmatic reinterpretation, or a combination.
2. Compare conditions in which these factors are independently varied.
3. Check whether behavioral evidence indicates reanalysis or a changed interpretation.
4. Avoid treating P600 as an exclusive marker of syntactic processing or of pragmatic processing.

**Mathematical formalization:** For each participant and item, estimate component amplitude `A`, latency `T`, and where justified a time-resolved response `r(t)`. Compare conditions with mixed-effects models that include participant and item variability. Correct for the number of tested time points, electrodes, regions, and contrasts.

**Key inference rule:** An ERP component is a measured pattern in electrical activity. Its cognitive interpretation is a hypothesis that must be supported by converging manipulations and behavioral evidence.

### Step 8 — Test whether the effect is language-selective or domain-general

**Purpose:** Determine whether the effect is tied to linguistic processing or can be explained by a more general demand.

1. Compare language conditions with appropriately matched non-linguistic tasks when this is relevant to the hypothesis.
2. Test whether the same region or network responds to memory, attention, control, social reasoning, or other non-language demands.
3. Estimate selectivity within individuals where possible, not only from group-level activation maps.
4. Distinguish language selectivity from absolute exclusivity. A region can be strongly tuned to language without being silent in every other condition.
5. Examine whether apparent overlap is caused by anatomical proximity, coarse spatial resolution, or genuinely shared computations.

**Mathematical formalization:** A selectivity contrast can be expressed as:

`S = response(language condition) - response(matched non-language condition)`

A normalized measure may divide by an estimate of response variability or overall activity, but the exact normalization must be justified. Classification accuracy or representational similarity can test whether neural patterns distinguish conditions; neither alone proves that the decoded information is used causally by the brain.

### Step 9 — Test network-boundary hypotheses without assuming sharp borders

**Purpose:** Represent the fact that language-selective, semantic, social, and executive systems can be functionally distinguishable while interacting.

1. Define candidate networks using independent localizers or connectivity data where possible.
2. Estimate within-network and between-network connectivity separately.
3. Test whether an area changes its coupling with other systems under a specific comprehension demand.
4. Evaluate alternative parcellations and individual network topographies.
5. Treat boundary regions, including areas sometimes assigned to both language and broader semantic or social functions, as uncertain rather than forcing them into a single category.

**Mathematical formalization:** Represent a network as a weighted graph `G=(V,E,W)`, where `V` denotes regions, `E` denotes estimated relationships, and `W` contains edge weights. Candidate measures include within-network connectivity, between-network connectivity, modularity, participation coefficient, and condition-dependent changes in edge weights.

Graph metrics depend on node definitions, thresholds, preprocessing, and the chosen connectivity measure. They are descriptive tools, not direct evidence that a brain region sends a particular message to another region.

### Step 10 — Distinguish correlation from causal contribution

**Purpose:** Avoid concluding that a region is necessary just because it activates during a task.

Use multiple forms of evidence where feasible:

- functional imaging to identify condition-related activity;
- lesion evidence to test whether damage is associated with a selective impairment;
- TMS or another suitable perturbation to test whether transiently altering a region changes performance;
- intracranial recordings or stimulation when clinically and ethically justified; and
- behavioral dissociations to determine whether two abilities can separate.

For lesion studies, account for lesion size, location, disconnection, reorganization, premorbid variation, and the limits of mapping a damaged brain onto a typical functional network. For perturbation studies, check whether the intervention reaches the intended target and whether any behavioral change is specific to the hypothesized computation.

**Mathematical formalization:** A randomized perturbation can estimate an average treatment effect:

`ATE = E[Y | intervention] - E[Y | control]`

This interpretation requires appropriate randomization, adherence, and measurement assumptions. A null effect does not necessarily prove that a region is irrelevant; compensation, inadequate stimulation, timing, and statistical power may matter.

### Step 11 — Use behavioral measures to constrain neural interpretation

**Purpose:** Ensure that neural changes explain comprehension rather than merely reflecting task difficulty or response execution.

Measure more than accuracy whenever feasible:

- response time;
- confidence;
- interpretation choice;
- eye movements or reading times;
- memory for explicit and inferred content;
- sensitivity to context; and
- whether participants can explain the intended meaning.

Separate comprehension from later decision or motor-response stages. For instance, a delayed button press may reflect response selection rather than delayed semantic access.

**Mathematical formalization:** Jointly model behavioral and neural observations where the data justify it. For example:

`P(behavior, neural_signal | H) = P(behavior | H) * P(neural_signal | behavior, H)`

This factorization is a modelling assumption, not a universal law. More elaborate models can represent shared latent variables, but they must be identifiable from the available observations.

### Step 12 — Test developmental, cultural, and language-background generalization

**Purpose:** Determine whether a proposed mechanism is robust or depends on the population and language used in the experiment.

1. State which populations the original evidence represents.
2. Test age and developmental stage where the research question concerns maturation.
3. Document language background, proficiency, literacy, dialect, and relevant cultural conventions.
4. Avoid treating a result obtained in one language or cultural context as universal without replication.
5. Use equivalent tasks only when their meanings and difficulty are genuinely comparable across groups.
6. Consider whether observed differences reflect strategy, experience, measurement bias, or a change in neural mechanism.

**Mathematical formalization:** Include prespecified moderators in hierarchical models and test measurement invariance before comparing group scores. If item difficulty differs across languages or populations, model item-level differences rather than interpreting all score differences as cognitive differences.

### Step 13 — Prevent overfitting and selective reporting

**Purpose:** Ensure that the explanation predicts new observations rather than merely describing the current dataset.

1. Preregister primary hypotheses, exclusions, outcome measures, and analysis choices where possible.
2. Separate confirmatory analyses from exploratory analyses.
3. Correct for multiple comparisons across electrodes, time windows, voxels, regions, and contrasts.
4. Use held-out participants, items, or datasets for out-of-sample evaluation when suitable.
5. Replicate critical effects with different stimuli and, where possible, independent laboratories or methods.
6. Report null, mixed, and contradictory findings rather than presenting only supportive contrasts.

**Mathematical formalization:** Use cross-validation to estimate out-of-sample predictive performance. Compare candidate models with appropriate likelihood-based, Bayesian, or information-criterion methods. Avoid choosing a model solely because it has more parameters and can fit the training data more closely.

### Step 14 — Compare models using explanatory coverage and simplicity

**Purpose:** Select the account that explains the evidence with the fewest unsupported assumptions, without automatically preferring oversimplified models.

For each candidate, score:

- behavioral fit;
- spatial predictions;
- temporal predictions;
- individual-level generalization;
- compatibility with lesion or perturbation evidence;
- ability to explain boundary conditions; and
- number and strength of additional assumptions.

**Mathematical formalization:** Candidate models may be compared using a Bayes factor:

`BF_12 = P(D | H_1) / P(D | H_2)`

or an information criterion such as AIC/BIC when their assumptions are appropriate. A model comparison is only as credible as the candidate set, data quality, and likelihood specification. If every candidate is poor, the algorithm must allow an “insufficient model” outcome rather than selecting the least-bad account as truth.

### Step 15 — Produce a calibrated conclusion

**Purpose:** Communicate what is supported, what is plausible, and what remains unknown.

Classify each proposed operation as one of:

- **Supported at the behavioral level:** reliable behavior demonstrates the relevant comprehension phenomenon.
- **Associated with a neural signal:** a region, network, or physiological measure covaries with the phenomenon.
- **Supported by converging evidence:** several distinct methods support a compatible functional interpretation.
- **Causally implicated under tested conditions:** perturbation or lesion evidence indicates that the region or pathway contributes causally, with limitations stated.
- **Mechanistically unresolved:** evidence does not yet distinguish alternative computations.

Avoid upgrading “associated” to “necessary,” or “consistent with” to “proves.”

The output should include a confidence estimate or qualitative confidence category and a list of the most informative next experiments.

## 12.5 Integrated research algorithm (pseudocode)

```text
ALGORITHM Evaluate_Neural_Model_of_Language_Comprehension

INPUT:
    target_phenomenon
    candidate_hypotheses
    behavioral_dataset
    neural_datasets
    participant_metadata
    experimental_design
    preregistered_analysis_plan

OUTPUT:
    ranked_hypotheses
    evidence_by_method
    confound_audit
    generalization_report
    uncertainty_report
    next_experiments

1. DEFINE target_phenomenon in observable behavioral terms.
2. DEFINE separate comprehension measures and task-response measures.
3. GENERATE multiple falsifiable candidate_hypotheses.
4. FOR each hypothesis:
       WRITE predicted computation, timing, network, and boundary conditions.
       WRITE observations that would weaken the hypothesis.
5. BUILD a causal diagram of manipulated, measured, and possible confounding variables.
6. AUDIT whether the experimental contrasts isolate the target computation.
7. IF major confounds are inseparable:
       MARK the mechanistic conclusion as underidentified.
       PROPOSE a better factorial or matched design.
8. PRESERVE participant-level and item-level data.
9. LOCALIZE functional regions individually where feasible.
10. ANALYZE behavior with participant and item variability represented.
11. ANALYZE fMRI, EEG/ERP, MEG, or other neural data using method-appropriate models.
12. EVALUATE N400/P600 or other neural effects as measured patterns,
    not as unique labels for cognitive operations.
13. TEST domain selectivity using suitable control tasks.
14. TEST within-network and between-network interaction.
15. COMPARE temporal, spatial, behavioral, and connectivity predictions.
16. EVALUATE lesion and perturbation evidence separately from correlational evidence.
17. TEST individual, developmental, cultural, and language-background moderators.
18. CORRECT for multiple comparisons and assess out-of-sample prediction.
19. COMPARE candidate hypotheses using justified model-comparison criteria.
20. IF all candidate models fit poorly:
       RETURN "No adequate model among the tested candidates."
21. OTHERWISE:
       RANK hypotheses by explanatory adequacy and robustness.
22. LABEL each claim as behavioral, correlational, convergent, causal,
    or unresolved.
23. REPORT assumptions, uncertainty, limitations, and contradictory evidence.
24. SELECT the next experiment with the highest expected ability
    to distinguish the remaining candidate hypotheses.
25. RETURN ranked_hypotheses and all supporting audit reports.

END ALGORITHM
```

## 12.6 Worked example: interpreting sarcasm without confusing it with general difficulty

Consider the utterance “Wonderful work!” spoken after an obvious mistake. The research question is whether listeners reinterpret the literal positive wording by integrating prosody, situation, and speaker intention.

**Candidate hypotheses**

- `H1`: prosodic cues change the interpretation during early linguistic/semantic processing;
- `H2`: the literal interpretation is computed first and then revised using discourse and speaker-intention inference;
- `H3`: the apparent neural effect reflects general conflict or working-memory demand rather than sarcasm-specific processing.

**Design**

Manipulate verbal content (positive/negative), prosody (approving/scolding), and situation (success/failure) independently. Include matched non-sarcastic conditions and a non-social conflict condition. Keep sentence length, lexical frequency, and task response format similar. Collect interpretation judgments, confidence, response time, and neural measures.

**Model**

For an interpretation response `Y`, one candidate logistic model is:

`logit P(Y=sarcastic) = b0 + b1*prosody + b2*situation + b3*prosody*situation + b4*lexical_valence + u_participant + u_item`

A neural analysis may model time-dependent responses to determine whether context and prosody influence early or later processing. A separate model can test whether any late activity is better predicted by mental-state inference, conflict, or memory load.

**Interpretation**

- If context/prosody changes interpretation reliably, that supports the behavioral claim that literal wording is not sufficient.
- If neural responses differ, this demonstrates condition-related neural association, not a unique sarcasm module.
- If an additional social-inference manipulation selectively changes a response after controlling for difficulty, that is evidence consistent with a mental-state contribution.
- If a non-social conflict condition produces the same effect, a general conflict account remains viable.
- Stronger causal claims require appropriate lesion or perturbation evidence and cannot be inferred from an fMRI contrast alone.

## 12.7 Worked example: testing language-network selectivity

Suppose a study reports left inferior frontal and temporal activation during a sentence-comprehension task. The question is whether this activity reflects language-selective processing or general task difficulty.

1. Use an independent language localizer to identify language-responsive regions for each participant.
2. Compare sentences with a matched non-linguistic task that controls response demands as closely as practical.
3. Include a difficulty manipulation within both language and non-language tasks.
4. Estimate language selectivity and difficulty sensitivity separately.
5. Check individual responses instead of relying only on group averages.
6. Examine whether the same pattern generalizes to new items or an independent sample.
7. Test whether the relevant region's activity or connectivity predicts language performance after controlling for general task difficulty.

A simplified model is:

`neural_response = beta0 + beta1*language + beta2*difficulty + beta3*language*difficulty + random_effects + error`

A reliable `beta1` under suitable matching supports language sensitivity. A reliable `beta2` indicates sensitivity to difficulty. Both may be present. The presence of a difficulty effect does not automatically make a region non-linguistic, and language sensitivity does not prove absolute exclusivity.

## 12.8 Worked example: interpreting individual variability

Suppose a group-average analysis suggests that a region is active during pragmatic inference, but individual results differ substantially.

1. Estimate each participant's effect and its uncertainty.
2. Plot the distribution rather than reporting only the mean.
3. Verify functional localization and data quality for participants with unusually high or low values.
4. Test prespecified moderators such as language proficiency or age.
5. Replicate the moderator in an independent sample.
6. Avoid post hoc subgroup claims unless they are clearly labelled exploratory.

A hierarchical model can estimate the population mean and participant variability simultaneously. If the between-person variance is large, the correct conclusion may be that the effect is heterogeneous, not that the group-average result describes every brain.

## 12.9 What mathematical operations belong to the brain, and what belongs to the researcher?

The following table prevents a central category error in computational neuroscience:

| Mathematical operation | Role in this part | Is it an established neural operation? |
|---|---|---|
| Linear regression / GLM | Estimate condition-related measurements | No; it is a researcher analysis method |
| Bayesian model comparison | Compare candidate scientific explanations | No; it is a model-evaluation method |
| Multiple-comparison correction | Control statistical false-positive risk | No; it is a research procedure |
| Cross-validation | Test generalization to held-out data | No; it is an evaluation procedure |
| Hierarchical modelling | Estimate population and individual variability | No; it is a statistical framework |
| Functional-connectivity estimation | Quantify relationships between measured signals | The statistic is an analysis method; biological interaction is real but not identical to the statistic |
| Prediction-error signalling | Candidate description of neural learning/inference processes | Plausible in several domains, but exact implementation depends on the circuit and task |
| Competition and inhibition | Candidate mechanism for selecting among representations | Biologically plausible and supported in multiple neural systems; not a universal explanation for all language effects |
| Gain modulation / attention | Candidate mechanism for changing processing priority | Biologically plausible; its precise role must be tested in each language task |

The first six entries should not be presented as equations the brain executes. They are used by scientists to decide whether a biological model is credible. Prediction error, competition, inhibition, and gain modulation are closer to candidate biological computations, but the source material does not establish a single universal circuit or equation for them.

## 12.10 Research decision rules and failure conditions

The algorithm should stop or downgrade its conclusion under any of these conditions:

- The critical experimental conditions differ on several uncontrolled dimensions.
- A neural result appears only after extensive exploratory analysis and fails replication.
- A group-average effect hides large individual differences that are relevant to the claim.
- The interpretation of an ERP component is not distinguished from competing interpretations.
- A functional-connectivity result is described as causal communication without a suitable causal test.
- A region is called language-selective without appropriate non-language comparisons.
- A lesion result is interpreted without considering disconnection or reorganization.
- A result from one language, population, or developmental group is generalized without evidence.
- Every candidate model fails its own predictions.

When a failure condition occurs, return a narrower statement, redesign the experiment, or mark the question unresolved. Do not fill a gap in the evidence with an invented neural mechanism.

## 12.11 Scientific principles preserved from the source

This algorithm translates the source's methodological cautions into explicit checks:

1. Literal and nonliteral sentence contrasts can differ in figurativeness, complexity, emotional content, and other dimensions; those differences can confound the result.
2. P600-like positivities have more than one proposed interpretation, including syntactic reanalysis, pragmatic reanalysis, and conflict-related accounts.
3. Frontal activation during difficult sentences may reflect working-memory or executive demand rather than a dedicated language computation.
4. Individual language-network topography varies, so group-averaged maps can obscure individual organization.
5. Pragmatic development, language proficiency, aging, and cultural experience can alter the use of context and the neural resources recruited.
6. Strong conclusions require converging behavioral, electrophysiological, imaging, lesion, and perturbation evidence where suitable.

## 12.12 Limitations

- There is no single statistical workflow that is optimal for every language-comprehension question.
- Statistical adjustment cannot rescue a fundamentally confounded experimental design.
- Neuroimaging measures are indirect and have different temporal and spatial limitations.
- Lesions and stimulation can affect networks rather than isolated regions.
- Model comparison can only compare the candidate models supplied; it cannot guarantee that the correct mechanism is among them.
- Cultural and linguistic diversity limits broad generalization from narrow samples.
- The equations in this part formalize scientific inference. They should not be inserted into a biologically inspired language algorithm as though they were literal neuronal calculations.

## 12.13 Completion and handoff

**Part 12 completes the source's methodological and debate-focused topic.** It contributes a rigorous algorithm for deciding how confidently a proposed neural computation can be claimed, and for distinguishing behavioral evidence, neural association, convergent evidence, and causal evidence.

The key result is a calibrated mechanistic account rather than a single asserted brain map. Where two explanations remain compatible with the evidence, both should remain in the model until a discriminating experiment is available.

**End of Part 12 — Methodological Considerations, Scientific Debates, and Model Selection.**
---

# Part 13 — Individual Differences and Developmental Changes in Language Comprehension

## 13.1 Scope and source alignment

This part translates the source's discussion of individual variability and development into a computational algorithm. The source makes three connected observations:

1. People differ in how strongly and efficiently they use linguistic context, social information, and background knowledge during comprehension.
2. Pragmatic integration and the use of other people's beliefs and intentions develop over time; younger children may initially favour more literal interpretations in situations where an older, more experienced comprehender can infer an unstated intention.
3. Second-language learners and people with differing verbal skills may use context and other cognitive resources differently. With aging, some people may compensate for slower lexical access by drawing more strongly on accumulated world knowledge and discourse context.

These are population-level tendencies and research hypotheses, not rules that apply to every person in a group. Age, language background, vocabulary, education, auditory or visual access, working memory, executive control, social experience, cultural conventions, and task design can interact. No single variable should be used as a substitute for measuring the actual process under study.

The algorithm below therefore does not assign a fixed processing profile based on an age label, a diagnosis, a first/second-language label, or a single vocabulary score. It estimates a person's current functional profile from multiple observations, preserves uncertainty, and adapts the computational model only where the evidence supports doing so.

## 13.2 Objective

Build an individualized, development-sensitive parameter profile that predicts how a comprehender processes words, sentences, discourse context, pragmatic cues, and speaker intentions. Use that profile to adjust the model's hypotheses about processing speed, cue reliability, contextual support, memory demands, prediction, reanalysis, and the recruitment of domain-general or social-cognitive resources.

The algorithm must satisfy five constraints:

- **Individualize rather than stereotype:** infer parameters from measured performance, not demographic category alone.
- **Separate component abilities:** lexical access, language knowledge, working memory, executive control, world knowledge, and pragmatic inference must not be collapsed into a single 'language ability' score.
- **Model change over time:** represent learning and development as gradual, experience-sensitive processes rather than abrupt switches at fixed ages.
- **Distinguish compensation from equivalence:** successful comprehension using stronger contextual support does not prove that every underlying process is identical across individuals.
- **Keep biological claims calibrated:** mathematical models below are computational descriptions. A fitted parameter is not automatically a literal neuronal quantity or a direct measurement of a brain region.

## 13.3 Inputs and outputs

### Inputs

1. A sequence of linguistic stimuli: spoken words, written words, sentences, or discourse.
2. Linguistic context: preceding words and sentences, local syntactic constraints, discourse topic, and communicative setting.
3. Participant-level measurements, when available:
   - age and developmental stage;
   - language history, proficiency, dominance, and exposure;
   - vocabulary breadth and depth;
   - word-recognition speed and accuracy;
   - auditory or visual access to the stimulus;
   - verbal and general working-memory measures;
   - inhibitory control and attentional measures;
   - world-knowledge and relevant domain-experience measures;
   - pragmatic, referential, and perspective-taking performance;
   - familiarity with the culture and conversational conventions represented in the task.
4. Trial-level measures such as response time, comprehension accuracy, eye movements, pupil response, EEG/ERP, or neuroimaging signals, depending on the experiment.
5. Uncertainty and measurement-quality estimates for every profile variable.

### Outputs

An individualized processing profile containing estimates and uncertainty for:

- lexical access efficiency;
- vocabulary and linguistic-knowledge coverage;
- phonological/orthographic processing reliability;
- use of local and global context;
- sensitivity to semantic and syntactic prediction;
- working-memory support requirements;
- susceptibility to interference and difficulty suppressing obsolete candidates;
- use of world knowledge;
- pragmatic and perspective-taking operations required by the task;
- processing strategy under low versus high contextual constraint;
- confidence and revision thresholds;
- expected variability across tasks and conditions.

The profile is used to configure the subsequent language-processing algorithms. It must not be treated as a permanent identity label. It is a task-sensitive estimate that can be revised with new evidence.

## 13.4 Core data structures

Represent the individual's current profile as a vector:

\[
\boldsymbol{\theta}_i = [L_i, V_i, P_i, W_i, I_i, K_i, T_i, C_i, R_i]
\]

where, for individual \(i\):

- \(L_i\): lexical retrieval efficiency;
- \(V_i\): vocabulary and linguistic-knowledge coverage;
- \(P_i\): perceptual reliability for the modality used;
- \(W_i\): working-memory support capacity or efficiency;
- \(I_i\): interference-control / inhibitory-control estimate;
- \(K_i\): relevant world-knowledge availability;
- \(T_i\): pragmatic and perspective-taking capability relevant to the task;
- \(C_i\): contextual-cue use or contextual facilitation;
- \(R_i\): reanalysis and revision efficiency.

These are not nine independent biological modules. They are separable modelling dimensions chosen to prevent one broad score from hiding different causes of success or difficulty. The values may be normalized scores, latent-variable estimates, or task-specific parameters. The uncertainty of each estimate must also be stored.

A trial record can be represented as:

```text
TRIAL = {
    stimulus_id,
    language,
    modality,
    context_constraint,
    target_word_predictability,
    syntactic_complexity,
    pragmatic_demand,
    response_time,
    accuracy,
    gaze_or_attention_measure,
    neural_measure,
    measurement_quality
}
```

A developmental/experience record should store continuous variables where possible:

```text
EXPERIENCE_PROFILE = {
    chronological_age,
    language_exposure_by_language,
    proficiency_by_language,
    education_and_literacy_experience,
    relevant_domain_experience,
    hearing_or_vision_measure,
    task_familiarity,
    testing_context
}
```

Chronological age is an informative descriptor, but it is not a mechanism. It should not stand in for lexical knowledge, sensory quality, cognitive control, or experience.

## 13.5 Detailed algorithmic steps

### Step 1 — Establish a task-specific baseline

Before estimating individual differences, define exactly what the participant must understand. A word-recognition task, a syntactically complex sentence, an indirect request, and a story about a character's mistaken belief place different demands on the system.

1. Identify the target operation: word recognition, semantic prediction, reference resolution, pragmatic inference, narrative integration, or another defined operation.
2. Specify the sensory modality and make the stimulus perceptually accessible.
3. Establish baseline performance in a condition with minimal contextual support.
4. Establish performance in a condition with strong, valid context.
5. Include control conditions that isolate sensory difficulty, linguistic complexity, and general task difficulty.
6. Measure both accuracy and processing cost. A fast response with poor accuracy is not equivalent to efficient comprehension.

A baseline can be expressed as a condition-specific vector:

\[
\mathbf{b}_i = [RT_{i,0}, Acc_{i,0}, ErrType_{i,0}, Neural_{i,0}]
\]

where the subscript 0 denotes the defined baseline condition. Response time should be interpreted with accuracy and task-specific measures rather than as a stand-alone ability score.

### Step 2 — Estimate component abilities separately

Collect multiple indicators for each process. For example, vocabulary knowledge should not be estimated solely from rapid word naming; working-memory support should not be inferred only from reading speed; pragmatic ability should not be inferred from a single sarcasm item.

1. Estimate word-form recognition from multiple items and, where possible, more than one modality.
2. Estimate vocabulary breadth and depth separately from the speed of accessing a known word.
3. Estimate working-memory performance using tasks that require retaining and manipulating relevant linguistic information.
4. Estimate interference control using tasks that distinguish suppressing a competitor from simply knowing the target.
5. Estimate contextual knowledge using familiar and unfamiliar topics, with care not to confuse lack of topic knowledge with a language deficit.
6. Estimate pragmatic comprehension using several communicative functions: reference, indirect requests, implicatures, irony, and belief-based interpretation where suitable.
7. Attach uncertainty to every estimate.

If multiple measures are noisy, use a latent-variable or hierarchical model instead of treating each raw score as a precise trait. A simple measurement model is:

\[
y_{ijm} = \lambda_{jm}\theta_{im} + \epsilon_{ijm}
\]

where \(y_{ijm}\) is measure \(j\) of component \(m\) for individual \(i\), \(\lambda_{jm}\) is the measure's loading on that component, \(\theta_{im}\) is the estimated component value, and \(\epsilon\) is measurement error. This is a statistical model used by the researcher or computational system; it is not proposed as a literal neural calculation.

### Step 3 — Characterize linguistic experience and proficiency

For each language used by the participant, estimate experience and proficiency rather than using a binary first-language/second-language flag.

1. Record age of acquisition and history of exposure, where known.
2. Estimate current proficiency in vocabulary, grammar, listening, reading, and comprehension.
3. Record the amount and recency of exposure and the contexts in which the language is used.
4. Estimate whether the target words and expressions are familiar, frequent, formulaic, or domain-specific for that individual.
5. When the task involves more than one language, test whether cue use and prediction differ by language.
6. Preserve language-specific parameters; do not assume that proficiency or prediction strategy transfers uniformly from one language to another.

A useful model may distinguish exposure from proficiency:

\[
Proficiency_{i,\ell} = f(Exposure_{i,\ell}, AgeOfAcquisition_{i,\ell}, Use_{i,\ell}, LearningHistory_{i,\ell}) + \epsilon
\]

where \(\ell\) indexes language. The function \(f\) must be estimated or selected for the task; the equation does not assert a universal linear relationship.

### Step 4 — Estimate how strongly context facilitates lexical access

Compare the processing of target words under weak and strong contextual constraint while matching the target words and sensory properties as carefully as possible.

1. Present targets in weakly constraining contexts.
2. Present matched targets in strongly constraining contexts.
3. Measure recognition accuracy, response time, eye movements, or other suitable markers.
4. Test whether the context benefit is reliable for the individual and whether it depends on vocabulary, working memory, or language proficiency.
5. Separate a genuine context benefit from a speed–accuracy trade-off.

A simple contextual-facilitation score can be defined as:

\[
CF_i = z(RT_{weak,i}) - z(RT_{strong,i})
\]

where larger positive values indicate a greater reduction in response time under strong context after standardization. Accuracy must be modelled alongside this score. One can instead fit a trial-level model that includes context strength, participant, target-word frequency, and their interactions.

This score describes behaviour. It does not directly measure the amount of contextual activity in a specific brain region.

### Step 5 — Estimate how the participant weights different cues

Comprehenders may rely on different combinations of acoustic, lexical, syntactic, semantic, discourse, visual, and social cues. Estimate cue use empirically.

1. Identify cues relevant to the task.
2. Manipulate cue reliability independently where possible.
3. Measure the effect of each cue on candidate interpretation and response.
4. Test for interactions: a cue may matter only when another cue is weak or ambiguous.
5. Estimate individual and language-specific cue weights.
6. Avoid assuming that a person's most-used cue is always the most reliable cue.

A computational cue-combination model is:

\[
Score(h) = \sum_{k=1}^{n} w_{ik} \, e_k(h)
\]

where \(h\) is a candidate interpretation, \(e_k(h)\) is evidence from cue \(k\), and \(w_{ik}\) is the estimated weight assigned by individual \(i\). Weights can be normalized with a softmax if the model requires relative candidate probabilities:

\[
P_i(h) = \frac{\exp(Score_i(h))}{\sum_{h'}\exp(Score_i(h'))}
\]

These are computational abstractions for cue integration. They do not establish that the brain explicitly computes this exact sum or probability distribution.

### Step 6 — Estimate context compensation separately from core linguistic efficiency

A person may understand a sentence successfully because context supports an uncertain word or relation. This can conceal reduced efficiency in a lower-level operation if only final comprehension accuracy is measured.

1. Compare isolated or weak-context word recognition with word recognition in sentence context.
2. Compare literal sentence interpretation with interpretation after adding relevant discourse.
3. Measure whether context improves accuracy, reduces time, or merely increases confidence.
4. Test whether the benefit depends on world knowledge or familiarity with the topic.
5. Examine whether strong context also increases false expectations when the sentence violates the prediction.
6. Retain separate estimates for the underlying component and the compensatory support it receives.

The model should keep two quantities distinct:

\[
CoreEfficiency_i \neq ContextBenefit_i
\]

A high context benefit does not imply poor core linguistic ability; a low context benefit does not prove that the person lacks context sensitivity. Both require task-specific evidence.

### Step 7 — Model developmental change as a trajectory, not a threshold

Children's vocabulary, grammar, discourse knowledge, pragmatic skills, social-cognitive abilities, and executive functions change through development and experience. Their rates of change differ, and skills can be uneven.

1. Measure the component being studied rather than assigning a fixed ability from age alone.
2. Include age-appropriate tasks that minimize confounds from reading ability, attention, or familiarity.
3. Distinguish language knowledge from the ability to infer a speaker's belief or communicative intention.
4. Examine whether a child can make a pragmatic inference with explicit contextual support even when spontaneous inference is not yet reliable.
5. Model within-person change over repeated measurements when possible.
6. Include linguistic environment, conversational experience, schooling, and cultural conventions as explanatory variables where relevant.

A developmental trajectory can be represented as:

\[
\theta_{im}(t) = f_m(Age_i(t), Experience_i(t), Exposure_i(t)) + u_{im} + \epsilon_{imt}
\]

where \(m\) denotes a component, \(t\) is measurement time, \(u_{im}\) is an individual-specific deviation, and \(\epsilon\) is measurement error. A spline, nonlinear growth curve, or other model may be more appropriate than a straight line. The biological system does not have to follow the chosen statistical curve; the curve is a summary of observed development.

### Step 8 — Model pragmatic and perspective-taking development conditionally

Do not treat all pragmatic meaning as equally dependent on theory of mind. Understanding a conventional indirect request, resolving a pronoun, detecting sarcasm, and reasoning about a character's false belief can rely on partially different information.

1. Identify which pragmatic operation the task requires.
2. Determine whether the relevant information is linguistic, conventional, situational, emotional, or belief-based.
3. Test the operation with and without explicit perspective information.
4. Measure the role of executive control and working memory in addition to social-cognitive measures.
5. Distinguish failure to infer an intention from failure to understand the sentence's literal content.
6. Preserve separate parameter estimates for different pragmatic tasks.

One conceptual model is:

\[
PragmaticSuccess_i = g(LinguisticKnowledge_i, Context_i, ToM_i, Executive_i, Experience_i)
\]

The function \(g\) is task-dependent. It should not be assumed that all pragmatic comprehension requires the same level of mental-state reasoning.

### Step 9 — Model second-language and multilingual processing without a single 'L2 penalty'

A later-learned language can differ from a first language in lexical accessibility, cue reliability, prediction, vocabulary depth, and experience. These differences are heterogeneous and depend on proficiency, exposure, language pair, and task.

1. Estimate the participant's proficiency separately for each language.
2. Match or model target-word frequency and familiarity within each language.
3. Determine whether the person uses semantic, syntactic, discourse, or form-based cues differently across languages.
4. Measure working memory and inhibitory control when the task requires revising an initial prediction.
5. Compare first- and later-learned language performance within the same participant where possible.
6. Do not assume that slower processing automatically means less accurate comprehension or absent prediction.
7. Update the profile with evidence from both comprehension and learning history.

A mixed-effects model may be used to estimate group and individual factors:

\[
Y_{ij} = \beta_0 + \beta_1 Language_{ij} + \beta_2 Proficiency_i + \beta_3 Context_{ij} + \beta_4 WM_i + \beta_5(Language\times Context)_{ij} + u_i + v_j + \epsilon_{ij}
\]

where \(u_i\) represents participant-level variation and \(v_j\) item-level variation. This model tests whether context effects differ by language and proficiency while acknowledging that individuals and items differ. It is a research/statistical model, not a neural equation.

### Step 10 — Model healthy aging as a change in component balance

The source notes that older adults may rely more on world knowledge and context to compensate for slower lexical access. The algorithm must treat compensation and interference as separate possibilities.

1. Measure auditory or visual access to the stimulus so sensory change is not misclassified as a language deficit.
2. Measure lexical access speed and accuracy separately from vocabulary knowledge.
3. Estimate the availability of accumulated semantic and world knowledge.
4. Test contextual facilitation under both high- and low-constraint conditions.
5. Measure whether previously activated or irrelevant candidates remain active and interfere with the current interpretation.
6. Test working-memory and inhibitory-control demands rather than assuming that all age-related differences are caused by one general decline.
7. Determine whether the person uses context effectively, inefficiently, or both depending on the task.

A compensatory-context model can be expressed as:

\[
Performance_i = \alpha + \beta_L L_i + \beta_C C_i + \beta_K K_i + \beta_{LC}(L_i\times C_i) + \epsilon_i
\]

where \(L_i\) is lexical efficiency, \(C_i\) contextual support, and \(K_i\) relevant knowledge. An age-related difference in \(L_i\) may be partly offset by stronger \(C_i\) or \(K_i\) in some tasks. A separate interference term should be included if the data support it:

\[
Performance_i^* = Performance_i - \gamma J_i
\]

where \(J_i\) is interference from irrelevant or previously activated information. The signs and magnitudes of these relationships must be estimated, not assumed.

### Step 11 — Estimate memory and executive support requirements

Some tasks place heavier demands on keeping earlier sentence material active, inhibiting an attractive but incorrect interpretation, or revising an initial prediction.

1. Vary sentence length and dependency distance.
2. Vary the number of competing referents or interpretations.
3. Test whether a context effect changes as memory load rises.
4. Determine whether reanalysis costs are related to inhibitory control or to linguistic knowledge.
5. Estimate the conditions under which the participant benefits from additional time or repetition.
6. Preserve independent estimates for working memory and inhibitory control because they need not covary perfectly.

A simple resource-demand model is:

\[
Demand = a_1 Length + a_2 DependencyLoad + a_3 CompetitorCount + a_4 RevisionDemand
\]

The task's demand can be compared with an estimated support capacity \(W_i\). A model may predict increased errors or latency when demand exceeds available support, but a hard threshold should be used only if supported by data.

### Step 12 — Separate stable differences from momentary state

Performance can change because of fatigue, attention, stress, motivation, hearing conditions, unfamiliarity, or temporary distraction. Do not automatically treat a single low score as a stable characteristic.

1. Repeat critical measures across trials or sessions.
2. Record stimulus quality and testing conditions.
3. Compare within-person variability with between-person variability.
4. Use trial-level state variables where available.
5. Update the individual's profile gradually rather than replacing it after one unusual observation.

One state-space description is:

\[
\theta_i(t+1) = \theta_i(t) + \eta_i(t)
\]

\[
y_i(t) = h(\theta_i(t), Stimulus_t, State_t) + \epsilon_i(t)
\]

where \(\eta\) represents slow change or learning and \(State_t\) represents temporary conditions. This is a modelling device for separating persistent and transient variation, not a claim that every component changes at the same rate.

### Step 13 — Estimate individual and group effects simultaneously

A group-average result may conceal important differences. Use hierarchical analysis when the data support it.

1. Estimate an overall population effect.
2. Estimate variation between participants.
3. Test prespecified moderators such as proficiency, age, context strength, or working memory.
4. Examine whether the moderator affects accuracy, response time, neural response, or more than one measure.
5. Use uncertainty intervals and avoid presenting small differences as established traits.
6. Replicate subgroup effects with independent data.

A hierarchical model is:

\[
\theta_i \sim \mathcal{N}(\mu_{group}, \Sigma_{group})
\]

\[
Y_{ij} = X_{ij}\beta + Z_{ij}b_i + \epsilon_{ij}
\]

where \(\mu_{group}\) describes the group-level tendency and \(\Sigma_{group}\) describes individual variation. Partial pooling can stabilize estimates for participants with fewer observations. The model should not be used to erase meaningful heterogeneity.

### Step 14 — Identify strategy shifts and compensatory pathways

The same successful answer can result from different processing strategies. One participant may retrieve a word rapidly; another may use a strong sentence context; a third may use visual scene information or world knowledge.

1. Compare performance under conditions that selectively weaken one cue at a time.
2. Observe whether the participant changes strategy when the usual cue becomes unreliable.
3. Identify whether a stronger context benefit is accompanied by greater false prediction when context is misleading.
4. Compare the participant's processing profile across familiar and unfamiliar topics.
5. Determine whether compensation is robust or narrowly tied to a particular task.
6. Record alternative pathways without claiming that they are anatomically separate routes unless supported by neural evidence.

One way to compare candidate strategies is to calculate their predictive fit to held-out trials:

\[
Score(Model_m) = -Loss_{heldout}(Model_m) - \lambda Complexity(Model_m)
\]

The penalty term discourages unnecessary complexity. The model with the best score is not automatically the true biological mechanism; it is the best-supported candidate among those tested.

### Step 15 — Adapt downstream processing parameters

Only after the profile has been estimated should the comprehension system alter its computational settings.

Examples:

- If perceptual reliability is low, retain more alternative word-form hypotheses rather than prematurely selecting one.
- If lexical access is slower but contextual prediction is reliable, allow context to support candidate activation while preserving the possibility of a surprising word.
- If vocabulary is limited in a particular language, avoid treating low activation of a word as evidence that the sentence is semantically anomalous.
- If a sentence creates high memory demand, preserve important referents and relations in the working-memory representation.
- If inhibition or revision is costly, maintain a trace of the previous interpretation and explicitly test whether new evidence invalidates it.
- If a pragmatic inference depends on speaker beliefs, recruit the relevant perspective-taking computation only when task evidence calls for it.

The adaptation must be reversible. New evidence can update the parameter profile; the model must not permanently lock into a strategy based on a small sample.

### Step 16 — Evaluate calibration and generalization

An individualized model should predict more than the data used to fit it.

1. Hold out trials, words, contexts, or sessions.
2. Test whether predicted context benefits match observed context benefits.
3. Test whether parameters transfer to new topics and sentence structures.
4. Test whether language-specific parameters generalize to new items in the same language.
5. Examine calibration: when the model reports 80% confidence, does it succeed at approximately that rate over suitable trials?
6. Re-estimate the model if it is systematically overconfident or underconfident.
7. Mark parameters as task-specific if they fail to transfer.

A calibration error can be summarized as:

\[
CE = \sum_{b=1}^{B}\frac{|S_b|}{N}\left|acc(S_b)-conf(S_b)\right|
\]

where \(S_b\) is a confidence bin, \(acc\) is observed accuracy, and \(conf\) is mean predicted confidence. This is a model-evaluation statistic, not a neural calculation.

## 13.6 Integrated individualized-processing algorithm

```text
ALGORITHM INDIVIDUALIZED_LANGUAGE_COMPREHENSION(
    participant,
    language_stimulus,
    discourse_context,
    task_requirements,
    observations
):

    1. Define the target comprehension operation and task demands.
    2. Verify sensory accessibility and record modality-specific reliability.
    3. Establish matched low-context and high-context baselines.
    4. Estimate lexical access, vocabulary, syntax, working memory,
       inhibitory control, world knowledge, and pragmatic skills separately.
    5. Estimate language exposure and proficiency for each language.
    6. Estimate individual cue weights and their uncertainty.
    7. Measure the context benefit without confusing it with core efficiency.
    8. If developmental data exist, fit component-specific trajectories.
    9. If pragmatic inference is required, identify the relevant type of
       perspective-taking and executive demand.
   10. If multiple languages are involved, preserve language-specific parameters.
   11. If aging is relevant, test both contextual compensation and interference.
   12. Separate stable participant parameters from temporary state effects.
   13. Estimate group and participant variation using suitable hierarchical models.
   14. Compare alternative processing strategies on held-out observations.
   15. Adapt downstream lexical, predictive, memory, and inference parameters.
   16. Preserve uncertainty and alternative hypotheses.
   17. Evaluate calibration and generalization.
   18. Update the profile only when new evidence supports a change.

    RETURN individualized_profile,
           adapted_processing_parameters,
           uncertainty_estimates,
           unresolved_hypotheses
```

## 13.7 Worked example A — A child interpreting an indirect request

A child hears an adult say, "It's cold in here," while a window is open.

1. The auditory system recognizes the words and their order.
2. Lexical and syntactic processing supplies the literal proposition that the temperature is low.
3. The child integrates the utterance with the visible window and the conversational situation.
4. The task may require inferring that the adult wants the window closed.
5. The algorithm measures whether the child recognizes the literal meaning, uses the situation, and infers the indirect request.
6. If the child does not infer the request, the model tests whether the difficulty is due to missing vocabulary, unfamiliarity with the convention, weak contextual integration, or difficulty representing the speaker's intention.
7. It then compares performance when the adult adds a contextual cue, such as looking toward the open window.
8. The profile preserves the distinction between understanding the sentence and inferring its communicative function.

The model must not conclude from one missed indirect request that the child lacks theory of mind. More tasks and controls are needed to identify the relevant factor.

## 13.8 Worked example B — A later-learned language and contextual prediction

A proficient but non-native English reader encounters a sentence whose next noun is strongly constrained by the preceding verb and discourse.

1. The algorithm estimates the reader's proficiency and familiarity with the target noun.
2. It records the sentence context and the target's predictability.
3. It estimates whether context facilitates the target more strongly in this language than in the participant's first language.
4. It includes verbal working memory and inhibitory control if the task requires revising a prediction.
5. It measures reading time, accuracy, and rereading rather than using response speed alone.
6. It tests whether the profile generalizes to new nouns and new sentence frames.
7. It adapts candidate activation and uncertainty without assuming that all later-learned language processing is less predictive.

Different cue weights or slower lexical access may coexist with strong contextual use. The algorithm therefore models them separately.

## 13.9 Worked example C — Context use in healthy aging

An older adult listens to a sentence in a noisy environment. The final word is partially masked, but the preceding words and the topic strongly constrain what is likely.

1. Measure audibility and speech perception first.
2. Estimate the lexical recognition of the partially masked word with and without context.
3. Measure the person's relevant vocabulary and world knowledge.
4. Measure whether contextual information increases the target's activation and improves final comprehension.
5. Include plausible but irrelevant referents to test interference.
6. Determine whether context compensates for perceptual or lexical difficulty and whether competing information remains active.
7. Retain both results: contextual support may be beneficial, while interference may still impose a cost.

The algorithm should not infer a uniform aging effect. The outcome depends on sensory quality, task structure, knowledge, attention, and the specific participant.

## 13.10 Mathematical operations: neural candidate versus research model

| Operation | Use in the algorithm | Interpretation |
|---|---|---|
| Nonlinear growth curves | Describe developmental trajectories | Statistical description of change; not a literal neuronal equation |
| Hierarchical Bayesian or mixed-effects estimation | Separate group tendencies from individual differences | Research model for heterogeneous data |
| Cue-weighted evidence integration | Represent how multiple cues influence interpretation | Computational hypothesis; exact biological implementation is unresolved |
| Contextual facilitation score | Quantify behavioural benefit from context | Measurement statistic, not a brain computation |
| State-space update | Separate stable profile from temporary state | Dynamic modelling abstraction |
| Competition and inhibition | Represent competing interpretations and interference | Biologically plausible operations, but task-specific implementation must be tested |
| Predictive activation | Represent pre-activation of likely words or features | Supported as a processing phenomenon in some tasks; exact neural form is debated |
| Synaptic plasticity | Candidate mechanism for experience-dependent change | Biological process; the equations used to model it are approximations |
| Confidence calibration | Evaluate whether model confidence matches accuracy | Computational evaluation, not a direct neural calculation |

A central rule is to avoid converting every useful equation into a proposed brain operation. Regression, latent-variable modelling, confidence calibration, and model selection belong to the analyst or artificial system. Biological claims require independent evidence.

## 13.11 Evidence and research cautions

The source's developmental and individual-difference discussion is consistent with a broader research picture, but the algorithm should preserve heterogeneity:

- Research on healthy aging reports that older adults can use semantic context and accumulated world knowledge to support prediction, while also showing greater activation of some irrelevant or previously mentioned information in certain tasks. This supports modelling compensation and interference separately rather than treating context reliance as uniformly beneficial.
- Reviews of first- and later-learned language processing report that prediction and cue weighting can differ across languages and individuals. Proficiency, working memory, target-word knowledge, and task demands can all affect the observed pattern.
- Studies of children show that pragmatic comprehension, theory-of-mind reasoning, and executive functions develop and interact; development should therefore be represented as a set of related but non-identical trajectories.
- Language development is shaped by the linguistic and social environments in which children communicate. The model should not reduce development to age alone.

### Selected research grounding

1. Altmann, L. J. P., & colleagues (2023 issue; online research published 2021), *Increased reliance on world knowledge during language comprehension in healthy aging: Evidence from verb-argument prediction*. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/34353231/) and [open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC8818061/).
2. Schlenter, J. (2023), *Prediction in bilingual sentence processing: How prediction differs in a later learned language from a first language*. [Cambridge University Press](https://doi.org/10.1017/S1366728922000736).
3. Kosaka, T. (2025), *Higher reading spans mitigate context effects on lexical processing by low-proficiency L2 learners: A self-paced reading study*. [Springer](https://doi.org/10.1007/s11145-024-10583-y).
4. *Pragmatic comprehension and inferential ability: Interplay with executive functions and Theory of Mind* (2025), [Royal Society / PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC12351312/).
5. *Language Development in Context* (Annual Review of Developmental Psychology), [Annual Reviews](https://www.annualreviews.org/content/journals/10.1146/annurev-devpsych-042220-121816).
6. *Inferential evaluation and revision in L1 and L2 text comprehension: An eye movement study*, [Cambridge University Press](https://www.cambridge.org/core/journals/bilingualism-language-and-cognition/article/inferential-evaluation-and-revision-in-l1-and-l2-text-comprehension-an-eye-movement-study/6ECA12E7E048BC7349156D4BC36F67DC).

## 13.12 Limitations and safeguards

1. **No deterministic age rules:** do not assign a processing strategy solely from chronological age.
2. **No single proficiency label:** later-learned language processing varies with proficiency, exposure, linguistic distance, task, and individual characteristics.
3. **No single pragmatic score:** irony, indirect requests, reference, implicature, and belief inference are not interchangeable tasks.
4. **No inference from group average to every person:** report variability and uncertainty.
5. **No conflation of compensation and impairment:** successful contextual support may coexist with different underlying processing costs; neither should be assumed without evidence.
6. **No neural localization from a behavioural parameter:** a context-benefit score does not identify mPFC, TPJ, IFG, or any other single region.
7. **No causal claim from correlation:** individual differences in brain activation or connectivity require convergent evidence before being interpreted as causal mechanisms.
8. **No cultural universalization:** pragmatic conventions and the meaning of contextual cues can vary across communities and languages.
9. **No fixed model parameters:** profiles should be updated when new evidence, learning, or task conditions warrant it.
10. **No false precision:** if a component cannot be measured reliably, preserve a broad uncertainty interval rather than fabricating a precise parameter.

## 13.13 Completion and handoff

Part 13 adds an individualized and developmental layer to the earlier language-comprehension algorithms. It does not replace the core processing stages. Instead, it changes how their parameters are estimated and how much support is expected from context, world knowledge, working memory, executive control, and pragmatic reasoning.

The central computational idea is **adaptive parameterization under uncertainty**: use measured evidence about the current comprehender to tune the processing model, preserve alternative explanations, and revise the profile over time. The central biological caution is that these estimated parameters are functional descriptions, not direct labels for neural modules or literal neuronal calculations.

**End of Part 13 — Individual Differences and Developmental Changes in Language Comprehension.**
