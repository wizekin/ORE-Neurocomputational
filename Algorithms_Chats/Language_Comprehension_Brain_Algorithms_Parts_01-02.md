# Language Comprehension in the Human Brain — Algorithmic Reconstruction

## Working document: Parts 1–2 of a sequential series

**Source document:** `Language Comprehension_ From Words to Contextual Meaning.txt` (Library file supplied by the user)  
**Coverage in this cumulative file:** Parts 1 and 2 — *perceptual input is converted into preliminary spoken or written word forms; those forms then activate lexical entries and word-level semantic knowledge*.  
**Status:** Part 1 complete. Later source points are intentionally not included yet.

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
