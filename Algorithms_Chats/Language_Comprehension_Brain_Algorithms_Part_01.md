# Language Comprehension in the Human Brain — Algorithmic Reconstruction

## Working document: Part 1 of a sequential series

**Source document:** `Language Comprehension_ From Words to Contextual Meaning.txt` (Library file supplied by the user)  
**Coverage in this installment:** the first processing point only — *perceptual input is converted into preliminary spoken or written word-form representations*.  
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
