# Language Comprehension in the Human Brain — Algorithmic Reconstruction

## Working document: Parts 1–3 of a sequential series

**Source document:** `Language Comprehension_ From Words to Contextual Meaning.txt` (Library file supplied by the user)  
**Coverage in this cumulative file:** Parts 1–3 — *perceptual input is converted into preliminary spoken or written word forms; those forms activate lexical entries and word-level semantic knowledge; retrieved lexical items are organized into an incremental grammatical structure*.  
**Status:** Parts 1, 2, and 3 complete. Part 4 and later source points are intentionally not included yet.

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
