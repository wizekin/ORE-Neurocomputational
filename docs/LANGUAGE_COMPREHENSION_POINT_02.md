# Language Comprehension — Point 2: Lexical Access

## Scope and handoff

This package implements Point 2 from
\`Language_Comprehension_Brain_Algorithms_Parts_01-15.md\`: retrieve word-level lexical
entries, semantic features, grammatical information and possible senses from the uncertain
spoken or written word-form hypotheses produced by Point 1.

\`Point2LexicalAccessProcessor\` exposes the source's 12 ordered operations
(\`LC-P02-01\`–\`LC-P02-12\`). It reuses \`ore_lexical_access\` for the lower-level seven-stage,
36-operation lexical identity, meaning, grammatical-category, morphology, incremental-activation
and LIF reference computations (\`BLA-001\`–\`BLA-036\`). The adapter keeps the new source-level
trace separate from the underlying BLA trace.

Point 2 is a lexical stage, not a sentence parser. Its output can include word meanings,
lexical categories, expressed grammatical/morphological features, and stored argument-taking
properties, but it does not assign actual sentence participants, build syntactic dependencies,
compose a proposition, or infer speaker intention.

## Source-level operation sequence

| ID | Operation | Main mathematical operations | Biological reference |
|---|---|---|---|
| LC-P02-01 | Accept graded form evidence | Feature reliability weighting, alternative distributions, uncertainty and boundary descriptors | Graded sensory-language population activity and recurrent feedback |
| LC-P02-02 | Select auditory/visual routes | Modality-specific transformations and likelihoods; explicit shared-entry convergence | Auditory temporal speech pathway and ventral occipitotemporal orthographic pathway |
| LC-P02-03 | Generate lexical candidates | Weighted feature matching, sequence/edit distance, position-sensitive evidence and permissive thresholding | Partial co-activation of overlapping phonological/orthographic populations |
| LC-P02-04 | Add familiarity and frequency | Log-frequency prior, exposure/readiness update, decaying recent-activation trace | Experience-dependent plasticity and distributed pattern readiness |
| LC-P02-05 | Add local expectations | Bayesian-style cue combination, weighted evidence score, reliability-dependent context gain | Top-down temporal/frontal feedback; no broad pragmatic inference |
| LC-P02-06 | Activate morphology | Segmentation search, reconstruction check, stem/affix retrieval, whole-form/decomposition competition | Distributed learned word-form and morpheme-family representations |
| LC-P02-07 | Resolve competition gradually | Recurrent excitation, lateral inhibition, sigmoid/softmax, entropy and evidence accumulation | Recurrent and inhibitory population interactions; control is demand-dependent |
| LC-P02-08 | Retrieve semantic features | Associative spread, nonlinear feature activation, sparse distributed semantic vectors | Distributed temporal and modality-linked conceptual networks; ATL-style convergence is one model |
| LC-P02-09 | Retrieve lexical properties | Attribute-vector activation and conditional feature support; soft constraints for later parsing | Temporal/frontal exchange of lexical and grammatical information |
| LC-P02-10 | Preserve homonym/polysemy ambiguity | Within-entry sense posterior, between-entry identity competition, top-k retention and margin | Overlapping related-sense populations and distinct lexical/concept patterns |
| LC-P02-11 | Reconcile with new sensory evidence | Incremental score update, mismatch/prediction error and confidence/margin stopping gate | Recurrent form-to-lexical feedback and conditional selection |
| LC-P02-12 | Package for syntactic parsing | Candidate normalization, confidence/margin reporting and structured lexical record assembly | Distributed lexical-semantic activation made available to subsequent parsing |

## Functional architecture

### Visual route

Point 1's position-ordered grapheme candidates are beam-composed into candidate orthographic forms.
The candidate generator retains the originating glyph IDs, region, order and confidence. Each
visual form is compared with the caller-supplied lexical inventory using the existing lexical
engine's Unicode-normalized form lookup, bounded edit-distance search, orthographic similarity,
stored frequency and familiarity terms.

A missing or unlabelled grapheme is not replaced with an invented character. The route requires
Point 1 to supply grapheme labels (normally via caller-supplied prototypes).

### Auditory route

Point 1's time-ordered phonetic segments and their alternatives are beam-composed into candidate
phoneme sequences. They are matched only against pronunciations explicitly stored in
\`LexicalEntry.phonemes\`. The implementation deliberately does not guess a pronunciation from
spelling or use an unstated grapheme-to-phoneme rule. A lexical entry without stored phonemes
remains available to the visual route but is not fabricated as an auditory match.

Both routes can converge on the same lexical entry ID when the same caller-supplied entry contains
the relevant orthography and pronunciation. The route evidence remains visible in the output.

### Lexical inventory and semantic knowledge

The processor consumes the existing \`ore_lexical_context.LexicalEntry\`,
\`LexicalMeaning\`, and \`MorphemeEntry\` records. It does not download or invent a lexicon.
A meaning candidate carries its feature map, prior, gloss, verifier and source reference.
A lexical candidate carries the entry ID, word form, phonological form if supplied, lexical
category, frequency/exposure, morphological alternatives, stem/affix retrieval and any additional
caller-supplied lexical properties.

The optional \`lexical_properties\` mapping is keyed by a supplied entry ID. It can hold
\`grammatical_features\`, \`morphological_features\`, \`argument_structure\`, and
\`complement_properties\`. These are lexical attributes for future syntax, not parsed sentence roles.

## Scoring and mathematical operations

The adapter composes multiple evidence channels. A simplified candidate score is:

\[
S_i = \log P(x\mid w_i)
      + \log P_{\mathrm{route}}(w_i)
      + \alpha\log(1+f_i)
      + \beta(1-r_x)C_i
      + \delta R_i
      + \log(a_i+\epsilon)
\]

where:

- \(P(x\mid w_i)\) is the spoken or written form likelihood;
- \(P_{\mathrm{route}}\) is the relative candidate support within the modality route;
- \(f_i\) is the supplied lexical frequency/exposure estimate;
- \(r_x\) is form reliability, so context contribution is restricted when the sensory form is clear;
- \(C_i\) is cosine compatibility between the supplied local context and stored meaning features;
- \(R_i\) is inventory-supported morphological support;
- \(a_i\) is the final incremental lexical activation from the existing engine.

Candidate scores are converted into relative posteriors with a temperature-controlled softmax:

\[
p_i = \frac{\exp(S_i/T)}{\sum_j\exp(S_j/T)}
\]

Within one lexical entry, possible senses are ranked separately using their stored priors and
available local semantic-feature compatibility. This maintains the distinction between:

- **Homonymy / lexical identity:** two different entry IDs share a written or spoken form.
- **Polysemy / sense ambiguity:** one entry ID has more than one associated sense.

The feature summary is a posterior-weighted combination of the stored sense feature maps:

\[
V_k = \min\left(1,\sum_i p_i\sum_j p(s_j\mid w_i,c)\,v_{ijk}\right)
\]

This is a transparent software summary of candidate feature evidence, not a claim that the cortex
performs literal vector addition or stores a word in one vector.

The existing lexical engine also supplies Levenshtein/sequence matching, logarithmic frequency
normalization, cosine similarity, Bayesian-style priors, morphological reconstruction, component
feature composition, recurrent activation, competition, and entropy. The 12 source steps map to
the corresponding BLA operations in \`_POINT2_TO_BLA\`.

## Biological reference

When enabled, each lower-level lexical operation advances the existing seven-population,
current-based leaky-integrate-and-fire reference network:

- orthographic
- lexical identity
- semantic
- grammatical category
- morphological
- selection/control
- recurrent candidate

The Point 2 biological trace aggregates these actual BLA population-activity updates under the
12 source-level steps. It remains distinct from the functional trace.

This is a computational reference, not an anatomically fitted or validated reproduction of a
human brain. Its voltages, currents, spike counts, firing rates and simulated time are not
measurements from a person. The software's edit distance, softmax, cosine, entropy and weighted
scores are explicit engineering approximations, not claims that neurons literally execute those
equations.

## Python API

Pass the \`form_hypotheses\` mapping from the Point 1 result and your own verified lexical inventory:

\`\`\`python
from ore_language_comprehension import (
    Point2LexicalAccessProcessor,
    Point2Request,
)
from ore_lexical_access import LexicalAccessConfig
from ore_lexical_context import LexicalEntry, LexicalMeaning

lexicon = (
    LexicalEntry(
        entry_id="cat",
        orthography="cat",
        frequency=24,
        phonemes=("k", "æ", "t"),
        meanings=(
            LexicalMeaning(
                "feline",
                {"ANIMAL": 1.0, "FELINE": 1.0},
                gloss="a small domesticated feline",
                verified_by="your-lexicon-source",
                source_reference="dataset://your-lexicon/cat",
            ),
        ),
        lexical_category="noun",
        language="en",
        verified_by="your-lexicon-source",
        source_reference="dataset://your-lexicon/cat",
    ),
)

# point1_result is the Point1Result returned by Point1PerceptualProcessor.
request = Point2Request.from_point1_result(
    sequence_id="reading-001",
    point1_result=point1_result,
    lexicon=lexicon,
    language="en",
    context_features={"ANIMAL": 0.8},
)
processor = Point2LexicalAccessProcessor(
    config=LexicalAccessConfig(simulate_biology=True),
)
result = processor.process(request)

print(len(result.functional_trace))  # 12
print(result.lexical_candidates)
print(result.word_level_semantic_features)
print(result.unresolved_ambiguities)
print(len(result.low_level_operation_trace))  # one 36-operation trace per lexical-engine run
\`\`\`

For a visual-only request, supply the orthographic hypotheses from Point 1. For auditory access,
the corresponding lexical entries must contain their own phoneme sequences. Optional extra sensory
evidence can be supplied through \`Point2Request.sensory_updates\`; it is processed as a new evidence
pass and the before/after candidate comparison is reported in source step 11.

## SQLite annotations

On initialization, \`Point2LexicalAccessProcessor\` calls
\`register_point02_annotations\` by default. This synchronizes class, function and method
docstrings for \`ore_language_comprehension\`, \`ore_lexical_access\` and
\`ore_lexical_context\` into the existing \`ore_logger.annotations\` SQLite table.
The default database is \`ore_logs/ore_logger.sqlite3\`; override it with \`ORE_LOGGER_DB\`
or the \`database_path\` argument. The helper uses scoped indexing and does not create a
Point-2-specific annotation database.

Execution runs remain compatible with the existing ORE logger's \`runs\` and \`events\` tables.
The adapter does not store raw audio or image payloads in annotation rows.

## Validation

Run the focused Point 2 acceptance tests:

\`\`\`bash
python -m pytest tests/test_language_comprehension_point02.py -q
\`\`\`

The tests cover the 12-step registry, Point 1 visual and auditory handoffs, explicit pronunciation
matching, homonym/polysemy separation, inventory-backed morphology, sensory evidence revision,
the seven-population LIF reference and shared SQLite annotation registration.

Software tests validate implementation contracts and trace integrity. They do not establish
biological equivalence, clinical validity, or language-wide lexical coverage.
