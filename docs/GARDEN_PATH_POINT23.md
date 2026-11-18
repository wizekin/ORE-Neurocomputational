# Point 23 — Garden-Path Sentences and Syntactic Ambiguity

## Implementation boundary

The ore_garden_path package implements the seventeen operations in the Point 23 source document. It models incremental word input, lexical-category competition, grammar-supplied parse hypotheses, token-by-token structural evidence, semantic plausibility, working-memory dependency maintenance, disambiguation, revision, thematic-role reassignment, partial/good-enough interpretation, surface-cue integration, and experience adaptation.

**Important scope distinction:** this is a configurable parsing and reanalysis research prototype, not a general-purpose grammar for every language. A grammar/analysis front end supplies candidate parse hypotheses, word-category assignments, structural likelihoods, dependency links, choice points, and thematic roles. The package scores and revises those candidates transparently. When no grammar candidates are supplied, it creates only a surface-order fallback and explicitly returns an unresolved syntactic result.

## The seventeen source-ordered operations

| Step | Operation | Functional implementation | Broad biological reference |
|---|---|---|---|
| 1 | Segment input | Tokenize text while preserving punctuation and word offsets | Auditory/visual input pathways and attention |
| 2 | Activate lexicon | Retrieve candidate lemmas, senses, categories, and lexical priors | Distributed temporal lexical/semantic representations |
| 3 | Generate parses | Extend supplied grammar-generated syntactic hypotheses | Temporal and inferior-frontal language networks |
| 4 | Estimate expectations | Incrementally score lexical and structural likelihoods; calculate surprisal | Experience-dependent temporal/frontal language activity |
| 5 | Integrate semantics | Combine syntactic log evidence with semantic plausibility | Temporal semantic systems and frontal language regions |
| 6 | Select provisional parse | Preserve a bounded beam and prefix-leading initial parse | Distributed incremental commitment and competition |
| 7 | Maintain dependencies | Track open dependencies and thematic roles under decay/interference | Working memory, attention, frontotemporal pathways |
| 8 | Detect conflict | Track token surprisal, posterior shift, entropy, and parse switching | Medial frontal monitoring and control systems |
| 9 | Test viability | Retain/reject candidates by posterior viability and preserve audit data | Temporal constraints and inferior-frontal unification |
| 10 | Trigger revision | Request reanalysis when an alternative displaces the initial structure or conflict is high | Left IFG and domain-general control interactions |
| 11 | Locate choice point | Find the earliest changed structural decision | Memory reactivation and temporal-frontal interaction |
| 12 | Reconstruct structure | Select a viable alternative supplied by the grammar front end | Cooperative frontotemporal language circuitry |
| 13 | Reassign roles | Recompute agent, patient, subject, object, and dependency relations | Temporal semantic representations and event-role integration |
| 14 | Reconcile interpretations | Keep initial activation separate from final parse confidence | Persistence and updating of distributed representations |
| 15 | Choose reanalysis depth | Select full, good-enough, or unresolved status according to task demands | Working memory, attention, domain-general control |
| 16 | Integrate surface cues | Fuse prosody, punctuation, context, and discourse likelihoods | Auditory prosody, visual cues, language/discourse networks |
| 17 | Adapt expectations | Update construction counts from explicit feedback; exposure alone is not correctness | Distributed memory and learning systems |

Each trace row records its P23 operation ID, mathematical abstraction, functional result, and biological-reference output.

## Core mathematical operations

### Parse ranking and surprisal

For candidate parse \(P_i\) and observed prefix \(X\):

\[
P(P_i\mid X)\propto P(X\mid P_i)P(P_i)
\]

The implementation accumulates log prior, lexical-category likelihood, and parse-specific structural likelihood. For token \(x_t\):

\[
Surprisal(x_t)=-\log P(x_t\mid x_{1:t-1})
\]

The predictive probability is calculated as a mixture over current candidate parses. Surprisal predicts processing cost in some settings; it is not a complete theory of syntactic difficulty.

### Parse uncertainty and conflict

\[
H(P)=-\sum_iP(P_i\mid X)\log P(P_i\mid X)
\]

Normalized entropy allows comparison across different candidate counts. The conflict score combines normalized surprisal with a total-variation measure of parse-posterior change. These are engineering diagnostics, not measured neural conflict signals.

### Working-memory accessibility

\[
A_i(t)=A_i(t_0)e^{-\lambda\Delta t}-\sum_j\rho_{ij}A_j(t)
\]

The prototype uses distance-dependent decay, same-relation interference, and an explicit capacity-pressure indicator. It does not claim that all linguistic memory follows a single exponential law.

### Revision, role recomputation, and confidence

Revision is requested when the initial parse becomes non-viable, an alternative has a sufficient posterior advantage, or conflict exceeds the configured threshold. The system identifies changed choice points and recomputes roles from the selected structure.

Separate confidence estimates are reported for syntax, semantic interpretation, and thematic roles. The earlier parse is not assumed to be erased merely because a revised parse wins.

### Task-sensitive stopping and learning

\[
EU_{continue}=P(correct\mid more\ processing)\cdot Value_{accuracy}-Cost_{processing}
\]

A precise task requires a high-confidence, complete parse; a gist task can accept a clearly labelled good-enough interpretation; unresolved dependencies or missing grammar candidates prevent a claim of full syntactic resolution.

Construction expectations use smoothed counts. Exposure increments are logged separately from correctness feedback: only a supplied feedback_parse_id changes parse-outcome counts.

## API example

~~~python
from ore_garden_path import (
    LexicalCandidate, ParseChoicePoint, ParseHypothesis,
    Point23Config, Point23GardenPathProcessor, SentenceRequest,
)

request = SentenceRequest(
    sentence_id="garden-001",
    text="While the man hunted the deer ran into the woods.",
    language="en",
    lexicon={
        "the": (LexicalCandidate("determiner"),),
        "man": (LexicalCandidate("noun"),),
        "hunted": (LexicalCandidate("verb"),),
        "deer": (LexicalCandidate("noun"),),
        "ran": (LexicalCandidate("verb"),),
    },
    parse_hypotheses=(
        ParseHypothesis(
            parse_id="deer-object",
            description="deer is object of hunted",
            prior=0.8,
            token_categories={1:"determiner",2:"noun",3:"verb",4:"determiner",5:"noun",6:"verb"},
            structural_likelihoods={6:0.01},
            thematic_roles={"hunted.agent":2,"hunted.patient":5},
            choice_points=(ParseChoicePoint("deer-role",5,"object"),),
        ),
        ParseHypothesis(
            parse_id="deer-subject",
            description="deer is subject of ran",
            prior=0.2,
            token_categories={1:"determiner",2:"noun",3:"verb",4:"determiner",5:"noun",6:"verb"},
            structural_likelihoods={6:0.99},
            thematic_roles={"hunted.agent":2,"ran.agent":5},
            choice_points=(ParseChoicePoint("deer-role",5,"matrix-subject"),),
        ),
    ),
)
result = Point23GardenPathProcessor(Point23Config()).process(request)
print(result.initial_parse_id, result.final_parse_id)
print(result.thematic_roles, result.confidence, result.interpretation_status)
~~~

The SentenceRequest contract supports written and spoken modalities, language-specific lexicons, lexical ambiguity, dependency/role templates, construction identifiers, context/prosody/punctuation/discourse likelihoods, and optional explicit feedback.

## SQLite annotation and operation logging

~~~python
from ore_garden_path import record_result_to_sqlite, register_annotations
catalogue = register_annotations()
report = record_result_to_sqlite(result)
print(report["run_id"], report["database_path"])
~~~

All package classes, functions, and methods are synchronized into the shared ore_logger.annotations catalogue. The result logger writes one event for each of the seventeen operations into the existing runs and events tables.

## Tests

Run:

~~~bash
python -m pytest tests/test_garden_path_point23.py -q
~~~

The focused suite covers incremental parse switching, surprisal at the disambiguating word, earliest choice-point tracing, thematic-role correction, no-revision controls, missing-grammar fallback, context/prosody cue integration, good-enough interpretations, unresolved dependencies, working-memory pressure, explicit feedback, written/spoken separation, biological-track disabling, validation, and SQLite logging.

## Scientific interpretation and limitations

- The biological track is a coarse distributed population-rate reference. Its drives and activations are illustrative engineering parameters, not measured neural activity.
- Temporal language regions, left inferior frontal cortex, and domain-general frontoparietal control are broad systems-level references. The implementation does not map one parse operation to one brain region.
- The relative contribution of prediction, memory retrieval, structural revision, semantic plausibility, and good-enough comprehension is debated and varies by construction and task.
- A high-surprisal word is not automatically syntactically difficult; structural integration, dependency length, cue conflict, and memory demands also matter.
- Rereading/regression is not treated as proof that reanalysis succeeded. P600 and N400 are not used as unique labels for individual algorithmic operations.
- The parser requires grammar hypotheses or an upstream syntax component for robust interpretation. The surface-only fallback is explicitly unresolved.
- This is research software, not a clinical assessment tool or a claim of complete human sentence comprehension.

## Research references from the source point

- *Fractionating difficulty during sentence comprehension using functional neuroimaging* (2024), *Cerebral Cortex*: https://pubmed.ncbi.nlm.nih.gov/38314589/ and https://pmc.ncbi.nlm.nih.gov/articles/PMC10847905/
- Christianson et al. (2024), *Retracing the garden-path: Nonselective rereading and no reanalysis*, *Journal of Memory and Language*, 137, 104515: https://doi.org/10.1016/j.jml.2024.104515
- *Neural Mechanisms Underlying the Dynamic Updating of Native Language* (2023): https://pmc.ncbi.nlm.nih.gov/articles/PMC10158591/
- *Beyond initial misanalysis: the disruptive nature of garden-path sentences processing* (2025): https://doi.org/10.1080/0163853X.2025.2513210
