# ORE — Schema-driven inference and reconstructive memory (Point 20)

## Purpose and relationship to Point 10

Point 20 implements the sequence in
\`Point_20_Schema_Driven_Inference_and_Reconstructive_Memory.md\`. It addresses a different
problem from Point 10's general schema-guided classification and prediction:

- Point 10 asks which reusable schema best explains a current case and predicts likely missing
  properties or relations.
- Point 20 asks what can responsibly be reconstructed about one particular past event when the
  available episodic fragments are incomplete, later information has been introduced, and
  generic schemas may bias recollection.

The two packages remain separate. Point 20 can consume schema descriptions in its own
\`SchemaProfile\` contract and does not make schema predictions equivalent to episode-specific
facts.

The central invariant is that a plausible or coherent completion is not an observation. The
processor therefore keeps direct evidence, episodic retrieval, schema inference, post-event
information, source-uncertain details, contradictions, and unknowns separate.

## Biological and computational scope

The biological-reference track models interacting functional populations corresponding broadly
to:

- hippocampal event binding and cue-driven pattern completion;
- medial prefrontal schema and prior-knowledge use;
- lateral prefrontal controlled retrieval and source monitoring;
- parietal/angular-gyrus accessibility of episodic details;
- sensory association-cortex reinstatement;
- temporal semantic-memory systems; and
- amygdala/neuromodulatory salience modulation.

The populations are recurrently coupled rather than independent one-region/one-operation units.
Their rates are simulated engineering variables—not neural measurements, biophysical neuron
models, fitted physiological constants, or truth detectors. Emotional salience can affect
priority, but it cannot improve a detail's source reliability merely because the detail is
emotionally vivid. The implementation also does not assume that every retrieval triggers
reconsolidation.

## The sixteen source-ordered operations

| Step | Operation | Main computation and safeguard |
|---:|---|---|
| 1 | Define the retrieval target | Encode the question, target feature classes, event identity, context, and required precision |
| 2 | Reactivate episodic fragments | Weighted cue summation and temperature-scaled competition; preserve each episode and relation |
| 3 | Separate retrieved from missing | Feature ledger distinguishes direct, retrieved, weak, not-retrieved, contradicted, and unqueried |
| 4 | Activate candidate schemas | Evidence fit, contextual fit, and goal relevance combined into a normalized schema distribution |
| 5 | Generate inferred completions | Generate multiple schema-conditioned candidates for missing features; label them \`inferred\` immediately |
| 6 | Combine evidence and priors | Bayesian/log-likelihood combination; preserve the prior and evidence contributions separately |
| 7 | Check event constraints | Evaluate supplied spatial, temporal, participant, and causal constraints without smoothing conflicts away |
| 8 | Monitor candidate sources | Estimate original perception, episodic retrieval, schema inference, imagination, and post-event sources |
| 9 | Separate familiarity and recollection | Maintain independent familiarity and source-specific recollection dimensions |
| 10 | Search competing episodes | Compare similar events and flag details that may have migrated from another episode |
| 11 | Calibrate detail confidence | Logistic confidence estimate from evidence, recollection, inferred proportion, contradiction, and source uncertainty |
| 12 | Select reporting policy | Minimize expected loss across report, qualify, clarify, and omit actions |
| 13 | Monitor retelling provenance | Preserve source timestamps and a directed acyclic provenance graph |
| 14 | Update from reliable feedback | Bayesian likelihood-ratio update using reliability and independent-source groups |
| 15 | Update general schema separately | Update beta counts only from verified, independent feedback explicitly authorized for schema learning |
| 16 | Finalize the report | Separate supported, inferred, source-uncertain, contradicted, and unresolved information |

## Mathematical implementation

The registry contains the complete equations and biological references. The major implemented
operations are summarized below.

### Cue-driven episodic reactivation

For fragment \(M_j\) and cue features \(c_i\):

\[
A_j = \sum_i w_{ij}c_i
\]

The implementation uses an inspectable weighted cue/context score and a normalized distribution:

\[
P(M_j\mid c) = \operatorname{softmax}(\beta A_j)
\]

This distribution ranks retrieval candidates. It does not prove that the top-ranked episode or
feature is the correct one.

### Schema activation and candidate generation

For schema \(S_k\), evidence \(E\), context \(C\), and goal relevance \(G_k\):

\[
Z_k = \alpha\,\operatorname{sim}(S_k,E)
      + \gamma\,\operatorname{fit}(S_k,C) + \eta G_k
\]

The activation distribution is normalized with a temperature-scaled softmax. Missing feature
values are proposed from the candidate schema distributions. Each proposal records its schema
source, prior, and initial \`inferred\` status.

### Evidence-prior combination

For candidate detail \(h\):

\[
P(h\mid E,S,C)\propto P(E\mid h,S,C)P(h\mid S,C)
\]

The code calculates a log score from the schema-conditioned prior and fragment likelihoods,
including explicit contradiction penalties. Unknown or un-retrieved features are neutral: they
are not counted as negative evidence.

A high posterior among a small candidate set is not equivalent to high absolute evidence. The
final confidence model separately considers source-specific evidence and inferred-detail
proportion.

### Source monitoring and confidence

Candidate source probabilities are estimated from explicit source records, provenance links,
and event-specific fragment characteristics. A source assignment requires a configured
threshold. Familiarity and recollection are tracked independently:

\[
D(h)=\theta_F F+\theta_R R-\theta_U U
\]

Confidence uses an illustrative logistic model:

\[
\operatorname{Conf}(h)=
\sigma(\beta_0+\beta_1E+\beta_2R-\beta_3I-\beta_4X-\beta_5U)
\]

where \(E\) is evidence strength, \(R\) source-specific recollection, \(I\) inferred proportion,
\(X\) contradiction burden, and \(U\) source uncertainty. Coherence is deliberately not an
input to this confidence estimate. The coefficients require empirical calibration before
real-world use.

### Corrective feedback and separate schema learning

For independent feedback \(F\), candidate odds are updated by:

\[
\operatorname{PosteriorOdds}(h)=
\operatorname{PriorOdds}(h)
\frac{P(F\mid h)}{P(F\mid\neg h)}
\]

Reports sharing an independence group are not counted as independent corroborations. A feedback
record can update an event candidate without changing general schema knowledge. Schema counts
are updated only when the feedback is both verified and explicitly marked for schema learning:

\[
\hat p_{t+1}=\hat p_t+\eta_t(y_t-\hat p_t)
\]

The beta-count state is stored separately from the episode record, preventing one reconstruction
from silently rewriting the event or turning a generated inference into a reusable observation.

### Confidence calibration

Externally labelled outcomes can be used to compute a Brier score:

\[
BS=\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2
\]

This provides a way to audit confidence separately for direct details, episodic retrieval, and
schema-generated inferences.

## Quick start

\`\`\`python
from ore_reconstructive_memory import (
    EpisodicFragment,
    Point20Config,
    Point20Request,
    ReconstructiveMemoryProcessor,
    SchemaProfile,
)

request = Point20Request(
    event_id="story-001",
    cue={"setting": "river"},
    target_features=["setting", "object_type", "cause"],
    episodic_fragments=[
        EpisodicFragment(
            detail_id="setting-1",
            feature_type="setting",
            content="riverbank",
            status="episodically_retrieved",
            retrieval_strength=0.9,
            episodic_context=0.9,
            source_type="episodic_retrieval",
            source_id="original-fragment-1",
            episode_id="story-001",
            cue_features={"setting": "river"},
        )
    ],
    schemas=[
        SchemaProfile(
            schema_id="travel-script",
            name="water journey",
            prior_probability=0.7,
            feature_probabilities={
                "object_type": {"canoe": 0.7, "boat": 0.3},
                "cause": {"storm": 0.6, "collision": 0.4},
            },
            context_profile={"setting": 0.9},
        )
    ],
    context={"gist": "People travelled across water before an unusual event."},
    precision="exact",
)

processor = ReconstructiveMemoryProcessor(Point20Config(simulate_biology=True))
result = processor.process(request)

print(len(result.operation_trace))  # 16
print(result.reconstruction["supported_details"])
print(result.reconstruction["inferred_details"])
print(result.reconstruction["unresolved_features"])
print(result.biological["operation_count"])  # 16
\`\`\`

To represent later testimony or a correction, add a \`SourceRecord\` or \`FeedbackRecord\`.
Schema learning requires verified feedback with \`use_for_schema_update=True\`. Reuse the same
processor instance to retain versioned event records and general schema-count updates across
successive reconstruction rounds.

## SQLite annotations and operation logging

The package reuses the repository's existing \`ore_logger\` SQLite database. The AST-backed
catalogue indexes every module, class, function, and method in \`ore_reconstructive_memory\`
into the shared \`annotations\` table. By default the database is
\`ore_logs/ore_logger.sqlite3\`.

\`record_result_to_sqlite(result)\` registers annotations and persists the complete sixteen-step
execution trace in the logger's \`runs\` and \`events\` tables. The operation metadata records the
mathematical description and separate functional and biological outputs.

Run the focused suite:

\`\`\`bash
python -m pytest tests/test_reconstructive_memory_point20.py -q
\`\`\`

## Bartlett's "The War of the Ghosts" as a conceptual validation case

The testable pattern is conditional rather than absolute: unfamiliar or weakly encoded details
may be omitted, transformed, rationalized, or replaced by culturally familiar schema details over
successive retellings. The algorithm should retain the broad story gist while leaving an
unsupported object subtype or causal explanation unknown. Schema-consistent knowledge can also
improve recall; schema use is not intrinsically erroneous.

The model therefore does not "correct" a memory merely because a familiar script predicts a
different event. It searches for source-specific evidence, tracks later suggestions separately,
and qualifies or withholds unsupported details.

## Explicit limitations

- The model does not establish what a person actually experienced from sparse data.
- A missing retrieval cue does not prove that a feature was absent or never encoded.
- Schema priors may help organize evidence but cannot establish a specific historical detail.
- Familiarity, confidence, coherence, and accuracy are separate quantities.
- Provenance labels are fallible and should be based on supplied evidence.
- Repeated reports from a common source are not independent confirmations.
- General schema learning uses only verified evidence and never silently rewrites an episode.
- Numerical weights and population dynamics are transparent engineering parameters, not universal
  human constants or literal neuronal arithmetic.
