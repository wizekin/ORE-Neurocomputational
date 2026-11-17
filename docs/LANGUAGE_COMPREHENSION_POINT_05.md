# Language comprehension — Point 5: Contextual and pragmatic integration

## Objective and boundary

Point 5 receives the developing literal proposition from Point 4 and evaluates it against
discourse history, the physical or imagined situation, speaker perspective, common ground,
world/event knowledge and the current conversational goal. It enriches interpretation without
discarding the literal proposition.

The implementation is a 20-step source-level adapter over the existing
`ore_discourse` eight-stage / 39-operation engine. This avoids duplicating ORE's established
coreference, discourse-link, speaker-intention, indirect-request, irony, situation/event,
temporal and coherence operations.

## The 20 ordered steps

| Step | Functional operation | Output |
|---:|---|---|
| 1 | Receive and preserve literal proposition | Immutable literal payload, roles, operators, alternatives and confidence |
| 2 | Retrieve relevant discourse and background knowledge | Bounded prior-turn and active-information records |
| 3 | Construct source-sensitive context state | Separate discourse, scene, speaker, common-ground, world and goal channels |
| 4 | Identify focus, topic, entities and events | Topic distribution, salient entities and active events |
| 5 | Resolve references | Coreference links, ranked candidates or unresolved status |
| 6 | Generate context-sensitive predictions | Topic/event expectations and prediction-related evidence |
| 7 | Compare with local discourse | Proposition links, coherence and knowledge conflicts |
| 8 | Compare with situation model | Scene/event records and explicit scene evidence |
| 9 | Evaluate world knowledge and event plausibility | Caller-supplied world evidence and conflict records |
| 10 | Decide whether further inference is needed | Coherence/uncertainty estimate and control signal |
| 11 | Generate bridging and causal inferences | Supported causal and temporal links |
| 12 | Generate pragmatic interpretations | Speaker-intention and indirect-speech-act hypotheses |
| 13 | Infer speaker goals and perspective | Speaker model and intention hypotheses |
| 14 | Interpret irony and sarcasm | Context/prosody-sensitive irony hypotheses |
| 15 | Integrate presuppositions, implicatures and indirect acts | Common-ground and indirect-act records |
| 16 | Rank, combine or defer interpretations | Evidence bundle, coherence and uncertainty |
| 17 | Reanalyse earlier assignments | Recurrent revision and knowledge-conflict records |
| 18 | Update discourse/event/situation models | Updated discourse context and situation graph |
| 19 | Maintain enriched interpretation | Active information; detailed retention is deferred to Point 6 |
| 20 | Deliver enriched interpretation | Literal proposition, contextual additions, uncertainty and Point 6 handoff |

The 39 underlying operations are partitioned, in original order, into 20 non-empty contiguous
groups. This mapping preserves traceability to the existing discourse engine; it does not
pretend that each of the 20 source steps is a separate measured brain operation.

## Mathematical operations

### Retrieval and context construction

A context record can be scored using semantic match, recency, causal relevance and task relevance:

`R_i=a,sim(q,m_i)+b,e^{-\Delta t_i/\tau}+c,k_i+d,g_i`

A context state preserves its separate sources:

`C_t=\{D_t,S_t,M_t,G_t,W_t,Q_t\}`

Contextual signals can be gated by relevance and reliability:

`c_t=\sum_j gate_j c_j,quad gate_j=\sigma(relevance_j\,reliability_j)`

### Focus and reference resolution

A discourse entity's salience can be modelled as:

`salience(e)=a r+b f+c topic+d syntax`

Candidate reference scores can be normalized:

`P(a_i\mid e,C)=\frac{e^{s_i/T}}{\sum_j e^{s_j/T}}`

Uncertainty is measured by entropy:

`H(A)=-\sum_i P(a_i)\log P(a_i)`

A system can abstain when the leading candidate's probability or margin is too small.

### Prediction, coherence and situation matching

Surprisal and prediction error can be represented as:

`I(x)=-\log(P(x\mid C_t)+\epsilon),\quad \delta_t=observed_t-expected_t`

Discourse relation support can be represented by:

`P(r\mid P_t,P_{t-1},C)`

Scene alignment may use a similarity score between linguistic participants and perceived/remembered scene entities. Temporal evidence can use interval overlap:

`O(i,j)=\max(0,\min(e_i,e_j)-\max(s_i,s_j))`

### Pragmatic and social inference

Candidate interpretations can be compared using evidence from literal language, discourse, situation,
world knowledge, speaker perspective and common ground:

`S(H_i)=w_L L_i+w_D D_i+w_S S_i+w_W W_i+w_M M_i+w_G G_i-\lambda A_i`

where (A_i) penalizes unsupported assumptions. Posterior normalization, entropy and a confidence
margin support selection or deferral. Irony can be assessed through a Bayes factor:

`BF=P(E\mid irony,C)/P(E\mid sincere,C)`

The implementation retains literal and nonliteral readings separately. These equations are useful
computational models, not claims that the brain explicitly runs a conventional probabilistic program.

### Recurrent updating and working-memory handoff

A simplified recurrent update is:

`P_{new}=BayesUpdate(P_{old},E_{new})`

and an activity/priority model can be written:

`a_i(t+1)=\lambda a_i(t)+input_i`

Point 5 emits the enriched representation to Point 6; sustained maintenance and capacity management
are implemented in that next stage rather than duplicated here.

## Biological-reference track

The adapter reuses the discourse subsystem's nine-population current-based LIF reference:
discourse memory, proposition binding, speaker model, pragmatic control, incongruity monitor,
situation model, temporal order, narrative coherence and prediction error. The 39 lower-level
operation records are grouped into 20 source-level biological trace records.

The LIF model integrates membrane voltage with a forward-Euler approximation, applies threshold
and reset behavior, refractory periods, recurrent excitation and global inhibition, and reports
simulated voltage (mV), current (pA), time (ms), firing rate (Hz) and spike counts.

This is a deterministic computational analogy. It is not an anatomically fitted simulation,
not a literal model of all language neurons, and not measured human EEG/fMRI or single-cell data.
mPFC, TPJ, temporal language regions and frontoparietal control systems are distributed,
task-sensitive contributors; the implementation does not assign one source step to one region.

## API

```python
from ore_discourse import DiscourseConfig
from ore_language_comprehension import (
    Point5Request, Point5ContextualPragmaticIntegrator,
)

request = Point5Request(
    sequence_id="dialogue-001",
    sentences=["The deployment failed.", "Great work on the release!"],
    prior_turns=["The team expected the release to succeed."],
    prosody={1: {"intonation_mismatch": 0.8, "exaggeration": 0.6}},
)
result = Point5ContextualPragmaticIntegrator(
    DiscourseConfig(simulate_biology=False)
).process(request)
assert len(result.functional_trace) == 20
print(result.enriched_interpretation)
```

To consume the preceding algorithm's output, use
`Point5Request.from_point4_result(sequence_id, point4_result, sentences=[...])`.
Surface sentences are required for discourse analysis. Explicit entities, prior turns, common
ground, speaker beliefs/goals, scene evidence, world knowledge, prosody and temporal evidence can
be supplied as separate fields.

## Annotation registration and tests

`register_point05_annotations` indexes the language-comprehension and discourse packages in
the shared `ore_logger.annotations` SQLite table at `ore_logs/ore_logger.sqlite3`.
It does not create a new Point 5 database.

Run:

```bash
python -m pytest tests/test_language_comprehension_point05.py -q
```

## Scientific limitations and safeguards

- Missing speaker beliefs, scene facts, causal relations and world knowledge remain unknown unless supplied or supported by the discourse engine's explicit conservative cues.
- Some underlying discourse operations use transparent lexical heuristics; they are not a full semantic parser or a validated theory of pragmatic inference.
- Irony and sarcasm are hypotheses, not automatic polarity reversals.
- Inferences remain defeasible and are kept separate from literal entailments.
- A mismatch with the scene does not prove that a linguistic assertion is false; it may be hypothetical, remembered, fictional or mistaken.
- The 20 source-level steps are a functional decomposition. Neural processing is parallel, recurrent and task-dependent.
- Detailed propositional working-memory maintenance is handed to Point 6.
