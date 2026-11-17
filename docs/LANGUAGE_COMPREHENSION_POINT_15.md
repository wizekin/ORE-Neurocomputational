# Language Comprehension — Point 15
## Open problems, temporal coordination, causal mechanisms, and research limitations

This is a scientist-facing research and model-validation algorithm, not a sixteenth
cognitive stage. It investigates how distributed systems support the earlier language
comprehension operations. Its equations are analyst-side abstractions unless independently
validated as biological mechanisms.

## Scope and source mapping

The implementation follows the source's 25 ordered stages: define falsifiable claims;
compare language-only, context, social-inference, domain-general-control, and recurrent
accounts; derive discriminating predictions; design controlled conditions; manipulate
factors; match baseline difficulty; mark informative events; define behavioural outcomes;
select complementary neural methods; quality-check signals; preserve individual networks;
align events; estimate time-resolved representations; estimate timing uncertainty; test
overlap; build candidate network graphs; distinguish functional/effective connectivity;
model dynamic coupling; test context modulation; narrow causal claims; select ethical
causal methods; grade evidence; validate across populations; test held-out predictions;
and report uncertainty and falsifiable next steps.

## Data contract

`Point15Request` accepts:
- `research_question`: required non-empty question.
- `conditions`: condition dictionaries with `condition_id`, `factors`, and optional
  baseline variables (lexical_frequency, word_length, sentence_length, etc.).
- `event_times`: `event` and `time` for informative stimulus onsets.
- `behavioural_trials`: outcome/accuracy, response_time, confidence, condition, context,
  group_id, split, and predicted_probability.
- `neural_records`: participant_id, region, signal, time, event_time, value, modality.
- `network_edges`: source, target, weight, optional time and kind.
- `interventions`: method, target, outcome, control_outcome, condition, approved.
- `population_groups`: group_id, language, cultural_context, and measurement_invariance_tested.
- `hypotheses`: hypothesis_id, family, prediction, and alternative.

All input records are copied. Missing and invalid measurements remain unknown.

## Mathematical operations and biological interpretation

| Research operation | Mathematical/algorithmic treatment | Plausible biological level | Critical limitation |
|---|---|---|---|
| Event timing | Differences between successive event timestamps | Continuous sensory and neural response timing | Timing is stimulus- and method-dependent |
| Neural representation | Signal means, variance, decoding/representational analyses (when labels exist) | Population activity carrying task-relevant information | Decodability is not explicit symbolic coding |
| Temporal overlap | Pairwise correlation at shared timestamps; lagged correlation can be added to richer datasets | Overlapping and recurrent neural dynamics | Correlation does not identify direction |
| Network representation | Weighted adjacency matrix (W_{rs}(t)) | Synaptic communication and anatomical pathways | Estimated edges are not direct anatomy |
| Dynamic coupling | Candidate state-space model (z_{t+1}=A_tz_t+Bu_t+\epsilon_t), (y_t=Cz_t+\nu_t) | Synaptic, membrane, population, and neuromodulatory dynamics | Matrix multiplication is a model abstraction |
| Causal intervention | Outcome minus matched control outcome | Perturbation-sensitive contribution in distributed circuitry | Ethical approval, sham/control, and specificity required |
| Cross-population variation | Group means and hierarchical parameters | Learning-dependent adaptation and individual variation | Measurement invariance and representative sampling required |
| Prediction validation | Held-out Brier score (N^{-1}\sum_i(p_i-y_i)^2) | A model should predict new behaviour | Predictive accuracy alone does not prove biological truth |

The implementation only reports temporal samples and correlation from the supplied data.
It deliberately does not claim a state-space fit without synchronized multivariate time
series and an explicit model specification. It does not calculate a causal effect from
unapproved or incomplete intervention records.

## Causal evidence safeguards

Behavioural association, neural association, temporal precedence, model-based directed
influence, perturbational evidence, and convergent evidence are kept distinct. An active
region, a lagged signal, or functional connectivity is not causal proof. The processor
reports intervention differences only for explicitly approved records with both outcome
and control outcome.

## Cross-linguistic and cultural validation

The output summarizes participant groups separately and only marks comparability as
established when every supplied group explicitly records measurement-invariance testing.
This is a bookkeeping guard, not a substitute for psychometric invariance analysis,
translation review, local norming, or independent replication.

## Use

```python
from ore_language_comprehension import Point15Request, Point15MechanisticResearchProcessor

request = Point15Request(
    research_question="Does discourse context alter network coupling?",
    conditions=[{"condition_id": "neutral"}, {"condition_id": "supportive"}],
    event_times=[{"event": "critical_word", "time": 420}],
)
result = Point15MechanisticResearchProcessor().process(request)
print(result.report)
print(len(result.trace))  # 25
```

SQLite annotations are registered through `register_point15_annotations()`. The function
reuses the repository's `ore_logger.annotations` table and scopes synchronization to
`ore_language_comprehension`; it creates no separate annotation database.

## Scientific limitations

No diagnostic classification, reverse inference, fabricated neural data, or universal
processing timetable is produced. A software test verifies code contracts, not biological
validity. Findings must be preregistered, measured with appropriate modalities, replicated,
and interpreted within task, participant, and language/cultural limits.
