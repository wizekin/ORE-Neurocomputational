# ORE discourse integration — eight stages / 39 operations

**Package:** `ore_discourse/`  
**Registry:** `ORE-DISCOURSE-39`  
**Version:** 2.0.0  
**Status:** Executable functional prototype with an optional coupled LIF population reference.  
**N9 slot:** Implements the existing N7 `predict(payload)` extension contract.

## 1. Purpose and boundaries

This package adds eight discourse/pragmatic stages after sentence-level lexical,
syntactic and semantic composition. It maintains a model of what has already
been said, connects propositions across sentences, estimates communicative
intent, distinguishes literal form from inferred action, evaluates irony cues,
constructs an event/situation representation, tracks temporal relations, and
maintains global topic and narrative consistency.

The input can contain plain sentences and/or explicit proposition records.
Caller-provided entities, aliases, common-ground facts, speaker beliefs/goals,
prosody and temporal intervals are retained as separate evidence. Surface
heuristics are marked as inferences; they do not overwrite the caller's facts.
Unknown antecedents, mental states, event properties and world facts remain
unknown or retain alternatives.

The included English-oriented lexical cues are a deterministic software
baseline, not a general language-understanding system. There is no learned
world-knowledge database, unrestricted theory-of-mind capability, full
prosodic signal analyzer, or validated human-brain simulation.

## 2. The eight stages and all 39 registered operations

The stable operation IDs are `DC-001` through `DC-039`. The source of truth
for each operation's functional responsibility, mathematical sketch, biological
reference and mapping to the detailed stage outputs is `ore_discourse/registry.py`.
Every request returns exactly 39 trace rows in registry order. Some registered
operations intentionally bundle closely related lower-level calculations;
their individual output values remain visible in each trace row under
`functional_output.source_stage_outputs`.

| Stage | Operations | Registered functional operations | Main representation |
|---|---:|---|---|
| S1 — Maintain discourse context | DC-001–005 | Initialize context/retrieve memory; extract mentions/resolve references; update salience; integrate common ground and given/new status; apply working-memory decay | Active entity and discourse-memory state |
| S2 — Integrate consecutive sentences | DC-006–010 | Extract/link propositions; detect connectives; score semantic continuity; infer coherence/bridging; build proposition graph/check compatibility | Proposition graph with typed links |
| S3 — Identify speaker intention | DC-011–015 | Initialize beliefs/infer communicative act; estimate goals; infer mental-state cues; compare common ground/epistemic commitment; rank intentions/preserve alternatives | Ranked communicative-act and goal hypotheses |
| S4 — Understand indirect requests | DC-016–020 | Preserve literal meaning/detect indirect forms; map intended action; integrate listener/scene context; estimate politeness/distinguish question/request; score action/preserve literal distinction | Separate literal interpretation and action hypothesis |
| S5 — Interpret irony | DC-021–024 | Detect polarity/retrieve outcome; measure valence mismatch; integrate prosody/evaluate incongruity; rank irony/sarcasm alternatives and uncertainty | Literal and ironic alternatives with cue scores |
| S6 — Construct situation models | DC-025–029 | Instantiate participants/extract events; assign roles; track entity states; infer goals/causal links; build/validate situation graph | People, events, participants, states, goals and causal graph |
| S7 — Track temporal discourse relations | DC-030–034 | Extract time cues/order events; estimate intervals; detect overlap/containment; resolve connectives/time shifts; normalize timeline/validate constraints | Event intervals, order links and validation |
| S8 — Maintain topic and narrative coherence | DC-035–039 | Aggregate topic features; estimate topic continuity; track narrative progression; integrate temporal/causal structure/detect contradictions; revise global state/compute coherence/produce summary | Global topic, progression, consistency and summary |

The stage implementations retain the finer calculations described below. The
39-operation registry is the stable public execution contract; its biological
reference advances once per registered operation.

### Stage 1 — Discourse context

Each discourse sample begins with a context state:

\[
D_0 = \{M_0, E_0, G_0, I_0\}
\]

where \(M_0\) is caller-supplied prior-turn/episodic memory, \(E_0\) is the
entity inventory, \(G_0\) is declared common ground, and \(I_0\) is active
information status. Entity mention records preserve sentence locations and
whether a candidate originated from a caller alias or a surface heuristic.

Coreference candidates use recency, current salience and optional entity-type
compatibility:

\[
s(e,m) = w_r R(e,m) + w_s S(e) + w_f F(e,m)
\]
\[
P(e\mid m) = \operatorname{softmax}(s(e,m)/T)
\]

The baseline resolves only when the best candidate has enough separation from
the runner-up. Ambiguous/unresolved cases are retained. Entity salience is a
bounded combination of recency, mention frequency and topical relevance:

\[
S_e(t)=aR_e(t)+b(1-e^{-n_e})+cT_e
\]
\[
R_e(t)=e^{-\lambda (t-t_{\mathrm{last}})}
\]

The implementation reads prior turns and episodic records without mutating
them. The decay expression is a computational forgetting/activation policy;
it is not a claim that human memory follows one universal exponential law.

### Stage 2 — Sentence integration

A proposition record contains a stable ID, sentence index, predicate,
arguments, polarity, modality and topic features. Explicit proposition records
take precedence over conservative surface cues.

Adjacent propositions are linked using entity overlap, content-word similarity
and explicit connectives:

\[
\operatorname{continuity}(i,j)=
w_e J(E_i,E_j)+w_t\cos(T_i,T_j)
\]

A relation hypothesis is selected from continuation, elaboration, contrast,
explanation, result, condition and temporal sequence. The package keeps the
connective and evidence source in the edge record. Bridging links are added
only for a pronoun reference that Stage 1 resolved to a supported entity.
Proposition compatibility and contradictions are reported separately from
link confidence.

### Stage 3 — Speaker intention

The speaker model keeps caller-declared beliefs and goals apart from
surface-inferred hypotheses. Communicative acts include assertion, question,
request, command, promise, apology and greeting. Cues such as modal verbs,
interrogative form, politeness markers and mental-state verbs contribute to
ranked hypotheses:

\[
P(a\mid u,c)=
\frac{\exp(s(a,u,c)/T)}{\sum_{a'}\exp(s(a',u,c)/T)}
\]

Speaker goals and mental-state expressions are inferred only from explicit
language patterns or supplied fields. The model does not claim direct access
to a speaker's actual beliefs. Close alternatives are preserved using a
top-two margin and an abstention flag.

### Stage 4 — Indirect requests

Literal sentence type and inferred listener action are independent fields.
Conventional patterns include ability/willingness questions (for example,
"Could you open the window?"), willingness questions ("Would you mind ..."),
permission-prefaced requests, and imperative forms. Scene hints are used only
when the caller explicitly supplies the relevant scene state.

\[
P(\text{request}\mid u,c)=
\sigma\left(k\left(s_{\mathrm{action}}(u,c)
-s_{\mathrm{information}}(u,c)\right)\right)
\]

The returned object retains the literal question/assertion/imperative, action
candidate, construction type, listener-feasibility cue, directness, politeness
markers and confidence. Ability questions can remain ambiguous; a question
mark alone is not treated as proof of a request.

### Stage 5 — Irony

The baseline compares lexical valence with prior contextual event valence,
then optionally incorporates caller-supplied prosodic cues. It does not
extract pitch or duration from audio itself.

\[
I =
w_s M_{\mathrm{semantic}}+
w_c M_{\mathrm{context}}+
w_p M_{\mathrm{prosody}}
\]
\[
P(\mathrm{irony})=\sigma(k(I-\theta))
\]

The output includes lexical polarity, contextual event polarity, semantic
mismatch, contextual mismatch, prosodic incongruity, explicit markers,
irony probability and subtype hypothesis. Sarcasm is a candidate label only;
subtype judgments need richer interactional context and human evaluation.

### Stage 6 — Situation model

The event model stores typed people/participants, events, arguments/roles,
explicit states, goal hypotheses and causal links:

\[
G_S=(V_{\mathrm{people}}\cup V_{\mathrm{event}}\cup V_{\mathrm{state}},
E_{\mathrm{role}}\cup E_{\mathrm{cause}})
\]

Caller-provided event arguments and causal fields are retained. Surface
extraction is intentionally conservative. States are created from explicit
state/copular predicates or caller records; the code does not assign hidden
traits or motivations from stereotypes. Graph validation checks endpoint IDs,
participant references and explicit inconsistencies.

### Stage 7 — Temporal relations

Each event can carry a start, end, duration, time expression, anchor and tense.
Caller interval values are preferred over simple lexical time cues.
For numeric intervals \(I_i=[s_i,e_i]\), overlap is computed by:

\[
O(i,j)=\max(0,\min(e_i,e_j)-\max(s_i,s_j))
\]

Before-links create a directed graph. A deterministic topological sort checks
whether ordering constraints contain a cycle. Negative durations and
end-before-start intervals are invalid. If time expressions or absolute
anchors are absent, the interval remains explicitly unresolved rather than
being assigned a fabricated date.

### Stage 8 — Topic and narrative coherence

Content-word frequency creates a sparse topic vector \(T_d\), and adjacent
sentence vectors are compared by cosine similarity. Narrative progression
tracks recurring participants/predicates, introduced events, and declared or
inferred goal hypotheses. Contradiction checks require the same predicate and
at least one shared explicit participant with opposite negation evidence.

The global coherence score is a normalized weighted combination:

\[
C = w_e E + w_t T + w_r R + w_k K,\qquad
\sum_i w_i=1
\]

A bounded update can blend the global score with the mean confidence of local
proposition links:

\[
C'=(1-\rho)C+\rho\overline{q}_{\mathrm{links}}
\]

The original sentence and observation channels remain preserved. The summary
reports the dominant topic term, event/proposition counts and explicit
conflicts; it is not an unconstrained generative summary.

## 3. Biological-reference track

When `DiscourseConfig(simulate_biology=True)`, the simulator advances a
coupled population reference once for every registered operation (39 steps). The
populations are:

- `discourse_memory`
- `proposition_binding`
- `speaker_model`
- `pragmatic_control`
- `incongruity_monitor`
- `situation_model`
- `temporal_order`
- `narrative_coherence`
- `prediction_error`

For population \(i\), a current-based leaky integrate-and-fire reference uses:

\[
V_i(t+\Delta t)=V_i(t)+\frac{\Delta t}{\tau_m}
\left(V_{\mathrm{rest}}-V_i(t)+10^{-3}R_m I_i(t)\right)
\]

A threshold crossing increments a spike count, resets voltage and starts a
refractory interval. Recurrent excitation and global inhibition provide
coupling. The result reports membrane voltage (mV), current (pA), time (ms),
spikes and firing rate (Hz). Population routing is a declared hypothesis for
software experiments. It is not a fitted cortical network, cellular model,
proof of anatomical localization, or evidence that the brain literally
executes the 39 registered computational operations.

Disable this reference with:

```python
DiscourseProcessor(DiscourseConfig(simulate_biology=False))
```

## 4. API and N9 integration

Standalone use:

```python
from ore_discourse import DiscourseProcessor, DiscourseRequest

request = DiscourseRequest(
    sequence_id="story-001",
    sentences=["Mira arrived.", "Then she opened the door."],
    entities=[{"entity_id": "mira", "label": "Mira", "entity_type": "person"}],
    common_ground={"facts": ["Mira was expected."]},
)
result = DiscourseProcessor().process(request)
print(result.discourse_context["coreference"])
print(result.temporal_relations)
print(result.coherence["score"])
print(len(result.operation_trace))  # 39
```

N9 already accepts an object exposing `predict(payload)` in its `n7`
adapter slot. The new `DiscourseN7Adapter` implements this contract and
returns `intents`, `predictions`, `situation_model`, `coherence`,
`uncertainty`, `temporal_links`, `causal_links`,
`proposition_compatibility` and `knowledge_conflicts`.

```python
from ore_discourse import DiscourseN7Adapter
from ore_n9 import N9Adapters, N9Processor

processor = N9Processor(adapters=N9Adapters(n7=DiscourseN7Adapter()))
integrated_result = processor.process(n9_request)
```

The adapter reads the existing N9 proposition, semantic-graph, lexical-hypothesis
and discourse-context payloads. It does not modify upstream results.

## 5. Annotation logging

Every public and private class/function in the new Python modules has an
English docstring. The existing `ore_logger.catalog.sync_repository()`
AST catalogue reads these docstrings into the repository's
`annotations` table in `ore_logs/ore_logger.sqlite3`. The automatic runtime
logger links function call/return/exception events to those annotation rows
when an ORE Python entry point is executed with auto-logging enabled.

The generated SQLite database is runtime output and is intentionally excluded
from Git. It is created locally when the package is run from an installed
repository; this commit adds the annotated source definitions and does not
commit a machine-specific database file.

## 6. Installation, tests and limits

The package is registered in `pyproject.toml` and uses the repository's
existing NumPy dependency.

```bash
python -m pip install -e ".[dev]"
python -m ore_discourse
python -m pytest tests/test_discourse.py -q
python -m pytest
```

Included software tests cover registry order/count, one biological update
per trace row, entity/coreference preservation, common ground, connective
integration, indirect-request separation, contextual/prosodic irony cues,
temporal ordering/duration, N9 adapter fields, JSON serialization and input
validation.

Acceptance categories must remain separate:

- **Software integrity:** established only for tested contracts and code paths.
- **Behavioral validity:** requires language-specific, held-out dialogue and
  narrative corpora with independent human labels and calibration.
- **Biological validity:** requires task-matched, independently collected
  neural measurements, preregistered predictions, held-out evaluation and
  replication.

Passing unit tests does not establish behavioral or biological validity.
