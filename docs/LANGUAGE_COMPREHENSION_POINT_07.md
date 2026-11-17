# Language comprehension — Point 7: long-term memory storage and decision use

## Purpose and boundary

Point 7 receives the structured proposition and situation-model state maintained by Point 6. It does not repeat perception, lexical access, parsing, semantic composition, pragmatic integration, or temporary working-memory maintenance.

It separates three things that must not be confused:

1. **Episodic memory:** memory for a particular event, including its participants, context, source, and relations.
2. **Semantic knowledge:** generalizable knowledge that may be updated only after checking scope, source, consistency, and reliability.
3. **Immediate task use:** retrieval, question answering, judgment, option comparison, response selection, and planning.

A proposition may follow multiple routes. A report can guide an immediate decision and also be encoded as an episode of hearing that report, without turning the reported content into a verified general fact.

## Architecture and implementation

The implementation contains 28 ordered source-level steps in ore_language_comprehension/point07.py.

- Point7Request accepts a proposition, component-level provenance/confidence, context, task, retrieval query, answer candidates, action options, and optional outcome feedback.
- Point7Request.from_point6_result(...) carries forward Point 6's active propositions and situation model.
- Point7MemoryStore stores episodic traces and semantic candidates in separate SQLite tables. The default is :memory:; pass a database path to persist across process restarts.
- Point7MemoryDecisionProcessor performs the functional operations.
- Point7LIFReference optionally simulates ten current-based LIF populations as a biological-reference track.

The Point 7 memory database is separate from ore_logger.sqlite3. Annotation registration still uses the repository's shared ore_logger.annotations table.

## The 28 ordered steps

| Step | Operation | Main result |
|---:|---|---|
| 1 | Receive the maintained proposition | Point 6 proposition and context handoff |
| 2 | Validate content, provenance and unresolved fields | Validated components and uncertainty |
| 3 | Represent the current task goal | Goal and time horizon |
| 4 | Estimate durable-retention priority | Normalized retention score |
| 5 | Select transient, episodic, semantic and immediate-use routes | One or more processing routes |
| 6 | Extract the central event and its participants | Predicate and role–filler event core |
| 7 | Bind event content to its context | Event-context trace candidate |
| 8 | Separate item and context while preserving links | Distinct but associated feature sets |
| 9 | Differentiate similar episodes | Similarity/interference estimate |
| 10 | Encode event and relational structure | Participant, temporal and discourse relations |
| 11 | Initiate activity-dependent encoding | Initial labile trace and plasticity analogy |
| 12 | Preserve source and confidence metadata | Component-level provenance |
| 13 | Detect generalizable semantic candidates | Candidate claim with scope and source |
| 14 | Compare with existing knowledge | Confirmation/extension/conflict status |
| 15 | Integrate without blindly overwriting | Accepted or pending semantic record |
| 16 | Model consolidation and reorganization | Reactivation/interference state |
| 17 | Translate a later question into retrieval cues | Query cue representation |
| 18 | Reactivate candidate traces | Ranked episodic/semantic candidates |
| 19 | Complete partial memories while marking reconstruction | Stored fields distinguished from inferred fields |
| 20 | Resolve competition among retrieved memories | Selected candidate or abstention |
| 21 | Parse the immediate answer/judgment request | Question type and target semantic role |
| 22 | Generate and evaluate candidate answers | Evidence and assumption scores |
| 23 | Monitor conflict and calibrate the answer | Selected, qualified or deferred answer |
| 24 | Generate feasible actions and predict outcomes | Constraint-filtered action candidates |
| 25 | Evaluate options and accumulate evidence | Transparent option scores |
| 26 | Apply a task-sensitive decision threshold | Select or defer |
| 27 | Select and prepare a response/action plan | Output plan, not execution |
| 28 | Monitor outcomes and update task state | Prediction error and feedback record |

## Mathematical operations

### Retention priority

A transparent retention score is:

\[
E = 0.20N + 0.30R + 0.15D + 0.25G + 0.10S
\]

where novelty N, task relevance R, distinctiveness D, intentional-learning goal G, and salience S are normalized to [0,1]. These weights are engineering defaults, not measured neural constants.

### Episodic binding and differentiation

A candidate trace binds event content e, context c, and relations r:

\[
m_{\mathrm{episode}} = f_{\mathrm{bind}}(e,c,r)
\]

The implementation stores item and context features separately, plus the links between them. Its baseline episode similarity uses Jaccard overlap over normalized tokens. This is deliberately inspectable and should be replaced by a trained semantic representation when available.

A simplified plasticity analogy is:

\[
\Delta w_{ij} = \eta x_i x_j
\]

This is not a complete model of hippocampal plasticity, spike timing, neuromodulation, or cellular mechanisms.

### Semantic integration

Candidate knowledge is compared with prior records using both similarity and a separate consistency check. A Bayesian update can describe the normative problem:

\[
P(H\mid D)=\frac{P(D\mid H)P(H)}{P(D)}
\]

The software does **not** automatically convert one sentence into accepted truth. A candidate remains pending_verification unless the caller explicitly marks it verified and its components are asserted or reported. Provenance and valid-time information remain attached.

### Retrieval and competition

A cue-based score can combine content, context, and task relevance:

\[
a_j=\lambda_1\,sim(q,m_j)+\lambda_2\,context_j+\lambda_3\,goal_j
\]

Candidate competition can be modelled with:

\[
P(m_j\mid q)=\frac{e^{a_j/\tau}}{\sum_k e^{a_k/\tau}}
\]

The implementation uses a transparent token-overlap score and a margin-based abstention rule. Episodic retrieval increments a reactivation counter and a bounded trace-strength estimate. Pattern completion never fabricates missing trace fields: it returns only stored relations and marks reconstruction status.

### Answer selection

For candidate answer a:

\[
Score(a)=w_e Evidence(a)+w_c ContextFit(a)+w_g GoalFit(a)-w_u UnsupportedAssumptions(a)
\]

The current adapter gives direct semantic-role matches priority over caller-supplied candidates. Its confidence is an uncalibrated model score, not an empirically validated probability. A low evidence score or narrow candidate margin leads to abstention.

### Decision and action planning

A normative option score may be written:

\[
EU(a)=\sum_o P(o\mid a,s,K)U(o)-Cost(a)
\]

The implementation accepts caller-supplied option feasibility, expected utility, cost, and risk. It filters hard-constraint violations before scoring. High-consequence decisions require an explicit evidence_sufficient flag; otherwise the processor defers.

A simplified evidence accumulator is:

\[
X_{t+1}=X_t+\delta_t+\epsilon_t
\]

A task-sensitive threshold theta can increase with the cost of error. These are useful models for selected decision tasks, not universal brain algorithms.

### Outcome monitoring

\[
\delta_t=Outcome_t-PredictedOutcome_t
\]

When numeric feedback is supplied, the processor reports this difference and stores the feedback in a separate outcome-feedback table. It does not silently rewrite previously accepted semantic knowledge.

## Biological-reference track

The optional LIF model contains ten functional populations:

1. hippocampal binding;
2. entorhinal interface;
3. perirhinal item processing;
4. parahippocampal context processing;
5. neocortical semantic representation;
6. medial-prefrontal schema/context coordination;
7. dorsolateral-prefrontal goal and retrieval control;
8. anterior-cingulate conflict monitoring;
9. orbitofrontal value/outcome evaluation;
10. basal-ganglia action gating.

The simulator uses a current-based leaky integrate-and-fire approximation with membrane integration, threshold/reset, and refractory periods. These labels are functional analogies, not one-to-one anatomical mappings. It is not fitted to human recordings and is not physiological validation.

## API example

```python
from ore_language_comprehension import (
    Point7MemoryDecisionProcessor,
    Point7MemoryStore,
    Point7Request,
)

# A path enables persistence across process restarts.
store = Point7MemoryStore("ore_logs/point7_memory.sqlite3")
processor = Point7MemoryDecisionProcessor(store)

request = Point7Request(
    sequence_id="meeting-info-001",
    proposition={
        "predicate": "MEETING_START",
        "semantic_roles": {"time": "3 p.m.", "location": "Room 12"},
    },
    context={"source": "calendar message", "valid_time": "2026-11-17"},
    task={
        "goal": "attend the meeting",
        "encode_episode": True,
        "relevance": 0.9,
        "intentional_learning": 0.8,
    },
)
result = processor.process(request)
print(result.episodic_trace)
print(result.diagnostics)
```

To use the Point 6 output directly:

```python
request = Point7Request.from_point6_result(
    "p7-from-p6",
    point6_result,
    task={"requires_answer": True},
    query="Who apologized?",
)
```

## Annotation registration and tests

register_point07_annotations() refreshes the Point 7 definitions in the existing shared ore_logger.annotations SQLite catalogue.

Run focused tests:

```bash
python -m pytest tests/test_language_comprehension_point07.py -q
```

## Limitations and safeguards

- The default store is process-local SQLite memory. A supplied path is required for persistence across restarts.
- Episodic and semantic records use separate tables and schemas.
- A reported or inferred component is never silently relabelled as an explicit observation.
- Semantic candidates remain pending until caller verification; this is not an external truth checker.
- Token overlap is a baseline retrieval measure, not semantic understanding.
- Retrieved content is reconstructed and can be incomplete; missing fields are not invented.
- Decision option scores require caller-supplied estimates and are not objective forecasts.
- High-consequence choices defer unless evidence sufficiency is explicitly supplied.
- The LIF reference is a simplified computational analogy, not an anatomically or physiologically validated brain simulation.
- The SQLite memory store is distinct from the annotation catalogue and does not imply that every understood sentence is permanently remembered.
