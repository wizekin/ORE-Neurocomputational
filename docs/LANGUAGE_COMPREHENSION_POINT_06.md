# Language comprehension — Point 6: Propositional working-memory representation and maintenance

## Purpose and scientific boundary

Point 6 receives the developing interpretation from Point 5 and holds the task-relevant
propositions, participants, event relations, operators, provenance and unresolved elements
in a temporary working representation. It is not a long-term memory-writing algorithm.
An inferred recipient, for example, must remain tagged as inferred rather than becoming
indistinguishable from an explicitly stated argument.

The implementation is an 18-step adapter over the repository's existing
`ore_working_memory` subsystem (five stages and 40 canonical WMCC operations). It reuses
the established buffers, incomplete-structure tracking, semantic competition, executive
control, task-state management, and optional LIF reference rather than implementing a
second working-memory engine.

## Inputs and outputs

The input can be created from Point 5 using
`Point6Request.from_point5_result(sequence_id, point5_result)`. The factory carries
forward the literal proposition, discourse and situation context, common ground, speaker
intention hypotheses and unresolved alternatives. It extracts Point 5's surface sentences
unless the caller supplies an explicit text string.

Optional upstream syntax results, partial structures, candidate interpretations, task
sequences, goal features, and context cues can also be supplied. Missing evidence remains
unknown; the adapter does not manufacture a missing role filler or antecedent.

The output contains the active proposition/situation handoff, bounded working-memory
buffers, incomplete structures, semantic-resolution alternatives, executive-control state,
task state, operation traces, and a separate biological-reference summary. It explicitly
reports that no durable memory write occurred.

## The 18 ordered steps

| Step | Operation | Maintained output |
|---:|---|---|
| 1 | Receive the developing meaning representation | Point 5 literal/enriched tracks, text, source and context |
| 2 | Segment into proposition candidates | Tokens, proposition records and clause-grouping evidence |
| 3 | Establish entity and event identities | Coreference and situation-model entity/event identities |
| 4 | Bind predicates to arguments and roles | Proposition graph, thematic roles and unresolved role slots |
| 5 | Attach operators, modifiers and scope | Negation, tense/aspect, modality, quantifiers, modifiers and scope evidence |
| 6 | Preserve stated, inferred and assumed content | Literal/inferred separation, provenance and uncertainty |
| 7 | Link propositions into event/discourse structure | Temporal, causal, contrastive, conditional and referential links |
| 8 | Construct or update the situation model | Supported entities, events, states and situation relations |
| 9 | Integrate linguistic-semantic and event information | Evidence ledger and ranked semantic hypotheses |
| 10 | Allocate limited working-memory resources | Bounded buffers, activation priorities and unresolved-item importance |
| 11 | Maintain active information | Buffer activation, rehearsal and retention traces |
| 12 | Update as words and clauses arrive | Per-token order and explicit per-request update boundary |
| 13 | Protect representations from interference | Similarity/interference estimates and identity-sensitive buffers |
| 14 | Retrieve maintained information | Retrieval candidates and pending dependencies |
| 15 | Monitor proposition completeness | Filled/open slots and task-relative completion report |
| 16 | Compress and organize information | Chunks and bounded buffer structures with preserved cross-references |
| 17 | Recruit executive control when needed | Conflict, control-demand, task and attention state |
| 18 | Hand off the coherent active state | Propositions, situation links, provenance, uncertainty and reactivation pointers |

The 40 canonical working-memory operations are partitioned in order into 18 non-empty
groups. The adapter also exposes the engine's two additive source-extension operations:
`WMX-001` evaluates semantic prediction error only when prior predictions were supplied,
and `WMX-002` records temporary interpretation consolidation only when a candidate was
accepted. These extensions are attached to the later revision and handoff source steps,
respectively. Their outputs are not counted as additional canonical WMCC operations.

## Mathematical operations and computational meaning

### State reception and proposition organization

A useful update abstraction is:

`x_t=F(x_{t-1},u_t,c_t)`

where (u_t) is new linguistic input, (c_t) is relevant context, and (x_t) is the
maintained state. Proposition segmentation may be formulated as a boundary classifier:

`B_t=\sigma(w^T f_t+b)`

A proposition can be stored as a node in a graph
(G=(V,E,\lambda_V,\lambda_E)), with proposition, entity, event and state nodes and
labelled role/discourse edges.

### Identity and role–filler binding

For a mention (m) and candidate antecedent (e_j), evidence can be combined as:

`s_j=w_a A_j+w_g G_j+w_r R_j+w_d D_j+w_c C_j`

and converted to a graded candidate distribution:

`P(e_j\mid m)=softmax(s_j/\tau)`

Entropy (H=-\sum_j p_j\log p_j) expresses remaining ambiguity. A role–filler binding
can be modelled as (b_r=\phi(r,e)); tensor-product representations are one possible
formal device for preserving role and participant identity, not an established literal
neural code.

### Operators and provenance

Operators should have explicit targets and scope:
`operator_edge=(operator,target,scope)`.
The simple truth-valued sketch (NOT(P)=1-P) is not a general account of natural-language
negation, which interacts with scope, presupposition and uncertainty.

Provenance is stored independently from confidence. Labels include explicit, entailed,
inferred, presupposed, assumed and unresolved. A generic graded update is:

`q_{t+1}=q_t+\eta(e_t-q_t)`

This is a smoothing model for evidence strength, not a claim about a particular neural
learning rule.

### Situation integration and resource allocation

Situation updates can be expressed as:

`G_t=G_{t-1}\cup\Delta G_t,\quad S_{t+1}=T(S_t,E_t)`

Linguistic, semantic and event channels can be integrated without collapsing provenance:

`z=W_l l+W_s s+W_e e,\quad h=f(z)`

A priority score can combine relevance, unresolved importance, task relevance and dependency
importance:

`\pi_i=aR_i+bU_i+cT_i+dD_i`

A limited attention budget can be represented with normalized weights
`\alpha_i=softmax(\pi_i)` and `\sum_i m_i\le C`. These are engineering models, not
measured brain resource units.

### Maintenance, updating and interference

A leaky maintenance approximation is:

`x_{t+1}=\lambda x_t+u_t`

or, in continuous time:

`dx/dt=-x/\tau+u(t)`

A recurrent population model may use:

`\tau dx/dt=-x+W_{rec}f(x)+Bu(t)+\epsilon(t)`

The actual implementation uses bounded activation, decay and rehearsal in software and can
optionally run a population LIF reference. It does not assume persistent firing is the only
biological working-memory mechanism.

Similarity-based interference can be approximated as:

`sim(x_i,x_j)=\frac{x_i^T x_j}{\|x_i\|\|x_j\|}`

and retrieval cues may combine similarity, role fit and context:

`r_i=\beta_1 sim(q,x_i)+\beta_2 role(q,x_i)+\beta_3 context(q,x_i)`

Task-relative completeness can be summarized as:

`K(P)=\frac{\sum_j w_j c_j}{\sum_j w_j}`

Chunking can be framed as a loss-aware compression objective:

`\min_{\widetilde M} D(M,\widetilde M)+\lambda Cost(\widetilde M)`

Control demand can rise with open dependencies, interference, ambiguity and task load:

`D_t=aN_{open}+bI+cA+dT`

No universal fixed capacity is assumed.

## Biological-reference track

The existing eight-population current-based LIF simulator is reused:
phonological loop, lexical maintenance, syntactic workspace, semantic workspace, parietal
attention, prefrontal control, conflict monitor and task-set network. The 40 canonical
operation records are mapped to 18 source-level biological-reference groups.

The LIF reference uses a forward-Euler membrane update, recurrent excitation, global
inhibition, threshold/reset behavior and refractory periods. Its outputs include simulated
voltage (mV), current (pA), time (ms), firing rate (Hz) and spike counts.

The population labels are functional analogies, not one-to-one anatomical assignments.
The simulation is not an anatomically fitted model, physiological validation, or measured
human EEG/fMRI/single-cell data. Biological working memory may involve persistent activity,
activity-silent mechanisms, reactivation and interaction with long-term memory; this adapter
does not settle that scientific debate.

## API example

```python
from ore_working_memory import WorkingMemoryConfig
from ore_language_comprehension import (
    Point6Request, Point6PropositionalWorkingMemory,
)

request = Point6Request(
    sequence_id="working-memory-001",
    text="Maya said that Omar had left.",
    partial_structures=[{
        "structure_id": "embedded-clause",
        "label": "embedded proposition",
        "expected_slots": ["content"],
        "filled_slots": ["content"],
    }],
)
result = Point6PropositionalWorkingMemory(
    WorkingMemoryConfig(simulate_biology=False)
).process(request)
assert len(result.functional_trace) == 18
print(result.handoff)
```

For the previous stage, pass its actual result to
`Point6Request.from_point5_result(sequence_id, point5_result)`. Explicit text, syntax,
task sequence, goals, unresolved structures and additional context can override or extend
the extracted values.

## Annotation registration and tests

`register_point06_annotations` indexes `ore_language_comprehension` and
`ore_working_memory` in the existing `ore_logger.annotations` SQLite table at
`ore_logs/ore_logger.sqlite3`. No separate Point 6 database is created.

Run:

```bash
python -m pytest tests/test_language_comprehension_point06.py -q
```

## Limitations

- The adapter processes one supplied sequence per request; it does not maintain a persistent
  online state across independent calls.
- The underlying engine is deterministic and evidence-driven, not a trained general-purpose
  parser or a complete cognitive model.
- Neutral values are explicit fallbacks when a score is not supplied; they are not biological
  or empirical probability estimates.
- The working-memory episode and temporary accepted-interpretation record are not durable
  episodic or semantic learning.
- A bounded software buffer is an engineering abstraction, not a universal human memory
  capacity or literal neural storage slot.
- Executive control is conditional on estimated demands, not assumed to be equally engaged
  for every sentence.
