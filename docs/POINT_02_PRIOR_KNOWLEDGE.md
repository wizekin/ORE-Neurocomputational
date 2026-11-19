# Point 02 — Connect to Prior Knowledge

## Scope

\`ore_prior_knowledge\` implements the 14-stage algorithm in \`Point_02_Connect_to_Prior_Knowledge_Brain_Algorithm.md\`. It integrates new material with a learner's existing semantic nodes, episodic traces, weighted relations, and schema candidates. Its core safeguard is to avoid treating the most familiar or most activated memory as necessarily relevant or correct.

The package is separate from Point 01 Active Engagement. Point 01 handles goal-directed engagement, questioning, evidence attention, self-explanation, and the active-learning episode. Point 02 consumes a new information unit and connects it to the learner's existing knowledge network.

## Package map

- \`contracts.py\`: validated input units, explicit relations, graph nodes and edges, schemas, configuration, persistent learner state, and result contract.
- \`registry.py\`: source-ordered \`P02-01\` through \`P02-14\` operation registry with purposes, equations, inputs/outputs, biological interpretations, and evidence cautions.
- \`functional.py\`: deterministic feature extraction, cue construction, bounded graph spreading, retrieval scoring, comparison, relation alignment, analogy limits, schema mismatch and novelty estimates, strategy selection, retrieval route construction, and route checks.
- \`biology.py\`: recurrent population activity, hippocampal-style pattern completion and episode binding, cortical associative learning, medial-prefrontal schema selection, frontoparietal retrieval control, and a bounded novelty/reliability gate.
- \`pipeline.py\`: 14-stage coordinator and integration with the shared SQLite annotation/event logger.
- \`annotations.py\`: package-specific entry point for AST registration of every package class and callable.
- \`tests/test_prior_knowledge_point02.py\`: functional, contract, state reuse, numerical, biology, conflict-preservation, retrieval-route, and SQLite tests.

## Mathematical operations represented

The functional implementation uses deterministic signed-hash feature vectors, Jaccard set overlap, cosine similarity, weighted candidate scores, bounded spreading activation, temperature-scaled softmax probabilities, Shannon entropy, surprisal, relation-pattern matching, and bounded associative updates. Schema updates are reliability-gated and versioned. Conflicting or novel input is retained as an episodic item rather than silently incorporated as a semantic rule.

The biological surrogate uses recurrent population dynamics of the form:

\`\`\`text
activity[t+1] = tanh(gain * input + W_recurrent @ activity[t] - inhibition)
\`\`\`

Hippocampal-style pattern completion is represented by similarity-based reinstatement from partial vector cues. Cortical associative strengths use a bounded co-activation update. Medial prefrontal schema selection uses normalized competition. A salience/reliability gate controls modeled association learning.

These equations are software-level models. They do not imply that the brain explicitly computes the same variables, uses these exact equations, or stores full symbolic graphs in a single anatomical region.

## Important safeguards and limits

1. **Provenance stays visible.** New relations retain whether they were observed, reported, inferred, or hypotheses. Generic links created by the pipeline are explicitly marked \`inferred\`.
2. **Analogy is not identity.** Mapping records preserve unmatched relations and state that transferred inferences require independent evidence.
3. **Conflict is not auto-corrected away.** Strongly schema-inconsistent or explicitly negated inputs are stored separately or trigger a revision proposal. Existing schemas are not overwritten automatically.
4. **Low evidence defers broad generalization.** Reliability and schema thresholds gate integration and updates.
5. **Repeated baseline inputs do not roll back learned state.** A newer schema version remains in place when the same older seed schema is supplied again.
6. **Retrieval route validation is a software simulation.** It checks that cue-to-node routes have been constructed; it is not a test of a human learner's actual recall.
7. **Consolidation is a computational surrogate.** The package records reactivation/association updates but does not simulate sleep, schedule future work, or claim biological systems-consolidation fidelity.
8. **Text processing is deliberately inspectable.** Explicit entities, features, and relations are preferred. Fallback extraction is deterministic and heuristic, not a full natural-language understanding system.
9. **Biology is not neuron-level simulation.** Population states and associative weights are model diagnostics, not measured neuronal activity.

## Basic usage

\`\`\`python
from ore_prior_knowledge import (
    LearningUnit, PriorKnowledgeRequest, PriorKnowledgeProcessor,
)

request = PriorKnowledgeRequest(
    learner_id="learner-1",
    new_input=LearningUnit(
        unit_id="lesson-1",
        text="A cell membrane selectively regulates movement of substances.",
        entities=("cell membrane", "substances"),
        features={"membrane": 1.0, "selective": 0.9, "transport": 0.8},
    ),
    current_context={"topic": "biology", "goal": "understand membranes"},
    # Supply learner-specific KnowledgeNode, KnowledgeRelation and KnowledgeSchema
    # records to retrieve and compare against prior knowledge.
)

result = PriorKnowledgeProcessor("learner-1").process(request)
print(result.output)
print([step["operation_id"] for step in result.operation_trace])
\`\`\`

For a full example with a prior knowledge graph, schema, and a familiar analogy, run:

\`\`\`bash
python examples/point02_prior_knowledge.py
\`\`\`

## SQLite annotations and operation history

\`register_annotations(database_path)\` uses \`ore_logger.catalog.sync_repository\` to index every class and callable in \`ore_prior_knowledge/\`. \`record_result_to_sqlite(result, database_path)\` registers those definitions and writes 14 ordered operation events to the shared ORE \`runs\` and \`events\` tables. Each event includes its operation ID, functional output, biological diagnostics, mathematical form, and evidence-level caveat.

## Validation

The test suite checks the 14-stage registry, normalized retrieval probabilities, candidate ranking, analogy boundaries, schema contradiction handling, separate episodic encoding, state reuse, repeat-input identity, bounded neural-population activity, and SQLite annotations/event persistence. The complete repository CI remains the final authority for dependency and integration validation.
