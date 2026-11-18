# Point 12 — Context-Dependent Interpretation

This package implements the 16 operations in the Point 12 research specification:
**Context-Dependent Interpretation: Prior Knowledge, Goals, and Situation Shape Meaning**.

## Processing sequence

1. Encode the incoming linguistic or environmental event.
2. Retrieve the immediately preceding discourse state.
3. Retrieve relevant long-term knowledge.
4. Estimate the current task goal.
5. Extract physical and perceptual scene context.
6. Retrieve cultural, conventional, and speaker-linked knowledge.
7. Construct a provisional situation model.
8. Generate candidate interpretations, retaining an open-set alternative.
9. Estimate context-conditioned priors.
10. Score direct input evidence separately from contextual expectations.
11. Combine prior and likelihood, then preserve uncertainty where warranted.
12. Integrate supported interpretations and explicit claims into the situation model.
13. Detect contradictions while retaining provenance for both sources.
14. Revise candidate probabilities when new explicit evidence conflicts with an interpretation.
15. Adapt context-source weights conservatively after confident outcomes.
16. Produce a task-appropriate interpretation, alternative set, or clarification request.

The canonical source-ordered registry is exposed as `OPERATION_REGISTRY`; every result includes one operation trace entry per stage.

## Evidence and uncertainty safeguards

- Incoming evidence, retrieved knowledge, discourse, scene cues, conventions, and generated hypotheses retain distinct provenance labels.
- A contextual prior is not allowed to replace direct input evidence. The two are scored independently and combined using a Bayesian-style update.
- The open-set `unknown` alternative is retained whenever explicit candidate alternatives are supplied.
- Ambiguity is preserved when posterior confidence, the top-two margin, or normalized entropy fails configured selection criteria.
- Contradictory input is recorded as a conflict; an unusual statement is not automatically rejected as false.
- Candidate-derived facts are marked as inferred. They are not labelled as directly observed or explicitly stated.
- Context-source adaptation is bounded and happens only after a confident candidate selection.

## Mathematical model

For candidate interpretation (h), current input (x), context (C), and goal (G):

[
P(h mid x,C,G) propto P(x mid h,C)P(h mid C,G)
]

The implementation obtains a contextual prior from relevance-weighted evidence, estimates input likelihood separately, and normalizes their product. Entropy and posterior margin are used as uncertainty diagnostics. Similarity, recency decay, softmax, and explicit conflict penalties are engineering abstractions; they are not claims that neurons literally calculate these expressions.

## Biological interpretation

The `biology.py` classes are interacting functional surrogates for language encoding, hippocampal discourse retrieval, temporal-semantic activation, goal maintenance, perceptual scene processing, situation-model integration, and controlled selection. Their boundaries are not one-region/one-function claims, anatomical replicas, or digital twins of neural tissue.

## API quick start

```python
from ore_context_dependent_interpretation import (
    CandidateHypothesis,
    ContextDependentInterpretationProcessor,
    ContextEvidence,
    ContextState,
    InputEvent,
)

state = ContextState(
    discourse_history=[
        ContextEvidence(
            evidence_id="previous-turn",
            source_type="discourse",
            content="I need to withdraw money from my account.",
            concepts=("account", "money", "withdrawal"),
            goal_tags=("financial_task",),
        )
    ],
    current_goal="financial_task",
)

event = InputEvent(
    event_id="sentence-1",
    content="The bank is closed.",
    candidates=(
        CandidateHypothesis("finance", "financial institution",
                            cue_terms=("money", "account", "deposit")),
        CandidateHypothesis("river", "river bank",
                            cue_terms=("river", "shore", "water")),
    ),
)

result = ContextDependentInterpretationProcessor(state=state).process(event)
print(result.output)
```

## SQLite annotations and operation logging

`register_annotations(database_path)` indexes class and function docstrings/signatures in the ORE annotation catalogue. `record_result_to_sqlite(result, database_path)` indexes the package and writes the 16 operation-trace entries to the ORE logger's `runs` and `events` tables.

The test suite exercises context reversal, input-versus-context separation, ambiguity preservation, explicit contradiction handling, source provenance, biological diagnostics, and SQLite annotation/event persistence.
