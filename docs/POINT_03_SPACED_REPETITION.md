# Point 03 — Spaced Repetition

## Scope

\`ore_spaced_repetition\` implements the 12-step algorithm in \`Point_03_Spaced_Repetition_Brain_Algorithm.md\`. It models an external review scheduler together with coarse, biologically inspired processes related to encoding, reactivation, activity-dependent plasticity, and offline consolidation.

This package is deliberately separate from Point 04, **Testing Effect**. Retrieval appears here only as a review operation needed for spaced practice. It does not attempt to implement the full testing-effect analysis, error taxonomy, or testing programme.

## Package layout

- \`contracts.py\`: validated learning items, review observations, offline-context records, configuration, memory records, learner state, and result contracts.
- \`registry.py\`: the canonical \`P03-01\` through \`P03-12\` source-ordered operation registry.
- \`functional.py\`: timestamp arithmetic, encoding-quality estimation, stability initialization, exponential/power-law accessibility estimates, review decisions, retrieval scoring, state updates, adaptive intervals, maintenance assessment, consolidation influence, and calibration metrics.
- \`biology.py\`: recurrent population activity, hippocampal-style item binding/reactivation, cortical spacing-sensitive plasticity, prefrontal retrieval control, a learning gate, and an offline-context monitor.
- \`pipeline.py\`: the 12-stage coordinator and SQLite annotation/execution-event recorder.
- \`annotations.py\`: AST annotation registration through the shared ORE SQLite catalogue.
- \`tests/test_spaced_repetition_point03.py\`: scheduling, outcome-sensitive updates, recognition/familiarity protection, maintenance, calibration, biological diagnostics, and SQLite tests.

## Three separate state concepts

The package keeps these concepts distinct:

1. **Accessibility:** a model estimate of how available an item may be at a particular time.
2. **Stability:** a coarse, item-specific timescale updated from the review history.
3. **Scheduling:** an external decision that calculates a future due timestamp.

None of these variables is a direct measurement of a single neural property. The calendar and due time are software state, not a claim that the brain maintains a review calendar.

## Mathematical operations represented

The functional layer implements:
- UTC timestamp normalization and elapsed-time arithmetic;
- weighted encoding-quality estimates with missing-data uncertainty;
- exponential and power-law accessibility functions;
- a recall-probability target band and due/deadline policy;
- separate accuracy, completeness, relation, fluency, effort, and hint fields;
- outcome-specific stability updates;
- a stability-based interval estimate \`I=-S ln(p*)\`, with difficulty, hints, and retention-horizon constraints;
- repeated-success maintenance criteria;
- optional evidence-gated offline adjustment;
- Brier score and signed prediction-error calibration.

The default numerical parameters are explicit software starting values, not universal scientific constants. They need calibration for the learner, content type, task, cue mode, and retention horizon. The package does not assume that an expanding schedule is always optimal or that one decay function describes all memory.

## Biological computational surrogates

The biological module includes:
- a recurrent neural-population activity model;
- hippocampal-style pattern binding and partial cue reactivation;
- a cortical co-activity update modulated by the interval between learning episodes;
- prefrontal retrieval-control and cue-support modulation;
- an illustrative learning gate;
- an offline-consolidation context monitor.

The models are inspectable engineering surrogates, not biophysical or neuron-level simulations. Their weights and activity values are not measured human neural data.

## Safeguards

1. **Recognition is not independent recall.** Recognition-only outcomes do not count as independent success and do not by themselves reset the meaningful-learning timestamp.
2. **First responses are retained.** Review records preserve the initial learner response, cue level, aid condition, and whether feedback was supplied.
3. **Corrected relearning is not unaided success.** The event type and feedback status remain explicit.
4. **Failure shortens the next interval.** Outcome-specific factors and updated stability make weak retrieval lead to a nearer next review.
5. **Mastery remains provisional.** A stability label requires multiple independent successes across separated reviews and does not claim permanent retention.
6. **No automatic sleep boost.** When offline/sleep evidence is absent, the stability estimate is unchanged by the consolidation stage. An optional adjustment is small, configurable, and gated by the supplied observation and evidence confidence.
7. **Uncertainty is retained.** Missing encoding measures and inconsistent review outcomes increase model uncertainty.
8. **Predicted recall is calibrated behaviorally.** Brier score and prediction error compare predictions with independent binary recall outcomes; they do not reveal a neural state.

## Basic usage

\`\`\`python
from ore_spaced_repetition import (
    LearningItem, ReviewObservation,
    SpacedRepetitionProcessor, SpacedRepetitionRequest,
)

item = LearningItem(
    item_id="synapse-definition",
    content="A synapse is a specialized junction through which one neuron communicates with another cell.",
    components=("specialized junction", "communication", "neuron", "another cell"),
    difficulty=0.45,
    target_retention_hours=24 * 30,
    encoding_measurements={
        "attention": 0.9, "comprehension": 0.85,
        "relational_organization": 0.8, "meaningful_practice": 0.75,
    },
)

processor = SpacedRepetitionProcessor("learner-1")
first = processor.process(SpacedRepetitionRequest(
    "learner-1", item, "2026-01-01T00:00:00Z"
))
print(first.memory_record["next_due_at_utc"])

# At the next review, submit an observed result rather than assuming success.
review = ReviewObservation(
    occurred_at_utc=first.memory_record["next_due_at_utc"],
    outcome="partial",
    accuracy=0.8, completeness=0.65, relation_accuracy=0.7,
    effort=0.75, response_mode="free_recall",
)
updated = processor.process(SpacedRepetitionRequest(
    "learner-1", item, review.occurred_at_utc, review_observation=review
))
print(updated.output)
\`\`\`

A complete reproducible example is available at:

\`\`\`bash
python examples/point03_spaced_repetition.py
\`\`\`

## SQLite annotation and operation logging

\`register_annotations(database_path)\` indexes each class and callable in \`ore_spaced_repetition/\` through the existing AST catalogue. \`record_result_to_sqlite(result, database_path)\` stores all 12 ordered operation events in the shared \`runs\` and \`events\` tables. Each event retains its operation identifier, equations, functional outputs, biological diagnostics, and evidence-level caution.

## Validation

The tests cover:
- exact 12-stage registry order;
- nonzero initial intervals and multiple forgetting curves;
- delayed recall success, partial recall, failure, corrective relearning, and recognition-only handling;
- shortened intervals after weak retrieval;
- repeated independent success before provisional maintenance;
- optional/no-op offline consolidation;
- prediction calibration;
- bounded biological surrogate activity;
- learner state reuse across multiple items;
- SQLite annotation and 12-event persistence.

The full repository GitHub Actions checks remain the authoritative integration validation.
