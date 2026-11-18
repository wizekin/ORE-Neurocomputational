# Point 18 — Attentional Failure: Distraction, Fatigue, and Missing Information

## Scope

This package implements the 17 ordered operations in the Point 18 Library specification. It models a task-sensitive process for detecting possible attentional lapses, identifying information that was missed or weakly encoded, recovering it from available evidence, auditing completeness, preserving uncertainty, and gating downstream comprehension.

It intentionally does not implement later failure mechanisms such as working-memory overload, schema-driven false inference, confirmation bias, heuristic error, ambiguous syntax, or misconception entrenchment.

## Architecture

- \`contracts.py\`: validated task requirements, attentional state, configuration, experience, and result contracts.
- \`functional.py\`: mathematical and functional operations for alertness, priority, sensory gain, working-memory admission, lapse detection, source-based recovery, confidence, and release gating.
- \`biology.py\`: distributed bounded population-rate surrogates for goal control, attention selection, sensory gain, arousal, monitoring, recovery, and memory re-encoding.
- \`registry.py\`: source-ordered operation registry \`P18-01\` through \`P18-17\`.
- \`pipeline.py\`: full pipeline, per-operation traces, SQLite catalogue synchronization, and event persistence.
- \`tests/test_attentional_failure_point18.py\`: functional, mathematical, safety, provenance, biological, and persistence tests.
- \`examples/attentional_failure.py\`: runnable source-backed recovery example.

## Seventeen source-ordered operations

1. Establish task goal and weighted information requirements.
2. Estimate alertness, fatigue, distraction, and attentional availability.
3. Detect competing external and internal inputs while preserving urgent safety signals.
4. Compute adaptive input priority from relevance, urgency, salience, novelty, selection history, and processing cost.
5. Bias sensory feature gain without deleting raw or low-priority source traces.
6. Gate information into working memory under a task-sensitive cost budget.
7. Track sustained attention with a leaky integrator and moving-window stability estimate.
8. Detect possible lapses from converging indicators and temporal hysteresis.
9. Reconstruct missing information only from available source records; do not use schema guesses as recovery evidence.
10. Audit required content, confidence, weighted coverage, and dependency gaps.
11. Select a feasible recovery action using expected benefit minus action cost.
12. Reorient to the current task goal and last verified processing position.
13. Re-encode recovered information with source and confidence provenance.
14. Verify repaired coverage and residual high-priority gaps.
15. Adapt future attention allocation using context-specific repeated feedback.
16. Preserve unresolved content as unknown and prevent unsupported completion.
17. Release a complete representation only when confidence, coverage, and critical requirements pass; otherwise return a marked partial or blocked result.

Every operation trace includes its operation ID, purpose, mathematical form, functional output, and biological population activity.

## Mathematical operations

- Cosine goal relevance, normalized requirement importance, and priority softmax.
- Logistic state/lapse estimates and multiplicative effective processing.
- Weighted distraction load and task-sensitive priority scores.
- Feature gain modulation and cost-budgeted greedy admission.
- Leaky integration and moving-window mean/variance.
- Hysteresis-based lapse detection.
- Missingness \(M_i=1-c_i\), weighted coverage, and conservative geometric confidence.
- Expected recovery benefit minus cost.
- Evidence-weighted re-encoding, residual gap score, bounded context-specific adaptation, and entropy-ready uncertainty records.
- Confidence/coverage/critical-gap release gate.

Coefficients and thresholds are inspectable model parameters, not universal psychological constants. They must be calibrated against task-specific behavioural data before predictive claims are made.

## Provenance and safety safeguards

- Sensory exposure is not treated as proof of adequate encoding.
- Low-priority inputs retain source pointers where possible.
- A missing element is reconstructed only from a supplied source record or an explicit repeated/clarified input; plausible schema completion is not accepted as evidence.
- Original input, recovered input, and inference are kept distinct.
- Unrecovered information remains unknown.
- Safety-critical or urgent information is not suppressed merely because it is salient or unexpected.
- No diagnosis, stable personal attention trait, or medical conclusion is inferred from a single task.
- Downstream modules receive either a verified representation, a clearly marked partial representation, or a blocked result.

## Biological modeling boundary

The network model uses bounded rate-coded updates with input gain, recurrence, inhibition, and decay. It is an inspectable distributed software surrogate. It is not a neural recording, biophysical simulation, or a one-region/one-operation anatomical claim. The network mapping is functional and task-dependent.

## SQLite annotations and operation logging

\`register_annotations(database_path)\` indexes Point 18 classes and callables in the ORE annotation catalogue.

\`record_result_to_sqlite(result, database_path)\` synchronizes the annotations and stores all 17 operation records in the existing ORE logger \`runs\` and \`events\` tables. Each event includes operation identity, formula, functional output, and biological output.

## Quick start

\`\`\`python
from ore_attentional_failure import (
    AttentionalExperience, AttentionalFailureProcessor, InformationRequirement,
)

event = AttentionalExperience(
    experience_id="example-1",
    task_goal="Understand the instruction",
    required_information=(
        InformationRequirement("actor", importance=1.0, critical=True),
        InformationRequirement("action", importance=1.0, critical=True),
    ),
    incoming_items=(
        {"key": "actor", "content": "the operator", "goal_relevance": 1.0},
        {"key": "action", "content": "stopped the process", "goal_relevance": 1.0},
    ),
    metrics={"alertness": 0.8, "fatigue": 0.1, "distraction_load": 0.1},
)
result = AttentionalFailureProcessor().process(event)
print(len(result.operation_trace))  # 17
print(result.release_decision["status"])
\`\`\`

## Validation plan

The focused suite covers stable attention, external/internal distraction, fatigue, priority and safety override, admission under budget, lapse hysteresis, source-based recovery, missing-information integrity, dependency-aware coverage, adaptive recovery, reorientation, re-encoding provenance, repair verification, context-specific adaptation, conservative release, biological recurrence, and SQLite logging.

The software tests establish code behaviour, not empirical human attention accuracy. The source research and published references remain the evidence basis for biological motivation; real-world claims require separate behavioural and neural validation.
