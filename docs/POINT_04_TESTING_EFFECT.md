# Point 04 — Testing Effect

## Scope

\`ore_testing_effect\` implements the source algorithm in \`Point_04_Testing_Effect_Brain_Algorithm.md\`. It treats retrieval as both a measurement of currently accessible knowledge and a learning event that can influence subsequent access, error correction, cue diversity, and delayed retention.

The implementation keeps four evidence categories separate:

1. The first retrieval attempt, which is recorded before any feedback.
2. Corrective feedback and the verified representation, if one is available.
3. The second or alternative-cue attempts, which test immediate correction and route flexibility.
4. Delayed attempts, which provide separate evidence about later retention.

The package does not assume that every failed attempt improves memory, that immediate retest success proves mastery, or that a fluent answer is necessarily correct.

## Package layout

- \`contracts.py\`: immutable target, attempt, feedback, and request contracts; persistent learner state and result objects.
- \`registry.py\`: the 22-stage \`P04-01\` through \`P04-22\` source-ordered operation registry, including mathematical forms, biological mappings, inputs, outputs, and evidence cautions.
- \`functional.py\`: cue alignment, weighted coverage, relation and order scoring, error classification, feedback selection, calibration, accessibility estimates, action selection, quality guards, graph updates, delayed retention, and practice priorities.
- \`biology.py\`: recurrent population activity, hippocampal-style pattern completion, prefrontal retrieval control, anterior-cingulate-inspired discrepancy monitoring, a neuromodulatory gate, and cortical associative plasticity.
- \`pipeline.py\`: the end-to-end coordinator plus SQLite annotation registration and operation-event recording.
- \`annotations.py\`: the Point 04 entry point for the repository's shared AST annotation catalogue.
- \`examples/point04_testing_effect.py\`: a reproducible example.
- \`tests/test_testing_effect_point04.py\`: contract, functional, biological, pipeline, and SQLite tests.

## The 22 operations

| Stage | Operation | Implemented behavior |
|---|---|---|
| 1 | Define target | Register content units, relations, order, conditions, weights, and a fixed criterion |
| 2 | Select cue | Estimate relevance, leakage, ambiguity, and cue utility |
| 3 | Establish retrieval condition | Record whether the answer was hidden and the attempt was unaided or assisted |
| 4 | Activate memory search | Reactivate candidates from a partial cue and manage competition |
| 5 | Reconstruct response | Preserve direct recollections separately from inferred components |
| 6 | Organize working memory | Inspect content, relation, sequence, and condition coverage |
| 7 | Monitor confidence | Compare confidence with evidenced accuracy and track prediction discrepancy |
| 8 | Commit original response | Store the first response immutably before correction |
| 9 | Validate criterion | Gate definitive error judgments on source reliability |
| 10 | Compare response | Score components, relations, order, and conditions separately |
| 11 | Diagnose error | Estimate omission, substitution, relation, sequence, condition, cue, and initial-learning explanations |
| 12 | Select feedback | Choose confirmation, targeted explanation, contrast, or re-encoding |
| 13 | Correct representation | Use only verified feedback or sufficiently supported retrieval evidence for plasticity updates |
| 14 | Retest | Evaluate a new attempt after hiding feedback and compute immediate gain |
| 15 | Vary cues | Test alternative valid cues and estimate route coverage and transfer |
| 16 | Update access | Update cue-specific EMA and Beta-binomial estimates from independent attempts |
| 17 | Select next action | Compare retrieval, feedback, re-study, transfer, delay, and pause utilities |
| 18 | Guard test quality | Flag cue mismatch, leakage, ambiguity, guessing, fatigue-related concerns, and confidence illusion |
| 19 | Update knowledge graph | Strengthen conceptual relations only when the evidence criterion is met |
| 20 | Test delayed retention | Keep delayed accuracy distinct from immediate retest gain |
| 21 | Prioritize practice | Combine inaccuracy, uncertainty, recurring errors, and transfer weakness |
| 22 | Handoff and audit | Return an auditable result and a suggested delayed-practice handoff to Point 03 |

## Mathematical models

The implementation makes the source document's proposed computations explicit and testable:

- **Cue alignment:** lexical Jaccard alignment and cosine similarity for vector representations.
- **Competition:** stable softmax normalization of candidate activations and diagnostic scores.
- **Population activation:** logistic activation of weighted input plus recurrent activity and inhibition.
- **Coverage:** importance-weighted component and relation ratios; optional pairwise sequence-order agreement and condition scoring.
- **Discrepancy:** target-versus-response gaps and confidence-versus-evidenced-accuracy error.
- **Error classification:** normalized candidate probabilities with a retained diagnostic confidence.
- **Feedback emphasis:** importance, uncertainty, recurrence, and confidence-sensitive error weighting.
- **Contrastive/associative plasticity:** bounded, gated Hebbian/Oja-style weight updates, applied only when evidence is sufficiently verified.
- **Accessibility:** exponential moving-average update \(p_{t+1}=p_t+\eta(r_t-p_t)\), alongside a Beta-binomial posterior for independent outcomes.
- **Calibration:** Brier score over independent retrieval observations only.
- **Test quality:** relevance, validity, clarity, ambiguity, and answer leakage.
- **Future priority:** a weighted score over inaccuracy, uncertainty, recurring errors, and transfer weakness.
- **Action policy:** choose the next action using a utility score adjusted for fatigue and remaining time.

The numerical weights are transparent engineering defaults, not universal psychological constants. They need calibration for learner, content, language, cue type, and task.

## Biological computational surrogates

The biology module coordinates candidate contributors that overlap in real learning:

- **Hippocampal pattern completion:** stores a target feature vector and estimates reinstatement from a partial cue.
- **Distributed population activity:** recurrent rate-based activations represent integration and competition over feature patterns.
- **Prefrontal retrieval control:** models goal focus, cue biasing, and reduction of irrelevant competition.
- **Anterior-cingulate-inspired monitoring:** computes model-level discrepancy and candidate-conflict diagnostics.
- **Neuromodulatory gate:** uses outcome quality, surprise, relevance, effort, and verified feedback to modulate learning.
- **Cortical associative plasticity:** updates a bounded association matrix using an Oja-like correction to Hebbian co-activity.

These are inspectable functional surrogates. They are not simulations of real neurons, do not estimate neurotransmitter concentrations, and do not establish that the living brain calculates these exact equations. Hippocampal, cortical, prefrontal, monitoring, and neuromodulatory roles overlap and vary with task and memory type.

## Scoring and evidence safeguards

### Explicit response evidence

The target's \`components\` and \`relations\` are rubric units. A caller may provide exact \`recalled_components\`, \`recalled_relations\`, or explicit status mappings (\`correct\`, \`missing\`, \`substitution\`, \`uncertain\`, \`contradicted\`; relations additionally support \`reversed\` and \`incorrect\`). This is preferable to treating raw word overlap as semantic understanding.

When explicit rubric evidence is absent, the package uses conservative lexical-overlap heuristics and marks the result as an estimate. It cannot independently determine whether a free-text explanation is scientifically correct. A trusted answer key, instructor judgment, validated parser, or another semantic evaluator must supply the criterion and/or evidence labels for high-stakes scoring.

### Original response and correction

The initial \`RetrievalAttempt\` is a frozen dataclass. The pipeline stores it before feedback, then records the corrected retest separately. It never overwrites the first response with the correction.

An unverified answer key cannot trigger definitive diagnosis or verified relation strengthening. A confident wrong answer receives a dedicated warning rather than being mistaken for strong knowledge.

### Recall, recognition, and assistance

\`aid_level=0\` and \`guessed=False\` are required for an attempt to count as independent retrieval evidence in the accessibility posterior and confidence calibration. Assisted, hinted, or guessed responses remain in the record but are not pooled as independent recall successes. Recognition should be represented through an appropriate cue type and must not be interpreted as free recall.

### Immediate versus delayed learning

The second retrieval measures immediate correction. Only \`delayed_attempts\` populate delayed-retention estimates. The package intentionally does not claim that one immediate improvement demonstrates durable retention.

### Handoff to Point 03

The output includes a \`review_handoff\` payload with a suggested delay and priority. The calendar and interval algorithm remain owned by Point 03 Spaced Repetition; Point 04 does not duplicate that scheduler or silently schedule external reminders.

## SQLite annotations and operation logging

\`register_annotations(database_path)\` uses the shared \`ore_logger.catalog.sync_repository\` AST indexer with \`include_paths={"ore_testing_effect"}\`. It registers module, class, and function definitions in the same SQLite annotation catalogue as the other ORE packages.

\`record_result_to_sqlite(result, database_path)\` first synchronizes the Point 04 annotations, creates one logger run, and stores exactly 22 ordered \`operation\` events. Every event includes its \`P04\` identifier, operation name, mathematical form, functional output, biological diagnostics, and evidence-level caution.

## Basic usage

\`\`\`python
from ore_testing_effect import (
    Feedback,
    LearningTarget,
    RetrievalAttempt,
    TestingEffectProcessor,
    TestingEffectRequest,
    record_result_to_sqlite,
)

target = LearningTarget(
    target_id="testing-effect-definition",
    prompt="Explain why retrieval practice can improve later retention.",
    components=(
        "active generation from memory",
        "improved later accessibility",
        "retrieval reveals knowledge gaps",
        "benefits depend on learning conditions",
    ),
    relations=(
        "retrieval practice requires generating information without direct exposure",
        "retrieval attempts can improve later access under suitable conditions",
    ),
    component_weights={
        "active generation from memory": 1.0,
        "improved later accessibility": 1.0,
        "retrieval reveals knowledge gaps": 0.8,
        "benefits depend on learning conditions": 1.0,
    },
    criterion_source="validated_lesson_answer",
    criterion_reliability=0.95,
)

initial = RetrievalAttempt(
    cue="Explain why actively trying to recall a learned idea may improve later memory.",
    response="It makes the learner generate the information and can improve later access.",
    confidence=0.72,
    recalled_components=(
        "active generation from memory",
        "improved later accessibility",
    ),
    recalled_relations=(
        "retrieval practice requires generating information without direct exposure",
    ),
)

feedback = Feedback(
    correct_representation=(
        "Retrieval practice makes learners generate information from memory; "
        "it can improve later access and expose gaps, but benefits depend on conditions."
    ),
    explanation="A failed attempt is not guaranteed to strengthen memory; correct errors before retesting.",
    source="validated_lesson_answer",
    source_reliability=0.95,
    verified=True,
)

processor = TestingEffectProcessor()
result = processor.process(TestingEffectRequest(
    target=target,
    initial_attempt=initial,
    feedback=feedback,
    current_time_utc="2026-01-01T00:00:00Z",
    delay_after_hours=72.0,
))
print(result.output)
# Optional: save 22 operations and AST annotations to the shared ORE SQLite logger.
# record_result_to_sqlite(result, "ore_logs/point04.sqlite3")
\`\`\`

Run the example with:

\`\`\`bash
python examples/point04_testing_effect.py
\`\`\`

Run the focused tests with:

\`\`\`bash
python -m pytest -q tests/test_testing_effect_point04.py
\`\`\`

The full repository test suite remains the integration check.
