# Point 13 — Emotion and Motivation: Modulating Attention, Learning, and Memory

This package implements the 16 source-ordered operations in `Point_13_Emotion_Motivation_Attention_and_Memory_Encoding.md`.

## Processing sequence

1. Encode incoming event features and their relations.
2. Estimate affective significance, keeping valence and arousal distinct.
3. Update fast arousal and slower stress-related state separately.
4. Compute bottom-up and goal-directed attentional priority under a finite resource budget.
5. Apply state-dependent gain and divisive normalization.
6. Encode attended features and their relations into a selective episodic trace.
7. Estimate task-specific control capacity under stress and fatigue.
8. Estimate reward value relative to effort and resource cost.
9. Update action/state values from observed outcomes using a temporal-difference prediction error.
10. Select effort and persistence according to expected progress and cost.
11. Elaborate new information through relevant associative links when motivated effort is justified.
12. Apply a time-limited, arousal-dependent consolidation update, separate from initial encoding.
13. Retrieve prior memories by cues while distinguishing vividness, confidence, and accuracy.
14. Measure attention concentration, central/peripheral retention, and contextual completeness.
15. Reappraise outcomes and update future cue associations without overwriting the stored episode.
16. Retain the episode and cautiously update generalized knowledge through shared features.

The exact source order is exposed by `OPERATION_REGISTRY`. Every processing result contains exactly 16 operation-trace entries.

## Mathematical operations

The model implements feature-vector extraction, temporal/state smoothing, valence and arousal appraisal, weighted salience/goal/context priority, softmax competition, finite-budget allocation, inverted-U gain, divisive normalization, feature-weighted encoding, utility-minus-cost selection, temporal-difference reward prediction error, associative learning, cue similarity, softmax retrieval, Shannon entropy, central/peripheral ratios, context-completeness measurements, time-limited consolidation, Bayesian-style reappraisal framing, and confidence-weighted concept updates.

Equations and scalar parameters are engineering abstractions. They are not direct estimates of neural firing, norepinephrine concentration, cortisol concentration, or dopamine release.

## Biological functional surrogates

The `biology.py` module coordinates interacting surrogates for:
- Amygdala/BLA-style significance appraisal and memory modulation.
- Locus-coeruleus norepinephrine adaptive gain.
- HPA-axis slow stress/glucocorticoid load.
- Dopaminergic/striatal reward valuation and prediction-error learning.
- Hippocampal episodic binding and consolidation.
- Prefrontal goal maintenance and effort allocation.
- Cingulo-opercular progress/error monitoring.
- Distributed cortical semantic association and elaboration.

The components are not anatomical replicas or a one-region/one-function model. Fast NE and slower HPA-like states are explicitly distinct. The amygdala modulates memory priority and consolidation; it is not treated as the storage site for the entire episode.

## Selective encoding and uncertainty safeguards

- Central, peripheral, and context details can receive different attention and encoding strength.
- Higher arousal is not assumed to improve all cognitive operations monotonically.
- Valence, arousal, significance, reward value, effort cost, encoding, consolidation, vividness, confidence, and accuracy are represented separately.
- Emotional strength is not an accuracy guarantee.
- Consolidation is distinct from initial encoding and includes a time-limited modulation proxy and replay-support input.
- Reward prediction error is updated only when an observed outcome is supplied.
- Generalized concept weights update only through feature/tag overlap; episodic records remain separately available.
- Physiology-related variables are clearly labeled as simulation proxies rather than measured quantities.

## SQLite logging

`register_annotations(database_path)` indexes all Point 13 Python classes and functions in the ORE logger's annotation catalogue. `record_result_to_sqlite(result, database_path)` persists all 16 operation events in the logger's `runs` and `events` tables.

## Quick start

```python
from ore_emotion_motivation import (
    EmotionMotivationMemoryProcessor, EventFeature, LearningExperience,
)

processor = EmotionMotivationMemoryProcessor()
experience = LearningExperience(
    event_id="example-1",
    goal="learn_a_concept",
    reward_relevance=0.8,
    features=(
        EventFeature("main_idea", salience=0.9, goal_relevance=1.0,
                     centrality=1.0, role="central", tags=("concept",)),
        EventFeature("background", salience=0.2, goal_relevance=0.2,
                     centrality=0.1, role="peripheral", tags=("context",)),
    ),
)
result = processor.process(experience)
print(result.output)
```

## Validation coverage

Tests exercise salient versus peripheral encoding, top-down goal relevance, moderate versus excessive arousal, non-monotonic control capacity, reward prediction errors, effort-cost decisions, elaborative links, separate consolidation, cue retrieval, vividness/confidence/accuracy distinction, reappraisal, provenance, and SQLite annotations plus all 16 logged operations.
