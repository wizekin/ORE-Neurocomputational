# Point 14 — Social Interaction and Conversational Learning

This package implements the 16 source-ordered operations in `Point_14_Social_Interaction_and_Conversational_Learning.md`.

## Processing sequence

1. Establish a learning episode with learner, partner, shared environment, goal, and multimodal interaction stream.
2. Detect and prioritize social partner cues such as speech, gaze, face, gesture, timing, and action.
3. Estimate joint attention to a shared referent while preserving uncertain or conflicting focus.
4. Infer communicative intention and probabilistic partner-knowledge states.
5. Integrate speech, prosody, gaze, gesture, facial signals, and action with modality provenance retained.
6. Ground words and relations to shared-scene referents and incrementally update word–referent associations.
7. Predict conversational turn state (continue, yield, repair) and response readiness.
8. Adapt response vocabulary and detail to partner knowledge and communicative utility.
9. Detect mismatch and misunderstanding, reopen alternatives, and initiate clarification/repair.
10. Learn from informational correction, task outcomes, and social-affective feedback as separate channels.
11. Adapt scaffold level to competence–difficulty gap and fade support after success.
12. Evaluate self-explanation coverage, relation completeness, gaps, and retrieval-strength updates.
13. Compare perspectives with source-reliability weighting while retaining disagreements and source dependence.
14. Construct a temporal/causal narrative graph and agent-specific goal/knowledge states.
15. Bind social learning into episodic memory containing content, partner/source, context, modality, feedback, and confidence.
16. Update partner models cautiously and generalize across partners only when a pattern is repeated and sufficiently consistent.

The exact ordered registry is exposed as `OPERATION_REGISTRY`. Every result has one trace entry for each of the 16 operations.

## Mathematical operations

Implemented model-level computations include:
- Weighted social attention and softmax cue selection.
- Bayesian-style referent cue integration and logistic joint-attention confidence.
- Inverse-planning intention inference and KL divergence as a belief-mismatch proxy.
- Reliability/precision-weighted multimodal fusion and temporal alignment windows.
- Word-to-referent scoring and prediction-error association updates.
- Turn-state softmax and a pause-duration hazard proxy.
- Audience design through expected shared information minus cost/ambiguity.
- Response mismatch residuals and negative-evidence scoring.
- Separate social reward/action-value learning and factual belief updates.
- Competence–difficulty gap and bounded scaffold allocation.
- Explanation coverage, relation coverage, and retrieval-strength updates.
- Reliability-weighted perspective comparison and Bayesian/evidence framing.
- Typed temporal-causal narrative graphs and surprisal-oriented event prediction.
- Social memory priority, source binding, time-dependent consolidation, and cue retrieval.
- Cautious partner adaptation and cross-partner generalization counts.

These formulas are computational abstractions. They are not literal neuronal arithmetic or direct measurements of neural activity.

## Biological functional surrogates

The `biology.py` module coordinates interacting surrogates for:
- Superior temporal social-perception cue analysis.
- Temporoparietal joint attention.
- Medial prefrontal mentalizing and communicative intention estimates.
- Frontotemporal language/multimodal integration.
- Inferior frontal auditory-motor turn coordination.
- Prefrontal audience design and adaptive scaffolding.
- Ventral striatal social/outcome reward learning.
- Hippocampal social-episodic source binding.
- Posterior medial narrative/event-model construction.

The systems are distributed and task-dependent. These classes are functional abstractions, not an anatomical connectome or a one-region/one-function mapping. Interpersonal neural synchrony is not interpreted as direct brain-to-brain transfer or proof of shared understanding.

## Provenance and truth safeguards

- Partner intentions, beliefs, and referent choices remain uncertain estimates.
- Speech, gaze, gesture, prosody, face, and action retain separate modality/source information.
- Informational feedback, observed task outcomes, and social approval update different records.
- Praise or agreement never establishes factual correctness by itself.
- Perspective disagreement is preserved; evidence reliability and source dependence affect updates.
- The source of a claim is retained in the source ledger and social episode memory.
- One speaker's habits are not generalized as universal conventions. Cross-partner patterns require at least two partners and repeated consistency.
- Narrative character knowledge is stored by agent rather than copied from narrator/learner knowledge.
- Scaffolding aims to provide the minimum useful support and gradually fade it.

## SQLite annotations and operation logging

`register_annotations(database_path)` indexes all Point 14 package classes and functions in ORE's annotation catalogue. `record_result_to_sqlite(result, database_path)` stores the 16 operation events in the ORE logger's `runs` and `events` tables.

## Quick start

```python
from ore_social_conversational_learning import (
    LearnerSocialState, ReferentCandidate, SocialInteractionEvent,
    SocialInteractionLearningProcessor, SocialSignal,
)

state = LearnerSocialState(learner_id="learner")
processor = SocialInteractionLearningProcessor("learner", state=state)
event = SocialInteractionEvent(
    event_id="conversation-1",
    learner_id="learner",
    partner_id="partner",
    utterance="Look at the red cup.",
    signals=(
        SocialSignal("speech", "speech", "Look at the red cup.", reliability=0.95),
        SocialSignal("point", "gesture", "points to cup", target_id="cup", reliability=0.9),
    ),
    referents=(ReferentCandidate("cup", "red cup", features=("red", "cup")),),
    learner_focus="cup",
    target_word="cup",
    learning_goal="learn_object_words",
)
result = processor.process(event)
print(result.output)
```

## Validation coverage

Tests exercise joint attention and cue conflict, intention uncertainty, multimodal reliability, referent grounding, turn coordination, audience design, conversational repair, independent factual/reward channels, scaffolding, self-explanation, perspective comparison, narrative agent knowledge, source-bound memory, cross-partner generalization, and SQLite persistence.
