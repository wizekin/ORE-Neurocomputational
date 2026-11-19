# Point 08 — Stay Motivated and Curious

## Scope and design

The `ore_curiosity_motivation` package implements the 24 ordered steps in `Point_08_Stay_Motivated_and_Curious_Brain_Algorithm.md`. It models information gaps, value, learnability, curiosity, intrinsic/extrinsic motivation, effort allocation, goals, exploration, attention, encoding, emotion, sustainable engagement, autonomy, consolidation, delayed understanding, and future policy updates.

This is a **computational model inspired by distributed neuroscience**, not a validated simulation of the human brain. The specification itself cautions that the brain does not execute a fixed 24-step serial program and that its equations are candidate abstractions rather than literal neuronal calculations.

## The 24 operations

| ID | Operation | Main mathematical idea |
|---|---|---|
| P08-01 | Parse material into learning units | Set segmentation and goal-linked relevance |
| P08-02 | Detect novelty and information gaps | Prediction mismatch, entropy, relevance × learnability |
| P08-03 | Estimate information value | Weighted intrinsic, utility, transfer, and personal value |
| P08-04 | Appraise learnability and safety | Logistic learnability and value-minus-cost appraisal |
| P08-05 | Estimate curiosity | `C = sigmoid(b + w_g*IG + w_v*V + w_l*L - w_t*T - w_f*F)` |
| P08-06 | Separate motivation sources | Distinct intrinsic, extrinsic, personal, mastery, and avoidance values |
| P08-07 | Estimate effort and opportunity cost | Expected gain minus effort, time, and stress costs |
| P08-08 | Set goals and progress | Milestone progress and challenge-fit control |
| P08-09 | Generate questions and actions | Expected information gain relative to action cost |
| P08-10 | Allocate attention | Stable temperature-scaled softmax and finite attention budget |
| P08-11 | Explore and sample evidence | Caller-supplied action/evidence sampling and source weighting |
| P08-12 | Update expectations | `delta = observed_learning - expected_learning`; action-value update |
| P08-13 | Encode target information | Bounded trace strengthening from attention, elaboration, relevance, evidence, and engagement |
| P08-14 | Encode incidental information | Context-link × moderate arousal × attention proxy |
| P08-15 | Appraise emotional significance | Separate valence/arousal/control/relevance and inverted-U arousal term |
| P08-16 | Integrate curiosity, emotion, and engagement | Separate signals combined into an engagement and exploration gate |
| P08-17 | Detect disengagement | Logistic boredom/fatigue/error/switching/progress estimate |
| P08-18 | Adapt challenge and explanation | Increase depth, build prerequisites, clarify, reduce distraction, or pause |
| P08-19 | Reinforce progress and competence | Update competence only from observed performance |
| P08-20 | Preserve autonomy and agency | Learner choice weighted by control and goal fit |
| P08-21 | Prioritize memory retention | Weighted relevance, elaboration, emotional significance, novelty, and future utility |
| P08-22 | Support consolidation over time and sleep | `M(t)=M0+(Mpotential-M0)*(1-exp(-k*t))` |
| P08-23 | Test durable understanding | Separate factual, relational, transfer, and confidence measurements |
| P08-24 | Update future curiosity and motivation | Prediction-error action values and future softmax policy |

## Input and output model

The API uses typed dataclasses:

- `LearningUnit`: a concept/claim/example with relevance, novelty, uncertainty, utility, intrinsic interest, transfer value, personal relevance, learnability, context linkage, and reliability.
- `LearningAction`: a possible question, reading step, explanation, experiment, or practice action with expected information/learning gain and explicit effort/time/stress costs.
- `LearningOutcome`: an optional observed action result. Missing observed learning is kept as unknown.
- `RetrievalCheck`: delayed factual, relational, transfer, and confidence measures. Missing values are never fabricated.
- `LearningRequest`: goal, units, candidate actions, caller-provided outcomes, affect, attention, resources, and review context.
- `LearningState`: persistent knowledge, topic/action values, memory candidates, unresolved questions, and learning history.
- `MotivationConfig`: adjustable weights, thresholds, and learning rates.

## Quick start

```python
from ore_curiosity_motivation import (
    LearningAction, LearningOutcome, LearningRequest, LearningUnit,
    Point08MotivatedLearningProcessor, RetrievalCheck,
)

request = LearningRequest(
    request_id="moon-phase-001",
    goal="Explain lunar phases from Sun-Earth-Moon geometry",
    units=(LearningUnit(
        "lunar-phases", "Why does the Moon's illuminated appearance change?",
        relevance=1.0, novelty=0.8, uncertainty=0.7,
        utility=0.9, intrinsic_interest=0.95,
        conceptual_distance=0.35, available_support=0.9,
    ),),
    actions=(LearningAction(
        "inspect-diagram", "lunar-phases", "inspect_diagram",
        expected_learning_gain=0.7, expected_information_gain=0.8,
        effort_cost=0.2, time_cost=0.1, learner_choice=True,
    ),),
    outcomes=(LearningOutcome("inspect-diagram", observed_learning=0.85),),
    retrieval_checks=(RetrievalCheck(
        "delayed", factual_accuracy=0.9, relational_accuracy=0.85,
        transfer_accuracy=0.8, confidence=0.85, delayed_days=1.0,
    ),),
)

result = Point08MotivatedLearningProcessor().process(request)
assert len(result.operation_trace) == 24
print(result.functional_outputs["step_05_estimate_curiosity"])
print(result.functional_outputs["step_23_test_durable_understanding"])
```

The implementation consumes caller-provided material and observations. It does not silently browse, invent evidence, or infer learning merely from curiosity or interest.

## Biological-reference model

`DistributedMotivationNetwork` maintains interacting coarse populations for:

- Prefrontal goal control and anterior-cingulate monitoring.
- Frontoparietal attention.
- Midbrain dopamine-related and ventral-striatal value proxies.
- Hippocampal memory and cortical plasticity.
- Amygdala-linked emotional salience.
- Locus-coeruleus arousal and hypothalamic stress-axis proxies.

Each population uses a bounded leaky rate approximation:

```text
r(t+dt) = r(t) + (1-exp(-dt/tau)) * (sigmoid(b + input + recurrent) - r(t))
```

Connection weights and time constants are transparent software parameters, not measured physiological constants. The dopamine-related population receives only an explicitly observed outcome prediction error. Novelty, semantic mismatch, curiosity, or emotional arousal alone are **not** automatically relabelled as dopamine reward-prediction error.

This model cannot establish neural fidelity. The broad biological mappings are many-to-many, context-dependent, and scientifically incomplete. Validation as a biological model would require empirical behavioral/neural data, parameter fitting, independent datasets, and model comparisons.

## SQLite annotations and event logging

The package reuses the existing `ore_logger.LoggerDatabase` and AST-backed annotation catalogue:

```python
from ore_curiosity_motivation import register_annotations, record_result_to_sqlite

catalogue = register_annotations()  # ore_logs/ore_logger.sqlite3 by default
saved = record_result_to_sqlite(result)
print(saved["events_written"])       # 24
```

Registration indexes the package's modules, classes, methods, and functions into the shared `annotations` table. Each operation event references its functional-step annotation and the `Point08MotivatedLearningProcessor` class annotation. The existing shared `runs` and `events` tables are used; no parallel logging database schema is introduced.

## Safeguards and interpretation

1. Curiosity is not the same as attention or learning.
2. Intrinsic and extrinsic value are distinct state variables.
3. Novelty alone does not imply useful curiosity.
4. High stress/fatigue can trigger regulation, task reduction, or pause recommendations rather than ever-increasing effort.
5. Emotional arousal has a task-specific non-linear proxy; stronger emotion is not assumed to be uniformly better.
6. Target and incidental memory candidates remain distinguishable.
7. Factual, relational, transfer, and calibration measures remain separate.
8. Missing outcome labels stay missing.
9. Reward prediction error is not conflated with generic cognitive surprise.
10. Biological population names are explanatory reference labels, not isolated neural modules.

## Validation

Run the focused suite:

```bash
python -m pytest tests/test_curiosity_motivation_point08.py -q
```

Run the normal repository checks as configured:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

The normal GitHub Actions workflow remains enabled; Point 08 adds tests rather than replacing or disabling existing automated checks.
