# Point 22 — Cognitive Heuristics and Systematic Reasoning Errors

## Implementation scope

The ore_cognitive_heuristics package implements the seventeen source-ordered operations from Point 22. It treats heuristics as fast strategies that can be useful in environments where their assumptions fit, not as inherently irrational mechanisms.

The implementation explicitly separates retrieved-example accessibility from population frequency; prototype similarity from posterior probability; numeric anchors from independent evidence estimates; processing fluency from independent corroboration; correlation from causal identification; intuitive judgment from statistical verification; and prediction error from attribution of an error to a particular cue.

The package includes typed contracts, an operation registry, a functional model, a distributed population-rate biological-reference model, an ordered pipeline, package-scoped annotation registration, and SQLite operation logging.

## Source-ordered operation map

| Step | Operation | Functional computation | Broad biological reference |
|---|---|---|---|
| 1 | Receive judgment problem | Parse target type, categories, uncertainty, time budget, and decision cost | Sensory/language processing, working memory, frontoparietal control |
| 2 | Activate knowledge | Retrieve episodic examples, semantic prototypes, and accessibility features | Hippocampal/medial-temporal retrieval and distributed cortical memory |
| 3 | Select strategy | Select candidate shortcuts and record assumptions and risks | Learned associations, valuation, and prefrontal strategy control |
| 4 | Availability | Score recall accessibility using recency, salience, fluency, and learned strength | Distributed memory retrieval, cortical associations, attention |
| 5 | Representativeness | Weighted prototype similarity using cosine-style similarity | Distributed feature coding and pattern completion |
| 6 | Base-rate check | Normalize priors and combine supplied diagnostic likelihoods using Bayes' rule | Learned regularities, memory, working memory, and control |
| 7 | Anchor adjustment | Compute anchor + rho * (evidence estimate - anchor) | Parietal magnitude processing, memory, prefrontal control |
| 8 | Fluency monitor | Quantify familiarity while preserving separate evidence confidence | Distributed perceptual/conceptual fluency and metacognition |
| 9 | Sample uncertainty | Compute standard error and an approximate interval where inputs permit | Statistical pattern learning and working-memory support |
| 10 | Causal check | Audit correlation, temporal order, confounders, reverse causality, and selection | Event-memory and distributed prefrontal/association reasoning |
| 11 | Affect and urgency | Softmax cue priorities using utility, salience, and time pressure | Amygdala/insula, vmPFC, striatum, and neuromodulation |
| 12 | Intuitive judgment | Combine accessibility, similarity, and prior cues into an initial distribution | Distributed retrieval, valuation, and readout |
| 13 | Conflict detection | Compare intuitive and base-rate/likelihood judgments; trigger checking | Medial frontal monitoring and lateral prefrontal control |
| 14 | Verification | Run explicit prior/likelihood pass and retain sampling/causal caveats | Working memory, statistical reasoning, and control networks |
| 15 | Ecological fit | Evaluate cue validity and stability; require held-out risk assessment | Learning and valuation adapting to environmental regularities |
| 16 | Feedback learning | Record Brier error and update cue weights only when attribution is provided | Corticostriatal, dopaminergic, hippocampal, and prefrontal learning |
| 17 | Store rule and decide | Preserve context, minimize supplied expected loss, or defer under uncertainty | Memory systems plus prefrontal/valuation decision processes |

Each trace row includes its source-order index, mathematical abstraction, functional output, and biological-reference output.

## Mathematical operations

### Accessibility and availability

For example i, the model uses this illustrative bounded accessibility score:

\[
A_i=0.25R_i+0.25S_i+0.20F_i+0.30L_i
\]

where R is recency, S is salience, F is fluency, and L is learned strength. Accessibility is not treated as event frequency.

### Representativeness and base rates

Weighted cosine-style similarity compares the current feature vector with category prototypes:

\[
Sim_w(x,p)=\frac{\sum_jw_jx_jp_j}
{\sqrt{\sum_jw_jx_j^2}\sqrt{\sum_jw_jp_j^2}}
\]

Similarity is not interpreted as a posterior by itself. Where priors and likelihoods are supplied:

\[
P(C_k\mid E)=
\frac{P(E\mid C_k)P(C_k)}
{\sum_jP(E\mid C_j)P(C_j)}
\]

Missing likelihoods are neutral rather than fabricated as negative evidence. If no diagnostic likelihoods are supplied, the posterior remains the normalized prior.

### Anchoring and fluency

The numeric anchor model is:

\[
\hat{x}=a+\rho(x^*-a),\qquad 0\leq\rho\leq1
\]

An arbitrary value without an independent target estimate is not silently promoted into evidence. Fluency is represented separately:

\[
Conf_{subjective}=
\min(1,Conf_{evidence}+0.2\,Fluency)
\]

The verified confidence path does not treat repeated exposure as independent corroboration.

### Sampling and causal inference

For independent observations:

\[
SE(\bar{x})=\frac{s}{\sqrt{n}},\qquad
SE(\hat{p})=\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
\]

A supplied design effect reduces effective sample size. The normal-approximation interval is a diagnostic, not a universal confidence interval for every sampling design.

Correlation can be calculated from paired finite observations:

\[
r_{XY}=\frac{Cov(X,Y)}{\sigma_X\sigma_Y}
\]

It does not establish direction or causality. The causal audit separately asks about temporal precedence, confounding, reverse causality, selection, and intervention evidence.

### Cue weighting, confidence, and feedback

Cue priority uses a softmax over utility- and salience-related signals:

\[
W_i=\frac{\exp(\beta u_i+\gamma s_i)}
{\sum_j\exp(\beta u_j+\gamma s_j)}
\]

The intuitive score combines accessibility, resemblance, and prior cues. A separate verification stage uses the prior/likelihood model. For known outcomes, the multiclass Brier score is:

\[
BS=\sum_k(p_k-y_k)^2
\]

Cue weights change only when cue-level feedback is explicitly supplied. An outcome by itself is not enough to identify which cue caused a prediction error. Updates are bounded to avoid extreme overcorrection.

## API example

~~~python
from ore_cognitive_heuristics import (
    JudgmentProblem, Point22Config, Point22CognitiveHeuristicsProcessor,
)

problem = JudgmentProblem(
    problem_id="case-001",
    question="Which category best explains the observation?",
    categories=("A", "B"),
    base_rates={"A": 0.1, "B": 0.9},
    similarity_scores={"A": 0.95, "B": 0.25},
    evidence={"likelihoods": {"A": 0.7, "B": 0.2}},
    decision_cost=0.8,
)
processor = Point22CognitiveHeuristicsProcessor(Point22Config(simulate_biology=True))
result = processor.process(problem)
print(result.initial_answer, result.verified_answer, result.probabilities)
print(result.warnings)
~~~

The result operation trace contains all seventeen rows. Set simulate_biology=False to disable population-rate dynamics without disabling the functional algorithm.

### SQLite annotations and operation logging

~~~python
from ore_cognitive_heuristics import record_result_to_sqlite
report = record_result_to_sqlite(result)
print(report["run_id"], report["database_path"])
~~~

The package uses the shared ore_logger.annotations, runs, and events tables. Package-scoped annotation registration is available through register_annotations().

## Tests

Run:

~~~bash
python -m pytest tests/test_cognitive_heuristics_point22.py -q
~~~

The suite checks operation ordering, Bayesian denominator correction, accessibility versus frequency, prototype similarity, anchor adjustment, fluency separation, sample-size uncertainty, correlation/causation safeguards, urgency escalation, feedback attribution, ecological-fit uncertainty, biological-track disabling, validation, and SQLite integration.

## Scientific and engineering limitations

- Coefficients, thresholds, population drives, and rate updates are illustrative engineering parameters, not measured neural constants.
- The biological-reference network is a coarse distributed activity analogy, not an anatomically fitted or validated neuronal simulation.
- Brain-region names denote broad systems-level associations; no single region is claimed to implement a named heuristic or guarantee unbiased judgment.
- The small-sample threshold prompts inspection; it is not a universal statistical rule.
- Normal-approximation intervals may fail under dependence, skew, selection, or very small samples.
- Expected-loss decisions are only as appropriate as the user-supplied loss function and category model.
- Heuristics can be adaptive when cue validity and environmental structure support them. Ecological fit remains unknown when no cue-validity evidence is supplied.
- This research/software abstraction is not a medical, legal, financial, or safety decision authority. High-cost cases should receive independent review.

## Source references

The originating Point 22 Library document cites Tversky and Kahneman on judgment under uncertainty and availability, Gigerenzer and Gaissmaier on heuristic decision-making, and later work on probabilistic inference and heuristics.
