# Point 22 — Cognitive Heuristics: Mental Shortcuts and Systematic Reasoning Errors

## Scope and relationship to the source document

This algorithm expands the source point that cognitive shortcuts can make comprehension and judgment faster, while sometimes producing systematic errors—for example, misreading statistical statements or treating correlation as causation. It follows the prior point on confirmation bias and precedes the source's point on language-parsing errors such as garden-path sentences.

The algorithm treats a heuristic as a **simplified strategy for selecting, retrieving, comparing, or evaluating information**. A heuristic is not automatically irrational. It can be efficient and accurate when its assumptions fit the environment; it can mislead when those assumptions do not fit. This ecological qualification is essential: a fast rule can be adaptive in a predictable environment and systematically wrong in a different one.

Three classical families—availability, representativeness, and anchoring-and-adjustment—are central, but the algorithm also includes familiarity/fluency, sample-size neglect, base-rate neglect, and causal shortcuts. The families can overlap; they are not mutually exclusive modules.

## Scientific and biological status labels

- **Established behavioural finding:** a repeatable pattern in human judgments, though its size depends on task, population, instructions, and context.
- **Supported systems-level account:** evidence implicates distributed neural systems in valuation, memory retrieval, cognitive control, conflict monitoring, or statistical reasoning.
- **Computational abstraction:** a useful mathematical or software representation of a process. It does not imply that neurons literally calculate the displayed equation.
- **Open or debated interpretation:** more than one cognitive mechanism can explain the same behavioural result.

There is no single “heuristics centre” in the brain. Judgment recruits interacting memory, valuation, attention, and control systems. Neural activation varies with the specific heuristic, task demands, expertise, stakes, and whether a person is correcting an initial response.

---

# Part I — Integrated algorithmic sequence

## Step 1 — Receive a judgment problem and identify what must be estimated

**Input:** a question, event, claim, choice, numerical estimate, or incomplete description.

1. Parse the task into a target variable: category membership, frequency, probability, magnitude, cause, or action choice.
2. Identify the requested output type. For example:
   - “How likely is this event?” → probability estimate.
   - “How many cases occur?” → frequency or count estimate.
   - “Which explanation fits?” → causal or explanatory selection.
   - “Which option should I choose?” → action valuation under constraints.
3. Detect whether the task is uncertain, time-limited, information-poor, or cognitively expensive.
4. Establish a provisional confidence requirement. A low-stakes, rapid choice may tolerate a rough answer; a high-stakes medical, legal, financial, or safety decision requires explicit evidence checking and should not rely on an unaudited shortcut.

**Biological interpretation:** sensory and language networks identify the problem; working-memory and frontoparietal control systems maintain the goal and relevant task instructions. The hippocampus and cortical semantic memory can supply previously learned examples and rules. These functions are distributed; the sequence is a computational description rather than a strict neural conveyor belt.

**Possible computational operations:**

- Represent the task as a feature vector \(q=[q_1,\ldots,q_m]\).
- Assign an output type \(T\in\{\text{class},\text{frequency},\text{probability},\text{magnitude},\text{cause},\text{choice}\}\).
- Set a required evidence threshold \(\tau_T\), which increases with decision cost and consequence.
- Estimate processing cost \(C_{full}\) and time budget \(B\).

A heuristic becomes more likely when \(C_{full}>B\), but time pressure is not the only cause: people also use shortcuts because information is missing, the environment is uncertain, or a learned rule has become habitual.

## Step 2 — Activate relevant knowledge, memories, and expectations

1. Retrieve semantic knowledge about the topic.
2. Activate episodic examples that resemble the current case.
3. Retrieve category prototypes, familiar scripts, causal stories, and prior estimates.
4. Register the current emotional and motivational state because salience can alter which information becomes accessible.
5. Keep the difference between retrieved information and verified evidence explicit.

**Biological interpretation:** hippocampal and medial-temporal systems contribute to episodic retrieval and relational completion; distributed temporal and association cortices contribute semantic knowledge; amygdala and other valuation-related systems can modulate salience; prefrontal networks maintain task goals and regulate retrieval. Which contribution dominates depends on the task.

**Computational abstraction:** construct a candidate evidence set \(M=\{m_1,\ldots,m_k\}\). Each candidate can have an accessibility value

\[
A_i=\alpha R_i+\beta S_i+\gamma F_i+\delta E_i,
\]

where \(R_i\) is recency, \(S_i\) is emotional or perceptual salience, \(F_i\) is retrieval fluency, and \(E_i\) is strength of prior learning. The coefficients are model parameters, not known fixed neural constants.

## Step 3 — Choose a fast judgment strategy when full analysis is costly

1. Compare the current problem with previously encountered problems.
2. Estimate whether a learned shortcut is likely to fit the environment.
3. Select one or more quick strategies: recall examples, compare resemblance, start from an anchor, use a single strong cue, or infer a likely cause from a familiar pattern.
4. Record which shortcut appears to be driving the initial response.

**Biological interpretation:** learned associations and habitual policies can produce fast responses; prefrontal control may maintain a deliberate strategy when the task demands it. It is misleading to describe all fast judgments as exclusively “limbic” or all slow judgments as exclusively “prefrontal.” Speed and accuracy depend on interacting networks.

**Selection rule:** a heuristic is more defensible when the environment is stable, the cue has a known relationship with the target, and the cost of error is low. It is less defensible when base rates are unusual, evidence is selectively reported, samples are tiny, or causal structure is unknown.

A useful model-selection expression is

\[
H^*=\arg\min_H\left(\mathbb{E}[L\mid H,\mathcal{E}]+\lambda\,Cost(H)\right),
\]

where \(H\) is a candidate strategy, \(L\) is expected loss, \(\mathcal{E}\) is the environment, and \(\lambda\) expresses the cost of computation. The brain is not assumed to solve this optimization explicitly; it formalizes the trade-off.

## Step 4 — Availability heuristic: estimate likelihood from ease of recall

1. Search memory for examples of the event or category.
2. Notice how quickly examples arrive and how vivid or numerous they feel.
3. Use retrieval ease as a proxy for frequency or probability.
4. Generate an initial estimate.

**Example:** after repeatedly seeing dramatic news reports about a rare event, a person may judge it more common than local base-rate data justify.

**Biological interpretation:** memory accessibility is supported by distributed memory networks. Hippocampal retrieval, cortical associations, emotional salience, and attention can all affect which examples become available. A vivid recent episode can dominate a frequency judgment even when it is not representative.

**Computational abstraction:**

\[
\widehat{P}_{avail}(E)=\sigma\left(b+w_1N_{retrieved}+w_2Ease+w_3Vividness\right),
\]

where \(\sigma\) maps an unconstrained score into a probability-like value. This is one possible model, not a uniquely established neural formula.

A different retrieval model lets each example contribute according to activation:

\[
A_i=\exp(-d_i/\tau)\,s_i\,r_i,
\qquad
\widehat{P}(E)\propto\sum_{i\in E}A_i,
\]

where \(d_i\) is recency or memory distance, \(s_i\) is salience, and \(r_i\) is retrieval strength.

**Failure condition:** retrieval probability and real-world frequency diverge because examples differ in media exposure, novelty, emotion, recency, or ease of imagination.

## Step 5 — Representativeness heuristic: estimate probability from resemblance

1. Extract salient features of the current case.
2. Retrieve a category prototype or familiar pattern.
3. Compare the case with the prototype.
4. Interpret high similarity as high probability of category membership or as evidence that a particular process generated the case.
5. Select the best-fitting explanation.

**Example:** a description that sounds stereotypically like a particular occupation may lead a person to overestimate that occupation's probability while neglecting how common each occupation is in the population.

**Biological interpretation:** semantic memory, distributed feature representations, and pattern-completion processes support similarity judgments. Attention may weight diagnostic-looking features more strongly than statistically informative but less vivid base-rate information.

**Computational abstraction:**

\[
Sim(x,p)=\frac{x\cdot p}{\|x\|\,\|p\|}
\]

or, for weighted features,

\[
D_w(x,p)=\sum_jw_j(x_j-p_j)^2,
\qquad
Score_{rep}(C\mid x)=-D_w(x,p_C).
\]

The closer the case is to the prototype, the higher the similarity score. Similarity is not itself probability. A statistically valid estimate must also account for prior category frequency and the likelihood of the observed features under each category.

## Step 6 — Base-rate neglect and the missing denominator check

1. Identify the relevant population or reference class.
2. Estimate how common each candidate category is before considering the specific case.
3. Ask how strongly the observed evidence distinguishes the categories.
4. Combine prior probability and likelihood rather than allowing resemblance alone to determine the answer.

**Computational correction:** Bayes’ rule

\[
P(C\mid x)=\frac{P(x\mid C)P(C)}{\sum_kP(x\mid C_k)P(C_k)}.
\]

Here \(P(C)\) is the base rate and \(P(x\mid C)\) is the likelihood. If the base rate is very small, even a plausible match may have a low posterior probability unless the evidence is highly diagnostic.

**Biological interpretation:** integrating base rates, likelihoods, and uncertainty plausibly draws on memory, learned statistical regularities, working memory, and control. There is no accepted evidence that one dedicated neural area calculates Bayes’ theorem for every judgment. Bayesian equations are normative and computational tools for describing how evidence could be combined.

## Step 7 — Anchoring and adjustment: start from an initial value

1. Detect a numeric value already present in the problem, suggested by another person, or retrieved from memory.
2. Use it as a starting point for an estimate.
3. Adjust upward or downward using available evidence.
4. Stop when the answer feels sufficiently plausible or when effort/time runs out.
5. Check whether the adjustment was large enough and whether the anchor was relevant.

**Example:** a suggested starting price or an arbitrary initial number can shift a later estimate even when the starting value has little evidential relevance.

**Computational abstraction:**

\[
\hat{x}=a+\rho(x^*-a),\qquad 0\leq\rho\leq1,
\]

where \(a\) is the anchor, \(x^*\) is the fully adjusted target estimate, and \(\rho\) is the fraction of the required adjustment completed. When \(\rho\) is too small, the result remains close to the anchor.

Another model represents the final response as a weighted blend:

\[
\hat{x}=w_a a+(1-w_a)\,x_{evidence}.
\]

The anchor weight \(w_a\) varies with relevance, confidence, knowledge, and task conditions.

**Biological interpretation:** numerical estimation can recruit parietal magnitude representations, memory, and prefrontal control. Anchoring effects have several proposed mechanisms; the equation above is a behavioural model, not a claim that a single brain circuit literally adds an anchor to a correction value.

## Step 8 — Familiarity and processing fluency: treat ease as a cue

1. Detect how familiar, clear, or easy to process a claim feels.
2. Interpret fluency as a possible signal of truth, frequency, importance, or prior exposure.
3. Increase subjective confidence when processing is easy.
4. Risk confusing repetition or familiarity with independent evidence.

**Example:** repeatedly encountered statements can feel more credible even when repetition is not accompanied by new supporting evidence.

**Computational abstraction:**

\[
Conf_{subjective}=Conf_{evidence}+\eta\,Fluency,
\]

where \(\eta\) represents the degree to which fluency influences confidence. A calibrated system should estimate fluency separately from evidential support.

**Biological interpretation:** perceptual and conceptual fluency arise from processing dynamics across the systems handling the stimulus. Familiarity judgments can involve medial-temporal and cortical processes; confidence also depends on metacognitive evaluation. No single region is a universal “truth detector.”

## Step 9 — Sample-size neglect and overinterpretation of small samples

1. Observe a sample or a short sequence of outcomes.
2. Detect a pattern, streak, or unusual result.
3. Infer that the pattern reflects the broader population or a stable process.
4. Fail to account for sample size and sampling variability.

**Computational correction:** for independent observations with variance \(\sigma^2\), the standard error of the sample mean is

\[
SE(\bar{x})=\frac{\sigma}{\sqrt{n}}.
\]

A small \(n\) produces a wider range of plausible sample means. The exact formula changes for dependent observations, unequal sampling, or non-normal distributions.

**Common errors:** overgeneralizing from one anecdote; assuming a short streak proves a changed process; or expecting a small sample to mirror the population's proportions exactly.

**Biological interpretation:** humans learn statistical structure from sequences and repeated experience, but intuitive sampling expectations are not always correctly calibrated. Working memory, pattern detection, and prior expectations can make local sequences feel more meaningful than they are.

## Step 10 — Correlation-to-causation shortcut

1. Detect that two variables occur together or change together.
2. Construct a causal story linking them.
3. Prefer a familiar, coherent explanation.
4. Overlook reverse causality, common causes, selection effects, measurement artifacts, or coincidence.
5. Treat the observed association as proof of a causal effect.

**Computational distinction:** correlation measures association, for example

\[
r_{XY}=\frac{Cov(X,Y)}{\sigma_X\sigma_Y}.
\]

It does not identify causal direction. A causal graph may contain \(X\rightarrow Y\), \(Y\rightarrow X\), or \(Z\rightarrow X\) and \(Z\rightarrow Y\). Observational association alone may not distinguish these structures.

**Correction sequence:**

- Ask whether X precedes Y.
- Identify plausible common causes Z.
- Consider reverse causation.
- Check how the sample was selected.
- Seek interventions, natural experiments, longitudinal evidence, or a defensible causal design.
- Separate “associated with” from “causes.”

**Biological interpretation:** causal inference uses learned event structure, memory, prediction, and reasoning. Medial prefrontal and other association networks may contribute to model-based causal reasoning, but no simple one-region explanation is adequate.

## Step 11 — Affect and urgency can amplify a shortcut

1. Detect emotional salience, reward, threat, or urgency.
2. Prioritize salient examples and immediate outcomes.
3. Increase the weight of emotionally accessible information.
4. Reduce the time available for checking alternative explanations.
5. Produce a confident initial judgment that may be poorly calibrated.

**Computational abstraction:**

\[
W_i=\frac{\exp(\beta u_i+\gamma s_i)}{\sum_j\exp(\beta u_j+\gamma s_j)},
\]

where \(u_i\) is estimated utility or relevance and \(s_i\) is salience. Emotional arousal can change the effective weights; the formula is a convenient model and not a literal neural computation.

**Biological interpretation:** amygdala, insula, ventromedial prefrontal cortex, striatal reward systems, and neuromodulatory pathways can influence valuation, salience, and control. Their effects are task- and context-dependent. Emotion can sometimes improve attention to relevant information rather than simply impair reasoning.

## Step 12 — Generate an intuitive answer and confidence estimate

1. Combine the currently accessible examples, similarity, anchor, fluency, and affective signals.
2. Select the candidate with the strongest overall subjective support.
3. Produce an initial answer quickly.
4. Generate confidence from the strength and consistency of the available signals.
5. Keep confidence distinct from objective accuracy.

**Computational abstraction:**

\[
S_k=\theta_1A_k+\theta_2Sim_k+\theta_3Anchor_k+\theta_4Fluency_k+\theta_5Prior_k,
\]

\[
P_{intuitive}(k)=\frac{e^{S_k}}{\sum_je^{S_j}}.
\]

This softmax model illustrates how several cues can jointly influence a choice. It is not a settled neural formula.

## Step 13 — Detect conflict, uncertainty, and possible heuristic failure

1. Compare the initial answer with available constraints, base rates, sample size, and alternative explanations.
2. Detect incompatible cues—for example, high similarity but a very low base rate.
3. Notice when confidence is high but evidence quality is low.
4. Check for an arbitrary anchor, highly vivid anecdote, repeated claim, tiny sample, or causal leap.
5. Trigger a slower verification path when conflict or consequence is high.

**Biological interpretation:** anterior cingulate and related medial frontal systems are frequently implicated in monitoring conflict, errors, and the need for control. Lateral prefrontal systems contribute to maintaining rules and implementing corrective strategies. These are broad associations from cognitive-control research, not a claim that the ACC directly detects every named bias.

**Computational abstraction:** define a discrepancy signal

\[
D=\left|S_{intuitive}-S_{evidence}\right|+\kappa U+\mu R,
\]

where \(U\) is uncertainty and \(R\) is decision risk. If \(D>\tau\), or if consequence exceeds a preset threshold, require additional analysis.

## Step 14 — Run an explicit statistical and causal verification pass

1. Identify the reference population and base rate.
2. Check whether evidence is representative and sufficiently large.
3. Calculate a rate using the correct denominator.
4. Separate conditional probabilities, such as \(P(A\mid B)\), from \(P(B\mid A)\).
5. Distinguish sample association from causal effect.
6. Compare alternative hypotheses and identify what evidence would discriminate between them.
7. Seek independent corroboration rather than treating repeated exposure to one source as independent evidence.

**Possible operations:** Bayes’ rule, likelihood ratios, confidence intervals, standard errors, contingency tables, conditional probability, correlation, regression with appropriate assumptions, and causal diagrams. The method must match the data; a calculation does not repair biased sampling or invalid measurement.

For two hypotheses \(H_1,H_0\), posterior odds can be written

\[
\frac{P(H_1\mid E)}{P(H_0\mid E)}=
\frac{P(E\mid H_1)}{P(E\mid H_0)}\times
\frac{P(H_1)}{P(H_0)}.
\]

The likelihood ratio updates prior odds. This is a normative framework; actual human judgment may approximate it, depart from it, or use another strategy.

## Step 15 — Re-evaluate the environment and the ecological fit of the heuristic

1. Ask what information the heuristic discards.
2. Determine whether the discarded information is genuinely irrelevant or potentially decisive.
3. Test whether the cue historically predicts the outcome in this environment.
4. Evaluate the cost of false positives and false negatives.
5. Retain the heuristic when it is robust and well calibrated; switch strategies when assumptions fail.

A one-cue rule can outperform a complex weighted model in noisy, small-sample, or unstable environments. Conversely, ignoring base rates or causal structure can be costly in other settings. The correct strategy is an empirical question, not a blanket preference for either intuition or exhaustive analysis.

**Computational abstraction:** estimate out-of-sample performance for each strategy:

\[
Risk(H)=\frac{1}{N}\sum_{i=1}^{N}L(\hat{y}^{(H)}_i,y_i),
\]

and compare it with strategy cost. Avoid selecting the rule only because it explains examples already used to invent it; evaluate on independent cases.

## Step 16 — Update the knowledge system using outcome feedback

1. When the correct outcome becomes known, compare it with the initial judgment.
2. Compute the prediction error.
3. Determine whether the error arose from a poor cue, missing base rate, unrepresentative sample, anchoring, selection bias, or an invalid causal story.
4. Update cue reliability and memory associations.
5. Record the context in which the heuristic succeeded or failed.
6. Avoid overcorrecting from one surprising outcome.

**Computational abstraction:**

\[
\delta_t=y_t-\hat{y}_t,
\qquad
w_{t+1}=w_t+\alpha\,\delta_t x_t,
\]

where \(\delta_t\) is prediction error, \(x_t\) is the relevant cue vector, and \(\alpha\) is a learning rate. Real biological learning is more complex and may involve multiple timescales and neuromodulatory signals.

**Biological interpretation:** prediction-error learning, reinforcement learning, and memory updating involve distributed corticostriatal, dopaminergic, hippocampal, and prefrontal systems, depending on the type of feedback. Not every judgment error produces learning; feedback must be noticed, attributed correctly, and linked to the decision.

## Step 17 — Store a calibrated rule and make the final decision

1. Store the case, context, initial response, chosen heuristic, verification results, outcome, and confidence.
2. Store not just the answer but the conditions under which the heuristic is reliable.
3. Use the verified estimate for the final decision.
4. If uncertainty remains high, communicate a range or defer the decision rather than manufacturing precision.
5. Revisit the stored rule when the environment changes.

**Final output:** a judgment with an evidence basis, an uncertainty estimate, and an explicit record of the assumptions used.

---

# Part II — Integrated computational model

The following pseudocode combines the steps into an interpretable program. Its parameters are illustrative and would need empirical calibration for a particular task and population. It is not a biologically faithful simulation of neurons.

```python
from dataclasses import dataclass, field
from math import exp, sqrt
from typing import Callable, Optional

@dataclass
class JudgmentProblem:
    question: str
    target_type: str  # "probability", "frequency", "magnitude", "cause", "choice"
    examples: list[dict] = field(default_factory=list)
    base_rates: dict[str, float] = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    candidate_anchors: list[float] = field(default_factory=list)
    time_pressure: float = 0.0  # normalized 0..1
    emotional_salience: float = 0.0  # normalized 0..1
    decision_cost: float = 0.0  # normalized 0..1

@dataclass
class JudgmentResult:
    initial_answer: object
    verified_answer: object
    confidence: float
    heuristic_used: list[str]
    warnings: list[str]
    assumptions: list[str]


def logistic(x: float) -> float:
    if x >= 0:
        z = exp(-x)
        return 1.0 / (1.0 + z)
    z = exp(x)
    return z / (1.0 + z)


def retrieval_accessibility(example: dict) -> float:
    """Toy accessibility score: recency, salience and learned strength."""
    return max(0.0, min(1.0,
        0.30 * example.get("recency", 0.0)
        + 0.30 * example.get("salience", 0.0)
        + 0.40 * example.get("strength", 0.0)))


def availability_score(examples: list[dict]) -> float:
    if not examples:
        return 0.0
    return sum(retrieval_accessibility(x) for x in examples) / len(examples)


def representativeness_score(feature_similarity: float) -> float:
    return max(0.0, min(1.0, feature_similarity))


def anchored_estimate(anchor: float, evidence_estimate: float,
                      adjustment_fraction: float = 0.5) -> float:
    rho = max(0.0, min(1.0, adjustment_fraction))
    return anchor + rho * (evidence_estimate - anchor)


def bayesian_binary_update(prior: float, likelihood_if_true: float,
                           likelihood_if_false: float) -> float:
    """Posterior for a binary hypothesis, with basic input validation."""
    values = (prior, likelihood_if_true, likelihood_if_false)
    if any(not 0.0 <= v <= 1.0 for v in values):
        raise ValueError("Probabilities must be between 0 and 1")
    numerator = likelihood_if_true * prior
    denominator = numerator + likelihood_if_false * (1.0 - prior)
    if denominator == 0:
        raise ValueError("Evidence has zero probability under both hypotheses")
    return numerator / denominator


def sample_mean_standard_error(sample_sd: float, n: int) -> float:
    if n <= 0 or sample_sd < 0:
        raise ValueError("n must be positive and sample_sd non-negative")
    return sample_sd / sqrt(n)


def correlation_is_not_causation_check(evidence: dict) -> list[str]:
    warnings = []
    if evidence.get("observational_association"):
        warnings.append("Association alone does not establish causal direction.")
    if not evidence.get("temporal_order_known"):
        warnings.append("Temporal precedence has not been established.")
    if not evidence.get("confounders_examined"):
        warnings.append("Plausible common causes have not been examined.")
    if evidence.get("selection_bias_possible"):
        warnings.append("Selection effects may distort the observed association.")
    return warnings


def make_judgment(problem: JudgmentProblem,
                  similarity: float = 0.5,
                  evidence_estimate: Optional[float] = None,
                  category: Optional[str] = None) -> JudgmentResult:
    heuristics = []
    warnings = []
    assumptions = []

    # Fast intuitive stage: use available examples and similarity as cues.
    avail = availability_score(problem.examples)
    rep = representativeness_score(similarity)
    if problem.examples:
        heuristics.append("availability")
    heuristics.append("representativeness")

    prior = problem.base_rates.get(category, 0.5) if category else 0.5
    if category and problem.base_rates:
        initial_probability = 0.5 * rep + 0.3 * avail + 0.2 * prior
    else:
        initial_probability = 0.6 * rep + 0.4 * avail
    initial_probability = max(0.0, min(1.0, initial_probability))

    initial_answer: object = initial_probability

    if problem.candidate_anchors and evidence_estimate is not None:
        anchor = problem.candidate_anchors[0]
        initial_answer = anchored_estimate(anchor, evidence_estimate)
        heuristics.append("anchoring-and-adjustment")

    # Verification stage: use base rates and likelihood information when supplied.
    verified_answer = initial_answer
    likelihood_true = problem.evidence.get("likelihood_if_true")
    likelihood_false = problem.evidence.get("likelihood_if_false")
    if category and likelihood_true is not None and likelihood_false is not None:
        verified_answer = bayesian_binary_update(
            prior, likelihood_true, likelihood_false)
        assumptions.append("Binary Bayesian model with supplied prior and likelihoods")
    elif category and category in problem.base_rates:
        warnings.append("Base rate is available, but diagnostic likelihoods are missing.")

    if problem.target_type == "cause":
        warnings.extend(correlation_is_not_causation_check(problem.evidence))

    if problem.evidence.get("sample_n") is not None:
        n = int(problem.evidence["sample_n"])
        sd = float(problem.evidence.get("sample_sd", 0.0))
        se = sample_mean_standard_error(sd, n)
        if n < 30:  # illustrative prompt, not a universal statistical cutoff
            warnings.append(f"Small sample: inspect sampling uncertainty (SE={se:.3f}).")

    risk = max(problem.decision_cost, problem.emotional_salience)
    if risk > 0.7:
        warnings.append("High consequence or salience: require independent review.")
    if problem.time_pressure > 0.7:
        warnings.append("High time pressure: initial judgment may rely more on shortcuts.")

    confidence = 0.65 * float(verified_answer if isinstance(verified_answer, (int, float)) else 0.5)
    confidence += 0.35 * (1.0 - min(1.0, len(warnings) / 5.0))
    confidence = max(0.0, min(1.0, confidence))

    return JudgmentResult(
        initial_answer=initial_answer,
        verified_answer=verified_answer,
        confidence=confidence,
        heuristic_used=heuristics,
        warnings=warnings,
        assumptions=assumptions,
    )
```

**Implementation caution:** the code is an explanatory scaffold. Its weighted averages are not validated psychological laws; a production system should use task-specific models, calibrated probabilities, uncertainty intervals, provenance tracking, and out-of-sample evaluation. A generic `n < 30` warning is only a prompt to inspect sampling uncertainty, not a universal statistical decision rule.

---

# Part III — Candidate mathematical operation registry

| Operation | Purpose in the model | Important limitation |
|---|---|---|
| Feature extraction | Represent the case and candidate categories | Feature selection and weights can themselves be biased |
| Similarity / distance | Model representativeness | Similarity is not probability |
| Retrieval accessibility | Model availability | Ease of recall is not the same as real frequency |
| Weighted cue combination | Approximate intuitive judgment | Weights are task-dependent and not fixed neural constants |
| Bayesian updating | Combine base rates with diagnostic evidence | Requires appropriate priors and likelihoods |
| Anchor adjustment | Model estimates shifted toward a starting value | Several mechanisms can produce anchoring |
| Softmax choice | Convert candidate scores into relative choice strengths | Does not establish the actual neural decision rule |
| Standard error | Quantify sampling variability | Formula assumptions must fit the sampling design |
| Correlation | Quantify association | Cannot by itself identify causation |
| Causal graph comparison | Represent possible causal structures | Requires assumptions and suitable evidence for identification |
| Prediction error | Compare estimate with observed outcome | Error attribution may be ambiguous |
| Parameter update | Improve future cue weighting | Can overfit or overreact to single outcomes |
| Expected loss | Include consequence of errors | Loss functions reflect task and values, not only data |

---

# Part IV — Validation and falsifiable tests

The algorithm should be evaluated against behavioural tasks, not considered validated merely because its pseudocode executes.

1. **Availability manipulation:** present matched event descriptions while varying recency, vividness, or ease of recall. Test whether estimates shift even when objective frequency is held constant.
2. **Representativeness/base-rate test:** vary resemblance and base rates independently. Measure whether judgments follow similarity, prior frequency, or both.
3. **Anchor manipulation:** randomize an irrelevant starting number. Compare estimates across anchor conditions and estimate the adjustment fraction.
4. **Fluency manipulation:** vary repetition or processing ease while keeping evidence constant. Measure perceived truth, familiarity, and confidence separately.
5. **Sample-size manipulation:** show the same apparent proportion from samples of different sizes. Test whether confidence and generalization respond to the amount of data.
6. **Correlation/causation test:** compare observational association, temporal evidence, confounder information, and intervention evidence. Assess whether causal confidence changes appropriately.
7. **Control-demand manipulation:** vary time pressure, working-memory load, and incentives for accuracy. Do not assume that any one manipulation always increases every bias.
8. **Feedback-learning test:** provide outcome feedback over repeated trials. Determine whether participants learn cue validity and whether calibration improves on new cases.
9. **Ecological validity test:** compare heuristic performance across environments with different cue-validity structures. A heuristic should be judged by performance in the environment it is used in.
10. **Neural systems test:** where appropriate, use neuroimaging, lesion evidence, or stimulation to test contributions of memory, valuation, and control networks. Avoid reverse inference from activation in one area to a named heuristic.

## Performance metrics

- Accuracy and decision error cost.
- Brier score or log loss for probabilistic forecasts.
- Calibration: whether events assigned probability \(p\) occur approximately \(p\) proportion of the time.
- Sensitivity to irrelevant anchors and salience manipulations.
- Ability to use base rates and sample size.
- False causal attribution rate.
- Time and information cost.
- Generalization to unseen examples and changed environments.

A system that gives an accurate answer once may still be poorly calibrated. Repeated evaluation is needed to distinguish luck from a reliable strategy.

---

# Part V — Common failure modes and safeguards

| Failure mode | Why it occurs | Safeguard |
|---|---|---|
| Vivid anecdote dominates | Easy retrieval is mistaken for prevalence | Check representative rates and a reliable denominator |
| Stereotype replaces probability | Similarity is mistaken for posterior probability | Combine base rate and likelihood |
| Arbitrary anchor persists | Adjustment is incomplete or anchor remains salient | Generate an independent estimate before seeing the anchor; use ranges |
| Repetition feels like proof | Familiarity/fluency is misread as evidence | Track independent sources and provenance |
| Tiny sample seems decisive | Sampling variability is underweighted | Show sample size, uncertainty interval, and replication |
| Correlation becomes cause | A coherent causal story is generated too early | Draw alternative causal graphs and seek discriminating evidence |
| Emotion narrows the evidence set | Salience controls retrieval and attention | Deliberately search for disconfirming and base-rate information |
| Overcorrection after one error | A single outcome is treated as decisive feedback | Update gradually across multiple observations |
| Complexity is mistaken for accuracy | More inputs can add noise or overfit | Evaluate out-of-sample performance and compare simpler rules |
| One “bias checklist” is applied everywhere | The environmental fit of the strategy is ignored | Select the method according to task structure and error costs |

---

# Part VI — Neuroscientific interpretation and limitations

1. **Memory and retrieval:** episodic and semantic memory systems contribute examples, category knowledge, familiarity, and learned regularities. Availability can therefore be affected by retrieval dynamics, recency, and emotional salience.
2. **Similarity and representation:** cortical feature representations and memory-based comparison can support rapid resemblance judgments. Similarity-based estimation is useful but can omit population frequencies and diagnosticity.
3. **Magnitude and quantity:** parietal and associated networks contribute to numerical and magnitude processing. Numerical estimates are also shaped by language, learned scales, context, and memory.
4. **Valuation and affect:** vmPFC, striatal, insular, amygdala, and neuromodulatory systems can contribute to relevance, reward, uncertainty, and salience. Their roles differ across tasks.
5. **Conflict and control:** medial frontal monitoring and lateral prefrontal control are implicated in detecting conflict, maintaining rules, and implementing corrections. They do not provide a universal guarantee against bias.
6. **Learning from outcomes:** corticostriatal and dopaminergic learning systems, hippocampal memory, and prefrontal representations can contribute to updating cue values and strategies. Feedback must be appropriately attributed for learning to improve judgment.

Neural findings should be interpreted conservatively. If a region is active during a heuristic task, this does not prove that the region uniquely implements that heuristic. Many regions support multiple operations, and a single judgment can recruit several systems. The mathematical models in this document are best used to design simulations, define testable hypotheses, and clarify assumptions—not to claim that the human brain performs exact symbolic arithmetic.

---

# References and further reading

1. Tversky, A., & Kahneman, D. (1974). **Judgment under Uncertainty: Heuristics and Biases.** *Science, 185*(4157), 1124–1131. https://doi.org/10.1126/science.185.4157.1124
2. Tversky, A., & Kahneman, D. (1973). **Availability: A heuristic for judging frequency and probability.** *Cognitive Psychology, 5*(2), 207–232. https://doi.org/10.1016/0010-0285(73)90033-9
3. Gigerenzer, G., & Gaissmaier, W. (2011). **Heuristic decision making.** *Annual Review of Psychology, 62*, 451–482. https://doi.org/10.1146/annurev-psych-120709-145346
4. Kahneman, D., & Tversky, A. (Eds.). (1982). **Judgment under Uncertainty: Heuristics and Biases.** Cambridge University Press. https://doi.org/10.1017/CBO9780511809477
5. A brief history of heuristics: how did research on heuristics evolve? (2023). *Humanities and Social Sciences Communications.* https://www.nature.com/articles/s41599-023-01542-z
6. Probabilistic Biases Meet the Bayesian Brain. *Current Directions in Psychological Science.* https://doi.org/10.1177/0963721420954801

## Completion boundary

This file completes **Point 22** only. The next source point concerns language-comprehension pitfalls, including garden-path sentences and ambiguous syntax. It is intentionally not included here; proceed only when the user requests the next point.
