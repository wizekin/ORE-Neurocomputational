# Point 21 — Confirmation Bias: Belief-Consistent Attention, Interpretation, and Memory

## Scope and relationship to the source

This point expands the source document's statement that people can recall events in ways that fit their prior beliefs. It treats confirmation bias as a family of related processes through which an existing belief or hypothesis influences (1) which information is sought, (2) which information receives attention, (3) how ambiguous evidence is interpreted, (4) how evidence is weighted during judgment, and sometimes (5) what is later retrieved or reconstructed from memory.

The algorithm is a functional and neurocomputational model, not a claim that the brain runs a literal symbolic program. Biological systems implement these operations through distributed neural activity, synaptic plasticity, recurrent interactions, neuromodulation, and memory retrieval. The equations below are useful abstractions for describing and testing those operations. No single brain region is a dedicated “confirmation-bias centre,” and confirmation bias is not inevitable in every person, task, or situation.

## 1. Define the state of the current belief

**Purpose:** Establish the hypothesis or expectation that may bias subsequent processing.

1. Receive a current question, event, claim, or observation requiring judgment.
2. Retrieve relevant prior knowledge and beliefs from semantic and episodic memory.
3. Identify the focal hypothesis, such as `H: the event was caused by X`, and relevant alternatives, such as `¬H` or `H₂`.
4. Represent belief strength as a graded state rather than an all-or-none label. A useful computational representation is a probability or log-odds value.
5. Store the provenance of the belief where possible: direct experience, testimony, inference, repeated exposure, or an earlier judgment.

A simple belief state is:

\[
B_t = P(H\mid E_{1:t}), \qquad L_t = \log\frac{B_t}{1-B_t}
\]

where \(E_{1:t}\) is the evidence considered up to time \(t\). The probability representation is a model variable; it is not suggested that neurons explicitly store a decimal probability.

**Biological implementation (functional interpretation):** Retrieval can recruit distributed temporal and medial-temporal memory systems. Medial prefrontal regions participate in evaluating self-relevant knowledge, value, and prior judgments. The exact network depends on the content and task.

**Output:** An initial belief state, its confidence, and—if available—the remembered source and context of that belief.

## 2. Activate a belief-linked expectation template

**Purpose:** Convert the prior belief into expectations that can guide perception and information search.

1. Derive the features expected if the hypothesis is true.
2. Derive features that would be expected if the hypothesis were false or an alternative were true.
3. Use the active hypothesis to bias top-down attention toward diagnostic or familiar features.
4. Keep a representation of expected evidence separate from evidence actually observed. This distinction is essential for avoiding later source confusion.

For feature \(f_i\), an expected relevance score may be modelled as:

\[
T_i = P(f_i\mid H) - P(f_i\mid \neg H)
\]

A positive value indicates that the feature is more expected under \(H\); a negative value indicates that it favours the alternative. The difference is a simple discriminability measure, not a universal neural code.

**Possible neural operations:** Recurrent interactions between prefrontal control systems and sensory or association cortices can bias which features are prioritized. In visual tasks, goal-directed templates can affect attentional selection. Such top-down modulation is normally useful; it becomes problematic when the template is treated as proof rather than as a prediction to test.

**Output:** A feature-expectation template with explicit predictions for both confirming and disconfirming observations.

## 3. Select information and sources

**Purpose:** Determine what evidence enters the reasoning process.

1. Generate possible sources: direct observation, records, witnesses, experiments, experts, or alternative explanations.
2. Estimate the cost, relevance, accessibility, and perceived reliability of each source.
3. Select which source to consult or which evidence to inspect next.
4. Under confirmation bias, the selection policy may favour sources likely to support the current belief and avoid sources expected to challenge it.
5. Under an accuracy-oriented policy, select information with the greatest expected ability to distinguish between competing hypotheses, even if it is uncomfortable or belief-inconsistent.

A softmax selection model can represent the probability of selecting source \(s\):

\[
P(s)=\frac{\exp(\beta U(s))}{\sum_j\exp(\beta U(s_j))}
\]

where \(U(s)\) combines expected diagnostic value, accessibility, processing fluency, social trust, and anticipated emotional cost. A confirmation-bias term can be added:

\[
U(s)=D(s)+aC(s)-bK(s)+cA(s)
\]

where \(D\) is diagnostic value, \(C\) is expected belief congruence, \(K\) is challenge or conflict cost, and \(A\) is accessibility. Positive \(a\) and \(b\) increase preference for congenial, low-challenge sources. These coefficients are model parameters, not fixed biological constants.

**Biological interpretation:** Selection reflects interacting attentional, valuation, control, and memory systems. It need not involve conscious intent. A person can choose belief-consistent material simply because it is more familiar, easier to retrieve, or less effortful to process.

**Output:** A selected evidence stream and a record of sources not yet sampled.

## 4. Allocate attention to incoming evidence

**Purpose:** Decide which parts of the selected information receive deeper processing.

1. Parse incoming sensory, linguistic, or social information into candidate features or propositions.
2. Compare each candidate with the active expectation template.
3. Allocate attention according to relevance, salience, novelty, uncertainty, and current goals.
4. A belief-consistent feature may receive preferential selection; a contradictory feature may receive less elaboration or be disengaged from sooner.
5. Do not assume that unattended information was not encoded at all. Some evidence may be represented in sensory or neural activity without strongly influencing the final choice.

One simplified attentional gain model is:

\[
G_i = G_0 + \alpha R_i + \gamma S_i + \delta U_i + \eta C_i
\]

where \(R_i\) is task relevance, \(S_i\) salience, \(U_i\) uncertainty reduction, and \(C_i\) congruence with the current belief. A positive \(\eta\) models a congruence preference. In a well-calibrated accuracy-seeking system, the coefficient for expected information gain should dominate mere congruence.

**Possible neural operations:** Frontoparietal control networks can modulate sensory processing; parietal and other association regions may carry evidence representations used in a decision. Empirical work indicates that belief-consistent and belief-inconsistent information can be differentially weighted at later readout, even when contradictory evidence has been encoded. Therefore, “not used” must not automatically be interpreted as “not perceived.”

**Output:** An attended evidence representation with an estimated processing strength and confidence.

## 5. Encode the evidence with source and context tags

**Purpose:** Preserve what was actually observed and where it came from.

1. Encode the content of the evidence.
2. Bind it, as far as possible, to its source, time, context, and whether it was directly observed or inferred.
3. Preserve the direction of the evidence: supports \(H\), supports an alternative, or is non-diagnostic.
4. Distinguish evidence content from the person's reaction to it.
5. Update encoding strength according to attention, elaboration, novelty, emotional arousal, and task demands.

A conceptual evidence record is:

\[
M_i = (x_i, s_i, c_i, d_i, q_i, t_i)
\]

where \(x_i\) is content, \(s_i\) source, \(c_i\) context, \(d_i\) evidential direction, \(q_i\) estimated quality, and \(t_i\) time. The vector is an engineering representation of the information that memory systems may preserve in distributed form.

**Biological interpretation:** The hippocampal formation is important for binding item, context, and relational information into episodic representations. Cortical systems contribute distributed content representations. Binding is not perfect; source details can be lost even when the gist remains accessible.

**Output:** A set of evidence traces whose content can be evaluated separately from the belief they concern.

## 6. Interpret ambiguous evidence

**Purpose:** Assign meaning to evidence that permits more than one interpretation.

1. Generate candidate interpretations.
2. Use linguistic context, prior knowledge, schemas, and the active hypothesis to estimate their plausibility.
3. Compare each interpretation against the actual details of the evidence.
4. Confirmation bias can cause ambiguous evidence to be interpreted in a belief-consistent direction.
5. Generate at least one plausible alternative interpretation before treating the first interpretation as established.

A Bayesian form is:

\[
P(H\mid E)=\frac{P(E\mid H)P(H)}{P(E)}
\]

The prior \(P(H)\) appropriately influences inference, but it should not make the likelihood \(P(E\mid H)\) immune to evidence. A bias can emerge when the prior is over-weighted, when likelihoods are estimated asymmetrically, or when source reliability and dependencies are misrepresented.

For competing interpretations \(I_k\):

\[
P(I_k\mid E,H)\propto P(E\mid I_k,H)P(I_k\mid H)
\]

A useful anti-bias check is to repeat the interpretation with the hypothesis label hidden or reversed and ask whether the same evidence would receive the same reading.

**Biological interpretation:** Semantic, episodic, and contextual memory interact with control and valuation systems. The computation is distributed; no claim is made that one area performs the full Bayesian calculation.

**Output:** A ranked set of interpretations, including alternatives and uncertainty.

## 7. Estimate evidence quality and source reliability

**Purpose:** Prevent agreement with the belief from being confused with evidence quality.

1. Assess whether the source had direct access to the event.
2. Assess expertise, track record, independence, incentives, and possible conflicts.
3. Check whether apparently separate sources depend on the same original report.
4. Estimate measurement error, sample size, missing data, and reproducibility where relevant.
5. Apply the same reliability criteria to supportive and contradictory sources.

A simple reliability-adjusted weight is:

\[
w_i = r_i\,d_i\,m_i
\]

where \(r_i\) is source reliability, \(d_i\) is diagnosticity, and \(m_i\) is measurement quality. The terms may be normalized to a common range for a particular model.

**Important dependency correction:** Ten reports repeating one unverified source are not ten independent observations. If evidence items are dependent, multiplying their likelihoods as if independent exaggerates certainty.

A bounded-rationality model can produce confirmation effects when people simplify the relationships between beliefs and source reliability. This is one proposed explanation, not the only account of confirmation bias.

**Output:** Reliability- and dependence-adjusted evidence weights.

## 8. Compare confirming and disconfirming evidence symmetrically

**Purpose:** Avoid asymmetrical standards of evaluation.

1. Compute or qualitatively assess how strongly each evidence item favours each hypothesis.
2. Ask what observation would be unlikely if the current belief were true.
3. Compare supportive and contradictory items using the same standards for source quality and methodological rigor.
4. Do not dismiss contradictory evidence solely because it is surprising, inconvenient, or delivered by a disliked source.
5. If an item is weak, explain the weakness in terms that would also apply to a similarly weak supportive item.

For evidence \(E_i\), a likelihood ratio is:

\[
LR_i=\frac{P(E_i\mid H)}{P(E_i\mid \neg H)}
\]

Its log contribution is:

\[
\ell_i=\log LR_i
\]

Positive \(\ell_i\) favours \(H\); negative \(\ell_i\) favours \(\neg H\); values near zero are weakly diagnostic. This gives a common scale for evidence in either direction.

**Output:** A balanced evidence table, not merely a count of arguments on each side.

## 9. Read out evidence for a judgment

**Purpose:** Convert encoded evidence into a decision or confidence estimate.

1. Retrieve relevant evidence traces.
2. Weight each trace by estimated quality, diagnosticity, and independence.
3. Combine its evidential contribution with the prior belief.
4. A confirmation bias can occur at readout: belief-consistent evidence receives greater behavioural weight, while contradictory evidence is underweighted.
5. Separate the strength of a memory trace from the weight assigned to it during judgment.

A simplified accumulation rule is:

\[
L_{t+1}=L_t+\sum_i g_i w_i\ell_i
\]

where \(g_i\) represents the degree to which evidence is used at readout. If \(g_i\) is larger for belief-consistent evidence, the final judgment becomes biased even if the evidence itself was encoded accurately.

This distinction matters: a person may have registered contradictory information but give it little influence when deciding. Recent neural evidence has shown precisely this possibility in a particular perceptual decision task; it should not be generalized to every form of belief or memory.

**Output:** A provisional posterior belief and a record of the evidence that materially influenced it.

## 10. Update confidence and detect belief perseverance

**Purpose:** Revise confidence in proportion to evidence strength rather than familiarity or repetition.

1. Compare the new evidence-weighted belief with the previous belief.
2. Determine whether confidence should increase, decrease, or remain stable.
3. Check whether the change is driven by genuinely new information or repeated exposure to the same information.
4. Detect whether confidence remains high after important supporting assumptions have been challenged.
5. If a belief is not revised, identify whether the cause is evidence quality, prior strength, source distrust, or selective weighting.

A bounded update rule is:

\[
B_{t+1}=B_t+\lambda\,(B^*_t-B_t)
\]

where \(B^*_t\) is the evidence-supported target and \(\lambda\) is the learning or update rate. A very small \(\lambda\) can model excessive resistance to change. A value that is too large can model overreaction to weak or noisy evidence.

**Biological interpretation:** Confidence and evidence updating involve distributed decision, valuation, and control networks. Research has linked confidence-related confirmation effects to altered processing of post-decision evidence and to reduced sensitivity to the strength of disconfirming opinions in medial prefrontal regions in particular tasks. These findings identify candidate mechanisms, not a universal localization.

**Output:** An updated confidence estimate and an explanation of the update (or non-update).

## 11. Consolidate and later retrieve belief-related memories

**Purpose:** Explain how prior beliefs may shape later recollection without assuming that recall is always biased.

1. Retain accessible traces of the event and the judgment made about it.
2. During later retrieval, use cues and schemas to reconstruct the episode.
3. Reconstructed details may be influenced by current beliefs, subsequent information, and the remembered interpretation.
4. Source and context tags may weaken, allowing an inference or repeated claim to be misremembered as direct observation.
5. Retrieval can also be accurate or unbiased; the direction and size of any effect depends on task and conditions.

A conceptual reconstruction model is:

\[
\hat{E}=\alpha M+\beta S+\gamma B+\epsilon
\]

where \(M\) is retrieved episodic detail, \(S\) is schema-based completion, \(B\) is the current belief, and \(\epsilon\) is noise. Coefficients are not fixed and the equation is not a literal neural mechanism.

**Critical qualification:** Confirmation bias is not synonymous with belief-consistent memory. Research has found effects in information search and interpretation without necessarily finding a recall advantage for belief-consistent material. Therefore, the algorithm tests each stage separately rather than assuming that all stages are biased together.

**Output:** A reconstructed account with separate confidence estimates for remembered details, inferred details, and source attribution.

## 12. Identify the feedback loop that can stabilize an inaccurate belief

**Purpose:** Model how multiple small biases can reinforce one another over time.

1. An initial belief produces an expectation template.
2. The template influences source choice and attention.
3. Selective evidence is interpreted in a belief-consistent direction.
4. Evidence readout and confidence updating favour the initial belief.
5. The resulting judgment becomes a stronger prior for the next cycle.
6. Repeated retrieval can make the belief more accessible and fluent, which may be mistaken for truth.
7. The cycle continues unless contradictory information is deliberately sampled and fairly evaluated.

A simple dynamic representation is:

\[
B_{t+1}=F(B_t,\,S_t,\,A_t,\,I_t,\,R_t)
\]

where \(S_t\) is source selection, \(A_t\) attention, \(I_t\) interpretation, and \(R_t\) evidence readout. Feedback occurs because each of these is partly influenced by \(B_t\).

The loop is not necessarily irrational in every respect. Expectations reduce computational cost and often help perception. The failure arises when prediction is not adequately corrected by prediction error, when alternatives are not sampled, or when confidence becomes detached from evidence quality.

**Output:** A causal map of the particular feedback mechanisms maintaining the belief.

## 13. Introduce corrective control operations

**Purpose:** Make the system more accuracy-seeking without simply reversing the bias.

1. **Pre-register the test:** State what evidence would count against the current hypothesis before examining the next result.
2. **Generate alternatives:** Write at least one competing explanation and its predicted observations.
3. **Seek disconfirming tests:** Choose observations that distinguish the hypotheses, not merely material that sounds oppositional.
4. **Use symmetric criteria:** Apply the same source-quality and measurement standards to both sides.
5. **Check independence:** Determine whether multiple sources provide independent evidence.
6. **Separate observation from inference:** Record exactly what was seen or measured before interpreting it.
7. **Calibrate confidence:** State a confidence range and update it when evidence changes.
8. **Invite independent review:** Ask another person to identify missing alternatives or unsupported assumptions.
9. **Revisit the evidence after a delay:** Reduce the influence of immediate emotional reactions and familiarity.
10. **Keep an audit trail:** Preserve the original belief, predicted outcomes, evidence, and subsequent revisions.

An accuracy-oriented source-selection policy can maximize expected information gain:

\[
IG(s)=H(H\mid E)-\mathbb{E}_{e\sim s}[H(H\mid E,e)]
\]

where \(H(\cdot)\) is entropy. Select sources with high expected reduction in uncertainty, accounting for reliability and cost. The most useful next source may be one that challenges the current belief, but contradiction alone does not make a source informative or trustworthy.

**Biological interpretation:** Deliberate checking recruits cognitive control, working memory, and metacognitive monitoring. These processes can counteract automatic selection and readout tendencies, but they require time, attention, motivation, and sufficient information.

**Output:** A corrected evidence-sampling and updating policy.

## 14. Validate whether the algorithm is functioning

The model should be tested at separate processing stages rather than by one global “bias score.”

### Experimental task structure

1. Establish a belief or initial hypothesis in participants.
2. Present evidence that varies independently in congruence, diagnosticity, reliability, and emotional relevance.
3. Measure source choice, viewing time or attention, interpretation, confidence, final decision, and later memory.
4. Manipulate whether participants receive time, an alternative hypothesis prompt, or a pre-specified disconfirmation test.
5. Compare each stage independently.

### Measurable quantities

- **Selection bias:** proportion of chosen sources that support the prior belief, controlling for utility and reliability.
- **Attentional bias:** difference in attention to confirming and disconfirming evidence.
- **Interpretation bias:** difference in the interpretation of equally ambiguous evidence as a function of prior belief.
- **Readout bias:** difference in decision weight for evidence that was comparably encoded.
- **Confidence bias:** overconfidence relative to objective accuracy or calibration.
- **Memory bias:** differences in later recall, recognition, source memory, and confidence for confirming versus disconfirming information.

### Failure signatures

- **Selection failure:** contradictory or high-information sources are rarely sampled.
- **Attention failure:** disconfirming information receives less processing despite equal visibility.
- **Interpretation failure:** ambiguous evidence is repeatedly resolved in favour of the current belief.
- **Reliability failure:** source quality is judged differently depending on whether the source agrees.
- **Readout failure:** known contradictory evidence is given little weight at decision time.
- **Update failure:** confidence remains high despite strong, reliable counterevidence.
- **Source-memory failure:** inferred or second-hand information is later remembered as directly witnessed.

A valid system should be able to change its mind in response to strong contradictory evidence, but should not change its mind merely because an opposing claim is vivid or confidently stated.

## 15. Integrated high-level pseudocode

```text
FUNCTION evaluate_claim(claim, prior_belief, prior_confidence):
    hypotheses = generate_competing_hypotheses(claim)
    belief_state = initialize_belief_state(
        hypotheses,
        prior_belief,
        prior_confidence,
        provenance=retrieve_belief_source(claim)
    )

    predictions = derive_predictions_for_all(hypotheses)
    evidence_log = []

    WHILE more_information_needed(belief_state):
        candidate_sources = enumerate_sources(claim, hypotheses)
        source_scores = score_sources(
            candidate_sources,
            diagnostic_value=True,
            reliability=True,
            independence=True,
            accessibility=True
        )

        # Avoid selecting sources solely because they agree with the prior.
        next_source = select_high_expected_information_source(source_scores)
        raw_evidence = inspect_source(next_source)
        evidence_items = parse_into_testable_observations(raw_evidence)

        FOR item IN evidence_items:
            observation = separate_observation_from_interpretation(item)
            attended = allocate_attention(
                observation,
                relevance=True,
                uncertainty=True,
                salience=True
            )
            memory_trace = encode_with_source_context(attended, next_source)
            candidate_interpretations = generate_interpretations(memory_trace)
            interpretations = evaluate_interpretations_symmetrically(
                candidate_interpretations,
                hypotheses,
                predictions
            )
            quality = assess_reliability_and_dependence(
                memory_trace,
                next_source
            )
            evidence_log.append((memory_trace, interpretations, quality))

        belief_state = update_using_weighted_evidence(
            belief_state,
            evidence_log,
            symmetric_criteria=True,
            avoid_double_counting=True
        )
        belief_state = calibrate_confidence(belief_state)

        IF belief_state.is_overconfident OR evidence_is_conflicting(evidence_log):
            request_disconfirming_test(predictions, hypotheses)

    RETURN {
        "beliefs": belief_state,
        "evidence_audit": evidence_log,
        "alternatives_considered": hypotheses,
        "uncertainty": belief_state.uncertainty,
        "source_provenance": preserve_source_tags(evidence_log)
    }
```

## 16. Scientific limitations and interpretation safeguards

1. Confirmation bias is an umbrella term. Source selection, attention, interpretation, evidence weighting, confidence updating, and recall are separable phenomena and need not all occur together.
2. Prior beliefs are not inherently errors. Bayesian inference requires prior information; the key question is whether priors and evidence likelihoods are appropriately calibrated.
3. A person can rationally discount contradictory information when its source is genuinely unreliable. The bias is not simply “disagreeing with criticism”; it is asymmetrical or poorly calibrated processing.
4. Neural findings are task-specific. A result from perceptual decision-making should not be treated as a complete explanation of political, autobiographical, scientific, or social beliefs.
5. The proposed equations describe functional relationships. There is no evidence that the brain literally calculates these formulas in the explicit mathematical form shown.
6. Confidence and emotional investment can influence evidence processing, but the size and direction of these effects vary with task, motivation, time pressure, knowledge, and social context.
7. Debiasing methods should improve evidence quality and comparison rather than force equal weight onto strong and weak arguments.

## 17. Core computational summary

The full process can be summarized as a recurrent inference system:

\[
\text{Prior belief}\rightarrow\text{prediction}\rightarrow\text{source selection}\rightarrow
\text{attention}\rightarrow\text{encoding}\rightarrow\text{interpretation}\rightarrow
\text{evidence weighting}\rightarrow\text{belief update}\rightarrow\text{memory retrieval}
\]

The critical control loop compares predicted and observed evidence:

\[
\varepsilon_t = E_t - \mathbb{E}[E_t\mid H_t]
\]

where \(\varepsilon_t\) is a conceptual prediction error. An adaptive system uses reliable prediction errors to revise its model. A confirmation-biased system may reduce the influence of errors that conflict with its current belief, sample too little disconfirming information, or selectively read out the evidence that supports the prior.

The main design principle is therefore not “remove prior beliefs,” but **keep prior expectations distinct from observations, test alternative predictions, evaluate evidence symmetrically, preserve source provenance, and update confidence in proportion to reliable information**.

## Selected scientific references

1. Nickerson, R. S. (1998). *Confirmation bias: A ubiquitous phenomenon in many guises*. Review of General Psychology, 2(2), 175–220. https://doi.org/10.1037/1089-2680.2.2.175
2. Hart, W., Albarracín, D., Eagly, A. H., Brechan, I., Lindberg, M. J., & Merrill, L. (2009). *Feeling validated versus being correct: A meta-analysis of selective exposure to information*. Psychological Bulletin, 135(4), 555–588. https://doi.org/10.1037/a0015701
3. Kappes, A., Harvey, A. H., Lohrenz, T., Montague, P. R., & Sharot, T. (2020). *Confirmation bias in the utilization of others’ opinion strength*. Nature Neuroscience, 23, 130–137. https://doi.org/10.1038/s41593-019-0549-2
4. Rollwage, M., et al. (2020). *Confidence drives a neural confirmation bias*. Nature Communications, 11, 2634. https://doi.org/10.1038/s41467-020-16278-6
5. Pilgrim, C., Sanborn, A., Malthouse, E., & Hills, T. T. (2024). *Confirmation bias emerges from an approximation to Bayesian reasoning*. Cognition, 245, 105693. https://doi.org/10.1016/j.cognition.2023.105693
6. Born, R. T. (2024). *Stop Fooling Yourself! (Diagnosing and Treating Confirmation Bias)*. eNeuro, 11(10). https://doi.org/10.1523/ENEURO.0415-24.2024
7. *Confirmation bias through selective readout of information encoded in human parietal cortex* (2025). The study reports that, in a perceptual decision task, disconfirming evidence could be encoded but contribute less to the subsequent report. PubMed record: https://pubmed.ncbi.nlm.nih.gov/38979146/

---

**End of Point 21.** The next point in the source sequence concerns cognitive shortcuts (heuristics) and how they can create systematic errors, including misinterpreting statistical statements and perceiving causality in mere correlation.
