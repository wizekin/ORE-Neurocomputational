# Point 24 — Feedback, clarification, and misconception correction

## Scope

The ore_corrective_learning package implements the Point 24 corrective-learning loop as twenty inspectable operations. The central engineering goal is to update a working interpretation only when there is enough discriminating evidence, preserve the history and source of the prior belief, and test the replacement through retrieval and transfer.

This is a configurable computational synthesis, not an autonomous fact-checker, a general knowledge base, or a validated neural circuit. It does not browse for evidence, invent missing alternatives, or treat confidence, authority, familiarity, or surprise as proof.

## The twenty operations

| # | Operation | Core computation | Output |
|---:|---|---|---|
| 1 | Preserve knowledge state | \(M_t=(H_t,E_t,C_t,S_t,q_t)\) | Prior interpretation and provenance |
| 2 | Generate predictions | \(P(y\mid H_i,C)\) | Candidate predictive distributions |
| 3 | Detect mismatch | \(\delta=y-\hat y\); \(-\log P(y)\) | Discrepancy and surprisal |
| 4 | Classify error | \(P(z\mid x,H,C)\) | Multi-label error probabilities |
| 5 | Select information action | \(EIG(a)=H(H\mid D)-\mathbb E_r[H(H\mid D,r,a)]\) | Infer, clarify, check, or test |
| 6 | Acquire clarification | \(U(a)=EIG(a)-\lambda Cost(a)\) | Best question/check plan or unresolved status |
| 7 | Evaluate feedback | \(w_e=r_es_ed_em_e\) | Reliability-weighted independent evidence |
| 8 | Reactivate prior interpretation | \(M_{old}=(H_{old},source,t)\) | Prior answer retained as historical, not current truth |
| 9 | Generate correction candidates | Caller-supplied candidate set | Bounded alternatives |
| 10 | Compare hypotheses | \(P(H_i\mid E,C)\propto P(E\mid H_i,C)P(H_i\mid C)\) | Posterior over alternatives |
| 11 | Update knowledge components | \(\theta_{t+1}=\theta_t+\eta r g\Delta\) | Selectively changed feature/relation |
| 12 | Build relational explanation | \(G=(V,E,R)\) | Conditions, examples, non-examples, and boundaries |
| 13 | Retrieve corrected understanding | \(A(t)=A_0e^{-\lambda t}\) | Unaided retrieval result |
| 14 | Give specific feedback | Residual error indicator \(e_k\) | Targeted correction without reinforcing the old answer |
| 15 | Test transfer | \(\hat y=f_\theta(z(x))\) | New-context and counterexample performance |
| 16 | Calibrate confidence | \(BS=N^{-1}\sum_i(p_i-o_i)^2\) | Brier score where verified outcomes exist |
| 17 | Consolidate correction | \(S_{t+1}=S_t+\beta I_tR_tC_t\) | Accessibility and consolidation state |
| 18 | Monitor belief regression | \(B_{t+1}=B_t-\gamma M_c+\rho F_{old}\) | Old-versus-corrected accessibility and regression risk |
| 19 | Evaluate closed loop | Stop only when task gates pass | Stop or identify earliest failed stage |
| 20 | Store and validate | \(L=\lambda_1L_{recall}+\lambda_2L_{transfer}+\lambda_3L_{calibration}\) | Trace, provenance, unresolved questions, and SQLite events |

## Mathematical details

### Prediction and mismatch

For candidate hypotheses \(H_i\), the processor combines their predicted outcome distributions using the current prior. For a discrete observation \(y\):

\[
\text{surprisal}(y)=-\log \max(\epsilon,P(y)),\qquad
\text{mismatch}=1-P(y)
\]

For a numeric observation, it also calculates the residual:

\[
\delta=y-\hat y,\qquad
m=\min\left(1,\frac{|\delta|}{\max(1,|y|,|\hat y|)}\right)
\]

Surprise is a reason to investigate, not proof that the current interpretation is false.

### Clarification and information gain

Each ClarificationOption provides a response distribution under each hypothesis. Expected information gain compares prior entropy with expected posterior entropy. The action utility subtracts a configurable cost penalty. The selected action is a plan; the package does not actually contact a person, teacher, database, or external source.

### Evidence quality and independence

Each evidence item carries a likelihood per hypothesis and explicit estimates for source reliability, relevance, diagnosticity, and directness. Evidence contributions sharing an independence_key are collapsed into a single weighted log-likelihood contribution. This prevents duplicated copies of the same report from being counted as independent confirmation. Independence is supplied by the caller; the software cannot infer whether two sources share an upstream origin.

For each evidence item:

\[
w_e=r_e\,s_e\,d_e\,m_e
\]

where the factors represent reliability, relevance, diagnosticity, and a directness multiplier. Posterior probabilities use normalized log scores for numerical stability.

### Selective updates and testing

Only named affected_components are eligible for component revision. If none are named, the selected hypothesis's components are eligible. Numeric values are moved toward the new target by a bounded update gate; unrelated components remain untouched.

Retrieval and transfer are separate measurements. A successful repetition of the corrected answer in the original wording is not equivalent to transfer. A correction is only marked corrected when posterior confidence, residual uncertainty, evidence quality, and any supplied retrieval/transfer checks meet the task thresholds. Missing observations or missing corrective evidence remain explicitly uncertain.

### Memory and regression

The optional state tracks accessibility of the old and corrected interpretations, consolidation strength, a history of correction episodes, and confidence/outcome pairs. The exponential decay and reinforcement equations are engineering approximations. Sleep and replay are not simulated or assumed to have occurred.

## Quick start

~~~python
from ore_corrective_learning import (
    CorrectionRequest, FeedbackEvidence, KnowledgeHypothesis,
    Point24CorrectiveLearningProcessor, RetrievalAttempt,
)

hypotheses = (
    KnowledgeHypothesis(
        "old", "The valve remains open above the threshold.", prior=0.7,
        predictions={"open": 0.8, "closed": 0.2},
    ),
    KnowledgeHypothesis(
        "new", "The valve closes above the threshold.", prior=0.3,
        predictions={"open": 0.1, "closed": 0.9},
    ),
)

evidence = FeedbackEvidence(
    "verified-test", "The independent test records a closed valve.",
    likelihoods={"old": 0.05, "new": 0.95},
    source_name="independent test", source_type="measurement",
    reliability=0.98, relevance=1.0, diagnosticity=0.98,
    directness="direct", independence_key="test-series-1",
)

request = CorrectionRequest(
    request_id="example-001",
    hypotheses=hypotheses,
    current_hypothesis_id="old",
    observed_outcome="closed",
    evidence=(evidence,),
    verified_hypothesis_id="new",
    retrieval_attempt=RetrievalAttempt("new", confidence=0.9, delay_days=1),
)

result = Point24CorrectiveLearningProcessor().process(request)
print(result.stopping["status"])
print(result.posterior_probabilities)
print(len(result.operation_trace))  # 20
~~~

The source-of-truth for each likelihood is the caller. verified_hypothesis_id should only be supplied when correctness has been independently established. Do not use posterior selection as a substitute for an external ground-truth label in experiments.

## SQLite annotation and event logging

register_annotations() synchronizes Python functions and classes under the ore_corrective_learning package scope into the shared ore_logger annotation catalogue.

record_result_to_sqlite() creates a run record and writes one event for each of the twenty operations, including the mathematical expression, functional output, and biological-reference activity. It records completion or failure in the shared run table.

## Biological reference track

The optional population-rate reference includes coarse populations for sensory association, frontotemporal language, medial-frontal monitoring, prefrontal control, hippocampal retrieval, locus-coeruleus/noradrenergic modulation, dopaminergic feedback value, semantic relations, metacognitive monitoring, distributed consolidation, and cerebellar calibration.

These populations are interacting modelling labels, not one-region-per-step claims. Prediction error, reward prediction error, and semantic correction are not interchangeable. The network is not fitted to neural recordings, does not model individual neurons, and is not physiological validation. Disable it with CorrectiveConfig(simulate_biology=False) while retaining all twenty functional operations.

## Validation and limits

The tests cover posterior normalization, information gain, evidence-source dependence, selective component updates, retrieval feedback, transfer, confidence scoring, delayed accessibility, regression detection, uncertainty preservation, high-stakes thresholds, disabled biological execution, and SQLite trace persistence.

The implementation does not automatically generate missing candidate explanations; source reliability estimates and independence groups must be supplied; its numerical parameters are not measured biological constants; an accurate immediate response does not establish durable learning; and the algorithm is not a substitute for domain-specific verification.

## References from the source point

- Ofen & Chen (2025), Prediction error and memory updating: Challenges ahead, Neuroscience & Biobehavioral Reviews, 176, 106308. https://doi.org/10.1016/j.neubiorev.2025.106308
- Carneiro, Lapa, & Finn (2021), Memory updating after retrieval: when new information is false or correct, Memory, 29(9), 1156–1175. https://doi.org/10.1080/09658211.2021.1968438
- Maraver et al. (2022), Can we learn from errors?, PLOS ONE, 17(8), e0272427. https://doi.org/10.1371/journal.pone.0272427
- Swire-Thompson et al. (2023), Memory failure predicts belief regression after the correction of misinformation, Cognition, 230, 105276. https://doi.org/10.1016/j.cognition.2022.105276
- Roediger & Abel (2022), The double-edged sword of memory retrieval, Nature Reviews Psychology, 1, 708–720. https://doi.org/10.1038/s44159-022-00115-2
- Fernández, Boccia, & Pedreira (2016), The fate of memory: Reconsolidation and the case of prediction error, Neuroscience & Biobehavioral Reviews, 68, 423–441. https://doi.org/10.1016/j.neubiorev.2016.06.004

The evidence base supports parts of the loop, such as prediction-error learning, feedback effects, retrieval practice, and conditional memory updating. The twenty-operation decomposition and its parameterization are computational design choices, not established biological mechanisms.
