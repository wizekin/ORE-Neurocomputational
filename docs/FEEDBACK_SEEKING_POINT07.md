# Point 07 — Seek Feedback: Functional and Biological-Reference Implementation

## Scope

`ore_feedback_seeking` implements the 18 source steps in `Point_07_Seek_Feedback_Brain_Algorithm.md`. It provides a functional feedback-correction pipeline, a separate distributed population-rate biological reference, explicit mathematical operations, and source-aligned operation traces. It complements (rather than overwrites) the pre-existing `ore_corrective_learning` Point 24 package.

The caller supplies candidate interpretations, evidence, source properties, correctness labels, and transfer probes. The implementation does not search for evidence, invent labels, or treat a correction as verified merely because it was provided by a source.

## The 18 operations

| ID | Operation | Main computation | Main output |
|---|---|---|---|
| P07-01 | Define target | Normalized weighted completeness: `C = sum(w_i * m_i)` | Target requirements and completeness |
| P07-02 | Generate interpretation | Normalize competing hypothesis priors and retain reasoning steps | Candidate model and prior entropy |
| P07-03 | Monitor comprehension | `u = a(1-c) + b*d + g*ambiguity` | Uncertainty and focused-check decision |
| P07-04 | Estimate confidence | Shannon entropy and Brier score when labelled | Separate confidence/calibration record |
| P07-05 | Select feedback route | Expected entropy reduction minus source access cost | Ranked feedback sources |
| P07-06 | Obtain actionable feedback | Structured response/evidence discrepancy | Feedback items and missing fields |
| P07-07 | Verify feedback | Claim-specific source/evidence quality and Bayesian updating | Trusted feedback and posterior |
| P07-08 | Classify/localize error | Error-class distribution and earliest low-consistency node | Error type and first divergence |
| P07-09 | Separate knowledge/performance | Logistic performance model with fatigue/distraction/load | Knowledge estimate and performance load |
| P07-10 | Apply targeted correction | Bounded local parameter update; preserve unrelated components | Revised knowledge components |
| P07-11 | Integrate knowledge | Evidence-linked concept graph and boundary-condition edges | Provenance and retrieval cues |
| P07-12 | Explain correction | Weighted accuracy/relation/evidence/boundary quality | Corrected explanation model |
| P07-13 | Test retrieval | Residual error and improvement, only when an observed score exists | Retrieval result |
| P07-14 | Calibrate confidence | Brier score and expected calibration error (ECE) | Updated calibration history |
| P07-15 | Control feedback load | `L_f = a*N + b*I + g*U`; split into chunks | Load estimate and chunk plan |
| P07-16 | Test transfer | Mean observed accuracy on novel contexts | Transfer score and counterexample result |
| P07-17 | Schedule delayed check | `A(t)=A0*exp(-lambda*t)` | Estimated accessibility and review record |
| P07-18 | Update future strategy | Exponentially weighted error-category frequency | Adaptive monitoring policy |

## Quick start

```python
from ore_feedback_seeking import (
    FeedbackEvidence, FeedbackRequest, FeedbackSource,
    Point07FeedbackProcessor, TargetCriterion,
)

request = FeedbackRequest(
    request_id="causality-001",
    learning_target="Distinguish association from causation",
    candidate_response="Correlation always proves causation.",
    domain="research methods",
    criteria=(TargetCriterion("causal", "Do not infer causation from correlation", 1.0, 0.0),),
    confidence=0.9,
    feedback_sources=(FeedbackSource(
        "methods-review", "Methods-aware reviewer", reliability=0.98,
        relevance=1.0, specificity=0.95, evidence_quality=0.98,
    ),),
    feedback_evidence=(FeedbackEvidence(
        "study-design", "methods-review", "The study is observational.",
        correction="Association alone does not establish causation.",
        evidence_quality=0.98,
    ),),
    corrected_response="Association alone does not establish causation.",
)

result = Point07FeedbackProcessor().process(request)
print(len(result.operation_trace))                 # 18
print(result.functional_outputs["step_07_verify_feedback"])
print(result.diagnostics["valid"])
```

The package's tests use explicit labels to exercise the verification, retrieval, calibration, and transfer paths. When those labels are absent, the system returns `None`/`not_tested` rather than manufacturing correctness.

## SQLite annotation and operation logging

The implementation reuses `ore_logger.LoggerDatabase` and the existing shared `ore_logger.annotations`, `runs`, and `events` tables:

```python
from ore_feedback_seeking import register_annotations, record_result_to_sqlite

catalogue = register_annotations()  # defaults to ore_logs/ore_logger.sqlite3
persisted = record_result_to_sqlite(result, "ore_logs/ore_logger.sqlite3")
```

`register_annotations()` runs the repository's AST-backed `ore_logger.catalog.sync_repository()` for the `ore_feedback_seeking` package. Every module, class, function, and method receives an annotation record with its docstring or source-derived description. Each of the 18 event rows links to its functional-step annotation and the `Point07FeedbackProcessor` class annotation.

## Biological reference track

The optional `DistributedFeedbackNetwork` represents coarse interacting populations for prefrontal goal control, medial-frontal error monitoring, frontoparietal attention, working memory, temporal semantic representations, hippocampal relational memory, language explanation, amygdala-linked salience, striatal value learning, and cortical plasticity. Each population uses a leaky rate equation:

```text
r(t + dt) = r(t) + (1 - exp(-dt/tau)) * (sigmoid(b + input + recurrent) - r(t))
```

The connectivity is sparse and signed. The network exposes per-step rates and its limitations. A semantic comprehension mismatch is not passed into the optional striatal reward-prediction-error channel; only an explicitly supplied reward/outcome prediction error enters that channel.

This is a computational analogy, not a neuronal-level simulation, a fitted brain model, or proof that a named region performs an isolated algorithm step. The biological relationships are distributed, task-dependent, and many-to-many. Software tests establish implementation behavior, not biological validity.

## Scientific and engineering safeguards

- Confidence is not treated as correctness.
- Evidence from copied sources is grouped by a caller-supplied independence key and is not counted as independent reinforcement.
- Feedback is weighted by domain-specific reliability, relevance, specificity, actionability, evidence quality, and noise.
- Only named/justified components are revised; other knowledge is retained.
- Unverified corrections remain provisional. Caller-supplied ground truth is required for objective calibration.
- Lexical overlap is clearly labelled as a fallback proxy, not semantic understanding.
- Transfer is scored only from labelled probes; otherwise its status is `not_tested`.
- The biological population network is an engineering abstraction, not a claim about exact neural computations.

## Validation

Run the focused suite:

```bash
python -m pytest tests/test_feedback_seeking_point07.py -q
```

Run the full repository suite after installing the repository's declared development dependencies:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

A software pass does not demonstrate neural fidelity. Empirical validation would require preregistered behavioral and neural measurements, fitted parameters, independent datasets, and model comparisons against alternative explanations.
