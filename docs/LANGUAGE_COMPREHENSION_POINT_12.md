# Point 12 — Methodological Considerations, Scientific Debates, and Model Selection

## Scope and scientific boundary

Point 12 is a researcher-facing model-evaluation workflow, not another chronological stage of sentence comprehension and not a simulation of a brain's scientific reasoning. It operationalizes phenomena, compares competing hypotheses, audits confounds, evaluates measurement and generalization, and returns a calibrated account. Regression, Bayesian model comparison, multiple-comparison correction, cross-validation, and information criteria are analyst-side tools. They must not be described as literal neuronal calculations.

The implementation follows Part 12 of Language_Comprehension_Brain_Algorithms_Parts_01-15.md. It preserves the source's central caution: a plausible account must be falsifiable, methodologically controlled, and explicit about unresolved alternatives.

## The 25 ordered operations

1. **Define an observable phenomenon.** Specify the exact comprehension problem and the response that operationalizes it. Keep comprehension separate from reading answer choices, decision, and motor response.
2. **Generate competing falsifiable hypotheses.** For each candidate, record an operation, positive predictions, disconfirming observations, and boundary conditions. A hypothesis without a possible falsifier is not considered fully testable.
3. **Audit causal structure and confounds.** Accept a directed causal graph and detect cycles. Label manipulated, measured, confounding, and mediating variables in the study plan; do not infer causal direction from correlation.
4. **Evaluate contrast identifiability.** Compare supplied target/control feature summaries. A non-manipulated feature that differs and is not declared matched is flagged as a potential confound.
5. **Preserve participant and item records.** Keep participant-level metadata and expected sample counts; do not replace individual data with a group mean before inspecting variability.
6. **Separate spatial, temporal, and functional claims.** Keep fMRI localization, EEG/MEG timing, connectivity, and cognitive interpretation distinct.
7. **Interpret ERP patterns conservatively.** N400/P600 are measured response patterns, not unique labels for semantic integration, syntax, or pragmatics.
8. **Test language selectivity.** Compare supplied language and matched non-language response profiles. Sensitivity is not absolute exclusivity.
9. **Evaluate network boundaries and interactions.** Summarize within- and between-network edge weights while preserving parcellation metadata. Graph statistics do not establish direct anatomical wiring.
10. **Separate association from causal contribution.** Keep imaging, lesion, and perturbation evidence distinct. A causal claim requires an appropriate design and review of assumptions.
11. **Constrain neural interpretation with behavior.** Use accuracy, response time, confidence, and interpretation measures to constrain neural explanations without treating behavior as a direct map of brain activity.
12. **Audit population generalization.** Preserve language background, culture, developmental group, and age-group diversity when supplied. Do not assume universal generalization.
13. **Prevent overfitting and selective reporting.** Record preregistration, exploratory status, multiplicity control, and confirmatory versus exploratory distinctions.
14. **Evaluate out-of-sample prediction.** Compute MSE and MAE only for paired held-out observations. A generalization claim additionally requires a frozen model and an independent test set.
15. **Compare candidate models transparently.** Posterior weights are computed only if every hypothesis supplies comparable, valid likelihoods and priors. No likelihoods are invented.
16. **Test boundary conditions.** Retain the conditions under which each candidate is expected to operate; do not extrapolate outside them by default.
17. **Inspect individual and group heterogeneity.** Preserve individual effects and participant characteristics; group averages do not describe every participant.
18. **Audit measurement quality and missingness.** Retain quality metadata, exclusions, and missingness reasons. Unreported quality is not assumed to be good.
19. **Evaluate evidence dependence.** Group methods by supplied cohort/source identifiers so multiple measures from one sample are not presented as independent replications.
20. **Check model adequacy.** Permit the outcome that all supplied candidate models are inadequate. Never force selection of the least-bad candidate as truth.
21. **Calibrate claim strength.** Distinguish behavioral evidence, neural association, convergent evidence, causal contribution under tested conditions, and unresolved mechanism.
22. **Preserve uncertainty and alternatives.** Missing evidence is not evidence of absence; unmeasured confounding and competing explanations remain visible.
23. **Choose discriminating next experiments.** Use falsifiers and supplied next-experiment proposals to identify tests that could separate candidates. Expected information gain is not calculated without a specified outcome model.
24. **Construct a calibrated mechanistic account.** Synthesize evidence types, boundaries, competing accounts, and claim limits.
25. **Return an auditable research report.** Return method-specific evidence, confound audit, generalization, uncertainty, candidate ranking where justified, and next experiments.

## Mathematical operations and what they mean

| Operation | Research use | Biological status |
|---|---|---|
| P(Y given X,H) | Predict an observable outcome under a hypothesis | Statistical model, not a neuronal equation |
| DAG / topological sort | Audit proposed causal structure | Research design tool |
| Feature contrast Δj = mean(target) − mean(control) | Detect mismatched condition features | Analyst calculation |
| Mixed-effects model | Account for participant and item variation | Statistical framework |
| ERP contrast ERP_T(t) − ERP_C(t) | Compare event-locked responses | Analysis of measured electrical signals |
| Selectivity S = R_language − R_nonlanguage | Compare language and non-language responses | Empirical contrast, not absolute exclusivity |
| Graph G = (V,E,W) | Describe measured network relationships | Does not prove anatomical wiring |
| ATE = E[Y given intervention] − E[Y given control] | Estimate perturbational effect under assumptions | Causal research estimand |
| MSE / MAE | Score held-out predictions | Model evaluation |
| Bayes' rule | Compare hypotheses using supplied likelihoods and priors | Researcher-side inference; not proof of neural Bayesian arithmetic |
| Expected information gain | Choose a potentially discriminating experiment | Experimental planning |

## Python API

    from ore_language_comprehension import Point12Request, Point12ModelEvaluator

    request = Point12Request(
        target_phenomenon="sarcasm interpretation",
        observable_target={
            "measure": "interpretation_choice",
            "separates_comprehension_from_response": True,
        },
        candidate_hypotheses=[
            {
                "name": "contextual_revision",
                "operation": "revise literal meaning using discourse context",
                "predictions": ["context changes the selected interpretation"],
                "falsifiers": ["context manipulation has no effect"],
            },
            {
                "name": "general_conflict",
                "operation": "resolve conflict between sentence and situation",
                "predictions": ["conflict predicts response difficulty"],
                "falsifiers": ["matched non-social conflict has no effect"],
            },
        ],
        analysis_plan={
            "preregistered": True,
            "multiple_comparison_control": True,
            "manipulated_variables": ["context"],
            "matched_variables": ["length", "lexical frequency"],
        },
    )
    result = Point12ModelEvaluator().process(request)
    print(result.ranked_hypotheses)  # empty unless comparable likelihoods were supplied
    print(result.trace)              # 25 ordered audit records

Point12Request accepts the target phenomenon, observable outcome, candidate hypotheses, target/control feature summaries, behavioral data, neural data, participant-level metadata, analysis plan, and study metadata. Point12Result.to_dict() returns JSON-compatible data.

## Evidence inputs

- behavioral_data: target/control accuracy, response time, and confidence arrays.
- neural_data.fmri, eeg, meg: target/control measurements plus relevant QC metadata.
- neural_data.connectivity.edges: source/target network names and numeric weights.
- neural_data.lesion and perturbation: caller-supplied evidence and design safeguards.
- neural_data.task_response_profile: language and non-language control response arrays.
- analysis_plan.causal_edges: directed edges as two-item sequences or source/target mappings.
- analysis_plan.held_out_observed and held_out_predicted: paired numeric test observations.
- Each candidate may supply prior_probability and a likelihoods mapping keyed by identical dataset IDs. Posterior ranking is withheld unless all candidates have comparable valid likelihood sets.

No raw fMRI/EEG preprocessing, source reconstruction, lesion-symptom mapping, or full mixed-effects fitting is claimed. Missing values remain missing; the evaluator summarizes supplied records and flags limitations.

## Shared SQLite annotations

register_point12_annotations() indexes Point 12 classes and functions in the existing ore_logger.annotations SQLite catalogue. It does not create a separate Point 12 database. Each public class, method, and registration function has a docstring for the annotation synchronizer.

## Scientific safeguards

- A correlation or co-activation is not a causal direction or proof of necessity.
- N400/P600 labels do not uniquely identify a cognitive operation.
- A language-selective response is not necessarily exclusive to language.
- Connectivity statistics are not anatomical tracts.
- A posterior is only as defensible as its candidate set, likelihoods, priors, and dependence assumptions.
- Statistical model fit does not prove biological implementation.
- A group mean does not imply uniform individual organization.
- All candidate hypotheses may be wrong or incomplete.

## Focused tests

Run: python -m pytest tests/test_language_comprehension_point12.py -q

The acceptance tests cover operation order, no fabricated evidence, causal-cycle detection, confound flags, measurement separation, ERP caution, selectivity, explicit-likelihood model comparison, held-out metrics, immutable request snapshots, docstrings, and SQLite annotation registration.
