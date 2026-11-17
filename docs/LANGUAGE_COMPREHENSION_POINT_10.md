# Point 10 — Evidence-Based Identification and Validation of Language-Comprehension Mechanisms

## Scope

Point 10 is a **scientific research-analysis workflow**, not another chronological comprehension stage and not a neural simulation. It tests the mechanisms proposed in Points 1–9 against supplied behavioral, fMRI, EEG/ERP, MEG, lesion, stimulation, and computational evidence. Statistical summaries in this module are operations performed by the analyst on data; they must never be described as literal calculations performed by neurons.

The implementation follows the supplied *Language_Comprehension_Brain_Algorithms_Parts_01-15.md*, Part 10, “Evidence-Based Identification and Validation of Language-Comprehension Mechanisms.”

## The 15 operations

1. **Operationalize one mechanism.** Specify the smallest testable operation, inputs, outputs, triggering conditions, positive predictions, and disconfirming predictions. A broad phrase such as “meaning processing” is not a testable operational definition.
2. **Construct controlled contrasts.** Define target and control conditions, match relevant linguistic and task variables, counterbalance items and order, and document confounds. Participant and item variation should be modeled in a full inferential analysis.
3. **Establish behavioral signature.** Summarize accuracy, response time, confidence, and error types. A response-time increase alone does not identify a neural operation.
4. **Evaluate spatial fMRI evidence.** Summarize supplied target/control response estimates and record independent-localizer, motion, nuisance, and multiple-comparison safeguards. BOLD is an indirect, slow hemodynamic signal.
5. **Evaluate temporal EEG/MEG evidence.** Summarize event-locked measures, trial sufficiency, artifact control, and preregistered time windows. N400/P600 labels are recorded as temporal signatures, not unique process labels.
6. **Evaluate lesion evidence.** Relate supplied lesion features to performance while retaining lesion volume, disconnection, non-language abilities, diaschisis, and compensation as important caveats.
7. **Evaluate perturbational evidence.** Compare active stimulation with sham/control outcomes. Document randomization, non-specific effects, and safety. TMS/tES is not perfectly focal.
8. **Build method-by-prediction matrix.** Keep behavior, spatial association, timing, lesion, perturbation, and model-fit contributions distinct.
9. **Evaluate competing explanations.** Preserve alternatives such as general difficulty, working-memory load, attention, perception, response selection, and prediction violation. Attach discriminating evidence where supplied.
10. **Estimate effect size and uncertainty.** Compute descriptive Cohen’s d when data support it; record whether multiplicity is controlled. Confidence intervals and full mixed-effects inference must be supplied by a study-specific analysis when not directly provided.
11. **Test individual-level reliability.** Preserve participant-level reliability/localization information and distinguish it from group averages. This module does not fabricate ICC estimates from insufficient metadata.
12. **Integrate without double-counting.** Group evidence by explicit source/cohort identifiers. Multiple measures from the same participants or dataset are not independent replications. Bayesian posterior probabilities are not fabricated; the report records that likelihoods and dependence assumptions are required.
13. **Revise the mechanistic account.** Retain replicated findings, adjust unsupported regional assignments, preserve unresolved alternatives, and require new falsifiable predictions.
14. **Validate out of sample.** For supplied held-out observed/predicted numeric arrays, calculate MSE and MAE and optionally correlation. The record also states whether the model was frozen and the test set independent.
15. **Produce calibrated report.** Summarize method-specific evidence, quality-control flags, alternatives, independent source groups, generalization, and the next discriminating experiments.

## Mathematical operations and their interpretation

| Operation | Use in this research workflow | Biological interpretation |
|---|---|---|
| Condition contrast | Target minus control response | Analyst's comparison, not a neural arithmetic operation |
| Mixed-effects model | Account for participant and item variation | Statistical inference; not neural circuitry |
| Accuracy and mean RT | Behavioral summary | Measures of observed behavior |
| Cohen's d | Standardized difference where sample sizes/variance permit | Statistical effect size |
| fMRI GLM | Fit task and nuisance predictors to BOLD time series | BOLD is indirect and correlational in ordinary task designs |
| ERP average/difference | Event-locked temporal summary | Does not make N400/P600 unique process markers |
| Lesion regression/correlation | Relate lesion features to performance | Supports network contribution under lesion conditions; not a one-region/one-function proof |
| Active-sham contrast | Estimate perturbational effect | Can support causal contribution; stimulation may spread through networks |
| Bayesian updating | Formal evidence synthesis | Depends on likelihoods and independence assumptions; not proof of literal neural probability calculation |
| Cross-validation / MSE | Evaluate held-out predictions | Model evaluation, not brain computation |

For example, a behavioral condition contrast can be represented as
`d=(mean_target-mean_control)/SD_pooled`. A voxel-wise fMRI model may use `y_v=X beta_v+epsilon_v`; an ERP contrast may use `ERP_target(t)-ERP_control(t)`; and held-out prediction error may use `MSE=mean((y-y_hat)^2)`. These formulas describe analysis of measurements.

## Python API

```python
from ore_language_comprehension import Point10Request, Point10EvidenceValidator

request = Point10Request(
    mechanism={
        "name": "lexical_semantic_access",
        "operation": "retrieve lexical-semantic information after word-form recognition",
        "predictions": ["semantic constraint changes behavior and neural response"],
        "alternatives": ["general difficulty", "working-memory load"],
        "next_discriminating_experiments": ["match difficulty while manipulating semantic constraint"],
    },
    study_plan={
        "target_condition": "high semantic constraint",
        "control_condition": "low semantic constraint",
        "counterbalanced": True,
        "confounds_controlled": ["word frequency", "length", "response format"],
        "participant_and_item_effects": True,
        "multiple_comparison_control": True,
    },
    datasets={
        "behavior": {
            "target": {"accuracy": [1, 1, 1, 0], "response_time": [500, 520, 510, 530]},
            "control": {"accuracy": [1, 0, 0, 0], "response_time": [600, 620, 610, 630]},
        },
        "fmri": {
            "target": [2.0, 2.1, 2.2], "control": [1.0, 1.1, 1.2],
            "independent_roi": True, "motion_controlled": True,
            "multiple_comparison_control": True,
            "independence_group": "sample-A",
        },
    },
)
result = Point10EvidenceValidator().process(request)
print(result.report)
print(result.trace)
```

The module accepts caller-supplied numeric summaries and records. It does not load raw neuroimaging data, run a full fMRI GLM, perform EEG preprocessing/source reconstruction, lesion-symptom mapping, TMS protocol design, or replace domain-specific statistical packages. Missing data remain missing; the validator does not invent study results.

## Shared SQLite annotations

`register_point10_annotations()` indexes the module's classes and functions in the existing repository-wide `ore_logger.annotations` SQLite table. It does not create a separate Point 10 database.

## Limitations and safeguards

- Activation/association is not necessity or causation.
- fMRI BOLD is indirect; EEG/MEG source localization is an inverse problem.
- ERP components are not one-to-one process labels.
- Lesions can disconnect networks and are constrained by vascular anatomy; recovery and compensation matter.
- Stimulation is not perfectly focal and requires appropriate safety/ethics protocols.
- Group averages can conceal individual variation.
- Null results are ambiguous and do not prove absence.
- Model fit does not prove biological implementation.
- Shared participants, stimuli, and preprocessing create dependent evidence; source grouping is explicit.
- A high confidence label is never produced merely because multiple method fields exist: study safeguards and independent source groups also matter.

## Tests

`tests/test_language_comprehension_point10.py` covers the 15-step trace, missing-data behavior, behavioral effect sizes, fMRI caveats and QC, EEG/MEG trial safeguards, lesion and stimulation limitations, competing explanations, evidence dependence, held-out metrics, serialization, and shared SQLite annotation registration.
