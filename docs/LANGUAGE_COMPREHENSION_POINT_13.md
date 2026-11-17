# Point 13 — Individual Differences and Developmental Changes in Language Comprehension

## Scope and scientific boundary

Point 13 adds an individualized, developmental parameter layer to the earlier comprehension pipeline. It does not replace sensory processing, lexical access, parsing, semantic composition, discourse integration, or pragmatic inference. It estimates how strongly a current comprehender can use each operation and how much support is available from context, memory, executive control, experience, and world knowledge.

The implementation follows Part 13 of Language_Comprehension_Brain_Algorithms_Parts_01-15.md. It deliberately avoids assigning fixed profiles from age, diagnosis, first/second-language labels, or one broad language score. Missing measurements remain unknown rather than being treated as zero ability.

Statistical estimation, hierarchical modelling, regression, held-out scoring, and confidence calibration are researcher/computational-system operations. Candidate biological processes such as competition, predictive activation, inhibition, and synaptic plasticity are discussed as possible mechanisms, not as proven one-to-one mappings from fitted parameters to neurons.

## The 18 ordered operations

1. **Define a task-specific baseline.** Record the target operation, sensory modality, low-context and high-context conditions, controls, accuracy, response time, and error types.
2. **Estimate component abilities separately.** Keep lexical efficiency, vocabulary, perception, working memory, inhibition, world knowledge, pragmatic perspective-taking, context use, and reanalysis distinct. Retain sample size and standard error.
3. **Characterize linguistic experience and proficiency.** Store exposure, acquisition history, use, proficiency, and target-word familiarity separately for each language.
4. **Estimate contextual facilitation.** Compare matched low- and high-context response time and accuracy; interpret speed and accuracy jointly.
5. **Estimate cue weighting.** Preserve empirically supplied cue weights and normalize candidate scores with a numerically stable softmax where candidate evidence is supplied.
6. **Separate context compensation from core efficiency.** A context benefit is not interchangeable with lexical or perceptual efficiency.
7. **Model developmental trajectories.** Fit component-specific descriptive slopes only when repeated measurements at distinct times are available. Age is not a hard threshold.
8. **Model pragmatic development conditionally.** Distinguish reference, indirect requests, implicatures, irony, and belief inference. Do not assume every pragmatic task requires the same theory-of-mind computation.
9. **Model multilingual processing by language.** Keep proficiency and cue-use parameters language-specific; do not impose a universal L2 penalty.
10. **Test aging-related compensation and interference.** Inspect supplied lexical, world-knowledge, context-benefit, and interference measures without assuming a uniform aging profile.
11. **Estimate memory and executive support demands.** Combine task length, dependency load, competitors, and revision demand using configurable weights; do not invent a universal capacity threshold.
12. **Separate stable differences from momentary state.** Preserve session-level and state-labelled performance rather than interpreting one trial as a stable trait.
13. **Estimate individual and group variation.** Summarize participant-level effects and supplied moderators without treating a group mean as every person's profile.
14. **Compare compensatory strategies.** Score supplied strategy predictions on paired held-out observations and apply a complexity penalty only when explicitly configured.
15. **Adapt downstream parameters.** Configure alternative retention, context support, referent maintenance, and revision traces from measured profile values. Adaptations are reversible and thresholds are configurable.
16. **Preserve uncertainty and alternatives.** Missing components remain missing; low evidence creates unresolved hypotheses rather than fabricated precision.
17. **Evaluate calibration and generalization.** Compare predicted confidence with accuracy and compute held-out loss only for paired observations. A generalization claim requires a frozen model and independent test set.
18. **Update the profile reversibly.** Combine a previous estimate and new evidence with a bounded learning rate; do not permanently lock the model into a strategy.

## Mathematical operations

| Operation | Computational role | Scientific interpretation |
|---|---|---|
| Mean, variance, standard error | Summarize component measurements | Research measurement, not neural firing |
| Context benefit: RT_low − RT_high | Quantify response-time facilitation | Behavioural statistic; interpret with accuracy |
| Softmax | Normalize candidate evidence | Computational hypothesis, not proof of explicit neural probability calculation |
| Linear regression / nonlinear growth models | Describe repeated developmental measures | Statistical trajectory, not a biological growth law |
| Hierarchical models | Separate group tendencies and individual variation | Research model |
| Resource-demand weighted sum | Represent task load | Configurable computational abstraction |
| State-space update | Separate gradual profile change from temporary state | Dynamic modelling abstraction |
| Held-out MSE and calibration error | Evaluate prediction and confidence | Model evaluation, not direct brain computation |
| Competition and inhibition | Represent alternatives and interference | Biologically plausible operations requiring task-specific evidence |
| Predictive activation and synaptic plasticity | Represent expectation and experience-sensitive change | Candidate biological processes; implementation is not identified by this algorithm |

## API

Point13Request accepts:

- participant_id and target_operation;
- stimulus, discourse_context, and task_requirements;
- trial-level observations;
- component_measures for the nine distinct profile dimensions;
- experience_profile, including language-specific records;
- baseline_conditions with low_context and high_context;
- optional longitudinal_measures, cohort_observations, and candidate_strategies;
- optional previous_profile and metadata.

Run:

    from ore_language_comprehension import Point13IndividualizedProcessor, Point13Request

    request = Point13Request(
        participant_id="participant-01",
        target_operation="indirect-request-comprehension",
        stimulus={"modality": "auditory", "language": "en"},
        baseline_conditions={
            "low_context": {"response_time": [700, 720], "accuracy": [0, 1]},
            "high_context": {"response_time": [550, 570], "accuracy": [1, 1]},
        },
    )
    result = Point13IndividualizedProcessor().process(request)
    print(result.profile)
    print(result.adapted_parameters)
    print(result.uncertainty_estimates)
    print(result.trace)

All public classes and functions have docstrings and can be indexed through register_point13_annotations() in the existing shared ore_logger.annotations SQLite catalogue.

## Safeguards

- This is not a diagnostic algorithm.
- No age, diagnosis, or language-background label determines a person's profile.
- A context benefit does not prove a core language impairment or a particular neural locus.
- Pragmatic tasks are not interchangeable.
- Behavioural response time is not a direct neural measurement.
- Group averages do not describe every individual.
- Missing evidence is not evidence of absence.
- The algorithm preserves task-specific uncertainty and can revise its estimates when new evidence arrives.

## Focused tests

    python -m pytest tests/test_language_comprehension_point13.py -q
