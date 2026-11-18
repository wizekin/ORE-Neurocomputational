# Point 17 — Adult Individual Differences in Learning and Comprehension

This package implements the 15 source-ordered operations in the Point 17 Library specification. It models adult comprehension as a changing interaction between the current task, the person's relevant experience, and the available processing conditions. It does **not** assign fixed ability labels, estimate intelligence, diagnose cognitive decline, or infer cultural knowledge from demographic groups.

## Architecture

The learner profile keeps these dimensions separate:

- **Attention control and working-memory support:** task-sensitive availability shaped by distraction, fatigue, stress, motivation, and external support.
- **Processing speed:** component-specific estimates for perception, retrieval, integration, and decision, with time pressure represented explicitly.
- **Prior knowledge and expertise:** domain-indexed schemas, reliability, recency, relations, exceptions, and learned chunk definitions.
- **Language experience:** language-specific proficiency and learned reliability for individual cues.
- **Cultural and lived experience:** contextual conventions tied to demonstrated cues, individual memory, and current task evidence.
- **Memory specificity:** separate estimates for gist, episodic detail, and source/context retrieval.
- **Learning history:** semantic estimates, exception records, source-linked episodes, and domain performance histories.

These are interacting distributed systems, not isolated anatomical modules. The \`biology.py\` module provides inspectable, rate-coded network surrogates for goal attention, working memory, temporal coordination, semantic knowledge, expertise chunking, language processing, contextual knowledge, hippocampal binding, conflict monitoring, adaptive support, gist/detail/source memory, prediction-error learning, and confidence decisions.

## Fifteen-stage algorithm

1. **Detect the task and establish the goal.** Build a normalized objective vector covering detail, gist, relations, procedure, and source accuracy. Derive feature priorities from the actual task rather than assuming that content alone specifies the goal.
2. **Estimate current conditions and available resources.** Estimate task-specific attention and working-memory availability from present distraction, fatigue, stress, motivation, and external scaffolding. The model does not convert one task into a permanent ability score.
3. **Segment input into source-linked units.** Tokenize or preserve supplied proposition/event/instruction units in order. Attach source IDs, confidence, and provenance, and retain ambiguous boundary estimates.
4. **Retrieve relevant prior knowledge and expertise.** Rank domain-specific schemas using feature overlap, goal fit, reliability, recency, conflict, and the expertise associated with that domain. A skill in another domain is not treated as transferable expertise by default.
5. **Convert familiar details into chunks when justified.** Compare separate-representation cost with a schema-chunk cost. Compress only when schema match and confidence are adequate and the representation cost is lower; preserve exceptions and novel details.
6. **Estimate element interactivity and working-memory load.** Combine active units, dependencies, unresolved references, propositions, and competing alternatives. Compare load with the current working-memory availability and recommend decomposition when pressure is high.
7. **Account for processing speed and temporal coordination.** Estimate perception, retrieval, integration, and decision times separately. Expose component bottlenecks, time budget, and temporal availability; slower processing is not equated with low knowledge or intelligence.
8. **Weight language cues through proficiency and experience.** Combine candidate priors and cue likelihoods with proficiency-sensitive, learned cue reliabilities for the active language. Neutral evidence remains neutral; bilingual status alone produces no assumed universal benefit or disadvantage.
9. **Incorporate cultural and lived-experience associations.** Retrieve conventions only when individual context or current cues support them. Keep multiple contextual readings when evidence is weak; demographic group labels are not a substitute for demonstrated experience.
10. **Integrate input with active knowledge while preserving provenance.** Maintain separate observed, retrieved, and inferred records. A schema-driven assumption is not silently rewritten as something explicitly stated.
11. **Monitor interference and competing interpretations.** Quantify uncertainty and candidate separation. Retain a bounded set of interpretations when the evidence is close and the cost of a wrong commitment is meaningful; later evidence can reopen alternatives.
12. **Adapt support to the person–task match.** Provide more definitions, worked examples, smaller steps, and external notes when relevant knowledge is low or load is high. Reduce redundant guidance for an expert while retaining checks for exceptions and superficial schema matches.
13. **Separate gist, episodic detail, and source retrieval.** Estimate these outcomes independently using the actual cue quality, encoding time, distinctiveness, and interference. Age may be recorded as context, but no automatic age penalty is applied.
14. **Update knowledge without false assimilation.** Classify incoming evidence as a new observation, expected variation, novel exception, contradiction, or possible input error. Weight updates by reliability and repetition, and retain source-linked episodes and unresolved conflicts.
15. **Calibrate confidence and select the next action.** Combine posterior separation, uncertainty, evidence quality, schema reliability, provenance completeness, and inference share. Choose whether to proceed, slow down/decompose, retrieve a source, seek clarification, or revisit the interpretation.

The registry exposes \`P17-01\` through \`P17-15\`. Each processing result has one trace row per operation with purpose, equation, functional output, and biological-network activity.

## Mathematical operations implemented

- Goal-weight normalization and weighted feature relevance.
- Logistic activation for current attention/control availability.
- Ordered token and proposition representations with boundary confidence.
- Jaccard feature overlap, schema scoring, and softmax candidate normalization.
- Separate-versus-chunked representation cost.
- Dependency-graph cost and element-interactivity load pressure.
- Component time estimates, exponential temporal availability, and time-budget pressure.
- Proficiency-sensitive cue likelihood weighting and Bayesian-style posterior ranking.
- Context-hypothesis mixture weights from individually demonstrated cues.
- Provenance-separated observed/retrieved/inferred situation representations.
- Entropy and best-versus-second candidate conflict for uncertainty monitoring.
- Adaptive support as a function of load, expertise, relevant knowledge, and recent errors.
- Separate logistic retrieval estimates for gist, episodic details, and source context.
- Reliability/repetition/novelty-sensitive incremental knowledge updates and prediction-error classification.
- Confidence calibration from candidate probability, normalized uncertainty, evidence quality, schema reliability, and provenance.

The coefficients are transparent modeling choices, not universal psychological constants. They require empirical calibration before any predictive or clinical use.

## Biological modeling boundary

The network activity uses a bounded recurrent update:

    a_t = (1 - decay) * sigmoid(input_gain * drive
          + recurrent_gain * a_previous
          - inhibition_gain * a_previous**2)
          + decay * a_previous

This is an interpretable population-rate surrogate. It is not a direct neural measurement, a detailed biophysical simulation, or a one-region/one-function map.

## Important safeguards

- Working memory is not modeled as a fixed number of empty slots. Attention, maintenance, retrieval, knowledge, task structure, and external support contribute.
- Processing speed is task- and component-dependent; it is not an intelligence score.
- Prior knowledge and expertise are domain-specific and can improve or mislead interpretation.
- Schema similarity is not truth. Novel details and contradictions remain explicit.
- Language cue weights reflect a particular language experience and proficiency estimate; no universal bilingual advantage/disadvantage is assumed.
- Culture is represented through demonstrated individual experience and situation cues, not group stereotypes.
- Gist is not treated as equivalent to detail or source memory.
- Age is contextual metadata, not a deterministic cognitive decline parameter or a diagnosis.
- A single performance observation cannot create a permanent ability label.
- Mathematical and network equations are computational analogies, not literal neural calculations.

## SQLite annotations and operation logging

\`register_annotations(database_path)\` indexes Point 17 classes and callables in the ORE annotation catalogue.

\`record_result_to_sqlite(result, database_path)\` synchronizes the annotations and writes all 15 operation records to ORE logger \`runs\` and \`events\` tables. Each event records the operation ID, mathematical form, functional output, and biological output.

## Quick start

\`\`\`python
from ore_adult_individual_differences import (
    AdultComprehensionExperience,
    AdultIndividualDifferencesProcessor,
    AdultLearnerState,
)

learner = AdultLearnerState(
    prior_knowledge={
        "physics": [{
            "schema_id": "thermal-expansion",
            "schema": "thermal expansion",
            "features": ["metal", "heated", "temperature", "expansion"],
            "reliability": 0.95,
        }]
    },
    expertise_by_domain={"physics": 0.8},
    language_proficiency={"english": 0.95},
)

processor = AdultIndividualDifferencesProcessor(state=learner)
event = AdultComprehensionExperience(
    experience_id="physics-example-001",
    content_units=("Metal", "expands", "when", "heated"),
    domain="physics",
    task_goal={"detail": 0.3, "gist": 0.2, "relation": 0.4, "source": 0.1},
    language="english",
    dependency_pairs=((0, 1), (1, 3)),
    external_support=0.2,
    evidence_quality=0.9,
)
result = processor.process(event)
print(len(result.operation_trace))  # 15
print(result.calibrated_decision["next_action"])
\`\`\`

## Validation

The focused tests cover source ordering, normalized goals, task-sensitive resource estimates, source-linked segmentation, domain-specific knowledge retrieval, expertise chunking, working-memory load, processing-speed timing, language cue weighting, cultural-context safeguards, provenance, interference, adaptive support, gist/detail/source retrieval, contradiction-safe updates, confidence calibration, biological rate dynamics, and SQLite annotation and operation logging.

The scientific background and references are preserved in the Point 17 source research file. This implementation translates those principles into testable software abstractions without claiming that the equations are literal neural operations.
