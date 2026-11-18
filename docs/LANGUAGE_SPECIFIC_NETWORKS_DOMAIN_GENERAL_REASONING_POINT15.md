# Point 15 — Language-Specific Networks and Domain-General Reasoning

The package implements the 15 ordered stages in the Library specification.

## Architecture principle: separation with coordination

The model keeps six representational streams distinct and allows them to exchange information:

- L — language-selective representation: perceptual language units, lexical candidates, constructions, structural relations, and composed sentence meaning.
- K — conceptual knowledge: relevant long-term semantic knowledge cued by linguistic representations.
- C — control/working-memory state: demand estimates and resource allocation for maintenance, selection, and deliberate comparison.
- R — deliberate reasoning: conclusions derived from explicit rules or additional premises; these are marked as inferred.
- S — social-pragmatic model: uncertain speaker-intention hypotheses, recruited only when needed.
- E — mismatch/error state: observed-versus-predicted conflicts classified by affected layer.

This separation is functional rather than absolute. It does not imply independent anatomical boxes or a rigid one-region/one-function organization.

## Fifteen-stage processing sequence

1. Establish the task and preserve the ordered sensory signal, modality, timing/prosodic or orthographic cues, missing segments, and signal quality.
2. Convert perceptual features into candidate phonological/orthographic units and combine sensory fit with local context while preserving ambiguity.
3. Activate a recurrent state for the distributed frontotemporal language-selective network.
4. Retrieve lexical candidates and word-related linguistic properties using sensory, contextual, frequency, and learned-strength evidence.
5. Retrieve learned constructions and constraints; build an incremental structural hypothesis and retain unresolved dependencies.
6. Compose sentence meaning according to lexical and structural relations while retaining an exact direct-content record.
7. Retrieve relevant conceptual knowledge and distinguish it from information explicitly encoded by the sentence.
8. Estimate processing demand using ambiguity, working-memory/dependency load, novelty, and deliberate-comparison needs.
9. Recruit domain-general control and working-memory support according to demand relative to current capacity.
10. Keep language interpretation distinct from deliberate reasoning; apply explicit inference rules only when the task requires them.
11. Recruit social-pragmatic inference conditionally and preserve uncertainty over speaker-intention hypotheses.
12. Integrate language, knowledge, control, reasoning, and pragmatic outputs while recording provenance and assumptions.
13. Detect prediction mismatch, classify the affected layer, and reopen only the relevant representation.
14. Select a comprehension state, answer, response, or action plan without unsupported elaboration.
15. Update lexical and constructional learning gradually from reliable repeated feedback rather than one anomalous event.

The operation registry exposes the exact source order. Each result includes one trace row per stage, containing the operation identifier, explanation, mathematical form, functional output, and biological-surrogate diagnostics.

## Mathematical operations and their scope

Implemented computations include:
- Ordered token/feature extraction and signal-quality normalization.
- Weighted sensory/context candidate scoring and temperature-scaled softmax competition.
- Bayesian-style lexical candidate ranking using sensory fit, contextual priors, frequency, and learned strengths.
- Partial construction matching, dependency tracking, and confidence ranking.
- Compositional structure records and confidence aggregation.
- Concept relevance scoring from semantic/context overlap and source reliability.
- Demand estimation over ambiguity, dependency load, novelty, and task goal.
- Sigmoid recruitment as a graded proxy for control allocation.
- Explicit rule application with premises/conclusions separated from sentence content.
- Conditional speaker-intention hypotheses and entropy-based uncertainty.
- Provenance-preserving integration across subsystem outputs.
- Prediction residuals, mismatch classification, and targeted revision records.
- Evidence-constrained response/action selection.
- Reliability- and repetition-weighted lexical/construction strength updates.

These formulas are candidate computational descriptions. They are not direct measurements of neuronal arithmetic, fixed neural constants, or literal simulations of neural microcircuits.

## Biological functional surrogates

The biology module contains nine named network-level activity records:
- Auditory/visual perceptual pathways.
- Distributed frontotemporal language-selective network.
- Distributed conceptual-semantic systems.
- Multiple-demand/domain-general control.
- Deliberate reasoning.
- Social-pragmatic cognition.
- Cross-network integration and monitoring.
- Language production and task output.
- Hippocampal/medial-temporal and cortical learning.

The core language system remains responsible for linguistic representation and composition. Conceptual retrieval supplies world knowledge. Multiple-demand circuitry contributes control support when needed, and social-cognition systems contribute speaker-perspective inference when the task warrants it. The precise anatomy and interactions are distributed and task-dependent; these software objects are functional surrogates rather than an anatomical connectome.

## Scientific and data-provenance safeguards

- Speech input is represented as a transcript or symbolic feature sequence; the package does not decode raw audio waveforms.
- Text modality preserves orthographic/order information; a real OCR engine is outside this package.
- The structural builder is a lightweight partial parser, not a complete grammar engine.
- Conceptual memory is labeled as retrieved knowledge, never silently copied into direct utterance content.
- Deliberate reasoning conclusions are stored under inferred and do not rewrite said_or_encoded.
- Pragmatic inference is conditional; routine literal comprehension does not automatically invoke elaborate mind-reading.
- Speaker beliefs and intentions are uncertain hypotheses, not facts.
- Mismatch revisions reopen affected layers while preserving the original input record.
- Long-term lexical/constructional updates require reliable repeated evidence.
- Biological engagement counters are model diagnostics, not neural recordings.
- A difficult language task can recruit both language and domain-general control systems without making them one undifferentiated system.

## SQLite annotations and operation logging

The register_annotations function indexes Point 15 classes and callable annotations in the ORE annotation catalogue.

The record_result_to_sqlite function synchronizes the package annotation catalogue and stores all 15 operation trace entries in ORE logger runs and events tables. Each event carries its operation ID, status, evidence level, mathematical abstraction, functional output, and biological-surrogate output.

## Quick start

\`\`\`python
from ore_language_domain_reasoning import LanguageDomainReasoningProcessor, LanguageEvent

processor = LanguageDomainReasoningProcessor()
event = LanguageEvent(
    event_id="example-001",
    sensory_stream="The metal expanded after heating.",
    modality="text",
    conceptual_memory={
        "thermal_expansion": {
            "description": "many materials expand when heated",
            "reliability": 0.95,
        }
    },
)
result = processor.process(event)
print(result.task_result)
print(len(result.operation_trace))  # 15
\`\`\`

## Validation coverage

Tests cover source order, modality-specific signal handling, lexical ambiguity, separation of explicit language from conceptual knowledge, demand-sensitive control, inference provenance, conditional pragmatics, mismatch classification, learning thresholds, full biological/functional pipeline execution, and SQLite annotation/operation logging.
