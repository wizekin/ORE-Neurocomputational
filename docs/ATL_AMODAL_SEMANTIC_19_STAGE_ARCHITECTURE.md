# Amodal Semantic Processing Through the Anterior Temporal Lobe

## Implementation overview

This package implements the 19-stage algorithm described in the Library source file "Amodal semantic processing through the anterior temporal lobe.txt".

The architecture has two separate execution tracks:

1. **Functional track** — deterministic feature encoding, modality-specific semantic spokes, normalization and reliability gates, bilateral ATL convergence, common semantic representations, cross-modal prototype matching, top-down feedback, lexical/concept retrieval, adjective–noun composition, denotational constraints, temporal refinement, and Hebbian plasticity.
2. **Biological reference track** — a generic current-based leaky-integrate-and-fire (LIF) simulation over coarse sensory-spoke, distributed feature, left/right ATL, hub, feedback, semantic-memory, and phrase-composition populations. It reports membrane voltage (mV), synaptic current (pA), firing rate (Hz), spike counts, and simulated time (ms).

The biological track is intentionally a reference simulation rather than a claim that the human brain executes the exact equations or numerical parameters in this package. Imaging and lesion evidence constrains plausible system-level roles; it does not expose literal neuronal calculations.

## The 19 stages

| ID | Functional operation | Mathematical operations | Biological reference |
|---|---|---|---|
| ATL-01 | Receive written, spoken, visual, tactile, action, affective, and functional evidence. Keep channel availability and provenance. | Input sets, availability masks, contextual feature encoding. | Parallel sensory and language pathways preserve channel-specific signals. |
| ATL-02 | Extract modality-specific feature vectors. | Deterministic feature projections, tanh, vector pooling. | Visual, auditory, somatosensory, motor, and language association processing. |
| ATL-03 | Activate distributed semantic spokes. | s_m = sigmoid(W_m x_m + b_m), thresholding, population vectors. | Distributed modality-linked feature populations. |
| ATL-04 | Normalize and weight incoming evidence. | Centering, variance scaling, tanh, reliability estimates, softmax gates. | Context- and quality-sensitive changes in effective gain/coupling. |
| ATL-05 | Transmit information toward left and right ATL. | Matrix projections u_m = W_m s_m; reliability-weighted summation. | Heterogeneous cortico-temporal pathways to bilateral ATL subdivisions. |
| ATL-06 | Integrate information recurrently within bilateral ATL. | Coupled recurrence, tanh, vector norms, bounded iterative convergence. | Interacting left/right transmodal populations and recurrent connectivity. |
| ATL-07 | Establish a common semantic representation. | Weighted pooling, cosine similarity, route residuals r_m = z_m - z_shared. | Hub-like cross-modal overlap with preserved modality-specific information. |
| ATL-08 | Generalize across modalities. | Prototype scoring with cosine similarity and softmax candidate probabilities. | Shared concept access across visual, auditory, and linguistic routes. |
| ATL-09 | Complete representation through feedback. | Hub-to-spoke projections, prediction error e_m = s_m - s_hat_m, feedback blending. | Top-down reactivation of distributed semantic spokes. |
| ATL-10 | Retrieve a concept for understanding. | Semantic similarity plus lexical cue score, candidate ranking, confidence/margin. | Transmodal semantic access and lexical-to-concept interaction. |
| ATL-11 | Activate the noun representation. | Concept-prototype encoding, nonlinear projection, context bias. | Word-related distributed semantic activation in bilateral ATL-connected networks. |
| ATL-12 | Maintain noun information while later words are processed. | Leaky integration n[t+1] = rho*n[t] + (1-rho)*n_input; stability cosine. | Recurrent/sustained activity maintaining task-relevant noun information. |
| ATL-13 | Process adjective information. | Separate adjective vector, contextual projection, dynamic state. | Modifier-related information with context-sensitive temporal evolution. |
| ATL-14 | Establish adjective–noun relationship. | Elementwise interaction a ⊙ n; bilinear interaction a^T B n. | Distributed semantic interaction between modifier and noun representations. |
| ATL-15 | Construct the complete phrase. | Nonlinear combination p = tanh(W_n n + W_a a + W_r r + b). | ATL participation in semantic composition, beyond isolated word access. |
| ATL-16 | Determine denotational effect. | Separate resemblance from category membership. Privative modifiers reduce membership independently of resemblance. | Distinction between subsective and privative adjective semantics. |
| ATL-17 | Refine the phrase over time. | Recurrent/leaky update; Bernoulli membership entropy as an uncertainty proxy. | Time-dependent change in phrase interpretation and modifier integration. |
| ATL-18 | Produce unified expression meaning. | Final vector normalization and structured output of noun, adjective, relation, resemblance, membership, and confidence. | Shared semantic mechanisms contributing to word and phrase representations. |
| ATL-19 | Strengthen cross-modal links through experience. | Hebbian update Delta W = eta * outer(hub, spoke) - eta * lambda * W; clipping and generation tracking. | Co-activation-dependent synaptic plasticity as a learning hypothesis. |

## Scientific distinctions preserved by the implementation

- An absent modality is not encoded as negative evidence.
- The left and right ATL are represented by separate, interacting populations and projection matrices.
- A shared semantic state does not erase modality-specific residual information.
- Conceptual resemblance and category membership are different quantities. For example, an imitation diamond can resemble a diamond but is not necessarily a member of the genuine-diamond category.
- Noun maintenance and adjective dynamics are separate stages.
- Learning proposals are preview-only by default. Persistent model weights change only when learning_authorized=True on a request or in the configuration.
- Numerical LIF parameters are illustrative and configurable, not estimates of an individual's neural physiology.

## Python usage

    from ore_atl_amodal import (
        AmodalATLProcessor, AmodalATLRequest, AmodalConcept, AmodalATLConfig
    )

    diamond = AmodalConcept(
        concept_id="diamond",
        label="diamond",
        semantic_features={"MINERAL": 1.0, "CRYSTAL": 0.9, "HARD": 0.9},
        modality_features={
            "visual": {"FACETED": 0.9, "TRANSPARENT": 0.8},
            "tactile": {"HARD": 1.0},
        },
        source_reference="your://verified-concept-source",
    )

    processor = AmodalATLProcessor(
        concepts=[diamond],
        config=AmodalATLConfig(simulate_biology=True),
    )

    result = processor.process(AmodalATLRequest(
        sequence_id="phrase-001",
        text="fake diamond",
    ))

    print(result.output["denotation"])
    print(len(result.stage_trace))       # 19
    print(len(result.operation_trace))   # every registered sub-operation

Pass named feature values in the range [0, 1] through modality_inputs, noun_features, or adjective_features. Supported modality aliases include visual/vision/image, auditory/audio/sound, tactile/touch, motor/action, linguistic/text, affective/emotional, and functional/use.

adjective_type accepts "auto", "subsective", or "privative". The illustrative automatic prior treats fake, counterfeit, and imitation as privative; callers should explicitly set the type when the intended semantics differ.

## SQLite annotations and execution traces

The implementation reuses the repository's ore_logger infrastructure.

Index package annotations:

    from ore_atl_amodal import index_atl_amodal_annotations

    report = index_atl_amodal_annotations(".")
    print(report["database_path"])
    print(report["atl_amodal_symbol_count"])

Record a processed result:

    from ore_atl_amodal import record_atl_amodal_result

    metadata = record_atl_amodal_result(result, repository_root=".")
    print(metadata["run_id"])
    print(metadata["operation_count"])

By default the database is stored at ore_logs/ore_logger.sqlite3. Every module, class, and function in ore_atl_amodal is AST-indexed into the annotations table using its docstring/source analysis. Every registered sub-operation and all 19 stage records are also saved as run events, linked to synthetic operation annotations in the same database.

## Command line

    python -m ore_atl_amodal
    python -m ore_atl_amodal --no-biology

## Tests

The acceptance suite is in tests/test_atl_amodal.py and covers the 19-stage registry, multimodal concept access, lexical-only input, phrase composition, privative/subsective denotation, LIF units and stage count, learning authorization, and logger-backed SQLite annotations/events.

Run with:

    pytest -q tests/test_atl_amodal.py
