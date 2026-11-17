# Distributed Neural Semantic Composition and Integration — 36 stages

## Purpose and evidence boundary

ore_distributed_semantics implements the 36-stage algorithm in the supplied
Distributed Neural Semantic Composition and Integration source. It extends
ORE additively and reuses ore_semantic.SemanticConcept and the repository-wide
ore_logger SQLite catalogue and run recorder.

The implementation deliberately keeps three kinds of claims separate:

1. **Neuroscientific observations** — findings from behavioral, lesion,
   stimulation, electrophysiological, or neuroimaging studies.
2. **Neuroscientific interpretations** — proposed roles for networks such as
   bilateral anterior temporal lobes (ATL), inferior frontal gyrus (IFG),
   angular gyrus (AG), posterior temporal cortex, hippocampal memory systems,
   and the default-mode network.
3. **Computational abstractions** — mathematical functions used here to
   implement a transparent model, including vector aggregation, cosine
   similarity, softmax, Bayesian-style scoring, graph propagation, Euler
   integration, and current-based leaky integrate-and-fire (LIF) dynamics.

A passing software test shows that code behaves as specified; it does not
establish that the code is a biologically accurate reproduction of a human
brain. The LIF track is synthetic model activity, not recorded neural data.
N400 is represented by a difficulty proxy, not by a measured ERP voltage.

Stages 1–30 describe normal semantic processing, learning and adaptation.
Stages 31–35 are controlled impairment/intervention simulations. Stage 36
evaluates the model; these latter stages are not asserted to be serial steps
that every person performs on each comprehension episode.

## Stage registry

| Stage | Functional purpose | Main computational families | Principal biological reference |
|---:|---|---|---|
| 1 | Multimodal input acquisition | Typed feature maps, availability masks, provenance | Primary sensory and association pathways |
| 2 | Linguistic and lexical identification | Tokenization, lexical similarity, candidate ranking | Auditory/visual language and posterior temporal networks |
| 3 | Initial conceptual candidate activation | Associative activation spread, nonlinear response, softmax | Distributed temporal semantic populations |
| 4 | Modality-specific feature retrieval | Feature maps, graded activation, relevance weighting | Visual, auditory, motor, tactile and other spokes |
| 5 | Cross-modal feature association | Pairwise cosine similarity, weighted links, connection gain | Long-range association pathways |
| 6 | ATL multimodal convergence | Reliability weighting, pooling, nonlinear convergence | Bilateral anterior temporal semantic system |
| 7 | Conceptual abstraction/generalization | Prototype estimation, feature intersection, residuals | ATL and distributed semantic association cortex |
| 8 | Associative semantic graph | Typed graph, weighted edges, activation propagation | Distributed temporal semantic networks and AG pathways |
| 9 | Linguistic/relational frame | Dependency graph, structural hypotheses | Posterior temporal language regions and IFG |
| 10 | Early ATL composition | Feature interaction, modifier-head binding | Bilateral ATL and frontotemporal language networks |
| 11 | Thematic roles | Predicate frames, role probabilities, event structure | Posterior temporal, ATL, IFG and temporoparietal networks |
| 12 | Selectional compatibility | Conditional preference, cosine compatibility, surprisal | Middle temporal semantic and semantic-control networks |
| 13 | Anticipatory prediction | Conditional scoring, candidate distributions, entropy | Temporal semantic, AG, medial temporal, DMN and IFG |
| 14 | Context-conditioned interpretation | Feature gates, Bayesian-style posterior | ATL, middle temporal regions, AG and IFG |
| 15 | Ambiguity/polysemy competition | Weighted evidence, softmax, entropy | Temporal semantic populations and IFG |
| 16 | Semantic control/selection | Relevance gates, inhibition, candidate margins | Left IFG (including BA45/47) and posterior MTG |
| 17 | AG/posterior temporal integration | Weighted pooling, relational tensor-like interactions | Angular gyrus, posterior parietal/temporal cortex, DMN |
| 18 | Posterior temporal relational integration | Typed event edges, relation confidence | Posterior MTG and temporal association cortex |
| 19 | Prediction-error calculation | Vector difference, norm, negative log probability | Distributed temporal prediction/integration networks |
| 20 | N400-related evaluation | Surprisal, difficulty weighting, negative-going proxy | Distributed N400-sensitive cortical generators |
| 21 | Late reanalysis | Candidate rescoring, constrained optimization | IFG, posterior temporal and language-control networks |
| 22 | Episodic association | Cue construction, similarity retrieval, weighted recall | Hippocampus, entorhinal and medial temporal systems |
| 23 | Schema inference | Schema matching, conditional confidence | Medial temporal, vmPFC, ATL, AG and DMN |
| 24 | Discourse integration | Entity tracking, reference resolution, state update | Posterior temporal/AG, hippocampal and DMN systems |
| 25 | Executive modulation | Demand estimation, resource allocation | IFG and frontoparietal control networks |
| 26 | Semantic learning | Confidence-weighted Hebbian-style updates | Cortical plasticity and medial temporal consolidation |
| 27 | Developmental maturation | Experience-dependent parameter refinement | Developing sensory, temporal, language and memory networks |
| 28 | Aging adaptation | Latency model, efficiency scaling | Temporal semantic, frontal-control and memory systems |
| 29 | Multilingual control | Shared concept mapping, language-specific lexical selection | Temporal semantic and inferior-frontal language-control networks |
| 30 | Multimodal educational reinforcement | Coverage, cross-modal co-activation, learning proposal | Sensory association, ATL convergence and memory systems |
| 31 | Bilateral ATL degradation | Severity scaling, specificity and consistency proxies | Clinical reference: semantic dementia |
| 32 | Unilateral ATL damage | Left/right gain, residual capacity, compensation assumption | Clinical reference: unilateral temporal resection/focal damage |
| 33 | Temporary ATL perturbation | Reversible attenuation and recovery assumption | Clinical reference: temporary perturbation/TMS |
| 34 | Semantic-control impairment | Control-gain attenuation with preserved knowledge | Clinical reference: semantic aphasia/control impairment |
| 35 | Multimodal integration impairment | Modality-specific connection attenuation | Clinical reference: distributed/spoke disconnection |
| 36 | Semantic performance evaluation | Accuracy, speed proxy, cross-modal consistency, uncertainty | Behavioral/neuropsychological assessment; independent neural validation |

The stable registry is available from
ore_distributed_semantics.registry.STAGE_REGISTRY. Every stage has three
independently traceable sub-operations, yielding 108 operations with
identifiers DS-01.01 through DS-36.03.

## Functional architecture

The functional track uses an explicit SemanticConcept inventory rather than
silently inventing definitions. Its major reusable calculations are:

- **Sparse feature vectors and similarity:** cosine similarity on the union of
  named feature dimensions, with a defined zero-vector result.
- **Candidate competition:** a numerically stable softmax converts lexical,
  contextual, association, and task evidence into relative candidate weights.
- **Associative retrieval:** weighted edges propagate activation over only
  caller-supplied or explicitly learned associations.
- **ATL-like integration:** modality availability and reliability weight feature
  contributions into a shared concept state; distinctive information is
  retained separately.
- **Composition and roles:** phrase features, event frames, and typed relation
  edges are kept distinct from mere co-activation.
- **Prediction:** association-based continuation candidates are ranked and
  uncertainty is measured with Shannon entropy.
- **Prediction error:** vector/score mismatch and -log P(observed|context)
  provide complementary measures of error and surprisal.
- **Learning:** a bounded update is previewed by default. Persistent model
  changes occur only when allow_learning=True.
- **Clinical simulation:** requested severities are explicit inputs. They
  change modeled capacity or connection gains; they do not diagnose a person
  or predict an individual clinical outcome.

When syntax or semantic evidence is not supplied, the prototype uses only
bounded transparent heuristics and labels their source. It does not claim to
be a trained parser, a comprehensive world model, or a calibrated language
understanding system.

## Biological-reference architecture

SemanticLIFSimulator models 14 named populations: modality-linked spokes,
left/right ATL, IFG semantic control, AG, posterior temporal cortex,
hippocampal memory, default-mode network, and inhibitory interneurons.

For population i, a simplified current-based LIF update is:

    I_i(t) = I_drive,i + I_recurrent,i - I_inhibitory,i

    dV_i/dt = (V_rest - V_i + 10^-3 R_m I_i) / tau_m

where current is represented in pA, resistance in MΩ, and the factor
10^-3 converts R_m I_i to mV. Euler integration advances the membrane state.
When voltage reaches threshold, the model records a spike, resets voltage,
and enters a refractory period. Population activity feeds back into the next
time step.

This is an intentionally lightweight reference model. Parameters are
engineering defaults, not fitted human membrane constants; population labels
are proposed mappings, not proof of one-region/one-computation anatomy.
Synthetic spike counts and firing-rate proxies must not be reported as human
measurements.

## API

    from ore_semantic.contracts import SemanticConcept
    from ore_distributed_semantics import (
        DistributedSemanticConfig,
        DistributedSemanticProcessor,
        DistributedSemanticRequest,
    )

    concepts = [
        SemanticConcept(
            concept_id="instrument-violin",
            label="violin",
            semantic_features={"ARTIFACT": 1.0, "MUSICAL_INSTRUMENT": 1.0},
            modality_features={
                "visual": {"WOODEN_BODY": 0.9, "STRINGS": 1.0},
                "auditory": {"BOWED_TONE": 1.0},
            },
            lexical_category="noun",
            verified_by="example",
            source_reference="example://violin",
        )
    ]

    processor = DistributedSemanticProcessor(
        concepts,
        config=DistributedSemanticConfig(simulate_biology=True),
    )
    request = DistributedSemanticRequest(
        sequence_id="reading-001",
        text="violin",
        modality_inputs={"visual": {"STRINGS": 0.9}},
    )
    result = processor.process(request)

    print(len(result.stage_trace))       # 36
    print(len(result.operation_trace))   # 108
    print(len(result.biological_trace))  # 36 when enabled
    print(result.output["selected_concepts"])

To use pre-tokenized input, pass tokens instead of relying on the text
tokenizer. Caller-provided syntactic relations, thematic roles, multilingual
lexicons, episodic memories, schemas, discourse history, evaluation targets,
and interventions are all optional and remain traceable as supplied evidence.

## SQLite annotations and run history

The package reuses ore_logger rather than creating another database schema:

    from pathlib import Path
    from ore_distributed_semantics import (
        index_distributed_semantic_annotations,
        record_distributed_semantic_result,
    )

    root = Path.cwd()
    catalogue = index_distributed_semantic_annotations(root)
    run_metadata = record_distributed_semantic_result(result, root)

The annotation indexer invokes ore_logger.catalog.sync_repository, which
stores module, class, function, and method docstrings in the existing
annotations table. record_distributed_semantic_result stores the 108
operation descriptions and 36 stage traces as run events using the standard
runs and events tables. The default database is
ore_logs/ore_logger.sqlite3; ORE_LOGGER_DB remains supported by the shared
logger runtime. No new package-specific database is created.

## Run and test

    python -m ore_distributed_semantics
    python -m pytest tests/test_distributed_semantics.py -q
    python -m pytest

The demonstration indexes the package annotations and records a SQLite run.
The focused tests check registry completeness, end-to-end functional and LIF
traces, opt-in learning, symbol annotations, and run-event persistence.
