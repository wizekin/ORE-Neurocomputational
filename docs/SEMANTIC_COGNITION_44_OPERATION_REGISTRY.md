# ORE Semantic Cognition — 44-operation registry

**Package:** `ore_semantic/`  
**Registry:** `ORE-SEMANTIC-COGNITION-44`  
**Version:** 1.0.0  
**Status:** Executable functional prototype with an optional current-based LIF reference network. Language coverage, behavioral validity, and biological correspondence require independent evaluation.

## 1. Purpose and boundary

This package implements a semantic-access subsystem for a word or ordered sequence of words. It models the interaction of:

1. episodic word-form memories;
2. modality-specific semantic spokes;
3. distributed semantic feature populations;
4. an anterior-temporal-lobe (ATL)-inspired transmodal hub;
5. inferior-frontal-gyrus (IFG)-inspired semantic control;
6. controlled concept retrieval and selection;
7. abstract lexical/category prototypes.

The package is **additive**. It does not replace the existing `ore_reading` 72-operation pipeline, ORE N1–N9, or the canonical `ore_contracts` interfaces. The reading pipeline remains responsible for its upstream visual/orthographic/phonological work. `ore_semantic` starts from text or already-tokenized word forms, or from a semantic adapter payload. It does not perform OCR, image segmentation, full syntax parsing, unrestricted discourse interpretation, or general world-model reasoning.

The 44 stable operation identifiers are `SC-001` through `SC-044`. Their exact names, functional responsibilities, and biological-reference descriptions are maintained in `ore_semantic/registry.py`. Both execution tracks refer to the same registry. Every functional run returns all 44 ordered records; the neural track returns one matching record per operation or an explicit skipped record when disabled.

## 2. Two tracks, one interface

### Functional track

The functional track is a deterministic, inspectable computational model. It:

- validates and normalizes input while retaining the original surface form;
- retrieves exact and near-match concepts from a provenance-tracked inventory;
- retrieves relevant word-form episodes when those are supplied;
- scores independent modality evidence channels;
- builds sparse distributed semantic feature states;
- settles a common hub-like representation through bounded recurrent updates;
- estimates semantic-control demand from competition and uncertainty;
- reweights candidates using task and contextual constraints;
- compares candidate features with read-only category prototypes;
- selects or abstains using explicit confidence and margin gates.

The functional evidence channels are not collapsed into one unexplained score. Candidate records retain lexical similarity, frequency prior, episode fit, modality fit, context fit, hub fit and prototype fit separately.

### Biological-reference track

The second track uses current-based leaky integrate-and-fire (LIF) populations for a transparent neural-dynamics reference. The population set includes lexical, episodic word-form, modality-spoke, distributed-feature, ATL-hub, IFG-control and abstract-prototype groups.

For neuron (i), the reference membrane update is a forward-Euler approximation:

[
V_i(t+Delta t)=V_i(t)+rac{Delta t}{	au_m}
left(V_{mathrm{rest}}-V_i(t)+R_m I_i(t)ight)
]

A spike occurs when (V_i) reaches threshold. The voltage is reset and the neuron enters a refractory interval. The synaptic-current proxy includes external functional drive, bounded recurrent excitation and population-level inhibition:

[
I_i(t)=I_{mathrm{external},i}(t)+g_r a_i(t)-g_iar a(t)
]

The software converts the current/voltage scale explicitly and reports:

- membrane voltage in **mV**;
- synaptic current in **pA**;
- firing rate in **Hz**;
- simulated time in **ms**;
- spike count as a dimensionless count.

This is a generic computational population model, not a fitted model of the human ATL, IFG, hippocampus, or complete semantic system. The operation-to-population routing is an explicit hypothesis for experiments. It is not proof that the brain implements the same 44 operations or equations.

## 3. Operation boundaries

| Range | Functional boundary | Biological-reference interpretation |
|---|---|---|
| SC-001–SC-006 | Input integrity, surface preservation, lexical key normalization, token order, task goals, immutable concept snapshot | Distributed input and task-goal signals; frozen software inventory is not a neural-memory claim |
| SC-007–SC-012 | Word-form candidates, phonological episode cues, episodic retrieval, form matching, recency/context weighting | Lexical and episodic populations with cue-driven reinstatement |
| SC-013–SC-020 | Visual, auditory/phonological, action, somatosensory, chemosensory/interoceptive, affective, spatial/temporal spokes and reliability | Modality-linked association populations; absent modalities are neutral rather than negative |
| SC-021–SC-026 | Distributed feature activation, associative propagation, integration, sparsification, feature-state construction and residual error | Distributed semantic feature populations and local competition |
| SC-027–SC-032 | ATL-like hub initialization, modality convergence, recurrent settling, supported completion, hub readout and lexical feedback | Transmodal convergence and recurrent temporal/lexical interaction |
| SC-033–SC-039 | IFG-like control demand, task goal, competing candidates, lateral inhibition, context reweighting, controlled retrieval, ambiguity | Frontal semantic-control and temporal-frontal competition |
| SC-040–SC-042 | Category prototype retrieval, feature centroid estimate, exemplar/prototype comparison | Abstract category-level representations derived from multiple exemplars |
| SC-043–SC-044 | Selection/abstention and complete result/evidence/acceptance report | Network outcome and separately reported neural measures |

## 4. Evidence integration

For candidate concept (c), the implementation builds a transparent weighted score from independent channels:

[
S(c)=
w_o,L(c)+w_f,F(c)+w_e,E(c)+w_m,M(c)+w_x,X(c)
]

where:

- (L(c)) is orthographic label/alias similarity;
- (F(c)) is a bounded frequency prior;
- (E(c)) is episodic retrieval strength;
- (M(c)) is the average fit of supplied modality evidence;
- (X(c)) is semantic-context compatibility.

Candidate activations are normalized using softmax. A bounded recurrent competition update combines candidate feedback and lateral inhibition. The common semantic state is then estimated from candidate-weighted feature maps:

[
mathbf{h}=operatorname{Normalize}left(
sum_c P(cmid e),mathbf{f}_c
+lambda_xmathbf{x}
ight)
]

Here (mathbf{f}_c) is the concept feature map and (mathbf{x}) is the supplied contextual feature map. The operation is an engineering approximation: the weights, similarity values and softmax scores are not empirical neural constants.

The ATL-inspired hub settles this state with damped iterations. Feature completion is kept separate from observed features. The implementation does not silently rewrite the observed channel.

IFG-like control is driven by candidate entropy and competition. It adds task- and context-dependent bias, then re-runs bounded competition. The final decision is made only if both confidence and top-candidate margin meet configured thresholds. Otherwise the result explicitly abstains and preserves alternatives.

## 5. Episodic retrieval and abstract prototypes

Episodic inputs are explicit `WordFormEpisode` records. Each record can carry:

- a remembered orthographic form;
- an optional concept ID;
- a phonological sequence, when known;
- semantic and contextual features;
- retrieval strength and recency;
- a source reference.

The processor performs read-only retrieval. It does not invent episodes or learn new ones during recognition.

Category prototypes are computed from the existing concept inventory as frequency-weighted feature centroids. Prototype fitting is included as a bounded decision signal, while each original concept and its source remain intact. Prototype creation is not treated as proof of human abstraction or biological learning.

## 6. API

```python
from ore_semantic import (
    SemanticConcept,
    SemanticCognitionProcessor,
    SemanticRequest,
    WordFormEpisode,
)

concepts = [
    SemanticConcept(
        concept_id="concept-apple",
        label="apple",
        aliases=("fruit",),
        semantic_features={"FOOD": 1.0, "PLANT": 0.8, "ROUND": 0.7},
        modality_features={
            "visual": {"ROUND": 0.9, "RED_OR_GREEN": 0.9},
            "gustatory": {"SWEET": 0.8, "TART": 0.4},
        },
        lexical_category="noun",
        frequency=30.0,
        prototype_id="food",
        verified_by="reviewer",
        source_reference="dataset://release/apple",
    ),
]

episodes = [
    WordFormEpisode(
        episode_id="episode-001",
        orthography="apple",
        concept_id="concept-apple",
        phonological_form=("AE", "P", "AH", "L"),
        strength=0.9,
        recency=0.8,
        source_reference="dataset://episodes/001",
    )
]

processor = SemanticCognitionProcessor(concepts)
result = processor.process(SemanticRequest(
    sequence_id="sample-001",
    text="apple",
    language="en",
    task="comprehension",
    context={"FOOD": 0.8},
    modality_evidence={"visual": {"ROUND": 0.8}},
    episodes=episodes,
))
print(result.tokens[0]["selected_concept"])
print(len(result.operation_trace))  # 44
print(result.biological["layers"]["atl_hub"]["mean_firing_rate_hz"])
```

Run a self-contained synthetic example:

```bash
python -m ore_semantic
python -m ore_semantic --output semantic-demo.json
```

To disable the biological-reference simulation:

```python
from ore_semantic import SemanticConfig, SemanticCognitionProcessor

processor = SemanticCognitionProcessor(
    concepts,
    config=SemanticConfig(simulate_biology=False),
)
```

## 7. Integration with existing ORE

### From the integrated reading result

`semantic_request_from_reading_result()` adapts the existing `ore_reading` result (or its dictionary form) into a `SemanticRequest`. It preserves sequence identity and token order and does not modify the original result.

### As the N9 N8 adapter

`SemanticN8Adapter` implements the existing N9 adapter method `compose(payload)`. It reads the lexical hypotheses produced upstream and returns:

- a typed graph containing candidate concept nodes;
- an empty proposition list (this package does not perform syntax or predicate-argument composition);
- concept-access completeness and ambiguity;
- the complete nested semantic-cognition result.

It intentionally does **not** invent graph edges, thematic roles, propositions, or syntax. Connect it through the existing injection point:

```python
from ore_n9 import N9Adapters, N9Processor
from ore_semantic import SemanticConcept, SemanticCognitionProcessor, SemanticN8Adapter

semantic_processor = SemanticCognitionProcessor(concepts)
adapters = N9Adapters(n8=SemanticN8Adapter(semantic_processor))
n9_result = N9Processor(adapters=adapters).process(n9_request)
```

The current adapter expects upstream N9 lexical hypotheses to include `form`, or an equivalent usable token field. If N6 is not configured and N9 only has visual fallback hypotheses, their inferred form can be used as a lexical query; the semantic package still does not claim that the form itself was verified.

## 8. Read-only inference and validation

Recognition is read-only. The processor pins a fingerprint of the concept inventory and reports that fingerprint in result metadata. Episodic records and derived prototypes are not updated during inference. Any future learning path must be implemented separately, with explicit authorization, detached child-model proposals, provenance, rollback and held-out evaluation.

Run focused tests:

```bash
python -m pytest tests/test_semantic_cognition.py -q
```

Run all repository tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Acceptance categories must remain separate:

- **Software integrity:** registry ordering, serialization, evidence preservation, deterministic trace length, numerical stability and adapter contracts can be tested automatically.
- **Behavioral validity:** requires held-out real-word, ambiguous-word, semantic-feature and controlled retrieval tasks with independent human labels.
- **Biological validity:** requires independently collected and analyzed neural measurements, task-matched predictions, held-out comparison and replication.

Passing Python tests does not establish behavioral accuracy or biological correspondence.
