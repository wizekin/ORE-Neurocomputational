# Point 11 — Fedorenko's Language Network vs. Other Brain Systems

## Scope

Point 11 is a cross-cutting functional-separation and cooperation algorithm, not another chronological stage of sentence comprehension. It distinguishes sensory-form encoding, the core language-selective network, conceptual/world knowledge, mentalizing, multiple-demand control, and memory while permitting task-dependent exchange.

The implementation follows Part 11 of the supplied *Language_Comprehension_Brain_Algorithms_Parts_01-15.md*. It reflects the framework that a left-dominant frontotemporal language network retrieves lexical/constructional knowledge and builds linguistic representations, while other systems contribute sensory, conceptual, social, executive, and memory information. The precise boundaries and degree of overlap remain empirical questions.

## The 16 operations

1. **Decompose task demands.** Estimate sensory, linguistic, conceptual, social, working-memory, executive, and response requirements separately.
2. **Encode sensory form.** Keep modality-specific acoustic/visual form and signal quality separate from meaning.
3. **Identify the individual's language-selective network.** Prefer supplied participant-specific localizer contrasts. Otherwise label a supplied literature prior as approximate; never fabricate an individual map.
4. **Route lexical and grammatical operations.** Preserve word/construction access and linguistic composition as a distinct language-system representation.
5. **Separate composition from conceptual enrichment.** Retrieve relevant world knowledge and retain whether each feature came from literal language or conceptual knowledge.
6. **Conditionally recruit mentalizing.** Add agent beliefs, knowledge, intentions, and perspectives only when the task or context requires them.
7. **Estimate executive demand independently.** Separate sentence complexity from memory load, conflict, time pressure, and response selection.
8. **Exchange information while preserving specialization.** Use a sparse directed functional-information graph with gated messages. These edges are not anatomical tracts.
9. **Evaluate task-appropriate selectivity.** Preserve supplied multi-task response profiles rather than declaring a system absolutely exclusive.
10. **Separate co-activation from contribution.** Keep association, lesion, and perturbation evidence as distinct evidence types.
11. **Preserve separate language and thought measures.** Linguistic ability, non-linguistic reasoning, social cognition, and executive performance are not collapsed into one score.
12. **Incorporate individual network organization.** Preserve laterality, stability, individual localization, and between-person variation.
13. **Evaluate right-hemisphere contributions.** Narrative, discourse, semantic-control, and prosodic demands may alter recruitment; task demand is not itself measured activation.
14. **Update allocation incrementally.** Re-estimate demands as new words and context arrive.
15. **Integrate with provenance.** Combine literal propositions and supported enrichments while retaining source and confidence.
16. **Reduce unneeded additional recruitment.** Avoid adding speculative detail when its marginal task utility is below its cost.

## Functional systems

| System | Computational responsibility | Important boundary |
|---|---|---|
| Perceptual/motor | Encode speech or visual text form and relevant signal quality | Sensory response is not automatically semantic computation |
| Core language-selective | Access words and constructions; build grammatical and linguistic representations | Does not need to solve every social, logical, or world-knowledge problem |
| Conceptual/world knowledge | Retrieve entities, events, properties, and relations | Semantic knowledge is distributed, not one isolated “meaning centre” |
| Mentalizing / ToM | Represent beliefs, knowledge, intentions, and perspectives when needed | Not required strongly for every literal sentence |
| Multiple-demand / executive | Support selection, conflict, memory, inhibition, and demanding response control | General task difficulty is not synonymous with linguistic complexity |
| Memory | Maintain active propositions and support episodic/contextual retrieval | Maintenance is not the same computation as linguistic composition |

## Mathematical approximations

The module uses a demand vector:
`d=[d_sens,d_ling,d_concept,d_social,d_WM,d_exec,d_resp]`.

Conditional recruitment is represented by a logistic gate:
`g=sigmoid(score-threshold)`. This is a computational abstraction, not proof that neurons literally calculate a logistic function.

Functional exchange is represented by a directed graph and a recurrent update:
`h_i(t+1)=tanh(drive_i + recurrent_gain * sum(gated_weighted_messages))`.

Each enrichment is tagged with `(value, source, confidence, time_index)`. These explicit provenance fields are a software design choice; they are not claimed to be a known biological data structure.

## Python API

```python
from ore_language_comprehension import (
    Point11Request,
    Point11FunctionalSeparationCoordinator,
)

request = Point11Request(
    sequence_id="example",
    sensory_input="Maya thinks the keys are in the drawer.",
    stage_outputs={
        "lexical": {"candidates": ["Maya", "thinks", "keys", "drawer"]},
        "syntactic": {"candidates": ["subject-verb-complement"]},
        "semantic": {"propositions": [{"agent": "Maya", "belief": "keys_in_drawer"}]},
    },
    task={"mentalizing_required": True, "working_memory_load": 0.4},
    context={"text": "The keys were actually moved to the table."},
    agent_candidates=[{"agent": "Maya", "belief": "keys_in_drawer", "confidence": 0.8}],
    individual_profile={
        "language_localizer": {"language": [2.1, 2.4], "control": [1.0, 1.1]},
    },
)
result = Point11FunctionalSeparationCoordinator().process(request)
print(result.interpretation)
print(result.provenance)
```

The optional `Point11LIFReference` is a simple six-population LIF-style reference simulation. It is not fitted human data, a complete cellular model, or evidence that this architecture reproduces the brain.

## Shared SQLite annotations

`register_point11_annotations()` indexes the Point 11 classes and functions in the existing repository-wide `ore_logger.annotations` SQLite table. No separate Point 11 database is created.

## Scientific safeguards

- Functional specialization does not mean physical isolation; systems can synchronize and exchange information.
- Co-activation does not prove identical function, necessity, or causal direction.
- A group atlas is not an exact personal language-network map.
- A task-demand score is not a measured neural activation.
- Language/thought dissociation does not imply complete independence; language can influence some forms of cognition.
- Right-hemisphere and multiple-demand contributions are task-sensitive; they are not categorically irrelevant.
- Default-mode and conceptual systems are heterogeneous; no single non-language network is asserted to contain all semantic knowledge.
- Mathematical equations and software routing are explanatory approximations, not literal neuronal instructions.

## Tests

`tests/test_language_comprehension_point11.py` covers the 16-step trace, conditional mentalizing/executive recruitment, individual localization, provenance, graph validation, selectivity profiles, evidence labels, language/thought dissociation, right-hemisphere demand, incremental updates, stopping, optional LIF simulation, and SQLite annotation registration.
