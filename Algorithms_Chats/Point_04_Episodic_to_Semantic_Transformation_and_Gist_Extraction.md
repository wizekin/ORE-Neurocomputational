# Point 04 — Episodic-to-Semantic Transformation and Gist Extraction

## Scope and position in the source sequence

This algorithm formalizes the next process described in *Learning, Memory, and Concept Formation*: as memories are reorganized over time, some event-specific information (especially where/when context and incidental details) may become less accessible, while regularities and meaning are retained as a more general semantic representation. The source then distinguishes vivid episodic recollection, which is strongly hippocampus-dependent in multiple-trace and trace-transformation accounts, from generalized semantic knowledge, which can increasingly rely on distributed cortical representations.

The target is **not** a literal file-transfer operation in which a complete memory is copied from the hippocampus into one cortical location. It is a distributed, gradual change in the form, accessibility, and neural support of remembered information. A single experience can contribute both to an event-specific representation and to more general knowledge. These forms can coexist and interact.

## 1. Biological objective

Given one or more encoded experiences, the brain must support two partially different outcomes:

1. **Episodic reinstatement:** recover a particular event with its distinctive combinations of people, objects, place, time, sensory details, actions, and subjective state.
2. **Semantic extraction:** retain what the experience teaches about objects, people, events, causal relations, and recurring situations, even when the original circumstances are no longer recalled precisely.

For example, a person may later remember the general fact that a certain kind of bird has feathers and sings without recalling the particular afternoon, location, weather, or individual bird through which that knowledge was learned. The semantic representation is useful beyond the original episode; the episodic representation preserves the specific occurrence.

The hippocampus and medial-temporal-lobe structures bind distributed features of events. Neocortical systems represent perceptual, conceptual, and semantic content across modality-specific and association networks. Prefrontal and parietal regions can contribute to integration, organization, retrieval goals, and abstraction. The exact division of labor is distributed and task-dependent; this algorithm therefore uses functional roles rather than claiming one brain region performs one isolated software-like step.

## 2. Important scientific qualification

The source's description that contextual details tend to fade while semantic gist remains is a useful high-level account, but it should not be interpreted as a universal rule that every old memory loses its episodic character or becomes hippocampus-independent.

- **Standard systems-consolidation theory** emphasizes a time-dependent strengthening of cortical representations and a reduced need for the hippocampus for some remote memories, especially semanticized knowledge.
- **Multiple-trace theory** proposes that detailed episodic recollection continues to depend on hippocampal traces, regardless of memory age, while semantic knowledge can become more independent.
- **Trace-transformation theory** proposes qualitative transformation: an event can yield a more generalized, gist-like representation while a detailed episodic form can remain available through hippocampal involvement.
- Related indexing and interactive-reorganization accounts emphasize continued interactions between hippocampal indices and distributed cortical representations.

The content and resolution of a memory matter, not age alone. A remote autobiographical memory recalled with vivid, event-specific detail may still recruit the hippocampus. Conversely, a remote memory described as a summary, fact, or generalized script may depend more strongly on cortical knowledge. Current research continues to refine the theories and their boundary conditions.

Accordingly, the algorithm below models **parallel representational change**: semantic knowledge becomes more stable and generalized, while detailed episodic access may persist, weaken, or require hippocampal reinstatement depending on the memory, the retrieval task, and the theory being modeled.

---

# 3. Algorithmic input and output

### Inputs

For each encoded event `E_i`:

- `content_i`: objects, people, actions, words, and event identity;
- `spatial_i`: location, spatial layout, and relations;
- `temporal_i`: order, duration, and temporal context;
- `sensory_i`: visual, auditory, tactile, olfactory, and other perceptual features;
- `affective_i`: emotional value, arousal, and motivational relevance;
- `causal_i`: perceived causes, goals, consequences, and relations;
- `source_i`: information about how the event was learned (direct experience, testimony, inference, etc.);
- `confidence_i`: confidence or strength associated with retrieval;
- `repetition_i`: number and spacing of learning or retrieval opportunities;
- `related_events`: prior and later experiences that may overlap with the event.

These are computational variables, not literal neural data structures. The brain processes distributed patterns of activity rather than named fields stored in a record.

### Outputs

- `episodic_representation`: an event-specific pattern retaining distinctive contextual bindings to the extent they remain accessible;
- `semantic_representation`: generalized knowledge about recurring features, relations, and meanings;
- `schema_links`: connections from the new knowledge to prior concepts and event structures;
- `uncertainty`: ambiguity, source uncertainty, and confidence estimates;
- `retrieval_policy`: whether a cue should preferentially reconstruct a particular episode, retrieve a general fact, or combine both.

---

# 4. Detailed algorithmic steps

## Step 1 — Reactivate an encoded event or a set of related events

**Biological interpretation**

A memory may be reactivated during spontaneous replay, deliberate remembering, sleep-associated processing, or exposure to a related cue. Reactivation can reinstate parts of the distributed pattern established during the original experience. Hippocampal pattern completion can help recover an event from a partial cue, while cortical regions reactivate relevant sensory, conceptual, and associative information.

Reactivation is not guaranteed to reproduce the original event perfectly. It is a reconstruction influenced by the cue, current context, later knowledge, and the strength of the stored associations.

**Computational operation**

1. Receive a cue or internally generated replay trigger `q`.
2. Compare it with stored event indices and distributed cortical feature patterns.
3. Select one or more candidate events with sufficient associative match.
4. Reconstruct a partial feature pattern for each candidate.
5. Preserve uncertainty for missing or conflicting features rather than filling every gap as fact.

One abstract cue-match operation is:

\[
s_i = \operatorname{sim}(q, k_i)
\]

where `k_i` is a stored event key or distributed index and `sim` is a similarity function. Candidate probabilities can be represented as:

\[
P(E_i\mid q)=\frac{\exp(\beta s_i)}{\sum_j\exp(\beta s_j)}
\]

Here `β` controls how strongly differences in match scores affect selection. This softmax is a modeling device; the brain is not known to calculate this exact formula.

**Output:** a reactivated, incomplete event representation with confidence estimates.

## Step 2 — Reconstruct the event's feature components

**Biological interpretation**

The event is distributed across interacting neural systems. Perceptual regions represent sensory properties; medial-temporal regions contribute object and contextual representations; the hippocampus binds elements into a relational event pattern. Emotional significance can influence which information is reinstated and how strongly it is remembered.

**Computational operation**

Partition the reconstructed event into overlapping feature groups:

\[
\mathbf{x}_i=[\mathbf{x}^{content}_i,\mathbf{x}^{space}_i,
\mathbf{x}^{time}_i,\mathbf{x}^{sensory}_i,
\mathbf{x}^{affect}_i,\mathbf{x}^{relation}_i]
\]

For each feature `f`, estimate its activation and confidence:

\[
\hat{x}_{if}=a_{if}\,c_{if}
\]

where `a` is activation strength and `c` is confidence. The product is a convenient model of confidence-weighted evidence, not a known synaptic calculation.

Separate features that identify the particular occurrence (e.g., the exact room, date, or incidental sound) from features that may be meaningful across many occurrences (e.g., the object category or causal relation).

**Output:** a structured feature representation with uncertainty attached to individual details.

## Step 3 — Preserve event-specific bindings before abstraction

**Biological interpretation**

The hippocampal formation is important for binding distinct elements into a relational representation: which person interacted with which object, where an action occurred, and in what sequence. If abstraction begins without preserving the event's distinctive relations, a model can confuse details from different experiences.

**Computational operation**

Represent relations among features as a graph or tensor. An illustrative event graph is:

\[
G_i=(V_i,R_i)
\]

where `V_i` contains event elements and `R_i` contains relations such as `agent-of`, `located-at`, `before`, `causes`, or `interacts-with`.

For example:

\[
\text{person A}\xrightarrow{\text{feeds}}\text{bird B},
\qquad \text{bird B}\xrightarrow{\text{located-at}}\text{garden C}
\]

Retain a record of which features belong together in the same event. A binding-consistency score can be modeled as:

\[
B_i=\frac{\text{supported within-event relations}}
{\text{candidate within-event relations}+\epsilon}
\]

This protects against merging an event's features with similar features from another event.

**Output:** an event-specific relational pattern suitable for later comparison with other events.

## Step 4 — Retrieve and align related experiences

**Biological interpretation**

General knowledge can emerge when multiple experiences activate overlapping representations. Related events may be brought together through hippocampal-cortical interactions and through cortical semantic networks. Overlap allows the system to detect recurring properties and relations.

**Computational operation**

1. Search memory for events sharing one or more features or relations with the reactivated event.
2. Estimate similarity using content, context, and relational structure.
3. Align corresponding elements across events even when their surface details differ.
4. Keep nonmatching features as event-specific rather than forcing them into a common pattern.

A weighted similarity model is:

\[
S(E_i,E_j)=\sum_f w_f\,\operatorname{sim}(x_{if},x_{jf})
+\lambda\,\operatorname{sim}(R_i,R_j)
\]

where `w_f` weights feature dimensions and `λ` weights relational overlap. Weights depend on task and goals; no single universal weighting is implied.

**Output:** a set of aligned but distinguishable episodes.

## Step 5 — Estimate which features are invariant and which are incidental

**Biological interpretation**

Semanticization requires distinguishing what tends to remain true across experiences from what was unique to one occasion. Repeated co-occurrence can strengthen an association, but frequency alone is not sufficient: a feature can be frequent yet irrelevant, or rare yet highly diagnostic.

**Computational operation**

For each candidate feature `f`, estimate:

- recurrence across related episodes;
- consistency of its relation to other features;
- diagnostic value for the category or event type;
- dependence on a specific context;
- reliability and source quality.

A simple recurrence estimate is:

\[
P(f\mid C)=\frac{\sum_i r_i\,\mathbb{1}(f\in E_i)}
{\sum_i r_i}
\]

where `r_i` is an episode reliability weight and `C` is the candidate class or event family. A feature's discriminative value can be represented by information gain:

\[
IG(f)=H(C)-H(C\mid f)
\]

where `H` is entropy. A feature that appears in nearly every category may have high recurrence but low discriminative value. A feature's abstraction score can combine recurrence, discriminative value, relational stability, and reliability:

\[
A_f=\alpha P(f\mid C)+\gamma IG(f)+\delta R_f+\eta Q_f
\]

The coefficients are modeling parameters. The brain is not assumed to calculate this exact linear score.

**Output:** likely invariant features, distinctive features, and uncertain candidates.

## Step 6 — Extract recurring relations, not just recurring objects

**Biological interpretation**

General knowledge often concerns relationships and event structure, not merely a list of frequently seen objects. Across different episodes, the same causal or functional relation may recur despite changes in the people, objects, or setting. Relational representations are important for flexible transfer and later inference.

**Computational operation**

1. Compare event graphs from Step 3.
2. Identify relations that recur across different feature identities.
3. Determine whether the relation is supported by multiple observations or only inferred from one event.
4. Encode the recurring relation with variable slots.

For a relation `r`, estimate:

\[
P(r\mid C)=\frac{\text{reliable episodes containing }r}
{\text{reliable episodes considered}}
\]

A relation's confidence should also depend on the number and diversity of observations. A simple reliability-adjusted count is:

\[
N_{eff}=\frac{(\sum_i r_i)^2}{\sum_i r_i^2+\epsilon}
\]

This effective-sample-size expression is a statistical modeling aid, not a known neural operation.

**Output:** candidate abstract relations that are less tied to the surface details of any single event.

## Step 7 — Generate a semantic or gist-level representation

**Biological interpretation**

The brain may retain the central meaning of an event or recurring event type while the original episode's incidental context becomes less accessible. This is not necessarily a degraded copy. A semantic representation can be a qualitatively different, more abstract representation that supports generalization.

**Computational operation**

Construct a representation from features and relations that have sufficient support across related episodes. Exclude or down-weight details that are unique to one occurrence unless they are essential to the learned meaning.

One abstract weighted representation is:

\[
\mathbf{g}_C=\frac{\sum_i r_i\,\phi(E_i)}
{\sum_i r_i+\epsilon}
\]

where `φ(E_i)` maps an event into a feature-relation representation and `g_C` is the candidate gist for class `C`.

A prototype-style implementation can use a weighted centroid. An exemplar-style implementation can preserve individual episodes and compare new events against them. The biological system may use mixtures of prototype-like and exemplar-like information; this algorithm should not force a single theory.

**Output:** a compact, generalized representation with links to the episodes that support it.

## Step 8 — Reduce dependence on exact spatial and temporal context when context is not essential

**Biological interpretation**

The source identifies where/when information as particularly vulnerable during the development of semantic gist. In generalized knowledge, a person may retain what happened or what was learned without remembering the precise date, location, or incidental sequence. However, context can remain important for autobiographical memory, source monitoring, causal understanding, and situations where context changes meaning.

**Computational operation**

Estimate the relevance of contextual features to the target knowledge. Do not discard context indiscriminately. Instead, retain it according to its diagnostic and explanatory value.

For a contextual feature `c`, a retention weight may be modeled as:

\[
w_c=\sigma(\theta_0+\theta_1D_c+\theta_2U_c+\theta_3M_c)
\]

where `D_c` is diagnostic value, `U_c` is utility for future retrieval, `M_c` is relevance to meaning, and `σ` is the logistic function. Context can be down-weighted in the semantic representation when it is incidental, while remaining more prominent in the episodic representation.

**Output:** a semantic representation with context selectively abstracted, not indiscriminately erased.

## Step 9 — Maintain differentiated episodic and semantic forms

**Biological interpretation**

The same learning history can support an event-specific memory and a generalized fact. In multiple-trace and trace-transformation accounts, a vivid recollection of a particular event continues to depend on hippocampal involvement, whereas generalized semantic knowledge can increasingly rely on neocortical representations. The exact relationship remains debated.

**Computational operation**

Maintain two linked representational products:

- `EpisodicStore`: high-resolution details and event bindings;
- `SemanticStore`: abstract features, relations, and category-level knowledge.

They share references but have different retrieval criteria. The episodic representation is selected when the query asks which particular event occurred; the semantic representation is selected when the query asks what is generally true.

A dual-store objective can be expressed as:

\[
\mathcal{L}=\lambda_e\mathcal{L}_{episode}
+\lambda_s\mathcal{L}_{semantic}
+\lambda_c\mathcal{L}_{consistency}
\]

The first term rewards accurate event-specific reconstruction; the second rewards generalization across examples; the third penalizes contradictions between a general claim and its supporting episodes. This is a proposed computational architecture, not a claim that the brain optimizes an explicit scalar loss.

**Output:** linked episodic and semantic representations rather than one representation overwriting the other.

## Step 10 — Integrate the gist into distributed cortical knowledge

**Biological interpretation**

Semantic knowledge is distributed across cortical systems. Sensory and motor features can remain represented in modality-related cortices; association regions help combine information across modalities. The anterior temporal lobe is often discussed as a semantic integration hub, while prefrontal and parietal regions can contribute to controlled retrieval, relational integration, and schema-level organization. This is a network-level process, not a single storage location.

**Computational operation**

1. Map the new gist to existing semantic representations.
2. Identify compatible categories, concepts, and relations.
3. Strengthen supported links while preserving distinctions between similar concepts.
4. Place the gist into distributed networks associated with its properties and use.
5. Register contradictory evidence rather than silently replacing prior knowledge.

A simple associative update is:

\[
W_{new}=W_{old}+\eta\,\Delta W
\]

A Hebbian-style illustrative rule is:

\[
\Delta w_{ab}=\eta\,x_a x_b
\]

where `x_a` and `x_b` denote co-activity and `η` is a learning-rate parameter. Biological synaptic plasticity involves many additional factors, including spike timing, neuromodulators, inhibition, dendritic mechanisms, and homeostatic regulation. These equations are abstractions.

**Output:** a semantic pattern integrated into pre-existing cortical knowledge networks.

## Step 11 — Update existing knowledge without erasing useful distinctions

**Biological interpretation**

New knowledge can modify prior representations. Repeated integration may strengthen common structure, refine categories, and alter the way a past event is interpreted. However, overgeneralization can create false memories or erase distinctions between events. The brain must balance integration with separation.

**Computational operation**

1. Compare the candidate gist with existing concepts and schemas.
2. Merge overlapping features only when evidence supports shared meaning.
3. Preserve unique properties as subcategory or instance attributes.
4. Mark contradictions and uncertainty for later resolution.
5. Use novelty or mismatch signals to prevent inappropriate merging.

One update rule is:

\[
\mathbf{z}_{new}=(1-\rho)\mathbf{z}_{old}+\rho\mathbf{g}_C
\]

where `ρ` is a context-dependent update rate. A separate prediction-error term can be used:

\[
\delta = y-\hat{y}
\]

where `y` is the observed outcome and `ŷ` is the predicted outcome. A large mismatch can trigger revision or retention of a separate representation. This is a generic learning model; the nervous system's error signals are task- and circuit-dependent.

**Output:** an updated semantic network with preserved exceptions and boundaries.

## Step 12 — Reweight or weaken access to incidental episode details over time

**Biological interpretation**

The source states that episodic context tends to fade as semantic gist remains. Several mechanisms may contribute: changes in synaptic strength, interference from similar experiences, reduced accessibility, retrieval competition, and transformation into a more general representation. It is not scientifically justified to assume that every forgotten detail has been physically deleted. Some details may remain stored but become difficult to retrieve.

**Computational operation**

Represent accessibility separately from whether a trace exists. For feature `f` in an event representation:

\[
A_f(t)=A_f(0)e^{-\lambda_f t}
\]

This exponential is a simplified forgetting/accessibility curve, not a universal biological law. Retrieval, rehearsal, emotional relevance, sleep, and later learning can change the effective rate.

An interference term can be added:

\[
A_f(t)=A_f(0)e^{-\lambda_f t}\,e^{-\kappa I_f(t)}
\]

where `I_f(t)` measures competition from similar traces. Highly diagnostic or repeatedly reactivated details may have smaller effective decay or interference. Contextual details that do not contribute to the gist may become less accessible than stable semantic features.

**Output:** a differentiated accessibility profile; the algorithm does not equate low accessibility with guaranteed erasure.

## Step 13 — Preserve hippocampal support when a query demands a specific episode

**Biological interpretation**

A request to recall a specific personal event requires more than knowing the general facts about that type of event. The hippocampus contributes to reconstructing relationally bound, event-specific details. Under multiple-trace and trace-transformation theories, vivid episodic recollection remains hippocampus-dependent even for remote events. Other models disagree about the duration and conditions of hippocampal dependence.

**Computational operation**

Classify the retrieval request:

- `EPISODIC`: asks what happened on a particular occasion, where, when, or with whom;
- `SEMANTIC`: asks what is generally true or what a category means;
- `MIXED`: asks for a general conclusion and its supporting experiences.

For episodic queries, search event indices and reconstruct the specific episode. For semantic queries, retrieve the generalized cortical representation. For mixed queries, retrieve the gist and its supporting episode links.

**Output:** retrieval matched to the representational level required by the task.

## Step 14 — Reconstruct the memory while separating remembered details from inferred details

**Biological interpretation**

Remembering is reconstructive. A semantic schema can help fill gaps, but it can also introduce details that were not part of the original event. This can produce source-monitoring errors or false recollections. The generalization process must therefore not be mistaken for perfect recovery of the past.

**Computational operation**

When reconstructing an episode, assign each retrieved feature a provenance label:

- directly reinstated event detail;
- supported by a linked record or source;
- inferred from semantic knowledge;
- uncertain or conflicting.

A confidence estimate can combine evidence strength and source reliability:

\[
C_f=P(f\mid q,\text{trace},\text{source})
\]

Do not promote a high-probability schema-consistent inference into a factual episodic detail without event-specific evidence. This distinction is essential for avoiding confabulation in a computational implementation.

**Output:** a reconstructed response with provenance and uncertainty, preserving the difference between gist and exact episodic detail.

## Step 15 — Use retrieval and new experience to refine both representations

**Biological interpretation**

Reactivated memories can be modified, strengthened, or updated; new related experiences can also change the semantic representation. Memory is therefore dynamic rather than a one-time transfer. Reconsolidation and retrieval-induced updating are relevant frameworks, although the conditions that produce modification vary across tasks and memory types.

**Computational operation**

1. Compare retrieved knowledge with current evidence.
2. Update the semantic pattern when reliable evidence supports a change.
3. Re-encode distinctive new event details into the episodic representation.
4. Preserve provenance and time-stamp the update.
5. Avoid changing the episodic record merely because a later schema suggests a more typical event.

A confidence-weighted update can be expressed as:

\[
\mathbf{g}_{t+1}=\mathbf{g}_t+\eta c\left(\mathbf{e}_{new}-\mathbf{g}_t\right)
\]

where `c` is evidence confidence and `η` is an update rate. This is a general adaptive-filter form, not an identified neural equation.

**Output:** progressively refined knowledge that can incorporate new experience without automatically rewriting the original event record.

## Step 16 — Evaluate the quality of the transformation

**Biological interpretation**

Successful semanticization should improve useful generalization without destroying the ability to distinguish events when exact episodic details matter. A person should be able to apply a learned regularity to a new situation, but should not falsely claim that a detail from one example occurred in another.

**Computational operation**

Evaluate at least four dimensions:

1. **Semantic generalization:** can the representation support correct judgments about novel examples?
2. **Episodic specificity:** can the system distinguish the original episode from similar events?
3. **Source accuracy:** can the system identify whether a detail was experienced, heard, inferred, or generalized?
4. **Calibration:** do confidence estimates track actual accuracy?

For semantic generalization, measure classification or prediction accuracy on held-out examples. For episodic discrimination, measure whether the correct event is selected among similar lures. For calibration, use a Brier score:

\[
BS=\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2
\]

where `p_i` is predicted confidence and `y_i` is outcome (0 or 1). These are evaluation metrics for the computational model, not brain operations.

**Output:** quality scores and signals indicating whether the representation should be refined, separated, or retained.

---

# 5. Integrated mathematical model

The complete process can be represented as a joint model with an event-specific component and an abstract component.

Let `E_i` denote an experience, `H_i` its hippocampal-associated event representation, and `C` the set of related experiences. Let `G_C` be the gist or semantic representation derived from the set.

### Event representation

\[
H_i=F_{bind}(content_i,space_i,time_i,sensory_i,affect_i,relations_i)
\]

`F_bind` stands for distributed relational binding and is an abstract function, not a literal hippocampal software module.

### Candidate semantic representation

\[
G_C=F_{abstract}(H_1,H_2,\ldots,H_n;\,reliability,\,overlap,\,relations)
\]

`F_abstract` extracts stable features and relations across events while retaining exceptions and uncertainty.

### Parallel support

\[
Memory(C)=\{H_1,H_2,\ldots,H_n,\,G_C,\,Links(H_i,G_C)\}
\]

This representation allows the system to retrieve an individual episode, retrieve general knowledge, or retrieve both together.

### Query-dependent retrieval

\[
R(q)=
\begin{cases}
R_{episodic}(H_i), & q\text{ requires a particular event}\n\\
R_{semantic}(G_C), & q\text{ requires general knowledge}\n\\
R_{combined}(H_i,G_C), & q\text{ requires explanation and supporting examples}
\end{cases}
\]

### Overall objective

\[
\mathcal{J}=\lambda_1 L_{event}+\lambda_2 L_{generalization}
+\lambda_3 L_{source}+\lambda_4 L_{calibration}
+\lambda_5 L_{interference}
\]

The objective summarizes computational design goals. It should not be mistaken for a claim that the brain explicitly computes a single global loss function.

---

# 6. Integrated pseudocode

```text
FUNCTION Transform_Experience_Set_Into_Episodic_And_Semantic_Memory(
    event_set,
    retrieval_cue,
    existing_knowledge,
    task_goal
):

    reactivated_events = Reactivate_Matching_Events(
        event_set,
        retrieval_cue
    )

    event_representations = []

    FOR event IN reactivated_events:
        features = Reconstruct_Feature_Groups(event)
        confidence = Estimate_Feature_Confidence(features)
        relations = Bind_Event_Relations(features, confidence)

        event_representation = Preserve_Event_Specific_Pattern(
            features,
            relations,
            confidence
        )

        event_representations.APPEND(event_representation)

    related_events = Retrieve_And_Align_Related_Experiences(
        event_representations,
        existing_knowledge
    )

    aligned_features = Align_Common_Elements(related_events)
    invariant_features = Estimate_Invariant_Features(aligned_features)
    recurring_relations = Extract_Recurring_Relations(related_events)

    semantic_gist = Build_Generalized_Representation(
        invariant_features,
        recurring_relations,
        evidence_reliability=related_events.reliability
    )

    contextual_weights = Estimate_Context_Relevance(
        related_events,
        task_goal
    )

    semantic_gist = Abstract_Incidental_Context(
        semantic_gist,
        contextual_weights
    )

    episodic_store = Preserve_Event_Specific_Representations(
        event_representations
    )

    semantic_store = Integrate_Into_Cortical_Knowledge_Model(
        semantic_gist,
        existing_knowledge
    )

    semantic_store = Resolve_Overlap_And_Preserve_Exceptions(
        semantic_store,
        event_representations
    )

    Update_Accessibility_Without_Assuming_Trace_Erasure(
        episodic_store,
        semantic_store
    )

    retrieval_type = Classify_Retrieval_Goal(task_goal)

    IF retrieval_type == "EPISODIC":
        result = Reconstruct_Specific_Event(episodic_store)
    ELSE IF retrieval_type == "SEMANTIC":
        result = Retrieve_General_Knowledge(semantic_store)
    ELSE:
        result = Combine_Gist_With_Supporting_Events(
            semantic_store,
            episodic_store
        )

    result = Attach_Source_And_Uncertainty_Labels(result)
    quality = Evaluate_Generalization_Specificity_And_Calibration(result)

    IF quality indicates unsupported merging or poor calibration:
        Refine_Representations_And_Preserve_Conflicts()

    RETURN {
        "episodic_memory": episodic_store,
        "semantic_memory": semantic_store,
        "retrieved_result": result,
        "quality": quality
    }
```

---

# 7. Biological mechanisms and computational analogies

| Biological process | Computational analogy | Important limitation |
|---|---|---|
| Hippocampal pattern completion | Retrieve an event from a partial cue | Neural completion is distributed and not necessarily a deterministic lookup |
| Relational binding | Construct a graph of event elements and relations | Neural bindings are population and circuit dynamics, not explicit graph objects |
| Reactivation and replay | Reprocess stored event patterns | Replay can be partial, compressed, or reordered; it is not necessarily a faithful movie |
| Cross-event overlap | Align related examples | Similarity is task-dependent and can create false merges |
| Semantic abstraction | Extract stable features and relations | No single known arithmetic formula explains human abstraction |
| Cortical integration | Update distributed associative representations | Semantic knowledge is not stored in one isolated cortical location |
| Reduced contextual accessibility | Lower the retrieval weight of incidental details | Inaccessibility does not prove that a trace has been erased |
| Episodic retrieval | Reconstruct a particular event | Retrieval is reconstructive and can include errors |

# 8. Testable predictions for a computational implementation

A model implementing this algorithm should satisfy the following tests:

1. **Repeated-example generalization:** after exposure to several distinct examples of a category, it should identify common features in a novel example without requiring that novel example to be identical to a stored instance.
2. **Episode preservation:** it should still distinguish two similar events that occurred at different times or places when asked about a specific occurrence.
3. **Context-sensitive abstraction:** it should retain contextual information when context changes the meaning or causal interpretation, and down-weight it only when it is genuinely incidental to the target generalization.
4. **Gist-versus-detail distinction:** it should answer a general question from semantic memory without fabricating exact episodic details.
5. **Source monitoring:** it should distinguish information directly experienced from information inferred, learned from another person, or generalized across examples.
6. **Interference robustness:** adding similar experiences should improve general knowledge while not causing all episodes to collapse into one indistinguishable record.
7. **Retrieval-dependent update:** new reliable evidence should refine semantic knowledge, while an unsupported inference should not rewrite an event-specific record.
8. **Confidence calibration:** uncertain or reconstructed details should not be reported with the same confidence as strongly supported event details.

# 9. Experimental concepts that can inform validation

- Compare recall of an event's central meaning with recall of its perceptual and spatial-temporal details after different retention intervals.
- Contrast autobiographical recollection that includes vivid event-specific context with factual or gist-level recall.
- Examine whether retrieval cues that demand exact context recruit hippocampal-associated processes differently from cues that request generalized knowledge.
- Use similarity and lure tests to determine whether semantic abstraction causes beneficial generalization or excessive false recognition.
- Test whether reactivation improves access to event-specific information, changes later semanticization, or both.

These are broad experimental paradigms, not a claim that one measurement uniquely identifies a memory theory. Neural activity, lesion evidence, behavior, and task demands must be interpreted together.

# 10. Scientific interpretation and limitations

This algorithm is a biologically informed computational model of one account of memory transformation. The mathematical equations are tools for making assumptions explicit and for implementing a testable system. They are not assertions that neurons literally calculate entropy, softmax, cosine similarity, or a global loss function.

Three distinctions must remain explicit:

1. **Trace existence versus retrieval accessibility:** failure to recall a detail does not establish that its neural trace has been erased.
2. **Semantic gist versus degraded perception:** a gist can be an abstract representation of meaning and relations, not merely a blurry sensory copy.
3. **Remote age versus representational type:** memory age alone does not determine whether the hippocampus is required; vivid event-specific recollection and generalized knowledge can have different neural dependencies.

The algorithm is therefore deliberately dual-track. It allows semantic knowledge to become increasingly abstract and distributed while retaining the possibility that a detailed recollection requires hippocampal participation. It should be revised as theories and empirical findings develop.

---

# 11. Selected scientific references

1. **Time-dependent memory transformation in hippocampus and neocortex is semantic in nature.** *Nature Communications* (2023). Reports semantic changes in memory representations over time and discusses the distinction between semantic transformation and simple loss of perceptual detail. https://www.nature.com/articles/s41467-023-41648-1
2. **Systems Consolidation, Transformation, and Reorganization: Multiple Trace Theory, Trace Transformation Theory, and Their Competitors.** Oxford University Press (2024). Reviews competing theories and argues for attention to interactive reorganization across the memory lifetime. https://academic.oup.com/edited-volume/57928/chapter-abstract/475478430
3. **Details, gist and schema: hippocampal–neocortical interactions underlying recent and remote episodic and spatial memory.** *Current Opinion in Behavioral Sciences* (2017). Reviews how memory detail, gist, and schema relate to hippocampal and cortical systems. https://doi.org/10.1016/j.cobeha.2017.07.016
4. **Functional neuroanatomy of remote episodic, semantic and spatial memory: a unified account based on multiple trace theory.** *Journal of Anatomy* (2005). Discusses distinctions between detailed episodic and semantic memory and their neural dependencies. https://pmc.ncbi.nlm.nih.gov/articles/PMC1571502/
5. **The Cognitive Neuroscience of Memory Representations.** *Neuroscience & Biobehavioral Reviews* (2025). Reviews memory representations, systems consolidation, trace transformation, and the status of hippocampal engrams. https://doi.org/10.1016/j.neubiorev.2025.106417
6. **A generative model of memory construction and consolidation.** *Nature Human Behaviour* (2023). Develops a generative account linking consolidation, semantic knowledge, and episodic reconstruction. https://doi.org/10.1038/s41562-023-01799-z
7. **Role of the hippocampus in systems consolidation of remote fear memory.** *Experimental & Molecular Medicine* (2026). Reviews evidence for continuing hippocampal involvement in remote memory and the interactions between hippocampal and cortical systems. https://www.nature.com/articles/s12276-026-01680-9

---

**End of Point 04.** The next point in the source sequence concerns how novelty after learning can preserve or amplify hippocampal traces, in contrast with sleep-associated cortical integration.
