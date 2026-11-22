# Point 11 — Analogical Transfer: Applying Relational Knowledge Across Domains

## Scope and place in the sequence

This point develops the next mechanism in *Learning, Memory, and Concept Formation*: analogical transfer. Analogical transfer occurs when a learner uses a relational pattern understood in one domain (the **source**) to interpret, explain, predict, or solve a problem in another domain (the **target**). A familiar example is using water flowing through pipes as an analogy for electric current in a circuit.

The essential operation is not simply matching objects that look alike. It is identifying correspondences between the **relations** that connect elements in two situations. A pipe is not literally a wire, water is not literally electric charge, and water pressure is not identical to voltage. The potentially useful correspondence is structural: a difference in a driving quantity can produce a flow through a pathway, while resistance in the pathway constrains that flow. The analogy is useful only within the limits of that shared structure.

The sequence below is a biologically informed computational model. It is intended to make the processes explicit enough to implement and test. It does **not** claim that neurons execute the equations exactly as written, or that neuroscience has established a single serial algorithm for analogy. Neural processing is distributed, recurrent, parallel, and dependent on task, knowledge, development, and context.

## 1. Problem definition

**Goal:** Given a familiar source situation and a less familiar target situation, find a defensible mapping between their entities and relations; transfer only those inferences supported by the mapping; adapt them to the target; and test whether they work.

**Inputs**

- `source_case`: a previously learned event, explanation, story, diagram, or problem with known components and relations.
- `target_case`: a new situation or question that may share relational structure with the source.
- `goal`: the required inference (explain, predict, classify, plan, or solve).
- `context`: current task, prior knowledge, relevant constraints, and available evidence.
- `memory`: episodic cases, semantic concepts, schemas, and previously successful or unsuccessful analogies.

**Outputs**

- A ranked set of source–target correspondences.
- An explicit relational mapping or analogy schema.
- Candidate transferred inferences, each with confidence and scope conditions.
- A target-specific explanation, prediction, or proposed solution.
- A record of feedback that can revise the analogy and update memory.

## 2. Biological framing

Analogy relies on several interacting capacities rather than a single “analogy center.” Research reviews and imaging meta-analyses commonly implicate a distributed frontoparietal system. The left rostrolateral prefrontal cortex is repeatedly associated with integrating relations into higher-order correspondences. Dorsolateral prefrontal regions contribute to maintaining relevant relations and goals in working memory; ventrolateral prefrontal regions can help control interference from tempting but irrelevant features; and posterior parietal regions contribute to representing individual relations. Domain-specific temporal, occipital, and other posterior regions contribute the content being compared. The hippocampus can bind disparate elements and relations into relational memories and support retrieval or recombination of structured experiences. These are functional associations, not one-to-one assignments of a whole algorithm to a single anatomical region.

The source text mentions medial prefrontal and temporal-lobe networks in relational reasoning and the hippocampus in binding disparate features. This remains a useful broad framing, but the detailed literature distinguishes medial prefrontal cortex from rostrolateral and lateral prefrontal contributions. Which subregions are engaged depends on the analogy type, materials, difficulty, and comparison task.

## 3. Core representations

Represent a situation as a typed relational graph:

\[
G=(V,E,A,C)
\]

where:

- \(V\) is a set of entities or events;
- \(E\) is a set of directed or undirected relations between entities;
- \(A\) stores attributes of entities and relations;
- \(C\) stores contextual constraints, causal assumptions, and confidence values.

A relation is represented as:

\[
r=(\text{name},\text{arguments},\text{direction},\text{conditions},\text{confidence})
\]

For example, `drives(flow, pressure_difference)` and `constrains(flow, resistance)` express relationships. A relation may be first-order (between two elements) or higher-order (between relations or whole relational structures).

A mapping \(M\) is a partial function from source entities and relations to target entities and relations. It is partial because not every source component has a target counterpart. It must preserve role and relational structure sufficiently well to support a valid inference.

## 4. Algorithmic sequence

### Step 1 — Detect the need for transfer

**Purpose:** Determine whether the target problem can benefit from an already learned relational pattern.

1. Parse the current task and determine what must be explained or predicted.
2. Identify what is unknown, ambiguous, or computationally difficult in the target.
3. Search for cues that the target has a structure already encountered in prior experience (for example, flow, containment, competition, hierarchy, feedback, exchange, or proportional change).
4. Keep analogy as a candidate strategy rather than assuming every unfamiliar problem should be forced into a familiar template.

**Possible mathematical operations**

- Goal and cue activation can be represented as a weighted feature vector \(q\).
- A relevance score for a remembered schema \(s\) can be computed as:

\[
R(s\mid q)=\sum_i w_i q_i s_i
\]

- A softmax can rank candidate schemas without making selection deterministic:

\[
P(s\mid q)=\frac{e^{\beta R(s\mid q)}}{\sum_j e^{\beta R(s_j\mid q)}}
\]

Here \(w_i\) represents cue importance and \(\beta\) controls how sharply activation favours the highest-scoring candidate. These equations are modeling devices for cue-driven accessibility, not claims about a literal neural softmax.

**Biological interpretation:** Goal maintenance and controlled search recruit prefrontal systems; prior knowledge and domain cues activate distributed semantic and episodic representations. Hippocampal pattern completion may help retrieve a related episode from partial cues.

**Output:** A shortlist of candidate source cases or schemas.

### Step 2 — Retrieve a useful source case

**Purpose:** Retrieve a source that contains a potentially transferable relational pattern.

1. Search episodic memory for concrete prior situations and semantic memory for generalized schemas.
2. Retrieve more than one candidate when possible, especially when one highly familiar example may bias interpretation.
3. Prefer candidates that match the target's relevant goal and relational structure, not just its vocabulary or appearance.
4. Retrieve the source's known outcome and any prior warnings about where the analogy failed.

**Possible mathematical operations**

- Combine cue overlap, structural relevance, recency, familiarity, and prior transfer success:

\[
S_{retrieve}(s)=\alpha S_{cue}+\gamma S_{goal}+\delta S_{structure}+\eta S_{success}-\lambda S_{irrelevant}
\]

- A memory accessibility term may decay with elapsed time:

\[
A_s(t)=A_{s,0}e^{-kt}
\]

This simple decay is only an approximation; retrieval practice, emotional salience, sleep, and context can change accessibility.

**Biological interpretation:** The hippocampus is relevant to retrieval of relationally bound episodes. Cortical semantic systems contribute familiar concepts and schemas. Retrieval is reconstructive: the recalled source may be incomplete or influenced by current expectations.

**Output:** One or more candidate source graphs with retrieval confidence.

### Step 3 — Encode the target situation

**Purpose:** Build a sufficiently structured representation of the unfamiliar target before transferring anything.

1. Identify target entities, events, quantities, agents, and relevant environmental conditions.
2. Extract attributes that could matter to the current question.
3. Record observed relations separately from assumptions or inferred relations.
4. Mark unknown values and uncertainty rather than filling every gap immediately.
5. Represent time order and direction where they change the meaning (for example, `causes(A,B)` differs from `causes(B,A)`).

**Possible mathematical operations**

- Encode an entity as a feature vector \(x_i\in\mathbb{R}^d\), with a mask for missing features.
- Normalize numerical features only when scale differences would otherwise dominate:

\[
z_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j+\epsilon}
\]

- Attach an uncertainty estimate \(u_i\) or probability distribution to uncertain observations.

**Biological interpretation:** Perceptual and domain-specific cortical systems represent sensory or learned content. Working-memory systems maintain task-relevant elements. Hippocampal binding can link elements that occur together in a particular event. The exact representational format is distributed and varies with the material.

**Output:** A target relational graph \(G_T\), with observations distinguished from hypotheses.

### Step 4 — Reconstruct the source structure

**Purpose:** Represent the remembered source in terms that can be compared with the target.

1. Identify source entities and the roles they played.
2. Extract the relations that were important to the source's outcome.
3. Separate central causal or functional relations from incidental details.
4. Recover the source's boundary conditions: what made its relationships hold?
5. Mark relations that are known from experience separately from relations inferred during recollection.

**Possible mathematical operations**

- Construct a relation adjacency tensor or matrix. For a simple directed graph, \(A_{ij}^{(k)}\) can encode whether relation type \(k\) connects entity \(i\) to entity \(j\).
- Assign each relation a salience or reliability weight \(w_r\).
- Use a threshold only as a practical simplification:

\[
E_{core}=\{r\in E: w_r\geq\tau\}
\]

- Do not discard low-frequency relations automatically: a rare relation may be the key causal link.

**Biological interpretation:** Episodic recollection can reassemble features from distributed cortical traces through hippocampal–cortical interactions. Semantic knowledge supplies familiar role and relation meanings. Reconstruction may be imperfect, so the algorithm retains uncertainty.

**Output:** A source relational graph \(G_S\).

### Step 5 — Separate relational similarity from surface similarity

**Purpose:** Prevent superficial resemblance from dominating the analogy.

1. Compare entities' surface properties (appearance, names, material, location, and other attributes).
2. Independently compare what the entities do and how they relate to other entities.
3. Identify cases where the entities differ but the relational pattern is similar.
4. Identify cases where entities look similar but play different roles.
5. Give priority to relations relevant to the goal, while retaining surface properties when they impose genuine target constraints.

**Possible mathematical operations**

\[
Sim_{total}=\rho Sim_{surface}+(1-\rho)Sim_{relational}
\]

The coefficient \(\rho\) should be task-dependent. For a structural analogy, use a smaller \(\rho\); for a perceptual classification task, surface similarity may deserve greater weight.

A contrastive score can penalize a candidate that matches many irrelevant attributes but few important relations:

\[
S_{contrast}=S_{relevant\ relations}-\kappa S_{misleading\ surface\ matches}
\]

**Biological interpretation:** Prefrontal control can help suppress salient but irrelevant matches and maintain task-relevant relations. Posterior perceptual and semantic systems continue to provide attribute information; the aim is not to ignore attributes but to prevent them from substituting for structure.

**Output:** A weighted distinction between surface matches and relational matches.

### Step 6 — Abstract individual relations into role-independent forms

**Purpose:** Express the source and target relations at a level that can be compared across different domains.

1. Replace domain-specific labels with a more general relation label when the underlying role is the same.
2. Preserve argument order and direction.
3. Preserve the relation's functional meaning and causal status.
4. Preserve quantitative form when relevant (for example, proportional, inverse, threshold, or feedback relationships).
5. Avoid over-abstraction: `moves` and `causes movement` are not interchangeable in every context.

**Example**

- Source: a pressure difference contributes to water flow through a pipe; pipe resistance constrains the flow.
- Target: a voltage difference contributes to electric current through a conductor; electrical resistance constrains the current.
- Candidate abstraction: `driving_difference → flow_through_pathway`, with `pathway_resistance` constraining flow.

**Possible mathematical operations**

- Map domain-specific relation labels to a shared relational ontology.
- Compare relation embeddings or feature vectors with cosine similarity:

\[
Sim(r_s,r_t)=\frac{v_s\cdot v_t}{\|v_s\|\|v_t\|+\epsilon}
\]

- Compare typed relation signatures, including arity, argument roles, direction, and causal status. A high vector similarity must not override a type mismatch.

**Biological interpretation:** This step is a computational abstraction. Neuroscience supports representations of relations and relational integration, but does not establish that the brain explicitly translates every relation into a common symbolic ontology.

**Output:** Canonicalized source and target relation candidates.

### Step 7 — Generate candidate entity correspondences

**Purpose:** Determine which source element could correspond to which target element.

1. Compare each source entity's relational role with target entities' roles.
2. Include functional and causal roles, not just object names.
3. Allow a source element to remain unmapped when the target has no counterpart.
4. Allow one-to-one mappings as a default, but do not assume that complex real-world analogies always have perfectly one-to-one components.
5. Keep several plausible mappings until structural evidence can distinguish them.

**Possible mathematical operations**

For source entity \(i\) and target entity \(j\), define:

\[
C_{ij}=a\,Sim_{role}+b\,Sim_{relations}+c\,Sim_{attributes}+d\,Sim_{goal}
\]

Then select a partial assignment \(M\) maximizing total correspondence score while respecting type constraints. A linear assignment solver can be used for a simplified one-to-one case; graph matching or search is needed when mappings include higher-order relations or non-bijective structures.

**Biological interpretation:** Working memory maintains candidate correspondences while prefrontal systems integrate them. Hippocampal relational retrieval may supply candidate bindings. The assignment formulation is an engineering model, not a known neuronal algorithm.

**Output:** Candidate entity mappings with local confidence scores.

### Step 8 — Test relational consistency across the mapping

**Purpose:** Reject entity correspondences that fail to preserve the important relations.

1. For each source relation, check whether the proposed target correspondences form a relation with compatible meaning.
2. Check direction, argument order, causal role, temporal order, and quantitative constraints.
3. Score a mapping more highly when several connected relations are preserved together.
4. Penalize mappings that explain one relation but contradict another central relation.
5. Keep optional relations separate from core relations so that one incidental difference does not destroy an otherwise useful analogy.

**Possible mathematical operations**

A mapping score can combine node and relation matches:

\[
Score(M)=\sum_{(i,j)\in M}C_{ij}+\mu\sum_{(r_s,r_t)\in P(M)}W(r_s,r_t)-\nu\,Contradictions(M)
\]

where \(P(M)\) is the set of relation pairs preserved by the mapping. A structural consistency term can be written as:

\[
SC(M)=\frac{\sum_{r_s\in E_S}w_{r_s}\,I[\text{compatible target relation exists}]}{\sum_{r_s\in E_S}w_{r_s}+\epsilon}
\]

**Biological interpretation:** Relational integration is associated with rostrolateral prefrontal cortex; posterior parietal regions are implicated in representing first-order relations, while lateral prefrontal regions support working-memory maintenance and interference control. These assignments are evidence-based functional summaries, not exclusive modules.

**Output:** A set of structurally consistent candidate mappings.

### Step 9 — Integrate multiple relations into a higher-order pattern

**Purpose:** Identify the shared relational system rather than treating each matched pair as an isolated coincidence.

1. Group related first-order relations into a connected structure.
2. Detect whether the same ordering or dependency pattern appears in both source and target.
3. Identify higher-order relations such as `enables`, `constrains`, `regulates`, `precedes`, or `counteracts`.
4. Check whether the source and target structures support the same overall explanation.
5. Prefer a coherent relational system over a larger number of disconnected attribute matches.

**Possible mathematical operations**

- Represent first-order relations as graph edges and higher-order relations as edges between edges or as hyperedges.
- Use a coherence objective:

\[
Coherence(M)=\sum_{r\in E_S}w_r\,Preserve(r,M)+\xi\,Connectedness(M)-\zeta\,Conflict(M)
\]

- Connectedness can be estimated through the proportion of mapped core relations belonging to a common connected component.

**Biological interpretation:** This is a plausible computational counterpart of relational integration. Neuroimaging reviews identify rostrolateral prefrontal involvement in integrating first-order relations into higher-order structures. The exact neural code for a higher-order analogy schema remains an open question.

**Output:** A coherent source–target relational alignment.

### Step 10 — Retrieve and use relational knowledge from the hippocampus

**Purpose:** Bind elements from different experiences and make relevant relations available for comparison and recombination.

1. Retrieve episodes that contain one or more of the target's relevant relations.
2. Retrieve episodes that share a relational pattern even if their surface content differs.
3. Use partial cues to recover missing components when supported by memory.
4. Combine compatible relations from different experiences when the task requires it.
5. Keep the origin of each relation traceable so that a reconstructed detail is not mistaken for a directly observed fact.

**Possible mathematical operations**

- Associative retrieval can be modeled as pattern completion from a partial cue vector \(q\):

\[
\hat{x}=f_{retrieve}(q,W_{memory})
\]

- Relational binding can be modeled as a tensor or graph association between entities and relation roles.
- Retrieval confidence can be reduced when the recovered pattern depends heavily on unconstrained completion:

\[
Conf_{retrieval}=Evidence_{direct}+\alpha Evidence_{supporting}-\beta Uncertainty_{completion}
\]

**Biological interpretation:** The hippocampus is strongly implicated in binding otherwise separate elements into relational and episodic representations and in retrieving structured memories. It should not be treated as the sole storage site for all semantic relations or as a dedicated analogy engine.

**Output:** A set of retrieved relational fragments and their provenance.

### Step 11 — Control working-memory load and interference

**Purpose:** Keep the relevant source and target relations active while preventing irrelevant information from overwhelming the comparison.

1. Maintain the goal, source structure, target structure, and current mapping candidates.
2. Chunk several connected relations into a compact schema when possible.
3. Prioritize relations with high causal or goal relevance.
4. Suppress candidate matches that are attractive only because of shared names or appearances.
5. Revisit omitted relations if the current mapping produces contradictions.

**Possible mathematical operations**

- Assign each relation an activation priority:

\[
P(r)=w_1GoalRelevance+w_2CausalCentrality+w_3EvidenceStrength-w_4Interference
\]

- Under a limited capacity budget \(B\), select a subset \(R^*\):

\[
R^*=\arg\max_{R'\subseteq R}\sum_{r\in R'}P(r)\quad\text{subject to}\quad\sum_{r\in R'}cost(r)\leq B
\]

- Update activation recurrently as new consistency evidence arrives:

\[
a_i(t+1)=\sigma\left(\sum_j w_{ji}a_j(t)+b_i\right)
\]

**Biological interpretation:** Dorsolateral prefrontal systems contribute to maintaining relational information; ventrolateral prefrontal control is implicated in handling interference; parietal regions contribute to relational representations. Working memory is limited, so chunking and prioritization are important computational approximations.

**Output:** A manageable set of active relations and mappings.

### Step 12 — Derive candidate inferences from the source

**Purpose:** Generate knowledge that is true or supported in the source and might also hold in the target if the mapping is valid.

1. Inspect source relations not yet represented in the target.
2. Determine whether each source relation follows from the mapped core structure or merely from a source-specific detail.
3. Transfer the relation as a hypothesis, not as a guaranteed fact.
4. Generate candidate predictions about the target.
5. Record the chain of correspondence supporting each prediction.

**Example**

If the source relation says that increasing pathway resistance reduces flow under a fixed driving difference, the target hypothesis may be that increasing electrical resistance reduces current under a fixed voltage. The transferred relation is conditional; the relevant electrical assumptions must still be checked.

**Possible mathematical operations**

- Conditional transfer probability:

\[
P(h_T\mid M,G_S,G_T)=P(h_T\mid\text{mapped relations, target evidence})
\]

- A simple confidence update can combine mapping confidence and source-rule reliability:

\[
Conf(h_T)=Conf(M)\times Rel(rule_S)\times Fit(target\ conditions)
\]

The product is a conservative engineering heuristic, not a universal law of human confidence.

**Biological interpretation:** Candidate inference generation plausibly depends on relational integration, working-memory manipulation, and retrieval of source knowledge. The particular equations are computational proposals.

**Output:** Target hypotheses with explicit support chains.

### Step 13 — Check domain-specific constraints and disanalogies

**Purpose:** Prevent overextension of the analogy.

1. Identify what the source and target have in common structurally.
2. Identify important differences in material, mechanism, scale, boundary conditions, and causal dynamics.
3. Determine whether each transferred relation still holds under target-specific laws and evidence.
4. Remove or qualify inferences that depend on a feature absent from the target.
5. State where the analogy stops being useful.

**Example: water-flow and electrical-circuit analogy**

The analogy can help explain driving differences, flow/current, and resistance. It does not imply that water and electric charge are the same substance, that every hydraulic detail has an electrical counterpart, or that all circuit behaviours can be inferred from a simple pipe model. Capacitance, inductance, electromagnetic propagation, and circuit-specific effects require additional concepts.

**Possible mathematical operations**

- Constraint satisfaction: accept a transfer only if target constraints are not violated.
- Bayesian update:

\[
P(h\mid e,M)\propto P(e\mid h,M)P(h\mid M)
\]

- Penalize unsupported extensions:

\[
Score_{transfer}=Support_{structural}+Support_{target\ evidence}-\lambda ViolatedConstraints-\omega UnsupportedAssumptions
\]

**Biological interpretation:** Executive control and domain knowledge help evaluate whether an attractive analogy is appropriate. Temporal and other domain-specific cortical systems supply knowledge needed to detect differences. The hippocampus may provide relevant prior episodes, but it does not guarantee that a transfer is valid.

**Output:** A set of qualified, target-compatible inferences and explicit disanalogies.

### Step 14 — Adapt transferred knowledge to the target

**Purpose:** Convert the abstract relational match into a target-specific explanation or solution.

1. Replace source entities with their mapped target counterparts.
2. Translate source-specific units, values, and mechanisms into target-domain terms.
3. Recalculate quantitative consequences using target-domain equations where available.
4. Preserve the relation's direction and conditions while adapting its parameters.
5. Generate a clear explanation showing which parts came from the analogy and which parts came from target-specific knowledge.

**Possible mathematical operations**

- Parameter translation: \(\theta_T=T(\theta_S,\phi_T)\), where \(\phi_T\) contains target-domain conditions.
- Dimensional consistency checks ensure that units are compatible.
- If both domains have comparable quantitative laws, compare normalized functional forms rather than raw numerical values.
- For a circuit example, use the target's own electrical law (such as Ohm's law for an appropriate ohmic element) rather than copying numerical values from a hydraulic system.

**Biological interpretation:** Adaptation requires combining the abstract relation with domain-specific semantic knowledge and current perceptual or task evidence. It is not just copying a source answer.

**Output:** A target-domain explanation, prediction, or proposed solution.

### Step 15 — Evaluate predictions against evidence

**Purpose:** Test whether the analogy makes useful and accurate predictions.

1. Derive observable consequences from the adapted model.
2. Compare them with known target evidence, measurements, or the result of an action.
3. Check not only whether the final answer is correct but whether the mapped relations actually predicted it.
4. Distinguish a correct answer produced by a valid analogy from a lucky guess.
5. If the prediction fails, localize the failure: source retrieval, role mapping, relation preservation, transferred rule, target constraints, or parameter adaptation.

**Possible mathematical operations**

- Prediction error:

\[
\delta_t=y_t-\hat{y}_t
\]

- For probabilistic predictions, use log loss or another calibrated scoring rule:

\[
L=-\log P(y_{observed}\mid model)
\]

- Update confidence according to prediction quality and evidence reliability.

**Biological interpretation:** Comparison of expected and observed outcomes can recruit learning and control systems. Prediction error is a useful computational description, but different neural learning signals operate at different levels and timescales; no single error signal explains all analogy evaluation.

**Output:** A validated, revised, or rejected transfer.

### Step 16 — Revise the mapping and search for a better source when necessary

**Purpose:** Treat analogy as an iterative process rather than a one-shot match.

1. If the target evidence conflicts with a transferred inference, identify which mapping assumption caused the conflict.
2. Remove an unsupported correspondence, revise a relation, or narrow the scope of the analogy.
3. Consider an alternative source case if the original source only partially fits.
4. Recompute the structural-consistency score.
5. Stop when the mapping is sufficiently coherent for the task, or report that no suitable analogy has been found.

**Possible mathematical operations**

- Iterative mapping update:

\[
M_{t+1}=\arg\max_M\left[Score(M)-\lambda Conflict(M\mid evidence_t)\right]
\]

- Use a stopping condition such as:

\[
|Score(M_{t+1})-Score(M_t)|<\epsilon
\]

or stop when all critical constraints are satisfied and confidence exceeds a task-specific threshold.

**Biological interpretation:** Recurrent comparison and control are consistent with distributed, interactive processing. The exact sequence of remapping and stopping is a computational design choice, not an established fixed neural cycle.

**Output:** A refined mapping or an explicit no-transfer decision.

### Step 17 — Abstract a reusable analogy schema

**Purpose:** Learn a more general relational pattern that can support future transfer.

1. Compare multiple successful analogies and identify relations that recur across different domains.
2. Remove domain-specific entity names while preserving stable relational roles.
3. Keep conditions and exceptions that determine when the pattern is valid.
4. Store the schema with examples and counterexamples.
5. Use the schema as a future retrieval cue, but do not allow it to override new evidence.

**Possible mathematical operations**

- Estimate a relation's recurrence across cases:

\[
Frequency(r)=\frac{\#\text{cases containing }r}{\#\text{relevant cases}}
\]

- Estimate reliability with a smoothed proportion:

\[
\hat{p}(r)=\frac{k+\alpha}{n+\alpha+\beta}
\]

where \(k\) of \(n\) relevant cases support the relation and \(\alpha,\beta\) are prior pseudo-counts.

- Preserve exceptions as explicit conditions rather than averaging them away.

**Biological interpretation:** Repeated relational learning can contribute to generalized schemas and semantic knowledge. The hippocampus can support the binding of distinct experiences, while neocortical systems contribute to more distributed and generalized knowledge. The precise conditions under which a schema is abstracted are still researched.

**Output:** A reusable, conditional analogy schema.

### Step 18 — Consolidate and make the new knowledge available for future reasoning

**Purpose:** Retain the successful relational learning and its boundaries.

1. Store the episode: target, chosen source, mapping, predictions, evidence, and outcome.
2. Strengthen the semantic schema only to the degree justified by the evidence.
3. Retain episodic context when it is important to remember why an inference was accepted or rejected.
4. During later retrieval, use both the generalized schema and the specific episode when needed.
5. Re-evaluate the schema when new examples contradict it.

**Possible mathematical operations**

- Confidence-weighted update of a schema parameter:

\[
\theta_{t+1}=\theta_t+\eta c_t(y_t-\hat{y}_t)
\]

- A recency- and reliability-weighted evidence accumulator:

\[
W_r(t)=\sum_i q_i e^{-\lambda(t-t_i)}
\]

where \(q_i\) is the reliability of each experience. This is a practical memory model, not a literal account of synaptic consolidation.

**Biological interpretation:** Long-term storage depends on interactions among hippocampal and cortical memory systems and is influenced by reactivation, subsequent experience, and sleep. A newly formed analogy may remain episodic initially and become part of broader semantic knowledge through repeated use and integration.

**Output:** An updated episodic record and, when sufficiently supported, an updated generalized schema.

## 5. Integrated computational pseudocode

```python
from dataclasses import dataclass, field
from typing import Any, Iterable

@dataclass
class Entity:
    entity_id: str
    kind: str
    attributes: dict[str, Any] = field(default_factory=dict)

@dataclass
class Relation:
    name: str
    arguments: tuple[str, ...]
    direction: str = "directed"
    conditions: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    importance: float = 1.0

@dataclass
class SituationGraph:
    entities: dict[str, Entity]
    relations: list[Relation]
    constraints: list[str] = field(default_factory=list)
    provenance: dict[str, str] = field(default_factory=dict)

@dataclass
class TransferHypothesis:
    relation: Relation
    confidence: float
    support: list[str]
    caveats: list[str] = field(default_factory=list)


def retrieve_sources(target_cues, memory, goal, limit=5):
    """Return candidate source episodes/schemas ranked by cue and goal relevance."""
    candidates = memory.search(target_cues)
    ranked = sorted(
        candidates,
        key=lambda item: item.score_cues(target_cues)
                    + item.score_goal(goal)
                    + item.score_relation_relevance(target_cues),
        reverse=True,
    )
    return ranked[:limit]


def encode_target(observations, domain_knowledge) -> SituationGraph:
    """Create entities and typed relations; keep observations distinct from inferences."""
    return domain_knowledge.parse_situation(observations)


def canonicalize_relation(relation, ontology):
    """Map a domain-specific relation to a typed abstract relation when justified."""
    return ontology.normalize(
        relation.name,
        arity=len(relation.arguments),
        direction=relation.direction,
        conditions=relation.conditions,
    )


def candidate_entity_pairs(source: SituationGraph, target: SituationGraph):
    """Generate plausible role-based correspondences, not just name matches."""
    pairs = []
    for s in source.entities.values():
        for t in target.entities.values():
            score = role_similarity(s, source, t, target)
            if score > 0:
                pairs.append((s.entity_id, t.entity_id, score))
    return pairs


def mapping_score(source, target, mapping, ontology):
    node_score = sum(pair.score for pair in mapping.entity_pairs)
    preserved = 0.0
    contradictions = 0.0
    for source_relation in source.relations:
        mapped_args = tuple(mapping.get(a) for a in source_relation.arguments)
        if any(arg is None for arg in mapped_args):
            continue
        match = find_compatible_relation(
            canonicalize_relation(source_relation, ontology),
            mapped_args,
            target,
            ontology,
        )
        if match is not None:
            preserved += source_relation.importance * match.confidence
        elif source_relation.importance >= mapping.core_relation_threshold:
            contradictions += source_relation.importance
    return node_score + preserved - mapping.contradiction_penalty * contradictions


def build_candidate_inferences(source, target, mapping, ontology):
    hypotheses = []
    for relation in source.relations:
        if target_already_contains_mapped_relation(relation, target, mapping, ontology):
            continue
        if not relation_is_transferable(relation, source, target, mapping):
            continue
        mapped = map_relation_arguments(relation, mapping)
        if mapped is None:
            continue
        confidence = (
            mapping.confidence
            * relation.confidence
            * target_condition_fit(relation, target, mapping)
        )
        hypotheses.append(
            TransferHypothesis(
                relation=mapped,
                confidence=max(0.0, min(1.0, confidence)),
                support=["source relation", "entity-role mapping", "structural consistency"],
            )
        )
    return hypotheses


def check_target_constraints(hypotheses, target, domain_knowledge):
    accepted, rejected = [], []
    for hypothesis in hypotheses:
        result = domain_knowledge.check(hypothesis.relation, target)
        if result.is_compatible:
            accepted.append(hypothesis)
        else:
            hypothesis.caveats.append(result.explanation)
            rejected.append(hypothesis)
    return accepted, rejected


def analogical_transfer(observations, goal, memory, domain_knowledge, ontology):
    target = encode_target(observations, domain_knowledge)
    cues = extract_relational_cues(target, goal)
    source_candidates = retrieve_sources(cues, memory, goal)
    if not source_candidates:
        return {"status": "no_source_found", "target": target}

    evaluated = []
    for source_item in source_candidates:
        source = source_item.to_situation_graph()
        pairs = candidate_entity_pairs(source, target)
        mappings = search_structurally_consistent_mappings(source, target, pairs)
        for mapping in mappings:
            score = mapping_score(source, target, mapping, ontology)
            evaluated.append((score, source, mapping))

    if not evaluated:
        return {"status": "no_valid_mapping", "target": target}

    evaluated.sort(key=lambda item: item[0], reverse=True)
    score, source, mapping = evaluated[0]
    hypotheses = build_candidate_inferences(source, target, mapping, ontology)
    accepted, rejected = check_target_constraints(
        hypotheses, target, domain_knowledge
    )
    explanation = adapt_inferences_to_target(accepted, target, domain_knowledge)
    predictions = derive_testable_predictions(explanation, goal)
    validation = evaluate_predictions(predictions, target, domain_knowledge)

    memory.record_transfer(
        source=source,
        target=target,
        mapping=mapping,
        accepted=accepted,
        rejected=rejected,
        validation=validation,
    )
    return {
        "status": "validated" if validation.supports_mapping else "revise_or_reject",
        "source": source,
        "target": target,
        "mapping": mapping,
        "accepted_inferences": accepted,
        "rejected_inferences": rejected,
        "explanation": explanation,
        "validation": validation,
    }
```

The pseudocode deliberately leaves important operations (parsing, graph search, relation compatibility, constraint checking, and validation) as domain-aware functions. A realistic implementation needs a typed ontology, explicit confidence and provenance, and benchmarks that test both successful and misleading analogies. It should never accept an inference only because a language model or similarity score produced it.

## 6. A compact worked example: water flow and an electrical circuit

| Relational role | Source: hydraulic system | Target: electrical circuit |
|---|---|---|
| Driving difference | Pressure difference | Voltage difference |
| Flowing quantity | Water flow rate | Electric current |
| Pathway | Pipe | Conductive path / circuit element |
| Constraint on flow | Hydraulic resistance | Electrical resistance |
| Candidate shared relation | Greater resistance limits flow under suitable fixed conditions | Greater resistance limits current under suitable fixed conditions |

The useful product is not a claim that water and electricity are identical. It is a conditional relational schema: a driving difference produces a flow through a pathway, and a resistance-like property constrains that flow. The target-domain model must determine which quantitative law and boundary conditions apply.

## 7. Important mathematical operations collected

1. **Cue matching and retrieval ranking:** weighted dot products, similarity functions, and probabilistic ranking.
2. **Feature normalization:** centering and scaling numeric attributes when appropriate.
3. **Graph construction:** adjacency matrices, typed edges, higher-order relations, and hyperedges.
4. **Relational similarity:** vector similarity plus strict type, argument-order, and direction checks.
5. **Correspondence search:** partial assignment, graph matching, constraint satisfaction, and beam/search procedures.
6. **Structural consistency:** weighted counts of preserved relations and penalties for contradictions.
7. **Working-memory prioritization:** weighted selection under a capacity budget and recurrent activation updates.
8. **Inference generation:** conditional probabilities and confidence propagation.
9. **Domain adaptation:** parameter translation, dimensional analysis, and target-specific laws.
10. **Evaluation and learning:** prediction error, probabilistic loss, evidence-weighted updates, and schema reliability estimates.

These operations are available to an engineered model of analogical reasoning. They should not be presented as direct measurements of mathematical calculations performed by the brain.

## 8. Failure modes and safeguards

- **Surface-match trap:** two objects share appearance or labels but not the relevant roles. Safeguard: score relational structure separately.
- **Shallow relation match:** one relation is shared but the larger system is not. Safeguard: require multiple connected core relations for complex transfers.
- **Role reversal:** source and target arguments are mapped in the wrong order. Safeguard: preserve direction and argument roles.
- **Causal overreach:** a correlation in the source is treated as a causal law in the target. Safeguard: record causal status explicitly.
- **False one-to-one mapping:** one source element is forced to correspond to exactly one target element even when the target mechanism has a different composition. Safeguard: permit partial and, when warranted, grouped mappings.
- **Overgeneralization:** a valid analogy is applied outside its boundary conditions. Safeguard: store assumptions, scope conditions, and counterexamples.
- **Source-memory distortion:** a reconstructed memory is treated as an exact record. Safeguard: preserve provenance and retrieval confidence.
- **Confirmation bias:** only evidence supporting the selected analogy is considered. Safeguard: generate alternatives and search for disconfirming predictions.
- **Uncalibrated confidence:** a high similarity score is mistaken for high truth probability. Safeguard: calibrate confidence against held-out examples and actual outcomes.
- **Answer-only evaluation:** a lucky guess is rewarded as if the mapping were correct. Safeguard: evaluate the correspondences and predicted consequences, not just the final answer.

## 9. Validation and experimental tests

A computational implementation should be tested on analogy sets where surface similarity and structural similarity are independently manipulated. Include at least these conditions:

1. **High structure / low surface similarity:** different objects but matching relational roles.
2. **Low structure / high surface similarity:** familiar-looking entities with different relations.
3. **Partial mapping:** a source contains components with no target counterpart.
4. **Relational reversal:** the candidate target reverses causal direction or argument order.
5. **Multiple candidate sources:** only one source preserves the target's central relational system.
6. **Boundary-condition violation:** an apparently good analogy fails because a target-specific constraint is violated.
7. **Counterexample update:** after a failed transfer, measure whether the model narrows the schema rather than simply strengthening the original analogy.
8. **Novel transfer:** test whether an abstracted schema supports a new domain not used during training.

Measure mapping accuracy, relation-preservation F1, inference precision, false-transfer rate, calibration error, and performance on novel domains. Human studies can additionally measure response time, confidence, eye movements, and sensitivity to distracting surface features. Neuroimaging can test task-dependent recruitment and connectivity, but activation alone does not prove that a particular algorithmic step is implemented in a particular region.

## 10. Scientific grounding and limitations

The broad biological account is supported by research on relational integration and analogical reasoning. A review by Holyoak and Monti describes posterior parietal contributions to first-order relational representations, rostrolateral prefrontal contributions to integrating relations into higher-order structures, dorsolateral prefrontal involvement in maintaining relations in working memory, and ventrolateral prefrontal involvement in interference control. A meta-analysis of 27 imaging experiments found the left rostrolateral prefrontal cortex to be one of the most consistent regions across analogy studies, alongside domain-sensitive regions and broader frontoparietal systems. A review of analogy component processes distinguishes access, mapping, inference, and schema induction, while emphasizing that the links between computational subprocesses and actual neural mechanisms remain incomplete.

The user-provided source also names medial prefrontal and temporal-lobe networks and hippocampal binding. These are relevant parts of the broader network, but the current evidence does not justify assigning all analogy operations to those regions alone. Different analogy tasks can recruit overlapping but non-identical networks. The hippocampus is included here as a contributor to relational memory and retrieval, not as a dedicated analogy processor.

**Interpretive caution:** The algorithm is a biologically informed engineering hypothesis. The mathematical equations operationalize ideas such as similarity, consistency, uncertainty, and learning. They are not direct recordings of neuronal arithmetic, and the 18 steps should not be assumed to unfold as a strict serial pipeline in the living brain.

## 11. Selected references

1. Holyoak, K. J., & Monti, M. M. (2021). *Relational Integration in the Human Brain: A Review and Synthesis*. Journal of Cognitive Neuroscience, 33(3), 341–356. https://doi.org/10.1162/jocn_a_01619
2. Vartanian, O. et al. (2016). *General and specialized brain correlates for analogical reasoning: A meta-analysis of functional imaging studies*. Human Brain Mapping, 37(5), 1953–1969. https://doi.org/10.1002/hbm.23149
3. Green, A. E. et al. / related review: *The Neural Correlates of Analogy Component Processes* (2022). Cognitive Science. https://doi.org/10.1111/cogs.13116
4. Green, A. E., Kraemer, D. J. M., Fugelsang, J. A., Gray, J. R., & Dunbar, K. N. (2010). *Connecting long distance: Semantic distance in analogical reasoning modulates frontopolar cortex activity*. Cerebral Cortex, 20(1), 70–76. https://doi.org/10.1093/cercor/bhp081
5. Bunge, S. A., Wendelken, C., Badre, D., & Wagner, A. D. (2005). *Analogical reasoning and prefrontal cortex: Evidence for separable retrieval and integration mechanisms*. Cerebral Cortex, 15(3), 239–249. https://doi.org/10.1093/cercor/bhh126
6. Gentner, D. (1983). *Structure-mapping: A theoretical framework for analogy*. Cognitive Science, 7(2), 155–170. https://doi.org/10.1207/s15516709cog0702_3

---

**End of Point 11.** The next point should not be started until the user explicitly types `continue`.
