# Point 09 — Schema Formation: Structured Concepts, Roles, Slots, and Relations

## Scope and biological framing

This algorithm expands the document's statement that a schema is a structured concept or frame. For example, a *restaurant* schema is not merely a list of associated words such as *waiter*, *menu*, and *paying*. It is an organized model in which participants and objects occupy roles, relations connect them, events tend to unfold in partially predictable orders, and expectations depend on goals and context.

The algorithm starts from multiple concrete experiences and constructs a flexible relational representation. It covers: (1) detecting recurring event structures, (2) identifying roles and variable slots, (3) learning relations among slots, (4) encoding temporal and causal expectations, (5) organizing schemas hierarchically, and (6) using and revising schemas when new situations are encountered.

**Important scientific distinction:** the brain is not known to execute the equations below as explicit symbolic arithmetic. The equations are computational descriptions of operations that neural populations and synaptic networks may approximate. The biological evidence supports contributions from hippocampal and medial-temporal systems, medial and orbitofrontal prefrontal regions, and distributed cortical networks. The exact division of labour, representational format, and timescale are active research questions. A schema should therefore be treated as a useful computational abstraction, not as a single brain location or a literal data structure stored intact in one region.

---

## 1. Define the schema-learning problem

**Goal:** transform repeated, related episodes into a reusable relational model that preserves meaningful structure while allowing the people, objects, locations, and details to vary.

For a restaurant schema, individual episodes might contain different restaurants, servers, menus, meals, payment methods, and social situations. The reusable structure is more abstract:

- a customer enters or arrives;
- a customer obtains information about available food;
- an order is communicated to a service provider;
- food is delivered and consumed;
- payment or settlement occurs;
- participants may leave, provide feedback, or perform optional actions.

The schema must not force every episode to contain every event. A takeaway order, buffet, prepaid meal, or restaurant visit with no human waiter may instantiate a different path through the same broader structure.

**Computational representation:**

Let the experience set be

\[
\mathcal{E}=\{E_1,E_2,\ldots,E_N\}
\]

and let a candidate schema be

\[
S=(C,R,T,K,G)
\]

where:

- \(C\) = context or situation class;
- \(R\) = role/slot definitions;
- \(T\) = relations among roles and events;
- \(K\) = constraints, typicalities, and transition knowledge;
- \(G\) = goals, purposes, and outcome structure.

The schema is successful when it captures recurring relational structure, predicts useful aspects of new episodes, and remains revisable when evidence contradicts it.

**Biological interpretation:** hippocampal and adjacent medial-temporal systems can bind the elements of individual episodes. Repeated experience and reactivation can support integration across episodes. Medial prefrontal and orbitomedial prefrontal systems are implicated in representing, retrieving, and using structured prior knowledge, while relevant content remains distributed across cortical systems.

---

## 2. Receive and segment concrete experiences into events

**Input:** multimodal episodes containing people, objects, locations, actions, speech, intentions, and outcomes.

**Algorithmic operations:**

1. Receive the current episode representation from perceptual, language, memory, and action systems.
2. Detect changes in participants, locations, goals, actions, or outcomes.
3. Divide the continuous stream into candidate events or sub-events.
4. Preserve the links among sub-events so that the episode is not reduced to an unordered bag of features.
5. Attach temporal indices and contextual metadata to each event.

An event can be represented as

\[
 e_t=(x_t,a_t,o_t,c_t,\tau_t)
\]

where \(x_t\) is the current state or situation, \(a_t\) is an action, \(o_t\) is an outcome, \(c_t\) is context, and \(\tau_t\) is temporal information.

A simple event-boundary score can combine several changes:

\[
B_t=\alpha\,\Delta_{percept}(t)+\beta\,\Delta_{goal}(t)
      +\gamma\,\Delta_{action}(t)+\delta\,\Delta_{context}(t)
\]

A high \(B_t\) indicates a possible event boundary. This is a computational model, not a claim that the brain sums four explicit values.

**Neural interpretation:** event segmentation depends on distributed perceptual and cognitive systems. Hippocampal activity can distinguish event contexts and bind successive elements, while medial prefrontal and posterior cortical systems contribute to situation models and event structure. The particular boundary signals may vary by task and modality.

**Output:** an ordered collection of structured event fragments, each retaining its episode identity.

---

## 3. Encode each episode while preserving its individual identity

Before generalizing, the system must retain enough detail to distinguish one experience from another.

**Algorithmic operations:**

1. Encode perceptual attributes such as appearance, sound, spatial location, and movement.
2. Encode people and objects as distinct entities rather than conflating similar instances.
3. Bind each entity to its actions, attributes, and context.
4. Preserve the order and co-occurrence of events.
5. Store an episode-specific trace that can later be compared with other episodes.

A relational episode graph can be written as

\[
G_i=(V_i,E_i,A_i)
\]

where \(V_i\) contains entities and events, \(E_i\) contains relations, and \(A_i\) contains attributes. For example:

- `customer_1 --orders--> meal_1`
- `server_1 --delivers--> meal_1`
- `customer_1 --pays--> bill_1`

**Mathematical operations:** feature binding can be modeled with an entity-role assignment matrix \(M_i\), where \(M_{jk}\) measures how strongly entity or feature \(j\) is associated with role \(k\). A normalized assignment may be represented by

\[
M'_{jk}=\frac{\exp(M_{jk})}{\sum_l\exp(M_{jl})}
\]

This softmax expression is one possible computational formalization of competing role assignments, not a literal neural operation.

**Neural interpretation:** hippocampal circuits are strongly implicated in binding elements into relational and episodic representations. Pattern separation and pattern completion are useful functional descriptions of complementary hippocampal computations: keeping similar episodes distinguishable while enabling retrieval from partial cues. The specific microcircuit account is more complex than a single equation.

**Output:** episode-specific relational graphs that remain available for later cross-episode comparison.

---

## 4. Detect recurring situation classes across episodes

The learner must determine which episodes are relevant to one another. Similarity may depend on shared goals and relations, not just shared objects.

**Algorithmic operations:**

1. Compare episode contexts, goals, event sequences, and relational structures.
2. Weight meaningful similarities more strongly than incidental overlap.
3. Group episodes that plausibly instantiate the same broad situation.
4. Keep uncertain or borderline cases in separate candidate groups until evidence improves.

A candidate similarity score is

\[
Sim(E_i,E_j)=w_c S_c+w_g S_g+w_r S_r+w_t S_t+w_f S_f
\]

where the component scores measure context, goal, relation, temporal-structure, and feature similarity. Weights should be learned or task-dependent; a restaurant's goal and service relations may be more informative than the colour of its walls.

A probabilistic grouping rule can be expressed as

\[
P(z_i=z_j\mid E_i,E_j)\propto
P(E_i,E_j\mid z_i=z_j)P(z_i=z_j)
\]

where \(z_i\) denotes the latent situation class of episode \(i\).

**Mathematical operations:** feature weighting, similarity computation, clustering, posterior probability estimation, and thresholding with an uncertainty region.

**Neural interpretation:** memory reactivation and medial-temporal/prefrontal interactions may support comparison and integration. The brain need not perform a single global clustering operation; grouping can emerge incrementally as related experiences reactivate one another.

**Output:** candidate episode families for schema induction.

---

## 5. Identify recurring participants and functional roles

A role is defined by what an entity does in the relational structure, not merely by its name or appearance. A *waiter* and a *self-service ordering screen* can perform overlapping service functions even though they are different kinds of entities.

**Algorithmic operations:**

1. Extract entities from each episode.
2. Record their observed actions, properties, relations, and outcomes.
3. Compare the functional contribution of each entity across episodes.
4. Group entities that fill similar relational positions.
5. Assign provisional role labels such as `customer`, `service_provider`, `menu_or_information_source`, `order`, `food_item`, and `payment_mechanism`.
6. Retain multiple possible role assignments when an entity has more than one function.

A role assignment score can be modeled as

\[
Score(e,r)=\theta_1 ActionMatch(e,r)+\theta_2 RelationMatch(e,r)
          +\theta_3 GoalMatch(e,r)+\theta_4 ContextMatch(e,r)
\]

The selected role is

\[
r^*=\arg\max_r Score(e,r)
\]

subject to any constraints on role compatibility. If two roles are similarly plausible, the system should retain ambiguity rather than force a premature label.

**Biological interpretation:** role-like representations are likely distributed across conceptual, episodic, and control networks. The hippocampus can bind who did what to whom; prefrontal systems can emphasize task-relevant or goal-relevant relations. There is no evidence that one neuron or region stores a universal symbolic `customer` slot.

**Output:** provisional role classes linked to the specific entities that instantiated them in each episode.

---

## 6. Convert repeated entities into variable slots

A schema slot is a placeholder for a role or feature that can take different fillers across episodes.

For example:

- `customer` can be filled by different people;
- `restaurant` can be filled by different establishments;
- `meal` can be filled by different foods;
- `payment_method` can be cash, card, app, or prepaid credit.

**Algorithmic operations:**

1. Align the roles across episodes.
2. Find the specific entities that fill each recurring role.
3. Replace episode-specific identities with variable placeholders while preserving links to the original episodes.
4. Record allowable filler types and known exceptions.
5. Keep a distinction between required, typical, optional, and context-dependent slots.

Formally, a slot \(s_k\) has a filler set

\[
F(s_k)=\{f_{k1},f_{k2},\ldots,f_{km}\}
\]

and a filler distribution

\[
P(f\mid s_k,C)
\]

which represents how likely a filler is in context \(C\). A slot can also have type constraints, e.g. a `payment_method` slot expects a method or instrument, not an arbitrary location.

**Mathematical operations:** entity alignment, clustering, abstraction from individual identities, probability estimation, and type-constraint learning.

**Neural interpretation:** variable slots are a formal representation for stable relational positions across episodes. The underlying neural implementation may be distributed across population codes and learned associations; no literal slot register is implied.

**Output:** a candidate frame with roles represented as slots and episode-specific entities retained as their observed fillers.

---

## 7. Extract relations among slots

Slots alone do not make a schema. The crucial information is how the slots relate.

Examples:

- `customer --requests--> food_or_service`;
- `service_provider --delivers--> ordered_item`;
- `customer --transfers_payment_to--> provider_or_system`;
- `menu --describes_options_for--> meal`.

**Algorithmic operations:**

1. Retrieve the relational links from each episode graph.
2. Map episode-specific entities onto their corresponding schema slots.
3. Identify relations that recur with the same role structure.
4. Distinguish directed from undirected relations.
5. Preserve relation arguments and their order.
6. Record alternative relations when episodes differ.

A schema relation can be represented as a typed edge

\[
r_k=(s_a,\rho,s_b)
\]

where \(s_a\) and \(s_b\) are slots and \(\rho\) is a relation type.

The recurrence strength of a relation can be estimated as

\[
Conf(\rho)=\frac{N_{support}(\rho)+a}{N_{eligible}+a+b}
\]

where \(a,b\) are smoothing parameters. The denominator should include only episodes in which the relation could reasonably have occurred; otherwise optional actions may be incorrectly treated as absent evidence.

**Mathematical operations:** graph matching, relational alignment, frequency estimation, Bayesian smoothing, and direction-sensitive edge comparison.

**Neural interpretation:** hippocampal relational binding and prefrontal integration are relevant to learning relations across events. Relational structures may be represented as coordinated patterns of neural activity rather than as explicit graph edges.

**Output:** a relational network connecting schema slots.

---

## 8. Learn temporal order and event-transition structure

Many schemas describe what typically happens next. The order is often flexible, but some transitions are more probable than others.

**Algorithmic operations:**

1. Align the event sequences from different episodes.
2. Identify recurring event transitions.
3. Distinguish mandatory precedence constraints from common but optional orderings.
4. Estimate the probability of one event following another in a given context.
5. Represent branching paths for alternative versions of the same schema.

A first-order transition model is

\[
P(e_{t+1}=j\mid e_t=i,C)=
\frac{N(i\rightarrow j,C)+\lambda}
     {N(i,C)+\lambda |\mathcal{J}|}
\]

where \(N(i\rightarrow j,C)\) counts observed transitions, \(\lambda\) smooths sparse estimates, and \(\mathcal{J}\) is the possible next-event set.

More complex schemas may require higher-order or hierarchical transitions:

\[
P(e_{t+1}\mid e_t,e_{t-1},goal,C)
\]

A restaurant experience may branch into dine-in, takeaway, delivery, buffet, or prepaid paths. The schema should represent these alternatives instead of imposing a single rigid sequence.

**Mathematical operations:** sequence alignment, transition counting, conditional probability estimation, Markov or semi-Markov modeling, and branch detection.

**Neural interpretation:** temporal order and event structure involve distributed memory and predictive systems. Hippocampal sequence representations and prefrontal expectations may contribute, but the brain's event model is not necessarily a Markov chain.

**Output:** a probabilistic event-flow structure with alternative paths and context-sensitive transitions.

---

## 9. Infer functional and causal relations cautiously

Repeated temporal association does not automatically establish causation. A robust schema should distinguish observed co-occurrence, functional dependence, and a causal hypothesis.

**Algorithmic operations:**

1. Detect recurring associations among events and outcomes.
2. Test whether one event reliably precedes or enables another.
3. Use background knowledge and counterexamples to evaluate causal interpretations.
4. Mark relations as `observed`, `functional`, or `causal-hypothesis` with confidence values.
5. Avoid converting a single correlation into a deterministic rule.

A predictive association can be estimated with conditional probabilities:

\[
P(Y\mid X),\qquad P(Y\mid \neg X)
\]

A simple association contrast is

\[
\Delta=P(Y\mid X)-P(Y\mid\neg X)
\]

but \(\Delta\) alone does not prove that \(X\) causes \(Y\). Causal interpretation requires additional assumptions, interventions, temporal evidence, or a plausible mechanism.

**Neural interpretation:** prefrontal and hippocampal networks support relational inference and the use of prior knowledge. The computation of causal structure is distributed and can recruit additional systems depending on the domain.

**Output:** a relation set whose certainty and evidential status are explicit.

---

## 10. Separate invariant structure from incidental details

A schema should preserve what is stable and useful across episodes without absorbing every accidental feature.

**Algorithmic operations:**

1. Estimate how consistently each feature or relation occurs in relevant episodes.
2. Estimate whether it predicts the situation's goal, outcome, or next event.
3. Discount features that vary widely and do not improve prediction.
4. Preserve context-specific attributes when they are useful in a narrower sub-schema.
5. Avoid discarding rare but diagnostic features merely because they are infrequent.

One possible feature utility score is

\[
U(f)=\eta_1 Stability(f)+\eta_2 PredictiveValue(f)
      +\eta_3 GoalRelevance(f)-\eta_4 ContextSpecificityPenalty(f)
\]

Predictive value can be approximated with mutual information:

\[
I(F;Y)=\sum_{f,y}P(f,y)\log\frac{P(f,y)}{P(f)P(y)}
\]

However, mutual information can favour frequent correlates and does not by itself determine whether a feature belongs in every version of a schema. The system should retain a confidence distribution rather than making all schema properties binary.

**Neural interpretation:** abstraction and dimensionality reduction are plausible computational descriptions of how repeated experiences yield compact knowledge. A 2025 perspective proposes that prediction errors, hierarchical learning, and dimensionality reduction are principles that may contribute to schema learning; this is a theoretical synthesis, not a settled neural algorithm.

**Output:** a weighted set of core, typical, optional, and incidental properties.

---

## 11. Represent the schema as a structured relational frame

Combine slots, relations, temporal structure, and constraints into one integrated representation.

A frame can be represented as

\[
S=(V_s,E_r,P_t,C_s,G_s,\Theta)
\]

where:

- \(V_s\) = schema slots;
- \(E_r\) = typed relations among slots;
- \(P_t\) = temporal/event-transition model;
- \(C_s\) = constraints and compatibility conditions;
- \(G_s\) = goals and expected outcomes;
- \(\Theta\) = learned probabilities, weights, and confidence values.

**Algorithmic operations:**

1. Merge the stable role slots.
2. Attach the recurring relations.
3. Add temporal transitions and optional branches.
4. Add type and compatibility constraints.
5. Link goals and outcomes to the events that support them.
6. Retain pointers to example episodes so that the abstraction can be checked against its source evidence.

The result is a graph or frame with probabilistic attributes, not merely a flat category prototype.

**Mathematical operations:** graph construction, edge typing, constraint composition, parameter estimation, and uncertainty propagation.

**Neural interpretation:** a schema is better viewed as an organized pattern of distributed knowledge than as one isolated representation. The current literature considers hippocampal-prefrontal interactions and broader cortical networks; recent work also investigates structured low-dimensional population geometry as a possible representation format.

**Output:** a provisional schema graph with evidence links and uncertainty.

---

## 12. Organize schemas hierarchically and compositionally

Schemas can be nested. A broad `food-service` schema can include `restaurant-dining`, `takeaway`, and `delivery` sub-schemas. Each sub-schema inherits some slots and relations while adding specialized constraints.

**Algorithmic operations:**

1. Compare candidate schemas for shared roles, goals, and relations.
2. Form a higher-level schema when a stable common structure exists.
3. Preserve lower-level schemas for context-specific details.
4. Allow a schema to reuse a component frame, such as `payment`, `travel`, or `greeting`.
5. Avoid merging schemas solely because they share a few surface features.

A hierarchical model can be expressed as

\[
S_{child}\rightarrow S_{parent}+\Delta S_{child}
\]

where the child inherits the parent structure and contributes its specific additions or constraints.

A Bayesian model-selection objective might be

\[
S^*=\arg\max_S\left[\log P(\mathcal{E}\mid S)-\kappa Complexity(S)\right]
\]

The complexity penalty reduces the tendency to create an unnecessarily elaborate schema for every small variation.

**Neural interpretation:** hierarchical knowledge and abstraction are associated with interactions across hippocampal, prefrontal, and cortical systems. The exact neural format for nested schemas is not established.

**Output:** a reusable hierarchy of related schemas and component frames.

---

## 13. Bind goals, roles, and expected outcomes

Schemas are not only descriptions of what often happens. They can encode why the event sequence occurs and what participants are trying to achieve.

**Algorithmic operations:**

1. Identify likely goals for each participant or agent.
2. Connect actions to the goals they tend to support.
3. Identify expected outcomes and conditions for success.
4. Represent alternative strategies for reaching the same goal.
5. Use the goal structure to interpret unusual but coherent episodes.

A simple goal-conditioned action value can be modeled as

\[
Q(s,a,g)=\mathbb{E}[R_g+\gamma V(s')\mid s,a,g]
\]

where \(g\) is the goal, \(R_g\) is goal-related reward or progress, and \(\gamma\) discounts future outcomes. This is a reinforcement-learning formalization; it does not assert that human schema representations are literally stored as Q-tables.

**Neural interpretation:** orbitofrontal and medial prefrontal systems are important candidates for representing task structure, goals, and expected outcomes. Hippocampal-prefrontal interactions can help connect current experiences with remembered relational structures.

**Output:** goal-sensitive schema branches that support flexible action and interpretation.

---

## 14. Generate predictions from the schema

A schema is useful because it allows the brain to anticipate likely missing events, roles, or relations.

**Algorithmic operations:**

1. Match the current situation to one or more candidate schemas.
2. Estimate which slots have been filled and which remain unobserved.
3. Predict likely next events and relations using context and goals.
4. Maintain multiple hypotheses when several schemas fit.
5. Treat predictions as expectations, not facts.

The posterior probability of a candidate schema can be written as

\[
P(S\mid O,C)\propto P(O\mid S,C)P(S\mid C)
\]

where \(O\) is the observed evidence and \(C\) is current context.

A next-event prediction is

\[
\hat e_{t+1}=\arg\max_j P(e_{t+1}=j\mid S,O,C)
\]

**Neural interpretation:** schemas can guide retrieval, attention, interpretation, and expectation. Predictive-processing language is a useful theoretical account, but the exact relationship between prediction and neural activity varies across tasks and remains debated.

**Output:** a ranked set of predictions with confidence estimates.

---

## 15. Use schema-based inference to fill gaps—but preserve uncertainty

When information is missing, a schema can support a plausible inference. This is useful, but it can also create false memories or misconceptions if inferred details are later mistaken for observed details.

**Algorithmic operations:**

1. Identify a missing role, relation, or event.
2. Retrieve the candidate schema most consistent with the current context.
3. Generate likely fillers or transitions.
4. Assign each inference a confidence value and an `inferred` provenance label.
5. Keep observed, remembered, and inferred details distinguishable.
6. Request or seek additional evidence when uncertainty matters.

For an unfilled slot \(s\), an inferred filler can be estimated by

\[
\hat f=\arg\max_f P(f\mid s,S,C,O)
\]

Confidence should reflect posterior uncertainty, not just the maximum score. For example, a distribution of 0.45/0.40/0.15 is ambiguous even though one candidate ranks first.

**Neural interpretation:** schema-guided reconstruction is one reason memory can be efficient but fallible. Prior knowledge may shape recall and interpretation; this does not mean every inferred detail becomes a false memory. The distinction between actual neural inference and later conscious source attribution is important.

**Output:** plausible completions explicitly tagged as inferred and accompanied by uncertainty.

---

## 16. Compare new episodes with the schema and compute prediction error

New experience can confirm, refine, or contradict an existing schema.

**Algorithmic operations:**

1. Align the new episode's roles and events with the candidate schema.
2. Compare observed relations and transitions with their predicted probabilities.
3. Calculate a mismatch score.
4. Determine whether the mismatch is noise, a rare exception, a new sub-type, or evidence that the schema is wrong.
5. Preserve the new episode if it is highly distinctive or poorly explained by existing schemas.

A simple prediction error is

\[
\delta_t=o_t-\hat o_t
\]

For categorical events, a negative log-likelihood is more appropriate:

\[
L=-\log P(o_t\mid S,C)
\]

A high surprise value indicates that the observed event was unlikely under the schema. It does not automatically imply that the whole schema must be discarded.

**Neural interpretation:** prediction-error and novelty-related signals can influence learning and memory updating. The hippocampus can encode unexpected or schema-incongruent details, while prefrontal systems may participate in selecting and revising relevant knowledge. The balance depends on the degree and nature of mismatch.

**Output:** a classified mismatch signal with evidence for either integration or separate encoding.

---

## 17. Update, specialize, or split the schema

A flexible schema needs mechanisms for learning from new evidence without being destabilized by every unusual episode.

**Algorithmic operations:**

1. If the new event is strongly schema-consistent, update the relevant filler probabilities or relation strengths.
2. If it is a consistent variation, add an optional branch or specialized sub-schema.
3. If it repeatedly violates a core relation, revise the schema's structure.
4. If it appears to belong to a different situation class, create a separate schema rather than forcing a merge.
5. Retain the original episode trace so later evidence can reverse or refine the update.

A simple incremental update for a relation probability is

\[
\theta_{new}=(1-\eta)\theta_{old}+\eta x
\]

where \(x\) is the new evidence (e.g. 1 if the relation occurred, 0 if it did not, with eligibility handled correctly) and \(\eta\) is a learning rate. A Bayesian count update is often safer when data are sparse:

\[
P(\rho)=\frac{a+N_{support}}{a+b+N_{eligible}}
\]

A split decision can be based on whether a separate model explains the new episode sufficiently better to justify its added complexity.

**Neural interpretation:** schema updating is associated with hippocampal-prefrontal interactions and memory reactivation. Recent theories emphasize a continuum between integrating schema-consistent information and forming more distinct episodic traces for incongruent information. This is not a universal all-or-none rule.

**Output:** an updated schema, a new sub-schema, a split schema, or a preserved independent episode.

---

## 18. Preserve flexibility and guard against schema overreach

A schema becomes harmful when typical patterns are treated as universal rules. Robust schema learning must preserve exceptions and distinguish frequency from necessity.

**Algorithmic operations:**

1. Store evidence counts and source episodes for each relation.
2. Mark relations as required, typical, optional, rare, or uncertain.
3. Use context-sensitive probabilities rather than global assumptions.
4. Keep counterexamples accessible.
5. Lower confidence when evidence is mixed or the environment changes.
6. Re-evaluate a schema when it repeatedly generates incorrect predictions.

A confidence-weighted rule can be represented as

\[
P(\rho\mid C)=\frac{N(\rho,C)+\alpha}{N(eligible,C)+\alpha+\beta}
\]

A context-shift detector can compare the recent distribution with the stored distribution using KL divergence or another distribution-distance measure:

\[
D_{KL}(P_{recent}\Vert P_{stored})=
\sum_x P_{recent}(x)\log\frac{P_{recent}(x)}{P_{stored}(x)}
\]

In practice, smoothing and finite-sample safeguards are required; a large divergence should trigger review, not an automatic rewrite.

**Neural interpretation:** schemas support efficient prediction and generalization but can also bias memory or interpretation. The 2020 review by van Kesteren and Meeter discusses this trade-off between useful structured knowledge and overreliance that can contribute to false memories or misconceptions.

**Output:** a schema with calibrated confidence, exceptions, and an update policy.

---

## 19. Integrated end-to-end pseudocode

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class EpisodeEvent:
    event_type: str
    entities: Dict[str, Any]
    relations: List[Tuple[str, str, str]]
    timestamp: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Episode:
    episode_id: str
    events: List[EpisodeEvent]
    goals: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Slot:
    name: str
    allowed_types: List[str] = field(default_factory=list)
    fillers: List[Any] = field(default_factory=list)
    status: str = "candidate"  # core, typical, optional, uncertain

@dataclass
class Schema:
    name: str
    slots: Dict[str, Slot] = field(default_factory=dict)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    transitions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    goals: List[str] = field(default_factory=list)
    supporting_episode_ids: List[str] = field(default_factory=list)
    exceptions: List[Dict[str, Any]] = field(default_factory=list)
    confidence: Dict[str, float] = field(default_factory=dict)


def segment_episode(raw_experience: Any) -> Episode:
    """Convert a continuous experience into ordered event representations.

    A real implementation would call perceptual, language, and event-boundary
    modules. This function documents the interface, not a biological algorithm.
    """
    raise NotImplementedError


def episode_similarity(a: Episode, b: Episode) -> float:
    """Compare context, goals, event roles, relations, and temporal structure."""
    raise NotImplementedError


def group_related_episodes(episodes: List[Episode], threshold: float):
    """Return candidate episode groups and retain uncertain assignments."""
    raise NotImplementedError


def align_entities_to_roles(group: List[Episode]):
    """Infer recurring functional roles while preserving ambiguous mappings."""
    raise NotImplementedError


def induce_slots(role_alignments) -> Dict[str, Slot]:
    """Turn recurring role classes into variable slots and collect fillers."""
    raise NotImplementedError


def induce_relations(group, role_alignments):
    """Align episode-specific edges to typed schema relations."""
    raise NotImplementedError


def estimate_transitions(group, role_alignments):
    """Estimate context-sensitive event transitions and alternative paths."""
    raise NotImplementedError


def identify_goals_and_outcomes(group):
    """Infer supported goals and outcomes; distinguish inference from observation."""
    raise NotImplementedError


def classify_schema_properties(slots, relations, transitions):
    """Label properties as core, typical, optional, incidental, or uncertain."""
    raise NotImplementedError


def build_schema(name, group) -> Schema:
    role_alignments = align_entities_to_roles(group)
    slots = induce_slots(role_alignments)
    relations = induce_relations(group, role_alignments)
    transitions = estimate_transitions(group, role_alignments)
    goals = identify_goals_and_outcomes(group)
    classify_schema_properties(slots, relations, transitions)

    return Schema(
        name=name,
        slots=slots,
        relations=relations,
        transitions=transitions,
        goals=goals,
        supporting_episode_ids=[ep.episode_id for ep in group],
    )


def predict_missing_structure(schema: Schema, observed: Episode, context: dict):
    """Return ranked predictions with confidence and provenance='inferred'."""
    raise NotImplementedError


def evaluate_new_episode(schema: Schema, episode: Episode):
    """Return mismatch scores and identify which slots/relations were violated."""
    raise NotImplementedError


def update_or_split_schema(schema: Schema, episode: Episode, mismatch):
    """Integrate, specialize, revise, split, or preserve the episode separately."""
    raise NotImplementedError


def learn_schemas(experiences: List[Any]) -> List[Schema]:
    episodes = [segment_episode(x) for x in experiences]
    groups = group_related_episodes(episodes, threshold=0.7)
    schemas = [build_schema(f"schema_{i}", group)
               for i, group in enumerate(groups)]

    for episode in episodes:
        candidates = rank_candidate_schemas(episode, schemas)  # defined by system
        if not candidates:
            schemas.append(build_schema(f"new_schema_{len(schemas)}", [episode]))
            continue

        best = candidates[0]
        mismatch = evaluate_new_episode(best, episode)
        update_or_split_schema(best, episode, mismatch)

    return schemas
```

The pseudocode is deliberately modular. The placeholder functions require explicit implementation choices, training data, uncertainty handling, and evaluation. They are not a claim that a biological brain runs Python-like procedures.

---

## 20. Mathematical operation inventory

| Operation | Purpose in the algorithm | Example formalization |
|---|---|---|
| Event-boundary detection | Divide continuous experience into event units | Weighted change score \(B_t\) |
| Feature binding | Associate entities, attributes, actions, and context | Assignment matrix \(M\) |
| Similarity estimation | Identify related episodes | Weighted similarity or kernel function |
| Probabilistic grouping | Infer shared situation classes | Posterior \(P(z_i=z_j\mid E_i,E_j)\) |
| Role assignment | Map entities to functional positions | \(\arg\max_r Score(e,r)\) |
| Slot induction | Abstract recurring roles into variables | Filler distribution \(P(f\mid s,C)\) |
| Relational alignment | Identify common directed relations | Typed graph matching |
| Frequency estimation | Estimate how typical a relation is | Smoothed proportion / Beta-Binomial model |
| Temporal learning | Estimate event order and alternatives | Conditional transition probability |
| Association analysis | Identify predictive relations | Conditional probability, mutual information |
| Causal evaluation | Avoid treating correlation as cause | Counterfactual/interventional model (when available) |
| Abstraction | Preserve stable and useful structure | Feature weighting, dimensionality reduction |
| Hierarchical composition | Build parent and child schemas | Model selection with complexity penalty |
| Goal-conditioned prediction | Predict actions and outcomes | Expected value / reinforcement-learning model |
| Bayesian inference | Select candidate schema from current evidence | Bayes' rule |
| Prediction error | Detect unexpected information | Residual or negative log-likelihood |
| Incremental update | Refine probabilities with new evidence | Learning-rate or Bayesian count update |
| Context-shift detection | Detect environmental change | KL divergence or related distance |

---

## 21. Biological systems and their cautious functional mapping

| System | Possible contribution to schema formation and use | Important qualification |
|---|---|---|
| Hippocampus and medial-temporal lobe | Bind elements of individual episodes; support relational memory and integration across experiences | Not simply a permanent store for every schema |
| Medial prefrontal cortex | Retrieve and use prior knowledge; emphasize goal-relevant commonalities; support integration and inference | Contributions vary with task and schema familiarity |
| Orbitofrontal / ventromedial prefrontal regions | Represent task structure, latent context, expected outcomes, and structured knowledge | Subregional functions are under active investigation |
| Anterior cingulate cortex | Monitor outcomes, conflict, and changing task demands; may help guide model updating | Not a dedicated schema module |
| Distributed neocortex | Represents content such as objects, people, actions, places, language, and social knowledge | Different schema components are distributed across networks |
| Entorhinal and related medial-temporal networks | Support structured representations and communication with hippocampal circuits | The relationship between cognitive maps, schemas, and abstract structure remains under study |

Recent work emphasizes that schema learning, use, and updating may be continuous and distributed across hippocampal-prefrontal circuits rather than separate operations performed by one centre. A 2026 review discusses structured, low-dimensional population representations as one emerging account, while stressing the importance of new mechanistic evidence.

---

## 22. Validation and falsifiable tests

A computational implementation should be evaluated against behaviour, not merely judged by whether its output looks plausible.

1. **Role generalization:** train on several restaurant episodes with different people and establishments. Test whether the model recognizes the same functional roles when the surface entities change.
2. **Relation preservation:** test whether the model distinguishes `customer orders meal` from `server orders customer`. Correct role relations should generalize beyond word overlap.
3. **Optional-path handling:** include takeaway, buffet, self-service, and prepaid episodes. The model should represent alternatives without treating one path as universal.
4. **Novel filler test:** introduce an unfamiliar payment instrument or service technology. The model should assign it to the appropriate slot when its functional relations support that assignment.
5. **Schema-incongruent event test:** introduce a highly unexpected event. The system should retain it as a distinctive episode or candidate new schema rather than silently forcing it into the old frame.
6. **False-inference test:** omit an event that is typical but not required. The system may predict it, but must mark the prediction as inferred rather than observed.
7. **Context-shift test:** change the environment so earlier transition probabilities no longer apply. The model should reduce confidence and revise its schema.
8. **Ablation test:** disable role alignment, relation extraction, temporal structure, or uncertainty tracking in turn. Measure which generalization and error patterns deteriorate.

Useful metrics include slot-filler accuracy, relation F1, temporal prediction log-loss, calibration error, schema-selection accuracy, false-inference rate, and performance on novel combinations. A biologically motivated model should additionally compare its predictions with neuropsychological, electrophysiological, or neuroimaging findings where available.

---

## 23. Failure modes and safeguards

- **Surface-feature overfitting:** grouping episodes because they share colours, words, or objects while missing their functional relations. *Safeguard:* weight goals and relation structure, not just perceptual similarity.
- **Role confusion:** mapping an entity to a role from its label rather than its function. *Safeguard:* use action and relation evidence and retain ambiguity.
- **Rigid event order:** treating a common sequence as mandatory. *Safeguard:* represent transition probabilities and alternative branches.
- **Correlation-to-causation error:** treating frequently co-occurring events as causally linked. *Safeguard:* label evidential status and require stronger evidence for causal claims.
- **Over-generalization:** applying a schema outside the context in which it was learned. *Safeguard:* condition predictions on situation and goal.
- **False-memory completion:** confusing an expected but unobserved event with a remembered event. *Safeguard:* preserve provenance tags.
- **Premature schema merging:** combining two structures because they share a few slots. *Safeguard:* penalize structural mismatch and retain hierarchical or separate models.
- **Failure to update:** continuing to apply a schema after the environment changes. *Safeguard:* monitor prediction errors and distribution shifts.
- **Overreaction to exceptions:** rebuilding the schema after a single unusual episode. *Safeguard:* accumulate evidence and compare alternative models.

---

## 24. Final end-to-end algorithm summary

1. Receive multiple concrete, context-rich episodes.
2. Segment each episode into ordered events while preserving individual episode identity.
3. Encode entities, attributes, actions, and relations.
4. Group episodes by shared situation, goal, and relational structure.
5. Identify recurring participants and functional roles.
6. Abstract individual entities into variable slots.
7. Extract typed relations among slots.
8. Learn temporal transitions and alternative event paths.
9. Separate associations from supported functional or causal relations.
10. Estimate which features and relations are invariant, typical, optional, or incidental.
11. Construct an integrated graph/frame containing slots, relations, constraints, and goals.
12. Organize related schemas into a hierarchy and reusable components.
13. Use the schema to generate context-sensitive predictions.
14. Fill missing information only as uncertain inference, retaining provenance.
15. Compare new observations with predictions and calculate mismatch.
16. Integrate consistent evidence, specialize recurring variants, or split incompatible structures.
17. Preserve counterexamples, confidence values, and source episodes to prevent rigid overgeneralization.
18. Reuse the revised schema for understanding and guiding future situations.

**Core result:** a schema is a flexible, experience-derived relational model. It represents not only *what entities may appear*, but also *which roles they can fill, how those roles relate, what events may follow, which goals organize the situation, and where exceptions are possible*. This is the algorithmic bridge from a collection of individual memories to structured knowledge that supports inference and generalization.

---

## Selected scientific references

1. Varga, N. L., Morton, N. W., & Preston, A. R. (2024). *Schema, Inference, and Memory*. In **The Oxford Handbook of Human Memory**. Oxford University Press. https://doi.org/10.1093/oxfordhb/9780190917982.013.47
2. Bein, O., & Niv, Y. (2025). Schemas, reinforcement learning and the medial prefrontal cortex. **Nature Reviews Neuroscience, 26**, 141–157. https://doi.org/10.1038/s41583-024-00893-z
3. van Kesteren, M. T. R., & Meeter, M. (2020). How to optimize knowledge construction in the brain. **npj Science of Learning, 5**, 5. https://doi.org/10.1038/s41539-020-0064-y
4. Audrain, S., & McAndrews, M. P. (2022). Schemas provide a scaffold for neocortical integration of new memories over time. **Nature Communications, 13**, 5795. https://doi.org/10.1038/s41467-022-33517-0
5. Maor, I., Zong, W., Gostolupce, D., & Schoenbaum, G. (2026). The Schematic Brain: The Neurophysiology of Behavioral Schemas. **The Neuroscientist**. Published online 20 October 2026. https://doi.org/10.1177/10738584261489262
6. Ghosh, V. E., & Gilboa, A. (2014). What is a memory schema? A historical perspective on current neuroscience literature. **Neuropsychologia, 53**, 104–114. https://doi.org/10.1016/j.neuropsychologia.2013.11.010
7. Igarashi, K. M., Lee, J. Y., & Jun, H. (2022). Reconciling neuronal representations of schema, abstract task structure, and categorization under cognitive maps in the entorhinal-hippocampal-frontal circuits. **Current Opinion in Neurobiology, 76**, 102641. https://doi.org/10.1016/j.conb.2022.102641

**Reference-use note:** the mathematical formulations in this file are candidate computational abstractions inspired by the cited cognitive and neural literature. They should not be interpreted as established literal calculations performed by neurons.
