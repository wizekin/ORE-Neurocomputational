# Point 02 — Connect to Prior Knowledge
## A detailed computational and biologically informed algorithm for integrating new information with existing knowledge

**Source point:** “Connect to prior knowledge: Relate new information to what you already know. Building analogies or mental ‘hooks’ (schemas) integrates the new with existing networks, making retrieval easier later.”

---

## 1. Scope, interpretation, and scientific caution

This document expands the source point into a sequential algorithm for how a human learner may connect incoming information with existing memories, concepts, schemas, and relational knowledge. It is a computational model inspired by cognitive neuroscience, not a claim that the brain literally executes the equations below as written.

The brain does not need to contain explicit variables, symbolic graphs, or a single central “schema lookup” routine. The equations are useful abstractions for describing operations that may be implemented by distributed neural populations, recurrent activity, synaptic strengths, neuromodulatory signals, and interactions among hippocampal, prefrontal, and neocortical systems. The same cognitive function can be implemented by multiple biological mechanisms, and the precise mapping from a mathematical operation to a neural circuit is often not uniquely known.

The central idea is **relational integration**. New information is more useful when it becomes connected to relevant prior knowledge: not merely because two things share a word or appearance, but because the learner detects how the new information relates to already known objects, events, causes, categories, roles, and rules. Those connections can provide context during encoding and later serve as retrieval routes.

A safe algorithm must also handle the opposite case. Prior knowledge can help learning when the new information fits an established schema, but it can bias interpretation or obstruct learning when the new information conflicts with assumptions. Therefore, the algorithm must not blindly force a new observation into the nearest existing schema. It must estimate fit, preserve important contradictions, and allow a new or revised representation to form.

### 1.1 Main goals

Given a new information unit, the algorithm should:

1. Extract its meaningful entities, concepts, attributes, events, and relations.
2. Activate relevant memories using multiple kinds of cues.
3. Select candidate prior knowledge without allowing one familiar association to dominate automatically.
4. Compare the new information with candidates at both surface and relational levels.
5. Identify analogies, shared structures, complementary facts, and contradictions.
6. Decide whether to integrate, modify an existing schema, create a distinct memory, or defer the decision.
7. Bind the new information to useful contextual and retrieval cues.
8. Strengthen the resulting representation through later reactivation and consolidation.
9. Use retrieval success and prediction errors to revise the connections over time.

### 1.2 Input and output

**Inputs**

- `new_input`: a word, sentence, explanation, diagram, event, or larger lesson.
- `current_context`: the topic, task, environment, goal, and currently active working-memory contents.
- `knowledge_network`: previously learned concepts, episodic memories, semantic facts, schemas, causal relations, and their strengths.
- `learner_state`: attention, working-memory availability, motivation, arousal, uncertainty, and fatigue (conceptual variables, not directly measured neural quantities in this algorithm).

**Outputs**

- `integrated_representation`: a representation linked to relevant prior knowledge, where justified.
- `new_episode`: a separate trace for important novel or schema-inconsistent information that should not be prematurely assimilated.
- `schema_updates`: changes to categories, relations, rules, or expectations when evidence supports updating them.
- `retrieval_routes`: cues and associations that can help recover the information later.
- `confidence_and_uncertainty`: estimates of what is well-supported, weakly linked, contradictory, or unresolved.

---

## 2. Conceptual data model

The following structures are computational conveniences. Biological knowledge is distributed and does not have to resemble a computer database.

### 2.1 Concept representation

Represent a concept as a feature bundle or distributed activation pattern:

\[
\mathbf{x}=[x_1,x_2,\ldots,x_n]
\]

Features may encode semantic attributes, sensory properties, actions, functions, contexts, emotional associations, and relations. A value can be a graded activation rather than a binary flag.

### 2.2 Knowledge graph

Represent explicitly retrievable knowledge as a graph:

\[
G=(V,E)
\]

where:

- \(V\) is a set of concept, entity, event, and schema nodes.
- \(E\) is a set of typed relations such as `is_a`, `part_of`, `causes`, `used_for`, `located_in`, `precedes`, `contrasts_with`, or `supports`.
- An edge can have a strength, confidence, recency, context, and evidence count.

A connection can be represented as:

\[
e_{ij}=(v_i,r,v_j,w_{ij},c_{ij})
\]

where \(r\) is the relation type, \(w_{ij}\) is connection strength, and \(c_{ij}\) is confidence or evidential support.

### 2.3 Schema representation

A schema is a structured pattern containing:

- expected entities or roles;
- attributes associated with each role;
- relations among roles;
- typical event order;
- causal or functional constraints;
- allowed variations and exceptions;
- confidence in each component.

A schema should be flexible rather than a rigid template. Some slots can be missing, optional, or filled by several alternatives.

### 2.4 Memory classes

For the algorithm, distinguish three broad forms of memory:

1. **Episodic traces:** representations of particular experiences, including where and when something happened.
2. **Semantic knowledge:** generalized concepts, facts, categories, and relations.
3. **Schemas:** structured expectations that organize multiple related facts or episodes.

These categories interact. A particular episode can be interpreted using a schema, and repeated or consolidated experience can contribute to semantic and schematic knowledge.

---

## 3. Global algorithm overview

```text
ALGORITHM CONNECT_NEW_INFORMATION_TO_PRIOR_KNOWLEDGE

INPUT: new_input, current_context, knowledge_network, learner_state
OUTPUT: integrated_representation, new_episode, schema_updates,
        retrieval_routes, confidence_and_uncertainty

1. Prepare and segment the incoming information.
2. Extract candidate concepts, features, events, and relations.
3. Generate multiple retrieval cues from the new information and its context.
4. Activate and retrieve candidate prior memories.
5. Rank candidates by relevance, reliability, and contextual fit.
6. Compare new information with prior knowledge at several levels.
7. Align relational structures and generate possible analogies.
8. Compute fit, novelty, and contradiction signals.
9. Select an integration strategy without erasing unresolved conflict.
10. Bind the new information to relevant concepts and contextual cues.
11. Update schema strengths and expectations cautiously.
12. Encode unresolved novelty as a distinct episodic trace when needed.
13. Revisit and consolidate connections over time.
14. Test retrieval and use the result to refine the knowledge network.

RETURN all outputs and uncertainty labels.
```

The steps below expand this process in detail.

---

# 4. Detailed algorithmic steps

## Step 1 — Prepare the incoming information and define the learning context

### Purpose

Before linking anything to memory, the learner must establish what is being learned and under what circumstances. The same sentence can activate different prior knowledge depending on the surrounding topic, the learner's goal, and the question being answered.

For example, the word “cell” can activate biological knowledge in a biology lesson, spreadsheet knowledge in a spreadsheet tutorial, or prison-related knowledge in a different context. The word alone is insufficient to determine the intended concept.

### Algorithmic procedure

1. Receive the incoming sensory or symbolic information.
2. Identify its modality: speech, text, image, diagram, demonstration, or a combination.
3. Estimate the current task: understanding a definition, explaining a mechanism, solving a problem, learning a procedure, or remembering an event.
4. Collect context from the immediately preceding material, the broader topic, and the learner's stated goal.
5. Identify what is already active in working memory and what may be relevant to the current item.
6. Estimate whether the learner has enough attentional and working-memory capacity to process the item. If the input is too dense, retain the main structure and process smaller segments rather than attempting to integrate everything at once.
7. Create a provisional context representation. Do not treat it as final; later words or evidence may change it.

### Possible mathematical operations

**Context feature construction**

Represent the context as a vector:

\[
\mathbf{c}_t=[c_1,c_2,\ldots,c_m]
\]

Each dimension may represent a topic, goal, task, setting, recently active concept, or discourse relation.

**Weighted integration of context cues**

\[
C_j=\sum_{k=1}^{m} a_k f_{kj}
\]

where \(f_{kj}\) is the evidence that context cue \(k\) supports context interpretation \(j\), and \(a_k\) represents cue relevance or reliability.

**Context uncertainty**

If several contexts are possible, maintain a probability distribution rather than selecting one immediately:

\[
P(C_j\mid X,\text{goal})
=\frac{P(X\mid C_j,\text{goal})P(C_j\mid\text{goal})}
{\sum_l P(X\mid C_l,\text{goal})P(C_l\mid\text{goal})}
\]

This is a computational description of context-sensitive interpretation, not evidence that neurons explicitly calculate Bayes' rule.

**Capacity gating**

A simplified processing gate can be written as:

\[
g=\sigma(\alpha A+\beta W-\gamma L)
\]

where \(A\) is attention availability, \(W\) is available working-memory capacity, \(L\) is estimated input load, and \(\sigma\) is a bounded activation function. A low gate suggests segmenting or slowing the input.

### Possible biological implementation

Sensory and language-processing systems provide the incoming representation, while attention and working-memory systems maintain task-relevant information. Prefrontal systems are involved in goal maintenance and control; temporal and parietal cortical systems contribute stored conceptual and contextual knowledge. These functions are distributed and interactive rather than a single linear pipeline.

### Output

`context_frame`: a provisional representation of the topic, goal, modality, active concepts, and processing constraints.

---

## Step 2 — Extract meaningful concepts, features, and relations

### Purpose

Prior knowledge cannot be retrieved effectively from an undifferentiated stream. The incoming material must be represented in terms of meaningful components. These components include not only nouns or labels but also properties, actions, causal relations, spatial relations, temporal order, and functional roles.

### Algorithmic procedure

1. Segment the input into meaningful units.
2. Identify candidate entities and concepts.
3. Extract attributes that distinguish each concept.
4. Identify events and actions, including who or what performs them and what is affected.
5. Detect relations among concepts, including causal, hierarchical, temporal, spatial, functional, and contrastive relations.
6. Preserve negation, uncertainty, and conditional language. “X may cause Y” must not be represented as the unconditional fact “X causes Y.”
7. Assign confidence to each extracted element, especially when the input is ambiguous.
8. Keep the original input linked to the extracted representation so later reinterpretation remains possible.

### Possible mathematical operations

**Feature activation**

\[
\mathbf{x}= [x_1,x_2,\ldots,x_n],\quad 0\leq x_i\leq 1
\]

Feature values represent degrees of support or activation, not necessarily literal firing rates.

**Feature weighting**

\[
\tilde{x}_i=w_i x_i
\]

where \(w_i\) reflects the feature's current relevance to the learning task. A feature's importance is context-dependent; a superficial feature may be relevant in one task and irrelevant in another.

**Relation extraction as structured tuples**

\[
r_k=(\text{subject},\text{relation},\text{object},\text{qualifiers})
\]

For example, “the hippocampus helps bind elements of an episode” can be represented as an agent/system, a functional relation, and an affected process, while retaining the qualifier “helps” rather than converting it to an exclusive responsibility.

**Confidence aggregation**

If several cues support a feature, a simple noisy-OR approximation is:

\[
P(f)=1-\prod_{i=1}^{n}(1-p_i)
\]

This can model combined evidence under simplifying independence assumptions. Real neural and cognitive evidence is often correlated, so the independence assumption must not be taken literally.

### Possible biological implementation

Distributed cortical activity can represent conceptual features and relations. Language-related systems help derive meaning from words and sentences; temporal, parietal, and frontal networks contribute different aspects of semantic retrieval, selection, and relational processing. The exact division of labor depends on the task and representation.

### Output

`new_representation`: a structured, provisional representation containing entities, properties, actions, relations, qualifiers, and confidence values.

---

## Step 3 — Generate multiple retrieval cues

### Purpose

A new concept may connect to prior knowledge through several routes. A learner may retrieve something because it shares a category, a function, a cause, a context, a sensory feature, a word, or a relation. Using several cues makes retrieval more robust and reduces dependence on one obvious association.

### Algorithmic procedure

1. Generate a cue from each important entity or concept.
2. Generate additional cues from distinctive attributes and functions.
3. Generate relational cues, such as “something that transfers energy,” “a structure that stores information,” or “an event that occurs after a trigger.”
4. Generate contextual cues from the topic and goal.
5. Generate contrast cues by identifying what the new item is not, or what it differs from.
6. Give higher priority to distinctive, reliable, and task-relevant cues.
7. Avoid treating generic words or common features as decisive retrieval signals.
8. Preserve several cues for parallel or successive retrieval attempts.

### Possible mathematical operations

**Cue vector construction**

Let the cue set be:

\[
Q=\{q_1,q_2,\ldots,q_k\}
\]

Each cue may have a salience and relevance weight \(s_i\).

**Cue utility**

\[
U(q_i)=s_i\times r_i\times d_i
\]

where \(r_i\) is task relevance and \(d_i\) is cue distinctiveness. This is a heuristic scoring model, not an identified neural formula.

**Cue combination**

A candidate memory's activation can be approximated by:

\[
A(M_j)=b_j+\sum_i w_{ij}q_i+\sum_{i<l}\eta_{ilj}q_iq_l
\]

where \(b_j\) is baseline activation, \(w_{ij}\) is the cue-to-memory association, and the interaction term represents the possibility that two cues together are more informative than either alone.

**Normalization**

To compare cue weights on a common scale:

\[
\hat{s}_i=\frac{s_i}{\sum_l s_l+\epsilon}
\]

where \(\epsilon\) prevents division by zero.

### Possible biological implementation

A cue can reactivate overlapping cortical representations. Recurrent connectivity allows activation to spread through associated networks. The hippocampus can help reinstate patterns of activity related to a particular episode or association. Multiple cues may converge on overlapping representations, increasing the chance of recovering relevant knowledge.

### Output

`retrieval_cues`: a weighted set of semantic, relational, contextual, perceptual, and contrastive cues.

---

## Step 4 — Activate and retrieve candidate prior memories

### Purpose

The cue set initiates access to existing knowledge. Retrieval should be broad enough to find useful candidates but constrained enough to avoid flooding working memory with unrelated memories.

### Algorithmic procedure

1. Use each cue to activate associated concepts, facts, episodes, and schemas.
2. Allow activation to spread through connected representations, with stronger or more relevant connections contributing more.
3. Limit the spread by relevance, association strength, and processing resources.
4. Retrieve candidates from more than one memory type when appropriate: semantic facts, specific past episodes, and abstract schemas.
5. Keep the context attached to episodic memories so a fact learned in one situation is not automatically generalized to every situation.
6. Combine candidates retrieved by different cues, but preserve the cue that activated each candidate.
7. Maintain a manageable candidate set for comparison.

### Possible mathematical operations

**Spreading activation**

For a network with activation vector \(\mathbf{a}^{(t)}\) and weighted adjacency matrix \(W\):

\[
\mathbf{a}^{(t+1)}=\phi(W\mathbf{a}^{(t)}+\mathbf{u})
\]

where \(\mathbf{u}\) is new cue input and \(\phi\) is a nonlinear activation function. Repeated application can approximate recurrent activation spreading.

**Decay and relevance control**

\[
\mathbf{a}^{(t+1)}=\phi(\lambda W\mathbf{a}^{(t)}+\mathbf{u})
\]

with \(0<\lambda<1\) in this simplified model to prevent unlimited propagation. Biological networks use many interacting mechanisms; this scalar decay is only a computational approximation.

**Candidate activation accumulation**

\[
A_j=\sum_i q_i w_{ij}
\]

where candidate \(j\) receives support from several retrieval cues.

**Competition for limited access**

\[
P(M_j\mid Q)=\frac{e^{A_j/T}}{\sum_l e^{A_l/T}}
\]

A lower temperature \(T\) yields a sharper distribution; a higher temperature retains more alternatives. This softmax is a model for graded competition, not a literal neural instruction.

### Possible biological implementation

The hippocampal formation, medial temporal lobe, prefrontal cortex, and distributed neocortical networks interact during memory retrieval. Pattern completion is one useful computational description: partial cues can help reinstate a broader representation. Prefrontal control can bias retrieval toward task-relevant information and suppress irrelevant competitors.

### Output

`candidate_memories`: a set of potentially relevant memories with activation estimates, memory type, context, and source cues.

---

## Step 5 — Evaluate and rank candidate knowledge

### Purpose

The most strongly activated memory is not necessarily the most useful or correct. A highly familiar but irrelevant concept can dominate retrieval. Candidate knowledge must therefore be evaluated using relevance, reliability, contextual compatibility, specificity, and evidence quality.

### Algorithmic procedure

1. Estimate how directly each candidate relates to the new information.
2. Estimate whether the candidate belongs to the same task or context.
3. Distinguish a direct factual connection from a loose association.
4. Estimate reliability using prior evidence, repeated confirmation, source quality, and known exceptions.
5. Penalize candidates supported only by generic or superficial overlap.
6. Retain candidates that offer complementary information even if their vocabulary differs.
7. Keep a small set of plausible alternatives where evidence is close.
8. Do not convert retrieval strength into truth. Familiarity and correctness are separate dimensions.

### Possible mathematical operations

**Multi-factor relevance score**

\[
R_j=\alpha S_j+\beta K_j+\gamma C_j+\delta E_j-\mu N_j
\]

where:

- \(S_j\): semantic similarity;
- \(K_j\): relational or structural similarity;
- \(C_j\): contextual fit;
- \(E_j\): evidential reliability;
- \(N_j\): mismatch or irrelevance penalty;
- coefficients determine task-dependent importance.

**Cosine similarity**

\[
\operatorname{sim}(\mathbf{x},\mathbf{m})=
\frac{\mathbf{x}\cdot\mathbf{m}}
{\|\mathbf{x}\|\|\mathbf{m}\|+\epsilon}
\]

This compares distributed feature patterns. It captures one kind of similarity but cannot establish that two items share the same causal or relational structure.

**Reliability update**

A simple evidence-strength estimate can accumulate support and counterevidence:

\[
E_j=\sum_i \rho_i e_{ij}-\sum_k \kappa_k c_{kj}
\]

where \(e_{ij}\) and \(c_{kj}\) represent supporting and conflicting evidence, and \(\rho_i,\kappa_k\) weight their quality.

**Rank normalization**

\[
P_j=\frac{e^{R_j/T}}{\sum_l e^{R_l/T}}
\]

The resulting values are relative candidate weights, not objective probabilities that a memory is true.

### Possible biological implementation

Prefrontal control and attentional mechanisms may bias which stored representations become accessible. Familiarity, semantic association, and episodic reinstatement can contribute differently. The separation between activation, relevance, and truth is important because memory retrieval can be biased by expectations and existing beliefs.

### Output

`ranked_candidates`: a ranked but non-exclusive set of prior knowledge candidates, each with relevance, confidence, and provenance information.

---

## Step 6 — Compare the new information with prior knowledge at multiple levels

### Purpose

A learner should compare more than the words or surface appearance. Two things may look different but share a useful structure; two things may look alike but perform different functions. This step separates several comparison levels.

### Algorithmic procedure

1. Compare category membership: do the new and known items belong to the same broad class?
2. Compare features: which attributes overlap and which differ?
3. Compare function: do they serve a similar purpose?
4. Compare relations: do their components stand in corresponding relationships?
5. Compare causal structure: do similar conditions produce similar outcomes?
6. Compare temporal or spatial organization where relevant.
7. Identify complementary links, such as a new detail that fills a missing slot in an established schema.
8. Identify contradictions and exceptions instead of averaging them away.
9. Mark which level supports the connection: surface, semantic, functional, relational, causal, or contextual.

### Possible mathematical operations

**Feature overlap**

For binary or graded feature sets, one option is Jaccard similarity:

\[
J(A,B)=\frac{|A\cap B|}{|A\cup B|+\epsilon}
\]

For graded vectors, cosine similarity or a weighted distance can be used.

**Weighted distance**

\[
D(\mathbf{x},\mathbf{m})=
\sqrt{\sum_i w_i(x_i-m_i)^2}
\]

The weights allow task-relevant features to matter more than irrelevant ones.

**Relational matching score**

\[
K=\sum_{r\in\mathcal{R}}\omega_r\,\operatorname{Match}(r_{new},r_{old})
\]

where each relation is compared by type, direction, participants, and qualifiers.

**Complementarity score**

\[
C_{fill}=\sum_s \nu_s\,\mathbf{1}(\text{new information fills schema slot }s)
\]

A new detail can be useful even when it is not highly similar to a known fact, provided it fills an expected but previously missing role.

**Contradiction score**

\[
X=\sum_i \chi_i\,\mathbf{1}(\text{new constraint conflicts with prior constraint }i)
\]

The conflict should be weighted by the confidence and importance of the affected prior constraint.

### Possible biological implementation

Distributed semantic representations allow partial feature overlap, while prefrontal and parietal systems support relational reasoning and controlled comparison. The hippocampus can contribute to the retrieval and binding of specific associations. These are broad functional descriptions; no single region performs all comparison operations alone.

### Output

`comparison_profile`: a structured account of shared features, shared relations, complementary details, mismatches, and contradictions.

---

## Step 7 — Align relational structures and generate analogies

### Purpose

An analogy is most useful when it transfers a relationship or system of relationships, not merely a superficial resemblance. Connecting new information to prior knowledge can be especially powerful when a familiar domain provides a structure for understanding an unfamiliar one.

For example, describing electrical current as water flowing through a pipe can provide an initial relational scaffold. However, the analogy has limits: electrical charge, voltage, resistance, and water pressure are not identical phenomena. A useful algorithm must record both correspondences and boundaries.

### Algorithmic procedure

1. Select the most promising prior candidates from Step 5.
2. Represent the new domain and candidate familiar domain as relational graphs.
3. Identify possible role correspondences between entities in the two graphs.
4. Prefer mappings that preserve relations across several components.
5. Distinguish relational similarity from shared surface features.
6. Identify where the mapping breaks down.
7. Generate tentative inferences from the familiar structure to the new one.
8. Label each inference as supported, plausible, or speculative.
9. Check each transferred inference against the original information and other evidence.
10. Keep the analogy as a learning aid, not as proof that the two domains are identical.

### Possible mathematical operations

**Graph representation**

Let the new and familiar structures be:

\[
G_N=(V_N,E_N),\qquad G_F=(V_F,E_F)
\]

**Node mapping**

A candidate mapping \(f:V_N\rightarrow V_F\) associates roles in the new domain with roles in the familiar domain.

**Structure-preservation score**

\[
S_{map}(f)=\sum_{(u,r,v)\in E_N}
\omega_r\,\operatorname{Match}\big((f(u),r,f(v)),E_F\big)
\]

A mapping is stronger when it preserves many important relations.

**Mapping cost**

\[
L(f)=\lambda_1 U(f)+\lambda_2 I(f)+\lambda_3 O(f)
\]

where \(U\) is the number or weight of unmapped important elements, \(I\) is the number or weight of inconsistent mappings, and \(O\) is the cost of over-extending the analogy.

**Analogy quality**

\[
Q_{analogy}=S_{map}-L(f)
\]

**Inference transfer**

For a possible transferred relation \(r\):

\[
P(r_{new}\mid f,\text{evidence})\propto
P(\text{evidence}\mid r_{new},f)P(r_{new}\mid f)
\]

This expresses that analogy can suggest a hypothesis, but the hypothesis still requires evidence.

### Possible biological implementation

Analogical reasoning recruits a distributed network involving prefrontal and parietal areas, with rostral prefrontal regions implicated in relational integration in several neuroimaging studies. Retrieval of familiar source knowledge can involve temporal and medial temporal memory systems. The exact neural implementation depends on analogy complexity and task demands.

### Output

`analogy_map`: correspondences, preserved relations, transferred hypotheses, and explicit limits of the analogy.

---

## Step 8 — Estimate schema fit, novelty, and prediction error

### Purpose

The learner must decide whether the new information is consistent with an existing schema, merely incomplete, genuinely novel, or in conflict with prior knowledge. This decision governs whether rapid integration is appropriate or a distinct memory should be preserved.

A key principle is that both consistency and inconsistency can make information memorable, but they may do so through different mechanisms. Consistent information may benefit from an existing structure. Unexpected information may attract attention and be encoded as a distinctive event. An algorithm that treats mismatch only as noise will lose important learning opportunities.

### Algorithmic procedure

1. Compare the new information with the best-matching schema.
2. Check whether the expected roles and relations are present.
3. Check whether a missing detail is simply unspecified or actually contradictory.
4. Estimate how surprising the input is under the current schema.
5. Separate harmless variation from a contradiction that changes the schema's predictions.
6. Consider whether the schema itself is reliable and sufficiently general.
7. If fit is high, prepare to integrate while retaining the episode-specific details that matter.
8. If fit is moderate or uncertain, retain the new information and candidate schema as separate but linked representations pending more evidence.
9. If conflict is strong, preserve a distinct trace and flag the schema for review instead of forcing assimilation.

### Possible mathematical operations

**Schema compatibility**

\[
F(S,X)=\sum_i w_i\operatorname{Compat}(x_i,s_i)
-\sum_j v_j\operatorname{Conflict}(x_j,s_j)
\]

**Prediction error**

\[
\delta_t=x_t-\hat{x}_t
\]

For categorical or structured information, this difference is better understood as a mismatch measure rather than literal subtraction.

**Surprisal**

\[
I(X)=-\log P(X\mid S,C)
\]

An event with low predicted probability has high surprisal. High surprisal does not automatically mean that it is important or true.

**Information-theoretic divergence**

If a schema predicts a distribution \(P\) and the new evidence supports \(Q\), their discrepancy can be approximated using KL divergence:

\[
D_{KL}(Q\|P)=\sum_x Q(x)\log\frac{Q(x)}{P(x)}
\]

This is a computational analogy for the difference between expected and observed patterns. The brain is not known to calculate this exact expression in a single circuit.

**Novelty-fit decision**

\[
Z=\alpha F-\beta I+\gamma N
\]

where \(F\) is schema fit, \(I\) is contradiction, and \(N\) is novelty or distinctiveness. Thresholds should be adaptive and task-dependent.

### Possible biological implementation

Medial prefrontal systems are implicated in representing and using schematic knowledge; hippocampal and broader medial temporal systems are important for encoding distinctive episodes and associations. Research suggests that the interaction between these systems varies with the degree of schema congruence and novelty. Neuromodulatory systems, including dopamine-related pathways, may contribute to novelty and learning signals, but the detailed causal pathways remain an active research area.

### Output

`fit_state`: one of `schema_consistent`, `schema_complementary`, `ambiguous_or_partial_fit`, `schema_conflict`, or `high_novelty`, with supporting evidence and uncertainty.

---

## Step 9 — Choose the appropriate integration strategy

### Purpose

Not every new fact should be integrated in the same way. A schema-consistent detail may be attached to an existing structure. A new relationship may require modifying a schema. A surprising fact may need an independent episodic representation. An ambiguous fact may need to remain unresolved.

### Algorithmic procedure

1. Read the fit state and confidence from Step 8.
2. If the information is consistent and independently supported, link it to the relevant schema and preserve its specific source when useful.
3. If it fills a missing slot, add the detail without changing unrelated parts of the schema.
4. If it changes a relation or rule, create a candidate schema revision rather than overwriting the old schema immediately.
5. If it strongly conflicts with prior knowledge, encode it as a distinct episode and mark the conflict explicitly.
6. If evidence is weak, defer broad generalization and retain a provisional connection.
7. If multiple schemas explain the information, preserve competing interpretations until context or new evidence discriminates between them.
8. Record why the selected strategy was chosen.

### Possible mathematical operations

**Strategy selection as expected utility**

For each strategy \(a\):

\[
EU(a)=\sum_o P(o\mid a,X)U(o)-Cost(a)
\]

Possible strategies include assimilate, modify, store separately, or defer. The utility function rewards learning and future retrieval while penalizing distortion and unnecessary complexity.

**Thresholded decision rule**

\[
\text{Strategy}=
\begin{cases}
\text{Integrate}, & F\geq\theta_F,\ I<\theta_I,\ E\geq\theta_E\\
\text{Complement schema}, & \text{new detail fills a valid missing slot}\\
\text{Revise candidate schema}, & I\geq\theta_I\text{ and evidence is strong}\\
\text{Store separately}, & I\text{ is high or novelty is high}\\
\text{Defer}, & \text{confidence is insufficient}
\end{cases}
\]

**Learning-rate control**

\[
\eta=\eta_0\,g(E,N,I)
\]

where the update rate depends on evidence quality, novelty, and conflict. A high conflict should not automatically produce a large schema change; high-quality evidence is required.

### Possible biological implementation

The brain is thought to use multiple memory systems and pathways rather than one explicit decision gate. Schema congruence can facilitate integration, while distinctive or conflicting information may recruit hippocampal episodic encoding. Prefrontal monitoring and control can help maintain competing interpretations and guide strategic learning.

### Output

`integration_plan`: a documented action specifying what to link, what to revise, what to store separately, and what to leave unresolved.

---

## Step 10 — Bind new information to existing concepts and contextual cues

### Purpose

Once an appropriate strategy is chosen, the learner must establish useful associations. The goal is not just to remember the new item and the old item independently, but to build links that can later support retrieval and inference.

### Algorithmic procedure

1. Identify the concepts and relations that should be linked.
2. Bind each new element to its relevant role in the schema.
3. Connect it to the context in which it was learned.
4. Connect it to the reason it matters, such as its function, cause, consequence, or practical application.
5. Preserve source and confidence information where the distinction matters.
6. Link both directions when appropriate: new-to-old and old-to-new.
7. Avoid linking every co-occurring item indiscriminately; prioritize meaningful and supported relations.
8. Add retrieval cues such as a question, example, diagram, analogy, or contrast.
9. Preserve a pointer to the original episode when a generalized fact was extracted from a particular event.

### Possible mathematical operations

**Associative weight update**

A simplified Hebbian-style update is:

\[
\Delta w_{ij}=\eta a_i a_j
\]

where \(a_i\) and \(a_j\) represent co-activation and \(\eta\) is a learning rate. This is a broad learning principle, not a complete model of synaptic plasticity.

**Error-sensitive update**

\[
\Delta w_{ij}=\eta\,e_i\,a_j
\]

where \(e_i\) is a teaching, prediction, or outcome-related signal. The appropriate biological learning rule depends on circuit and timescale.

**Bounded weight update**

\[
w_{ij}^{new}=\operatorname{clip}(w_{ij}^{old}+\Delta w_{ij},w_{min},w_{max})
\]

This prevents unbounded weights in a computational implementation. Biological synaptic changes are regulated through more complex mechanisms.

**Associative network update**

\[
W_{new}=W_{old}+\eta\,\mathbf{a}\mathbf{a}^{T}
\]

This outer-product form illustrates how co-activation can strengthen distributed associations. It is a simplified learning model and does not capture the full diversity of plasticity rules.

**Contextual binding**

\[
M=(X,C,T,G,R)
\]

where \(X\) is content, \(C\) context, \(T\) temporal information, \(G\) goal relevance, and \(R\) relations to other knowledge. The representation is a model of what may need to be associated, not a claim about a single memory engram format.

### Possible biological implementation

Hippocampal circuits are strongly implicated in binding items and their relations within episodes. Neocortical networks represent distributed semantic content. Interaction between hippocampus and cortex can support linking new information to existing knowledge. Synaptic plasticity, recurrent network activity, and coordinated neural activity are candidate mechanisms for strengthening associations.

### Output

`bound_memory`: new information linked to selected prior concepts, relations, contexts, and retrieval cues.

---

## Step 11 — Update the schema cautiously and preserve exceptions

### Purpose

Learning does not stop after attaching a new fact. Repeated, reliable information can refine an existing schema. However, one unusual example should not necessarily change a general rule. The learner must decide whether the new information is an isolated exception, a context-specific variation, or evidence that the schema itself is incomplete or wrong.

### Algorithmic procedure

1. Identify which schema components are directly affected by the new information.
2. Estimate whether the evidence is independent and reliable.
3. Compare the new evidence with prior examples and known exceptions.
4. Update only the affected attributes or relations.
5. Preserve previous schema versions or confidence histories when the change is substantial.
6. Represent exceptions explicitly rather than hiding them inside an average.
7. Re-evaluate predictions that depended on the changed component.
8. Avoid overgeneralizing from one episode.
9. If repeated evidence supports a new regularity, gradually strengthen the updated relation.

### Possible mathematical operations

**Incremental estimate**

For a parameter \(\theta\) estimated from observations:

\[
\theta_{new}=(1-\eta)\theta_{old}+\eta x_{new}
\]

The update rate \(\eta\) should depend on evidence quality, sample size, reliability, and context. A single observation should usually have limited influence on a broad schema.

**Precision-weighted update**

\[
\theta_{new}=\frac{\tau_{old}\theta_{old}+\tau_{new}x_{new}}
{\tau_{old}+\tau_{new}}
\]

where \(\tau\) represents precision or inverse uncertainty. This captures the idea that more reliable evidence should exert more influence.

**Confidence accumulation**

For repeated independent evidence, odds can be updated as:

\[
\text{Posterior odds}=\text{Prior odds}\times\text{Likelihood ratio}
\]

This is a Bayesian description. Real learning evidence is rarely perfectly independent, so repeatedly encountering the same source should not be counted as multiple independent confirmations.

**Exception tracking**

\[
S=\{r_{general},e_1,e_2,\ldots,e_k\}
\]

The schema stores a general rule and a set of qualified exceptions. A context condition can specify where the exception applies.

### Possible biological implementation

Schema modification may involve coordinated changes across prefrontal and cortical networks, with hippocampal learning contributing new episodes and relations. Repeated co-activation and offline reactivation can support longer-term changes. The exact timing and route of integration vary; not all new memories become generalized semantic knowledge in the same way.

### Output

`updated_schema_candidate`: a refined schema with updated confidence, preserved exceptions, and a record of what changed.

---

## Step 12 — Encode unresolved or surprising information as a distinct episode

### Purpose

When new information strongly conflicts with a schema, it may be dangerous to assimilate it immediately. The learner needs a representation that preserves what happened, the context, and the discrepancy. This helps prevent prior knowledge from erasing a surprising fact.

### Algorithmic procedure

1. Detect whether schema conflict or novelty exceeds the integration threshold.
2. Preserve the exact new observation as accurately as possible.
3. Bind it to its source, time, place, task, and surrounding context when available.
4. Record the schema expectation it violated.
5. Store multiple plausible explanations if the cause is unclear.
6. Avoid revising the schema until sufficient corroboration exists.
7. Allow later retrieval to compare the episode with the schema and with other episodes.
8. If repeated evidence supports the surprising observation, consider promoting it into semantic knowledge or a revised schema.

### Possible mathematical operations

**Distinctiveness estimate**

\[
D(X)=\alpha I(X)+\beta U(X)+\gamma C_{new}
\]

where \(I\) is surprisal, \(U\) is uniqueness, and \(C_{new}\) is the amount of new contextual information.

**Episode representation**

\[
E_t=(X_t,Context_t,Time_t,Source_t,Expectation_t,Error_t)
\]

**Separate-trace decision**

\[
\text{StoreEpisode}=\mathbf{1}(I>\theta_I\ \lor\ X>\theta_X)
\]

The indicator is a simplified decision rule. Actual memory formation is graded and affected by attention, emotional significance, stress, sleep, and other factors.

**Later integration probability**

\[
P(\text{integrate later})=\sigma(\alpha R+\beta C+\gamma V-\delta U)
\]

where \(R\) is repetition or recurrence, \(C\) is corroboration, \(V\) is explanatory value, and \(U\) is unresolved uncertainty.

### Possible biological implementation

The hippocampus and medial temporal lobe are important for encoding distinctive relational and episodic information. Novelty-related neuromodulatory influences may affect encoding, while interactions with prefrontal systems help relate episodes to established knowledge. Strong emotional arousal can change memory processing, but emotional intensity does not guarantee accuracy.

### Output

`new_episode`: a distinct trace for novel or conflicting information, retaining context and the unresolved relation to prior knowledge.

---

## Step 13 — Consolidate and reorganize the connections over time

### Purpose

The connection formed during initial learning may be weak, incomplete, or dependent on active hippocampal support. With time, reactivation, sleep, and additional learning, some information may become integrated into broader cortical knowledge networks. This should be treated as a gradual and variable process, not a guaranteed fixed transfer from one region to another.

### Algorithmic procedure

1. Retain the newly encoded representation and its links.
2. During later reactivation, reinstate relevant parts of the new memory and associated prior knowledge.
3. Replay or reactivate relevant relations in the context of the broader knowledge network.
4. Strengthen connections that repeatedly prove useful and well-supported.
5. Weaken or revise connections that consistently lead to incorrect predictions.
6. Extract stable common structure across multiple episodes while preserving meaningful details and exceptions.
7. Reorganize retrieval routes so the concept can be reached from more than one cue.
8. Reassess the schema after new learning and after intervals of sleep or rest.

### Possible mathematical operations

**Reactivation-weighted strengthening**

\[
\Delta w_{ij}=\eta r_{ij}c_{ij}
\]

where \(r_{ij}\) is reactivation and \(c_{ij}\) is confidence or usefulness of the connection.

**Slow integration**

\[
W_{cortex}(t+1)=W_{cortex}(t)+\eta_s\,R_t
\]

where \(R_t\) is reactivated information and \(\eta_s\) is a slower update rate. This is a conceptual description of gradual cortical integration, not a literal transfer of a complete memory file.

**Generalization across episodes**

For a set of related examples \(X_1,\ldots,X_n\), an abstract representation can be modeled as:

\[
\mathbf{g}=\frac{\sum_i \omega_i\mathbf{x}_i}{\sum_i\omega_i}
\]

with weights that account for reliability and relevance. In practice, abstraction also requires preserving relational structure and detecting exceptions; a simple arithmetic mean can erase important distinctions.

**Connection decay or pruning**

\[
w_{ij}(t+1)=\rho w_{ij}(t)+\Delta w_{ij}
\]

where \(0<\rho\leq1\) is a conceptual decay factor. Real biological maintenance and weakening are more complex than uniform decay.

### Possible biological implementation

Systems consolidation theories emphasize interactions among hippocampal and neocortical systems over time. Sleep-associated reactivation and replay are proposed to contribute to memory stabilization and integration, but the degree and form of integration depend on memory type, prior knowledge, and task. Medial prefrontal and distributed cortical networks are implicated in schema-related consolidation.

### Output

`consolidated_network`: a more stable set of links, concepts, and schema relations, with episode-specific details retained when necessary.

---

## Step 14 — Test retrieval and refine the knowledge network

### Purpose

A connection is useful only if it can support later understanding, recall, or application. The learner should attempt to retrieve the new information using different cues and check whether the expected relations can be reconstructed. Retrieval success can strengthen access routes; retrieval failure can reveal a weak or misleading connection.

### Algorithmic procedure

1. Present a retrieval cue without immediately showing the answer.
2. Attempt to reconstruct the new concept from a prior concept, context, example, or relation.
3. Attempt retrieval in the reverse direction: use the new concept to recover the prior knowledge it connects to.
4. Check whether the correct relations are recovered, not merely whether familiar words come to mind.
5. Compare recalled content with the original evidence.
6. Identify omissions, distortions, and false inferences.
7. Correct the representation and strengthen accurate routes.
8. If the learner can retrieve the link only with one narrow cue, add or practice alternative cues.
9. Re-test after a delay and in a different context.
10. Feed test results back into confidence, schema fit, and future retrieval weights.

### Possible mathematical operations

**Retrieval success**

\[
S_{ret}=\frac{\text{accurately recovered required elements}}
{\text{total required elements}+\epsilon}
\]

A richer score should separately assess facts, relations, context, and confidence calibration.

**Retrieval error**

\[
E_{ret}=\sum_i w_i\,\operatorname{Mismatch}(y_i,\hat{y}_i)
\]

where \(y_i\) is the target relation and \(\hat{y}_i\) is the retrieved relation.

**Practice-driven association update**

\[
w_{ij}^{new}=w_{ij}^{old}+\eta\,S_{ret}\,R_i
\]

where \(R_i\) represents successful reactivation of a retrieval route. Incorrect retrieval should trigger correction, not simply strengthen the erroneous response.

**Calibration error**

\[
E_{cal}=|Conf_{reported}-Acc_{observed}|
\]

This helps distinguish genuine knowledge from a strong feeling of familiarity.

**Cue diversity**

If a memory can be retrieved through multiple independent cues, access may be more robust. One possible diversity measure is cue entropy:

\[
H(Q)=-\sum_i p(q_i)\log p(q_i)
\]

The goal is not to maximize entropy indiscriminately, but to avoid dependence on a single fragile route.

### Possible biological implementation

Retrieval can reactivate cortical representations and hippocampal associations. Successful retrieval may itself alter and strengthen memory, while reactivation can also introduce distortion. Feedback, prediction error, and later re-encoding can help correct mistakes. Repeated retrieval is therefore not simply passive reading in reverse; it can change memory accessibility and organization.

### Output

`retrieval_routes`: tested cues and connections, plus an error report indicating which links need reinforcement or correction.

---

# 5. Integrated algorithm with control flow

The full process is iterative. New information can cause the learner to revisit an earlier interpretation, retrieve a different schema, or maintain multiple hypotheses. The pseudocode below gives an implementation-level summary.

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FitState(Enum):
    CONSISTENT = "schema_consistent"
    COMPLEMENTARY = "schema_complementary"
    AMBIGUOUS = "ambiguous_or_partial_fit"
    CONFLICT = "schema_conflict"
    NOVEL = "high_novelty"


class Strategy(Enum):
    INTEGRATE = "integrate"
    COMPLEMENT = "complement_schema"
    REVISE_CANDIDATE = "revise_candidate_schema"
    STORE_EPISODE = "store_separately"
    DEFER = "defer"


@dataclass
class CandidateMemory:
    content: Any
    memory_type: str
    context: Any
    activation: float
    relevance: float = 0.0
    reliability: float = 0.0
    relations: list = field(default_factory=list)


@dataclass
class ConnectionResult:
    integrated_representation: Any = None
    new_episode: Any = None
    schema_updates: list = field(default_factory=list)
    retrieval_routes: list = field(default_factory=list)
    confidence_and_uncertainty: dict = field(default_factory=dict)


def connect_to_prior_knowledge(new_input, current_context,
                               knowledge_network, learner_state):
    # 1. Establish the provisional task and context.
    context_frame = infer_context(new_input, current_context, learner_state)

    # 2. Extract concepts, attributes, events, and typed relations.
    new_representation = parse_meaning_and_relations(new_input, context_frame)

    # 3. Generate several independent retrieval cues.
    cues = generate_semantic_relational_contextual_cues(
        new_representation, context_frame
    )

    # 4. Retrieve candidates through cue-driven activation.
    candidates = retrieve_associated_memories(
        cues, knowledge_network, context_frame
    )

    # 5. Rank without treating familiarity as proof of correctness.
    candidates = rank_by_relevance_reliability_and_context(
        candidates, new_representation, context_frame
    )

    # 6. Compare surface features, functions, relations, and constraints.
    comparisons = compare_at_multiple_levels(
        new_representation, candidates
    )

    # 7. Align relational structures and generate bounded analogies.
    analogy_maps = align_relations_and_generate_analogies(
        new_representation, candidates, comparisons
    )

    # 8. Estimate schema fit, novelty, and contradiction.
    fit_state = estimate_fit_novelty_and_prediction_error(
        new_representation, candidates, comparisons
    )

    # 9. Select a strategy that prevents premature assimilation.
    strategy = select_integration_strategy(
        fit_state, comparisons, evidence_quality=candidates
    )

    result = ConnectionResult()

    if strategy == Strategy.INTEGRATE:
        result.integrated_representation = bind_to_existing_knowledge(
            new_representation, candidates, analogy_maps, context_frame
        )
    elif strategy == Strategy.COMPLEMENT:
        result.integrated_representation, update = fill_schema_slots(
            new_representation, candidates, context_frame
        )
        result.schema_updates.append(update)
    elif strategy == Strategy.REVISE_CANDIDATE:
        proposal = create_schema_revision_proposal(
            new_representation, candidates, comparisons
        )
        result.new_episode = encode_distinct_episode(
            new_representation, context_frame, fit_state
        )
        result.schema_updates.append(proposal)  # proposal, not automatic overwrite
    elif strategy == Strategy.STORE_EPISODE:
        result.new_episode = encode_distinct_episode(
            new_representation, context_frame, fit_state
        )
    else:
        result.new_episode = store_provisional_representation(
            new_representation, context_frame
        )
        result.confidence_and_uncertainty["status"] = "deferred"

    # 10. Attach meaningful retrieval cues and context.
    result.retrieval_routes = build_bidirectional_retrieval_routes(
        new_representation, candidates, cues, analogy_maps
    )

    # 11. Preserve confidence, contradictions, and source information.
    result.confidence_and_uncertainty.update(
        assess_confidence_and_record_unresolved_questions(
            new_representation, comparisons, fit_state
        )
    )

    # 12. Later processes revisit, consolidate, and test these connections.
    schedule_for_later_reactivation(result, knowledge_network)

    return result
```

The functions in this pseudocode are conceptual operations. A software implementation would need explicit data structures, thresholds, memory policies, and validation tests. It should not claim to reproduce the full biological mechanism of human learning.

---

# 6. Mathematical operation inventory

The main mathematical families used in this model are summarized below.

| Operation family | What it models | Where it appears |
|---|---|---|
| Weighted sums | Combining context, cues, features, and evidence | Steps 1–5 |
| Vector similarity and distance | Comparing semantic features | Steps 5–6 |
| Graph matching | Aligning roles and relations across knowledge structures | Steps 6–7 |
| Bayesian updating | Combining prior expectations with new evidence | Steps 1, 7, 11 |
| Softmax / normalization | Maintaining graded competition among interpretations or candidates | Steps 4–5 |
| Entropy and surprisal | Uncertainty and unexpectedness | Steps 1, 8, 14 |
| Prediction error | Measuring the difference between expectation and observation | Steps 8, 12 |
| Constraint satisfaction | Checking whether new details fit schema roles and relations | Steps 6, 8–9 |
| Hebbian-like association updates | Modeling co-activity-dependent connection strengthening | Step 10 |
| Precision-weighted learning | Updating beliefs according to confidence and evidence quality | Step 11 |
| Clustering / abstraction | Extracting common structure across related experiences | Step 13 |
| Retrieval scoring and error | Testing whether connections support accurate recall | Step 14 |

These operations are candidate computational descriptions. They do not imply that one brain area computes one equation or that the brain uses explicit floating-point arithmetic. Neural computations are embodied in the dynamics of populations, synapses, circuits, and interactions across timescales.

---

# 7. Biological systems that may contribute

The following mapping is intentionally cautious. Brain regions participate in networks, and their roles overlap. No region should be treated as a standalone module responsible for the entire step.

## 7.1 Hippocampus and medial temporal lobe

Potential contributions include:

- binding elements of experiences into relational or episodic representations;
- retrieving a broader memory from partial cues (pattern completion as a computational description);
- encoding distinctive or schema-inconsistent events;
- linking new information with previously stored knowledge;
- supporting later reactivation and memory integration.

## 7.2 Medial prefrontal cortex

Potential contributions include:

- representing or using established schemas and structured knowledge;
- evaluating the relevance of incoming information to prior knowledge;
- supporting schema-guided encoding and retrieval;
- interacting with hippocampal systems when information fits or conflicts with existing knowledge.

## 7.3 Distributed neocortical networks

Potential contributions include:

- long-term representations of concepts, categories, attributes, and semantic relations;
- gradual integration of newly learned knowledge;
- distributed storage of features and associations;
- reactivation and generalization across repeated experiences.

## 7.4 Lateral prefrontal and frontoparietal systems

Potential contributions include:

- maintaining goals and relevant information;
- selecting useful retrieval cues;
- controlling interference from irrelevant associations;
- comparing candidate interpretations and supporting relational reasoning.

## 7.5 Neuromodulatory systems

Dopamine-related and other neuromodulatory influences may affect learning when information is novel, salient, rewarding, or prediction-error-related. These systems do not simply label every interesting fact as important. Their effects depend on context, state, circuit, and the kind of learning involved.

---

# 8. Important safeguards and failure modes

A good prior-knowledge algorithm must make use of existing knowledge without becoming trapped by it.

### 8.1 False analogy

**Failure:** Two things share a superficial feature, so the learner assumes they share the same mechanism.

**Safeguard:** Require relational alignment, record unmatched components, and test transferred inferences independently.

### 8.2 Confirmation bias

**Failure:** New information is interpreted only in ways that support the current schema.

**Safeguard:** Generate contrastive cues, search for disconfirming evidence, and preserve conflicting observations.

### 8.3 Overgeneralization from one example

**Failure:** One episode is converted into a universal rule.

**Safeguard:** Keep episode-specific context, track sample size, and use precision-weighted updates.

### 8.4 Similarity mistaken for causal relation

**Failure:** Two events co-occur or share features, so one is assumed to cause the other.

**Safeguard:** Store relation types explicitly and distinguish correlation, temporal order, function, and causation.

### 8.5 Schema rigidity

**Failure:** The learner rejects valid new information because it conflicts with an old expectation.

**Safeguard:** Use a separate episodic trace, allow competing hypotheses, and revise the schema when reliable evidence accumulates.

### 8.6 Schema overfitting

**Failure:** A schema is modified too aggressively in response to a rare or noisy event.

**Safeguard:** Use confidence and evidence quality to control update size; keep candidate revisions separate until supported.

### 8.7 Retrieval competition

**Failure:** A highly familiar but irrelevant memory wins retrieval.

**Safeguard:** Evaluate relevance separately from activation and preserve multiple candidates when rankings are close.

### 8.8 Context loss

**Failure:** A fact learned in one narrow context is applied universally.

**Safeguard:** Bind facts to contextual conditions and test them in other settings before generalizing.

### 8.9 Confidence inflation

**Failure:** A fluent analogy or familiar explanation feels true even when its details are wrong.

**Safeguard:** Label analogical inferences as hypotheses and compare them with independent evidence.

### 8.10 Memory distortion during reactivation

**Failure:** Recalling and retelling an event gradually changes its details.

**Safeguard:** Preserve original source information where possible, distinguish the original observation from later interpretation, and verify critical facts against reliable records.

---

# 9. Example walkthrough: learning a new concept through prior knowledge

Suppose a learner is introduced to the idea that **a cell membrane regulates what enters and leaves a cell**.

1. **Context preparation:** The learner recognizes a biology lesson about cell structure.
2. **Concept extraction:** The key entities are cell, membrane, and transported substances. The central relation is regulation of movement across a boundary.
3. **Cue generation:** Possible cues include boundary, selective passage, control, inside/outside, and transport.
4. **Retrieval:** The learner recalls a door, a filter, a security checkpoint, and perhaps a familiar household container.
5. **Candidate evaluation:** A door and a filter are both useful but incomplete. A filter captures selectivity; a door captures controlled access. Neither is identical to a biological membrane.
6. **Relational comparison:** The learner maps “boundary,” “substance,” and “selective passage” between the biological concept and the familiar filter analogy.
7. **Analogy limits:** A membrane is not a rigid sieve. It contains a lipid bilayer and proteins, and transport depends on molecular properties and specific mechanisms.
8. **Fit and novelty:** Existing knowledge about filters provides a useful scaffold, while the molecular mechanism is new and must be encoded distinctly.
9. **Integration:** The learner links selective transport to the existing idea of filtering, but does not equate the two mechanisms.
10. **Schema update:** The learner refines the general schema “a boundary can regulate passage” with a new biological example.
11. **Retrieval route:** Later, the cue “selective boundary” may help retrieve membrane function; “membrane proteins” may retrieve the biological details.
12. **Testing:** The learner explains why a cell membrane is selectively permeable and names at least one limitation of the filter analogy.

This example illustrates why the algorithm must combine familiar knowledge with careful preservation of new domain-specific details.

---

# 10. How this point supports comprehension and future learning

Connecting new information to prior knowledge can improve comprehension because the learner does not have to interpret every item as an isolated fact. Existing concepts provide context, expected relationships, and a framework for organizing new details. A connected representation can also provide more retrieval routes: a new idea may be remembered through its definition, function, example, cause, contrast, or relation to a familiar concept.

However, the benefit depends on the accuracy and flexibility of the prior knowledge. Incorrect schemas can mislead interpretation; superficial analogies can introduce false inferences; and highly unexpected evidence can be lost if it is forced into an old framework. A sound learning system therefore performs both integration and error-sensitive separation.

**Desired final state:** New knowledge is connected to relevant existing concepts and relations, its source and context remain recoverable where needed, unresolved contradictions are preserved, and the learner can retrieve and apply the connection in more than one context.

---

# 11. Scientific references and further reading

The sources below support the broad principles used in this algorithm. They do not validate every equation or every step as a literal description of neural activity.

1. **Gilboa, A., & Marlatte, H. (2017).** Neurobiology of schemas and schema-mediated memory. *Trends in Cognitive Sciences*, 21(8), 618–631. Discusses schema representations and the interaction between hippocampal and prefrontal systems. [https://pubmed.ncbi.nlm.nih.gov/28602485/](https://pubmed.ncbi.nlm.nih.gov/28602485/)
2. **Larsen, P. B., et al. (2020).** Naïve to expert: Considering the role of previous knowledge in memory. *Journal of Cognition*, 3(1). Reviews how prior knowledge can change encoding, consolidation, and retrieval. [https://pubmed.ncbi.nlm.nih.gov/32954007/](https://pubmed.ncbi.nlm.nih.gov/32954007/)
3. **van Kesteren, M. T. R., Ruiter, D. J., Fernández, G., & Henson, R. N. (2012).** How schema and novelty augment memory formation. *Trends in Neurosciences*, 35(4), 211–219. Reviews schema congruence, novelty, and medial temporal–prefrontal interactions. [https://pubmed.ncbi.nlm.nih.gov/22398180/](https://pubmed.ncbi.nlm.nih.gov/22398180/)
4. **van Kesteren, M. T. R., et al. (2013).** Differential roles for medial prefrontal and medial temporal cortices in schema-dependent encoding: From congruent to incongruent. *NeuroImage*, 67, 313–322. Examines neural differences as information varies in schema congruence. [https://pubmed.ncbi.nlm.nih.gov/23770537/](https://pubmed.ncbi.nlm.nih.gov/23770537/)
5. **Harris, J. A., & colleagues (2016).** Effects of prior knowledge on memory: Implications for education. *Mind, Brain, and Education*, 10(3), 153–161. Reviews how prior knowledge can help or hinder learning. [https://doi.org/10.1111/mbe.12110](https://doi.org/10.1111/mbe.12110)
6. **Krawczyk, D. C. and colleagues (2016).** General and specialized brain correlates for analogical reasoning: A meta-analysis of functional imaging studies. *Human Brain Mapping*, 37, 1953–1969. Reviews brain systems involved in relational analogy. [https://pmc.ncbi.nlm.nih.gov/articles/PMC6867453/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6867453/)
7. **Parsons, J.-D., & Davies, J. (2022).** The neural correlates of analogy component processes. *Cognitive Science*, 46(3), e13116. Reviews component processes and distributed neural contributions to analogy. [https://pubmed.ncbi.nlm.nih.gov/35297092/](https://pubmed.ncbi.nlm.nih.gov/35297092/)
8. **Audrain, S., & McAndrews, M. P. (2022).** Schemas provide a scaffold for neocortical integration of new memories over time. *Nature Communications*, 13, 5795. Examines schema effects on memory representation over time. [https://www.nature.com/articles/s41467-022-33517-0](https://www.nature.com/articles/s41467-022-33517-0)
9. **Heald, J. B., Wolpert, D. M., & Lengyel, M. (2023).** The computational and neural bases of context-dependent learning. *Annual Review of Neuroscience*, 46, 233–258. Reviews computational and neural accounts of context inference and context-dependent memory. [https://doi.org/10.1146/annurev-neuro-092322-100402](https://doi.org/10.1146/annurev-neuro-092322-100402)
10. **To update or to create? The influence of novelty and prior knowledge on memory networks (2024).** Reviews schema-consistent integration, novel episodic encoding, and hippocampal–prefrontal interactions. [https://pmc.ncbi.nlm.nih.gov/articles/PMC11343309/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11343309/)
11. **How to optimize knowledge construction in the brain (2020).** Reviews encoding, consolidation, reactivation, memory integration, predictive coding, and hippocampal–prefrontal interactions. [https://www.nature.com/articles/s41539-020-0064-y](https://www.nature.com/articles/s41539-020-0064-y)

---

## Completion checklist

- [x] Preserve the source point's meaning: relate new information to existing knowledge.
- [x] Model semantic, contextual, functional, and relational links.
- [x] Include analogy generation and explicit limits of analogies.
- [x] Include schema-fit, novelty, mismatch, and contradiction handling.
- [x] Include integration, separate episodic encoding, and cautious schema revision.
- [x] Explain candidate mathematical operations and their limits.
- [x] Connect computational stages to plausible distributed biological systems.
- [x] Include integrated pseudocode, failure modes, and a worked example.
- [x] Keep scientific claims distinguishable from modeling assumptions.

**End of Point 02.** Point 03 should not be started until the learner explicitly requests continuation.
