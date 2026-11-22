# Point 10 — Schema-Guided Inference and Generalization

## Turning familiar knowledge into predictions about an unfamiliar case

**Source sequence:** *Learning, Memory, and Concept Formation.txt* — the section immediately following schema formation. The source describes how inference and generalization emerge from structured knowledge: a new animal can be understood and acted toward by matching its features to an existing schema (for example, a dog-like animal may be inferred to be a dog). It also places these operations within interactions among medial prefrontal and temporal-lobe networks, with the hippocampus binding otherwise separate features.

**Scope of this algorithm:** This point formalizes the use of an existing schema to interpret incomplete or unfamiliar input, infer likely unobserved properties and relations, select an appropriate category, and generalize knowledge to a new instance. Analogical transfer between structurally different domains is treated separately in Point 11. The algorithm here may use a schema that was formed earlier, but does not re-derive the whole schema from scratch.

---

## 1. Biological interpretation and modeling boundary

A schema is a structured knowledge representation: a set of entities or roles, characteristic features, relationships, likely event sequences, constraints, and context-dependent expectations. When a new situation is encountered, the brain does not need to learn every element independently. Previously acquired knowledge can bias attention, perception, memory retrieval, prediction, and action selection. This allows a person to infer information that is not directly present in the current sensory input.

The source gives a simple animal example. A novel animal may share a subset of perceptual and behavioral cues with a familiar dog. A dog-related schema can make other properties more likely: a particular body plan, familiar vocalizations, likely behavior, or a suitable response. The inference is useful, but it is not guaranteed to be correct. A wolf, fox, or dog-like robot could activate overlapping features while requiring a different final classification or action. A sound implementation must therefore represent uncertainty, seek discriminating evidence, and avoid converting a plausible prediction into a fact.

Several neural systems contribute, with no single brain region implementing the complete algorithm:

- **Temporal and distributed semantic systems** contribute learned conceptual knowledge and feature associations. The current stimulus can activate visual, auditory, action-related, and other knowledge relevant to candidate categories.
- **Medial prefrontal and orbitomedial prefrontal regions** are implicated in using structured prior knowledge, selecting relevant dimensions, integrating information with existing schemas, and guiding goal-sensitive interpretation. Their exact contributions vary by task and are not reducible to a single “schema module.”
- **The hippocampal formation** can bind relational information among features and retrieve specific episodes. It is particularly relevant when a current combination of features must be linked to a remembered event, or when a new relation needs to be encoded.
- **Attention and control networks** help prioritize diagnostic features, maintain the current goal, and resolve conflict when the input does not fit the leading interpretation.

The mathematical operations below are computational descriptions that make the proposed process testable. They are not claims that neurons literally calculate cosine similarity, Bayes’ theorem, matrix multiplication, or software-style loops. Biological implementations may use population activity, synaptic strengths, recurrent dynamics, oscillatory coordination, neuromodulation, and plasticity to approximate some of the same functional computations.

A key scientific qualification is that schema-based completion can help understanding while also creating memory distortions. The model must therefore distinguish **what was observed**, **what was retrieved**, **what was inferred**, and **what remains uncertain**.

---

## 2. Inputs, persistent representations, and outputs

### 2.1 Inputs

1. **Current sensory evidence**: visual shape, color, movement, sound, texture, smell, spatial arrangement, or other available signals.
2. **Current context**: location, time, task, environment, social situation, and preceding events.
3. **Goal or question**: classify the object, predict what happens next, choose an action, explain an event, or identify a missing relation.
4. **Existing schema library**: previously learned structured categories and situation models.
5. **Episodic memory cues**: remembered encounters that may be similar to the current case.
6. **Reliability estimates**: how trustworthy each cue is in the current conditions (for example, a muffled sound is less reliable than a clear one).

### 2.2 Internal representations

Each candidate schema should contain:

- a category or situation identity;
- core, typical, optional, and exception features;
- roles and relations among features;
- context conditions under which the schema applies;
- expected but not yet observed attributes;
- alternative subtypes and neighboring categories;
- confidence or strength estimates for its associations;
- links to specific episodes and their source/context information.

The current case should be represented separately as an evidence record. Each property should carry a status such as `observed`, `not_observed`, `contradicted`, `inferred`, or `unknown`. “Not observed” must not automatically mean “absent.”

### 2.3 Outputs

- a ranked set of candidate schemas or categories;
- a structured interpretation of the current case;
- inferred features and relations, each marked with its evidential status;
- confidence and uncertainty estimates;
- predictions about likely next states or properties;
- a selected response or action, when the task requires one;
- a record of prediction errors and any information that should be learned from the new case.

---

# 3. Main algorithm: 17 sequential steps

## Step 1 — Construct a current-case evidence representation

**Purpose:** Convert incoming information into a structured description without prematurely assigning it to a familiar category.

**Biological interpretation:** Sensory pathways extract information at different levels and across modalities. Visual systems may contribute shape, contour, movement, and configuration; auditory systems may contribute call structure, pitch, and temporal patterns. These streams can activate distributed conceptual knowledge. Attention influences which features receive more processing. The representation available to higher-level systems is therefore selective and context-sensitive rather than a complete photographic copy of the world.

**Computational procedure:**

1. Receive available sensory observations and contextual cues.
2. Extract candidate features and relations (for example, `four_legs`, `fur_like_surface`, `barking_sound`, `approaches_person`).
3. Attach a reliability value to each observation.
4. Mark each feature as observed, contradicted, or unknown. Do not infer an unobserved feature at this stage.
5. Preserve temporal order when the input describes a sequence of actions.
6. Keep the raw evidence separate from later schema-derived predictions.

A simple evidence vector can be represented as

\[
\mathbf{x}=[x_1,x_2,\ldots,x_n],\qquad x_i\in\{1,0,\varnothing\},
\]

where 1 means observed present, 0 means observed absent/contradicted, and \(\varnothing\) means not measured or unknown. In a richer implementation, each feature also has an observation confidence \(r_i\in[0,1]\).

**Possible mathematical operations:** feature extraction; temporal differencing; normalization; reliability weighting; construction of a sparse feature vector or graph; missing-value representation.

**Why it matters:** If unknown features are silently treated as absent, the system will reject valid matches. If inferred features are inserted into the raw evidence, later processing may mistake its own prediction for an observation.

## Step 2 — Activate potentially relevant schemas from context and partial cues

**Purpose:** Retrieve a manageable set of candidate schemas before performing detailed inference.

**Biological interpretation:** Partial cues can reactivate distributed semantic knowledge and prior event structures. Context changes which knowledge becomes accessible. A furry animal in a park may activate a different candidate set from a plush object in a toy shop, even when some visual features overlap. Medial prefrontal and temporal systems are thought to interact during the use of structured knowledge; this is not a literal database lookup.

**Computational procedure:**

1. Use the current goal and context to identify relevant schema families.
2. Use a subset of observed features to retrieve candidate schemas associated with those cues.
3. Include more than one plausible candidate when ambiguity is substantial.
4. Include a broad “unknown/other” candidate so the system is not forced into a familiar category.
5. Avoid allowing one highly salient but unreliable cue to dominate retrieval.

One possible activation score is

\[
A(S_j)=b_j+\sum_i r_i\,w_{ij}\,g(x_i,S_j)+c_j,
\]

where \(S_j\) is a candidate schema, \(b_j\) is its baseline accessibility, \(w_{ij}\) is the learned association between feature \(i\) and the schema, \(r_i\) is cue reliability, \(g\) measures compatibility, and \(c_j\) summarizes contextual relevance. This is a modeling choice, not an established neural equation.

A softmax can turn scores into normalized candidate activations:

\[
P(S_j\mid\mathbf{x},C)=\frac{e^{A(S_j)/\tau}}{\sum_k e^{A(S_k)/\tau}},
\]

where \(\tau\) controls how sharply activation concentrates on the leading candidate.

**Possible mathematical operations:** associative retrieval; weighted summation; contextual gain modulation; competition; softmax normalization; top-k candidate selection.

## Step 3 — Retrieve the relational structure of each candidate schema

**Purpose:** Retrieve not just a category label, but its slots, roles, relations, constraints, and expected transitions.

**Biological interpretation:** A schema is structured knowledge, not merely a list of similar words or features. Temporal semantic systems contribute conceptual attributes; medial prefrontal systems help use event structure and relevant regularities; hippocampal retrieval can reinstate particular relational episodes. Neural representations are distributed and task-dependent.

**Computational procedure:**

For each candidate schema, retrieve a structured graph:

- nodes: entities, roles, attributes, events, and states;
- edges: relations such as `has_part`, `is_located_in`, `approaches`, `causes`, `precedes`, or `used_for`;
- slot constraints: which values can fill each role;
- typicality: how common a feature or relation is;
- importance: how diagnostic it is for the current question;
- contextual conditions: when the relation is expected to apply.

Example: a generic `dog` schema may contain attributes such as a canine body plan and common dog vocalizations, but these attributes must be treated as probabilistic rather than universal. A `friendly_dog_interaction` schema may additionally encode a person, an approaching animal, body-language cues, and a possible social response.

**Possible mathematical operations:** graph retrieval; associative spreading activation; relation indexing; constraint retrieval; hierarchy traversal.

## Step 4 — Align the observed case with schema roles and slots

**Purpose:** Determine which observed features could fill which schema slots.

**Biological interpretation:** The brain relates the current perceptual configuration to stored knowledge. Matching is influenced by attention and task demands. The hippocampus may support binding when the relevant information consists of multiple relations or features distributed across time and space.

**Computational procedure:**

1. Compare each observed feature with candidate slot values.
2. Compare relations among features, not only individual features. For example, four legs and fur are weaker evidence than a coherent combination of body shape, gait, and characteristic sound.
3. Permit one observed cue to support more than one candidate slot when appropriate.
4. Penalize incompatible relations, but only when the relevant property was actually observed.
5. Preserve multiple possible alignments when the case is ambiguous.

A feature-level match may use a similarity kernel:

\[
s_i=\exp\left(-\frac{d_i(\mathbf{x},\mathbf{v}_i)^2}{2\sigma_i^2}\right),
\]

where \(\mathbf{v}_i\) is a stored feature pattern and \(d_i\) is an appropriate distance. For categorical attributes, a compatibility table may be more suitable than Euclidean distance.

A relation-level match can be scored separately:

\[
M_{rel}=\sum_{(u,v)} q_{uv}\,\mathbb{1}[R_{obs}(u,v)\text{ is compatible with }R_S],
\]

where \(q_{uv}\) weights the importance of an observed relation.

**Possible mathematical operations:** pattern matching; similarity kernels; graph alignment; constraint satisfaction; weighted relation scoring; combinatorial assignment.

## Step 5 — Weight evidence by diagnosticity, reliability, and task relevance

**Purpose:** Give more influence to observations that genuinely distinguish among candidates.

**Biological interpretation:** Attention is selective. Goals and context can bias processing toward diagnostic dimensions. A feature that is common to many categories should contribute less to discrimination than a feature that separates close alternatives. The brain's weighting is adaptive and shaped by learning and task context.

**Computational procedure:**

1. Estimate how reliable each observed cue is.
2. Estimate how informative it is for separating current candidates.
3. Increase weight for features relevant to the current goal.
4. Reduce weight for redundant cues that provide almost the same information.
5. Do not use typicality alone as proof of category membership.

An information-theoretic diagnosticity term can be approximated by

\[
D(f_i)=\log\frac{P(f_i\mid S_j)}{P(f_i\mid \neg S_j)},
\]

for a particular candidate \(S_j\), with smoothing to avoid zero probabilities. The total contribution can be

\[
E_j=\sum_i r_i\,a_i\,D_{ij},
\]

where \(a_i\) is goal-dependent attentional relevance. The log-likelihood ratio can be positive or negative; contradictory evidence should lower a candidate's score.

**Possible mathematical operations:** reliability weighting; log-likelihood ratios; mutual information; feature selection; redundancy reduction; normalization.

## Step 6 — Estimate how well each schema explains the observed evidence

**Purpose:** Compare candidate explanations using the evidence actually available.

**Biological interpretation:** The nervous system combines bottom-up sensory input with top-down expectations. This can improve recognition under incomplete conditions, but it can also bias perception. Competing interpretations may remain active until enough evidence is available.

**Computational procedure:**

For each schema, calculate an evidence-fit score that includes:

- support from present observations;
- penalties for directly contradicted core features;
- uncertainty from missing observations;
- compatibility of the observed relations;
- contextual applicability.

A Bayesian expression is

\[
P(S_j\mid\mathbf{x},C)\propto P(\mathbf{x}\mid S_j,C)P(S_j\mid C).
\]

The prior \(P(S_j\mid C)\) expresses how plausible the schema was before considering the current evidence; the likelihood \(P(\mathbf{x}\mid S_j,C)\) expresses how expected the observations would be if that schema applied. Normalize across candidates after scoring.

Do not treat the Bayesian expression as evidence that the brain explicitly represents probability distributions in this exact form. It is a normative framework for describing how prior knowledge and evidence can be combined.

**Possible mathematical operations:** Bayesian updating; log probability addition; normalization; evidence accumulation; model comparison.

## Step 7 — Generate predictions for unobserved slots

**Purpose:** Use the best-supported schemas to estimate likely but unseen attributes and relations.

**Biological interpretation:** Prior knowledge can guide expectations about what is likely to be present next or what an object is likely to do. Schema-based predictions can affect attention and interpretation. This is the central generative aspect of inference: knowledge supplies plausible structure beyond the current input.

**Computational procedure:**

1. For each candidate schema, identify slots not yet observed.
2. Estimate conditional probabilities for their possible values given the observed features, context, goal, and schema.
3. Keep each predicted value separate from the observed record.
4. Generate only predictions that are relevant to the current task; do not fill every possible slot.
5. Preserve alternative values if their probabilities are close.

For slot \(z_k\),

\[
P(z_k=v\mid\mathbf{x}_{obs},S_j,C,G)
\]

is the predicted probability that the slot takes value \(v\), given observed data, schema, context, and goal. A simple conditional model might use

\[
P(z_k=v\mid\cdot)=\operatorname{softmax}_v\left(\theta_{kv}^{T}\phi(\mathbf{x}_{obs},S_j,C,G)\right),
\]

where \(\phi\) is a feature representation and \(\theta_{kv}\) contains learned parameters.

**Possible mathematical operations:** conditional probability estimation; generative prediction; softmax; pattern completion; sampling from a distribution.

## Step 8 — Complete missing relations while preserving uncertainty

**Purpose:** Infer how entities or events may be related when some links are not directly observed.

**Biological interpretation:** The hippocampus is important for binding relations among elements of an episode, while prefrontal and temporal networks contribute prior structure and semantic knowledge. A schema can make a relation more accessible, but the relation is still an inference if it was not witnessed.

**Computational procedure:**

1. Identify pairs of observed entities with a missing relation.
2. Retrieve relations permitted by the candidate schema.
3. Condition the relation estimate on context and observed sequence.
4. Compare alternative relations, including `unknown`.
5. Store the selected relation as inferred, not observed.

For a relation variable \(R_{ab}\) between entities \(a\) and \(b\), estimate

\[
P(R_{ab}=r\mid a,b,\mathbf{x},S_j,C).
\]

A graph-completion objective can balance schema compatibility and evidence:

\[
\hat{G}=\arg\max_G\left[\log P(G\mid S_j,C)+\lambda\log P(\mathbf{x}\mid G)\right].
\]

**Possible mathematical operations:** relational inference; graph completion; constraint propagation; conditional probability; maximum a posteriori estimation.

## Step 9 — Infer likely category membership and hierarchical level

**Purpose:** Decide whether the new case belongs to a familiar category, a subtype, a broader parent category, or no known category.

**Biological interpretation:** Categorization can occur at multiple levels of abstraction. A person may first recognize an animal as an animal, then as a mammal, then as a canine, and only later identify a specific species. Available evidence and expertise influence the level of specificity that can be justified.

**Computational procedure:**

1. Rank candidate categories using evidence and prior knowledge.
2. Evaluate category scores at multiple levels in the hierarchy.
3. Choose the most specific category whose confidence exceeds a threshold.
4. If no specific category passes the threshold, return a broader parent class or `uncertain`.
5. Do not force a dog label merely because several cues resemble a dog; preserve alternatives such as wolf, fox, or unknown animal if discriminative cues are absent.

A hierarchical decision rule can be written as

\[
S^*=\operatorname*{arg\,max}_{S_j}P(S_j\mid\mathbf{x},C),
\]

subject to a confidence threshold and a specificity constraint. One possible rule chooses the deepest node in the taxonomy for which posterior probability exceeds \(\theta\) and the margin over close competitors exceeds \(\delta\).

**Possible mathematical operations:** hierarchical classification; posterior ranking; thresholding; margin calculation; selective prediction; rejection option.

## Step 10 — Use temporal and causal structure to predict what may happen next

**Purpose:** Extend generalization from properties of an object to the likely progression of an event.

**Biological interpretation:** Event schemas encode typical sequences and relationships. The brain can anticipate likely next states, orient attention toward expected events, and prepare actions. These predictions are sensitive to goals and can be revised when the sequence differs from expectations.

**Computational procedure:**

1. Identify the current event state and its role in the schema.
2. Retrieve plausible transitions.
3. Estimate the probability of each transition under current context and goals.
4. Generate a limited set of next-event predictions.
5. Distinguish a temporal sequence (`A tends to occur before B`) from a causal claim (`A produces B`).
6. Avoid causal conclusions based only on repeated co-occurrence.

A transition model can be represented as

\[
P(s_{t+1}\mid s_t,S_j,C,G),
\]

or, for a multi-step event, as a state-transition matrix \(T\), where \(T_{ab}=P(s_{t+1}=b\mid s_t=a)\). The distribution after \(k\) steps in a simple stationary model is \(\mathbf{p}_{t+k}=\mathbf{p}_tT^k\). Real cognition is more context-sensitive than a fixed Markov chain, but this provides a testable approximation.

**Possible mathematical operations:** transition probability estimation; sequence prediction; state-space modeling; conditional expectation; temporal constraint checking.

## Step 11 — Integrate schema predictions with the current sensory evidence

**Purpose:** Produce a coherent interpretation without allowing expectations to overwrite what is actually perceived.

**Biological interpretation:** Perception and memory are interactive. Prior knowledge can guide interpretation of ambiguous input, while new sensory evidence can correct predictions. Strongly schema-consistent input may be assimilated efficiently; unexpected details may receive additional attention and require a more specific episodic representation.

**Computational procedure:**

1. Compare each predicted feature with the latest sensory evidence.
2. If confirmed, increase confidence in the relevant interpretation.
3. If contradicted, reduce confidence or revise the candidate schema.
4. If still unobserved, retain it as a prediction only.
5. Recompute category and relation estimates after significant new evidence arrives.

For a binary feature, a simple Bayesian update is

\[
P(H\mid e)=\frac{P(e\mid H)P(H)}{P(e\mid H)P(H)+P(e\mid\neg H)P(\neg H)}.
\]

A predictive-error term can be represented as

\[
\varepsilon_i=x_i-\hat{x}_i,
\]

for numeric or suitably encoded features. For categorical features, use a mismatch indicator or the negative log probability of the observed value:

\[
\mathcal{S}(e)=-\log P(e\mid S_j,C),
\]

where \(\mathcal{S}\) represents model-relative surprise. Surprise is not automatically evidence that the schema is wrong; it can also signal noisy input or an exceptional instance.

**Possible mathematical operations:** Bayesian evidence integration; prediction error; surprise; iterative posterior updating; confidence recalibration.

## Step 12 — Resolve conflict among schemas and seek discriminating information

**Purpose:** Handle cases in which several schemas explain the observations similarly well or one schema predicts a feature that another rejects.

**Biological interpretation:** Conflict and uncertainty can recruit attention and cognitive control. People can selectively inspect, listen, ask a question, or wait for additional information. The hippocampus may contribute specific remembered comparisons, while prefrontal systems can help maintain the task goal and select diagnostic evidence.

**Computational procedure:**

1. Detect close posterior scores or incompatible predictions.
2. Identify which missing cue would most clearly distinguish the top candidates.
3. Estimate whether the cue can be obtained safely and at reasonable cost.
4. Direct attention or behavior toward acquiring that cue.
5. Update the candidate set after observing it.
6. If no cue is available, report uncertainty rather than manufacturing a definitive answer.

An information-gain criterion is

\[
IG(q)=H(S\mid D)-\mathbb{E}_{a\sim P(a\mid q,D)}[H(S\mid D,a)],
\]

where \(q\) is a possible question or action, \(D\) is current evidence, \(a\) is its possible answer, and \(H\) is entropy. A practical selection rule is

\[
q^*=\arg\max_q\left[IG(q)-\kappa\,Cost(q)\right].
\]

**Possible mathematical operations:** entropy; expected information gain; active sampling; expected utility; conflict detection; decision thresholds.

## Step 13 — Produce a source-aware interpretation

**Purpose:** Keep direct observations, retrieved memories, and schema-based inferences distinguishable.

**Biological interpretation:** Human memory is reconstructive. Remembered details can be combined with generalized knowledge and expectations. A familiar schema may make a missing detail feel plausible, even when that detail was never present in the original event. Source-monitoring processes are therefore essential to avoiding confident but inaccurate recollection.

**Computational procedure:**

For every element in the final representation, retain:

- content;
- origin (`current_sensory_input`, `specific_episode`, `schema_prediction`, or `external_information`);
- confidence;
- time and context of acquisition, when available;
- whether it was directly observed or inferred;
- which candidate schema supports it.

Do not rewrite a predicted attribute into the original episode record. In a later recall, retrieve the source label along with the content.

**Possible mathematical operations:** provenance tracking; confidence annotation; source classification; evidence lineage; uncertainty propagation.

## Step 14 — Select a response or action appropriate to the inferred case

**Purpose:** Convert the interpretation into behavior or a task-specific answer.

**Biological interpretation:** Schemas can support action by predicting what is likely to happen and which responses are suitable. Actual behavior depends on goals, reward, safety, social rules, and current sensory evidence—not category inference alone. For example, even a strong dog classification does not imply that approaching an unfamiliar dog is safe.

**Computational procedure:**

1. Generate candidate actions that fit the current goal.
2. Estimate expected outcomes under the current interpretation and its uncertainty.
3. Include the cost of errors, especially when false-positive decisions are dangerous.
4. Prefer a conservative action when confidence is insufficient.
5. Continue monitoring the situation after acting.

An expected-utility formulation is

\[
U(a)=\sum_s P(s\mid D)R(a,s)-Cost(a),
\]

where \(R(a,s)\) is the value of action \(a\) if state \(s\) is true. Select \(a^*=\arg\max_a U(a)\), subject to safety constraints.

**Possible mathematical operations:** expected utility; risk-sensitive decision-making; threshold policies; action selection; cost-sensitive classification.

## Step 15 — Compare the outcome with the predicted outcome

**Purpose:** Determine which predictions were confirmed, violated, or never tested.

**Biological interpretation:** Learning is influenced by the difference between expectation and outcome. A mismatch can prompt further attention, episodic encoding, or schema updating. Different neural systems may respond to different kinds of mismatch; it is too simplistic to assign every schema prediction error to the hippocampus alone.

**Computational procedure:**

1. Record the actual outcome or newly observed property.
2. Match it to the predictions made in Steps 7–10.
3. Calculate an error for each tested prediction.
4. Separate true contradiction from missing measurement or noisy evidence.
5. Determine whether the mismatch is isolated, repeated, or explained by a context change.

A simple scalar prediction error is

\[
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t),
\]

which is a reinforcement-learning-style temporal-difference expression and is not a universal neural formula for semantic inference. For categorical prediction, use a log-loss or Brier score to evaluate calibration over repeated cases.

**Possible mathematical operations:** prediction-error calculation; residual analysis; likelihood comparison; calibration scoring; change detection.

## Step 16 — Update the case model and, only when justified, the schema

**Purpose:** Learn from the new instance without allowing one unusual example to corrupt a reliable generalization.

**Biological interpretation:** New experiences can be integrated with existing knowledge or encoded as distinct episodes. Schema-consistent information may be assimilated relatively efficiently, whereas incongruent information can receive stronger episodic encoding. The degree and pathway of updating depend on novelty, relevance, prior knowledge, task, and memory conditions.

**Computational procedure:**

1. Store the new episode with its specific context and source details.
2. If evidence repeatedly supports a change to a schema, update the relevant feature or relation probability.
3. If the case is a rare exception, add an exception or subtype rather than changing the general rule immediately.
4. Keep confidence intervals or uncertainty estimates for learned parameters.
5. Reduce the influence of noisy or low-reliability cases.
6. Preserve previous schema versions when the change is substantial, enabling later comparison.

For a simple Bernoulli feature with a Beta prior, update counts as

\[
\alpha' = \alpha + n_1,\qquad \beta' = \beta+n_0,
\]

where \(n_1\) and \(n_0\) are reliable observations of presence and absence. The posterior mean is \(\alpha'/ (\alpha'+\beta')\). More complex schemas require hierarchical or relational updates rather than independent feature counts.

**Possible mathematical operations:** incremental Bayesian updating; learning-rate adjustment; weighted counts; regularization; hierarchical parameter estimation; exception detection.

## Step 17 — Consolidate the interpretation into reusable knowledge while retaining episode-specific detail

**Purpose:** Make the new knowledge available for future inference without erasing the individual event.

**Biological interpretation:** Generalized knowledge and episodic memory are complementary. A new case may strengthen existing semantic structure, create a new subtype, or remain primarily an episodic memory. The hippocampus can preserve relational details of a specific event while cortical systems contribute to more generalized knowledge over time. These processes should not be modeled as a mandatory immediate conversion of every episode into a schema.

**Computational procedure:**

1. Store the detailed episode separately.
2. Store validated changes to the schema as probabilistic updates.
3. Link the episode to the schema and to relevant neighboring categories.
4. Preserve contradictory cases and boundary examples.
5. Make the updated schema available to subsequent inference while maintaining access to its evidence history.

**Possible mathematical operations:** memory indexing; graph linking; incremental model selection; hierarchical generalization; compression with retained exemplars; confidence-weighted consolidation.

---

# 4. Integrated algorithm in pseudocode

```text
FUNCTION SCHEMA_GUIDED_INFERENCE(raw_input, context, goal, schema_library):

    evidence = EXTRACT_FEATURES_AND_RELATIONS(raw_input, context)
    MARK_EACH_ITEM_AS_OBSERVED_CONTRADICTED_OR_UNKNOWN(evidence)
    ATTACH_RELIABILITY_AND_SOURCE(evidence)

    candidates = RETRIEVE_SCHEMAS(
        partial_cues=evidence.observed,
        context=context,
        goal=goal
    )
    candidates.ADD(UNKNOWN_OR_OTHER_SCHEMA)

    FOR each schema IN candidates:
        structure = RETRIEVE_SCHEMA_GRAPH(schema)
        alignment = ALIGN_EVIDENCE_TO_SLOTS_AND_RELATIONS(evidence, structure)
        diagnostic_weights = ESTIMATE_RELIABILITY_AND_DIAGNOSTICITY(
            alignment, context, goal
        )
        posterior[schema] = COMBINE_PRIOR_AND_EVIDENCE(
            schema, alignment, diagnostic_weights, context
        )

    NORMALIZE(posterior)
    ranked = RANK_CANDIDATES(posterior)

    FOR each candidate IN ranked:
        predictions[candidate] = INFER_UNOBSERVED_SLOTS_AND_RELATIONS(
            candidate.schema, evidence, context, goal
        )
        predictions[candidate] = MARK_AS_INFERRED_NOT_OBSERVED(predictions)
        next_event_predictions[candidate] = PREDICT_EVENT_TRANSITIONS(
            candidate.schema, evidence, context, goal
        )

    interpretation = SELECT_HIERARCHICAL_CATEGORY_IF_CONFIDENT(ranked)

    IF CANDIDATES_ARE_AMBIGUOUS(ranked) OR IMPORTANT_CONFLICT_EXISTS(predictions):
        query = SELECT_MOST_DIAGNOSTIC_INFORMATION_GATHERING_ACTION(
            ranked, predictions, goal
        )
        new_evidence = ACQUIRE_INFORMATION(query)
        evidence = UPDATE_EVIDENCE_WITH_PROVENANCE(evidence, new_evidence)
        RECOMPUTE_CANDIDATE_POSTERIORS_AND_PREDICTIONS()

    final_interpretation = INTEGRATE_EVIDENCE_AND_PREDICTIONS(
        evidence, ranked, predictions
    )
    PRESERVE_SOURCE_LABELS(final_interpretation)

    action = SELECT_GOAL_AND_RISK_APPROPRIATE_ACTION(
        final_interpretation, goal
    )
    outcome = MONITOR_OUTCOME(action)
    errors = COMPARE_OUTCOME_WITH_TESTED_PREDICTIONS(outcome, predictions)

    STORE_SPECIFIC_EPISODE(evidence, final_interpretation, outcome)
    UPDATE_SCHEMA_ONLY_WHEN_EVIDENCE_JUSTIFIES(errors, schema_library)
    LINK_EPISODE_TO_SCHEMA_WITH_SOURCE_INFORMATION()

    RETURN {
        interpretation,
        inferred_properties,
        uncertainty,
        predictions,
        selected_action: action,
        evidence_provenance,
        learning_updates
    }
```

---

# 5. Compact mathematical summary

A useful abstract description is a generative inference cycle:

\[
\boxed{\text{Current evidence} + \text{context} + \text{schema prior}
\longrightarrow \text{candidate interpretation}}
\]

For candidate schema \(S_j\), the posterior is

\[
P(S_j\mid X,C,G)\propto P(X\mid S_j,C,G)P(S_j\mid C,G).
\]

The schema then predicts missing variables and relations:

\[
P(Z_{miss},R_{miss}\mid X_{obs},S_j,C,G).
\]

New evidence \(E\) revises the interpretation:

\[
P(S_j\mid X,E,C,G)\propto P(E\mid S_j,X,C,G)P(S_j\mid X,C,G).
\]

The system should also keep a source-aware representation:

\[
M=\{(item,source,status,confidence,context)\},
\]

so that an inferred detail is never silently relabeled as directly observed. Across repeated experiences, validated prediction outcomes can update schema parameters while individual episodes remain separately retrievable.

---

# 6. Biological implementation map: functional contributions, not one-to-one modules

| Functional operation | Relevant systems | Caution in interpretation |
|---|---|---|
| Extract current perceptual features | Modality-specific sensory association cortices | Feature extraction is distributed and depends on the sensory task. |
| Activate conceptual features and category knowledge | Distributed temporal and modality-linked semantic systems | No single region stores every property of a concept. |
| Retrieve and use structured prior knowledge | Medial/orbitomedial prefrontal networks interacting with posterior and temporal regions | Schema use is a network-level process; precise subregional roles remain under study. |
| Bind features and relations from a specific event | Hippocampal formation and connected medial temporal regions | Hippocampal contributions depend on novelty, relational demands, and memory task. |
| Compare current information with episodic details | Hippocampal and broader memory networks | Mismatch responses may differ for episodic versus schematic expectations. |
| Select diagnostic cues and resolve conflict | Prefrontal control and attention networks | Control depends on goal and task difficulty. |
| Update predictions and actions | Interacting memory, prefrontal, and valuation/action systems | This is not a claim that schema inference is identical to reinforcement learning. |

---

# 7. Failure modes and protective mechanisms

1. **Premature categorization:** one vivid feature activates a schema too strongly. *Protection:* retain multiple candidates and weight cues by reliability and diagnosticity.
2. **Treating missing as absent:** an unobserved attribute is incorrectly used as negative evidence. *Protection:* preserve a separate unknown state.
3. **Schema-driven false detail:** a plausible but unseen feature is remembered as if it occurred. *Protection:* store provenance and inference status for every detail.
4. **Overgeneralization:** a broad schema is applied to an exceptional instance. *Protection:* use hierarchical confidence thresholds and an unknown/other category.
5. **Under-generalization:** a system refuses to transfer knowledge because the new case is not identical to a stored example. *Protection:* compare relational and diagnostic features, not only exact surface matches.
6. **Context neglect:** a schema is retrieved even though the current environment changes its applicability. *Protection:* condition schema activation and predictions on context.
7. **Correlated-cue double counting:** two highly related observations are treated as independent evidence. *Protection:* model covariance or group redundant features.
8. **Ignoring counterevidence:** contradictory information is explained away to preserve the first guess. *Protection:* explicitly score contradictory observations and update posterior confidence.
9. **Confusing sequence and causality:** repeated temporal order is taken as proof of causal influence. *Protection:* encode temporal and causal relations separately and require additional evidence for causal claims.
10. **Over-updating from one exception:** one unusual example corrupts a broadly reliable schema. *Protection:* preserve exceptions, uncertainty, and evidence counts; require replication or strong diagnostic evidence for structural changes.
11. **Uncalibrated certainty:** high confidence is assigned to a candidate even when the available cues cannot distinguish it from close alternatives. *Protection:* evaluate calibration, use abstention, and seek diagnostic evidence.
12. **Action based on category alone:** a plausible identification is treated as sufficient to choose a risky behavior. *Protection:* include action-specific utility and safety constraints.

---

# 8. Validation experiments and computational tests

The algorithm should be tested at multiple levels rather than judged only by whether it produces plausible explanations.

### Experiment A — Generalization to novel instances

Train participants or a model on several examples of a category, then present new examples that vary in irrelevant surface properties but preserve diagnostic structure. Measure classification accuracy, confidence, and transfer to unseen combinations. A useful computational baseline compares schema-based inference with nearest-exemplar retrieval.

### Experiment B — Partial-cue inference

Present incomplete or degraded inputs. Measure whether the model uses context and prior knowledge to recover likely interpretations without treating predictions as observations. Evaluate both accuracy and calibration.

### Experiment C — Schema conflict

Create cases where two familiar schemas share surface features but differ in a diagnostic relation. Test whether the algorithm preserves alternatives, searches for discriminating evidence, and avoids premature commitment.

### Experiment D — Source-memory protection

Show participants or a model a set of observed features, then ask later which properties were directly seen and which were merely consistent with the schema. Measure source-monitoring errors separately from category classification accuracy.

### Experiment E — Context-sensitive generalization

Hold the sensory features constant while changing the task or environment. Test whether candidate activation and inferred properties adjust appropriately.

### Experiment F — Prediction-error-driven updating

Provide a sequence of schema-consistent cases followed by controlled exceptions. Compare: (a) immediate global updating, (b) episodic storage only, and (c) confidence-weighted hierarchical updating. Evaluate which approach best predicts subsequent cases without catastrophic overgeneralization.

### Experiment G — Neural-network correspondence

In human studies, task contrasts can distinguish schema-consistent inference, schema-incongruent inference, retrieval of a specific episode, and simple perceptual matching. Use behavioral results alongside neuroimaging or electrophysiology; do not interpret activation in a region as proof that a single computational step occurs exclusively there.

### Suggested software metrics

- top-1 and hierarchical classification accuracy;
- negative log-likelihood and Brier score;
- expected calibration error;
- precision/recall for inferred relations;
- source-attribution accuracy;
- rate of false inferred-as-observed details;
- generalization accuracy on held-out categories and contexts;
- rate of unnecessary abstention and rate of unjustified certainty;
- schema-update stability after exceptions.

---

# 9. What is well supported and what remains a model choice

**Well-supported at a broad level:** prior knowledge and schemas influence memory, interpretation, and inference; hippocampal and prefrontal systems interact during schema-related learning and memory; new knowledge can be integrated with old knowledge; reconstruction can introduce schema-consistent distortions; and generalization allows behavior beyond directly encountered examples.

**More theory-dependent or task-specific:** the exact order of operations; whether a particular inference is best described as Bayesian prediction, pattern completion, or relational integration; the precise division of labor among medial prefrontal subregions; and the degree to which a given mismatch is computed by hippocampal versus non-hippocampal networks. Recent work suggests that mismatch responses can differ according to whether the violated expectation comes from a specific episodic memory or generalized schematic knowledge.

**Implementation rule:** treat equations as candidate computational mechanisms to test. Never write that the brain literally performs a particular software operation unless independent evidence supports that level of claim.

---

# 10. Selected scientific references

1. Varga, N. L., Morton, N. W., & Preston, A. R. (2024). *Schema, Inference, and Memory*. In **The Oxford Handbook of Human Memory**. Oxford University Press. https://doi.org/10.1093/oxfordhb/9780190917982.013.47
2. Bein, O., & Niv, Y. (2025). Schemas, reinforcement learning and the medial prefrontal cortex. **Nature Reviews Neuroscience, 26**, 141–157. https://doi.org/10.1038/s41583-024-00893-z
3. Eichenbaum, H. (2017). Prefrontal–hippocampal interactions in episodic memory. **Nature Reviews Neuroscience, 18**, 547–558. https://doi.org/10.1038/nrn.2017.74
4. Spens, E., & Burgess, N. (2024). A generative model of memory construction and consolidation. **Nature Human Behaviour, 8**, 526–543. https://doi.org/10.1038/s41562-023-01799-z
5. Varga, D. K., Raykov, P. P., Jefferies, E., Ben-Yakov, A., & Bird, C. M. (2025). Hippocampal mismatch signals are based on episodic memories and not schematic knowledge. **Proceedings of the National Academy of Sciences, 122**(34), e2503535122. https://doi.org/10.1073/pnas.2503535122
6. Cooper, S. E., Hennings, A. C., Bibb, S. A., Lewis-Peacock, J. A., & Dunsmoor, J. E. (2024). Semantic structures facilitate threat memory integration throughout the medial temporal lobe and medial prefrontal cortex. **Current Biology**. https://doi.org/10.1016/j.cub.2024.06.071
7. Frank, D., Montaldi, D., Wittmann, B., & Talmi, D. (2018). Beneficial and detrimental effects of schema incongruence on memory for contextual events. **Learning & Memory, 25**(8), 352–360. https://doi.org/10.1101/lm.047738.118

---

## Completion boundary

This file covers only **Point 10 — Schema-Guided Inference and Generalization**. It does not implement the next distinct topic, **analogical transfer between domains**. That will be handled as Point 11 only after the user requests continuation.
