# Point 06 — Detecting Commonalities Across Experiences to Form Concepts

**Source sequence:** *Learning, Memory, and Concept Formation.txt*  
**Source passage covered:** “Concepts and schemas emerge from these memory processes. A concept … is an abstract category distilled from many examples. The brain forms concepts by detecting commonalities across experiences …”  
**Scope boundary:** This file develops the commonality-detection and abstraction process only. The following source topics—prototype versus exemplar representations, distributed semantic attributes and hub regions, schema/role-slot formation, inference/generalization, and analogical transfer—are reserved for their own later points.

---

## 1. Objective and biological interpretation

The objective is to construct a biologically informed computational account of how a brain can move from several distinct experiences to a representation of what those experiences have in common. Consider encounters with several birds: a sparrow on a branch, a robin in a garden, and a pigeon in a square. Each episode differs in location, time, viewpoint, size, colour, movement, and background. Nevertheless, some recurring properties and relations—such as feathers, wings, a beak, characteristic movement, and particular sounds—can become associated with a common category.

The central computational problem is **invariance extraction under variation**. The learner must detect properties that recur across relevant instances, distinguish them from accidental details, and preserve enough information to avoid grouping unrelated things together. This is not simply averaging every sensory input. Some features are shared but uninformative; other features are infrequent but highly diagnostic. A reliable concept therefore depends on recurrence, feature relationships, contrast with other categories, attention, prior knowledge, prediction, and correction.

Biologically, concept learning is not assigned to one isolated “concept centre.” It can involve interactions among sensory and association cortices, medial-temporal structures including the hippocampus, prefrontal and parietal networks, and—when learning is guided by feedback or reward—corticostriatal and dopaminergic systems. The precise balance depends on the task and the kind of concept being acquired. The hippocampus can preserve relations among specific episodes and help integrate information across them; cortical systems can gradually develop more stable shared representations. These are complementary contributions, not a claim that every concept must follow one identical anatomical route.

### Evidence labels used in this algorithm

- **Established biological principle:** supported broadly by converging evidence, such as distributed plasticity, hippocampal relational memory, and interactions among memory, attention, perception, and learning systems.
- **Supported mechanistic account:** a biologically plausible interpretation backed by experimental findings, but not a complete or exclusive explanation.
- **Computational abstraction:** a mathematical operation useful for describing or simulating the process. It must not be interpreted as proof that neurons explicitly execute that equation.

The equations below are therefore explanatory models. Neural tissue uses electrochemical activity, synaptic plasticity, recurrent circuits, and population dynamics; it does not need to calculate named statistical quantities in a symbolic way.

---

## 2. Inputs, evolving state, and final output

### Inputs

1. A stream of episodes or encounters, each containing sensory and contextual information.
2. Candidate feature representations extracted from each episode: visual, auditory, tactile, motor, spatial, temporal, and other task-relevant properties.
3. Existing memories and prior category knowledge, which influence attention and interpretation.
4. Task goals, motivation, and feedback, where available.
5. Information about competing categories or background instances, needed to determine which recurring features are genuinely informative.

### Internal state

- An episodic record or partial trace for each encounter.
- A set of candidate features and relations associated with those encounters.
- Feature salience and reliability estimates.
- Estimates of recurrence, co-occurrence, variability, and diagnosticity.
- A provisional shared structure that changes as further experiences arrive.
- Prediction errors and confidence signals used to revise that structure.

### Output

A provisional abstract category representation: a structured pattern of features and relationships that captures recurring, useful regularities across the experiences while discounting irrelevant episode-specific variation. The result is not necessarily a single average vector. It may be distributed across neural populations and may preserve uncertainty, exceptions, and multiple levels of representation.

---

## 3. Integrated algorithm: 16 biological-computational steps

## Step 1 — Encode each encounter as a distinct experience

**Purpose:** Preserve the fact that each example occurred in a particular episode before trying to extract what the episodes share.

**Biological operations:** Sensory systems process the current input, while medial-temporal structures—especially hippocampal circuitry—can bind information about objects, attributes, relations, and context into an episode-specific representation. The hippocampus is important here because concept acquisition may require comparing experiences that happened at different times and places. Keeping some episode-specific structure prevents premature collapse of all examples into one undifferentiated trace.

The representation can include the encountered object, its features, the surrounding scene, temporal order, actions, and relevant outcomes. The exact information encoded depends on attention, task demands, novelty, and the learner’s prior knowledge. Not every feature is necessarily retained with equal strength.

**Computational interpretation:** For encounter `i`, construct an episode record:

\[
E_i = \{X_i, C_i, T_i, A_i, O_i\}
\]

where `X_i` is the set or vector of observed features, `C_i` is context, `T_i` is temporal information, `A_i` is action or interaction information, and `O_i` is any outcome or feedback.

A simple feature representation is:

\[
\mathbf{x}_i = [x_{i1},x_{i2},\ldots,x_{id}]
\]

The components may be graded activations rather than literal binary values.

**Possible mathematical operations:** feature encoding; concatenation or binding of feature indices; normalization; temporal tagging; weighted storage; and separation of episode identity from feature content.

**Important distinction:** This is a computational representation of episodic binding, not a claim that the hippocampus stores a literal computer record or a complete photographic copy.

## Step 2 — Extract candidate features through modality-specific processing

**Purpose:** Determine which properties of each encounter can be compared across episodes.

**Biological operations:** Early and intermediate sensory pathways extract structured information. For example, visual processing can represent shape, colour, contour, texture, and movement; auditory processing can represent pitch, rhythm, and spectral structure; somatosensory and motor systems can contribute tactile and action-related information. Higher association regions combine these properties into more complex representations.

Different encounters may expose different subsets of a category’s properties. A bird may be seen but not heard in one episode, heard but partly hidden in another, and observed flying in a third. A flexible concept-learning process cannot require every feature to be present in every episode.

**Computational interpretation:**

\[
\mathbf{x}_i = [x_{i1},x_{i2},\ldots,x_{id}], \qquad x_{ik}\in[0,1]
\]

Here, `x_ik` represents the estimated presence or strength of feature `k` in episode `i`. A missing or unobserved feature should be marked as unknown rather than automatically set to zero.

Let `m_ik` indicate whether feature `k` was observed:

\[
m_{ik}\in\{0,1\}
\]

**Possible mathematical operations:** feature detection; pooling across receptive fields; dimensionality reduction; normalization; temporal integration; and missing-data masking.

**Biological caution:** These calculations describe functional roles. Real sensory pathways use many interacting neuronal populations, and the exact feature code differs by modality and task.

## Step 3 — Select and weight features according to attention and current goals

**Purpose:** Avoid treating every detectable property as equally important.

**Biological operations:** Attention can amplify processing of task-relevant stimuli and features. Prefrontal and parietal control systems can bias sensory processing according to current goals, while learned salience and motivational state can influence what is selected. Attention is not an absolute gate: some regularities can be learned with limited deliberate awareness, but attention often affects what is integrated and how efficiently.

For example, while learning to recognize birds, the learner may prioritize wings, beaks, and characteristic movement over the colour of the fence behind the animal. If the task is to distinguish bird species by song, auditory properties receive greater weight.

**Computational interpretation:** Assign each feature an adaptive salience weight `a_k`:

\[
a_k = f(\text{goal},\text{attention},\text{prior knowledge},\text{salience})
\]

A normalized weighting scheme may be:

\[
\alpha_k = \frac{\exp(a_k)}{\sum_j \exp(a_j)}
\]

The softmax is a convenient model for competition among candidate features; it is not asserted to be the literal neuronal operation.

**Possible mathematical operations:** gain modulation; competitive normalization; weighted summation; thresholding; and dynamic reweighting as goals change.

**Failure mode:** If a background detail receives excessive weight, the learner may form an overly narrow concept—for example, treating “birds” as “small animals seen near trees.”

## Step 4 — Reactivate and retrieve related prior episodes

**Purpose:** Put the current encounter into relation with earlier encounters rather than processing it in isolation.

**Biological operations:** The hippocampus can support pattern completion: a partial cue may reactivate aspects of a previously encoded episode. Reinstatement of related memories can make information from separate learning events jointly available. Prefrontal systems may guide retrieval toward episodes relevant to the current learning goal. Retrieval can also bias subsequent encoding by emphasizing features that matter for an emerging regularity.

When a new bird is encountered, a partial cue such as its silhouette or call may reactivate memories of earlier birds. Those memories supply comparison cases that are not physically present.

**Computational interpretation:** For a current episode `E_t`, retrieve a set of related traces:

\[
R_t = \operatorname{Retrieve}(E_t, M, G)
\]

where `M` is stored memory and `G` represents current goals or context. Similarity-based retrieval can be illustrated as:

\[
s_i = \frac{\mathbf{x}_t\cdot\mathbf{x}_i}{\|\mathbf{x}_t\|\,\|\mathbf{x}_i\|+\epsilon}
\]

The retrieved set may include the highest-scoring traces, traces associated through relations, and memories selected by task relevance.

**Possible mathematical operations:** cue matching; similarity scoring; associative activation spread; thresholded retrieval; and context-dependent selection.

**Important caution:** Hippocampal pattern completion is a neural memory phenomenon, not necessarily a literal vector cosine calculation. The equation is one possible software-level approximation.

## Step 5 — Align comparable features despite changes in viewpoint and context

**Purpose:** Recognize that two episodes can contain the same underlying property even when it appears differently.

**Biological operations:** Perceptual systems exhibit some tolerance to changes in viewpoint, lighting, scale, position, and acoustic conditions. Association processes can link different sensory manifestations to a common property. Memory comparison may help identify that two differently observed objects share a relevant structure.

A wing seen from above and a wing seen from the side produce different retinal patterns, but they may still be recognized as the same type of part. Likewise, a bird call can vary in loudness and recording conditions while preserving a characteristic pattern.

**Computational interpretation:** Map raw observations into a more comparable feature space:

\[
\mathbf{z}_i = g(\mathbf{x}_i, C_i)
\]

where `g` discounts nuisance variation while retaining category-relevant information. A similarity metric may then compare `z_i` rather than raw sensory inputs.

One illustrative weighted distance is:

\[
d_w(i,j)=\sqrt{\sum_k \alpha_k(z_{ik}-z_{jk})^2}
\]

**Possible mathematical operations:** coordinate transformation; invariance learning; feature normalization; alignment; and weighted distance estimation.

**Failure mode:** Excessive invariance can remove meaningful distinctions. For example, treating all flying objects as equivalent would merge birds, insects, and aircraft. The system must preserve dimensions that matter for categorization.

## Step 6 — Detect recurring features across multiple episodes

**Purpose:** Identify which properties repeatedly occur among related examples.

**Biological operations:** Repeated exposure can strengthen shared patterns through plasticity in sensory and association cortex. Hippocampal reactivation and cross-episode integration may help make features from separate episodes jointly available. With repeated examples, recurring attributes can become more strongly represented than incidental details.

The learner may notice that different birds repeatedly exhibit wings and beaks, although their colours, sizes, and locations vary. Recurrence is informative, but frequency alone is insufficient: a feature common to nearly every object in the environment may not distinguish the target category.

**Computational interpretation:** Estimate how often a feature appears in the relevant episode set:

\[
\hat{P}(f_k\mid C)=\frac{\sum_i m_{ik}x_{ik}}{\sum_i m_{ik}+\epsilon}
\]

This estimates feature prevalence among observed category candidates. If observations are binary, the expression reduces to a proportion of episodes in which the feature was present.

An incremental update can avoid recomputing from the full history:

\[
q_k^{(t)}=(1-\eta)q_k^{(t-1)}+\eta x_{tk}
\]

where `η` controls how strongly the newest observation affects the running estimate. A biological process need not use this exact learning rule; it is an implementable approximation of gradual updating.

**Possible mathematical operations:** counting; accumulation; moving averages; frequency estimation; and synaptic strengthening with repeated co-activation.

**Caution:** The learner must distinguish absent from unobserved. A bird hidden behind a tree does not provide evidence that it lacks wings.

## Step 7 — Detect co-occurrence and relations among features

**Purpose:** Learn that certain properties tend to occur together, rather than treating features as an unrelated checklist.

**Biological operations:** Neural populations can become associated when their activity repeatedly co-occurs within episodes. Hippocampal relational binding can preserve links among elements, while cortical plasticity can reflect recurring feature combinations. The relevant relationship may be spatial, temporal, functional, causal, or part-based.

For example, wings, feathers, and a beak may co-occur in many bird encounters. These features together can be more informative than any one feature considered independently. A concept can therefore be supported by a structured pattern of relations, not merely a list of frequent attributes.

**Computational interpretation:** For two features `f_a` and `f_b`, estimate co-occurrence:

\[
P(f_a,f_b\mid C)
\]

Compare it with the probability expected if the two features were independent:

\[
\operatorname{PMI}(f_a,f_b)=\log\frac{P(f_a,f_b)}{P(f_a)P(f_b)}
\]

Positive pointwise mutual information indicates that the features co-occur more than expected under independence. Other measures may be preferable when counts are sparse.

A relation can be represented as a graph:

\[
G_C=(V_C,E_C)
\]

where nodes represent features and edges represent learned associations or relations.

**Possible mathematical operations:** co-activation accumulation; covariance estimation; conditional probability; mutual information; graph construction; and associative weight updates.

**Failure mode:** Spurious co-occurrence can occur because two features share a background cause. The learner should test whether the relationship survives across different contexts.

## Step 8 — Separate category-relevant regularities from background coincidences

**Purpose:** Avoid turning accidental correlations into defining properties.

**Biological operations:** Comparison among different objects and contexts helps reveal which features are characteristic of one category and which are widespread or incidental. Attention and prior expectations can influence these comparisons. Prefrontal systems may support rule evaluation and the use of contrastive information; learning systems can adjust feature weights when a candidate distinction repeatedly fails.

If every bird encountered so far was near a tree, “near a tree” is frequent in the sample but is not a reliable bird-defining property. If feathers occur across multiple bird examples but not in the non-bird comparison set, feathers are more category-informative.

**Computational interpretation:** Estimate both within-category and background prevalence:

\[
P(f_k\mid C), \qquad P(f_k\mid \neg C)
\]

A simple diagnosticity score is a log-likelihood ratio:

\[
D_k=\log\frac{P(f_k\mid C)+\epsilon}{P(f_k\mid\neg C)+\epsilon}
\]

A feature can be frequent but weakly diagnostic if it is equally common outside the category. A rarer feature may be strongly diagnostic if it reliably separates the target from alternatives.

Information gain can also be used to model how much observing a feature reduces uncertainty about category membership:

\[
IG(f_k)=H(C)-H(C\mid f_k)
\]

where `H` is entropy.

**Possible mathematical operations:** contrastive comparison; conditional probability; log ratios; entropy; information gain; and feature reweighting.

**Caution:** The brain may learn diagnosticity through distributed prediction and feedback mechanisms rather than explicitly calculating information gain.

## Step 9 — Estimate which regularities are stable and which vary naturally

**Purpose:** Prevent the concept from being defined by properties that change substantially across members, conditions, or time.

**Biological operations:** A category’s examples are not identical. The learner must tolerate natural variation while retaining category structure. Repeated exposure under different viewpoints and contexts can help distinguish stable properties from variable ones. Cortical representations can become less sensitive to irrelevant variation while remaining sensitive to meaningful distinctions.

For example, bird size and colour vary widely. The learner should not reject a bird merely because it differs from a remembered small, brown example. At the same time, some variations may signal that an example belongs to a different species or category, depending on the task.

**Computational interpretation:** Estimate within-category variability for feature `k`:

\[
\sigma_k^2=\frac{1}{N_k-1}\sum_i m_{ik}(x_{ik}-\bar{x}_k)^2
\]

A candidate feature with low within-category variability may be stable, but stability alone does not make it diagnostic. Combine stability with the contrastive score from Step 8.

A simple combined weight could be:

\[
w_k = \alpha_k\,D_k\,r_k
\]

where `α_k` is attention/task relevance, `D_k` is diagnosticity, and `r_k` is reliability. This is an illustrative model, not a known neural formula.

**Possible mathematical operations:** variance estimation; robust statistics; reliability weighting; outlier detection; and adaptive normalization.

**Failure mode:** If the learner treats every unusual member as noise, it may erase legitimate subtypes. If it treats every unusual member as a new category, it may over-fragment knowledge.

## Step 10 — Integrate information across episodes into a provisional shared structure

**Purpose:** Build a representation that captures common information without erasing all episodic differences.

**Biological operations:** The hippocampus may support integration across overlapping memories, and interactions with medial prefrontal systems can help extract regularities or latent structure. Cortical learning gradually strengthens patterns that recur across episodes. The result can be a provisional representation that is updated as new evidence arrives.

The process is iterative. A new encounter may confirm the current category structure, add a new feature, or reveal that an apparently common feature was accidental. Memory integration should therefore remain revisable.

**Computational interpretation:** Maintain a candidate latent representation `μ_C` or a more general parameter set `θ_C`. A simple incremental estimate is:

\[
\mu_C^{(t)}=\mu_C^{(t-1)}+\frac{1}{n_t}(\mathbf{z}_t-\mu_C^{(t-1)})
\]

where `n_t` is the number of relevant observations incorporated. For uncertain or noisy observations, use a confidence-weighted update:

\[
\mu_C^{(t)}=\mu_C^{(t-1)}+\eta_t r_t(\mathbf{z}_t-\mu_C^{(t-1)})
\]

where `r_t` is estimated reliability and `η_t` is a learning rate.

**Possible mathematical operations:** incremental integration; weighted estimation; recurrent activation; parameter updating; and confidence-weighted accumulation.

**Biological caution:** A mean vector is only one possible formalization. Neural concepts may be distributed, relational, multimodal, context-sensitive, and not reducible to a single centroid.

## Step 11 — Compress episode-specific details while preserving useful distinctions

**Purpose:** Reduce the burden of representing every episode independently, yet retain details that matter for discrimination and later retrieval.

**Biological operations:** Repeated learning can strengthen shared regularities while episodic traces retain some unique context. Prefrontal–hippocampal interactions have been proposed to support memory integration and abstraction by emphasizing common structure. Cortical representations can gradually become more stable and less dependent on all the details of a single encounter.

Compression must be selective. The exact location of one bird may be irrelevant to the broad concept “bird,” but a distinctive feature may be essential for identifying a rare subtype. The system should not discard all exceptions or context indiscriminately.

**Computational interpretation:** Seek a representation that minimizes unnecessary detail while preserving predictive and discriminative information. One explanatory objective is:

\[
J(\theta_C)=L_{\text{prediction}}(\theta_C)+\lambda\,C(\theta_C)
\]

where `L_prediction` measures how poorly the shared representation accounts for useful observations and `C` penalizes unnecessary complexity. The parameter `λ` controls the trade-off.

This resembles regularization or minimum-description-length reasoning. It is a model of the functional trade-off, not evidence that the brain computes an explicit compression objective.

**Possible mathematical operations:** dimensionality reduction; regularization; feature selection; information bottleneck approximations; and abstraction over repeated instances.

**Failure mode:** Over-compression produces a vague concept that merges different categories; under-compression leaves the learner dependent on individual episodes and prevents useful generalization.

## Step 12 — Stabilize common feature relationships through gradual cortical plasticity

**Purpose:** Allow shared regularities to become increasingly available beyond the immediate retrieval of the individual episodes.

**Biological operations:** Repeated co-activation can alter synaptic efficacy in cortical networks. Different sensory and association regions contribute feature-specific information, while higher-order networks help coordinate and use those representations. Plasticity is distributed and can depend on attention, task structure, feedback, and interactions with memory systems.

The cortical contribution is gradual rather than a single event. With repeated exposure, a new encounter can activate a pattern that reflects recurring features across many examples. This does not require the hippocampus to become irrelevant; hippocampal involvement can remain important when detailed episode retrieval, rapid relational learning, or flexible recombination is needed.

**Computational interpretation:** A simplified associative update is:

\[
\Delta w_{ij}=\eta\,x_i y_j
\]

A more biologically inspired learning rule may include pre- and post-synaptic activity, eligibility traces, neuromodulatory gating, and decay or normalization. For example:

\[
\Delta w_{ij}=\eta\,e_{ij}\,m_t
\]

where `e_ij` is an eligibility signal derived from recent activity and `m_t` is a modulatory factor associated with outcome or learning state.

**Possible mathematical operations:** synaptic weight updates; temporal integration; eligibility traces; normalization; and gradual reinforcement of shared activation patterns.

**Caution:** These equations are families of computational models. Biological long-term potentiation and depression involve complex cellular mechanisms and are not captured fully by a one-line rule.

## Step 13 — Use the provisional concept to predict features of a new instance

**Purpose:** Test whether the emerging shared structure captures meaningful regularities beyond the training episodes.

**Biological operations:** Concepts support expectations about unfamiliar instances. A new object activates candidate features and can trigger retrieval of related episodes or category knowledge. Prediction can influence perception and attention, while the mismatch between expected and observed information can guide further learning.

For example, after several bird encounters, the learner may expect a newly seen bird-like animal to have wings and a beak. The prediction must remain probabilistic: an object may resemble a bird but lack one expected property, or it may be an unusual bird that violates a typical expectation.

**Computational interpretation:** Estimate the likelihood of a new observation under the candidate concept:

\[
P(\mathbf{z}_{new}\mid C,\theta_C)
\]

A simple class decision model may compare:

\[
P(C\mid\mathbf{z}_{new})\propto P(\mathbf{z}_{new}\mid C)P(C)
\]

This Bayesian expression captures the influence of evidence and prior expectation. It does not imply that the brain consciously evaluates symbolic probabilities.

**Possible mathematical operations:** predictive activation; likelihood estimation; Bayesian updating; similarity comparison; and confidence estimation.

## Step 14 — Compute mismatch and revise the common structure when predictions fail

**Purpose:** Prevent an early hypothesis from becoming permanently fixed when new evidence contradicts it.

**Biological operations:** Prediction error and feedback can influence learning. A surprising example may trigger renewed attention, memory retrieval, and revision of the emerging category. The hippocampus may help link the new episode to related experiences, while prefrontal and other learning systems help evaluate whether the discrepancy is noise, an exception, or evidence that the concept should be divided or reorganized.

If a newly encountered animal has wings but is clearly an insect rather than a bird, the system should not simply strengthen the rule “wings means bird.” It should consider other diagnostic features and the broader pattern of evidence.

**Computational interpretation:** Define a prediction error:

\[
\delta_t=\mathbf{z}_t-\hat{\mathbf{z}}_t
\]

A scalar error measure may be:

\[
E_t=\delta_t^T\Sigma^{-1}\delta_t
\]

where `Σ` represents expected variability. Large error can trigger additional retrieval, slower updating, or reconsideration of the category boundary.

An adaptive learning rate may be:

\[
\eta_t=\eta_{min}+ (\eta_{max}-\eta_{min})\,g(E_t)
\]

where `g` maps prediction error to a bounded adjustment. This is one possible computational control scheme, not a literal neural control law.

**Possible mathematical operations:** subtraction; residual estimation; uncertainty normalization; threshold detection; and error-modulated learning.

**Failure modes:** Treating every surprising example as noise creates rigidity; treating every surprise as proof of a new category creates instability. The learner must consider reliability, context, and repeated counterexamples.

## Step 15 — Consolidate and index the emerging concept for future retrieval

**Purpose:** Make the shared structure efficiently accessible when related situations occur again.

**Biological operations:** Repeated learning can strengthen connections among relevant cortical representations. Associative links can connect the developing concept to sensory features, related memories, actions, language, and goals. The hippocampus can continue to support flexible retrieval and relational recombination, while cortical networks increasingly support stable semantic access.

A useful concept is not an isolated stored label. It is embedded in a network of associations and can be activated by different cues. Seeing a characteristic shape, hearing a familiar call, or encountering a relevant word may all contribute to retrieval, depending on the learner’s history.

**Computational interpretation:** Create an index from cue features to the current concept parameters:

\[
\operatorname{Index}(q)\rightarrow\{C,\theta_C,\text{related traces}\}
\]

Retrieval strength can be modeled as a weighted combination of cue match, recency, frequency, and relevance:

\[
A(C\mid q)=\beta_1S(q,C)+\beta_2R(C)+\beta_3F(C)+\beta_4G(C)
\]

The coefficients are task-dependent modeling parameters, not known fixed neural constants.

**Possible mathematical operations:** associative indexing; activation accumulation; thresholded access; and strengthening or weakening of connections through use.

## Step 16 — Produce a revisable abstract concept representation

**Purpose:** Deliver the result of commonality extraction while preserving uncertainty and the ability to learn from future examples.

**Biological operations:** The resulting knowledge can be distributed across sensory and association cortices and linked to episodic and contextual memories. The learner can now recognize a new example as related to previous instances without requiring exact repetition of one remembered encounter. The emerging representation remains plastic: new evidence can refine it, differentiate subcategories, or expose mistaken assumptions.

**Computational interpretation:** The output can be represented as:

\[
C=\{F_{shared},R_{shared},W,\mathcal{U},M_{support}\}
\]

where:

- `F_shared` contains recurring and relevant features;
- `R_shared` contains important relationships among features;
- `W` contains adaptive weights reflecting relevance, reliability, and diagnosticity;
- `U` records uncertainty, variability, and exceptions;
- `M_support` references the episodes or evidence supporting the current concept.

This representation is intentionally broader than a single prototype vector. The next source point addresses alternative representational theories, including prototype and exemplar accounts.

**Possible mathematical operations:** structured aggregation; uncertainty retention; graph or tensor representation; confidence calibration; and incremental updating.

---

## 4. End-to-end pseudocode

The following pseudocode is an implementable abstraction of the 16 steps. It is not a literal simulation of every neuron or synapse.

```python
from dataclasses import dataclass, field
from math import log
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class Episode:
    episode_id: str
    features: Dict[str, Optional[float]]
    context: Dict[str, Any] = field(default_factory=dict)
    relations: List[Tuple[str, str, str]] = field(default_factory=list)
    reliability: float = 1.0


@dataclass
class ConceptCandidate:
    feature_prevalence: Dict[str, float] = field(default_factory=dict)
    feature_variance: Dict[str, float] = field(default_factory=dict)
    diagnosticity: Dict[str, float] = field(default_factory=dict)
    feature_weights: Dict[str, float] = field(default_factory=dict)
    relation_strength: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    uncertainty: Dict[str, float] = field(default_factory=dict)
    supporting_episode_ids: List[str] = field(default_factory=list)


def safe_probability(value: float, eps: float = 1e-6) -> float:
    return min(1.0 - eps, max(eps, value))


def extract_comparable_features(episode: Episode) -> Dict[str, float]:
    """Return observed features; None means unobserved, not absent."""
    return {
        name: float(value)
        for name, value in episode.features.items()
        if value is not None
    }


def retrieve_related_episodes(
    current: Episode,
    memory: List[Episode],
    limit: int = 20,
) -> List[Episode]:
    """Simple overlap retrieval; a production model needs richer representations."""
    current_features = extract_comparable_features(current)
    scored = []
    for past in memory:
        past_features = extract_comparable_features(past)
        shared = set(current_features) & set(past_features)
        union = set(current_features) | set(past_features)
        score = len(shared) / max(1, len(union))
        scored.append((score * past.reliability, past))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [episode for score, episode in scored[:limit] if score > 0]


def estimate_prevalence(episodes: List[Episode]) -> Dict[str, float]:
    """Estimate feature prevalence using only episodes where the feature was observed."""
    observed: Dict[str, List[float]] = {}
    for episode in episodes:
        for name, value in extract_comparable_features(episode).items():
            observed.setdefault(name, []).append(value)
    return {
        name: sum(values) / len(values)
        for name, values in observed.items()
        if values
    }


def estimate_variance(episodes: List[Episode]) -> Dict[str, float]:
    observed: Dict[str, List[float]] = {}
    for episode in episodes:
        for name, value in extract_comparable_features(episode).items():
            observed.setdefault(name, []).append(value)

    result = {}
    for name, values in observed.items():
        if len(values) < 2:
            result[name] = 0.0
            continue
        mean = sum(values) / len(values)
        result[name] = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return result


def estimate_diagnosticity(
    target_episodes: List[Episode],
    comparison_episodes: List[Episode],
    eps: float = 1e-6,
) -> Dict[str, float]:
    target = estimate_prevalence(target_episodes)
    comparison = estimate_prevalence(comparison_episodes)
    features = set(target) | set(comparison)
    scores = {}
    for name in features:
        p_target = safe_probability(target.get(name, 0.0), eps)
        p_other = safe_probability(comparison.get(name, 0.0), eps)
        scores[name] = log(p_target / p_other)
    return scores


def estimate_relations(episodes: List[Episode]) -> Dict[Tuple[str, str, str], float]:
    counts: Dict[Tuple[str, str, str], float] = {}
    totals: Dict[Tuple[str, str, str], float] = {}
    for episode in episodes:
        for relation in episode.relations:
            counts[relation] = counts.get(relation, 0.0) + episode.reliability
        for relation in set(episode.relations):
            totals[relation] = totals.get(relation, 0.0) + episode.reliability
    return {
        relation: min(1.0, count / max(1.0, len(episodes)))
        for relation, count in counts.items()
    }


def build_concept_candidate(
    target_episodes: List[Episode],
    comparison_episodes: List[Episode],
    attention_weights: Optional[Dict[str, float]] = None,
) -> ConceptCandidate:
    if not target_episodes:
        raise ValueError("At least one target episode is required")

    attention_weights = attention_weights or {}
    prevalence = estimate_prevalence(target_episodes)
    variance = estimate_variance(target_episodes)
    diagnosticity = estimate_diagnosticity(target_episodes, comparison_episodes)

    weights = {}
    uncertainty = {}
    for feature, frequency in prevalence.items():
        attention = max(0.0, attention_weights.get(feature, 1.0))
        reliability = 1.0 / (1.0 + max(0.0, variance.get(feature, 0.0)))
        diagnostic = max(0.0, diagnosticity.get(feature, 0.0))
        weights[feature] = attention * reliability * diagnostic * frequency
        uncertainty[feature] = variance.get(feature, 0.0)

    relations = estimate_relations(target_episodes)
    return ConceptCandidate(
        feature_prevalence=prevalence,
        feature_variance=variance,
        diagnosticity=diagnosticity,
        feature_weights=weights,
        relation_strength=relations,
        uncertainty=uncertainty,
        supporting_episode_ids=[e.episode_id for e in target_episodes],
    )


def evaluate_new_episode(
    episode: Episode,
    concept: ConceptCandidate,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    observed = extract_comparable_features(episode)
    numerator = 0.0
    denominator = 0.0
    for feature, weight in concept.feature_weights.items():
        if feature in observed:
            numerator += weight * observed[feature]
            denominator += abs(weight)
    score = numerator / denominator if denominator else 0.0
    return {
        "score": score,
        "provisional_match": score >= threshold,
        "observed_features": list(observed),
    }


def update_from_new_evidence(
    concept: ConceptCandidate,
    new_episode: Episode,
    learning_rate: float = 0.1,
) -> None:
    """A deliberately simple incremental update; category reassessment may be needed."""
    if not 0.0 < learning_rate <= 1.0:
        raise ValueError("learning_rate must be in (0, 1]")

    for feature, value in extract_comparable_features(new_episode).items():
        old = concept.feature_prevalence.get(feature, value)
        concept.feature_prevalence[feature] = (1 - learning_rate) * old + learning_rate * value
        concept.supporting_episode_ids.append(new_episode.episode_id)

    for relation in new_episode.relations:
        old = concept.relation_strength.get(relation, 0.0)
        concept.relation_strength[relation] = min(
            1.0, (1 - learning_rate) * old + learning_rate
        )


def concept_learning_pipeline(
    target_episodes: List[Episode],
    comparison_episodes: List[Episode],
    attention_weights: Optional[Dict[str, float]] = None,
) -> ConceptCandidate:
    """Construct a provisional concept from recurring, contrastive evidence."""
    if len(target_episodes) < 2:
        raise ValueError("Commonality detection requires multiple examples")

    # 1–5: preserve episodes, extract features, weight attention, retrieve,
    # and align comparable features. Retrieval is represented by the input set.
    # 6–9: estimate prevalence, relations, diagnosticity, and variability.
    # 10–12: integrate and compress evidence into a stable candidate.
    concept = build_concept_candidate(
        target_episodes,
        comparison_episodes,
        attention_weights=attention_weights,
    )

    # 13–16: use the candidate to predict, evaluate new evidence, revise,
    # and retain an indexed, revisable representation.
    return concept
```

**Implementation note:** This pseudocode demonstrates data flow and illustrative statistics. It is not a validated neural simulator. A more realistic model would require modality-specific encoders, recurrent dynamics, biologically constrained plasticity, memory retrieval, uncertainty-aware feature alignment, temporal replay, and task-dependent learning-system interactions.

---

## 5. Mathematical operations inventory

| Computational operation | Why it is useful in this model | Biological interpretation / limitation |
|---|---|---|
| Feature encoding and normalization | Makes properties from different episodes comparable | Sensory populations and association circuits form distributed activity patterns; not literal symbolic vectors |
| Attention-weighted gain | Prioritizes properties relevant to current goals | Attention and control networks modulate processing; no single fixed weight vector is implied |
| Similarity and cue matching | Retrieves related episodes and compares examples | A model of associative retrieval and perceptual similarity, not a claim about exact neural arithmetic |
| Frequency and conditional-probability estimation | Detects recurring properties | Repeated co-activation and plasticity can encode regularity; explicit probability calculation is not required |
| Co-occurrence, covariance, and mutual information | Detects relationships among features | Captures statistical structure that may be represented in distributed circuits |
| Within-/between-category comparison | Distinguishes stable features from diagnostic features | The brain can learn from contrastive examples and feedback; exact statistics are model choices |
| Variance and reliability weighting | Tolerates natural variation and noisy observations | Neural systems are sensitive to uncertainty and reliability, but implementations are distributed |
| Incremental weighted updating | Integrates episodes over time | A simplified account of gradual plasticity and memory integration |
| Regularization / compression | Balances abstraction against loss of useful distinctions | A computational analogy for selective abstraction, not a literal neural objective function |
| Prediction and prediction error | Tests whether the emerging concept generalizes | Prediction-related activity and error-dependent learning are supported broadly; specific formula is illustrative |
| Associative indexing | Makes the concept accessible from multiple cues | Models learned connectivity and distributed retrieval pathways |

---

## 6. Functional roles of major systems

The following is a functional map, not a rigid one-region/one-step assignment.

| Neural system | Candidate contribution to this point | Important qualification |
|---|---|---|
| Sensory cortices | Extract modality-specific properties and become tuned through experience | The exact representation depends on modality and learned task |
| Association cortices | Combine features into more complex representations and support cross-modal regularities | Many distributed regions participate; no single association area performs all abstraction |
| Hippocampal formation | Rapidly binds episode details, supports relational memory and retrieval, and may integrate overlapping experiences | Hippocampal involvement varies with task and learning stage; concept acquisition is not exclusively hippocampal |
| Medial prefrontal cortex | May guide retrieval, integrate across experiences, and emphasize latent regularities or schema-relevant structure | Functions are network-dependent and still under investigation |
| Lateral prefrontal / parietal control networks | Support goal maintenance, attention, rule evaluation, and working-memory demands | Less necessary for some implicit or simple statistical regularities |
| Basal ganglia and dopaminergic systems | Contribute to feedback-guided, reward-related, and procedural aspects of category learning | Their role depends on task demands; not every concept is reward-trained |
| Distributed cortical networks | Gradually represent recurring properties and support stable access to learned regularities | A concept is not necessarily stored in one location or as one prototype |

---

## 7. Biological caveats and limits of the algorithm

1. **Commonality is not mere frequency.** A property can occur often because it belongs to the background context, not because it defines the category. Contrastive examples and task-relevant diagnosticity matter.
2. **The brain does not need to construct an explicit table of features.** Feature vectors, probability estimates, graphs, and equations are software abstractions that make the hypothesized computation testable.
3. **Concept learning can be incremental and predictive.** Categorization may influence perception from early processing stages through feedback, rather than appearing only after a completed sensory analysis. A recent 2026 perspective argues for this pervasive role of categorization; it is a theoretical synthesis, not a settled replacement for all stage-based accounts.
4. **Hippocampal and cortical roles overlap and interact.** It is misleading to say that the hippocampus only stores episodes while cortex alone forms concepts. Evidence suggests hippocampal memory integration can contribute directly to concept learning, while cortical plasticity supports more gradual representations.
5. **The same person can use different learning mechanisms in different tasks.** Explicit rules, similarity-based learning, implicit statistical learning, feedback, and memory-based integration can make different contributions.
6. **Abstraction has a bias–variance trade-off.** Too little abstraction leads to memorizing each example separately; too much abstraction erases important distinctions and creates false categories.
7. **The output should remain revisable.** A useful concept incorporates uncertainty and can change when new examples, context, or feedback contradict the current structure.

---

## 8. Suggested validation experiments for a computational implementation

A Python model based on this algorithm should be evaluated against more than its ability to reproduce the training examples.

### Test A — Held-out generalization

Train the model using only a subset of category members. Test whether it recognizes new members with different incidental features. Compare performance against a nearest-example baseline and a frequency-only baseline.

### Test B — Background-confound rejection

Place the same background feature in most training examples, but also in non-category examples. The model should not assign that feature a high diagnostic weight merely because it is frequent.

### Test C — Missing observations

Hide some features in individual examples. The system should distinguish unknown features from known absences and should not penalize an example for an unobserved attribute.

### Test D — Natural within-category variation

Vary size, colour, viewpoint, sound level, or other nuisance dimensions. Test whether the concept remains stable while preserving features that distinguish neighboring categories.

### Test E — Counterexample-driven revision

Introduce repeated, reliable examples that conflict with the current candidate. Measure whether the model revises its weights or reorganizes its representation without abandoning the concept after one noisy observation.

### Test F — Feature conjunctions and relations

Construct examples in which individual features are weakly diagnostic but their conjunction or relation is informative. Test whether the model can represent co-occurrence and relational structure rather than relying on independent feature counts.

### Test G — Learning trajectory

Measure accuracy and representation stability as examples accumulate. A suitable model should show gradual improvement, calibrated uncertainty, and a sensible response to changes in learning rate or evidence reliability.

### Test H — Ablation of memory integration

Disable cross-episode retrieval or relational integration in the model. Compare whether concept learning becomes less efficient when examples overlap in partial features but differ in context. This is a computational ablation, not direct proof of a specific brain-region function.

---

## 9. Scientific grounding and further reading

The following sources support the general mechanisms and identify important theoretical qualifications:

1. **Seger, C. A., & Miller, E. K. (2010). “Category learning in the brain.” *Annual Review of Neuroscience*, 33, 203–219.** Reviews the distributed neural systems involved in category learning, including neocortex, medial temporal lobe, basal ganglia, and midbrain systems, and discusses the balance between rapid learning and gradual abstraction. https://doi.org/10.1146/annurev.neuro.051508.135546
2. **Schlichting, M. L., & Preston, A. R. (2015). “Memory integration: neural mechanisms and implications for behavior.” *Current Opinion in Behavioral Sciences*, 1, 1–8.** Discusses how information from separate experiences can be integrated and how hippocampal–prefrontal interactions may support generalization. https://doi.org/10.1016/j.cobeha.2014.07.005
3. **Schlichting, M. L., & Preston, A. R. (2017). “Memory integration constructs maps of space, time, and concepts.” *Current Opinion in Behavioral Sciences*, 17, 161–168.** Reviews evidence that integrating across experiences can support conceptual structure and emphasizes common and distinguishing features. https://pmc.ncbi.nlm.nih.gov/articles/PMC5600494/
4. **Schapiro, A. C., & colleagues / related review (2018). “Building concepts one episode at a time: The hippocampus and concept formation.”** Presents the Episodes-to-Concepts framework and reviews memory integration, attentional biasing, and memory-based prediction error in concept acquisition. https://pmc.ncbi.nlm.nih.gov/articles/PMC5803467/
5. **Conway, C. M. (2020). “How does the brain learn environmental structure? Ten core principles for understanding the neurocognitive mechanisms of statistical learning.” *Neuroscience & Biobehavioral Reviews*, 112, 279–299.** Reviews modality-specific learning, cortical plasticity, attention, and the mechanisms supporting statistical learning. https://doi.org/10.1016/j.neubiorev.2020.01.032
6. **Mack, M. L., Love, B. C., & Preston, A. R. (2016/2018 conceptual work on category learning and memory integration).** Related studies investigate how hippocampal and prefrontal systems integrate examples and support generalization. See the reviews above for entry points.
7. **Bowman, C. R., Zeithamova, D., and colleagues (2020). “Tracking prototype and exemplar representations in the brain across learning.” *eLife*, 9, e59360.** Shows that distinct levels of category representation can be present during learning; it is particularly relevant to the next point on prototype and exemplar theories. https://doi.org/10.7554/eLife.59360
8. **Barrett, L. F., & Miller, E. K. (2026). “Categorization is ‘baked’ into the brain.” *Nature Reviews Neuroscience*.** A recent perspective arguing that categorization can shape processing from early stages through predictive feedback. It should be treated as a contemporary theoretical perspective rather than a final consensus. https://doi.org/10.1038/s41583-026-01036-2

---

## 10. Final algorithmic summary

```text
FOR each new encounter:
    1. Preserve an episode-specific memory trace.
    2. Extract candidate properties across relevant sensory modalities.
    3. Weight properties according to attention, goals, and current relevance.
    4. Retrieve related past episodes using available cues.
    5. Align comparable properties despite viewpoint and context variation.
    6. Estimate which properties recur across related examples.
    7. Detect co-occurrence and relationships among recurring properties.
    8. Compare target examples with alternative categories and background contexts.
    9. Estimate reliability, variability, and uncertainty for each property.
   10. Integrate overlapping evidence into a provisional shared structure.
   11. Discount incidental episode details while preserving useful distinctions.
   12. Strengthen recurring patterns through gradual distributed plasticity.
   13. Use the candidate structure to predict properties of unfamiliar examples.
   14. Evaluate mismatches and revise the representation when evidence warrants.
   15. Stabilize associative access to the emerging concept.
   16. Output a revisable abstract representation containing shared features,
       relations, weights, uncertainty, and supporting evidence.
END
```

**Core result:** The learner transforms multiple distinct episodes into a provisional representation of their meaningful common structure. This enables recognition and inference beyond exact repetition, while preserving enough uncertainty and episodic support to correct the concept when future evidence reveals that the original abstraction was incomplete or mistaken.
