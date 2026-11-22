# Point 20 — Schema-Driven Inference and Reconstructive Memory

## Scope and position in the source sequence

This document expands the next brain-related point in **Learning, Memory, and Concept Formation.txt**:

> Inference errors are common: the brain automatically fills in gaps based on schemas, which can create false memories or biases. Bartlett’s “War of the Ghosts” study is a classic demonstration: when people retold an unfamiliar story, they reshaped details to fit cultural expectations.

The algorithm below describes how a brain-like system can construct a coherent interpretation or recollection from incomplete evidence, how prior knowledge can help, how it can also introduce unsupported details, and how the system can detect and correct those additions.

The target is not simply “memory retrieval.” It is the interaction of **partial episodic evidence, semantic knowledge, schemas, expectations, source attribution, confidence, and corrective feedback**. The algorithm is deliberately written as a computational model of interacting neural systems. It does not claim that neurons literally execute the equations or that every brain uses one fixed sequence.

---

## 1. Biological and computational objective

### 1.1 The computational problem

An experience is rarely encoded with every detail equally well. Attention may be selective; some features may never be perceived; some details may weaken; and retrieval cues may reactivate only part of an episode. Nevertheless, the brain often produces a meaningful recollection. It combines the surviving evidence with knowledge about objects, events, people, causal relations, and cultural conventions.

This is adaptive. A schema can make it possible to understand a noisy scene, infer a likely event sequence, or remember the central meaning even when peripheral details are unavailable. The same process creates risk: a detail that is merely typical or plausible can be experienced as though it were actually observed. A coherent reconstruction is not necessarily a historically accurate reconstruction.

### 1.2 Main biological systems represented

The model distributes the work across interacting systems rather than assigning the whole process to one “memory centre.”

- **Hippocampal formation:** supports the binding and retrieval of relations among event elements. A partial cue can reinstate a broader pattern through pattern completion, although completion can also recover an incorrect or overlapping episode when cues are ambiguous.
- **Medial prefrontal cortex (mPFC):** participates in the use and integration of schemas, prior knowledge, and structured event models. Its contributions vary with the task and the kind of knowledge being retrieved.
- **Lateral prefrontal regions:** contribute to controlled retrieval, evaluation of competing representations, strategic search, and monitoring of whether a detail belongs to the event being remembered.
- **Parietal and angular-gyrus regions:** can contribute to the representation, attention, or subjective accessibility of retrieved episodic details; their exact role is debated and task-dependent.
- **Sensory association cortices:** can contribute modality-specific reinstatement, such as visual, auditory, or spatial features of an episode.
- **Temporal semantic-memory systems:** provide knowledge of concepts, typical features, scripts, and relations used to interpret incomplete information.
- **Amygdala and neuromodulatory systems:** can influence which emotionally salient information is prioritized and consolidated. They are modulators, not general truth detectors.

These systems are not independent modules with perfectly isolated functions. They interact recurrently, and a result obtained in one task should not be treated as a universal division of labour.

### 1.3 Algorithmic objective

Given:

1. a retrieval cue or an incomplete current observation;
2. a partial set of episodic features;
3. candidate schemas and semantic knowledge;
4. contextual constraints (time, place, task, speaker, culture, goals);
5. source information about where each candidate detail may have come from;

construct the most defensible interpretation or recollection while keeping separate:

- directly observed or clearly encoded details;
- details retrieved with episodic context;
- details inferred from a schema;
- details suggested by another person or a later source;
- details whose source is unknown or uncertain.

The system should prefer an incomplete but evidence-faithful report over an unsupported completion presented with certainty.

---

## 2. Inputs, internal representations, and outputs

### Inputs

- `cue`: a question, word, object, location, partial image, or other retrieval cue.
- `episodic_fragments`: available event details, each with a feature type and a retrieval-strength estimate.
- `context`: time, location, surrounding discourse, current goal, social setting, and relevant cultural information.
- `schema_library`: candidate event frames, scripts, category knowledge, role relations, and expected transitions.
- `source_history`: information about whether a candidate detail came from perception, the original event, imagination, inference, another person, or later information.
- `feedback`: corrections or corroborating evidence that may become available after a reconstruction.

### Internal record for each candidate detail

Each candidate is represented as a structured record:

`detail = {content, feature_type, episodic_strength, schema_support, source_candidates, contextual_fit, conflict_score, confidence, status}`

`status` should not be a single binary value. Useful statuses include `direct_evidence`, `episodically_retrieved`, `inferred`, `suggested_after_event`, `source_uncertain`, `contradicted`, and `unknown`.

### Outputs

- a ranked set of candidate reconstructions;
- a report containing supported details and explicit uncertainty;
- a source tag for every nontrivial detail;
- a list of conflicts or missing evidence;
- an update request when an important distinction cannot be resolved;
- a cautiously updated schema or episodic record after reliable feedback.

---

## 3. Detailed algorithmic sequence

## Step 1 — Detect a retrieval demand and define the question

**Purpose:** Establish what must be reconstructed before activating broad associative knowledge.

1. Detect whether the system is answering a question about a past event, interpreting an incomplete scene, retelling a story, or filling a missing relation.
2. Parse the requested target: person, object, action, sequence, location, time, causal link, or general meaning.
3. Determine the required precision. “What was the story about?” calls for gist; “What exact words did the speaker use?” requires highly specific evidence.
4. Preserve the distinction between a request for what happened and a request for what usually happens.
5. Attach the current context and the source of the retrieval cue to the retrieval task.

**Possible mathematical formalization**

Represent the query as a target vector `q` over feature classes. A relevance score for candidate feature `f_i` can be expressed as:

\[
r_i = \operatorname{sim}(q,f_i)\, a_i
\]

where `sim` is a similarity measure and `a_i` is an attention or task-relevance weight. This is a computational selection rule, not a claim that the brain calculates a literal vector dot product.

**Biological interpretation:** Goal-directed retrieval engages interactions among prefrontal control, semantic systems, and medial-temporal memory structures. The formulation of the question can bias which memories and schema features are activated.

## Step 2 — Reactivate the strongest available episodic fragments

**Purpose:** Recover evidence from the particular event before using generic expectations to fill gaps.

1. Search for event-specific cues linked to the requested target.
2. Attempt to reinstate sensory, spatial, temporal, and relational features associated with the episode.
3. Retrieve multiple fragments when possible rather than treating the first familiar association as decisive.
4. Preserve the relation among fragments: which person performed which action, in what location, and at what time.
5. Record weak or absent features as unavailable, not automatically as evidence that the feature was absent from the original event.

**Possible mathematical formalization**

A simplified cue-to-memory activation model is:

\[
A_j = \sum_i w_{ij}c_i
\]

where `c_i` denotes cue activation and `w_ij` denotes learned association strength from cue feature `i` to memory element `j`. Competing memories can be represented by a normalized activation:

\[
P(M_j\mid c)=\frac{\exp(\beta A_j)}{\sum_k \exp(\beta A_k)}
\]

The parameter `β` controls how sharply activation differences affect selection. These are useful computational abstractions of associative retrieval, not known neuronal equations.

**Biological interpretation:** Hippocampal pattern completion is one proposed contributor to reinstating an episode from partial cues. Similar events can overlap, so completion may retrieve the wrong instance or blend elements unless source and context are checked.

## Step 3 — Separate recovered evidence from missing information

**Purpose:** Prevent “not retrieved” from silently becoming “never happened” or “must have happened.”

1. Create a feature ledger for the target event.
2. For every feature, distinguish `retrieved`, `weakly retrieved`, `not retrieved`, `contradicted`, and `not queried`.
3. Record whether a missing feature was likely to have been attended to and encoded in the first place.
4. Estimate whether the feature is central or peripheral to the event.
5. Do not fill a missing cell at this stage.

**Possible mathematical formalization**

Let `e_i` be the evidence state for feature `i`. It may be represented categorically or probabilistically:

\[
e_i \in \{1,\;0,\;?\}
\]

where `1` means supported, `0` means evidence against, and `?` means unknown. A probabilistic implementation can use `P(e_i\text{ present}\mid retrieval)` rather than force an early yes/no decision.

**Biological interpretation:** Retrieval is reconstructive and cue-dependent. Failure to retrieve is not equivalent to complete erasure, and the absence of a vivid image does not prove that the event detail was absent.

## Step 4 — Activate relevant schemas, scripts, and cultural knowledge

**Purpose:** Retrieve useful prior structures that can organize incomplete evidence.

1. Identify the event class or conceptual domain suggested by the recovered fragments.
2. Retrieve multiple candidate schemas if the event is ambiguous.
3. Activate schema roles, typical objects, likely goals, causal relations, and expected event sequences.
4. Include context-specific and cultural knowledge only when relevant to the actual situation.
5. Keep the schema as a source of hypotheses, not as a record of what definitely occurred.

**Possible mathematical formalization**

For each candidate schema `S_k`, compute a contextual activation:

\[
Z_k = \alpha\,\operatorname{sim}(S_k,E)+\gamma\,\operatorname{fit}(S_k,C)+\eta\,G_k
\]

where `E` is the current evidence, `C` is context, `G_k` is goal relevance, and the coefficients control their contributions. Convert activations to normalized weights with a softmax or another bounded competition function.

**Biological interpretation:** Medial prefrontal and temporal systems contribute to schema use and semantic knowledge retrieval. The exact neural implementation of schema activation is distributed and remains under investigation.

## Step 5 — Generate candidate completions, but mark them as inferences

**Purpose:** Use schema knowledge to propose explanations for missing details without confusing proposals with retrieved facts.

1. For each missing or ambiguous feature, ask what values are compatible with each activated schema.
2. Generate more than one candidate when alternatives are plausible.
3. Mark each generated candidate `inferred` at creation.
4. Identify whether it is a typical feature, a causal expectation, a social convention, or a culturally learned assumption.
5. Prevent a generated completion from being promoted to episodic evidence merely because it is coherent.

**Possible mathematical formalization**

Let `h` be a candidate completion. A generative model can propose it with:

\[
P(h\mid S,C)\propto P(h\mid S)P(h\mid C)
\]

The system can retain the top `K` alternatives rather than committing to only the highest prior-probability candidate.

**Biological interpretation:** Schema-consistent expectations can guide interpretation and memory reconstruction. They are useful for prediction, but schema typicality alone cannot establish that a specific event detail occurred.

## Step 6 — Combine episodic evidence and prior expectations cautiously

**Purpose:** Determine which candidate best explains the actual evidence while preventing a strong prior from overwhelming contrary information.

1. Evaluate each completion against all retrieved fragments.
2. Determine whether the candidate predicts evidence that is actually present.
3. Penalize candidates that conflict with reliable episodic details.
4. Use the schema prior to resolve genuine ambiguity, not to erase contradictions.
5. Preserve the prior and evidence contributions separately for later source monitoring.

**Possible mathematical formalization: Bayesian inference**

\[
P(h\mid E,S,C)=\frac{P(E\mid h,S,C)P(h\mid S,C)}{P(E\mid S,C)}
\]

Here, `P(h|S,C)` is the prior plausibility of candidate `h`, while `P(E|h,S,C)` expresses how well it predicts the retrieved evidence. A candidate may be highly typical but still have low posterior support if it conflicts with strong evidence.

A practical log-score can be:

\[
L(h)=\lambda_E\log P(E\mid h)+\lambda_S\log P(h\mid S,C)-\lambda_X X(h)
\]

where `X(h)` is a contradiction penalty. The weights should be calibrated to the task; exact autobiographical recall should give greater weight to source-specific evidence than generic typicality.

**Biological interpretation:** Predictive and associative models are useful ways to formalize top-down and bottom-up influences. The equation is normative mathematics for the model, not a demonstrated literal neural computation.

## Step 7 — Test candidate details against temporal, spatial, and causal constraints

**Purpose:** Reject a plausible detail if it cannot fit the event’s actual structure.

1. Check whether the candidate can occur at the retrieved time.
2. Check whether the location and participants are compatible with known event fragments.
3. Check whether the proposed action is consistent with the order of events.
4. Test causal direction: do not assume that a familiar sequence is the sequence that occurred.
5. Detect impossible or mutually incompatible combinations.
6. Keep unresolved contradictions visible rather than smoothing them away.

**Possible mathematical formalization**

Represent the event as a graph `G=(V,R)`, where `V` contains entities and event states and `R` contains temporal, spatial, and causal relations. A candidate completion `h` receives a constraint score:

\[
C(h)=\sum_m \omega_m\,\phi_m(h,G)
\]

where `φ_m` is positive when constraint `m` is satisfied and negative when it is violated. A probabilistic version can multiply or combine constraint likelihoods, provided dependencies are not incorrectly assumed independent.

**Biological interpretation:** Hippocampal relational representations and prefrontal reasoning can contribute to checking whether retrieved elements belong together. The graph is an engineering abstraction for the relations that must be preserved.

## Step 8 — Perform source monitoring for each candidate detail

**Purpose:** Determine whether a detail came from the original event, inference, imagination, a later conversation, or another source.

1. Retrieve perceptual characteristics: visual, auditory, spatial, emotional, and temporal context.
2. Retrieve cognitive-operation characteristics: whether the detail was seen, heard, imagined, inferred, rehearsed, or merely considered.
3. Search for contextual tags identifying the original event and distinguish them from tags tied to later retellings.
4. Compare source candidates rather than assigning a source from familiarity alone.
5. If the source remains unclear, label it `source_uncertain`.

**Possible mathematical formalization**

Let `s` be a possible source and `f` the available source features. A classifier can estimate:

\[
P(s\mid f)=\frac{P(f\mid s)P(s)}{\sum_{s'}P(f\mid s')P(s')}
\]

Candidate sources might include `original_perception`, `episodic_retrieval`, `schema_inference`, `imagination`, `post_event_information`, and `unknown`. A confidence threshold should be required before assigning a specific source.

**Biological interpretation:** Source monitoring is a cognitive judgement based on characteristics of retrieved information and the operations that produced it. It is not a perfect tag stored beside every memory. Prefrontal control can contribute to evaluating source evidence; hippocampal and sensory reinstatement can contribute contextual detail.

## Step 9 — Detect familiarity without mistaking it for proof

**Purpose:** Prevent a candidate from being accepted solely because it feels familiar or fluent.

1. Estimate familiarity and recollective evidence separately.
2. Ask whether the candidate has unique event-specific context or only general semantic fluency.
3. Check whether repeated retellings have increased familiarity with a detail that was not present in the original event.
4. Reduce confidence when fluency is high but source evidence is weak.
5. Preserve the possibility that a true detail can feel unfamiliar and a false detail can feel familiar.

**Possible mathematical formalization**

Use two evidence dimensions:

\[
\mathbf{x}=[F,R]
\]

where `F` is familiarity strength and `R` is source-specific recollection strength. A decision function can be:

\[
D(h)=\theta_F F+\theta_R R-\theta_U U
\]

where `U` is source uncertainty. For high-precision autobiographical claims, `θ_R` should generally exceed `θ_F`. This is a decision model, not a claim that the brain uses exactly two scalar channels.

## Step 10 — Search for contradictory evidence and competing episodes

**Purpose:** Reduce errors caused by blending similar events or accepting the first coherent reconstruction.

1. Retrieve competing episodes with overlapping people, places, or actions.
2. Compare distinctive contextual details for each episode.
3. Identify whether a candidate belongs to a repeated routine rather than the particular instance being recalled.
4. Search for evidence that would falsify the leading candidate.
5. Re-rank candidates when a contradiction is found.

**Possible mathematical formalization**

For candidate `h_k`, define a score from support and contradiction:

\[
Q_k=\sum_i a_i e_{ik}-\sum_j b_j x_{jk}
\]

where `e_ik` represents supporting evidence and `x_jk` represents contradictions. The algorithm should not treat the absence of a cue as a contradiction unless the cue was likely to be encoded and retained.

**Biological interpretation:** Controlled retrieval and monitoring can help distinguish overlapping episodes. Similarity among memories is a source of interference and source confusion, especially for repeated or highly similar events.

## Step 11 — Estimate confidence independently of coherence

**Purpose:** Ensure a smooth, complete story does not automatically receive high confidence.

1. Compute evidence strength and source specificity.
2. Compute how many details were inferred rather than retrieved.
3. Account for unresolved contradictions and competing memories.
4. Estimate confidence separately for each detail, not just for the whole story.
5. Lower confidence when the reconstruction depends heavily on a schema, later suggestion, or generic expectation.

**Possible mathematical formalization**

A calibrated confidence model can use:

\[
\operatorname{Conf}(h)=\sigma(\beta_0+\beta_1E+\beta_2R-\beta_3I-\beta_4X-\beta_5U)
\]

where `E` is evidence strength, `R` source-specific recollection, `I` inferred-detail proportion, `X` contradiction burden, `U` source uncertainty, and `σ` is the logistic function. Parameters must be calibrated against an appropriate dataset; they are not universal biological constants.

## Step 12 — Choose a response policy: report, qualify, or request clarification

**Purpose:** Match the answer to the strength of evidence.

1. Report details with strong event-specific support as remembered, without adding embellishment.
2. State schema-derived interpretations as possibilities or inferences.
3. Explicitly say when the source of a detail is unclear.
4. Omit details that are unsupported and unnecessary.
5. Ask for clarification or further evidence when a high-impact distinction remains unresolved.
6. If the task requests gist rather than exact detail, summarize the stable meaning while avoiding invented specifics.

**Possible mathematical formalization**

Choose an action `a` that minimizes expected loss:

\[
a^*=\arg\min_a\sum_h P(h\mid E,S,C)\,\mathcal{L}(a,h)
\]

The loss function should penalize confident false claims more strongly than transparent uncertainty in high-stakes contexts. For low-stakes gist summaries, the relative costs can differ.

## Step 13 — Monitor the effect of retelling and post-event information

**Purpose:** Prevent later suggestions and repeated reconstructions from being mistaken for original evidence.

1. Record whether a detail appeared only after a question, a conversation, a photograph, a document, or another retelling.
2. Preserve the timeline of when each candidate detail entered the record.
3. Do not overwrite the original-source status when a later source repeats the same detail.
4. When new information is reliable, add it as a new source rather than silently changing the provenance of the earlier memory.
5. Distinguish a memory of the event from a memory of being told about the event.

**Possible mathematical formalization**

Represent provenance as a directed acyclic graph. Nodes represent source events or memory reports; edges represent transformations such as `perceived`, `inferred`, `suggested`, `rehearsed`, and `verified`. A candidate’s provenance confidence can decrease when it depends on an unverified chain of retellings.

**Biological interpretation:** Repeated retrieval and later information can influence subsequent remembering. The exact mechanisms vary with timing, task, and the memory’s state; reconsolidation should not be assumed to occur in every act of recall.

## Step 14 — Update the reconstruction when reliable corrective evidence arrives

**Purpose:** Learn from correction without replacing one unsupported belief with another.

1. Evaluate the reliability and independence of the feedback source.
2. Compare the correction with the prior evidence and source history.
3. Update the relevant candidate probabilities rather than globally rewriting the event.
4. Preserve the previous version and the reason for the update.
5. If evidence remains conflicting, keep multiple candidates rather than forcing certainty.

**Possible mathematical formalization**

For a candidate `h`, update odds using an evidence likelihood ratio:

\[
\text{Posterior odds}(h)=\text{Prior odds}(h)\times \frac{P(F\mid h)}{P(F\mid \neg h)}
\]

where `F` is the new feedback. Independent evidence can be more informative than multiple reports that all derive from the same original source. The likelihood ratio must reflect source reliability.

## Step 15 — Update schema knowledge separately from the memory of the specific episode

**Purpose:** Learn what is generally typical without rewriting what happened in one instance.

1. If multiple reliable experiences support a pattern, consider adjusting the corresponding schema.
2. Keep episode-specific details attached to their individual source records.
3. Do not treat one unusual event as proof that the general schema is wrong.
4. Do not let a general schema overwrite a reliable exception.
5. Store uncertainty and context boundaries with the schema.

**Possible mathematical formalization**

A simple incremental update for a feature probability is:

\[
\hat p_{t+1}=\hat p_t+\eta_t(y_t-\hat p_t)
\]

where `y_t` is a reliably observed feature (0 or 1) and `η_t` is a learning rate. A Bayesian beta-binomial representation can track both the estimate and uncertainty. The update should use verified observations, not details generated by the schema itself, or the system would reinforce its own assumptions.

## Step 16 — Finalize a provenance-aware reconstruction and retain uncertainty

**Purpose:** Produce a coherent answer without erasing the distinction between memory and inference.

1. Assemble only the details needed to answer the original question.
2. Preserve chronology and participant-role relations supported by evidence.
3. Label uncertain or inferred details in the output.
4. Keep the provenance record available for future review.
5. Store unresolved alternatives rather than collapsing them into one unsupported narrative.
6. Stop when further search is unlikely to improve accuracy or when the available evidence is exhausted.

**Output schema**

```text
Reconstruction:
  supported_details: [...]
  inferred_details: [...]
  source_uncertain_details: [...]
  contradicted_details: [...]
  unresolved_alternatives: [...]
  confidence_by_detail: {...}
  provenance_graph: ...
```

---

## 4. Mathematical operations and why they are useful

The following operations provide a rigorous computational vocabulary for the algorithm. They should not be interpreted as literal arithmetic performed consciously or explicitly by individual neurons.

| Operation | Computational role | Main caution |
|---|---|---|
| Weighted cue summation | Reactivates candidate memory elements from partial cues | Association strengths are distributed and context-dependent |
| Softmax / normalization | Represents competition among candidate memories or schemas | A normalized score is not automatically a true probability |
| Bayesian inference | Combines prior plausibility with likelihood of observed evidence | Priors can bias results when evidence is weak; model assumptions matter |
| Constraint satisfaction | Checks temporal, spatial, causal, and relational consistency | A coherent graph can still describe an event that did not happen |
| Source classification | Estimates whether a detail came from perception, inference, or later information | Source tags are fallible and often reconstructed |
| Signal detection | Separates evidence strength from decision threshold | Subjective confidence is not equivalent to accuracy |
| Logistic calibration | Maps evidence features to estimated confidence | Must be calibrated on representative data |
| Likelihood-ratio updating | Incorporates new evidence according to its diagnosticity | Correlated reports must not be counted as independent evidence |
| Incremental parameter update | Changes general schema knowledge from verified examples | Self-generated inferences must not be recycled as observations |
| Graph provenance | Tracks how details entered and changed across retellings | Missing provenance cannot be repaired by inference alone |

### 4.1 Gist and detail as distinct information channels

A useful model distinguishes a representation of the general meaning (`gist`) from a representation of specific episode details (`verbatim` or item-specific information). Both can be useful. A gist representation may support correct understanding even when exact details are unavailable. It can also produce false recognition when a plausible but unseen item matches the remembered theme.

One simplified representation is:

\[
M=(M_{gist},M_{detail})
\]

At retrieval, the response may be based on:

\[
R=w_gM_{gist}+w_dM_{detail}
\]

When detailed episodic evidence is weak, `w_g` may become relatively influential. This can preserve the broad meaning while increasing the risk of accepting schema-consistent details that were not actually present.

### 4.2 Prediction error and mismatch

A candidate schema predicts features that may or may not match the retrieved evidence. A simple mismatch term is:

\[
\delta_i=o_i-\hat{o}_i
\]

where `o_i` is an observed or retrieved feature and `ô_i` is the predicted feature. Large mismatch should trigger re-evaluation of the active schema or candidate episode. In this model, a mismatch is a reason to search again—not an instruction to alter the remembered evidence until it fits.

### 4.3 Confidence calibration

Confidence should be assessed empirically. For a set of claims with predicted confidence `p_i` and correctness `y_i∈{0,1}`, a Brier score is:

\[
BS=\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2
\]

Lower scores indicate better probabilistic calibration and accuracy together. Calibration can be assessed separately for directly retrieved details, inferred details, and source-uncertain details. This helps reveal whether the system is systematically overconfident in schema-generated completions.

---

## 5. Bartlett’s “The War of the Ghosts” as a worked conceptual case

Frederic Bartlett used an unfamiliar North American Indigenous folk tale, commonly titled **“The War of the Ghosts,”** to investigate remembering and repeated reproduction. The story contained content and causal relations that were unfamiliar to many of his British participants. Across retellings, reports could become shorter and more conventionalized: details could be omitted, transformed, rationalized, or reorganized in ways that made the account more compatible with the rememberer’s existing knowledge.

For this algorithm, the lesson is not that every memory is false or that schemas always overpower evidence. The important mechanism is conditional:

1. Some details are unfamiliar or weakly encoded.
2. The person retains a partial representation of the story’s gist and structure.
3. Familiar schemas provide candidate interpretations for gaps or unusual events.
4. A retelling may substitute a familiar element for an unfamiliar one, omit a detail, or rearrange an event relation.
5. Repeated retellings can make the revised version more accessible and familiar.
6. Unless source monitoring distinguishes the original story from later reconstruction, the person may report the transformed version as the remembered story.

A modern revisit of Bartlett’s schema account emphasizes an important qualification: schema-consistent material can be remembered relatively well, and distortion is more likely when material conflicts with or poorly fits available schemas. Therefore, schema use is not intrinsically erroneous. It is adaptive organization with a conditional risk of distortion. See Ost et al. (2022).

### Worked computational trace

Suppose a story contains an unfamiliar object and an unusual causal transition.

- **Retrieved evidence:** “a narrow vessel,” “people travelled across water,” “a sudden event interrupted the journey.”
- **Schema activated:** familiar local travel story involving a small boat and a conventional accident.
- **Candidate completion:** replace the unfamiliar vessel with a familiar boat type; explain the interruption using a familiar accident sequence.
- **Evidence check:** the exact vessel type and causal explanation are not supported by the recovered fragments.
- **Source-monitoring check:** the candidate arose from a familiar schema, not from a vivid event-specific detail.
- **Output:** retain the supported broad description; mark the vessel subtype and causal explanation unknown.

This trace illustrates the desired algorithmic safeguard: a system may use schemas to generate candidate explanations but must not promote those candidates to observed facts without independent evidence.

---

## 6. Integrated pseudocode

```python
from dataclasses import dataclass, field
from enum import Enum
from math import exp
from typing import Any, Iterable


class DetailStatus(str, Enum):
    DIRECT_EVIDENCE = "direct_evidence"
    EPISODICALLY_RETRIEVED = "episodically_retrieved"
    INFERRED = "inferred"
    SUGGESTED_AFTER_EVENT = "suggested_after_event"
    SOURCE_UNCERTAIN = "source_uncertain"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


@dataclass
class DetailCandidate:
    content: Any
    status: DetailStatus = DetailStatus.UNKNOWN
    evidence_strength: float = 0.0
    episodic_context: float = 0.0
    schema_support: float = 0.0
    source_uncertainty: float = 1.0
    contradiction: float = 0.0
    confidence: float = 0.0
    provenance: list[str] = field(default_factory=list)


def softmax(scores: list[float], temperature: float = 1.0) -> list[float]:
    if not scores:
        return []
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [s / temperature for s in scores]
    peak = max(scaled)
    values = [exp(s - peak) for s in scaled]
    total = sum(values)
    return [v / total for v in values]


def activate_schemas(evidence, context, schema_library):
    """Return candidate schemas with separate evidence and context scores."""
    ranked = []
    for schema in schema_library:
        evidence_fit = schema.evidence_fit(evidence)
        context_fit = schema.context_fit(context)
        score = 0.65 * evidence_fit + 0.35 * context_fit
        ranked.append((schema, score, evidence_fit, context_fit))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def generate_completions(missing_features, active_schemas):
    candidates = []
    for feature in missing_features:
        for schema, schema_score, _, _ in active_schemas:
            for proposal in schema.propose(feature):
                candidates.append(
                    DetailCandidate(
                        content=proposal,
                        status=DetailStatus.INFERRED,
                        schema_support=max(0.0, min(1.0, schema_score)),
                        provenance=[f"schema:{schema.name}"],
                    )
                )
    return candidates


def evaluate_candidate(candidate, evidence, context, source_history):
    # These methods are abstract interfaces supplied by the implementation.
    candidate.evidence_strength = evidence.support_for(candidate.content)
    candidate.episodic_context = evidence.episodic_specificity(candidate.content)
    candidate.contradiction = evidence.contradiction_for(candidate.content)
    candidate.source_uncertainty = source_history.uncertainty_for(candidate.content)

    # Evidence and episode-specific context outweigh generic schema support.
    raw = (
        0.40 * candidate.evidence_strength
        + 0.35 * candidate.episodic_context
        + 0.10 * candidate.schema_support
        - 0.10 * candidate.contradiction
        - 0.05 * candidate.source_uncertainty
    )
    candidate.confidence = max(0.0, min(1.0, raw))

    if candidate.contradiction >= 0.75:
        candidate.status = DetailStatus.CONTRADICTED
    elif candidate.episodic_context >= 0.75 and candidate.evidence_strength >= 0.75:
        candidate.status = DetailStatus.EPISODICALLY_RETRIEVED
    elif candidate.source_uncertainty >= 0.65:
        candidate.status = DetailStatus.SOURCE_UNCERTAIN
    else:
        # Keep the inferred label unless independent episodic evidence is found.
        candidate.status = DetailStatus.INFERRED
    return candidate


def reconstruct(cue, evidence, context, schema_library, source_history,
                feedback=None):
    target = define_retrieval_target(cue, context)
    fragments = evidence.retrieve_fragments(target)
    ledger = build_feature_ledger(target, fragments)

    # Unknown features remain unknown until a candidate is explicitly generated.
    missing = [item for item in ledger if item.state == "unknown"]
    active_schemas = activate_schemas(fragments, context, schema_library)
    proposals = generate_completions(missing, active_schemas[:3])

    candidates = [
        evaluate_candidate(c, evidence, context, source_history)
        for c in proposals
    ]
    candidates = check_temporal_spatial_causal_constraints(candidates, fragments)
    candidates = search_competing_episodes(candidates, fragments, evidence)

    if feedback is not None:
        candidates = update_with_source_weighted_feedback(candidates, feedback)

    report = assemble_provenance_aware_report(
        retrieved_fragments=fragments,
        candidates=candidates,
        feature_ledger=ledger,
        source_history=source_history,
    )
    return report
```

The pseudocode intentionally leaves neural mechanisms as abstract interfaces. It is an implementable computational architecture, not a biological simulator. Numeric weights are illustrative placeholders and require empirical calibration for any real application.

---

## 7. Failure modes and safeguards

| Failure mode | Why it occurs | Safeguard |
|---|---|---|
| Schema substitution | A typical feature replaces an unavailable episode detail | Preserve `inferred` provenance; require episode-specific evidence for promotion |
| Gist-to-detail error | Broad meaning is mistaken for a precise item or action | Maintain separate gist and detail records |
| Source confusion | A detail from imagination or later discussion is attributed to the original event | Maintain a source timeline and use source-monitoring checks |
| Repeated-retelling drift | Each reconstruction becomes input to the next reconstruction | Preserve the earliest available record and provenance graph |
| Familiarity inflation | Repetition makes a detail feel known | Separate familiarity from recollective context |
| Competing-episode blending | Similar events contribute details to one composite memory | Retrieve alternatives and compare distinctive context |
| Cultural overfitting | The system replaces unfamiliar content with culturally typical content | Keep culturally specific expectations as priors, never as proof |
| Overcorrection | A correction from an unreliable source overwrites valid evidence | Weight feedback by source reliability and independence |
| Excessive skepticism | The system refuses useful inference even when the task requires prediction | Permit inference but label it and represent uncertainty |
| False precision | A numerical confidence is treated as objective certainty | Calibrate confidence on external outcomes and report source class |

---

## 8. Validation and experimental tests

A computational implementation should be tested with controlled tasks rather than judged only by whether its generated stories sound plausible.

### Test 1 — Schema-consistent versus schema-inconsistent details

Present stories containing details that either fit or conflict with familiar scripts. Test immediate and delayed recall. The model should predict stronger schema influence for poorly encoded or unfamiliar details and comparatively faithful recall for familiar, well-structured material. It must not assume schema-consistency always creates an error.

### Test 2 — Gist and detail separation

Use material where participants can remember the theme but not exact words or objects. Assess gist recognition and detail recognition independently. The model should be able to produce a correct gist while withholding unsupported exact details.

### Test 3 — Source-monitoring challenge

Present details through different channels: original story, later suggestion, imagination instruction, and post-event discussion. Ask participants to identify the source. Measure source attribution accuracy separately from item familiarity.

### Test 4 — Repeated retelling

Collect multiple retellings at controlled intervals. Track omissions, substitutions, additions, and changes in event order. Include an immutable copy of the original stimulus so that drift can be measured objectively.

### Test 5 — Competing similar episodes

Present two similar events with one or more distinctive details. Later ask questions that cue both events. The model should compare context and avoid merging elements from different episodes.

### Test 6 — Confidence calibration

Collect confidence for each reported detail. Compare confidence with objective correctness using calibration curves and Brier score. The system should be penalized when inferred details are reported with the same certainty as directly supported details.

### Test 7 — Feedback reliability

Provide corrections from sources with different reliability and independence. The model should update more when evidence is diagnostic and independently corroborated, not merely repeated by several people who all received the same original suggestion.

### Evaluation metrics

- Detail-level accuracy and omission rate.
- Schema-consistent false-addition rate.
- Source-attribution accuracy.
- Event-order and relational accuracy.
- Gist accuracy versus verbatim-detail accuracy.
- Calibration error and Brier score.
- Rate of unsupported details reported as facts.
- Robustness across cultural and familiarity conditions.
- Ability to preserve uncertainty when evidence is insufficient.

---

## 9. Scientific interpretation and limitations

1. **Reconstructive does not mean uniformly unreliable.** Prior knowledge can improve memory organization and retrieval. Distortion is more likely under particular conditions, including unfamiliar material, weak item-specific evidence, source confusion, and repeated reconstruction.
2. **A schema is not a complete script that is always activated.** Knowledge structures are distributed, vary in specificity, and can compete. A person may use several partially overlapping schemas.
3. **False memories have multiple routes.** Semantic association, gist-based retrieval, misinformation, source-monitoring errors, imagination, and event overlap can all contribute. No single equation or brain region explains all false memories.
4. **Confidence is not a truth signal.** Confidence can be affected by familiarity, repetition, social reinforcement, fluency, and emotion. It must be evaluated separately from objective accuracy.
5. **Neural localization is not one-to-one.** The hippocampus, medial prefrontal cortex, lateral prefrontal cortex, temporal semantic systems, and parietal regions interact. Their contributions depend on task, retrieval demands, and the type of memory.
6. **Reconsolidation is not assumed on every recall.** Retrieval can sometimes make memories susceptible to updating, but whether and how this occurs depends on boundary conditions. A computational system should record new information and its source without presuming that a biological memory trace has been rewritten.
7. **Mathematical operations are models.** Bayesian inference, graph constraints, weighted sums, and source classifiers provide useful formal descriptions. They are not proof that the brain explicitly computes these formulas.

---

## 10. Selected scientific references

- Bartlett, F. C. (1932). *Remembering: A Study in Experimental and Social Psychology*. Cambridge University Press. (Foundational work on reconstructive remembering and “The War of the Ghosts.”)
- Ost, J., Udell, J., Dear, S., Zinken, J., Blank, H., & Costall, A. (2022). The serial reproduction of an urban myth: revisiting Bartlett's schema theory. *Memory, 30*(6), 775–783. https://doi.org/10.1080/09658211.2022.2059514
- Lentoor, A. G. (2023). Cognitive and neural mechanisms underlying false memories: misinformation, distortion or erroneous configuration? *AIMS Neuroscience, 10*(3), 255–268. https://doi.org/10.3934/Neuroscience.2023020
- Gatti, D., Rinaldi, L., Mazzoni, G., et al. (2024). Semantic and episodic processes differently predict false memories in the DRM task. *Scientific Reports, 14*, 256. https://doi.org/10.1038/s41598-023-50687-z
- Johnson, M. K., Hashtroudi, S., & Lindsay, D. S. (1993). Source monitoring. *Psychological Bulletin, 114*(1), 3–28. https://doi.org/10.1037/0033-2909.114.1.3

### Evidence notes used in preparing this algorithm

- The 2022 revisit of Bartlett’s theory is important because it cautions against the oversimplified claim that all schema use necessarily distorts memory. Familiar, schema-compatible material may be reproduced more accurately than unfamiliar material.
- The 2023 review summarizes several proposed mechanisms and neural correlates of false memories, including gist-based and associative accounts; it does not establish a single universal mechanism.
- The 2024 DRM study illustrates that semantic and episodic contributions can differ in predicting false memories, reinforcing the need to model multiple interacting routes.

---

## 11. Completion criterion

The algorithm is complete for this point when an implementation can:

1. reconstruct a candidate account from partial evidence;
2. use schemas to propose missing information without mislabelling it as observed;
3. compare candidates against episodic, temporal, spatial, and causal constraints;
4. monitor the source of each detail;
5. distinguish familiarity, gist, and event-specific recollection;
6. detect contradictions and competing episodes;
7. calibrate confidence and preserve unresolved uncertainty;
8. incorporate reliable feedback without erasing provenance; and
9. update general schema knowledge separately from the record of a particular event.

**Core design principle:** use prior knowledge to interpret incomplete evidence, but never allow plausibility alone to masquerade as evidence of what actually happened.
