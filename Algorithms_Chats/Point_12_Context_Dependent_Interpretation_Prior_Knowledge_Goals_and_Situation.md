# Point 12 — Context-Dependent Interpretation: Prior Knowledge, Goals, and Situation Shape Meaning

## Scope and position in the source sequence

This document develops the next brain-related point in **“Learning, Memory, and Concept Formation.txt”**:

> Understanding is never isolated from context or motivation. Prior knowledge and current goals frame interpretation at every level. Context—including the prior sentence, cultural norms, and the physical environment—constrains ambiguity: the same phrase can mean different things depending on the situation.

The focus here is **context-dependent interpretation**. Emotional modulation, motivation as a learning/reward process, social interaction, lifespan development, and common comprehension failures are separate topics later in the source and are not expanded as independent algorithms here. Current goals are included only insofar as they select and weight contextual information during interpretation.

The algorithm is a biologically informed computational reconstruction. It does not claim that the brain literally executes the equations below or that a single brain region performs the whole process. The equations formalize possible operations—prediction, evidence weighting, competition between interpretations, updating, and uncertainty reduction—that are compatible with several findings in cognitive neuroscience and language research.

---

## 1. Biological objective

The brain must interpret incoming information against a continuously changing internal model of the world. A word, phrase, gesture, or event rarely arrives in isolation. Its interpretation depends on what has already been said or perceived, what the listener knows, what is happening in the environment, who is communicating, and what the listener is trying to accomplish.

Consider the phrase **“The bank is closed.”** Depending on context, *bank* may refer to a financial institution or the side of a river. A preceding conversation about withdrawing money strongly favours the financial meaning; a discussion about fishing favours the riverbank interpretation. The sensory or lexical input may be identical, but the probability assigned to each interpretation changes because the context changes.

A plausible biological objective is therefore:

1. Maintain a compact, revisable representation of the current discourse and situation.
2. Retrieve relevant knowledge from long-term memory.
3. Use goals and environmental cues to determine which knowledge is relevant now.
4. Generate and rank candidate interpretations.
5. integrate incoming evidence with contextual expectations rather than allowing expectations to replace the evidence.
6. Update the current interpretation when new information confirms, weakens, or contradicts it.

The outcome is not necessarily a complete conscious narrative. It can be a temporary, task-sensitive situation model sufficient to identify a referent, understand an instruction, predict an event, or choose an action.

## 2. Main biological systems and their proposed contributions

The process is distributed. The following assignments are functional summaries, not exclusive one-region/one-operation mappings.

### 2.1 Language-selective frontotemporal network

Left frontal and temporal language-responsive regions support the processing of linguistic forms and the construction of linguistic meaning. Temporal regions contribute to lexical-semantic access and combinatorial interpretation; inferior frontal regions can contribute to controlled selection, syntactic/semantic composition, and resolving competition when the input is difficult. The language network is not the only system involved in context-rich comprehension, and it should not be treated as a general-purpose reasoning network.

### 2.2 Temporal and temporoparietal semantic systems

Lateral and anterior temporal regions participate in accessing and combining conceptual knowledge. Angular-gyrus and other temporoparietal areas have been associated with the integration of information across a discourse, retrieval of relevant knowledge, and construction of coherent event or situation representations. These functions are distributed and task dependent.

### 2.3 Medial prefrontal and posterior midline systems

Medial prefrontal cortex and posterior midline regions—including posterior cingulate/precuneus—are often recruited when comprehension depends on integrating a situation with prior knowledge, event structure, people, goals, or a broader context. Such activity is particularly evident in coherent narrative or discourse comprehension. These regions are also involved in processes beyond language, including internally directed cognition and social/event knowledge.

### 2.4 Hippocampal and medial temporal memory systems

The hippocampus can help retrieve and bind relations among people, places, events, and their temporal order. It is especially relevant when the current sentence or event must be linked to a specific earlier episode, when contextual details are separated in time, or when a new situation must be compared with a previously experienced one. It is not necessary to assume that every ordinary word is consciously retrieved as an episodic memory.

### 2.5 Perceptual and action systems

Visual, auditory, spatial, and action-related systems contribute information about the physical scene and can provide grounding for language. Seeing a person holding a key, standing beside a door, or pointing toward a river can change which interpretation is most plausible. Situation models can therefore incorporate perceptual and sensorimotor information as well as linguistic information.

### 2.6 Control and attention systems

Current task goals and attentional control influence which contextual features are selected. When several meanings compete, control processes can bias retrieval toward information relevant to the listener's current task. This is not equivalent to deliberately choosing a meaning at every word; many context effects are rapid and occur without explicit awareness.

---

# 3. Integrated algorithm

## Step 1 — Detect and encode the incoming linguistic or environmental event

**Objective:** Convert the new input into a usable event representation while preserving its relation to the ongoing stream.

**Biological interpretation:** For speech, auditory pathways extract acoustic and phonological structure. For reading, visual pathways identify orthographic patterns and map them onto linguistic representations. Language-responsive temporal and frontal systems support lexical and compositional processing. In parallel, visual or other sensory systems may encode relevant features of the surrounding scene.

**Computational operations:**

- Feature extraction from the input stream.
- Segmentation into candidate units (words, phrases, gestures, or events).
- Temporal indexing so that the new item can be associated with the current discourse state.
- Confidence estimation for uncertain or degraded input.

A simplified input vector can be represented as:

\[
\mathbf{x}_t = [x_{t,1},x_{t,2},\ldots,x_{t,n}]
\]

where the components encode available linguistic or perceptual features at time \(t\). These are abstract model features, not literal neural coordinates.

A confidence term can be represented as \(q_t\in[0,1]\), where low confidence means that the sensory evidence is noisy or incomplete. The model should preserve uncertainty instead of prematurely committing to one interpretation.

**Output:** A provisional representation of the incoming item and its time position in the ongoing stream.

## Step 2 — Retrieve the immediately preceding discourse state

**Objective:** Recover what has just been discussed so that the new item is not interpreted as an isolated fragment.

**Biological interpretation:** Working-memory and language systems maintain recently processed propositions and discourse referents. Temporal and parietal semantic systems help retrieve relevant meanings; frontal control systems may assist when an earlier referent is difficult to access. A discourse representation can include current topic, participants, unresolved references, event sequence, and causal relations.

**Computational operations:**

- Maintain a short-term state \(D_t\) containing recent propositions and active referents.
- Retrieve items with high semantic, syntactic, temporal, or referential relevance to the current input.
- Allow activation to decay with time, while preserving items that remain important to the topic or task.

One illustrative activation model is:

\[
a_i(t)=a_i(t_0)e^{-\lambda_i(t-t_0)}+r_i(t)
\]

where \(a_i\) is the activation of a discourse item, \(\lambda_i\) is its decay parameter, and \(r_i(t)\) represents renewed activation from current cues. The formula is a convenient memory model, not a measured law of neural decay.

**Output:** A compact active discourse context \(D_t\), including recently mentioned entities and events.

## Step 3 — Retrieve relevant long-term knowledge

**Objective:** Bring in world knowledge and learned concepts that can explain the current input.

**Biological interpretation:** Distributed semantic memory systems make conceptual knowledge available. Temporal association areas and the anterior temporal lobe participate in conceptual retrieval and integration; modality-related cortices may contribute feature information. The hippocampus may retrieve specific episodes when details of a previously experienced situation are relevant.

**Computational operations:**

- Generate memory retrieval cues from the input and active discourse.
- Search for concepts, event scripts, causal relations, and familiar regularities that overlap with those cues.
- Retrieve only a subset of available knowledge, because activating every possible association would create interference.
- Preserve provenance: distinguish what was explicitly stated from what is retrieved from prior knowledge.

A cue-to-memory similarity function may be written as:

\[
S(m_i,\mathbf{x}_t)=\frac{\mathbf{m}_i\cdot\mathbf{x}_t}{\|\mathbf{m}_i\|\|\mathbf{x}_t\|}
\]

where \(m_i\) is a stored representation. Similarity is one possible retrieval principle; biological memory retrieval is more complex than a single cosine calculation.

**Output:** A set of candidate knowledge structures \(K_t\), each linked to the cue that activated it and tagged as semantic, episodic, procedural, or perceptual when the model can make that distinction.

## Step 4 — Estimate the current task goal

**Objective:** Determine what the listener is trying to achieve, because the same context may be relevant differently for different tasks.

**Biological interpretation:** Goal maintenance involves interactions among prefrontal control systems, parietal attention systems, and memory networks. The goal may be explicit (find a route, identify a speaker's intended meaning, follow an instruction) or implicit (continue a conversation, locate an object, detect a warning).

**Computational operations:**

Represent the active goal as \(G_t\). Possible goal states include:

- Identify a word or referent.
- Understand a causal explanation.
- Follow an instruction.
- Predict what happens next.
- Evaluate a speaker's claim.
- Locate an object or event in the environment.

Assign a relevance weight \(w_j(G_t)\) to each available context feature. For example, the physical location may receive a high weight during navigation, whereas a speaker's identity and prior beliefs may become more important when interpreting an ambiguous social remark.

A normalized goal-weight model is:

\[
w_j(G_t)=\frac{\exp(\beta_g u_j(G_t))}{\sum_k\exp(\beta_g u_k(G_t))}
\]

where \(u_j\) estimates the usefulness of contextual feature \(j\) for the current goal and \(\beta_g\) controls how strongly the model concentrates on the most useful features. This softmax is a computational abstraction of selective attention, not a proposed literal neural equation.

**Output:** A task-conditioned relevance profile over context sources.

## Step 5 — Extract context from the physical and perceptual environment

**Objective:** Identify relevant properties of the current scene that constrain possible interpretations.

**Biological interpretation:** Visual and auditory systems encode objects, locations, actions, and environmental events. Parietal and temporal systems help relate these features spatially and semantically. The current scene may activate learned event knowledge—for example, a person holding a menu inside a restaurant makes restaurant-related interpretations more likely.

**Computational operations:**

- Detect objects, agents, actions, locations, and salient scene features.
- Estimate relations among them (near, inside, holding, looking at, moving toward).
- Link perceptual features to the active discourse and task goal.
- Reject irrelevant scene information or reduce its weight.

Represent the scene as a graph:

\[
E_t=(V_t,R_t)
\]

where \(V_t\) contains entities and \(R_t\) contains relations among them. The graph makes explicit that a scene is not just a list of objects; relations such as *person-at-counter* or *boat-near-bank* can be crucial to interpretation.

**Output:** A perceptually grounded context representation \(E_t\).

## Step 6 — Retrieve cultural, conventional, and shared background knowledge

**Objective:** Use learned conventions and shared knowledge without treating them as universally valid.

**Biological interpretation:** Cultural knowledge is acquired through repeated experience, language, education, and social interaction. It is stored across distributed semantic and episodic systems. Learned conventions can rapidly constrain expectations, such as the likely meaning of an idiom, a social form of address, or a familiar public sign.

**Computational operations:**

- Retrieve relevant conventions and event regularities.
- Estimate whether the current speaker and situation are likely to share those conventions.
- Maintain uncertainty when the listener's cultural experience may differ from the speaker's.
- Avoid allowing a stereotype or familiar script to override contradictory evidence.

A simple applicability term is:

\[
A(c)=P(\text{convention }c\text{ applies}\mid\text{speaker, setting, history})
\]

This term should be reduced when there is evidence that the speaker belongs to a different convention system or is using a nonstandard meaning.

**Output:** A set of culturally and conventionally informed expectations with explicit uncertainty.

## Step 7 — Construct a provisional situation model

**Objective:** Combine the active discourse, retrieved knowledge, goals, and environmental information into one coherent representation of the current situation.

**Biological interpretation:** Situation-model construction recruits interacting language, semantic, memory, and broader association networks. Research on discourse comprehension implicates posterior midline and medial prefrontal regions in context-sensitive event integration, while hippocampal and other memory systems can contribute relations among events and entities.

**Computational operations:**

Build a structured model containing, where available:

- Agents and entities.
- Actions and states.
- Locations and spatial relations.
- Temporal order and duration.
- Causes, intentions, and consequences.
- Current topic and discourse referents.
- What is known, inferred, and still uncertain.

A model state can be represented as:

\[
M_t=\{A_t,O_t,\mathcal{R}_t,T_t,L_t,C_t,Q_t\}
\]

where \(A_t\) are agents, \(O_t\) objects, \(\mathcal{R}_t\) relations, \(T_t\) temporal information, \(L_t\) locations, \(C_t\) causal or intentional structure, and \(Q_t\) uncertainty/provenance information.

**Output:** A provisional situation model \(M_t\) against which new input can be interpreted.

## Step 8 — Generate candidate interpretations for ambiguous input

**Objective:** Keep multiple plausible meanings active when the input permits more than one reading.

**Biological interpretation:** Lexical-semantic representations can activate multiple related meanings. Contextual cues influence their relative accessibility. In ambiguous language, competition and selection can involve temporal semantic systems and frontal control mechanisms. Contextual facilitation can occur rapidly; resolving difficult or weakly constrained ambiguity may require additional processing.

**Computational operations:**

For an ambiguous item \(x\), generate candidates:

\[
H_t=\{h_1,h_2,\ldots,h_n\}
\]

Each candidate should include its semantic features, grammatical fit, contextual assumptions, and predicted consequences. Avoid deleting alternatives too early, especially when the input is compatible with more than one interpretation.

**Output:** A hypothesis set with an initial prior probability for each candidate.

## Step 9 — Estimate prior plausibility from context

**Objective:** Use the available context to determine which candidate interpretations are initially more plausible.

**Biological interpretation:** Expectations are shaped by the preceding discourse, world knowledge, scene information, and speaker-related cues. Predictive influences can alter processing difficulty and neural responses to incoming words. However, contextual prediction does not imply that the brain always predicts a single exact next word.

**Computational operations:**

Estimate a prior probability:

\[
P(h_i\mid C_t,G_t)
\]

where \(C_t\) combines the active discourse, long-term knowledge, scene, and shared conventions, and \(G_t\) is the current goal. For the phrase “the bank is closed,” a financial conversation increases the prior probability of the financial-institution meaning; a river-crossing discussion increases the prior probability of the river-edge meaning.

One way to combine independent-ish contextual sources is a weighted log-linear score:

\[
 s_i=\theta_0+\sum_j w_j(G_t)\,f_j(h_i,C_t)
\]

\[
P(h_i\mid C_t,G_t)=\frac{e^{s_i}}{\sum_k e^{s_k}}
\]

where \(f_j\) scores how well candidate \(h_i\) fits context source \(j\). This model is useful for simulation, but real sources are correlated and should not be treated as statistically independent without justification.

**Output:** Context-sensitive prior probabilities over candidate meanings.

## Step 10 — Evaluate the incoming evidence against each candidate

**Objective:** Determine how well each interpretation explains the actual sensory and linguistic input.

**Biological interpretation:** The auditory or visual evidence, phonological structure, syntax, lexical meaning, and discourse constraints interact over time. The incoming signal can support or disconfirm a contextually preferred interpretation. A context-favoured meaning must remain revisable if the actual word or sentence does not fit it.

**Computational operations:**

Compute a likelihood term:

\[
P(x_t\mid h_i,C_t)
\]

This represents how expected the observed input would be if candidate \(h_i\) were correct. Evidence can include:

- Phonological or orthographic compatibility.
- Syntactic fit.
- Semantic coherence.
- Referential consistency.
- Compatibility with visible or audible events.

For a simple model, the candidate score becomes:

\[
\log S_i=\log P(h_i\mid C_t,G_t)+\alpha\log P(x_t\mid h_i,C_t)
\]

where \(\alpha\) controls the relative strength assigned to direct input evidence. A robust system should not set \(\alpha\) so low that strong contextual expectation can ignore clear contrary input.

**Output:** A candidate score that combines contextual expectation and observed evidence.

## Step 11 — Resolve competition while preserving uncertainty

**Objective:** Select the best-supported interpretation when one is sufficiently favoured, but avoid false certainty when candidates remain close.

**Biological interpretation:** Neural representations can compete through recurrent activation and inhibition, while control processes may bias relevant representations. The precise neural implementation depends on task and timescale. Behaviourally, ambiguity resolution can be graded rather than a single all-or-none event.

**Computational operations:**

Use Bayes' rule as a normative model:

\[
P(h_i\mid x_t,C_t,G_t)=\frac{P(x_t\mid h_i,C_t)P(h_i\mid C_t,G_t)}{\sum_k P(x_t\mid h_k,C_t)P(h_k\mid C_t,G_t)}
\]

Calculate uncertainty with entropy:

\[
\mathcal{H}(H_t)=-\sum_i P(h_i\mid x_t,C_t,G_t)\log P(h_i\mid x_t,C_t,G_t)
\]

If the best candidate exceeds a confidence threshold and has a clear margin over alternatives, it can be used provisionally. If not, preserve multiple candidates, seek additional context, or defer commitment.

**Output:** A selected or unresolved interpretation accompanied by confidence and residual ambiguity.

## Step 12 — Integrate the selected interpretation into the ongoing model

**Objective:** Update the situation model so that the new meaning becomes part of the coherent discourse or event representation.

**Biological interpretation:** New information is integrated with active discourse representations and long-term knowledge. The hippocampus may help bind new elements to previously encoded events; temporal, parietal, and midline association systems contribute to broader semantic and situation-level integration. The extent of hippocampal involvement depends on the memory demands and novelty of the situation.

**Computational operations:**

- Add the new proposition to the active model.
- Link its entities to existing referents.
- Update event sequence, temporal order, causal structure, and topic state.
- Track the source of each element: directly observed, explicitly stated, retrieved, or inferred.
- Preserve contradictions rather than silently overwriting them.

A generic update is:

\[
M_{t+1}=U(M_t,x_t,h^*,q_t)
\]

where \(U\) is an update operation and \(h^*\) is the chosen interpretation. The function should retain earlier information that remains relevant and revise only those parts affected by the new evidence.

**Output:** An updated situation model \(M_{t+1}\).

## Step 13 — Check coherence and detect context conflicts

**Objective:** Test whether the current interpretation fits the broader situation rather than only the immediately preceding words.

**Biological interpretation:** Coherence monitoring and integration can recruit language, memory, semantic, and control systems. A contextually anomalous or surprising word can elicit an increased N400 response, while later reanalysis or syntactic repair can recruit additional processes. The N400 is a context-sensitive neural response associated with meaning-related processing; it should not be interpreted as a direct meter of surprise or as a one-to-one readout of a single computation.

**Computational operations:**

- Check consistency with earlier propositions.
- Check whether entities and referents remain stable.
- Identify causal, temporal, or spatial contradictions.
- Compare the model's predictions with the new evidence.
- Mark unresolved conflict for later repair.

A contradiction score can be modeled as:

\[
\Delta_t=\sum_j \rho_j\,\mathbf{1}[r_j\text{ conflicts with }M_t]
\]

where \(r_j\) are relevant constraints and \(\rho_j\) are their importance weights. This is a schematic error score, not a neural measurement.

**Output:** A coherence estimate and list of unresolved conflicts.

## Step 14 — Revise the interpretation when new evidence requires it

**Objective:** Prevent an early context-based guess from becoming an irreversible commitment.

**Biological interpretation:** Comprehension unfolds incrementally. Later words, sentences, perceptual cues, or speaker clarification can force reinterpretation. Reanalysis may require renewed retrieval, competition among alternatives, and updating of the situation model.

**Computational operations:**

- Reopen candidate meanings when the coherence check fails.
- Generate new candidates from the disconfirming evidence.
- Reweight prior expectations according to the reliability of the new information.
- Revise the interpretation and downstream inferences.
- Retain a record of the earlier hypothesis when it matters for explaining the processing history.

A simple prediction-error update is:

\[
\delta_t=y_t-\hat{y}_t
\]

where \(y_t\) is the observed feature or event and \(\hat{y}_t\) is what the current model expected. The direction and size of revision depend on reliability, uncertainty, and the structure of the model; the brain does not necessarily use a single scalar prediction error for all comprehension.

**Output:** A corrected or qualified situation model, with confidence recalculated.

## Step 15 — Adapt contextual weighting for the next input

**Objective:** Use the current episode to maintain a context model that is responsive to changing topics, settings, speakers, and goals.

**Biological interpretation:** Listeners adapt expectations as discourse and environmental regularities unfold. The active topic may change; a new speaker may use terms differently; a new task may make previously irrelevant information important. Frontal control and memory systems help maintain and update longer-timescale context, while semantic processing remains sensitive to local linguistic cues.

**Computational operations:**

Update the relevance weights based on the usefulness and reliability of previous contextual cues:

\[
w_j^{(t+1)}=(1-\eta)w_j^{(t)}+\eta\,\tilde{w}_j^{(t)}
\]

where \(\tilde{w}_j^{(t)}\) is a revised weight based on recent evidence and \(\eta\) is an adaptation rate. A small \(\eta\) gives stable context expectations; a larger value adapts faster but may overreact to a single unusual event.

The model should also detect context boundaries—for example, the start of a new topic, a change of location, or a speaker shift—so that old assumptions do not automatically carry over.

**Output:** An updated context state for subsequent interpretation.

## Step 16 — Produce a task-appropriate interpretation or action

**Objective:** Make the result useful for the current goal without requiring every possible inference to be resolved.

**Biological interpretation:** Comprehension supports behaviour: responding in conversation, following an instruction, answering a question, locating an object, or predicting an event. Different goals require different levels of detail. A person following “meet me by the bank” may need a location, while someone reading a financial report may need the institutional meaning and related facts.

**Computational operations:**

- Select the relevant portion of the situation model.
- Produce a verbal interpretation, decision, prediction, or action plan.
- Preserve uncertainty where the evidence is insufficient.
- Request clarification when ambiguity materially affects the action.

**Output:** A goal-conditioned interpretation with confidence and provenance.

---

# 4. Full mathematical model

A simplified end-to-end representation can be expressed as a Bayesian update over candidate interpretations:

\[
P(h\mid x,C,G)\propto P(x\mid h,C)P(h\mid C,G)
\]

where:

- \(h\) is a candidate meaning or situation hypothesis.
- \(x\) is the incoming linguistic/perceptual evidence.
- \(C\) is the composite context (discourse, knowledge, scene, and conventions).
- \(G\) is the current goal.

The context itself is dynamic. It is constructed from a set of sources:

\[
C_t=\{D_t,K_t,E_t,S_t,M_t\}
\]

where \(D_t\) is recent discourse, \(K_t\) is retrieved long-term knowledge, \(E_t\) is environmental evidence, \(S_t\) is speaker/social/conventional information, and \(M_t\) is the current situation model. In practice these sources overlap and are not statistically independent. The expression is an organizational model, not a claim that the brain stores five separate registers.

Contextual selection may be written as:

\[
C_t^*=\operatorname{Select}(C_t\mid G_t,\text{relevance},\text{reliability})
\]

The interpretation is then updated:

\[
P(h_t\mid x_{1:t},C_{1:t},G_{1:t})
\]

and used to update the situation model:

\[
M_{t+1}=U(M_t,x_t,h_t^*)
\]

The important computational principle is **reciprocal constraint**: context changes the interpretation of the input, but the input also changes the context model. Context is not a static label applied before comprehension begins.

## 5. Integrated pseudocode

```python
from dataclasses import dataclass, field
from math import exp, log
from typing import Any, Dict, List, Optional


@dataclass
class Candidate:
    meaning: str
    prior_score: float = 0.0
    evidence_score: float = 0.0
    posterior: float = 0.0
    assumptions: List[str] = field(default_factory=list)


@dataclass
class SituationModel:
    entities: Dict[str, Any] = field(default_factory=dict)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    inferences: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_conflicts: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ContextState:
    discourse: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_knowledge: List[Dict[str, Any]] = field(default_factory=list)
    scene: Dict[str, Any] = field(default_factory=dict)
    conventions: List[Dict[str, Any]] = field(default_factory=list)
    goal: Optional[str] = None
    source_weights: Dict[str, float] = field(default_factory=dict)
    situation: SituationModel = field(default_factory=SituationModel)


def softmax(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    peak = max(scores.values())
    values = {key: exp(value - peak) for key, value in scores.items()}
    total = sum(values.values())
    if total <= 0:
        raise ValueError("Candidate scores cannot all be non-positive after normalization")
    return {key: value / total for key, value in values.items()}


def entropy(probabilities: Dict[str, float]) -> float:
    return -sum(p * log(p) for p in probabilities.values() if p > 0)


def construct_context(
    input_item: Dict[str, Any],
    state: ContextState,
    retrieve_knowledge,
    inspect_scene,
    estimate_goal_weights,
) -> ContextState:
    """Prepare a goal-conditioned context; callbacks supply domain-specific models."""
    state.retrieved_knowledge = retrieve_knowledge(
        input_item=input_item,
        discourse=state.discourse,
        current_situation=state.situation,
    )
    state.scene = inspect_scene(input_item)
    state.source_weights = estimate_goal_weights(
        goal=state.goal,
        discourse=state.discourse,
        knowledge=state.retrieved_knowledge,
        scene=state.scene,
    )
    return state


def interpret_with_context(
    input_item: Dict[str, Any],
    state: ContextState,
    generate_candidates,
    score_context_fit,
    score_input_fit,
    threshold: float = 0.70,
) -> Dict[str, Any]:
    """Rank interpretations while retaining ambiguity when evidence is weak."""
    candidates: List[Candidate] = generate_candidates(input_item, state)
    if not candidates:
        return {
            "status": "unresolved",
            "reason": "No candidate interpretation was generated",
            "situation": state.situation,
        }

    raw_scores: Dict[str, float] = {}
    for candidate in candidates:
        candidate.prior_score = score_context_fit(candidate, state)
        candidate.evidence_score = score_input_fit(candidate, input_item, state)
        raw_scores[candidate.meaning] = (
            candidate.prior_score + candidate.evidence_score
        )

    probabilities = softmax(raw_scores)
    for candidate in candidates:
        candidate.posterior = probabilities[candidate.meaning]

    ranked = sorted(candidates, key=lambda item: item.posterior, reverse=True)
    best = ranked[0]
    margin = best.posterior - (ranked[1].posterior if len(ranked) > 1 else 0.0)
    uncertainty = entropy(probabilities)

    if best.posterior < threshold or margin < 0.15:
        return {
            "status": "ambiguous",
            "candidates": ranked,
            "uncertainty": uncertainty,
            "situation": state.situation,
        }

    return {
        "status": "provisional_selection",
        "selected": best,
        "candidates": ranked,
        "uncertainty": uncertainty,
        "situation": state.situation,
    }


def update_situation_model(
    state: ContextState,
    input_item: Dict[str, Any],
    interpretation: Candidate,
    integrate_event,
    check_coherence,
) -> ContextState:
    """Integrate only supported content and preserve unresolved contradictions."""
    state.situation = integrate_event(
        existing_model=state.situation,
        input_item=input_item,
        interpretation=interpretation,
        provenance={"input": "observed_or_stated", "meaning": interpretation.meaning},
    )
    conflicts = check_coherence(state.situation)
    state.situation.unresolved_conflicts.extend(conflicts)
    state.situation.confidence = interpretation.posterior
    state.discourse.append({
        "input": input_item,
        "interpretation": interpretation.meaning,
        "confidence": interpretation.posterior,
    })
    return state


# The complete cycle is iterative:
# 1. Encode new input.
# 2. Retrieve active discourse and relevant knowledge.
# 3. Weight context according to the current goal.
# 4. Generate candidate interpretations.
# 5. Combine contextual priors with input evidence.
# 6. Preserve ambiguity when confidence is insufficient.
# 7. Update the situation model and check coherence.
# 8. Reinterpret when later evidence contradicts the current model.
```

The pseudocode is a software-level sketch. Functions such as `retrieve_knowledge`, `generate_candidates`, `score_context_fit`, and `check_coherence` require explicit implementations and should not be mistaken for direct descriptions of individual brain regions.

---

# 6. Important mathematical operations and their interpretation

| Mathematical operation | Computational role | Biological interpretation (cautious) |
|---|---|---|
| Feature extraction | Convert sensory/language input into usable features | Distributed sensory and language processing |
| Similarity / cue matching | Retrieve relevant discourse and long-term knowledge | Associative memory retrieval |
| Weighted relevance | Prioritize context according to goal | Selective attention and control |
| Bayesian prior and likelihood | Combine expectation with actual evidence | Predictive/context-sensitive interpretation |
| Softmax normalization | Express relative candidate strength | Abstract model of graded competition |
| Entropy | Quantify unresolved ambiguity | Uncertainty monitoring in a computational model |
| Graph construction | Represent entities and their relations | Situation/event representation |
| Prediction error | Detect mismatch and trigger revision | Error-sensitive updating; not a single universal neural signal |
| Incremental state update | Integrate new content with the current model | Discourse and memory updating |

No single equation above should be interpreted as a direct quantitative measurement of neuronal firing. Neural activity is implemented by populations of neurons, synapses, oscillatory coordination, recurrent circuits, and neuromodulatory influences. The equations formalize computational functions that can be implemented by many different neural mechanisms.

# 7. Biological evidence and observable signatures

## 7.1 Context changes semantic processing

Electrophysiological research shows that the N400 response to meaningful input is influenced by preceding linguistic and world knowledge. Words that fit a supportive context typically elicit a reduced N400 relative to less expected or contextually incongruent words. This is evidence that context affects meaning-related processing, but the N400 should not be reduced to a simple prediction-error meter: it is sensitive to several factors affecting the accessibility and processing of meaning.

## 7.2 Context-rich comprehension recruits broader networks

Studies comparing contextually supported and unsupported descriptions show that adding a meaningful scene or discourse context can change activity beyond traditional language-responsive regions. Medial prefrontal and posterior midline regions are among the areas implicated in integrating text with a broader situation model. This supports a distributed view in which language systems interact with memory and world-knowledge systems.

## 7.3 Situation models are continuously updated

Research on discourse and narrative comprehension describes situation models as dynamic representations that integrate text information with prior knowledge. New events are indexed to information such as time, location, and participants, then integrated into the evolving model. Newer reviews extend situation-model work beyond written text to spoken language, visual narratives, and real-world events.

## 7.4 Contextual predictions remain flexible

Context can influence expectations before the next word arrives, but prediction varies with the strength of the context, the listener's knowledge, and the task. The system must balance anticipation with sensitivity to unexpected evidence. A model that predicts too aggressively will misinterpret novel or surprising information; a model that ignores context will be inefficient and incoherent.

# 8. Failure modes and safeguards

### 8.1 Context overrules the actual input

A strongly expected interpretation can bias the listener toward hearing or reading what was anticipated. Safeguard: maintain a separate score for direct input fit and require strong sensory/linguistic evidence before accepting a contextually attractive candidate.

### 8.2 Irrelevant context is retrieved

Associative retrieval may activate knowledge that is related but not useful. Safeguard: use the active goal and current discourse to weight relevance; retain the ability to suppress low-relevance memories.

### 8.3 Old context persists after a topic change

A previous topic or speaker model can bias interpretation after the situation has changed. Safeguard: detect context boundaries and reduce the weight of stale assumptions.

### 8.4 Cultural assumptions are treated as universal

A familiar convention may not be shared by the current speaker. Safeguard: model convention applicability probabilistically and seek clarification when the stakes are high.

### 8.5 The model fills gaps with unsupported inferences

A situation model often requires inference, but inferred content is not the same as explicitly stated content. Safeguard: keep provenance labels and distinguish observation, statement, memory retrieval, and inference.

### 8.6 Premature commitment to one meaning

The system may settle on an interpretation before disambiguating information arrives. Safeguard: preserve multiple candidates when posterior probabilities are close or overall uncertainty remains high.

### 8.7 Excessive uncertainty prevents useful action

The system may retain too many alternatives even when one is clearly adequate for the task. Safeguard: use task-sensitive thresholds and allow provisional decisions with calibrated confidence.

# 9. Validation and experimental predictions

A computational implementation inspired by this algorithm should be tested against controlled conditions rather than judged only by whether its final answer seems plausible.

1. **Context reversal test:** Present the same ambiguous phrase after two different preceding contexts. The preferred interpretation should shift with the context, while the lexical input remains fixed.
2. **Context strength test:** Vary how strongly the preceding discourse supports one meaning. Candidate probabilities should become more concentrated as context becomes more constraining, but contradictory input should still be able to reverse the decision.
3. **Goal manipulation test:** Present identical information under different tasks (for example, identify a location versus interpret a financial statement). The system should change which context features it weights most strongly.
4. **Physical-scene congruence test:** Pair identical spoken phrases with different relevant scenes. The model should use scene evidence when it is informative and avoid using unrelated visual details.
5. **Topic-shift test:** Introduce an explicit discourse boundary or a new speaker. Context weights should adapt rather than carrying the former assumptions forward unchanged.
6. **Delayed-disambiguation test:** Present an initially ambiguous phrase followed by a disambiguating sentence. The model should revise an earlier provisional interpretation when the later evidence requires it.
7. **Contradiction test:** Make the incoming statement conflict with a strong world-knowledge expectation. The model should detect the conflict but must not automatically reject the statement; unusual events can be real.
8. **Provenance test:** Ask the system to distinguish what was stated, what was visible, what came from prior knowledge, and what it inferred. This tests whether contextual completion is being mistaken for direct evidence.
9. **Uncertainty calibration test:** Compare confidence values with actual accuracy across many examples. Confidence should fall when contexts are weak, noisy, or conflicting.
10. **Neural-signature comparison (research setting):** In human experiments, vary contextual support and examine behavioural facilitation, eye movements, and electrophysiological responses such as the N400. Neuroimaging can test whether context-rich comprehension recruits broader memory and midline networks. These are group-level research measures and should not be interpreted as direct readings of the algorithm's internal variables.

# 10. What is established and what is modeled

**Well-supported at the broad level:** Meaning-related processing is context sensitive; prior discourse and world knowledge influence interpretation; situation models integrate information across sentences and events; task and scene context can change comprehension; and context-rich understanding can engage networks beyond the core language-selective system.

**Model-dependent or still debated:** The exact timing and representational format of predictions, how much specific content is pre-activated before it appears, the precise division of labour among semantic, memory, and control networks, and whether a given neural signal reflects prediction, retrieval, integration, or several processes together.

**Engineering abstractions in this document:** The explicit Bayesian equations, source-weight vectors, entropy thresholds, graph update rules, and pseudocode are implementable ways to reproduce selected functional properties. They are not claims that neurons literally calculate probabilities, cosine similarity, entropy, or softmax values in the same form.

# 11. References and further reading

1. Hagoort, P. (2017). The meaning-making mechanism(s) behind the eyes and between the ears. *Philosophical Transactions of the Royal Society B*, 372(1714), 20160094. https://doi.org/10.1098/rstb.2016.0094
2. Kutas, M., & Federmeier, K. D. (2011). Thirty years and counting: Finding meaning in the N400 component of the event-related brain potential (ERP). *Annual Review of Psychology*, 62, 621–647. https://doi.org/10.1146/annurev.psych.093008.131123
3. Zwaan, R. A., & Radvansky, G. A. (1998). Situation models in language comprehension and memory. *Psychological Bulletin*, 123(2), 162–185. https://doi.org/10.1037/0033-2909.123.2.162
4. Zwaan, R. A. (2025). From words to worlds: Twenty-five years of advances in situation model research. *Current Directions in Psychological Science*. https://doi.org/10.1177/09637214251326812
5. Willems, R. M., & colleagues (2018). Grounding the neurobiology of language in first principles: The necessity of non-language-centric explanations for language comprehension. *Cognition*, 180, 135–144. https://pmc.ncbi.nlm.nih.gov/articles/PMC6145924/
6. “How context changes the neural basis of perception and language.” (2021). Review article. https://pmc.ncbi.nlm.nih.gov/articles/PMC8091044/
7. “Semantic Representations during Language Comprehension Are Affected by Context.” (2023). *Journal of Neuroscience*. https://pmc.ncbi.nlm.nih.gov/articles/PMC10146529/
8. “The neural and cognitive basis of expository text comprehension.” (2024). Review article. https://pmc.ncbi.nlm.nih.gov/articles/PMC10957871/
9. “Prediction during language comprehension: what is next?” (2024). Review article. https://pmc.ncbi.nlm.nih.gov/articles/PMC11614350/
10. Kawasaki, E. (2024; available online 2025). Research trends in situation models for text comprehension: From construction to verification processes in situation models. *Japanese Journal of Psychology*, 67(2), 191–209. https://doi.org/10.24602/sjpr.67.2_191

---

## End of Point 12

This file covers only context-dependent interpretation through prior knowledge, current goals, discourse, culture/conventions, and physical environment. The next point in the source sequence concerns emotion and motivation as modulators of attention and learning. That point is intentionally not developed here.
