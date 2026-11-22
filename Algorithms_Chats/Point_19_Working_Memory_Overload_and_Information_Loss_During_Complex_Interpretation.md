# Point 19 — Working-Memory Overload and Information Loss During Complex Interpretation

## Source-sequence position

This is the nineteenth brain-related point in **Learning, Memory, and Concept Formation.txt**. It follows the point on attentional failure, distraction, and missed information.

The source statement is:

> Overloaded working memory (e.g. trying to absorb an overly long sentence) causes parts of a message to be forgotten mid-interpretation.

This document expands that statement into a biologically informed computational algorithm. It focuses on the temporary maintenance, retrieval, and integration of information while a sentence or other complex message is still being interpreted.

---

## 1. Objective and computational problem

The algorithm must construct a coherent interpretation from a sequence of incoming units while earlier units may still be needed. A word or phrase can be understood locally but cannot yet be fully interpreted until a later word arrives. For example, a listener may need to retain the subject of a sentence until the verb appears, preserve the beginning of a relative clause until its ending is reached, or keep the referent of a pronoun accessible until the pronoun is resolved.

The core problem is therefore not merely to store a list of words. The system must:

1. encode the identity and relevant properties of recent words or phrases;
2. maintain relationships that have not yet been resolved;
3. retrieve the appropriate earlier item when a later cue arrives;
4. integrate retrieved information with the current phrase and the evolving event or situation model;
5. manage competition from similar or irrelevant representations;
6. detect when processing demands are exceeding the system's current ability to maintain and use the required information; and
7. recover, revise, or preserve uncertainty when a dependency can no longer be resolved reliably.

A long sentence is not automatically difficult simply because it contains many words. Familiar vocabulary, predictable structure, meaningful chunking, prior knowledge, and a coherent context can reduce processing demands. Conversely, a shorter sentence can be difficult when it contains unfamiliar terms, several competing referents, nested dependencies, ambiguity, or a concurrent task.

### Required input

- A stream of perceptual-linguistic units (spoken or written words, morphemes, phrases, punctuation, prosody, and pauses).
- Current task goals and attentional priorities.
- Activated lexical, grammatical, semantic, and world-knowledge representations.
- The partial interpretation already constructed.
- Available contextual cues and recent discourse information.
- Current processing conditions, including competing tasks and fatigue-related constraints when relevant.

### Required output

- An incrementally updated interpretation or situation model.
- A set of active unresolved dependencies and their links to earlier material.
- Confidence values for retrieved and integrated information.
- A load estimate and indicators of possible overload.
- A record of unresolved, weakly supported, or lost information that requires reanalysis or clarification.

---

## 2. Biological architecture: interacting systems, not a single memory container

Working memory is a functional description of information that is temporarily accessible for ongoing cognition. It is not a single anatomical box, and research does not establish one universal fixed number of items that every person can maintain in every task. Performance depends on the nature of the material, the operations being performed, attentional control, interference, familiarity, task goals, and the opportunities to form meaningful chunks.

For language comprehension, several partly overlapping systems contribute:

### 2.1 Language-processing circuitry

Left-lateralized frontotemporal language regions contribute to lexical, phonological, syntactic, and semantic operations. Posterior and middle temporal regions participate in mapping linguistic forms to lexical and conceptual information. Inferior frontal regions contribute to aspects of controlled retrieval, unification, and processing under demanding conditions. These are interacting components of a distributed language network; no single region should be described as the entire working-memory system or as the sole location of sentence comprehension.

### 2.2 Attentional and executive control

Frontal and parietal control systems help maintain task goals, prioritize relevant information, select among competing representations, and regulate interference. Dorsolateral and ventrolateral prefrontal regions, anterior cingulate and related control circuitry, and parietal regions may contribute differently depending on the task. Their involvement is dynamic rather than a fixed one-region/one-function mapping.

### 2.3 Temporary verbal and phonological maintenance

Phonological representations can remain temporarily accessible, especially when speech sounds or word forms must be retained. Rehearsal and attentional refreshing are often proposed as maintenance mechanisms. The relative contribution of phonological maintenance changes with linguistic modality, task, and the kind of information that must be remembered.

### 2.4 Event and episodic representation

Comprehenders construct an evolving representation of who did what, to whom, where, when, and why. Medial cortical and episodic-memory systems may contribute to constructing or maintaining this broader event representation, particularly when the task requires connecting a sentence to prior context. This is not equivalent to claiming that every sentence is stored as an episodic memory.

### 2.5 Retrieval and interference

An earlier element may become less accessible as more material is processed. Similar items can compete: several people mentioned in a sentence may all partially match a later pronoun or verb cue. Retrieval can therefore fail even when the original information has not simply vanished. Interference-based explanations are important alongside time-dependent decreases in activation.

### 2.6 Important scientific qualification

The exact architecture of verbal working memory and sentence comprehension remains debated. Some theories emphasize shared general-purpose resources, others emphasize distinct maintenance mechanisms, and others model memory as activated long-term representations plus a narrow focus of attention. The algorithm below is a functional synthesis of well-supported operations, not a claim that the brain literally executes the pseudocode or that every proposed operation has a unique anatomical location.

---

## 3. Formal representation of the running interpretation

Maintain a structured state rather than a raw unorganized token buffer.

Let the state at processing step \(t\) be:

\[
S_t = (B_t, D_t, E_t, G_t, C_t, L_t, U_t)
\]

where:

- \(B_t\): currently active linguistic and conceptual representations;
- \(D_t\): unresolved dependencies, such as an expected verb, an open clause, or a referent awaiting resolution;
- \(E_t\): the evolving event/situation representation;
- \(G_t\): active goals and task instructions;
- \(C_t\): contextual cues and activated long-term knowledge;
- \(L_t\): estimated processing load;
- \(U_t\): uncertainty and confidence annotations.

Each active representation \(i\) can be described by a feature vector:

\[
\mathbf{x}_i = [\text{lexical identity},\text{semantic features},\text{syntactic role},\text{referent},\text{time},\text{source}]
\]

The feature vector is a computational abstraction. Neural representations are distributed and context-sensitive; there is no evidence that the brain stores each word as this exact fixed vector.

An unresolved dependency can be represented as:

\[
d_j = (q_j, k_j, r_j, a_j, c_j)
\]

where \(q_j\) is the missing information or expected relation, \(k_j\) is the cue expected to resolve it, \(r_j\) is the set of candidate representations, \(a_j\) is current accessibility, and \(c_j\) is confidence in the current partial interpretation.

---

# 4. Detailed algorithmic sequence

## Step 1 — Establish the current comprehension goal

**Purpose:** Determine what the system is trying to understand before allocating limited attention and maintenance effort.

**Biological interpretation:** Goal maintenance and attentional selection recruit distributed control systems. The current task can change which features of a message are prioritized: remembering exact wording, identifying the main event, following instructions, or evaluating an argument impose different demands.

**Operations:**

1. Read the task context and identify the required level of detail.
2. Activate the relevant linguistic and conceptual knowledge.
3. Set priorities for entities, relations, temporal order, negation, and other task-critical information.
4. Establish whether exact form or approximate meaning is required.

**Computational model:** Assign a relevance weight to each information class:

\[
w_f = \operatorname{softmax}(\alpha_g G_f + \alpha_c C_f + \alpha_r R_f)
\]

Here, \(G_f\) is goal relevance, \(C_f\) is contextual relevance, and \(R_f\) is task-defined importance. The softmax is a modeling device for normalized priorities, not a literal neural calculation.

**Output:** An active goal state and an initial priority profile.

## Step 2 — Segment the incoming stream into candidate units

**Purpose:** Convert continuous speech or written text into manageable words, phrases, and structural units.

**Biological interpretation:** Auditory and visual language systems extract speech or orthographic information and progressively organize it into lexical and phrase-level structures. Prosody, punctuation, learned word boundaries, and syntactic expectations assist segmentation.

**Operations:**

1. Detect word or morpheme boundaries where possible.
2. Retrieve candidate lexical entries.
3. identify phrase boundaries and clause markers.
4. preserve punctuation, prosodic boundaries, and pauses as structural cues.
5. keep ambiguous segmentation alternatives alive when evidence is insufficient.

**Mathematical model:** For candidate segmentation \(z\), estimate:

\[
P(z\mid x_{1:t},C_t) \propto P(x_{1:t}\mid z)P(z\mid C_t)
\]

where \(x_{1:t}\) is the observed sequence and \(C_t\) is context. The highest-probability segmentation may be provisionally selected, but alternatives should remain available when posterior probabilities are close.

**Output:** A sequence of candidate linguistic units with confidence values.

## Step 3 — Encode each unit with the features needed for later integration

**Purpose:** Store more than the word's sound or spelling. Preserve features that may be needed later.

**Biological interpretation:** Lexical access and language processing activate information about word form, grammatical properties, meaning, and possible referents. The degree to which a feature remains accessible depends on task demands and ongoing processing.

**Operations:**

1. Retrieve lexical identity and likely meanings.
2. activate grammatical properties such as number, person, tense, and syntactic category;
3. identify possible semantic roles and referents;
4. attach source and position information where useful;
5. assign a provisional confidence score.

**Mathematical abstraction:**

\[
\mathbf{x}_i = f_{lex}(u_i,C_t,G_t)
\]

where \(u_i\) is the incoming unit and \(f_{lex}\) is a function representing context-sensitive feature activation.

**Output:** A feature-rich representation for each incoming unit.

## Step 4 — Create an incremental structural representation

**Purpose:** Record how the new unit relates to what has already been processed.

**Biological interpretation:** Comprehension is incremental. The system often builds partial structure before the entire sentence is available. Syntactic and semantic expectations guide the interpretation of subsequent material.

**Operations:**

1. Assign the unit a provisional phrase or clause position.
2. determine whether it opens or closes a phrase, clause, or dependency;
3. bind it to currently available entities or relations when supported;
4. create an unresolved dependency when a required element has not arrived;
5. preserve competing analyses if the current input is ambiguous.

**Computational model:** Update the structural graph:

\[
E^{parse}_{t+1} = E^{parse}_t \cup \{v_i, e_{i,*}\}
\]

where \(v_i\) is the new node and \(e_{i,*}\) are the supported relations to prior nodes.

**Output:** An updated partial parse and a dependency list.

## Step 5 — Maintain unresolved dependencies

**Purpose:** Keep earlier information available when later words will be needed to complete the interpretation.

Examples include retaining a subject before its verb, a main-clause opening across an embedded clause, or a candidate antecedent until a pronoun appears.

**Biological interpretation:** The system maintains selected representations through active accessibility, attentional focus, temporary verbal maintenance, and retrieval from activated long-term knowledge. These functions are distributed and may rely on partly different neural mechanisms.

**Operations:**

1. Enumerate each unresolved dependency.
2. identify the features needed to resolve it;
3. preserve the most diagnostic information rather than every detail;
4. periodically refresh or reactivate task-relevant representations when possible;
5. reduce priority for information that is no longer needed, without deleting it prematurely.

**Model:** Let accessibility of representation \(i\) be:

\[
A_i(t)=B_i + R_i(t) - I_i(t) - D_i(t)
\]

where \(B_i\) is baseline activation from frequency or prior learning, \(R_i\) is contextual/goal-based support, \(I_i\) is interference, and \(D_i\) is a time-related accessibility loss term. This decomposition is theoretical; it is not a direct reading of neural activation.

**Output:** A prioritized set of still-needed representations.

## Step 6 — Estimate current processing load

**Purpose:** Determine whether ongoing processing is becoming difficult relative to current resources.

**Important qualification:** Do not equate load with a universal item count. The effective demand depends on the kind of representation, number and structure of unresolved relations, interference, processing complexity, familiarity, and chunking.

**Candidate load contributors:**

- number of simultaneously unresolved dependencies;
- complexity of phrase and clause embedding;
- distance between an item and the point where it must be retrieved;
- similarity among competing referents;
- ambiguity and uncertainty;
- concurrent memory or reasoning tasks;
- unfamiliar vocabulary and low-quality prior representations;
- the amount of active restructuring or reanalysis.

**Candidate load reducers:**

- familiar vocabulary and syntax;
- predictive context;
- meaningful chunking;
- stable and distinctive referents;
- well-learned schemas;
- clear prosody, punctuation, and discourse structure;
- external notes or visual structure where the task permits them.

**Illustrative equation:**

\[
L_t = \beta_1N_D + \beta_2C_S + \beta_3D_R + \beta_4I_S + \beta_5A_M + \beta_6U_t - \beta_7H_t - \beta_8K_t
\]

where \(N_D\) is unresolved-dependency count, \(C_S\) structural complexity, \(D_R\) retrieval distance, \(I_S\) similarity-based interference, \(A_M\) additional task demand, \(U_t\) uncertainty, \(H_t\) chunking support, and \(K_t\) knowledge/context support. Coefficients must be empirically estimated for a particular model or task; this equation is a useful engineering abstraction, not an established brain formula.

**Output:** A relative load estimate and a list of its main causes.

## Step 7 — Group information into meaningful chunks

**Purpose:** Reduce the number of separately maintained elements by encoding meaningful combinations.

**Biological interpretation:** Expertise and prior knowledge permit several elements to be treated as one meaningful unit. Familiar phrase structures, named entities, and schemas can reduce the burden of maintaining disconnected details.

**Operations:**

1. Detect familiar phrases and grammatical constructions.
2. bind related words into a phrase-level representation;
3. compress predictable or redundant detail while preserving task-relevant distinctions;
4. retain the internal relations of a chunk so it can later be unpacked;
5. avoid combining elements whose distinction is crucial, such as two similar people or opposing propositions.

**Mathematical model:** If a group \(G\) contains \(m\) elements, represent it as one higher-level node with an internal relation structure:

\[
\operatorname{Chunk}(G)=\left(v_G,\{r_{ij}: i,j\in G\}\right)
\]

An illustrative effective-demand measure is:

\[
N_{eff}=N_{unbound}+\sum_G \gamma_G
\]

where \(0<\gamma_G\leq m_G\) is the estimated cost of maintaining a chunk with \(m_G\) elements. A highly familiar chunk may have a lower estimated cost, but chunking does not make its information free or immune to interference.

**Output:** A more compact structured representation.

## Step 8 — Allocate attention between processing and maintenance

**Purpose:** Balance the need to analyze new material with the need to preserve earlier material.

**Biological interpretation:** Working memory involves both maintaining task-relevant representations and controlling attention. Reading or listening while holding earlier material in mind creates a coordination demand. The allocation is flexible and may change during comprehension.

**Operations:**

1. identify the most urgent unresolved dependency;
2. estimate which current operation is necessary to interpret the new input;
3. prioritize critical dependencies over redundant details;
4. refresh or retrieve needed material when its accessibility is low;
5. avoid spending all available control on one interpretation if alternatives remain plausible.

**Optimization abstraction:**

\[
\max_{\mathbf{a}}\; \sum_i v_i a_i - \lambda\,\operatorname{Cost}(\mathbf{a})
\]

subject to a task-dependent resource constraint. Here \(a_i\) is attention allocated to operation \(i\), \(v_i\) its task value, and \(\lambda\) the cost sensitivity. This formalizes a trade-off; it is not evidence that neural systems solve an explicit mathematical optimization problem.

**Output:** A current priority schedule for comprehension and maintenance.

## Step 9 — Protect relevant representations from similarity-based interference

**Purpose:** Reduce confusion when several active items share features.

**Example:** If a sentence mentions two or more people of the same grammatical gender, a later pronoun may match multiple antecedents. A complex clause may also contain several nouns that compete to be retrieved at a verb.

**Biological interpretation:** Retrieval cues activate candidate representations. Similarity-based competition can reduce the accessibility of the intended item. Control systems help select goal-relevant information and downregulate irrelevant competitors.

**Operations:**

1. identify representations sharing the current retrieval cues;
2. compare their syntactic, semantic, positional, and discourse features;
3. privilege candidates supported by multiple independent cues;
4. suppress or deprioritize clearly irrelevant competitors;
5. retain uncertainty when the evidence does not uniquely identify a candidate.

**Illustrative model:**

\[
I_i = \sum_{j\ne i} s(\mathbf{x}_i,\mathbf{x}_j)\,a_j\,g_j
\]

where \(s\) is feature similarity, \(a_j\) is competitor accessibility, and \(g_j\) captures current cue overlap. A possible similarity function is cosine similarity:

\[
s(\mathbf{x}_i,\mathbf{x}_j)=\frac{\mathbf{x}_i\cdot\mathbf{x}_j}{\|\mathbf{x}_i\|\|\mathbf{x}_j\|}
\]

The formula is one possible computational implementation, not a literal biological computation.

**Output:** Ranked retrieval candidates with an interference estimate.

## Step 10 — Retrieve earlier material when a new cue arrives

**Purpose:** Recover the specific prior item required to interpret the current word or phrase.

**Biological interpretation:** Retrieval is cue-dependent. A verb, pronoun, connective, or semantic relation can reactivate earlier material. The strongest candidate is not always the correct one if interference is high or the cue is weak.

**Operations:**

1. extract retrieval cues from the current unit;
2. search accessible active and long-term representations;
3. compute candidate match from grammatical, semantic, positional, and discourse cues;
4. include accessibility and competition in the decision;
5. retrieve the best-supported candidate or preserve multiple candidates.

**Probabilistic model:**

\[
P(i\mid q)=\frac{\exp(\tau M(i,q))}{\sum_j\exp(\tau M(j,q))}
\]

where \(M(i,q)\) is the match between candidate \(i\) and retrieval cue \(q\), and \(\tau\) controls how sharply the model favors the best match. If the top two candidates have similar probabilities, retrieval confidence should remain low.

**Output:** A retrieved representation and confidence estimate, or an unresolved dependency.

## Step 11 — Integrate the retrieved item with the incoming material

**Purpose:** Bind the current word or phrase to the appropriate earlier element and update the meaning.

**Biological interpretation:** Language networks integrate lexical, syntactic, and semantic information incrementally. Under higher demand, additional control and memory systems may be recruited. Event-level integration may connect sentence meaning with discourse context.

**Operations:**

1. check grammatical compatibility;
2. check semantic plausibility;
3. check consistency with discourse context;
4. close or update the relevant dependency;
5. revise the evolving event representation;
6. preserve unresolved ambiguity where no candidate is sufficiently supported.

**Gated-update model:**

\[
E_{t+1}=E_t + \eta_t\,\Delta E_t
\]

where \(\Delta E_t\) is the proposed update and \(\eta_t\) is a confidence-sensitive integration weight. A high-confidence update receives greater weight; an uncertain update is provisional. This is an abstract update equation, not a claim about one neural learning rule.

**Output:** A revised partial meaning and updated dependency list.

## Step 12 — Detect overload through converging performance signals

**Purpose:** Detect when the current interpretation is becoming unreliable.

**Possible behavioral signs:**

- slowing or unusually long pauses;
- repeated rereading or regressions to earlier words;
- loss of a previously available referent;
- failure to connect a clause to its opening;
- inconsistent answers to questions about the same sentence;
- omitted relations or reduced recall of earlier elements;
- repeated reinterpretation or unstable parsing.

No single sign proves that working memory is overloaded. Slowing can also reflect careful analysis, unfamiliarity, visual difficulty, or low confidence. Overload should be inferred from a pattern of signals in relation to task demands.

**Mathematical detection model:**

\[
O_t = \sigma(\theta_0 + \theta_1L_t + \theta_2T_t + \theta_3R_t + \theta_4E_t - \theta_5C_t)
\]

where \(L_t\) is estimated load, \(T_t\) is processing-time deviation, \(R_t\) is reanalysis frequency, \(E_t\) is observed error evidence, \(C_t\) is contextual support, and \(\sigma\) is a logistic function. The output \(O_t\) is a modeled overload probability, not a directly measurable neural quantity.

**Output:** A graded overload warning with an explanation of likely contributing factors.

## Step 13 — Identify which information is at risk

**Purpose:** Determine whether the problem concerns word identity, order, syntactic role, referent, or a relation between phrases.

**Biological interpretation:** Not all representations fail in the same way. The lexical item may remain familiar while its sentence role is lost; the entity may be remembered while its relation to another entity becomes unclear. Retrieval interference and incomplete binding can produce different errors.

**Operations:**

1. inspect each unresolved dependency;
2. inspect the accessibility of its required elements;
3. determine whether the missing information is an item, order, binding, or relation;
4. identify competing candidates;
5. label the partial interpretation with the specific uncertainty.

**Information-risk score:**

\[
R_i = w_1(1-A_i)+w_2I_i+w_3D_i+w_4U_i
\]

where \(A_i\) is accessibility normalized to [0,1], \(I_i\) is interference, \(D_i\) is dependency demand, and \(U_i\) is uncertainty. This score prioritizes recovery but is a designed model.

**Output:** A ranked list of at-risk representations and relations.

## Step 14 — Apply a recovery strategy before information is irretrievably omitted from the active interpretation

**Purpose:** Reduce load or restore access to crucial information.

**Possible recovery operations:**

1. **Reactivation:** return attention to the earlier phrase or reactivate its representation from context.
2. **Rehearsal or refreshing:** revisit a critical word or relation where task conditions permit.
3. **Chunk restructuring:** group a phrase into a familiar grammatical or semantic unit.
4. **Selective prioritization:** retain the main event relation and task-critical distinctions before peripheral detail.
5. **Reanalysis:** reconsider the current parse using the newly available cue.
6. **Contextual support:** use discourse, world knowledge, prosody, punctuation, or a diagram to disambiguate.
7. **External support:** in an engineered or practical system, keep notes, highlight dependencies, or request that the sentence be divided into shorter units.

**Computational decision rule:** choose action \(a\) to maximize expected recovery while limiting added demand:

\[
a^*=\arg\max_a \left[\mathbb{E}(\text{accuracy gain}\mid a)-\lambda\,\text{extra load}(a)-\mu\,\text{time cost}(a)\right]
\]

**Output:** A selected recovery action and an updated state.

## Step 15 — Distinguish weak retrieval from actual omission

**Purpose:** Avoid treating every retrieval failure as permanent forgetting.

**Biological interpretation:** An item may remain encoded but temporarily inaccessible because the wrong cue was used, a similar competitor won retrieval, or attention moved elsewhere. In other cases, the item may not have been encoded sufficiently or its relation may never have been bound successfully.

**Operations:**

1. test an alternative cue;
2. check whether the item can be recovered from a paraphrase or context;
3. determine whether the item is recognized but its role is unknown;
4. distinguish failed encoding, failed maintenance, failed retrieval, and failed integration;
5. preserve the appropriate uncertainty label.

**Diagnostic states:**

- `ENCODING_WEAK`: the representation was not established reliably;
- `MAINTENANCE_WEAK`: the representation became inaccessible while needed;
- `RETRIEVAL_INTERFERENCE`: a competing representation dominated;
- `BINDING_FAILURE`: the items are present but their relation is unclear;
- `INTEGRATION_FAILURE`: the retrieved elements were not incorporated coherently;
- `RECOVERABLE`: an alternative cue restores the information.

**Output:** A failure-type diagnosis rather than a generic “forgotten” label.

## Step 16 — Reconstruct only what is supported; do not silently invent missing relations

**Purpose:** Prevent overload-related gaps from being filled automatically with an unjustified interpretation.

**Biological interpretation:** Comprehension uses prediction and prior knowledge to fill gaps, which is normally useful but can also create confident errors. When input is incomplete, the system may select a schema-consistent interpretation even when the original message did not justify it.

**Operations:**

1. generate candidate reconstructions from surviving cues;
2. compare each candidate with grammar, semantics, context, and source evidence;
3. mark inferred information separately from directly retained information;
4. lower confidence when several alternatives remain plausible;
5. request clarification or preserve an unresolved state when necessary.

**Bayesian abstraction:**

\[
P(h\mid e,C)\propto P(e\mid h,C)P(h\mid C)
\]

where \(h\) is a candidate interpretation, \(e\) is surviving evidence, and \(C\) is context. Prior knowledge can help recover a likely interpretation but cannot guarantee that it matches the original message.

**Output:** A supported reconstruction, an explicitly inferred reconstruction, or an unresolved interpretation.

## Step 17 — Check global coherence and consistency

**Purpose:** Verify that the updated sentence or message makes sense as a whole.

**Operations:**

1. check agreement and dependency closure;
2. verify entity identity and semantic roles;
3. check temporal and causal relations;
4. compare the new event representation with prior discourse;
5. detect contradictions or missing links;
6. reopen the relevant dependency if coherence fails.

**Model:**

\[
Q_t = \alpha G_t + \beta S_t + \gamma D_t - \delta X_t
\]

where \(G_t\) is grammatical consistency, \(S_t\) semantic coherence, \(D_t\) discourse fit, and \(X_t\) unresolved contradictions. The result is a model-specific coherence score.

**Output:** A coherence assessment and any dependency that must be reopened.

## Step 18 — Commit the current partial interpretation with confidence labels

**Purpose:** Preserve a usable representation without pretending that uncertain details are certain.

**Operations:**

1. store the main proposition or event structure;
2. attach confidence to entities, actions, roles, and relations;
3. preserve unresolved alternatives;
4. separate explicitly stated content from inferred content;
5. retain only details relevant to the current task in the active state;
6. allow later context to revise the interpretation.

**Output:** A compact, revisable interpretation with provenance and uncertainty annotations.

## Step 19 — Update the model of task difficulty for the next segment

**Purpose:** Adapt processing strategy as the sentence or conversation continues.

**Operations:**

1. compare predicted difficulty with actual processing signals;
2. adjust load-estimation parameters;
3. identify whether complexity, similarity, ambiguity, or divided attention caused the greatest difficulty;
4. change priorities or chunking strategy;
5. prepare the unresolved dependency set for the next input.

**Mathematical model:**

\[
\beta_{t+1}=\beta_t+\rho\,\nabla_{\beta}\mathcal{L}_{error}
\]

This describes an adaptive engineering model. It should not be interpreted as literal gradient descent in the brain. Biological adaptation occurs through complex learning and control mechanisms over different timescales.

**Output:** Updated task parameters and a state ready for the next input.

---

# 5. Integrated pseudocode

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import exp
from typing import Any, Iterable


class FailureType(str, Enum):
    NONE = "none"
    ENCODING_WEAK = "encoding_weak"
    MAINTENANCE_WEAK = "maintenance_weak"
    RETRIEVAL_INTERFERENCE = "retrieval_interference"
    BINDING_FAILURE = "binding_failure"
    INTEGRATION_FAILURE = "integration_failure"
    UNRESOLVED = "unresolved"


@dataclass
class Representation:
    identifier: str
    text: str
    features: dict[str, Any]
    accessibility: float = 1.0
    confidence: float = 0.5
    task_relevance: float = 0.5
    timestamp: int = 0
    chunk_id: str | None = None


@dataclass
class Dependency:
    identifier: str
    required_features: tuple[str, ...]
    candidate_ids: list[str] = field(default_factory=list)
    confidence: float = 0.5
    open: bool = True
    failure_type: FailureType = FailureType.NONE


@dataclass
class InterpretationState:
    active: dict[str, Representation] = field(default_factory=dict)
    dependencies: dict[str, Dependency] = field(default_factory=dict)
    event_model: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    current_goal: dict[str, float] = field(default_factory=dict)
    estimated_load: float = 0.0
    overload_probability: float = 0.0
    uncertainty: list[str] = field(default_factory=list)
    next_index: int = 0


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def similarity(a: Representation, b: Representation) -> float:
    """Simple feature-overlap proxy; not a neural similarity measure."""
    common = set(a.features) & set(b.features)
    if not common:
        return 0.0
    matches = sum(a.features[k] == b.features[k] for k in common)
    return matches / max(1, len(common))


def estimate_interference(
    target: Representation,
    candidates: Iterable[Representation],
) -> float:
    total = 0.0
    for candidate in candidates:
        if candidate.identifier == target.identifier:
            continue
        total += similarity(target, candidate) * candidate.accessibility
    return clamp01(total / 3.0)


def estimate_load(state: InterpretationState, new_unit_complexity: float,
                  extra_task_load: float = 0.0) -> float:
    open_dependencies = [d for d in state.dependencies.values() if d.open]
    dependency_cost = min(1.0, len(open_dependencies) / 5.0)
    competition = 0.0
    active = list(state.active.values())
    if active:
        competition = sum(
            estimate_interference(item, active)
            for item in active
        ) / len(active)
    chunk_credit = sum(1 for x in active if x.chunk_id is not None)
    chunk_relief = min(0.25, chunk_credit * 0.02)
    load = (
        0.35 * dependency_cost
        + 0.25 * competition
        + 0.25 * clamp01(new_unit_complexity)
        + 0.15 * clamp01(extra_task_load)
        - chunk_relief
    )
    return clamp01(load)


def encode_unit(unit: str, state: InterpretationState,
                features: dict[str, Any]) -> Representation:
    identifier = f"u{state.next_index}"
    return Representation(
        identifier=identifier,
        text=unit,
        features=features,
        accessibility=1.0,
        confidence=0.65,
        task_relevance=0.5,
        timestamp=state.next_index,
    )


def retrieve_candidate(dependency: Dependency,
                       candidates: list[Representation]) -> tuple[Representation | None, float]:
    scored: list[tuple[float, Representation]] = []
    for item in candidates:
        score = 0.0
        for feature in dependency.required_features:
            if feature in item.features:
                score += 1.0
        score = score / max(1, len(dependency.required_features))
        score = 0.65 * score + 0.35 * item.accessibility
        scored.append((score, item))

    if not scored:
        return None, 0.0
    scored.sort(key=lambda pair: pair[0], reverse=True)
    best_score, best_item = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    confidence = clamp01(best_score * (0.5 + 0.5 * (best_score - second_score)))
    return best_item, confidence


def detect_overload(state: InterpretationState, observed_slowing: float,
                    reanalysis_rate: float, error_rate: float) -> float:
    z = (
        -2.0
        + 1.0 * state.estimated_load
        + 0.6 * clamp01(observed_slowing)
        + 0.8 * clamp01(reanalysis_rate)
        + 1.0 * clamp01(error_rate)
    )
    probability = 1.0 / (1.0 + exp(-z))
    state.overload_probability = probability
    return probability


def apply_recovery(state: InterpretationState) -> None:
    """Prioritize recovery of open, high-importance dependencies."""
    open_dependencies = sorted(
        (d for d in state.dependencies.values() if d.open),
        key=lambda d: d.confidence,
    )
    for dependency in open_dependencies:
        candidate, confidence = retrieve_candidate(
            dependency, list(state.active.values())
        )
        if candidate is not None and confidence >= 0.65:
            dependency.candidate_ids = [candidate.identifier]
            dependency.confidence = confidence
            dependency.open = False
            dependency.failure_type = FailureType.NONE
        else:
            dependency.failure_type = FailureType.RETRIEVAL_INTERFERENCE
            state.uncertainty.append(
                f"Dependency {dependency.identifier} remains unresolved"
            )


def process_stream(units: list[str], feature_map: dict[str, dict[str, Any]],
                   goals: dict[str, float] | None = None) -> InterpretationState:
    state = InterpretationState(current_goal=goals or {})

    for unit in units:
        # 1–3: select, segment, and encode the incoming unit.
        features = feature_map.get(unit, {"surface": unit})
        representation = encode_unit(unit, state, features)

        # 4: integrate a provisional representation into the active state.
        state.active[representation.identifier] = representation

        # 5: open dependencies supplied by the feature annotation.
        for dependency_spec in features.get("opens_dependency", []):
            dep = Dependency(
                identifier=f"d{len(state.dependencies)}",
                required_features=tuple(dependency_spec),
            )
            state.dependencies[dep.identifier] = dep

        # 6–9: estimate load and protect high-priority information.
        state.estimated_load = estimate_load(
            state,
            new_unit_complexity=float(features.get("complexity", 0.2)),
            extra_task_load=float(features.get("extra_load", 0.0)),
        )

        # 10–11: use current features to resolve compatible dependencies.
        for dep in state.dependencies.values():
            if not dep.open:
                continue
            candidate, confidence = retrieve_candidate(
                dep, list(state.active.values())
            )
            if candidate is not None and confidence >= 0.70:
                dep.candidate_ids = [candidate.identifier]
                dep.confidence = confidence
                dep.open = False

        # 12–16: detect load-related risk and attempt controlled recovery.
        overload = detect_overload(
            state,
            observed_slowing=float(features.get("slowing", 0.0)),
            reanalysis_rate=float(features.get("reanalysis", 0.0)),
            error_rate=float(features.get("error", 0.0)),
        )
        if overload >= 0.60:
            apply_recovery(state)

        # 17–19: commit a provisional event update and advance.
        state.event_model.append({
            "unit": unit,
            "representation_id": representation.identifier,
            "confidence": representation.confidence,
            "load": state.estimated_load,
            "overload_probability": state.overload_probability,
        })
        state.next_index += 1

    # Do not silently close dependencies that were never resolved.
    for dependency in state.dependencies.values():
        if dependency.open:
            dependency.failure_type = FailureType.UNRESOLVED
            state.uncertainty.append(
                f"Unresolved dependency at end: {dependency.identifier}"
            )

    return state
```

### Pseudocode interpretation and limitations

This code demonstrates how an artificial system could implement the functional sequence. It is not a biological simulation of neurons, synapses, oscillations, or neurotransmitters. The thresholds, coefficients, feature maps, and similarity functions are placeholders that must be calibrated against behavioral data. A more advanced implementation should use validated language parsing, explicit dependency graphs, empirically estimated retrieval functions, and separate models for spoken and written input.

---

# 6. Failure modes and their distinct computational signatures

| Failure mode | What fails | Observable consequence | Recovery direction |
|---|---|---|---|
| Weak encoding | A word or phrase is not represented reliably | The item is not recognized later | Re-expose input; improve segmentation and attention |
| Maintenance loss | Needed information becomes temporarily inaccessible | Earlier subject, referent, or clause opening is unavailable | Reactivate, rehearse, or re-read |
| Similarity interference | Several candidates match the retrieval cue | Wrong antecedent or noun is retrieved | Use more diagnostic cues; preserve alternatives |
| Binding failure | Elements are retained but their relation is not | Correct words, incorrect roles or order | Rebuild structural links |
| Integration failure | Retrieved information is not combined with current input | Locally understood phrases fail to form a coherent sentence | Reanalyse clause structure and event roles |
| Goal overload | Too many competing task demands | Detail retention falls while the main task continues | Reduce concurrent demand; prioritize task-critical relations |
| Premature compression | Distinct details are merged into one chunk | Important contrasts or negation are lost | Split the chunk and restore internal distinctions |
| Schema-driven completion | Missing information is filled by expectation | A fluent but unsupported interpretation | Separate observed evidence from inference |

A crucial distinction is that reduced accessibility does not necessarily mean erasure. Information may be present in long-term memory or in the preceding text but unavailable under the current retrieval conditions.

---

# 7. Mathematical operations catalogue

The following operations are plausible tools for formalizing the functional algorithm. They are not claims that the brain performs literal symbolic arithmetic.

| Operation | Formula or procedure | Computational purpose | Biological caution |
|---|---|---|---|
| Feature encoding | \(\mathbf{x}=f(u,C,G)\) | Represent task-relevant word and phrase features | Neural codes are distributed and context-sensitive |
| Probabilistic segmentation | \(P(z\mid x,C)\) | Select likely word/phrase boundaries | Perceptual inference is not known to use this exact formula |
| Activation/accessibility | \(A_i=B_i+R_i-I_i-D_i\) | Estimate whether a representation can be retrieved | Decay and interference are competing/combined theoretical accounts |
| Similarity | Cosine or feature overlap | Estimate competition between candidates | Neural similarity depends on population activity and task |
| Retrieval competition | Softmax over candidate matches | Rank possible antecedents or dependencies | A softmax is an engineering choice, not an identified neural equation |
| Load estimation | Weighted sum of dependencies, complexity, interference, and support | Estimate current demand | There is no universal biological load scalar |
| Chunking | Hierarchical grouping of related elements | Reduce separately managed units | Chunk quality depends on knowledge and task |
| Attention allocation | Constrained optimization or normalized priorities | Balance maintenance and new processing | No evidence for a literal optimizer operating over explicit variables |
| Confidence gating | \(E_{t+1}=E_t+\eta\Delta E\) | Prevent weak retrieval from becoming a firm interpretation | Integration involves multiple neural mechanisms |
| Overload detection | Logistic model of load and error signals | Detect probable processing difficulty | Behavioral markers are indirect and non-specific |
| Bayesian reconstruction | \(P(h\mid e,C)\propto P(e\mid h,C)P(h\mid C)\) | Compare candidate interpretations | A normative model; not proof of exact neural Bayesian computation |
| Coherence checking | Weighted constraint score | Detect contradictions and missing links | Coherence is multidimensional and context-sensitive |

---

# 8. Validation and experimental predictions

A computational implementation should not be accepted merely because it can produce a plausible interpretation. It should reproduce measurable patterns and distinguish competing explanations.

## Experiment 1 — Sentence length versus dependency complexity

Compare sentences that differ in length while controlling lexical familiarity and syntactic structure. Also compare sentences with similar length but different numbers and distances of unresolved dependencies.

**Prediction:** Length alone should not explain comprehension difficulty. Dependency structure, interference, predictability, and chunking should explain additional variance.

**Measurements:** Reading time by region, listening time, comprehension accuracy, recall of referents, and regressions to earlier text.

## Experiment 2 — Similarity-based retrieval interference

Manipulate the similarity of competing nouns or referents while keeping sentence structure relatively stable.

**Prediction:** More similar competitors should increase retrieval difficulty and the likelihood of role or antecedent errors, especially when the retrieval cue is not distinctive.

**Measurements:** Comprehension accuracy, retrieval latency, pronoun-resolution choices, and memory for thematic roles.

## Experiment 3 — Concurrent memory load

Have participants maintain an additional set of words or symbols while processing sentences of different structural complexity.

**Prediction:** A secondary load should disproportionately affect structures that require maintaining and retrieving earlier material, though the magnitude will depend on similarity between the secondary material and sentence representations.

**Measurements:** Dual-task accuracy, sentence comprehension, response latency, and interference-sensitive errors.

## Experiment 4 — Chunking and expertise

Compare familiar constructions or domain-relevant material with unfamiliar material, and manipulate whether the sequence can be grouped into meaningful chunks.

**Prediction:** Familiarity and knowledge-based chunking should reduce the effective burden of maintaining separate elements, but should not eliminate interference when several candidates remain similar.

**Measurements:** Recall, reading time, dependency resolution, and accuracy under secondary load.

## Experiment 5 — Recovery after an induced lapse

Present a difficult sentence, then allow a rereading or paraphrase cue before asking comprehension questions.

**Prediction:** Some errors should be recoverable with a better cue, consistent with temporary inaccessibility or retrieval interference. Other errors should persist when crucial information was never encoded or when the structural relation was misanalysed.

**Measurements:** Initial and post-cue accuracy, confidence calibration, and ability to identify the exact source of uncertainty.

## Experiment 6 — Separate item memory from relation memory

Test whether a participant remembers the individual nouns but confuses who performed the action or which clause a noun belonged to.

**Prediction:** Item recognition can remain intact while relational binding is impaired. A model that stores only word identities will not capture this dissociation.

**Measurements:** Word recognition, thematic-role questions, order reconstruction, and event-model accuracy.

## Experiment 7 — Neural network recruitment under increasing demand

Use fMRI, EEG/MEG, or another suitable technique while manipulating complexity, interference, and secondary-task load.

**Prediction:** Increasing demand should alter activity and coordination across language and domain-general control/memory systems. The precise regions and lateralization should vary with task design, participant characteristics, and the type of load.

**Caution:** Regional activation alone cannot prove that a region performs a single specific memory operation. Functional connectivity and carefully controlled contrasts are needed, and even these do not directly reveal an algorithm executed by the brain.

---

# 9. Engineering interpretation for a brain-inspired AI architecture

A useful artificial implementation should avoid treating working memory as a flat list with a fixed token limit. Instead, it should maintain a structured, inspectable state containing:

1. active lexical and conceptual representations;
2. open dependencies and their expected retrieval cues;
3. a compact event/situation model;
4. candidate referents and similarity-based competition;
5. confidence and provenance for every important relation;
6. a dynamic estimate of processing demand;
7. explicit unresolved or uncertain information;
8. recovery operations that can re-query the original input or retrieve earlier representations.

### Suggested modular architecture

- **Input Segmentation Module:** detects word, phrase, and clause units.
- **Lexical-Feature Module:** retrieves lexical, grammatical, and semantic features.
- **Incremental Parser:** maintains partial structures and unresolved dependencies.
- **Working-State Manager:** tracks active representations, accessibility, and goal relevance.
- **Interference Controller:** ranks competing retrieval candidates and manages similarity conflicts.
- **Event-Model Builder:** integrates entities, actions, roles, time, and discourse links.
- **Load Monitor:** estimates demand and identifies possible overload.
- **Recovery Controller:** selects reactivation, re-parsing, chunk restructuring, or clarification.
- **Confidence and Provenance Logger:** records what was stated, retrieved, inferred, or left unresolved.

### Critical design principle

When information cannot be retrieved confidently, the architecture should not silently fabricate a link. It should preserve an unresolved dependency, seek a better cue, revisit the original input, or request clarification. This is the computational equivalent of acknowledging that parts of a long message may have been lost during interpretation.

---

# 10. Scientific interpretation and limits

The source's statement that working-memory overload can cause parts of a long message to be forgotten is broadly consistent with research showing that sentence comprehension depends on maintaining and retrieving partial products of interpretation. However, several qualifications are essential:

- Comprehension difficulty cannot be explained by a simple word-count threshold.
- Working-memory performance reflects interacting storage, processing, attention-control, and retrieval mechanisms.
- Similarity-based interference can be as important as, or more explanatory than, simple time-based decay in some sentence-processing conditions.
- Meaningful structure and prior knowledge can reduce demand by allowing chunking and prediction.
- The system can lose access to a relation while retaining the individual words.
- An incorrect interpretation may result from retrieval competition or premature inference, not only from literal forgetting.
- Brain activation findings support a distributed, demand-sensitive network; they do not establish one fixed neural location or one universally accepted working-memory architecture.

The mathematical models in this file are proposed computational abstractions for constructing a testable brain-inspired system. They should be treated as hypotheses to compare against human behavioral and neural data, not as direct measurements of the brain's internal arithmetic.

---

# 11. Selected scientific references

1. **Lewis, R. L., Vasishth, S., & Van Dyke, J. A. (2006).** Computational principles of working memory in sentence comprehension. *Trends in Cognitive Sciences, 10*(10), 447–454. https://doi.org/10.1016/j.tics.2006.08.007. PMID: 16949330. This review describes the role of a limited attentional focus, retrieval interference, and time-dependent accessibility in sentence comprehension.

2. **Van Dyke, J. A., & Johns, C. L. (2012).** Memory interference as a determinant of language comprehension. *Language and Linguistics Compass.* https://doi.org/10.1002/lnc3.330. This review discusses interference-based accounts of comprehension difficulty and contrasts them with simple capacity/decay explanations.

3. **Gordon, P. C., Hendrick, R., & Levine, W. H. (2002).** Memory-load interference in syntactic processing. *Psychological Science, 13*(5). https://doi.org/10.1111/1467-9280.00475. The study found that an additional memory load affected comprehension of syntactically complex sentences, with greater effects when memory items were similar to sentence items.

4. **Newman, S. D., Malaia, E., Seo, R., & Cheng, H. (2013).** The effect of individual differences in working memory capacity on sentence comprehension: An fMRI study. *Brain and Language.* https://doi.org/10.1007/s10548-012-0264-8. The study links working-memory differences to changing recruitment and connectivity during sentence reading and retrieval.

5. **The interaction between language and working memory: A systematic review of fMRI studies in the past two decades (2021).** *Frontiers in Psychology.* https://pmc.ncbi.nlm.nih.gov/articles/PMC7815476/. This review summarizes how language complexity and working-memory demands engage overlapping and task-dependent neural systems.

6. **Neural mechanisms of interference control in working memory capacity (2018).** *Brain and Behavior.* https://pmc.ncbi.nlm.nih.gov/articles/PMC6866570/. This study discusses active maintenance and interference regulation, including contributions from prefrontal and parietal networks.

7. **Differential Effects of Working Memory Constraints on Syntactic and Semantic Processing: Evidence from ERPs (2025).** https://pmc.ncbi.nlm.nih.gov/articles/PMC11993935/. This recent study investigates how working-memory constraints interact with syntactic and semantic processing demands.

8. **The nature of working memory capacity in sentence comprehension: Evidence against domain-specific working memory resources (2006).** Fedorenko, E., Gibson, E., & Rohde, D. *Journal of Memory and Language, 54*, 541–553. https://web.mit.edu/evlab/assets/papers/Fedorenko_et_al_2006_JML.pdf. This work examines shared versus domain-specific resource accounts using dual-task manipulations.

---

## Completion status

**Point 19 is complete.** This file contains the detailed algorithmic sequence for working-memory overload and information loss during complex interpretation. The next source point should be handled only after the user explicitly requests continuation.
