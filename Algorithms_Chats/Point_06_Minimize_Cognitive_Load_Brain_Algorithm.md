# Point 06 — Minimize Cognitive Load
## A detailed computational and biologically informed algorithm for managing information during comprehension and learning

**Source point:** *Improving Comprehension and Learning.txt* (Library)

> “Break information into chunks, focus on one idea at a time, and avoid multitasking. When learning complex material (like a dense paper), actively outline its structure or draw diagrams to offload working memory.”

---

## 1. Scope and interpretation

This document converts the source point into a sequential, reusable algorithm. It describes how a learner can reduce unnecessary demands on limited, task-dependent working-memory and attentional resources while preserving the information needed to understand a complex subject.

The algorithm contains four connected goals:

1. **Reduce avoidable processing demands.** Remove irrelevant search, confusing presentation, unnecessary switching, and poorly organized material.
2. **Organize meaningful information.** Segment material into coherent units, connect related elements, and build chunks that can be handled as integrated structures.
3. **Control the flow of information.** Attend to a manageable subgoal, keep only currently useful intermediate information active, and retrieve relevant knowledge when needed.
4. **Use external representations.** Put relationships, intermediate results, and the overall structure into outlines, notes, or diagrams so they do not all have to be maintained mentally at once.

The aim is not to make learning effortless or to remove all difficulty. Some difficulty is inherent in the subject and may be necessary for learning. The aim is to prevent *unnecessary* difficulty from consuming resources that should be spent understanding, comparing, inferring, and remembering.

### 1.1 Three levels of scientific interpretation

Throughout this document, statements are marked conceptually as one of three kinds:

- **Established behavioral finding or broad evidence:** A pattern supported by experimental work, such as performance costs in many dual-task situations, the benefits of coherent organization for difficult material, and the relevance of prior knowledge to task complexity.
- **Computational model:** A mathematical or algorithmic abstraction that can help implement the behavior in software. The brain is not known to calculate the equations literally.
- **Possible biological implementation:** A cautious account of brain systems and neural processes that may contribute. It is not a one-to-one mapping between an algorithmic step and a single brain region.

A numerical “load score,” a chunk boundary, a threshold, or a scheduling decision in this document is therefore a model variable. It should not be interpreted as a direct reading of a biological meter inside the brain.

---

## 2. Overall algorithm

### 2.1 Input

- A learning or comprehension goal.
- Learning material: text, speech, diagrams, examples, or a combination.
- The learner's relevant prior knowledge and vocabulary.
- The current task context, including time, distractions, and available tools.
- Optional evidence of difficulty: pauses, errors, rereading, loss of the goal, subjective effort, or failure to explain the material.

### 2.2 Internal state to maintain in the computational model

For each current learning episode, maintain:

- `goal`: what the learner is trying to understand or produce.
- `units`: candidate information elements extracted from the material.
- `relations`: prerequisite, causal, temporal, contrastive, and part-whole links among units.
- `chunks`: groups of units that can be treated as meaningful structures.
- `active_subgoal`: the single principal question or operation currently being handled.
- `working_set`: the intermediate facts, relations, and partial results currently needed.
- `external_state`: what has been written, drawn, labeled, or otherwise represented outside the mind.
- `load_estimate`: a qualitative or quantitative estimate of current demand.
- `interference_estimate`: the degree to which competing tasks or irrelevant cues are disrupting the target task.
- `understanding_estimate`: evidence that the learner can reconstruct and use the material.
- `support_level`: how much scaffolding, segmentation, or external structure is still required.

### 2.3 Output

A manageable sequence of learning steps in which:

- essential information is retained;
- relationships among concepts remain visible;
- avoidable processing is reduced;
- intermediate results are preserved externally when useful;
- the learner can reconstruct the complete explanation from its organized parts;
- support is reduced gradually as the learner becomes more fluent.

---

# 3. Sequential algorithmic steps

## Step 1 — Define the comprehension goal before processing the material

### Purpose

Prevent the learner's attention from being spread across every available detail. A clear goal creates a selection criterion: information relevant to the goal receives priority, while unrelated information can be deferred.

### Detailed procedure

1. Convert the broad request “understand this material” into one or more explicit outcomes. Examples include identifying the central claim, explaining a mechanism, comparing two theories, solving a class of problems, or reconstructing the sequence of a process.
2. Determine the expected depth: recognition, explanation, application, comparison, or independent derivation.
3. Identify what evidence would demonstrate success. For example, the learner may need to explain the process without looking, draw its causal structure, or apply it to a new example.
4. Separate the current goal from questions that are interesting but not necessary for the current pass.
5. Keep the goal visible in a short phrase or question.

### Possible mathematical operations

Represent the learning goal as a vector of target features:

\[
\mathbf{g}=[g_1,g_2,\ldots,g_m]
\]

For each candidate information unit \(u_i\), estimate its relevance:

\[
R_i=\operatorname{sim}(\mathbf{u}_i,\mathbf{g})\times I_i
\]

where `sim` is a semantic or task-relevance measure and \(I_i\) is the importance of the unit to the intended outcome. The score is a computational prioritization device, not a literal neuronal calculation.

A threshold can divide material into:

\[
\text{priority}(u_i)=
\begin{cases}
\text{process now}, & R_i\geq\tau_g\\
\text{defer or omit}, & R_i<\tau_g
\end{cases}
\]

The threshold \(\tau_g\) should be adjusted to the task. A literature review may require broad coverage, whereas a focused procedural task may require only a small subset.

### Possible biological implementation

Goal maintenance and selective attention are associated with interactions among prefrontal and parietal control systems and sensory or association regions. These systems can bias processing toward goal-relevant input and help resist distraction. The exact implementation is distributed and context-dependent; it is not simply a single “attention center” filtering all information.

### Failure detection and recovery

If the learner repeatedly follows unrelated details, restate the goal and identify the next answerable question. Do not automatically suppress an unexpected detail: mark it as a possible important exception and return to it after the main structure is understood.

---

## Step 2 — Inspect the material and estimate its initial complexity

### Purpose

Determine whether the main challenge comes from the subject itself, the way the material is presented, or the learner's unfamiliarity with its terminology and structure.

### Detailed procedure

1. Inspect headings, paragraph boundaries, diagrams, examples, equations, and the order in which concepts are introduced.
2. Identify unfamiliar terms and identify which terms are defined by the material itself.
3. Detect information that is repeated, peripheral, visually separated from the text it explains, or presented in an order that forces repeated searching.
4. Identify dependencies: which concepts must be understood before others can be interpreted?
5. Estimate how many elements must be considered together to understand each major explanation.
6. Use the learner's prior knowledge to estimate whether several details can already be treated as one familiar unit.

### Mathematical model: element interactivity

Let the material contain elements \(e_1,\ldots,e_n\). A simple model of task complexity is:

\[
C_{\mathrm{task}}=\sum_{i=1}^{n} a_i+
\lambda\sum_{i<j}b_{ij}
\]

where:

- \(a_i\) represents the processing demand of element \(i\);
- \(b_{ij}\) represents the strength or importance of the interaction between elements \(i\) and \(j\);
- \(\lambda\) controls how strongly interacting elements contribute to the estimate.

This makes an important distinction: ten independent labels may be easier to process than five concepts whose meanings depend on one another. The number of words or bullet points alone is not a reliable measure of complexity.

An expertise-adjusted estimate can be represented as:

\[
C_{\mathrm{effective}}=C_{\mathrm{task}}\,(1-\eta K)
\]

where \(K\) is a normalized estimate of relevant prior knowledge and \(\eta\) controls how strongly that knowledge reduces the demand. In an implementation, constrain the product so that the estimate never becomes negative. This is a simplified model of the fact that knowledge and schemas can make multiple elements easier to process together.

### Possible biological implementation

Prior knowledge distributed across long-term memory can change how incoming information is interpreted and grouped. Familiar structures can be activated as patterns, reducing the need to treat every component as a separate novel item. The hippocampal and cortical memory systems contribute to retrieving and relating knowledge, but the formula above is an engineering representation rather than a known neural equation.

### Failure detection and recovery

If complexity is high, do not immediately assume the learner has poor capacity. Check for missing prerequisites, unclear definitions, fragmented presentation, or many simultaneous dependencies. Select the correction that addresses the actual source of difficulty.

---

## Step 3 — Identify prerequisite knowledge and retrieve only what is needed

### Purpose

Reduce the burden of interpreting each new term from scratch. At the same time, avoid activating a large set of loosely related memories that distract from the current goal.

### Detailed procedure

1. For every central new concept, list its prerequisites.
2. Check whether each prerequisite is already understood well enough for the current task.
3. Retrieve the most relevant definition, rule, or example—not every memory associated with the topic.
4. If a prerequisite is missing, pause the main sequence and teach or review that prerequisite in a compact unit.
5. Link the prerequisite to the new concept with an explicit relation such as “depends on,” “causes,” “is an example of,” or “contrasts with.”
6. Return to the original learning goal after the prerequisite gap has been repaired.

### Mathematical operations

A prerequisite structure can be modeled as a directed graph:

\[
G=(V,E)
\]

where nodes are concepts and a directed edge \(u_i\to u_j\) means that understanding \(u_i\) supports understanding \(u_j\).

For a target concept \(v\), retrieve its prerequisite closure:

\[
P(v)=\{u\in V: u\leadsto v\}
\]

In practice, retrieve the smallest subset of prerequisites sufficient for the current goal, rather than activating the entire graph.

Candidate memories can be ranked using:

\[
S(m_i)=w_r\,\operatorname{Rel}(m_i,g)+w_p\,\operatorname{Prereq}(m_i,v)
-w_d\,\operatorname{Distraction}(m_i)
\]

A ranking model can prioritize relevant memories and penalize tangential associations.

### Possible biological implementation

Cue-dependent retrieval activates distributed representations associated with a concept. Hippocampal pattern completion may help reinstate related episode or relational information, while cortical systems represent semantic knowledge. Executive control may help select which retrieved information is relevant. These are broad functional descriptions, not a claim that the brain explicitly traverses a symbolic graph.

---

## Step 4 — Segment the material into coherent information units

### Purpose

Replace an unstructured stream of words or symbols with manageable units. Segmentation should follow meaning and relations, not arbitrary equal-length cuts.

### Detailed procedure

1. Identify natural boundaries: a definition, one causal transition, one stage of a process, one argument, one example, or one operation in a procedure.
2. Give each unit a short label that preserves its central meaning.
3. Keep tightly related details together when they must be interpreted jointly.
4. Separate details that belong to different logical stages or answer different questions.
5. Preserve the relationships that connect one unit to the next.
6. Check whether the unit can be paraphrased in one or two sentences without losing its essential relation.
7. If it cannot, determine whether it contains several sub-operations that should be separated or whether the elements are inherently interdependent and must remain together.

### Mathematical operations

Let a source contain ordered elements \(X=(x_1,\ldots,x_n)\). Segmentation partitions it into sets:

\[
\mathcal{P}=\{B_1,B_2,\ldots,B_k\},\quad
\bigcup_jB_j=X,\quad B_i\cap B_j=\varnothing
\]

One possible computational objective is to maximize within-chunk coherence while retaining important cross-chunk relations:

\[
J(\mathcal{P})=
\sum_j\operatorname{Coherence}(B_j)
-\alpha\sum_j\operatorname{Overload}(B_j)
-\beta\operatorname{LostRelations}(\mathcal{P})
\]

Here \(\alpha\) and \(\beta\) balance manageable unit size against the risk of fragmenting meaningful relations.

A candidate boundary can be placed where there is a strong semantic or structural transition:

\[
\operatorname{Boundary}(i)=f(\Delta\text{topic},\Delta\text{goal},\Delta\text{relation},\text{discourse cues})
\]

No single numeric chunk size is universally optimal. Familiarity, language, task, and element interactivity all matter.

### Possible biological implementation

Perceptual and language systems detect boundaries and transitions; attention selects relevant units; long-term knowledge helps group details into familiar patterns. A meaningful chunk is not necessarily a fixed-size packet stored in one brain location. Its usefulness depends on the structure represented and on what the learner already knows.

### Failure detection and recovery

- **Over-segmentation:** If each word or tiny detail is isolated, restore the relations that make the pieces meaningful.
- **Under-segmentation:** If a whole chapter or complex mechanism is treated as one unit, split it by functional or causal stage.
- **False chunking:** If a group is only visually adjacent but does not form a meaningful unit, separate it.

---

## Step 5 — Convert useful groups into meaningful chunks or schemas

### Purpose

Allow multiple details to be treated as one familiar structure. This can reduce the number of independently managed items, but only after the learner has understood the internal relations.

### Detailed procedure

1. Find recurring patterns among the segmented units.
2. Identify the rule or organizing relationship that makes those units belong together.
3. Create a concise label for the pattern.
4. Store important internal roles or relations, not just a memorized label.
5. Attach a representative example and, when helpful, a counterexample.
6. Test whether the learner can reconstruct the components from the chunk label.
7. Revise the chunk if it merges concepts that behave differently.

### Mathematical operations

A chunk can be represented as a structured object:

\[
C_j=(L_j,\{r_1,\ldots,r_q\},\mathcal{R}_j)
\]

where \(L_j\) is the chunk label, \(r_i\) are its roles or elements, and \(\mathcal{R}_j\) specifies relations among them.

The effective number of units can be approximated by:

\[
N_{\mathrm{eff}}=N_{\mathrm{unbound}}+N_{\mathrm{chunks}}
\]

This approximation is meaningful only if each chunk is understood and retrievable. A learner who memorizes an unexplained acronym has not necessarily reduced the underlying conceptual demand.

Chunk validity can be evaluated with reconstruction accuracy:

\[
Q_{\mathrm{chunk}}=\frac{\text{essential relations correctly reconstructed}}
{\text{total essential relations}}
\]

A high-quality chunk retains the information required for the task while making the structure easier to handle.

### Possible biological implementation

Expertise-dependent chunking is often explained by the interaction of working memory with long-term knowledge. Familiar patterns can be retrieved as integrated structures, and repeated practice may make their use more efficient. The neural representation is distributed and depends on the knowledge domain; no universal neural “chunk register” is established.

---

## Step 6 — Order learning from prerequisites to dependent operations

### Purpose

Avoid requiring the learner to process several unfamiliar elements and their relationships before the basic components are understood.

### Detailed procedure

1. Use the prerequisite graph to identify concepts with no unmet dependencies.
2. Begin with these foundational units.
3. Explain each unit with a clear example.
4. Add one or a small number of dependent elements at a time.
5. Revisit the relations connecting the new material to previously learned units.
6. Combine units into a larger process only after their individual roles are understood.
7. Preserve exceptions and conditions; do not simplify a rule so far that it becomes false.

### Mathematical operations

For a dependency graph, a topological order \(v_1,\ldots,v_n\) satisfies:

\[
(v_i\to v_j)\in E\implies i<j
\]

This provides a valid teaching sequence when the graph is acyclic. If concepts depend on one another cyclically, present the smallest useful cycle as an integrated system, then revisit the components.

An incremental complexity schedule may be written:

\[
C_{t+1}=C_t+\delta_t
\]

where \(\delta_t\) is the amount of new interacting information introduced at step \(t\). Reduce \(\delta_t\) when errors or overload indicators rise; increase it when the learner reliably integrates the current material.

### Possible biological implementation

Prior knowledge supports the processing of new information, while attentional control and working memory support the temporary maintenance of current relations. Gradual learning may allow cortical representations and schemas to become more organized over repeated experiences. The equation is a teaching-policy model, not a measured neural growth law.

---

## Step 7 — Select one principal subgoal for the current processing cycle

### Purpose

Reduce competition among unrelated goals and preserve a stable task set while the learner completes one meaningful operation.

### Detailed procedure

1. Convert the next learning unit into one question or operation.
2. State what information is required to complete it.
3. Temporarily defer independent tasks such as checking unrelated messages or exploring tangential links.
4. Complete the subgoal or reach a clear stopping point.
5. Record any unresolved question externally rather than repeatedly switching to it.
6. Move to the next subgoal after updating the outline or working notes.

### Mathematical operations

A goal-selection model can use:

\[
P(g_i\mid s)=\frac{\exp(\beta U(g_i,s))}
{\sum_j\exp(\beta U(g_j,s))}
\]

where \(U\) is the estimated usefulness of goal \(g_i\) in the current state \(s\), and \(\beta\) controls selectivity. This is a computational way of expressing prioritization; it does not establish that the brain computes a softmax over explicit goals.

Switching between tasks can be represented as adding a cost:

\[
T_{\mathrm{total}}=\sum_iT_i+\sum_{i=2}^{n}S_{i-1,i}
\]

where \(S_{i-1,i}\) is the time or performance cost associated with changing task set. Actual costs depend on task similarity, preparation, cues, and practice.

### Possible biological implementation

Frontoparietal control systems and other task-control circuits contribute to maintaining goals, selecting relevant actions, and managing interference. Dual-task performance costs are widely observed, although their size and source vary. Multitasking should not be described as absolutely impossible: some activities can be combined when they are highly practiced or draw on separable resources.

### Failure detection and recovery

If the learner switches tasks repeatedly, capture the unrelated thought in a “later” list, return to the active subgoal, and complete a small meaningful unit. If the tasks are genuinely complementary—for example, reading a diagram while listening to the explanation—integrate them rather than imposing an artificial separation.

---

## Step 8 — Reduce avoidable presentation and search demands

### Purpose

Prevent the learner from wasting working-memory resources on finding, matching, or repeatedly reinterpreting information that could be presented more coherently.

### Detailed procedure

1. Place definitions near the concepts that use them.
2. Label diagrams and connect labels clearly to the relevant visual elements.
3. Avoid requiring repeated movement between distant sections when information must be compared.
4. Remove decorative or redundant details that compete with the learning goal.
5. Use headings, consistent terminology, and explicit transitions.
6. Highlight only the information that deserves attention; excessive highlighting becomes another source of noise.
7. Keep examples and their explanations spatially or temporally close when the learner must compare them.

### Mathematical operations

Let the total demand be divided for modeling purposes into:

\[
L_{\mathrm{total}}=L_{\mathrm{essential}}+L_{\mathrm{avoidable}}
\]

This is a practical decomposition, not a universally accepted neural partition. Redesign aims to reduce \(L_{\mathrm{avoidable}}\) without removing information needed for understanding.

A presentation-cost estimate might be:

\[
L_{\mathrm{avoidable}}=\alpha S+\beta N+\gamma A+\delta R
\]

where:

- \(S\): split-attention or search demands;
- \(N\): irrelevant novelty or visual noise;
- \(A\): ambiguity in labels or wording;
- \(R\): redundant processing that does not improve the learner's mental model.

Coefficients depend on the learner and material. The model helps compare designs; it is not a physiological measurement.

### Possible biological implementation

When the learner must retain one item while searching for its matching definition elsewhere, attention and working memory must maintain and coordinate both. Better-integrated presentation can reduce this coordination demand. Visual attention, language processing, and executive control interact during this work.

---

## Step 9 — Externalize intermediate information with notes, outlines, and diagrams

### Purpose

Keep the overall structure and intermediate results available without requiring the learner to maintain every detail internally at once.

### Detailed procedure

1. Choose the external representation suited to the relationship: an outline for hierarchy, a flowchart for sequence, a causal diagram for mechanisms, a table for comparison, or a concept map for a network of relations.
2. Write the central question at the top.
3. Add only the concepts and relations necessary for the current explanation.
4. Label arrows with meaningful relations such as “causes,” “inhibits,” “requires,” or “is an example of.”
5. Record intermediate calculations or assumptions so they do not have to be mentally recomputed.
6. Keep the representation readable; an overcrowded diagram can become a new source of load.
7. Use the representation to resume after a pause or interruption.
8. Periodically attempt to reconstruct the structure without looking, then use the external representation to check omissions.

### Mathematical operations

Represent the knowledge structure as a graph:

\[
G=(V,E,W)
\]

where \(V\) contains concepts, \(E\) contains relationships, and \(W\) optionally weights the importance or confidence of each relationship.

For an outline, represent hierarchical membership as a tree. For a process, represent transitions as directed edges. For a comparison, represent shared and distinguishing attributes as a matrix:

\[
M_{ij}=\text{value of feature }j\text{ for concept }i
\]

The external-state update can be modeled as:

\[
E_{t+1}=E_t\cup\Delta E_t
\]

where \(\Delta E_t\) contains newly established concepts, links, examples, or unresolved questions.

One can estimate how much internal maintenance is avoided by writing down a relation, but there is no simple universal conversion from a number of written symbols to a fixed quantity of biological working-memory capacity. External tools can also impose costs: the learner must read, navigate, interpret, and coordinate them.

### Possible biological implementation

The brain still needs to perceive and interpret the written or drawn structure. External representation does not literally transfer a memory out of the brain. It changes the task by making information available in the environment, reducing the need to internally maintain or reconstruct some intermediate states. This is an example of interaction between internal cognition and external resources.

### Failure detection and recovery

If the diagram becomes too dense, split it into linked subdiagrams. If the outline lists facts but contains no explanatory relationships, add verbs or labeled links. If the learner copies the diagram without understanding it, use retrieval and explanation to test whether the structure has been internalized.

---

## Step 10 — Maintain only the working set required for the current subgoal

### Purpose

Reduce interference and unnecessary maintenance by keeping the current operation's relevant facts active while leaving completed or future information in long-term memory or external notes.

### Detailed procedure

1. Identify the minimal set of facts needed for the current subgoal.
2. Bring those facts into active consideration using recall, perception, or the external outline.
3. Mark intermediate results that will be needed later.
4. Remove completed, irrelevant, or superseded items from the active working set, while ensuring they remain retrievable or externally recorded.
5. Refresh information when it is still needed but has become uncertain.
6. Avoid discarding a premise merely because it is not immediately active; retain its location in the outline or note structure.

### Mathematical operations

Represent the active working set at time \(t\) as:

\[
W_t=\{w_1,w_2,\ldots,w_k\}
\]

A relevance score can govern selection:

\[
A_i(t)=\alpha R_i+\beta D_i+\gamma U_i-\delta I_i
\]

where \(R_i\) is goal relevance, \(D_i\) is dependency on the current operation, \(U_i\) is uncertainty or need for review, and \(I_i\) is interference or distraction risk.

An item remains in the working set when its score and current task role justify maintenance. This is a control model, not a literal neural ranking mechanism.

A simple state-decay approximation is:

\[
x_i(t+\Delta t)=x_i(t)e^{-\Delta t/\tau_i}
\]

where \(\tau_i\) is a model parameter for persistence. Real working-memory maintenance is more complex than passive exponential decay; attention, refreshing, interference, binding, and task demands all matter.

### Possible biological implementation

Working memory is supported by distributed, dynamic interactions among cortical and subcortical systems. Depending on the task, representations may be sustained through active neural activity, changes in synaptic states, or combinations of mechanisms. Frontoparietal systems contribute to selection and control, while specialized sensory or association areas may represent task content. There is no single universally accepted physical storage mechanism for every form of working memory.

---

## Step 11 — Monitor signs of overload and distinguish them from productive effort

### Purpose

Detect when the current organization or pace is preventing effective comprehension, without treating every difficult moment as overload.

### Observable indicators

Possible signs that the current demand exceeds the learner's available resources include:

- repeatedly rereading the same sentence without gaining meaning;
- losing track of the original question;
- forgetting a premise before the next inference is completed;
- making errors in combining details that were understood separately;
- confusing terms or relations that were just introduced;
- switching repeatedly between sections to recover missing context;
- being unable to summarize the last completed unit;
- a subjective sense of excessive effort accompanied by declining accuracy.

No single sign is conclusive. Fatigue, low motivation, poor prior knowledge, unclear writing, sensory difficulty, and anxiety can produce similar behavior.

### Detailed procedure

1. Collect multiple indicators rather than relying on one feeling.
2. Compare current accuracy and comprehension with the learner's recent baseline.
3. Determine whether the difficulty is caused by too many interacting elements, missing prerequisites, distraction, unclear presentation, or fatigue.
4. Choose a targeted intervention: reduce the number of active elements, clarify a term, retrieve a prerequisite, redraw the structure, or take a break.
5. Reassess after the intervention.

### Mathematical operations

A monitoring system can calculate a composite indicator:

\[
O_t=w_1E_t+w_2R_t+w_3S_t+w_4F_t
\]

where:

- \(E_t\): current error rate;
- \(R_t\): rereading or repeated-search rate;
- \(S_t\): task-switching or goal-loss rate;
- \(F_t\): subjective fatigue or effort rating.

Normalize each variable before combining them. The weights \(w_i\) should be calibrated to the learner and task.

An overload response can be triggered when:

\[
O_t>\theta_O
\]

but the threshold should be interpreted as a heuristic. It does not diagnose a brain state.

### Possible biological implementation

Monitoring and control involve interactions among prefrontal, parietal, cingulate, and other systems that support performance monitoring, attention allocation, and behavioral adjustment. Subjective effort and performance errors are indirect indicators; they do not provide a direct neural measurement of cognitive load.

---

## Step 12 — Adapt the next learning step to the current load estimate

### Purpose

Regulate the material dynamically rather than using a fixed chunk size or fixed pace for every learner and every topic.

### Detailed procedure

1. If the learner can explain the current chunk accurately and integrate it with prior material, increase complexity modestly.
2. If accuracy is adequate but effort is high, maintain the current complexity and improve clarity or external support.
3. If the learner repeatedly loses relations, reduce the number of simultaneously interacting elements.
4. If a prerequisite is missing, teach it before continuing.
5. If the presentation is causing search or navigation costs, redesign the presentation rather than simplifying the actual concept.
6. If performance is declining because of fatigue, pause rather than splitting the material indefinitely.
7. Reassess after every meaningful adjustment.

### Mathematical operations

Let \(L_t\) be estimated demand and \(C_t\) the learner's current available resource estimate. Define utilization:

\[
\rho_t=\frac{L_t}{C_t+\epsilon}
\]

where \(\epsilon\) prevents division by zero in a computational implementation. The value is a relative index, not a direct biological capacity measurement.

A simple policy is:

\[
\text{action}_t=
\begin{cases}
\text{increase complexity slightly}, & \rho_t<\theta_1\text{ and comprehension is stable}\\
\text{maintain or clarify}, & \theta_1\leq\rho_t\leq\theta_2\\
\text{reduce or scaffold}, & \rho_t>\theta_2\text{ or comprehension declines}
\end{cases}
\]

The thresholds depend on task difficulty, learner experience, and measurement quality. Do not equate a high effort rating by itself with failure; effort can be productive when comprehension remains sound.

### Possible biological implementation

Adaptive behavior depends on monitoring performance, updating goals, reallocating attention, and drawing on memory. The algorithmic policy is an external model of that regulation rather than a literal neural controller with explicit thresholds.

---

## Step 13 — Use worked examples and guided steps when the learner lacks a usable schema

### Purpose

Prevent a novice from having to discover all relevant procedures and relationships at once. A worked example can reduce unproductive search and show how elements fit together.

### Detailed procedure

1. Present a complete, correct example that matches the learning goal.
2. Label the role of each step and explain why it is necessary.
3. Point out which information is essential and which is incidental to that example.
4. Ask the learner to predict the next step before revealing it when appropriate.
5. Move from a complete example to a partially completed example.
6. Gradually ask the learner to complete more steps independently.
7. Transfer to a new example to test whether the underlying schema—not merely the surface wording—was learned.

### Mathematical operations

Let the number of steps supplied by the instructional scaffold be \(s_t\). A fading policy can reduce support as performance improves:

\[
s_{t+1}=s_t-\Delta s
\]

provided that accuracy remains above a chosen criterion. If accuracy falls, restore the smallest useful amount of support.

An instructional policy can balance independent effort and unnecessary search:

\[
J=\alpha\,\text{LearningGain}-\beta\,\text{UnproductiveSearch}-\gamma\,\text{SupportDependence}
\]

This objective is a design tool. It does not imply that all unguided discovery is harmful or that worked examples are always best. The best support depends on expertise and task structure.

### Possible biological implementation

Examples provide structured sensory and conceptual input that can be related to existing knowledge. Repeated successful processing may support the formation and refinement of schemas in long-term memory. As schemas become more accessible, a learner can often process previously separate elements as a meaningful whole.

---

## Step 14 — Integrate the current chunk into the larger structure

### Purpose

Prevent chunking from producing isolated fragments. The learner must understand how each local unit contributes to the main explanation.

### Detailed procedure

1. Finish the current chunk and state its central point.
2. Identify what question it answers in the larger subject.
3. Connect it to the preceding and following chunks.
4. State whether the relationship is causal, temporal, hierarchical, evidential, or contrastive.
5. Update the outline or diagram.
6. Reconstruct the sequence so far, using external notes only when needed.
7. Correct missing or contradictory links before adding further complexity.

### Mathematical operations

Let chunk representations be \(C_1,\ldots,C_k\). Define a relation matrix:

\[
R_{ij}=\begin{cases}
1 & \text{if a meaningful relation from }C_i\text{ to }C_j\text{ is established}\\
0 & \text{otherwise}
\end{cases}
\]

In richer models, entries may be weighted by relation type and confidence. Structural completeness can be estimated as:

\[
Q_{\mathrm{structure}}=
\frac{\sum_{i,j}w_{ij}\,\mathbf{1}(\text{required link }i,j\text{ is correct})}
{\sum_{i,j}w_{ij}\,\mathbf{1}(\text{link }i,j\text{ is required})}
\]

This measures whether the learner has preserved required relations, not merely remembered a list of chunk labels.

### Possible biological implementation

Comprehension requires integration across successive items and across semantic or event relationships. Recurrent interactions among language, memory, and control systems can support this integration. The relation matrix is a computational representation, not a literal cortical table.

---

## Step 15 — Consolidate, retrieve, and gradually reduce the external support

### Purpose

Ensure that external outlines and chunk labels support learning rather than becoming the only route by which the learner can perform the task.

### Detailed procedure

1. After a meaningful section, pause and explain its structure from memory.
2. Compare the explanation with the outline or diagram.
3. Identify omissions and restore only the necessary links.
4. Revisit difficult relations after a delay rather than rereading everything immediately.
5. Use the external representation less often as independent reconstruction becomes reliable.
6. Preserve the external artifact as a reference for later complex tasks, even when it is no longer needed at every step.
7. Review the full structure after several chunks have been integrated.

### Mathematical operations

Retrieval accuracy can be represented as:

\[
A_t=\frac{\text{correctly reconstructed target units and relations}}
{\text{total target units and relations}}
\]

A scaffold-fading policy may set:

\[
H_{t+1}=H_t(1-\mu A_t)
\]

where \(H_t\) is support level and \(\mu\) is a small adaptation parameter. In practice, use bounds so support does not become negative and do not fade support based on one successful attempt.

### Possible biological implementation

Retrieval, repeated use, and rest or sleep can contribute to memory stabilization and integration. The relationship among active study, consolidation, and later accessibility is complex; no simple equation captures all biological changes.

---

## Step 16 — Verify comprehension at both local and global levels

### Purpose

Check that reducing cognitive load has not removed important content or created an oversimplified mental model.

### Detailed procedure

1. Ask the learner to explain each chunk in their own words.
2. Ask what would happen if one important condition changed.
3. Ask the learner to identify the relation between two non-adjacent chunks.
4. Ask for a new example or application.
5. Compare the explanation with the original source for missing conditions, exceptions, or causal links.
6. If understanding fails, determine whether the problem is local (one chunk), relational (links among chunks), or global (overall structure).
7. Repair only the failing level, then repeat the test.

### Mathematical operations

A comprehension score can combine distinct dimensions:

\[
Q=\alpha A_{\mathrm{local}}+\beta A_{\mathrm{relations}}
+\gamma A_{\mathrm{transfer}}+\delta A_{\mathrm{explanation}}
\]

where the weights sum to one. Do not let fluent verbal repetition substitute for correct relational or transfer performance.

A simple error signal is:

\[
\mathbf{e}=\mathbf{y}_{\mathrm{target}}-\mathbf{y}_{\mathrm{response}}
\]

The error indicates which concepts or relationships need repair. It is a computational analogy to discrepancy-driven correction, not a claim that the brain represents all comprehension errors as a single vector.

### Possible biological implementation

Self-monitoring, retrieval, and comparison with feedback recruit distributed memory and control mechanisms. A learner's confidence can be informative but is not perfectly calibrated; objective explanation and transfer tests provide additional evidence.

---

# 4. Integrated control loop

The steps above can be implemented as a repeated loop rather than a one-way pipeline:

1. Define or refresh the goal.
2. Select the next prerequisite-ready unit.
3. Estimate its complexity relative to the learner's knowledge.
4. Segment and chunk the information while preserving meaningful relations.
5. Select one principal subgoal.
6. Present the information coherently and externalize intermediate structure when helpful.
7. Maintain only the working set required for the subgoal.
8. Monitor performance and signs of excessive load.
9. Adapt chunk size, pace, scaffolding, or presentation.
10. Integrate the chunk into the global structure.
11. Test local and global understanding.
12. Continue, repair, or pause based on the result.

### Integrated pseudocode

```text
FUNCTION MINIMIZE_COGNITIVE_LOAD(material, learning_goal, learner_profile):
    state = INITIALIZE_STATE(material, learning_goal, learner_profile)
    source_map = INSPECT_STRUCTURE(material)
    prerequisites = BUILD_PREREQUISITE_GRAPH(source_map)
    outline = CREATE_INITIAL_OUTLINE(source_map, learning_goal)

    WHILE NOT GOAL_SATISFIED(state):
        unit = SELECT_NEXT_PREREQUISITE_READY_UNIT(prerequisites, state)
        prior_knowledge = RETRIEVE_RELEVANT_PREREQUISITES(unit, state)

        complexity = ESTIMATE_ELEMENT_INTERACTIVITY(
            unit,
            prior_knowledge,
            learner_profile
        )

        segments = SEGMENT_BY_MEANING_AND_RELATION(unit)
        chunks = FORM_COHERENT_CHUNKS(segments, prior_knowledge)
        active_subgoal = SELECT_ONE_PRINCIPAL_SUBGOAL(chunks, learning_goal)

        presentation = REDUCE_AVOIDABLE_SEARCH_AND_SPLIT_ATTENTION(
            active_subgoal
        )
        external_map = UPDATE_OUTLINE_OR_DIAGRAM(
            outline,
            active_subgoal,
            presentation
        )

        working_set = SELECT_MINIMAL_RELEVANT_WORKING_SET(
            active_subgoal,
            prior_knowledge,
            external_map
        )

        PROCESS_AND_EXPLAIN(working_set)
        indicators = MONITOR_ERRORS_EFFORT_SWITCHING_AND_GOAL_LOSS()
        load = ESTIMATE_CURRENT_DEMAND(complexity, indicators)

        IF MISSING_PREREQUISITE(indicators):
            TEACH_PREREQUISITE_AND_RETURN()
        ELSE IF PRESENTATION_CAUSES_AVOIDABLE_LOAD(indicators):
            REDESIGN_PRESENTATION()
        ELSE IF OVERLOAD_LIKELY(load, indicators):
            REDUCE_ACTIVE_ELEMENTS()
            INCREASE_SCAFFOLDING_OR_EXTERNAL_SUPPORT()
        ELSE IF COMPREHENSION_STABLE(indicators):
            INCREASE_COMPLEXITY_SLIGHTLY()

        INTEGRATE_CHUNK_WITH_GLOBAL_STRUCTURE(outline, active_subgoal)
        result = TEST_LOCAL_GLOBAL_AND_TRANSFER_UNDERSTANDING()

        IF result is inadequate:
            REPAIR_FAILED_UNIT_OR_RELATION(result)
        ELSE:
            UPDATE_LEARNER_PROFILE_AND_FADE_SUPPORT_GRADUALLY()

    RETURN outline, integrated_knowledge_state, comprehension_evidence
```

---

# 5. Worked example — Understanding a dense explanation of cellular respiration

This example demonstrates the logic of the algorithm. It is not intended to be a complete biology lesson.

### Initial problem

A learner reads a dense explanation that includes glucose, glycolysis, pyruvate, mitochondria, the citric acid cycle, electron carriers, the electron transport chain, a proton gradient, ATP synthase, and oxygen. The learner tries to hold all terms in mind simultaneously and loses the relationship among the stages.

### Application of the algorithm

**1. Define the goal.** The immediate goal is to explain how energy from glucose is transferred into ATP, not to memorize every enzyme in one pass.

**2. Inspect the material.** Identify the main stages and the dependencies among them. Notice which details are central to the energy-transfer sequence and which are specialized details for a later pass.

**3. Retrieve prerequisites.** Confirm basic understanding of glucose, oxidation, ATP, mitochondria, and electron carriers. Review a missing prerequisite before proceeding.

**4. Segment the material.** Divide it into broad stages: glycolysis, pyruvate processing and citric acid cycle, electron transport, and chemiosmosis/ATP synthesis.

**5. Create meaningful chunks.** Give each stage a concise role label, such as “initial glucose breakdown,” “electron-carrier loading,” “gradient generation,” and “ATP production.” Preserve the underlying details and conditions within each stage.

**6. Order by dependency.** Explain the inputs and outputs of each stage before presenting the next stage's dependencies.

**7. Select one subgoal.** First answer: “What does glycolysis produce?” Avoid switching to the detailed mechanism of ATP synthase before this is stable.

**8. Reduce presentation demand.** Put the explanation of each stage next to its diagram and label each arrow with what moves or changes.

**9. Externalize.** Draw a four-stage flowchart. Record the main input and output below each stage rather than trying to remember every intermediate while reading the next paragraph.

**10. Maintain the working set.** While learning electron transport, keep the needed information about electron carriers and membrane location active. Keep completed details in the chart.

**11. Monitor overload.** If the learner remembers the names but cannot explain why the proton gradient matters, the issue is a missing causal relation, not simply too many vocabulary items.

**12. Adapt.** Pause and explain the proton gradient with a simpler membrane example; then return to the biological sequence.

**13. Use a worked example.** Trace one simplified electron-transfer sequence and show how it contributes to proton movement. Gradually remove the labels and ask the learner to reconstruct the sequence.

**14. Integrate.** Connect the electron transport chain to the proton gradient and the gradient to ATP synthase. This makes the stages a causal chain rather than four unrelated facts.

**15. Retrieve.** Close the material and reconstruct the four-stage explanation from memory. Use the diagram only to check missing links.

**16. Verify transfer.** Ask what would be affected if oxygen were unavailable at the end of the electron transport chain. The learner must reason from the structure, not merely repeat the stage labels.

### What the example demonstrates

The learner's difficulty was not solved by deleting important biology. It was addressed by clarifying the goal, retrieving prerequisites, reducing simultaneous demands, preserving intermediate information externally, and making causal links explicit. The final explanation still contains complex biology, but it is approached through manageable and integrated stages.

---

# 6. Mathematical model summary

The equations below are useful engineering abstractions for a computational implementation. They should not be interpreted as a complete quantitative theory of the human brain.

| Operation | Example model | Computational purpose |
|---|---|---|
| Goal relevance | \(R_i=\operatorname{sim}(u_i,g)I_i\) | Prioritize information relevant to the current learning outcome |
| Element interactivity | \(C=\sum_i a_i+\lambda\sum_{i<j}b_{ij}\) | Estimate the number and strength of interacting elements |
| Prior-knowledge adjustment | \(C_{eff}=C(1-\eta K)\) | Approximate how expertise changes effective task demand |
| Prerequisite graph | \(G=(V,E)\) | Represent which concepts depend on other concepts |
| Segmentation | Partition \(X\) into coherent blocks | Organize material into meaningful units |
| Chunk quality | Correctly reconstructed relations / required relations | Ensure that chunk labels preserve internal structure |
| Goal selection | Softmax over goal utility | Model prioritization among competing goals |
| Switching cost | \(T=\sum_iT_i+\sum_iS_i\) | Represent overhead from changing task sets |
| External representation | \(G=(V,E,W)\) or outline tree | Preserve concepts and their relations outside the active working set |
| Load monitoring | Weighted composite of errors, rereading, switching, effort | Detect when the current approach needs adjustment |
| Load utilization | \(\rho=L/(C+\epsilon)\) | Compare estimated task demand with learner-specific resources |
| Scaffold fading | Decrease support as demonstrated performance improves | Transfer more processing to the learner as schemas develop |
| Comprehension verification | Weighted local, relational, and transfer scores | Ensure simplification has not destroyed understanding |

### 6.1 Important mathematical cautions

1. **No fixed universal capacity constant.** Do not hard-code a universal number of items that every person can hold in working memory. The effective amount depends on item complexity, chunking, prior knowledge, task, and interference.
2. **Do not equate word count with load.** A short but relationally dense equation may be harder than a long list of familiar names.
3. **Do not treat load as directly observable.** Errors, pauses, self-reports, and task performance are indirect indicators.
4. **Do not assume all mental effort is harmful.** Effort can accompany productive inference and learning. The relevant question is whether the effort supports the goal and whether comprehension is progressing.
5. **Do not treat cognitive-load categories as settled neural compartments.** The distinction between essential task complexity and avoidable presentation demands is useful, but theoretical definitions and measurement approaches continue to be debated.
6. **Do not interpret the model's equations as neural code.** They are explicit engineering representations of hypothesized functions.

---

# 7. Possible biological systems involved

The algorithm is distributed across interacting functions. The labels below identify broad contributions, not isolated brain modules that each execute one software-like step.

## 7.1 Prefrontal and frontoparietal control systems

These systems contribute to goal maintenance, selection among competing task demands, regulation of attention, and control of behavior. They are relevant when the learner maintains a question, resists irrelevant information, or changes strategy after detecting difficulty.

## 7.2 Parietal and domain-related cortical representations

Parietal and other cortical systems participate in attention, numerical and spatial processing, integration, and task-relevant representation. Which regions contribute depends strongly on the kind of material and task.

## 7.3 Language and semantic networks

Language-related systems process words, sentence structure, and meaning. Broader semantic knowledge is distributed across cortical systems. Familiar terms and established semantic relationships can make new explanations easier to interpret.

## 7.4 Hippocampal and medial temporal memory systems

These systems contribute to relational and episodic memory, retrieval of relevant experiences, and the formation of new memories. They can help link the current material to prior information. It would be too simple to identify the hippocampus as a general-purpose working-memory container.

## 7.5 Sensory and association systems

Visual, auditory, and multimodal systems process the words, diagrams, sounds, and other representations used during learning. A diagram can make a spatial relation visible, but it still requires perception, attention, and interpretation.

## 7.6 Distributed neural dynamics

Working memory and cognitive control are not adequately described as a single fixed-size box in one location. Different tasks depend on distributed networks and dynamic interactions. Candidate mechanisms include sustained activity, transient activity, activity-silent synaptic states, and interactions between attention and long-term memory. The relative contributions remain an active research topic.

---

# 8. Evaluation criteria for a software implementation

A computational system inspired by this algorithm should be tested against observable performance, not merely whether it produces shorter chunks or smaller load scores.

### 8.1 Unit-level criteria

- Each chunk expresses one coherent idea or operation.
- Important internal relations are retained.
- The learner can paraphrase the chunk accurately.
- The chunk is neither so large that it becomes unmanageable nor so small that it loses meaning.

### 8.2 Structure-level criteria

- Prerequisite order is respected.
- The outline represents the true conceptual hierarchy.
- Causal and contrastive relationships are explicit.
- The learner can reconstruct the overall sequence.

### 8.3 Regulation criteria

- The system detects repeated loss of the active goal.
- It distinguishes missing prerequisites from presentation problems.
- It adjusts support in response to performance.
- It does not automatically simplify intrinsically complex but necessary content.
- It allows productive effort when understanding is progressing.

### 8.4 Transfer criteria

- The learner can explain the material without simply reading the notes.
- The learner can apply the underlying structure to a new example.
- The learner can identify which parts of a familiar schema do not apply to an unusual case.
- Performance remains robust after a delay, not only immediately after exposure.

### 8.5 Suggested experimental comparisons

Compare at least these conditions:

1. Unstructured presentation versus conceptually segmented presentation.
2. Split-attention materials versus integrated text-and-diagram materials.
3. Continuous task switching versus focused subgoal blocks.
4. No external outline versus a learner-generated outline.
5. Fixed chunk sizes versus expertise- and performance-adaptive chunking.
6. Permanent scaffolding versus gradual scaffold fading.

Measure accuracy, explanation quality, completion time, error type, delayed retention, and transfer. Self-reported effort can supplement these measures but should not be the only outcome.

---

# 9. Limitations and safeguards

- **Chunking is not always beneficial.** If elements are tightly interdependent, separating them can make the explanation harder. Keep related elements together when their interaction is the concept being learned.
- **One-idea-at-a-time is a practical default, not an absolute law.** Real comprehension often requires integrating language, diagrams, and context in parallel. The aim is to avoid competing unrelated tasks, not to prevent useful integration.
- **Multitasking costs vary.** Some practiced or compatible activities can be combined. The algorithm should avoid categorical claims that the brain can never process two tasks at once.
- **External notes can become clutter.** An overly detailed outline or diagram may add navigation and interpretation costs. Externalization must be designed for the learner's current question.
- **Simplification can destroy accuracy.** Preserve essential exceptions, boundary conditions, and uncertainty.
- **Load estimates are uncertain.** Use multiple indicators and adapt conservatively. A performance problem may arise from fatigue, motivation, sensory issues, anxiety, or lack of knowledge rather than overload alone.
- **Cognitive-load theory is not a complete neural account.** It is a framework for instructional design and information processing, not a detailed map of every brain operation.
- **The biological implementation is distributed.** Avoid mapping a software function to one brain region without evidence.

---

# 10. Key conceptual conclusion

Minimizing cognitive load is best understood as **organizing the work of comprehension**, not simply reducing the quantity of information. The learner needs to preserve all essential concepts while reducing unnecessary search, irrelevant switching, unhelpful fragmentation, and avoidable demands on active maintenance.

The central cycle is:

\[
\text{Goal}\rightarrow\text{Select}\rightarrow\text{Segment}\rightarrow\text{Relate}\rightarrow\text{Externalize}\rightarrow\text{Process}\rightarrow\text{Monitor}\rightarrow\text{Adapt}\rightarrow\text{Integrate}\rightarrow\text{Verify}
\]

A successful system keeps the learner's attention on the current meaningful operation, stores intermediate structure in a usable form, and repeatedly checks that local simplification still produces global understanding.

---

# 11. Selected scientific references

The following sources support the conceptual discussion. They do not validate every equation in this document; the equations are explicitly proposed computational abstractions.

1. **Chen, O., Paas, F., & Sweller, J. (2023).** A Cognitive Load Theory Approach to Defining and Measuring Task Complexity Through Element Interactivity. *Educational Psychology Review*, 35, Article 63. https://doi.org/10.1007/s10648-023-09782-w
   - Reviews task complexity and element interactivity, including how prior expertise changes effective task demands.

2. **Hitch, G. J., Allen, R. J., & Baddeley, A. D. (2024).** The multicomponent model of working memory fifty years on. *Quarterly Journal of Experimental Psychology*. https://doi.org/10.1177/17470218241290909
   - Reviews the development of the multicomponent working-memory model and issues concerning executive control.

3. **Langerock, N., Oberauer, K., Throm, E., & Vergauwe, E. (2025).** The cognitive load effect in working memory: Refreshing the empirical landscape, removing outdated explanations. *Journal of Memory and Language*, 140, Article 104558. https://doi.org/10.1016/j.jml.2024.104558
   - Shows that cognitive-load effects in working-memory tasks are task-dependent and cautions against overly broad explanations.

4. **Garner, K. G., & Dux, P. E. (2023).** Knowledge generalization and the costs of multitasking. *Nature Reviews Neuroscience*, 24, 98–112. https://doi.org/10.1038/s41583-022-00653-x
   - Reviews multitasking costs and possible links between flexible knowledge generalization and cognitive control.

5. **Musslick, S., & Cohen, J. D. (2021).** Rationalizing constraints on the capacity for cognitive control. *Trends in Cognitive Sciences*, 25(9), 757–775. https://doi.org/10.1016/j.tics.2021.06.001
   - Reviews theories of limits on simultaneous control-dependent tasks and trade-offs in cognitive control.

6. **Seufert, T., Hamm, V., Vogt, A., et al. (2024).** The Interplay of Cognitive Load, Learners’ Resources and Self-regulation. *Educational Psychology Review*, 36, Article 50. https://doi.org/10.1007/s10648-024-09890-1
   - Examines how task demand, learner resources, and self-regulation interact.

7. **Wang, T., & Lajoie, S. P. (2023).** How Does Cognitive Load Interact with Self-Regulated Learning? A Dynamic and Integrative Model. *Educational Psychology Review*, 35, Article 69. https://doi.org/10.1007/s10648-023-09794-6
   - Discusses the dynamic relationship between cognitive load and self-regulation.

---

**End of Point 06 — Minimize Cognitive Load.**

The next source point is **Point 07 — Seek Feedback**. Do not begin it until the user explicitly requests continuation.
