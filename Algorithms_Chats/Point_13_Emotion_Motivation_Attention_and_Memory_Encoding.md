# Point 13 — Emotion and Motivation: Modulating Attention, Learning, and Memory

## Scope and source position

This algorithm expands the following point from **Learning, Memory, and Concept Formation.txt**:

> Emotion and motivation modulate learning and attention: positive interest or stress hormones (norepinephrine) can boost the encoding of related material, whereas anxiety may narrow focus. Emotionally charged words or arousing situations can produce stronger memory through interactions involving the amygdala and hippocampus. Motivation can encourage deeper processing and sustain effort when material is difficult.

The target is the brain's **modulation of what receives attention, how strongly an experience is encoded, how it is consolidated, and whether cognitive effort is sustained**. It is not a claim that emotion creates a separate memory system or that every emotional experience is remembered accurately.

## 1. Biological and computational framing

Emotion and motivation are related but distinct control influences.

- **Emotion / affective appraisal** estimates the significance of an event for safety, threat, opportunity, attachment, or other concerns. It can change arousal, attention, action readiness, and memory priority.
- **Motivation** changes the value assigned to possible outcomes and the willingness to allocate effort toward them. It helps determine which goal is pursued and how long a difficult task is maintained.
- **Attention** selects and amplifies some information relative to competing information. Selection is limited and depends on both bottom-up salience and top-down goals.
- **Learning and memory** alter representations through encoding, synaptic plasticity, and consolidation. Arousal and reward can modulate these operations without guaranteeing that the stored representation is complete or unbiased.

Several neural systems interact. The amygdala, especially basolateral amygdala (BLA) circuits, helps evaluate biologically significant events and can modulate memory consolidation in other structures. The hippocampus binds event elements and their relations. The locus coeruleus (LC) supplies widespread norepinephrine (NE), influencing arousal, vigilance, and neural gain. The hypothalamic–pituitary–adrenal (HPA) axis and glucocorticoids participate in stress responses. Dopaminergic circuits, including midbrain–striatal and midbrain–prefrontal pathways, contribute to reward learning, value, and effort allocation. Prefrontal and cingulo-opercular systems help maintain goals, regulate attention, and choose whether to persist or switch tasks. These are interacting networks, not isolated modules with one-to-one functions.

**Important biological caution:** norepinephrine is not simply a “stress hormone.” It is a neuromodulator involved in arousal and attention, with effects depending on receptor type, concentration, timing, brain region, and current task. Cortisol is a glucocorticoid stress hormone. They interact, but they are not interchangeable.

The mathematical operations below are **computational descriptions** useful for building an explicit model. Neurons do not need to calculate the written equations symbolically. Neural populations can approximate the described operations through firing rates, synaptic weights, oscillatory coordination, neuromodulation, and recurrent circuit dynamics.

---

# 2. The complete algorithm

## Step 1 — Receive an experience and construct an initial event representation

**Purpose:** Establish the information that could be attended to, evaluated emotionally, and learned.

1. Receive incoming sensory and internal signals: visual patterns, sounds, language, bodily sensations, remembered information, and current task state.
2. Extract elementary features in modality-specific pathways. For a spoken sentence, these may include acoustic features, phonetic structure, words, prosody, and speaker cues. For a physical event, they may include objects, motion, spatial layout, and bodily state.
3. Bind features that appear to belong to the same event. Hippocampal and cortical systems contribute to representing relationships among the event's elements; the exact division of labor depends on task and timescale.
4. Preserve a provisional representation rather than committing immediately to a final interpretation. Later appraisal and attention can alter which features are emphasized.

**Possible computational operations**

- Feature extraction: \(\mathbf{x}_t = [x_1, x_2, \ldots, x_n]\), where each component represents a candidate feature or feature activation.
- Temporal integration: \(\bar{\mathbf{x}}_t = \sum_{k=0}^{K-1} w_k\mathbf{x}_{t-k}\), where recent samples receive weights \(w_k\). This is a generic model of integration over time, not a literal hippocampal formula.
- Event binding: represent relations between features as a graph \(G_t=(V_t,E_t)\), where nodes are features or entities and edges encode relations such as “near,” “caused,” or “belongs to.”

**Biological interpretation:** Early sensory processing and language-processing pathways provide candidate content. Attention, appraisal, and memory systems then influence which parts of this content receive further processing.

## Step 2 — Estimate affective significance

**Purpose:** Evaluate whether the event matters to the organism's current needs, goals, or learned concerns.

1. Compare the event with learned associations, current goals, expected outcomes, and the surrounding context.
2. Estimate whether it signals threat, reward, novelty, loss, social relevance, safety, or uncertainty.
3. Distinguish the event's **valence** (positive, negative, or mixed) from its **arousal** (how activating or urgent it is). These dimensions are related but not identical.
4. Update the appraisal as additional context arrives. A loud sound may first trigger orienting, then be reclassified as harmless when its source is identified.

**Possible computational operations**

- Feature-based appraisal: \(S_t = f(\mathbf{x}_t,\mathbf{c}_t,\mathbf{g}_t,\mathbf{m}_t)\), where \(\mathbf{c}_t\) is context, \(\mathbf{g}_t\) current goals, and \(\mathbf{m}_t\) relevant memories.
- A simple decomposed estimate: \(S_t = aT_t+bR_t+cN_t+dU_t\), with threat \(T\), reward relevance \(R\), novelty \(N\), uncertainty \(U\), and weights that vary by person and context.
- Valence and arousal can be represented separately: \(v_t\in[-1,1]\), \(a_t\in[0,1]\). A negative event need not have high arousal, and a highly arousing event need not be negative.

**Biological interpretation:** Amygdala and broader salience-related circuits contribute to evaluating biologically important events. Appraisal is not performed by the amygdala alone; cortical, hippocampal, hypothalamic, brainstem, and bodily signals contribute.

## Step 3 — Generate and regulate the arousal response

**Purpose:** Adjust alertness and readiness in response to the event's significance.

1. If the event warrants a response, recruit autonomic and neuromodulatory systems.
2. LC activity can change NE release across widespread cortical and subcortical targets, helping adjust vigilance and responsiveness.
3. Depending on the event and duration, the HPA axis can produce a slower hormonal response, including cortisol release.
4. Match the response to current demands: a brief orienting response differs from prolonged vigilance or sustained stress.
5. Reappraise or down-regulate arousal when the event is understood and no longer requires an emergency response.

**Possible computational operations**

- State update: \(A_{t+1}=\rho A_t+(1-\rho)\,q(S_t)\), where \(A_t\) is a latent arousal state and \(0<\rho<1\) models persistence.
- Multi-timescale response: \(A_t=A_t^{fast}+A_t^{slow}\), representing rapid neuromodulatory/autonomic changes and slower endocrine influences.
- Saturating response: \(q(S)=S/(K+S)\), as one possible model when response growth slows at high input. Actual physiological dose-response relationships vary by receptor and tissue.

**Biological interpretation:** NE, acetylcholine, autonomic output, and glucocorticoids operate on different timescales and have different effects. The brain does not have one single arousal dial.

## Step 4 — Compute attentional priority

**Purpose:** Select the information that should receive enhanced processing now.

1. Estimate bottom-up salience: contrast, novelty, abrupt change, emotional intensity, or biological significance.
2. Estimate top-down relevance: whether a feature helps answer a question, reach a goal, avoid a danger, or obtain a reward.
3. Combine these signals to prioritize competing items.
4. Allocate processing to the highest-priority information, while retaining some capacity for monitoring other relevant cues.
5. Recalculate priority when goals or environmental conditions change.

**Possible computational operations**

- Priority score: \(P_i=\alpha S_i+\beta G_i+\gamma C_i\), where \(S_i\) is salience, \(G_i\) goal relevance, and \(C_i\) contextual relevance.
- Normalized competition: \(p_i=\exp(P_i/\tau)/\sum_j\exp(P_j/\tau)\). The temperature \(\tau\) controls how sharply priority is concentrated.
- Limited resource allocation: choose attentional weights \(a_i\) under \(\sum_i a_i\leq B\), where \(B\) is a finite processing budget.

**Biological interpretation:** Attention is shaped by competition among neural representations and by top-down control. Arousal can increase the gain of selected signals, but it can also reduce flexible, broad monitoring when it becomes excessive or threat-focused.

## Step 5 — Apply neuromodulatory gain to selected representations

**Purpose:** Change the responsiveness of neural populations so that selected information has a stronger or weaker influence on ongoing processing.

1. Use current arousal and task state to modulate the gain of sensory and association circuits.
2. Increase responsiveness to cues that predict a relevant outcome or require action.
3. Allow goal-directed attention to counteract irrelevant emotional capture when control resources are available.
4. Avoid assuming that all NE release uniformly increases all neural activity; its effects depend on receptor distribution, dose, and circuit state.

**Possible computational operations**

- Gain modulation: \(y_i=g_i(A_t)\,x_i\), where \(x_i\) is incoming activation and \(g_i\) is state-dependent gain.
- A conceptual inverted-U relation for some cognitive functions: \(g(A)=g_{max}\exp[-(A-A^*)^2/(2\sigma^2)]\). Here \(A^*\) is an illustrative optimal state. This is not a universal law for all brain functions; strong arousal can improve some threat-related processing while impairing working memory or flexible control.
- Divisive normalization: \(y_i=x_i/(\epsilon+\sum_j w_jx_j)\), a useful model of population competition and gain control.

**Biological interpretation:** LC–NE and basal-forebrain acetylcholine systems both influence arousal and attention, with distinct and overlapping effects. Cortical inhibitory networks also regulate population gain.

## Step 6 — Encode emotional and goal-relevant features more strongly

**Purpose:** Give important features a higher chance of influencing memory formation.

1. Keep the event's core sensory, semantic, spatial, and temporal features available to encoding systems.
2. Use attentional selection to emphasize the parts most relevant to the current emotional or motivational significance.
3. Allow the amygdala, especially BLA circuits, to modulate memory-related systems when arousal is significant.
4. The hippocampus binds the selected event elements and relations into an episode representation; cortical systems encode modality-specific and semantic information.
5. Preserve the possibility that central and peripheral details are encoded unequally. A threatening central object may be remembered well while background details are not.

**Possible computational operations**

- Salience-weighted feature vector: \(\mathbf{x}'=\mathbf{m}\odot\mathbf{x}\), where \(\mathbf{m}\) contains attentional or salience weights and \(\odot\) denotes elementwise multiplication.
- Associative binding: \(\mathbf{h}_{new}=F(\mathbf{x}',\mathbf{r},\mathbf{c})\), where \(\mathbf{r}\) represents relations and \(\mathbf{c}\) context.
- A schematic encoding-strength model: \(E_i=E_{0i}+\lambda_1a_i+\lambda_2s_i+\lambda_3g_i\), with attention \(a_i\), salience \(s_i\), and goal relevance \(g_i\). This is an explanatory abstraction, not a measured biological formula.

**Biological interpretation:** Emotional arousal can enhance memory through BLA interactions with hippocampal and other memory-related structures. Enhancement is selective and can coexist with distortions, omissions, or narrowed attention.

## Step 7 — Integrate stress and arousal effects without assuming “more is always better”

**Purpose:** Predict how the current physiological state changes different cognitive operations.

1. Estimate whether arousal is low, moderate, or high relative to the task's demands.
2. For moderate arousal, allow alertness and prioritization to improve when the task benefits from them.
3. Under intense or prolonged stress, check for reduced working-memory capacity, attentional flexibility, or prefrontal control.
4. Recognize that emotional memory may still be strong for central salient features even when broader context or peripheral details are weak.
5. Separate effects on encoding, consolidation, and retrieval; the same stressor may affect these phases differently.

**Possible computational operations**

- State-dependent capacity: \(B(A)=B_0\,f(A)\), where \(B\) is the available control budget and \(f(A)\) may be task-specific and non-monotonic.
- Detail allocation: \(D_i\propto a_i\,r_i\), where \(r_i\) denotes relevance; high salience may concentrate detail encoding on selected items.
- Phase-specific model: \(M_{final}=F(E_{encode},C_{consolidate},R_{retrieve})\), rather than treating memory strength as a single state.

**Biological interpretation:** Stress effects depend on intensity, duration, predictability, controllability, timing, and individual differences. High stress does not simply erase memory; it can enhance some components and impair others.

## Step 8 — Evaluate reward value and motivational relevance

**Purpose:** Determine whether learning or continuing the task is worth the effort.

1. Estimate expected outcomes: reward, progress, information gain, social approval, safety, or avoidance of loss.
2. Estimate costs: time, physical effort, cognitive effort, uncertainty, and opportunity cost.
3. Compare expected value with the cost of acting or continuing.
4. Use current goals and prior reward history to prioritize which material deserves sustained engagement.
5. Update this estimate when the reward is delayed, reduced, unexpectedly increased, or no longer available.

**Possible computational operations**

- Expected value: \(Q(a)=\sum_o P(o\mid a)V(o)\), where action \(a\) can lead to outcomes \(o\).
- Net utility: \(U(a)=Q(a)-\kappa C(a)\), where \(C(a)\) is effort/cost and \(\kappa\) expresses cost sensitivity.
- Choice probability: \(P(a)=\exp(U(a)/\tau)/\sum_b\exp(U(b)/\tau)\).

**Biological interpretation:** Dopaminergic circuits contribute to reward prediction, incentive motivation, and learning about the value of actions. Dopamine is not simply a “pleasure chemical”; its functions include teaching and motivating signals, and other systems also contribute to reward experience.

## Step 9 — Compute prediction errors and update reward expectations

**Purpose:** Learn whether an outcome was better or worse than expected.

1. Maintain an estimate of expected reward or progress for the current state and action.
2. Observe the actual outcome.
3. Compare the outcome with the prediction.
4. Increase or decrease the expected value according to the mismatch.
5. Use the updated value when deciding whether to repeat, change, or stop the behavior.

**Possible computational operations**

- Temporal-difference reward prediction error: \(\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)\).
- Value update: \(V_{t+1}(s_t)=V_t(s_t)+\alpha\delta_t\), where \(\alpha\) is a learning rate.
- For multiple cues, update associative strengths: \(w_i\leftarrow w_i+\alpha\delta_t x_i\).

**Biological interpretation:** Phasic dopamine signals in some midbrain pathways are often modeled as reward-prediction-error signals, although dopamine activity is heterogeneous across regions, cell populations, tasks, and timescales. The equations are a formal learning model, not a claim that every dopamine neuron encodes the same scalar error.

## Step 10 — Use motivation to sustain cognitive control and effort

**Purpose:** Maintain engagement when understanding requires repeated, demanding operations.

1. Maintain an explicit or implicit task goal.
2. Monitor progress and the remaining difficulty.
3. Estimate whether additional cognitive effort is likely to produce useful progress.
4. Recruit working-memory maintenance, inhibition of distraction, task switching, and error monitoring as required.
5. Persist when expected progress is worth the effort; reduce effort, change strategy, seek help, or disengage when costs exceed expected benefit.
6. Reassess rather than treating persistence as unconditionally beneficial.

**Possible computational operations**

- Effort allocation: \(e^*=\arg\max_e[\mathbb{E}(R\mid e)-\kappa C(e)]\).
- Goal maintenance: \(G_{t+1}=\rho_GG_t+\eta I_{goal}-\zeta D_t\), where distraction \(D_t\) competes with goal activation.
- Persistence rule: continue when \(\mathbb{E}[\Delta Progress\mid e]>\kappa\Delta Cost\), subject to constraints such as fatigue and available resources.

**Biological interpretation:** Prefrontal and cingulate systems participate in goal maintenance, monitoring, and control allocation. Dopamine may influence both the value of effortful actions and the functional state of circuits used for working memory and control. The everyday term “grit” describes a behavioral pattern, not one isolated brain circuit.

## Step 11 — Promote deeper, elaborative processing when the learner is engaged

**Purpose:** Connect new material to existing knowledge instead of processing it only superficially.

1. Relate new information to prior concepts, examples, causal models, and personal goals.
2. Generate explanations, comparisons, questions, or predictions.
3. Organize details into meaningful groups and relations.
4. Test the resulting interpretation by retrieval, paraphrase, or application.
5. Correct contradictions and fill gaps with evidence rather than merely repeating an initial guess.

**Possible computational operations**

- Associative expansion: \(A_{new}=A_{old}+\sum_i w_i\,sim(x,new_i)\), where similarities link a new item to existing representations.
- Elaboration score: \(L=\sum_i q_i\,r_i\), where \(q_i\) is the number/quality of meaningful links and \(r_i\) their relevance.
- Prediction testing: \(\epsilon_t=y_t-\hat y_t\), followed by model revision when prediction error is systematic.

**Biological interpretation:** Elaborative encoding recruits interactions among hippocampal memory systems, semantic knowledge networks, and prefrontal control. Motivation can make sustained elaboration more likely, but motivation does not guarantee that elaboration is accurate or well organized.

## Step 12 — Coordinate emotional modulation with hippocampal encoding and consolidation

**Purpose:** Translate significance at learning time into a possible advantage in later retention.

1. During encoding, allow arousal and attention to influence the quality and composition of the initial trace.
2. After the event, allow amygdala-related modulation to influence consolidation processes in the hippocampus and other memory systems.
3. Consider the timing of neuromodulatory signals: a signal arriving during or shortly after learning can have different effects from one arriving much later.
4. Preserve the distinction between initial encoding strength and later consolidation; a vivid initial response is not itself proof of a durable memory.
5. Integrate later sleep, replay, retrieval, and repeated learning with the earlier emotional/motivational modulation.

**Possible computational operations**

- Salience-modulated consolidation rate: \(w_{t+1}=w_t+\eta_0 m_t\,\Delta w_t\), where \(m_t\) is a state-dependent modulation factor.
- Time-limited modulation kernel: \(m(t)=m_0e^{-t/\tau}\), as a simplified representation of a decaying modulatory window.
- Retention estimate: \(R(t)=R_0e^{-kt}\), where \(k\) depends on rehearsal, interference, sleep, and consolidation. This simple decay curve is only a baseline approximation.

**Biological interpretation:** The BLA's influence on memory consolidation is supported by extensive research. This does not mean the amygdala stores the entire emotional memory or that it independently determines what will be recalled.

## Step 13 — Retrieve the memory and assess how emotion affects recollection

**Purpose:** Use learned information while accounting for emotional influences on retrieval.

1. Receive a retrieval cue or a new situation that overlaps with the stored event.
2. Reconstruct relevant information from hippocampal and cortical representations.
3. Allow current mood, goals, and context to influence which memories become accessible.
4. Distinguish subjective vividness or confidence from objective accuracy.
5. Compare retrieved content with external evidence when precision matters.

**Possible computational operations**

- Cue-to-memory similarity: \(sim(\mathbf{c},\mathbf{m}_j)=\frac{\mathbf{c}\cdot\mathbf{m}_j}{\|\mathbf{c}\|\|\mathbf{m}_j\|}\).
- Retrieval competition: \(P(m_j\mid c)\propto P(c\mid m_j)P(m_j)\).
- Confidence calibration: \(Cal=1-|\hat p-acc|\), where \(\hat p\) is subjective confidence and \(acc\) is observed accuracy over repeated trials.

**Biological interpretation:** Emotional state can influence access and reconstruction, but emotional intensity, confidence, and accuracy are not interchangeable measures.

## Step 14 — Detect attentional narrowing and memory trade-offs

**Purpose:** Identify when heightened focus improves memory for central information but weakens memory for context or peripheral information.

1. Compare what was highly salient with what was merely present.
2. Estimate which central features received the strongest attention.
3. Check whether background, temporal, spatial, or contextual information was neglected.
4. When anxiety or threat dominates, use available control resources to broaden the search for relevant cues if the task requires a wider view.
5. Avoid inferring that a vivid memory is a complete record of the event.

**Possible computational operations**

- Attentional concentration: \(H(A)=-\sum_i p_i\log p_i\). Lower attention-distribution entropy means resources are concentrated on fewer items.
- Central/peripheral ratio: \(T=\frac{M_{central}}{M_{peripheral}+\epsilon}\), where the ratio describes a possible trade-off in recalled details.
- Context completeness: \(C=\frac{\text{relevant context features retrieved}}{\text{relevant context features available}}\).

**Biological interpretation:** Arousal-biased competition and related accounts predict that arousal can amplify processing of prioritized information while suppressing competing information. Whether this occurs depends on task, timing, arousal, and the relationship between emotional and non-emotional features.

## Step 15 — Reappraise and update the emotional and motivational model

**Purpose:** Learn from the consequences of the event and refine future responses.

1. Compare the predicted significance or reward with the actual outcome.
2. Determine whether an emotional response was appropriate, exaggerated, or based on incomplete context.
3. Update associations between cues, outcomes, and actions.
4. Revise goals or strategies when the environment changes.
5. Store the corrected interpretation without assuming that all earlier emotional traces are erased.

**Possible computational operations**

- Appraisal error: \(\epsilon_S=S_{observed}-S_{predicted}\).
- Associative update: \(w_{t+1}=w_t+\alpha\epsilon_Sx_t\).
- Bayesian revision: \(P(H\mid E)=\frac{P(E\mid H)P(H)}{P(E)}\), where evidence \(E\) changes belief in hypothesis \(H\).

**Biological interpretation:** Reappraisal and learning involve distributed interactions among prefrontal, amygdala, hippocampal, and other systems. The degree and direction of top-down regulation depend on context and task demands.

## Step 16 — Produce an updated, state-dependent learning outcome

**Purpose:** Combine attention, emotional significance, motivation, encoding, and consolidation into a useful memory and action tendency.

1. Preserve the event representation, including the features that were strongly encoded.
2. Preserve uncertainty about omitted details and possible distortions.
3. Update semantic knowledge and action values where the event provides generalizable information.
4. Carry forward the current goal and revised expectations to future situations.
5. Permit later retrieval, sleep-dependent consolidation, and additional experiences to modify the representation.

**Possible computational operations**

- Composite learning state: \(Z_t=F(X_t,A_t,P_t,U_t,E_t,C_t)\), where \(X\) is event information, \(A\) arousal, \(P\) attentional priority, \(U\) motivational utility, \(E\) encoding, and \(C\) consolidation.
- Confidence-weighted update: \(K_{new}=K_{old}+\omega_t\Delta K\), where \(\omega_t\) reflects evidence quality and uncertainty.
- Generalization: update a concept only to the degree that the event's shared features and relations support transfer beyond the individual episode.

---

# 3. Integrated algorithmic pseudocode

```text
INPUT:
    sensory_event
    current_context
    current_goals
    prior_memories
    current_body_and_arousal_state

1. event = extract_features_and_bind_relations(sensory_event)
2. appraisal = estimate_valence_arousal_and_significance(
       event, current_context, current_goals, prior_memories
   )
3. arousal_state = update_autonomic_and_neuromodulatory_state(
       appraisal, current_body_and_arousal_state
   )
4. for each event_feature:
       salience = estimate_bottom_up_salience(event_feature)
       relevance = estimate_goal_and_context_relevance(event_feature)
       priority = combine(salience, relevance, appraisal)
5. attention = allocate_limited_resources(priority, arousal_state)
6. neural_gain = estimate_state_dependent_gain(arousal_state, task_demands)
7. attended_event = modulate_features(event, attention, neural_gain)
8. memory_trace = encode_event_and_relations(
       attended_event, hippocampal_binding, cortical_feature_systems
   )
9. motivation = estimate_expected_value_minus_effort_cost(
       current_goals, expected_outcomes, available_resources
   )
10. while task_is_active:
        progress = monitor_progress_and_errors()
        prediction_error = observed_outcome - expected_outcome
        update_reward_and_action_values(prediction_error)
        if expected_progress_exceeds_effort_cost:
            maintain_goal_and_continue_elaborative_processing()
        else:
            change_strategy_or_pause_or_stop()
11. consolidate(memory_trace, state_dependent_amygdala_modulation)
12. retrieved_memory = retrieve_when_cued(memory_trace, current_context)
13. compare_vividness_confidence_and_accuracy(retrieved_memory)
14. detect_missing_context_and_attentional_tradeoffs(memory_trace)
15. reappraise_outcomes_and_update_associations()
16. return updated_memory, updated_values, revised_goal_state
```

This pseudocode is an engineering-level decomposition. Real neural processes are parallel, recurrent, distributed, and continuously active; they do not wait for one numbered stage to finish before another begins.

---

# 4. Mathematical operations and their interpretation

| Operation | Mathematical expression | What it models | Biological caution |
|---|---|---|---|
| Feature representation | \(\mathbf{x}=[x_1,\ldots,x_n]\) | Candidate event features | Neural representations are population activity patterns, not necessarily explicit arrays |
| Valence/arousal appraisal | \((v,a)=f(x,c,g,m)\) | Significance in context | Appraisal is distributed and context-sensitive |
| Priority weighting | \(P_i=\alpha S_i+\beta G_i+\gamma C_i\) | Bottom-up and top-down attention | The weights are model parameters, not known neural constants |
| Softmax competition | \(p_i=\exp(P_i/\tau)/\sum_j\exp(P_j/\tau)\) | Relative attentional selection | A useful choice model, not a literal neural calculation |
| Gain modulation | \(y_i=g(A)x_i\) | Neuromodulatory change in responsiveness | NE effects depend on region, receptor, dose, and state |
| Limited capacity | \(\sum_i a_i\leq B\) | Finite processing resources | Capacity varies with task and state |
| Reward prediction error | \(\delta=r+\gamma V(s')-V(s)\) | Learning from better/worse-than-expected outcomes | Dopamine signals are heterogeneous; not all encode the same error |
| Value learning | \(V\leftarrow V+\alpha\delta\) | Updating expected outcomes | Simplified reinforcement-learning rule |
| Effort allocation | \(e^*=\arg\max_e[E(R\mid e)-\kappa C(e)]\) | Choosing how much effort to invest | Human decisions involve multiple interacting motives and constraints |
| Salience-modulated plasticity | \(\Delta w=\eta m\,\Delta w_0\) | State-dependent memory updating | A schematic expression, not a complete synaptic-plasticity law |
| Cue similarity | \(cos(\mathbf{c},\mathbf{m})\) | Matching cues to stored content | Actual retrieval includes pattern completion and distributed dynamics |
| Bayesian revision | \(P(H\mid E)\propto P(E\mid H)P(H)\) | Updating interpretation with evidence | Normative model; brains may approximate it imperfectly |

## 5. Biological pathways and functional roles

### Amygdala and memory modulation

The amygdala is involved in detecting and learning the significance of stimuli and in modulating memory systems. The BLA is especially important in research on the enhancement of long-term memory for emotionally arousing experiences. Its influence can reach the hippocampus, striatum, basal forebrain, and cortical systems. It is therefore more accurate to describe the amygdala as a **modulator and evaluator within a distributed system** than as a storage location for all emotional memories.

### Hippocampus and event binding

The hippocampus supports the binding of people, objects, places, actions, and temporal relations into event memories. Emotional significance can influence how strongly aspects of an event are encoded or consolidated, but the resulting memory may be selective. Attention to central emotional information can come at the cost of peripheral contextual detail.

### Locus coeruleus and norepinephrine

LC neurons provide widespread noradrenergic input. NE can influence cortical and hippocampal excitability, responsiveness, attention, and plasticity. Effects depend on firing mode, receptor subtype, local circuit, timing, and concentration. NE participates in arousal and vigilance; it should not be equated with the endocrine stress hormone cortisol.

### HPA axis and glucocorticoids

When stress is sustained or sufficiently significant, the hypothalamus, pituitary, and adrenal glands can produce a hormonal response culminating in glucocorticoid release. Glucocorticoids have time-dependent and region-dependent effects on memory. Acute stress, chronic stress, and stress occurring before, during, or after learning can have different consequences.

### Dopamine, reward, and effort

Dopamine contributes to reinforcement learning, incentive motivation, and the willingness to invest cognitive effort. Reward prediction error is a useful computational framework, but dopamine should not be reduced to pleasure or motivation alone. Prefrontal control networks, striatum, cingulate cortex, and other neuromodulators also contribute to effort selection.

### Prefrontal control and elaborative processing

Prefrontal systems help maintain goals, select strategies, inhibit irrelevant responses, and organize working memory. When a learner is motivated to understand something, they may spend more time elaborating, comparing, retrieving, explaining, and testing the material. These operations can create richer retrieval routes than passive exposure alone.

---

# 6. Validation experiments for a computational implementation

A Python simulation based on this algorithm should test the following independently. These tests validate the model's behavior; they do not prove that the simulated operations are the literal operations of the brain.

1. **Neutral versus emotionally salient encoding:** Hold the objective material constant and manipulate an abstract salience variable. Confirm that salience can raise later memory strength for prioritized details without forcing every detail to improve.
2. **Central versus peripheral detail:** Present a simulated event with one high-priority central feature and several peripheral features. Increase arousal and test whether the model can produce stronger central recall but weaker peripheral recall.
3. **Moderate versus excessive arousal:** Use a task in which broad working-memory control matters. Compare low, moderate, and high arousal settings. Ensure the model does not assume a monotonic “higher arousal is always better” relationship.
4. **Goal-directed attention:** Make one feature relevant to the learner's current goal but less emotionally salient than a distractor. Test whether top-down relevance can change priority allocation.
5. **Reward prediction error:** Deliver better-than-expected, expected, and worse-than-expected outcomes. Verify that positive, near-zero, and negative prediction errors update expected values in the appropriate direction.
6. **Effort-cost tradeoff:** Increase expected reward or reduce estimated effort cost. Confirm that modeled engagement rises only within the specified resource limits.
7. **Elaborative processing:** Compare shallow repetition with active explanation, comparison, and retrieval. Evaluate whether the latter creates more independent retrieval cues and better transfer in the simulation.
8. **Consolidation modulation:** Apply an arousal-related modulation factor during a plausible consolidation window, then compare delayed retention. Keep encoding and consolidation parameters separate.
9. **Retrieval confidence calibration:** Make the simulated memory vivid but incomplete. Confirm that confidence can be high even when objective recall accuracy is not perfect.
10. **Reappraisal:** Provide later evidence that changes the meaning of an initially ambiguous event. Check whether appraisal and future expectations update without rewriting every stored detail automatically.

## 7. Failure modes and safeguards

- **Arousal treated as universally beneficial:** use region-, task-, phase-, and state-dependent modulation rather than a single multiplier that always increases memory.
- **Norepinephrine confused with cortisol:** represent neuromodulatory and endocrine processes separately, with distinct timescales.
- **Amygdala treated as the memory store:** model its role as evaluation and modulation interacting with hippocampal and cortical memory systems.
- **Emotion treated as an accuracy guarantee:** separately represent memory strength, detail completeness, confidence, and accuracy.
- **Dopamine treated as pleasure alone:** model reward prediction, incentive value, effort allocation, and learning as related but distinct variables.
- **Motivation treated as unlimited willpower:** include effort cost, fatigue, available resources, uncertainty, and opportunity cost.
- **Attention treated as a single spotlight:** allow multiple features and goals to compete, including contextual information needed to interpret a salient event.
- **“Grit” treated as a single neural mechanism:** model persistence as a behavioral outcome of value, control, learning history, and environmental constraints.
- **Computational equations mistaken for measurements:** label parameters as hypotheses, calibrate them against behavioral and physiological data, and avoid interpreting them as known neurotransmitter concentrations or firing rates.

## 8. Limits of the model

This algorithm is a high-level synthesis, not a complete biophysical simulation. It omits many details, including receptor kinetics, cell-type diversity, oscillatory coordination, endocrine feedback loops, developmental changes, individual differences, sleep-dependent replay, social context, and interactions with pain or fatigue. The relationship between stress and memory is especially dependent on timing and task conditions. Evidence that emotional arousal can enhance consolidation is strong, but it does not imply that the full episode becomes more accurate. Reward and motivation effects on declarative memory are supported, but the precise pathways and effects differ between tasks and individuals.

For a biological implementation, each computational variable should be explicitly labeled as one of:

1. **Biologically observed variable:** for example, measured behavior, pupil-linked arousal, hormone level, or neural activity in a particular experiment.
2. **Mechanistic hypothesis:** for example, a hypothesized pathway or neuromodulatory interaction.
3. **Engineering abstraction:** for example, a softmax priority score or scalar effort-cost equation used to make the system computable.

Do not silently translate category 3 into category 1.

---

# 9. Scientific references

1. McGaugh, J. L. (2002). *Memory consolidation and the amygdala: a systems perspective.* Trends in Neurosciences, 25(9), 456–461. https://doi.org/10.1016/S0166-2236(02)02211-7. Review of BLA involvement in emotional memory consolidation and interactions with other memory systems.
2. McGaugh, J. L. (2004). *The amygdala modulates the consolidation of memories of emotionally arousing experiences.* Annual Review of Neuroscience, 27, 1–28. https://doi.org/10.1146/annurev.neuro.27.070203.144157.
3. Aston-Jones, G., & Cohen, J. D. (2005). *An integrative theory of locus coeruleus–norepinephrine function: adaptive gain and optimal performance.* Annual Review of Neuroscience, 28, 403–450. https://doi.org/10.1146/annurev.neuro.28.061604.135709.
4. Sara, S. J. (2009). *The locus coeruleus and noradrenergic modulation of cognition.* Nature Reviews Neuroscience, 10, 211–223. https://doi.org/10.1038/nrn2573.
5. Mather, M., & Sutherland, M. R. (2011). *Arousal-biased competition in perception and memory.* Perspectives on Psychological Science, 6(2), 114–133. https://doi.org/10.1177/1745691611400234.
6. Clewett, D., & Mather, M. (2014). *The locus coeruleus and regulation of attention to emotional stimuli.* In: The Oxford Handbook of Attention. Relevant work on arousal-biased competition and neuromodulatory influences.
7. Adcock, R. A., Thangavel, A., Whitfield-Gabrieli, S., Knutson, B., & Gabrieli, J. D. E. (2006). *Reward-motivated learning: mesolimbic activation precedes memory formation.* Neuron, 50(3), 507–517. https://doi.org/10.1016/j.neuron.2006.03.036.
8. Murayama, K., & Kuhbandner, C. (2011). *Money enhances memory consolidation—but only for boring material.* Cognition, 119(1), 120–124. https://doi.org/10.1016/j.cognition.2011.01.001.
9. Miendlarzewska, E. A., Bavelier, D., & Schwartz, S. (2016). *Influence of reward motivation on human declarative memory.* Neuroscience & Biobehavioral Reviews, 61, 156–176. https://doi.org/10.1016/j.neubiorev.2015.11.015.
10. Botvinick, M. M., & Braver, T. S. (2015). *Motivation and cognitive control: from behavior to neural mechanism.* Annual Review of Psychology, 66, 83–113. https://doi.org/10.1146/annurev-psych-010814-015044.
11. Westbrook, A., & Braver, T. S. (2016). *Dopamine does double duty in motivating cognitive effort.* Neuron, 89(4), 695–710. https://doi.org/10.1016/j.neuron.2015.12.029.
12. *Role of the locus coeruleus and basal forebrain in arousal and attention.* Brain Research Bulletin (2022). https://doi.org/10.1016/j.brainresbull.2022.07.014.
13. *Locus coeruleus in memory formation and Alzheimer's disease.* European Journal of Neuroscience (2021). https://doi.org/10.1111/ejn.15045.
14. Periche-Tomas, E., Husain, M., Apps, M. A. J., Little, S., & Harmer, C. J. (2026). *Neurochemical drivers of effort: The roles of dopamine and beyond in physical and cognitive exertion.* Neuroscience & Biobehavioral Reviews, 186, 106696. https://doi.org/10.1016/j.neubiorev.2026.106696. Recent review; it emphasizes that effort depends on valuation and interacting neurochemical systems rather than dopamine acting alone.

---

## Completion note

This document covers **Point 13 only**. The next point in the source document moves to social interaction and its influence on understanding and learning. That next point is intentionally not included here.
