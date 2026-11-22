# Point 02 — Gradual Cortical Learning From Repeated Experiences

## Scope and place in the sequence

This algorithm formalizes the second mechanism in *Learning, Memory, and Concept Formation*: repeated experiences produce gradual changes in cortical networks, while the hippocampal system can encode individual episodes rapidly. The target is **the slow, cumulative acquisition of regularities by distributed neocortical representations**.

This point is deliberately limited to repeated-experience cortical learning and the complementary-learning-systems principle. It does not yet implement the later stages in the source document that concern sleep-dependent replay, systems consolidation, the loss of episodic detail, concept formation, schemas, inference, or analogy. Those require separate algorithms.

The central computational idea is:

1. Each encounter activates a distributed pattern of cortical neurons.
2. Repeated encounters activate some overlapping patterns and relationships.
3. Synaptic plasticity gradually strengthens reliable, repeatedly supported relationships and weakens or fails to strengthen unreliable ones.
4. Because updates are gradual and distributed, new information can be incorporated with less disruption to previously acquired cortical knowledge than would occur with a large one-shot change.

The brain does not literally execute the vector and matrix operations below as symbolic calculations. The equations are **computational descriptions** of possible functions implemented through population activity, synaptic plasticity, inhibition, neuromodulation, and network dynamics. No single agreed-upon microscopic algorithm explains every form of cortical learning.

---

## 1. Biological objective

### 1.1 What the system is trying to achieve

The neocortex must continuously update its internal model of the world. It receives activity associated with sensory features, actions, language, events, and relations. A single experience can be unusual or noisy. If cortical synapses changed too strongly after every isolated event, the newest event could overwrite or distort previously useful knowledge.

Repeated exposure provides evidence about which relationships are stable. When different experiences repeatedly activate overlapping cortical representations, the network can gradually tune its synaptic connections to reflect the regularities shared across those experiences.

The resulting representation should support three outcomes:

- **Retention:** preserve previously useful cortical representations while learning new information.
- **Integration:** incorporate recurring features and relationships into existing distributed networks.
- **Generalization:** use learned regularities to respond appropriately to a new example that resembles, but is not identical to, previous examples.

### 1.2 Complementary learning systems

The complementary learning systems (CLS) framework proposes a division of computational labor. The hippocampal system is capable of rapidly encoding specific episodes, while neocortical networks generally learn more gradually by integrating information across experiences. The fast system can preserve information about an individual encounter; the slower system can acquire regularities that recur across encounters.

This is a useful systems-level model, not an absolute anatomical rule. Cortical learning can begin during and shortly after experience, and newer research also finds that the hippocampus can participate in learning regularities across multiple experiences. The distinction is therefore one of relative learning dynamics and representational functions, not a claim that the hippocampus only learns once or that the cortex is inactive during first exposure.

---

## 2. Inputs, state, and outputs

### 2.1 Inputs for each encounter

For each new encounter `e_t`, the algorithm receives or constructs:

- `sensory_activity_t`: neural activity produced by current sensory input (visual, auditory, somatosensory, and other relevant modalities).
- `action_activity_t`: motor, action-planning, and action-outcome activity, when relevant.
- `context_t`: spatial, temporal, task, social, and situational signals that help distinguish the current encounter.
- `goal_t`: current behavioral objective or task demand.
- `prior_cortical_state`: existing distributed cortical representations and synaptic strengths.
- `episode_trace_t`: a fast episodic record that may be available from hippocampal and medial-temporal systems. It is an input to the broader learning system, but the detailed replay operation is outside this point.
- `neuromodulatory_state_t`: modulatory influences associated with attention, novelty, reward, arousal, and behavioral relevance.

### 2.2 Persistent internal state

The cortical learner maintains:

- `A`: a set of cortical populations or representational units.
- `W`: effective connection strengths between populations. In a simplified model, `W[i,j]` represents the influence of population `i` on population `j`.
- `M`: activity and co-activity statistics accumulated across encounters.
- `P`: predictions or expected downstream activations generated by the current cortical model.
- `R`: a reliability estimate for features and relationships across encounters.
- `H`: homeostatic state, including estimates of average activity and excitability.
- `C`: context-sensitive information used to avoid treating unrelated events as identical.

These are abstract state variables. Real cortex contains many cell types, laminar circuits, dendritic compartments, recurrent connections, inhibitory interneurons, and region-specific learning rules; it is not one homogeneous matrix.

### 2.3 Outputs

After a sequence of repeated experiences, the algorithm produces:

- slowly updated cortical connection strengths;
- more stable population responses to recurring features and relations;
- increased predictive reliability for recurring patterns;
- reduced dependence on any single encounter for regularities that have become well supported;
- retained uncertainty or context-dependence where the evidence is inconsistent.

---

# 3. The complete algorithm

## Step 1 — Receive a new encounter and preserve its distinct features

**Biological purpose:** Start cortical processing of a new event without assuming that it is identical to prior events.

1. Receive concurrent sensory, motor, contextual, and task-related activity.
2. Preserve information about which features occurred together and in what context.
3. Allow specialized cortical pathways to process different information types in parallel. Visual properties, sounds, body states, actions, and language-related information need not be represented in the same cortical location.
4. Maintain enough event-specific information for the system to distinguish this encounter from other encounters.
5. Pass the distributed activity onward for relational and cross-experience comparison.

**Possible mathematical abstraction:** Represent the encounter as a feature-activity vector:

\[
\mathbf{x}_t = [x_{t1},x_{t2},\ldots,x_{tn}]^T
\]

Each component is an abstract measure of activity in a population or feature channel. A value could represent firing rate, spike count within a time window, or a normalized population response. The vector is a modeling convenience; individual neurons communicate through electrochemical signals, not by storing a literal numeric vector.

To preserve context, represent the same encounter with a context vector \(\mathbf{c}_t\), or treat context as a gating signal that changes which connections are eligible to update.

**Important distinction:** A cortical representation should not be stripped of context too early. Two events may share an object or sound but have different meanings because their relationships and circumstances differ.

## Step 2 — Construct distributed cortical representations

**Biological purpose:** Encode features and relations across populations rather than placing the entire memory in a single neuron or isolated cortical location.

1. Allow early sensory cortices to encode modality-specific properties.
2. Allow association areas to combine information from multiple features and modalities.
3. Allow recurrent cortical circuits to stabilize and refine the current population response.
4. Allow the same feature to participate in multiple distributed representations, depending on context and task.
5. Preserve overlap between related encounters so that recurring structures can influence common connections.

**Possible mathematical abstraction:** Let the cortical representation be:

\[
\mathbf{z}_t = f(\mathbf{W}_{in}\mathbf{x}_t + \mathbf{b})
\]

where:

- \(\mathbf{x}_t\) is the incoming feature activity;
- \(\mathbf{W}_{in}\) is an effective connectivity mapping;
- \(\mathbf{b}\) represents baseline influences;
- \(f\) is a nonlinear population-response function;
- \(\mathbf{z}_t\) is the resulting distributed cortical activity.

A simple rate-model choice for \(f\) might be a sigmoid or rectified function. Biological neurons have more complex membrane, synaptic, dendritic, and spike-generation dynamics, so this equation should be interpreted as a coarse population-level approximation.

**Why distributed coding matters:** If related experiences activate overlapping populations, learning can modify shared connections. A new experience can then influence an existing representation without requiring the entire network to be rewritten.

## Step 3 — Compare the current pattern with previous cortical activity

**Biological purpose:** Determine which components of the current encounter resemble patterns already represented in cortical networks.

1. Reactivate the current cortical pattern through feedforward and recurrent pathways.
2. Let the pattern interact with existing connection strengths.
3. Identify populations that respond similarly to the current encounter and earlier encounters.
4. Preserve important differences between examples rather than treating all partial similarities as identity.
5. Use contextual signals to reduce inappropriate matching between superficially similar events.

**Possible mathematical abstraction — similarity:** For two normalized activity patterns \(\mathbf{z}_t\) and \(\mathbf{z}_k\), cosine similarity can be used in a model:

\[
s(t,k)=\frac{\mathbf{z}_t^T\mathbf{z}_k}{\|\mathbf{z}_t\|\,\|\mathbf{z}_k\|+\epsilon}
\]

where \(\epsilon\) prevents division by zero. High similarity suggests overlapping population activity; it does not prove that the brain explicitly computes cosine similarity.

A second option is a distance measure:

\[
d(t,k)=\|\mathbf{z}_t-\mathbf{z}_k\|_2
\]

A network can be modeled as more likely to treat two patterns as related when their distance is small, subject to context and task constraints.

## Step 4 — Detect features and relationships that recur across encounters

**Biological purpose:** Let repeated co-activation reveal reliable regularities.

1. Track which features are activated repeatedly.
2. Track which pairs or groups of features tend to co-occur.
3. Track which cortical populations reliably precede or predict other populations.
4. Distinguish repeated relationships from accidental co-occurrences by considering the number and diversity of encounters.
5. Retain context labels so that a relationship that is valid in one situation is not automatically generalized to every situation.

**Possible mathematical abstractions:**

An exponentially weighted estimate of a feature's activation frequency can be modeled as:

\[
F_i(t)= (1-\alpha)F_i(t-1)+\alpha x_{ti}
\]

For co-activation of two populations:

\[
C_{ij}(t)= (1-\alpha)C_{ij}(t-1)+\alpha x_{ti}x_{tj}
\]

For centered activity, a covariance-like statistic is:

\[
\operatorname{Cov}_{ij}=E[(x_i-\mu_i)(x_j-\mu_j)]
\]

Here \(\alpha\) is a small update factor, not a measured universal cortical learning rate. These formulas express how repeated evidence can accumulate gradually. Neural circuits may approximate such integration through synaptic traces, biochemical processes, recurrent activity, and plasticity mechanisms.

## Step 5 — Estimate whether a detected relationship is reliable

**Biological purpose:** Avoid strengthening every relationship that happens to appear once.

1. Estimate how often each feature or relationship has been observed.
2. Consider whether it occurs in different examples or only in one repeated, highly similar setting.
3. Consider the consistency of the relationship and the amount of competing evidence.
4. Consider whether the relationship is relevant to the organism's current goals and behavioral outcomes.
5. Reduce the influence of inconsistent or noisy coincidences.

**Possible mathematical abstraction:** Estimate conditional probability:

\[
P(B\mid A)=\frac{N(A\cap B)+\beta}{N(A)+\beta+\gamma}
\]

where \(N(A\cap B)\) is the number of observed cases in which A and B co-occur, \(N(A)\) is the number of cases containing A, and \(\beta,\gamma\) are smoothing constants. This is a statistical model, not a literal neuronal division operation.

A confidence score may combine recurrence and consistency:

\[
R_{ij}=g(\text{frequency}_{ij},\text{consistency}_{ij},\text{context diversity}_{ij})
\]

The exact function \(g\) is unspecified biologically. In a simulation it could be a bounded weighted sum or Bayesian estimate.

**Caution:** Repetition alone is not sufficient evidence of truth. A repeated coincidence can still be misleading, especially when all observations come from the same narrow context. Generalization should depend on reliability and context, not exposure count alone.

## Step 6 — Activate existing cortical predictions

**Biological purpose:** The current cortical network does not merely register input; its existing connections shape the activity expected from that input.

1. Use the current pattern to activate downstream populations through learned connections.
2. Allow recurrent activity and feedback to influence the representation.
3. Generate expected feature, category, or outcome activity where the circuit architecture supports prediction.
4. Compare the expected response with the activity actually observed.
5. Preserve both matching and mismatching information for possible learning.

**Possible mathematical abstraction:**

\[
\hat{\mathbf{y}}_t = f(\mathbf{W}_{pred}\mathbf{z}_t)
\]

where \(\hat{\mathbf{y}}_t\) is predicted activity. The observed downstream pattern is \(\mathbf{y}_t\). A simple discrepancy is:

\[
\mathbf{e}_t=\mathbf{y}_t-\hat{\mathbf{y}}_t
\]

This is a useful predictive-learning formalization. Cortical prediction and error signaling are implemented through distributed circuits and are not necessarily represented as one explicit vector subtraction.

## Step 7 — Determine whether plasticity should be induced

**Biological purpose:** Synaptic change depends on more than simple simultaneous activity. Timing, postsynaptic state, neuromodulators, inhibition, and prior synaptic state can all influence plasticity.

1. Determine whether the relevant pre- and postsynaptic populations were active in a suitable temporal relationship.
2. Determine whether postsynaptic depolarization and local dendritic conditions support potentiation or depression.
3. Consider neuromodulatory signals associated with attention, novelty, reward, or behavioral significance.
4. Consider inhibitory circuit activity that can regulate which populations are allowed to change.
5. Gate or scale the plasticity signal so that a low-confidence event does not produce an excessively large persistent update.

**Possible mathematical abstractions:** A simplified plasticity eligibility factor can be written as:

\[
E_{ij}(t)=\phi(x_{ti},y_{tj},\Delta t,\text{local state})
\]

where \(\Delta t\) represents the relative timing of pre- and postsynaptic activity. A modulatory gate can be modeled as:

\[
G_t = h(\text{attention},\text{novelty},\text{reward},\text{arousal})
\]

and an effective update eligibility as:

\[
Q_{ij}(t)=E_{ij}(t)\,G_t\,R_{ij}(t)
\]

The formula is an abstract model. It does not assert that the cortex multiplies a single neuromodulator value by a synaptic eligibility number. Different neuromodulators and cell types have distinct actions.

## Step 8 — Apply small, cumulative synaptic modifications

**Biological purpose:** Strengthen or weaken connections in response to repeated, appropriately correlated activity, while avoiding abrupt global rewriting of cortical networks.

1. Identify the synapses eligible for modification.
2. Increase the effective influence of a connection when presynaptic activity repeatedly contributes to postsynaptic activation under appropriate plasticity conditions.
3. Decrease or fail to reinforce connections when evidence supports weakening, when activity patterns are anti-correlated, or when competing plasticity mechanisms favor depression.
4. Apply local changes rather than changing every connection in the network.
5. Let the effects accumulate across many encounters.

**Possible mathematical abstractions:**

A simplified Hebbian relation is:

\[
\Delta w_{ij}=\eta x_i y_j
\]

A covariance-based version is:

\[
\Delta w_{ij}=\eta(x_i-\mu_i)(y_j-\mu_j)
\]

A rate-based error-corrective update could be modeled as:

\[
\Delta\mathbf{W}= -\eta\nabla_{\mathbf{W}}\mathcal{L}
\]

where \(\mathcal{L}\) measures a mismatch between desired and predicted responses. These rules are candidate computational approximations; no single rule explains all cortical plasticity. Long-term potentiation (LTP), long-term depression (LTD), spike-timing-dependent plasticity, and other plasticity mechanisms differ across synapses, brain regions, developmental stages, and behavioral conditions.

The essential feature for this algorithm is the **small cumulative nature** of the cortical update, not a claim that every cortical synapse uses the same learning equation.

## Step 9 — Preserve distributed knowledge while integrating new evidence

**Biological purpose:** Update the cortical network without erasing useful connections that represent earlier knowledge.

1. Restrict changes to the synapses and populations involved in the current learning event.
2. Use distributed overlap so that new information can be incorporated into relevant existing representations.
3. Avoid allowing one encounter to dominate a large portion of the network.
4. Preserve well-established connections unless substantial new evidence warrants changing them.
5. Allow old and new representations to coexist when they refer to distinct contexts or conditions.

**Possible mathematical abstraction:** A regularized update can be modeled as minimizing:

\[
\mathcal{J}(\mathbf{W})=\mathcal{L}_{new}(\mathbf{W})+\lambda\,\Omega(\mathbf{W},\mathbf{W}_{old})
\]

where \(\mathcal{L}_{new}\) represents the new learning objective and \(\Omega\) penalizes excessive disruption to important existing weights. A simple illustrative penalty is:

\[
\Omega=\sum_{ij}F_{ij}(w_{ij}-w_{ij}^{old})^2
\]

where \(F_{ij}\) represents the modeled importance or stability of a connection. This resembles regularization approaches used in machine learning. It is **not** evidence that the brain computes this exact objective or stores an explicit importance matrix.

Biological protection against interference can arise from several mechanisms acting together: distributed representations, sparse or context-dependent activation, synaptic and intrinsic plasticity constraints, inhibitory control, replay and interleaving, and multiple memory systems. This step represents the functional objective, not a single proven mechanism.

## Step 10 — Integrate examples across time rather than relying on one exposure

**Biological purpose:** Accumulate evidence from different encounters so that cortical changes reflect a recurring structure rather than the accidental properties of a single event.

1. Receive a later encounter that shares some features or relations with earlier encounters.
2. Reactivate overlapping cortical populations.
3. Allow the new activity to modify existing synaptic traces incrementally.
4. Include examples that vary in irrelevant details, so common relations can become more stable than surface-specific details.
5. Retain context-specific branches where apparently similar events have different outcomes.

**Possible mathematical abstraction:** An incremental mean of a feature across N encounters is:

\[
\boldsymbol{\mu}_N=\boldsymbol{\mu}_{N-1}+\frac{1}{N}(\mathbf{z}_N-\boldsymbol{\mu}_{N-1})
\]

A weighted running estimate can use:

\[
\boldsymbol{\mu}_t=(1-\alpha)\boldsymbol{\mu}_{t-1}+\alpha\mathbf{z}_t
\]

The first equation gives equal weight to all observations; the second gives more recent observations greater influence. The nervous system may combine multiple time constants rather than follow either formula exactly.

**Why different examples matter:** If the same object is encountered under different lighting, positions, backgrounds, and goals, common features can be reinforced while variable features are less likely to dominate the learned representation. In a computational model, this can be represented by comparing within-category variability with between-category differences.

## Step 11 — Regulate activity and synaptic strength

**Biological purpose:** Long-term learning must remain stable. Unchecked potentiation could increase network activity excessively; unchecked depression could weaken useful responses.

1. Track typical activity levels of local neuronal populations.
2. Allow inhibitory circuits to constrain excessive activation and sharpen competition between populations.
3. Allow homeostatic plasticity and synaptic scaling to help maintain appropriate activity ranges.
4. Preserve the network's capacity to respond to new input.
5. Balance local Hebbian changes with stabilizing mechanisms.

**Possible mathematical abstraction:** A target-activity controller can be modeled as:

\[
\Delta w_{ij}^{homeo}=\kappa(\rho-\bar{a}_j)w_{ij}
\]

where \(\rho\) is a target activity level, \(\bar{a}_j\) is the recent average activity of the postsynaptic population, and \(\kappa\) is a small regulation factor. A simple bounded update is:

\[
w_{ij}^{new}=\operatorname{clip}(w_{ij}+\Delta w_{ij},w_{min},w_{max})
\]

These are engineering abstractions of stabilizing functions. Biological homeostasis is implemented through multiple cellular and circuit mechanisms and may act over different timescales.

## Step 12 — Test whether the updated network responds to recurring structure

**Biological purpose:** Determine whether accumulated learning has made cortical responses more reliable, without assuming that every repeated association should generalize.

1. Present a later example that shares the relevant relation but differs in superficial details.
2. Observe whether the cortical network activates the expected populations.
3. Check whether the response is supported by multiple learned features rather than one misleading cue.
4. Check whether the current context changes the response appropriately.
5. Update reliability estimates if the prediction succeeds or fails.

**Possible mathematical abstraction:** A prediction error over a set of test examples can be represented as:

\[
E_{test}=\frac{1}{N}\sum_{k=1}^{N}\|\mathbf{y}_k-\hat{\mathbf{y}}_k\|_2^2
\]

A decreasing error on new, varied examples suggests improved generalization in the model. A decreasing error only on previously repeated examples may instead indicate memorization or overfitting.

The biological system does not necessarily maintain a consciously accessible test set. Its predictions are evaluated continuously through sensory feedback, behavior, and outcome-related signals.

## Step 13 — Retain uncertainty when evidence is inconsistent

**Biological purpose:** Avoid forcing all experiences into one stable cortical rule when the environment is ambiguous or variable.

1. Detect that the same apparent cue is followed by different outcomes across encounters.
2. Check whether the difference is explained by context, hidden features, goals, or changes in the environment.
3. Preserve alternative responses or context-dependent representations if the evidence supports them.
4. Reduce confidence in a general rule when its predictions repeatedly fail.
5. Permit future learning to revise the model.

**Possible mathematical abstraction:** A model can track uncertainty in a relationship estimate, for example with a Beta distribution for a binary outcome:

\[
p\sim\operatorname{Beta}(a,b),\qquad E[p]=\frac{a}{a+b}
\]

Each confirming or disconfirming observation updates the parameters. This is a convenient statistical representation, not a claim that a neuron stores Beta-distribution parameters.

## Step 14 — Continue gradual learning across subsequent encounters

**Biological purpose:** Repeat the cycle over many experiences, allowing stable cortical knowledge to emerge over time.

1. Return to Step 1 for each new encounter.
2. Reuse existing cortical connections as the starting state.
3. Allow repeated relationships to accumulate support.
4. Allow inconsistent evidence to decrease confidence or create context-sensitive representations.
5. Keep synaptic modifications bounded and distributed.
6. Continue using the hippocampal system for rapid episode-specific learning when required; do not assume the cortical learner must replace the episodic system.

The outcome is not a single instant at which a memory is moved from one structure to another. It is a gradual change in the relative contribution and stability of distributed representations across interacting systems.

---

# 4. Integrated mathematical model

The following compact model combines the major computational functions. It is intended for simulation and conceptual analysis, not as a literal biophysical description of cortical tissue.

Let:

- \(\mathbf{x}_t\): incoming activity at encounter t;
- \(\mathbf{z}_t\): distributed cortical representation;
- \(\mathbf{W}_t\): trainable effective cortical connections;
- \(\hat{\mathbf{y}}_t\): predicted downstream activity;
- \(\mathbf{y}_t\): observed downstream activity;
- \(\mathbf{e}_t\): prediction error;
- \(\mathbf{M}_t\): accumulated co-activity statistics;
- \(\eta_t\): small effective learning rate;
- \(G_t\): modulatory gate;
- \(R_t\): reliability weighting;
- \(\Omega\): regularization term that discourages damaging changes to important prior weights.

### 4.1 Representation

\[
\mathbf{z}_t=f(\mathbf{W}_{in}\mathbf{x}_t+\mathbf{b})
\]

### 4.2 Prediction

\[
\hat{\mathbf{y}}_t=f(\mathbf{W}_t\mathbf{z}_t)
\]

### 4.3 Prediction error

\[
\mathbf{e}_t=\mathbf{y}_t-\hat{\mathbf{y}}_t
\]

### 4.4 Recurrence statistics

\[
\mathbf{M}_t=(1-\alpha)\mathbf{M}_{t-1}+\alpha\mathbf{z}_t\mathbf{z}_t^T
\]

### 4.5 Reliability- and modulation-weighted update

One possible abstract update is:

\[
\mathbf{W}_{t+1}=\mathbf{W}_t-\eta_t G_t R_t\nabla_{\mathbf{W}}\mathcal{L}_t-\eta_t\lambda\nabla_{\mathbf{W}}\Omega
\]

This combines error correction with protection of established knowledge. A different implementation might use Hebbian or spike-timing-dependent updates instead of gradient descent. The equation should not be interpreted as proof that cortex uses backpropagation.

### 4.6 Stability constraint

\[
\|\mathbf{W}_{t+1}-\mathbf{W}_t\|\leq\delta
\]

This represents the design objective that a single ordinary encounter should not cause an arbitrarily large global change. In a biological model, this constraint could be approximated through bounded synaptic updates, local plasticity, inhibitory regulation, and homeostatic mechanisms.

---

# 5. Integrated algorithmic pseudocode

```text
INITIALIZE cortical connection state W
INITIALIZE running feature statistics M_feature
INITIALIZE running co-activity statistics M_relation
INITIALIZE reliability estimates R
INITIALIZE population homeostatic state H

FOR each encounter e_t:

    1. RECEIVE sensory, action, context, and goal signals
       x_t = encode_current_input(e_t)

    2. BUILD distributed cortical activity
       z_t = cortical_population_response(x_t, W, context_t)

    3. COMPARE with existing cortical representations
       similar_patterns = retrieve_overlapping_patterns(z_t, context_t)

    4. UPDATE recurring feature and relation statistics
       M_feature = update_feature_statistics(M_feature, z_t)
       M_relation = update_coactivity_statistics(M_relation, z_t)

    5. ESTIMATE reliability of recurring relationships
       R = update_reliability(M_relation, context_t, observed_outcomes)

    6. GENERATE expected downstream activity
       predicted = cortical_prediction(z_t, W)
       observed = collect_downstream_activity()
       error = observed - predicted

    7. COMPUTE plasticity eligibility
       eligibility = local_activity_and_timing_rule(z_t, observed)
       modulation = estimate_modulatory_gate(attention, novelty, reward, arousal)

    8. FORM a bounded local update
       candidate_update = plasticity_rule(
           eligibility,
           error,
           modulation,
           reliability=R
       )

    9. PROTECT important existing representations
       candidate_update = limit_interference(
           candidate_update,
           existing_connection_state=W,
           prior_importance=estimated_stability
       )

   10. APPLY gradual cortical modification
       W = apply_local_bounded_update(W, candidate_update)

   11. STABILIZE population activity
       W, H = homeostatic_and_inhibitory_regulation(W, H)

   12. EVALUATE generalization on subsequent varied experiences
       update confidence and context dependence using new evidence

END FOR

RETURN updated cortical connection state W,
       accumulated regularity statistics M_relation,
       reliability estimates R
```

---

# 6. Mathematical operations potentially involved

The following table summarizes the kinds of mathematical relationships represented in this algorithm. These are possible computational descriptions, not a catalogue of arithmetic operations consciously or explicitly executed by neurons.

| Operation or model | Where it appears | Computational purpose | Biological interpretation / caution |
|---|---|---|---|
| Weighted summation | Population activation | Combine inputs from many synapses | Postsynaptic integration is electrochemical and time-dependent, not a perfect digital sum |
| Nonlinear transformation | Cortical response | Convert input drive into population activity | Reflects thresholding, saturation, adaptation, and other neuronal dynamics in simplified form |
| Vector similarity or distance | Cross-encounter comparison | Estimate representational overlap | A model metric for population patterns; not necessarily computed explicitly |
| Running average | Repeated exposure | Accumulate stable statistics over time | Can approximate the effects of plasticity and biochemical traces at a population level |
| Correlation / covariance | Co-activity learning | Detect recurring relationships | Related to Hebbian principles, but correlation alone is not sufficient to explain learning |
| Conditional probability | Reliability estimation | Separate robust regularity from coincidence | Statistical description; context and causal structure remain essential |
| Error subtraction | Prediction comparison | Represent mismatch between expected and observed activity | Prediction-error signals are distributed; no universal single subtraction circuit is established |
| Multiplication / gating | Plasticity eligibility | Scale synaptic change by timing and modulatory state | Abstractly captures interactions among local activity, neuromodulators, and cellular state |
| Incremental weight update | Synaptic learning | Gradually strengthen or weaken connections | Corresponds broadly to long-lasting synaptic plasticity, with many biological variants |
| Regularization / bounded update | Interference control | Limit destructive changes to prior knowledge | Computational analogy; biology uses multiple stabilizing processes, not necessarily explicit penalty minimization |
| Normalization / homeostasis | Activity stabilization | Keep network activity within functional ranges | Related to inhibitory control and homeostatic plasticity, which operate on different timescales |
| Mean squared error | Model evaluation | Quantify prediction accuracy on varied inputs | A simulation metric, not a literal neural quantity in all cortical circuits |

---

# 7. Biological interpretation and limits

## 7.1 Why slower cortical learning is useful

A system that changed its long-term cortical connections dramatically after every isolated encounter would be vulnerable to interference. A new experience might overwrite old associations, particularly when the new event is unusual or noisy. Gradual learning lets repeated evidence influence shared connections over time. The CLS framework describes this as a way to combine rapid episodic learning with slower integration into existing cortical knowledge.

## 7.2 What “slow” means

“Slow” is relative, not a fixed number of hours or days. Some forms of cortical plasticity occur rapidly, and learning can be accelerated when new information is consistent with an established knowledge structure. Other changes require repeated practice and may unfold over longer periods. Different cortical regions, layers, synapses, tasks, and developmental states have different dynamics.

## 7.3 What repeated co-activation can and cannot explain

Hebbian-style principles provide an important starting point: activity in connected populations can contribute to changes in synaptic efficacy. But a realistic account also needs inhibition, competition, neuromodulation, timing, dendritic integration, synaptic tagging, homeostatic processes, and network architecture. Simple correlation learning can reinforce accidental co-occurrence; context and outcome information are needed to constrain the learned relationship.

## 7.4 Why the hippocampus–cortex division is not absolute

The classic CLS distinction is helpful but simplified. The hippocampus is not only a one-shot episodic recorder: recent work reviews mechanisms that support both single-shot and many-shot learning within hippocampal circuits. Likewise, neocortical representations can be active during initial learning rather than waiting passively for later transfer. The algorithm therefore treats “fast hippocampal” and “gradual cortical” learning as complementary tendencies, not mutually exclusive stages.

## 7.5 Generalization must be selective

Repeated experiences do not always reveal a useful rule. The environment can contain noise, changing contingencies, and context-specific exceptions. Over-integrating every remembered detail can make a model overfit and harm generalization. A well-designed learner should strengthen relationships that are reliable across relevant contexts while preserving uncertainty or context-specific distinctions when evidence is mixed.

## 7.6 What this point does not claim

This algorithm does not claim that:

- the brain stores one global cortical matrix;
- every synapse follows the same Hebbian or gradient-descent equation;
- neurons explicitly calculate cosine similarity, covariance, or Bayesian probabilities;
- all hippocampal memories are eventually transferred completely into the cortex;
- repeated exposure alone guarantees accurate knowledge;
- cortical learning must wait until sleep;
- a specific mathematical learning rate or number of repetitions applies universally.

---

# 8. Suggested measurable variables for a computational implementation

A Python simulation inspired by this point could expose the following variables for inspection and testing:

| Variable | Suggested representation | Why log it |
|---|---|---|
| `encounter_id` | Integer or UUID | Preserve encounter order |
| `context_id` | Symbolic or vector identifier | Prevent false merging across contexts |
| `cortical_activity` | NumPy vector | Inspect distributed activation |
| `pattern_similarity` | Float | Quantify overlap between encounters |
| `coactivity_matrix` | NumPy matrix | Track recurring population relationships |
| `relationship_reliability` | Float or probability distribution | Track evidence strength and uncertainty |
| `prediction` | NumPy vector | Inspect expected downstream activity |
| `prediction_error` | NumPy vector | Inspect model mismatch |
| `plasticity_eligibility` | Matrix or sparse synapse map | Track which connections can change |
| `modulatory_gate` | Bounded scalar or vector | Model state-dependent scaling |
| `weight_delta` | Sparse matrix | Audit the size and location of updates |
| `interference_score` | Float | Detect damage to earlier learned patterns |
| `generalization_error` | Float | Test whether learning transfers to novel examples |
| `homeostatic_activity` | Vector | Monitor network stability |

For scientific honesty, every log field should distinguish a simulated quantity from a directly observed neural measurement. A simulation's `prediction_error`, for example, is a model variable and should not be labeled as a measured brain signal unless it is tied to a specific empirical measurement and analysis method.

---

# 9. Suggested validation experiments for the algorithm

These are computational tests of the proposed model, not claims that the experiments have already been run.

### Test A — One encounter versus repeated encounters

- Train one model on one example and another on multiple related examples.
- Compare their accuracy on new examples sharing the same underlying relation.
- Expected computational result: the repeated-experience model should better capture stable regularities when the examples are representative and sufficiently consistent.

### Test B — Interference after abrupt learning

- Train the model on Task A.
- Introduce Task B that uses overlapping representations.
- Compare a large one-shot update with gradual bounded updates.
- Measure retention of Task A and acquisition of Task B.
- Expected computational result: gradual or protected updates should often reduce interference, although the outcome depends on representation overlap and task conflict.

### Test C — Context sensitivity

- Present the same cue in two contexts with different outcomes.
- Compare a model that ignores context with one that uses contextual gating.
- Measure false generalization across the two situations.
- Expected computational result: context-aware learning should reduce inappropriate transfer when context reliably predicts the difference.

### Test D — Noisy repeated observations

- Add random noise to a subset of examples.
- Compare unfiltered accumulation with reliability-weighted accumulation.
- Measure performance on held-out, clean examples.
- Expected computational result: reliability-aware learning may resist some noise, but an incorrect reliability estimator can itself introduce bias.

### Test E — Generalization versus memorization

- Train on a limited set of examples and vary the amount of repeated updating.
- Track training error and held-out error separately.
- Expected computational result: excessive fitting to noisy or unrepresentative examples may improve training performance while worsening generalization.

---

# 10. References and scientific grounding

1. McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). *Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory.* Psychological Review, 102(3), 419–457. https://doi.org/10.1037/0033-295X.102.3.419
   - Foundational computational account of rapid hippocampal learning and gradual neocortical learning, including the catastrophic-interference problem.

2. Kumaran, D., Hassabis, D., & McClelland, J. L. (2016). *What learning systems do intelligent agents need? Complementary learning systems theory updated.* Trends in Cognitive Sciences, 20(7), 512–534. https://doi.org/10.1016/j.tics.2016.05.004
   - Review and update of complementary learning systems, including interactions between episodic learning and structured knowledge.

3. Squire, L. R., & Dede, A. J. O. (2015). *Conscious and unconscious memory systems.* Cold Spring Harbor Perspectives in Biology, 7(3), a021667. https://doi.org/10.1101/cshperspect.a021667
   - Broad review of memory systems and their neural organization.

4. Sun, W., Advani, M., Spruston, N., Saxe, A., & Fitzgerald, J. E. (2023). *Organizing memories for generalization in complementary learning systems.* Nature Neuroscience, 26, 1438–1448. https://doi.org/10.1038/s41593-023-01382-9
   - A mathematical theory proposing that systems consolidation should be regulated by its value for generalization; emphasizes that unregulated transfer can overfit noisy experience.

5. Liao, Z., & Losonczy, A. (2024). *Learning, Fast and Slow: Single- and Many-Shot Learning in the Hippocampus.* Annual Review of Neuroscience, 47, 187–209. https://doi.org/10.1146/annurev-neuro-102423-100258
   - Reviews evidence for both single-shot and many-shot learning modes in hippocampal circuits; useful caution against an overly rigid hippocampus-fast / cortex-slow division.

6. Bhasin, B. J., Raymond, J. L., & Goldman, M. S. (2024). *Synaptic weight dynamics underlying memory consolidation: Implications for learning rules, circuit organization, and circuit function.* Proceedings of the National Academy of Sciences, 121(41), e2406010121. https://doi.org/10.1073/pnas.2406010121
   - Computational and synaptic perspective on how plasticity in a later-learning circuit can accumulate activity patterns generated by an earlier-learning circuit. The paper models cerebellar circuitry, so it is used here as a general illustration of synaptic integration principles, not as direct evidence for a neocortical implementation.

7. Moscovitch, M., et al. (2022). *Has the concept of systems consolidation outlived its usefulness? Identification and evaluation of premises underlying systems consolidation.* https://pmc.ncbi.nlm.nih.gov/articles/PMC9720899/
   - Critical review highlighting limitations and contested assumptions in traditional systems-consolidation theories.

8. Singh, D., & Schapiro, A. C. (2026). *Evidence for complementary learning systems within the hippocampus.* Philosophical Transactions of the Royal Society B, 381(1954). https://doi.org/10.1098/rstb.2025.0243
   - Recent review of evidence that distinct hippocampal pathways may contribute differently to rapid episodic encoding and gradual structure learning.

---

## Completion boundary

This document completes **Point 02: gradual cortical learning from repeated experiences**. The next source mechanism concerns hippocampal replay during rest and sleep and its contribution to strengthening cortical representations. That next mechanism is intentionally not implemented here.
