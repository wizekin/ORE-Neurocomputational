# Point 03 — Spaced Repetition: Distributing Learning and Retrieval Over Time

## Scope and source statement

**Source:** `Improving Comprehension and Learning.txt`

> Spaced repetition: Distribute study or review over time rather than cramming. Repeated retrieval after delays has been shown to dramatically boost retention (the “spacing effect”). Each retrieval practice strengthens the memory network more than an equivalent amount of re-reading.

This document expands only this third point. It develops a sequential computational model for the processes that could support spaced learning in a human learner. It does not implement the later source points as separate algorithms. Retrieval is included here only as the means by which a spaced review can reactivate a memory; a complete analysis of the testing effect belongs to Point 04.

## 1. Scientific and modeling boundary

Spaced practice is a well-established learning effect: learning episodes distributed over time often produce better long-term retention than the same or a similar amount of practice massed into one session. The best interval is not a universal fixed number. It depends on factors such as the desired retention period, the type of material, prior learning, the learner's ability to retrieve the material, and the spacing schedule. Meta-analytic work finds that the interval between study episodes and the final retention interval interact; longer intended retention periods can call for longer study gaps. Research also cautions against assuming that expanding intervals are always superior to equal intervals.

At the neural level, learning involves activity-dependent changes in distributed neural circuits and synapses. Repeated, temporally separated training can engage molecular and cellular processes differently from tightly massed training. Sleep and offline periods can support memory consolidation; hippocampal replay and coordinated activity between hippocampal and cortical systems are important candidate mechanisms, although details and boundary conditions remain active research topics.

**Critical distinction:** the brain has not been shown to calculate a calendar date, evaluate a single scalar called “memory strength,” or run the equations below as written. The equations are computational abstractions that make the learning process explicit enough to simulate. Their variables are useful model constructs, not directly measured biological quantities. Biological explanations are stated as plausible functional correspondences, not one-to-one identifications.

The model separates three related but distinct ideas:

1. **Memory availability:** how accessible a representation is at a particular moment.
2. **Longer-term learning/stability:** how durable the representation has become following encoding, reactivation, and consolidation.
3. **Review scheduling:** an external algorithmic decision about when to present another learning opportunity. Human learners may experience a sense of familiarity, effort, or confidence, but there is no evidence that the brain literally maintains a spaced-repetition calendar.

## 2. Overall computational objective

Given a set of learned items or concepts, the algorithm should:

1. Create a memory representation after initial learning.
2. Record what was learned, when it was learned, and how successfully it could be retrieved.
3. Allow memory accessibility to change over elapsed time.
4. Estimate whether a review is likely to be too easy, appropriately effortful, or too difficult.
5. Present a review after a meaningful delay rather than immediately repeating the same information without a gap.
6. Reactivate the relevant representation and update its associations when retrieval succeeds or when corrective learning occurs.
7. Choose the next interval using performance, the target retention horizon, and uncertainty about the memory estimate.
8. Permit longer intervals as durable learning develops, while shortening an interval after forgetting or weak retrieval.
9. Allow offline consolidation to influence later stability without assuming that sleep automatically guarantees retention.
10. Repeat the cycle until the target durability or learning criterion is reached.

The central control loop is:

**Encode → allow time to pass → estimate accessibility → retrieve or relearn → update the trace → select the next delay → consolidate → repeat.**

## 3. Notation and state variables

For each memory item `i`, maintain a model state:

- `x_i`: the representation or content of the learned item.
- `t_i`: time of the most recent meaningful learning or retrieval event.
- `Δt_i`: elapsed time since that event.
- `A_i`: modeled accessibility or retrieval activation at the current time.
- `S_i`: modeled long-term stability/durability parameter.
- `C_i`: confidence in the current estimate of memory state; this is model confidence, not necessarily the learner's subjective confidence.
- `q_i`: observed retrieval outcome, for example success, partial success, or failure.
- `e_i`: retrieval effort or difficulty rating, measured behaviorally or reported by the learner.
- `h_i`: desired future retention horizon, the period for which the learner wants the material to remain usable.
- `I_i`: chosen interval until the next review.
- `d_i`: item difficulty, optionally estimated from prior learning outcomes.
- `u_i`: degree of uncertainty or variability in prior observations.
- `c_i`: contextual conditions such as fatigue, distraction, testing format, and cue availability.
- `L_i`: a trace of the learner's prior review events and outcomes.

The variables `A_i`, `S_i`, and `C_i` should not be conflated. A learner can have relatively weak immediate access but a representation that becomes retrievable with a useful cue; an item can also feel familiar while failing to be available for independent recall. The model may approximate these distinctions, but should not treat a single score as a complete description of human memory.

For a set of items, maintain a collection:

`MemoryBank = {x_1, x_2, ..., x_n}`

Each item has its own history and should normally receive its own review interval. A single schedule for all topics ignores differences in difficulty, encoding quality, and prior knowledge.

## 4. Detailed sequential algorithm

### Step 1 — Detect that a learning episode has occurred

**Purpose**

Establish that the learner has encountered information that could form or modify a memory. Spaced repetition cannot operate on an item that has not been represented at least well enough to be recognized as a learning target.

**Procedure**

1. Receive the learner's new material or experience.
2. Divide it into candidate memory units, such as a definition, factual association, procedure, rule, event, or conceptual relationship.
3. Preserve the relations needed to understand each unit. Do not split a multi-part idea into fragments that cannot be interpreted independently.
4. Assign each unit an identifier and retain its source/context.
5. Determine whether the unit is genuinely new, a refinement of an existing item, or a duplicate encounter.
6. Create or update the corresponding memory record.
7. Mark the event time as the initial learning time.

**Possible mathematical operations**

If a lesson contains `N` candidate units, a segmentation procedure produces:

`X = {x_1, x_2, ..., x_N}`

A similarity function can compare a candidate unit `x` with previously stored units `m_j`:

`sim(x, m_j) = (x · m_j) / (||x|| ||m_j||)`

This cosine-similarity expression is one possible computational tool for detecting overlap in a vector representation. It is not a claim that neurons compute cosine similarity explicitly. A threshold can support a duplicate/refinement decision:

`match(x) = argmax_j sim(x, m_j)`

If the maximum similarity exceeds a chosen threshold, the algorithm can inspect whether the new encounter should update an existing item. If not, it creates a new item.

A simple initial-state assignment could be:

`S_i(0) = S_0(d_i, encoding_quality_i)`

where `S_0` is an initialized stability estimate. The model should use a low or uncertain initial estimate when attention or comprehension was poor. No universal numerical value is implied.

**Possible biological correspondence**

Perceptual and language systems process the incoming material; attention and working-memory systems help keep relevant information available; hippocampal and cortical systems participate in forming new episodic and semantic representations. The particular systems involved depend on whether the material is declarative, procedural, emotional, perceptual, or another type. The computational “item record” is a simplified abstraction of distributed neural activity, not a literal file stored in one brain location.

**Output**

A memory item exists with an identifier, content representation, initial learning time, and an initial estimate of learning quality.

### Step 2 — Establish an initial encoding-quality estimate

**Purpose**

Avoid treating every first exposure as equally successful. A learner who carefully understands and organizes an idea may begin with a different memory state from a learner who was distracted or only glanced at it.

**Procedure**

1. Estimate whether the learner attended to the material.
2. Check whether the learner can identify its central meaning or perform the relevant action.
3. Record whether the learner understood the relationships among the parts.
4. Record the presence of cues, examples, and context that may later support retrieval.
5. Assign an encoding-quality estimate and its uncertainty.
6. Use this estimate to initialize the memory-state model, but do not infer long-term retention from a single immediate success.

**Possible mathematical operations**

A bounded encoding-quality score can be represented as a weighted combination:

`E_i = w_a Atn_i + w_c Compr_i + w_r Rel_i + w_p Prac_i`

where each input is normalized to a common range and the weights sum to one. The symbols represent attention, comprehension, relational organization, and meaningful practice. These are modeling variables, not a validated universal biological formula.

One can constrain the score to `[0,1]`:

`E_i = min(1, max(0, E_i))`

The initial state could be:

`S_i(0) = S_min + E_i (S_max - S_min)`

A Bayesian model could instead retain a probability distribution over possible stability values, updating it when future retrieval outcomes are observed. This is often preferable to pretending the first estimate is precise.

**Possible biological correspondence**

Attention, elaborative processing, and the organization of the information influence what is encoded and which neural populations are coactivated. Neuromodulatory systems can affect plasticity and encoding, but a simple score such as `E_i` is only a computational summary of many interacting processes.

**Output**

An initial, explicitly uncertain estimate of the item's encoded state.

### Step 3 — Register the temporal origin and allow a meaningful delay

**Purpose**

Create separation between learning encounters. Immediate repetition can increase short-term familiarity without requiring the learner to reconstruct the memory after it has become less accessible.

**Procedure**

1. Store the time of the last meaningful learning event.
2. Do not automatically schedule the next exposure at zero delay.
3. Choose an initial gap appropriate to the item's difficulty and the intended retention horizon.
4. Allow ordinary activity and time to intervene.
5. Avoid assuming that every gap must be identical or that all items should be reviewed at the same time.
6. When the gap has elapsed, evaluate the item's current accessibility before choosing the next action.

**Possible mathematical operations**

Elapsed time is:

`Δt_i = t_now - t_i`

The review is eligible when:

`Δt_i ≥ I_i`

A simple initial interval may be based on difficulty and target retention:

`I_i,0 = clip(k × h_i × g(d_i, E_i), I_min, I_max)`

Here, `k` is a calibration factor, `h_i` is the desired retention horizon, and `g` decreases the interval for harder or weakly encoded items. The function and constants require empirical calibration. This formula describes a scheduler; it is not a neural calculation.

**Possible biological correspondence**

The biological relevance of time gaps may involve changes in the state of neural representations, synaptic and molecular processes, interference from other experiences, and opportunities for offline consolidation. There is no need to suppose the brain deliberately waits for a computed time interval. The human algorithm's scheduling clock is an external operational representation of the learning strategy.

**Output**

A review time is separated from the original learning event by a nonzero interval.

### Step 4 — Estimate the memory's accessibility after the delay

**Purpose**

Model the fact that the likelihood of retrieving a memory can change as time passes. The estimate helps determine whether the next review will be productive rather than either trivial or impossibly difficult.

**Procedure**

1. Calculate elapsed time since the last meaningful encounter.
2. Retrieve the item's stability estimate and event history.
3. Estimate its current accessibility.
4. Incorporate the number and quality of previous learning/retrieval events.
5. Represent uncertainty in the estimate.
6. Do not equate subjective familiarity with successful recall.

**Possible mathematical operations**

One illustrative exponential-decay model is:

`A_i(Δt) = A_0 × exp(-Δt / τ_i)`

where `A_0` is initial modeled accessibility and `τ_i` is a time constant related to stability. Larger `τ_i` means slower modeled decline. This is a convenient approximation, not a universal law of human forgetting.

A power-law alternative is:

`A_i(Δt) = A_0 × (1 + α Δt)^(-β)`

where `α` and `β` are fitted parameters. Different forgetting functions can fit different tasks and timescales; the model should select and validate a form rather than claim one equation governs all memories.

A possible mapping from accessibility to recall probability is a logistic function:

`P_recall_i = 1 / (1 + exp(-(A_i - θ_i) / σ_i))`

where `θ_i` is an effective retrieval threshold and `σ_i` controls uncertainty/variability. For a model using stability directly, a common alternative is:

`P_recall_i(Δt) = exp(-Δt / S_i)`

This last expression is a compact scheduler approximation and should not be interpreted as a literal neural firing probability.

**Possible biological correspondence**

Retrieval depends on the state of distributed neural representations and the availability of effective cues. Changes in accessibility can arise from competition, interference, cue mismatch, and altered activation patterns, not only passive time-based decay. Neural populations do not simply fade at one fixed rate. The mathematical decay term summarizes the net behavioral effect for a particular task.

**Output**

A predicted accessibility or recall-probability estimate, plus an uncertainty estimate.

### Step 5 — Decide whether the time is appropriate for review

**Purpose**

Choose a review opportunity at a point where successful retrieval is plausible but not effortless. This balance can make the review informative: too little delay may leave the answer in short-term availability; too much delay may make retrieval fail entirely.

**Procedure**

1. Read the predicted recall probability and its uncertainty.
2. Compare it with a target difficulty band.
3. If the item is predicted to be nearly perfectly accessible, consider a longer interval, unless immediate practice is needed for fluency or safety-critical performance.
4. If it is predicted to be retrievable with effort, proceed with a review.
5. If the model predicts near-certain failure, provide a cue or brief relearning before expecting independent retrieval.
6. Account for the target date by which knowledge must be available.
7. Do not lengthen the interval solely because the learner recognizes the item when it is shown.

**Possible mathematical operations**

Define a target recall probability `p*` for a particular review task. For example, a scheduler may aim for a region such as `[p_low, p_high]`, but the values are task-dependent and must not be treated as universal constants.

A simple decision function is:

`Action_i =`

- `REVIEW` if `p_low ≤ P_recall_i ≤ p_high`;
- `DEFER` if `P_recall_i > p_high` and the final deadline permits;
- `RELEARN_WITH_CUE` if `P_recall_i < p_low`;
- `URGENT_REVIEW` if the final retention deadline is near.

A scheduler can calculate the estimated time at which recall probability will reach a target value `p*`. Under the exponential model:

`Δt_target = -S_i × ln(p*)`

This gives an interval estimate from a chosen memory model. It should be limited by minimum and maximum practical intervals and adjusted for the final test/usage date.

**Possible biological correspondence**

The brain can experience different levels of retrieval accessibility and effort. Neuromodulatory and control systems may be involved in effort, attention, and learning from successful or failed attempts. But the numerical difficulty band is imposed by the computational model; it is not a measured neural set point.

**Output**

A decision to retrieve, defer, or briefly relearn with support.

### Step 6 — Attempt retrieval rather than immediately re-reading

**Purpose**

Reactivate the stored representation after a delay. The source point emphasizes repeated retrieval after delays; therefore, the review should initially test what can be produced from memory rather than simply show the answer again.

**Procedure**

1. Present a cue that points to the target without displaying the full answer.
2. Give the learner time to reconstruct the information.
3. Ask for free recall, cued recall, an explanation, a diagram, or performance of the learned procedure, depending on the target.
4. Record the response before showing the correct answer.
5. Distinguish complete recall, partial recall, a plausible but inaccurate response, recognition only, and failure to retrieve.
6. Record retrieval latency or effort if useful.
7. Avoid interpreting an unanswered item as proof that the memory trace has been erased.

**Possible mathematical operations**

A response can be scored with a weighted rubric:

`Q_i = w_1 Accuracy_i + w_2 Completeness_i + w_3 Relation_i + w_4 Fluency_i`

Each component is scaled to `[0,1]`; weights depend on what the material requires. For a mathematical rule, accuracy may dominate. For a conceptual explanation, preserving relations may matter more than matching exact wording.

The retrieval outcome can be categorized:

`q_i ∈ {0, partial, 1}`

or represented continuously as `Q_i ∈ [0,1]`.

A response-time term could be included:

`Effort_i = f(response_latency, hints_used, subjective_difficulty)`

This measure should be kept separate from correctness. Slow but correct recall may be meaningfully different from fast recognition.

**Possible biological correspondence**

A cue can trigger pattern completion or partial reactivation of a distributed representation. Hippocampal and cortical networks can contribute depending on the memory. The reactivation process is not a single “read from storage” operation; it is reconstructive and cue-dependent.

**Output**

An observed retrieval result, effort estimate, and cue-condition record.

### Step 7 — Provide corrective information when retrieval is incomplete or wrong

**Purpose**

Prevent a wrong response from becoming the most recently rehearsed version of the item. Retrieval is followed by comparison with reliable information and correction when needed.

**Procedure**

1. Compare the response with the correct target representation.
2. Identify which components were correct, missing, or incorrect.
3. Present the correct answer or demonstration.
4. Explain the specific difference, rather than merely labeling the response wrong.
5. Ask the learner to reconstruct the corrected version.
6. Update the stored representation or its links if the original learning was incomplete.
7. Record that this review included feedback/relearning, because it is not equivalent to an unaided successful recall.

**Possible mathematical operations**

Let `y_i` be the correct target and `ŷ_i` the learner's response. An error measure can be:

`Loss_i = d(y_i, ŷ_i)`

For a numeric target, `d` may be squared error. For a structured concept, it can combine missing relations, incorrect attributes, and contradictory claims:

`Loss_i = λ_1 L_missing + λ_2 L_incorrect + λ_3 L_relation`

An update can be represented abstractly as:

`x_i,new = x_i,old + η × CorrectionSignal_i`

where `η` is a learning-rate parameter. This is an algorithmic analogy, not a literal claim that a single scalar learning rate is used throughout the brain.

The scheduler should store correction separately from recall success:

`EventType ∈ {successful_retrieval, partial_retrieval, failed_retrieval, corrected_relearning}`

**Possible biological correspondence**

Feedback and error-sensitive learning can alter subsequent processing. Reactivated memories can sometimes be modified, but reconsolidation does not occur identically for all memories and should not be assumed to happen every time a person recalls something. New encoding and corrective learning can contribute even without invoking reconsolidation.

**Output**

A corrected or clarified representation and a detailed record of the review event.

### Step 8 — Update the modeled memory state based on the review

**Purpose**

Represent the fact that a review can change later retention. A successful, effortful retrieval may strengthen future accessibility; a failed retrieval followed by relearning may also improve learning, but it should not be scored as identical to a successful independent recall.

**Procedure**

1. Read the outcome, accuracy, effort, and feedback condition.
2. Estimate whether the item was retrieved independently or only recognized.
3. Increase the stability estimate after reliable retrieval, with a gain related to outcome and task difficulty.
4. Apply an appropriate update after partial recall or corrected relearning.
5. Reduce confidence in the state estimate when outcomes are inconsistent.
6. Retain the event history so later estimates can be recalibrated.
7. Avoid assuming every retrieval causes equal strengthening.

**Possible mathematical operations**

A simple update rule is:

`S_i,new = S_i,old + η_i × G(q_i, e_i, feedback_i)`

where `G` is an outcome function. For example, it could give a positive update for successful retrieval, a smaller or different update for partial retrieval, and a distinct update for relearning after failure. The function must be empirically calibrated.

A bounded version is:

`S_i,new = min(S_max, max(S_min, S_i,new))`

A probabilistic implementation can update a Beta distribution over the probability of successful retrieval:

`p_i ~ Beta(a_i, b_i)`

After a binary success, update `a_i ← a_i + 1`; after a failure, update `b_i ← b_i + 1`. Partial outcomes can be represented using a more elaborate likelihood or weighted fractional update, but the simple Beta-Bernoulli example is only a model of observed performance, not a biological memory molecule.

One possible difficulty-sensitive gain is:

`η_i = η_0 × h(e_i) × r(q_i)`

where `h` and `r` are chosen so that meaningful effort and correct retrieval can influence the update while very easy recognition does not automatically produce a large gain. This is a hypothesis for a scheduler, not a universal empirical law.

**Possible biological correspondence**

Reactivation can coincide with synaptic plasticity and changes in the strength or coordination of neural connections. Repeated spaced training has been associated with molecular signaling, transcription, and dendritic-spine remodeling in experimental systems. The mapping from those distributed biological processes to the single state variable `S_i` is intentionally coarse.

**Output**

An updated estimate of the item's durability, with an event history retained for future recalibration.

### Step 9 — Choose the next review interval adaptively

**Purpose**

Use the review result to determine when the next learning opportunity should occur. The interval should be lengthened when retention is robust and shortened when the learner repeatedly struggles, subject to the final retention goal.

**Procedure**

1. Start from the updated stability/accessibility estimate.
2. Consider accuracy and whether retrieval was independent.
3. Consider retrieval effort and cue dependence.
4. Consider the desired retention horizon.
5. Consider the learner's actual schedule and the practical minimum/maximum intervals.
6. Set the next interval.
7. Store the next due time and the reason for the choice.
8. If the learner has a fixed examination or performance date, ensure that the final review plan includes a suitable opportunity before that date.

**Possible mathematical operations**

A simple multiplicative interval update is:

`I_i,next = I_i,current × m(q_i, e_i, d_i)`

where `m` is greater than one after robust successful recall, close to one after an uncertain/partial result, and less than one after failure or high cue dependence. Its values should be calibrated for the task rather than copied as universal constants.

A stability-based alternative is:

`I_i,next = clip(-S_i × ln(p*), I_min, I_max)`

where `p*` is a target recall probability at the next review. If the target horizon is `h_i`, the scheduler can also constrain the interval to leave enough time for at least one later review:

`I_i,next ≤ ρ × h_remaining`

where `ρ` is a planning proportion chosen by the scheduler. This is a practical scheduling constraint, not a neural law.

A Bayesian adaptive scheduler can choose the interval that maximizes expected utility:

`I* = argmax_I E[ U(success, effort, time_cost, final_retention) | history_i ]`

This makes explicit the trade-off between review burden, retrieval success, and eventual retention. The expected utility can be learned from the learner's own history.

**Possible biological correspondence**

The human brain adapts learning and retrieval based on prior experience, error, success, and context, but the calendar interval is chosen by a human or software scheduler. A learner's sense of fluency or difficulty may influence voluntary review behavior, yet it is not equivalent to an explicit interval optimization algorithm inside the brain.

**Output**

A personalized next-review interval and a reasoned scheduling record.

### Step 10 — Repeat the cycle while monitoring long-term retention

**Purpose**

Turn spaced repetition into a continuing learning process rather than a one-time delayed test.

**Procedure**

1. Wait until the next scheduled interval.
2. Re-estimate current accessibility.
3. Run the retrieval attempt.
4. Provide correction or relearning if needed.
5. Update the memory-state estimate.
6. Select the next interval.
7. Compare performance across successive sessions.
8. Mark the item as provisionally stable only after it meets a predefined criterion across meaningful delays.
9. Continue occasional checks if the item must remain available for a long period.
10. Reopen the cycle when later performance reveals forgetting or when new knowledge changes the target representation.

**Possible mathematical operations**

The review history can be represented as:

`L_i = {(t_1,q_1,e_1), (t_2,q_2,e_2), ..., (t_n,q_n,e_n)}`

The model can estimate a trend in performance:

`Trend_i = slope(Q_i over review time)`

A stable criterion may require:

- performance above a chosen threshold;
- success after more than one nontrivial delay;
- acceptable effort or latency for the intended task;
- and retention at a time horizon relevant to the learner's goal.

A simple stopping condition is:

`STOP_i = (Q_i ≥ θ_Q) AND (Δt_i ≥ h_required) AND (independent_retrieval = true)`

In practice, the scheduler should not permanently stop reviewing information that must remain accessible for years. A low-frequency maintenance schedule may be more appropriate.

**Possible biological correspondence**

Repeated learning episodes can strengthen and reorganize distributed representations. The long-term result may include changes in accessibility, stability, and the degree to which a memory is supported by cortical networks. A computational stopping rule is only a practical criterion; no single test can prove that a memory will never be forgotten.

**Output**

A continuing schedule or a provisional maintenance state, with a retained record of performance over time.

### Step 11 — Include offline consolidation as a separate state transition

**Purpose**

Represent the possibility that memory processing continues between deliberate study sessions. Spaced repetition is not simply a sequence of conscious reviews; offline periods, including sleep, can contribute to consolidation.

**Procedure**

1. Mark periods after learning or retrieval when the learner is not actively studying the item.
2. Do not artificially insert a “sleep boost” as a guaranteed fixed percentage.
3. Allow the model to represent consolidation as a probabilistic or uncertain influence on later stability.
4. If relevant, record sleep timing and quality only when available and appropriate.
5. Keep consolidation separate from deliberate retrieval practice.
6. Avoid claiming that every memory is replayed or consolidated equally during every sleep period.

**Possible mathematical operations**

One abstract state transition is:

`S_i(t + Δt) = S_i(t) + C_i(Δt, state, trace_strength) - D_i(Δt, interference)`

where `C_i` represents a possible consolidation contribution and `D_i` represents destabilizing influences such as interference or decay. Both terms are conceptual functions, not directly measurable universal quantities.

A probabilistic model could represent an offline consolidation event as:

`S_i,new = S_i,old + Z_i × κ_i`

where `Z_i` is a random or evidence-conditioned event indicator and `κ_i` is the potential effect. This must not be interpreted as a literal all-or-none biological switch; it is a convenient simulation device for uncertainty.

**Possible biological correspondence**

Research on sleep-dependent memory consolidation describes repeated reactivation of neuronal activity patterns, including hippocampal replay during non-REM sleep and coordination with thalamic and cortical rhythms. This is associated with the stabilization and reorganization of memory representations. The exact transformation of memories, the contribution of different sleep stages, and the mechanisms for different memory types remain under investigation.

**Output**

A cautious update to the modeled memory state that acknowledges offline consolidation without assuming a guaranteed gain.

### Step 12 — Protect against misleading familiarity and measurement errors

**Purpose**

Ensure the algorithm measures useful retention rather than mere recognition, repeated exposure, or a temporary feeling of fluency.

**Procedure**

1. Use at least one review mode that requires production from memory.
2. Vary the cue format when appropriate, while keeping the target knowledge constant.
3. Compare recognition with free or cued recall.
4. Record when hints or answer choices were supplied.
5. Do not mark a memory mastered solely because the learner says it looks familiar.
6. Treat ambiguous responses as uncertain rather than automatically correct.
7. Recalibrate the scheduler if its predicted probability repeatedly disagrees with observed performance.

**Possible mathematical operations**

Separate performance channels:

`P_total = w_r P_recall + w_c P_cued_recall + w_n P_recognition`

For a goal requiring independent recall, set `w_r` higher than `w_n`. The weights depend on the intended real-world task.

Prediction error can be monitored:

`ε_i = observed_success_i - predicted_success_i`

Calibration can be assessed by grouping predictions into probability bins and comparing predicted with observed success rates. A systematic positive error indicates overconfidence by the model; a systematic negative error indicates underconfidence.

**Possible biological correspondence**

Familiarity and recollection can be partly dissociable memory signals and can depend on different processes. A computational scoring system should therefore avoid treating familiarity as complete evidence of retrievable knowledge.

**Output**

A more reliable estimate of usable retention and a scheduler that can correct its own prediction errors.

## 5. Integrated algorithmic pseudocode

```text
ALGORITHM SPACED_REPETITION(memory_item, target_retention_horizon):

    1. item <- SEGMENT_AND_REGISTER(memory_item)
    2. item.encoding_quality, item.uncertainty <- ESTIMATE_INITIAL_ENCODING(item)
    3. item.stability <- INITIALIZE_STABILITY(item.encoding_quality)
    4. item.last_learning_time <- CURRENT_TIME()
    5. item.target_horizon <- target_retention_horizon
    6. item.history <- EMPTY_LIST
    7. item.next_interval <- CHOOSE_INITIAL_INTERVAL(
           item.difficulty,
           item.encoding_quality,
           item.target_horizon
       )

    8. WHILE NOT MAINTENANCE_CRITERION_MET(item):

           WAIT_UNTIL(item.last_learning_time + item.next_interval)

           elapsed <- CURRENT_TIME() - item.last_learning_time
           predicted_accessibility, uncertainty <- ESTIMATE_ACCESSIBILITY(
               item.stability,
               elapsed,
               item.history
           )

           IF predicted_accessibility is too high AND deadline permits:
               item.next_interval <- EXTEND_INTERVAL(item)
               CONTINUE

           IF predicted_accessibility is too low:
               cue_level <- SELECT_SUPPORTIVE_CUE(item)
           ELSE:
               cue_level <- MINIMAL_CUE

           response <- ATTEMPT_RETRIEVAL(item, cue_level)
           outcome <- SCORE_RESPONSE(response, item.correct_representation)

           IF outcome is incomplete OR incorrect:
               feedback <- PRESENT_CORRECTION(item, response)
               RECONSTRUCT_CORRECT_RESPONSE(item, feedback)
               event_type <- CORRECTED_RELEARNING
           ELSE:
               event_type <- INDEPENDENT_RETRIEVAL_SUCCESS

           APPEND(item.history, {
               time: CURRENT_TIME(),
               elapsed: elapsed,
               outcome: outcome,
               effort: response.effort,
               cue_level: cue_level,
               event_type: event_type
           })

           item.stability, item.uncertainty <- UPDATE_MEMORY_STATE(
               item.stability,
               item.uncertainty,
               outcome,
               response.effort,
               event_type
           )

           item.next_interval <- CHOOSE_NEXT_INTERVAL(
               item.stability,
               item.history,
               item.target_horizon,
               item.difficulty,
               item.uncertainty
           )

           APPLY_OPTIONAL_OFFLINE_CONSOLIDATION_MODEL(item)
           CHECK_MODEL_CALIBRATION(item)

    9. RETURN item, item.history, item.next_interval
```

This is a high-level algorithm for a learning-support system. It does not imply that a biological brain carries out a symbolic `WHILE` loop or stores the fields as a conventional data structure.

## 6. Mathematical model set — interpretation and limitations

### 6.1 Time and interval arithmetic

`Δt = t_now - t_last`

This is ordinary elapsed-time bookkeeping in the software model. It defines the temporal gap between learning events.

### 6.2 Forgetting/accessibility functions

Possible functions include exponential decay, power-law decay, or empirically fitted alternatives. They are candidate behavioral descriptions. Different material, tasks, cue conditions, and retention periods may require different models.

### 6.3 Probability estimation

A logistic or exponential mapping can estimate recall probability. The result is a prediction that should be calibrated against actual retrieval outcomes, not an objective reading of a memory trace.

### 6.4 State updating

A weighted update or Bayesian update can revise the estimated state after observed success or failure. The exact update should be selected for the task and tested on learner data.

### 6.5 Interval optimization

An optimization rule can choose the next interval by balancing retention, effort, and time cost. A fixed expanding schedule is easy to implement but is not guaranteed to be optimal for every learner or material type.

### 6.6 Error and uncertainty

Prediction error and confidence tracking help prevent an algorithm from becoming overconfident. A model that never tests itself against actual retrieval outcomes is not a reliable adaptive scheduler.

## 7. Possible biological implementation — from algorithmic steps to neural processes

The following correspondences are deliberately broad. Brain networks overlap in function, and the same memory task can recruit different systems depending on content, attention, and context.

### 7.1 Initial encoding and representation formation

When information is first learned, sensory and association cortices process its features. Attention and working-memory processes help select and maintain relevant information. The hippocampal formation can bind distributed components into an episodic representation, while cortical systems represent content and meaning. The algorithm's `memory_item` is a computational stand-in for this distributed representation.

### 7.2 Reactivation during review

A review cue can reactivate part of a stored representation. Successful recall may involve pattern completion and coordinated activity across hippocampal and cortical networks. The recalled representation is reconstructed from partial information and current context, rather than retrieved like an unchanged computer file.

### 7.3 Activity-dependent plasticity

Learning-related coactivation can alter synaptic efficacy and the future responsiveness of neural circuits. In many experimental models, plasticity involves multiple time-dependent processes, including receptor trafficking, kinase and signaling cascades, gene expression, protein synthesis, and structural changes. Spaced training may permit these processes to interact differently from massed training. These are distributed mechanisms; no single one is a complete explanation of the spacing effect.

### 7.4 Why temporal separation may matter

One family of explanations proposes that closely massed presentations can occur while some relevant processing remains active or refractory, producing less effective encoding of a later presentation. A spaced presentation can occur after those states have changed, allowing renewed processing and plasticity. Other accounts emphasize retrieval of the earlier learning episode, contextual variability, attention, and the time course of consolidation. There is no single explanation accepted as sufficient for every task.

### 7.5 Offline processing and sleep

During sleep, especially non-REM sleep, hippocampal activity patterns can be reactivated and coordinated with cortical slow oscillations and thalamic spindles. This coordinated activity is implicated in systems consolidation, in which memory representations can become more stable and reorganized across networks. The details vary by memory type and remain a research topic. A review scheduler should therefore treat sleep as a potentially helpful biological condition, not as a guaranteed memory-strength multiplier.

### 7.6 Neuromodulation and motivation

Neuromodulatory systems can influence attention, salience, plasticity, and learning. Dopamine and other neuromodulators may affect learning under particular conditions, but the algorithm should not reduce spacing to a simplistic “dopamine increase per review” rule. Point 08 in the source separately addresses motivation and curiosity.

## 8. Worked example — learning a scientific definition

Suppose a learner studies a definition: “A synapse is a specialized junction through which one neuron communicates with another cell.” The learner wants to retain the definition and explain its meaning accurately after several weeks.

1. **Initial encounter:** The definition is registered as a memory item, with its key elements: specialized junction, communication, neuron, and target cell.
2. **Initial encoding estimate:** The learner explains the sentence in their own words. The model records whether the explanation preserves the essential relation.
3. **First delay:** The software selects a nonzero initial review gap based on the learner's performance and intended retention horizon. This is a scheduler choice, not a neural time calculation.
4. **Accessibility estimate:** The model predicts that recall may be less fluent after the gap, with uncertainty retained.
5. **Retrieval:** The learner is asked, “What is a synapse?” without being shown the original definition.
6. **Outcome:** The learner recalls “a junction where neurons communicate” but omits that the other cell need not be a neuron.
7. **Correction:** The missing element is shown and explained. The learner reconstructs the full meaning.
8. **State update:** The algorithm records partial independent recall followed by corrected relearning, rather than recording a perfect unaided success.
9. **Next interval:** The scheduler chooses a moderate next gap. A later accurate explanation after a longer delay can justify extending the interval.
10. **Long-term check:** Before the desired retention date, the learner explains the definition and applies it to an example. The model marks the item provisionally stable only if the retrieval meets the actual learning goal.

Numerical scores and intervals in a real implementation should come from calibration data. The example intentionally avoids presenting arbitrary numbers as scientifically established values.

## 9. Edge cases and control rules

### 9.1 Immediate failure after a long delay

Do not simply increase the interval. Supply a useful cue, repair the encoding, and schedule another attempt sooner. A single failure may reflect cue mismatch or distraction rather than total loss.

### 9.2 Easy recognition but poor free recall

Do not label the item mastered. Use a retrieval mode closer to the learner's real goal and record the difference between recognition and recall.

### 9.3 Inconsistent performance

Increase the uncertainty of the estimate. Consider task difficulty, fatigue, distraction, cue format, and interference before making a large interval change.

### 9.4 A deadline is approaching

The scheduler may need to use shorter intervals or more frequent practice even when long-term efficiency would otherwise favor a longer gap. The user's goal determines the optimization target.

### 9.5 Very simple or highly familiar content

Avoid spending excessive review time on items already reliably accessible. However, do not assume that familiarity guarantees future independent performance.

### 9.6 Complex concepts with multiple components

Track component-level performance as well as whole-concept performance. The learner may remember the label but lose the causal relation, boundary condition, or procedural sequence.

### 9.7 Interference or changed knowledge

When new information changes the correct explanation, update the target representation and retain the previous version only where historically relevant. Do not reinforce an outdated answer merely because it was previously learned.

### 9.8 Poor sleep or fatigue

Treat these as contextual factors that may affect attention and performance. Do not infer a precise amount of lost memory or apply a fixed numerical penalty without validated data.

## 10. Suggested modular software architecture

An implementation can separate the scheduler from the modeled memory process:

- `MemoryItem`: identity, content, components, source, difficulty, target horizon.
- `EncodingEstimator`: records the quality and uncertainty of initial learning.
- `AccessibilityModel`: estimates accessibility as a function of time and event history.
- `RetrievalInterface`: presents a cue and captures a response.
- `ResponseScorer`: scores correctness, completeness, effort, and cue dependence.
- `FeedbackEngine`: supplies corrections and checks reconstruction.
- `StabilityUpdater`: updates the item's estimated durability.
- `IntervalScheduler`: selects the next review time.
- `ConsolidationModel`: optional research-oriented simulation of offline influences.
- `CalibrationMonitor`: compares predicted recall with actual performance and revises model parameters.
- `EventLogger`: records each learning/retrieval event for auditability and later analysis.

Every function should record its inputs, outputs, model assumptions, and parameter version. This allows the system to distinguish observed learning outcomes from model-generated estimates.

## 11. Validation requirements for a computational implementation

A software implementation should be tested rather than assuming its formulas work because they look plausible.

1. **Baseline comparison:** compare spaced practice against an equal-time massed-practice condition.
2. **Equal practice opportunities:** ensure conditions receive comparable amounts of study and testing time.
3. **Delayed final test:** assess retention after a meaningful delay, not only immediately after training.
4. **Different item difficulty:** test whether the scheduler adapts rather than giving every item the same interval.
5. **Calibration:** compare predicted recall probabilities with observed success frequencies.
6. **Ablation:** remove adaptive interval selection, retrieval scoring, or feedback one at a time to determine which components contribute.
7. **Robustness:** vary cue type, learner state, and material type.
8. **Retention horizon:** test more than one final delay, since an interval helpful for a short-term test may not be optimal for a much later test.
9. **No misleading biological claim:** label computational variables as modeled constructs and distinguish them from measured neural variables.
10. **Reproducibility:** store random seeds, parameters, event histories, and evaluation procedures.

## 12. Key assumptions and what this algorithm does not claim

- It does not claim that every repeated review strengthens memory by an equal amount.
- It does not claim that an expanding schedule is always better than fixed spacing.
- It does not claim that a single forgetting curve describes all forms of memory.
- It does not claim that the brain calculates review intervals or stores a literal next-review timestamp.
- It does not claim that a successful response proves permanent retention.
- It does not claim that sleep guarantees consolidation of every item.
- It does not treat subjective familiarity as equivalent to accurate independent retrieval.
- It does not equate a software variable such as `stability` with a single synaptic property.
- It does not assume a one-to-one mapping between a cognitive operation and one brain region.

The strongest defensible claim is that distributed learning and retrieval can improve long-term retention across many tasks, while the best schedule and the underlying mechanisms depend on the learner, material, and target delay.

## 13. Summary of the complete algorithmic sequence

1. Detect and register a new learning item.
2. Estimate initial encoding quality and uncertainty.
3. Record the temporal origin and schedule a meaningful delay.
4. Estimate accessibility after the delay.
5. Decide whether the item is ready for review or needs more support.
6. Attempt retrieval before re-exposure.
7. Provide corrective feedback when needed.
8. Update the estimated memory state using the observed outcome.
9. Choose the next interval adaptively.
10. Repeat and monitor retention over progressively meaningful delays.
11. Represent offline consolidation as a possible, uncertain influence.
12. Protect the model against familiarity, cueing, and measurement errors.

**End of Point 03.** The next source point is **Point 04 — Testing Effect**. It has not been developed in this file.

## 14. References and further reading

1. Carpenter, S. K., Pan, S. C., & Butler, A. C. (2022). “The science of effective learning with spacing and retrieval practice.” *Nature Reviews Psychology*, 1, 496–511. https://doi.org/10.1038/s44159-022-00089-1. A broad review of spacing and retrieval practice, including implications for applied learning and metacognition.
2. Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., & Rohrer, D. (2006). “Distributed practice in verbal recall tasks: A review and quantitative synthesis.” *Psychological Bulletin*, 132(3), 354–380. https://doi.org/10.1037/0033-2909.132.3.354. A meta-analysis showing that spacing and the final retention interval interact in determining retention.
3. Smolen, P., Zhang, Y., & Byrne, J. H. (2016). “The right time to learn: mechanisms and optimization of spaced learning.” *Nature Reviews Neuroscience*, 17, 77–88. https://doi.org/10.1038/nrn.2015.18. Reviews candidate molecular and cellular mechanisms for spaced learning, including signaling dynamics, dendritic spine remodeling, and transcription.
4. Born, J., & Wilhelm, I. (2012). “System consolidation of memory during sleep.” *Psychological Research*, 76, 192–203. https://doi.org/10.1007/s00426-011-0335-6. Reviews the role of sleep in systems consolidation.
5. “Sleep—A brain-state serving systems memory consolidation.” *Neuron* (2023). https://doi.org/10.1016/j.neuron.2023.03.005. Reviews replay and coordinated activity during sleep and wakefulness.
6. Lutz, N. D., Harkotte, M., & Born, J. “Sleep’s contribution to memory formation.” *Physiological Reviews* (final version listed in the journal as Volume 106, Issue 1). https://doi.org/10.1152/physrev.00054.2024. Reviews active systems consolidation, sleep-associated reactivation, synaptic processes, and open questions.
7. Camp, G. (2026). “A Meta-Analysis of the Effectiveness and Boundary Conditions of Retrieval Practice, Distributed Practice, and Interleaved Practice in Authentic School Classrooms.” *Educational Psychology Review*. https://doi.org/10.1007/s10648-026-10225-5. Reports average effects and moderators in classroom studies, emphasizing context-dependent outcomes.

---

**Modeling note:** The equations and pseudocode in this document are proposed computational tools for constructing a brain-inspired learning model. They are not direct measurements of neural computation and should be tested against behavioral and, where relevant, neuroscientific evidence before being treated as explanatory mechanisms.
