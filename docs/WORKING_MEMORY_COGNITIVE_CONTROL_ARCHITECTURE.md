# ORE working memory and cognitive control — five stages / 30 operations

**Package:** ore_working_memory/  
**Registry:** ORE-WMC-30  
**Operation IDs:** WMC-001 through WMC-030  
**Status:** Executable functional prototype with an optional coupled LIF population reference

## 1. Purpose and scientific boundary

This subsystem implements five language-processing and control functions:

1. Linguistic working-memory buffers.
2. Maintenance of incomplete syntactic structures.
3. Executive control over sentence interpretations.
4. Resolution of competing semantic interpretations.
5. Cognitive task switching.

The processor accepts a sentence or a token sequence, optional syntax output, explicit unfinished
structures, candidate interpretations, goal features, context cues, and a task sequence. It returns
the preserved buffer contents, open syntactic frames, candidate readings, posterior probabilities,
abstention state, task transitions, and a 30-operation trace.

This is a deterministic research prototype. The functional operations are computational models of
useful information-processing requirements; the biological track is an illustrative population-level
reference. Neither is a claim that the human brain uses the exact data structures or equations in
this package.

## 2. The 30 registered operations

| Stage | ID range | Operation sequence |
|---|---|---|
| S1 — Linguistic working-memory buffers | WMC-001–006 | Tokenize and encode; allocate modality-specific buffers; calculate activation; apply decay/capacity; rehearse active items; retrieve context cues |
| S2 — Maintain incomplete syntactic structures | WMC-007–012 | Extract incomplete frames; build hierarchical stack; track unresolved dependencies; score retrieval cues; bind supported slots; emit open-structure state |
| S3 — Executive control over sentence interpretations | WMC-013–018 | Prepare hypotheses; measure conflict; allocate control; evaluate goal fit; apply top-down bias; select temporary focus |
| S4 — Resolve competing semantic interpretations | WMC-019–024 | Integrate evidence; normalize posteriors; run recurrent competition; measure entropy/margin; apply abstention; build resolution report |
| S5 — Cognitive task switching | WMC-025–030 | Encode task sets; detect switches; calculate switch cost; reconfigure task control; manage interference/resumption; emit task state |

Each operation is implemented in ore_working_memory/pipeline.py, has a stable registry ID, records
its input/output summary in operation_trace, and advances one biological-reference operation when
simulation is enabled.

### Stage 1 — Linguistic working-memory buffers

Token input is converted into ordered records while preserving supplied token boundaries. The same
token references are then maintained in distinct functional channels: phonological-form proxy,
lexical items, syntactic markers, content-bearing semantic items, and discourse-relevant items.
These channels are separate software views; they are not asserted to be five literal anatomical buffers.

For token i at time t, a simple activation estimate is:

    a_i(t) = s_i * exp(-lambda * age_i)

where s_i is a supplied-focus salience multiplier, lambda is the configured decay rate, and age is
represented by token distance. Each buffer retains at most K active items:

    B'_m = TopK(B_m, activation_i, K)

Rehearsal refreshes only a small active set:

    a'_i = min(1, a_i + eta * r_i)

Context retrieval ranks items by activation and explicit cue match:

    R_i = alpha * a_i + (1-alpha) * cue_match_i

The values are software controls, not measured human memory constants or estimates of a person's
true working-memory capacity.

**Biological-reference interpretation:** phonological-loop activity, distributed temporal language
representations, inferior-parietal maintenance/attention, and prefrontal control are represented by
coupled population labels. Recurrent activity and cue-dependent reinstatement are mechanistic
analogies, not literal simulations of human rehearsal.

### Stage 2 — Maintaining incomplete syntactic structures

This stage accepts explicit partial phrase/clause frames or unresolved dependencies from an upstream
syntax processor. An open frame contains a stable ID, label, nesting depth, parent ID, expected slots,
filled slots, unresolved slots, and optional candidate bindings.

    U_f = E_f set_minus F_f

where E_f is the set of expected slots and F_f is the set already filled. Frames are ordered by depth
in a stack. Long-distance dependencies are retained only when upstream evidence marks them unresolved
or open.

A candidate c can fill slot s only if:

    q(c,s) = 0.6 * confidence(c) + 0.4 * slotMatch(c,s)
    bind(s,c) iff q(c,s) >= retrieval_threshold

The processor never generates a missing word to close a frame. If a candidate does not meet the
configured threshold, the slot remains unresolved in the result.

**Biological-reference interpretation:** posterior temporal syntactic representations, inferior
frontal language/control mechanisms, and parietal working-memory/retrieval processes are used as a
broad functional hypothesis for maintaining expectations across intervening material. A software
stack is not claimed to exist literally in cortex.

### Stage 3 — Executive control over sentence interpretations

Candidate readings remain distinct records. Conflict combines normalized posterior entropy and the
margin between the top two preliminary readings:

    C = 0.5 * H(p)/log(k) + 0.5 * (1 - (p1-p2))

Control demand combines conflict, unresolved syntax load, and explicit goal priority:

    D = clip(0.45*C + 0.35*U + 0.20*G, 0, 1)

Candidate features are compared with supplied goal features. The top-down bias is stored separately
from lexical, semantic, syntactic, discourse, and world evidence:

    b(h) = gain * D * (0.65*goalFit(h) + 0.35*prior(h))

The control-focus operation selects a temporary processing focus but does not delete other hypotheses.
This separation prevents a control preference from being mistaken for independent semantic evidence.

**Biological-reference interpretation:** lateral prefrontal control, anterior cingulate conflict
monitoring, and frontoparietal attention are broad functional analogies. The implementation does not
estimate neural activity from an actual person.

### Stage 4 — Resolving competing semantic interpretations

For hypothesis h, the initial integrated score is:

    z_h = 0.20*L_h + 0.28*S_h + 0.20*Y_h + 0.16*D_h + 0.16*W_h + b(h) + 0.05*P_h

where L is lexical evidence, S semantic evidence, Y syntactic evidence, D discourse evidence,
W supplied world/context evidence, and P the prior. Candidate scores are transformed into a posterior:

    p(h) = softmax(z_h / T)

A damped recurrent update adds a simple lateral-competition term:

    p_(t+1) = (1-rho)*p_t + rho*softmax(z/T - gamma*(1-p_t))

Normalized entropy and top-two margin quantify uncertainty:

    H_n = -sum(p_i*log(p_i+epsilon))/log(k)
    m = p_1 - p_2

The processor commits only if both the maximum posterior and the margin exceed their configured
thresholds. Otherwise, it returns abstention plus the ranked alternatives, evidence channels, and
convergence trace. One candidate has zero normalized entropy and a margin of one by definition.

**Biological-reference interpretation:** distributed temporal semantic representations, inferior
frontal selection, recurrent excitation/inhibition, and conflict monitoring provide a broad functional
analogy. The softmax distribution and entropy are software variables, not direct measurements of cortical firing.

### Stage 5 — Cognitive task switching

The task stage reads an explicit task sequence or a single current task from context. A switch occurs
when the task label changes. Switch cost is an estimate:

    C_switch = C_0 * (1 + inertia*similarity) * (1-cue_strength)

where C_0 is a configured baseline cost, inertia is task inertia, similarity is a coarse task-family
similarity, and cue_strength is the supplied cue strength. The returned unit is milliseconds for
convenient interpretation, but the output is not a human reaction-time prediction.

Task-specific control profiles redistribute software attention weights over phonological, lexical,
syntactic, semantic, and executive channels. Suspended task records are retained, and resumption is
reported when a prior task reappears later in the sequence.

**Biological-reference interpretation:** frontoparietal control, pre-supplementary motor/cingulo-
opercular task-set updating, and basal-ganglia gating are used as a deliberately coarse functional
analogy. No anatomical localization is established by these software outputs.

## 3. Biological-reference simulator

When WorkingMemoryConfig(simulate_biology=True), the simulator advances a small set of coupled
current-based leaky integrate-and-fire populations once per WMC operation:

- phonological_loop
- lexical_maintenance
- syntactic_workspace
- semantic_workspace
- parietal_attention
- prefrontal_control
- conflict_monitor
- task_set_network

A forward-Euler membrane update is used:

    V(t+dt) = V(t) + dt/tau_m * (V_rest - V(t) + 1e-3*R_m*I(t))

Threshold crossing increments a spike count, resets voltage, and starts a refractory interval.
Recurrent excitation and global inhibition couple the population activities. The simulator reports
membrane voltage (mV), current (pA), simulated time (ms), firing rate (Hz), spike count, and population
activity.

Population names, routing weights, and parameters are engineering hypotheses. This implementation
is not a neuron-by-neuron simulation, not a fitted human language circuit, and not evidence that the
five software stages map one-to-one onto discrete brain regions. Passing software tests establishes
code behavior only, not biological validity.

Disable the simulator with:

    WorkingMemoryProcessor(WorkingMemoryConfig(simulate_biology=False))

## 4. API

    from ore_working_memory import WorkingMemoryProcessor, WorkingMemoryRequest

    request = WorkingMemoryRequest(
        sequence_id="ambiguous-001",
        text="I saw her duck.",
        interpretations=[
            {
                "hypothesis_id": "bird",
                "interpretation": "I saw the duck belonging to her",
                "lexical_score": 0.8,
                "semantic_score": 0.75,
                "syntactic_score": 0.7,
            },
            {
                "hypothesis_id": "action",
                "interpretation": "I saw her lower her head",
                "lexical_score": 0.78,
                "semantic_score": 0.76,
                "syntactic_score": 0.72,
            },
        ],
        task_sequence=["reading", "parsing", "semantic_resolution"],
    )
    result = WorkingMemoryProcessor().process(request)
    print(result.semantic_resolution["alternatives"])
    print(result.incomplete_structures)
    print(result.task_switching)
    print(len(result.operation_trace))  # 30

To maintain an unfinished structure, provide explicit evidence:

    request = WorkingMemoryRequest(
        sequence_id="open-clause-001",
        text="The report that the analyst reviewed",
        partial_structures=[
            {
                "structure_id": "relative-1",
                "label": "relative clause",
                "expected_slots": ["object"],
            }
        ],
    )

The slot remains open until an explicit candidate binding matches it and clears the retrieval threshold.
An upstream syntax result can be supplied using syntax_result; unresolved_dependencies,
incomplete_structures, and explicitly unresolved long_distance_dependencies are read when present.

## 5. SQLite annotation logging

Every class and function in ore_working_memory has a plain-English Python docstring. The existing
ORE AST catalogue indexes those docstrings into the annotations table of
ore_logs/ore_logger.sqlite3. The automatic runtime logger links calls, returns, and exceptions to
the annotation rows when an ORE entry point is launched with auto-logging enabled. Set ORE_AUTO_LOG=0
to disable automatic tracing. The SQLite file is runtime output and is intentionally not committed.

The test suite also invokes sync_repository against a temporary SQLite database and checks that
class and method annotations were stored with annotation_source='docstring'.

## 6. Validation and limitations

The software tests cover registry order, all 30 trace rows, one neural-reference update per operation,
bounded modality-specific buffers, activation and rehearsal, explicit incomplete frames, supported-only
slot binding, unresolved long-distance dependency preservation, semantic alternatives and abstention,
goal-biased executive focus, task switches and resumption, JSON serialization, docstring completeness,
and SQLite annotation persistence.

The implementation is English-oriented and uses deterministic heuristics. It does not include an
acoustic phonological encoder, a trained language parser, an episodic-memory database, a learned
executive controller, or a validated human task-switching model. Behavioral validity requires held-out
language/task datasets, independent labels, calibration, and baseline comparisons. Biological validity
requires task-matched neural data, preregistered predictions, and independent replication.
