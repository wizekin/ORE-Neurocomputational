# ORE working memory and cognitive control — five stages / 40 operations

**Package:** ore_working_memory/  
**Canonical source:** Uploaded Library document Brain Language Algorithm.md  
**Registry:** WMCC-001 through WMCC-040  
**Stage distribution:** 8 / 6 / 7 / 9 / 10  
**Status:** Executable functional prototype with a separate, optional coupled LIF population reference

## 1. Purpose and scientific boundary

The package implements the exact 40-operation sequence in the source document. It connects five interacting subsystems:

1. Linguistic working-memory buffers.
2. Maintaining incomplete syntactic structures.
3. Executive control over sentence interpretations.
4. Resolving competing semantic interpretations.
5. Cognitive task switching.

The ordering is an inspectable execution schedule. It does not claim that the brain executes these functions as a rigid serial pipeline; language, memory, attention, and control can interact recurrently and in parallel.

The functional implementation is evidence-driven. Supplied syntax, semantic knowledge, contextual features, task history, and new revision evidence remain distinguishable from neutral defaults and surface heuristics. It does not use a trained parser or invent missing constituents, world facts, or reference resolutions.

## 2. Canonical 40-operation registry

| Stage | Operation IDs | Exact operation sequence |
|---|---|---|
| **S1 — Linguistic working-memory buffers** | WMCC-001–008 (8) | Initialize cognitive processing environment; receive linguistic information; encode incoming linguistic representations; distribute representations across working-memory buffers; organize and chunk stored information; maintain activation; regulate memory capacity and attention; perform content-addressable retrieval |
| **S2 — Maintaining incomplete syntactic structures** | WMCC-009–014 (6) | Detect incomplete syntactic structures; generate grammatical expectations; maintain unresolved dependencies; maintain nested syntactic structures; update incomplete structures; maintain dependencies through working-memory updates |
| **S3 — Executive control over sentence interpretations** | WMCC-015–021 (7) | Maintain cognitive goals; monitor sentence-processing progress; detect competing processing demands; allocate attention; regulate competing representations; determine interpretation acceptance; coordinate processing subsystems |
| **S4 — Resolving competing semantic interpretations** | WMCC-022–030 (9) | Identify semantic ambiguity; generate alternative interpretations; retrieve supporting semantic knowledge; evaluate grammatical compatibility; evaluate contextual compatibility; calculate the overall evidence supporting each interpretation; allow competing interpretations to interact; select an interpretation or preserve multiple possibilities; revise an interpretation when new information arrives |
| **S5 — Cognitive task switching** | WMCC-031–040 (10) | Identify the currently active cognitive task; detect when a task switch is necessary; preserve the current task state before switching; activate the new cognitive task; reduce interference from the previous task; monitor the new task and determine when to return; resume suspended tasks; learn from task-switching history; integrate the cognitive state; evaluate whether processing is complete |

The registry stores each operation's responsibility, inputs, outputs, mathematical form, and qualified biological-reference note. The pipeline emits one trace row for each ID in this exact order. Helper calculations used inside an operation are deliberately not emitted as extra registry operations.

## 3. Stage implementation

### Stage 1 — Linguistic working-memory buffers (8 operations)

The processor receives the original text or caller-tokenized input and creates ordered token records. It distributes token references into five functional channels:

- phonological: phonological-form proxy and ordered input units.
- lexical_semantic: lexical item references.
- syntactic: supplied or surface-detected syntactic markers.
- event_discourse: content-bearing references for event and discourse processing.
- executive_goal: caller-supplied goal features.

The channels are software views with cross-references, not five anatomically isolated storage compartments. Chunks preserve their source token positions. Caller-defined chunks are marked as caller-supplied; automatically grouped chunks are marked as surface-contiguity groupings and are not treated as semantic propositions.

A simplified activation model is:

\[
A_i(t)=B_i+\sum_j W_jS_{ji}+\sum_r\rho_rR_{ir}
-\lambda_i\Delta t-\mu I_i
\]

where retrieval cues and refresh events can increase accessibility, elapsed time can reduce it, and similar active items can produce interference. Buffer limits are engineering resource constraints, not a universal number of words the human brain can store.

Capacity and attention are modeled with:

\[
\alpha_i=\frac{\exp(A_i/\tau)}{\sum_j\exp(A_j/\tau)},
\qquad \sum_i r_i\le K
\]

Content-addressable retrieval ranks representations from activation and explicit cue match:

\[
P(M_i\mid Q)=\operatorname{softmax}_i\left(\beta\,\operatorname{Match}(Q,M_i)\right)
\]

### Stage 2 — Maintaining incomplete syntactic structures (6 operations)

This stage consumes explicitly supplied partial structures and upstream syntax results. Each open frame retains its ID, label, depth, parent, expected slots, filled slots, unresolved slots, and supplied candidate bindings.

\[
U_t=\{v\in V:\operatorname{complete}(v)=0\}
\]

Nested frames are ordered in a depth-aware stack/agenda, while long-distance dependencies remain explicit. Grammatical expectations are created from declared grammar expectations and unresolved slot labels. Slot-binding candidates are not accepted unless their evidence reaches the configured retrieval threshold:

\[
q(c,s)=0.6\,\operatorname{confidence}(c)+0.4\,\operatorname{slotMatch}(c,s)
\]

\[
\operatorname{bind}(s,c)\iff q(c,s)\ge\theta_{\mathrm{retrieval}}
\]

The dependency-maintenance operation marks known referenced buffer items for reactivation and emits links when token positions were actually supplied. It does not guess a missing filler or gap location.

### Stage 3 — Executive control over sentence interpretations (7 operations)

The controller maintains a declared comprehension goal and monitors memory availability, syntactic completeness, dependency completion, reference status, semantic consistency, and contextual relevance.

An example processing-quality estimate is:

\[
Q_t=\sum_k w_k Z_{tk}
\]

where the monitored status vector can include syntax, memory availability, reference resolution, semantic consistency, interpretation confidence, and contextual relevance.

Conflict can be estimated through candidate entropy, the difference between competing activations, unresolved structure load, and overall processing demand:

\[
H(P)=-\sum_i P_i\log P_i,
\qquad
T_c=w_1H(P)+w_2U+w_3D
\]

Attention and top-down bias are separate from the source evidence. The executive controller then regulates preliminary candidate activations through goal-sensitive excitation and lateral inhibition, preserving all candidates. It makes a provisional accept/continue decision using confidence, top-two margin, and any explicitly declared blocking conflict. This decision is not final: Stage 4 performs the fuller evidence integration and can select, abstain, or revise when new evidence arrives. Coordination messages express requests among memory, syntax, executive control, semantic resolution, and task management without allowing one subsystem to overwrite another's state.

### Stage 4 — Resolving competing semantic interpretations (9 operations)

The semantic resolver identifies ambiguity from caller-supplied alternative readings, ambiguity annotations, and upstream structures. If no interpretation candidates are supplied, it creates one neutral surface-reading hypothesis rather than pretending to know its meaning.

For each candidate it keeps separate lexical, semantic, grammatical, contextual, knowledge, pragmatic, prior, and goal-bias channels. Semantic knowledge is accepted only from explicit candidate scores or context.semantic_knowledge. Grammar and contextual compatibility use supplied syntax scores, support maps, or context feature mappings; missing evidence remains identified as neutral or unavailable.

An illustrative evidence score is:

\[
S_i=w_gG_i+w_lL_i+w_cC_i+w_kK_i+w_pP_i
\]

Candidate activation is converted to a posterior distribution and passed through bounded recurrent competition. A reading is selected only when both confidence and top-two margin pass the configured thresholds:

\[
P^*=\arg\max_i P_i,
\qquad
P(P^*)\ge\theta_c,
\qquad
P(P^*)-P(P_2)\ge\theta_m
\]

Otherwise the result abstains and preserves ranked alternatives. New information supplied in context.revision_evidence can update named candidate channels and trigger a new posterior and selection pass. No candidate is silently discarded.

### Stage 5 — Cognitive task switching (10 operations)

The task manager reads the explicit task sequence or a declared current-task default. Switch decisions and estimated switch costs are retained separately from observed transition costs. Before switching, it creates checkpoints containing references to working-memory items, open syntactic structures, candidate IDs, token position, and the current goal.

Task profiles reconfigure the five buffer priorities. Interference reduces the priority of the previous task but does not delete its checkpoint. Completion is marked only when explicit task-completion information is supplied; otherwise it remains unknown. A suspended task can resume when it reappears in the task sequence and the resumption policy permits it.

Task-transition estimates are updated only from explicit observations in context.task_history or context.observed_transition_costs. Estimated switch latency is not misrepresented as an observed reaction time.

The final integration operation creates a joined, inspectable state. The final completion operation checks input-end status, syntactic completion, semantic readiness, reference status, and unresolved conflict. It can report complete, partial_or_unresolved, or pending_input.

## 4. Functional and biological execution tracks

The functional track executes all 40 operations and emits an operation trace with:

- Stable WMCC-001–WMCC-040 ID and exact registry name.
- Functional output summary.
- Mathematical form used as the operation's computational specification.
- Qualified biological-reference note.
- Neural-reference update from the corresponding stage population, when enabled.

The optional biological-reference track uses eight coupled current-based leaky integrate-and-fire populations:

phonological_loop, lexical_maintenance, syntactic_workspace, semantic_workspace, parietal_attention, prefrontal_control, conflict_monitor, and task_set_network.

It advances once per operation and reports membrane voltage (mV), current (pA), simulated time (ms), firing rate (Hz), spike counts, and activity. The five WMCC stages drive the most relevant populations, while recurrent excitatory and global inhibitory terms allow coupling.

This is a configurable population-level computational hypothesis. It is not a fitted anatomical network, a cellular model of language, or empirical proof that the brain implements the registry's equations.

## 5. API and example

    from ore_working_memory import WorkingMemoryProcessor, WorkingMemoryRequest

    request = WorkingMemoryRequest(
        sequence_id="wmcc-example-001",
        text="I saw the man with the telescope. He was holding it.",
        interpretations=[
            {
                "hypothesis_id": "instrument",
                "interpretation": "I used the telescope to see the man",
                "syntactic_score": 0.72,
                "semantic_score": 0.70,
            },
            {
                "hypothesis_id": "possession",
                "interpretation": "The man had the telescope",
                "syntactic_score": 0.70,
                "semantic_score": 0.72,
            },
        ],
        task_sequence=["reading", "reference_retrieval", "semantic_resolution"],
        context={
            "revision_evidence": {
                "possession": {"discourse_score": 0.95, "semantic_score": 0.9}
            }
        },
    )
    result = WorkingMemoryProcessor().process(request)
    print(result.semantic_resolution["alternatives"])
    print(result.task_switching["switch_events"])
    print(len(result.operation_trace))  # 40

Inputs for task completion, unresolved references, semantic knowledge, grammar support, and contextual features are optional and must be supplied explicitly when those channels are to contribute.

## 6. Logger and SQLite annotations

Every new class and function has a plain-English docstring. The repository's ORE logger indexes these docstrings into its SQLite annotation catalogue when a logged Python entry point runs.

The default local database is:

ore_logs/ore_logger.sqlite3

It is generated at runtime and intentionally ignored by Git. Operation trace rows also contain the registry descriptions and mathematical/biological notes needed to inspect an execution. The SQLite test indexes the repository into a temporary database and checks that the new class/function annotations are persisted with annotation_source='docstring'.

## 7. Verification

Run the focused suite and then all repository tests:

    python -m pytest tests/test_working_memory.py -q
    python -m pytest

Tests cover the exact 40-operation name/order, 8/6/7/9/10 stage allocation, one-to-one trace and LIF operation counts, bounded buffers, explicit syntactic binding, unresolved dependencies, ambiguity/abstention, goal bias, semantic knowledge and revision, task checkpoint/resumption, observed transition-cost learning, completion gating, JSON serialization, and SQLite docstring indexing.

Passing these tests demonstrates software behavior and trace integrity only. Biological correspondence and behavioral validity require independent datasets, preregistered predictions, calibrated comparisons, and ablation studies.
