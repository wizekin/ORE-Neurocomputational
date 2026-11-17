# Language comprehension — Point 4: Semantic composition

Point 4 builds a sentence-level proposition from the lexical-semantic candidates delivered by Point 2 and the grammatical relations delivered by Point 3. It transforms a collection of recognized concepts into a relational meaning representation. For example, it aims to preserve the distinction between:

```text
CHASE(agent = DOG, patient = CAT)
CHASE(agent = CAT, patient = DOG)
```

The same concepts can form different propositions when their semantic roles are different.

## Implementation strategy

The repository already contains `ore_semantic_composition`, a nine-stage, 54-operation processor. Point 4 reuses that implementation instead of creating a duplicate semantic engine. The adapter adds the source's 14 ordered steps, maps its lower-level operation traces, creates modifier/property edges from Point 3 dependency relations, records semantic compatibility, and recomputes a partial proposition state for each growing word prefix.

The integration expects a Point 3 result with its ordered tokens, POS decisions, dependency relations, incremental prefix snapshots, and lexical records. The Point 3 records normally embed the Point 2 lexical candidates. When those candidates contain sense-specific feature maps, Point 4 creates distinct concept-inventory records from them. It does not invent missing semantic features.

## The 14 source-level operations

| Step | Operation | Main output |
|---|---|---|
| 1 | Receive lexical-semantic candidates and syntax | Token-aligned lexical, semantic, syntactic, and morphology evidence |
| 2 | Activate relevant semantic features | Sense-specific concept features and a semantic workspace |
| 3 | Identify semantic heads | Event/state predicates, entity heads, property heads, and relational heads |
| 4 | Create or update entity/event variables | Sentence-local entity and event references |
| 5 | Map grammatical functions to semantic roles | Candidate Agent, Patient, Experiencer, Stimulus, Recipient, Theme, and other roles |
| 6 | Bind arguments to predicates | Proposition shells with role/filler links and open slots |
| 7 | Compose modifiers and properties | Adjective, adverb, possessive, compound, and nominal-relation edges |
| 8 | Compose grammatical operators | Negation, modality, quantifier, tense/aspect, and scope records |
| 9 | Evaluate semantic compatibility | Selectional fit and possible literal semantic tensions |
| 10 | Preserve competing readings | Separate candidate propositions, sense alternatives, and role alternatives |
| 11 | Update composition incrementally | A partial semantic-state snapshot after each incoming word |
| 12 | Check proposition completeness | Filled, pending, optional, missing, and unspecified required roles |
| 13 | Consolidate the proposition graph | Typed nodes, semantic edges, confidence, provenance, and validation |
| 14 | Hand off to contextual/pragmatic integration | Literal proposition plus ambiguity and unresolved information |

## Mathematical operations

### Semantic feature activation

A lexical sense is represented by a sparse concept-feature vector:

`x = [x_1, x_2, \ldots, x_n]`

Feature gating gives:

`x_rel = g \odot x`

A population-style weighted activation can be expressed as:

`a_j = phi(sum_i w_ij x_i - theta_j)`

These equations are computational models of selective feature activation, not evidence that individual neurons execute those exact operations.

### Predicate–argument binding

A predicate frame lists roles and the selectional preferences attached to those roles. For a candidate filler `x` and role `r`, a model can combine grammatical and semantic support:

`S(r,x) = w_s S_syntax(r,x) + w_m S_semantic(r,x)`

Role competition is normalized with a softmax, while low-confidence or low-margin cases remain unresolved.

A completed argument relation can be written as:

`P(e) = CHASE(e)`
`AGENT(e, dog)`
`PATIENT(e, cat)`

This graph preserves the event-participant relationships instead of flattening all words into one pooled feature vector.

### Modifier composition

The adapter uses dependency labels such as `amod`, `advmod`, `compound`, `nmod`, and `poss` to connect modifiers to their syntactic targets. It preserves both the modifier and target feature maps, together with a bounded feature union and any shared feature intersection:

`F_union(k) = max(F_head(k), F_modifier(k))`
`F_shared(k) = min(F_head(k), F_modifier(k))`

These are transparent feature-map operations. The graph edge records the syntactic provenance and semantic relation; mere co-occurrence is not treated as composition.

### Operators and scope

The underlying semantic engine represents negation, modality, and quantifiers as explicit operators. Examples include:

`NOT(CHASE(agent=DOG, patient=CAT))`

`FOR_ALL(x, DOG(x), EXISTS(y, CAT(y) AND CHASE(x,y)))`

The latter is one schematic scope reading, not the only possible interpretation.

### Compatibility and uncertainty

Selectional compatibility can be modelled as:

`C(type(x), role_r) in [0,1]`

with a mismatch penalty:

`E_sem = sum_i lambda_i (1-C_i)`

Competing interpretations can be ranked using lexical, syntactic, and compositional evidence:

`P(M_k | E) proportional to P(E | M_k) P(M_k)`

The output preserves alternatives rather than blending incompatible readings into one representation.

### Incremental state

For each prefix of the sentence, Point 4 invokes the existing composition engine with the corresponding Point 3 syntax snapshot. It records newly created and revised propositions, active role slots, modifier links, and the partial state:

`M_t = U(M_(t-1), L_t, S_t)`

This implementation recomputes each prefix through the existing batch engine. It is an inspectable approximation of incremental/recurrent composition, not a persistent neural-state simulation.

## Biological-reference track

The adapter maps the underlying 54 semantic-composition operations into 14 source-level biological records. When enabled, the reused nine-population current-based leaky-integrate-and-fire reference reports activity, firing-rate summaries, voltage, current, and simulated time.

The populations are broad functional analogies spanning concept retrieval, semantic binding, predicate–argument composition, thematic roles, propositions, operator scope, causality, and sentence integration. The model is not anatomically fitted, is not a literal cellular simulation of language, and is not measured EEG/fMRI data. Regional contributions attributed to ATL, posterior/middle temporal cortex, LIFG, and angular gyrus remain distributed and task-sensitive; no single region is asserted to perform the whole algorithm.

## Quick start

```python
from ore_language_comprehension import Point4Request, Point4SemanticComposer
from ore_semantic_composition import PredicateFrame, SemanticCompositionConfig

# point3_result is the Point3Result (or its to_dict() mapping) from Point 3.
chase_frame = PredicateFrame(
    predicate="chase",
    roles=("Agent", "Patient"),
    required_roles=("Agent", "Patient"),
    selectional_preferences={
        "Agent": {"ANIMATE": 1.0},
        "Patient": {"ANIMATE": 0.9},
    },
    source_reference="lexicon://reviewed/chase-frame",
)

request = Point4Request.from_point3_result(
    "sentence-001",
    point3_result,
    predicate_frames=(chase_frame,),
)
result = Point4SemanticComposer(
    config=SemanticCompositionConfig(simulate_biology=False)
).process(request)

print(result.propositions)
print(result.proposition_graph["edges"])
print(len(result.functional_trace))  # 14
print(result.handoff["target"])      # Point 5 contextual/pragmatic integration
```

The request factory reads lexical candidates from the Point 3 handoff. When a Point 2 lexical record contains `meaning_candidates[*].features`, each supplied sense becomes a separate concept candidate with its entry ID, sense ID, feature map, frequency, and source reference. A caller-provided concept inventory can be supplied when the upstream result does not contain sufficient semantic features.

Predicate frames should be reviewed and supplied where possible. They can specify roles, required roles, and role-specific selectional preferences. The adapter also reads explicit frames embedded in Point 2 lexical properties. When no explicit frame is available, the underlying engine uses a marked generic fallback; that is not a universal argument structure for the word.

## Outputs

`Point4Result` exposes:

- `word_concepts` and the semantic workspace;
- semantic heads, entity variables, and event variables;
- predicate argument frames and thematic-role candidates;
- predicate propositions, modifiers, operators, quantifier scopes, and explicit causal relations;
- semantic-compatibility records and separate competing proposition readings;
- one incremental partial-state record per token prefix;
- required-role completeness and unresolved slot status;
- a validated proposition graph, 14 source-level functional traces, and 14 mapped biological-reference traces;
- a handoff bundle for contextual/pragmatic integration.

Upstream lexical confidence and sense priors remain attached as provenance. The reused 54-operation engine independently recalculates its local concept support from the concept inventory and its configured lexical, context, frequency, and syntax weights; it does not simply copy Point 2's posterior into the final proposition.

## SQLite annotations

On initialization, `register_point04_annotations` indexes class, function, and method docstrings/signatures from:
`ore_language_comprehension`, `ore_semantic_composition`, `ore_semantic`, and `ore_syntax`.

It writes into the existing `ore_logger.annotations` SQLite table, by default
`ore_logs/ore_logger.sqlite3`. The location can be overridden through `ORE_LOGGER_DB` or
`database_path`. No separate Point 4 database is created.

Run the focused acceptance suite with:

```bash
python -m pytest tests/test_language_comprehension_point04.py -q
```

## Scope and limitations

- Semantic features are only as grounded as the lexical records supplied by Point 2 or the caller.
- Correct role assignment depends on predicate-frame quality, grammatical construction, voice, and morphology. Subject is not a synonym for Agent.
- The compact underlying composition engine has English-oriented lexical and operator rules; a language label alone does not provide a language-specific semantic grammar.
- The modifier adapter uses Point 3 dependency relations and Point 2 feature maps. It does not infer unattested modifier meanings.
- Selectional mismatch is a literal compositional tension, not a claim that an utterance is impossible. Metaphor, personification, irony, and broader context are left to the next stage.
- Quantifier formulas and operator scope are model sketches; unresolved readings are retained.
- The output does not infer discourse history, physical context, hidden speaker intention, sarcasm, irony, or unstated causal explanations.

The next source point is contextual and pragmatic integration. Point 4 passes its literal proposition, alternatives, unresolved roles, and uncertainty to that stage without performing its work.
