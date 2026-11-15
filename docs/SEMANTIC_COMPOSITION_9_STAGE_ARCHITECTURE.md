# ORE Sentence-Level Semantic Composition — nine stages / 54 operations

**Package:** `ore_semantic_composition/`  
**Registry:** `ORE-SEMANTIC-COMPOSITION-54`  
**Version:** 1.0.0  
**Status:** Executable research prototype with an optional current-based LIF reference track.

## 1. Purpose and boundaries

This package takes an ordered sentence (plus optional upstream syntax and lexical evidence) and builds an explicit semantic representation. It implements the nine requested stages:

1. Retrieve individual word concepts
2. Combine word meanings
3. Predicate-argument composition
4. Assign thematic roles
5. Construct propositions
6. Represent negation and modality
7. Represent quantifiers and their scope
8. Represent cause and effect
9. Generate sentence-level semantic representations

It is additive. `ore_semantic` continues to retrieve and select individual word concepts; `ore_syntax` continues to provide POS, phrase, dependency, and predicate-argument structure. This package connects those outputs and continues beyond individual concept retrieval to proposition and sentence meaning. It does not replace either upstream package.

A reviewed concept inventory is required for meaningful word-concept retrieval. Predicate valency frames are optional but recommended. Without them, the processor preserves syntax-based candidate roles and labels fallback assumptions. It never silently invents a missing participant or causal edge.

The registry contains **54 stable operations**, `SMC-001` through `SMC-054`, with exactly six operations in each stage. Each trace row includes the operation ID, description, mathematical form, evidence, output summary, confidence, and biological-reference state.

## 2. Stage and operation registry

| Stage | Operations | Functional work |
|---|---|---|
| 1. Retrieve individual word concepts | SMC-001–006 | Normalize word cues; retrieve exact/near lexical matches; apply upstream hypotheses; retrieve features; compare context; compute concept posteriors |
| 2. Combine word meanings | SMC-007–012 | Initialize sentence workspace; project concepts into token nodes; modulate with context; pool shared features; compute pairwise associations; preserve token-local distinctions |
| 3. Predicate-argument composition | SMC-013–018 | Identify predicates; retrieve valency frames; read syntactic dependents; instantiate argument slots; score compatibility; mark completeness or unresolved roles |
| 4. Assign thematic roles | SMC-019–024 | Generate frame roles; map dependency/voice evidence; add semantic selectional evidence; combine scores; normalize competition; select or abstain |
| 5. Construct propositions | SMC-025–030 | Create predicate propositions; bind entity references; attach role-bearing arguments; encode event attributes; validate; estimate representational support |
| 6. Represent negation and modality | SMC-031–036 | Detect negation; determine scope; apply polarity; detect modal cues; estimate modal force; compose scoped operators |
| 7. Represent quantifiers and their scope | SMC-037–042 | Detect quantifiers; locate nominal domains; create variables; enumerate bounded scope orders; interpret operators; preserve ambiguity |
| 8. Represent cause and effect | SMC-043–048 | Detect explicit causal markers; align event clauses; orient cause/effect; add temporal evidence separately; score support; construct causal edges |
| 9. Generate sentence-level semantic representations | SMC-049–054 | Create typed graph nodes; create typed edges; integrate sentence structure; estimate uncertainty; validate references; serialize the full result |

The exact names, inputs, outputs, equations, and biological-reference notes are available programmatically:

```python
from ore_semantic_composition import OPERATION_REGISTRY, STAGE_REGISTRY, registry_as_dicts

print(len(OPERATION_REGISTRY))  # 54
print(len(STAGE_REGISTRY))      # 9
for operation in OPERATION_REGISTRY:
    print(operation.operation_id, operation.name, operation.mathematical_form)
```

## 3. Functional computations

### 3.1 Word-concept retrieval

For each normalized token (w_i), exact label/alias matches are preferred. Near matches are retained only above the configured lexical-similarity floor. Optional `concept_hypotheses` restrict candidates to concept IDs proposed by an upstream component.

A candidate score is a transparent weighted combination:

[
S(c_i)=w_l L(c_i,w_i)+w_x X(c_i,mathbf{x})+w_f F(c_i)
]

where (L) is lexical similarity, (X) is cosine similarity between the candidate feature map and caller-supplied context, and (F) is a bounded log-frequency prior. Candidate posteriors use a numerically stable softmax:

[
P(c_imid w_i,mathbf{x})=
rac{exp(S(c_i)/T)}{sum_jexp(S(c_j)/T)}
]

The output keeps candidate probabilities, lexical similarity, context fit, features, lexical category, and source references separate. Confidence also accounts for the top candidate's lexical similarity, so a lone weak fuzzy match does not automatically become a certain concept. Low confidence or a small top-two margin causes abstention.

### 3.2 Sentence feature composition

Each word contributes a posterior-weighted sparse feature vector:

[
mathbf{h}_i=sum_c P(cmid w_i)mathbf{f}_c
]

Context features modulate a separate copy of each token vector. The sentence-level feature summary is a bounded weighted pool:

[
mathbf{H}=
operatorname{clip}left(
rac{sum_ialpha_imathbf{h}'_i}{sum_ialpha_i+epsilon},0,1
ight)
]

Pairwise token association is computed as:

[
A_{ij}=rac{mathbf{h}'_icdotmathbf{h}'_j}
{|mathbf{h}'_i||mathbf{h}'_j|+epsilon}
]

The processor retains token-indexed semantic nodes and the association matrix. It does not reduce the sentence to an unordered bag of words; downstream predicate and argument bindings preserve composition structure.

### 3.3 Predicate-argument composition and thematic roles

A predicate frame states the roles licensed by a predicate, required roles, and optional role-specific semantic feature preferences. Dependency relations provide syntactic evidence (for example, subject, object, indirect object, oblique, or passive-agent relations).

For participant (a), predicate (v), and role (r):

[
S(a,rmid v)=w_s E_{syn}(a,r)+w_m
operatorname{cosine}(mathbf{f}_a,mathbf{p}_r)
]

Role candidates are normalized with softmax. A role is selected only when both its confidence and its margin over the second candidate pass configured thresholds. Otherwise, the full ranked alternatives remain in the output.

The syntax-based role mapping is deliberately conservative and English-oriented. It is not a general semantic-role labeler and does not guarantee that Agent/Patient labels are correct without appropriate syntax, lexical frames, and evaluated role evidence.

### 3.4 Propositions, negation, and modality

Each supported predicate creates a proposition shell. The processor binds participant entity references, role assignments, and available event attributes (tense, aspect, and voice). Missing required arguments are reported rather than filled by assumption.

Negation is represented as a scoped operator. For a proposition with (k) scoped negative cues, polarity is represented by parity:

[
polarity(p)=
egin{cases}
negative,& kmod 2=1\
positive,& kmod 2=0
end{cases}
]

Modal markers are mapped to transparent labels and force scores (for example, possibility, necessity/obligation, ability, or prediction). These scores encode the prototype's lexical mapping, not calibrated probabilities of truth. Modal and negative operators remain attached to the target proposition, with scope warnings where the current syntax does not provide a fully resolved scope.

### 3.5 Quantifiers and scope

Quantifier cues are linked to nominal heads using dependency determiner arcs when available, otherwise a marked nearest-following-nominal fallback. The system creates a variable, a restrictor sketch, and a symbolic operator. Examples include:

[
llbracket everyrbracket=orall,quad
llbracket somerbracket=exists,quad
llbracket norbracket=
egexists
]

When several quantifiers occur, the processor enumerates bounded scope-order alternatives (up to `max_scope_readings`). It preserves all generated readings rather than claiming to have resolved scope from token order alone. The predicate (P) in a formula is a placeholder when the available syntax does not establish a complete logical form.

### 3.6 Cause and effect

The causal stage recognizes explicit connectives (such as because, therefore, hence, and thus) and a small set of causative predicate forms. It aligns candidate event propositions, uses cue-specific direction rules, and reports temporal order separately.

A relation confidence is an interpretable weighted support score:

[
C_{cause}=0.60C_{cue}+0.25C_{structure}+0.15C_{time}
]

Temporal succession alone never creates a causal relation. Relations without two explicit proposition endpoints can still be returned as argument-level candidates, but they are not converted into proposition-to-proposition graph edges.

### 3.7 Sentence-level graph and uncertainty

The final representation is a typed graph containing word-concept nodes, propositions, quantifiers, and edges for argument binding, thematic roles, operator scope, quantification, and causality. It retains the proposition list and the full operator/causal structures alongside the graph.

Uncertainty diagnostics include concept entropy, role entropy, unresolved concept count, role abstentions, and quantifier-scope ambiguity. These are indicators of ambiguity within the implemented candidate sets, not calibrated measures of a person's comprehension.

## 4. Biological-reference track

A separate nine-population current-based leaky integrate-and-fire (LIF) reference is available. Its populations correspond functionally to the nine stages: concept retrieval, semantic binding, predicate-argument processing, thematic roles, proposition construction, negation/modality, quantifier scope, causal integration, and sentence integration.

The membrane update is a forward-Euler approximation:

[
V_i(t+Delta t)=V_i(t)+rac{Delta t}{	au_m}
left(V_{rest}-V_i(t)+10^{-3}R_m I_i(t)ight)
]

The current proxy combines an external functional drive, recurrent gain, and population-level inhibition. A threshold crossing produces a spike and voltage reset.

Reported units are kept explicit:

- Membrane voltage: **mV**
- Synaptic current: **pA**
- Firing rate: **Hz**
- Simulated time: **ms**
- Spike count: dimensionless count

This simulator is a configurable computational reference, not a fitted model of the ATL, IFG, hippocampus, or complete human semantic system. Its populations and stage routing are engineering hypotheses; they are not measured neural constants and do not establish biological validity.

Set `SemanticCompositionConfig(simulate_biology=False)` to skip this track while preserving the functional 54-operation trace.

## 5. Python API

### Create a concept inventory and predicate frame

The package accepts the existing `ore_semantic.SemanticConcept` records:

```python
from ore_semantic import SemanticConcept
from ore_semantic_composition import (
    PredicateFrame, SemanticCompositionProcessor,
    SemanticCompositionRequest, SemanticCompositionConfig,
)

concepts = [
    SemanticConcept(
        concept_id="scientist",
        label="scientist",
        semantic_features={"HUMAN": 1.0, "AGENTIVE": 0.9},
        lexical_category="noun",
        verified_by="reviewer",
        source_reference="dataset://concepts/scientist",
    ),
    SemanticConcept(
        concept_id="study",
        label="studies",
        aliases=("study",),
        semantic_features={"EVENT": 0.8, "COGNITION": 1.0},
        lexical_category="verb",
        verified_by="reviewer",
        source_reference="dataset://concepts/study",
    ),
    SemanticConcept(
        concept_id="neuron",
        label="neurons",
        aliases=("neuron",),
        semantic_features={"BIOLOGICAL_ENTITY": 1.0, "CELL": 0.9},
        lexical_category="noun",
        verified_by="reviewer",
        source_reference="dataset://concepts/neuron",
    ),
]

frames = [
    PredicateFrame(
        predicate="studies",
        predicate_concept_id="study",
        roles=("Agent", "Patient"),
        required_roles=("Agent", "Patient"),
        selectional_preferences={
            "Agent": {"HUMAN": 1.0, "AGENTIVE": 0.8},
            "Patient": {"BIOLOGICAL_ENTITY": 1.0},
        },
        source_reference="dataset://frames/studies",
    )
]

processor = SemanticCompositionProcessor(
    concepts,
    frames,
    SemanticCompositionConfig(simulate_biology=True),
)
result = processor.process(SemanticCompositionRequest(
    sequence_id="sample-001",
    tokens=("scientist", "studies", "neurons"),
))
print(len(result.operation_trace))   # 54
print(result.propositions)
print(result.sentence_representation["edges"])
print(result.diagnostics)
```

### Supply an upstream syntax result

```python
from ore_syntax import SyntaxProcessor, SyntaxRequest
from ore_semantic_composition import composition_request_from_results

syntax_result = SyntaxProcessor().process(SyntaxRequest(
    sequence_id="sample-001",
    text="The scientist studies neurons.",
))
request = composition_request_from_results(
    sequence_id="sample-001",
    syntax_result=syntax_result,
    # semantic_result=semantic_result,  # Optional ore_semantic output
)
result = processor.process(request)
```

The adapter transfers token order, dependency structure, and (when supplied) selected concept candidate IDs. The concept inventory still must be supplied by the caller; an adapter output is not a substitute for a reviewed knowledge base.

### Run the bundled demonstration

```bash
python -m pip install -e ".[dev]"
python -m ore_semantic_composition
python -m ore_semantic_composition --no-biology
python -m ore_semantic_composition --output semantic-composition-demo.json
python -m pytest tests/test_semantic_composition.py -q
```

## 6. SQLite annotation and execution logging

Every class and function in the new package has a plain-English Python docstring. The existing `ore_logger` AST catalogue indexes module, class, method, nested-function, and function definitions, taking those docstrings as their initial annotations. On an ORE script run, `ore_logger.runtime` synchronizes changed source definitions into the SQLite `annotations` table and records execution events with references to the stored annotations.

The default database remains:

```text
ore_logs/ore_logger.sqlite3
```

It is created locally when the repository logger starts; it is not committed as a source file. To catalogue the package immediately from the repository root, run:

```bash
python -m ore_semantic_composition
```

The run creates/updates its annotation entries and logs the run under the existing logger behavior. Manual annotation edits continue to be preserved by `ore_logger` when source definitions are refreshed.

## 7. Validation and scientific limitations

The tests cover registry integrity, argument-role binding, proposition assembly, negation, modality, quantifier domains, explicit causal markers, graph validity, and LIF output units. They establish software behavior on synthetic examples only.

A research-grade evaluation still requires:
- language-specific and provenance-tracked concept inventories;
- reviewed predicate valency and selectional-preference frames;
- held-out semantic-role and proposition datasets;
- minimal pairs for negation, modal scope, quantifier scope, and causal direction;
- calibration and abstention evaluation;
- independently collected, task-matched neural data before making biological-validity claims.

Passing the software tests does not demonstrate human-level comprehension, a complete theory of sentence meaning, or a biologically faithful model of the human brain.
