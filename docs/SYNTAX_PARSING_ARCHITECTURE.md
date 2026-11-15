# ORE Syntax Parsing — Nine-Stage Architecture

**Package:** `ore_syntax/`  
**Registry:** ORE-SYNTAX-56  
**Status:** Executable rule-based research prototype; language coverage and biological correspondence require independent validation.

## Scope and integration

The syntax processor consumes caller-provided text or tokenized word forms. It can also consume token-indexed lexical category distributions from `ore_lexical_access` through `SyntaxRequest.lexical_evidence`, plus semantic feature vectors from upstream semantic processing through `SyntaxRequest.semantic_features`. Inventories are read-only. This package does not alter N1–N9, lexical access, semantic cognition, or integrated reading.

Two separate tracks are exposed:

1. **Functional track:** POS candidate inference, phrase spans, dependency arcs, a weighted CKY-style constituency chart, a nested hierarchy, predicate argument frames, thematic-role distributions, recurrent ambiguity updates, and enhanced filler-gap links.
2. **Biological reference track:** nine coupled current-based leaky-integrate-and-fire populations with membrane leak, recurrent input, global inhibition, threshold, reset, refractory time, spike counts and rates.

This is a compact English-oriented implementation. It does not use a pretrained language model, a large hand-curated grammar, a treebank-trained parser or a language-specific morphology engine. The parser preserves uncertainty and exposes its rules so they can be expanded or replaced.

## Ordered algorithm

### Stage 1 — Part-of-speech identification (SYN-001–SYN-007)

1. Normalize the lexical key with Unicode NFC and case folding while retaining the source string.
2. Retrieve caller-supplied lexical category evidence and optional inventory labels.
3. Generate candidate categories from the closed-class vocabulary, ambiguous-word inventory, suffix cues, capitalization and lexical records.
4. Estimate contextual evidence from nearby function words and category patterns.
5. Combine evidence in log space.
6. Normalize candidate scores with temperature-scaled softmax:
   `P_i(c) = exp(s_i(c)/T) / sum_k exp(s_i(k)/T)`.
7. Preserve alternatives and report confidence, margin and abstention.

The default vocabulary is a demonstration inventory, not a general-purpose English lexicon. A real ORE deployment should pass the lexical candidates already produced upstream rather than relying on the compact defaults.

### Stage 2 — Phrase detection (SYN-008–SYN-013)

1. Mark boundary cues from determiners, adpositions, conjunctions, particles and punctuation.
2. Detect nominal heads.
3. Expand noun phrases over determiners, adjectives, numerals and compound nouns.
4. Assemble auxiliary/verb/adverb/particle predicate chains into VP spans.
5. Detect PP, ADJP and ADVP spans.
6. Score spans from head confidence and compatible boundaries; preserve overlapping hypotheses where useful.

Phrase detection is a fast span proposal stage. The later constituency chart performs a separate hierarchical analysis.

### Stage 3 — Dependency parsing (SYN-014–SYN-020)

1. Generate possible governors for each token.
2. Extract distance, direction, POS, boundary and phrase-membership features.
3. Propose relation labels such as `nsubj`, `obj`, `det`, `amod`, `aux`, `case`, `advmod`, `obl:agent` and `xcomp`.
4. Score each candidate arc:
   `score(i->j,r) = w_r^T f(i,j) - lambda*distance`.
5. Choose a predicate-like root, with a content-word fallback.
6. Choose one governor per non-root token and repair cycles.
7. Validate index bounds, one root and one head per other token.

The present arc selector is a transparent greedy approximation with deterministic cycle repair, not a trained graph-based dependency parser or exact maximum spanning arborescence implementation.

### Stage 4 — Constituency parsing (SYN-021–SYN-026)

1. Convert POS alternatives into lexical chart leaves.
2. Initialize all contiguous span cells.
3. Apply explicit weighted binary phrase-structure rules.
4. Apply bounded unary closure without recursive cycles.
5. Rank and return up to `max_parse_alternatives` full parses.
6. If the grammar cannot cover the complete input, emit a `FRAG` tree rather than silently dropping tokens.

The chart is CKY-style dynamic programming over the supplied compact grammar. Parse scores combine POS log posteriors and log rule weights; these are not corpus-calibrated PCFG probabilities.

### Stage 5 — Hierarchical syntax representation (SYN-027–SYN-031)

1. Instantiate typed nodes with spans, labels, heads and children.
2. Store immediate parent-child dominance and node depth.
3. Align constituency leaves with dependency heads and relations.
4. Attach phrase and clause span annotations.
5. Serialize the result with stable node identifiers and token provenance.

The result deliberately preserves constituency and dependency views rather than collapsing them into one structure.

### Stage 6 — Subject/object assignment (SYN-032–SYN-037)

1. Identify lexical predicate centres.
2. Collect candidate nominal dependents and local NP heads.
3. Assign subjects using dependency labels and local order.
4. Assign direct/indirect objects and oblique arguments.
5. Estimate active/passive voice from auxiliary and participle cues.
6. Build predicate-centred argument frames.

If no dependency relation is available, a local-order fallback is explicitly marked in the argument record. Passive voice uses a deliberately small auxiliary/participle detector and will not cover every passive construction.

### Stage 7 — Agent/patient thematic roles (SYN-038–SYN-043)

1. Retrieve a compact verb-frame expectation, overridable by the caller.
2. Retrieve semantic features supplied for each argument token.
3. Compute feature compatibility using cosine similarity:
   `compat(x,F) = (x dot F)/(||x|| ||F|| + epsilon)`.
4. Combine syntactic/voice mapping with semantic compatibility.
5. Normalize candidate roles with softmax.
6. Return selected roles, alternatives and warnings where semantic evidence is absent.

When the caller provides no semantic features, agent/patient labels are primarily syntactic and voice-based. They must not be mistaken for independently verified world knowledge.

### Stage 8 — Syntactic ambiguity resolution (SYN-044–SYN-050)

1. Collect close POS/structural hypotheses.
2. Score syntactic compatibility from local constraints and the current argument structure.
3. Score semantic evidence only when caller features are available.
4. Apply a bounded recurrent belief update:
   `q^(t+1) = (1-rho)q^t + rho*softmax((log(q^t)+g_s*S_syn+g_m*S_sem)/T)`.
5. Calculate posterior margins and entropy.
6. Select only when confidence and margin exceed configured thresholds; otherwise abstain.
7. Preserve ranked alternatives, convergence deltas and changed POS decisions.

When POS decisions change, the package reconciles phrases, dependencies, constituency and argument/role outputs with the updated tags. Original and final POS labels remain available in the ambiguity record.

### Stage 9 — Long-distance dependency processing (SYN-051–SYN-056)

1. Detect wh-fillers and relative pronouns.
2. Identify possible clause boundaries and predicate sequences.
3. Generate unsatisfied argument slots from predicate frames and compact valency cues.
4. Score filler-gap pairs using role fit, valency, intervening nominal evidence and distance:
   `score(f,g) = role_fit + valency + clause_fit - lambda*distance`.
5. Add an enhanced remote dependency from filler to the best-supported predicate.
6. Validate the link and preserve unresolved cases.

Remote filler-gap links are stored separately from the basic dependency tree, so the tree remains a single-head rooted structure. This is a bounded heuristic, not a full movement theory or a general unbounded dependency solver.

## Biological reference populations

| Population | Computational role |
|---|---|
| `orthographic_input` | Incoming word-form drive |
| `lexical_pos` | Category candidate activity |
| `temporal_syntax` | Lexical and structural integration |
| `frontal_control` | Selection and ambiguity control |
| `phrase_binding` | Local and hierarchical phrase binding |
| `dependency_relations` | Head/relation competition |
| `thematic_roles` | Event participant-role activity |
| `working_memory` | Maintained filler and unresolved argument cues |
| `prediction_error` | Mismatch/uncertainty monitoring |

For each population, the simulator advances a current-based LIF approximation:

`V_i(t+dt) = V_i(t) + (dt/tau_m) * (V_rest - V_i(t) + 10^-3 R_m I_i[pA])`.

Threshold crossing increments a spike count, resets voltage and starts a refractory interval. Rates are reported in Hz; voltage in mV; current in pA; time in ms. This is a small deterministic computational reference, not a biologically complete cortical circuit, fitted model or demonstration that the brain literally executes these operations.

## Python API

```python
from ore_syntax import SyntaxConfig, SyntaxProcessor, SyntaxRequest

processor = SyntaxProcessor(config=SyntaxConfig(simulate_biology=True))
result = processor.process(SyntaxRequest(
    sequence_id="sample-001",
    text="The cat eats fish.",
    language="en",
    semantic_features={
        1: {"ANIMATE": 1.0},
        3: {"FOOD": 1.0},
    },
))

print(result.selected_pos)
print(result.dependency_arcs)
print(result.constituency_tree)
print(result.predicate_arguments)
print(result.thematic_roles)
print(result.operation_trace)  # 56 individually numbered operations
print(result.diagnostics)
```

To bridge lexical access, pass per-token grammatical evidence:

```python
request = SyntaxRequest(
    sequence_id="from-lexical-access",
    tokens=("The", "quokka", "runs", "."),
    lexical_evidence={1: {"NOUN": 0.99, "VERB": 0.01}},
)
```

Run the demonstration with `python -m ore_syntax`. Run tests with `python -m pytest tests/test_syntax.py -q`, then the complete repository suite with `python -m pytest`.

## Validation and limitations

Unit tests establish interface and numerical behavior for the included synthetic cases. They do not establish psycholinguistic validity or neural correspondence. A research-grade evaluation needs language-specific lexicons and grammars, held-out treebank parsing (UAS/LAS and constituency F1), labelled argument-role data, ambiguity-focused minimal pairs, filler-gap acceptability/reading-time tasks, calibration and independent neural measurements.
