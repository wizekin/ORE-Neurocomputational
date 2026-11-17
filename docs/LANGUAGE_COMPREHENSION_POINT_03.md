# Language comprehension — Point 3: Syntactic parsing

Point 3 converts the uncertain word-level lexical records from Point 2 into an incremental-style
grammatical representation. It returns part-of-speech alternatives, phrase constituents, head–
dependent relations, ranked constituent trees, agreement evidence, unresolved dependencies, and
confidence values. It does **not** compose the sentence proposition or expose thematic/event roles;
those belong to later stages.

## Implementation strategy

The package reuses the repository's existing `ore_syntax` engine instead of implementing a second
parser. That engine supplies nine functional stages and 58 traceable operations for category
identification, phrase detection, dependency parsing, constituency parsing, hierarchical
representation, grammatical subject/object relations, ambiguity resolution, and long-distance
dependencies. Point 3 maps those operations into the source document's 12 algorithmic steps
(`LC-P03-01`–`LC-P03-12`).

The adapter accepts one ordered lexical item per word. As each word arrives, it reparses the
available prefix and records the changing categories, phrase groupings, dependency links, parse
alternatives, active expectations, unresolved dependencies, and any category/attachment revisions.
This is an incremental-prefix approximation: the current prefix is recomputed through the existing
batch syntax engine, rather than preserving a mutable parser state inside that engine. The
intermediate prefixes run without LIF dynamics; the final prefix can run the biological reference
once. Each item can provide a surface form, candidate grammatical categories, morphology, and
inherited confidence. Use
`Point3Request.from_point2_results` when a preceding Point 2 result is available for each token;
alternatively, pass `lexical_items` directly.

## The 12 source-level operations

| Step | Operation | Functional result |
|---|---|---|
| 1 | Receive and align lexical input | Ordered token records and inherited confidence |
| 2 | Estimate grammatical category | Ranked POS alternatives |
| 3 | Generate next-category expectations | Local continuation expectations and syntactic surprisal |
| 4 | Extend phrase constituents | NP/VP/PP and other phrase groupings, plus constituency alternatives |
| 5 | Identify heads and dependencies | Directed grammatical head–dependent relations |
| 6 | Check morphosyntactic agreement | Feature compatibility and soft mismatch penalties from supplied features |
| 7 | Combine grammatical cues | Word order, function words, case/morphology, punctuation, and optional supplied prosody |
| 8 | Maintain unresolved dependencies | Long-distance links and phrase-closure candidates |
| 9 | Rank structural alternatives | Normalized relative support for constituent parses |
| 10 | Detect structural conflict | Category surprisal, weak support, and explicit agreement mismatches |
| 11 | Reanalyse and repair | Revisions made by the underlying syntax engine's recurrent ambiguity pass |
| 12 | Consolidate syntactic output | A sentence-level grammatical frame passed to Point 4 |

## Mathematical operations

Category alternatives are represented as graded evidence. The syntax engine combines lexical and
local grammatical cues and normalizes candidate scores:

`P(c\mid evidence) = exp(s_c/T) / sum_k exp(s_k/T)`

For an explicitly supplied head/dependent feature pair, agreement compatibility is in `[0,1]`.
The adapter applies a soft penalty:

`E_agreement = sum_r lambda_r * mismatch_r`

`adjusted_edge_score = original_edge_score - E_agreement`

Missing features are treated as unknown, not as mismatches. This is important because a lexical
inventory may omit inflectional features, and some languages do not overtly mark every feature.

The adapter estimates category surprisal as:

`Surprisal_i = -log2 P(category_i | current evidence)`

and normalizes ranked constituency alternatives with a stable softmax. These are engineering
scores; they are not calibrated human-processing probabilities or claims about literal cortical
arithmetic.

Unresolved dependency availability may be modelled as:

`A_d(t) = A_0 * exp(-lambda * Delta_t) * CueMatch(d,t)`

This is an explanatory memory-accessibility model. The code exposes the syntax engine's actual
long-distance dependency records rather than asserting that neurons implement this exact equation.

## Biological-reference track

When `SyntaxConfig(simulate_biology=True)` is enabled, the existing nine-population current-based
leaky-integrate-and-fire network runs alongside the 58 syntax operations. The adapter aggregates
the corresponding operation outputs into 12 source-level biological records. It reports simulated
time, network activity, active populations, and end-of-parse population summaries.

This is a generic computational reference. It is not anatomically fitted, not a simulation of
individual human neurons, and not measured EEG/fMRI data. Neural descriptions in the step registry
are candidate functional analogies, not definitive localization claims.

## Quick start

```python
from ore_language_comprehension import Point3Request, Point3SyntacticParser
from ore_syntax import SyntaxConfig

lexical_items = [
    {"form": "The", "category_candidates": {"DET": 0.99}, "confidence": 0.98},
    {"form": "dog", "category_candidates": {"NOUN": 0.98}, "confidence": 0.97},
    {"form": "chased", "category_candidates": {"VERB": 0.98}, "confidence": 0.96},
    {"form": "the", "category_candidates": {"DET": 0.99}, "confidence": 0.98},
    {"form": "cat", "category_candidates": {"NOUN": 0.98}, "confidence": 0.97},
    {"form": ".", "category_candidates": {"PUNCT": 0.99}, "confidence": 1.0},
]
request = Point3Request(
    sequence_id="sentence-001",
    lexical_items=lexical_items,
    morphology_features={
        1: {"number": "singular"},
        2: {"number": "singular"},
        4: {"number": "singular"},
    },
)
result = Point3SyntacticParser(
    config=SyntaxConfig(simulate_biology=False)
).process(request)

print(result.dependency_relations)
print(len(result.functional_trace))  # 12
print(result.diagnostics["sentence_proposition_composed"])  # False
```

The handoff from Point 2 can be created with one result per word:

```python
request = Point3Request.from_point2_results(
    "sentence-002", [point2_the, point2_dog, point2_chased, point2_the2, point2_cat]
)
result = Point3SyntacticParser().process(request)
```

## SQLite annotations

On processor initialization, `register_point03_annotations` indexes module, class, function, and
method docstrings/signatures from `ore_language_comprehension` and `ore_syntax` into the existing
`ore_logger.annotations` SQLite table. The default database remains
`ore_logs/ore_logger.sqlite3`; use `ORE_LOGGER_DB` or `database_path` to select another path.
Point 3 does not create a separate annotation database.

## Scope and limitations

- The underlying compact grammar is English-oriented. Supplying a different `language` label does
  not automatically provide that language's word order, case, agreement, clitic, or phrase rules.
- POS continuation expectations and phrase-closure indicators added by the adapter are transparent
  heuristics.
- Agreement penalties are calculated only from explicit caller-provided morphology.
- Prosody contributes only when the caller supplies speech boundary cues; punctuation remains a
  visual convention rather than a direct equivalent of intonation.
- The lower-level ORE syntax engine has internal subject/object and thematic-role modules. Point 3
  returns grammatical dependencies but intentionally omits semantic/thematic-role outputs.
- No proposition such as `CHASE(dog, cat)`, discourse interpretation, speaker-intention inference,
  or pragmatic enrichment is performed.

Run the focused acceptance suite with:

```bash
python -m pytest tests/test_language_comprehension_point03.py -q
```
