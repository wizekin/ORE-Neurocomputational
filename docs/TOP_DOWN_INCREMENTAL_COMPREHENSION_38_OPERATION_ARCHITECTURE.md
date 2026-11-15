# TDPIA: Top-Down Prediction and Incremental Comprehension

## Purpose and scientific scope

The ore_comprehension package implements the seven-stage, 38-operation specification in
Brain Comprehension Algorithm.md as an inspectable Python reference system.
It connects predictive language processing, word-by-word integration,
semantic mismatch, syntactic reanalysis, and experience-dependent adaptation.

This is a computational model, not a claim that the human brain contains these
software modules or literally computes Bayesian formulae, logarithms, cosine
similarities, or matrix products. The neural dynamics are a separate, coarse
leaky-integrate-and-fire (LIF) reference simulation. They are not anatomically
mapped, calibrated to human EEG, or validated as a biological circuit.

## Seven stages and all 38 operations

| Stage | Operations | Responsibility |
|---|---:|---|
| A — Sentence-level prediction | 1–3 | Reconstruct context, infer likely situation, predict phrase continuations |
| B — Syntactic prediction | 4–8 | Maintain partial parse, grammatical category, dependencies, agreement, competing parses |
| C — Semantic prediction | 9–12 | Activate concepts, predict semantic categories, thematic roles, expected features |
| D — World-knowledge prediction | 13–16 | Retrieve facts and event schemas, assess plausibility and prior reliability |
| E — Word-level surprisal | 17–20 | Recognize input, estimate word probability, surprisal and entropy |
| F — N400/P600 modeling | 21–28 | Compare expected and actual features, model semantic mismatch, conflict, reanalysis and ERP proxies |
| G — Incremental comprehension | 29–38 | Integrate each word, update interpretations/context, predict next word, complete and consolidate, learn |

The canonical operation IDs are TDPI-001 through TDPI-038. The
OPERATION_REGISTRY stores each operation's stage, number, method name,
plain-English purpose, mathematical expression, and biological reference.

## Mathematical mechanisms

The functional pipeline includes the following families of operations:

1. **Probability and normalization:** softmax distributions over continuations,
   syntactic categories, semantic roles, and event interpretations.
2. **Bayesian-style updating:** prior interpretation support is multiplied by
   the likelihood of newly observed evidence and normalized.
3. **Information theory:** word surprisal is -log2(P(word | context));
   predictive uncertainty is Shannon entropy -sum(p * log2(p)).
4. **Semantic vector operations:** weighted feature pooling, cosine similarity,
   feature residuals, and mean squared/precision-weighted error.
5. **Representational conflict:** Jensen–Shannon divergence compares role
   distributions; agreement mismatch and parse uncertainty contribute to
   a reanalysis score.
6. **Incremental integration:** actual semantic features update predicted
   features through a precision-weighted residual.
7. **Learning:** word-pair associations are strengthened with an incremental
   Hebbian-style update, and event frequencies are accumulated across runs
   of one processor instance.

All probabilities are normalized; entropy uses base-2 logarithms; and numerical
floors avoid log(0). Model weights and thresholds are engineering parameters,
not fitted neural constants.

## Functional and biological tracks

ComprehensionProcessor.process() runs all 38 methods in registry order for each
input token. Each trace includes the operation ID, stage, formula, biological
rationale, functional output, and corresponding LIF population activity.

The biological reference has coarse populations for sentence context, syntax,
semantics, world knowledge, lexical input, prediction error, conflict monitoring,
working memory, and learning. Its membrane state follows a leaky-integrate-and-fire
update with recurrent excitation, global inhibition, heterogeneous thresholds,
refractory periods, and numerical diagnostics.

The LIF track receives the functional operation's activation and error/conflict
signals. It does not feed its spikes back into the functional probability
calculation; this separation makes the symbolic/statistical outputs reproducible
and allows the biological analogy to be switched off for faster tests.

## N400 and P600 estimates

- **N400-like estimate:** a negative scalar based on semantic-feature mismatch
  and normalized word surprisal. N400 is a measured EEG ERP component with
  multiple contributing factors; the scalar is not an EEG waveform or direct
  measurement.
- **P600-like estimate:** a positive scalar based on syntactic/semantic
  representational conflict and reanalysis demand. P600 is not simply a
  surprisal measure, and its functional interpretation remains debated.

The gain parameters are configurable and must be fitted to experimental EEG
before any comparison to measured amplitudes is attempted.

## Inputs and extensibility

The default lexicon and event schemas are intentionally small and transparent.
Applications can extend the model with:

- lexical_categories: mapping words to POS labels such as NOUN, VERB, ADJ, DET,
  PRON, ADP, or AUX;
- semantic_features: mapping words to sparse numeric feature vectors;
- world_knowledge: mapping words to caller-supplied entity/event features;
- discourse_context: previously available context strings.

This keeps the package deterministic and dependency-light while allowing ORE
upstream components or domain knowledge stores to supply richer evidence.

## Run

From the repository root, run:

    python -m ore_comprehension "The chef prepared the meal with a knife."

Disable the LIF reference if only functional output is needed:

    python -m ore_comprehension --no-biology "The chef prepared the meal with a knife."

By default the CLI synchronizes Python class/function docstrings into
ore_logs/ore_logger.sqlite3 before processing. Set ORE_LOGGER_DB or pass
--db-path to use a different SQLite file.

## Logger annotations

Every class, public helper, pipeline method, and CLI main() has a plain-English
docstring. The repository's ore_logger.catalog.sync_repository() AST-scans
these docstrings and stores them as reusable rows in the logger's annotations
table, including module, qualified name, source path, signature, source hash,
and annotation text. Each CLI run also creates standard run/event records through
the existing sitecustomize/ore_logger.runtime integration when enabled.

The annotation catalogue is synchronized when running the CLI, or can be
re-synchronized independently using the existing LoggerDatabase and
sync_repository APIs. The SQLite database is a runtime artifact and is not
committed to Git; the source annotations are committed with the Python code.
