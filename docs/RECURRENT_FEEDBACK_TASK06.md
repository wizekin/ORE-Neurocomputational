# Task 6 — Create recurrent feedback and corrective learning

## Scope and design boundary

The `ore_corrective_learning` package already implements Point 24's twenty corrective-learning operations for a single evidence-weighted correction episode. Task 6 adds the outer `RecurrentFeedbackController`: it detects a possible semantic conflict, routes targeted information actions to the relevant ORE structures, validates new evidence, invokes Point 24 again from the previous posterior, and stops when the evidence and uncertainty gates are satisfied or the bounded budget is exhausted.

The controller reuses rather than replaces:

- Point 24 — hypothesis comparison, evidence-quality weighting, Bayesian update, corrective explanation, retrieval and calibration state.
- Point 21 — confirmation-bias auditing and symmetric evidence review.
- Point 22 — reasoning-heuristic and judgment-conflict diagnostics.
- Point 07 — feedback seeking and information acquisition.
- Point 08 — curiosity and information-gap handling.
- Task 5 memory integration — read-only episodic/semantic recall and provenance-preserving memory evidence.
- N9 — optional targeted semantic reprocessing on a copied request.

A missing module is reported as `unavailable`; no replacement result or evidence is invented. Typed processor requests must be supplied through `RecurrentFeedbackRequest.module_inputs`, or the application may provide a named `module_handlers` adapter.

## Processing cycle

```text
freeze current input and initial interpretation
           |
           v
Point 24 initial hypothesis/evidence pass
           |
           v
measure compatibility dimensions and detect conflict
           |
           v
classify probable error causes; identify affected modules
           |
           v
confirmation-bias / heuristic / curiosity / memory checks
           |
           v
rank supplied information actions by information gain minus cost
           |
           v
invoke one targeted action; validate returned evidence
           |
           v
apply only the new evidence batch from the previous posterior
           |
           v
reprocess affected structures (N9 only when explicitly supplied)
           |
           v
re-evaluate conflict, posterior movement, confidence, and uncertainty
           |
           +---- supported + conflict resolved ---> finalize correction episode
           |
           +---- no progress / no useful actions --> preserve uncertainty
           |
           +---- budget remains ------------------> next bounded feedback round
```

The original input and interpretation are kept in the episode snapshot. A revised hypothesis is recorded as `supersedes_in_context`; the old hypothesis is not deleted. Every feedback episode has a maximum iteration count and a shared action/cost budget.

## Mathematical operations

### Semantic compatibility

For measured compatibility signals (s_jin[0,1]) and non-negative weights (w_j):

[
C=rac{sum_{jin J}w_j(1-s_j)}{sum_{jin J}w_j}
]

(J) contains only dimensions with an actual measurement and positive weight. A missing score is omitted; it is not assigned zero compatibility. The controller detects a possible conflict when (C) reaches its configured threshold or an explicit review was requested. A conflict score is a trigger for investigation, not proof that the existing interpretation is false.

### Information-action selection

For the current hypothesis distribution (P(H)), a caller-supplied response model for action (a), and its expected response (Y):

[
IG(a)=H(H)-mathbb{E}_{Ymid a}[H(Hmid Y,a)]
]

[
U(a)=IG(a)-lambda,Cost(a)
]

Only options with positive utility and cost within the shared budget are considered. Actions are fingerprinted from their module, action identifier, target, context, and evidence-set identity. Repeated unsuccessful actions with unchanged inputs are skipped. New context or new evidence changes the fingerprint and can make the action eligible again.

### Bayesian evidence update

Point 24 collapses dependent evidence into source-independence groups before computing likelihood contributions. The conceptual update is:

[
P_t(H_i)propto P_{t-1}(H_i)
prod_{gin G}L(E_gmid H_i)^{w_g}
]

For every recurrent update, the previous posterior becomes the next prior and **only the newly validated evidence batch** is passed to Point 24. Earlier evidence remains in the episode history but is not counted again. Evidence likelihoods must be explicitly supplied for the hypotheses being considered; the controller does not infer likelihoods from prose, confidence, or memory similarity.

### Progress and convergence

The controller separately measures posterior and conflict movement:

[
Delta P=max_i|P_t(H_i)-P_{t-1}(H_i)|
]

[
Delta C=|C_t-C_{t-1}|
]

A round is considered to make no measurable progress when both values are below the configured convergence threshold. Consecutive no-progress rounds are bounded by a patience value. **Convergence does not imply correctness**: a stable posterior with insufficient evidence, unresolved conflict, or high uncertainty remains `UNCERTAIN`.

### Confidence and calibration

The controller reports posterior concentration using normalized Shannon entropy:

[
H_N(P)=rac{-sum_iP_ilog P_i}{log |H|}
]

Evidence quality is separately calculated with Point 24's reliability, relevance, diagnosticity, directness, and source-independence logic. An independent-verification result is possible only when the selected hypothesis matches the caller-supplied `verified_hypothesis_id`. A Brier score is reported only when an observed verified outcome is actually available from the underlying Point 24 calibration record.

## Stop outcomes and safeguards

The controller returns one of:

- `NO_CORRECTION_NEEDED`
- `SUPPORTED_CORRECTION`
- `INDEPENDENTLY_VERIFIED`
- `UNCERTAIN`
- `MAXIMUM_ITERATIONS_REACHED`

The default semantic loop is limited to four rounds, with an absolute configurable hard limit of eight. Action attempts and total action cost are also bounded. An unavailable module, invalid evidence, duplicated evidence ID, unknown hypothesis ID, or malformed adapter output is recorded in diagnostics; it cannot silently enter the posterior.

Ordinary inference never mutates model parameters or promotes a candidate into permanent knowledge. The optional N9 adapter deep-copies the original `N9Request`, adds revision context, and forces `learning_authorized=False`. A separate `execute_authorized_learning_transaction` method delegates to `LearningTransactionManager`; it requires an explicitly authorized `ore_contracts.LearningRequest` and regression gates, returns a detached child, and never promotes it into the active model.

## API example

```python
from ore_corrective_learning import (
    ConflictDimension,
    CorrectionRequest,
    FeedbackEvidence,
    KnowledgeHypothesis,
    RecurrentFeedbackController,
    RecurrentFeedbackRequest,
)

correction = CorrectionRequest(
    request_id="sentence-001",
    hypotheses=(
        KnowledgeHypothesis("sense-a", "The familiar word sense", prior=0.5),
        KnowledgeHypothesis("sense-b", "The contextual word sense", prior=0.5),
    ),
    current_hypothesis_id="sense-a",
)

request = RecurrentFeedbackRequest(
    request_id="feedback-001",
    correction_request=correction,
    input_snapshot={"text": "A sentence with an ambiguous word"},
    compatibility_dimensions=(
        ConflictDimension(
            dimension="semantic_selection",
            compatibility_score=0.20,
            affected_modules=("semantic_processor", "memory_retrieval"),
        ),
    ),
)

controller = RecurrentFeedbackController()
result = controller.process(request)
print(result.status)
print(result.episode.correction_record)
```

For a real correction, the application supplies typed requests or module handlers that can return Point 24 `FeedbackEvidence` with explicit likelihoods, plus `ClarificationOption` objects with response models. The controller does not browse, ask a human, or manufacture missing source responses by itself. In production, pass a persistent logger path or set `ORE_LOGGER_DB`; controller initialization registers all recurrent-controller and N9 symbols in the existing logger `annotations` table, while each episode can record its fourteen operation events in the same SQLite database.

## SQLite annotations and operation audit

The implementation registers every class, top-level function, and method in `ore_corrective_learning`, `ore_n9`, `ore_feedback_seeking`, `ore_confirmation_bias`, `ore_cognitive_heuristics`, `ore_curiosity_motivation`, and `ore_memory_integration` with `ore_logger.catalog.sync_repository`. Each recurrent result contains one source-ordered row for each RF-001 through RF-014 operation; repeated rounds are grouped under their corresponding operation entry. `record_recurrent_result_to_sqlite` writes those rows and a final episode event to `ore_logger.sqlite3`, linking each method event to both its method and class annotation symbols.

The logger database is instrumentation, not the long-term semantic store. Retrieved episodes and semantic claims remain in their existing Point 7 database; the recurrent controller only records the evidence and references required to explain its decision.

## Validation and scientific limits

Run:

```bash
python -m pytest -q tests/test_recurrent_feedback_controller.py
python -m pytest -q
```

Tests cover conflict/no-conflict routing, missing compatibility data, Bayesian revision from explicit feedback, preservation of alternatives, action fingerprinting, no-progress and iteration stops, unresolved stable beliefs, optional N9 read-only reprocessing, unauthorized model-learning rejection, and SQLite annotations.

Weights and thresholds are transparent engineering defaults, not biologically measured constants. The implementation is a functional integration prototype, not evidence of a biologically faithful feedback circuit or autonomous fact-checking system. Its correctness is bounded by the accuracy, independence, and provenance of caller-supplied evidence and adapters.
