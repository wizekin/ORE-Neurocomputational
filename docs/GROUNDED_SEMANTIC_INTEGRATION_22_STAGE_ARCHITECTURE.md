# Grounded Semantic Integration and Predictive Language — 22-stage implementation

## Purpose and scientific scope

The ore_grounded_integration package implements the 22 stages in the source specification
"Grounded Semantic Integration and Predictive Language Algorithm.txt".

The design maintains two separate interpretations:

- **Functional model:** executable Python operations over feature vectors, probability distributions,
  semantic graphs, transition counts, memory records and learner metrics.
- **Biological reference:** explicitly named neural populations and a configurable current-based
  leaky-integrate-and-fire (LIF) simulation, used as a measurable systems analogy.

The LIF output is not EEG, MEG, fMRI, electrophysiology or a recording from a participant.
The implementation does not claim that a human brain literally performs Transformer attention,
feature hashing, backpropagation, a specific Hebbian equation or next-token prediction in the same
form as the software. Those operations are computational models for testing functional hypotheses.

## Stage map

| ID | Stage | Main executable operations |
| --- | --- | --- |
| GS-01 | Acquire and synchronize multimodal information | Capture channel inputs; calculate temporal gaps and exponential co-occurrence weights |
| GS-02 | Extract modality-specific representations | Encode and normalize distinct modality feature vectors |
| GS-03 | Construct distributional linguistic representations | Build token context counts and lexical vectors |
| GS-04 | Determine the current semantic context | Score context hypotheses and normalize a context posterior |
| GS-05 | Align cross-modal representations | Pairwise cosine similarity and contrastive margin objective |
| GS-06 | Connect concepts to actions and affordances | Estimate action cue strengths and state-transition affordances |
| GS-07 | Integrate linguistic, sensory and motor information | Reliability-weighted projection into a common concept representation |
| GS-08 | Build a semantic relationship network | Create associative links and spread activation with decay |
| GS-09 | Select the meaning appropriate to context | Score and rank explicit candidate senses |
| GS-10 | Compose word meanings into sentence meaning | Contextual sentence vector and a transparent event-role frame |
| GS-11 | Generate predictions about language and events | Learn next-token transition counts from observed sequences |
| GS-12 | Compare predictions with incoming information | Compare integrated input against stored prototypes and compute residual error |
| GS-13 | Regulate attention by relevance and uncertainty | Score each modality and normalize attention allocation |
| GS-14 | Retrieve semantic and episodic memories | Rank concept prototypes and episodes by similarity, strength and recency evidence |
| GS-15 | Perform semantic inference and generalization | Traverse one- and two-hop semantic graph paths |
| GS-16 | Produce language or behavior | Generate conservative task-conditioned output from observed and retrieved evidence |
| GS-17 | Learn from feedback and update associations | Update co-occurrence, graph edges, transition counts, Hebbian-style matrices and TD estimates |
| GS-18 | Consolidate learning and maintain knowledge | Decay accessibility, update prototypes and preserve episodic records |
| GS-19 | Construct multisensory teaching representations | Preserve teaching modalities and measure pairwise coherence/reliability |
| GS-20 | Construct meaningful usage examples | Keep caller examples and generate context-variation practice prompts |
| GS-21 | Evaluate learning using retrieval and prediction | Score recall, identification, action, context and generalization |
| GS-22 | Adapt instruction to learner knowledge | Choose an activity targeting the weakest measured learner dimension |

Every stage is represented in registry.py by a stable stage ID and one or more
OperationSpec records. Each operation states its mathematical form, functional responsibility
and biological analogy.

## Mathematical implementation

The implementation uses the following core forms:

- Temporal co-occurrence: A_ij = exp(-abs(t_i - t_j) / tau).
- Feature standardization, unit normalization and cosine similarity.
- Stable softmax: p_i = exp((s_i - max(s))/T) / sum_j exp((s_j - max(s))/T).
- Cross-modal contrastive margin: max(0, margin - positive_similarity + hardest_negative).
- Modality integration: C = sum_i(alpha_i * H_i), with alpha = softmax(reliability + relevance).
- Graph spreading: a_i(t+1) = f(sum_j(W_ji * a_j(t)) - decay * a_i(t)).
- Prediction residual: e = observed - predicted; squared error is 0.5 * dot(e,e).
- Attention allocation: temperature-scaled softmax over relevance, reliability and uncertainty cues.
- Memory retrieval: cosine similarity plus bounded memory strength.
- Hebbian-style update: DeltaW = eta * outer(post, pre) - decay * W, clipped to a bounded range.
- Temporal-difference error: delta = reward + gamma * next_value - current_value.
- Memory accessibility: S(t) = S0 * exp(-lambda * t), with retrieval strengthening.
- Entropy: H = -sum_i(p_i * log(p_i)).
- Learner score: weighted mean over the learner dimensions actually provided.
- Adaptive activity: choose the activity associated with the weakest measured learner dimension.

These equations are engineering approximations from the source's mathematical discussion.
They should be compared against behavioral or neural data before making mechanistic claims.

## Encoders and input contract

GroundedSemanticRequest accepts text or caller-supplied tokens and any subset of:
linguistic, visual, auditory, motor and context.

Modality inputs may be named feature maps, strings, numeric vectors or arrays. The default
feature encoder uses deterministic feature hashing/binning. It is dependency-light and
reproducible, but it is not a vision model and does not extract image semantics from raw
pixels. For raw images, audio waveforms or richer text features, pass a corresponding
callable encoder to GroundedSemanticIntegrationProcessor(encoders={...}).

Unobserved modalities are not fabricated. Candidate senses and learner scores are
caller-provided evidence. If no candidate senses are supplied, the module does not invent
a fine-grained word sense. If no learner response is supplied, Stage 22 asks for baseline
evidence rather than inventing a learner profile.

## Quick start

~~~python
from ore_grounded_integration import (
    GroundedSemanticIntegrationProcessor,
    GroundedSemanticRequest,
)

processor = GroundedSemanticIntegrationProcessor()
request = GroundedSemanticRequest(
    sample_id="bird-001",
    text="The bird builds a nest",
    modality_inputs={
        "visual": {"wings": 0.9, "feathers": 0.8},
        "auditory": {"chirp": 0.7},
        "motor": {"build": 0.8, "fly": 0.6},
    },
    context={"forest": 1.0, "tree": 0.7},
    task="describe",
    candidate_senses={
        "bird_animal": {"bird": 1.0, "feathers": 0.5},
        "sports_bat": {"bat": 1.0, "ball": 0.5},
    },
)
result = processor.process(request)
print(result.output)
assert len(result.stage_trace) == 22
~~~

The processor preserves its learned word transitions, semantic graph, concept prototypes,
episodes and association weights between successive calls on the same instance.

## SQLite annotations and run traces

ORE's existing ore_logger package is used; no separate database schema is introduced.

~~~python
from ore_grounded_integration import (
    register_grounded_annotations,
    record_grounded_result,
)

# Index every class and function docstring across the repository.
annotation_report = register_grounded_annotations()

# After processor.process(request), persist every registered mathematical operation
# and all 22 stage traces to the same SQLite logger.
run_report = record_grounded_result(result)
print(annotation_report["database_path"])
print(run_report["database_summary"])
~~~

Default storage: ore_logs/ore_logger.sqlite3, overridable via ORE_LOGGER_DB
or the database_path argument. AST indexing records class/function annotations,
qualified names, signatures, line ranges and source hashes in ore_logger.annotations.
Execution records use the logger's existing runs, events and annotations tables.
The binary database is runtime output and should not be committed.

Run a local demonstration and SQLite registration:

~~~bash
python -m ore_grounded_integration
python -m ore_grounded_integration --index-only
python -m ore_grounded_integration --no-biology --database /tmp/ore-grounded.sqlite3
~~~

## Validation

The subsystem tests validate:

- All 22 stable stage IDs and the stage-operation registry.
- Complete functional traces and 22 optional LIF snapshots.
- Learning across repeated inputs: word transitions, semantic graph and memory retrieval.
- Core numerical primitives.
- SQLite class/function annotation registration.
- Persisted operation and stage events with linked annotation text.

The default implementation is deterministic and local. Tests do not require external pretrained
models or access to sensors.
