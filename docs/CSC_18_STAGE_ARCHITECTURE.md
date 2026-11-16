# Controlled Semantic Cognition (CSC) — 18-stage architecture

**Package:** `ore_csc/`  
**Registry:** ORE-CSC-18  
**Version:** 1.0.0  
**Status:** Functional computational prototype with an optional current-based LIF reference network.

## 1. Purpose and scope

CSC separates semantic representation from semantic control. The representation side retrieves distributed conceptual knowledge from modality-linked spokes and a transmodal integration state associated here with the anterior temporal lobes (ATL). The control side uses task and context to influence which features are amplified, inhibited, or deliberately retrieved. Left inferior frontal gyrus (IFG) and posterior middle temporal gyrus (pMTG) are represented as important control/retrieval nodes. The implementation also records pITG, dorsomedial and dorsolateral prefrontal, and task-dependent parietal-attention contributions.

This package is additive. It reuses the existing provenance-tracked SemanticConcept contract from ore_semantic, but it does not replace the existing 44-operation semantic cognition registry or the 21-stage ore_hub_spoke architecture.

The numbered stages describe logical dependencies and traceability, not a claim that the human brain performs a strict serial pipeline. Several computations can proceed in parallel and the control loop revisits representations when the current retrieval is weak or incorrect.

## 2. Eighteen stages

| Stage | Name | Main functional and mathematical operations |
|---:|---|---|
| 1 | Sensory and linguistic input encoding | Signal/feature transformation, feature-vector construction, normalization, temporal integration |
| 2 | Candidate word and concept activation | Lexical similarity, frequency prior, log-evidence accumulation, parallel candidate activation |
| 3 | Distributed semantic memory activation | Weighted spreading activation, association weighting, exponential decay, recurrent state updates |
| 4 | ATL hub integration and concept formation | Reliability-weighted modality integration, cosine similarity, supported pattern completion, recurrent stabilization |
| 5 | Context construction and contextual constraint extraction | Context-feature extraction, exponential moving average, contextual similarity, Bayesian-style reweighting |
| 6 | Task and intention representation | Task-vector encoding, task-feature similarity, priority weighting, goal maintenance |
| 7 | Estimation of semantic control demand | Candidate entropy, conflict, target weakness, task novelty, sigmoid demand estimate |
| 8 | Recruitment of the semantic control network | Task/context/demand control signal, region activity, recurrent communication, demand-dependent coupling |
| 9 | Generation of targeted semantic retrieval signals | Target-feature vector, feature relevance, sigmoid gain, weak-but-relevant exploration priority |
| 10 | Amplification of task-relevant semantic features | Multiplicative gain, additive excitation, recurrent amplification, element-wise feature modulation |
| 11 | Inhibition of task-irrelevant semantic features | Subtractive inhibition, non-negative rectification, lateral competition, task-dependent suppression, combined E/I update |
| 12 | Controlled retrieval of weak or non-dominant knowledge | Target-weighted spreading activation, retrieval softmax, iterative retrieval, convergence criterion |
| 13 | Candidate meaning and feature evaluation | Semantic and task similarity, feature relevance, weighted evidence combination, candidate softmax |
| 14 | Competition resolution and semantic selection | Winner ranking, confidence margin, selection threshold, posterior confidence, multi-feature selection |
| 15 | Performance monitoring and control adjustment | Retrieval error, gain adjustment, inhibition adjustment, demand adaptation, iterative correction |
| 16 | Integration of selected semantic information | Weighted feature aggregation, feature-relation construction, final integration, relation consistency |
| 17 | Generation of the appropriate response or behaviour | Response-option similarity, best response, output gate, task-specific structured response |
| 18 | Adaptive learning and control updating | Hebbian co-activation, error-driven update, reward modulation, frequency/recency priors, parameter adaptation |

The registry in ore_csc/registry.py gives each stage and operation a stable ID. All operation records include a mathematical description, functional responsibility, and stage-level biological reference. The current registry contains 78 operations.

## 3. Functional track

The functional implementation accepts:
- A pinned sequence of SemanticConcept records with provenance, aliases, frequency, semantic features, and optional modality-specific features.
- One text cue or an ordered token sequence.
- Optional one-dimensional numeric signal samples, transformed with z-score normalization and a real FFT; raw audio decoding and image feature extraction remain external front-end responsibilities.
- Optional graded current and previous contextual/task feature maps, combined with exponential moving averages.
- Optional modality feature maps and response options.

Named feature activations are finite values in [0, 1]. The model retains all candidate meanings through initial lexical access. It uses weak frequency priors during candidate activation, then evaluates candidates again using semantic, task, feature, and contextual compatibility.

The core stages can be summarized as:

1. Candidate retrieval: lexical similarity plus a weak log-frequency prior.
2. Representation: sparse feature activation and feature co-occurrence spreading.
3. Integration: reliability-weighted spoke evidence and a recurrent ATL-inspired hub state.
4. Control: task/context relevance, uncertainty and conflict, distributed control activity, feature gain and inhibition.
5. Retrieval and selection: bounded recurrent search, candidate evidence integration, posterior/margin gate, and explicit abstention.
6. Feedback: control changes are adjusted in a bounded loop when task fit is insufficient.
7. Output and learning: a selected concept/response is returned; longer-term updates require explicit external feedback and authorization.

Candidate scores preserve separate evidence channels. They are inspectable, not hidden inside a single opaque score. The default response is a selected concept label or a caller-supplied option; this package is not an unrestricted natural-language generator.

## 4. Mathematical implementation

The equations are implemented as useful computational abstractions, not as experimentally measured neuron-level arithmetic.

**Activation propagation**

A_j(t+1) = f(sum_i W_ij A_i(t) + B_j)

Feature links are derived from co-occurrence inside the supplied concept inventory or are passed explicitly as associations. The nonlinear bounded activation is implemented with clipping to [0, 1].

**Selective gain**

A_i(controlled) = G_i A_i

The gain is computed from relevance to the current task/context target. Unrelated long-term concept features are not deleted.

**Candidate competition**

P(m_i | X,T,C) = exp(Z_i/tau) / sum_j exp(Z_j/tau)

The candidate score Z_i combines lexical, semantic, task, feature, and contextual evidence. Temperature tau controls how sharply score differences affect posterior values.

**Inhibition**

A_i(new) = max(0, A_i - I_i)

Inhibitory drive combines task irrelevance and a small competitive component. This reduces current influence, not the stored concept representation.

**Control demand**

H(M) = -sum_i P_i log(P_i)

K = 1 - (P_best - P_second)

U = 1 - A_target

N = 1 - sim(T, T_familiar)

D = sigmoid(w_h H + w_k K + w_u U + w_n N + b)

Entropy is normalized for comparability across candidate counts. The weights and bias are configurable modeling choices, not fitted biological constants.

**LIF reference**

For a generic population, the reference simulator uses a forward-Euler update of:

dV/dt = (V_rest - V + R_m I) / tau_m

A spike is emitted when membrane voltage reaches threshold; voltage is reset and a refractory interval is applied. Current/voltage scale conversion is explicit in code (MOhm × pA = 10^-3 mV). Output units are mV, pA, Hz, ms, and spike counts.

## 5. Biological-reference track and limitations

The optional LIF simulator includes named populations for sensory input, lexical candidates, distributed semantics, ATL hub, context/task, left IFG, posterior MTG/ITG, dorsomedial and dorsolateral PFC, optional parietal attention, response, and plasticity.

The population drive mapping is intended to make assumptions observable and testable. It does not reproduce anatomical connectivity, neuron counts, real firing-rate distributions, or an experimentally fitted human circuit. The exact anatomical boundaries of semantic-control networks and the role of parietal areas are model- and task-dependent. The LIF simulation must therefore be interpreted as a reference model, not evidence that the brain executes these equations or stages literally.

The source document presents Fourier analysis as one possible signal transformation. The package implements a numerical real FFT for caller-supplied one-dimensional signal samples, but does not acquire or decode audio recordings, perform cochlear modeling, or process raw images. Caller-supplied visual and other modality features remain feature-map inputs.

## 6. Learning safety and reproducibility

Learning is preview-only by default. The concept inventory is immutable. An update is committed only when the caller sets learning_authorized=True, provides feedback_reward in [-1, 1], and identifies feedback_concept_id in the request. This prevents self-selected outputs from being treated automatically as ground truth.

The current prototype keeps learned association/prior updates in the CSCProcessor instance. Durable semantic-memory persistence is a separate integration boundary and is not silently coupled to the execution logger.

## 7. Logger and annotations

The project-wide AST catalogue already stores module, class, function, and async-function annotations in ore_logs/ore_logger.sqlite3. CSC exposes:

- index_csc_annotations(repository_root, database_path=None): invokes ore_logger.sync_repository and verifies the ore_csc symbols in the annotations table.
- record_csc_result(result, repository_root, database_path=None): records all 78 mathematical/functional/biological operation descriptions and the 18-stage trace through SQLiteRunRecorder, linked to the indexed symbol annotations.

Every public class and callable in ore_csc has a docstring suitable for the shared annotation catalogue. Run the package demonstration from the repository root with python -m ore_csc to populate the logger database and save one sample run.

## 8. API example

    from ore_semantic.contracts import SemanticConcept
    from ore_csc import CSCProcessor, CSCRequest

    concepts = [
        SemanticConcept(
            concept_id="bank-river",
            label="river bank",
            aliases=("bank", "shore"),
            semantic_features={"RIVER": 1.0, "WATER": 0.9, "SHORE": 0.8},
            source_reference="dataset://bank/river",
        ),
        SemanticConcept(
            concept_id="bank-financial",
            label="financial bank",
            aliases=("bank",),
            semantic_features={"MONEY": 1.0, "TRANSACTIONS": 0.9},
            frequency=20.0,
            source_reference="dataset://bank/financial",
        ),
    ]

    processor = CSCProcessor(concepts)
    request = CSCRequest(
        sequence_id="bank-example",
        text="The fisherman sat beside the bank.",
        task="interpret the geographical meaning",
        context_features={"RIVER": 0.9, "WATER": 0.9},
        task_features={"SHORE": 0.8, "RIVER": 0.9},
    )
    result = processor.process(request)
    print(result.selected_concept)
    print(len(result.stage_trace), len(result.operation_trace))

The functional and biological tracks are both optional/configurable, and their outputs are reported separately.
