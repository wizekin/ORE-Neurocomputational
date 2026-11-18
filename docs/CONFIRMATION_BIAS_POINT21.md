# ORE — Confirmation bias and belief-consistent attention, interpretation, and memory (Point 21)

## Purpose

Point 21 implements the source sequence in the file Point_21_Confirmation_Bias_and_Belief_Consistent_Memory.md. It models confirmation bias as a set of separable processing effects rather than a single yes/no trait:

1. a prior belief activates expectations;
2. the belief can influence which information sources are selected;
3. attention can be allocated unevenly to incoming evidence;
4. ambiguous observations can receive belief-consistent interpretations;
5. evidence can be weighted asymmetrically at decision readout;
6. confidence can resist reliable contradictory evidence; and
7. later retrieval can make belief-consistent details more accessible.

These stages can occur independently. The implementation therefore reports stage-specific measurements and does not infer a memory bias merely because source selection or judgment was biased.

The pipeline is a functional and neurocomputational engineering model. It does not claim that neurons literally calculate the equations shown below, that confirmation bias has one anatomical centre, or that every prior belief is irrational. Priors are useful; the failure is poorly calibrated use of evidence, failure to sample alternatives, or confidence that is detached from evidence quality.

## Fourteen source-ordered operations

| ID | Operation | Functional result |
|---|---|---|
| P21-01 | Define the current belief state | Normalized competing-hypothesis priors, current confidence, and belief provenance |
| P21-02 | Activate expectation templates | Predictions for the focal and alternative hypotheses, kept separate from observations |
| P21-03 | Select information sources | Ranked sources with diagnosticity, reliability, accessibility, trust, challenge cost, and congruence shown separately |
| P21-04 | Allocate attention | Per-observation attentional gain and processing-strength estimates |
| P21-05 | Encode evidence with provenance | Content, source, time, context, evidential direction, and attention strength |
| P21-06 | Interpret ambiguous evidence | Per-item probability distributions across candidate hypotheses and alternative interpretations |
| P21-07 | Assess evidence quality | Reliability-, diagnosticity-, measurement-, and dependence-adjusted weights |
| P21-08 | Compare evidence symmetrically | Centered log-likelihood contributions for every hypothesis using a common scale |
| P21-09 | Read out evidence for a judgment | A provisional evidence posterior and item-specific readout gates |
| P21-10 | Update belief and confidence | Bounded belief update, uncertainty, and possible perseverance warning |
| P21-11 | Retrieve belief-related memories | Cue- and belief-sensitive accessibility with original content and source tags preserved |
| P21-12 | Detect the reinforcing feedback loop | Separate selection, attention, interpretation, readout, and memory indicators |
| P21-13 | Apply corrective control | Pre-registered predictions, alternative hypotheses, and ranked discriminating tests |
| P21-14 | Validate stages separately | Measurable process metrics and explicit failure signatures |

Every operation produces a registry-aligned trace containing its purpose, mathematical form, functional output, biological-reference output, and evidence-level label.

## Mathematical operations

### Belief representation

For competing hypotheses \(H_k\), priors form a normalized distribution:

\[
\sum_k P(H_k)=1,\qquad L_k=\log P(H_k)
\]

A persisted belief from an earlier run can initialize the next evaluation of the same claim ID. The original request priors remain available in the request object and are not rewritten.

### Expectation templates

For feature \(f\), the model stores the predicted feature value separately under each hypothesis:

\[
T_{k,f}=P(f\mid H_k)
\]

A discriminability score compares the focal prediction with the mean prediction under alternatives. Predictions are not entered into the evidence ledger as observations.

### Information-source selection

A source utility combines diagnostic value \(D\), source reliability \(R\), accessibility \(A\), social trust \(T\), expected belief congruence \(C\), and anticipated challenge cost \(K\):

\[
U(s)=0.45D(s)R(s)+0.20A(s)+0.10T(s)
+a(2C(s)-1)-bK(s)
\]

A softmax converts these utilities into relative selection probabilities. The coefficients are tunable model parameters, not biological constants. In accuracy-first mode, the congruence and challenge-cost preference terms are disabled. Reliability and diagnostic value remain active.

### Attention and evidence encoding

Attention gain is a weighted combination of relevance, salience, novelty, uncertainty reduction, and belief congruence:

\[
G_i=0.10+0.28R_i+0.22S_i+0.12N_i+0.22U_i
+\eta(2C_i-1)
\]

A temperature-scaled softmax gives relative allocation. Attention strength is not evidence truth and is not the same as memory encoding. An item from an unselected source remains in the evidence audit but is marked as not deeply processed.

An encoded evidence trace is represented as:

\[
M_i=(x_i,s_i,c_i,d_i,q_i,t_i)
\]

where the terms preserve the original content, source, context, evidence direction, quality, and time. The stored content is never overwritten by a later belief-consistent interpretation.

### Interpretation and evidence quality

For one observation and hypothesis \(H_k\), the interpretation model combines its likelihood and prior:

\[
P(H_k\mid E_i)\propto P(E_i\mid H_k)\,P(H_k)^\rho
\]

The implementation separately exposes the likelihood-only distribution so the prior's effect can be audited.

Evidence weight is computed independently of whether the observation supports the focal belief:

\[
w_i=r_i\,d_i\,m_i\,a_i\,\frac{1}{\sqrt{n_g}}
\]

where \(r_i\) is source reliability, \(d_i\) diagnosticity, \(m_i\) measurement quality, \(a_i\) attention strength, and \(n_g\) is the number of selected evidence items sharing an independence group. Reports from the same underlying source therefore do not count as independent corroboration.

### Symmetric comparison and readout

The log-likelihood for each hypothesis is centered against the mean across hypotheses:

\[
\ell_{i,k}=\log P(E_i\mid H_k)
-\frac{1}{K}\sum_j\log P(E_i\mid H_j)
\]

This produces positive contributions for relatively favoured hypotheses and negative contributions for alternatives on one common scale.

A separate readout gate can model congruence-dependent use of otherwise encoded evidence:

\[
g_i=1+\zeta(2C_i-1),\qquad
L_k=\log P(H_k)+\sum_i w_i g_i\ell_{i,k}
\]

When accuracy-first mode is enabled, \(g_i=1\). The gate is not allowed to modify the original observation, source reliability, or memory trace.

### Confidence, memory retrieval, and corrective control

The evidence-supported posterior \(P^*_k\) is combined with the prior using a bounded update rate:

\[
B_{t+1,k}=(1-\lambda)B_{t,k}+\lambda P^*_k
\]

The rate is configurable and is not a universal learning constant. A perseverance warning is raised only when the focal prior was high, the evidence-supported target materially lowers it, and the bounded update still leaves a substantial residual.

Memory accessibility is modeled from the stored trace strength and cue match, with an optional belief-congruence retrieval gate. The retrieved output includes the immutable original content and source attribution so that accessibility is not mistaken for historical accuracy.

Corrective source ranking uses an approximate entropy-reduction score:

\[
IG(s)=H(H\mid E)-H(H\mid E,e_s)
\]

This implementation uses the supplied candidate observations to estimate information gain. It is a transparent ranking heuristic, not a full generative model of every possible future outcome.

## Biological-reference implementation

ConfirmationBiasNetwork contains eight recurrently interacting population-rate variables:

- prefrontal prior/expectation;
- prefrontal control and monitoring;
- frontoparietal attention;
- parietal evidence readout;
- hippocampal context binding;
- temporal semantic associations;
- salience/valuation; and
- metacognitive conflict monitoring.

Each operation advances the full network through leak, recurrent excitation, and cross-population inhibition. The registry maps broad functions to candidate distributed systems, not to isolated one-region/one-operation mechanisms. The activity values are simulated reference signals, not measured brain activity, biophysical neuron models, anatomical localization, or fitted physiological parameters. The network cannot determine whether a belief is true.

Set Point21Config(simulate_biology=False) to disable this separate track while retaining all functional operations and their trace records.

## Quick start

~~~python
from ore_confirmation_bias import (
    EvidenceItem,
    EvidenceSource,
    Hypothesis,
    Point21Config,
    Point21ConfirmationBiasProcessor,
    Point21Request,
)

request = Point21Request(
    claim_id="wet-street-001",
    claim="Why was the street wet?",
    hypotheses=(
        Hypothesis(
            "rain",
            "Rain caused the street to become wet.",
            0.70,
            {"wet_street": 0.9, "rain_gauge": 0.9},
        ),
        Hypothesis(
            "cleaning",
            "Street cleaning caused the street to become wet.",
            0.30,
            {"wet_street": 0.7, "rain_gauge": 0.1},
        ),
    ),
    sources=(
        EvidenceSource(
            "camera",
            "direct_record",
            reliability=0.95,
            diagnostic_value=0.8,
            independence_group="camera-system",
        ),
        EvidenceSource(
            "gauge",
            "instrument_record",
            reliability=0.92,
            diagnostic_value=0.9,
            independence_group="weather-station",
        ),
    ),
    evidence=(
        EvidenceItem(
            "wet-road",
            "camera",
            "The road surface appears wet.",
            feature="wet_street",
            likelihoods={"rain": 0.85, "cleaning": 0.65},
            direct_observation=True,
        ),
        EvidenceItem(
            "no-rain",
            "gauge",
            "The rainfall gauge recorded no rain.",
            feature="rain_gauge",
            likelihoods={"rain": 0.10, "cleaning": 0.90},
            direct_observation=True,
        ),
    ),
    focus_hypothesis_id="rain",
    context={"memory_cues": ["wet_street", "rain_gauge"]},
)

processor = Point21ConfirmationBiasProcessor(
    Point21Config(accuracy_first=False, simulate_biology=True)
)
result = processor.process(request)

print(result.beliefs)
print(result.validation["failure_signatures"])
print(len(result.operation_trace))  # 14
~~~

The request requires at least two hypotheses with priors summing to 1. Each evidence item's likelihoods maps hypothesis IDs to positive relative likelihoods. Missing likelihood entries are treated as neutral likelihood 1.0. Evidence source IDs must refer to sources in the same request.

## Shared SQLite annotation and execution logging

The package reuses the repository's ore_logger database. register_annotations() indexes every Point 21 package class, function, and method in the shared annotations catalogue. record_result_to_sqlite(result) registers the symbols and writes the fourteen operation traces to the existing runs and events tables.

Run the focused tests:

~~~bash
python -m pytest tests/test_confirmation_bias_point21.py -q
~~~

Run the complete repository suite:

~~~bash
python -m pytest
~~~

## Scientific and epistemic safeguards

- Confirmation bias is a family of possible process effects; the model does not force all stages to be biased together.
- Priors influence inference but do not make evidence immune to revision.
- Source reliability is evaluated independently of agreement with the current belief.
- Dependent reports sharing an independence group are downweighted.
- Unselected information remains visible; lack of deep attention is not treated as non-existence.
- Original observation, source tag, interpretation, evidence weight, belief, and confidence are separately represented.
- A belief-consistent interpretation does not rewrite the original evidence trace.
- Confidence is not equivalent to truth or objective accuracy.
- Corrective controls seek diagnostic evidence, not merely an opposing claim.
- The coefficients are illustrative engineering parameters requiring empirical calibration.
- The biological track is a functional analogy, not a validated model of human neural activity.
