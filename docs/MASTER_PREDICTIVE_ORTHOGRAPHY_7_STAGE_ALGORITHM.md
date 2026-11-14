# ORE Master Predictive Orthography — Seven Stages / 40 Operations

**Package:** ore_predictive_orthography/  
**Functional backends:** ore_n3 (visual evidence and orthographic recurrence) and ore_n4 (attention and precision-gated inference)  
**Biological reference:** seven coupled current-based leaky-integrate-and-fire populations  
**Input boundary:** one already-segmented N2 patch or finite feature vector per grapheme position, in logical reading order  
**Scientific status:** computational research implementation; biological equivalence and human reading performance are not established.

## Architecture and scope

This additive master processor composes existing N3 and N4 implementations rather than replacing them. It emits a stable 40-operation registry, MPO-001–MPO-040, grouped across the seven requested stages. Its functional path produces candidate distributions, contextual messages, attention and precision values, attended posteriors, divergence signals and feature-space residuals.

A parallel biological path receives bounded summaries of the stage outputs and advances seven interacting neural populations. Neural activity does not feed back into or mutate the functional posterior. The LIF network is a computational reference, not a fitted brain simulation.

Character patches must already be segmented and ordered. N3 must be fitted on labelled feature samples from the target script/rendering domain. Transition statistics are learned from labelled grapheme sequences supplied to N3. This package is not an end-to-end OCR segmenter.

## Stage map and operation IDs

| Stage | Requested stage | Operations |
|---|---|---|
| S1 | Bottom-up character evidence | MPO-001–008 |
| S2 | Orthographic transition probabilities | MPO-009–015 |
| S3 | Predictions from neighboring grapheme beliefs | MPO-016–020 |
| S4 | Attention-dependent sensory precision | MPO-021–028 |
| S5 | Prior precision modulation | MPO-029–032 |
| S6 | Iterative posterior updates | MPO-033–037 |
| S7 | Prediction divergence and feature-space residual diagnostics | MPO-038–040 |

The executable registry in ore_predictive_orthography/registry.py contains the exact name, equation, biological interpretation and implementation source for every operation.

## Mathematical specification

### S1 — Bottom-up character evidence

For labelled patterns (x_j,c_j), estimate feature-wise mean and scale, standardize and normalize the descriptor, then build each grapheme prototype:

z_j = normalize((x_j - mu) / sigma)  
w_c = normalize(mean(z_j where c_j = c))

For a query descriptor z_i, compute prototype activation a_i(c)=z_i dot w_c, then:

P(c | known) = softmax(a_i(c) / T)

An uncertainty/confidence estimate r_i preserves an unknown outcome rather than forcing every patch into a known class: q_i(c)=r_i P(c|known) and q_i(UNK)=1-r_i. N2 descriptors summarize engineered visual-structure channels; they are not a direct neural or pixel-level grapheme code.

### S2 — Orthographic transition probabilities

Count grapheme exposures N(c) and ordered adjacent pairs N(a,b). Add Dirichlet/Laplace smoothing alpha, then calculate:

P(c) = (N(c)+alpha) / (sum_k N(k)+alpha*|C|)  
P(b|a) = (N(a,b)+alpha) / sum_j(N(a,j)+alpha)  
P(a|b) = (N(a,b)+alpha) / sum_i(N(i,b)+alpha)

The same ordered pair-count matrix supports both conditional directions: rows normalize forward transitions, columns normalize reverse transitions. A fingerprint identifies the frozen transition model.

### S3 — Predictions from neighboring grapheme beliefs

For position i, left and right messages are:

m_i_left = b_(i-1) @ P(next|current)  
m_i_right = P(previous|next) @ b_(i+1)

The N3 recurrent update integrates available neighbor messages with a smoothed unigram prior and normalizes the combined support. Boundary messages are explicitly marked unavailable; no fictitious neighbor is added. The master result separately exposes both message matrices and availability masks.

### S4 — Attention-dependent sensory precision

Visual uncertainty is categorical entropy normalized by class count:

H_i = -sum_c q_i(c) log(q_i(c)+epsilon)  
U_i = H_i / log(max(|C|,2))

N4 combines uncertainty, classwise prediction error and symmetric KL disagreement between sensory and predicted distributions to construct bottom-up salience. Optional task relevance, class goals and seed attention add top-down relevance. Competitive attention uses a temperature-softmax, a uniform coverage floor and local smoothing.

pi_s_i = clip(pi_s_0 * (0.5+0.5*r_i) * (1+g_a*N*a_i), pi_min, pi_max)

Here r_i is segmentation reliability and a_i is the spatial attention share. N4 also preserves class-selective attention gates.

### S5 — Prior precision modulation

Prior and sensory precision are separate control signals:

pi_p_i = clip(g_p*(0.5+U_i)*(2-r_i)*(1+g_t*t_i), pi_p_min, pi_p_max)

Predictive gain increases with sensory uncertainty/unreliability and may increase with top-down relevance. These values are configurable engineering hypotheses, not measured physiological constants.

### S6 — Iterative posterior updates

The posterior proposal combines precision-weighted sensory and contextual evidence in log space:

qhat_i = softmax(pi_s_i*g_i(c)*log(q_i(c)) + pi_p_i*log(p_i(c)))

Damped inference settles recurrent beliefs:

b_i^(k+1) = (1-rho)*b_i^k + rho*qhat_i

The iteration stops at a configured change tolerance or a maximum iteration count. N4 creates a new attended state and does not alter N3's input arrays.

### S7 — Prediction divergence and feature-space residual diagnostics

Candidate-wise mismatch is retained as:

e_i(c) = b_i(c) * (log(b_i(c)+epsilon) - log(p_i(c)+epsilon))

The master processor also computes symmetric KL divergence between visual evidence and contextual priors. In standardized descriptor space:

r_i_context = z_i - sum_c p_i(c)*w_c  
r_i_attended = z_i - sum_c b_i(c)*w_c

The UNK class has no learned visual prototype, so known-class posterior mass is renormalized before the attended expected feature vector is calculated. If known-class mass is zero, the expected feature vector is zero. A precision-weighted KL scalar is reported as an energy proxy, not as validated biological free energy.

## Biological reference implementation

Each of the seven stage-level pools contains configurable LIF neurons and receives an external stage drive, feedforward current from connected pools, recurrent current, global inhibition and spike-triggered adaptation. A current-based membrane update is:

V(t+dt) = V(t) + (dt/tau_m) * (E_rest - V(t) + R_m*I_syn(t))

When voltage reaches threshold, a spike is counted, voltage resets and a refractory interval is applied. Adaptation and a smoothed firing-rate state decay exponentially. Outputs include voltage (mV), synaptic current (pA), firing rates (Hz), spike counts and simulation time (ms).

This generic network is not anatomically mapped to seven brain regions. The seven functional stages are not claimed to be seven literal neural modules, and the listed mathematical functions are not claims that neurons explicitly execute software equations. Stronger biological claims require empirical fitting and independent neural/behavioral evaluation.

## API

Example:
- Fit N3 with labelled reference patches and grapheme sequences.
- Construct PredictiveOrthographyProcessor(fitted_n3).
- Call process(N3Sequence(...), task_relevance=optional_vector).
- Inspect result.stage_trace, result.operation_trace, result.stage_outputs, result.posterior_beliefs and result.biological.

Set simulate_biology=False to run only the functional path. The full registry and functional outputs remain available.

## Validation

Run focused tests with python -m pytest tests/test_predictive_orthography.py -q; run all tests with python -m pytest. Tests cover registry completeness, probability normalization, boundary-aware neighbor messages, finite residuals, LIF population output and separation of functional and biological tracks. Software tests establish code behavior only; they do not establish human reading accuracy or biological validity.
