# ORE N9 — Integrated Neurocomputational ORE

**Specification:** Proposed N9 integrated architecture, v1.0  
**Parent stages:** Newer ORE N1–N8 implementation line  
**Algorithm size:** 120 ordered operations across 10 computational subsystems  
**Status:** Architecture proposal; software and independent scientific validation are pending  
**Compatibility:** Additive integration layer. It must not silently replace or mutate N1–N8, the earlier ORE A–J baseline, or frozen model checkpoints.

---

## 0. Scope, naming and scientific status

The archived `ORE Neurocomputational Development Roadmap v1.0` uses a different phase register: in that document, N6 is mechanism instrumentation/ablations, N7 is the Stage A bridge/demo, and N8 is independent neurocognitive and real-corpus validation. The newer implementation line developed afterward uses N6 for hierarchical sequence/lexical access, N7 for predictive/contextual processing, and N8 for lexical-semantic access. This specification follows the **newer N1–N8 implementation line**. The archived independent-validation gate remains necessary and is not converted into a processing module.

N9 is not another isolated recognizer. It is a system-level controller and recurrent-inference layer that coordinates:

- N1: early visual feature extraction;
- N2: visual organization and distributed grapheme hypotheses;
- N3: recurrent orthographic prediction and error;
- N4: attention allocation and precision control;
- N5: explicitly authorized experience-dependent learning;
- N6: sequence decoding, boundaries and lexical access;
- N7: hierarchical predictive/contextual inference;
- N8: exact-form lexical-semantic access and explicit relation composition;
- proposed N9 extensions: phonological interface, syntactic hypothesis construction, proposition composition, discourse/situation models, integrated feedback, and system-wide decision/learning governance.

N9 accepts either an already prepared sequence of visual patches or an image/text-region input through a separately validated adapter. It must preserve the original image, all patch alternatives, uncertainty, sequence order and source provenance. It must keep bottom-up visual evidence distinct from learned orthographic priors, semantic knowledge and discourse context.

### Scientific interpretation rules

1. Neural systems operate through cascaded, parallel and recurrent processes. The numbered order below is an inspectable software execution schedule, not a claim that the brain waits for each operation to finish before beginning the next.
2. Some operations have plausible biological analogues (oriented filtering, normalization, population coding, recurrent prediction, attention, memory and plasticity). The exact neural implementation and anatomical localization are often debated.
3. Equations below are computational abstractions. A softmax, matrix multiplication, explicit graph, parser or optimizer is a software implementation choice; it is not asserted to be a literal neural operation.
4. N8's supplied semantic entries and relation evidence are evidence sources, not truth guarantees. N9 must not invent an unprovided sense, relation, syntactic dependency or factual claim and report it as verified.
5. Inference is read-only. Learning is a separate, explicitly authorized operation that creates a versioned child model and never silently modifies the active/frozen parent.
6. Passing unit tests or synthetic demonstrations establishes software behavior only. Real-script generalization and biological correspondence require independent evidence.

## 1. Global state and notation

Let:

- `I` be the source image or image region;
- `P = {p_i}` be the ordered set of candidate patches/regions;
- `x_i` be the feature vector for patch `i`;
- `q_i(g)` be the visual probability of grapheme `g` at patch `i`;
- `r_i` be patch/segmentation reliability;
- `a_i` be attention allocated to patch `i`;
- `πᵥ,i` and `πₚ,i` be sensory and prior precision proxies;
- `B` be the sequence-level grapheme belief state;
- `W` be lexical candidates and their word/sense hypotheses;
- `Φ` be optional phonological-form hypotheses;
- `T` be syntactic/constituency/dependency hypotheses;
- `S` be semantic senses and concept hypotheses;
- `G` be a typed graph of concepts, propositions and explicitly supported relations;
- `D` be discourse/context state;
- `M` be a read-only snapshot of long-term model/memory resources during inference.

The integrated latent state is:

`Z = {P, x, q, B, W, Φ, T, S, G, D}`.

A useful global objective is a factorized posterior:

`p(Z | I, C) ∝ ψvisual(I, x) · ψgrapheme(x, q) · ψposition(q, B) · ψlexical(B, W) · ψphonology(W, Φ) · ψsyntax(W, T) · ψsemantics(W, T, S, G) · ψcontext(S, G, D, C)`.

This factorization is a design proposal. Each factor must be independently inspectable and ablatable. The software may approximate inference using message passing, beam search, dynamic programming or variational updates; no single inference algorithm is claimed to be the brain's definitive implementation.

---

# PART A — Input orchestration and system-state initialization

**Operations 1–12**  
**Goal:** Accept a visual-language observation, establish its provenance and task, initialize independent evidence channels, and create a coordinated execution state without prematurely collapsing ambiguity.

### 1. Validate the incoming observation contract

Check that the source image, patch sequence or feature representation exists and has a declared schema. Reject non-finite arrays, invalid dimensions, corrupted image metadata and unsupported tensor shapes. Do not infer that a missing field has a default biological value.

**Mathematical operation:** Shape and domain constraints; for numeric input `X`, require `X ∈ R^(H×W)` or the declared tensor shape and `∀x∈X: finite(x)`. Validate pixel range when applicable: `0 ≤ I(x,y) ≤ 1`.

### 2. Identify the input pathway and script profile

Determine whether the request is a single glyph, a word, a line, multiple text regions or an ordered patch sequence. Load the declared script profile, reading direction, grapheme conventions, Unicode policy and known layout conventions. Latin, Arabic, vertical scripts and mixed-script text must not share hidden left-to-right assumptions.

**Mathematical operation:** Select a profile-conditioned transform `R_s`; for a right-to-left horizontal script, the logical coordinate can be defined as `u_i = -x_i`, while physical coordinate `x_i` remains preserved. Profile choice is metadata, not a learned inference.

### 3. Preserve the raw observation and source identity

Create an immutable input snapshot containing original pixels, file/record IDs, patch bounds, source/license metadata, timestamps and any supplied labels. Every later transformation must point back to this snapshot.

**Mathematical operation:** Compute a content digest `h_I = SHA256(serialize(I, metadata))`. This is an integrity operation, not a neural computation.

### 4. Construct explicit missing-data and occlusion masks

Identify pixels/patches that are absent, occluded, cropped or unusable. Preserve the difference between “not observed” and “observed with low activation.” Do not replace missing values with a plausible character.

**Mathematical operation:** Define mask `m(x,y)∈{0,1}` and valid-support ratio `v = Σm / (H·W)`. Reliability is reduced when support is missing, but the missing region remains explicitly marked.

### 5. Establish spatial and temporal coordinates

Attach physical coordinates, patch bounds, acquisition order and script-aware logical order. If eye-tracking or time-resolved data are supplied, retain gaze/time separately from text order. Do not infer eye movements from an ordinary static image.

**Mathematical operation:** Normalize non-degenerate physical coordinates: `p_i = (u_i - min(u)) / (max(u)-min(u)+ε)`. Keep an `ordinal_index` as a distinct field.

### 6. Estimate initial observation reliability

Combine supplied segmentation confidence, contrast quality, blur, clipping, occlusion and source quality into an initial reliability estimate. These are measurable input-quality cues, not semantic correctness.

**Mathematical operation:** `r_i = clip(∏_k r_ik^(w_k), 0, 1)` or a calibrated logistic model `r_i=σ(wᵀf_i+b)`. If calibration data are unavailable, mark reliability as heuristic.

### 7. Initialize the system's latent-state container

Create separate storage for visual features, grapheme hypotheses, sequence positions, lexical candidates, phonological candidates, syntax, semantics, discourse and errors. Use immutable snapshots for externally owned objects.

**Mathematical operation:** Initialize probability vectors with declared priors, for example `q_i^(0)(g)=1/|G|` only when no measured prior exists. Record whether a value is observed, learned, caller-supplied or a neutral initialization.

### 8. Validate compatibility of upstream N1–N8 contracts

Check schema versions, model fingerprints, vocabulary order, script profile compatibility and required fields. N9 must reject incompatible upstream objects rather than silently reshaping or relabeling them.

**Mathematical operation:** Compare schema IDs and SHA-256 fingerprints; verify vector/matrix dimensions such as `W_N6.classes = W_N5.classes` wherever that contract is required.

### 9. Snapshot all read-only model resources

Snapshot N1/N2 encoders, N3 transitions, N5 model state, N6 lexical resources, N7 configuration/context and N8 semantic lexicon. Record their versions and provenance before inference starts.

**Mathematical operation:** `h_M = SHA256(serialize(M))`. After inference, recompute the digest and require equality. Any mismatch signals mutation or inconsistent serialization.

### 10. Encode the task and operating mode

Represent whether the task is recognition, reading, transcription, lexical access, semantic interpretation, comparison or a caller-specified operation. The task may influence attention and output format but must not be treated as proof of the expected answer.

**Mathematical operation:** Convert task category to a one-hot or bounded vector `t`; task-conditioned priors can be represented as `p(Z|t)` with an explicit, configurable task gain.

### 11. Load contextual and working-memory snapshots

Read caller-supplied prior sentences, domain, topic, dialogue state, recent context or task goals. Context remains a distinct channel; it is not merged into visual evidence. Memory is read-only during recognition.

**Mathematical operation:** Decay recent activation: `m_k(t)=m_k(t-1)·exp(-Δt/τ_k)`. Combine with current context using a declared mixture `c = λc_recent +(1-λ)c_long`.

### 12. Build the execution dependency graph and launch eligible branches

Construct the directed dependency graph for all modules. Run independent computations concurrently when their inputs are available: low-level visual filtering, metadata checks, context loading and model validation. Downstream stages wait only for their required dependencies, not for unrelated branches.

**Mathematical operation:** Represent execution dependencies as a directed acyclic graph `G_exec=(V,E)`. A node is schedulable when all predecessors have completed; recurrent inference edges are handled in the iterative loop, not as an unbounded task cycle.

**Output of Part A:** Validated observation, profile, task state, provenance, model snapshots, initial uncertainty and execution graph.

---

# PART B — Early visual processing and visual organization

**Operations 13–24**  
**Goal:** Convert pixels into reliable, spatially organized visual evidence while preserving uncertainty about boundaries and character identity.

### 13. Convert luminance into an ink/salience signal

Convert RGB to a declared luminance representation, or preserve color channels if the experiment requires them. For dark text on a bright background, create a polarity-consistent ink-strength map. Avoid destructive thresholding as the only representation.

**Mathematical operation:** `Y=0.2126R+0.7152G+0.0722B`; for dark-ink convention, `I_ink=1-Y`. For inverse polarity, estimate the background polarity from the sample rather than assume it.

### 14. Compute local center-surround contrast

Approximate early contrast sensitivity by comparing activity at a location with activity in its surround. Use multiple spatial scales so thin strokes, thick strokes and local gaps can all be represented.

**Mathematical operation:** Difference-of-Gaussians response `D_σ1,σ2 = G_σ1*I - k(G_σ2*I)`, where `σ2>σ1`; retain ON-like and OFF-like channels separately if polarity is relevant.

### 15. Normalize local contrast

Reduce variation caused by broad illumination and contrast differences. Keep the original luminance map, normalized map and normalization parameters for later analysis.

**Mathematical operation:** `I_n=(I-μ_local)/sqrt(σ²_local+ε)` or divisive normalization `r_i=f_i/(ε+Σ_j w_ij|f_j|)^α`.

### 16. Extract orientation-selective energy

Apply a set of oriented filters to detect strokes, edges and line junctions. A quadrature pair gives a phase-tolerant local energy estimate. These filters are an engineering analogue of orientation-sensitive visual responses, not fitted biological receptive fields.

**Mathematical operation:** `E_(θ,λ)=sqrt((I_n*g_even(θ,λ))²+(I_n*g_odd(θ,λ))²)`. Use multiple orientations `θ` and wavelengths `λ`.

### 17. Separate spatial-frequency bands

Extract coarse, medium and fine structure. Coarse bands help locate regions and broad shapes; fine bands preserve narrow strokes and small diacritics. No one band should erase the others.

**Mathematical operation:** Gaussian/Laplacian pyramid `L_s=G_s-G_(s+1)` or a filter-bank transform `F_s=K_s*I_n`.

### 18. Detect edge, contour and junction evidence

Build candidate edges and junctions from local gradients and orientation continuity. Preserve multiple contour hypotheses where strokes touch, cross or become occluded.

**Mathematical operation:** Gradient magnitude `M=√(I_x²+I_y²)`; orientation `θ=atan2(I_y,I_x)`; contour continuity can use a local compatibility score `c_ij=exp(-d_ij²/(2σ_d²))·exp(-Δθ_ij²/(2σ_θ²))`.

### 19. Pool local features into shape descriptors

Aggregate neighboring filter responses while preserving both average activity and strong local responses. Use multiple pooling sizes to retain small diacritics as well as larger grapheme structure.

**Mathematical operation:** Mean pooling `μ_R=|R|⁻¹Σr`; max pooling `m_R=max(r)`; normalize concatenated descriptor `x=x_raw/(||x_raw||₂+ε)`.

### 20. Construct visual population features

Pass descriptors through N1/N2 visual encoders. Compare them with learned or caller-supplied prototypes and retain a vector of candidate affinities instead of committing prematurely to one symbol.

**Mathematical operation:** Cosine similarity `s_g=(x·w_g)/(||x||||w_g||+ε)`; prototype activation `q(g)=softmax(s_g/T_v)`.

### 21. Generate alternative region/patch segmentation hypotheses

Propose candidate text regions, stroke groupings and patch boundaries. Each candidate has a span, boundary confidence and visual evidence. Keep multiple possible segmentations when the image is ambiguous.

**Mathematical operation:** Boundary probability `b_k=σ(z_k)`; candidate segmentation score `L(P)=Σ_i log P(x_i|g_i)+Σ_k log P(boundary_k)`. Retain the top `K` alternatives with beam search or a segmentation lattice.

### 22. Estimate figure-ground and stroke grouping

Group fragments that plausibly belong to the same symbol. Use proximity, orientation, continuity, common region and script-specific diacritic geometry. Do not assume that connected pixels always form one grapheme.

**Mathematical operation:** Pair affinity `A_ij=w_dD_ij+w_oO_ij+w_cC_ij+w_rR_ij`; cluster through a graph cut, spectral clustering or connected-component proposal. Treat these as alternatives.

### 23. Align visual fragments with candidate grapheme regions

For each segmentation hypothesis, aggregate the fragments that would form a patch. Keep alternative alignments and attach their confidence to the resulting candidate distribution.

**Mathematical operation:** Soft assignment `z_ik=softmax_k(a_ik/T)`; expected feature `x_i=Σ_k z_ik x_ik`. Alignment entropy indicates uncertainty.

### 24. Emit the N2 visual evidence package

Return visual probabilities, shape features, region/patch maps, candidate segmentation paths, confidence and unknown/novelty estimates. Do not yet replace visual probabilities with lexical or semantic expectations.

**Mathematical operation:** Check `Σ_g q_i(g)=1`, `q_i(g)≥0`, and calculate entropy `H_i=-Σ_g q_i(g)log(q_i(g)+ε)`. Preserve raw and normalized evidence separately.

**Output of Part B:** Multiscale visual features, contour/shape descriptors, segmentation alternatives and per-region visual probability distributions.

---

# PART C — Position coding, orthographic populations and sequence hypotheses

**Operations 25–36**  
**Goal:** Represent character identity and character order as distinct but interacting information channels.

### 25. Create the grapheme candidate vocabulary

Load the script-specific grapheme inventory and reserve an explicit unknown state. In scripts with combining marks or contextual forms, use grapheme-cluster conventions rather than treating every Unicode code point as an independent character.

**Mathematical operation:** Build a class index `G={g_1,…,g_K,UNK}`. Validate uniqueness and stable ordering. Unicode normalization is a canonicalization operation, not a semantic rewrite.

### 26. Compute distributed grapheme activations

Use N2 visual features and population prototypes to compute an activation for each grapheme. Retain top-k alternatives and the complete probability vector for the inference path.

**Mathematical operation:** `a_ig=β·cos(x_i,w_g)+b_g`; `q_i(g)=exp(a_ig/T)/Σ_h exp(a_ih/T)`.

### 27. Estimate novelty and unknown probability

Compare a patch with known prototypes. When all known candidates fit poorly, assign probability to `UNK` rather than forcing a known grapheme. The novelty threshold must be calibrated on held-out, writer/font-disjoint samples.

**Mathematical operation:** `ν_i=1-max_g cos(x_i,w_g)`; `q_i(UNK)=σ((ν_i-τ)/T_u)` or use a calibrated OOD model. Renormalize known-class mass to `1-q_i(UNK)`.

### 28. Build absolute-position coding (A channel)

Represent the estimated absolute location of a grapheme within the text line or region. Absolute position can be useful for initial/medial/final position effects, while remaining separate from character identity.

**Mathematical operation:** For slot center `μ_k`, `A_ik=exp(-(p_i-μ_k)²/(2σ_k²))`. Normalize slot activation only where a probability distribution is intended.

### 29. Build relative/open-bigram coding (B channel)

Represent directed relations between candidate graphemes at two locations. A relative code should preserve which candidate precedes which other candidate and an estimate of their spacing/gap.

**Mathematical operation:** `Δp_ij=p_j-p_i`; `B_ij(g,h)=q_i(g)q_j(h)K(Δp_ij)`. Preserve direction; `B_ij(g,h)` is not identical to `B_ji(h,g)` if directional features or context are included.

### 30. Build distributed-overlap coding (O channel)

Represent a candidate's activation across several ordinal slots with overlapping tuning functions. This supports displacement tolerance without discarding order information.

**Mathematical operation:** `O_ik=Σ_j q_j(g)exp(-(rank_j-μ_k)²/(2σ_o²))`. Normalize or retain activation mass explicitly; do not mix normalized probabilities and unbounded activations without recording the conversion.

### 31. Preserve and compare A/B/O channels

Keep absolute, relative and overlap representations separately addressable. N9 may compute a combined representation, but the components must remain available for ablation and counterfactual analysis.

**Mathematical operation:** `z_i=[A_i;B_i;O_i]`; optional fusion `z_f=W_AA+W_BB+W_OO`, with each weight group exposed in configuration. A learned fusion is trained only on training data.

### 32. Transform physical coordinates into script-aware logical order

For right-to-left scripts, vertical scripts and mixed-direction spans, transform physical coordinates into the declared logical reading order. Retain both the original physical position and the logical coordinate.

**Mathematical operation:** `u_i=f_script(x_i,y_i,writing_mode)` and `rank_i=rank(u_i)`. Validate with script-profile fixtures for RTL, vertical and bidirectional text.

### 33. Integrate N5 transition memory as a prior

Read the frozen N5 forward/reverse transition statistics and fast/slow prototype state. Apply configured smoothing and class alignment. N9 must not update N5 during recognition.

**Mathematical operation:** `T_f(g,h)=(C_f(g,h)+α)/(Σ_h C_f(g,h)+Kα)`; construct reverse conditional probabilities with their declared normalization. Validate rows/columns and fingerprint.

### 34. Generate sequence-level candidate paths

Construct top-k character/grapheme sequences from patch distributions, order codes and transition priors. Preserve non-word and unfamiliar sequences as legitimate hypotheses.

**Mathematical operation:** `Score(G)=Σ_i log q_i(g_i)+λ_fΣ_i log T_f(g_i,g_(i+1))+λ_p S_position(G)`. Retain the top `K_seq` paths using beam search or Viterbi/k-best decoding.

### 35. Reconcile sequence hypotheses with patch alignment

Score whether each candidate path is consistent with the candidate segmentation. Keep alternatives in which a boundary shift changes the number or identity of graphemes.

**Mathematical operation:** `P(G,P|I)∝P(I|G,P)P(G|A,B,O,T_f)P(P)`. Use log-sum-exp to combine alternative segmentation paths without numerical underflow.

### 36. Emit orthographic uncertainty and evidence decomposition

Output the ranked sequences, position-wise marginals, boundary uncertainty and A/B/O representations. Store a separate visual-fit score and orthographic-prior score.

**Mathematical operation:** For each candidate `G`, store `L_visual=Σ_i log q_i(g_i)` and `L_prior=Σ_i log T_f(g_i,g_(i+1))`; never collapse them into one undocumented score.

**Output of Part C:** Orthographic sequence hypotheses with multiple position codes, transition evidence and explicit uncertainty.

---

# PART D — Attention, precision and dynamic resource allocation

**Operations 37–48**  
**Goal:** Allocate additional computation where the evidence is uncertain, surprising, task-relevant or important to resolving competing interpretations.

### 37. Calculate local visual uncertainty

Estimate uncertainty for each patch/position. High uncertainty may justify more attention, but should not automatically imply that the patch is important to the current task.

**Mathematical operation:** `U_i=H(q_i)/log(K)` where `H(q_i)=-Σ_g q_i(g)log(q_i(g)+ε)`.

### 38. Calculate local prediction error

Compare N3's expected grapheme or feature evidence with current N2 observations. Keep classwise and feature-space residuals, since they measure different things.

**Mathematical operation:** `ε_i(g)=q_i(g)-p_i(g)` or classwise divergence `D_i=KL(B_i||prior_i)`. For a feature model, `r_i=x_i- x̂_i`.

### 39. Add externally supplied task and salience cues

Integrate caller-provided relevance, fixation/gaze (only if actually measured), region salience or task focus. Keep cue source and reliability. Do not fabricate eye position or neural attention data.

**Mathematical operation:** `t_i=clip(wᵀc_i,0,1)` with cue vector `c_i`; if a cue is missing, use a documented neutral/default path rather than infer it from the desired answer.

### 40. Construct attention logits

Combine uncertainty, residual magnitude, segmentation reliability and task relevance into an attention score. Use configurable weights to enable ablation.

**Mathematical operation:** `ℓ_i=w_uU_i+w_eE_i+w_sS_i+w_tt_i`, with each input normalized and each weight declared.

### 41. Allocate competitive attention

Normalize the scores across positions. A uniform floor prevents low-scoring patches from being permanently eliminated, preserving the possibility of later correction.

**Mathematical operation:** `a_i=(1-λ_u)softmax(ℓ_i/T_a)+λ_u/N`. Check `Σ_i a_i=1`.

### 42. Smooth or spatially regularize attention

Reduce unstable single-patch spikes when adjacent patches belong to the same region, while respecting word boundaries and script layout. Do not smooth across known region boundaries indiscriminately.

**Mathematical operation:** `a'_i=(1-γ)a_i+γΣ_jK_ij a_j`; require non-negative `K` and re-normalize. The kernel may be boundary-aware.

### 43. Estimate sensory precision

Convert observation reliability into a precision-like gain. Better segmented, clearer visual input should generally receive more sensory weight than highly corrupted input.

**Mathematical operation:** `πᵥ,i=clip(πᵥ,0·r_i·(1+g_a·N a'_i),πᵥ,min,πᵥ,max)`.

### 44. Estimate prior precision

Estimate how strongly contextual or orthographic expectations should influence the current update. Prior precision should not rise without bound simply because a model is confident.

**Mathematical operation:** `πₚ,i=clip(πₚ,0·f(U_i,context_quality_i),πₚ,min,πₚ,max)`. Use independently calibrated context-quality estimates.

### 45. Compute relative precision and channel gates

Normalize sensory, transition, lexical and semantic precision proxies so different numerical scales do not silently dominate one another.

**Mathematical operation:** `ρ_k=π_k/(Σ_jπ_j+ε)`. Keep one set of relative weights per inference factor or layer.

### 46. Weight residuals by precision

Prioritize mismatches that are both large and reliable. A large residual from an occluded/poorly segmented patch should not be treated as equivalent to one from a clean patch.

**Mathematical operation:** `ε̃_i=π_i·ε_i`; for Gaussian errors, energy `E_i=0.5·ε_iᵀΠ_iε_i`.

### 47. Pass attention state to lexical and semantic branches

Send a copy of attention and precision metadata to N6–N8. Use it to select which alternatives receive deeper scoring, while keeping the full original candidate list for audit.

**Mathematical operation:** Candidate budget `K_i=K_min+floor((K_max-K_min)·a'_i)` or a threshold on normalized posterior mass. Ensure the pruned mass is measured and recorded.

### 48. Reallocate attention after higher-level feedback

When N6/N7/N8 detects ambiguity, a lexical conflict or an unresolved graph node, send a request for additional computation to the relevant low-level region. The request changes computational priority, not the source image.

**Mathematical operation:** `ℓ_i^(t+1)=ℓ_i^t+κ·u_i^(high)`; then recompute `a_i^(t+1)=softmax(ℓ_i^(t+1)/T_a)` with a stop condition and a maximum iteration budget.

**Output of Part D:** A bounded attention allocation, sensory/prior precision estimates, weighted errors and an inspectable reprocessing schedule.

---

# PART E — Recurrent sequence decoding, lexical and phonological interfaces

**Operations 49–60**  
**Goal:** Convert visual sequence hypotheses into orthographic lexical candidates and, where the task requires, phonological candidates.

### 49. Initialize recurrent sequence beliefs

Initialize N3's recurrent state from N2 visual distributions and N5 transition priors. Keep a separate untouched copy of bottom-up probabilities.

**Mathematical operation:** `B_i^(0)=normalize(q_i)`; `B_visual` is immutable during the recurrent update.

### 50. Generate bidirectional contextual predictions

Use left and right neighbors to estimate candidate support. At sequence edges, use the declared unigram prior or boundary state rather than indexing outside the sequence.

**Mathematical operation:** `L_i(g)=Σ_hB_(i-1)(h)T_f(h,g)` and `R_i(g)=Σ_hB_(i+1)(h)T_r(h,g)`; combine using configured weights.

### 51. Update beliefs with a precision-weighted proposal

Combine visual likelihood and orthographic prior. Use damping to prevent unstable oscillation and preserve the pre-update state.

**Mathematical operation:** `b̂_i=softmax(πᵥ,i log q_i + πₚ,i log prior_i)`; `B_i^(t+1)=normalize((1-ρ)B_i^t+ρb̂_i)`.

### 52. Calculate forward/backward sequence marginals

Run forward and backward messages through the transition model to estimate the probability of each grapheme at each position given the complete sequence.

**Mathematical operation:** `α_i(g)=log e_i(g)+LSE_h[α_(i-1)(h)+log T(h,g)]`; `β_i(h)=LSE_g[log T(h,g)+log e_(i+1)(g)+β_(i+1)(g)]`; `P(g_i|I)∝exp(α_i+β_i)`.

### 53. Decode maximum and alternative paths

Produce a best path and k-best alternatives. An unfamiliar but visually supported string must survive even when it is absent from the current lexicon.

**Mathematical operation:** Viterbi recurrence `δ_i(g)=log e_i(g)+max_h[δ_(i-1)(h)+log T(h,g)]`; use a k-best extension to retain alternatives.

### 54. Re-estimate boundaries and segmentation alternatives

Combine supplied segmentation evidence with lexical and orthographic span support. The lexicon may help identify plausible boundaries but must not erase a visually supported non-word segmentation.

**Mathematical operation:** `P(Boundary_k=1|I,G)∝P(I|boundary)P(boundary|lexicon,script)`; score complete paths in a segmentation DAG and retain normalized top-k paths.

### 55. Perform N6 lexical access

Look up exact orthographic forms against the validated lexicon, using language/script and provenance constraints. Return matched entries and explicit unlexicalized fallback units.

**Mathematical operation:** Exact lookup is a deterministic index operation. Candidate posterior may be `P(w|G,C)∝P(G|w)P(w|C)` when alternatives are present. Exact equality alone does not prove semantic correctness.

### 56. Generate cautious morphological candidates

Where a script/language morphology resource is supplied, generate possible stems, affixes, clitics or inflectional analyses. This is a proposed N9 extension; do not apply generic prefix/suffix stripping as if it were validated morphology.

**Mathematical operation:** `P(morph|form,language)∝P(form|morph,language)P(morph|language)`. Retain source/rule IDs, and include a no-analysis candidate.

### 57. Generate optional grapheme-to-phoneme hypotheses

For reading-aloud or phonological tasks, map orthographic forms to one or more phonological forms using an explicitly versioned pronunciation resource or a separately trained grapheme-to-phoneme model.

**Mathematical operation:** `P(φ|G,script,language)=softmax(score_φ/T_φ)`. Preserve dialect and pronunciation variants rather than forcing one form.

### 58. Combine lexical and phonological routes without collapsing them

The ventral lexical route and a sublexical/phonological route can provide complementary evidence. Keep both route scores and let the task determine whether phonological evidence is relevant.

**Mathematical operation:** `P(w,φ|I)∝P(G|w)P(φ|w)P(w)` with separate route factors; a route mixture can be `P=λP_lex+(1-λ)P_sublex` where `λ` is learned/calibrated.

### 59. Rank word, morpheme and pronunciation alternatives

Rank candidates from visual fit, sequence coherence, lexicon prior and optional phonology. Do not hide the contribution of word frequency or allow it to overwhelm strong visual mismatch.

**Mathematical operation:** `L(c)=λ_vL_visual+λ_oL_orthographic+λ_lL_lexicon+λ_φL_phonology`; normalize using softmax only after scale calibration.

### 60. Apply lexical unknown and abstention states

Label each item as resolved, ambiguous, low confidence, unlexicalized or unknown. Maintain the top alternatives and evidence for every abstention.

**Mathematical operation:** Resolve only if `P_top≥τ_conf`, `P_top-P_second≥τ_margin` and source reliability passes its threshold. Otherwise return `ABSTAIN`/`AMBIGUOUS`/`UNKNOWN` with the ranked alternatives.

**Output of Part E:** Grapheme sequences, boundary alternatives, lexical entries, optional phonological hypotheses and explicit unknown/abstention states.

---

# PART F — Syntax, sentence structure and semantic composition

**Operations 61–72**  
**Goal:** Build candidate sentence structures and compose word senses into proposition-like representations. This is a proposed extension beyond current N8's exact-form semantic lookup and explicit relation graph.

### 61. Estimate lexical category and morphosyntactic features

For each lexical alternative, retrieve or infer candidate part-of-speech, inflection, agreement, case, gender/number/person or other script/language-appropriate features. Distinguish features directly supplied by the lexicon from inferred candidates.

**Mathematical operation:** `P(pos,features|w,context)` from a calibrated tagger or finite-state grammar. Keep a null/unknown feature state.

### 62. Generate constituent and dependency candidates

Create possible head-dependent relations, phrase boundaries and grammatical roles. For Arabic and other morphologically rich languages, morphology and clitic segmentation can be inputs to these candidates.

**Mathematical operation:** Candidate arc score `s(i→j,r)=f(h_i,h_j,e_r,features)`; arc probability `P(i→j,r)=softmax_r(s/T)`. A relation is a hypothesis until validated.

### 63. Score word-order and grammatical constraints

Apply language-profile-specific word-order, case and agreement constraints. These constraints must be soft when the language permits flexible order and must not be copied uncritically between scripts/languages.

**Mathematical operation:** `Score(T)=Σ arcs s_arc+Σ constraints λ_k·c_k(T)`; penalize violated constraints instead of banning every non-canonical structure.

### 64. Build a constituency or dependency structure

Run a bounded parser over the best lexical/morphological alternatives. Retain multiple parse trees when a sentence is ambiguous or incomplete.

**Mathematical operation:** `P(T|W)=softmax(Σ_{a∈T}s_a/T_p)`; use chart parsing, beam search or a constraint solver. Track pruned parse mass.

### 65. Assign candidate thematic roles

Identify candidate agent, patient, experiencer, recipient, instrument, location, source, goal and other roles supported by the predicate and the parse. A syntactic subject is not always an agent.

**Mathematical operation:** `P(role_k|predicate,argument,T,S)` from role compatibility scores. Normalize roles per predicate, and permit `UNRESOLVED_ROLE`.

### 66. Construct hierarchical phrase representations

Combine words into noun phrases, verb phrases, clauses and other language-specific units according to the candidate structure. Preserve the original word nodes and links.

**Mathematical operation:** Recursive composition `h_parent=f_comp(h_child1,…,h_childn, relation)`; in a symbolic implementation use typed constructors; in a vector model use a learned tensor/MLP composition function.

### 67. Represent negation, modality, quantification and scope

Identify operators such as not/never, possibility/necessity, all/some, tense/aspect and their scope. Do not flatten an operator into a bag-of-words feature.

**Mathematical operation:** Represent operator stack `O=[o_1,…,o_k]` and scope graph `o_i→span_j`; score candidate scopes with `P(scope|syntax,operator,context)`. Preserve alternative scope orders.

### 68. Detect connectives and clause relations

Identify explicit discourse and clause connectives: conjunction, contrast, cause, condition, concession, temporal sequence and explanation. Determine which proposition(s) each connective links.

**Mathematical operation:** `P(rel_type,source_prop,target_prop|connective,syntax,context)`. Require explicit text spans or a validated parser/annotator source for graph edges.

### 69. Resolve candidate reference and coreference

Link pronouns, repeated noun phrases and definite descriptions to possible antecedents. Keep unresolved or ambiguous references when multiple antecedents are plausible.

**Mathematical operation:** `P(antecedent=j|mention_i,features,discourse)=softmax_j(s_ij/T_c)`. Include a null antecedent option.

### 70. Compose predicate-argument propositions

Combine selected lexical senses, predicate frames, roles, modifiers and operator scopes into proposition records. Keep a proposition's source spans and each contributing sense ID.

**Mathematical operation:** Example typed form `P = Predicate(s) + {Role_k: Argument_k} + Operators + Modifiers`. A neural approximation may use `h_P=f(h_pred,Σ_k W_k h_arg,k,h_ops)`; symbolic execution should retain typed argument slots.

### 71. Construct semantic graph nodes and edges

Create nodes for concepts, entities, events, propositions, time, location and discourse units. Include only supported relation edges. Separate asserted relations from inferred candidate relations and mark their provenance.

**Mathematical operation:** `G_sem=(V,E,τ_V,τ_E)` with node/edge types `τ`; edge confidence may be `c_ij=c_source·P(source_sense)·P(target_sense)·P(relation)`. Validate type constraints.

### 72. Validate proposition completeness and preserve alternatives

Check that each proposition has a supported predicate and required argument fields, or mark it partial. Preserve competing senses, parses, roles and scope assignments for downstream contextual selection.

**Mathematical operation:** Completeness score `C_P=Σ required_fields_present/Σ required_fields`; proposition posterior `P(P|I)=Σ_{W,T,S}P(P|W,T,S)P(W,T,S|I)`. Do not normalize away the `PARTIAL`/`AMBIGUOUS` status.

**Output of Part F:** Ranked syntax trees, dependency/constituent hypotheses, semantic role assignments, proposition graph and compositional alternatives.

---

# PART G — Contextual, discourse and situation-model integration

**Operations 73–84**  
**Goal:** Relate the current sentence to preceding discourse, task, world knowledge and episodic context without confusing plausibility with truth.

### 73. Load discourse context and topic state

Read prior propositions, speaker/author identifiers, document section, conversation state and task topic when supplied. Distinguish source text from user-provided assumptions and system-generated hypotheses.

**Mathematical operation:** Maintain a context state `D_t={propositions, entities, temporal anchors, topic}` with stable IDs and provenance.

### 74. Retrieve relevant semantic memory

Retrieve semantic entries related to the activated concepts and predicates. Avoid unrestricted nearest-neighbor retrieval that returns loosely similar but irrelevant facts.

**Mathematical operation:** Similarity `sim(q,m)=cos(e_q,e_m)` or typed graph neighborhood scoring; candidate memory activation `a_m=exp(-d(q,m)/τ)·recency(m)`.

### 75. Retrieve episodic or document-local context

Where available, retrieve earlier propositions from the same text or interaction. Keep document-local references distinct from general world knowledge.

**Mathematical operation:** `score(m)=w_topic·sim_topic+w_entity·overlap_entity+w_time·compat_time+w_recency·decay`. Use source identity as a hard boundary when required.

### 76. Derive intent and domain marginals

Use N7's joint intent/domain posterior, then combine with current lexical and proposition cues. N7 intent/domain probabilities are contextual signals, not semantic truth.

**Mathematical operation:** `P(intent=i)=Σ_dP(i,d)` and `P(domain=d)=Σ_iP(i,d)`. Validate both marginal sums equal 1.

### 77. Compute candidate proposition-context compatibility

Compare each proposition with previous propositions, topic and task. A high compatibility score should support a candidate but should not override direct contradictory evidence.

**Mathematical operation:** `C(P,D)=w_e·entity_overlap+w_t·topic_similarity+w_r·relation_compatibility+w_s·sequence_continuity`.

### 78. Generate predictions for upcoming or missing information

Use syntactic, semantic and discourse constraints to predict possible next concepts, arguments or connective relations. This is anticipatory inference, not a license to insert unobserved words into the recognized transcript.

**Mathematical operation:** `P(z_next|D_t,P_t)=Σ_z P(z_next|z,D_t)q_t(z)`. Keep predictions in a separate `expected_next` channel.

### 79. Build temporal and causal candidate relations

Use explicit tense, aspect, temporal expressions and connectives to create candidate ordering or causal relations. Mere sentence order is insufficient evidence for causality.

**Mathematical operation:** Temporal order constraints `t_i≤t_j`; interval overlap and Allen-style relation scores; causal edge confidence requires an explicit connective, construction or external validated relation signal.

### 80. Construct a situation-model state

Represent entities, events, attributes, locations, times, goals and their relationships as a structured model of what the text describes. Distinguish text assertions from background knowledge.

**Mathematical operation:** Situation state `M_t=(E_t,V_t,R_t,T_t,A_t)`; update `M_(t+1)=Update(M_t,P_t)` using a typed operation (add, revise, retract, unresolved), not blind vector averaging.

### 81. Integrate multiple propositions into discourse coherence

Check whether neighboring and distant propositions fit together, refer to consistent entities and respect temporal/causal constraints. Maintain a coherence score and specific conflict explanations.

**Mathematical operation:** `C_disc=Σ_i w_i C_local(P_i,P_(i-1))+Σ_(i,j) w_ij C_global(P_i,P_j)`; measure entity-chain consistency and relation contradictions separately.

### 82. Compare current interpretation with retrieved knowledge

Retrieve corroborating or conflicting background knowledge only when the task permits it. Mark each item as external knowledge, not as evidence directly present in the text.

**Mathematical operation:** `L_world=log P(world_facts|knowledge)` as a compatibility factor; preserve `L_text` and `L_world` separately. If they disagree, expose the discrepancy instead of rewriting the text.

### 83. Estimate discourse-level uncertainty

Calculate ambiguity across senses, parses, antecedents, propositions and situation models. Local word certainty does not guarantee sentence or discourse certainty.

**Mathematical operation:** `H_global=-Σ_hP(h|I,D)logP(h|I,D)`; optionally calculate entropy per level and mutual information between levels.

### 84. Return the contextualized semantic state

Return a versioned discourse/situation model with proposition links, source spans, uncertainty and provenance. The original N6/N8 outputs remain unchanged.

**Mathematical operation:** Validate node/edge referential integrity, probability bounds and graph acyclicity where a graph type requires it. Hash the context snapshot and record the update delta.

**Output of Part G:** Contextualized proposition graph, situation model, entity references, temporal/discourse links and uncertainty.

---

# PART H — Hierarchical feedback, predictive coding and conflict resolution

**Operations 85–96**  
**Goal:** Let high-level lexical, semantic and discourse hypotheses send controlled predictions downward, while lower-level evidence can revise higher-level interpretations.

### 85. Build a hierarchy of predicted states

For each level, generate a prediction about the level below: discourse predicts likely proposition roles; semantics predicts compatible lexical senses; lexical access predicts orthographic forms; orthography predicts grapheme/feature patterns.

**Mathematical operation:** `ŷ_l = f_l(z_(l+1), context_l)` for levels `l`; preserve the generator and prediction target for each message.

### 86. Project semantic predictions onto lexical candidates

Use the current semantic frame and N7 context to assign contextual support to N8's alternative senses. Do not change exact spelling or the original visual posterior.

**Mathematical operation:** `P(s|w,D)∝P(w|s)P(s|D)`. Update sense scores only; the bottom-up orthographic evidence is immutable.

### 87. Project lexical and sequence predictions onto grapheme candidates

For a candidate word, compute the grapheme pattern it predicts and compare it with N2/N3 distributions. Alternative segmentations remain eligible when the word prediction conflicts with the image.

**Mathematical operation:** `P(G|w)=∏_iP(g_i|w,position_i)`; score compatibility `L_o=Σ_i log(q_i(g_i)+ε)`.

### 88. Predict lower-level visual-feature evidence

If a learned generative feature decoder exists, predict the expected visual feature map for each candidate word/grapheme. This is a proposed extension: the current N3/N7 class-belief algorithms do not implement a complete pixel-generative cortical hierarchy.

**Mathematical operation:** `x̂_i=Decoder(z_i)`; visual residual `r_i=x_i-x̂_i`. If no trained decoder is present, mark this operation `NOT_AVAILABLE` and do not substitute a prototype residual as if it were pixel prediction.

### 89. Compute layer-specific prediction errors

Compare each predicted layer with actual evidence: visual features, grapheme distributions, lexical candidates, semantic cues and context. Keep one error stream per layer.

**Mathematical operation:** `ε_l=y_l-ŷ_l`; categorical layers may use KL divergence `D_l=KL(q_l||p_l)`. Record raw and precision-weighted errors separately.

### 90. Weight errors by estimated precision

Use reliability/precision to regulate which errors drive updates. Low-confidence context should not dominate a high-quality visual observation.

**Mathematical operation:** `ε̃_l=Π_lε_l`; total error energy `E=0.5Σ_l ε_lᵀΠ_lε_l`.

### 91. Calculate the integrated free-energy-like objective

Combine prediction-error energy with divergence from declared priors. The weights are experimental hyperparameters and should be calibrated/ablated.

**Mathematical operation:** `F(q)=E_q[E(Z)]+β·KL(q(Z)||p(Z|C))`. A factorized approximation can use a mean-field posterior `q(Z)=∏_lq_l(z_l)`.

### 92. Revise posterior beliefs with a bounded update

Update candidate sequence, lexical, syntactic, semantic and discourse states in a controlled sequence or by message passing. Accept only updates that satisfy numerical and contract constraints.

**Mathematical operation:** `q_l^(t+1)=softmax(log q_l^t-η∇_(q_l)F)`; use damping and backtracking. For discrete factors, use loopy belief propagation or bounded beam re-ranking as an alternative.

### 93. Resolve conflicts between channels

When visual evidence and lexical/semantic expectation disagree, inspect the source of the conflict. Options include retaining multiple candidates, requesting another segmentation hypothesis, lowering prior precision, or abstaining.

**Mathematical operation:** Conflict `K=|L_visual-L_prior|` or Jensen–Shannon divergence between channel posteriors. A conflict policy maps `K`, reliability and posterior margin to a controlled action.

### 94. Request targeted reprocessing when useful

Select only the patches, boundaries, token candidates or semantic nodes that could alter the result. Do not rerun every module for a single localized ambiguity.

**Mathematical operation:** Expected value of computation `EVC(a)=E[loss_before-loss_after|a]-cost(a)`. Reprocess only when estimated EVC exceeds a declared threshold and a budget remains.

### 95. Iterate until convergence or budget exhaustion

Repeat prediction, comparison and bounded revision. Keep the best valid state and all accepted/rejected update records. Stop if posterior change and energy change are small, or if the computation limit is reached.

**Mathematical operation:** Stop when `max_l||q_l^(t+1)-q_l^t||₁<ε_q` and `|F_(t+1)-F_t|<ε_F`; otherwise stop at `T_max` and set a non-convergence flag.

### 96. Preserve evidence separation after feedback

Even after feedback, store bottom-up visual evidence, learned orthographic prior, lexical likelihood, semantic likelihood and discourse prior as separate channels. A final posterior must not conceal which source produced the result.

**Mathematical operation:** Store a factor ledger `L_total=Σ_k λ_kL_k` with all `L_k` and `λ_k`; validate that the total can be reconstructed from the ledger.

**Output of Part H:** Iteratively revised beliefs, layer-wise errors, conflict records, targeted reprocessing events and convergence diagnostics.

---

# PART I — Global integration, decision and output

**Operations 97–108**  
**Goal:** Produce a coherent system result while preserving evidence, ambiguity, alternatives, module status and provenance.

### 97. Assemble the integrated hypothesis graph

Unify the identifiers of graphemes, words, senses, propositions, entities, events and discourse relations. Use links between modules rather than copying and overwriting their data.

**Mathematical operation:** Construct a typed graph `G_N9=(V,E)` with referential-integrity constraints. Each node stores source module, confidence, source span and model version.

### 98. Reconcile candidate interpretations

Combine lexical, syntactic, semantic and discourse candidates into joint interpretations. Use marginalization across alternatives when possible, rather than forcing early top-1 decisions.

**Mathematical operation:** `P(H|I,C)=Σ_(W,T,S,D)P(H,W,T,S,D|I,C)`. In practice use k-best joint hypotheses or a factor graph approximation.

### 99. Evaluate internal consistency

Check that selected senses match orthographic forms, relation endpoints exist, syntactic roles are compatible with predicates, operator scopes are valid and discourse links do not contradict the candidate itself.

**Mathematical operation:** Constraint penalty `C(H)=Σ_jλ_j·violation_j(H)`; invalidate only hard schema/type errors. Linguistic plausibility violations remain scored and explainable.

### 100. Evaluate evidence support and contradiction

For every output claim, identify its supporting spans, lexicon source, relation evidence, contextual prior and any external knowledge used. Contradictory evidence must remain visible.

**Mathematical operation:** Evidence support `S(c)=Σ_iw_i·support_i(c)` and contradiction `K(c)=Σ_jv_j·contradict_j(c)`. Report both rather than only `S-K`.

### 101. Calibrate confidence by level

Calibrate visual, sequence, lexical, syntactic, semantic and proposition confidence independently on held-out data. A high word confidence is not a guarantee of high sentence-level correctness.

**Mathematical operation:** Temperature scaling `p'=softmax(logits/T)` or isotonic calibration; evaluate Brier score `BS=N⁻¹Σ(p-y)²` and ECE. Never fit calibrators on the final test set.

### 102. Decide whether to answer, preserve alternatives or abstain

Choose an output state per layer: resolved, ambiguous, partial, unknown, abstained or failed. Use task-specific risk thresholds. It is acceptable to return a partial parse or uncertain semantic graph.

**Mathematical operation:** Choose action `a*=argmin_a E[loss(a,H)|evidence]`; abstain if expected error cost exceeds abstention cost.

### 103. Construct the structured recognition result

Return visual observations, grapheme candidates, sequence hypotheses, lexical entries, optional phonological alternatives, syntactic parses, semantic graph, propositions, discourse state and feedback history.

**Mathematical operation:** Serialize to a versioned schema; ensure probabilities are normalized, IDs unique and every graph edge references existing nodes.

### 104. Construct task-specific outputs without losing the canonical result

A caller may request transcription, tokenization, semantic interpretation, reading aloud, structured knowledge or a short explanation. Derive each output from the same canonical state and preserve the canonical state unchanged.

**Mathematical operation:** `output_k=Render_k(Z_N9)`; the renderer is a deterministic mapping where possible. Pronunciation output uses the phonological hypothesis channel, not inferred semantics.

### 105. Produce explanations and traceable evidence paths

For every selected word/sense/proposition, provide a compact trace of the operations and evidence that influenced the choice. Do not fabricate a biological explanation from a software trace.

**Mathematical operation:** Backtrack selected hypothesis through factor graph ancestors; compute contribution ledger `ΔL_k=λ_kL_k`. A feature attribution method may be used only with its own limitations stated.

### 106. Handle failure and rollback

If an intermediate module fails, preserve the last valid state, label missing outputs and prevent invalid downstream composition. Do not turn exceptions into plausible empty results.

**Mathematical operation:** Transactional checkpoint `(state,hash,version)`; rollback to last valid checkpoint if invariant checks fail. Failure rate is recorded separately from abstention.

### 107. Estimate processing cost and allocate remaining budget

Measure latency, candidate counts, iteration counts and module cost. Use this to set future compute budgets without silently pruning evidence needed for correctness.

**Mathematical operation:** `cost=Σ_l(n_ops,l·c_l)`; resource scheduler optimizes `quality_estimate-cost_penalty` under fixed time/memory budgets. Log all pruning decisions.

### 108. Verify final output and immutable parent state

Recompute upstream fingerprints, schema checks, finite values, evidence preservation and operation order. N9 must return an explicit validity state, not just a result object.

**Mathematical operation:** Require `hash(N1…N8)_after=hash(N1…N8)_before`; validate all arrays finite and all distributions sum to one; validate trace completeness and graph references.

**Output of Part I:** One integrated N9 result with alternative hypotheses, evidence ledger, confidence/abstention, explanations, provenance and resource diagnostics.

---

# PART J — Controlled learning, consolidation and empirical acceptance

**Operations 109–120**  
**Goal:** Allow the integrated system to improve over time without contaminating inference, introducing unverified labels or claiming biological validity from software success.

### 109. Freeze the inference result before considering learning

Inference must not update model weights, transition counts, semantic lexicons or persistent context memory. Store the result first, then create a separate learning request only if the user or experiment protocol authorizes it.

**Mathematical operation:** Snapshot `M_t`; compute result `R_t`; verify `hash(M_t_after_inference)=hash(M_t_before_inference)`.

### 110. Detect candidate learning events

Identify repeated prediction errors, novel grapheme patterns, low-confidence confusions or verified corrections that may justify a learning update. An error alone is not a trustworthy label.

**Mathematical operation:** Novelty `ν=1-max_g q(g)`; error `e=1-q(g_gold)` only when an independently supplied gold label exists. Otherwise create a review candidate, not a training target.

### 111. Require independent verification and explicit authorization

Require a verified target, reviewer/source provenance and explicit authorization before plasticity. Maintain a no-learning path for ordinary recognition.

**Mathematical operation:** Boolean gate `learn_allowed = verified ∧ authorized ∧ provenance_valid ∧ split=train`. If false, reject learning.

### 112. Update a detached child representation

Clone the parent and apply a bounded prototype update to the child. The N5 Oja-style rule is a proposed population-learning approximation and must not overwrite the original checkpoint.

**Mathematical operation:** `Δw=η·q·y·(x-yw)`; `w_child=normalize(w_parent+clip_norm(Δw,δ_max))`.

### 113. Update transition and lexical exposure statistics

Increment exposure counts and transition statistics only from verified sequences. Keep grapheme counts, forward/reverse transitions, lexical frequencies and semantic evidence stores separately typed.

**Mathematical operation:** `C_(t+1)=C_t+ω·count(sequence)`; normalized transition update uses pseudocount smoothing. The input sequence is checked for order and provenance.

### 114. Consolidate recurring patterns cautiously

Promote a chunk or sublexical pattern only when it occurs sufficiently often and improves held-out prediction without increasing false recognition. Avoid turning a frequent but spurious fragment into a permanent word/semantic rule.

**Mathematical operation:** Minimum Description Length criterion `ΔMDL=L(model_new)+L(data|model_new)-[L(model_old)+L(data|model_old)]`; promote only if improvement survives held-out evaluation and confidence interval checks.

### 115. Apply stability, homeostasis and anti-forgetting controls

Bound prototype drift, monitor rare classes and protect previously validated representations. A new pattern must not degrade unrelated scripts or vocabulary without that trade-off being measured.

**Mathematical operation:** Parent-relative drift `d=||w_child-w_parent||₂`; constrain `d≤δ`; class exposure can regulate gain `g_c←clip(g_c·exp(η(h*-h_c)),g_min,g_max)`.

### 116. Update version, provenance and immutable model fingerprints

Assign the child a new version, record the parent version, training examples, reviewer IDs, source references, script profile, config and update statistics. Do not create an untraceable in-place update.

**Mathematical operation:** Canonical serialization plus SHA-256 fingerprint; model lineage `parent_hash→child_hash`. Every model artifact must be reproducible from its manifest.

### 117. Run mechanism ablations and counterfactual tests

Disable recurrence, one direction of context, position channels, attention, semantic priors, syntax, or discourse prediction one at a time. Use identical samples and frozen model states to isolate effects.

**Mathematical operation:** `Δmetric_k=metric_full-metric_ablated(k)`; calculate paired confidence intervals and effect sizes. Ablation effects show software contribution, not biological necessity.

### 118. Evaluate on leakage-resistant real corpora

Use rights-cleared, provenance-tracked datasets. Split by writer, font family, source document and string family. Test Latin and Arabic, including RTL, morphology, diacritics and contextual forms; expand to other scripts only with appropriate profiles and data.

**Mathematical operation:** Report CER, WER when word recognition is in scope, top-k accuracy, parse/role F1, semantic relation precision/recall, calibration, risk/coverage, latency and per-condition confidence intervals. Run multiple fixed seeds for stochastic variants.

### 119. Apply separate software, task and biological acceptance gates

A passing test suite is a software gate. Real OCR/linguistic performance is an empirical task gate. Human/neural correspondence is a separate biological gate requiring preregistered, quantitative predictions and independent behavioral/neurophysiological evidence.

**Mathematical operation:** A conjunctive release gate `accept = G_software ∧ G_task ∧ G_biological` only when all three are explicitly in scope. Otherwise report each independently as `PASS`, `FAIL`, `PENDING` or `UNPROVEN`.

### 120. Return the versioned N9 integrated result and model state

Package the full N9 output, trace, source evidence, selected and alternative hypotheses, confidence, abstentions, diagnostics and model lineage. Return a child model only if learning was separately authorized and accepted.

**Mathematical operation:** Validate schema, graph integrity, normalized probabilities, immutable-parent hashes, ordered trace `1…120` and result digest. Mark scientific validation status explicitly.

**Output of Part J:** Integrated recognition/interpretation result, complete trace, optional detached child model, empirical evaluation status and provenance manifest.

---

# 2. Core mathematical model for the integrated system

The 120 operations can be summarized by an iterative hierarchical model. Let `y_l` be the observed state at level `l`, `z_(l+1)` the next higher-level latent state and `f_l` the top-down prediction mapping.

**Prediction:**

`ŷ_l = f_l(z_(l+1), c_l)`

**Prediction error:**

`ε_l = y_l - ŷ_l`

**Precision-weighted error:**

`ε̃_l = Π_l ε_l`, where `Π_l` is a precision matrix or scalar gain.

**Joint objective:**

`F(q) = E_q[ 1/2 Σ_l ε_lᵀΠ_lε_l ] + β·KL(q(Z)||p(Z|C)) + λ·C_constraints(q)`

Here:

- the first term measures precision-weighted mismatch;
- the KL term regularizes the inferred state toward declared priors;
- `C_constraints` penalizes incompatible syntax, graph types, scope and provenance;
- `β` and `λ` are tunable engineering parameters, not neural constants.

A bounded approximate update is:

`q_(t+1) = normalize((1-ρ)q_t + ρ·softmax(log q_t - η∇F(q_t)))`.

A production system should use separate update rules for continuous feature states, categorical grapheme/sense states and discrete graph structures. This compact equation is a conceptual objective, not a claim that all brain subsystems optimize one globally accessible scalar function.

## 3. Information-flow rules

1. Bottom-up information may move from N1/N2 toward N6/N8/N9.
2. Top-down predictions may move from N7/N8/N9 toward lower levels, but they remain predictions, not observations.
3. N4 attention changes resource allocation and precision, not the identity of the source pixels.
4. N5 learning is separate from inference and requires explicit verification/authorization.
5. N8 semantic candidates require external lexicon provenance; N9 syntax/relations require explicit evidence, a validated grammar/parser or a labelled candidate status.
6. Every selected output keeps alternative hypotheses, unknowns, conflicts and abstentions.
7. Model, data, script profile, source, configuration and trace versions are recorded.

## 4. Proposed neural-system correspondences (hypotheses, not one-to-one mappings)

| ORE function | Candidate neural-system correspondence | What the software approximates | Main limitation |
|---|---|---|---|
| Early visual feature extraction | Retina/LGN/V1 and later visual areas | Contrast, orientation, frequency and pooling | Does not simulate cells, layers or retinal dynamics |
| Grapheme/word-form coding | Ventral occipitotemporal visual word processing | Distributed candidate populations and position codes | Prototypes are not measured neural populations |
| Orthographic prediction | Recurrent visual-orthographic pathways | Transition priors and error residuals | Current beliefs are not full pixel-generative neural states |
| Attention/precision | Frontoparietal and sensory gain modulation | Softmax allocation and precision weights | Parameters are not physiological gain measurements |
| Phonological interface | Dorsal temporoparietal/temporal-frontal pathways | Optional G2P candidate probabilities | Requires independent phonological resources and validation |
| Lexicosemantic access | Ventral temporal/temporal pole/inferior frontal pathways | Exact-form semantic lookup and sense ranking | External lexicon quality determines much of the result |
| Syntactic/semantic composition | Distributed frontotemporal language network | Parse candidates, role assignment and proposition graphs | Symbolic parsers are functional abstractions, not neuronal simulations |
| Discourse and situation model | Temporal, parietal, prefrontal and default-mode systems | Entity/event memory and proposition coherence | Does not reproduce episodic memory or the full DMN |
| Plasticity | Hebbian/Oja-like population adaptation | Verified child-model updates | Not a detailed synaptic/cellular model |

## 5. Suggested execution policy

The controller should not run all 120 operations as a rigid serial chain. Use three execution modes:

- **Feed-forward initialization:** operations 1–84 create the first structured interpretation. Independent visual, profile and context computations can run in parallel.
- **Recurrent refinement:** operations 85–96 send controlled feedback and perform a bounded number of update cycles.
- **Finalization/governance:** operations 97–120 validate, report and optionally learn. Learning runs only after inference has been frozen and a separate authorization gate has passed.

The controller should use explicit dependency edges, bounded candidate budgets, per-module timeouts, deterministic tie-breaking and a maximum iteration count. If a module is unavailable, the system should mark that channel unavailable instead of silently replacing it with an unrelated signal.

## 6. Required tests for a future N9 software implementation

### Contract and evidence integrity

- Every module receives schema-compatible inputs.
- N1–N8 inputs and model snapshots remain unchanged after inference.
- Every result field has source provenance and an evidence-channel label.
- Unknown/ambiguous states survive all downstream steps.
- Every graph edge references valid nodes and carries a support status.

### Numerical invariants

- Probability vectors are finite, non-negative and normalized.
- Attention sums to one and precision remains within declared positive bounds.
- Log-domain sequence inference avoids underflow.
- Iterative updates are bounded and report convergence or budget exhaustion.
- Confidence calibration is computed only on held-out data.

### Linguistic/orthographic adversarial tests

- Adjacent and distant letter transpositions;
- substitutions, insertions and deletions;
- ambiguous segmentation and touching characters;
- low contrast, blur, occlusion, skew, font and size changes;
- out-of-vocabulary words and novel strings;
- lexical ambiguity (same spelling, different senses);
- reversed agent/patient roles;
- negation, modality, quantifier and scope changes;
- anaphora/coreference ambiguity;
- contrast, condition, cause and temporal relations;
- Arabic RTL, diacritics, contextual forms, clitics and Unicode normalization;
- contexts that strongly predict the wrong word, to measure hallucination/false-correction risk.

### Ablation matrix

At minimum compare:

1. N1–N2 visual only;
2. visual + A/B/O position codes;
3. visual + N3 recurrent transitions;
4. visual + N3 + N4 precision/attention;
5. complete orthographic N1–N6 path;
6. N1–N6 + N7 contextual prediction;
7. N1–N7 + N8 semantic access;
8. full N9 syntax/semantic/discourse composition;
9. full N9 without top-down feedback;
10. full N9 without phonological route;
11. full N9 without discourse memory;
12. full N9 without one module at a time.

Use identical splits, seeds, input samples and calibration policy across conditions.

## 7. Scientific foundations and further reading

These papers support selected architectural hypotheses; none validates this N9 proposal as a whole:

1. Gagl et al. (2020), *An orthographic prediction error as the basis for efficient visual word recognition*. The authors report evidence for top-down orthographic expectations shaping early visual representations and an orthographic prediction-error signal. https://pmc.ncbi.nlm.nih.gov/articles/PMC7284316/
2. Ghuman, Fiez and Boring (2026), *The reading brain — Transforming vision into language*. Review of visual-to-language pathways and the role of ventral occipitotemporal cortex in connecting visual, phonological and lexicosemantic systems. https://pmc.ncbi.nlm.nih.gov/articles/PMC13491551/
3. He et al. (2025), *Diverse Frontoparietal Connectivity Supports Semantic Prediction and Integration in Sentence Comprehension*. Reports dynamic frontoparietal coupling during anticipatory prediction and bottom-up semantic integration. https://pmc.ncbi.nlm.nih.gov/articles/PMC11780348/
4. *From form to concept: a dynamic spatiotemporal brain network cascade supports language comprehension* (Cerebral Cortex, 2026). Reports a proposed cascade from occipitotemporal word processing through semantic retrieval, inferential, syntactic-semantic and concept-coherence networks. https://doi.org/10.1093/cercor/bhag097
5. *Distributed Sensitivity to Syntax and Semantics throughout the Language Network* (Journal of Cognitive Neuroscience, 2024). Reports distributed sensitivity to both syntactic and semantic information across a broad frontotemporal network. https://doi.org/10.1162/jocn_a_02164
6. Lambon Ralph et al. (2017), *The neural and computational bases of semantic cognition*. Reviews interacting semantic representation and semantic-control systems. https://www.nature.com/articles/nrn.2016.150

## 8. Final architectural interpretation

N9 should make ORE a coordinated visual-to-linguistic inference system in which:

- visual evidence remains traceable and independently inspectable;
- grapheme identity and letter order are represented separately;
- lexical, phonological and semantic alternatives can coexist;
- syntax determines how candidate meanings are composed;
- discourse and memory supply context without rewriting the observation;
- prediction errors and precision regulate bounded feedback;
- learning is explicit, verified and versioned;
- software correctness, task accuracy and biological validity are separate claims.

The intended achievement is not “a brain in Python.” It is a transparent, testable computational architecture in which each proposed mechanism has a mathematical definition, an ablation switch, an evidence source, a failure state and a validation experiment.
