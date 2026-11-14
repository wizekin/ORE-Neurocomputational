# Lexical Context and Predictive Processing — 40-operation registry

This document specifies the additive `ore_lexical_context` subsystem. It follows the 40 operations in `ore_lexical_context.registry.OPERATION_REGISTRY`, in order. Each operation has a stable ID (`LC-001`–`LC-040)), a responsibility, an explicit mathematical form, and a biological interpretation stored in code. The registry is executable metadata, not merely a prose checklist.

## Scope and evidence policy

The processor receives an explicit lexical inventory, optional morpheme inventory, optional grapheme-to-phoneme (G2P) rules, transition counts, and contextual feature evidence. It does not infer a dictionary, invent morphemes, or silently correct spellings. Every meaning and morpheme feature map can carry verification/provenance information. The mathematical functions below are computational hypotheses that make the implementation testable; they are not claims that the human brain evaluates these exact equations.

## The 40-operation execution sequence

| ID | Operation | Main computation | Biological interpretation |
|---|---|---|---|
| LC-001 | Retrieve accumulated word exposure | (n_w=n_{lexicon}(w)+n_{session}(w)) | Familiarity/exposure history influences readiness |
| LC-002 | Calculate logarithmic frequency strength | (F_w=\log(1+n_w)/\log(1+n_{max})) | Frequency-dependent graded population response |
| LC-003 | Construct lexical resting activation | (R_w=b_0+\alpha_FF_w) | Baseline excitability of lexical form populations |
| LC-004 | Adjust lexical activation thresholds | (\theta_w=\theta_0-\alpha_\theta F_w+\beta_AA_w) | Threshold modulation and adaptation |
| LC-005 | Generate frequency-conditioned activation | (a_w=\sigma((R_w+g_sS_w+g_rH_w-\theta_w-g_AA_w)/T)) | Stimulus-driven lexical population activation |
| LC-006 | Retrieve recent lexical activation history | (H_w=\{(a_k,t_k):k<t\}) | Recurrent/synaptic trace of recent activity |
| LC-007 | Calculate stimulus similarity | (S_w=1-d_{Lev}(input,w)/\max(|input|,|w|)) | Distributed orthographic feature match |
| LC-008 | Calculate temporal decay | (D_w=e^{-\Delta t/\tau_H}) | Time-dependent decay of short-term traces |
| LC-009 | Generate repetition-dependent facilitation | (P_w=g_P\log(1+n_{recent})/\log(1+n_{cap})\cdot D_w) | Repetition priming/facilitation |
| LC-010 | Apply repetition-related neural adaptation | (A_w(t)=e^{-\Delta t/\tau_A}A_w(t-1)+\eta a_w(t-1)) | Activity-dependent adaptation/refractoriness |
| LC-011 | Detect possible morphological boundaries | (B=\{(prefix,stem,suffix):input=prefix+stem+suffix\}) | Candidate decomposition of word-form patterns |
| LC-012 | Evaluate stem and affix candidates | (M=F_{stem}\prod_j(P_{affix,j}F_{affix,j})) | Joint stem/affix evidence |
| LC-013 | Retrieve morpheme-associated meanings | (\mu_m=feature\_map(m)) | Learned semantic associations of supplied morphemes |
| LC-014 | Evaluate morphological composition | (\mu_{comp}(f)=clip(\mu_{stem}(f)+\sum_j\eta_j\mu_{affix,j}(f),0,1)) | Distributed semantic feature combination |
| LC-015 | Integrate whole-word and decomposed representations | (\mu_w=q\mu_{whole}+(1-q)\mu_{comp}) | Co-activation of whole-form and decomposed routes |
| LC-016 | Generate grapheme-to-phoneme correspondences | (G2P(w)=\{\phi_{lex},\phi_{rule},...\}) | Interaction of orthographic and phonological representations |
| LC-017 | Calculate mapping consistency | (C_\phi=coverage\cdot mean(rule\_reliability)) | Reliability of learned mappings |
| LC-018 | Activate competing pronunciations | (\Phi=\{\phi_1,...,\phi_k\}) | Parallel activation of pronunciation candidates |
| LC-019 | Weight pronunciation alternatives | (P(\phi_i|w)=softmax(\log prior_i+\log reliability_i+\beta C_i)) | Differential evidence among phonological candidates |
| LC-020 | Send phonological evidence into lexical recognition | (a'_w=clip(a_w+g_\phi sim(\phi_w,\phi_{input}),0,1)) | Bidirectional phonology-to-lexicon feedback |
| LC-021 | Generate orthographic neighbors | (N(w)=\{v:d_{Lev}(w,v)\le d_{max},v\ne w\}) | Co-activation of similar word forms |
| LC-022 | Calculate neighborhood strength | (NS_w=\sum_{v\in N(w)}F_vS_v/(|N(w)|+\epsilon)) | Neighborhood density/frequency contribution |
| LC-023 | Calculate shared orthographic activation | (SO_w=\sum_v Dice(ngrams(w,v))a_v) | Shared letter/n-gram feature activity |
| LC-024 | Apply competitive inhibition | (a^{inh}_w=max(0,a_w-\lambda\sum_v sim(w,v)a_v)) | Lateral inhibition among candidates |
| LC-025 | Settle lexical competition | (a(t+1)=(1-\rho)a(t)+\rho\sigma(u+g_ea-g_iWa)) | Recurrent excitation and lateral inhibition |
| LC-026 | Detect semantic ambiguity | (K_w=\#\{supported\ meaning\ hypotheses\}\) | Co-activation of multiple senses |
| LC-027 | Generate prior probabilities for meanings | (P(m)=prior_m\,a_m/\sum_j prior_j a_j) | Frequency/access bias on sense candidates |
| LC-028 | Retrieve meaning-specific features | (\mu_m=\{f:x_f\}\) | Feature-specific semantic populations |
| LC-029 | Calculate contextual compatibility | (C_m=cosine(\mu_m,context)) | Context-sensitive semantic modulation |
| LC-030 | Select or retain competing meanings | (P(m|c)=softmax(\log P(m)+\beta_CC_m)) with confidence/margin gate | Competitive selection or unresolved alternatives |
| LC-031 | Construct the current sentence context | (c_t=normalize(c_{external}+\gamma_c\sum_{i<t}\mu_i)) | Distributed sentence/discourse state |
| LC-032 | Generate upcoming word predictions | (P(w_{t+1}|c_t,w_t)\propto n(w_t,w_{t+1})+feature\_prior(w|c_t)) | Predictive pre-activation of likely words |
| LC-033 | Calculate contextual predictability | (surprisal(w_t)=-\log(P(w_t|context_{t-1})+\epsilon)) | Unexpected input generates a larger surprise signal |
| LC-034 | Pre-activate expected features | (\mu_{pred}=\sum_wP(w|context)\mu_w) | Top-down semantic feature pre-activation |
| LC-035 | Modulate predictions by reliability | (r_{pred}=evidence\_coverage(1-H(P)/\log K)) | Precision depends on contextual certainty |
| LC-036 | Construct incoming semantic representation | (\mu_{in}=\eta\mu_{observed}+(1-\eta)\sum_mP(m|c)\mu_m) | Integration of bottom-up and lexical-semantic evidence |
| LC-037 | Calculate semantic prediction residual | (\epsilon_f=\mu_{in}(f)-\mu_{pred}(f)) | Feature-wise mismatch signal |
| LC-038 | Calculate precision-weighted error | (\epsilon_{\pi}=\pi_s/(\pi_s+\pi_pr_{pred}+\epsilon)\cdot\epsilon_f) | Precision-weighted prediction-error hypothesis |
| LC-039 | Update semantic interpretation | (\mu_{post}=clip(\mu_{pred}+\eta_e\epsilon_\pi,0,1)) | Recurrent semantic state is revised by incoming evidence |
| LC-040 | Update predictions for subsequent words | (c_{t+1}=normalize(c_t+\gamma\mu_{post});\ P_{t+1}=Predict(c_{t+1})) | Updated context changes later anticipatory activity |

The registry in Python is authoritative for exact labels and machine-readable equations. This table summarizes each operation for reading.

## Functional execution

`LexicalContextProcessor` applies the 40 operations to each token in order. Each trace row includes the operation ID, token index, token string, subsystem, and the computed intermediate output. A multi-token input therefore has (40\times N) trace rows.

The implementation uses:
- logarithmic frequency compression and sigmoid activation for lexical readiness;
- Levenshtein similarity and configurable edit-distance neighborhoods;
- exponential decay and saturating repetition facilitation, alongside a separate adaptation state;
- supplied prefix/root/suffix candidates, feature composition, and a confidence-weighted whole-word/decomposition blend;
- competing lexical/G2P pronunciation candidates with softmax weighting;
- n-gram Dice overlap, neighborhood drive, lateral inhibition, and damped iterative settling;
- prior/posterior sense probabilities, cosine context compatibility, and an abstention gate;
- supplied bigram transition counts, predictive semantic features, surprisal, reliability, feature residuals, and precision-weighted updates.

The lexical exposure state is session-local to a processor instance. Set `record_exposures=False` for read-only exposure tracking. No parameter-fitting or model-training occurs implicitly.

## Biological reference track

When `simulate_biology=True`, every functional operation also advances six recurrent populations:
`orthographic`, `lexical`, `morphological`, `phonological`, `semantic`, and `context_predictive`.

The reference uses a current-based leaky integrate-and-fire update, membrane leak, recurrent excitation, global inhibition, refractory periods, spike reset and a decaying adaptation current. It reports membrane voltage in mV, synaptic input current in pA, firing rates in Hz and simulation time in ms. Two LIF time steps are advanced per operation by default.

The functional and neural tracks are linked through an explicit mapping from operation outputs to population drives. This is a generic, deterministic research reference. It is not anatomically mapped, fitted to human recordings, or a claim that the brain executes software operations sequentially or uses these exact equations. Biological parameters are configurable hypotheses requiring empirical calibration.

## Quick start

```python
from ore_lexical_context import (
    LexicalContextProcessor, LexicalContextRequest, LexicalEntry,
    LexicalMeaning, MorphemeEntry, G2PRule,
)

lexicon = [
    LexicalEntry(
        "bank", "bank", frequency=25,
        meanings=(
            LexicalMeaning("river-bank", {"RIVER": 1.0, "LAND": 0.7}),
            LexicalMeaning("financial-bank", {"MONEY": 1.0, "INSTITUTION": 0.7}),
        ),
        verified_by="curated-example",
        source_reference="dataset://example/bank",
    )
]
processor = LexicalContextProcessor(lexicon)
result = processor.process(LexicalContextRequest(
    "example-001", text="bank", context_features={"RIVER": 0.9}
))
print(result.tokens[0]["selected_meaning"])
print(len(result.operation_trace))  # 40 for one token
print(result.biological["layers"]["lexical"]["mean_firing_rate_hz"])
```

Run the included demonstration with `python -m ore_lexical_context`. Run focused tests with `python -m pytest tests/test_lexical_context.py -q`.
