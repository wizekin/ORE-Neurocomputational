# ORE Final Lexical Architecture — seven stages / 36 operations

The additive `ore_lexical_access` package implements the seven stages of lexical access and morphology as an executable, numbered operation registry (`BLA-001`–`BLA-036`). It is designed to sit alongside the existing `ore_lexical_context`, `ore_reading`, and `ore_semantic` packages. It reuses the existing `LexicalEntry`, `LexicalMeaning`, and `MorphemeEntry` records from `ore_lexical_context`; no existing package or result contract is modified.

## Scope and evidence policy

The processor consumes an explicit lexical inventory and an explicit morpheme inventory. It does not create dictionary entries, infer undocumented meanings, or invent prefixes, roots, suffixes, or grammatical rules. A morphological split is admitted only when supplied prefix/suffix forms reconstruct the input exactly and the resulting stem resolves to a supplied lexical entry or root morpheme. Whole-word access remains a competing route; the algorithm never forces a decomposition.

The mathematical operations are engineering hypotheses for transparent computation. The biological track is a separate seven-population current-based leaky integrate-and-fire (LIF) reference. It is not a simulation of the complete human brain, not anatomically fitted, and not validated against human electrophysiology. Default values are demonstration parameters, not physiological constants.

## Stage overview

| Stage | Function | Operations | Main result |
|---|---|---:|---|
| S1 | Retrieve stored word identity | 5 | Ranked lexical-form candidates and identity posterior |
| S2 | Retrieve word meaning | 5 | Meaning alternatives, features, contextual compatibility and posterior |
| S3 | Retrieve grammatical category | 4 | Category posterior and ambiguity margin |
| S4 | Morphological decomposition | 5 | Exact whole-word and prefix–stem–suffix hypotheses |
| S5 | Retrieve stems and affixes | 4 | Component records, semantic features and category cues |
| S6 | Distinguish morphological alternatives | 5 | Ranked analyses and confidence/abstention decision |
| S7 | Activate lexical candidates incrementally | 8 | Per-character activation trajectories and final ranking |
| **Total** | | **36** | Ordered functional trace and biological reference trace |

## Detailed algorithm and mathematics

Let `x` be the normalized input word, `e` a lexical entry, `m` a meaning, `g` a grammatical category, `h` a morphological hypothesis, and `t` a character-arrival step. `softmax(z/T)` denotes a normalized exponential over alternatives; `d_Lev` is Levenshtein distance; `sim` is normalized string similarity; `cos` is cosine similarity.

### S1 — Retrieve stored word identity (BLA-001–BLA-005)

1. **Normalize the incoming word form (BLA-001).** Unicode NFC normalization and case folding make lookup stable without stripping diacritics: `x = casefold(NFC(input))`.
2. **Retrieve lexical-form candidates (BLA-002).** Search the supplied inventory for exact forms and bounded near-form candidates: `C(x) = {e : d_Lev(x, form(e)) <= d_max AND sim(x, form(e)) >= s_min}`.
3. **Calculate orthographic likelihood (BLA-003).** Convert edit distance into graded likelihood: `L(e|x) = exp(-lambda * d_Lev(x, form(e)) / max(|x|, |form(e)|, 1))`.
4. **Retrieve frequency and exposure priors (BLA-004).** Read frequency and exposure from the lexical record and compress the scale: `F(e) = log(1 + frequency(e) + exposure(e))`.
5. **Estimate stored-identity posterior (BLA-005).** Combine form likelihood and prior: `P(e|x) = softmax((log L(e|x) + beta_F * F(e)) / T_id)`. Multiple homographs and near forms are preserved.

**Biological reference.** The primary high-level analogy is the distributed ventral occipitotemporal / visual word-form network interacting with lexical representations. It is not a claim that one cortical location or neuron stores a complete word. Frequency and exposure values are software proxies for learned familiarity.

### S2 — Retrieve word meaning (BLA-006–BLA-010)

6. **Enumerate stored meaning hypotheses (BLA-006).** Collect the meanings attached to active lexical entries: `M = union_e meanings(e)`.
7. **Retrieve semantic feature patterns (BLA-007).** Load each stored, provenance-bearing feature map: `mu_m = feature_map(m)`. Missing features remain missing.
8. **Calculate lexical meaning priors (BLA-008).** Weight a sense by its lexical identity posterior and supplied sense prior: `q(m) = P(e(m)|x) * prior(m)`.
9. **Calculate semantic-context compatibility (BLA-009).** When context is supplied, compare feature vectors: `C(m,c) = (mu_m dot c) / (||mu_m|| ||c|| + epsilon)`.
10. **Normalize and retain meaning alternatives (BLA-010).** `P(m|x,c) = softmax((log(q(m)+epsilon) + beta_c*C(m,c)) / T_sem)`. The result retains all supported senses, not only the top one.

**Biological reference.** This stage represents distributed lexical-semantic features and their convergence through temporal semantic networks, including an anterior temporal lobe (ATL)-inspired transmodal hub and modality-linked spokes. Contextual/controlled retrieval is treated as interaction with frontal language-control systems, not as an isolated semantic center.

### S3 — Retrieve grammatical category (BLA-011–BLA-014)

11. **Retrieve category labels (BLA-011).** Read each entry's supplied `lexical_category` field.
12. **Pool category evidence (BLA-012).** `E(g) = sum_e P(e|x) * I(category(e)=g)`.
13. **Integrate syntactic priors (BLA-013).** Add optional caller-supplied category evidence: `S(g) = log(E(g)+epsilon) + beta_g*prior_context(g)`. This prior biases attested evidence; it does not create a new lexical entry.
14. **Estimate category posterior (BLA-014).** `P(g|x,c) = softmax(S(g)/T_pos)`. Report the top-two posterior margin to expose uncertainty.

**Biological reference.** Category retrieval is modeled as distributed lexical and syntactic evidence in temporal/frontal language networks, including posterior temporal and inferior frontal interactions. The implementation uses a category distribution rather than a hard-coded brain-region lookup.

### S4 — Morphological decomposition (BLA-015–BLA-019)

15. **Retrieve the supplied morphology inventory (BLA-015).** Separate known prefixes, roots and suffixes.
16. **Generate candidate boundaries (BLA-016).** Test supplied prefix and suffix forms around candidate stems: `h=(p,s,u)` with `x=p+s+u`.
17. **Verify reconstruction (BLA-017).** Require exact concatenative reconstruction: `I_reconstruct = I[x == prefix + stem + suffix]`.
18. **Resolve the stem (BLA-018).** Require an exact lexical stem entry or a supplied root morpheme: `R(s)=lexical_entries(s) union root_morphemes(s)`.
19. **Preserve whole-word and segmented routes (BLA-019).** Add an intact whole-word hypothesis whenever an exact lexical entry exists. Preserve both routes where both are supported.

**Biological reference.** Morphological form processing is modeled as interaction between stored word forms and recurring morpheme representations, with frontal control available for competition. The software's boundary enumeration is an algorithmic approximation, not a literal claim that the brain runs a prefix/suffix loop.

### S5 — Retrieve stems and affixes (BLA-020–BLA-023)

20. **Retrieve stem records (BLA-020).** Load lexical and root records for the candidate stem, including supplied meanings, category labels and frequency.
21. **Retrieve affix records (BLA-021).** Load identified prefix/suffix forms, frequency, productivity, semantic features, and provenance.
22. **Retrieve component cues (BLA-022).** Collect component semantic features and optional caller-supplied grammatical-category cues. No category rule is inferred from spelling alone.
23. **Compose morpheme evidence (BLA-023).** A transparent feature-composition approximation is `mu_comp(f) = clip(mu_stem(f) + sum_j eta_j*productivity(a_j)*mu_affix_j(f), 0, 1)`. The output retains the contribution of each component.

**Biological reference.** Stem and affix activation is interpreted as reactivation of distributed form, semantic and morphosyntactic representations. The additive feature formula is a symbolic readout used for testability, not a claim that cortical tissue literally adds dictionary feature maps.

### S6 — Distinguish morphological alternatives (BLA-024–BLA-028)

24. **Measure stem support (BLA-024).** `S_stem = log(1+freq_stem) / log(1+freq_max)`.
25. **Measure affix support/productivity (BLA-025).** `S_affix = mean_j(log(1+freq_j)/log(1+freq_max) * productivity_j)`.
26. **Measure semantic/category compatibility (BLA-026).** `S_sem = cosine(mu_comp, mu_active)` and `S_pos = sum_g P(g|x,c)*cue(g)`.
27. **Compute decomposition posterior (BLA-027).** `z(h)=b_h + w_s*S_stem + w_a*S_affix + w_m*S_sem + w_g*S_pos`; `P(h)=softmax(z/T_morph)`.
28. **Apply confidence and ambiguity gate (BLA-028).** Select `h_1` only when `P(h_1)>=theta` and `P(h_1)-P(h_2)>=delta`. Otherwise set `preferred_morphology=None` and preserve all alternatives.

**Biological reference.** Whole-form and segmented hypotheses are treated as competing population patterns. Temporal semantic fit and frontal selection/control can bias the result, while abstention prevents software from pretending that weakly supported analyses are certain.

### S7 — Activate lexical candidates incrementally (BLA-029–BLA-036)

29. **Initialize candidate populations (BLA-029).** Create an activation state for every candidate and expose the word one character at a time: `x_t=x[0:t]`, `a_i(0)=a_0`.
30. **Accumulate bottom-up prefix evidence (BLA-030).** `B_i(t)=sim(x_t, form_i[0:|x_t|])`.
31. **Retrieve frequency-based pre-activation (BLA-031).** `F_i=log(1+freq_i+exposure_i)/max_j log(1+freq_j+exposure_j)`.
32. **Propagate recurrent excitation (BLA-032).** `R_i(t)=g_e*a_i(t-1)`.
33. **Calculate lateral inhibition (BLA-033).** `I_i(t)=g_i*sum_{j!=i}W_ij*a_j(t-1)/max(1,N-1)`, where `W_ij` is orthographic similarity between candidate forms.
34. **Integrate semantic/category/morphology feedback (BLA-034).** `T_i(t)=g_sem*M_i + g_pos*G_i + g_morph*H_i`.
35. **Update leaky activation (BLA-035).** `a_i(t)=(1-rho)*a_i(t-1)+rho*sigmoid((b+B_i+F_i+R_i-I_i+T_i)/T_a)`.
36. **Rank final candidates (BLA-036).** Return final activity, ranking, and per-character trajectories; retain close alternatives rather than forcing a winner.

**Biological reference.** The network combines an orthographic population, lexical-form populations, semantic/category/morphological populations, a selection-control population, and recurrent candidate activity. The LIF reference advances current-based membrane dynamics, recurrence, global inhibition, refractory resets and adaptation. The character stream is an explicit computational representation of incremental evidence arrival.

## Biological reference implementation

`LexicalAccessLIFSimulator` exposes seven coupled populations:

1. `orthographic`
2. `lexical_identity`
3. `semantic`
4. `grammatical_category`
5. `morphological`
6. `selection_control`
7. `recurrent_candidate`

For a population, a simplified current-based membrane update is:

`I_i(t) = I_drive + I_recurrent - I_global_inhibition - I_adaptation`

`V_i(t+dt) = V_i(t) + (dt/tau_m) * (V_rest - V_i(t) + 10^-3 * R_m[MΩ] * I_i[pA])`

The implementation converts `R_m` in megaohms and `I` in picoamps into millivolts, applies a threshold/reset and refractory period, and updates a decaying spike/adaptation state. Outputs include voltage (mV), current (pA), spike count, firing rate (Hz), activity fraction and simulated time (ms).

The LIF branch is diagnostic and separate from functional posterior probabilities. The three `selected_*` outputs are populated only when their confidence and ambiguity gates pass; otherwise the ranked alternatives remain available and the processor abstains. A valid numerical simulation does not prove that the human brain uses these exact operations or parameters.

## API

```python
from ore_lexical_context import LexicalEntry, LexicalMeaning, MorphemeEntry
from ore_lexical_access import (
    LexicalAccessConfig, LexicalAccessProcessor, LexicalAccessRequest,
)

lexicon = [
    LexicalEntry(
        "cat", "cat", frequency=24,
        meanings=(LexicalMeaning("feline", {"ANIMAL": 1.0, "FELINE": 1.0}),),
        lexical_category="noun", language="en",
        verified_by="curated-example", source_reference="dataset://cat",
    ),
]
morphemes = [
    MorphemeEntry(
        "plural-s", "s", "suffix", {"PLURAL": 1.0},
        frequency=20, productivity=0.95,
        verified_by="curated-example", source_reference="dataset://plural-s",
    ),
]
processor = LexicalAccessProcessor(
    lexicon, morphemes=morphemes,
    morpheme_category_cues={"plural-s": {"noun": 0.9}},
    config=LexicalAccessConfig(simulate_biology=True),
)
result = processor.process(LexicalAccessRequest("sample-001", "cats", language="en"))
print(result.preferred_morphology)
print(len(result.operation_trace))             # 36
print(result.incremental_activation[-1])
print(result.biological["layers"]["lexical_identity"]["mean_firing_rate_hz"])
```

`LexicalAccessProcessor` consumes the existing lexical/morpheme data records read-only. `LexicalAccessRequest.context_features` supplies optional semantic context; `contextual_category_priors` supplies optional grammatical-category priors. `morpheme_category_cues` is optional evidence provided by the caller and is never generated from suffix spelling. `result.to_dict()` returns a JSON-friendly representation.

Run the included demonstration with `python -m ore_lexical_access`. Run focused tests with `python -m pytest tests/test_lexical_access.py -q` and the full repository suite with `python -m pytest`.

## Limitations and next validation steps

- The package expects one already-tokenized word. Sentence parsing, OCR, and character segmentation remain upstream responsibilities.
- Lexical forms, meanings, category labels, roots, and affixes must be supplied and provenance-tracked by the caller.
- The included similarity, posterior and activation equations are computational models, not unique biological mechanisms.
- English-style prefix/root/suffix concatenation is the current morphology interface. Non-concatenative morphology, templatic morphology, infixation, tone, and script-specific allomorphy need explicit future extensions.
- Biological parameters and population mappings require empirical calibration. Software tests establish interface behavior and numerical sanity only; they do not establish cognitive or neural validity.
