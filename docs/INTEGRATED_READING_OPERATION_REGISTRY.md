# Integrated Reading Algorithm — Formal Operation Registry

**Registry:** ORE-INTEGRATED-READING-72  
**Package:** ore_reading/  
**Version:** 1.0.0  
**Status:** Executable research prototype; language coverage, behavioral accuracy and biological correspondence require independent validation.

## Scope and integration boundary

The algorithm starts from text or caller-tokenized orthographic word forms. It connects orthographic word-form processing, lexical access, phonological conversion, phonological word recognition, Bayesian competition and semantic activation.

The package is additive. It does not replace ORE N1–N9, canonical F-A–F-J / B-N1–B-N9 registers, or N6 orthographic segmentation. Upstream OCR/ORE can pass recognized tokens into ReadingRequest.tokens; an N6 result can be converted by reading_request_from_n6_result(). This package does not infer characters from pixels or segment a full image.

Two separate branches are exposed:

1. **Functional branch:** orthographic lexicon access, interactive activation, dual-route reading, grapheme-to-phoneme conversion, phonological assembly, TRACE-style feedback, Cohort narrowing, Shortlist-B-like Bayesian selection, and a connectionist spelling-sound-meaning triangle.
2. **Biological reference branch:** named neural populations simulated with current-based leaky integrate-and-fire (LIF) dynamics, membrane leak, threshold, reset, refractory state, recurrent excitation and population-level inhibition. It reports voltage, current, spike counts, firing rate and simulated time as distinct measures.

The biological branch is a generic computational population model. It is not fitted, anatomically complete or experimentally validated as a reproduction of human reading circuitry. Functional model names identify computational inspiration, not exact replications of the original published models.

## Formal operation registry

Every operation has one stable ID and appears exactly once in the runtime trace, in increasing order. Operations IR-066–IR-070 are marked SKIPPED when biological simulation is disabled; the trace still contains all 72 records.

### A. Input integrity and orthographic representation — IR-001 to IR-008

| ID | Operation | Executable responsibility |
|---|---|---|
| IR-001 | Validate request | Check sequence identity, exclusive input form and context contract |
| IR-002 | Unicode normalization | NFKC-normalize and case-fold while retaining original input |
| IR-003 | Token boundary preservation | Preserve token order and surface boundaries |
| IR-004 | Punctuation isolation | Separate leading/trailing punctuation from word core |
| IR-005 | Grapheme-cluster extraction | Group base code points with combining marks |
| IR-006 | Visual evidence initialization | Build normalized grapheme evidence; do not invent pixel confidence |
| IR-007 | Script and direction encoding | Preserve language, script hints and LTR/RTL metadata |
| IR-008 | Position-coded orthographic form | Encode grapheme identity and normalized serial position |

### B. Orthographic lexicon and interactive activation — IR-009 to IR-016

| ID | Operation | Executable responsibility |
|---|---|---|
| IR-009 | Lexicon provenance validation | Validate IDs, frequencies, pronunciations and semantic feature values |
| IR-010 | Lexicon indexing | Build normalized direct and first-grapheme indexes |
| IR-011 | Exact orthographic retrieval | Retrieve exact spelling matches |
| IR-012 | Approximate candidate retrieval | Retrieve near neighbors by grapheme edit similarity |
| IR-013 | Letter-level bottom-up drive | Score spelling/position support for candidates |
| IR-014 | Lexical frequency prior | Add bounded log-frequency exposure prior |
| IR-015 | Interactive activation initialization | Initialize word activation from bottom-up evidence and prior |
| IR-016 | Recurrent excitation and competition | Run damped recurrence with candidate feedback and lateral inhibition |

For candidate word \(w\), the initial activation score is

\[
s_w=2.2\,\mathrm{sim}(o,w)+0.12\log(1+f_w),\qquad
a_w^{(0)}=\mathrm{softmax}(s_w).
\]

The recurrent update is

\[
\tilde{s}_w^{(t)}=2.2\,\mathrm{sim}(o,w)+0.12\log(1+f_w)
+g_fa_w^{(t)}-g_i\sum_{v\ne w}a_v^{(t)},
\]
\[
a^{(t+1)}=0.65a^{(t)}+0.35\,\mathrm{softmax}(\tilde{s}^{(t)}).
\]

### C. Dual-route reading and grapheme-to-phoneme conversion — IR-017 to IR-024

| ID | Operation | Executable responsibility |
|---|---|---|
| IR-017 | Lexical phonology lookup | Retrieve stored pronunciations from lexical entries |
| IR-018 | Sublexical rule preparation | Load caller-supplied or default G2P correspondences |
| IR-019 | Grapheme chunking | Segment spelling by longest matching grapheme rules |
| IR-020 | Grapheme-to-phoneme conversion | Convert chunks; preserve explicit unknown-phone markers |
| IR-021 | Route compatibility scoring | Score lexical and sublexical evidence separately |
| IR-022 | Route gain calculation | Normalize route strengths into comparable weights |
| IR-023 | Route evidence integration | Select the best-supported route but retain both outputs |
| IR-024 | Pronunciation uncertainty | Compute route entropy and confidence |

The lexical route uses stored entries. The sublexical route uses longest-match G2P rules. An unknown grapheme yields <UNK_PHONEME> rather than disappearing. The route selector uses a softmax over route evidence. Both route outputs remain available for inspection.

### D. Phonological assembly — IR-025 to IR-032

| ID | Operation | Executable responsibility |
|---|---|---|
| IR-025 | Phoneme feature encoding | Map phones to a compact articulatory feature sketch |
| IR-026 | Syllabic nucleus estimation | Estimate vowel-like nuclei |
| IR-027 | Stress assignment | Preserve lexical stress only when supplied |
| IR-028 | Phonological buffer construction | Build the ordered phoneme buffer |
| IR-029 | Serial position coding | Add stable phoneme indices and normalized positions |
| IR-030 | Recurrent rehearsal | Maintain buffer activation through bounded leaky cycles |
| IR-031 | Phonological similarity | Compare output with stored phonological word forms |
| IR-032 | Phonological output preparation | Produce a sequence for word-form recognition |

### E. TRACE-style word recognition and Cohort narrowing — IR-033 to IR-040

| ID | Operation | Executable responsibility |
|---|---|---|
| IR-033 | Phonological input feature map | Aggregate phoneme feature evidence |
| IR-034 | TRACE feature-to-phoneme activation | Activate compatible phoneme candidates |
| IR-035 | TRACE phoneme-to-word activation | Activate lexical forms from the phonological stream |
| IR-036 | TRACE top-down feedback | Allow word candidates to reinforce constituent phonemes |
| IR-037 | TRACE lateral inhibition | Normalize and suppress competing hypotheses |
| IR-038 | Cohort initialization | Initialize candidates compatible with the initial phoneme |
| IR-039 | Cohort incremental elimination | Filter the cohort as each phoneme arrives |
| IR-040 | Cohort recognition point | Record the first unique-prefix point, when present |

TRACE is applied to the phonological stream assembled during reading. It models a phonological word-form competition stage downstream of orthographic conversion; it does not imply that print is literally processed as acoustic input. Cohort narrowing is exact-prefix by default and is exposed as an incremental diagnostic.

### F. Shortlist-B-like Bayesian recognition — IR-041 to IR-048

| ID | Operation | Executable responsibility |
|---|---|---|
| IR-041 | Shortlist hypothesis generation | Retrieve bounded phonologically similar word candidates |
| IR-042 | Noisy-channel likelihood | Score phoneme evidence using edit-distance noise |
| IR-043 | Lexical prior probability | Normalize supplied lexicon frequency into priors |
| IR-044 | Context prior estimation | Estimate semantic fit against caller-provided context |
| IR-045 | Bayesian posterior update | Combine evidence likelihood, lexical prior and context prior |
| IR-046 | Shortlist B noise competition | Retain multiple alternatives under uncertainty |
| IR-047 | Posterior calibration | Expose normalized posterior, entropy and margin |
| IR-048 | Lexical abstention gate | Abstain below confidence or separation thresholds |

The posterior is a noisy-channel approximation:

\[
P(w\mid e,c)\propto P(e\mid w)P(w)P(c\mid w).
\]

The evidence likelihood uses normalized phoneme edit cost and configured noise; the lexical prior comes from supplied frequency; the context term measures semantic-feature compatibility. These are model scores and require corpus calibration before being interpreted as empirical probabilities.

### G. Connectionist spelling-sound-meaning triangle — IR-049 to IR-056

| ID | Operation | Executable responsibility |
|---|---|---|
| IR-049 | Orthographic vector | Encode grapheme evidence in a shared alphabet space |
| IR-050 | Phonological vector | Encode phones in a shared phoneme inventory |
| IR-051 | Semantic feature space | Build a shared semantic feature vocabulary |
| IR-052 | Orthography-phonology associations | Estimate O-P co-occurrence weights |
| IR-053 | Orthography-semantics associations | Estimate O-S association weights |
| IR-054 | Phonology-semantics associations | Estimate P-S association weights |
| IR-055 | Triangular recurrent settling | Propagate damped activation across O/P/S layers |
| IR-056 | Meaning activation readout | Combine direct lexical semantics and triangle outputs |

Association matrices are induced from the caller-supplied lexicon, weighted by \(\log(1+\mathrm{frequency})\) and row-normalized. Direct semantic features are integrated with orthography-to-semantics and phonology-to-semantics activity. If the lexicon has no semantic features, that channel is explicitly unavailable.

### H. Semantic and sequence-level integration — IR-057 to IR-064

| ID | Operation | Executable responsibility |
|---|---|---|
| IR-057 | Sequence-level lexical integration | Retain ordered token hypotheses and boundaries |
| IR-058 | Neighbor-context aggregation | Pool bounded semantic evidence from adjacent words |
| IR-059 | Context-sensitive sense resolution | Reweight candidate interpretations using contextual fit |
| IR-060 | Typed relation proposal | Propose relations only from explicit lexical categories |
| IR-061 | Situation/context state update | Maintain a confidence-weighted semantic context |
| IR-062 | Semantic coherence estimate | Compute feature-space similarity as a diagnostic |
| IR-063 | Final lexical-phonological-semantic decision | Resolve or abstain while preserving alternatives |
| IR-064 | Explanation and evidence ledger | Expose route scores, alternatives, confidence and provenance |

This subsystem does not claim to perform a full syntactic parse, discourse model or unrestricted natural-language understanding. Category-based relations are heuristic proposals and are labeled accordingly.

### I. Biological simulation and acceptance — IR-065 to IR-072

| ID | Operation | Executable responsibility |
|---|---|---|
| IR-065 | Functional-to-neural drive mapping | Map functional representations to bounded neural drives |
| IR-066 | Orthographic population dynamics | Run the orthographic LIF population |
| IR-067 | Lexical and phonological population dynamics | Run lexical/phonological populations with competition |
| IR-068 | Auditory-wordform and semantic dynamics | Run word-form, semantic and control populations |
| IR-069 | Neural-state measurement | Report spikes, rates, voltage, current and time with units |
| IR-070 | Prediction-error comparison | Compare functional confidence with neural firing summaries |
| IR-071 | Integrity and operation-trace validation | Validate probability sums, finite outputs, token coverage and trace order |
| IR-072 | Structured result and scope declaration | Return outputs, trace, configuration, provenance and limitations |

The LIF population update uses a forward-Euler approximation:

\[
\tau_m\frac{dV_i}{dt}=V_{\mathrm{rest}}-V_i+R_m I_i(t).
\]

A spike occurs when voltage reaches threshold, followed by reset and a refractory interval. The implemented current includes external functional drive, a bounded recurrent term and global population inhibition. Spike counts are converted to Hz using simulated duration; membrane voltage is reported in mV and synaptic current in pA.

## Python API

```python
from ore_reading import IntegratedReadingProcessor, ReadingLexiconEntry, ReadingRequest

lexicon = [
    ReadingLexiconEntry(
        entry_id="word-001",
        orthography="read",
        phonemes=("R", "IY", "D"),
        frequency=40.0,
        semantic_features={"READING": 1.0, "KNOWLEDGE": 0.7},
        lexical_category="verb",
        verified_by="independent-reviewer",
        source_reference="dataset://release/read",
    ),
]
result = IntegratedReadingProcessor(lexicon).process(
    ReadingRequest("sample-001", text="read")
)
print(result.decoded_text)
print(result.tokens[0]["pronunciation"])
print(len(result.operation_trace))  # 72
print(result.biological["layers"]["orthographic"]["mean_firing_rate_hz"])
```

Use ReadingRequest.tokens when upstream OCR/ORE has produced token strings. ReadingRequest.context accepts caller-provided semantic-feature weights. The lexicon is not auto-expanded, and recognition never updates it.

## Validation

Run focused tests:

```bash
python -m pytest tests/test_integrated_reading.py -q
```

Run the complete repository suite:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Software tests establish software invariants only. Research validation still needs language-specific lexicons/rules, word and nonword naming tasks, lexical-decision benchmarks, pseudoword and exception-word conditions, human response times and error patterns, independent behavioral data, and independently sourced neural recordings. Word accuracy, nonword accuracy, phoneme error, latency distributions, calibration, abstention behavior and split-controlled generalization should be reported separately from neural correspondence.

## Existing ORE integration

Use reading_request_from_n6_result() to pass N6's already-segmented token output into this package. It preserves N6 sequence ID and token order; it does not change N6 probabilities, segmentation or model state.

```python
from ore_reading import IntegratedReadingProcessor, reading_request_from_n6_result

request = reading_request_from_n6_result(n6_result, language="en")
reading_result = IntegratedReadingProcessor(lexicon).process(request)
```
