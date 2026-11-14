# Integrated Reading Algorithm — Formal Operation Registry

**Registry:** \`ORE-INTEGRATED-READING-72\`  
**Package:** \`ore_reading/\`  
**Version:** \`1.0.0\`  
**Execution:** functional symbolic/computational branch plus optional biological LIF population branch  
**Status:** executable research prototype; language coverage, behavioral accuracy and biological correspondence remain to be independently validated

## Scope and integration boundary

The Integrated Reading Algorithm starts from text or caller-tokenized orthographic word forms. It connects orthographic word-form processing, lexical access, phonological conversion, phonological word recognition, Bayesian competition and semantic activation in one inspectable sequence.

The package is additive. It does not replace ORE N1-N9, the canonical F-A–F-J / B-N1–B-N9 registers, or N6 orthographic segmentation. An upstream OCR/ORE caller can pass recognized token strings to \`ReadingRequest.tokens\`; an upstream N6 caller can pass decoded token orthographies. This package does not infer characters from pixels.

Two implementation branches are distinct:

1. **Functional branch:** deterministic operations implementing computational ideas associated with orthographic lexicons, interactive activation, dual-route reading, grapheme-to-phoneme conversion, TRACE-style feedback, Cohort narrowing, Shortlist-B-like Bayesian selection and a connectionist spelling-sound-meaning triangle.
2. **Biological branch:** named neural populations simulated with current-based leaky integrate-and-fire (LIF) dynamics, membrane leak, threshold, reset, refractory state, recurrent excitation and population-level inhibition. It receives bounded activations from the functional branch and reports voltage, current, spike counts, firing rate and simulated time separately.

The biological branch is a generic computational population model, not a fitted, anatomically complete or experimentally validated reproduction of human reading circuitry. Functional model names identify inspiration, not exact replications of the original published models.

## Operation registry

Each operation has a stable ID, functional rule, computational form, biological analogue and named output. Runtime traces contain exactly one record for each operation in increasing ID order. Optional biological operations are marked \`SKIPPED\` when simulation is disabled; the registry is never shortened.

### A. Input integrity and orthographic representation — IR-001 to IR-008

| ID | Operation | Executable responsibility | Main output |
|---|---|---|---|
| IR-001 | Validate request | Validate sequence identity, exactly one input form and finite context values | \`validated_request\` |
| IR-002 | Unicode normalization | NFKC-normalize and case-fold while retaining original input | \`normalized_text\` |
| IR-003 | Token boundary preservation | Preserve token order and split text into surface tokens | \`surface_tokens\` |
| IR-004 | Punctuation isolation | Keep leading/trailing punctuation separate from word core | \`token_boundaries\` |
| IR-005 | Grapheme-cluster extraction | Group base code points with combining marks | \`grapheme_sequences\` |
| IR-006 | Visual evidence initialization | Build normalized grapheme/letter evidence; mark confidence as absent unless supplied | \`visual_letter_evidence\` |
| IR-007 | Script and direction encoding | Preserve language, script hints and LTR/RTL metadata | \`script_profile\` |
| IR-008 | Position-coded orthographic form | Encode grapheme identity and normalized serial position | \`orthographic_codes\` |

### B. Orthographic lexicon and interactive activation — IR-009 to IR-016

| ID | Operation | Executable responsibility | Main output |
|---|---|---|---|
| IR-009 | Lexicon provenance validation | Validate entry IDs, frequency, pronunciations and semantic-feature values | \`validated_lexicon\` |
| IR-010 | Lexicon indexing | Build normalized direct and first-grapheme indexes | \`lexicon_index\` |
| IR-011 | Exact orthographic retrieval | Retrieve exact spelling matches | \`exact_candidates\` |
| IR-012 | Approximate candidate retrieval | Retrieve near neighbors by grapheme edit similarity | \`approximate_candidates\` |
| IR-013 | Letter-level bottom-up drive | Score spelling/position evidence for each candidate | \`letter_support\` |
| IR-014 | Lexical frequency prior | Add bounded log-frequency exposure prior | \`frequency_priors\` |
| IR-015 | Interactive activation initialization | Initialize word candidate activation from bottom-up evidence and prior | \`initial_word_activation\` |
| IR-016 | Recurrent excitation and competition | Run damped candidate recurrence with word feedback and lateral inhibition | \`settled_word_activation\` |

For candidate word \(w\), a simple initial score is

\[
s_w = 2.2\,\mathrm{sim}(o,w) + 0.12\log(1+f_w),\qquad
a_w^{(0)}=\mathrm{softmax}(s_w).
\]

The interactive update applies bounded feedback and competition before renormalization:

\[
\tilde{s}_w^{(t)}=2.2\,\mathrm{sim}(o,w)+0.12\log(1+f_w)
+g_f a_w^{(t)}-g_i\sum_{v\ne w}a_v^{(t)},
\]
\[
a^{(t+1)}=0.65a^{(t)}+0.35\,\mathrm{softmax}(\tilde{s}^{(t)}).
\]

### C. Dual-route reading and grapheme-to-phoneme conversion — IR-017 to IR-024

| ID | Operation | Executable responsibility | Main output |
|---|---|---|---|
| IR-017 | Lexical phonology lookup | Retrieve stored pronunciations from candidate lexical entries | \`lexical_pronunciations\` |
| IR-018 | Sublexical rule preparation | Load caller-supplied or default G2P correspondences | \`g2p_rules\` |
| IR-019 | Grapheme chunking | Segment spelling using longest matching grapheme rules | \`grapheme_chunks\` |
| IR-020 | Grapheme-to-phoneme conversion | Convert chunks, retaining explicit unknown-phone markers | \`sublexical_pronunciation\` |
| IR-021 | Route compatibility scoring | Score lexical and sublexical evidence separately | \`route_scores\` |
| IR-022 | Route gain calculation | Normalize route strengths into comparable weights | \`route_weights\` |
| IR-023 | Route evidence integration | Select best-supported route but retain both outputs | \`integrated_pronunciation\` |
| IR-024 | Pronunciation uncertainty | Compute route entropy and confidence | \`pronunciation_confidence\` |

Both routes are preserved. The lexical route uses stored entries and their frequencies. The sublexical route uses a longest-match grapheme rule inventory. An unknown grapheme yields \`<UNK_PHONEME>\` rather than disappearing. Route selection uses a softmax over weighted route evidence; not all words are assumed to follow a single deterministic pronunciation rule.

### D. Phonological assembly — IR-025 to IR-032

| ID | Operation | Executable responsibility | Main output |
|---|---|---|---|
| IR-025 | Phoneme feature encoding | Map phones to a compact articulatory feature sketch | \`phoneme_features\` |
| IR-026 | Syllabic nucleus estimation | Estimate vowel-like nuclei | \`syllable_nuclei\` |
| IR-027 | Stress assignment | Preserve stress only when supplied in the lexicon | \`stress_pattern\` |
| IR-028 | Phonological buffer construction | Build the ordered phoneme buffer | \`phonological_buffer\` |
| IR-029 | Serial position coding | Add stable phoneme indices and normalized positions | \`phoneme_positions\` |
| IR-030 | Recurrent rehearsal | Maintain buffer activation through bounded leaky cycles | \`rehearsal_state\` |
| IR-031 | Phonological similarity | Compare output with stored word-form pronunciations | \`phonological_similarity\` |
| IR-032 | Phonological output preparation | Produce a sequence for phonological lexical recognition | \`prepared_phonology\` |

### E. TRACE-style word recognition and Cohort narrowing — IR-033 to IR-040

| ID | Operation | Executable responsibility | Main output |
|---|---|---|---|
| IR-033 | Phonological input feature map | Aggregate phoneme feature evidence | \`trace_feature_input\` |
| IR-034 | TRACE feature-to-phoneme activation | Activate compatible phoneme candidates | \`trace_phoneme_activation\` |
| IR-035 | TRACE phoneme-to-word activation | Activate lexical forms from the generated phoneme stream | \`trace_word_activation\` |
| IR-036 | TRACE top-down feedback | Allow word candidates to reinforce constituent phonemes | \`trace_feedback\` |
| IR-037 | TRACE lateral inhibition | Normalize and suppress competing hypotheses | \`trace_competition\` |
| IR-038 | Cohort initialization | Initialize candidates compatible with the initial phoneme | \`cohort_initial\` |
| IR-039 | Cohort incremental elimination | Filter the cohort as each phoneme arrives | \`cohort_trace\` |
| IR-040 | Cohort recognition point | Record first unique-prefix point, when one exists | \`cohort_recognition_point\` |

TRACE is applied to the phonological stream produced by reading. It therefore models a phonological word-form competition stage downstream of orthographic conversion; it does not imply printed text is literally processed as an acoustic signal. Cohort narrowing is exact-prefix by default and is exposed as an incremental diagnostic.

### F. Shortlist-B-like Bayesian recognition — IR-041 to IR-048

| ID | Operation | Executable responsibility | Main output |
|---|---|---|---|
| IR-041 | Shortlist hypothesis generation | Retrieve bounded phonologically similar word candidates | \`shortlist\` |
| IR-042 | Noisy-channel likelihood | Score phoneme evidence with edit-distance noise | \`evidence_likelihoods\` |
| IR-043 | Lexical prior probability | Normalize lexicon frequency into priors | \`lexical_priors\` |
| IR-044 | Context prior estimation | Estimate semantic fit against caller-provided context | \`context_priors\` |
| IR-045 | Bayesian posterior update | Combine evidence likelihood, lexical prior and context prior | \`bayesian_posterior\` |
| IR-046 | Shortlist B noise competition | Retain multiple alternatives under uncertainty | \`shortlist_b_alternatives\` |
| IR-047 | Posterior calibration | Normalize and expose entropy/margin | \`calibrated_posterior\` |
| IR-048 | Lexical abstention gate | Abstain below confidence or separation thresholds | \`lexical_abstention\` |

The posterior is a noisy-channel approximation:

\[
P(w\mid e,c)\propto P(e\mid w)P(w)P(c\mid w).
\]

\(P(e\mid w)\) is estimated from normalized phoneme edit cost and configured noise. \(P(w)\) comes from supplied entry frequency. \(P(c\mid w)\) is semantic-feature compatibility with supplied context. These are computational scores and require calibration on a language-specific corpus before being interpreted as empirical probabilities.

### G. Connectionist spelling-sound-meaning triangle — IR-049 to IR-056

| ID | Operation | Executable responsibility | Main output |
|---|---|---|---|
| IR-049 | Orthographic vector | Encode grapheme evidence in a shared alphabet space | \`orthographic_vector\` |
| IR-050 | Phonological vector | Encode phones in a shared phoneme inventory | \`phonological_vector\` |
| IR-051 | Semantic feature space | Build a shared semantic feature vocabulary | \`semantic_feature_space\` |
| IR-052 | Orthography-phonology associations | Estimate O-P co-occurrence matrix | \`weight_orthography_phonology\` |
| IR-053 | Orthography-semantics associations | Estimate O-S association matrix | \`weight_orthography_semantics\` |
| IR-054 | Phonology-semantics associations | Estimate P-S association matrix | \`weight_phonology_semantics\` |
| IR-055 | Triangular recurrent settling | Propagate damped activation across O/P/S layers | \`triangle_activations\` |
| IR-056 | Meaning activation readout | Combine direct lexical semantics and triangle outputs | \`semantic_activation\` |

Association matrices are induced from the supplied lexicon, weighted by \(\log(1+\mathrm{frequency})\) and row-normalized. The implementation combines direct lexical semantic features with orthography-to-semantics and phonology-to-semantics activity. Empty semantic inventories remain explicitly unavailable.

### H. Semantic and sequence-level integration — IR-057 to IR-064

| ID | Operation | Executable responsibility | Main output |
|---|---|---|---|
| IR-057 | Sequence-level lexical integration | Retain ordered token hypotheses | \`sequence_lexical_state\` |
| IR-058 | Neighbor-context aggregation | Pool semantic evidence from adjacent words | \`neighbor_context\` |
| IR-059 | Context-sensitive sense resolution | Reweight candidates by contextual semantic fit | \`contextual_senses\` |
| IR-060 | Typed relation proposal | Propose relations only from explicit lexical categories | \`relation_proposals\` |
| IR-061 | Situation/context state update | Maintain confidence-weighted semantic context | \`situation_state\` |
| IR-062 | Semantic coherence estimate | Compute feature-space similarity as a diagnostic | \`coherence_scores\` |
| IR-063 | Final lexical-phonological-semantic decision | Resolve or abstain while preserving alternatives | \`token_decisions\` |
| IR-064 | Explanation and evidence ledger | Expose route scores, lexical alternatives and provenance | \`evidence_ledger\` |

This subsystem does not claim to perform a full syntactic parse, discourse model or unrestricted natural-language understanding. Category-based relation proposals are explicitly labeled heuristic.

### I. Biological simulation and acceptance — IR-065 to IR-072

| ID | Operation | Executable responsibility | Main output |
|---|---|---|---|
| IR-065 | Functional-to-neural drive mapping | Map functional vectors to bounded neural drives | \`neural_drives\` |
| IR-066 | Orthographic population dynamics | Run orthographic LIF population | \`orthographic_neural_state\` |
| IR-067 | Lexical and phonological population dynamics | Run lexical/phonological recurrent-competition populations | \`lexical_phonological_neural_state\` |
| IR-068 | Auditory-wordform and semantic dynamics | Run phonological-wordform/semantic/control populations | \`semantic_neural_state\` |
| IR-069 | Neural-state measurement | Report spikes, rates, voltage, current and time with units | \`neural_measurements\` |
| IR-070 | Prediction-error comparison | Compare functional confidence with neural firing summaries | \`functional_neural_residual\` |
| IR-071 | Integrity and operation-trace validation | Validate probabilities, finite values, token coverage and trace order | \`integrity_report\` |
| IR-072 | Structured result and scope declaration | Return outputs, trace, config, provenance and limitations | \`reading_result\` |

The LIF update uses a forward-Euler approximation to

\[
\tau_m\frac{dV_i}{dt}=V_{\mathrm{rest}}-V_i+R_m I_i(t),
\]

with spike generation when \(V_i\) reaches threshold, membrane reset and a refractory interval. The implemented current includes external functional drive, a bounded recurrent term and global population inhibition. Spike counts are converted to Hz using simulated duration; membrane voltage is reported in mV and synaptic current in pA.

## Python API

\`\`\`python
from ore_reading import IntegratedReadingProcessor, ReadingLexiconEntry, ReadingRequest

lexicon = [
    ReadingLexiconEntry(
        entry_id="word-001", orthography="read", phonemes=("R", "IY", "D"),
        frequency=40.0, semantic_features={"READING": 1.0, "KNOWLEDGE": 0.7},
        lexical_category="verb", verified_by="independent-reviewer",
        source_reference="dataset://release/read",
    ),
]
result = IntegratedReadingProcessor(lexicon).process(ReadingRequest("sample-001", text="read"))
print(result.decoded_text)
print(result.tokens[0]["pronunciation"])
print(len(result.operation_trace))  # 72
print(result.biological["layers"]["orthographic"]["mean_firing_rate_hz"])
\`\`\`

Use \`ReadingRequest.tokens\` when upstream OCR/ORE has produced token strings. \`ReadingRequest.context\` accepts caller-provided semantic-feature weights. The lexicon is not auto-expanded, and recognition never updates it.

## Validation

Run focused tests:

\`\`\`bash
python -m pytest tests/test_integrated_reading.py -q
\`\`\`

Run the repository test suite:

\`\`\`bash
python -m pip install -e ".[dev]"
python -m pytest
\`\`\`

Software tests establish implementation invariants only. Research validation still needs language-specific lexicons/rules, word and nonword naming tasks, lexical-decision benchmarks, pseudoword and exception-word conditions, human response times and error patterns, independent behavioral data, and independently sourced neural recordings. Report word accuracy, nonword accuracy, phoneme error, latency distributions, calibration, abstention behavior and split-controlled generalization separately from neural correspondence.

## Existing ORE integration

Use \`reading_request_from_n6_result()\` to pass N6's already-segmented token output into this package. It preserves the N6 sequence ID and token order; it does not change N6 probabilities, segmentation or model state.

\`\`\`python
from ore_reading import IntegratedReadingProcessor, reading_request_from_n6_result

request = reading_request_from_n6_result(n6_result, language="en")
reading_result = IntegratedReadingProcessor(lexicon).process(request)
\`\`\`
