# Point 23 — Language-Comprehension Pitfalls: Garden-Path Sentences and Ambiguous Syntax

## Scope and relationship to the source document

This point expands the source document's statement that language comprehension can be disrupted by garden-path sentences (for example, **“The old man the boats”**) and by ambiguous syntax. It specifies a biologically informed, computationally explicit account of how a listener or reader incrementally builds a sentence interpretation, commits to a temporarily preferred structure, encounters disambiguating evidence, and may or may not revise the earlier analysis.

The target is the human brain's online sentence-comprehension process. It is not a claim that the brain executes a literal symbolic parser or performs the equations below as conscious arithmetic. Equations are formal descriptions that make hypotheses measurable and implementable in a computational model. The neural mechanisms are distributed and interactive; no single brain region should be treated as a complete “grammar parser.”

## 1. Operational definition and boundaries

A syntactic ambiguity occurs when a word sequence supports more than one grammatical structure or attachment. The ambiguity may be temporary or remain unresolved until later context.

A **garden-path sentence** is a special case: the early portion strongly encourages one analysis, but later material makes that initial analysis difficult or impossible to maintain. The comprehender must detect conflict, revise the parse, construct a different structure, or settle for a partial/good-enough interpretation.

Classic example:

> The old man the boats.

A frequent initial reading treats “the old man” as a noun phrase. The later “the boats” makes that analysis unsuitable. A less expected parse treats “the old” as a noun phrase (meaning elderly people) and “man” as a verb (to operate or crew), yielding the meaning that elderly people operate the boats. The sentence is unusual in vocabulary and construction, so it is useful as a teaching illustration, but it should not be treated as a universal diagnostic of every reader's parsing strategy.

Another common family is the temporary main-clause/relative-clause ambiguity, illustrated by sentences such as “While the man hunted the deer ran into the woods.” Early words may encourage the reader to treat “the deer” as the object of “hunted”; the following verb “ran” indicates that “the deer” is instead the subject of a new clause. Exact effects depend on sentence wording, language, context, prosody, frequency, and the comprehender.

Distinguish at least four cases:

1. **Lexical ambiguity:** one word has multiple meanings or syntactic categories.
2. **Structural ambiguity:** the same words permit multiple phrase structures or dependency relations.
3. **Attachment ambiguity:** a phrase can attach to more than one head or clause.
4. **Temporary ambiguity:** the available prefix supports multiple future continuations, and subsequent input changes their relative plausibility.

The algorithm below concentrates on syntactic and attachment ambiguity, while allowing lexical category and semantic plausibility to contribute cues.

## 2. Biological architecture: cooperating systems, not a single parser

Sentence comprehension recruits a predominantly left-lateralized frontotemporal language network, with important contributions from temporal cortex, inferior frontal cortex, and connecting white-matter pathways. The exact division of labor is task-, construction-, and individual-dependent.

### 2.1 Temporal language regions

Middle and posterior portions of the temporal lobe contribute to accessing and combining lexical, syntactic, and sentence-level information. They provide constraints from word identity, argument structure, combinatorial patterns, and the evolving interpretation. Anterior temporal regions can contribute to combining structured linguistic and conceptual information. These functions are distributed, and the temporal cortex should not be reduced to a passive word store.

### 2.2 Inferior frontal cortex

Left inferior frontal regions, including pars opercularis and neighboring areas, are often engaged when sentence interpretation requires demanding structural operations, resolving competition, or revising an analysis. However, increased activation does not prove that a single region performs reanalysis by itself. It may reflect several interacting demands, including selection, unification, controlled retrieval, and maintenance of alternatives.

### 2.3 Multiple-demand and domain-general control systems

Difficult or unexpected sentences can also recruit parts of the multiple-demand/frontoparietal control network, including lateral frontal and parietal regions. These systems support domain-general operations such as maintaining task goals, monitoring conflict, and allocating control. Evidence from garden-path research suggests that language-selective and domain-general regions can make distinguishable contributions rather than being interchangeable.

### 2.4 Working memory and attention

Working memory helps maintain recently encountered words, phrase boundaries, thematic roles, and competing interpretations while new material arrives. Attention prioritizes the current input and controls which cues receive processing resources. Limited resources, distraction, fatigue, or a long distance between a head and its dependent can reduce the probability that the correct structure is maintained or successfully recovered.

### 2.5 Auditory prosody and visual punctuation

For spoken language, stress, pitch, duration, rhythm, and pauses provide cues to phrase boundaries and intended grouping. In reading, punctuation, capitalization, line breaks, and typography can provide partial analogues. These cues interact with syntax and meaning; they do not simply decorate an already completed sentence representation.

## 3. Computational representation of the incoming sentence

Represent the incoming sentence as a sequence of word or morpheme units:

\[
X = (x_1,x_2,\ldots,x_T)
\]

For each incoming unit \(x_t\), the comprehension system has a current state containing:

- lexical candidates and their grammatical properties;
- one or more candidate syntactic structures;
- dependencies and argument-role assignments;
- semantic and discourse interpretations;
- prosodic or punctuation cues;
- confidence values and unresolved ambiguities;
- the current working-memory state.

A candidate parse \(P_i\) is not merely a string label. It can be represented as a structured graph or tree whose nodes encode words or phrases and whose edges represent grammatical dependencies, constituency, attachment, and thematic relations.

The algorithm maintains a set of hypotheses rather than assuming that exactly one parse is always active. The number and persistence of hypotheses may vary with the task, the ambiguity, and processing demands.

## 4. Integrated algorithmic sequence

### Step 1 — Receive and segment the linguistic stream

**Biological operation:** Auditory pathways analyze the speech signal over time, while visual pathways analyze written characters and word forms. Attention selects the relevant stream. The system extracts candidate phonological/orthographic units and establishes provisional word boundaries.

**Computational operation:** Convert the continuous input into a sequence of candidate lexical units. For speech, estimate word-boundary probabilities rather than assuming that the acoustic signal contains explicit spaces. For text, use orthographic segmentation while retaining punctuation and formatting cues.

A boundary estimate can be expressed as:

\[
P(b_t=1\mid a_{1:T},c_t)
\]

where \(b_t\) indicates a possible boundary, \(a_{1:T}\) is acoustic evidence, and \(c_t\) is contextual information. This is a model-level description, not a literal neuronal probability register.

**Output:** A time-ordered sequence of candidate words with confidence values and available prosodic or punctuation features.

### Step 2 — Activate lexical entries and grammatical category candidates

**Biological operation:** Word forms activate learned lexical and phonological representations, with semantic and grammatical information becoming available through distributed temporal and frontal language circuits.

**Computational operation:** For each word, retrieve possible lemmas, meanings, syntactic categories, inflectional features, and argument-structure information. A category may be ambiguous: a form can be a noun, verb, adjective, or another category depending on language and context.

For candidate category \(c\):

\[
P(c\mid x_t,h_t)\propto P(x_t\mid c)P(c\mid h_t)
\]

where \(h_t\) is the preceding linguistic and discourse context. Lexical frequency and learned construction frequencies can influence the prior, but neither should be assumed to determine the choice alone.

**Output:** A ranked set of lexical/category hypotheses for the current word.

### Step 3 — Generate syntactic continuations incrementally

**Biological operation:** As words arrive, the comprehension system combines lexical and structural expectations to form a provisional representation. Processing is incremental: the brain does not necessarily wait until the complete sentence is heard or read.

**Computational operation:** Extend each currently active parse with grammatical continuations licensed by the new word. Candidate structures may include alternative phrase boundaries, head assignments, attachment sites, and thematic-role mappings.

The parser's state after word \(t\) is:

\[
\mathcal{P}_t=\{P_1,P_2,\ldots,P_k\}
\]

where each \(P_i\) has a score, confidence, and representation of unresolved dependencies.

**Output:** An updated set of partial syntactic hypotheses.

### Step 4 — Estimate structural expectations

**Biological operation:** Previously learned language regularities shape expectations about likely next words and structures. These expectations are sensitive to exposure, construction frequency, verb bias, discourse, and language-specific grammar.

**Computational operation:** Estimate which continuations are expected from the current prefix. A simple probabilistic model computes:

\[
P(x_{t+1},P_{t+1}\mid x_{1:t},P_t)
\]

A model may use construction frequencies, lexical preferences, and contextual information. A simple “minimal structure first” heuristic is one historical account of some garden-path effects, but modern accounts are broader and do not assume one universal parsing rule.

**Output:** A probability distribution over likely next words and structures.

### Step 5 — Integrate semantic plausibility and thematic-role constraints

**Biological operation:** Meaning-related information becomes available while syntax is being built. Knowledge about entities, events, and typical relations can support or discourage candidate parses.

**Computational operation:** For each candidate structure, estimate whether its thematic roles and event interpretation are plausible. For example, knowledge about which entities can typically perform an action may alter the ranking of parses. Plausibility is a constraint, not a guarantee: unusual but grammatical sentences remain possible.

A combined parse score may be:

\[
S(P_i)=\alpha S_{syntax}(P_i)+\beta S_{lexical}(P_i)+\gamma S_{semantic}(P_i)+\delta S_{context}(P_i)
\]

The weights are fitted parameters in a computational model. There is no evidence that the brain applies one fixed linear scoring formula in this exact form.

**Output:** Candidate structures ranked using syntax, lexical preferences, semantic plausibility, and context.

### Step 6 — Select a provisional interpretation under uncertainty

**Biological operation:** The system often behaves as though it has settled on a likely interpretation sufficiently to continue comprehension. This can be efficient, but early commitment can make later revision difficult.

**Computational operation:** Select the highest-scoring parse, or retain several candidates if uncertainty is high. A beam-search implementation can preserve the top \(K\) hypotheses:

\[
\mathcal{B}_t=\operatorname{TopK}_{P_i\in\mathcal{P}_t}S(P_i)
\]

A single-parse strategy is computationally cheap but vulnerable to temporary ambiguity. A multi-hypothesis strategy can reduce premature commitment, but increases memory and processing cost. These are competing implementation assumptions, not established descriptions of a unique neural mechanism.

**Output:** One provisional interpretation or a small set of active alternatives.

### Step 7 — Maintain syntactic dependencies and thematic roles

**Biological operation:** The comprehension system keeps track of relationships between words that may be separated in the sentence, such as a verb and its subject or object. Working memory and predictive representations contribute to maintaining these relations.

**Computational operation:** Store unresolved dependencies, open phrases, expected arguments, and candidate attachment sites. Update them as new words arrive. A dependency record may contain:

- head word or phrase;
- dependent expected or observed;
- grammatical relation;
- thematic role;
- activation/confidence;
- distance or time since activation.

A simple activation-decay approximation is:

\[
A_j(t)=A_j(t_0)e^{-\lambda(t-t_0)}
\]

with interference terms added for competing items. This is a useful formalization of declining accessibility, not a claim that all linguistic memory decays exponentially.

**Output:** A maintained representation of current dependencies and open structural expectations.

### Step 8 — Detect a disambiguating or conflicting cue

**Biological operation:** New input may fit the current parse, support an alternative, or conflict with the predicted continuation. Conflict can increase processing demand and trigger additional analysis.

**Computational operation:** Compare the incoming word and its features against the predicted continuations of active parses. Measure mismatch or surprise. One common information-theoretic quantity is surprisal:

\[
I(x_t)=-\log_2 P(x_t\mid x_{1:t-1})
\]

Lower predicted probability produces greater surprisal. Surprisal can help predict processing difficulty, but it is not the only determinant: structural ambiguity, integration cost, memory load, and plausibility also matter.

A parse-specific mismatch score can be:

\[
E_i(t)=D\bigl(F(x_t),\widehat{F}_i(t)\bigr)
\]

where \(F(x_t)\) is the observed feature vector, \(\widehat{F}_i(t)\) is the feature pattern predicted by parse \(P_i\), and \(D\) is a suitable distance or incompatibility measure.

**Output:** A conflict/surprise estimate and an indication of which candidate parses remain compatible.

### Step 9 — Determine whether the initial parse is still viable

**Biological operation:** The system evaluates whether the new material can be integrated into the current interpretation. Some apparent conflicts can be resolved through local integration; others require changing phrase structure or role assignment.

**Computational operation:** Test grammatical compatibility, lexical category compatibility, argument requirements, and semantic coherence. If a parse violates a strong constraint, sharply reduce its score or remove it from the active beam. Do not discard a parse solely because it is unusual if it remains grammatically possible.

A viability score may be:

\[
V(P_i,t)=G_i(t)\cdot L_i(t)\cdot C_i(t)
\]

where each factor represents a normalized compatibility estimate. A practical implementation should use log probabilities or a calibrated scoring model to avoid numerical underflow.

**Output:** A set of viable parses and a record of rejected or low-confidence analyses.

### Step 10 — Trigger revision when the existing interpretation fails or becomes costly

**Biological operation:** When the current analysis is inadequate, the system may increase control and attempt to revise its representation. Left inferior frontal and temporal language regions, together with domain-general control systems, can contribute depending on task and type of difficulty.

**Computational operation:** Trigger revision when the top parse falls below a compatibility threshold, when a competitor becomes substantially more probable, or when the cost of preserving the current parse exceeds the cost of rebuilding it.

For example:

\[
\text{revise if }V(P_{best},t)<\theta_V
\quad\text{or}\quad
S(P_{alt})-S(P_{best})>\theta_\Delta
\]

Thresholds are model parameters and should be estimated from behavior, not invented as universal brain constants.

**Output:** A revision request with the words and structural decisions that need reconsideration.

### Step 11 — Locate the earliest consequential ambiguity point

**Biological operation:** Successful recovery may require revisiting an earlier word or phrase that supported the initial analysis. This can be difficult because the original decision may no longer be highly accessible.

**Computational operation:** Search the active structure for choice points where alternative categories, attachments, or clause boundaries were available. Identify which earlier decision caused the current incompatibility. Maintain a set of possible revision points rather than presuming readers always return to one syntactically optimal location.

A causal trace can record:

\[
(x_j, d_j, P_{before},P_{after},c_j)
\]

where \(x_j\) is the relevant earlier word, \(d_j\) is the structural decision, and \(c_j\) is confidence at the time.

**Output:** A candidate set of revision points and structural decisions.

### Step 12 — Reconstruct an alternative syntactic structure

**Biological operation:** The system attempts to reinterpret the ambiguous material in a way that is compatible with the new evidence. Revision can involve changing a word's category, moving a phrase attachment, assigning a different subject/object relation, or identifying a clause boundary.

**Computational operation:** Generate alternative parses by changing one or more earlier decisions while preserving constraints that remain valid. For “The old man the boats,” one candidate revision treats “the old” as a noun phrase and “man” as a verb. In other examples, a noun phrase initially treated as a direct object may instead become the subject of a subordinate clause.

A revised parse can be selected by:

\[
P(P_i\mid X)\propto P(X\mid P_i)P(P_i)
\]

where \(X\) is the observed sequence and the prior is influenced by language experience. The equation formalizes evidence-based ranking; it does not imply explicit Bayesian calculation in the brain.

**Output:** One or more alternative structures compatible with the complete input so far.

### Step 13 — Reassign thematic roles and rebuild sentence meaning

**Biological operation:** A syntactic revision may change who performed an action, what was affected, and how phrases relate. Semantic representations must be updated accordingly.

**Computational operation:** Recompute subject, object, agent, patient, instrument, location, and other roles as appropriate. Reconstruct the event or proposition and check whether the new role assignment is consistent with the lexical meanings and discourse model.

An event representation can be represented as:

\[
E=\langle predicate,agent,patient,recipient,instrument,location,time\rangle
\]

Fields may be null or uncertain when not specified. This representation is a modeling convenience; natural-language meaning can be richer and more context-dependent.

**Output:** Updated syntactic structure and event/semantic representation.

### Step 14 — Reconcile old and new interpretations

**Biological operation:** The previously activated analysis may continue to influence comprehension even after new information suggests a different structure. Revision may therefore be partial, delayed, or incomplete.

**Computational operation:** Compare the old and revised interpretations, preserve compatible information, withdraw unsupported role assignments, and reduce confidence in unresolved components. Avoid assuming that a late correction completely erases the earlier analysis.

A confidence update may use:

\[
C_{new}=C_{old}+\eta\bigl(C_{evidence}-C_{old}\bigr)
\]

where \(\eta\) is a learning/update rate. Separate confidence should be maintained for grammatical structure, word meaning, and overall interpretation.

**Output:** A reconciled interpretation with uncertainty annotations.

### Step 15 — Decide between full reanalysis and a good-enough interpretation

**Biological operation:** Human comprehenders do not always arrive at a fully grammatical and semantically accurate final analysis. Sometimes they retain an initial interpretation, infer a rough meaning, or answer a comprehension question using partial information.

**Computational operation:** Evaluate whether the current representation is sufficient for the task. A full parse may be pursued when accuracy is important or when the sentence is explicitly analyzed. A good-enough representation may be accepted when processing resources are limited and the task does not require precise structural detail.

A task-sensitive stopping condition can be:

\[
\text{stop if }U(P)+R_{task}(P) < \theta_{stop}
\]

where \(U(P)\) is uncertainty and \(R_{task}(P)\) is the expected risk of using the current interpretation. Alternatively, compare expected accuracy benefit against additional processing cost.

**Output:** A complete parse, a partial/good-enough interpretation, or an unresolved-ambiguity flag.

### Step 16 — Use prosody, punctuation, and discourse to constrain the final interpretation

**Biological operation:** Prosodic boundaries in speech and punctuation or formatting in writing contribute information about phrase grouping and discourse organization. Context can make a previously unlikely analysis more plausible.

**Computational operation:** Fuse structural, acoustic/orthographic, semantic, and discourse evidence. A cue-integration model may be represented as:

\[
S(P_i)=\sum_m w_m\log P(c_m\mid P_i)
\]

where \(c_m\) denotes a cue (syntax, prosody, punctuation, semantic plausibility, discourse), and \(w_m\) reflects cue reliability. Cue independence is a simplifying assumption; in real language, cues can be correlated.

**Output:** A contextually integrated interpretation with remaining uncertainty made explicit.

### Step 17 — Store the outcome and adapt expectations from experience

**Biological operation:** Repeated exposure to constructions and feedback can change later expectations. The comprehender may become more sensitive to a construction, verb bias, or contextual cue after encountering it repeatedly.

**Computational operation:** Update construction and lexical preference estimates using a gradual learning rule. For a simple frequency-based model:

\[
\widehat{P}_{t+1}(c)=\frac{N_t(c)+\alpha}{N_t+\alpha K}
\]

where \(N_t(c)\) is the count for construction \(c\), \(N_t\) is the total count, \(K\) is the number of alternatives, and \(\alpha\) is a smoothing parameter. More advanced models can use hierarchical priors, recency weighting, or prediction-error learning.

**Output:** Updated expectations that can reduce or alter future garden-path effects, while preserving sensitivity to genuine ambiguity.

## 5. Mathematical operations and candidate models

The following operations can be implemented in a computational model. They should be treated as formal hypotheses or useful abstractions, not as a list of literal calculations performed by identifiable neurons.

### 5.1 Probabilistic parse ranking

\[
P(P_i\mid X)\propto P(X\mid P_i)P(P_i)
\]

Ranks candidate parses by how well they explain the words and how consistent they are with prior language experience.

### 5.2 Surprisal

\[
Surprisal(x_t)=-\log P(x_t\mid x_{1:t-1})
\]

Provides a measurable estimate of how unexpected a word is in its preceding context. It is associated with processing difficulty in many settings, but it is not a complete explanation of syntactic processing.

### 5.3 Entropy over competing parses

\[
H(P)=-\sum_i P(P_i\mid X)\log P(P_i\mid X)
\]

High entropy indicates uncertainty spread across several interpretations. Entropy can help model ambiguity and the need for additional evidence.

### 5.4 Prediction error

\[
\delta_t = observed_t - predicted_t
\]

For categorical or structured data, this should be expressed with a suitable loss function rather than simple numeric subtraction. Large error can trigger candidate revision or increased attention.

### 5.5 Similarity and structural distance

Candidate parses can be compared by tree-edit distance, dependency-edge overlap, role-assignment agreement, or a learned structural similarity measure. A parse with high lexical similarity but different thematic roles should not be treated as semantically equivalent.

### 5.6 Working-memory accessibility

A simple model can combine temporal decay and interference:

\[
A_i(t)=A_i(t_0)e^{-\lambda\Delta t}-\sum_j\rho_{ij}A_j(t)
\]

The interference term is schematic and should be constrained so that activation remains within a valid range. More realistic models can use competitive retrieval or cue-dependent accessibility.

### 5.7 Revision cost

\[
Cost_{revise}=Cost_{retrieval}+Cost_{restructure}+Cost_{role	ext{-}update}+Cost_{integration}
\]

This decomposition allows different garden-path constructions to produce different recovery profiles. Revision may be less likely when the initial analysis is strongly activated, the critical region is distant, or several dependencies must be updated.

### 5.8 Expected utility of further analysis

\[
EU_{continue}=P(correct\mid more\ processing)\cdot Value_{accuracy}-Cost_{processing}
\]

This captures why comprehenders may stop with a good-enough representation when precision is not required or resources are constrained.

## 6. Observable behavioral and neural signatures

A computational implementation should be evaluated against more than whether it outputs the correct final parse.

### 6.1 Reading-time measures

Self-paced reading and eye-tracking can reveal longer processing times at disambiguating regions, regressions to earlier text, rereading, and spillover effects in later regions. These measures indicate processing difficulty but do not uniquely identify the mental operation that produced it.

### 6.2 Comprehension accuracy

Questions about who did what to whom can reveal whether a participant derived the correct thematic-role structure. A person may report that a sentence was difficult yet still preserve the initial interpretation, or may answer correctly without a complete syntactic representation.

### 6.3 Event-related potentials

The P600 is often observed in response to syntactic anomalies or difficult structural reanalysis, but it is not a uniquely specific “reparse meter.” Depending on the construction and account, it may relate to structural revision, conflict between interpretations, integration, or other processes. The N400 is commonly associated with semantic access/integration difficulty, though its interpretation is also context-sensitive. Neither component should be treated as a one-to-one readout of a single algorithmic step.

### 6.4 Neuroimaging

Functional neuroimaging studies have associated garden-path processing with left frontal language regions and with areas of the multiple-demand network. The pattern depends on the particular sentence manipulation and the type of difficulty. Neuroimaging activation alone cannot prove that a region performs a specific computational step.

### 6.5 Individual and language variation

The likelihood of a garden-path effect and success of recovery can vary with language, age, reading skill, working memory, syntactic experience, vocabulary, and familiarity with a construction. Prosody and context may change the preferred analysis. Avoid hard-coding one English-specific example as a universal model of sentence comprehension.

## 7. Important theoretical qualifications

1. **No single universal parsing strategy is established.** Minimal-attachment and late-closure heuristics explain some patterns, but constraint-based, expectation-based, memory-based, and interactive accounts explain other effects. A realistic model should combine cues and be tested across constructions.
2. **Surprise is not the same as syntactic difficulty.** A word can be unexpected but easy to integrate, or expected locally but difficult to attach over a long dependency.
3. **Reanalysis is not guaranteed.** Readers may fail to revise the initial parse, revise only part of it, or arrive at a good-enough interpretation.
4. **Rereading does not prove successful reanalysis.** Eye regressions and rereading can be nonselective or confirmatory; their presence should not automatically be interpreted as deliberate correction.
5. **P600 is not a unique marker of syntactic repair.** It is informative but theoretically debated and sensitive to task and materials.
6. **A grammatical sentence can be semantically unusual.** Plausibility should constrain parsing without becoming an absolute veto.
7. **Neural localization is approximate.** Sentence comprehension relies on interacting networks, not a one-region-per-operation architecture.

## 8. Integrated pseudocode

```text
FUNCTION comprehend_sentence(input_stream, context, task_goal):
    lexical_state = initialize_lexical_state()
    parse_beam = initialize_parse_hypotheses()
    dependency_memory = initialize_dependency_memory()
    discourse_model = load_relevant_context(context)

    FOR each incoming unit x_t IN input_stream:
        cues = extract_input_cues(
            x_t,
            prosody_or_punctuation=True,
            attention_state=current_attention()
        )

        lexical_candidates = retrieve_lexical_and_category_candidates(
            x_t, context=discourse_model
        )

        parse_beam = incrementally_extend_parses(
            parse_beam,
            lexical_candidates,
            cues
        )

        FOR each parse IN parse_beam:
            update_dependencies(parse, dependency_memory)
            integrate_syntax(parse)
            integrate_semantics_and_thematic_roles(parse)
            score_parse_using_available_cues(
                parse,
                syntax=True,
                lexical_preferences=True,
                semantic_plausibility=True,
                discourse=True,
                prosody=True
            )

        parse_beam = retain_top_hypotheses(parse_beam)
        prediction_error = compare_input_with_predictions(x_t, parse_beam)
        ambiguity = estimate_parse_uncertainty(parse_beam)

        IF current_best_parse_is_incompatible(parse_beam) OR
           revision_is_worthwhile(prediction_error, ambiguity, task_goal):

            revision_points = identify_candidate_choice_points(parse_beam)
            alternatives = generate_alternative_structures(
                parse_beam,
                revision_points,
                lexical_candidates,
                context
            )

            alternatives = rebuild_thematic_roles(alternatives)
            alternatives = score_full_sentence_compatibility(alternatives)
            parse_beam = reconcile_old_and_new_interpretations(
                parse_beam, alternatives
            )

        update_working_memory(parse_beam, dependency_memory)

        IF good_enough_for_task(parse_beam, task_goal):
            mark_current_representation_as_sufficient_but_uncertain_if_needed()

    final_interpretation = select_task_appropriate_interpretation(parse_beam)
    confidence = estimate_separate_confidence(
        syntax=final_interpretation.syntax,
        semantics=final_interpretation.semantics,
        thematic_roles=final_interpretation.roles
    )

    update_language_expectations_from_experience(
        input_stream,
        final_interpretation,
        feedback_available=True_or_false
    )

    RETURN final_interpretation, confidence, unresolved_ambiguities
```

## 9. Validation and falsifiable tests

A useful model should predict both successful and unsuccessful comprehension. Recommended tests include:

1. **Temporary ambiguity manipulation:** compare garden-path sentences with unambiguous controls while holding lexical frequency and length as constant as possible. Predict increased reading time or other processing costs at the disambiguating region, with construction-dependent effect sizes.
2. **Thematic-role probes:** ask who performed the action and who was affected. This tests whether the model has updated roles rather than simply detected a conflict.
3. **Prosody manipulation:** present the same lexical sequence with different prosodic boundaries. Test whether predicted parse preferences and comprehension accuracy change.
4. **Context manipulation:** provide discourse that favors one analysis before the ambiguous sentence. Test how strongly contextual information changes the initial parse and later recovery.
5. **Working-memory manipulation:** vary dependency length or add competing material. Test whether increased memory demands reduce accurate revision.
6. **Repeated-exposure adaptation:** expose participants to a structural construction repeatedly. Test whether its expectation profile and garden-path cost change over trials.
7. **Rereading test:** measure regressions and comprehension accuracy separately. Do not assume that rereading predicts accurate reinterpretation.
8. **Good-enough test:** compare immediate plausibility judgments with detailed thematic-role questions. A model may predict a plausible overall gist while retaining an inaccurate syntactic detail.
9. **Cross-language test:** adapt materials to languages with different word-order and morphological cues. The algorithm should learn language-specific constraints rather than rely on English-only heuristics.

Evaluation metrics should include parse accuracy, thematic-role accuracy, confidence calibration, word-by-word surprisal fit, reading-time prediction, and recovery probability. Neural measures may be used as converging evidence, not as a substitute for behavioral validation.

## 10. Failure modes the algorithm must explicitly represent

- **Premature commitment:** one parse receives excessive confidence before adequate evidence arrives.
- **Lexical-category error:** a word is assigned the wrong category or sense.
- **Attachment error:** a phrase is linked to an inappropriate head.
- **Dependency loss:** a relevant earlier word or open dependency becomes inaccessible.
- **Role-assignment persistence:** the initial agent/patient assignment remains active after structural revision.
- **Cue underweighting:** prosody, punctuation, discourse, or semantic plausibility is ignored or assigned an inappropriate reliability.
- **Overreliance on plausibility:** unusual but grammatical interpretations are rejected.
- **Insufficient alternative generation:** the correct structural candidate is never considered.
- **Incomplete reanalysis:** the structure changes but some semantic consequences are not updated.
- **Good-enough misinterpretation:** the model accepts a rough interpretation that is inadequate for the actual task.
- **False confidence:** a fluent or familiar interpretation is treated as certain despite unresolved ambiguity.
- **Overcorrection:** the model revises a valid parse merely because a rare or unexpected continuation appears.

## 11. Research grounding and selected references

The algorithm draws on established psycholinguistic and neuroscience findings while recognizing active debate about how garden-path recovery works.

- **Fractionating difficulty during sentence comprehension using functional neuroimaging** (2024). Functional neuroimaging work that examines garden-path and other sentence-processing difficulties across temporal and frontal cortex. *Cerebral Cortex*. PubMed record: https://pubmed.ncbi.nlm.nih.gov/38314589/ . Open-access article: https://pmc.ncbi.nlm.nih.gov/articles/PMC10847905/ .
- **Retracing the garden-path: Nonselective rereading and no reanalysis** (Christianson et al., 2024). Two eye-tracking studies report that rereading patterns did not support a simple account in which readers selectively return to critical points and reliably repair the parse. *Journal of Memory and Language*, 137, 104515. https://doi.org/10.1016/j.jml.2024.104515 .
- **Neural Mechanisms Underlying the Dynamic Updating of Native Language** (2023). Reports garden-path-related effects in left inferior frontal cortex and left anterior insula/frontal operculum, with changes after repeated exposure. https://pmc.ncbi.nlm.nih.gov/articles/PMC10158591/ .
- **Beyond initial misanalysis: the disruptive nature of garden-path sentences processing** (2025). Reviews evidence that recovery can be partial and that initial analyses may persist; useful for distinguishing full reanalysis from good-enough interpretations. https://doi.org/10.1080/0163853X.2025.2513210 .
- **Strategies of Discourse Comprehension** (van Dijk & Kintsch, 1983). Classic treatment of incremental parsing strategies and the “The old man the boats” example. https://discourses.org/wp-content/uploads/2022/06/Teun-A-van-Dijk-Walter-Kintsch-1983-Strategies-Of-Discourse-Comprehension.pdf .

## 12. Scientific interpretation and implementation boundary

The algorithm is a computational synthesis of candidate mechanisms, not a direct map of neurons to software functions. Biological evidence supports incremental language processing, sensitivity to structural and lexical expectations, increased processing demands at disambiguation, contributions from frontotemporal language regions, and domain-general control under some difficult conditions. The relative importance of prediction, memory retrieval, structural reanalysis, semantic plausibility, and good-enough interpretation remains disputed and varies by construction and experimental task.

For implementation, keep the model modular: lexical access, structure generation, prediction, cue integration, dependency maintenance, conflict detection, revision, semantic-role recomputation, confidence calibration, and adaptation should be separable components. Log every operation with its input evidence, active candidate parses, confidence scores, selected action, and reason for revision. This allows later tests to determine whether a failure originated in lexical access, structure selection, memory maintenance, cue weighting, or incomplete repair.

**End of Point 23.** The next point in the source sequence should address how feedback or clarification can prevent misconceptions from becoming entrenched.
