"""Numbered 56-operation registry for the nine-stage syntax algorithm.

Biological references are computational analogies, not anatomical claims.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class StageSpec:
    stage_id: str
    stage_name: str
    operation_count: int
    primary_output: str
    biological_systems: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OperationSpec:
    operation_id: str
    stage_id: str
    stage_name: str
    operation_name: str
    functional_responsibility: str
    mathematical_form: str
    biological_reference: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


_STAGE_ROWS = (
    ("S1", "Part-of-speech identification", 7, "Token-wise POS candidate distributions", "temporal lexical representations; inferior-frontal selection"),
    ("S2", "Phrase detection", 6, "NP, VP, PP, ADJP and ADVP spans", "posterior temporal phrase grouping; inferior-parietal binding"),
    ("S3", "Dependency parsing", 7, "Head-dependent arcs and a rooted dependency tree", "frontotemporal syntactic relation populations"),
    ("S4", "Constituency parsing", 6, "Ranked phrase-structure trees", "temporal-frontal structural assembly"),
    ("S5", "Hierarchical syntax representation", 5, "Nested syntax tree aligned to dependency relations", "working-memory binding and hierarchical integration"),
    ("S6", "Subject/object assignment", 6, "Predicate-centred grammatical argument frames", "left inferior frontal and posterior temporal argument mapping"),
    ("S7", "Agent/patient thematic roles", 6, "Role distributions for event participants", "temporal semantic features and event/action networks"),
    ("S8", "Syntactic ambiguity resolution", 7, "Updated POS/parse beliefs, margin and alternatives", "recurrent frontotemporal competition and control"),
    ("S9", "Long-distance dependency processing", 6, "Filler-gap links and unresolved dependencies", "phonological/syntactic working memory and frontal control"),
)

# Each tuple: operation, executable responsibility, mathematical sketch, biological reference.
_ROWS = (
    # S1 — POS identification (7)
    ("Normalize lexical form", "Create Unicode NFC/case-folded lookup keys while retaining the original token.", "x_i = casefold(NFC(token_i))", "Distributed orthographic/lexical populations provide a stable word-form cue."),
    ("Retrieve lexical category evidence", "Read caller-supplied category distributions and optional lexicon entries without changing them.", "E_i(c) = sum_e P(e|x_i) I(category(e)=c)", "Temporal lexical representations can activate grammatical-category features in parallel."),
    ("Generate category candidates", "Combine lexical labels, closed-class inventories, capitalization and morphological suffix cues.", "C_i = lexical_i union rule_i union caller_i", "Multiple category-selective populations can remain co-active for an ambiguous word."),
    ("Compute local contextual evidence", "Measure category fit from neighbouring tags and function-word constraints.", "K_i(c) = sum_j w(i,j) phi(c, x_j)", "Posterior temporal and inferior-frontal networks contribute context-sensitive syntactic expectations."),
    ("Compute category log scores", "Combine lexical, morphological and local-context evidence with bounded weights.", "s_i(c) = log(prior_i(c)+eps) + beta K_i(c) + M_i(c)", "Converging evidence changes relative population drive rather than switching a single category cell."),
    ("Normalize POS beliefs", "Apply temperature-scaled softmax and retain every candidate above numerical floor.", "P_i(c) = exp(s_i(c)/T) / sum_k exp(s_i(k)/T)", "Competition and normalization approximate graded category-population activity."),
    ("Select or preserve POS alternatives", "Record top category, confidence, margin and an abstention flag when evidence is weak.", "margin_i = P_i(c1)-P_i(c2)", "Frontal selection can bias a winner while parallel alternatives remain available."),
    // S2 — phrase detection (6)
    ("Initialize phrase boundary cues", "Mark likely phrase starts/ends from determiners, adpositions, conjunctions, punctuation and POS transitions.", "B_i = f(POS_i, POS_(i-1), POS_(i+1))", "Sequential attention and boundary-sensitive temporal responses provide grouping cues."),
    ("Detect nominal heads", "Find noun, proper-noun and pronoun anchors with local head confidence.", "H_NP(i) = P_i(NOUN)+P_i(PROPN)+P_i(PRON)", "Temporal lexical populations activate head-word representations."),
    ("Expand noun phrases", "Expand nominal anchors over compatible determiners, adjectives, numerals and compounds.", "NP[a:b] = closure(head, compatible_left/right)", "Inferior-parietal and temporal binding can associate modifiers with a nominal head."),
    ("Assemble verb phrases", "Group verbs, auxiliaries, particles and adverbs; retain argument boundaries as separate phrase spans.", "VP[a:b] = chain(AUX, VERB, ADV, PART)", "Posterior temporal/frontal populations bind a predicate with its local functional material."),
    ("Detect prepositional/adjectival/adverbial phrases", "Recognize adposition-plus-NP, adjective and adverb spans with rule-based templates.", "PP = ADP + NP; ADJP = ADV? + ADJ; ADVP = ADV+", "Distributed syntactic populations support phrase-type distinctions."),
    ("Score and prune phrase spans", "Score spans from POS posterior, head strength and boundary compatibility; preserve overlaps when licensed.", "score(p) = mean(log P_i(tag_i)) + boundary_bonus", "Competitive grouping can suppress weak groupings while retaining alternatives."),
    // S3 — dependency parsing (7)
    ("Generate candidate heads", "For every token, enumerate possible heads and an artificial ROOT, excluding self-arcs.", "A_i = {ROOT} union {j: j != i}", "Potential relation populations receive parallel input from candidate word positions."),
    ("Extract arc features", "Compute distance, direction, POS compatibility, function-word cues, phrase co-membership and predicate proximity.", "f(i,j) = [distance, direction, POS, boundary, phrase]", "Frontotemporal networks integrate lexical, positional and structural cues."),
    ("Propose dependency relations", "Assign candidate labels such as nsubj, obj, det, amod, aux, case, advmod, obl and acl.", "r(i,j) = argmax_r P(r|f(i,j))", "Relation-sensitive populations encode grammatical functions in context."),
    ("Score head-relation candidates", "Combine local relation templates and distance penalties into a scalar arc score.", "score(i->j,r) = w_r^T f(i,j) - lambda*distance", "Relative activation supports competition among possible syntactic links."),
    ("Select the sentence root", "Choose the strongest finite/lexical predicate, falling back to the strongest available content word.", "root = argmax_j [verb_bonus(j)+clause_evidence(j)]", "A clause-level event/predicate representation acts as an integration anchor."),
    ("Build a single-head dependency tree", "Select one governor per non-root token and repair cycles deterministically.", "head(i) = argmax_j score(i->j); cycle repair => arborescence", "Recurrent syntactic binding promotes a coherent relation network."),
    ("Validate dependency structure", "Check one root, one head per non-root token, index validity and acyclicity.", "indegree(i)=1 for i!=root; indegree(root)=0; DAG", "Error monitoring detects inconsistent relation assignments; it is not a literal tree verifier in cortex."),
    // S4 — constituency parsing (6)
    ("Create lexical preterminals", "Convert each token and its POS alternatives into chart leaves with log posterior weights.", "chart[i,i+1] = {(tag_i, log P_i(tag_i))}", "Word-category populations provide lexical leaves for larger structural units."),
    ("Initialize span chart", "Create dynamic-programming cells for every contiguous token interval.", "C[i,j] = empty for 0<=i<j<=n", "Local and global structural hypotheses can be maintained concurrently."),
    ("Apply binary grammar productions", "Combine adjacent chart spans through the explicit weighted phrase-structure grammar.", "score(A->BC) = score(B)+score(C)+log p(A->BC)", "Frontal-temporal integration binds two constituent populations into a larger phrase."),
    ("Apply unary closure", "Promote lexical and phrase categories through unary productions while preventing cycles.", "score(A->B) = score(B)+log p(A->B)", "Hierarchical category activation allows a local unit to participate in a larger constituent."),
    ("Rank N-best full parses", "Keep bounded alternatives per chart cell and rank complete sentence trees by log score.", "P(tree) proportional to exp(score(tree)/T)", "Competing structural interpretations remain active before selection."),
    ("Handle incomplete grammar coverage", "Return a transparent FRAG tree with token leaves when no complete S production is licensed.", "fallback = FRAG(token_1,...,token_n)", "Uncertain input can retain a partial representation rather than being discarded."),
    // S5 — hierarchical representation (5)
    ("Instantiate hierarchical nodes", "Create typed nodes with stable IDs, token spans, lexical heads and child lists.", "node = (label, start, end, head, children)", "Distributed groups are bound into nested phrase and clause representations."),
    ("Encode parent-child relations", "Store immediate dominance separately from transitive ancestor relationships.", "parent(v)=u; ancestor^+(u,v)", "Hierarchical working-memory binding maintains local and nested relations."),
    ("Align constituency and dependency views", "Attach dependency head/relation metadata to leaves and constituent heads.", "align(token_i, leaf_i, head(i), relation_i)", "Frontal-temporal systems coordinate phrase structure with grammatical relations."),
    ("Attach phrase and clause annotations", "Mark clause spans, phrase types, and their head tokens without flattening nesting.", "span(node)=[start,end); depth(child)=depth(parent)+1", "Nested representations are maintained across distributed cortical populations."),
    ("Emit traceable syntax hierarchy", "Expose tree, token alignment, dependency references and provenance in JSON-safe form.", "H = serialize(root, arcs, spans, evidence)", "A coherent multi-level representation is the computational target, not a literal neural data structure."),
    // S6 — subject/object assignment (6)
    ("Identify predicate centres", "Collect lexical verbs and predicate-like auxiliaries with clause membership.", "V = {i: POS_i=VERB}", "Event/predicate representations recruit frontotemporal language populations."),
    ("Collect nominal argument candidates", "Find nominal dependents and nearby NP heads within the predicate's clause.", "N_i = {j: POS_j in nominal and clause(j)=clause(i)}", "Temporal lexical and frontal structural systems jointly represent possible arguments."),
    ("Assign subject relations", "Use dependency labels, order, agreement-like cues and clause boundaries to identify subjects.", "P(nsubj|j,v) = softmax(subject_features)", "Frontal-temporal relation mapping links a predicate to its grammatical subject."),
    ("Assign direct/indirect objects", "Identify object dependents and recipient/theme candidates from dependency and verb-frame cues.", "P(obj|j,v), P(iobj|j,v) = softmax(object_features)", "Argument structure populations distinguish predicate complements."),
    ("Estimate voice and grammatical function", "Detect passive auxiliaries/participles and reinterpret grammatical subject versus object.", "voice = passive when aux_pass and participle evidence exceed threshold", "Syntactic control integrates auxiliary, morphology and argument order cues."),
    ("Build predicate argument frames", "Return predicate-centred subject, object, indirect-object and oblique records with confidence.", "Frame(v)={nsubj,obj,iobj,obl}", "Distributed event representations bind participants to a predicate."),
    // S7 — thematic roles (6)
    ("Retrieve verb semantic frame", "Load a compact, overridable verb-to-role expectation inventory.", "F_v(role,features) = caller_frame or default_frame(v)", "Temporal semantic representations carry event knowledge and selectional preferences."),
    ("Retrieve argument semantic features", "Read supplied token semantic features; do not infer missing features as facts.", "x_j = caller_semantic_features(j)", "Distributed semantic feature populations provide properties of candidate referents."),
    ("Compute role compatibility", "Compare argument features with role expectations using weighted cosine/overlap.", "compat(j,r) = dot(x_j,F_v(r))/(||x_j|| ||F_v(r)||+eps)", "Event-semantic populations integrate participant features with predicate expectations."),
    ("Integrate syntactic and voice evidence", "Use active/passive grammatical mapping as a prior, not as an infallible semantic label.", "s(j,r)=syntax_prior(j,r)+beta*compat(j,r)", "Temporal event knowledge interacts with frontal syntactic structure."),
    ("Normalize thematic-role beliefs", "Produce role probabilities over agent, patient, experiencer, theme and recipient.", "P(r|j,v)=softmax(s(j,r)/T_role)", "Population competition approximates graded thematic-role activation."),
    ("Report role confidence and evidence", "Expose structural source, semantic match, posterior and missing-feature warnings.", "confidence=max_r P(r|j,v)", "Control/monitoring systems track uncertainty in event interpretation."),
    // S8 — ambiguity resolution (7)
    ("Collect competing interpretations", "Collect low-margin POS, phrase, dependency, parse and role alternatives.", "U = {h: posterior(h)<threshold or margin(h)<delta}", "Multiple linguistic populations can remain co-active during ambiguity."),
    ("Compute syntactic compatibility", "Score each POS/parse alternative against local phrase and dependency constraints.", "S_syn(h)=sum_i log P(tag_i|context)+arc_support(h)", "Frontal-temporal recurrence supports structural competition."),
    ("Compute semantic plausibility", "Add caller-supplied semantic feature compatibility with verb-frame role expectations.", "S_sem(h)=sum_(j,r) compat(j,r)", "Distributed semantic and event knowledge biases syntactic interpretation."),
    ("Run damped belief updates", "Iteratively combine prior, syntax and semantic evidence without deleting alternatives.", "q^(t+1)=(1-rho)q^t+rho*softmax((log q^t+g_s S_syn+g_m S_sem)/T)", "Recurrent excitation and inhibition provide a generic settling analogy."),
    ("Calculate posterior margins and entropy", "Quantify separation between top hypotheses and uncertainty across alternatives.", "margin=p1-p2; H=-sum_i p_i log(p_i+eps)", "Population dispersion is a proxy for unresolved competition."),
    ("Select or abstain", "Choose only when confidence and margin clear thresholds; otherwise expose an unresolved decision.", "select iff p1>=tau and p1-p2>=delta", "Frontal control biases a stable interpretation while monitoring confidence."),
    ("Preserve resolved and rejected hypotheses", "Return ranked alternatives, evidence contributions, iterations and convergence diagnostics.", "alternatives=sort(q); delta_t=||q^t-q^(t-1)||_1", "Resolved and suppressed interpretations remain traceable in the model's output."),
    // S9 — long-distance dependencies (6)
    ("Detect dependency fillers", "Find wh-phrases and relative pronouns that may bind a remote argument position.", "F={who,whom,whose,what,which,where,when,why,how}", "Working-memory representations maintain an early filler cue."),
    ("Identify clause boundaries", "Find finite predicates, complementizers, relative-clause markers and punctuation boundaries.", "clauses = intervals induced by SCONJ/relative markers/predicate cues", "Frontal and temporal systems maintain clause-level structure over intervening material."),
    ("Generate candidate gap sites", "Find predicates with an unsatisfied subject/object/oblique slot licensed by verb valency.", "G(v)={role: expected(v,role) and role not overtly filled}", "Working memory keeps unresolved argument expectations active."),
    ("Score filler-gap compatibility", "Combine role fit, clause membership, valency, distance and intervening-subject evidence.", "score(f,g)=role_fit+valency+clause_fit-lambda*distance", "Frontal control prioritizes plausible retrieval targets across intervening words."),
    ("Create enhanced remote dependency", "Attach a non-tree enhanced arc from the filler to the best supported gap predicate.", "edge=(head=predicate, dependent=filler, relation=role, remote=True)", "A maintained filler representation is linked to a later syntactic retrieval site."),
    ("Validate binding and unresolved cases", "Check index validity, preserve the basic tree, and report ambiguity when no gap clears threshold.", "valid(edge) and basic_tree_unchanged; abstain if max_score<tau", "Monitoring detects retrieval failure or competing gap hypotheses."),
)

STAGE_REGISTRY = tuple(
    StageSpec(stage_id, name, count, output, biology)
    for stage_id, name, count, output, biology in _STAGE_ROWS
)

_operation_rows = []
_cursor = 1
for (stage_id, stage_name, _, _, _), start, end in zip(
    _STAGE_ROWS,
    (1, 8, 14, 21, 27, 32, 38, 44, 51),
    (7, 13, 20, 26, 31, 37, 43, 50, 56),
):
    for operation_name, responsibility, equation, biology in _ROWS[start - 1:end]:
        _operation_rows.append(OperationSpec(
            f"SYN-{_cursor:03d}", stage_id, stage_name, operation_name,
            responsibility, equation, biology,
        ))
        _cursor += 1

OPERATION_REGISTRY = tuple(_operation_rows)


def registry_as_dicts() -> dict[str, list[dict[str, object]]]:
    return {
        "stages": [item.to_dict() for item in STAGE_REGISTRY],
        "operations": [item.to_dict() for item in OPERATION_REGISTRY],
    }
