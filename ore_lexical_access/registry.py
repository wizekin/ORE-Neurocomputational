"""Executable registry for the seven-stage brain-inspired lexical access algorithm."""
from __future__ import annotations

from dataclasses import asdict, dataclass


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


# Seven ordered stages and 36 individually traceable operations.
_ROWS = (
    # Stage 1 — Retrieve stored word identity (5)
    ("S1", "Retrieve stored word identity", "Normalize the incoming word form",
     "Convert to Unicode NFC and case-fold without deleting diacritics.",
     "x = casefold(NFC(input))",
     "Stable orthographic input is represented across distributed visual/orthographic populations."),
    ("S1", "Retrieve stored word identity", "Retrieve lexical-form candidates",
     "Search the supplied orthographic lexicon; retain exact and bounded near-form matches.",
     "C(x) = {e : d_Lev(x, form(e)) <= d_max or sim(x, form(e)) >= s_min}",
     "Left ventral occipitotemporal word-form networks are a useful high-level analogue for learned orthographic patterns; no single neuron stores a word."),
    ("S1", "Retrieve stored word identity", "Calculate orthographic likelihood",
     "Convert edit distance to a graded match likelihood for each stored form.",
     "L(e|x) = exp(-lambda * d_Lev(x, form(e)) / max(|x|, |form(e)|, 1))",
     "Distributed letter-position evidence can support graded activation rather than an all-or-none lookup."),
    ("S1", "Retrieve stored word identity", "Retrieve frequency and exposure priors",
     "Read lexical frequency and accumulated exposure from the supplied entry.",
     "F(e) = log(1 + frequency(e) + exposure(e)); p(e) proportional to exp(beta_F * F(e))",
     "Lifetime familiarity and repeated exposure can bias readiness; inventory counts are software proxies."),
    ("S1", "Retrieve stored word identity", "Estimate stored-identity posterior",
     "Combine orthographic likelihood and lexical prior; preserve homographic and near-form alternatives.",
     "P(e|x) = softmax((log L(e|x) + beta_F * F(e)) / T_id)",
     "Parallel word-form populations can coactivate; relative activity can reflect evidence and familiarity."),
    # Stage 2 — Retrieve word meaning (5)
    ("S2", "Retrieve word meaning", "Enumerate stored meaning hypotheses",
     "Load all provenance-bearing meanings attached to active lexical entries.",
     "M = union_e meanings(e), for e in C(x)",
     "Lexical-semantic access engages distributed temporal and modality-linked feature networks."),
    ("S2", "Retrieve word meaning", "Retrieve semantic feature patterns",
     "Read the stored feature vector for each meaning; do not fabricate missing features.",
     "mu_m = feature_map(m)",
     "An anterior temporal transmodal hub is a useful model of feature convergence with distributed modality-specific spokes."),
    ("S2", "Retrieve word meaning", "Calculate lexical meaning priors",
     "Weight each meaning by the posterior of its lexical entry and its supplied sense prior.",
     "q(m) = P(e(m)|x) * prior(m)",
     "Meaning frequency and word-form activation can bias initial semantic access."),
    ("S2", "Retrieve word meaning", "Calculate semantic-context compatibility",
     "Compare each meaning feature vector with optional context using cosine similarity.",
     "C(m,c) = (mu_m dot c) / (||mu_m|| ||c|| + epsilon)",
     "Contextual and controlled retrieval involve interactions across temporal semantic and inferior frontal networks."),
    ("S2", "Retrieve word meaning", "Normalize and retain meaning alternatives",
     "Combine log prior and context fit; retain multiple senses when evidence is close.",
     "P(m|x,c) = softmax((log(q(m)+epsilon) + beta_c*C(m,c)) / T_sem)",
     "Competing semantic feature patterns can remain active until context or control resolves them."),
    # Stage 3 — Retrieve grammatical category (4)
    ("S3", "Retrieve grammatical category", "Retrieve category labels from lexical entries",
     "Read each candidate entry's supplied lexical_category field.",
     "G = {category(e) : e in C(x)}",
     "Lexical category information is distributed across temporal lexical representations and frontal-temporal language networks."),
    ("S3", "Retrieve grammatical category", "Pool category evidence across candidates",
     "Accumulate identity-posterior mass for each attested grammatical category.",
     "E(g) = sum_e P(e|x) * I(category(e)=g)",
     "Category-selective populations may receive graded evidence from coactive lexical candidates."),
    ("S3", "Retrieve grammatical category", "Integrate caller-supplied syntactic priors",
     "Combine lexical category evidence with optional contextual category priors without replacing the lexical evidence.",
     "S(g) = log(E(g)+epsilon) + beta_g*log(prior_context(g)+epsilon)",
     "Sentence structure and task context can bias category activation through temporal-frontal interactions."),
    ("S3", "Retrieve grammatical category", "Estimate category posterior and ambiguity",
     "Normalize category scores and report confidence and the top-two margin.",
     "P(g|x,c) = softmax(S(g)/T_pos); margin = P(g_1)-P(g_2)",
     "Inferior frontal and posterior temporal systems contribute to syntactic selection/control; the equation is a computational abstraction."),
    # Stage 4 — Morphological decomposition (5)
    ("S4", "Morphological decomposition", "Retrieve supplied affix and root inventory",
     "Index only caller-supplied prefix, root and suffix records.",
     "A = A_prefix union A_root union A_suffix",
     "Morphological processing can recruit distributed word-form and morpheme representations; inventories stand for learned experience."),
    ("S4", "Morphological decomposition", "Generate candidate boundaries",
     "Test every supplied prefix and suffix boundary around a non-empty candidate stem.",
     "b = (p,s,u) where x = p + s + u",
     "Candidate segmentation can be computed from recurring orthographic/morphological patterns."),
    ("S4", "Morphological decomposition", "Verify exact concatenative reconstruction",
     "Reject any split that does not reconstruct the normalized input exactly.",
     "I_reconstruct = I[x == prefix + stem + suffix]",
     "A decomposition hypothesis should be supported by a legal learned form, not by unconstrained substring guessing."),
    ("S4", "Morphological decomposition", "Resolve a stem against lexical/root memory",
     "Require an exact lexical stem entry or supplied root morpheme for each segmentation.",
     "R(s) = lexical_entries(s) union root_morphemes(s)",
     "Stem access can reactivate stored lexical/morphemic form representations."),
    ("S4", "Morphological decomposition", "Preserve whole-word and segmented routes",
     "Add an intact whole-word hypothesis when an exact lexical entry exists; never force decomposition.",
     "H = {whole_word if exact(x)} union {valid prefix-stem-suffix splits}",
     "Whole-form and decomposed routes may coactivate; morphology is not obligatory for every word."),
    # Stage 5 — Retrieve stems and affixes (4)
    ("S5", "Retrieve stems and affixes", "Retrieve stem lexical and morphemic records",
     "Load exact stem entries, root records, frequencies, category labels and stored meaning features.",
     "R_s = lookup_stem(s); mu_s = feature_map(R_s)",
     "Stem representations may reactivate lexical, semantic and grammatical feature populations."),
    ("S5", "Retrieve stems and affixes", "Retrieve prefix and suffix records",
     "Load each identified affix's form, type, frequency, productivity and semantic features.",
     "R_a = lookup(prefix_id, suffix_id)",
     "Affix-form populations can carry learned morphosyntactic and semantic associations."),
    ("S5", "Retrieve stems and affixes", "Retrieve component semantic/category cues",
     "Collect only explicit semantic features and optional caller-supplied category cues for components.",
     "mu_components = {mu_stem, mu_prefix, mu_suffix}; G_a = supplied_category_cues(a)",
     "Distributed temporal semantic and frontal-temporal grammatical networks can represent component contributions."),
    ("S5", "Retrieve stems and affixes", "Compose morpheme-level evidence",
     "Compose stem and affix features while recording each component's contribution.",
     "mu_comp(f) = clip(mu_stem(f) + sum_j eta_j*productivity(a_j)*mu_a_j(f), 0, 1)",
     "Co-activation of morpheme-linked features is a useful functional hypothesis; it is not a claim of literal symbolic addition in cortex."),
    # Stage 6 — Distinguish morphological alternatives (5)
    ("S6", "Distinguish morphological alternatives", "Measure stem support",
     "Convert lexical/root frequency into a bounded support term for each decomposition.",
     "S_stem = log(1+freq_stem) / log(1+freq_max)",
     "Familiar stems may have stronger prior activation; frequency is a proxy."),
    ("S6", "Distinguish morphological alternatives", "Measure affix support and productivity",
     "Combine affix frequency and supplied productivity for each candidate segmentation.",
     "S_affix = mean_j(log(1+freq_j)/log(1+freq_max) * productivity_j)",
     "Productive morpheme patterns can be more available; productivity values must come from an explicit inventory."),
    ("S6", "Distinguish morphological alternatives", "Measure semantic and category compatibility",
     "Compare composed features with active meanings/context and supplied morpheme category cues with category posterior.",
     "S_sem = cosine(mu_comp, mu_active); S_pos = sum_g P(g|x,c)*cue(g)",
     "Temporal semantic and frontal syntactic-control signals can bias competing structural interpretations."),
    ("S6", "Distinguish morphological alternatives", "Compute decomposition log scores and posterior",
     "Combine whole-word prior, structural validity, stem/affix support, semantic fit and category fit.",
     "z(h)=b_h + w_s*S_stem + w_a*S_affix + w_m*S_sem + w_g*S_pos; P(h)=softmax(z/T_morph)",
     "Competition between whole-form and decomposed hypotheses can be approximated by parallel population activity."),
    ("S6", "Distinguish morphological alternatives", "Apply confidence and ambiguity gate",
     "Select a preferred analysis only when posterior confidence and margin meet configured criteria; otherwise abstain and retain alternatives.",
     "select h_1 iff P(h_1)>=theta and P(h_1)-P(h_2)>=delta",
     "Frontal control can bias selection while leaving unresolved alternatives available; abstention prevents forced certainty."),
    # Stage 7 — Activate lexical candidates incrementally (8)
    ("S7", "Activate lexical candidates incrementally", "Initialize candidate populations and character stream",
     "Create one activation state per candidate and expose the input as successively longer prefixes.",
     "x_t = x[0:t], a_i(0)=a_0",
     "Orthographic and lexical populations can become active before the whole word has arrived."),
    ("S7", "Activate lexical candidates incrementally", "Accumulate bottom-up prefix evidence",
     "At each character, compare the observed prefix with the corresponding candidate prefix.",
     "B_i(t)=sim(x_t, form_i[0:|x_t|])",
     "Incremental letter-position evidence can continuously alter candidate activation."),
    ("S7", "Activate lexical candidates incrementally", "Retrieve frequency-based lexical pre-activation",
     "Add a bounded prior readiness term from supplied frequency and exposure counts.",
     "F_i = log(1+freq_i+exposure_i) / max_j log(1+freq_j+exposure_j)",
     "Familiar word forms may start from a higher baseline state."),
    ("S7", "Activate lexical candidates incrementally", "Propagate recurrent excitation",
     "Allow each candidate's recent activation to reinforce its own next state.",
     "R_i(t)=g_e*a_i(t-1)",
     "Recurrent excitatory interactions can maintain and amplify compatible lexical population activity."),
    ("S7", "Activate lexical candidates incrementally", "Calculate lateral inhibition from competitors",
     "Subtract similarity-weighted activity of competing word-form candidates.",
     "I_i(t)=g_i*sum_{j!=i} W_ij*a_j(t-1)/max(1,N-1)",
     "Local inhibitory competition can suppress overlapping alternatives; the weight matrix is a model abstraction."),
    ("S7", "Activate lexical candidates incrementally", "Integrate semantic, category and morphology feedback",
     "Feed current meaning, category and decomposition support back into the candidate population.",
     "T_i(t)=g_sem*M_i + g_pos*G_i + g_morph*H_i",
     "Temporal semantic and frontal control signals can feed back to lexical representations."),
    ("S7", "Activate lexical candidates incrementally", "Update leaky candidate activation",
     "Apply a sigmoid target and damped recurrent update for each arriving character.",
     "a_i(t)=(1-rho)*a_i(t-1)+rho*sigmoid((b+B_i+F_i+R_i-I_i+T_i)/T_a)",
     "A bounded rate-state approximation of recurrent excitation, inhibition and evidence integration; not a literal neuronal equation."),
    ("S7", "Activate lexical candidates incrementally", "Rank final candidates and preserve uncertainty",
     "Return final activations and per-character trajectories, retaining alternatives when activation is close.",
     "rank_i = a_i(T); margin = a_(1)(T)-a_(2)(T)",
     "A distributed lexical population code can settle into a dominant interpretation or remain ambiguous."),
)

OPERATION_REGISTRY = tuple(
    OperationSpec(f"BLA-{index:03d}", *row) for index, row in enumerate(_ROWS, start=1)
)
STAGE_REGISTRY = tuple(
    {"stage_id": stage_id, "stage_name": stage_name,
     "first_operation": next(x.operation_id for x in OPERATION_REGISTRY if x.stage_id == stage_id),
     "last_operation": next(x.operation_id for x in reversed(OPERATION_REGISTRY) if x.stage_id == stage_id),
     "operation_count": sum(x.stage_id == stage_id for x in OPERATION_REGISTRY)}
    for stage_id, stage_name in (
        ("S1", "Retrieve stored word identity"),
        ("S2", "Retrieve word meaning"),
        ("S3", "Retrieve grammatical category"),
        ("S4", "Morphological decomposition"),
        ("S5", "Retrieve stems and affixes"),
        ("S6", "Distinguish morphological alternatives"),
        ("S7", "Activate lexical candidates incrementally"),
    )
)

if len(OPERATION_REGISTRY) != 36 or [x.operation_id for x in OPERATION_REGISTRY] != [
    f"BLA-{i:03d}" for i in range(1, 37)
]:
    raise RuntimeError("The seven-stage registry must contain 36 ordered unique operations")


def registry_as_dicts() -> list[dict[str, str]]:
    return [operation.to_dict() for operation in OPERATION_REGISTRY]
