"""Canonical 44-operation semantic-cognition registry.

The biological descriptions are testable implementation hypotheses, not claims
that the human brain literally executes these software steps in this order.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    operation_id: str
    subsystem: str
    name: str
    functional_responsibility: str
    biological_reference: str


_OPERATION_ROWS = (
    ("SC-001", "Input and lexical access", "Validate semantic request",
     "Validate sequence identity, exclusive text/token input, context and modality contracts.",
     "Distributed language/input pathways provide a usable word-form signal; no single validation neuron is assumed."),
    ("SC-002", "Input and lexical access", "Preserve surface evidence",
     "Retain the original string and supplied token boundaries without rewriting source evidence.",
     "Early word-form traces remain distinguishable from later semantic interpretations."),
    ("SC-003", "Input and lexical access", "Normalize lexical access keys",
     "Create Unicode-normalized, case-folded lookup keys while preserving the surface form.",
     "Abstract lexical access tolerates selected surface variation; normalization is a software approximation."),
    ("SC-004", "Input and lexical access", "Establish token units",
     "Extract or retain ordered word tokens and stable token indices.",
     "Serial word-form access is maintained alongside parallel distributed feature activation."),
    ("SC-005", "Input and lexical access", "Set task and contextual goals",
     "Represent the requested task and caller-supplied contextual feature prior.",
     "Prefrontal and task-control systems bias processing toward task-relevant information."),
    ("SC-006", "Input and lexical access", "Load a provenance-pinned concept snapshot",
     "Freeze the concept inventory, aliases, frequencies and source provenance for read-only inference.",
     "Long-term lexical-semantic knowledge is available to retrieval; the software snapshot is not a neural storage model."),
    ("SC-007", "Episodic word-form memory", "Activate orthographic word-form candidates",
     "Match normalized input against concept labels and stored aliases.",
     "Visual word-form representations can cue lexical access; the package receives text, not raw visual cortex activity."),
    ("SC-008", "Episodic word-form memory", "Generate lexical alternatives",
     "Retrieve exact matches first, then bounded near-neighbor candidates while preserving alternatives.",
     "Partially matching word forms can coactivate competing lexical representations."),
    ("SC-009", "Episodic word-form memory", "Retrieve phonological word-form cues",
     "Collect pronunciation-like cues from concept-linked episodes without inventing missing phonology.",
     "Phonological word-form representations can interact with lexical-semantic access."),
    ("SC-010", "Episodic word-form memory", "Retrieve relevant episodic traces",
     "Search supplied word-form episodes using orthography, concept identity and context.",
     "Hippocampal/medial-temporal episodic retrieval can reinstate bound word-form and context information."),
    ("SC-011", "Episodic word-form memory", "Match remembered surface forms",
     "Compute exact and graded orthographic similarity between input tokens and episodes.",
     "Pattern completion can retrieve an episode from partial cues; the similarity metric is an engineering proxy."),
    ("SC-012", "Episodic word-form memory", "Weight episode context and recency",
     "Combine episode strength, recency and contextual feature overlap as a separate evidence channel.",
     "Retrieval strength and contextual reinstatement can vary across remembered encounters."),
    ("SC-013", "Modality-specific semantic spokes", "Activate visual feature spoke",
     "Score candidate concepts against caller-provided visual feature evidence.",
     "Visual association populations contribute color, shape, surface and visual-form features."),
    ("SC-014", "Modality-specific semantic spokes", "Activate auditory and phonological spoke",
     "Score candidate concepts against sound, phonology and auditory feature evidence.",
     "Auditory association and phonological populations contribute sound-related features."),
    ("SC-015", "Modality-specific semantic spokes", "Activate action and motor spoke",
     "Score candidate concepts against action, manipulation and motor feature evidence.",
     "Premotor, motor and action-observation networks contribute action-related features."),
    ("SC-016", "Modality-specific semantic spokes", "Activate somatosensory spoke",
     "Score candidate concepts against tactile, texture and body-sensation evidence.",
     "Somatosensory association networks contribute tactile and bodily feature information."),
    ("SC-017", "Modality-specific semantic spokes", "Activate olfactory, gustatory and interoceptive spokes",
     "Score candidate concepts against smell, taste and internal-state evidence.",
     "Olfactory, gustatory and interoceptive systems provide modality-specific associations."),
    ("SC-018", "Modality-specific semantic spokes", "Activate affective spoke",
     "Score candidate concepts against caller-provided affective feature evidence.",
     "Affective and valuation networks contribute emotional and motivational associations."),
    ("SC-019", "Modality-specific semantic spokes", "Activate spatial and temporal spoke",
     "Score candidate concepts against location, scale, duration and temporal feature evidence.",
     "Parietal, medial-temporal and association systems contribute spatial/temporal knowledge."),
    ("SC-020", "Modality-specific semantic spokes", "Estimate spoke reliability",
     "Measure evidence coverage and uncertainty per modality; do not treat missing modalities as negative evidence.",
     "Modality-specific signals have context-dependent precision and reliability."),
    ("SC-021", "Distributed semantic feature networks", "Initialize distributed feature activity",
     "Assemble candidate-linked semantic and modality feature activations without collapsing their provenance.",
     "Concept knowledge is represented across distributed modality-linked and association networks."),
    ("SC-022", "Distributed semantic feature networks", "Propagate feature associations",
     "Spread bounded activation across shared semantic features and concept associations.",
     "Feature coactivation and learned association provide a distributed route to related knowledge."),
    ("SC-023", "Distributed semantic feature networks", "Integrate feature evidence",
     "Pool modality-weighted evidence into a sparse, typed feature representation.",
     "Distributed feature populations jointly constrain concept identity."),
    ("SC-024", "Distributed semantic feature networks", "Apply feature competition and sparsification",
     "Suppress weak or redundant feature activations while retaining channel-specific values.",
     "Local inhibition and sparse population activity are plausible mechanisms; the software threshold is not measured biology."),
    ("SC-025", "Distributed semantic feature networks", "Construct a semantic feature state",
     "Build the token-level feature vector and retain the source channel of each contribution.",
     "Distributed semantic features form a state that can be read by a multimodal hub."),
    ("SC-026", "Distributed semantic feature networks", "Compute feature prediction error",
     "Compare incoming feature evidence with candidate-conditioned expectations and retain residuals.",
     "Predictive-processing accounts propose error-sensitive updating; the residual here is a model diagnostic."),
    ("SC-027", "Anterior temporal semantic hub", "Initialize ATL hub activity",
     "Seed a common multimodal concept state from lexical and distributed feature evidence.",
     "Anterior temporal regions are modeled as a transmodal convergence/hub contribution, not a single isolated store."),
    ("SC-028", "Anterior temporal semantic hub", "Converge modality-specific inputs",
     "Combine modality spokes with adaptive reliability weights while preserving individual spoke outputs.",
     "ATL-centered transmodal representations interact with distributed modality-specific spokes."),
    ("SC-029", "Anterior temporal semantic hub", "Settle recurrent hub representation",
     "Run damped recurrent refinement until the common feature state stabilizes or the iteration bound is reached.",
     "Recurrent interactions among temporal and distributed cortical populations can support stable concept states."),
    ("SC-030", "Anterior temporal semantic hub", "Complete partially cued concepts",
     "Use learned concept associations to infer missing but supported features without overwriting observed evidence.",
     "Associative pattern completion can reactivate linked features from partial cues; inferred and observed evidence stay distinct."),
    ("SC-031", "Anterior temporal semantic hub", "Read out hub-level concept activation",
     "Project the settled feature state onto candidate concepts and compute hub compatibility.",
     "A transmodal hub supports access to concept-level representations across input modalities."),
    ("SC-032", "Anterior temporal semantic hub", "Return hub feedback to lexical candidates",
     "Send bounded hub compatibility back to candidate activations while retaining bottom-up lexical scores.",
     "Bidirectional temporal-frontal and lexical-semantic interactions can refine candidate competition."),
    ("SC-033", "Inferior frontal semantic control", "Estimate semantic control demand",
     "Use candidate entropy, conflict and weak evidence to estimate the need for controlled retrieval.",
     "Inferior frontal semantic-control regions are recruited when retrieval is weak, competing or task demanding."),
    ("SC-034", "Inferior frontal semantic control", "Set control goal and task relevance",
     "Convert task instructions and control demand into explicit feature/goal weights.",
     "Frontal control biases retrieval toward task-relevant aspects of knowledge."),
    ("SC-035", "Inferior frontal semantic control", "Generate competing interpretations",
     "Maintain candidate-level alternatives and quantify pairwise competition.",
     "Multiple semantic candidates can be coactivated before selection."),
    ("SC-036", "Inferior frontal semantic control", "Apply lateral inhibitory control",
     "Apply bounded competition to suppress inconsistent candidates without deleting their evidence.",
     "Frontal-temporal control and local inhibitory interactions can reduce competing semantic activations."),
    ("SC-037", "Inferior frontal semantic control", "Reweight context-sensitive evidence",
     "Reweight candidate scores using explicit context and task goals, keeping each evidence channel separate.",
     "Top-down control can emphasize currently relevant semantic features."),
    ("SC-038", "Semantic retrieval and concept selection", "Run controlled retrieval cycles",
     "Perform a bounded retrieve-evaluate-refine loop and stop on convergence or the configured limit.",
     "Controlled semantic retrieval can involve repeated interactions between temporal semantic representations and frontal control."),
    ("SC-039", "Semantic retrieval and concept selection", "Measure ambiguity and selection margin",
     "Calculate posterior entropy, top-candidate margin and evidence sufficiency.",
     "Selection uncertainty is represented computationally; it is not a direct neural scalar measurement."),
    ("SC-040", "Abstract lexical prototypes", "Retrieve abstract lexical prototypes",
     "Retrieve read-only category-level feature prototypes from provenance-tracked concept groups.",
     "Abstract semantic knowledge may be supported by generalized representations across experienced exemplars."),
    ("SC-041", "Abstract lexical prototypes", "Estimate prototype feature structure",
     "Compute frequency-weighted feature centroids without changing stored concept entries.",
     "Statistical regularities across exemplars can support category-level abstraction."),
    ("SC-042", "Abstract lexical prototypes", "Compare exemplar and prototype",
     "Measure concept-to-prototype fit and use it as bounded generalization evidence.",
     "Specific word/concept representations interact with more abstract category knowledge."),
    ("SC-043", "Semantic retrieval and concept selection", "Resolve or abstain",
     "Combine controlled candidate evidence with prototype fit; select only when confidence and margin gates pass.",
     "Selection is a network-level outcome; the software abstention rule is an explicit governance mechanism."),
    ("SC-044", "Output and validation", "Emit semantic state, evidence trace and acceptance report",
     "Return selected concepts, alternatives, source-linked evidence, biological measures and invariant checks.",
     "Neural observables and functional decisions are reported separately; independent empirical validation remains necessary."),
)

OPERATION_REGISTRY: tuple[OperationSpec, ...] = tuple(
    OperationSpec(*row) for row in _OPERATION_ROWS
)

if len(OPERATION_REGISTRY) != 44:
    raise RuntimeError(f"semantic operation registry must contain 44 operations, got {len(OPERATION_REGISTRY)}")
if len({item.operation_id for item in OPERATION_REGISTRY}) != 44:
    raise RuntimeError("semantic operation IDs must be unique")
if tuple(item.operation_id for item in OPERATION_REGISTRY) != tuple(
    f"SC-{index:03d}" for index in range(1, 45)
):
    raise RuntimeError("semantic operation registry must be contiguous and ordered")


def registry_as_dicts() -> list[dict[str, str]]:
    """Return a JSON-friendly copy of the canonical operation registry."""
    return [asdict(item) for item in OPERATION_REGISTRY]
