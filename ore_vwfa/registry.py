"""Canonical six-stage, 24-operation VWFA registry with equations and biological qualifications."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one numbered VWFA operation, including its inputs, outputs, mathematics, and scope."""

    operation_id: str
    stage: str
    stage_name: str
    step: int
    operation_name: str
    objective: str
    mathematical_operations: tuple[str, ...]
    biological_reference: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping for documentation, traces, and experiment reports."""
        return asdict(self)


_ROWS = (
    ("A", "Early visual feature extraction", "Acquire and represent the visual input",
     "Sample pixels; estimate local luminance and variance; calculate local contrast and normalized contrast.",
     ("spatial sampling", "Gaussian local mean", "local variance=max(E[I²]-E[I]²,0)", "C=I-μ", "I_norm=C/sqrt(σ²+ε)"),
     "Retinal photoreceptors, center-surround retinal/LGN pathways, and contrast adaptation; image arrays are engineering measurements rather than literal retinal signals.",
     ("visual_image",), ("normalized_image", "local_luminance", "local_contrast")),
    ("A", "Early visual feature extraction", "Extract elementary visual features",
     "Apply even/odd oriented Gabor filters across orientations and spatial frequencies; calculate phase-tolerant energy and gain control.",
     ("Gabor convolution R=I*g", "quadrature energy E=sqrt(Re²+Ro²)", "rectification", "multi-scale filtering", "divisive normalization"),
     "V1 orientation/spatial-frequency-selective populations and earlier contrast pathways; Gabor kernels approximate receptive-field tuning, not individual neurons.",
     ("normalized_image", "filter_config"), ("orientation_energy_maps", "simple_cell_responses", "normalized_feature_maps")),
    ("A", "Early visual feature extraction", "Combine visual features into more complex structures",
     "Integrate neighboring feature channels into candidate conjunctions such as stroke intersections, curves, endpoints, and letter-like shapes.",
     ("feature-vector construction", "weighted sums W x+b", "nonlinear activation", "feature conjunction", "population competition"),
     "Intermediate visual populations combine simpler responses into more complex configurations; the implemented conjunction units are explicit computational proxies.",
     ("normalized_feature_maps",), ("conjunction_maps", "shape_evidence")),
    ("A", "Early visual feature extraction", "Preserve spatial relationships between visual components",
     "Find candidate visual regions, normalize their coordinates, and retain the identity/feature evidence at each ordered position.",
     ("connected-component labeling", "bounding-box/centroid estimation", "relative position r=(x-xmin)/(xmax-xmin)", "ordinal sequence sorting"),
     "Retinotopic visual coding and spatial relationships that support later position-sensitive orthographic processing; regions are not asserted to be correctly segmented characters.",
     ("conjunction_maps", "normalized_feature_maps"), ("candidate_regions", "position_codes", "visual_descriptors")),
    ("B", "Hierarchical visual populations", "Construct populations of simple visual detectors",
     "Represent each candidate region as a distributed vector over orientation, scale, and local-shape populations.",
     ("vector construction", "weighted dot products", "bias addition", "nonlinear activation", "population normalization"),
     "Distributed neural population coding; a vector channel is a model feature, not a one-neuron-per-feature claim.",
     ("visual_descriptors",), ("simple_population_activity", "population_vectors")),
    ("B", "Hierarchical visual populations", "Build increasingly complex receptive-field populations",
     "Compose simple responses into complex visual configurations through multiple affine/nonlinear hierarchy levels.",
     ("matrix multiplication h_l=f(W_l h_(l-1)+b_l)", "pooling", "nonlinear activation", "channel normalization"),
     "V2/V4 and downstream ventral visual hierarchy motivate increasing receptive-field complexity; this is a compact hierarchy, not a full cortical map.",
     ("population_vectors", "normalized_feature_maps"), ("hierarchical_maps", "complex_population_vectors")),
    ("B", "Hierarchical visual populations", "Build position-tolerant visual representations",
     "Pool local feature evidence to tolerate small position/scale changes while keeping an independent ordinal-position channel.",
     ("adaptive spatial pooling", "mean/max pooling", "feature normalization", "identity-position channel concatenation"),
     "Increasing position tolerance along ventral visual pathways while preserving information needed to encode letter order; complete position invariance would be harmful to reading.",
     ("hierarchical_maps", "position_codes"), ("pooled_identity_features", "position_preserving_features")),
    ("B", "Hierarchical visual populations", "Establish candidate orthographic representations",
     "Compare visual descriptors with caller-supplied grapheme prototypes using variance-scaled distance and population competition.",
     ("diagonal Mahalanobis distance", "RBF tuning exp(-d²/2)", "softmax", "uncertainty/unknown mass"),
     "Learned grapheme-sensitive distributed populations; labelled prototypes are user-provided model references, not claims of a biological lookup table.",
     ("position_preserving_features", "grapheme_prototypes"), ("grapheme_posteriors", "orthographic_candidates", "unknown_mass")),
    ("C", "Posterior-to-anterior VWFA specialization", "Activate posterior VWFA populations",
     "Encode local letter-like evidence while retaining detailed visual and positional information.",
     ("affine population transform", "ReLU/softplus", "feature normalization", "posterior evidence pooling"),
     "Posterior ventral occipitotemporal/VWFA-1 is relatively visually detailed and sensitive to local orthographic form; boundaries are functional hypotheses, not precise anatomical parcels.",
     ("grapheme_posteriors", "position_preserving_features"), ("posterior_vwfa_activity", "posterior_orthographic_code")),
    ("C", "Posterior-to-anterior VWFA specialization", "Integrate multiple letters into sublexical units",
     "Build overlapping bigrams, trigrams, quadrigrams, and directed non-adjacent letter pairs with position-distance weighting.",
     ("ordered n-gram extraction", "pairwise probability products", "distance kernel K(d)=exp(-d/τ)", "weighted aggregation"),
     "Intermediate orthographic populations can encode recurring letter combinations; n-gram features are a computational description, not a proven discrete neural token set.",
     ("posterior_orthographic_code", "grapheme_posteriors"), ("sublexical_ngrams", "directed_pair_activity")),
    ("C", "Posterior-to-anterior VWFA specialization", "Activate anterior VWFA populations",
     "Transform local and sublexical evidence into more abstract whole-form representations with reduced sensitivity to font and exact retinal layout.",
     ("layer-wise weighted transforms", "nonlinear activation", "identity pooling", "feature conjunction"),
     "More anterior ventral occipitotemporal regions show responses to higher-order orthographic structure; exact division of VWFA subregions varies across studies and individuals.",
     ("posterior_vwfa_activity", "sublexical_ngrams"), ("anterior_vwfa_activity", "abstract_orthographic_code")),
    ("C", "Posterior-to-anterior VWFA specialization", "Integrate visual familiarity and orthographic invariance",
     "Compare alternate visual views and normalized forms, balancing visual-form invariance against letter identity and order preservation.",
     ("cosine similarity", "view-consistency loss", "identity contrast", "case/Unicode normalization", "position-sensitive similarity"),
     "Literacy supports learned tolerance to fonts and visual transformations while reading retains orientation and ordinal distinctions; indiscriminate mirror/position invariance is not assumed.",
     ("abstract_orthographic_code", "visual_views", "orthographic_sequence"), ("invariance_score", "orthographic_identity_code")),
    ("D", "Whole-word and morpheme-sensitive representations", "Generate whole-word representations",
     "Score supplied whole-word forms using ordered orthographic similarity and an exposure-frequency prior; preserve alternatives.",
     ("edit/cosine similarity", "frequency term λ log(1+F)", "softmax competition", "candidate entropy"),
     "Familiar written forms can activate whole-word representations; the lexicon and frequencies must be supplied with provenance.",
     ("orthographic_identity_code", "lexicon"), ("whole_word_candidates", "whole_word_posterior")),
    ("D", "Whole-word and morpheme-sensitive representations", "Detect potential morphological components",
     "Enumerate bounded concatenative analyses from a caller-supplied morpheme inventory and score form/frequency compatibility.",
     ("substring matching", "boundary search", "log-frequency prior", "P(M|O)∝P(O|M)P(M)", "bounded search"),
     "Visual word processing can be sensitive to recurring morphemic form; morphology and its meanings must not be inferred from an unverified spelling list.",
     ("orthographic_identity_code", "morpheme_inventory"), ("morphology_candidates", "morpheme_boundaries")),
    ("D", "Whole-word and morpheme-sensitive representations", "Activate morphological representations in parallel with whole words",
     "Maintain separate whole-word, stem, prefix, suffix, and inflection activation vectors and combine them without serially forcing decomposition.",
     ("parallel activation", "role-wise weighted sum", "normalization", "recurrent interaction"),
     "Whole-word and constituent-form processing may co-activate; the implementation keeps these evidence channels distinct.",
     ("whole_word_posterior", "morphology_candidates"), ("parallel_morphology_activity", "combined_orthographic_evidence")),
    ("D", "Whole-word and morpheme-sensitive representations", "Access lexical and semantic information associated with morphemes",
     "Retrieve only supplied stem/affix semantic and grammatical features and compose a transparent feature vector in the wider language network.",
     ("associative retrieval", "weighted vector composition", "feature binding", "grammatical-feature union", "context similarity"),
     "Semantic and grammatical composition involves distributed temporal/frontal language systems; VWFA contributes orthographic input and does not alone contain word meaning.",
     ("morphology_candidates", "lexicon", "context_features"), ("morpheme_semantic_evidence", "composed_language_features")),
    ("D", "Whole-word and morpheme-sensitive representations", "Select and stabilize the appropriate word representation",
     "Iteratively combine visual, lexical, morphological, recurrent, and contextual evidence with bounded inhibition and an explicit convergence test.",
     ("recurrent state update", "Bayesian/log-evidence addition", "softmax competition", "confidence and margin", "L2 convergence criterion"),
     "Interactive visual-word recognition includes recurrent and context-sensitive exchange; this does not imply context can override absent visual evidence without cost.",
     ("whole_word_candidates", "parallel_morphology_activity", "composed_language_features", "context_features"), ("stabilized_candidates", "selected_candidate", "convergence_diagnostics")),
    ("E", "Biological cortical connectivity", "Establish bottom-up connectivity from visual regions",
     "Propagate visual hierarchy activity through a directed visual-to-VWFA graph with damped time filtering.",
     ("graph adjacency multiplication", "synaptic weighted summation", "nonlinear transfer", "exponential low-pass filtering"),
     "Feedforward visual pathways convey signals from early visual cortex toward ventral occipitotemporal areas; edge weights are configurable model parameters.",
     ("visual_hierarchy_activity", "posterior_vwfa_activity"), ("bottom_up_network_state", "visual_network_trace")),
    ("E", "Biological cortical connectivity", "Establish connectivity with language-related brain regions",
     "Exchange orthographic activity with IFG, MTG, ATL, STG, angular gyrus, and posterior parietal populations over a recurrent graph.",
     ("adjacency matrix", "graph propagation x_(t+1)=φ(Ax_t+Bu_t)", "row normalization", "functional correlation"),
     "VWFA participates in distributed reading-language networks; connectivity patterns vary by subregion, task, age, and individual.",
     ("bottom_up_network_state", "composed_language_features"), ("regional_activity", "connectivity_matrix", "network_readout")),
    ("E", "Biological cortical connectivity", "Integrate top-down and bottom-up information",
     "Generate an orthographic prediction from higher-level language/context state, calculate residual error, and combine it with bottom-up evidence by precision.",
     ("prediction matrix product", "error e=x-x_hat", "precision Π=1/(σ²+ε)", "precision-weighted fusion", "recurrent update"),
     "Predictive feedback is a broad network hypothesis; model prediction errors and precision terms are explicit computational variables, not direct neuronal measurements.",
     ("regional_activity", "bottom_up_network_state", "context_features"), ("prediction_error", "precision_weighted_state", "refined_orthography")),
    ("E", "Biological cortical connectivity", "Coordinate attention and visual-language processing",
     "Allocate normalized attention over candidate positions and regions, pool attended features, and apply bounded task gain.",
     ("attention logits", "softmax", "weighted sum", "gain modulation", "spatial normalization"),
     "Visual attention, gaze and task demands modulate reading networks; the model's eye-position variable is an abstract coordinate, not an eye tracker.",
     ("refined_orthography", "position_codes", "context_features"), ("attention_weights", "attended_representation", "task_gain")),
    ("F", "Developmental specialization through literacy", "Establish initial visual and linguistic learning associations",
     "Pair hashed orthographic features with caller-supplied phonological targets using an error-correcting associative delta rule.",
     ("feature hashing", "prediction y_hat=xW", "error δ=y-y_hat", "Hebbian/delta update ΔW=η xᵀδ", "bounded weight update"),
     "Literacy learning changes associations between visual forms and pre-existing spoken-language representations; the examples and targets must be explicitly supplied.",
     ("learning_examples", "learning_authorized"), ("visual_phonological_weights", "association_learning_report")),
    ("F", "Developmental specialization through literacy", "Develop specialized word-sensitive neural populations",
     "Estimate word-versus-control selectivity from explicit labelled examples and compute a contrastive specialization objective.",
     ("S_word=R_word-R_control", "softmax", "cross-entropy", "population selectivity contrast", "supervised update"),
     "Literacy is associated with increasing word selectivity in ventral visual populations; this model objective is an engineering hypothesis, not a biological learning law.",
     ("learning_examples", "word_control_labels"), ("word_selectivity", "specialization_loss", "category_response_summary")),
    ("F", "Developmental specialization through literacy", "Consolidate efficient orthographic processing",
     "Update fast and slow orthographic prototypes, summarize recurring patterns, and estimate pattern-completion readiness from experience counts.",
     ("EMA μ_fast←(1-η)μ_fast+ηx", "slow consolidation EMA", "frequency accumulation", "similarity-based pattern completion", "regularization"),
     "Repeated literacy experience supports stable, efficient word-form recognition; consolidation time scales are proxies and are not measured synaptic constants.",
     ("authorized_learning_examples", "learned_fast_prototypes"), ("fast_memory", "slow_memory", "consolidation_report")),
)


STAGE_STEP_COUNTS = {"A": 4, "B": 4, "C": 4, "D": 5, "E": 4, "F": 3}

OPERATION_REGISTRY = tuple(
    OperationSpec(
        operation_id=f"VWFA-{index:03d}",
        stage=row[0], stage_name=row[1], step=index,
        operation_name=row[2], objective=row[3], mathematical_operations=tuple(row[4]),
        biological_reference=row[5], inputs=tuple(row[6]), outputs=tuple(row[7]),
    )
    for index, row in enumerate(_ROWS, start=1)
)

if len(OPERATION_REGISTRY) != 24:
    raise RuntimeError("VWFA registry must contain exactly 24 operations")
if tuple(item.operation_id for item in OPERATION_REGISTRY) != tuple(f"VWFA-{i:03d}" for i in range(1, 25)):
    raise RuntimeError("VWFA operation identifiers must be ordered VWFA-001 through VWFA-024")


def registry_as_dicts() -> dict[str, list[dict[str, Any]]]:
    """Return the six-stage operation registry as JSON-ready dictionaries."""
    return {"operations": [item.to_dict() for item in OPERATION_REGISTRY]}
