"""Stable registry for the sixteen functional and biological-reference visual-orthography operations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one algorithm step, its mathematical operations and biological interpretation."""

    operation_id: str
    stage: str
    stage_name: str
    operation_name: str
    objective: str
    mathematical_operations: tuple[str, ...]
    biological_reference: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the operation specification as a JSON-ready dictionary."""
        return asdict(self)


_OPERATION_ROWS = (
    ("A", "Visual acquisition and early feature processing", "Acquire the visual pattern",
     "Sample luminance, estimate local contrast and normalize local intensity.",
     ("spatial sampling I(x,y)->I[i,j]", "local contrast C=I-I_local",
      "local z-score (I-mu)/(sigma+epsilon)"),
     "Photoreceptors, retinal center-surround pathways and LGN contrast transmission; the sampled array is a software representation.",
     ("visual_image",), ("normalized_image", "contrast_map")),
    ("A", "Visual acquisition and early feature processing", "Extract elementary visual features",
     "Detect oriented edges, bars, curves and spatial-frequency components.",
     ("Gabor convolution I*g(theta,lambda)", "quadrature energy sqrt(Re^2+Ro^2)",
      "rectification", "multi-scale feature stacking", "divisive normalization"),
     "V1-like orientation and spatial-frequency selective receptive fields; Gabor filters approximate response profiles, not individual neurons.",
     ("normalized_image",), ("orientation_energy_maps", "feature_stack")),
    ("A", "Visual acquisition and early feature processing", "Integrate features into candidate letter shapes",
     "Combine strokes, curves, corners and intersections into candidate grapheme-form feature patterns.",
     ("feature-vector pooling", "weighted feature integration", "prototype distance",
      "nonlinear population tuning", "softmax competition"),
     "Hierarchical V2/V4 and ventral visual feature integration; candidate masks and templates remain engineering constructs.",
     ("feature_stack", "candidate_regions", "optional prototypes"),
     ("candidate_shape_features", "visual_form_candidates")),
    ("B", "Recognizing letters across fonts and visual variations", "Map visual variants to abstract letter identities",
     "Associate different glyph appearances with a shared labelled orthographic identity.",
     ("diagonal Mahalanobis distance", "RBF-like tuning exp(-d^2/2)", "population summation",
      "softmax identity probabilities"),
     "Progressively abstract ventral occipitotemporal and VWFA-sensitive representations; visual experience can still modulate responses.",
     ("visual_form_candidates", "letter_prototypes"), ("letter_identity_populations",)),
    ("B", "Recognizing letters across fonts and visual variations", "Learn invariance across visual transformations",
     "Use explicitly labelled variants to update a shared prototype and measure within-identity dispersion.",
     ("exponential moving average m'=(1-eta)m+eta*x", "within-class cosine distance",
      "variance update", "identity-preserving regularization"),
     "Experience-dependent synaptic plasticity can associate visual forms with common identities; the update rule is a computational learning hypothesis.",
     ("letter_prototypes", "labeled_variants"), ("updated_prototypes", "invariance_report")),
    ("C", "Position-specific letter conjunctions", "Estimate approximate letter positions",
     "Retain each letter's horizontal location and normalize it within the written string.",
     ("center-of-mass estimate", "relative position r=(x-x_min)/(x_max-x_min)",
      "spatial normalization"),
     "Retinotopic visual maps and position-sensitive orthographic populations preserve spatial information before and during reading.",
     ("letter_identity_populations", "glyph_regions"), ("relative_letter_positions",)),
    ("C", "Position-specific letter conjunctions", "Represent positional uncertainty",
     "Represent each location as a distribution rather than a perfectly rigid ordinal slot.",
     ("Gaussian tuning exp(-(r-mu)^2/(2*sigma^2))", "probability normalization",
      "uncertainty-dependent population spread"),
     "Population activity can represent noisy spatial estimates; Gaussian distributions are a useful model, not a claim about explicit neural probability tables.",
     ("relative_letter_positions", "position_uncertainty"), ("position_probability_maps",)),
    ("C", "Position-specific letter conjunctions", "Build position-specific letter conjunctions",
     "Detect ordered letter pairs with compatibility for their relative spacing and anchor position.",
     ("pairwise multiplication L_i(a)*L_j(b)", "ordered enumeration i<j",
      "Gaussian distance tuning", "activation aggregation"),
     "Position-specific conjunction populations are a candidate account of letter-combination sensitivity; no unique circuit is established.",
     ("letter_identity_populations", "position_probability_maps"),
     ("position_specific_conjunctions",)),
    ("C", "Position-specific letter conjunctions", "Construct flexible ordered letter pairs",
     "Encode all forward pairs, including non-adjacent letters, while discounting longer distances.",
     ("open-bigram enumeration", "pairwise multiplication", "distance decay exp(-lambda*d)",
      "pair activation aggregation"),
     "Open-bigram and overlapping-position codes are computational theories for flexible letter order, not a literal confirmed cortical mechanism.",
     ("letter_identity_populations",), ("open_bigram_populations",)),
    ("D", "Parallel and recurrent processing", "Activate multiple letter detectors in parallel",
     "Update candidate letters across all positions in one vectorized population operation.",
     ("matrix construction", "vectorized affine activation", "parallel softmax",
      "population-wise normalization"),
     "Neural populations coding different positions can show temporally overlapping activity; vectorization models concurrent computation, not hardware parallelism in the brain.",
     ("letter_identity_evidence",), ("parallel_letter_activity",)),
    ("D", "Parallel and recurrent processing", "Process letters and conjunctions concurrently",
     "Maintain individual-letter and letter-combination activity during overlapping processing intervals.",
     ("matrix-vector multiplication", "letter-to-conjunction projection",
      "weighted integration", "joint activation normalization"),
     "Distinct letter and conjunction-sensitive populations can be active over overlapping time windows.",
     ("parallel_letter_activity", "position_specific_conjunctions", "open_bigram_populations"),
     ("joint_letter_conjunction_activity",)),
    ("D", "Parallel and recurrent processing", "Resolve activity through recurrence and lateral competition",
     "Refine ambiguous letter hypotheses using neighbor feedback, recurrent excitation and competition.",
     ("recurrent matrix updates", "neighbor message passing", "lateral inhibition",
      "damped iteration", "softmax normalization"),
     "Recurrent excitatory and inhibitory interactions are broad cortical motifs; this simplified update is not a fitted microcircuit.",
     ("joint_letter_conjunction_activity",), ("settled_letter_populations", "recurrent_deltas")),
    ("E", "Transposed-letter effects and orthographic integration", "Compare flexible letter order",
     "Measure how much ordered-pair information remains after transposition versus letter replacement.",
     ("set intersection and union", "Jaccard overlap |A∩B|/|A∪B|",
      "identity multiset overlap", "ordered edit similarity"),
     "Transposed-letter effects indicate flexible position coding, while preserved order information remains necessary to distinguish words.",
     ("orthographic_sequence", "comparison_strings"), ("orthographic_similarity",)),
    ("E", "Transposed-letter effects and orthographic integration", "Integrate orthographic evidence channels",
     "Combine letter identity, approximate position, position-specific conjunction and open-bigram information.",
     ("channel-wise L2 normalization", "weighted feature fusion",
      "population-vector concatenation", "vector normalization"),
     "A distributed ventral visual word-form representation can integrate multiple levels of orthographic information; this fused vector is a proposed model representation.",
     ("identity_channel", "position_channel", "conjunction_channel", "bigram_channel"),
     ("integrated_orthographic_vector",)),
    ("E", "Transposed-letter effects and orthographic integration", "Activate candidate word representations",
     "Rank only caller-supplied lexical candidates using orthographic evidence and an optional frequency prior.",
     ("letter-count similarity", "normalized edit distance", "conjunction and bigram Jaccard",
      "weighted score", "softmax competition"),
     "Stored orthographic word representations can compete within ventral occipitotemporal pathways; no missing lexical item is fabricated.",
     ("integrated_orthographic_vector", "supplied_lexicon"), ("ranked_word_candidates",)),
    ("E", "Transposed-letter effects and orthographic integration", "Stabilize the orthographic interpretation",
     "Track recurrent candidate settling and report convergence or confidence without treating a threshold as a literal neural switch.",
     ("damped posterior accumulation", "Euclidean state-delta norm",
      "confidence estimation", "convergence and threshold checks"),
     "Neural population states evolve over time; stability criteria summarize computational settling and are not literal brain decision thresholds.",
     ("ranked_word_candidates", "settled_letter_populations"),
     ("stabilized_interpretation", "settling_history")),
)

OPERATION_REGISTRY = tuple(
    OperationSpec(
        operation_id=f"OVO-{index:03d}",
        stage=row[0],
        stage_name=row[1],
        operation_name=row[2],
        objective=row[3],
        mathematical_operations=tuple(row[4]),
        biological_reference=row[5],
        inputs=tuple(row[6]),
        outputs=tuple(row[7]),
    )
    for index, row in enumerate(_OPERATION_ROWS, start=1)
)

if len(OPERATION_REGISTRY) != 16:
    raise RuntimeError("Visual orthography registry must contain exactly 16 operations")
if tuple(item.operation_id for item in OPERATION_REGISTRY) != tuple(
    f"OVO-{index:03d}" for index in range(1, 17)
):
    raise RuntimeError("Operation identifiers must be ordered OVO-001 through OVO-016")


def registry_as_dicts() -> dict[str, list[dict[str, Any]]]:
    """Return all five stages and sixteen ordered operation specifications."""
    return {"operations": [item.to_dict() for item in OPERATION_REGISTRY]}
