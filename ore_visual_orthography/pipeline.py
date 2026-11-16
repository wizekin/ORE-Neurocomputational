"""Integrated implementation of the visual acquisition and orthographic stages A-E."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi

from ore_brain_networks.contracts import BNBEAConfig
from ore_brain_networks.grapheme_populations import segment_candidate_regions
from ore_brain_networks.visual_gabor import (
    GaborVisualProcessor,
    extract_region_feature_vector,
    to_grayscale,
)

from .annotations import register_visual_orthography_annotations
from .biology import BIOLOGICAL_POPULATIONS, BiologicalReferenceSimulator
from .contracts import (
    LetterPrototype,
    VisualOrthographyConfig,
    VisualOrthographyRequest,
    VisualOrthographyResult,
)
from .math_ops import (
    compare_orthographic_strings,
    gaussian_position_distributions,
    integrate_representation_channels,
    jaccard_similarity,
    matrix_from_evidence,
    multiset_similarity,
    normalized_edit_similarity,
    normalized_positions,
    open_bigram_activations,
    position_specific_conjunctions,
    recurrent_letter_update,
    softmax,
)
from .registry import OPERATION_REGISTRY


_STAGE_NAMES = {
    "A": "Visual acquisition and early feature processing",
    "B": "Recognizing letters across fonts and visual variations",
    "C": "Position-specific letter conjunctions",
    "D": "Parallel and recurrent processing",
    "E": "Transposed-letter effects and orthographic integration",
}

_BIOLOGICAL_DRIVES = {
    1: ("retina_lgn",),
    2: ("v1",),
    3: ("v2_v4",),
    4: ("vwfa_posterior", "letter_identity"),
    5: ("letter_identity",),
    6: ("letter_position",),
    7: ("letter_position",),
    8: ("letter_conjunction",),
    9: ("open_bigram",),
    10: ("letter_identity", "letter_position"),
    11: ("letter_identity", "letter_conjunction", "open_bigram"),
    12: ("letter_identity", "letter_conjunction", "open_bigram"),
    13: ("orthography",),
    14: ("orthography", "letter_conjunction", "open_bigram"),
    15: ("lexical",),
    16: ("orthography", "lexical"),
}


class VisualOrthographyProcessor:
    """Run the complete sixteen-step functional algorithm and a separate LIF neural reference.

    Image-to-letter decoding is deliberately conditional on a supplied, labelled prototype bank.
    The processor never invents OCR labels. A caller may instead provide a known orthographic
    sequence, which is useful for testing the position, conjunction, recurrence and word stages
    independently of image segmentation.
    """

    def __init__(
        self,
        config: VisualOrthographyConfig | None = None,
        *,
        register_annotations: bool = True,
        repository_root: str | Path | None = None,
        database_path: str | Path | None = None,
    ) -> None:
        """Initialize feature extraction, the neural reference and optional ORE SQLite registration."""
        self.config = config or VisualOrthographyConfig()
        self.annotation_report = (
            register_visual_orthography_annotations(repository_root, database_path)
            if register_annotations
            else {"status": "annotation_registration_disabled"}
        )
        gabor_config = BNBEAConfig(
            gabor_size=self.config.gabor_size,
            orientations=self.config.orientations,
            spatial_frequencies=self.config.spatial_frequencies,
            gabor_sigma=self.config.gabor_sigma,
            gabor_gamma=self.config.gabor_gamma,
            visual_threshold_quantile=self.config.visual_threshold_quantile,
            max_candidate_regions=self.config.max_candidate_regions,
            simulate_biology=False,
        )
        self.visual_processor = GaborVisualProcessor(gabor_config)
        self.neural_simulator = (
            BiologicalReferenceSimulator(self.config) if self.config.simulate_biology else None
        )

    def make_letter_prototype(
        self,
        label: str,
        glyph_image: Any,
        *,
        form_id: str = "",
    ) -> LetterPrototype:
        """Build a labelled prototype vector from one cropped, single-glyph training image.

        The caller is responsible for supplying a crop containing one glyph. This helper does not
        infer its label; it computes Gabor energy and pools the thresholded glyph region.
        """
        image = to_grayscale(glyph_image)
        feature_stack = self.visual_processor.process(image)["feature_stack"]
        energy = np.max(feature_stack, axis=0)
        threshold = float(np.quantile(energy, self.config.visual_threshold_quantile))
        mask = energy > threshold
        if not np.any(mask):
            mask = np.ones_like(energy, dtype=bool)
        vector = extract_region_feature_vector(feature_stack, mask)
        return LetterPrototype(
            label=label,
            mean=vector,
            variance=np.ones_like(vector),
            form_id=form_id,
        )

    def update_letter_prototype(
        self,
        prototype: LetterPrototype,
        observed_vector: Sequence[float],
        learning_rate: float | None = None,
    ) -> LetterPrototype:
        """Apply an exponential-moving-average identity update to a labelled visual prototype."""
        vector = np.asarray(observed_vector, dtype=np.float64)
        mean = np.asarray(prototype.mean, dtype=np.float64)
        variance = np.asarray(prototype.variance, dtype=np.float64)
        if vector.shape != mean.shape or not np.all(np.isfinite(vector)):
            raise ValueError("observed_vector must be finite and match prototype dimensions")
        rate = (
            self.config.prototype_learning_rate
            if learning_rate is None else float(learning_rate)
        )
        if not 0.0 <= rate <= 1.0:
            raise ValueError("learning_rate must lie in [0, 1]")
        updated_mean = (1.0 - rate) * mean + rate * vector
        residual = (vector - updated_mean) ** 2
        updated_variance = (
            (1.0 - rate) * variance + rate * np.maximum(residual, 1e-4)
        )
        return LetterPrototype(
            label=prototype.label,
            mean=updated_mean,
            variance=updated_variance,
            form_id=prototype.form_id,
        )

    def _decode_regions(
        self,
        regions: Sequence[Mapping[str, Any]],
        prototypes: Sequence[LetterPrototype],
    ) -> tuple[list[dict[str, float]], list[str], list[float], list[dict[str, Any]]]:
        """Map visual region feature vectors to labelled letter populations when prototypes exist."""
        if not prototypes:
            return [], [], [], []
        decoded_evidence: list[dict[str, float]] = []
        decoded_labels: list[str] = []
        centers: list[float] = []
        details: list[dict[str, Any]] = []
        for region in regions:
            vector = np.asarray(region["feature_vector"], dtype=np.float64)
            prototype_scores: list[float] = []
            prototype_labels: list[str] = []
            label_scores: dict[str, float] = defaultdict(float)
            for prototype in prototypes:
                mean = np.asarray(prototype.mean, dtype=np.float64)
                variance = np.asarray(prototype.variance, dtype=np.float64)
                if mean.shape != vector.shape:
                    raise ValueError(
                        f"prototype {prototype.label!r} has {mean.size} features; "
                        f"visual region extractor produced {vector.size}"
                    )
                delta = vector - mean
                distance_squared = float(np.sum((delta**2) / variance))
                prototype_scores.append(-0.5 * distance_squared)
                prototype_labels.append(prototype.label)
            probabilities = softmax(
                np.asarray(prototype_scores, dtype=np.float64),
                temperature=self.config.prototype_temperature,
            )
            for label, probability in zip(prototype_labels, probabilities):
                label_scores[label] += float(probability)
            total = sum(label_scores.values())
            evidence = {
                label: score / max(total, 1e-12)
                for label, score in label_scores.items()
            }
            ranked = sorted(evidence.items(), key=lambda item: item[1], reverse=True)
            if not ranked:
                continue
            decoded_evidence.append(evidence)
            decoded_labels.append(ranked[0][0])
            bbox = region["bbox"]
            centers.append(float((bbox[0] + bbox[2] - 1) / 2.0))
            details.append({
                "region_id": region["region_id"],
                "bbox": tuple(int(value) for value in bbox),
                "candidate_probabilities": dict(evidence),
                "best_label": ranked[0][0],
                "confidence": float(ranked[0][1]),
                "feature_dimensions": int(vector.size),
            })
        return decoded_evidence, decoded_labels, centers, details

    def _update_labeled_variants(
        self,
        prototypes: Sequence[LetterPrototype],
        labeled_variants: Mapping[str, Sequence[Sequence[float]]],
    ) -> tuple[list[LetterPrototype], dict[str, Any]]:
        """Update only explicitly labelled prototypes and report within-identity feature dispersion."""
        updated = list(prototypes)
        index_by_label: dict[str, list[int]] = defaultdict(list)
        for index, prototype in enumerate(updated):
            index_by_label[prototype.label].append(index)
        update_counts: dict[str, int] = {}
        for label, vectors in labeled_variants.items():
            indices = index_by_label.get(label, [])
            if not indices:
                continue
            for vector in vectors:
                target_index = indices[0]
                updated[target_index] = self.update_letter_prototype(
                    updated[target_index], vector
                )
                update_counts[label] = update_counts.get(label, 0) + 1
        dispersion: dict[str, float] = {}
        for label, indices in index_by_label.items():
            means = np.stack([
                np.asarray(updated[index].mean, dtype=np.float64) for index in indices
            ])
            center = np.mean(means, axis=0)
            center_norm = float(np.linalg.norm(center))
            if center_norm <= 1e-12:
                dispersion[label] = 0.0
                continue
            normalized_center = center / center_norm
            similarities = []
            for vector in means:
                norm = float(np.linalg.norm(vector))
                similarities.append(
                    float(np.dot(vector / max(norm, 1e-12), normalized_center))
                )
            dispersion[label] = float(np.mean(similarities))
        return updated, {
            "prototype_count": len(updated),
            "labeled_update_counts": update_counts,
            "mean_within_identity_cosine_similarity": dispersion,
            "learning_rule": "m'=(1-eta)*m+eta*x; variance tracks residual dispersion",
            "status": "updated_from_explicit_labels" if update_counts else "no_labeled_updates_supplied",
        }

    def _lexical_candidates(
        self,
        observed: str,
        identity_evidence: Sequence[Mapping[str, float]],
        lexicon: Sequence[Any],
    ) -> list[dict[str, Any]]:
        """Score and normalize only supplied words using identity, order, conjunction and frequency."""
        if not observed or not lexicon:
            return []
        observed_evidence = [
            {character: 1.0} for character in observed
        ]
        observed_bigrams = set(
            open_bigram_activations(observed_evidence, self.config.bigram_distance_decay)[
                "pair_activations"
            ]
        )
        observed_conjunctions = set(
            position_specific_conjunctions(
                observed_evidence,
                normalized_positions(range(len(observed))),
                self.config.conjunction_distance_sigma,
            )["pair_activations"]
        )
        log_frequencies = np.asarray(
            [np.log1p(float(candidate.frequency)) for candidate in lexicon],
            dtype=np.float64,
        )
        max_frequency = max(float(np.max(log_frequencies)), 1e-12)
        scores: list[float] = []
        rows: list[dict[str, Any]] = []
        weights = {
            "identity": self.config.identity_weight,
            "position": self.config.position_weight,
            "conjunction": self.config.conjunction_weight,
            "bigram": self.config.bigram_weight,
            "frequency": self.config.frequency_weight,
        }
        for candidate, log_frequency in zip(lexicon, log_frequencies):
            word = candidate.word
            candidate_evidence = [{character: 1.0} for character in word]
            candidate_bigrams = set(
                open_bigram_activations(
                    candidate_evidence, self.config.bigram_distance_decay
                )["pair_activations"]
            )
            candidate_conjunctions = set(
                position_specific_conjunctions(
                    candidate_evidence,
                    normalized_positions(range(len(word))),
                    self.config.conjunction_distance_sigma,
                )["pair_activations"]
            )
            identity_score = multiset_similarity(observed, word)
            position_score = normalized_edit_similarity(observed, word)
            conjunction_score = jaccard_similarity(observed_conjunctions, candidate_conjunctions)
            bigram_score = jaccard_similarity(observed_bigrams, candidate_bigrams)
            frequency_score = float(log_frequency / max_frequency)
            component_scores = {
                "identity": identity_score,
                "position": position_score,
                "conjunction": conjunction_score,
                "bigram": bigram_score,
                "frequency": frequency_score,
            }
            score = sum(weights[name] * value for name, value in component_scores.items())
            scores.append(float(score))
            rows.append({
                "word": word,
                "source_reference": candidate.source_reference,
                "component_scores": component_scores,
                "score": float(score),
            })
        probabilities = softmax(
            np.asarray(scores, dtype=np.float64),
            temperature=self.config.lexical_temperature,
        )
        for row, probability in zip(rows, probabilities):
            row["probability"] = float(probability)
        rows.sort(key=lambda item: item["score"], reverse=True)
        return rows

    def _settle_candidates(
        self,
        ranked_candidates: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Iteratively accumulate a fixed candidate posterior and measure state convergence."""
        if not ranked_candidates:
            return {
                "status": "awaiting_supplied_lexicon_or_orthographic_evidence",
                "candidate_probabilities": [],
                "settling_deltas": [],
                "converged": False,
                "confidence_threshold_reached": False,
                "confidence": 0.0,
            }
        words = [str(item["word"]) for item in ranked_candidates]
        target = np.asarray(
            [float(item["probability"]) for item in ranked_candidates],
            dtype=np.float64,
        )
        state = np.full(target.shape, 1.0 / len(target), dtype=np.float64)
        deltas: list[float] = []
        converged = False
        threshold_reached = False
        for _ in range(self.config.max_settling_steps):
            updated = (
                (1.0 - self.config.settling_damping) * state
                + self.config.settling_damping * target
            )
            delta = float(np.linalg.norm(updated - state))
            deltas.append(delta)
            state = updated / max(float(np.sum(updated)), 1e-12)
            if delta < self.config.stability_epsilon:
                converged = True
                break
            if float(np.max(state)) >= self.config.confidence_threshold:
                threshold_reached = True
                break
        ranked = sorted(
            [
                {"word": word, "probability": float(probability)}
                for word, probability in zip(words, state)
            ],
            key=lambda item: item["probability"],
            reverse=True,
        )
        confidence = float(ranked[0]["probability"])
        return {
            "status": "converged" if converged else (
                "confidence_threshold_reached" if threshold_reached else "iteration_limit_reached"
            ),
            "candidate_probabilities": ranked,
            "settling_deltas": deltas,
            "converged": converged,
            "confidence_threshold_reached": threshold_reached,
            "confidence": confidence,
            "best_candidate": ranked[0]["word"],
            "confidence_sufficient": confidence >= self.config.confidence_threshold,
        }

    def _advance_biology(
        self,
        operation_index: int,
        strength: float,
    ) -> dict[str, Any] | None:
        """Advance the separate LIF populations with normalized evidence for one algorithm step."""
        if self.neural_simulator is None:
            return None
        drive = float(np.clip(strength, 0.0, 1.0))
        populations = _BIOLOGICAL_DRIVES[operation_index]
        snapshot = self.neural_simulator.advance(
            {name: drive for name in populations},
            steps=self.config.lif_steps_per_operation,
        )
        return {
            "operation_id": f"OVO-{operation_index:03d}",
            "driven_populations": list(populations),
            "normalized_drive": drive,
            "time_ms": snapshot["time_ms"],
            "population_rates_hz": {
                name: snapshot["populations"][name]["firing_rate_hz"]
                for name in BIOLOGICAL_POPULATIONS
            },
        }

    def _build_trace(
        self,
        statuses: Mapping[int, str],
        outputs: Mapping[int, str],
        biology_steps: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build the fixed sixteen-row trace with the functional and neural evidence kept distinct."""
        biological_by_id = {
            item["operation_id"]: item for item in biology_steps
        }
        trace: list[dict[str, Any]] = []
        for index, spec in enumerate(OPERATION_REGISTRY, start=1):
            biological = biological_by_id.get(spec.operation_id)
            trace.append({
                "operation_id": spec.operation_id,
                "stage": spec.stage,
                "stage_name": spec.stage_name,
                "step": index,
                "operation_name": spec.operation_name,
                "objective": spec.objective,
                "functional_status": statuses.get(index, "not_evaluated"),
                "functional_output": outputs.get(index, "not_available"),
                "mathematical_operations": list(spec.mathematical_operations),
                "biological_reference": spec.biological_reference,
                "biological_status": (
                    "simulated" if biological is not None else (
                        "disabled_by_config" if self.neural_simulator is None
                        else "not_advanced"
                    )
                ),
                "biological_output": biological,
            })
        return trace

    def process(self, request: VisualOrthographyRequest) -> VisualOrthographyResult:
        """Execute stages A-E in order and return sixteen traceable functional and biological steps."""
        if not isinstance(request, VisualOrthographyRequest):
            raise TypeError("request must be a VisualOrthographyRequest")

        statuses: dict[int, str] = {}
        outputs: dict[int, str] = {}
        biology_steps: list[dict[str, Any]] = []
        functional: dict[str, Any] = {
            "stage_a": {},
            "stage_b": {},
            "stage_c": {},
            "stage_d": {},
            "stage_e": {},
        }
        visual_output: dict[str, Any] | None = None
        feature_stack: np.ndarray | None = None
        region_details: list[dict[str, Any]] = []
        raw_regions: list[dict[str, Any]] = []
        region_centers: list[float] = []

        # Stage A / Step 1: sample the input and encode local luminance contrast.
        if request.visual_image is not None:
            image = np.asarray(request.visual_image, dtype=np.float64)
            if image.ndim == 3 and image.shape[2] == 1:
                image = image[..., 0]
            grayscale = to_grayscale(image)
            local_mean = ndi.gaussian_filter(grayscale, sigma=2.0)
            contrast = grayscale - local_mean
            functional["stage_a"]["spatial_sampling"] = {
                "height": int(grayscale.shape[0]),
                "width": int(grayscale.shape[1]),
                "contrast_map": contrast,
                "normalized_luminance": (grayscale - np.mean(grayscale)) / (
                    np.std(grayscale) + 1e-8
                ),
                "mathematics": [
                    "I(x,y) -> I[i,j]",
                    "C(x,y)=I(x,y)-I_local(x,y)",
                    "(I-mu_local)/(sigma_local+epsilon)",
                ],
                "biological_analogue": "Retinal photoreceptors, center-surround contrast pathways and LGN transmission.",
            }
            statuses[1] = "completed"
            outputs[1] = "spatial image array and local contrast representation computed"
            biology_steps.append(self._advance_biology(1, float(np.mean(np.abs(contrast)) * 3.0)))
        else:
            statuses[1] = "bypassed_no_visual_image"
            outputs[1] = "no image supplied; processing can continue from an explicit orthographic sequence"

        # Stage A / Step 2: generate orientation- and spatial-frequency-sensitive feature maps.
        if request.visual_image is not None:
            visual_output = self.visual_processor.process(image)
            feature_stack = np.asarray(visual_output["feature_stack"], dtype=np.float64)
            functional["stage_a"]["early_visual_features"] = visual_output
            statuses[2] = "completed"
            outputs[2] = "Gabor feature maps, quadrature energy and divisive normalization computed"
            biology_steps.append(self._advance_biology(2, float(np.mean(feature_stack) * 3.0)))
        else:
            statuses[2] = "bypassed_no_visual_image"
            outputs[2] = "no visual feature maps; explicit sequence may still drive later orthographic operations"

        # Stage A / Step 3: combine visual parts into candidate glyph-form feature vectors.
        if feature_stack is not None:
            raw_regions = segment_candidate_regions(
                feature_stack,
                self.config.visual_threshold_quantile,
                self.config.max_candidate_regions,
            )
            raw_regions.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
            region_centers = [
                float((item["bbox"][0] + item["bbox"][2] - 1) / 2.0)
                for item in raw_regions
            ]
            region_details = [
                {
                    "region_id": item["region_id"],
                    "bbox": tuple(int(value) for value in item["bbox"]),
                    "feature_dimensions": int(np.asarray(item["feature_vector"]).size),
                    "feature_vector": np.asarray(item["feature_vector"], dtype=np.float64),
                }
                for item in raw_regions
            ]
            functional["stage_a"]["candidate_visual_forms"] = {
                "candidate_count": len(region_details),
                "regions": region_details,
                "mathematics": [
                    "weighted feature integration",
                    "nonlinear population tuning",
                    "connected-region grouping and feature pooling",
                ],
                "biological_analogue": "Hierarchical visual-form integration across intermediate ventral visual populations.",
            }
            statuses[3] = "completed" if raw_regions else "no_candidate_regions_detected"
            outputs[3] = f"{len(raw_regions)} candidate visual region(s) represented as feature vectors"
            biology_steps.append(self._advance_biology(3, min(len(raw_regions) / 8.0, 1.0)))
        else:
            statuses[3] = "bypassed_no_visual_features"
            outputs[3] = "candidate visual-form feature extraction unavailable without an image"

        # Stage B / Step 4: associate observed visual forms with shared abstract letter identities.
        if raw_regions and request.prototypes:
            evidence, decoded_labels, decoded_centers, decoded = self._decode_regions(
                raw_regions, request.prototypes
            )
            region_centers = decoded_centers
            sequence = tuple(decoded_labels)
            recognition_source = "labelled_visual_prototypes"
            confidence = float(np.mean([
                max(item.values()) for item in evidence
            ])) if evidence else 0.0
            statuses[4] = "completed"
        elif request.orthographic_sequence is not None:
            sequence = tuple(request.orthographic_sequence)
            evidence = [{label: 1.0} for label in sequence]
            decoded = []
            recognition_source = "explicit_orthographic_sequence"
            confidence = 1.0
            statuses[4] = "completed_from_explicit_sequence"
        else:
            sequence = ()
            evidence = []
            decoded = []
            recognition_source = "no_labelled_identity_source"
            confidence = 0.0
            statuses[4] = "awaiting_labelled_prototypes_or_explicit_sequence"
        functional["stage_b"]["letter_identity"] = {
            "sequence": list(sequence),
            "position_evidence": evidence,
            "decoded_regions": decoded,
            "recognition_source": recognition_source,
            "mean_identity_confidence": confidence,
            "mathematics": [
                "diagonal Mahalanobis distance",
                "RBF-like population tuning",
                "softmax identity competition",
            ],
            "biological_analogue": "Progressively abstract ventral occipitotemporal and VWFA-sensitive letter representations.",
        }
        outputs[4] = (
            f"{len(sequence)} letter identity position(s) represented from {recognition_source}"
        )
        biology_steps.append(self._advance_biology(4, confidence))

        # Stage B / Step 5: learn only from explicitly labelled form variants.
        updated_prototypes, invariance_report = self._update_labeled_variants(
            request.prototypes, request.labeled_variants
        )
        functional["stage_b"]["visual_invariance_learning"] = {
            "updated_prototypes": updated_prototypes,
            "invariance_report": invariance_report,
            "mathematics": [
                "EMA prototype update m'=(1-eta)m+eta*x",
                "within-identity cosine similarity",
                "residual variance tracking",
            ],
            "biological_analogue": "Experience-dependent plasticity linking different visual forms to shared letter identity.",
        }
        statuses[5] = (
            "updated_from_labeled_variants"
            if invariance_report["labeled_update_counts"]
            else "no_labeled_variants_supplied"
        )
        outputs[5] = invariance_report["status"]
        biology_steps.append(self._advance_biology(5, float(bool(invariance_report["labeled_update_counts"]))))

        # If recognition was explicitly requested but no evidence was decoded, preserve the supplied sequence.
        if not sequence and request.orthographic_sequence is not None:
            sequence = tuple(request.orthographic_sequence)
            evidence = [{label: 1.0} for label in sequence]
        observed = "".join(sequence)
        if evidence:
            if len(region_centers) != len(evidence):
                region_centers = list(np.linspace(0.0, 1.0, len(evidence)))
            positions = normalized_positions(region_centers)
        else:
            positions = np.zeros(0, dtype=np.float64)

        # Stage C / Step 6: preserve approximate letter locations.
        functional["stage_c"]["relative_positions"] = {
            "positions": positions,
            "source_centers": region_centers,
            "coordinate_system": "normalized horizontal coordinate in [0,1]",
            "mathematics": "r_i=(x_i-x_min)/(x_max-x_min)",
            "biological_analogue": "Retinotopic and position-sensitive visual/orthographic population activity.",
        }
        statuses[6] = "completed" if evidence else "awaiting_letter_identity_evidence"
        outputs[6] = f"{len(positions)} approximate relative letter position(s)"
        biology_steps.append(self._advance_biology(6, float(np.mean(positions)) if positions.size else 0.0))

        # Stage C / Step 7: express position as a probability-like population over a spatial grid.
        uncertainty = request.position_uncertainty or self.config.position_sigma
        position_distributions = gaussian_position_distributions(
            positions, uncertainty, self.config.position_grid_size
        )
        functional["stage_c"]["position_uncertainty"] = {
            "grid": np.linspace(0.0, 1.0, self.config.position_grid_size),
            "distributions": position_distributions,
            "uncertainty_sigma": uncertainty,
            "row_sums": np.sum(position_distributions, axis=1),
            "mathematics": "P_i(r) proportional to exp(-(r-mu_i)^2/(2*sigma_i^2))",
            "biological_analogue": "Population coding of uncertain spatial position; Gaussian tuning is a computational model, not an explicit neural probability table.",
        }
        statuses[7] = "completed" if evidence else "awaiting_letter_identity_evidence"
        outputs[7] = f"{position_distributions.shape[0]} normalized position distributions"
        biology_steps.append(self._advance_biology(7, float(np.mean(position_distributions)) if position_distributions.size else 0.0))

        # Stage C / Step 8: combine identities and position to activate position-specific conjunctions.
        conjunctions = position_specific_conjunctions(
            evidence, positions, self.config.conjunction_distance_sigma
        )
        functional["stage_c"]["position_specific_conjunctions"] = conjunctions
        statuses[8] = "completed" if evidence else "awaiting_letter_identity_evidence"
        outputs[8] = f"{len(conjunctions['pairs'])} ordered, position-weighted conjunction instances"
        conjunction_strength = (
            float(np.mean(list(conjunctions["pair_activations"].values())))
            if conjunctions["pair_activations"] else 0.0
        )
        biology_steps.append(self._advance_biology(8, conjunction_strength))

        # Stage C / Step 9: generate open bigrams that preserve forward order with distance decay.
        bigrams = open_bigram_activations(evidence, self.config.bigram_distance_decay)
        functional["stage_c"]["open_bigrams"] = bigrams
        statuses[9] = "completed" if evidence else "awaiting_letter_identity_evidence"
        outputs[9] = f"{len(bigrams['pairs'])} forward ordered-pair instances"
        bigram_strength = (
            float(np.mean(list(bigrams["pair_activations"].values())))
            if bigrams["pair_activations"] else 0.0
        )
        biology_steps.append(self._advance_biology(9, bigram_strength))

        # Stage D / Step 10: update the letter populations across positions as one vectorized operation.
        letter_matrix, alphabet = matrix_from_evidence(evidence)
        functional["stage_d"]["parallel_letter_populations"] = {
            "activation_matrix": letter_matrix,
            "alphabet": alphabet,
            "position_count": int(letter_matrix.shape[0]),
            "candidate_count": int(letter_matrix.shape[1]),
            "mathematics": "L_i(t+1)=softmax(W_i*F+b_i); all positions updated in one population step",
            "biological_analogue": "Temporally overlapping activity in distinguishable letter-position populations.",
        }
        statuses[10] = "completed" if evidence else "awaiting_letter_identity_evidence"
        outputs[10] = f"vectorized activation matrix shape {letter_matrix.shape}"
        biology_steps.append(self._advance_biology(10, float(np.mean(letter_matrix)) if letter_matrix.size else 0.0))

        # Stage D / Step 11: maintain letter and conjunction activations concurrently.
        conjunction_keys = sorted(conjunctions["pair_activations"])
        bigram_keys = sorted(bigrams["pair_activations"])
        joint_vector = np.concatenate((
            letter_matrix.ravel(),
            np.asarray([conjunctions["pair_activations"][key] for key in conjunction_keys]),
            np.asarray([bigrams["pair_activations"][key] for key in bigram_keys]),
        ))
        functional["stage_d"]["joint_letter_conjunction_activity"] = {
            "letter_matrix": letter_matrix,
            "conjunction_labels": conjunction_keys,
            "conjunction_activity": np.asarray([
                conjunctions["pair_activations"][key] for key in conjunction_keys
            ]),
            "bigram_labels": bigram_keys,
            "bigram_activity": np.asarray([
                bigrams["pair_activations"][key] for key in bigram_keys
            ]),
            "joint_population_vector": joint_vector,
            "mathematics": "B(t+1)=f(W_LB*L(t)+W_BB*B(t)); simultaneous population integration",
            "biological_analogue": "Overlapping individual-letter and letter-conjunction population activity.",
        }
        statuses[11] = "completed" if evidence else "awaiting_letter_identity_evidence"
        outputs[11] = f"joint letter/conjunction vector with {joint_vector.size} component(s)"
        biology_steps.append(self._advance_biology(11, float(np.mean(joint_vector)) if joint_vector.size else 0.0))

        # Stage D / Step 12: allow recurrent neighbor messages and lateral competition to settle.
        settled_matrix, recurrent_deltas = recurrent_letter_update(
            letter_matrix,
            iterations=self.config.recurrent_steps,
            damping=self.config.recurrent_damping,
            lateral_inhibition=self.config.lateral_inhibition,
        )
        settled_evidence: list[dict[str, float]] = []
        for row in settled_matrix:
            settled_evidence.append({
                label: float(row[index])
                for index, label in enumerate(alphabet)
            })
        functional["stage_d"]["recurrent_competition"] = {
            "settled_activation_matrix": settled_matrix,
            "activation_change_norms": recurrent_deltas,
            "iterations": len(recurrent_deltas),
            "final_delta": recurrent_deltas[-1] if recurrent_deltas else 0.0,
            "mathematics": "a(t+1)=f(W_bottom*x+W_recurrent*a+W_feedback*h-W_inhibition*a)",
            "biological_analogue": "Recurrent excitatory and inhibitory integration among visual/orthographic populations.",
        }
        statuses[12] = "completed" if evidence else "awaiting_letter_identity_evidence"
        outputs[12] = f"recurrent state updated for {len(recurrent_deltas)} iteration(s)"
        biology_steps.append(self._advance_biology(12, float(np.mean(settled_matrix)) if settled_matrix.size else 0.0))

        # Stage E / Step 13: explicitly compare requested alternatives such as JUGDE and JUPTE.
        comparisons = {
            candidate: compare_orthographic_strings(observed, candidate)
            for candidate in request.comparison_strings
        }
        functional["stage_e"]["transposition_similarity"] = {
            "observed_sequence": observed,
            "comparisons": comparisons,
            "mathematics": "Jaccard(open-bigram sets), multiset overlap and ordered edit similarity",
            "biological_analogue": "Flexible but order-sensitive orthographic position coding; transposition tolerance is not position indifference.",
        }
        statuses[13] = "completed" if observed else "awaiting_orthographic_sequence"
        outputs[13] = f"{len(comparisons)} comparison string(s) scored"
        biology_steps.append(self._advance_biology(
            13,
            float(np.mean([item["combined_similarity"] for item in comparisons.values()]))
            if comparisons else (1.0 if observed else 0.0),
        ))

        # Stage E / Step 14: normalize and combine complementary representation channels.
        identity_channel = settled_matrix.ravel()
        position_channel = position_distributions.ravel()
        conjunction_channel = np.asarray([
            conjunctions["pair_activations"][key] for key in conjunction_keys
        ], dtype=np.float64)
        bigram_channel = np.asarray([
            bigrams["pair_activations"][key] for key in bigram_keys
        ], dtype=np.float64)
        channel_weights = {
            "identity": self.config.identity_weight,
            "position": self.config.position_weight,
            "conjunction": self.config.conjunction_weight,
            "bigram": self.config.bigram_weight,
        }
        integrated, normalized_channels = integrate_representation_channels(
            {
                "identity": identity_channel,
                "position": position_channel,
                "conjunction": conjunction_channel,
                "bigram": bigram_channel,
            },
            channel_weights,
        )
        functional["stage_e"]["integrated_orthography"] = {
            "representation_vector": integrated,
            "normalized_channels": normalized_channels,
            "channel_weights": channel_weights,
            "channel_dimensions": {
                "identity": int(identity_channel.size),
                "position": int(position_channel.size),
                "conjunction": int(conjunction_channel.size),
                "bigram": int(bigram_channel.size),
            },
            "mathematics": "O=alpha*identity+beta*position+gamma*conjunction+delta*open_bigrams",
            "biological_analogue": "Proposed distributed orthographic population code integrating identity, position and pair information.",
        }
        statuses[14] = "completed" if integrated.size else "empty_no_orthographic_evidence"
        outputs[14] = f"integrated feature vector has {integrated.size} component(s)"
        biology_steps.append(self._advance_biology(14, float(np.linalg.norm(integrated))))

        # Stage E / Step 15: rank only known lexical candidates, never fabricating a dictionary entry.
        ranked_candidates = self._lexical_candidates(
            observed, settled_evidence, request.lexicon
        )
        functional["stage_e"]["lexical_candidates"] = {
            "candidates": ranked_candidates,
            "candidate_source": "caller_supplied_lexicon_only",
            "mathematics": "weighted identity/order/conjunction/bigram/frequency score followed by softmax",
            "biological_analogue": "Competitive activation of stored word-form representations in the ventral visual reading network.",
        }
        statuses[15] = (
            "completed" if ranked_candidates else
            ("awaiting_supplied_lexicon" if observed else "awaiting_orthographic_sequence")
        )
        outputs[15] = f"{len(ranked_candidates)} supplied lexical candidate(s) ranked"
        biology_steps.append(self._advance_biology(
            15,
            float(ranked_candidates[0]["probability"]) if ranked_candidates else 0.0,
        ))

        # Stage E / Step 16: stabilize the final orthographic/lexical state and report confidence.
        settling = self._settle_candidates(ranked_candidates)
        stable_interpretation = {
            "orthographic_sequence": observed or None,
            "selected_word": settling.get("best_candidate"),
            "confidence": settling["confidence"],
            "confidence_sufficient": settling.get("confidence_sufficient", False),
            "settling": settling,
        }
        functional["stage_e"]["stabilization"] = stable_interpretation
        statuses[16] = (
            "completed" if settling["status"] in {"converged", "confidence_threshold_reached"}
            else settling["status"]
        )
        outputs[16] = (
            f"settling status {settling['status']} with confidence {settling['confidence']:.4f}"
        )
        biology_steps.append(self._advance_biology(
            16,
            float(settling["confidence"]),
        ))

        trace = self._build_trace(statuses, outputs, biology_steps)
        biological = {
            "enabled": self.neural_simulator is not None,
            "operation_advances": biology_steps,
            "lif": self.neural_simulator.snapshot() if self.neural_simulator else None,
            "population_names": list(BIOLOGICAL_POPULATIONS),
            "scope": (
                "Population-level current-based leaky-integrate-and-fire computational hypothesis. "
                "It is not an anatomically fitted or empirically validated simulation of human reading."
            ),
        }
        diagnostics = {
            "operation_count": len(trace),
            "registered_operation_count": len(OPERATION_REGISTRY),
            "operation_ids_complete": [item["operation_id"] for item in trace] == [
                item.operation_id for item in OPERATION_REGISTRY
            ],
            "stage_operation_counts": {
                stage: sum(item.stage == stage for item in OPERATION_REGISTRY)
                for stage in ("A", "B", "C", "D", "E")
            },
            "functional_steps_completed": sum(
                item["functional_status"] == "completed" for item in trace
            ),
            "steps_pending_or_bypassed": sum(
                item["functional_status"] != "completed" for item in trace
            ),
            "annotation_registration": self.annotation_report,
            "separate_functional_and_biological_tracks": True,
        }
        metadata = {
            "algorithm": "Visual Processing and Orthographic Representations",
            "package": "ore_visual_orthography",
            "version": "1.0.0",
            "sample_id": request.sample_id,
            "language": request.language,
            "configuration": self.config.to_dict(),
            "stage_sequence": _STAGE_NAMES,
            "scientific_limitations": [
                "A visual image is not automatically transcribed; use an explicit sequence or labelled visual prototypes.",
                "Connected-region thresholding is a compact research segmentation method, not production OCR.",
                "Gabor filters, prototype populations, Gaussian position distributions, open bigrams and recurrent equations are computational hypotheses.",
                "Open-bigram coding is one theoretical account of flexible position effects, not an established literal neural circuit.",
                "LIF population names and numerical parameters are not subject-specific physiological measurements.",
                "A passing software test does not establish biological validity; independent behavioral and neural data are required.",
            ],
        }
        return VisualOrthographyResult(
            sample_id=request.sample_id,
            functional=functional,
            biological=biological,
            operation_trace=trace,
            diagnostics=diagnostics,
            metadata=metadata,
        )
