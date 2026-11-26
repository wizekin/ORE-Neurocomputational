"""Executable seven-stage lexical identity, meaning, morphology and activation pipeline."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence
import math
import unicodedata

from .biology import POPULATIONS, LexicalAccessLIFSimulator
from .contracts import LexicalAccessConfig, LexicalAccessRequest, LexicalAccessResult
from .registry import OPERATION_REGISTRY


def _sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def _softmax(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    if not scores:
        return []
    temp = max(float(temperature), 1e-8)
    scaled = [max(-100.0, min(100.0, float(score) / temp)) for score in scores]
    peak = max(scaled)
    exponentials = [math.exp(score - peak) for score in scaled]
    total = sum(exponentials)
    return [value / total for value in exponentials] if total else [1.0 / len(scores)] * len(scores)


def _levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row_index, left_char in enumerate(left, start=1):
        current = [row_index]
        for column_index, right_char in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column_index] + 1,
                previous[column_index - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def _similarity(left: str, right: str) -> float:
    denominator = max(len(left), len(right), 1)
    return max(0.0, 1.0 - _levenshtein(left, right) / denominator)


def _cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    dot = sum(float(left.get(key, 0.0)) * float(right.get(key, 0.0)) for key in keys)
    left_norm = math.sqrt(sum(float(left.get(key, 0.0)) ** 2 for key in keys))
    right_norm = math.sqrt(sum(float(right.get(key, 0.0)) ** 2 for key in keys))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _bounded_features(features: Mapping[str, float]) -> dict[str, float]:
    return {str(key): max(0.0, min(1.0, float(value))) for key, value in features.items()}


def _combine_features(*feature_maps: Mapping[str, float]) -> dict[str, float]:
    combined: dict[str, float] = {}
    for feature_map in feature_maps:
        for feature, value in feature_map.items():
            combined[feature] = max(combined.get(feature, 0.0), float(value))
    return _bounded_features(combined)


def _max_frequency(values: Sequence[float]) -> float:
    # Keep logarithmic normalization defined even for an all-zero inventory.
    return max(1.0, max((float(value) for value in values), default=0.0))


class LexicalAccessProcessor:
    """Seven-stage processor over explicit lexical and morpheme inventories.

    `lexicon` and `morphemes` accept the existing `ore_lexical_context` records.
    Entries are consumed read-only; no exposure count or inventory is modified.
    """

    def __init__(
        self,
        lexicon: Sequence[Any],
        *,
        morphemes: Sequence[Any] = (),
        morpheme_category_cues: Mapping[str, Mapping[str, float]] | None = None,
        morpheme_affix_categories: Mapping[str, str] | None = None,
        config: LexicalAccessConfig | None = None,
    ) -> None:
        self.entries = tuple(lexicon)
        self.morphemes = tuple(morphemes)
        if not self.entries:
            raise ValueError("lexicon must contain at least one LexicalEntry")
        entry_ids = [str(entry.entry_id) for entry in self.entries]
        if len(set(entry_ids)) != len(entry_ids):
            raise ValueError("lexical entry_id values must be unique")
        morpheme_ids = [str(item.morpheme_id) for item in self.morphemes]
        if len(set(morpheme_ids)) != len(morpheme_ids):
            raise ValueError("morpheme_id values must be unique")
        self.config = config or LexicalAccessConfig()
        self.by_form: dict[str, list[Any]] = defaultdict(list)
        for entry in self.entries:
            normalized_form = unicodedata.normalize("NFC", str(entry.orthography)).casefold()
            self.by_form[normalized_form].append(entry)
        self.morpheme_by_id = {str(item.morpheme_id): item for item in self.morphemes}
        self.morpheme_category_cues = {
            str(morpheme_id): _bounded_features(cues)
            for morpheme_id, cues in (morpheme_category_cues or {}).items()
        }
        unknown_cues = set(self.morpheme_category_cues) - set(self.morpheme_by_id)
        if unknown_cues:
            raise ValueError(f"category cues reference unknown morphemes: {sorted(unknown_cues)}")
        self.morpheme_affix_categories = {
            str(morpheme_id): str(category).strip().casefold()
            for morpheme_id, category in (morpheme_affix_categories or {}).items()
        }
        unknown_affix_categories = set(self.morpheme_affix_categories) - set(self.morpheme_by_id)
        if unknown_affix_categories:
            raise ValueError(
                f"affix categories reference unknown morphemes: {sorted(unknown_affix_categories)}"
            )
        invalid_affix_categories = {
            morpheme_id: category for morpheme_id, category in self.morpheme_affix_categories.items()
            if category not in {"derivational", "inflectional"}
            or self.morpheme_by_id[morpheme_id].morpheme_type not in {"prefix", "suffix"}
        }
        if invalid_affix_categories:
            raise ValueError(
                "affix categories must classify known prefixes/suffixes as derivational or inflectional: "
                f"{invalid_affix_categories}"
            )

    def _entry_frequencies(self) -> list[float]:
        return [max(0.0, float(entry.frequency) + float(getattr(entry, "exposure_count", 0.0)))
                for entry in self.entries]

    def _identity_candidates(self, word: str) -> list[dict[str, Any]]:
        cfg = self.config
        frequency_max = _max_frequency(self._entry_frequencies())
        records = []
        for entry in self.entries:
            form = unicodedata.normalize("NFC", str(entry.orthography)).casefold()
            distance = _levenshtein(word, form)
            similarity = max(0.0, 1.0 - distance / max(len(word), len(form), 1))
            if distance > cfg.max_edit_distance or similarity < cfg.minimum_similarity:
                continue
            raw_frequency = max(0.0, float(entry.frequency) + float(getattr(entry, "exposure_count", 0.0)))
            frequency_strength = math.log1p(raw_frequency) / math.log1p(frequency_max)
            likelihood = math.exp(-2.0 * distance / max(len(word), len(form), 1))
            score = math.log(max(likelihood, 1e-12)) + 0.25 * frequency_strength
            records.append({
                "entry_id": str(entry.entry_id),
                "orthography": form,
                "lexical_category": str(getattr(entry, "lexical_category", "unknown") or "unknown"),
                "distance": distance,
                "orthographic_similarity": similarity,
                "orthographic_likelihood": likelihood,
                "frequency": float(entry.frequency),
                "exposure_count": float(getattr(entry, "exposure_count", 0.0)),
                "frequency_strength": frequency_strength,
                "language": str(getattr(entry, "language", "und")),
                "semantic_concept_ids": list(getattr(entry, "semantic_concept_ids", ())),
                "semantic_link_status": str(getattr(entry, "semantic_link_status", "UNLINKED")),
                "lemma_entry_id": getattr(entry, "lemma_entry_id", None),
                "morphology_link_status": str(getattr(entry, "morphology_link_status", "NOT_APPLICABLE")),
                "script": getattr(entry, "script", None),
                "grammatical_information_json": getattr(entry, "grammatical_information_json", None),
                "verified_by": str(getattr(entry, "verified_by", "unspecified")),
                "source_reference": str(getattr(entry, "source_reference", "unspecified")),
                "_score": score,
            })
        records.sort(key=lambda record: record["_score"], reverse=True)
        records = records[:cfg.max_candidates]
        weights = _softmax([record["_score"] for record in records], cfg.identity_temperature)
        for record, weight in zip(records, weights):
            record["posterior"] = weight
            record.pop("_score", None)
        return records

    def _meaning_candidates(
        self, identities: Sequence[Mapping[str, Any]], context: Mapping[str, float]
    ) -> list[dict[str, Any]]:
        cfg = self.config
        entry_by_id = {str(entry.entry_id): entry for entry in self.entries}
        records: list[dict[str, Any]] = []
        for identity in identities:
            entry = entry_by_id[str(identity["entry_id"])]
            for meaning in tuple(getattr(entry, "meanings", ())):
                features = _bounded_features(getattr(meaning, "features", {}))
                prior = max(float(getattr(meaning, "prior", 1.0)), 1e-12)
                context_fit = _cosine(features, context)
                records.append({
                    "entry_id": str(entry.entry_id),
                    "orthography": str(entry.orthography),
                    "meaning_id": str(meaning.meaning_id),
                    "semantic_concept_ids": list(getattr(meaning, "semantic_concept_ids", ())),
                    "semantic_link_status": str(getattr(meaning, "semantic_link_status", "UNLINKED")),
                    "gloss": str(getattr(meaning, "gloss", "")),
                    "features": features,
                    "prior": prior,
                    "lexical_identity_posterior": float(identity["posterior"]),
                    "context_compatibility": context_fit,
                    "verified_by": str(getattr(meaning, "verified_by", "unspecified")),
                    "source_reference": str(getattr(meaning, "source_reference", "unspecified")),
                    "_score": math.log(max(float(identity["posterior"]) * prior, 1e-12))
                              + cfg.context_semantic_gain * context_fit,
                })
        weights = _softmax([record["_score"] for record in records], cfg.semantic_temperature)
        for record, weight in zip(records, weights):
            record["posterior"] = weight
            record.pop("_score", None)
        records.sort(key=lambda record: record["posterior"], reverse=True)
        return records

    def _category_candidates(
        self, identities: Sequence[Mapping[str, Any]], contextual_priors: Mapping[str, float]
    ) -> list[dict[str, Any]]:
        cfg = self.config
        evidence: dict[str, float] = defaultdict(float)
        for identity in identities:
            category = str(identity.get("lexical_category", "unknown")).casefold()
            if category and category != "unknown":
                evidence[category] += float(identity["posterior"])
        normalized_context = {str(key).casefold(): float(value) for key, value in contextual_priors.items()}
        categories = sorted(set(evidence) | set(normalized_context))
        if not categories:
            return []
        scores = []
        for category in categories:
            lexical_support = evidence.get(category, 0.0)
            context_support = normalized_context.get(category, 0.0)
            score = math.log(lexical_support + 1e-9)
            score += 0.35 * context_support
            scores.append(score)
        weights = _softmax(scores, cfg.category_temperature)
        records = [
            {"category": category, "lexical_evidence": evidence.get(category, 0.0),
             "contextual_prior": normalized_context.get(category, 0.0),
             "posterior": weight}
            for category, weight in zip(categories, weights)
        ]
        records.sort(key=lambda record: record["posterior"], reverse=True)
        return records

    def _decompositions(self, word: str) -> list[dict[str, Any]]:
        prefixes = [item for item in self.morphemes if item.morpheme_type == "prefix"]
        suffixes = [item for item in self.morphemes if item.morpheme_type == "suffix"]
        roots = [item for item in self.morphemes if item.morpheme_type == "root"]
        records: list[dict[str, Any]] = []
        exact_entries = self.by_form.get(word, [])
        for entry in exact_entries:
            records.append({
                "hypothesis_id": f"whole:{entry.entry_id}", "kind": "whole_word",
                "prefix_id": None, "stem": word, "root_morpheme_id": None,
                "suffix_id": None, "stem_entry_ids": [str(entry.entry_id)],
                "morpheme_ids": [], "reconstruction": word,
                "structural_validity": 1.0, "_base_score": self.config.whole_word_bias,
            })
        for prefix in [None, *prefixes]:
            prefix_form = str(prefix.form).casefold() if prefix is not None else ""
            if prefix_form and not word.startswith(prefix_form):
                continue
            after_prefix = word[len(prefix_form):]
            for suffix in [None, *suffixes]:
                suffix_form = str(suffix.form).casefold() if suffix is not None else ""
                if suffix_form and not after_prefix.endswith(suffix_form):
                    continue
                stem = after_prefix[:-len(suffix_form)] if suffix_form else after_prefix
                if not stem or (prefix is None and suffix is None):
                    continue
                if prefix_form + stem + suffix_form != word:
                    continue
                stem_entries = self.by_form.get(stem, [])
                stem_roots = [root for root in roots if str(root.form).casefold() == stem]
                if not stem_entries and not stem_roots:
                    continue
                for root in [None, *stem_roots]:
                    if root is None and not stem_entries:
                        continue
                    prefix_id = str(prefix.morpheme_id) if prefix is not None else None
                    root_id = str(root.morpheme_id) if root is not None else None
                    suffix_id = str(suffix.morpheme_id) if suffix is not None else None
                    ids = [item for item in (prefix_id, root_id, suffix_id) if item]
                    root_marker = root_id or (stem_entries[0].entry_id if stem_entries else "")
                    records.append({
                        "hypothesis_id": f"morph:{prefix_id or '-'}:{root_marker}:{suffix_id or '-'}",
                        "kind": "segmented", "prefix_id": prefix_id, "stem": stem,
                        "root_morpheme_id": root_id, "suffix_id": suffix_id,
                        "stem_entry_ids": [str(entry.entry_id) for entry in stem_entries],
                        "morpheme_ids": ids, "reconstruction": prefix_form + stem + suffix_form,
                        "structural_validity": 1.0, "_base_score": 0.0,
                    })
        unique: dict[str, dict[str, Any]] = {}
        for record in records:
            unique[record["hypothesis_id"]] = record
        return list(unique.values())[: self.config.max_candidates]

    def _retrieve_components(self, decompositions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        entry_by_id = {str(entry.entry_id): entry for entry in self.entries}
        retrieved: list[dict[str, Any]] = []
        for hypothesis in decompositions:
            stem_entries = [entry_by_id[item] for item in hypothesis["stem_entry_ids"] if item in entry_by_id]
            stem_morpheme = self.morpheme_by_id.get(str(hypothesis.get("root_morpheme_id")))
            stem_meanings = [meaning for entry in stem_entries for meaning in tuple(getattr(entry, "meanings", ()))]
            stem_feature_maps = [getattr(meaning, "features", {}) for meaning in stem_meanings]
            if stem_morpheme is not None:
                stem_feature_maps.append(getattr(stem_morpheme, "semantic_features", {}))
            stem_features = _combine_features(*stem_feature_maps)
            components: list[dict[str, Any]] = []
            for morpheme_id in (hypothesis.get("prefix_id"), hypothesis.get("root_morpheme_id"), hypothesis.get("suffix_id")):
                if not morpheme_id:
                    continue
                record = self.morpheme_by_id[str(morpheme_id)]
                components.append({
                    "morpheme_id": str(record.morpheme_id), "form": str(record.form),
                    "morpheme_type": str(record.morpheme_type),
                    "affix_category": self.morpheme_affix_categories.get(
                        str(record.morpheme_id), "unspecified"
                    ) if record.morpheme_type in {"prefix", "suffix"} else "not_applicable",
                    "semantic_features": _bounded_features(getattr(record, "semantic_features", {})),
                    "frequency": float(record.frequency), "productivity": float(record.productivity),
                    "verified_by": str(getattr(record, "verified_by", "unspecified")),
                    "source_reference": str(getattr(record, "source_reference", "unspecified")),
                    "category_cues": dict(self.morpheme_category_cues.get(str(record.morpheme_id), {})),
                })
            prefix_record = self.morpheme_by_id.get(str(hypothesis.get("prefix_id"))) if hypothesis.get("prefix_id") else None
            suffix_record = self.morpheme_by_id.get(str(hypothesis.get("suffix_id"))) if hypothesis.get("suffix_id") else None
            affixes = [item for item in (prefix_record, suffix_record) if item is not None]
            component_by_id = {item["morpheme_id"]: item for item in components}
            derivational_affixes = [
                component_by_id[str(item.morpheme_id)] for item in affixes
                if self.morpheme_affix_categories.get(str(item.morpheme_id)) == "derivational"
            ]
            inflectional_affixes = [
                component_by_id[str(item.morpheme_id)] for item in affixes
                if self.morpheme_affix_categories.get(str(item.morpheme_id)) == "inflectional"
            ]
            unspecified_affixes = [
                component_by_id[str(item.morpheme_id)] for item in affixes
                if str(item.morpheme_id) not in self.morpheme_affix_categories
            ]
            composed = dict(stem_features)
            contributions = {"stem": dict(stem_features)}
            for affix in affixes:
                gain = 0.5 * float(affix.productivity)
                affix_features = getattr(affix, "semantic_features", {})
                contributions[str(affix.morpheme_id)] = {
                    key: gain * float(value) for key, value in affix_features.items()
                }
                for feature, value in affix_features.items():
                    composed[feature] = min(1.0, composed.get(feature, 0.0) + gain * float(value))
            category_cues: dict[str, float] = {}
            for entry in stem_entries:
                category = str(getattr(entry, "lexical_category", "unknown")).casefold()
                if category and category != "unknown":
                    category_cues[category] = max(category_cues.get(category, 0.0), 1.0)
            for component in components:
                for category, value in component["category_cues"].items():
                    category_cues[category] = max(category_cues.get(category, 0.0), float(value))
            retrieved.append({
                "hypothesis_id": str(hypothesis["hypothesis_id"]),
                "kind": str(hypothesis["kind"]), "stem": str(hypothesis["stem"]),
                "stem_entry_ids": list(hypothesis["stem_entry_ids"]),
                "stem_morpheme_id": hypothesis.get("root_morpheme_id"),
                "components": components,
                "derivational_affixes": derivational_affixes,
                "inflectional_affixes": inflectional_affixes,
                "unspecified_affixes": unspecified_affixes,
                "affix_classification_source": "explicit_caller_inventory" if (derivational_affixes or inflectional_affixes) else "unclassified",
                "stem_features": stem_features,
                "composed_features": _bounded_features(composed),
                "feature_contributions": contributions,
                "category_cues": _bounded_features(category_cues),
                "stem_frequency": max((float(entry.frequency) for entry in stem_entries), default=(
                    float(stem_morpheme.frequency) if stem_morpheme is not None else 0.0)),
                "affix_frequency": [float(item.frequency) for item in affixes],
                "affix_productivity": [float(item.productivity) for item in affixes],
            })
        return retrieved

    def _score_morphology(
        self,
        decompositions: Sequence[Mapping[str, Any]],
        retrieved: Sequence[Mapping[str, Any]],
        meanings: Sequence[Mapping[str, Any]],
        categories: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        cfg = self.config
        retrieval_by_id = {str(item["hypothesis_id"]): item for item in retrieved}
        active_semantic_features: dict[str, float] = defaultdict(float)
        for meaning in meanings:
            for feature, value in meaning["features"].items():
                active_semantic_features[feature] += float(meaning["posterior"]) * float(value)
        category_weights = {str(item["category"]): float(item["posterior"]) for item in categories}
        max_stem_frequency = _max_frequency([
            float(item["stem_frequency"]) for item in retrieved
        ])
        all_affix_freq = [value for item in retrieved for value in item["affix_frequency"]]
        max_affix_frequency = _max_frequency(all_affix_freq)
        ranked: list[dict[str, Any]] = []
        for hypothesis in decompositions:
            retrieved_item = retrieval_by_id.get(str(hypothesis["hypothesis_id"]), {})
            stem_frequency = float(retrieved_item.get("stem_frequency", 0.0))
            stem_support = math.log1p(stem_frequency) / math.log1p(max_stem_frequency) if stem_frequency > 0 else 0.0
            affix_terms = [
                (math.log1p(float(frequency)) / math.log1p(max_affix_frequency)) * float(productivity)
                for frequency, productivity in zip(
                    retrieved_item.get("affix_frequency", []),
                    retrieved_item.get("affix_productivity", []),
                )
            ]
            affix_support = sum(affix_terms) / len(affix_terms) if affix_terms else 0.5
            component_affixes = [
                component for component in retrieved_item.get("components", [])
                if component.get("morpheme_type") in {"prefix", "suffix"}
            ]
            derivational_components = [
                component for component in component_affixes
                if component.get("affix_category") == "derivational"
            ]
            inflectional_components = [
                component for component in component_affixes
                if component.get("affix_category") == "inflectional"
            ]
            unspecified_components = [
                component for component in component_affixes
                if component.get("affix_category") == "unspecified"
            ]
            def class_support(items: Sequence[Mapping[str, Any]]) -> float:
                """Calculate normalized frequency-productivity support within one affix class."""
                if not items:
                    return 0.0
                terms = [
                    (math.log1p(float(item["frequency"])) / math.log1p(max_affix_frequency))
                    * float(item["productivity"])
                    for item in items
                ]
                return sum(terms) / len(terms)
            derivational_support = class_support(derivational_components)
            inflectional_support = class_support(inflectional_components)
            unspecified_support = class_support(unspecified_components)
            composed_features = retrieved_item.get("composed_features", {})
            semantic_fit = _cosine(composed_features, active_semantic_features)
            category_cues = retrieved_item.get("category_cues", {})
            category_fit = sum(category_weights.get(category, 0.0) * cue for category, cue in category_cues.items())
            score = (
                float(hypothesis.get("_base_score", 0.0))
                + cfg.stem_frequency_gain * stem_support
                + cfg.affix_frequency_gain * affix_support
                + cfg.affix_productivity_gain * affix_support
                + 0.5 * cfg.affix_productivity_gain * (
                    derivational_support + inflectional_support
                )
                + cfg.semantic_consistency_gain * semantic_fit
                + cfg.category_consistency_gain * category_fit
                + cfg.morphology_gain * float(hypothesis["structural_validity"])
            )
            ranked.append({
                "hypothesis_id": str(hypothesis["hypothesis_id"]), "kind": str(hypothesis["kind"]),
                "prefix_id": hypothesis.get("prefix_id"), "stem": str(hypothesis["stem"]),
                "root_morpheme_id": hypothesis.get("root_morpheme_id"), "suffix_id": hypothesis.get("suffix_id"),
                "stem_entry_ids": list(hypothesis["stem_entry_ids"]),
                "morpheme_ids": list(hypothesis["morpheme_ids"]),
                "reconstruction": str(hypothesis["reconstruction"]),
                "stem_support": stem_support, "affix_support": affix_support,
                "derivational_affix_support": derivational_support,
                "inflectional_affix_support": inflectional_support,
                "unspecified_affix_support": unspecified_support,
                "derivational_affix_ids": [item["morpheme_id"] for item in derivational_components],
                "inflectional_affix_ids": [item["morpheme_id"] for item in inflectional_components],
                "unspecified_affix_ids": [item["morpheme_id"] for item in unspecified_components],
                "affix_classification_complete": not unspecified_components,
                "semantic_compatibility": semantic_fit, "category_compatibility": category_fit,
                "score": score,
            })
        weights = _softmax([item["score"] for item in ranked], cfg.morphology_temperature)
        for item, weight in zip(ranked, weights):
            item["posterior"] = weight
        ranked.sort(key=lambda item: item["posterior"], reverse=True)
        return ranked

    def _incremental_activation(
        self,
        word: str,
        identities: Sequence[Mapping[str, Any]],
        meanings: Sequence[Mapping[str, Any]],
        categories: Sequence[Mapping[str, Any]],
        morphology: Sequence[Mapping[str, Any]],
        simulator: LexicalAccessLIFSimulator | None,
    ) -> list[dict[str, Any]]:
        cfg = self.config
        entry_ids = [str(item["entry_id"]) for item in identities]
        by_id = {str(entry.entry_id): entry for entry in self.entries}
        frequencies = {
            item["entry_id"]: float(item["frequency"]) + float(item["exposure_count"])
            for item in identities
        }
        freq_max = _max_frequency(list(frequencies.values()))
        semantic_by_entry: dict[str, float] = defaultdict(float)
        for item in meanings:
            semantic_by_entry[str(item["entry_id"])] += float(item["posterior"])
        category_weights = {str(item["category"]): float(item["posterior"]) for item in categories}
        morph_by_entry: dict[str, float] = defaultdict(float)
        for item in morphology:
            if item["kind"] != "segmented":
                continue
            for entry_id in item["stem_entry_ids"]:
                morph_by_entry[str(entry_id)] += float(item["posterior"])
        activations = {entry_id: 0.04 for entry_id in entry_ids}
        trajectories: list[dict[str, Any]] = []
        similarity_matrix: dict[tuple[str, str], float] = {}
        for left_id in entry_ids:
            for right_id in entry_ids:
                if left_id != right_id:
                    similarity_matrix[(left_id, right_id)] = _similarity(
                        str(by_id[left_id].orthography).casefold(),
                        str(by_id[right_id].orthography).casefold(),
                    )
        operation_frame_summaries: dict[str, list[dict[str, Any]]] = {
            f"BLA-{number:03d}": [] for number in range(29, 37)
        }
        operation_frame_summaries["BLA-029"].append({
            "candidate_ids": entry_ids,
            "initial_activations": dict(activations),
            "character_count": len(word),
        })
        if simulator is not None:
            simulator.step("BLA-029", {"lexical_identity": 0.08, "recurrent_candidate": 0.04})

        for char_index in range(1, len(word) + 1):
            prefix = word[:char_index]
            bottom_up: dict[str, float] = {}
            prior_drive: dict[str, float] = {}
            semantic_drive: dict[str, float] = {}
            category_drive: dict[str, float] = {}
            morphology_drive: dict[str, float] = {}
            for identity in identities:
                entry_id = str(identity["entry_id"])
                form = str(identity["orthography"]).casefold()
                target_prefix = form[:char_index]
                if len(form) < char_index:
                    prefix_score = max(0.0, 1.0 - (char_index - len(form)) / max(len(word), 1))
                else:
                    prefix_score = _similarity(prefix, target_prefix)
                bottom_up[entry_id] = prefix_score
                prior_drive[entry_id] = math.log1p(frequencies[entry_id]) / math.log1p(freq_max)
                semantic_drive[entry_id] = min(1.0, semantic_by_entry.get(entry_id, 0.0))
                category = str(by_id[entry_id].lexical_category).casefold()
                category_drive[entry_id] = category_weights.get(category, 0.0)
                morphology_drive[entry_id] = min(1.0, morph_by_entry.get(entry_id, 0.0))

            mean_bottom = sum(bottom_up.values()) / len(bottom_up) if bottom_up else 0.0
            mean_prior = sum(prior_drive.values()) / len(prior_drive) if prior_drive else 0.0
            mean_semantic = sum(semantic_drive.values()) / len(semantic_drive) if semantic_drive else 0.0
            mean_category = sum(category_drive.values()) / len(category_drive) if category_drive else 0.0
            mean_morph = sum(morphology_drive.values()) / len(morphology_drive) if morphology_drive else 0.0
            operation_frame_summaries["BLA-030"].append({
                "character_index": char_index, "observed_prefix": prefix,
                "bottom_up_evidence": dict(bottom_up), "mean_bottom_up_evidence": mean_bottom,
            })
            operation_frame_summaries["BLA-031"].append({
                "character_index": char_index, "observed_prefix": prefix,
                "frequency_prior": dict(prior_drive), "mean_frequency_prior": mean_prior,
            })
            if simulator is not None:
                simulator.step("BLA-030", {"orthographic": mean_bottom})
                simulator.step("BLA-031", {"lexical_identity": mean_prior})

            mean_recurrent = 0.0
            mean_inhibitory = 0.0
            mean_top_down = 0.0
            for _ in range(cfg.settle_iterations_per_character):
                next_activations: dict[str, float] = {}
                recurrent_values: list[float] = []
                inhibitory_values: list[float] = []
                top_down_values: list[float] = []
                for entry_id in entry_ids:
                    inhibitory = sum(
                        similarity_matrix.get((entry_id, other_id), 0.0) * activations[other_id]
                        for other_id in entry_ids if other_id != entry_id
                    ) / max(1, len(entry_ids) - 1)
                    top_down = (
                        cfg.semantic_feedback_gain * semantic_drive[entry_id]
                        + cfg.category_feedback_gain * category_drive[entry_id]
                        + cfg.morphology_feedback_gain * morphology_drive[entry_id]
                    )
                    recurrent = cfg.recurrent_excitation * activations[entry_id]
                    inhibition_term = cfg.lateral_inhibition * inhibitory
                    evidence = (
                        cfg.activation_bias
                        + cfg.bottom_up_gain * bottom_up[entry_id]
                        + cfg.frequency_prior_gain * prior_drive[entry_id]
                        + top_down + recurrent - inhibition_term
                    )
                    target = _sigmoid(evidence / cfg.activation_temperature)
                    next_activations[entry_id] = (
                        (1.0 - cfg.activation_damping) * activations[entry_id]
                        + cfg.activation_damping * target
                    )
                    recurrent_values.append(recurrent)
                    inhibitory_values.append(inhibition_term)
                    top_down_values.append(top_down)
                activations = next_activations
                mean_recurrent = sum(recurrent_values) / max(1, len(recurrent_values))
                mean_inhibitory = sum(inhibitory_values) / max(1, len(inhibitory_values))
                mean_top_down = sum(top_down_values) / max(1, len(top_down_values))

            mean_activation = sum(activations.values()) / max(1, len(activations))
            max_activation = max(activations.values(), default=0.0)
            operation_frame_summaries["BLA-032"].append({
                "character_index": char_index, "observed_prefix": prefix,
                "mean_recurrent_excitation": mean_recurrent,
            })
            operation_frame_summaries["BLA-033"].append({
                "character_index": char_index, "observed_prefix": prefix,
                "mean_lateral_inhibition": mean_inhibitory,
            })
            operation_frame_summaries["BLA-034"].append({
                "character_index": char_index, "observed_prefix": prefix,
                "mean_top_down_drive": mean_top_down,
                "mean_semantic_feedback": mean_semantic,
                "mean_category_feedback": mean_category,
                "mean_morphology_feedback": mean_morph,
            })
            operation_frame_summaries["BLA-035"].append({
                "character_index": char_index, "observed_prefix": prefix,
                "mean_activation": mean_activation, "maximum_activation": max_activation,
            })

            frame = {
                "character_index": char_index,
                "observed_prefix": prefix,
                "candidates": [
                    {"entry_id": entry_id, "orthography": str(by_id[entry_id].orthography),
                     "bottom_up_evidence": bottom_up[entry_id], "frequency_prior": prior_drive[entry_id],
                     "semantic_feedback": semantic_drive[entry_id], "category_feedback": category_drive[entry_id],
                     "morphology_feedback": morphology_drive[entry_id], "activation": activations[entry_id]}
                    for entry_id in entry_ids
                ],
            }
            frame["candidates"].sort(key=lambda item: item["activation"], reverse=True)
            trajectories.append(frame)
            ranking = [{"entry_id": item["entry_id"], "activation": item["activation"]}
                       for item in frame["candidates"][:2]]
            operation_frame_summaries["BLA-036"].append({
                "character_index": char_index, "observed_prefix": prefix, "leading_candidates": ranking,
            })
            if simulator is not None:
                simulator.step("BLA-032", {"recurrent_candidate": min(1.0, mean_recurrent)})
                simulator.step("BLA-033", {"selection_control": min(1.0, mean_inhibitory)})
                simulator.step("BLA-034", {
                    "semantic": mean_semantic, "grammatical_category": mean_category,
                    "morphological": mean_morph,
                })
                simulator.step("BLA-035", {
                    "orthographic": mean_bottom, "lexical_identity": max_activation,
                    "semantic": mean_semantic, "grammatical_category": mean_category,
                    "morphological": mean_morph, "recurrent_candidate": max_activation,
                }, cycles=cfg.steps_per_character)

        if simulator is not None:
            final_activation = max(activations.values(), default=0.0)
            simulator.step("BLA-036", {
                "lexical_identity": final_activation,
                "selection_control": 1.0 if len(activations) <= 1 else 0.5,
                "recurrent_candidate": final_activation,
            })
        if trajectories:
            final_frame = trajectories[-1]
            operation_frame_summaries["BLA-036"].append({
                "final_ranking": [{"entry_id": item["entry_id"], "activation": item["activation"]}
                                  for item in final_frame["candidates"]],
            })
        else:
            operation_frame_summaries["BLA-036"].append({"final_ranking": [], "reason": "no lexical candidates"})
        for operation_number in range(29, 37):
            operation_id = f"BLA-{operation_number:03d}"
            self._pending_stage7_trace.append((operation_id, operation_frame_summaries[operation_id]))
        return trajectories

    def _confident_choice(self, records: Sequence[Mapping[str, Any]], *, identity: bool = False) -> dict[str, Any] | None:
        if not records:
            return None
        top = records[0]
        top_probability = float(top.get("posterior", 0.0))
        second_probability = float(records[1].get("posterior", 0.0)) if len(records) > 1 else 0.0
        margin = top_probability - second_probability
        if top_probability < self.config.confidence_threshold:
            return None
        if len(records) > 1 and margin < self.config.ambiguity_margin:
            return None
        if identity and float(top.get("orthographic_similarity", 0.0)) < 0.85:
            return None
        return dict(top)

    def process(self, request: LexicalAccessRequest) -> LexicalAccessResult:
        """Execute stages S1-S7. The supplied lexical inventories remain read-only."""
        cfg = self.config
        original = request.word
        word = unicodedata.normalize("NFC", original.strip()).casefold()
        simulator = LexicalAccessLIFSimulator(cfg) if cfg.simulate_biology else None
        trace: list[dict[str, Any]] = []
        self._pending_stage7_trace: list[tuple[str, list[dict[str, Any]]]] = []

        def mark(operation_number: int, output: Mapping[str, Any], drives: Mapping[str, float]) -> None:
            spec = OPERATION_REGISTRY[operation_number - 1]
            clean_output = dict(output)
            trace.append({
                "operation_id": spec.operation_id, "stage_id": spec.stage_id,
                "stage_name": spec.stage_name, "operation_name": spec.operation_name,
                "mathematical_form": spec.mathematical_form,
                "output": clean_output,
            })
            if simulator is not None:
                simulator.step(spec.operation_id, drives)

        # S1 — Stored word identity.
        mark(1, {"normalized_word": word}, {"orthographic": 0.15})
        identities = self._identity_candidates(word)
        selected_identity = self._confident_choice(identities, identity=True)
        mark(2, {"candidate_ids": [item["entry_id"] for item in identities]},
             {"orthographic": 0.45 if identities else 0.0})
        mark(3, {"orthographic_likelihoods": {item["entry_id"]: item["orthographic_likelihood"] for item in identities}},
             {"orthographic": max((item["orthographic_similarity"] for item in identities), default=0.0)})
        mark(4, {"frequency_strengths": {item["entry_id"]: item["frequency_strength"] for item in identities}},
             {"lexical_identity": sum((item["frequency_strength"] for item in identities), 0.0) / max(1, len(identities))})
        mark(5, {"identity_posterior": {item["entry_id"]: item["posterior"] for item in identities}},
             {"lexical_identity": max((item["posterior"] for item in identities), default=0.0)})

        # S2 — Meaning retrieval.
        meanings = self._meaning_candidates(identities, request.context_features)
        selected_meaning = self._confident_choice(meanings)
        mark(6, {"meaning_ids": [item["meaning_id"] for item in meanings]},
             {"semantic": min(1.0, len(meanings) / 3.0)})
        mark(7, {"retrieved_feature_maps": {item["meaning_id"]: item["features"] for item in meanings}},
             {"semantic": max((sum(item["features"].values()) / max(1, len(item["features"])) for item in meanings), default=0.0)})
        mark(8, {"meaning_priors": {item["meaning_id"]: item["lexical_identity_posterior"] * item["prior"] for item in meanings}},
             {"semantic": max((item["lexical_identity_posterior"] for item in meanings), default=0.0)})
        mark(9, {"context_compatibility": {item["meaning_id"]: item["context_compatibility"] for item in meanings}},
             {"semantic": max((item["context_compatibility"] for item in meanings), default=0.0)})
        meaning_distribution: dict[str, float] = defaultdict(float)
        for item in meanings:
            meaning_distribution[item["meaning_id"]] += item["posterior"]
        meaning_top = max(meaning_distribution.values(), default=0.0)
        mark(10, {"meaning_posterior": dict(meaning_distribution), "meaning_confidence": meaning_top},
             {"semantic": meaning_top})

        # S3 — Grammatical category.
        mark(11, {"retrieved_categories": sorted({item["lexical_category"] for item in identities if item["lexical_category"] != "unknown"})},
             {"grammatical_category": 0.25 if identities else 0.0})
        category_evidence: dict[str, float] = defaultdict(float)
        for item in identities:
            category = item["lexical_category"].casefold()
            if category != "unknown":
                category_evidence[category] += item["posterior"]
        mark(12, {"category_evidence": dict(category_evidence)},
             {"grammatical_category": max(category_evidence.values(), default=0.0)})
        mark(13, {"contextual_category_priors": dict(request.contextual_category_priors)},
             {"grammatical_category": max(request.contextual_category_priors.values(), default=0.0)})
        categories = self._category_candidates(identities, request.contextual_category_priors)
        selected_category = self._confident_choice(categories)
        mark(14, {"category_posterior": categories,
                  "top_two_margin": (categories[0]["posterior"] - categories[1]["posterior"]) if len(categories) > 1 else (categories[0]["posterior"] if categories else 0.0)},
             {"grammatical_category": max((item["posterior"] for item in categories), default=0.0)})

        # S4 — Morphological decomposition.
        mark(15, {"supplied_prefixes": [str(item.morpheme_id) for item in self.morphemes if item.morpheme_type == "prefix"],
                  "supplied_roots": [str(item.morpheme_id) for item in self.morphemes if item.morpheme_type == "root"],
                  "supplied_suffixes": [str(item.morpheme_id) for item in self.morphemes if item.morpheme_type == "suffix"]},
             {"morphological": 0.15 if self.morphemes else 0.0})
        hypotheses = self._decompositions(word)
        mark(16, {"candidate_hypothesis_count": len(hypotheses)},
             {"morphological": min(1.0, len(hypotheses) / 3.0)})
        mark(17, {"exact_reconstruction_checks": {item["hypothesis_id"]: item["reconstruction"] == word for item in hypotheses}},
             {"morphological": 1.0 if hypotheses else 0.0})
        mark(18, {"stem_resolution": {item["hypothesis_id"]: item["stem_entry_ids"] for item in hypotheses}},
             {"morphological": min(1.0, sum(bool(item["stem_entry_ids"] or item.get("root_morpheme_id")) for item in hypotheses) / max(1, len(hypotheses)))})
        mark(19, {"preserved_routes": [{"hypothesis_id": item["hypothesis_id"], "kind": item["kind"]} for item in hypotheses]},
             {"morphological": 0.25 if hypotheses else 0.0})

        # S5 — Retrieve stem and affix representations.
        retrieved = self._retrieve_components(hypotheses)
        mark(20, {"stem_records": [{"hypothesis_id": item["hypothesis_id"], "stem": item["stem"],
                                     "entry_ids": item["stem_entry_ids"], "root_morpheme_id": item["stem_morpheme_id"]} for item in retrieved]},
             {"morphological": min(1.0, len(retrieved) / 3.0)})
        mark(21, {"affix_records": [{
                    "hypothesis_id": item["hypothesis_id"], "components": item["components"],
                    "derivational_affixes": item["derivational_affixes"],
                    "inflectional_affixes": item["inflectional_affixes"],
                    "unspecified_affixes": item["unspecified_affixes"],
                    "classification_source": item["affix_classification_source"],
                } for item in retrieved]},
             {"morphological": max((len(item["components"]) / 3.0 for item in retrieved), default=0.0)})
        mark(22, {"component_semantic_and_category_cues": [{"hypothesis_id": item["hypothesis_id"],
                                                              "stem_features": item["stem_features"],
                                                              "category_cues": item["category_cues"]} for item in retrieved]},
             {"semantic": max((sum(item["stem_features"].values()) / max(1, len(item["stem_features"])) for item in retrieved), default=0.0),
              "grammatical_category": max((max(item["category_cues"].values(), default=0.0) for item in retrieved), default=0.0)})
        mark(23, {"composed_morpheme_features": [{"hypothesis_id": item["hypothesis_id"],
                                                   "composed_features": item["composed_features"],
                                                   "feature_contributions": item["feature_contributions"]} for item in retrieved]},
             {"semantic": max((sum(item["composed_features"].values()) / max(1, len(item["composed_features"])) for item in retrieved), default=0.0),
              "morphological": 0.5 if retrieved else 0.0})

        # S6 — Distinguish morphological alternatives.
        mark(24, {"stem_support": {item["hypothesis_id"]: item["stem_frequency"] for item in retrieved}},
             {"morphological": max((item["stem_frequency"] for item in retrieved), default=0.0) / max(1.0, _max_frequency([item["stem_frequency"] for item in retrieved]))})
        mark(25, {"affix_support": {item["hypothesis_id"]: {"frequency": item["affix_frequency"], "productivity": item["affix_productivity"]} for item in retrieved}},
             {"morphological": max((sum(item["affix_productivity"]) / len(item["affix_productivity"]) for item in retrieved if item["affix_productivity"]), default=0.0)})
        mark(26, {"semantic_category_compatibility_inputs": [{"hypothesis_id": item["hypothesis_id"],
                                                               "features": item["composed_features"],
                                                               "category_cues": item["category_cues"]} for item in retrieved]},
             {"semantic": 0.35 if meanings and retrieved else 0.0, "selection_control": 0.15 if len(retrieved) > 1 else 0.0})
        morphology = self._score_morphology(hypotheses, retrieved, meanings, categories)
        mark(27, {"morphology_scores_and_posteriors": morphology},
             {"morphological": max((item["posterior"] for item in morphology), default=0.0)})
        top_morph = morphology[0] if morphology else None
        second_morph = morphology[1]["posterior"] if len(morphology) > 1 else 0.0
        morph_margin = float(top_morph["posterior"] - second_morph) if top_morph else 0.0
        morph_selected = bool(
            top_morph and top_morph["posterior"] >= cfg.confidence_threshold
            and (len(morphology) == 1 or morph_margin >= cfg.ambiguity_margin)
        )
        preferred = dict(top_morph) if morph_selected and top_morph else None
        mark(28, {"preferred_morphology": preferred, "abstained": not morph_selected,
                  "top_two_margin": morph_margin},
             {"selection_control": 1.0 if morph_selected else 0.35})

        # S7 — Incremental interactive activation.
        incremental = self._incremental_activation(word, identities, meanings, categories, morphology, simulator)
        for operation_id, frame_summaries in self._pending_stage7_trace:
            operation_number = int(operation_id.rsplit("-", 1)[1])
            spec = OPERATION_REGISTRY[operation_number - 1]
            trace.append({
                "operation_id": spec.operation_id, "stage_id": spec.stage_id,
                "stage_name": spec.stage_name, "operation_name": spec.operation_name,
                "mathematical_form": spec.mathematical_form,
                "output": {"per_character": frame_summaries},
            })
        final_candidates = incremental[-1]["candidates"] if incremental else []
        top_activation = float(final_candidates[0]["activation"]) if final_candidates else 0.0
        second_activation = float(final_candidates[1]["activation"]) if len(final_candidates) > 1 else 0.0
        activation_margin = top_activation - second_activation
        if simulator is not None:
            biological = simulator.snapshot()
        else:
            biological = {
                "enabled": False, "model": "disabled", "layers": {},
                "layer_order": list(POPULATIONS), "simulated_time_ms": 0.0,
                "numerically_valid": True, "operation_trace": [],
                "scope": "Biological reference disabled by configuration.",
            }
        valid = (
            len(trace) == len(OPERATION_REGISTRY)
            and all(math.isfinite(float(candidate["activation"])) for frame in incremental for candidate in frame["candidates"])
            and biological.get("numerically_valid", True) is True
        )
        diagnostics = {
            "valid": valid,
            "stage_count": 7,
            "operation_count": len(trace),
            "registry_operation_count": len(OPERATION_REGISTRY),
            "operations_per_word": len(OPERATION_REGISTRY),
            "lexical_candidate_count": len(identities),
            "meaning_candidate_count": len(meanings),
            "grammatical_category_count": len(categories),
            "morphological_alternative_count": len(morphology),
            "incremental_character_frames": len(incremental),
            "incremental_top_activation": top_activation,
            "incremental_activation_margin": activation_margin,
            "lexical_identity_abstained": selected_identity is None,
            "meaning_abstained": selected_meaning is None and bool(meanings),
            "grammatical_category_abstained": selected_category is None and bool(categories),
            "morphology_abstained": not morph_selected,
            "biological_numerically_valid": biological.get("numerically_valid", True),
        }
        metadata = {
            "language": request.language,
            "source_reference": request.source_reference,
            "timestamp_ms": request.timestamp_ms,
            "parameters": cfg.to_dict(),
            "inventory_policy": "read-only; only caller-supplied lexical entries and morphemes are used",
            "affix_classification_policy": (
                "derivational and inflectional affixes are distinguished only when explicitly "
                "classified by morpheme_affix_categories; unclassified affixes remain unspecified"
            ),
            "biological_scope": "generic seven-population recurrent LIF reference; not fitted to human data",
            "contextual_records": [
                {"context_id": item.context_id, "context_type": item.context_type,
                 "value": item.value, "language_profile": item.language_profile, "scope": item.scope}
                for item in request.contextual_records
            ],
            "evidence_records": [
                {"evidence_id": item.evidence_id, "source_id": item.source_id,
                 "evidence_type": item.evidence_type, "support_status": item.support_status,
                 "source_rights_status": item.source_rights_status,
                 "source_redistribution": item.source_redistribution,
                 "source_license_id": item.source_license_id,
                 "usable_for_local_reasoning": item.usable_for_local_reasoning,
                 "verified_fact": False, "interpretation": item.interpretation}
                for item in request.evidence_records
            ],
            "source_provenance": [
                {"source_id": item.source_id, "provider": item.provider,
                 "source_type": item.source_type, "rights_status": item.rights_status,
                 "redistribution": item.redistribution, "attribution": item.attribution,
                 "license_id": item.license_id, "usable_for_local_import": item.usable_for_local_import,
                 "redistributable": item.redistributable}
                for item in request.source_provenance
            ],
            "evidence_is_verified_fact": False,
        }
        return LexicalAccessResult(
            request.sequence_id, original, word, identities, selected_identity,
            meanings, selected_meaning, categories, selected_category,
            morphology, retrieved, preferred, incremental, biological, trace,
            diagnostics, metadata,
        )
