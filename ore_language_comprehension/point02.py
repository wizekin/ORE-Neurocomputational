"""Source-ordered Point 2 lexical access over ORE's established lexical engine.

This module turns Point 1's ordered auditory/visual form hypotheses into ranked lexical
identities, word-level semantic features, grammatical properties and unresolved sense options.
The seven-stage/36-operation ore_lexical_access engine supplies the lower-level functional and
LIF-reference computations; this adapter preserves the 12-step sequence in the source document.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
import unicodedata
from typing import Any, Mapping, Sequence

from ore_lexical_access import LexicalAccessConfig, LexicalAccessProcessor, LexicalAccessRequest
from ore_lexical_context import LexicalEntry, MorphemeEntry
from ore_sensory_perception.contracts import to_jsonable


POINT2_STEP_SPECS: tuple[dict[str, Any], ...] = (
    {
        "operation_id": "LC-P02-01",
        "name": "Accept graded form evidence without forcing a final word decision",
        "objective": "Receive Point 1 auditory phoneme or visual grapheme alternatives, preserve their order, boundaries, confidence and uncertainty.",
        "mathematical_operations": ("reliability-weighted features x_tilde[k]=r[k]*x[k]", "alternative hypothesis distributions", "uncertainty descriptors and boundary intervals"),
        "biological_reference": "Graded activity in auditory/visual sensory-language populations with recurrent feedback; explicit covariance matrices are modelling conveniences, not claimed neural data structures.",
    },
    {
        "operation_id": "LC-P02-02",
        "name": "Select modality-specific lexical access routes",
        "objective": "Keep auditory phonological matching and visual orthographic matching distinct while allowing both to reach the same lexicon when caller-supplied entries link the forms.",
        "mathematical_operations": ("z_a=f_a(x_a) and z_v=f_v(x_v)", "modality-specific likelihoods", "shared lexical identity by explicit entry ID and form links"),
        "biological_reference": "Auditory temporal speech pathways and ventral occipitotemporal orthographic pathways converge on overlapping temporal lexical-semantic networks; no single dictionary area is assumed.",
    },
    {
        "operation_id": "LC-P02-03",
        "name": "Generate initial lexical candidates from partial and complete form similarity",
        "objective": "Activate entries compatible with the incoming phoneme or grapheme sequence, including partial matches, and preserve plausible competitors.",
        "mathematical_operations": ("weighted feature match M_i=x^T*phi_i", "sequence/edit distance d_i", "position-sensitive reliability sum", "permissive candidate threshold and beam search"),
        "biological_reference": "Overlapping phonological/orthographic populations coactivate as shared form features arrive; activation can precede completion of a spoken or written word.",
    },
    {
        "operation_id": "LC-P02-04",
        "name": "Add long-term familiarity and word-frequency influences",
        "objective": "Use supplied frequency and exposure information as a baseline readiness term, separately from current context.",
        "mathematical_operations": ("log-frequency bias b_i=alpha*log(1+f_i)", "experience strength update q_i[t+1]=q_i[t]+eta*delta_i", "short-term trace r_i[t+1]=lambda*r_i[t]+u_i[t]"),
        "biological_reference": "Repeated co-activation and experience can alter distributed connectivity and pattern readiness; an inventory frequency is an engineering proxy, not a neural counter.",
    },
    {
        "operation_id": "LC-P02-05",
        "name": "Incorporate only currently available local expectations",
        "objective": "Use supplied nearby semantic features and category priors as a limited bias without allowing context to override incompatible sensory evidence or perform discourse pragmatics.",
        "mathematical_operations": ("P(w_i|x,c) proportional to P(x|w_i)*P(w_i|c)", "weighted score S_i=alpha*M_i+beta*F_i+gamma*C_i", "reliability-dependent context gain gamma=gamma0*(1-r_x)"),
        "biological_reference": "Top-down feedback among temporal and frontal language networks can bias lexical populations; the source keeps this local prediction separate from later discourse and intention reasoning.",
    },
    {
        "operation_id": "LC-P02-06",
        "name": "Activate supported morphological analyses",
        "objective": "Compare stored whole-word entries with legal, inventory-supported stem/prefix/suffix decompositions and retain alternatives.",
        "mathematical_operations": ("candidate segmentation search", "P(parts|x) proportional to P(x|parts)*P(parts)", "morpheme feature composition v_word=g(v_stem,v_affix)", "whole-form/decomposition competition"),
        "biological_reference": "Learned whole-form and morpheme-family representations may coactivate in distributed language networks; morphology is language-specific and not obligatory for every word.",
    },
    {
        "operation_id": "LC-P02-07",
        "name": "Resolve lexical-form competition gradually",
        "objective": "Accumulate compatible evidence, suppress contradicted alternatives and preserve ambiguity when the candidate margin is small.",
        "mathematical_operations": ("recurrent excitation and lateral inhibition", "sigmoid activation", "softmax candidate distribution", "evidence accumulation and entropy H=-sum(p_i*log(p_i))"),
        "biological_reference": "Recurrent excitation, inhibitory interactions, adaptation and changing sensory evidence provide a plausible population-level analogue; effortful frontal control is conditional, not mandatory for every familiar word.",
    },
    {
        "operation_id": "LC-P02-08",
        "name": "Retrieve distributed word-level semantic features",
        "objective": "Activate conventional category, attribute, action, function and associative features for each active lexical candidate without composing a sentence proposition.",
        "mathematical_operations": ("associative spread y=W*a+b", "nonlinear feature activation sigmoid(W*a+b)", "distributed feature vector v_w", "sparse/thresholded coding and feature-reliability weighting"),
        "biological_reference": "Distributed temporal and modality-linked association systems contribute auditory, visual, motor, affective, social and other concept features; an ATL-style convergence account is a model, not an exclusive storage claim.",
    },
    {
        "operation_id": "LC-P02-09",
        "name": "Retrieve grammatical and lexical properties for later parsing",
        "objective": "Expose stored lexical category, expressed morphology, and caller-supplied argument/complement properties as candidate constraints without assigning sentence roles.",
        "mathematical_operations": ("lexical attribute vector g_i", "conditional feature probability P(g|w,x)", "soft constraint weights for later parsing"),
        "biological_reference": "Temporal and frontal language-responsive populations exchange lexical and grammatical information; the syntactic parser, not Point 2, assigns actual participants and relations.",
    },
    {
        "operation_id": "LC-P02-10",
        "name": "Preserve homonym and polysemy alternatives",
        "objective": "Keep unrelated entries with the same form distinct and keep multiple related senses attached to one entry when current local evidence cannot resolve them.",
        "mathematical_operations": ("sense posterior P(s_j|w,c) proportional to P(c|s_j,w)*P(s_j|w)", "shared-core plus sense offsets", "top-k sense retention", "confidence margin Delta=p_best-p_second"),
        "biological_reference": "Overlapping semantic populations can support related senses while distinct lexical/concept patterns support homonyms; later syntax and discourse may revise the preferred sense.",
    },
    {
        "operation_id": "LC-P02-11",
        "name": "Reconcile candidates with new sensory evidence",
        "objective": "Update existing candidate scores from additional phoneme/grapheme evidence, request a less committed Point 1 hypothesis when necessary and never equate stability with certainty.",
        "mathematical_operations": ("incremental score update S_i[t+1]=S_i[t]+Delta*S_i", "prediction error e=x-x_hat", "recurrent state h[t+1]=f(W_in*x+W_rec*h+W_top*c)", "confidence-and-margin stopping gate"),
        "biological_reference": "Recurrent connections among sensory form areas, temporal language regions and conditional frontal selection networks permit interactive correction and word-boundary revision.",
    },
    {
        "operation_id": "LC-P02-12",
        "name": "Package lexical-semantic output for syntactic parsing",
        "objective": "Return ranked word identities, semantic features, lexical properties, morphology, senses, confidence and unresolved alternatives as word-level records.",
        "mathematical_operations": ("candidate probability normalization", "confidence and margin calibration", "structured lexical-semantic record assembly"),
        "biological_reference": "Distributed lexical-semantic activation makes information available to subsequent syntactic processing; this stage does not construct a sentence-level predicate-argument structure.",
    },
)


# Source-level steps are mapped to the existing 36-operation BLA lexical engine.
_POINT2_TO_BLA: dict[int, tuple[int, ...]] = {
    1: (1,),
    2: (1,),
    3: (2, 3),
    4: (4,),
    5: (6, 8, 9, 10),
    6: (15, 16, 17, 18, 19, 20, 21, 22, 23),
    7: (5, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36),
    8: (6, 7),
    9: (11, 12, 13, 14),
    10: (6, 7, 8, 9, 10),
    11: (29, 30, 31, 32, 33, 34, 35, 36),
    12: tuple(range(1, 37)),
}


@dataclass(frozen=True, slots=True)
class Point2Request:
    """Supply Point 1 form hypotheses, an explicit lexicon and optional local lexical evidence.

    The form_hypotheses field is the Point1Result.form_hypotheses mapping. Lexical entries and
    morphemes are caller-provided ore_lexical_context records; missing pronunciations, meanings,
    or grammatical properties are never fabricated.
    """

    sequence_id: str
    form_hypotheses: Mapping[str, Any]
    lexicon: Sequence[LexicalEntry]
    morphemes: Sequence[MorphemeEntry] = ()
    language: str = "und"
    context_features: Mapping[str, float] = field(default_factory=dict)
    contextual_category_priors: Mapping[str, float] = field(default_factory=dict)
    lexical_properties: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    morpheme_category_cues: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    morpheme_affix_categories: Mapping[str, str] = field(default_factory=dict)
    sensory_updates: Sequence[Mapping[str, Any]] = ()
    source_reference: str | None = None

    def __post_init__(self) -> None:
        """Validate the form handoff, explicit lexical inventory and all caller-supplied priors."""
        if not self.sequence_id.strip():
            raise ValueError("sequence_id must be non-empty")
        if not isinstance(self.form_hypotheses, Mapping):
            raise TypeError("form_hypotheses must be the Point 1 form-hypothesis mapping")
        if not self.lexicon:
            raise ValueError("lexicon must contain at least one explicit LexicalEntry")
        entry_ids = [str(item.entry_id) for item in self.lexicon]
        if len(entry_ids) != len(set(entry_ids)):
            raise ValueError("lexicon entry_id values must be unique")
        for name, values in (
            ("context_features", self.context_features),
            ("contextual_category_priors", self.contextual_category_priors),
        ):
            for key, raw in values.items():
                value = float(raw)
                if not str(key).strip() or not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"{name} values must be finite and within [0, 1]")
        if any(str(key) not in set(entry_ids) for key in self.lexical_properties):
            raise ValueError("lexical_properties may only reference supplied lexical entry IDs")
        for update in self.sensory_updates:
            if update.get("modality") not in {"auditory", "visual"}:
                raise ValueError("each sensory update must specify modality='auditory' or 'visual'")
            if update.get("modality") == "visual" and not str(update.get("form", "")).strip():
                raise ValueError("visual sensory updates require a non-empty form")
            if update.get("modality") == "auditory" and not update.get("phonemes"):
                raise ValueError("auditory sensory updates require a non-empty phonemes sequence")
        object.__setattr__(self, "lexicon", tuple(self.lexicon))
        object.__setattr__(self, "morphemes", tuple(self.morphemes))
        object.__setattr__(self, "sensory_updates", tuple(dict(item) for item in self.sensory_updates))

    @classmethod
    def from_point1_result(
        cls,
        sequence_id: str,
        point1_result: Any,
        lexicon: Sequence[LexicalEntry],
        **kwargs: Any,
    ) -> "Point2Request":
        """Construct a Point 2 request directly from a Point1Result or its serialized mapping."""
        hypotheses = getattr(point1_result, "form_hypotheses", None)
        if hypotheses is None and isinstance(point1_result, Mapping):
            hypotheses = point1_result.get("form_hypotheses", point1_result)
        if not isinstance(hypotheses, Mapping):
            raise TypeError("point1_result must expose a form_hypotheses mapping")
        return cls(
            sequence_id=sequence_id,
            form_hypotheses=hypotheses,
            lexicon=lexicon,
            **kwargs,
        )


@dataclass(slots=True)
class Point2Result:
    """Store ranked lexical entries, word meanings, grammatical properties and separate biological traces."""

    sequence_id: str
    modality_routes: Mapping[str, Any]
    form_candidates: Sequence[Mapping[str, Any]]
    lexical_candidates: Sequence[Mapping[str, Any]]
    selected_candidate: Mapping[str, Any] | None
    meaning_candidates: Sequence[Mapping[str, Any]]
    word_level_semantic_features: Mapping[str, float]
    lexical_properties: Mapping[str, Any]
    sense_candidates: Sequence[Mapping[str, Any]]
    unresolved_ambiguities: Sequence[Mapping[str, Any]]
    confidence: float
    functional_trace: Sequence[Mapping[str, Any]]
    biological_trace: Sequence[Mapping[str, Any]]
    low_level_operation_trace: Sequence[Mapping[str, Any]]
    diagnostics: Mapping[str, Any]
    annotation_report: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete lexical-access result into bounded JSON-compatible values."""
        return to_jsonable({
            "schema": "ore.language_comprehension.point02.v1",
            "sequence_id": self.sequence_id,
            "modality_routes": self.modality_routes,
            "form_candidates": self.form_candidates,
            "lexical_candidates": self.lexical_candidates,
            "selected_candidate": self.selected_candidate,
            "meaning_candidates": self.meaning_candidates,
            "word_level_semantic_features": self.word_level_semantic_features,
            "lexical_properties": self.lexical_properties,
            "sense_candidates": self.sense_candidates,
            "unresolved_ambiguities": self.unresolved_ambiguities,
            "confidence": self.confidence,
            "functional_trace": self.functional_trace,
            "biological_trace": self.biological_trace,
            "low_level_operation_trace": self.low_level_operation_trace,
            "diagnostics": self.diagnostics,
            "annotation_report": self.annotation_report,
            "scope": {
                "sentence_structure_built": False,
                "predicate_argument_structure_built": False,
                "discourse_pragmatics_performed": False,
                "biological_trace_is_measured_human_data": False,
            },
        })


def _softmax(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Convert finite candidate scores into a numerically stable relative distribution."""
    if not scores:
        return []
    temp = max(float(temperature), 1e-9)
    scaled = [max(-100.0, min(100.0, float(score) / temp)) for score in scores]
    peak = max(scaled)
    weights = [math.exp(item - peak) for item in scaled]
    total = sum(weights)
    return [weight / total for weight in weights] if total else [1.0 / len(scores)] * len(scores)


def _edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    """Calculate Levenshtein distance over characters or phoneme symbols."""
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (left_item != right_item),
            ))
        previous = current
    return previous[-1]


def _cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Measure overlap between sparse semantic feature maps without inventing absent dimensions."""
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    dot = sum(float(left.get(key, 0.0)) * float(right.get(key, 0.0)) for key in keys)
    norm_left = math.sqrt(sum(float(left.get(key, 0.0)) ** 2 for key in keys))
    norm_right = math.sqrt(sum(float(right.get(key, 0.0)) ** 2 for key in keys))
    return dot / (norm_left * norm_right) if norm_left > 0.0 and norm_right > 0.0 else 0.0


class Point2LexicalAccessProcessor:
    """Run the 12-step Point 2 adapter over ORE's 36-operation lexical engine and LIF reference.

    Visual glyph alternatives are beam-composed into candidate orthographies. Auditory phoneme
    alternatives are compared with caller-supplied lexical-entry pronunciations. Both routes then
    retrieve meaning, morphology and lexical properties from the same explicit lexical inventory.
    """

    def __init__(
        self,
        config: LexicalAccessConfig | None = None,
        *,
        top_k: int = 5,
        max_form_candidates: int = 12,
        register_annotations: bool = True,
        repository_root: str | None = None,
        database_path: str | None = None,
    ) -> None:
        """Configure lexical evidence, candidate retention and optional shared SQLite annotation indexing."""
        if top_k < 1 or max_form_candidates < 1:
            raise ValueError("top_k and max_form_candidates must be positive")
        self.config = config or LexicalAccessConfig()
        self.top_k = int(top_k)
        self.max_form_candidates = int(max_form_candidates)
        if register_annotations:
            from .annotations import register_point02_annotations

            self.annotation_report = register_point02_annotations(repository_root, database_path)
        else:
            self.annotation_report = {"status": "annotation_registration_disabled"}

    @staticmethod
    def _candidate_probability(candidate: Mapping[str, Any]) -> float:
        """Read a bounded confidence value from a Point 1 or direct sensory form candidate."""
        value = float(candidate.get("confidence", candidate.get("probability", 0.0)))
        return min(1.0, max(0.0, value))

    def _visual_word_segments(
        self,
        glyphs: Sequence[Mapping[str, Any]],
    ) -> list[tuple[int, list[Mapping[str, Any]], str]]:
        """Group ordered glyphs into line/word candidates using only clear Point 1 geometry gaps.

        The gap heuristic is deliberately conservative. It only proposes a split when the
        horizontal whitespace exceeds a fraction of the median glyph width. When bounding boxes
        are missing or unreliable, it returns one unresolved region segment instead of guessing.
        """
        usable = []
        heights = []
        widths = []
        for glyph in glyphs:
            bbox = glyph.get("bounding_box_y0_x0_y1_x1")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            y0, x0, y1, x1 = [float(value) for value in bbox]
            if y1 <= y0 or x1 <= x0:
                continue
            usable.append((glyph, y0, x0, y1, x1))
            heights.append(y1 - y0)
            widths.append(x1 - x0)
        if len(usable) != len(glyphs) or not usable:
            ordered = sorted(
                glyphs,
                key=lambda item: int((item.get("position_code") or {}).get("ordinal_position", 0)),
            )
            return [(0, ordered, "word_boundary_unresolved_no_complete_glyph_geometry")]

        median_height = sorted(heights)[len(heights) // 2]
        median_width = sorted(widths)[len(widths) // 2]
        line_tolerance = max(2.0, 0.55 * median_height)
        rows: list[list[tuple[Mapping[str, Any], float, float, float, float]]] = []
        for item in sorted(usable, key=lambda value: ((value[1] + value[3]) / 2.0, value[2])):
            center_y = (item[1] + item[3]) / 2.0
            target = None
            best_distance = float("inf")
            for row in rows:
                row_center = sum((member[1] + member[3]) / 2.0 for member in row) / len(row)
                distance = abs(center_y - row_center)
                if distance <= line_tolerance and distance < best_distance:
                    target = row
                    best_distance = distance
            if target is None:
                rows.append([item])
            else:
                target.append(item)

        segments: list[tuple[int, list[Mapping[str, Any]], str]] = []
        word_gap_threshold = max(2.0, 0.45 * median_width)
        for line_index, row in enumerate(rows):
            row.sort(key=lambda value: value[2])
            current: list[Mapping[str, Any]] = []
            previous_right: float | None = None
            word_index = 0
            for item in row:
                glyph, _, left, _, right = item
                gap = left - previous_right if previous_right is not None else 0.0
                if current and gap > word_gap_threshold:
                    segments.append((
                        line_index * 1000 + word_index,
                        current,
                        "candidate_word_boundary_from_clear_point1_horizontal_gap",
                    ))
                    word_index += 1
                    current = []
                current.append(glyph)
                previous_right = right
            if current:
                segments.append((
                    line_index * 1000 + word_index,
                    current,
                    "candidate_word_boundary_from_clear_point1_horizontal_gap",
                ))
        return segments

    def _visual_forms(self, hypotheses: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Beam-compose position-ordered grapheme alternatives into uncertainty-aware word-form candidates."""
        visual = hypotheses.get("visual", {}) if isinstance(hypotheses, Mapping) else {}
        glyphs = list(visual.get("form_hypotheses", []))
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for item in glyphs:
            region = str(item.get("region_id") or "default")
            grouped[region].append(item)
        forms: list[dict[str, Any]] = []
        for region, rows in grouped.items():
            segments = self._visual_word_segments(rows)
            for segment_index, (_, segment_glyphs, boundary_status) in enumerate(segments):
                beam: list[tuple[str, float, list[str]]] = [("", 1.0, [])]
                ordered_glyphs = sorted(
                    segment_glyphs,
                    key=lambda item: int((item.get("position_code") or {}).get("ordinal_position", 0)),
                )
                for glyph in ordered_glyphs:
                    options = list(glyph.get("candidates", []))
                    options = [
                        (str(option.get("grapheme", option.get("label", ""))),
                         max(0.0, float(option.get("probability", 0.0))))
                        for option in options
                        if str(option.get("grapheme", option.get("label", "")))
                    ]
                    total = sum(probability for _, probability in options)
                    if not options or total <= 0.0:
                        beam = []
                        break
                    options = [(grapheme, probability / total) for grapheme, probability in options]
                    expanded = [
                        (prefix + grapheme, score * probability, glyph_ids + [str(glyph.get("glyph_id", ""))])
                        for prefix, score, glyph_ids in beam
                        for grapheme, probability in options
                    ]
                    expanded.sort(key=lambda item: item[1], reverse=True)
                    beam = expanded[:self.top_k]
                for form, confidence, glyph_ids in beam:
                    forms.append({
                        "modality": "visual",
                        "form": unicodedata.normalize("NFC", form).casefold(),
                        "observed_form": form,
                        "confidence": confidence,
                        "region_id": region,
                        "word_segment_index": segment_index,
                        "glyph_ids": glyph_ids,
                        "boundary_status": boundary_status,
                        "evidence_origin": "point1",
                    })
        # Preserve the source's spatial/ordinal order; confidence ranks alternatives within a word.
        return forms[:self.max_form_candidates]

    def _auditory_forms(self, hypotheses: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Beam-compose time-ordered phoneme alternatives into candidate spoken word forms."""
        auditory = hypotheses.get("auditory", {}) if isinstance(hypotheses, Mapping) else {}
        segments = list(auditory.get("form_hypotheses", []))
        if not segments:
            return []
        beam: list[tuple[tuple[str, ...], float, list[Any]]] = [((), 1.0, [])]
        for segment in segments:
            options = list(segment.get("alternatives", []))
            if not options and segment.get("selected_symbol"):
                options = [{
                    "symbol": segment["selected_symbol"],
                    "probability": segment.get("confidence", 1.0),
                }]
            cleaned = [
                (str(option.get("symbol", "")), max(0.0, float(option.get("probability", 0.0))))
                for option in options
                if str(option.get("symbol", ""))
            ]
            total = sum(probability for _, probability in cleaned)
            if not cleaned or total <= 0.0:
                beam = []
                break
            cleaned = [(symbol, probability / total) for symbol, probability in cleaned]
            expanded = [
                (prefix + (symbol,), score * probability, spans + [segment.get("frame_span")])
                for prefix, score, spans in beam
                for symbol, probability in cleaned
            ]
            expanded.sort(key=lambda item: item[1], reverse=True)
            beam = expanded[:self.top_k]
        forms = [{
            "modality": "auditory",
            "phonemes": list(phonemes),
            "observed_form": " ".join(phonemes),
            "confidence": confidence,
            "frame_spans": spans,
            "boundary_status": "Point 1 acoustic frame segmentation estimate",
            "evidence_origin": "point1",
        } for phonemes, confidence, spans in beam]
        return forms[:self.max_form_candidates]

    def _sensory_update_forms(self, updates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Convert additional caller-supplied phoneme or orthographic evidence into update candidates."""
        output = []
        for update in updates:
            modality = str(update["modality"])
            confidence = min(1.0, max(0.0, float(update.get("confidence", 1.0))))
            if modality == "visual":
                form = unicodedata.normalize("NFC", str(update["form"]).strip()).casefold()
                output.append({
                    "modality": "visual", "form": form, "observed_form": str(update["form"]),
                    "confidence": confidence, "evidence_origin": "sensory_update",
                    "boundary_status": update.get("boundary_status", "caller_update"),
                })
            else:
                phonemes = tuple(str(item) for item in update["phonemes"])
                output.append({
                    "modality": "auditory", "phonemes": list(phonemes),
                    "observed_form": " ".join(phonemes), "confidence": confidence,
                    "evidence_origin": "sensory_update",
                    "boundary_status": update.get("boundary_status", "caller_update"),
                })
        return output

    def _run_lexical_engine(
        self,
        request: Point2Request,
        form: Mapping[str, Any],
        entries: Sequence[LexicalEntry],
        *,
        observed_word: str,
    ) -> Any:
        """Run the existing seven-stage lexical engine on an explicit candidate-entry subset."""
        engine = LexicalAccessProcessor(
            entries,
            morphemes=request.morphemes,
            morpheme_category_cues=request.morpheme_category_cues,
            morpheme_affix_categories=request.morpheme_affix_categories,
            config=self.config,
        )
        return engine.process(LexicalAccessRequest(
            sequence_id=f"{request.sequence_id}-{form['modality']}-{observed_word}",
            word=observed_word,
            language=request.language,
            context_features=request.context_features,
            contextual_category_priors=request.contextual_category_priors,
            source_reference=request.source_reference,
        ))

    def _auditory_entry_matches(
        self,
        request: Point2Request,
        form: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Match a spoken phoneme sequence against stored pronunciations without guessing missing ones."""
        phonemes = tuple(str(item) for item in form.get("phonemes", []))
        if not phonemes:
            return []
        frequencies = [
            max(0.0, float(entry.frequency) + float(getattr(entry, "exposure_count", 0.0)))
            for entry in request.lexicon
        ]
        maximum_frequency = max(frequencies, default=1.0)
        max_log_frequency = math.log1p(maximum_frequency) or 1.0
        rows = []
        for entry in request.lexicon:
            stored = tuple(str(item) for item in getattr(entry, "phonemes", ()))
            if not stored:
                continue
            distance = _edit_distance(phonemes, stored)
            similarity = max(0.0, 1.0 - distance / max(len(phonemes), len(stored), 1))
            if distance > self.config.max_edit_distance or similarity < self.config.minimum_similarity:
                continue
            frequency = max(0.0, float(entry.frequency) + float(getattr(entry, "exposure_count", 0.0)))
            frequency_strength = math.log1p(frequency) / max_log_frequency
            likelihood = math.exp(-2.0 * distance / max(len(phonemes), len(stored), 1))
            score = math.log(max(likelihood * float(form.get("confidence", 0.0)), 1e-12))
            rows.append({
                "entry": entry,
                "entry_id": str(entry.entry_id),
                "orthography": unicodedata.normalize("NFC", str(entry.orthography)).casefold(),
                "phoneme_distance": distance,
                "phoneme_similarity": similarity,
                "phoneme_likelihood": likelihood,
                "frequency_strength": frequency_strength,
                "score": score,
            })
        rows.sort(key=lambda item: item["score"], reverse=True)
        rows = rows[:self.config.max_candidates]
        probabilities = _softmax([item["score"] for item in rows], self.config.identity_temperature)
        for item, probability in zip(rows, probabilities):
            item["phoneme_posterior"] = probability
        return rows

    def _process_forms(
        self,
        request: Point2Request,
        forms: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Resolve visual or auditory form candidates with the explicit lexical inventory."""
        runs: list[dict[str, Any]] = []
        entries_by_id = {str(entry.entry_id): entry for entry in request.lexicon}
        for form in forms[:self.max_form_candidates]:
            modality = str(form["modality"])
            if modality == "visual":
                observed = str(form.get("form", "")).strip()
                if not observed:
                    continue
                result = self._run_lexical_engine(
                    request, form, request.lexicon, observed_word=observed
                )
                scores = {
                    str(item["entry_id"]): {
                        "form_match": float(item["orthographic_likelihood"]),
                        "form_similarity": float(item["orthographic_similarity"]),
                        "frequency_strength": float(item["frequency_strength"]),
                        "route_prior": 1.0,
                    }
                    for item in result.lexical_identity_candidates
                }
                runs.append({
                    "modality": modality, "form": dict(form), "result": result,
                    "identity_evidence": scores, "entries": entries_by_id,
                })
                continue

            matches = self._auditory_entry_matches(request, form)
            by_orthography: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for match in matches:
                by_orthography[match["orthography"]].append(match)
            for orthography, grouped in by_orthography.items():
                group_entries = [item["entry"] for item in grouped]
                result = self._run_lexical_engine(
                    request, form, group_entries, observed_word=orthography
                )
                scores = {
                    item["entry_id"]: {
                        "form_match": float(item["phoneme_likelihood"]),
                        "form_similarity": float(item["phoneme_similarity"]),
                        "phoneme_distance": int(item["phoneme_distance"]),
                        "frequency_strength": float(item["frequency_strength"]),
                        "route_prior": float(item["phoneme_posterior"]),
                    }
                    for item in grouped
                }
                runs.append({
                    "modality": modality, "form": dict(form), "result": result,
                    "identity_evidence": scores,
                    "entries": {str(item.entry_id): item for item in group_entries},
                })
        return runs

    def _activation_by_entry(self, result: Any) -> dict[str, float]:
        """Read final recurrent lexical activations for the supplied entry IDs."""
        if not result.incremental_activation:
            return {}
        frame = result.incremental_activation[-1]
        return {
            str(item.get("entry_id")): float(item.get("activation", 0.0))
            for item in frame.get("candidates", [])
        }

    def _morphology_by_entry(self, result: Any) -> dict[str, float]:
        """Associate whole-word and stem-supported morphology scores with their explicit lexical entries."""
        scores: dict[str, float] = defaultdict(float)
        for item in result.morphological_alternatives:
            ids = list(item.get("stem_entry_ids", []))
            if str(item.get("kind")) == "whole_word":
                hypothesis_id = str(item.get("hypothesis_id", ""))
                if hypothesis_id.startswith("whole:"):
                    ids.append(hypothesis_id.split(":", 1)[1])
            for entry_id in ids:
                scores[str(entry_id)] = max(scores[str(entry_id)], float(item.get("posterior", 0.0)))
        return dict(scores)

    def _candidate_records(self, request: Point2Request, runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Merge form evidence by lexical ID while keeping each modality's likelihood and alternatives visible."""
        per_modality: dict[str, dict[str, dict[str, Any]]] = {
            "auditory": {}, "visual": {},
        }
        low_level_traces: list[Mapping[str, Any]] = []
        for run in runs:
            result = run["result"]
            low_level_traces.extend(result.operation_trace)
            activations = self._activation_by_entry(result)
            morphology_scores = self._morphology_by_entry(result)
            meaning_by_entry: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for meaning in result.meaning_candidates:
                meaning_by_entry[str(meaning["entry_id"])].append(dict(meaning))
            for identity in result.lexical_identity_candidates:
                entry_id = str(identity["entry_id"])
                evidence = run["identity_evidence"].get(entry_id)
                if evidence is None:
                    continue
                form_confidence = float(run["form"].get("confidence", 0.0))
                form_match = float(evidence["form_match"])
                frequency_strength = float(evidence["frequency_strength"])
                meanings = meaning_by_entry.get(entry_id, [])
                context_fit = max(
                    (float(item.get("context_compatibility", 0.0)) for item in meanings),
                    default=0.0,
                )
                morphology_support = morphology_scores.get(entry_id, 0.0)
                activation = activations.get(entry_id, float(identity.get("posterior", 0.0)))
                reliability_adjusted_context = (
                    self.config.context_semantic_gain * (1.0 - form_confidence) * context_fit
                )
                score = (
                    math.log(max(form_confidence, 1e-12))
                    + math.log(max(form_match, 1e-12))
                    + math.log(max(float(evidence.get("route_prior", 1.0)), 1e-12))
                    + self.config.frequency_prior_gain * frequency_strength
                    + reliability_adjusted_context
                    + self.config.morphology_gain * morphology_support
                    + math.log(max(activation, 1e-8))
                )
                entry = run["entries"].get(entry_id)
                if entry is None:
                    continue
                sense_scores = [
                    math.log(max(float(item.get("prior", 1.0)), 1e-12))
                    + self.config.context_semantic_gain * float(item.get("context_compatibility", 0.0))
                    for item in meanings
                ]
                sense_probabilities = _softmax(sense_scores, self.config.semantic_temperature)
                senses = [
                    {**item, "posterior_within_entry": probability}
                    for item, probability in zip(meanings, sense_probabilities)
                ]
                candidate = {
                    "entry_id": entry_id,
                    "canonical_word_form": str(entry.orthography),
                    "language": str(getattr(entry, "language", request.language)),
                    "lexical_category": str(getattr(entry, "lexical_category", "unknown")),
                    "frequency": float(getattr(entry, "frequency", 0.0)),
                    "exposure_count": float(getattr(entry, "exposure_count", 0.0)),
                    "familiarity_strength": frequency_strength,
                    "phonological_form": list(getattr(entry, "phonemes", ())),
                    "form_match_score": form_match,
                    "form_reliability": form_confidence,
                    "context_compatibility": context_fit,
                    "morphology_support": morphology_support,
                    "recurrent_activation": activation,
                    "lexical_identity_engine_posterior": float(identity.get("posterior", 0.0)),
                    "meaning_candidates": senses,
                    "morphological_alternatives": [
                        dict(item) for item in result.morphological_alternatives
                        if entry_id in item.get("stem_entry_ids", [])
                        or str(item.get("hypothesis_id", "")).endswith(":" + entry_id)
                        or item.get("kind") == "whole_word" and item.get("stem") == str(entry.orthography).casefold()
                    ],
                    "stem_affix_retrieval": [
                        dict(item) for item in result.stem_affix_retrieval
                        if entry_id in item.get("stem_entry_ids", [])
                    ],
                    "lexical_properties": dict(request.lexical_properties.get(entry_id, {})),
                    "verified_by": str(getattr(entry, "verified_by", "unspecified")),
                    "source_reference": str(getattr(entry, "source_reference", "unspecified")),
                }
                route_evidence = {
                    "modality": run["modality"],
                    "observed_form": run["form"].get("observed_form"),
                    "confidence": form_confidence,
                    "form_match_score": form_match,
                    "form_similarity": float(evidence["form_similarity"]),
                    "evidence_origin": run["form"].get("evidence_origin", "point1"),
                }
                existing = per_modality[run["modality"]].get(entry_id)
                if existing is None or score > existing["score"]:
                    candidate["route_evidence"] = [route_evidence]
                    candidate["score"] = score
                    per_modality[run["modality"]][entry_id] = candidate
                else:
                    existing["route_evidence"].append(route_evidence)

        modality_lists: dict[str, list[dict[str, Any]]] = {}
        combined: dict[str, dict[str, Any]] = {}
        for modality, rows in per_modality.items():
            ordered = sorted(rows.values(), key=lambda item: item["score"], reverse=True)
            probabilities = _softmax([item["score"] for item in ordered])
            modality_lists[modality] = []
            for item, probability in zip(ordered, probabilities):
                item["modality_posterior"] = probability
                item["modality"] = modality
                modality_lists[modality].append(item)
                previous = combined.get(item["entry_id"])
                if previous is None:
                    combined[item["entry_id"]] = {
                        **item,
                        "modality_evidence": {modality: probability},
                        "route_evidence": list(item["route_evidence"]),
                    }
                else:
                    previous["modality_evidence"][modality] = probability
                    previous["route_evidence"].extend(item["route_evidence"])
                    if item["score"] > previous["score"]:
                        preserve = previous["modality_evidence"]
                        routes = previous["route_evidence"]
                        combined[item["entry_id"]] = {
                            **item, "modality_evidence": preserve, "route_evidence": routes,
                        }
        combined_rows = sorted(combined.values(), key=lambda item: sum(item["modality_evidence"].values()), reverse=True)
        total = sum(sum(item["modality_evidence"].values()) for item in combined_rows)
        for item in combined_rows:
            raw = sum(item["modality_evidence"].values())
            item["posterior"] = raw / total if total > 0.0 else 0.0
            item.pop("score", None)
        return {
            "by_modality": modality_lists,
            "combined": combined_rows,
            "low_level_operation_trace": low_level_traces,
        }

    def _semantic_feature_pattern(self, candidates: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        """Pool sense-specific feature activations by lexical candidate confidence without making a proposition."""
        combined: dict[str, float] = defaultdict(float)
        for candidate in candidates:
            candidate_probability = float(candidate.get("posterior", 0.0))
            senses = candidate.get("meaning_candidates", [])
            for sense in senses:
                sense_probability = float(sense.get("posterior_within_entry", 0.0))
                for feature, value in sense.get("features", {}).items():
                    combined[str(feature)] += candidate_probability * sense_probability * float(value)
        return {key: min(1.0, value) for key, value in combined.items()}

    def _sense_candidates(self, candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Keep each sense tied to its lexical identity so homonyms and polysemy remain distinguishable."""
        senses = []
        for candidate in candidates:
            for sense in candidate.get("meaning_candidates", []):
                senses.append({
                    "entry_id": candidate["entry_id"],
                    "word_form": candidate["canonical_word_form"],
                    "meaning_id": sense.get("meaning_id"),
                    "gloss": sense.get("gloss", ""),
                    "features": dict(sense.get("features", {})),
                    "posterior": float(candidate.get("posterior", 0.0))
                    * float(sense.get("posterior_within_entry", 0.0)),
                    "context_compatibility": float(sense.get("context_compatibility", 0.0)),
                    "source_reference": sense.get("source_reference", "unspecified"),
                })
        senses.sort(key=lambda item: item["posterior"], reverse=True)
        return senses

    def _lexical_properties(self, candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Expose lexical categories, inflectional features and stored argument/complement data without parsing roles."""
        entries = []
        for candidate in candidates:
            entries.append({
                "entry_id": candidate["entry_id"],
                "canonical_word_form": candidate["canonical_word_form"],
                "lexical_category": candidate["lexical_category"],
                "morphological_features": candidate["lexical_properties"].get("morphological_features", {}),
                "grammatical_features": candidate["lexical_properties"].get("grammatical_features", {}),
                "argument_structure": candidate["lexical_properties"].get("argument_structure", {}),
                "complement_properties": candidate["lexical_properties"].get("complement_properties", {}),
                "posterior": candidate.get("posterior", 0.0),
                "uncertainty": "category_and_properties_are_lexical_candidates_not_sentence_parse",
            })
        return {
            "candidate_properties": entries,
            "sentence_arguments_assigned": False,
            "predicate_argument_structure_built": False,
        }

    def _ambiguities(
        self,
        candidates: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Classify same-form lexical identities and within-entry multiple senses without prematurely resolving them."""
        ambiguities: list[dict[str, Any]] = []
        by_form: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        by_entry: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            by_form[str(candidate["canonical_word_form"]).casefold()].append(candidate)
            by_entry[str(candidate["entry_id"])].extend(candidate.get("meaning_candidates", []))
        for form, entries in by_form.items():
            identities = sorted({str(item["entry_id"]) for item in entries})
            if len(identities) > 1:
                ambiguities.append({
                    "type": "homonym_or_lexical_identity_ambiguity",
                    "form": form,
                    "entry_ids": identities,
                    "resolution": "preserve_distinct_lexical_entries",
                })
        for entry_id, entry_senses in by_entry.items():
            meaning_ids = sorted({str(item.get("meaning_id")) for item in entry_senses})
            if len(meaning_ids) > 1:
                ambiguities.append({
                    "type": "polysemy_or_sense_ambiguity",
                    "entry_id": entry_id,
                    "meaning_ids": meaning_ids,
                    "resolution": "retain_sense_candidates_for_later_sentence_and_discourse_evidence",
                })
        if candidates and len(candidates) > 1:
            margin = float(candidates[0].get("posterior", 0.0)) - float(candidates[1].get("posterior", 0.0))
            if margin < self.config.ambiguity_margin:
                ambiguities.append({
                    "type": "lexical_candidate_competition",
                    "top_two_margin": margin,
                    "resolution": "retain_ranked_alternatives",
                })
        return ambiguities

    def _biological_trace(self, runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Aggregate the seven-population LIF activity generated by BLA operations into 12 source-level steps."""
        low_level_by_id: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        any_enabled = False
        for run in runs:
            biological = run["result"].biological
            if not biological.get("enabled"):
                continue
            any_enabled = True
            for item in biological.get("operation_trace", []):
                identifier = str(item.get("operation_id", ""))
                try:
                    number = int(identifier.rsplit("-", 1)[1])
                except (IndexError, ValueError):
                    continue
                low_level_by_id[number].append(item)
        trace = []
        for step, spec in enumerate(POINT2_STEP_SPECS, start=1):
            indices = _POINT2_TO_BLA[step]
            selected = [item for index in indices for item in low_level_by_id.get(index, [])]
            if not any_enabled:
                status = "biological_simulation_disabled"
            elif not selected:
                status = "no_lexical_engine_run_for_this_step"
            else:
                status = "simulated_reference_not_measured_human_data"
            populations: dict[str, list[float]] = defaultdict(list)
            for item in selected:
                for name, activity in item.get("population_activity", {}).items():
                    populations[str(name)].append(float(activity))
            trace.append({
                "operation_id": spec["operation_id"],
                "status": status,
                "underlying_BLA_operation_ids": [f"BLA-{index:03d}" for index in indices],
                "mean_population_activity": {
                    name: sum(values) / len(values) for name, values in populations.items() if values
                },
                "population_names": sorted(populations),
                "biological_scope": "generic_current_based_LIF_reference_not_anatomically_fitted_or_measured",
            })
        return trace

    def process(self, request: Point2Request) -> Point2Result:
        """Execute all 12 Point 2 operations and return uncertainty-aware lexical-semantic candidates."""
        if not isinstance(request, Point2Request):
            raise TypeError("request must be a Point2Request")
        visual_forms = self._visual_forms(request.form_hypotheses)
        auditory_forms = self._auditory_forms(request.form_hypotheses)
        initial_forms = visual_forms + auditory_forms
        initial_runs = self._process_forms(request, initial_forms)
        initial_state = self._candidate_records(request, initial_runs)
        update_forms = self._sensory_update_forms(request.sensory_updates)
        update_runs = self._process_forms(request, update_forms) if update_forms else []
        final_runs = initial_runs + update_runs
        final_state = self._candidate_records(request, final_runs)
        candidates = list(final_state["combined"])
        candidates.sort(key=lambda item: float(item.get("posterior", 0.0)), reverse=True)
        meanings = self._sense_candidates(candidates)
        semantic_features = self._semantic_feature_pattern(candidates)
        properties = self._lexical_properties(candidates)
        ambiguities = self._ambiguities(candidates)

        top_probability = float(candidates[0]["posterior"]) if candidates else 0.0
        second_probability = float(candidates[1]["posterior"]) if len(candidates) > 1 else 0.0
        confidence_margin = top_probability - second_probability
        selected = None
        if (
            candidates
            and top_probability >= self.config.confidence_threshold
            and (len(candidates) == 1 or confidence_margin >= self.config.ambiguity_margin)
        ):
            selected = dict(candidates[0])

        modality_routes = {
            "auditory_active": bool(initial_state["by_modality"]["auditory"]),
            "visual_active": bool(initial_state["by_modality"]["visual"]),
            "auditory_candidate_count": len(initial_state["by_modality"]["auditory"]),
            "visual_candidate_count": len(initial_state["by_modality"]["visual"]),
            "shared_lexical_inventory": "caller-supplied entries with explicit phonemes and/or orthography",
            "cross_modal_convergence": "same lexical entry IDs may collect independent route evidence",
        }
        sensory_feedback = {
            "update_count": len(update_forms),
            "updates_processed": len(update_runs),
            "new_evidence_received": bool(update_forms),
            "existing_candidates_re_scored": bool(update_forms),
            "previous_candidate_count": len(initial_state["combined"]),
            "updated_candidate_count": len(candidates),
            "candidate_changes": [
                {
                    "entry_id": item["entry_id"],
                    "previous_posterior": next(
                        (float(old["posterior"]) for old in initial_state["combined"]
                         if old["entry_id"] == item["entry_id"]), 0.0
                    ),
                    "updated_posterior": float(item["posterior"]),
                }
                for item in candidates
            ],
            "feedback_to_point1": (
                "request_revised_or_less_committed_form_hypothesis"
                if not candidates or (candidates and float(candidates[0].get("form_match_score", 0.0)) < 0.55)
                else "no_additional_form_feedback_required"
            ),
            "feedback_is_request_metadata_not_an_automatic_neural_or_external_action": True,
        }

        functional_outputs = (
            {
                "input_modalities": {"visual": len(visual_forms), "auditory": len(auditory_forms)},
                "form_hypothesis_count": len(initial_forms),
                "original_hypotheses_preserved": True,
                "uncertainty_preserved": True,
                "candidate_boundaries_retained": True,
            },
            modality_routes,
            {
                "form_candidates": initial_forms,
                "candidate_entry_ids": sorted({item["entry_id"] for rows in initial_state["by_modality"].values() for item in rows}),
                "candidate_counts_by_modality": {
                    key: len(value) for key, value in initial_state["by_modality"].items()
                },
                "partial_match_policy": "bounded sequence/edit distance and phoneme/orthographic alternatives",
            },
            {
                "candidate_priors": [
                    {
                        "entry_id": item["entry_id"],
                        "frequency": item["frequency"],
                        "exposure_count": item["exposure_count"],
                        "familiarity_strength": item["familiarity_strength"],
                    }
                    for item in candidates
                ],
                "frequency_and_context_are_separate": True,
            },
            {
                "context_features_used": dict(request.context_features),
                "contextual_category_priors": dict(request.contextual_category_priors),
                "context_compatibility": {
                    item["entry_id"]: item["context_compatibility"] for item in candidates
                },
                "reliability_dependent_context_weight": "context contribution is scaled by (1 - form confidence)",
                "broad_pragmatics_performed": False,
            },
            {
                "morphological_alternatives": [
                    {"entry_id": item["entry_id"], "alternatives": item["morphological_alternatives"],
                     "stem_affix_retrieval": item["stem_affix_retrieval"]}
                    for item in candidates if item["morphological_alternatives"] or item["stem_affix_retrieval"]
                ],
                "whole_word_route_preserved": True,
                "only_supplied_morphemes_used": True,
            },
            {
                "ranked_candidates": [
                    {
                        "entry_id": item["entry_id"],
                        "posterior": item["posterior"],
                        "recurrent_activation": item["recurrent_activation"],
                        "modalities": sorted(item["modality_evidence"]),
                    }
                    for item in candidates
                ],
                "top_two_margin": confidence_margin,
                "winner_take_all_forced": False,
                "ambiguity_retained": selected is None and len(candidates) > 1,
            },
            {
                "word_level_semantic_features": semantic_features,
                "meaning_candidates": meanings,
                "feature_source": "explicit lexical meaning records only",
                "sentence_proposition_composed": False,
            },
            properties,
            {
                "homonym_and_polysemy_analysis": ambiguities,
                "sense_candidates": meanings,
                "sense_disambiguation_scope": "limited caller-supplied local features only",
            },
            sensory_feedback,
            {
                "selected_or_unresolved_entry": selected,
                "ranked_candidates": candidates,
                "word_level_semantic_features": semantic_features,
                "lexical_properties": properties,
                "sense_candidates": meanings,
                "confidence": top_probability,
                "unresolved_ambiguities": ambiguities,
                "handoff_target": "Point 3 syntactic parsing",
                "sentence_structure_built": False,
                "predicate_argument_structure_built": False,
            },
        )
        functional_trace = []
        for specification, output in zip(POINT2_STEP_SPECS, functional_outputs, strict=True):
            functional_trace.append({
                **specification,
                "status": "completed",
                "functional_result": to_jsonable(output),
            })

        low_level_trace = list(final_state["low_level_operation_trace"])
        biological_trace = self._biological_trace(final_runs)
        diagnostics = {
            "valid": len(functional_trace) == 12,
            "source_step_count": len(functional_trace),
            "lexical_candidate_count": len(candidates),
            "auditory_candidate_count": len(final_state["by_modality"]["auditory"]),
            "visual_candidate_count": len(final_state["by_modality"]["visual"]),
            "meaning_candidate_count": len(meanings),
            "sense_ambiguity_count": len(ambiguities),
            "selected_candidate": selected is not None,
            "low_level_BLA_operation_count": len(low_level_trace),
            "biological_simulation_enabled": any(
                run["result"].biological.get("enabled") for run in final_runs
            ),
            "biological_reference_numerically_valid": all(
                run["result"].biological.get("numerically_valid", True) for run in final_runs
            ),
        }
        return Point2Result(
            sequence_id=request.sequence_id,
            modality_routes=modality_routes,
            form_candidates=initial_forms,
            lexical_candidates=candidates,
            selected_candidate=selected,
            meaning_candidates=meanings,
            word_level_semantic_features=semantic_features,
            lexical_properties=properties,
            sense_candidates=meanings,
            unresolved_ambiguities=ambiguities,
            confidence=top_probability,
            functional_trace=functional_trace,
            biological_trace=biological_trace,
            low_level_operation_trace=low_level_trace,
            diagnostics=diagnostics,
            annotation_report=self.annotation_report,
        )
