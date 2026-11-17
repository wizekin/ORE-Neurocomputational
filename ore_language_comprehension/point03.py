"""Source-ordered Point 3 syntactic parsing over ORE's established syntax engine.

The adapter accepts uncertainty-aware lexical items from Point 2, invokes the existing
nine-stage/58-operation syntax processor, and returns only grammatical structure. Semantic
composition and thematic-role interpretation are deliberately excluded from its public output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from ore_sensory_perception.contracts import to_jsonable
from ore_syntax import SyntaxConfig, SyntaxProcessor, SyntaxRequest


POINT3_STEP_SPECS: tuple[dict[str, Any], ...] = (
    {
        "operation_id": "LC-P03-01",
        "name": "Receive and align the next lexical item with the active parse",
        "objective": "Preserve lexical alternatives, word order, inherited confidence, morphology, and boundary evidence at each token position.",
        "mathematical_operations": ("joint support J_i(T,k)=P(T)*q_i(k)", "token-index alignment", "reliability-weighted lexical evidence"),
        "biological_reference": "Sequential lexical input becomes available to distributed temporal and frontal language populations; an explicit software queue is only an engineering representation.",
    },
    {
        "operation_id": "LC-P03-02",
        "name": "Estimate grammatical-category alternatives",
        "objective": "Combine Point 2 lexical categories with morphology and local grammatical evidence while retaining category ambiguity.",
        "mathematical_operations": ("P(c|k,m,C) proportional to P(m|c,k)*P(c|k,C)", "log category scores", "temperature softmax and top-two margin"),
        "biological_reference": "Temporal lexical representations and interacting frontal-temporal language populations support graded category activation; a literal cortical softmax is not assumed.",
    },
    {
        "operation_id": "LC-P03-03",
        "name": "Generate expectations for upcoming grammatical material",
        "objective": "Use incomplete phrases and active heads to estimate graded next-category and next-phrase expectations.",
        "mathematical_operations": ("P(c_next|S_i)=sum_T P(c_next|T)P(T|S_i)", "surprisal=-log2 P(c_i|S_(i-1))"),
        "biological_reference": "Predictive structure-building and syntactic surprisal are associated with fronto-temporal language activity, without implying a single neural probability calculator.",
    },
    {
        "operation_id": "LC-P03-04",
        "name": "Open and extend hierarchical phrase constituents",
        "objective": "Group words into nested NP, VP, PP, adjective, adverb, and clause-level structures, retaining licensed alternatives and incomplete spans.",
        "mathematical_operations": ("chart constituents C[A,i,j]", "Score(C)=Score(children)+Score(rule)+Score(boundary)", "bounded parse-forest expansion"),
        "biological_reference": "Posterior temporal phrase grouping and interacting frontal-temporal structure building are candidate network-level analogies.",
    },
    {
        "operation_id": "LC-P03-05",
        "name": "Identify phrase heads and construct grammatical dependencies",
        "objective": "Build directed head-dependent relations for subjects, objects, modifiers, determiners, complements, and adjuncts without assigning conceptual event roles.",
        "mathematical_operations": ("edge=(head,dependent,relation,confidence)", "edge score=sum_j w_j*f_j", "single-head tree and cycle constraints"),
        "biological_reference": "Distributed temporal and inferior-frontal language populations may contribute differentiated phrase and dependency sensitivities; no region is treated as a literal parser.",
    },
    {
        "operation_id": "LC-P03-06",
        "name": "Check supplied morphosyntactic feature compatibility",
        "objective": "Compare explicitly supplied number, person, gender, case, and noun-class features on relevant grammatical links; missing values remain unknown.",
        "mathematical_operations": ("Compat(F_h,F_d) in [0,1]", "E_agreement=sum_r lambda_r*mismatch_r", "Score_new=Score_old-E_agreement"),
        "biological_reference": "Agreement-sensitive processing is distributed across the language network; EEG effects vary and are not one-to-one labels for a neural subroutine.",
    },
    {
        "operation_id": "LC-P03-07",
        "name": "Combine order, case, function-word, and boundary cues",
        "objective": "Use positional patterns, closed-class words, caller-supplied inflection/case, punctuation, and optional speech prosody as graded evidence.",
        "mathematical_operations": ("Score_cues(T)=sum_j alpha_j*log P(C_j|T)", "reliability-weighted cue fusion", "boundary evidence aggregation"),
        "biological_reference": "Syntactic networks interact with auditory prosody and visual punctuation cues; spoken and written boundary signals are not treated as identical.",
    },
    {
        "operation_id": "LC-P03-08",
        "name": "Maintain unresolved dependencies and incomplete structures",
        "objective": "Keep remote fillers, open grammatical slots, and incomplete phrase links accessible while tracking retrieval uncertainty.",
        "mathematical_operations": ("A_d(t)=A0*exp(-lambda*Delta_t)*CueMatch(d,t)", "A'_d=A_d-gamma*sum_j Similarity(d,d_j)"),
        "biological_reference": "Distributed recurrent language activity and memory/control support can maintain dependencies; a literal decaying stack is not claimed.",
    },
    {
        "operation_id": "LC-P03-09",
        "name": "Rank structural alternatives without premature commitment",
        "objective": "Retain top constituency parses and competing attachments, using relative scores and confidence margins rather than forced certainty.",
        "mathematical_operations": ("LogScore(T)=prior+lexical+morphology+order+boundary-penalties", "P(T|input)=softmax(LogScore(T))", "bounded top-B alternatives"),
        "biological_reference": "Competing syntactic interpretations and recurrent fronto-temporal evaluation are plausible; a literal top-B beam is an engineering approximation.",
    },
    {
        "operation_id": "LC-P03-10",
        "name": "Detect structural conflict and unexpected grammatical evidence",
        "objective": "Flag low-probability category continuations, agreement mismatches, and dependency/constituent conflicts while distinguishing surprise from ungrammaticality.",
        "mathematical_operations": ("Conflict_i=-log P(W_i|preferred_parse,grammar)", "flag iff Conflict_i>theta_conflict", "category surprisal in bits"),
        "biological_reference": "Syntactic expectancy and reanalysis can evoke fronto-temporal and ERP responses; ELAN/LAN/P600 are variable and not unique operation markers.",
    },
    {
        "operation_id": "LC-P03-11",
        "name": "Reanalyse and repair an earlier structural decision",
        "objective": "Use the syntax engine's recurrent ambiguity pass to revise category and structural preferences, preserving changed and unresolved alternatives.",
        "mathematical_operations": ("DeltaScore=LogScore(T_new)-LogScore(T_old)", "revised score=LogScore(T_new)-rho*RevisionCost", "posterior margin and entropy"),
        "biological_reference": "Re-evaluation and repair likely involve interacting language and control processes; no single dedicated repair centre is assumed.",
    },
    {
        "operation_id": "LC-P03-12",
        "name": "Close completed constituents and consolidate the syntactic handoff",
        "objective": "Return token categories, phrase groupings, grammatical dependencies, ranked trees, open dependencies, and confidence for semantic composition.",
        "mathematical_operations": ("Complete(C) in [0,1]", "ParseOutput={tree,alternatives,categories,phrases,arcs,features,open_links,confidence}"),
        "biological_reference": "A sentence-level grammatical representation is made available to later meaning-composition processes through distributed fronto-temporal language circuitry.",
    },
)


_POINT3_TO_SYN: dict[int, tuple[str, ...]] = {
    1: ("SYN-001", "SYN-002"),
    2: tuple(f"SYN-{index:03d}" for index in range(2, 8)),
    3: ("SYN-004", "SYN-005"),
    4: tuple(f"SYN-{index:03d}" for index in range(8, 14))
       + tuple(f"SYN-{index:03d}" for index in range(21, 27)),
    5: tuple(f"SYN-{index:03d}" for index in range(14, 21)),
    6: ("SYN-015", "SYN-017", "SYN-018")
       + tuple(f"SYN-{index:03d}" for index in range(32, 38)),
    7: ("SYN-001", "SYN-004", "SYN-005", "SYN-015", "SYN-016", "SYN-017", "SYN-018"),
    8: tuple(f"SYN-{index:03d}" for index in range(51, 59)),
    9: tuple(f"SYN-{index:03d}" for index in range(44, 51)),
    10: tuple(f"SYN-{index:03d}" for index in range(44, 51)),
    11: tuple(f"SYN-{index:03d}" for index in range(47, 51)),
    12: tuple(f"SYN-{index:03d}" for index in range(27, 32)),
}

_CATEGORY_ALIASES = {
    "N": "NOUN", "NOUN": "NOUN", "PROPER_NOUN": "PROPN", "PROPN": "PROPN",
    "V": "VERB", "VERB": "VERB", "AUX": "AUX", "AUXILIARY": "AUX",
    "ADJ": "ADJ", "ADJECTIVE": "ADJ", "ADV": "ADV", "ADVERB": "ADV",
    "DET": "DET", "DETERMINER": "DET", "PRON": "PRON", "PRONOUN": "PRON",
    "ADP": "ADP", "PREP": "ADP", "PREPOSITION": "ADP",
    "CCONJ": "CCONJ", "COORDINATING_CONJUNCTION": "CCONJ",
    "SCONJ": "SCONJ", "SUBORDINATING_CONJUNCTION": "SCONJ",
    "NUM": "NUM", "NUMBER": "NUM", "PUNCT": "PUNCT", "PART": "PART",
    "INTJ": "INTJ", "X": "X",
}


def _category_name(value: Any) -> str:
    """Normalize lexical category names to the inventory used by the ORE syntax engine."""
    key = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    return _CATEGORY_ALIASES.get(key, key if key in {"NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV", "DET", "ADP", "PRON", "CCONJ", "SCONJ", "NUM", "PUNCT", "PART", "INTJ", "X"} else "X")


def _category_distribution(item: Mapping[str, Any]) -> dict[str, float]:
    """Collect category probabilities from direct labels or Point 2 lexical candidates."""
    totals: dict[str, float] = {}
    raw = item.get("category_candidates", item.get("pos_candidates", {}))
    if isinstance(raw, Mapping):
        for label, weight in raw.items():
            category = _category_name(label)
            value = max(0.0, float(weight))
            totals[category] = totals.get(category, 0.0) + value
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for candidate in raw:
            if not isinstance(candidate, Mapping):
                continue
            label = candidate.get("category", candidate.get("tag", candidate.get("lexical_category")))
            weight = candidate.get("posterior", candidate.get("probability", candidate.get("confidence", 0.0)))
            category = _category_name(label)
            totals[category] = totals.get(category, 0.0) + max(0.0, float(weight))
    candidates = item.get("lexical_candidates", ())
    if not totals and isinstance(candidates, Sequence):
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            category = _category_name(candidate.get("lexical_category", "X"))
            totals[category] = totals.get(category, 0.0) + max(0.0, float(candidate.get("posterior", 0.0)))
    direct = item.get("category", item.get("lexical_category"))
    if not totals and direct:
        totals[_category_name(direct)] = 1.0
    total = sum(totals.values())
    return {key: value / total for key, value in totals.items()} if total > 0.0 else {}


def _morphology_record(item: Mapping[str, Any]) -> dict[str, Any]:
    """Merge only caller-supplied lexical morphology and grammatical features."""
    output: dict[str, Any] = {}
    for key in ("morphology", "morphological_features", "grammatical_features"):
        value = item.get(key)
        if isinstance(value, Mapping):
            output.update(dict(value))
    candidates = item.get("lexical_candidates", ())
    if isinstance(candidates, Sequence):
        ranked = sorted(
            (row for row in candidates if isinstance(row, Mapping)),
            key=lambda row: float(row.get("posterior", 0.0)),
            reverse=True,
        )
        if ranked:
            properties = ranked[0].get("lexical_properties", {})
            if isinstance(properties, Mapping):
                for key in ("morphological_features", "grammatical_features"):
                    value = properties.get(key)
                    if isinstance(value, Mapping):
                        for feature, setting in value.items():
                            output.setdefault(str(feature), setting)
    return output


def _softmax(scores: Sequence[float]) -> list[float]:
    """Normalize arbitrary finite parse scores with a stable softmax calculation."""
    if not scores:
        return []
    finite = [float(score) if math.isfinite(float(score)) else -1e9 for score in scores]
    peak = max(finite)
    weights = [math.exp(max(-100.0, score - peak)) for score in finite]
    total = sum(weights)
    return [weight / total for weight in weights] if total > 0.0 else [1.0 / len(scores)] * len(scores)


@dataclass(frozen=True, slots=True)
class Point3Request:
    """Represent an ordered lexical stream and its optional morphology and boundary evidence.

    Each lexical item accepts a 'form' or 'token', category alternatives or Point 2 candidates,
    optional morphology, boundary cues, and inherited confidence. No spelling, pronunciation,
    grammatical feature, or semantic role is inferred from missing inventory data.
    """

    sequence_id: str
    lexical_items: Sequence[Mapping[str, Any]]
    language: str = "en"
    morphology_features: Mapping[int, Mapping[str, Any]] = field(default_factory=dict)
    boundary_cues: Mapping[int, Mapping[str, Any]] = field(default_factory=dict)
    source_reference: str | None = None

    def __post_init__(self) -> None:
        """Validate sequence identity, lexical forms, candidate weights, and token-indexed evidence."""
        if not self.sequence_id.strip():
            raise ValueError("sequence_id must be non-empty")
        if not self.lexical_items:
            raise ValueError("lexical_items must contain at least one recognized word/token")
        normalized = []
        for index, raw in enumerate(self.lexical_items):
            if not isinstance(raw, Mapping):
                raise TypeError(f"lexical_items[{index}] must be a mapping")
            item = dict(raw)
            form = item.get("form", item.get("token", item.get("canonical_word_form")))
            if not isinstance(form, str) or not form.strip():
                raise ValueError(f"lexical_items[{index}] needs a non-empty form/token")
            confidence = float(item.get("confidence", item.get("form_reliability", 1.0)))
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("lexical item confidence must be finite and in [0, 1]")
            distribution = _category_distribution(item)
            for label, value in distribution.items():
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"invalid category probability for token {index}: {label}")
            item["form"] = form.strip()
            item["confidence"] = confidence
            item["category_candidates"] = distribution
            item["morphology"] = _morphology_record(item)
            normalized.append(item)
        for field_name, evidence in (
            ("morphology_features", self.morphology_features),
            ("boundary_cues", self.boundary_cues),
        ):
            for index, values in evidence.items():
                if not isinstance(index, int) or index < 0 or index >= len(normalized):
                    raise ValueError(f"{field_name} indices must identify an existing token")
                if not isinstance(values, Mapping):
                    raise TypeError(f"{field_name}[{index}] must be a mapping")
        object.__setattr__(self, "lexical_items", tuple(normalized))
        object.__setattr__(self, "morphology_features", {
            int(index): dict(value) for index, value in self.morphology_features.items()
        })
        object.__setattr__(self, "boundary_cues", {
            int(index): dict(value) for index, value in self.boundary_cues.items()
        })

    @classmethod
    def from_point2_results(
        cls,
        sequence_id: str,
        point2_results: Sequence[Any],
        **kwargs: Any,
    ) -> "Point3Request":
        """Build an ordered lexical stream from one Point2Result per word in the sentence."""
        if not point2_results:
            raise ValueError("point2_results must contain one Point 2 result for each token")
        items: list[dict[str, Any]] = []
        for result in point2_results:
            data = result.to_dict() if hasattr(result, "to_dict") else result
            if not isinstance(data, Mapping):
                raise TypeError("each Point 2 result must be a result object or serialized mapping")
            candidates = list(data.get("lexical_candidates", ()))
            selected = data.get("selected_candidate")
            best = selected if isinstance(selected, Mapping) else (
                candidates[0] if candidates else {}
            )
            form = best.get("canonical_word_form")
            if not form:
                form_candidates = data.get("form_candidates", ())
                form = next(
                    (row.get("form") for row in form_candidates
                     if isinstance(row, Mapping) and row.get("form")),
                    None,
                )
            if not form:
                raise ValueError("Point 2 result has no recognized lexical form to pass to syntax")
            items.append({
                "form": str(form),
                "lexical_candidates": candidates,
                "category_candidates": [
                    {"category": _category_name(row.get("lexical_category", "X")),
                     "posterior": float(row.get("posterior", 0.0))}
                    for row in candidates
                ],
                "confidence": float(best.get("form_reliability", data.get("confidence", 0.0))),
                "morphology": {
                    **dict(best.get("lexical_properties", {}).get("morphological_features", {})),
                    **dict(best.get("lexical_properties", {}).get("grammatical_features", {})),
                } if isinstance(best.get("lexical_properties", {}), Mapping) else {},
                "source_reference": best.get("source_reference"),
            })
        return cls(sequence_id=sequence_id, lexical_items=items, **kwargs)


@dataclass(slots=True)
class Point3Result:
    """Expose syntactic categories, constituents, dependencies, alternatives, and biological traces."""

    sequence_id: str
    tokens: Sequence[str]
    lexical_item_records: Sequence[Mapping[str, Any]]
    incremental_updates: Sequence[Mapping[str, Any]]
    pos_candidates: Sequence[Sequence[Mapping[str, Any]]]
    selected_pos: Sequence[Mapping[str, Any]]
    constituent_groups: Sequence[Mapping[str, Any]]
    dependency_relations: Sequence[Mapping[str, Any]]
    constituency_alternatives: Sequence[Mapping[str, Any]]
    constituency_tree: Mapping[str, Any]
    syntax_hierarchy: Mapping[str, Any]
    unresolved_dependency_slots: Sequence[Mapping[str, Any]]
    agreement_checks: Sequence[Mapping[str, Any]]
    cue_evidence: Mapping[str, Any]
    active_expectations: Sequence[Mapping[str, Any]]
    structural_conflicts: Sequence[Mapping[str, Any]]
    revision_events: Sequence[Mapping[str, Any]]
    confidence: float
    functional_trace: Sequence[Mapping[str, Any]]
    biological_trace: Sequence[Mapping[str, Any]]
    biological_summary: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    annotation_report: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize Point 3 output while asserting that sentence meaning remains a later stage."""
        return to_jsonable({
            "schema": "ore.language_comprehension.point03.v1",
            "sequence_id": self.sequence_id,
            "tokens": self.tokens,
            "lexical_item_records": self.lexical_item_records,
            "incremental_updates": self.incremental_updates,
            "pos_candidates": self.pos_candidates,
            "selected_pos": self.selected_pos,
            "constituent_groups": self.constituent_groups,
            "dependency_relations": self.dependency_relations,
            "constituency_alternatives": self.constituency_alternatives,
            "constituency_tree": self.constituency_tree,
            "syntax_hierarchy": self.syntax_hierarchy,
            "unresolved_dependency_slots": self.unresolved_dependency_slots,
            "agreement_checks": self.agreement_checks,
            "cue_evidence": self.cue_evidence,
            "active_expectations": self.active_expectations,
            "structural_conflicts": self.structural_conflicts,
            "revision_events": self.revision_events,
            "confidence": self.confidence,
            "functional_trace": self.functional_trace,
            "biological_trace": self.biological_trace,
            "biological_summary": self.biological_summary,
            "diagnostics": self.diagnostics,
            "annotation_report": self.annotation_report,
            "scope": {
                "syntactic_relations_built": True,
                "sentence_proposition_composed": False,
                "thematic_roles_assigned": False,
                "semantic_event_roles_assigned": False,
                "discourse_pragmatics_performed": False,
                "biological_trace_is_measured_human_data": False,
            },
        })


class Point3SyntacticParser:
    """Execute the 12-step source algorithm using ORE's established 58-operation syntax engine.

    The lower-level engine computes category, phrase, dependency, constituency, hierarchy,
    grammatical subject/object, ambiguity, and long-distance syntax. This adapter does not
    expose predicate-argument semantic interpretation or thematic-role outputs.
    """

    def __init__(
        self,
        config: SyntaxConfig | None = None,
        *,
        register_annotations: bool = True,
        repository_root: str | None = None,
        database_path: str | None = None,
    ) -> None:
        """Configure the syntax engine, optional LIF reference, and shared SQLite annotations."""
        self.config = config or SyntaxConfig()
        if register_annotations:
            from .annotations import register_point03_annotations

            self.annotation_report = register_point03_annotations(repository_root, database_path)
        else:
            self.annotation_report = {"status": "annotation_registration_disabled"}

    @staticmethod
    def _expectations(selected_pos: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Estimate next-category expectations from the current best category and open local pattern."""
        continuations = {
            "DET": ("ADJ", "NOUN", "PROPN"),
            "ADJ": ("ADJ", "NOUN", "PROPN"),
            "NOUN": ("VERB", "ADP", "CCONJ", "PUNCT"),
            "PROPN": ("VERB", "ADP", "CCONJ", "PUNCT"),
            "PRON": ("VERB", "AUX", "ADP", "PUNCT"),
            "AUX": ("VERB", "ADV", "PUNCT"),
            "VERB": ("NOUN", "PROPN", "PRON", "ADP", "ADV", "PUNCT"),
            "ADP": ("DET", "NOUN", "PROPN", "PRON"),
            "SCONJ": ("PRON", "NOUN", "PROPN", "DET"),
            "CCONJ": ("NOUN", "PROPN", "PRON", "VERB", "DET"),
            "ADV": ("VERB", "ADJ", "ADV", "PUNCT"),
        }
        output = []
        for index, item in enumerate(selected_pos):
            tag = str(item.get("tag", "X"))
            output.append({
                "after_token_index": index,
                "after_token": item.get("token"),
                "expected_next_categories": list(continuations.get(tag, ("X",))),
                "expectation_source": "transparent local continuation heuristic; grammar/language-specific training not claimed",
            })
        return output

    @staticmethod
    def _feature_value_compatibility(left: Any, right: Any) -> float | None:
        """Return explicit feature compatibility, or None when either side lacks evidence."""
        if left is None or right is None:
            return None
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            left_keys = {
                str(key) for key, value in left.items()
                if (float(value) > 0.0 if isinstance(value, (int, float)) else bool(value))
            }
            right_keys = {
                str(key) for key, value in right.items()
                if (float(value) > 0.0 if isinstance(value, (int, float)) else bool(value))
            }
            if not left_keys or not right_keys:
                return None
            return 1.0 if left_keys & right_keys else 0.0
        if isinstance(left, (list, tuple, set)) or isinstance(right, (list, tuple, set)):
            left_values = {str(value).casefold() for value in (left if isinstance(left, (list, tuple, set)) else [left])}
            right_values = {str(value).casefold() for value in (right if isinstance(right, (list, tuple, set)) else [right])}
            return 1.0 if left_values & right_values else 0.0
        return 1.0 if str(left).casefold() == str(right).casefold() else 0.0

    def _check_agreement(
        self,
        request: Point3Request,
        arcs: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Score only morphosyntactic comparisons licensed by available features and relation labels."""
        noun_relations = {"det", "amod", "nummod"}
        subject_relation = {"nsubj", "nsubj:pass"}
        records: list[dict[str, Any]] = []
        adjusted: list[dict[str, Any]] = []
        for raw_arc in arcs:
            arc = dict(raw_arc)
            dependent = int(arc.get("dependent_index", -1))
            head = arc.get("head_index")
            if head is None or not 0 <= dependent < len(request.lexical_items) or not 0 <= int(head) < len(request.lexical_items):
                adjusted.append(arc)
                continue
            head_index = int(head)
            relation = str(arc.get("relation", ""))
            if relation in subject_relation:
                features = ("number", "person")
            elif relation in noun_relations:
                features = ("number", "gender", "case", "noun_class")
            else:
                adjusted.append(arc)
                continue
            head_features = dict(request.lexical_items[head_index].get("morphology", {}))
            head_features.update(request.morphology_features.get(head_index, {}))
            dependent_features = dict(request.lexical_items[dependent].get("morphology", {}))
            dependent_features.update(request.morphology_features.get(dependent, {}))
            checks = []
            penalty = 0.0
            for feature in features:
                compatibility = self._feature_value_compatibility(
                    head_features.get(feature), dependent_features.get(feature)
                )
                if compatibility is None:
                    continue
                mismatch = 1.0 - compatibility
                checks.append({
                    "feature": feature,
                    "head_value": head_features.get(feature),
                    "dependent_value": dependent_features.get(feature),
                    "compatibility": compatibility,
                    "mismatch": mismatch,
                })
                penalty += 0.35 * mismatch
            if checks:
                original_score = float(arc.get("score", 0.0))
                adjusted_score = original_score - penalty
                arc["score_before_agreement"] = original_score
                arc["score"] = adjusted_score
                arc["agreement_penalty"] = penalty
                arc["agreement_compatibility"] = max(0.0, 1.0 - penalty / 0.35 / len(checks))
                arc["confidence_before_agreement"] = float(arc.get("confidence", 0.0))
                arc["confidence"] = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, adjusted_score - 2.0))))
            else:
                arc["agreement_compatibility"] = None
                arc["agreement_status"] = "unknown_missing_features"
            arc["agreement_checks"] = checks
            if checks:
                record = {
                    "head_index": head_index, "head": arc.get("head"),
                    "dependent_index": dependent, "dependent": arc.get("dependent"),
                    "relation": relation, "checks": checks, "penalty": penalty,
                    "status": "mismatch" if penalty > 0.0 else "compatible",
                }
                records.append(record)
            adjusted.append(arc)
        return records, adjusted

    @staticmethod
    def _cue_evidence(
        request: Point3Request,
        tokens: Sequence[str],
        selected_pos: Sequence[Mapping[str, Any]],
        dependency_relations: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Summarize order, function-word, inflection, punctuation, and supplied prosody cues."""
        punctuation = [
            {"token_index": index, "token": token}
            for index, token in enumerate(tokens)
            if token and not any(character.isalnum() for character in token)
        ]
        function_words = [
            {"token_index": index, "token": item.get("token"), "category": item.get("tag")}
            for index, item in enumerate(selected_pos)
            if item.get("tag") in {"DET", "AUX", "ADP", "CCONJ", "SCONJ", "PART"}
        ]
        supplied = {
            index: dict(cues)
            for index, cues in request.boundary_cues.items()
        }
        return {
            "word_order": {
                "token_positions": list(range(len(tokens))),
                "dependency_relations_considered": len(dependency_relations),
                "source": "ordered lexical stream plus syntax-engine positional features",
            },
            "function_words": function_words,
            "case_and_inflection": {
                str(index): request.morphology_features.get(index, item.get("morphology", {}))
                for index, item in enumerate(request.lexical_items)
                if request.morphology_features.get(index) or item.get("morphology")
            },
            "punctuation_and_visual_boundaries": punctuation,
            "caller_supplied_speech_prosody_or_boundary_cues": supplied,
            "limitations": "Prosody is used only when supplied; punctuation is not treated as a direct neural equivalent of intonation.",
        }

    @staticmethod
    def _structural_conflicts(
        pos_candidates: Sequence[Sequence[Mapping[str, Any]]],
        selected_pos: Sequence[Mapping[str, Any]],
        agreement_checks: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Flag surprising/weak categories and explicit agreement mismatches without equating surprise with error."""
        conflicts: list[dict[str, Any]] = []
        surprisal_rows: list[dict[str, Any]] = []
        agreement_mismatches = [row for row in agreement_checks if row.get("status") == "mismatch"]
        for index, (candidates, selected) in enumerate(zip(pos_candidates, selected_pos, strict=True)):
            tag = str(selected.get("tag", "X"))
            probability = next(
                (float(item.get("posterior", 0.0)) for item in candidates if item.get("tag") == tag),
                float(selected.get("confidence", 0.0)),
            )
            surprisal = -math.log2(max(probability, 1e-12))
            surprisal_rows.append({
                "token_index": index, "token": selected.get("token"),
                "selected_category": tag, "category_probability": probability,
                "syntactic_surprisal_bits": surprisal,
            })
            if selected.get("abstained") or surprisal > 3.0:
                conflicts.append({
                    "type": "low_support_or_unexpected_category",
                    "token_index": index, "token": selected.get("token"),
                    "category": tag, "surprisal_bits": surprisal,
                    "interpretation": "low category support or unexpectedness; not automatically ungrammatical",
                })
        for mismatch in agreement_mismatches:
            conflicts.append({
                "type": "morphosyntactic_agreement_mismatch",
                "head_index": mismatch["head_index"],
                "dependent_index": mismatch["dependent_index"],
                "relation": mismatch["relation"],
                "checks": mismatch["checks"],
            })
        return conflicts, surprisal_rows

    @staticmethod
    def _rank_parses(parses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Convert syntax-engine constituency log scores into normalized relative parse support."""
        rows = [dict(item) for item in parses]
        probabilities = _softmax([float(item.get("log_score", 0.0)) for item in rows])
        for item, probability in zip(rows, probabilities, strict=True):
            item["posterior"] = probability
        return rows

    @staticmethod
    def _unresolved_dependencies(
        links: Sequence[Mapping[str, Any]],
        phrases: Sequence[Mapping[str, Any]],
        tokens: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Return unresolved remote links and incomplete phrase spans without inventing obligatory slots."""
        output = []
        for link in links:
            if str(link.get("status", "")).casefold() != "linked":
                output.append({
                    **dict(link),
                    "kind": "long_distance_dependency",
                    "status": link.get("status", "unresolved"),
                })
        for phrase in phrases:
            start, end = int(phrase.get("start", 0)), int(phrase.get("end", 0))
            if end <= start or end > len(tokens):
                continue
            next_token = tokens[end] if end < len(tokens) else None
            is_boundary = next_token is None or next_token in {".", "!", "?", ";", ":"}
            output.append({
                "kind": "constituent_closure_candidate",
                "phrase_id": phrase.get("phrase_id"),
                "phrase_type": phrase.get("type"),
                "start": start, "end": end,
                "next_token": next_token,
                "closure_supported": bool(is_boundary),
                "closure_is_heuristic": True,
            })
        return output

    @staticmethod
    def _biological_trace(
        operation_trace: Sequence[Mapping[str, Any]],
        biological_summary: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Aggregate selected syntax-engine LIF operations into the 12 source-level biological steps."""
        by_id = {str(item.get("operation_id")): item for item in operation_trace}
        layers = biological_summary.get("layers", {}) if isinstance(biological_summary, Mapping) else {}
        output = []
        for index, specification in enumerate(POINT3_STEP_SPECS, start=1):
            selected = [
                by_id[operation_id]["biological_reference_step"]
                for operation_id in _POINT3_TO_SYN[index]
                if operation_id in by_id
            ]
            active = sorted({
                name
                for item in selected
                for name in item.get("active_populations", [])
            })
            status = (
                "simulated_reference_not_measured_human_data"
                if selected and any(item.get("status") != "disabled" for item in selected)
                else "biological_simulation_disabled"
            )
            output.append({
                "operation_id": specification["operation_id"],
                "status": status,
                "underlying_syntax_operation_ids": list(_POINT3_TO_SYN[index]),
                "operation_count": len(selected),
                "last_simulated_time_ms": max(
                    (float(item.get("simulated_time_ms", 0.0)) for item in selected),
                    default=0.0,
                ),
                "mean_network_activity": (
                    sum(float(item.get("mean_network_activity", 0.0)) for item in selected) / len(selected)
                    if selected else 0.0
                ),
                "active_populations": active,
                "population_summary_at_parse_end": dict(layers),
                "biological_scope": "generic coupled current_based_LIF syntax reference; not anatomically fitted or measured",
            })
        return output

    def process(self, request: Point3Request) -> Point3Result:
        """Execute Point 3 and return a grammatical representation for the later semantic-composition stage."""
        if not isinstance(request, Point3Request):
            raise TypeError("request must be a Point3Request")
        tokens = [str(item["form"]) for item in request.lexical_items]
        lexical_evidence = {
            index: dict(item.get("category_candidates", {}))
            for index, item in enumerate(request.lexical_items)
            if item.get("category_candidates")
        }
        lexicon = {
            str(item["form"]): {"pos": dict(item.get("category_candidates", {}))}
            for item in request.lexical_items
            if item.get("category_candidates")
        }
        incremental_updates: list[dict[str, Any]] = []
        previous_pos: dict[int, str] = {}
        previous_dependencies: dict[int, tuple[Any, Any, str]] = {}
        syntax_result = None
        for prefix_end in range(1, len(tokens) + 1):
            # Prefixes are parsed without LIF dynamics. The complete final prefix runs the
            # configured biological reference once, keeping functional and biological tracks
            # separate without multiplying simulation cost by sentence length.
            prefix_config = self.config if prefix_end == len(tokens) else SyntaxConfig(
                **{**self.config.to_dict(), "simulate_biology": False}
            )
            prefix_tokens = tuple(tokens[:prefix_end])
            prefix_evidence = {
                index: evidence for index, evidence in lexical_evidence.items()
                if index < prefix_end
            }
            prefix_result = SyntaxProcessor(
                lexicon=lexicon, config=prefix_config
            ).process(SyntaxRequest(
                sequence_id=f"{request.sequence_id}-prefix-{prefix_end}",
                tokens=prefix_tokens,
                language=request.language,
                lexical_evidence=prefix_evidence,
                source_reference=request.source_reference,
            ))
            current_pos = {
                int(item["token_index"]): str(item["tag"])
                for item in prefix_result.selected_pos
            }
            current_dependencies = {
                int(item["dependent_index"]): (
                    item.get("head_index"), item.get("head"), str(item.get("relation", ""))
                )
                for item in prefix_result.dependency_arcs
            }
            revisions = [
                {
                    "token_index": index,
                    "old_category": previous_pos[index],
                    "new_category": tag,
                }
                for index, tag in current_pos.items()
                if index in previous_pos and previous_pos[index] != tag
            ]
            dependency_revisions = [
                {
                    "dependent_index": index,
                    "previous": previous_dependencies[index],
                    "revised": relation,
                }
                for index, relation in current_dependencies.items()
                if index in previous_dependencies and previous_dependencies[index] != relation
            ]
            prefix_parses = self._rank_parses(prefix_result.constituency_parses)
            incremental_updates.append({
                "prefix_end_exclusive": prefix_end,
                "new_token_index": prefix_end - 1,
                "new_token": tokens[prefix_end - 1],
                "selected_pos": [dict(item) for item in prefix_result.selected_pos],
                "constituent_groups": [dict(item) for item in prefix_result.phrases],
                "dependency_relations": [dict(item) for item in prefix_result.dependency_arcs],
                "constituency_alternatives": [
                    {
                        "rank": int(item["rank"]),
                        "label": item["label"],
                        "log_score": float(item["log_score"]),
                        "posterior": float(item["posterior"]),
                    }
                    for item in prefix_parses
                ],
                "active_expectation": self._expectations(prefix_result.selected_pos)[-1],
                "unresolved_dependencies": self._unresolved_dependencies(
                    prefix_result.long_distance_dependencies,
                    prefix_result.phrases,
                    prefix_tokens,
                ),
                "category_revision_events": revisions,
                "dependency_revision_events": dependency_revisions,
                "dependency_tree_valid": bool(prefix_result.diagnostics.get("dependency_tree_valid")),
                "partial_parse_is_final_sentence_parse": prefix_end == len(tokens),
            })
            previous_pos = current_pos
            previous_dependencies = current_dependencies
            syntax_result = prefix_result
        assert syntax_result is not None
        agreement_checks, dependency_relations = self._check_agreement(
            request, syntax_result.dependency_arcs
        )
        constituency_alternatives = self._rank_parses(syntax_result.constituency_parses)
        selected_pos = [dict(item) for item in syntax_result.selected_pos]
        pos_candidates = [[dict(item) for item in row] for row in syntax_result.pos_candidates]
        active_expectations = self._expectations(selected_pos)
        conflicts, surprisal_rows = self._structural_conflicts(
            pos_candidates, selected_pos, agreement_checks
        )
        revision_events = [
            {
                "token_index": int(item.get("token_index", index)),
                "token": item.get("token"),
                "old_category": item.get("previous_tag", item.get("original_tag")),
                "new_category": item.get("tag"),
                "changed": bool(item.get("changed", False)),
                "reason": item.get("revision_reason", "syntax engine recurrent ambiguity resolution"),
            }
            for index, item in enumerate(selected_pos)
            if item.get("changed")
        ]
        unresolved = self._unresolved_dependencies(
            syntax_result.long_distance_dependencies,
            syntax_result.phrases,
            tokens,
        )
        biological = dict(syntax_result.biological)
        biological_trace = self._biological_trace(syntax_result.operation_trace, biological)
        parse_posterior = float(constituency_alternatives[0]["posterior"]) if constituency_alternatives else 0.0
        category_confidence = (
            sum(float(item.get("confidence", 0.0)) for item in selected_pos) / len(selected_pos)
            if selected_pos else 0.0
        )
        confidence = max(0.0, min(1.0, 0.5 * parse_posterior + 0.5 * category_confidence))
        functional_outputs = (
            {
                "aligned_tokens": tokens,
                "token_indices": list(range(len(tokens))),
                "lexical_uncertainty_preserved": True,
                "inherited_confidence": [float(item["confidence"]) for item in request.lexical_items],
                "prefix_parse_count": len(incremental_updates),
                "incremental_updates": incremental_updates,
            },
            {
                "pos_candidates": pos_candidates,
                "selected_pos": selected_pos,
                "category_ambiguity_preserved": True,
            },
            {
                "active_expectations": active_expectations,
                "expectation_history": [item["active_expectation"] for item in incremental_updates],
                "surprisal_by_token": surprisal_rows,
            },
            {
                "constituent_groups": syntax_result.phrases,
                "constituency_alternatives": constituency_alternatives,
                "prefix_constituent_history": [
                    {"prefix_end_exclusive": item["prefix_end_exclusive"],
                     "constituent_count": len(item["constituent_groups"])}
                    for item in incremental_updates
                ],
                "hierarchical_groups_created": bool(syntax_result.constituency_parses),
            },
            {
                "dependency_relations": dependency_relations,
                "dependency_revision_history": [
                    {"prefix_end_exclusive": item["prefix_end_exclusive"],
                     "revisions": item["dependency_revision_events"]}
                    for item in incremental_updates
                    if item["dependency_revision_events"]
                ],
                "single_head_dependency_tree_valid": syntax_result.diagnostics["dependency_tree_valid"],
            },
            {
                "agreement_checks": agreement_checks,
                "missing_features_treated_as_unknown": True,
            },
            {
                "cue_evidence": self._cue_evidence(request, tokens, selected_pos, dependency_relations),
            },
            {
                "unresolved_dependency_slots": unresolved,
                "open_slot_history": [
                    {"prefix_end_exclusive": item["prefix_end_exclusive"],
                     "unresolved_dependencies": item["unresolved_dependencies"]}
                    for item in incremental_updates
                ],
                "long_distance_dependencies": syntax_result.long_distance_dependencies,
            },
            {
                "ranked_constituency_alternatives": constituency_alternatives,
                "alternative_count": len(constituency_alternatives),
                "winner_take_all_forced": False,
            },
            {
                "structural_conflicts": conflicts,
                "surprisal_by_token": surprisal_rows,
                "surprise_is_not_equivalent_to_ungrammaticality": True,
            },
            {
                "revision_events": revision_events,
                "incremental_category_revisions": [
                    {"prefix_end_exclusive": item["prefix_end_exclusive"],
                     "revisions": item["category_revision_events"]}
                    for item in incremental_updates
                    if item["category_revision_events"]
                ],
                "recurrent_ambiguity_resolution_executed": True,
                "unresolved_alternatives_preserved": True,
            },
            {
                "consolidated_constituency_tree": syntax_result.constituency_tree,
                "syntax_hierarchy": syntax_result.hierarchy,
                "dependency_relations": dependency_relations,
                "open_dependencies_preserved": unresolved,
                "semantic_proposition_composed": False,
                "thematic_roles_assigned": False,
                "handoff_target": "Point 4 semantic composition",
            },
        )
        functional_trace = []
        for specification, output in zip(POINT3_STEP_SPECS, functional_outputs, strict=True):
            functional_trace.append({
                **specification,
                "status": "completed",
                "functional_result": to_jsonable(output),
                "underlying_syntax_operation_ids": list(_POINT3_TO_SYN[len(functional_trace) + 1]),
            })
        cue_evidence = self._cue_evidence(request, tokens, selected_pos, dependency_relations)
        diagnostics = {
            "valid": len(functional_trace) == 12 and len(syntax_result.operation_trace) == 58,
            "source_step_count": len(functional_trace),
            "incremental_prefix_parse_count": len(incremental_updates),
            "incremental_prefix_count_matches_tokens": len(incremental_updates) == len(tokens),
            "underlying_syntax_operation_count": len(syntax_result.operation_trace),
            "underlying_operation_order_valid": bool(syntax_result.diagnostics.get("operation_order_valid")),
            "dependency_tree_valid": bool(syntax_result.diagnostics.get("dependency_tree_valid")),
            "constituency_alternative_count": len(constituency_alternatives),
            "agreement_check_count": len(agreement_checks),
            "agreement_mismatch_count": sum(item["status"] == "mismatch" for item in agreement_checks),
            "structural_conflict_count": len(conflicts),
            "revision_event_count": len(revision_events),
            "unresolved_dependency_slot_count": len(unresolved),
            "biological_simulation_enabled": bool(biological.get("enabled")),
            "biological_reference_is_measured_data": False,
            "sentence_proposition_composed": False,
            "thematic_roles_assigned": False,
            "warnings": list(syntax_result.diagnostics.get("warnings", ())) + [
                "Agreement checks only compare caller-supplied morphology; missing features are unknown.",
                "Next-category expectations and phrase-closure indicators are transparent heuristics.",
                "The existing ORE syntax engine has an English-oriented compact grammar; passing another language label does not create a language-specific grammar.",
                "The underlying reference engine also computes internal downstream fields, but Point 3 does not expose predicate-argument or thematic-role interpretations.",
            ],
        }
        return Point3Result(
            sequence_id=request.sequence_id,
            tokens=tokens,
            lexical_item_records=request.lexical_items,
            incremental_updates=incremental_updates,
            pos_candidates=pos_candidates,
            selected_pos=selected_pos,
            constituent_groups=syntax_result.phrases,
            dependency_relations=dependency_relations,
            constituency_alternatives=constituency_alternatives,
            constituency_tree=syntax_result.constituency_tree,
            syntax_hierarchy=syntax_result.hierarchy,
            unresolved_dependency_slots=unresolved,
            agreement_checks=agreement_checks,
            cue_evidence=cue_evidence,
            active_expectations=active_expectations,
            structural_conflicts=conflicts,
            revision_events=revision_events,
            confidence=confidence,
            functional_trace=functional_trace,
            biological_trace=biological_trace,
            biological_summary=biological,
            diagnostics=diagnostics,
            annotation_report=self.annotation_report,
        )
