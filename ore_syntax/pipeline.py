"""Nine-stage functional syntax parser with explicit evidence and trace outputs.

This is a deterministic, rule-based English research prototype. Its numerical
scores are model scores, not calibrated psycholinguistic probabilities.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence
import math
import re
import unicodedata

from .biology import POPULATIONS, SyntaxLIFSimulator
from .contracts import SyntaxConfig, SyntaxRequest, SyntaxResult
from .registry import OPERATION_REGISTRY, STAGE_REGISTRY


POS_TAGS = (
    "NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV", "DET", "ADP",
    "PRON", "CCONJ", "SCONJ", "PART", "NUM", "PUNCT", "INTJ", "X",
)
_NOMINAL = {"NOUN", "PROPN", "PRON", "NUM"}
_CLOSED_CLASS = {"DET", "ADP", "PRON", "AUX", "CCONJ", "SCONJ", "PART"}
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*|\d+(?:[.,]\d+)*|[^\w\s]", re.UNICODE)

_LEXICAL_CATEGORIES: dict[str, dict[str, float]] = {}
def _add_words(category: str, words: str, strength: float = 0.92) -> None:
    for word in words.split():
        _LEXICAL_CATEGORIES.setdefault(word, {})[category] = strength

_add_words("DET", "a an the this that these those some any every each no", 0.98)
_add_words("PRON", "i me we us you he him she her it they them who whom whose what which myself yourself", 0.98)
_add_words("AUX", "am is are was were be been being do does did have has had can could may might must shall should will would", 0.98)
_add_words("ADP", "in on at by with from to into onto over under through after before of for about around against between during without", 0.98)
_add_words("CCONJ", "and but or nor yet so", 0.98)
_add_words("SCONJ", "because although though while if unless when since that whether", 0.93)
_add_words("PART", "not n't to", 0.90)
_add_words("ADV", "very quite rather too also only just already still often always never here there now then quickly slowly carefully yesterday today tomorrow", 0.88)
_add_words("ADJ", "good bad new old large small little big happy sad red blue green fast slow important possible difficult easy", 0.82)
_add_words("VERB", "be have do say go get make know think take see come want use find give tell work call try ask need feel become leave put mean keep let begin seem help talk turn start show hear play run move live believe bring happen write provide sit stand lose pay meet include continue set learn change lead understand watch follow stop create read eat drink buy build open close chase love like", 0.78)
_add_words("NOUN", "time person year way day man woman child world life hand part eye place work week case point government company number group problem fact house water book school family question", 0.72)
_add_words("INTJ", "oh hey hello yes yeah no wow", 0.75)
_add_words("PROPN", "mary john alice bob london paris", 0.88)
_add_words("NOUN", "cat cats dog dogs fish apple apples boy girl man woman child children cake book books ball teacher student scientist computer city car mouse mice", 0.78)
_add_words("VERB", "eats ate eaten drinks drank drunk reads read writes wrote written buys bought chases chased sees saw gives gave given makes made builds built likes liked watches watched runs ran walks walked", 0.84)

_AMBIGUOUS_LEXEMES = {
    "book": {"NOUN": 0.55, "VERB": 0.45},
    "can": {"AUX": 0.58, "VERB": 0.22, "NOUN": 0.20},
    "saw": {"VERB": 0.58, "NOUN": 0.42},
    "watch": {"VERB": 0.53, "NOUN": 0.47},
    "light": {"NOUN": 0.38, "VERB": 0.28, "ADJ": 0.34},
    "that": {"DET": 0.42, "SCONJ": 0.32, "PRON": 0.26},
    "like": {"VERB": 0.55, "ADP": 0.25, "ADJ": 0.20},
    "flies": {"NOUN": 0.48, "VERB": 0.52},
    "close": {"ADJ": 0.36, "ADV": 0.22, "VERB": 0.42},
    "record": {"NOUN": 0.55, "VERB": 0.45},
}

_SUFFIX_CUES = {
    "VERB": (("ing", 0.65), ("ed", 0.62), ("ize", 0.42), ("ise", 0.42), ("ify", 0.42)),
    "NOUN": (("tion", 0.72), ("sion", 0.66), ("ment", 0.58), ("ness", 0.58), ("ity", 0.52), ("ship", 0.42)),
    "ADJ": (("ous", 0.62), ("ful", 0.58), ("less", 0.56), ("able", 0.56), ("ible", 0.50), ("ive", 0.50), ("al", 0.34)),
    "ADV": (("ly", 0.58),),
}

_BINARY_RULES = (
    ("S", "NP", "VP", 1.00),
    ("S", "S", "CCONJ", 0.82),
    ("S", "CCONJ", "S", 0.78),
    ("NP", "DET", "NOM", 0.98),
    ("NOM", "ADJ", "NOM", 0.92),
    ("NOM", "NUM", "NOM", 0.78),
    ("NP", "NP", "PP", 0.72),
    ("VP", "AUX", "VP", 0.96),
    ("VP", "VERB", "NP", 0.96),
    ("VP", "VERB", "PP", 0.82),
    ("VP", "VP", "NP", 0.88),
    ("VP", "VP", "PP", 0.80),
    ("VP", "VP", "ADVP", 0.68),
    ("VP", "VERB", "ADJP", 0.76),
    ("PP", "ADP", "NP", 0.98),
    ("ADJP", "ADV", "ADJ", 0.88),
    ("ADVP", "ADV", "ADV", 0.58),
    ("SBAR", "SCONJ", "S", 0.92),
    ("SBAR", "PRON", "S", 0.72),
    ("S", "S", "PUNCT", 0.98),
)
_UNARY_RULES = (
    ("NP", "NOUN", 0.88), ("NP", "PROPN", 0.92), ("NP", "PRON", 0.96),
    ("NP", "NUM", 0.65), ("NP", "NOM", 0.94),
    ("NOM", "NOUN", 0.94), ("NOM", "PROPN", 0.96),
    ("VP", "VERB", 0.96), ("VP", "AUX", 0.62),
    ("ADJP", "ADJ", 0.94), ("ADVP", "ADV", 0.90),
    ("S", "VP", 0.50),
)
_VERB_FRAMES: dict[str, dict[str, dict[str, float]]] = {
    "eat": {"agent": {"ANIMATE": 1.0, "HUMAN": 0.7}, "patient": {"FOOD": 1.0, "OBJECT": 0.2}},
    "drink": {"agent": {"ANIMATE": 1.0}, "patient": {"LIQUID": 1.0, "DRINK": 1.0}},
    "read": {"agent": {"ANIMATE": 1.0, "HUMAN": 0.7}, "patient": {"TEXT": 1.0, "BOOK": 0.8, "INFORMATION": 0.6}},
    "write": {"agent": {"ANIMATE": 1.0, "HUMAN": 0.7}, "patient": {"TEXT": 1.0, "DOCUMENT": 0.7}},
    "see": {"agent": {"ANIMATE": 0.8}, "patient": {"VISIBLE": 1.0, "OBJECT": 0.4}},
    "watch": {"agent": {"ANIMATE": 0.8}, "patient": {"EVENT": 0.6, "OBJECT": 0.5}},
    "chase": {"agent": {"ANIMATE": 1.0}, "patient": {"ANIMATE": 0.8}},
    "build": {"agent": {"ANIMATE": 0.8}, "patient": {"ARTIFACT": 1.0, "OBJECT": 0.6}},
    "make": {"agent": {"ANIMATE": 0.8}, "patient": {"ARTIFACT": 0.7, "OBJECT": 0.5}},
    "buy": {"agent": {"ANIMATE": 0.9}, "patient": {"OBJECT": 0.8, "GOODS": 1.0}},
    "give": {"agent": {"ANIMATE": 0.9}, "patient": {"OBJECT": 0.6}, "recipient": {"ANIMATE": 1.0}},
    "tell": {"agent": {"ANIMATE": 0.9}, "patient": {"INFORMATION": 0.8}, "recipient": {"ANIMATE": 0.7}},
    "open": {"agent": {"ANIMATE": 0.7}, "patient": {"OBJECT": 0.8}},
    "close": {"agent": {"ANIMATE": 0.7}, "patient": {"OBJECT": 0.8}},
    "find": {"agent": {"ANIMATE": 0.8}, "patient": {"OBJECT": 0.7, "INFORMATION": 0.5}},
    "love": {"agent": {"ANIMATE": 0.8}, "patient": {"ANIMATE": 0.8}},
    "like": {"agent": {"ANIMATE": 0.8}, "patient": {"OBJECT": 0.6, "ANIMATE": 0.6}},
}
_PARTICIPLES = {"eaten", "seen", "written", "built", "made", "known", "taken", "given", "found", "done", "shown", "drunk"}
_WH_WORDS = {"who", "whom", "whose", "what", "which", "where", "when", "why", "how"}
_RELATIVE_WORDS = {"who", "whom", "whose", "which", "that"}
_COMPLEMENTIZERS = {"that", "whether", "if", "because", "although", "while", "since"}


def _softmax(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    if not scores:
        return []
    temp = max(float(temperature), 1e-8)
    scaled = [max(-80.0, min(80.0, float(score) / temp)) for score in scores]
    peak = max(scaled)
    exps = [math.exp(value - peak) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps] if total > 0 else [1.0 / len(scores)] * len(scores)


def _cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float | None:
    keys = set(left) | set(right)
    if not keys:
        return None
    dot = sum(float(left.get(key, 0.0)) * float(right.get(key, 0.0)) for key in keys)
    ln = math.sqrt(sum(float(left.get(key, 0.0)) ** 2 for key in keys))
    rn = math.sqrt(sum(float(right.get(key, 0.0)) ** 2 for key in keys))
    return dot / (ln * rn) if ln > 0 and rn > 0 else None


def _normalize(token: str) -> str:
    return unicodedata.normalize("NFC", token).casefold()


def _tree_node(label: str, start: int, end: int, children: Sequence[dict[str, Any]] = (),
               *, token_index: int | None = None, score: float = 0.0) -> dict[str, Any]:
    return {
        "label": label, "start": start, "end": end, "token_index": token_index,
        "children": [dict(child) for child in children], "score": float(score),
    }


class SyntaxProcessor:
    """Nine-stage syntax processor that accepts upstream ORE evidence read-only.

    Lexicon entries can contain POS labels, frequencies and semantic features.
    SyntaxRequest.lexical_evidence carries token-indexed category weights from
    upstream lexical access. The processor never trains or mutates the lexicon.
    """

    def __init__(
        self,
        lexicon: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
        *,
        verb_frames: Mapping[str, Mapping[str, Mapping[str, float]]] | None = None,
        config: SyntaxConfig | None = None,
    ) -> None:
        self.config = config or SyntaxConfig()
        self.lexicon: dict[str, dict[str, Any]] = {}
        if isinstance(lexicon, Mapping):
            for form, record in lexicon.items():
                self.lexicon[_normalize(str(form))] = self._clean_lexical_record(record)
        elif lexicon is not None:
            for record in lexicon:
                if not isinstance(record, Mapping):
                    raise TypeError("lexicon sequence entries must be mappings")
                form = record.get("orthography", record.get("form", record.get("word")))
                if not isinstance(form, str) or not form.strip():
                    raise ValueError("lexicon sequence records need orthography/form/word")
                self.lexicon[_normalize(form)] = self._clean_lexical_record(record)
        self.verb_frames = {
            _normalize(verb): {str(role).casefold(): dict(features) for role, features in roles.items()}
            for verb, roles in (verb_frames or _VERB_FRAMES).items()
        }

    @staticmethod
    def _clean_lexical_record(record: Any) -> dict[str, Any]:
        if isinstance(record, str):
            return {"pos": {record.upper(): 1.0}}
        if isinstance(record, Mapping):
            data = dict(record)
            category = data.get("lexical_category", data.get("category"))
            if category and "pos" not in data:
                data["pos"] = {str(category).upper(): 1.0}
            return data
        if hasattr(record, "lexical_category"):
            return {
                "pos": {str(getattr(record, "lexical_category", "X")).upper(): 1.0},
                "frequency": float(getattr(record, "frequency", 1.0)),
            }
        return {}

    def _tokenize(self, request: SyntaxRequest) -> tuple[list[str], str]:
        if request.tokens is not None:
            tokens = [str(token).strip() for token in request.tokens]
            original = " ".join(tokens)
        else:
            original = str(request.text)
            tokens = _TOKEN_RE.findall(original)
        if not tokens:
            raise ValueError("no tokens could be extracted from the syntax request")
        if len(tokens) > self.config.max_tokens:
            raise ValueError(f"token count {len(tokens)} exceeds max_tokens={self.config.max_tokens}")
        return tokens, original

    def _base_pos_scores(
        self, token: str, index: int, tokens: Sequence[str], request: SyntaxRequest
    ) -> dict[str, float]:
        word = _normalize(token)
        scores: dict[str, float] = {}
        for tag, strength in _LEXICAL_CATEGORIES.get(word, {}).items():
            scores[tag] = max(scores.get(tag, 0.0), strength)
        if word in _AMBIGUOUS_LEXEMES:
            scores.update(_AMBIGUOUS_LEXEMES[word])
        if word.isdigit():
            scores["NUM"] = 0.99
        if token in {".", ",", ";", ":", "!", "?", "(", ")", "[", "]", '"', "—", "-", "…"}:
            scores["PUNCT"] = 0.99
        if token[:1].isupper() and index > 0 and word not in {"i"}:
            scores["PROPN"] = max(scores.get("PROPN", 0.0), 0.78)
        for tag, cues in _SUFFIX_CUES.items():
            for suffix, strength in cues:
                if len(word) > len(suffix) + 2 and word.endswith(suffix):
                    scores[tag] = max(scores.get(tag, 0.0), strength)
        if not scores:
            scores.update({"NOUN": 0.48, "VERB": 0.32, "ADJ": 0.12, "X": 0.08})

        lexical = self.lexicon.get(word, {})
        lex_pos = lexical.get("pos", lexical.get("pos_candidates", {}))
        if isinstance(lex_pos, str):
            lex_pos = {lex_pos.upper(): 1.0}
        if isinstance(lex_pos, Mapping):
            for tag, value in lex_pos.items():
                tag = str(tag).upper()
                if tag in POS_TAGS:
                    scores[tag] = max(scores.get(tag, 0.0), float(value))
        supplied = request.lexical_evidence.get(index, {})
        supplied_pos = supplied.get("pos", supplied.get("pos_candidates", supplied))
        if isinstance(supplied_pos, Mapping):
            for tag, value in supplied_pos.items():
                tag = str(tag).upper()
                if tag in POS_TAGS and isinstance(value, (int, float)):
                    scores[tag] = max(scores.get(tag, 0.0), float(value))

        prev = _normalize(tokens[index - 1]) if index > 0 else ""
        nxt = _normalize(tokens[index + 1]) if index + 1 < len(tokens) else ""
        prev_tag = next(iter(_LEXICAL_CATEGORIES.get(prev, {})), "")
        next_tag = next(iter(_LEXICAL_CATEGORIES.get(nxt, {})), "")
        if prev in {"a", "an", "the", "this", "that", "these", "those", "each", "some"}:
            scores["NOUN"] = scores.get("NOUN", 0.0) + 0.22
            scores["ADJ"] = scores.get("ADJ", 0.0) + 0.12
        if prev in {"to", "will", "would", "can", "could", "may", "might", "must", "should", "did", "does", "do"}:
            scores["VERB"] = scores.get("VERB", 0.0) + 0.24
        if prev in {"in", "on", "at", "by", "with", "from", "of", "for", "about", "under"}:
            for tag in ("NOUN", "PROPN", "PRON", "NUM"):
                scores[tag] = scores.get(tag, 0.0) + 0.16
        if next_tag in {"NOUN", "PROPN"} and word in {"a", "an", "the", "this", "that", "these", "those"}:
            scores["DET"] = max(scores.get("DET", 0.0), 0.99)
        if prev_tag in {"PRON", "NOUN", "PROPN"} and word in _AMBIGUOUS_LEXEMES:
            scores["VERB"] = scores.get("VERB", 0.0) + 0.12
        return {tag: math.log(max(1e-5, min(1.0, float(value)))) for tag, value in scores.items()}

    def _trace_stage(
        self, stage_index: int, payloads: Sequence[Mapping[str, Any]], *,
        drives: Mapping[str, float], trace: list[dict[str, Any]],
        simulator: SyntaxLIFSimulator | None,
    ) -> None:
        stage = STAGE_REGISTRY[stage_index - 1]
        operations = [item for item in OPERATION_REGISTRY if item.stage_id == stage.stage_id]
        if len(operations) != len(payloads):
            raise RuntimeError(f"{stage.stage_id} trace payload count mismatch")
        for spec, payload in zip(operations, payloads):
            neural = simulator.step(spec.operation_id, drives) if simulator else {
                "operation_id": spec.operation_id, "status": "disabled"
            }
            trace.append({
                **spec.to_dict(), "status": "executed",
                "output_summary": dict(payload), "biological_reference_step": neural,
            })

    def _pos_identification(self, tokens, request, trace, simulator):
        candidates, selected = [], []
        for i, token in enumerate(tokens):
            scores = self._base_pos_scores(token, i, tokens, request)
            tags = sorted(scores)
            probabilities = _softmax([scores[tag] for tag in tags], self.config.pos_temperature)
            ranked = sorted(
                [{"tag": tag, "score": float(scores[tag]), "posterior": float(prob)}
                 for tag, prob in zip(tags, probabilities)],
                key=lambda item: item["posterior"], reverse=True,
            )
            candidates.append(ranked)
            top = ranked[0]
            margin = top["posterior"] - (ranked[1]["posterior"] if len(ranked) > 1 else 0.0)
            selected.append({
                "token_index": i, "token": token, "tag": top["tag"],
                "confidence": top["posterior"], "margin": margin,
                "abstained": top["posterior"] < self.config.confidence_threshold,
            })
        self._trace_stage(1, [
            {"token_count": len(tokens), "normalized_tokens": [_normalize(t) for t in tokens]},
            {"tokens_with_lexical_evidence": sum(bool(self.lexicon.get(_normalize(t))) for t in tokens)},
            {"candidate_count": sum(len(row) for row in candidates)},
            {"contextual_evidence_used": True},
            {"score_rule": "log lexical/morphology evidence + local-context cue"},
            {"posterior_rows": len(candidates), "temperature": self.config.pos_temperature},
            {"top_tag_count": len(selected), "abstention_count": sum(x["abstained"] for x in selected)},
        ], drives={"orthographic_input": 0.8, "lexical_pos": 0.8, "temporal_syntax": 0.35, "frontal_control": 0.25},
           trace=trace, simulator=simulator)
        return candidates, selected

    @staticmethod
    def _best_tags(selected):
        return [str(item["tag"]) for item in selected]

    def _phrase_detection(self, tokens, selected, trace, simulator):
        tags = self._best_tags(selected)
        phrases, seen = [], set()

        def add(label, start, end, head, confidence, rule):
            key = (label, start, end)
            if start < end and key not in seen:
                seen.add(key)
                phrases.append({
                    "phrase_id": f"{label}-{start}-{end}", "type": label,
                    "start": start, "end": end, "head_index": head,
                    "text": " ".join(tokens[start:end]),
                    "confidence": float(max(0.0, min(1.0, confidence))), "rule": rule,
                })

        nominal_heads = [i for i, tag in enumerate(tags) if tag in _NOMINAL]
        for head in nominal_heads:
            start, end = head, head + 1
            while start > 0 and tags[start - 1] in {"DET", "ADJ", "NUM"}:
                start -= 1
            while start > 0 and tags[start - 1] in {"NOUN", "PROPN"} and tags[start] in {"NOUN", "PROPN"}:
                start -= 1
            while end < len(tags) and tags[end] in {"NOUN", "PROPN"}:
                end += 1
            add("NP", start, end, head, min(float(selected[head]["confidence"]), 0.95),
                "nominal head expanded over determiner/adjective/numeral/compound cues")
        for i, tag in enumerate(tags):
            if tag == "ADP" and i + 1 < len(tags):
                np_candidates = [p for p in phrases if p["type"] == "NP" and p["start"] == i + 1]
                if np_candidates:
                    np = min(np_candidates, key=lambda p: p["end"])
                    add("PP", i, int(np["end"]), int(np["head_index"]), 0.82, "ADP followed by NP")
            if tag == "ADJ":
                start = i - 1 if i > 0 and tags[i - 1] == "ADV" else i
                add("ADJP", start, i + 1, i, 0.76, "adverb? + adjective template")
            if tag == "ADV":
                add("ADVP", i, i + 1, i, 0.70, "adverb lexical category")
        for i, tag in enumerate(tags):
            if tag in {"VERB", "AUX"}:
                start = i
                while start > 0 and tags[start - 1] in {"AUX", "ADV"} and i - start < 3:
                    start -= 1
                end = i + 1
                while end < len(tags) and tags[end] in {"AUX", "VERB", "ADV", "PART"}:
                    end += 1
                add("VP", start, end, i, float(selected[i]["confidence"]),
                    "predicate chain: auxiliary/verb/adverb/particle")
        phrases.sort(key=lambda p: (p["start"], -(p["end"] - p["start"]), p["type"]))
        self._trace_stage(2, [
            {"boundary_cues": [i for i, tag in enumerate(tags) if tag in _CLOSED_CLASS or tag == "PUNCT"]},
            {"nominal_head_count": len(nominal_heads)},
            {"noun_phrase_count": sum(p["type"] == "NP" for p in phrases)},
            {"verb_phrase_count": sum(p["type"] == "VP" for p in phrases)},
            {"other_phrase_count": sum(p["type"] in {"PP", "ADJP", "ADVP"} for p in phrases)},
            {"phrase_span_count": len(phrases), "overlaps_preserved": True},
        ], drives={"temporal_syntax": 0.65, "phrase_binding": 0.8, "frontal_control": 0.25},
           trace=trace, simulator=simulator)
        return phrases

    @staticmethod
    def _relation_and_score(dep, head, tags, tokens, phrase_membership):
        dep_tag, head_tag = tags[dep], tags[head]
        dep_word = _normalize(tokens[dep])
        distance = abs(dep - head)
        score = 0.25 + 1.0 / (1.0 + distance)
        relation = "dep"
        if dep_tag == "PUNCT":
            relation, score = "punct", 4.5 - 0.05 * distance
        elif dep_tag == "DET" and head_tag in {"NOUN", "PROPN"} and head > dep:
            relation, score = "det", 4.4 - 0.08 * distance
        elif dep_tag == "ADJ" and head_tag in {"NOUN", "PROPN"}:
            relation, score = "amod", 3.8 - 0.09 * distance
        elif dep_tag == "NUM" and head_tag in {"NOUN", "PROPN"}:
            relation, score = "nummod", 3.8 - 0.08 * distance
        elif dep_tag == "ADP" and head_tag in _NOMINAL and head > dep:
            relation, score = "case", 3.9 - 0.08 * distance
        elif dep_tag == "AUX" and head_tag == "VERB" and abs(head - dep) <= 3:
            relation, score = "aux", 4.1 - 0.08 * distance
        elif dep_tag == "ADV" and head_tag in {"VERB", "ADJ", "ADV"}:
            relation, score = "advmod", 2.8 - 0.06 * distance
        elif dep_tag == "PART" and head_tag == "VERB":
            relation, score = "mark", 2.9 - 0.06 * distance
        elif dep_tag == "CCONJ":
            relation, score = "cc", 2.0 - 0.05 * distance
        elif dep_tag == "SCONJ":
            relation, score = "mark", 2.7 - 0.05 * distance
        elif dep_tag in _NOMINAL and head_tag == "VERB":
            if dep_word in {"who", "whom", "what", "which"}:
                relation, score = ("nsubj" if dep < head else "obj"), 2.9
            else:
                relation = "nsubj" if dep < head else "obj"
                score = 3.0 - 0.04 * distance
                if dep > 0 and _normalize(tokens[dep - 1]) == "by":
                    relation, score = "obl:agent", 3.65 - 0.05 * distance
        elif dep_tag in {"VERB", "AUX"} and head_tag == "VERB":
            relation, score = ("conj" if dep > 0 and tags[dep - 1] == "CCONJ" else "xcomp"), 2.4 - 0.07 * distance
        elif dep_tag in _NOMINAL and head_tag in _NOMINAL:
            if dep < head and head - dep <= 3:
                relation, score = "compound", 1.8 - 0.07 * distance
            elif dep > head and dep - head <= 3:
                relation, score = "nmod", 1.6 - 0.07 * distance
        if (dep, head) in phrase_membership:
            score += 0.12
        return relation, score

    def _dependency_parse(self, tokens, selected, phrases, trace, simulator):
        tags, n = self._best_tags(selected), len(tokens)
        verb_indices = [i for i, tag in enumerate(tags) if tag == "VERB"]
        if not verb_indices:
            verb_indices = [i for i, tag in enumerate(tags) if tag == "AUX"]
        root = max(verb_indices, key=lambda i: float(selected[i]["confidence"])) if verb_indices else (
            max(range(n), key=lambda i: float(selected[i]["confidence"]))
        )
        phrase_membership = set()
        for phrase in phrases:
            head = phrase.get("head_index")
            if head is not None:
                for i in range(int(phrase["start"]), int(phrase["end"])):
                    if i != int(head):
                        phrase_membership.add((i, int(head)))
        arcs = []
        for dep in range(n):
            if dep == root:
                arcs.append({
                    "dependent_index": dep, "dependent": tokens[dep], "head_index": None,
                    "head": "ROOT", "relation": "root", "score": 5.0, "confidence": 1.0,
                })
                continue
            candidates = []
            for head in range(n):
                if head == dep:
                    continue
                relation, score = self._relation_and_score(dep, head, tags, tokens, phrase_membership)
                if tags[dep] in _NOMINAL and tags[head] == "VERB":
                    score += 0.25
                    if (dep < head and relation == "nsubj") or (dep > head and relation == "obj"):
                        score += 0.22
                if tags[dep] in {"DET", "ADJ", "NUM", "ADP", "AUX"}:
                    score += 0.25 if relation in {"det", "amod", "nummod", "case", "aux"} else -0.4
                candidates.append((score, head, relation))
            candidates.sort(reverse=True)
            chosen = candidates[0] if candidates else (0.0, root, "dep")
            arcs.append({
                "dependent_index": dep, "dependent": tokens[dep],
                "head_index": chosen[1], "head": tokens[chosen[1]],
                "relation": chosen[2], "score": float(chosen[0]),
                "confidence": float(1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, chosen[0] - 2.0))))),
            })
        arcs = self._repair_dependency_cycles(arcs, tokens, root)
        roots = [arc for arc in arcs if arc["relation"] == "root"]
        valid = len(roots) == 1 and len(arcs) == n and all(
            arc["head_index"] is None or 0 <= int(arc["head_index"]) < n for arc in arcs
        )
        self._trace_stage(3, [
            {"candidate_head_pairs": n * max(n - 1, 0)},
            {"arc_features": ["distance", "direction", "POS compatibility", "phrase membership"]},
            {"relation_inventory": sorted({arc["relation"] for arc in arcs})},
            {"scored_arc_count": n * max(n - 1, 0)},
            {"root_index": root, "root_token": tokens[root]},
            {"selected_arc_count": len(arcs), "cycle_repairs": sum(a["relation"] == "dep:cycle_repair" for a in arcs)},
            {"valid_single_root_tree": valid, "root_count": len(roots)},
        ], drives={"temporal_syntax": 0.6, "dependency_relations": 0.9, "frontal_control": 0.45},
           trace=trace, simulator=simulator)
        return arcs

    @staticmethod
    def _repair_dependency_cycles(arcs, tokens, root):
        by_dep = {int(arc["dependent_index"]): arc for arc in arcs}
        for _ in range(len(tokens)):
            repaired = False
            for start in range(len(tokens)):
                path, seen, node = [], {}, start
                while node in by_dep and by_dep[node]["head_index"] is not None:
                    if node in seen:
                        cycle = path[seen[node]:]
                        victim = min(cycle, key=lambda idx: float(by_dep[idx]["score"]))
                        old_head = by_dep[victim]["head_index"]
                        by_dep[victim].update({
                            "head_index": root, "head": tokens[root], "relation": "dep:cycle_repair",
                            "score": 0.0, "confidence": 0.5, "cycle_repair_from": old_head,
                        })
                        repaired = True
                        break
                    seen[node] = len(path)
                    path.append(node)
                    node = int(by_dep[node]["head_index"])
                if repaired:
                    break
            if not repaired:
                break
        return [by_dep[i] for i in range(len(tokens))]

    @staticmethod
    def _unary_closure(nodes, limit):
        result = list(nodes)
        seen = {(node["label"], node["start"], node["end"], tuple(c["label"] for c in node["children"])) for node in result}
        cursor = 0
        while cursor < len(result) and len(result) < limit * 5:
            child = result[cursor]
            cursor += 1
            for parent, child_label, weight in _UNARY_RULES:
                if child["label"] != child_label:
                    continue
                key = (parent, child["start"], child["end"], (child_label,))
                if key in seen:
                    continue
                seen.add(key)
                result.append(_tree_node(
                    parent, child["start"], child["end"], [child],
                    token_index=child.get("token_index"),
                    score=float(child["score"] + math.log(max(weight, 1e-8))),
                ))
        return result

    def _constituency_parse(self, tokens, pos_candidates, selected, trace, simulator):
        n = len(tokens)
        chart: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for i in range(n):
            lexical_nodes = [
                _tree_node(str(item["tag"]), i, i + 1, [], token_index=i,
                           score=math.log(max(float(item["posterior"]), 1e-8)))
                for item in pos_candidates[i]
            ]
            chart[(i, i + 1)] = self._unary_closure(lexical_nodes, self.config.max_parse_alternatives)
        rule_uses = 0
        for width in range(2, n + 1):
            for start in range(0, n - width + 1):
                end = start + width
                generated: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
                for split in range(start + 1, end):
                    for left in chart.get((start, split), []):
                        for right in chart.get((split, end), []):
                            for parent, left_label, right_label, weight in _BINARY_RULES:
                                if left["label"] == left_label and right["label"] == right_label:
                                    node = _tree_node(
                                        parent, start, end, [left, right],
                                        score=float(left["score"] + right["score"] + math.log(max(weight, 1e-8))),
                                    )
                                    key = (parent, (left_label, right_label))
                                    if key not in generated or node["score"] > generated[key]["score"]:
                                        generated[key] = node
                                    rule_uses += 1
                nodes = self._unary_closure(list(generated.values()), self.config.max_parse_alternatives)
                grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for node in nodes:
                    grouped[node["label"]].append(node)
                chart[(start, end)] = []
                for hypotheses in grouped.values():
                    chart[(start, end)].extend(
                        sorted(hypotheses, key=lambda x: x["score"], reverse=True)[:self.config.max_parse_alternatives]
                    )
        full = chart.get((0, n), [])
        complete = [node for node in full if node["label"] == "S"]
        ranked = sorted(complete or full, key=lambda node: node["score"], reverse=True)
        if not ranked:
            leaves = [
                _tree_node(str(selected[i]["tag"]), i, i + 1, [], token_index=i,
                           score=math.log(max(float(selected[i]["confidence"]), 1e-8)))
                for i in range(n)
            ]
            ranked = [_tree_node("FRAG", 0, n, leaves, score=-float(n))]
        result, max_score = [], ranked[0]["score"]
        for rank, node in enumerate(ranked[:self.config.max_parse_alternatives], start=1):
            result.append({
                "rank": rank, "label": node["label"], "log_score": float(node["score"]),
                "relative_score": float(node["score"] - max_score), "tree": node,
            })
        self._trace_stage(4, [
            {"lexical_preterminal_count": sum(len(row) for row in pos_candidates)},
            {"chart_cell_count": n * (n + 1) // 2},
            {"binary_rule_activations": rule_uses},
            {"unary_rule_count": len(_UNARY_RULES), "unary_closure": "bounded/cycle-safe"},
            {"complete_parse_count": len(complete), "returned_alternatives": len(result)},
            {"fallback_used": not bool(complete), "fallback_label": result[0]["label"] if result else None},
        ], drives={"temporal_syntax": 0.7, "phrase_binding": 0.7, "frontal_control": 0.55},
           trace=trace, simulator=simulator)
        return result

    @staticmethod
    def _build_hierarchy(tree, arcs, phrases, tokens):
        arc_by_dep = {int(arc["dependent_index"]): dict(arc) for arc in arcs}
        def clone(node, depth=0, parent_id=None):
            start, end = int(node["start"]), int(node["end"])
            node_id = f"{node['label']}:{start}:{end}:{depth}"
            token_index = node.get("token_index")
            head_index = token_index if token_index is not None else next(
                (i for i in range(start, end) if i in arc_by_dep and arc_by_dep[i]["relation"] == "root"), start
            )
            result = {
                "node_id": node_id, "label": node["label"], "span": [start, end],
                "depth": depth, "parent_id": parent_id, "head_index": head_index,
                "token": tokens[token_index] if token_index is not None else None,
                "dependency": arc_by_dep.get(token_index) if token_index is not None else None,
                "children": [],
            }
            result["children"] = [clone(child, depth + 1, node_id) for child in node.get("children", [])]
            return result
        phrase_index = [
            {key: phrase[key] for key in ("phrase_id", "type", "start", "end", "head_index", "confidence")}
            for phrase in phrases
        ]
        return {
            "root": clone(tree), "token_count": len(tokens),
            "dependency_edges": [dict(arc) for arc in arcs], "phrase_spans": phrase_index,
            "representation": "nested constituency nodes aligned with dependency heads and phrase spans",
        }

    def _hierarchy_stage(self, parse_alternatives, arcs, phrases, tokens, trace, simulator):
        tree = dict(parse_alternatives[0]["tree"])
        hierarchy = self._build_hierarchy(tree, arcs, phrases, tokens)
        node_count, stack = 0, [tree]
        while stack:
            node = stack.pop()
            node_count += 1
            stack.extend(node.get("children", []))
        self._trace_stage(5, [
            {"root_label": tree["label"], "root_span": [tree["start"], tree["end"]]},
            {"hierarchy_node_count": node_count},
            {"dependency_leaf_alignment_count": len(arcs)},
            {"phrase_annotation_count": len(phrases), "maximum_depth": self._max_depth(tree)},
            {"serialized_hierarchy": True, "token_alignment_count": len(tokens)},
        ], drives={"phrase_binding": 0.85, "temporal_syntax": 0.65, "working_memory": 0.45},
           trace=trace, simulator=simulator)
        return tree, hierarchy

    @staticmethod
    def _max_depth(node):
        return 1 + max((SyntaxProcessor._max_depth(child) for child in node.get("children", [])), default=0)

    def _subject_object_assignment(self, tokens, selected, arcs, trace, simulator):
        tags = self._best_tags(selected)
        arc_by_dep = {int(arc["dependent_index"]): arc for arc in arcs}
        predicates, nominal = [], [i for i, tag in enumerate(tags) if tag in _NOMINAL]
        for verb_index, tag in enumerate(tags):
            if tag != "VERB":
                continue
            verb = _normalize(tokens[verb_index])
            dependents = [arc for arc in arcs if arc.get("head_index") == verb_index]
            subjects = [int(a["dependent_index"]) for a in dependents if str(a["relation"]).startswith(("nsubj", "csubj"))]
            objects = [int(a["dependent_index"]) for a in dependents if a["relation"] in {"obj", "dobj"}]
            indirect = [int(a["dependent_index"]) for a in dependents if a["relation"] == "iobj"]
            agents = [int(a["dependent_index"]) for a in dependents if str(a["relation"]).startswith("obl:agent")]
            if not subjects:
                left = [i for i in nominal if i < verb_index and abs(i - verb_index) <= 8]
                if left:
                    subjects = [max(left)]
            if not objects:
                right = [i for i in nominal if i > verb_index and abs(i - verb_index) <= 8 and i not in agents]
                if right:
                    objects = [min(right)]
            auxiliary_tokens = [
                _normalize(tokens[i]) for i in range(max(0, verb_index - 3), verb_index)
                if tags[i] == "AUX"
            ]
            passive = bool(
                any(aux in {"be", "am", "is", "are", "was", "were", "been", "being", "get", "got"} for aux in auxiliary_tokens)
                and (verb.endswith("ed") or verb.endswith("en") or verb in _PARTICIPLES)
            )
            if passive and not agents:
                by_indices = [
                    i + 1 for i in range(verb_index)
                    if _normalize(tokens[i]) == "by" and i + 1 < len(tokens) and tags[i + 1] in _NOMINAL
                ]
                agents = by_indices[-1:] if by_indices else []
            subjects_role, objects_role = ("patient", "agent") if passive else ("agent", "patient")
            predicates.append({
                "predicate_index": verb_index, "predicate": tokens[verb_index],
                "voice": "passive" if passive else "active_or_unmarked",
                "subjects": [{"token_index": i, "token": tokens[i], "grammatical_relation": "nsubj",
                              "provisional_role": subjects_role,
                              "source": "dependency_arc" if i in arc_by_dep and arc_by_dep[i].get("head_index") == verb_index else "local_order_fallback"}
                             for i in subjects],
                "objects": [{"token_index": i, "token": tokens[i], "grammatical_relation": "obj",
                             "provisional_role": objects_role,
                             "source": "dependency_arc" if i in arc_by_dep and arc_by_dep[i].get("head_index") == verb_index else "local_order_fallback"}
                            for i in objects],
                "indirect_objects": [{"token_index": i, "token": tokens[i], "grammatical_relation": "iobj",
                                      "provisional_role": "recipient"} for i in indirect],
                "oblique_agents": [{"token_index": i, "token": tokens[i], "grammatical_relation": "obl:agent",
                                    "provisional_role": "agent"} for i in agents],
            })
        self._trace_stage(6, [
            {"predicate_count": len(predicates), "predicate_indices": [p["predicate_index"] for p in predicates]},
            {"nominal_candidate_count": len(nominal)},
            {"subject_assignment_count": sum(len(p["subjects"]) for p in predicates)},
            {"object_assignment_count": sum(len(p["objects"]) for p in predicates)},
            {"passive_predicate_count": sum(p["voice"] == "passive" for p in predicates)},
            {"argument_frame_count": len(predicates)},
        ], drives={"dependency_relations": 0.75, "temporal_syntax": 0.55, "frontal_control": 0.6},
           trace=trace, simulator=simulator)
        return predicates

    def _thematic_roles(self, tokens, request, predicate_arguments, trace, simulator):
        roles = []
        for frame in predicate_arguments:
            verb_frame = self.verb_frames.get(_normalize(str(frame["predicate"])), {})
            for group in ("subjects", "objects", "indirect_objects", "oblique_agents"):
                for argument in frame[group]:
                    index = int(argument["token_index"])
                    features = request.semantic_features.get(index, {})
                    provisional = str(argument["provisional_role"])
                    options = ["agent", "patient"] + (["recipient"] if provisional == "recipient" else [])
                    scores, compatibilities = {}, {}
                    for role in options:
                        expectation = verb_frame.get(role, {})
                        compatibility = _cosine(features, expectation)
                        compatibilities[role] = compatibility
                        structural = 1.15 if role == provisional else 0.05
                        semantic = 0.55 * compatibility if compatibility is not None else 0.0
                        scores[role] = structural + semantic
                    probabilities = _softmax([scores[role] for role in options], 0.42)
                    ranked = sorted(
                        [{"role": role, "posterior": float(prob), "semantic_compatibility": compatibilities[role]}
                         for role, prob in zip(options, probabilities)],
                        key=lambda item: item["posterior"], reverse=True,
                    )
                    roles.append({
                        "predicate_index": int(frame["predicate_index"]), "predicate": frame["predicate"],
                        "argument_index": index, "argument": tokens[index],
                        "grammatical_relation": argument["grammatical_relation"],
                        "role": ranked[0]["role"], "confidence": ranked[0]["posterior"],
                        "role_candidates": ranked,
                        "evidence_source": argument.get("source", "grammatical_mapping"),
                        "semantic_features_supplied": bool(features),
                        "semantic_evidence_warning": None if features else "No caller semantic features; role is primarily syntax/voice based.",
                    })
        self._trace_stage(7, [
            {"verb_frame_count": len(predicate_arguments), "custom_or_default_frames": "caller override or compact defaults"},
            {"arguments_with_semantic_features": sum(bool(request.semantic_features.get(r["argument_index"])) for r in roles)},
            {"role_compatibility_scores": len(roles)},
            {"syntax_voice_evidence_used": len(roles)},
            {"role_distributions": len(roles), "temperature": 0.42},
            {"role_records": len(roles), "missing_semantic_feature_count": sum(not r["semantic_features_supplied"] for r in roles)},
        ], drives={"thematic_roles": 0.8, "dependency_relations": 0.45, "frontal_control": 0.4},
           trace=trace, simulator=simulator)
        return roles

    def _resolve_ambiguity(self, tokens, request, candidates, selected, predicate_arguments, trace, simulator):
        current = [{str(item["tag"]): float(item["posterior"]) for item in row} for row in candidates]
        initial_tags = self._best_tags(selected)
        predicate_indices = {int(frame["predicate_index"]) for frame in predicate_arguments}
        argument_indices = {
            int(argument["token_index"]) for frame in predicate_arguments
            for group in ("subjects", "objects", "indirect_objects", "oblique_agents")
            for argument in frame[group]
        }
        iteration_deltas, syntactic_support, semantic_support = [], [], []
        for index, token in enumerate(tokens):
            prev = _normalize(tokens[index - 1]) if index > 0 else ""
            nxt = _normalize(tokens[index + 1]) if index + 1 < len(tokens) else ""
            syn, sem = {}, {}
            for item in candidates[index]:
                tag = str(item["tag"])
                value = 0.0
                if prev in {"a", "an", "the", "this", "that", "these", "those"}:
                    value += 0.7 if tag in {"NOUN", "PROPN", "ADJ"} else -0.18
                if prev in {"did", "does", "do", "will", "can", "could", "may", "might", "must", "to"}:
                    value += 0.72 if tag == "VERB" else -0.10
                if prev in {"in", "on", "at", "by", "with", "from", "of", "for"}:
                    value += 0.55 if tag in _NOMINAL else -0.08
                if nxt in {"a", "an", "the", "this", "that"}:
                    value += 0.45 if tag in {"ADJ", "NOUN", "PROPN"} else 0.0
                if tag == "VERB" and index in predicate_indices:
                    value += 0.85
                if tag in _NOMINAL and index in argument_indices:
                    value += 0.62
                syn[tag] = value
                features = request.semantic_features.get(index, {})
                if features and tag in _NOMINAL:
                    sem[tag] = 0.35 * (features.get("ANIMATE", 0.0) + features.get("OBJECT", 0.0) + features.get("ENTITY", 0.0))
                elif features and tag == "VERB":
                    sem[tag] = 0.35 * (features.get("EVENT", 0.0) + features.get("ACTION", 0.0))
                else:
                    sem[tag] = 0.0
            syntactic_support.append(syn)
            semantic_support.append(sem)
        for _ in range(self.config.inference_iterations):
            total_delta, next_rows = 0.0, []
            for i, row in enumerate(current):
                tags = list(row)
                logits = [
                    math.log(max(row[tag], 1e-8))
                    + self.config.syntactic_gain * syntactic_support[i][tag]
                    + self.config.semantic_gain * semantic_support[i][tag]
                    for tag in tags
                ]
                target = _softmax(logits, self.config.syntax_temperature)
                updated = {
                    tag: (1.0 - self.config.recurrent_damping) * row[tag]
                         + self.config.recurrent_damping * probability
                    for tag, probability in zip(tags, target)
                }
                norm = sum(updated.values()) or 1.0
                updated = {tag: value / norm for tag, value in updated.items()}
                total_delta += sum(abs(updated[tag] - row[tag]) for tag in tags)
                next_rows.append(updated)
            current = next_rows
            iteration_deltas.append(total_delta)
        resolved_candidates, resolved, ambiguous = [], [], []
        for i, row in enumerate(current):
            ranked = sorted(
                [{"tag": tag, "posterior": probability,
                  "syntactic_support": syntactic_support[i][tag],
                  "semantic_support": semantic_support[i][tag]}
                 for tag, probability in row.items()],
                key=lambda item: item["posterior"], reverse=True,
            )
            resolved_candidates.append(ranked)
            margin = ranked[0]["posterior"] - (ranked[1]["posterior"] if len(ranked) > 1 else 0.0)
            abstained = ranked[0]["posterior"] < self.config.confidence_threshold or margin < self.config.ambiguity_margin
            resolved.append({
                "token_index": i, "token": tokens[i], "tag": ranked[0]["tag"],
                "confidence": ranked[0]["posterior"], "margin": margin, "abstained": abstained,
                "previous_tag": initial_tags[i], "changed": ranked[0]["tag"] != initial_tags[i],
            })
            if len(ranked) > 1 and margin < self.config.ambiguity_margin:
                ambiguous.append({
                    "token_index": i, "token": tokens[i], "alternatives": ranked[:4],
                    "status": "unresolved" if abstained else "resolved_with_low_margin",
                })
        self._trace_stage(8, [
            {"ambiguous_token_count": len(ambiguous), "token_count": len(tokens)},
            {"syntax_evidence_terms": sum(len(row) for row in syntactic_support)},
            {"semantic_evidence_terms": sum(bool(x) for x in request.semantic_features.values())},
            {"belief_update_iterations": len(iteration_deltas), "deltas_l1": iteration_deltas},
            {"posterior_entropy": [-sum(p * math.log(max(p, 1e-12)) for p in row.values()) for row in current]},
            {"selected_count": sum(not item["abstained"] for item in resolved),
             "abstained_count": sum(item["abstained"] for item in resolved)},
            {"alternatives_preserved": sum(len(row) for row in resolved_candidates),
             "changed_pos_count": sum(item["changed"] for item in resolved)},
        ], drives={"frontal_control": 0.82, "temporal_syntax": 0.75, "prediction_error": 0.4,
                   "working_memory": 0.35},
           trace=trace, simulator=simulator)
        return resolved_candidates, resolved, {
            "ambiguous_tokens": ambiguous, "iteration_deltas_l1": iteration_deltas,
            "converged": bool(iteration_deltas and iteration_deltas[-1] < 1e-3),
            "initial_pos": initial_tags, "final_pos": [item["tag"] for item in resolved],
            "abstention_count": sum(item["abstained"] for item in resolved),
        }

    def _release_resolved_dependency_memory(self, links):
        """Release only strongly linked dependencies and retain unresolved retrieval cues."""
        release_threshold = 0.75
        released = []
        retained = []
        for link in links:
            is_releasable = link["status"] == "linked" and float(link.get("score", 0.0)) >= release_threshold
            link["memory_status"] = "released_after_resolution" if is_releasable else "retained_for_retrieval"
            link["memory_release_threshold"] = release_threshold
            if is_releasable:
                released.append(link["filler_index"])
            else:
                retained.append(link["filler_index"])
        return {
            "release_threshold": release_threshold,
            "released_dependency_count": len(released),
            "retained_dependency_count": len(retained),
            "released_filler_indices": released,
            "retained_filler_indices": retained,
            "unresolved_cues_preserved": bool(retained),
        }

    def _manage_nested_competing_dependencies(self, links):
        """Annotate nested spans and competition groups without deleting alternative gap sites."""
        parents = list(range(len(links)))

        def find(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left, right):
            a, b = find(left), find(right)
            if a != b:
                parents[b] = a

        nested_pairs = []
        competing_pairs = []
        spans = []
        for link in links:
            head = link.get("head_index")
            filler = int(link["filler_index"])
            spans.append((min(filler, int(head)) if head is not None else filler,
                          max(filler, int(head)) if head is not None else filler))
        for left in range(len(links)):
            for right in range(left + 1, len(links)):
                a0, a1 = spans[left]
                b0, b1 = spans[right]
                nested = (a0 < b0 < b1 < a1) or (b0 < a0 < a1 < b1)
                same_filler = links[left]["filler_index"] == links[right]["filler_index"]
                same_target = (
                    links[left].get("head_index") is not None
                    and links[left].get("head_index") == links[right].get("head_index")
                )
                if nested:
                    nested_pairs.append((left, right))
                if same_filler or same_target:
                    competing_pairs.append((left, right))
                    union(left, right)
        groups = {}
        for index, link in enumerate(links):
            root = find(index)
            groups.setdefault(root, []).append(index)
        competition_group_for = {}
        group_number = 0
        for members in groups.values():
            if len(members) > 1:
                group_number += 1
                group_id = f"dependency-competition-{group_number}"
                for member in members:
                    competition_group_for[member] = group_id
        for index, link in enumerate(links):
            link["nested_with"] = [
                other if pair[0] == index else pair[0]
                for pair in nested_pairs for other in [pair[1]]
                if pair[0] == index or pair[1] == index
            ]
            link["competition_group"] = competition_group_for.get(index)
            sites = list(link.get("candidate_sites", ()))
            if sites:
                scores = [max(-50.0, min(50.0, float(site.get("score", 0.0)))) for site in sites]
                peak = max(scores)
                exponentials = [math.exp(score - peak) for score in scores]
                total = sum(exponentials) or 1.0
                for site, value in zip(sites, exponentials):
                    site["competition_probability"] = value / total
                link["candidate_sites"] = sites
            link["alternatives_preserved"] = len(sites) > 1 or link["status"] == "unresolved"
        return {
            "nested_pair_count": len(nested_pairs),
            "competing_pair_count": len(competing_pairs),
            "competition_group_count": group_number,
            "alternative_hypothesis_count": sum(len(item.get("candidate_sites", ())) for item in links),
            "unresolved_links_retained": sum(item["status"] == "unresolved" for item in links),
        }

    def _long_distance_dependencies(self, tokens, selected, predicate_arguments, trace, simulator):
        tags = self._best_tags(selected)
        predicates = [int(frame["predicate_index"]) for frame in predicate_arguments]
        frames_by_verb = {int(frame["predicate_index"]): frame for frame in predicate_arguments}
        fillers = []
        for i, token in enumerate(tokens):
            word = _normalize(token)
            relative_context = word in _RELATIVE_WORDS and i > 0 and any(
                tags[j] in _NOMINAL for j in range(max(0, i - 3), i)
            )
            question_context = word in _WH_WORDS and (
                i == 0 or any(_normalize(t) == "?" for t in tokens)
            )
            if relative_context or question_context:
                fillers.append((i, word, "relative" if relative_context else "wh_question"))
        links = []
        for filler_index, filler_word, filler_type in fillers:
            candidate_sites = []
            for predicate in predicates:
                if predicate <= filler_index:
                    continue
                frame = frames_by_verb[predicate]
                has_subject, has_object = bool(frame["subjects"]), bool(frame["objects"])
                is_who = filler_word in {"who", "whom", "whose"}
                if is_who and not has_subject:
                    role, role_fit = "nsubj", 1.1
                elif not is_who and not has_object:
                    role, role_fit = "obj", 1.15
                elif not is_who and not has_subject and filler_word in {"what", "which", "that"}:
                    role, role_fit = "nsubj", 0.82
                else:
                    continue
                expected_transitive = _normalize(tokens[predicate]) in {
                    "see", "saw", "eat", "ate", "read", "write", "wrote", "buy", "bought",
                    "build", "built", "make", "made", "find", "found", "watch", "chase", "open", "close",
                }
                score = role_fit + (0.45 if expected_transitive and role == "obj" else 0.0)
                score -= self.config.gap_distance_penalty * abs(predicate - filler_index)
                if filler_type == "relative":
                    score += 0.15
                intervening = [
                    j for j in range(filler_index + 1, predicate)
                    if tags[j] in _NOMINAL and j != filler_index
                ]
                if role == "obj" and intervening:
                    score += 0.22
                candidate_sites.append({
                    "predicate_index": predicate, "predicate": tokens[predicate],
                    "relation": role, "score": score, "intervening_nominals": intervening,
                })
            candidate_sites.sort(key=lambda item: (item["score"], item["predicate_index"]), reverse=True)
            best = candidate_sites[0] if candidate_sites else None
            if best and best["score"] > 0.25:
                links.append({
                    "filler_index": filler_index, "filler": tokens[filler_index],
                    "filler_type": filler_type, "head_index": best["predicate_index"],
                    "head": best["predicate"], "relation": best["relation"],
                    "distance_tokens": best["predicate_index"] - filler_index,
                    "score": float(best["score"]), "remote": True,
                    "candidate_sites": candidate_sites[:4], "status": "linked",
                    "basic_dependency_tree_modified": False,
                })
            else:
                links.append({
                    "filler_index": filler_index, "filler": tokens[filler_index],
                    "filler_type": filler_type, "head_index": None, "head": None,
                    "relation": None, "score": float(best["score"]) if best else 0.0,
                    "remote": True, "candidate_sites": candidate_sites[:4],
                    "status": "unresolved", "basic_dependency_tree_modified": False,
                })
        memory_release = self._release_resolved_dependency_memory(links)
        nested_competition = self._manage_nested_competing_dependencies(links)
        self._trace_stage(9, [
            {"filler_count": len(fillers), "fillers": [{"index": i, "word": w, "type": t} for i, w, t in fillers]},
            {"predicate_clause_candidates": len(predicates)},
            {"candidate_gap_count": sum(len(link["candidate_sites"]) for link in links)},
            {"gap_scoring": "role fit + valency + intervening-subject cue - distance penalty"},
            {"remote_link_count": sum(link["status"] == "linked" for link in links)},
            {"unresolved_filler_count": sum(link["status"] == "unresolved" for link in links)},
            {"memory_release": memory_release},
            {"nested_and_competing_dependencies": nested_competition},
        ], drives={"working_memory": 0.9, "frontal_control": 0.82, "dependency_relations": 0.55,
                   "prediction_error": 0.5},
           trace=trace, simulator=simulator)
        return links

    def process(self, request: SyntaxRequest) -> SyntaxResult:
        tokens, original_text = self._tokenize(request)
        token_records = [
            {"token_index": i, "surface": token, "normalized": _normalize(token),
             "punctuation": bool(token and not any(ch.isalnum() for ch in token))}
            for i, token in enumerate(tokens)
        ]
        trace = []
        simulator = SyntaxLIFSimulator(self.config) if self.config.simulate_biology else None
        pos_candidates, selected_pos = self._pos_identification(tokens, request, trace, simulator)
        phrases = self._phrase_detection(tokens, selected_pos, trace, simulator)
        dependency_arcs = self._dependency_parse(tokens, selected_pos, phrases, trace, simulator)
        constituency_parses = self._constituency_parse(tokens, pos_candidates, selected_pos, trace, simulator)
        constituency_tree, hierarchy = self._hierarchy_stage(
            constituency_parses, dependency_arcs, phrases, tokens, trace, simulator
        )
        predicate_arguments = self._subject_object_assignment(
            tokens, selected_pos, dependency_arcs, trace, simulator
        )
        thematic_roles = self._thematic_roles(tokens, request, predicate_arguments, trace, simulator)
        resolved_candidates, resolved_pos, ambiguity = self._resolve_ambiguity(
            tokens, request, pos_candidates, selected_pos, predicate_arguments, trace, simulator
        )

        # Reconcile structural outputs if recurrent ambiguity resolution changed POS decisions.
        if any(item["changed"] for item in resolved_pos):
            phrases = self._phrase_detection_reconcile(tokens, resolved_pos, phrases)
            dependency_arcs = self._dependency_parse_reconcile(tokens, resolved_pos, phrases)
            constituency_parses = self._constituency_parse_reconcile(tokens, resolved_candidates, resolved_pos)
            constituency_tree = dict(constituency_parses[0]["tree"])
            hierarchy = self._build_hierarchy(constituency_tree, dependency_arcs, phrases, tokens)
            predicate_arguments = self._subject_object_reconcile(tokens, resolved_pos, dependency_arcs)
            thematic_roles = self._thematic_reconcile(tokens, request, predicate_arguments)

        long_distance = self._long_distance_dependencies(
            tokens, resolved_pos, predicate_arguments, trace, simulator
        )
        biological = simulator.snapshot() if simulator else {
            "enabled": False, "reason": "simulate_biology=False",
            "scope": "The functional syntax parser still executes; no LIF dynamics were run.",
        }
        root_count = sum(arc["relation"] == "root" for arc in dependency_arcs)
        valid_tree = (
            len(dependency_arcs) == len(tokens) and root_count == 1
            and all(arc["head_index"] is None or 0 <= int(arc["head_index"]) < len(tokens) for arc in dependency_arcs)
        )
        diagnostics = {
            "valid": bool(valid_tree and len(trace) == len(OPERATION_REGISTRY)),
            "dependency_tree_valid": bool(valid_tree), "root_count": root_count,
            "token_count": len(tokens), "operation_count": len(trace),
            "expected_operation_count": len(OPERATION_REGISTRY),
            "operation_order_valid": [item["operation_id"] for item in trace]
                == [item.operation_id for item in OPERATION_REGISTRY],
            "complete_constituency_parse": bool(constituency_parses and constituency_parses[0]["label"] == "S"),
            "unresolved_pos_count": sum(bool(item["abstained"]) for item in resolved_pos),
            "unresolved_long_distance_count": sum(link["status"] == "unresolved" for link in long_distance),
            "released_dependency_memory_count": sum(link.get("memory_status") == "released_after_resolution" for link in long_distance),
            "nested_dependency_links": sum(bool(link.get("nested_with")) for link in long_distance),
            "competing_dependency_groups": len({link.get("competition_group") for link in long_distance if link.get("competition_group")}),
            "warnings": [
                "The grammar and lexical inventory are compact, rule-based and English-oriented.",
                "Posterior-like values are model scores, not corpus-calibrated probabilities.",
                "LIF activity is a coarse reference simulation, not evidence of neural equivalence.",
            ],
        }
        metadata = {
            "language": request.language, "source_reference": request.source_reference,
            "lexicon_entries": len(self.lexicon), "registry": "ORE-SYNTAX-56",
            "configuration": self.config.to_dict(),
            "model_scope": "rule-based syntax prototype; caller evidence is read-only",
        }
        return SyntaxResult(
            sequence_id=request.sequence_id, original_text=original_text, tokens=list(tokens),
            token_records=token_records, pos_candidates=resolved_candidates, selected_pos=resolved_pos,
            phrases=phrases, dependency_arcs=dependency_arcs,
            constituency_parses=constituency_parses, constituency_tree=constituency_tree,
            hierarchy=hierarchy, predicate_arguments=predicate_arguments, thematic_roles=thematic_roles,
            ambiguity=ambiguity, long_distance_dependencies=long_distance, biological=biological,
            operation_trace=trace, diagnostics=diagnostics, metadata=metadata,
        )

    # These reconciliation passes avoid duplicate registry events.
    def _phrase_detection_reconcile(self, tokens, selected, previous):
        tags, phrases, seen = self._best_tags(selected), [], set()
        for head, tag in enumerate(tags):
            if tag not in _NOMINAL:
                continue
            start, end = head, head + 1
            while start > 0 and tags[start - 1] in {"DET", "ADJ", "NUM", "NOUN", "PROPN"}:
                start -= 1
            while end < len(tags) and tags[end] in {"NOUN", "PROPN"}:
                end += 1
            key = ("NP", start, end)
            if key not in seen:
                seen.add(key)
                phrases.append({"phrase_id": f"NP-{start}-{end}", "type": "NP", "start": start,
                                "end": end, "head_index": head, "text": " ".join(tokens[start:end]),
                                "confidence": float(selected[head]["confidence"]), "rule": "post-ambiguity reconciliation"})
        for i, tag in enumerate(tags):
            if tag in {"VERB", "AUX"}:
                start = i
                while start > 0 and tags[start - 1] in {"AUX", "ADV"}:
                    start -= 1
                end = i + 1
                while end < len(tags) and tags[end] in {"AUX", "VERB", "ADV", "PART"}:
                    end += 1
                key = ("VP", start, end)
                if key not in seen:
                    seen.add(key)
                    phrases.append({"phrase_id": f"VP-{start}-{end}", "type": "VP", "start": start,
                                    "end": end, "head_index": i, "text": " ".join(tokens[start:end]),
                                    "confidence": float(selected[i]["confidence"]), "rule": "post-ambiguity reconciliation"})
        for phrase in previous:
            if phrase["type"] in {"PP", "ADJP", "ADVP"} and (phrase["start"], phrase["end"]) not in {
                (p["start"], p["end"]) for p in phrases
            }:
                phrases.append(dict(phrase))
        return sorted(phrases, key=lambda p: (p["start"], -(p["end"] - p["start"]), p["type"]))

    def _dependency_parse_reconcile(self, tokens, selected, phrases):
        tags, n = self._best_tags(selected), len(tokens)
        verbs = [i for i, tag in enumerate(tags) if tag == "VERB"] or [i for i, tag in enumerate(tags) if tag == "AUX"]
        root = max(verbs, key=lambda i: float(selected[i]["confidence"])) if verbs else max(
            range(n), key=lambda i: float(selected[i]["confidence"])
        )
        membership = set()
        for phrase in phrases:
            head = phrase.get("head_index")
            if head is not None:
                membership.update((i, int(head)) for i in range(int(phrase["start"]), int(phrase["end"])) if i != int(head))
        arcs = []
        for dep in range(n):
            if dep == root:
                arcs.append({"dependent_index": dep, "dependent": tokens[dep], "head_index": None,
                             "head": "ROOT", "relation": "root", "score": 5.0, "confidence": 1.0})
                continue
            choices = []
            for head in range(n):
                if head != dep:
                    rel, score = self._relation_and_score(dep, head, tags, tokens, membership)
                    choices.append((score, head, rel))
            score, head, rel = max(choices, default=(0.0, root, "dep"))
            arcs.append({"dependent_index": dep, "dependent": tokens[dep], "head_index": head,
                         "head": tokens[head], "relation": rel, "score": float(score),
                         "confidence": float(1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, score - 2.0)))))})
        return self._repair_dependency_cycles(arcs, tokens, root)

    def _constituency_parse_reconcile(self, tokens, candidates, selected):
        n, chart = len(tokens), {}
        for i in range(n):
            leaves = [_tree_node(str(item["tag"]), i, i + 1, [], token_index=i,
                                 score=math.log(max(float(item["posterior"]), 1e-8)))
                     for item in candidates[i]]
            chart[(i, i + 1)] = self._unary_closure(leaves, self.config.max_parse_alternatives)
        for width in range(2, n + 1):
            for start in range(n - width + 1):
                end, generated = start + width, {}
                for split in range(start + 1, end):
                    for left in chart.get((start, split), []):
                        for right in chart.get((split, end), []):
                            for parent, llabel, rlabel, weight in _BINARY_RULES:
                                if left["label"] == llabel and right["label"] == rlabel:
                                    key = (parent, llabel, rlabel)
                                    node = _tree_node(parent, start, end, [left, right],
                                                      score=left["score"] + right["score"] + math.log(max(weight, 1e-8)))
                                    if key not in generated or node["score"] > generated[key]["score"]:
                                        generated[key] = node
                nodes = self._unary_closure(list(generated.values()), self.config.max_parse_alternatives)
                grouped = defaultdict(list)
                for node in nodes:
                    grouped[node["label"]].append(node)
                chart[(start, end)] = [
                    node for label in grouped
                    for node in sorted(grouped[label], key=lambda x: x["score"], reverse=True)[:self.config.max_parse_alternatives]
                ]
        full = chart.get((0, n), [])
        complete = [node for node in full if node["label"] == "S"]
        ranked = sorted(complete or full, key=lambda node: node["score"], reverse=True)
        if not ranked:
            ranked = [_tree_node("FRAG", 0, n, [
                _tree_node(str(selected[i]["tag"]), i, i + 1, [], token_index=i,
                           score=math.log(max(float(selected[i]["confidence"]), 1e-8)))
                for i in range(n)
            ], score=-float(n))]
        top = ranked[0]["score"]
        return [{"rank": i + 1, "label": node["label"], "log_score": float(node["score"]),
                 "relative_score": float(node["score"] - top), "tree": node}
                for i, node in enumerate(ranked[:self.config.max_parse_alternatives])]

    def _subject_object_reconcile(self, tokens, selected, arcs):
        tags = self._best_tags(selected)
        frames, nominals = [], [i for i, tag in enumerate(tags) if tag in _NOMINAL]
        for verb_index, tag in enumerate(tags):
            if tag != "VERB":
                continue
            deps = [a for a in arcs if a["head_index"] == verb_index]
            subjects = [int(a["dependent_index"]) for a in deps if str(a["relation"]).startswith("nsubj")]
            objects = [int(a["dependent_index"]) for a in deps if a["relation"] == "obj"]
            if not subjects:
                left = [i for i in nominals if i < verb_index]
                subjects = [max(left)] if left else []
            if not objects:
                right = [i for i in nominals if i > verb_index]
                objects = [min(right)] if right else []
            aux = [_normalize(tokens[i]) for i in range(max(0, verb_index - 3), verb_index) if tags[i] == "AUX"]
            passive = bool(any(a in {"is", "are", "was", "were", "been", "being", "get", "got"} for a in aux)
                           and (tokens[verb_index].casefold().endswith(("ed", "en")) or _normalize(tokens[verb_index]) in _PARTICIPLES))
            frames.append({
                "predicate_index": verb_index, "predicate": tokens[verb_index],
                "voice": "passive" if passive else "active_or_unmarked",
                "subjects": [{"token_index": i, "token": tokens[i], "grammatical_relation": "nsubj",
                              "provisional_role": "patient" if passive else "agent",
                              "source": "reconciled_dependency_or_order"} for i in subjects],
                "objects": [{"token_index": i, "token": tokens[i], "grammatical_relation": "obj",
                             "provisional_role": "agent" if passive else "patient",
                             "source": "reconciled_dependency_or_order"} for i in objects],
                "indirect_objects": [], "oblique_agents": [],
            })
        return frames

    def _thematic_reconcile(self, tokens, request, frames):
        roles = []
        for frame in frames:
            verb_frame = self.verb_frames.get(_normalize(frame["predicate"]), {})
            for group in ("subjects", "objects", "indirect_objects", "oblique_agents"):
                for argument in frame[group]:
                    index = int(argument["token_index"])
                    features = request.semantic_features.get(index, {})
                    role = argument["provisional_role"]
                    compatibility = _cosine(features, verb_frame.get(role, {}))
                    confidence = 0.78 if compatibility is None else min(0.98, 0.72 + 0.2 * compatibility)
                    roles.append({
                        "predicate_index": frame["predicate_index"], "predicate": frame["predicate"],
                        "argument_index": index, "argument": tokens[index],
                        "grammatical_relation": argument["grammatical_relation"], "role": role,
                        "confidence": confidence,
                        "role_candidates": [{"role": role, "posterior": confidence,
                                             "semantic_compatibility": compatibility}],
                        "evidence_source": argument["source"], "semantic_features_supplied": bool(features),
                        "semantic_evidence_warning": None if features else "No caller semantic features; syntax/voice mapping dominates.",
                    })
        return roles


OREParser = SyntaxProcessor
