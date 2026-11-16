"""Persistent episodic word-form memory and experience-driven lexical prototypes.

Learning is explicit: semantic inference never writes memory. This module provides
a SQLite-backed implementation boundary for capture, encoding, pattern separation,
binding, storage, retrieval, consolidation and prototype strengthening.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import re
import sqlite3
import unicodedata
import uuid
from typing import Any, Mapping, Sequence

from .contracts import WordFormEpisode


MEMORY_OPERATION_REGISTRY = (
    {"operation_id": "SM-001", "source_step_id": "29", "name": "Capture current word-form episode"},
    {"operation_id": "SM-002", "source_step_id": "30", "name": "Encode episode into memory representation"},
    {"operation_id": "SM-003", "source_step_id": "31", "name": "Separate similar episodic traces"},
    {"operation_id": "SM-004", "source_step_id": "32", "name": "Bind orthographic, phonological, semantic and context features"},
    {"operation_id": "SM-005", "source_step_id": "33", "name": "Store durable episodic memory trace"},
    {"operation_id": "SM-006", "source_step_id": "34", "name": "Retrieve matching episodic traces"},
    {"operation_id": "SM-007", "source_step_id": "35", "name": "Perform pattern completion from partial cues"},
    {"operation_id": "SM-008", "source_step_id": "36", "name": "Send retrieved episodic information into semantic processing"},
    {"operation_id": "SM-009", "source_step_id": "37", "name": "Collect multiple word-form experiences"},
    {"operation_id": "SM-010", "source_step_id": "38", "name": "Identify common features across episodes"},
    {"operation_id": "SM-011", "source_step_id": "39", "name": "Build abstract orthographic representation"},
    {"operation_id": "SM-012", "source_step_id": "40", "name": "Build abstract phonological representation"},
    {"operation_id": "SM-013", "source_step_id": "41", "name": "Construct abstract lexical identity"},
    {"operation_id": "SM-014", "source_step_id": "42", "name": "Strengthen prototype through experience"},
    {"operation_id": "SM-015", "source_step_id": "43", "name": "Consolidate abstract representation"},
    {"operation_id": "SM-016", "source_step_id": "44", "name": "Generalize across new experiences"},
)


def _normalize(text: str) -> str:
    """Normalize a word form without discarding the original spelling stored in memory."""
    return re.sub(r"^\W+|\W+$", "", unicodedata.normalize("NFKC", text).casefold(), flags=re.UNICODE)


def _ngrams(text: str, n: int = 2) -> dict[str, float]:
    """Return a normalized character n-gram profile for orthographic generalization."""
    padded = f"^{text}$"
    if len(padded) < n:
        return {padded: 1.0}
    counts: dict[str, float] = {}
    for index in range(len(padded) - n + 1):
        gram = padded[index:index + n]
        counts[gram] = counts.get(gram, 0.0) + 1.0
    total = sum(counts.values()) or 1.0
    return {key: value / total for key, value in counts.items()}


def _cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Calculate cosine similarity for sparse feature maps."""
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    a = [float(left.get(key, 0.0)) for key in keys]
    b = [float(right.get(key, 0.0)) for key in keys]
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _json(value: Any) -> str:
    """Serialize stored feature values deterministically."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))



@dataclass(frozen=True, slots=True)
class SemanticPrototypeEvidence:
    """Abstract lexical evidence derived from consolidated experiences, not a new episode."""

    episode_id: str
    orthography: str
    concept_id: str | None
    phonological_form: tuple[str, ...]
    semantic_features: Mapping[str, float]
    context_features: Mapping[str, float]
    strength: float
    recency: float
    source_reference: str
    evidence_type: str = "abstract_lexical_prototype"
    prototype_id: str = ""


def _numeric_features(values: Mapping[str, Any]) -> dict[str, float]:
    """Keep only finite numeric feature entries from an optional context mapping."""
    features: dict[str, float] = {}
    for key, raw in values.items():
        if isinstance(raw, bool):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            features[str(key)] = value
    return features


def _sequence_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    """Return normalized edit similarity for ordered phoneme sequences."""
    a, b = list(left), list(right)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            current.append(min(
                current[-1] + 1, previous[j] + 1,
                previous[j - 1] + (item_a != item_b),
            ))
        previous = current
    return max(0.0, 1.0 - previous[-1] / max(len(a), len(b), 1))


class SemanticMemoryStore:
    """Persist episodic word-form memories and consolidate them into lexical prototypes.

    The SQLite database is deliberately separate from SemanticCognitionProcessor's
    read-only inference path. Call store_episode() or capture_word_form() to learn.
    """

    def __init__(self, database_path: str = "ore_semantic_memory.sqlite3") -> None:
        """Open or create a SQLite memory database at database_path."""
        self.database_path = str(database_path)
        self._connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._trace: list[dict[str, Any]] = []
        self._initialize_schema()

    @property
    def operation_trace(self) -> list[dict[str, Any]]:
        """Return a copy of the most recent memory-learning and retrieval trace."""
        return [dict(row) for row in self._trace]

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._connection.close()

    def _initialize_schema(self) -> None:
        """Create normalized episode, prototype and operation-ledger tables."""
        self._connection.executescript(
            """
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                normalized_form TEXT NOT NULL,
                orthography TEXT NOT NULL,
                concept_id TEXT,
                prototype_id TEXT NOT NULL,
                phonological_form TEXT NOT NULL,
                semantic_features TEXT NOT NULL,
                context_features TEXT NOT NULL,
                strength REAL NOT NULL CHECK(strength >= 0 AND strength <= 1),
                recency REAL NOT NULL CHECK(recency >= 0 AND recency <= 1),
                source_reference TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_episode_form ON episodes(normalized_form);
            CREATE INDEX IF NOT EXISTS idx_episode_concept ON episodes(concept_id);
            CREATE INDEX IF NOT EXISTS idx_episode_prototype ON episodes(prototype_id);
            CREATE TABLE IF NOT EXISTS prototypes (
                prototype_id TEXT PRIMARY KEY,
                experience_count INTEGER NOT NULL,
                orthographic_profile TEXT NOT NULL,
                phonological_profile TEXT NOT NULL,
                phonological_sequence TEXT NOT NULL DEFAULT '[]',
                semantic_features TEXT NOT NULL,
                context_features TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_operation_ledger (
                ledger_id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_id TEXT NOT NULL,
                episode_id TEXT,
                status TEXT NOT NULL,
                evidence TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        columns = {
            row["name"] for row in self._connection.execute("PRAGMA table_info(prototypes)")
        }
        if "phonological_sequence" not in columns:
            self._connection.execute(
                "ALTER TABLE prototypes ADD COLUMN phonological_sequence TEXT NOT NULL DEFAULT '[]'"
            )
        self._connection.commit()

    def _record(self, operation_id: str, status: str, evidence: Mapping[str, Any],
                episode_id: str | None = None) -> None:
        """Append and persist one memory operation with its own execution evidence."""
        row = {
            "operation_id": operation_id,
            "status": status,
            "episode_id": episode_id,
            "evidence": dict(evidence),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._trace.append(row)
        self._connection.execute(
            "INSERT INTO memory_operation_ledger(operation_id, episode_id, status, evidence, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (operation_id, episode_id, status, _json(evidence), row["timestamp"]),
        )
        self._connection.commit()

    def capture_word_form(
        self,
        orthography: str,
        *,
        concept_id: str | None = None,
        phonological_form: Sequence[str] = (),
        semantic_features: Mapping[str, float] | None = None,
        context_features: Mapping[str, float] | None = None,
        strength: float = 0.75,
        recency: float = 1.0,
        source_reference: str = "observed_experience",
        prototype_id: str | None = None,
    ) -> WordFormEpisode:
        """Capture a new word-form encounter and send it through explicit encoding and storage."""
        if not isinstance(orthography, str) or not orthography.strip():
            raise ValueError("orthography must be a non-empty string")
        episode = WordFormEpisode(
            episode_id=uuid.uuid4().hex,
            orthography=orthography,
            concept_id=concept_id,
            phonological_form=tuple(str(phone) for phone in phonological_form),
            semantic_features=dict(semantic_features or {}),
            context_features=dict(context_features or {}),
            strength=float(strength),
            recency=float(recency),
            source_reference=source_reference,
        )
        self._record("SM-001", "executed", {
            "orthography": orthography, "concept_id": concept_id,
            "phoneme_count": len(episode.phonological_form),
            "semantic_feature_count": len(episode.semantic_features),
            "context_feature_count": len(episode.context_features),
        }, episode.episode_id)
        self.store_episode(episode, prototype_id=prototype_id)
        return episode

    def _encode(self, episode: WordFormEpisode) -> dict[str, Any]:
        """Build a multimodal memory code while retaining independently addressable channels."""
        normalized = _normalize(episode.orthography)
        encoded = {
            "normalized_form": normalized,
            "orthographic_profile": _ngrams(normalized),
            "phonological_profile": {phone: episode.phonological_form.count(phone)
                                     for phone in set(episode.phonological_form)},
            "semantic_features": dict(episode.semantic_features),
            "context_features": dict(episode.context_features),
            "concept_id": episode.concept_id,
        }
        self._record("SM-002", "executed", {
            "normalized_form": normalized,
            "orthographic_ngram_count": len(encoded["orthographic_profile"]),
            "phonological_unit_count": len(encoded["phonological_profile"]),
            "semantic_feature_count": len(encoded["semantic_features"]),
            "context_feature_count": len(encoded["context_features"]),
        }, episode.episode_id)
        return encoded

    def _pattern_separation(self, episode: WordFormEpisode, encoded: Mapping[str, Any],
                            threshold: float = 0.96) -> tuple[str, bool, float]:
        """Reuse only near-identical encounters; keep similar but distinct experiences separate."""
        rows = self._connection.execute(
            "SELECT episode_id, phonological_form, context_features FROM episodes "
            "WHERE normalized_form=? AND COALESCE(concept_id,'')=COALESCE(?,'') "
            "ORDER BY created_at DESC LIMIT 50",
            (encoded["normalized_form"], episode.concept_id),
        ).fetchall()
        best_id: str | None = None
        best_fit = 0.0
        for row in rows:
            old_phones = tuple(json.loads(row["phonological_form"]))
            phone_fit = 1.0 if old_phones == episode.phonological_form else 0.0
            old_context = json.loads(row["context_features"])
            context_fit = _cosine(old_context, episode.context_features)
            fit = 0.65 * phone_fit + 0.35 * context_fit
            if fit > best_fit:
                best_id, best_fit = row["episode_id"], fit
        duplicate = best_id is not None and best_fit >= threshold
        selected_id = best_id if duplicate else episode.episode_id
        self._record("SM-003", "merged_duplicate" if duplicate else "separated", {
            "candidate_episode_count": len(rows), "best_similarity": float(best_fit),
            "threshold": threshold, "distinct_episode_preserved": not duplicate,
        }, selected_id)
        return selected_id, duplicate, float(best_fit)

    def store_episode(self, episode: WordFormEpisode, *,
                      prototype_id: str | None = None) -> dict[str, Any]:
        """Encode, pattern-separate, bind and durably store one episodic word-form trace."""
        if not isinstance(episode, WordFormEpisode):
            raise TypeError("episode must be a WordFormEpisode")
        encoded = self._encode(episode)
        selected_id, duplicate, similarity = self._pattern_separation(episode, encoded)
        resolved_prototype = prototype_id or (
            f"concept:{episode.concept_id}" if episode.concept_id else f"form:{encoded['normalized_form']}"
        )
        bound = {
            "episode_id": selected_id,
            "orthography": episode.orthography,
            "normalized_form": encoded["normalized_form"],
            "concept_id": episode.concept_id,
            "prototype_id": resolved_prototype,
            "phonological_form": list(episode.phonological_form),
            "semantic_features": dict(episode.semantic_features),
            "context_features": dict(episode.context_features),
            "strength": float(episode.strength),
            "recency": float(episode.recency),
            "source_reference": episode.source_reference,
        }
        self._record("SM-004", "executed", {
            "bound_channels": ["orthography", "phonology", "semantics", "context"],
            "prototype_id": resolved_prototype,
        }, selected_id)
        now = datetime.now(timezone.utc).isoformat()
        if duplicate:
            self._connection.execute(
                "UPDATE episodes SET strength=MIN(1.0, MAX(strength, ?)), "
                "recency=MAX(recency, ?), updated_at=? WHERE episode_id=?",
                (episode.strength, episode.recency, now, selected_id),
            )
        else:
            self._connection.execute(
                """INSERT INTO episodes(
                    episode_id, normalized_form, orthography, concept_id, prototype_id,
                    phonological_form, semantic_features, context_features, strength,
                    recency, source_reference, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (selected_id, encoded["normalized_form"], episode.orthography,
                 episode.concept_id, resolved_prototype, _json(list(episode.phonological_form)),
                 _json(dict(episode.semantic_features)), _json(dict(episode.context_features)),
                 episode.strength, episode.recency, episode.source_reference, now, now),
            )
        self._connection.commit()
        self._record("SM-005", "merged" if duplicate else "stored", {
            "durable_storage": True, "prototype_id": resolved_prototype,
            "pattern_similarity": similarity,
        }, selected_id)
        return {**bound, "episode_id": selected_id, "duplicate_merged": duplicate,
                "pattern_similarity": similarity}

    def recall(self, query: str, *, context_features: Mapping[str, float] | None = None,
               concept_id: str | None = None, limit: int = 8) -> list[WordFormEpisode]:
        """Retrieve episodic traces using surface overlap, context fit and optional concept identity."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if limit < 1:
            raise ValueError("limit must be positive")
        normalized = _normalize(query)
        rows = self._connection.execute(
            "SELECT * FROM episodes WHERE (? IS NULL OR concept_id=?)",
            (concept_id, concept_id),
        ).fetchall()
        query_profile = _ngrams(normalized)
        context_features = _numeric_features(context_features or {})
        ranked: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            form_fit = _cosine(query_profile, _ngrams(row["normalized_form"]))
            exact_bonus = 1.0 if normalized == row["normalized_form"] else 0.0
            context_fit = _cosine(context_features or {}, json.loads(row["context_features"]))
            score = 0.70 * form_fit + 0.20 * exact_bonus + 0.10 * context_fit
            score *= 0.5 + 0.5 * float(row["strength"]) * float(row["recency"])
            ranked.append((score, row))
        ranked.sort(key=lambda pair: (pair[0], pair[1]["created_at"]), reverse=True)
        selected = [row for score, row in ranked[:limit] if score > 0.0]
        self._record("SM-006", "executed", {
            "query": query, "candidate_count": len(rows),
            "returned_count": len(selected), "limit": limit,
        })
        return [
            WordFormEpisode(
                episode_id=row["episode_id"], orthography=row["orthography"],
                concept_id=row["concept_id"],
                phonological_form=tuple(json.loads(row["phonological_form"])),
                semantic_features=json.loads(row["semantic_features"]),
                context_features=json.loads(row["context_features"]),
                strength=float(row["strength"]), recency=float(row["recency"]),
                source_reference=row["source_reference"],
            )
            for row in selected
        ]

    def pattern_complete(
        self,
        *,
        orthography: str | None = None,
        phonological_form: Sequence[str] = (),
        semantic_features: Mapping[str, float] | None = None,
        context_features: Mapping[str, float] | None = None,
        concept_id: str | None = None,
        minimum_similarity: float = 0.15,
    ) -> dict[str, Any]:
        """Complete missing channels from the best-supported stored episode using partial cues."""
        normalized = _normalize(orthography or "")
        phones = tuple(str(phone) for phone in phonological_form)
        semantic = _numeric_features(semantic_features or {})
        context = _numeric_features(context_features or {})
        if not (normalized or phones or semantic or context or concept_id):
            raise ValueError("pattern completion requires at least one non-empty cue")
        if not math.isfinite(minimum_similarity) or not 0.0 <= minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must lie in [0, 1]")
        rows = self._connection.execute(
            "SELECT * FROM episodes WHERE (? IS NULL OR concept_id=?)",
            (concept_id, concept_id),
        ).fetchall()
        query_profile = _ngrams(normalized) if normalized else {}
        cue_weights = {
            "orthography": 0.45 if normalized else 0.0,
            "phonology": 0.25 if phones else 0.0,
            "semantics": 0.20 if semantic else 0.0,
            "context": 0.10 if context else 0.0,
        }
        active_weight = sum(cue_weights.values()) or 1.0
        ranked: list[tuple[float, dict[str, float], sqlite3.Row]] = []
        for row in rows:
            channels: dict[str, float] = {}
            if normalized:
                channels["orthography"] = _cosine(query_profile, _ngrams(row["normalized_form"]))
            if phones:
                channels["phonology"] = _sequence_similarity(
                    phones, tuple(json.loads(row["phonological_form"]))
                )
            if semantic:
                channels["semantics"] = _cosine(
                    semantic, json.loads(row["semantic_features"])
                )
            if context:
                channels["context"] = _cosine(context, json.loads(row["context_features"]))
            score = sum(cue_weights[key] * channels.get(key, 0.0) for key in cue_weights) / active_weight
            score *= 0.65 + 0.35 * float(row["strength"]) * float(row["recency"])
            ranked.append((score, channels, row))
        ranked.sort(key=lambda item: (item[0], item[2]["created_at"]), reverse=True)
        if not ranked or ranked[0][0] < minimum_similarity:
            result = {
                "status": "no_supported_completion",
                "similarity": float(ranked[0][0]) if ranked else 0.0,
                "completed_episode": None,
                "completed_channels": [],
                "candidate_count": len(rows),
            }
            self._record("SM-007", "no_supported_completion", {
                "cue_channels": [key for key, value in cue_weights.items() if value],
                "minimum_similarity": minimum_similarity,
                "candidate_count": len(rows),
            })
            return result
        score, channel_scores, row = ranked[0]
        episode = WordFormEpisode(
            episode_id=row["episode_id"],
            orthography=row["orthography"],
            concept_id=row["concept_id"],
            phonological_form=tuple(json.loads(row["phonological_form"])),
            semantic_features=json.loads(row["semantic_features"]),
            context_features=json.loads(row["context_features"]),
            strength=float(row["strength"]),
            recency=float(row["recency"]),
            source_reference=row["source_reference"],
        )
        supplied = {
            "orthography": bool(normalized), "phonology": bool(phones),
            "semantics": bool(semantic), "context": bool(context),
        }
        completed = [key for key, observed in supplied.items() if not observed]
        result = {
            "status": "completed",
            "similarity": float(score),
            "channel_scores": channel_scores,
            "completed_episode": episode,
            "completed_episode_id": episode.episode_id,
            "completed_channels": completed,
            "candidate_count": len(rows),
            "completion_source": "stored_episodic_trace",
            "no_unobserved_feature_invention": True,
        }
        self._record("SM-007", "executed", {
            "cue_channels": [key for key, value in cue_weights.items() if value],
            "channel_scores": channel_scores,
            "selected_episode_id": episode.episode_id,
            "completed_channels": completed,
            "similarity": float(score),
        }, episode.episode_id)
        return result

    def record_semantic_feedback(
        self, query: str, *, episodic_ids: Sequence[str] = (),
        prototype_ids: Sequence[str] = (), evidence_count: int = 0,
    ) -> None:
        """Record that retrieved memory evidence was returned to the semantic inference path."""
        self._record("SM-008", "executed", {
            "query": query,
            "episodic_ids": list(episodic_ids),
            "prototype_ids": list(prototype_ids),
            "evidence_count": int(evidence_count),
            "destination": [
                "ATL semantic hub", "modality spokes", "distributed semantic features",
                "phonological representation", "semantic control",
            ],
        })

    def generalize(
        self,
        *,
        orthography: str | None = None,
        phonological_form: Sequence[str] = (),
        semantic_features: Mapping[str, float] | None = None,
        context_features: Mapping[str, float] | None = None,
        limit: int = 4,
        minimum_similarity: float = 0.15,
    ) -> list[dict[str, Any]]:
        """Match a new word-form cue against consolidated abstract lexical prototypes."""
        normalized = _normalize(orthography or "")
        phones = tuple(str(phone) for phone in phonological_form)
        semantic = _numeric_features(semantic_features or {})
        context = _numeric_features(context_features or {})
        if not (normalized or phones or semantic or context):
            raise ValueError("generalization requires at least one non-empty cue")
        if limit < 1:
            raise ValueError("limit must be positive")
        if not math.isfinite(minimum_similarity) or not 0.0 <= minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must lie in [0, 1]")
        rows = self._connection.execute(
            "SELECT * FROM prototypes ORDER BY prototype_id"
        ).fetchall()
        query_profile = _ngrams(normalized) if normalized else {}
        weights = {
            "orthography": 0.45 if normalized else 0.0,
            "phonology": 0.25 if phones else 0.0,
            "semantics": 0.20 if semantic else 0.0,
            "context": 0.10 if context else 0.0,
        }
        active_weight = sum(weights.values()) or 1.0
        ranked: list[dict[str, Any]] = []
        for row in rows:
            proto_phones = json.loads(row["phonological_profile"])
            prototype_sequence = tuple(json.loads(row["phonological_sequence"]))
            channels: dict[str, float] = {}
            if normalized:
                channels["orthography"] = _cosine(
                    query_profile, json.loads(row["orthographic_profile"])
                )
            if phones:
                expected = prototype_sequence or tuple(
                    phone for phone, _ in sorted(proto_phones.items(), key=lambda pair: pair[0])
                )
                channels["phonology"] = _sequence_similarity(phones, expected)
            if semantic:
                channels["semantics"] = _cosine(
                    semantic, json.loads(row["semantic_features"])
                )
            if context:
                channels["context"] = _cosine(
                    context, json.loads(row["context_features"])
                )
            score = sum(weights[key] * channels.get(key, 0.0) for key in weights) / active_weight
            if score < minimum_similarity:
                continue
            ranked.append({
                "prototype_id": row["prototype_id"],
                "experience_count": int(row["experience_count"]),
                "similarity": float(score),
                "channel_scores": channels,
                "orthographic_profile": json.loads(row["orthographic_profile"]),
                "phonological_profile": proto_phones,
                "phonological_sequence": list(prototype_sequence),
                "semantic_features": json.loads(row["semantic_features"]),
                "context_features": json.loads(row["context_features"]),
                "generalization_status": "matched_consolidated_prototype",
            })
        ranked.sort(key=lambda item: (item["similarity"], item["experience_count"]), reverse=True)
        matches = ranked[:limit]
        self._record("SM-016", "matched" if matches else "no_match", {
            "orthography_cue": normalized,
            "candidate_prototype_count": len(rows),
            "returned_prototype_ids": [item["prototype_id"] for item in matches],
            "minimum_similarity": minimum_similarity,
        })
        return matches

    def consolidate(self, prototype_id: str | None = None, *,
                    learning_rate: float = 0.20) -> dict[str, Any]:
        """Consolidate stored episodes and strengthen category-level orthographic, phonological and semantic prototypes."""
        if not math.isfinite(learning_rate) or not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must lie in (0, 1]")
        groups = [prototype_id] if prototype_id else [
            row["prototype_id"] for row in self._connection.execute(
                "SELECT DISTINCT prototype_id FROM episodes ORDER BY prototype_id"
            ).fetchall()
        ]
        updated: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc).isoformat()
        for group in groups:
            rows = self._connection.execute(
                "SELECT * FROM episodes WHERE prototype_id=? ORDER BY created_at", (group,)
            ).fetchall()
            if not rows:
                continue
            total = sum(max(float(row["strength"]), 1e-6) for row in rows)
            ortho: dict[str, float] = {}
            phon: dict[str, float] = {}
            semantic: dict[str, float] = {}
            context: dict[str, float] = {}
            ordered_episodes: list[tuple[tuple[str, ...], float]] = []
            for row in rows:
                weight = max(float(row["strength"]), 1e-6) / total
                episode_phones = tuple(json.loads(row["phonological_form"]))
                if episode_phones:
                    ordered_episodes.append((episode_phones, weight))
                for target, features in (
                    (ortho, _ngrams(row["normalized_form"])),
                    (phon, {phone: 1.0 for phone in episode_phones}),
                    (semantic, json.loads(row["semantic_features"])),
                    (context, json.loads(row["context_features"])),
                ):
                    for key, value in features.items():
                        target[key] = target.get(key, 0.0) + weight * float(value)
            representative_sequence: tuple[str, ...] = ()
            if ordered_episodes:
                # Select a weighted medoid sequence so the prototype retains learned phoneme order.
                representative_sequence = max(
                    ordered_episodes,
                    key=lambda candidate: sum(
                        other_weight * _sequence_similarity(candidate[0], other_sequence)
                        for other_sequence, other_weight in ordered_episodes
                    ),
                )[0]
            old = self._connection.execute(
                "SELECT * FROM prototypes WHERE prototype_id=?", (group,)
            ).fetchone()
            if old:
                def blend(old_map: Mapping[str, float], new_map: Mapping[str, float]) -> dict[str, float]:
                    keys = set(old_map) | set(new_map)
                    return {key: (1.0 - learning_rate) * float(old_map.get(key, 0.0))
                            + learning_rate * float(new_map.get(key, 0.0)) for key in keys}
                ortho = blend(json.loads(old["orthographic_profile"]), ortho)
                phon = blend(json.loads(old["phonological_profile"]), phon)
                semantic = blend(json.loads(old["semantic_features"]), semantic)
                context = blend(json.loads(old["context_features"]), context)
                experience_count = int(old["experience_count"]) + len(rows)
            else:
                experience_count = len(rows)
            self._connection.execute(
                """INSERT INTO prototypes(
                    prototype_id, experience_count, orthographic_profile,
                    phonological_profile, phonological_sequence,
                    semantic_features, context_features, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(prototype_id) DO UPDATE SET
                    experience_count=excluded.experience_count,
                    orthographic_profile=excluded.orthographic_profile,
                    phonological_profile=excluded.phonological_profile,
                    phonological_sequence=excluded.phonological_sequence,
                    semantic_features=excluded.semantic_features,
                    context_features=excluded.context_features,
                    updated_at=excluded.updated_at""",
                (group, experience_count, _json(ortho), _json(phon),
                 _json(list(representative_sequence)), _json(semantic), _json(context), now),
            )
            # Consolidation increases accessibility without exceeding the bounded strength scale.
            for row in rows:
                strengthened = min(1.0, float(row["strength"]) + learning_rate * 0.05)
                self._connection.execute(
                    "UPDATE episodes SET strength=?, updated_at=? WHERE episode_id=?",
                    (strengthened, now, row["episode_id"]),
                )
            updated.append({
                "prototype_id": group, "experience_count": experience_count,
                "orthographic_dimensions": len(ortho), "phonological_dimensions": len(phon),
                "semantic_dimensions": len(semantic), "context_dimensions": len(context),
            })
        self._connection.commit()
        prototype_ids = [item["prototype_id"] for item in updated]
        total_episodes = sum(item["experience_count"] for item in updated)
        self._record("SM-009", "executed", {
            "prototype_groups": prototype_ids, "episode_count_collected": total_episodes,
        })
        self._record("SM-010", "executed", {
            "common_feature_dimensions": {
                item["prototype_id"]: {
                    "orthographic": item["orthographic_dimensions"],
                    "phonological": item["phonological_dimensions"],
                    "semantic": item["semantic_dimensions"],
                    "context": item["context_dimensions"],
                } for item in updated
            },
            "experience_weighting": "normalized_episode_strength",
        })
        self._record("SM-011", "executed", {
            "orthographic_profiles_built": prototype_ids,
            "position_sensitive_n_gram_encoding": "ordered_character_bigrams",
        })
        self._record("SM-012", "executed", {
            "phonological_profiles_built": prototype_ids,
            "ordered_phonological_prototypes_built": prototype_ids,
            "phonological_order_policy": "weighted medoid of stored ordered phonological sequences",
        })
        self._record("SM-013", "executed", {
            "abstract_lexical_identities": prototype_ids,
            "identity_links": ["orthography", "phonology", "semantics", "context", "episode references"],
        })
        self._record("SM-014", "executed", {
            "experience_strength_updates": total_episodes,
            "learning_rate": learning_rate,
            "bounded_strength": True,
        })
        self._record("SM-015", "executed", {
            "consolidated_prototype_count": len(updated),
            "prototype_ids": prototype_ids,
            "slow_cortical_rate_approximation": learning_rate,
        })
        return {"updated_prototypes": updated, "learning_rate": learning_rate}

    def get_prototype(self, prototype_id: str) -> dict[str, Any] | None:
        """Return the consolidated prototype record without mutating memory."""
        row = self._connection.execute(
            "SELECT * FROM prototypes WHERE prototype_id=?", (prototype_id,)
        ).fetchone()
        return dict(row) if row else None

    def episode_count(self) -> int:
        """Return the number of durable episodic traces."""
        return int(self._connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0])
