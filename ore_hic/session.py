"""Compilation session lifecycle, immutable input records, artifact versioning.

Implements spec Steps 1-2: a unique session binds every later artifact without
modifying the original input, and raw input is preserved byte-for-byte as an
immutable ``InputRecord`` with integrity digests and source offsets for
traceability.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .contracts import (
    ArtifactKind,
    ArtifactPointer,
    CanonicalArtifact,
    GateViolation,
    InputModality,
    QualityFlag,
    SourceSpan,
    canonical_json,
    digest_payload,
    digest_text,
)
from .provenance import ProvenanceLedger


@dataclass(frozen=True, slots=True)
class Attachment:
    """An immutable attached source file preserved with its own digest."""

    attachment_id: str
    filename: str
    content: str
    modality: InputModality
    notes: str = ""

    @property
    def content_digest(self) -> str:
        return digest_text(self.content)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attachment_id": self.attachment_id,
            "filename": self.filename,
            "content": self.content,
            "modality": self.modality.value,
            "notes": self.notes,
            "content_digest": self.content_digest,
        }


class InputRecord(CanonicalArtifact):
    """Immutable record of one raw human input (spec 2.2, Step 2).

    The exact original content, modality, source identity, timestamp,
    attachments and integrity digest are preserved. Derived normalized views
    are stored separately and never replace the original.
    """

    kind = ArtifactKind.INPUT_RECORD

    def __init__(
        self,
        *,
        artifact_id: str,
        session_id: str,
        content: str,
        modality: InputModality,
        source_identity: str,
        language: str = "en",
        transcription_confidence: float | None = None,
        quality_flags: Iterable[QualityFlag] = (),
        attachments: Iterable[Attachment] = (),
        created_at: str | None = None,
        version: int = 1,
    ) -> None:
        if not content:
            raise ValueError("InputRecord content must not be empty")
        if modality is InputModality.SPEECH_TRANSCRIPT:
            if transcription_confidence is None:
                raise ValueError("speech transcripts require a transcription confidence")
            if not 0.0 <= transcription_confidence <= 1.0:
                raise ValueError("transcription confidence must lie in [0, 1]")
        self.artifact_id = artifact_id
        self.session_id = session_id
        self.content = content
        self.modality = modality
        self.source_identity = source_identity
        self.language = language
        self.transcription_confidence = transcription_confidence
        self.quality_flags = tuple(sorted(quality_flags, key=lambda f: f.value))
        self.attachments = tuple(attachments)
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.version = version
        self._span_text_cache: str | None = None

    @property
    def integrity_digest(self) -> str:
        """Digest over the exact raw bytes of the original content."""
        return digest_text(self.content)

    def span(self, start: int, end: int) -> SourceSpan:
        """Create a traceable source span into the preserved original content."""
        if not 0 <= start <= end <= len(self.content):
            raise ValueError(f"span [{start}, {end}) outside input of length {len(self.content)}")
        return SourceSpan(artifact_id=self.artifact_id, start=start, end=end, text=self.content[start:end])

    def content_payload(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "session_id": self.session_id,
            "content": self.content,
            "integrity_digest": self.integrity_digest,
            "modality": self.modality.value,
            "source_identity": self.source_identity,
            "language": self.language,
            "transcription_confidence": self.transcription_confidence,
            "quality_flags": [flag.value for flag in self.quality_flags],
            "attachments": [a.to_dict() for a in self.attachments],
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InputRecord:
        return cls(
            artifact_id=payload["artifact_id"],
            session_id=payload["session_id"],
            content=payload["content"],
            modality=InputModality(payload["modality"]),
            source_identity=payload["source_identity"],
            language=payload.get("language", "en"),
            transcription_confidence=payload.get("transcription_confidence"),
            quality_flags=[QualityFlag(f) for f in payload.get("quality_flags", [])],
            attachments=[
                Attachment(
                    attachment_id=a["attachment_id"],
                    filename=a["filename"],
                    content=a["content"],
                    modality=InputModality(a["modality"]),
                    notes=a.get("notes", ""),
                )
                for a in payload.get("attachments", [])
            ],
            created_at=payload.get("created_at"),
            version=int(payload.get("version", 1)),
        )


class SessionSettings:
    """Deterministic compilation settings bound to one session (spec Step 1)."""

    def __init__(
        self,
        *,
        deterministic: bool = False,
        language_preferences: str = "en",
        allow_codegen_without_approval: bool = False,
        property_test_cases: int = 40,
        property_test_seed: int = 20261123,
        sandbox_cpu_seconds: int = 5,
        sandbox_memory_mb: int = 256,
        sandbox_wall_seconds: float = 10.0,
        sandbox_output_byte_limit: int = 65536,
        target_language: str = "python",
        context_facts: Mapping[str, str] | None = None,
    ) -> None:
        if property_test_cases <= 0:
            raise ValueError("property_test_cases must be positive")
        if sandbox_cpu_seconds <= 0 or sandbox_memory_mb <= 0 or sandbox_wall_seconds <= 0:
            raise ValueError("sandbox limits must be positive")
        self.deterministic = deterministic
        self.language_preferences = language_preferences
        self.allow_codegen_without_approval = allow_codegen_without_approval
        self.property_test_cases = property_test_cases
        self.property_test_seed = property_test_seed
        self.sandbox_cpu_seconds = sandbox_cpu_seconds
        self.sandbox_memory_mb = sandbox_memory_mb
        self.sandbox_wall_seconds = sandbox_wall_seconds
        self.sandbox_output_byte_limit = sandbox_output_byte_limit
        self.target_language = target_language
        self.context_facts = dict(context_facts or {})

    def payload(self) -> dict[str, Any]:
        return {
            "deterministic": self.deterministic,
            "language_preferences": self.language_preferences,
            "allow_codegen_without_approval": self.allow_codegen_without_approval,
            "property_test_cases": self.property_test_cases,
            "property_test_seed": self.property_test_seed,
            "sandbox_cpu_seconds": self.sandbox_cpu_seconds,
            "sandbox_memory_mb": self.sandbox_memory_mb,
            "sandbox_wall_seconds": self.sandbox_wall_seconds,
            "sandbox_output_byte_limit": self.sandbox_output_byte_limit,
            "target_language": self.target_language,
            "context_facts": dict(sorted(self.context_facts.items())),
        }


class CompilationSession:
    """One compilation session binding artifacts, versions, and provenance (Step 1)."""

    def __init__(
        self,
        *,
        user: str,
        project: str = "default-project",
        workspace: str = "default-workspace",
        settings: SessionSettings | None = None,
        session_id: str | None = None,
        created_at: str | None = None,
    ) -> None:
        if not user.strip():
            raise ValueError("session user must be a non-empty identity")
        self.user = user
        self.project = project
        self.workspace = workspace
        self.settings = settings or SessionSettings()
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.session_id = session_id or self._derive_session_id()
        self.ledger = ProvenanceLedger()
        self._artifacts: dict[ArtifactKind, list[CanonicalArtifact]] = {}
        self._counter_by_kind: dict[ArtifactKind, int] = {}

    def _derive_session_id(self) -> str:
        seed = canonical_json(
            {
                "user": self.user,
                "project": self.project,
                "workspace": self.workspace,
                "settings": self.settings.payload(),
            }
        )
        return f"sess-{digest_payload(seed)[:16]}"

    # -- artifact store ------------------------------------------------------

    def next_artifact_id(self, kind: ArtifactKind) -> str:
        """Return the next stable artifact identifier for ``kind``."""
        count = self._counter_by_kind.get(kind, 0) + 1
        self._counter_by_kind[kind] = count
        return f"{kind.value.lower()}-{self.session_id}-{count:03d}"

    def register(self, artifact: CanonicalArtifact) -> ArtifactPointer:
        """Store a versioned artifact and return its stable pointer."""
        versions = self._artifacts.setdefault(artifact.kind, [])
        if not hasattr(artifact, "version"):
            raise GateViolation("canonical artifacts must expose a version")
        if any(a.content_digest == artifact.content_digest for a in versions):
            pointer = ArtifactPointer(
                artifact_kind=artifact.kind,
                artifact_id=getattr(artifact, "artifact_id", artifact.kind.value),
                version=getattr(artifact, "version", 1),
                content_digest=artifact.content_digest,
            )
            return pointer
        versions.append(artifact)
        return ArtifactPointer(
            artifact_kind=artifact.kind,
            artifact_id=getattr(artifact, "artifact_id", artifact.kind.value),
            version=getattr(artifact, "version", 1),
            content_digest=artifact.content_digest,
        )

    def latest(self, kind: ArtifactKind) -> CanonicalArtifact | None:
        versions = self._artifacts.get(kind, [])
        return versions[-1] if versions else None

    def versions_of(self, kind: ArtifactKind) -> tuple[CanonicalArtifact, ...]:
        return tuple(self._artifacts.get(kind, ()))

    def create_input_record(
        self,
        *,
        content: str,
        modality: InputModality,
        source_identity: str,
        language: str = "en",
        transcription_confidence: float | None = None,
        quality_flags: Iterable[QualityFlag] = (),
        attachments: Iterable[Attachment] = (),
    ) -> InputRecord:
        """Step 2: preserve the raw input as an immutable, digested record."""
        record = InputRecord(
            artifact_id=self.next_artifact_id(ArtifactKind.INPUT_RECORD),
            session_id=self.session_id,
            content=content,
            modality=modality,
            source_identity=source_identity,
            language=language,
            transcription_confidence=transcription_confidence,
            quality_flags=quality_flags,
            attachments=attachments,
        )
        self.register(record)
        self.ledger.record(
            stage="A",
            step_id="HIC02",
            description="Preserve raw input as immutable InputRecord",
            outputs=[record],
            rule_references=["HIC-STEP-02"],
        )
        return record

    def describe(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user": self.user,
            "project": self.project,
            "workspace": self.workspace,
            "created_at": self.created_at,
            "settings": self.settings.payload(),
            "artifact_counts": {kind.value: len(v) for kind, v in sorted(self._artifacts.items())},
        }
