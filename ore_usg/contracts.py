"""Typed results for read-only USG knowledge imports into ORE."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

USG_KNOWLEDGE_CONTRACT = "USG-KNOWLEDGE/0.1.0"
USG_KNOWLEDGE_SCHEMA_VERSION = "0.1.0"
USG_LEXICAL_ADAPTER_VERSION = "0.1.0"
ORE_USG_BUNDLE_CONTRACT = "ORE.USG.KNOWLEDGE_BUNDLE"
ORE_USG_LEXICON_CONTRACT = "ORE.USG.LEXICON"


@dataclass(frozen=True, slots=True)
class USGImportExclusion:
    """Explain why one lexical record was not exposed to ORE."""

    record_id: str
    record_kind: str
    reason: str


@dataclass(frozen=True, slots=True)
class USGSourceSummary:
    """Retain the source rights and attribution fields used by the import policy."""

    source_id: str
    provider: str
    rights_status: str
    redistribution: str
    attribution: str
    usable_for_local_import: bool
    license_id: str | None = None
    source_locator: str | None = None
    retrieved_at: str = ""
    content_digest: str | None = None


@dataclass(frozen=True, slots=True)
class USGEvidenceSummary:
    """Expose evidence type and support/contradiction state without copying evidence text."""

    evidence_id: str
    source_id: str
    evidence_type: str
    support_status: str
    locator: str | None
    text_digest: str | None
    captured_at: str
    transformation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class USGContextSummary:
    """Preserve language, domain, and other explicit context scopes by stable ID."""

    context_id: str
    context_type: str
    value: str
    language_profile: str | None
    scope: str | None


@dataclass(frozen=True, slots=True)
class USGLexiconImportReport:
    """Summarize deterministic lexical import decisions without asserting truth."""

    bundle_id: str
    records_seen: int
    lexical_records_seen: int
    entries_created: int
    meanings_created: int
    imported_record_ids: tuple[str, ...]
    excluded_records: tuple[USGImportExclusion, ...]
    source_summaries: tuple[USGSourceSummary, ...]
    evidence_summaries: tuple[USGEvidenceSummary, ...]
    context_summaries: tuple[USGContextSummary, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class USGLexiconResource:
    """Versioned ORE lexical resource with the complete import report attached."""

    entries: tuple[Any, ...]
    bundle_id: str
    namespace: str
    bundle_digest: str
    adapter_version: str
    report: USGLexiconImportReport

    def __post_init__(self) -> None:
        """Reject incomplete resource identities and duplicate ORE entry IDs."""
        if not self.bundle_id.strip() or not self.namespace.strip():
            raise ValueError("bundle_id and namespace must be non-empty")
        if (
            len(self.bundle_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.bundle_digest)
        ):
            raise ValueError("bundle_digest must be a lowercase SHA-256 hex digest")
        entry_ids = [str(getattr(item, "entry_id", "")) for item in self.entries]
        if any(not item for item in entry_ids):
            raise ValueError("every imported lexical entry must have a non-empty entry_id")
        if len(entry_ids) != len(set(entry_ids)):
            raise ValueError("imported lexical entry IDs must be unique")
