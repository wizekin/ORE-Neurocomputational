"""Typed input adapters, modality detection, and loss-free normalization.

Implements spec Steps 2-4 and Section 2.1: written text, speech transcripts
(with preserved confidence), files, diagrams/tables as source artifacts, and
an optional, constrained BCI adapter that is disabled unless explicit neural
measurements and a task-specific decoder are supplied. Normalization corrects
encoding and whitespace only as a *derived view* with source mapping; the raw
source is never replaced (spec Step 4).
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import (
    GateViolation,
    HICError,
    InputModality,
    QualityFlag,
    SourceSpan,
)
from .session import Attachment, CompilationSession, InputRecord

_MODAITY_HINTS: tuple[tuple[InputModality, tuple[str, ...]], ...] = (
    (InputModality.CODE, ("def ", "import ", "class ", "function ", "return ")),
    (InputModality.TABLE, ("|---", "\t", "col1,", "csv", ",,,"),
     ),
)


class InputAdapter:
    """Base class for typed input adapters (spec 2.1)."""

    modality: InputModality = InputModality.TEXT

    def validate(self, content: str, **kwargs: Any) -> tuple[QualityFlag, ...]:
        """Return modality-specific quality warnings without mutating content."""
        return ()

    def capture(self, session: CompilationSession, content: str, **kwargs: Any) -> InputRecord:
        """Preserve the input as an immutable InputRecord bound to ``session``."""
        flags = list(self.validate(content, **kwargs))
        record = session.create_input_record(
            content=content,
            modality=self.modality,
            source_identity=str(kwargs.get("source_identity", "user")),
            language=str(kwargs.get("language", "en")),
            transcription_confidence=kwargs.get("transcription_confidence"),
            quality_flags=flags,
            attachments=kwargs.get("attachments", ()),
        )
        session.ledger.record(
            stage="A",
            step_id="HIC03",
            description=f"Classify input modality {self.modality.value} and quality",
            inputs=[record],
            outputs=[record],
            rule_references=["HIC-STEP-03"],
        )
        return record


class TextInputAdapter(InputAdapter):
    """Written natural-language ideas, descriptions, and requirements."""

    modality = InputModality.TEXT

    def validate(self, content: str, **kwargs: Any) -> tuple[QualityFlag, ...]:
        flags: list[QualityFlag] = []
        if not content.strip():
            flags.append(QualityFlag.EMPTY_INPUT)
        try:
            content.encode("utf-8")
        except UnicodeEncodeError:
            flags.append(QualityFlag.ENCODING_CORRECTED)
        return tuple(flags)


class SpeechTranscriptAdapter(InputAdapter):
    """Speech transcribed to text; confidence links back to the audio."""

    modality = InputModality.SPEECH_TRANSCRIPT

    def validate(self, content: str, **kwargs: Any) -> tuple[QualityFlag, ...]:
        confidence = kwargs.get("transcription_confidence")
        if confidence is None:
            raise HICError("speech input requires transcription_confidence in [0, 1]")
        if not 0.0 <= float(confidence) <= 1.0:
            raise HICError("transcription_confidence must lie in [0, 1]")
        flags: list[QualityFlag] = []
        if float(confidence) < 0.85:
            # Uncertain extraction must remain visible, never silently accepted.
            flags.append(QualityFlag.LOW_TRANSCRIPTION_CONFIDENCE)
        return tuple(flags)

    def capture(self, session: CompilationSession, content: str, **kwargs: Any) -> InputRecord:
        kwargs.setdefault("attachments", ())
        audio_note = kwargs.pop("audio_reference", None)
        if audio_note is not None:
            kwargs["attachments"] = tuple(kwargs["attachments"]) + (
                Attachment(
                    attachment_id=f"audio-{len(content)}",
                    filename=str(audio_note),
                    content="[audio source retained externally; transcript is the working view]",
                    modality=InputModality.SPEECH_TRANSCRIPT,
                    notes="link to original audio",
                ),
            )
        return super().capture(session, content, **kwargs)


class FileSourceAdapter(InputAdapter):
    """Existing code, specifications, or workflows supplied as context."""

    modality = InputModality.STRUCTURED_FILE

    def validate(self, content: str, **kwargs: Any) -> tuple[QualityFlag, ...]:
        if not content.strip():
            return (QualityFlag.EMPTY_INPUT,)
        if any(hint in content for hint in ("def ", "import ", "class ")):
            self.modality = InputModality.CODE
        return ()

    def capture(self, session: CompilationSession, content: str, **kwargs: Any) -> InputRecord:
        self.modality = InputModality.CODE if any(
            hint in content for hint in ("def ", "import ", "class ")
        ) else InputModality.STRUCTURED_FILE
        return super().capture(session, content, **kwargs)


class DiagramSourceAdapter(InputAdapter):
    """Diagrams and tables represented as source artifacts (spec 2.1)."""

    modality = InputModality.DIAGRAM

    def validate(self, content: str, **kwargs: Any) -> tuple[QualityFlag, ...]:
        flags: list[QualityFlag] = []
        if "ocr_confidence" in kwargs and float(kwargs["ocr_confidence"]) < 0.9:
            flags.append(QualityFlag.OCR_UNCERTAINTY)
        if kwargs.get("missing_pages"):
            flags.append(QualityFlag.MISSING_PAGES)
        return tuple(flags)


class BCIStreamAdapter(InputAdapter):
    """Optional experimental BCI command stream (spec 2.1, Section 11).

    The adapter stays disabled unless a task-specific decoder over measured
    neural signals is provided. It is a constrained input modality, never a
    claim that unrestricted private thoughts can be read.
    """

    modality = InputModality.BCI_STREAM

    def __init__(self, decoder: Any | None = None) -> None:
        if decoder is None:
            self.enabled = False
        else:
            if not callable(decoder):
                raise HICError("BCI decoder must be callable over measured neural signals")
            self.enabled = True
        self.decoder = decoder

    def validate(self, content: str, **kwargs: Any) -> tuple[QualityFlag, ...]:
        if not self.enabled:
            raise GateViolation(
                "BCI input is disabled: no measured neural signals and task-specific decoder available"
            )
        return (QualityFlag.UNCONFIRMED_EXTRACTION,)

    def decode_commands(self, samples: Sequence[float]) -> tuple[str, ...]:
        """Decode a constrained command alphabet from measured signals."""
        if not self.enabled or self.decoder is None:
            raise GateViolation("BCI decoding requires an enabled decoder")
        return tuple(self.decoder(list(samples)))


def detect_modality(content: str, *, filename: str | None = None) -> InputModality:
    """Step 3: classify the modality of raw content deterministically."""
    if not content.strip():
        return InputModality.TEXT
    lowered = content.lower()
    if filename:
        suffix = filename.rsplit(".", 1)[-1].lower()
        if suffix in {"png", "jpg", "svg", "drawio"}:
            return InputModality.DIAGRAM
        if suffix in {"csv", "tsv"}:
            return InputModality.TABLE
        if suffix in {"py", "js", "rs", "cpp", "java"}:
            return InputModality.CODE
    code_hints = sum(1 for hint in ("def ", "import ", "class ", "return ", "function ") if hint in lowered)
    if code_hints >= 2:
        return InputModality.CODE
    if "|" in content and "---" in content:
        return InputModality.TABLE
    if re.search(r"\b(transcript|asr|speech-to-text)\b", lowered):
        return InputModality.SPEECH_TRANSCRIPT
    return InputModality.TEXT


@dataclass(frozen=True, slots=True)
class NormalizedView:
    """A derived normalization of the input with a full source mapping (Step 4)."""

    record_id: str
    text: str
    corrections: tuple[str, ...]
    mapping: tuple[tuple[int, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "text": self.text,
            "corrections": list(self.corrections),
            "mapping": [list(m) for m in self.mapping],
        }


class Normalizer:
    """Normalize encoding and whitespace without semantic loss (spec Step 4)."""

    _RE_MULTISPACE = re.compile(r"[ \t]+")
    _RE_MULTINL = re.compile(r"\n{3,}")

    def normalize(self, record: InputRecord) -> NormalizedView:
        """Produce a derived view; every corrected region maps back to source offsets."""
        corrections: list[str] = []
        text = record.content
        if text != unicodedata.normalize("NFC", text):
            corrections.append("unicode-normalized-to-NFC")
            text = unicodedata.normalize("NFC", text)
        if "\r\n" in text or "\r" in text:
            corrections.append("crlf-normalized-to-lf")
            text = text.replace("\r\n", "\n").replace("\r", "\n")
        if text != text.strip():
            corrections.append("outer-whitespace-trimmed")
        text = self._RE_MULTISPACE.sub(" ", text)
        text = self._RE_MULTINL.sub("\n\n", text)
        mapping = self._build_mapping(record.content, text)
        return NormalizedView(
            record_id=record.artifact_id,
            text=text,
            corrections=tuple(corrections),
            mapping=tuple(mapping),
        )

    @staticmethod
    def _build_mapping(original: str, normalized: str) -> list[tuple[int, int]]:
        """Greedy character alignment between original and normalized offsets."""
        mapping: list[tuple[int, int]] = []
        orig_idx = 0
        norm_idx = 0
        run_start: tuple[int, int] | None = None
        while orig_idx < len(original) and norm_idx < len(normalized):
            o_char = original[orig_idx]
            n_char = normalized[norm_idx]
            if o_char == n_char:
                if run_start is None:
                    run_start = (orig_idx, norm_idx)
                orig_idx += 1
                norm_idx += 1
            else:
                if run_start is not None:
                    mapping.append(run_start)
                    run_start = None
                if o_char.isspace() and n_char.isspace():
                    orig_idx += 1
                    norm_idx += 1
                elif o_char.isspace():
                    orig_idx += 1
                else:
                    norm_idx += 1
        if run_start is not None:
            mapping.append(run_start)
        return mapping

    @staticmethod
    def map_span_to_original(view: NormalizedView, start: int, end: int) -> SourceSpan:
        """Map a span in the normalized view back to original source offsets."""
        best: tuple[int, int] | None = None
        for orig_start, norm_start in view.mapping:
            if norm_start <= start:
                best = (orig_start, norm_start)
        if best is None:
            raise HICError(f"cannot map normalized offset {start} back to source")
        orig_start, norm_start = best
        delta = start - norm_start
        return SourceSpan(
            artifact_id=view.record_id,
            start=orig_start + delta,
            end=orig_start + delta + (end - start),
            text="",
        )
