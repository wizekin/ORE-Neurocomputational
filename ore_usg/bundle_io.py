"""Deterministic, read-only discovery and loading of exported USG knowledge bundles."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from .lexical import _canonical_json
from .validation import require_valid_knowledge_bundle

_BUNDLE_FILENAMES = (
    "knowledge_bundle.json",
    "usg_knowledge_bundle.json",
    "bundle.json",
)


class USGBundleLoadError(ValueError):
    """Raised when a USG bundle cannot be discovered, decoded, or verified."""


@dataclass(frozen=True, slots=True)
class USGBundleSnapshot:
    """A validated detached snapshot and its stable source identity."""

    bundle: Mapping[str, Any]
    source_path: str | None
    bundle_digest: str
    byte_length: int


def discover_knowledge_bundle(directory: str | Path) -> Path:
    """Find one conventional bundle file; fail rather than guess when several exist."""
    root = Path(directory).expanduser()
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    matches = [root / name for name in _BUNDLE_FILENAMES if (root / name).is_file()]
    if not matches:
        raise FileNotFoundError(
            f"No conventional USG bundle found in {root}; expected one of "
            + ", ".join(_BUNDLE_FILENAMES)
        )
    if len(matches) > 1:
        raise USGBundleLoadError(
            "Multiple conventional USG bundles found; pass the intended JSON file explicitly: "
            + ", ".join(path.name for path in matches)
        )
    return matches[0]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise USGBundleLoadError(f"Duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def load_knowledge_bundle(
    source: str | Path | Mapping[str, Any],
    *,
    expected_digest: str | None = None,
    max_bytes: int = 256 * 1024 * 1024,
) -> USGBundleSnapshot:
    """Load and validate a detached bundle from a mapping, JSON file, or directory.

    Directory discovery uses only documented conventional filenames and refuses
    ambiguous directories. The loader never writes to the source or USG database.
    """
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    source_path: str | None = None
    if isinstance(source, Mapping):
        try:
            raw = _canonical_json(source)
            bundle = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
        except (TypeError, ValueError, UnicodeError) as exc:
            raise USGBundleLoadError(f"Mapping cannot be serialized as a valid JSON bundle: {exc}") from exc
    else:
        path = Path(source).expanduser()
        if path.is_dir():
            path = discover_knowledge_bundle(path)
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_file():
            raise USGBundleLoadError(f"Bundle source is not a regular file: {path}")
        size = path.stat().st_size
        if size > max_bytes:
            raise USGBundleLoadError(
                f"Bundle is {size} bytes, exceeding configured limit of {max_bytes}"
            )
        try:
            raw = path.read_bytes()
            bundle = json.loads(
                raw.decode("utf-8-sig"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    USGBundleLoadError(f"Non-finite JSON number is not allowed: {value}")
                ),
            )
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise USGBundleLoadError(f"Cannot decode USG JSON bundle at {path}: {exc}") from exc
        source_path = str(path.resolve())
    if not isinstance(bundle, Mapping):
        raise USGBundleLoadError("USG bundle JSON root must be an object")
    if len(raw) > max_bytes:
        raise USGBundleLoadError(
            f"Bundle is {len(raw)} bytes, exceeding configured limit of {max_bytes}"
        )
    try:
        require_valid_knowledge_bundle(bundle)
        digest = sha256(_canonical_json(bundle)).hexdigest()
    except (TypeError, ValueError) as exc:
        raise USGBundleLoadError(f"USG bundle validation failed: {exc}") from exc
    if expected_digest is not None and digest.casefold() != expected_digest.casefold():
        raise USGBundleLoadError(
            f"Bundle digest mismatch: expected {expected_digest}, calculated {digest}"
        )
    # Round-trip through canonical JSON detaches nested mappings/lists from caller state.
    detached = json.loads(_canonical_json(bundle).decode("utf-8"))
    return USGBundleSnapshot(
        bundle=detached,
        source_path=source_path,
        bundle_digest=digest,
        byte_length=len(raw),
    )


__all__ = [
    "USGBundleLoadError",
    "USGBundleSnapshot",
    "discover_knowledge_bundle",
    "load_knowledge_bundle",
]
