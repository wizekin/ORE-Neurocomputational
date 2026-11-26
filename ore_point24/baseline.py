"""Phase I, Step 1 - freeze the integration baseline.

Record the repository identity (URL, branch, commit SHA, capture time), the
exact runtime versions (Python, dependencies, optional backends), the source
assessment identity and SHA-256 digests of the architecture documents used for
integration decisions.  The baseline manifest is immutable once computed and
every later test result and architectural decision must identify the baseline
it refers to.
"""
from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .core import POINT24_VERSION, SPEC_TITLE, canonical_json, sha256_digest, utc_now_iso

BASELINE_SCHEMA = "ore.point24.baseline.v1"
_DEFAULT_DEPENDENCIES = ("numpy", "scipy")


def _git_output(repository: Path, *args: str) -> str | None:
    """Run one read-only git query; return None when git metadata is absent."""
    try:
        completed = subprocess.run(
            ["git", *args], cwd=str(repository), capture_output=True, text=True,
            timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _dependency_versions(names: tuple[str, ...]) -> dict[str, str]:
    """Record exact installed versions of tracked dependencies."""
    versions: dict[str, str] = {}
    for name in names:
        try:
            module = importlib.import_module(name)
        except ImportError:
            versions[name] = "unavailable"
            continue
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


@dataclass(frozen=True, slots=True)
class IntegrationBaseline:
    """The immutable Step 1 baseline manifest with repository provenance."""

    schema: str
    spec_title: str
    spec_version: str
    captured_at: str
    repository_root: str
    repository_url: str | None
    branch: str | None
    commit_sha: str | None
    commit_dirty: bool
    python_version: str
    platform: str
    dependency_versions: Mapping[str, str]
    source_assessment: str
    point_relevance: Mapping[str, int]
    architecture_document_hashes: Mapping[str, str]
    baseline_id: str = field(default_factory=lambda: f"point24_baseline_{sha256_digest(utc_now_iso())[:12]}")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible baseline manifest."""
        return {
            "schema": self.schema,
            "baseline_id": self.baseline_id,
            "spec_title": self.spec_title,
            "spec_version": self.spec_version,
            "captured_at": self.captured_at,
            "repository": {
                "root": self.repository_root,
                "url": self.repository_url,
                "branch": self.branch,
                "commit": self.commit_sha,
                "dirty": self.commit_dirty,
            },
            "runtime": {
                "python": self.python_version,
                "platform": self.platform,
                "dependencies": dict(self.dependency_versions),
            },
            "source_assessment": self.source_assessment,
            "point_relevance": dict(self.point_relevance),
            "architecture_document_hashes": dict(self.architecture_document_hashes),
        }

    def digest(self) -> str:
        """Return the canonical SHA-256 digest of the baseline manifest."""
        return sha256_digest(self.to_dict())


def freeze_integration_baseline(
    repository_root: str | Path,
    *,
    architecture_documents: Mapping[str, str | Path] | None = None,
    tracked_dependencies: tuple[str, ...] = _DEFAULT_DEPENDENCIES,
    source_assessment: str = "Most Intelligent Systems 01-24.txt (24-point assessment)",
    point_relevance: Mapping[str, int] | None = None,
) -> IntegrationBaseline:
    """Freeze the Step 1 baseline without changing implementation code.

    The 24-point relevance ranking is recorded from the source assessment
    (rank 1 = most useful family, 24 = the integration point itself).
    """
    root = Path(repository_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"repository root does not exist: {root}")

    url = _git_output(root, "remote", "get-url", "origin")
    branch = _git_output(root, "rev-parse", "--abbrev-ref", "HEAD")
    commit = _git_output(root, "rev-parse", "HEAD")
    status = _git_output(root, "status", "--porcelain")
    dirty = bool(status)

    hashes: dict[str, str] = {}
    for label, document in (architecture_documents or {}).items():
        path = Path(document)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(f"architecture document not found: {path}")
        hashes[label] = sha256_digest(path.read_bytes())

    relevance: dict[str, int] = dict(point_relevance or {})
    if not relevance:
        from .spec import POINT_FAMILIES
        relevance = {f"point_{family.point:02d}": 25 - family.point for family in POINT_FAMILIES}

    return IntegrationBaseline(
        schema=BASELINE_SCHEMA,
        spec_title=SPEC_TITLE,
        spec_version=POINT24_VERSION,
        captured_at=utc_now_iso(),
        repository_root=str(root),
        repository_url=url,
        branch=branch,
        commit_sha=commit,
        commit_dirty=dirty,
        python_version=platform.python_version(),
        platform=platform.platform(),
        dependency_versions=_dependency_versions(tracked_dependencies),
        source_assessment=source_assessment,
        point_relevance=relevance,
        architecture_document_hashes=hashes,
    )


__all__ = [
    "BASELINE_SCHEMA",
    "IntegrationBaseline",
    "freeze_integration_baseline",
]
