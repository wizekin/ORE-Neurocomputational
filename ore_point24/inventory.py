"""Phase I, Step 2 - inventory executable modules rather than documentation claims.

Enumerate Python packages, module manifests, runtime backend bindings and the
tests that exercise each module's actual behaviour.  For every module a state
vector ``v_m = (s_m, r_m, b_m, t_m, i_m)`` is produced in which source
presence, manifest registration, backend binding, validation and
integration-test coverage are *separately evidenced* Booleans.  A README
entry, class name or docstring alone can never set backend-bound or validated
status.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .core import ModuleStateVector, new_id, sha256_digest
from .spec import POINT_FAMILIES

INVENTORY_SCHEMA = "ore.point24.inventory.v1"

_REPOSITORY_PACKAGES: tuple[str, ...] = (
    "ore_contracts", "ore_workspace", "ore_usg", "ore_logger",
)


@dataclass(frozen=True, slots=True)
class PackageInventoryEntry:
    """One importable Python package found in the repository tree."""

    package_name: str
    source_path: str
    python_files: int
    declares_manifests: bool
    declares_registration: bool
    ships_sibling_tests: bool
    file_count: int
    source_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_name": self.package_name,
            "source_path": self.source_path,
            "python_files": self.python_files,
            "declares_manifests": self.declares_manifests,
            "declares_registration": self.declares_registration,
            "ships_sibling_tests": self.ships_sibling_tests,
            "file_count": self.file_count,
            "source_digest": self.source_digest,
        }


def _iter_python_sources(package_dir: Path) -> Iterable[Path]:
    for path in sorted(package_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def _package_flags(package_dir: Path) -> tuple[bool, bool]:
    """Detect manifest and registration declarations by AST, not docstrings."""
    declares_manifests = False
    declares_registration = False
    for path in _iter_python_sources(package_dir):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == "ModuleManifest":
                        declares_manifests = True
            if isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Name) and func.id == "ModuleManifest"
                ) or (
                    isinstance(func, ast.Attribute) and func.attr in {"register", "bind_backend"}
                ):
                    if isinstance(func, ast.Name):
                        declares_manifests = True
                    else:
                        declares_registration = True
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name.lower()
                if name.startswith("register_") and any(
                    token in name for token in ("module", "specialist", "backend")
                ):
                    declares_registration = True
        if declares_manifests and declares_registration:
            break
    return declares_manifests, declares_registration


def scan_package(package_dir: str | Path) -> PackageInventoryEntry:
    """Inventory one importable package directory (source facts only)."""
    root = Path(package_dir).expanduser().resolve()
    if not root.is_dir() or not (root / "__init__.py").is_file():
        raise FileNotFoundError(f"not an importable package directory: {root}")
    python_files = list(_iter_python_sources(root))
    all_files = [p for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    declares_manifests, declares_registration = _package_flags(root)
    sibling_tests = any(
        (root.parent / candidate).is_dir()
        for candidate in ("tests",)
    ) and (root.parent / "tests").is_dir()
    return PackageInventoryEntry(
        package_name=root.name,
        source_path=str(root),
        python_files=len(python_files),
        declares_manifests=declares_manifests,
        declares_registration=declares_registration,
        ships_sibling_tests=sibling_tests,
        file_count=len(all_files),
        source_digest=sha256_digest(
            {str(p.relative_to(root)): sha256_digest(p.read_bytes()) for p in all_files[:512]}
        ),
    )


def scan_repository(
    repository_root: str | Path,
    *,
    extra_packages: Iterable[str] = (),
) -> dict[str, Any]:
    """Inventory the integration repository: infrastructure, points, others.

    The result separates (a) reused ORE infrastructure, (b) the 23 integrated
    point packages and (c) any other importable packages.  Documentation-only
    descriptions, examples and test fixtures are counted as files, never as
    executable backends.
    """
    root = Path(repository_root).expanduser().resolve()
    point_packages = {family.package for family in POINT_FAMILIES if family.package}
    infrastructure = set(_REPOSITORY_PACKAGES) | set(extra_packages)

    entries: dict[str, PackageInventoryEntry] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not (child / "__init__.py").is_file():
            continue
        if child.name.startswith(".") or child.name in {"__pycache__", "node_modules"}:
            continue
        entries[child.name] = scan_package(child)

    classification: dict[str, str] = {}
    for name in entries:
        if name in infrastructure:
            classification[name] = "infrastructure"
        elif name in point_packages:
            classification[name] = "point_package"
        else:
            classification[name] = "specialist"

    return {
        "schema": INVENTORY_SCHEMA,
        "repository_root": str(root),
        "classification": classification,
        "packages": {name: entry.to_dict() for name, entry in sorted(entries.items())},
        "counts": {
            "total_packages": len(entries),
            "infrastructure": sum(1 for v in classification.values() if v == "infrastructure"),
            "point_packages": sum(1 for v in classification.values() if v == "point_package"),
            "specialist": sum(1 for v in classification.values() if v == "specialist"),
        },
    }


def module_state_vector(
    *,
    source_present: bool,
    manifest_registered: bool,
    backend_bound: bool,
    validated: bool,
    integration_tested: bool,
) -> ModuleStateVector:
    """Build the Step 2 state vector with each field separately evidenced."""
    return ModuleStateVector(
        source_present=source_present,
        manifest_registered=manifest_registered,
        backend_bound=backend_bound,
        validated=validated,
        integration_tested=integration_tested,
    )


__all__ = [
    "INVENTORY_SCHEMA",
    "PackageInventoryEntry",
    "module_state_vector",
    "scan_package",
    "scan_repository",
]
