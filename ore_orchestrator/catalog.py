"""Static discovery of the Python packages that make up ORE."""
from __future__ import annotations

import ast
import importlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_INFRASTRUCTURE = {"ore_contracts", "ore_workspace", "ore_logger", "ore_runtime", "ore_runtime_governance"}
_ORCHESTRATION = {"ore_point24", "ore_control", "ore_blackboard", "ore_neurosymbolic", "ore_macl", "ore_cros", "ore_hic", "ore_planning"}
_PRIMARY_PIPELINES = {
    "ore_n1", "ore_n2", "ore_n3", "ore_n4", "ore_n5", "ore_n6", "ore_n9",
    "ore_bnbea", "ore_comprehension", "ore_reading", "ore_semantic",
    "ore_semantic_composition", "ore_syntax", "ore_discourse",
    "ore_working_memory", "ore_sensory_perception",
}


@dataclass(frozen=True, slots=True)
class PackageDescriptor:
    """Static facts about one importable package found at the repository root."""
    name: str
    import_path: str
    category: str
    version: str | None
    python_file_count: int
    public_exports: tuple[str, ...]
    entrypoint_modules: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OREPackageCatalog:
    """Discover, describe, and lazily import repository packages."""

    def __init__(self, repository_root: str | Path | None = None) -> None:
        self.repository_root = Path(repository_root or Path(__file__).resolve().parents[1]).expanduser().resolve()
        self._packages: dict[str, PackageDescriptor] = {}
        self.refresh()

    @staticmethod
    def _read_init_metadata(init_file: Path) -> tuple[str | None, tuple[str, ...]]:
        version: str | None = None
        exports: list[str] = []
        try:
            tree = ast.parse(init_file.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeError):
            return version, ()
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value = node.value
                if any(isinstance(target, ast.Name) and target.id == "__version__" for target in targets):
                    try:
                        candidate = ast.literal_eval(value)
                        if isinstance(candidate, str):
                            version = candidate
                    except (ValueError, TypeError):
                        pass
                if any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
                    try:
                        candidate = ast.literal_eval(value)
                        if isinstance(candidate, (list, tuple)):
                            exports.extend(item for item in candidate if isinstance(item, str))
                    except (ValueError, TypeError):
                        pass
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
                try:
                    candidate = ast.literal_eval(node.value)
                    if isinstance(candidate, (list, tuple)):
                        exports.extend(item for item in candidate if isinstance(item, str))
                except (ValueError, TypeError):
                    pass
        return version, tuple(dict.fromkeys(exports))

    @staticmethod
    def _category(name: str) -> str:
        if name == "examples":
            return "examples"
        if name == "experiment_workbench":
            return "developer_tooling"
        if name in _INFRASTRUCTURE:
            return "infrastructure"
        if name in _ORCHESTRATION:
            return "orchestration"
        if name in _PRIMARY_PIPELINES:
            return "primary_pipeline"
        return "specialist"

    def refresh(self) -> tuple[PackageDescriptor, ...]:
        """Re-scan direct child packages without importing them."""
        if not self.repository_root.is_dir():
            raise FileNotFoundError(f"ORE repository root does not exist: {self.repository_root}")
        found: dict[str, PackageDescriptor] = {}
        for directory in sorted(self.repository_root.iterdir()):
            if (
                not directory.is_dir()
                or directory.name.startswith(".")
                or directory.name == "__pycache__"
                or not (directory / "__init__.py").is_file()
            ):
                continue
            python_files = [item for item in directory.rglob("*.py") if "__pycache__" not in item.parts]
            direct_modules = {item.stem for item in directory.glob("*.py")}
            entrypoints = tuple(
                name for name in ("__main__", "pipeline", "coordinator", "registration")
                if name in direct_modules
            )
            version, exports = self._read_init_metadata(directory / "__init__.py")
            found[directory.name] = PackageDescriptor(
                name=directory.name,
                import_path=directory.name,
                category=self._category(directory.name),
                version=version,
                python_file_count=len(python_files),
                public_exports=exports,
                entrypoint_modules=entrypoints,
            )
        self._packages = found
        return self.packages

    @property
    def packages(self) -> tuple[PackageDescriptor, ...]:
        return tuple(self._packages[name] for name in sorted(self._packages))

    @property
    def package_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.packages)

    def list_packages(self, category: str | None = None) -> tuple[PackageDescriptor, ...]:
        if category is None:
            return self.packages
        return tuple(item for item in self.packages if item.category == category)

    def get_package(self, name: str) -> PackageDescriptor:
        try:
            return self._packages[name]
        except KeyError as exc:
            raise KeyError(f"package is not present in this repository inventory: {name}") from exc

    def import_package(self, name: str):
        """Import one known package on demand; discovery itself never imports packages."""
        descriptor = self.get_package(name)
        return importlib.import_module(descriptor.import_path)

    def to_dict(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for item in self.packages:
            counts[item.category] = counts.get(item.category, 0) + 1
        return {
            "repository_root": str(self.repository_root),
            "package_count": len(self._packages),
            "category_counts": dict(sorted(counts.items())),
            "packages": [item.to_dict() for item in self.packages],
        }


__all__ = ["OREPackageCatalog", "PackageDescriptor"]
