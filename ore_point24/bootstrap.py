"""Bootstrap helper: install the 23 point packages from their zip layout.

Reproduces the integration layout used by the Point 24 audit:

- each point package directory is copied to the repository root as an
  importable ``ore_*`` package;
- the remaining materials (docs, examples, sibling tests) are preserved under
  ``ore_point_sources/<folder>/`` as integration evidence;
- the Point 22 kernel package ``ore_runtime`` is renamed to
  ``ore_runtime_governance`` to resolve the name conflict with the Point 19
  execution engine ``ore_runtime`` (documented architectural decision), and
  its sibling tests are import-patched accordingly.

Usage::

    python -m ore_point24.bootstrap --repo /path/to/ORE-Neurocomputational \\
        --sources /path/to/extracted/23_extra_ore_algorithms
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# (zip folder, package dir inside the folder, final package name, source point)
POINT_PACKAGE_LAYOUT: tuple[tuple[str, str, str, int], ...] = (
    ("ORE-Distributed Systems", "ore_distributed", "ore_distributed", 17),
    ("ORE-HIC", "ore_hic", "ore_hic", 16),
    ("ORE-Probabilistic Data Structures", "ore_data", "ore_data", 18),
    ("ORE-RLMS", "ore_learning", "ore_learning", 14),
    ("ORE_Bayesian_Causal_Reasoning", "ore_bayesian_causal", "ore_bayesian_causal", 7),
    ("ORE_CROS", "ore_cros", "ore_cros", 23),
    ("ORE_CWM_Cognitive_World_Model", "ore_cwm", "ore_cwm", 4),
    ("ORE_Cryptography_Privacy_Verifiable_Computation", "ore_security", "ore_security", 20),
    ("ORE_HACOL_Hardware_Awareness", "ore_hacol", "ore_hacol", 21),
    ("ORE_RUNTIME_Kernel_Isolation_OS", "ore_runtime", "ore_runtime_governance", 22),
    ("ORE_Runtime_Engine", "ore_runtime", "ore_runtime", 19),
    ("ORE_rl_Implementation Traceability", "ore_rl", "ore_rl", 10),
    ("ORE_rl_Implementation Traceability", "ore_logs", "ore_logs", 10),
    ("ore_Scientific Geometric and Physical-World", "ore_science", "ore_science", 12),
    ("ore_Search Algorithms", "ore_planning", "ore_planning", 6),
    ("ore_agentic", "ore_agentic", "ore_agentic", 15),
    ("ore_blackboard_implementation", "ore_blackboard", "ore_blackboard", 1),
    ("ore_control_implementation", "ore_control", "ore_control", 2),
    ("ore_evolution_intelligence", "ore_evolution_intelligence", "ore_evolution_intelligence", 9),
    ("ore_formal_reasoning_implementation", "ore_formal_reasoning", "ore_formal_reasoning", 5),
    ("ore_graph_intelligence", "ore_graph_intelligence", "ore_graph_intelligence", 8),
    ("ore_macl_multi_agent_swarm_coordination", "ore_macl", "ore_macl", 11),
    ("ore_neurosymbolic_implementation", "ore_neurosymbolic", "ore_neurosymbolic", 3),
    ("ore_temporal", ".", "ore_temporal", 13),
)


def install_point_packages(
    sources_root: str | Path,
    repository_root: str | Path,
    *,
    overwrite: bool = True,
) -> dict[str, object]:
    """Copy the 23 point packages and their evidence into the repository."""
    sources = Path(sources_root).expanduser().resolve()
    repo = Path(repository_root).expanduser().resolve()
    if not sources.is_dir():
        raise FileNotFoundError(f"sources directory not found: {sources}")
    if not repo.is_dir():
        raise FileNotFoundError(f"repository directory not found: {repo}")
    point_sources = repo / "ore_point_sources"
    point_sources.mkdir(exist_ok=True)

    installed: list[str] = []
    for folder, package_dir, final_name, point in POINT_PACKAGE_LAYOUT:
        src_pkg = sources / folder / package_dir
        dst_pkg = repo / final_name
        if final_name in installed:
            continue
        if not src_pkg.exists():
            print(f"warning: source package missing: {src_pkg}")
            continue
        if dst_pkg.exists():
            if not overwrite:
                print(f"skip (exists): {dst_pkg}")
                continue
            shutil.rmtree(dst_pkg)
        shutil.copytree(src_pkg, dst_pkg, ignore=shutil.ignore_patterns("__pycache__"))
        installed.append(final_name)

        evidence = point_sources / folder
        if not evidence.exists():
            evidence.mkdir(parents=True)
            for item in (sources / folder).iterdir():
                if item.name == "__pycache__" or item.name == package_dir:
                    continue
                if item.is_dir():
                    shutil.copytree(item, evidence / item.name,
                                    ignore=shutil.ignore_patterns("__pycache__"))
                elif item.suffix.lower() in {".zip", ".patch"}:
                    continue
                else:
                    shutil.copy2(item, evidence / item.name)

    _patch_kernel_test_imports(point_sources)
    return {"installed": sorted(installed), "count": len(installed)}


def _patch_kernel_test_imports(point_sources: Path) -> None:
    """Repoint the renamed kernel package's sibling tests."""
    kernel_tests = point_sources / "ORE_RUNTIME_Kernel_Isolation_OS" / "tests"
    if not kernel_tests.is_dir():
        return
    for test_file in kernel_tests.glob("*.py"):
        text = test_file.read_text(encoding="utf-8")
        new = text.replace("from ore_runtime import", "from ore_runtime_governance import")
        new = new.replace("from ore_runtime.", "from ore_runtime_governance.")
        new = new.replace("import ore_runtime\n", "import ore_runtime_governance\n")
        if new != text:
            test_file.write_text(new, encoding="utf-8")
            print(f"patched imports in {test_file.name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="ORE-Neurocomputational repository root")
    parser.add_argument("--sources", required=True,
                        help="extracted '23 Extra ORE Algorithms' directory")
    parser.add_argument("--no-overwrite", action="store_true",
                        help="do not overwrite existing packages")
    args = parser.parse_args(argv)
    result = install_point_packages(
        args.sources, args.repo, overwrite=not args.no_overwrite,
    )
    installed = result["installed"]  # type: ignore[index]
    print(f"installed {result['count']} point packages:")
    for name in installed:  # type: ignore[union-attr]
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
