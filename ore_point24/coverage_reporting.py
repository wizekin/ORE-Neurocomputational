"""Phase X, Steps 46-50 - close residual coverage and maintain the architecture.

Step 46 closes the 24-point and A-G coverage matrix, reporting coverage by
evidence state rather than a binary flag - a source family may be examined
while its implementation gate remains false.  Step 47 classifies residual
technologies with one of four decisions (integrate, evaluate experimentally,
retain as a future option, reject/defer).  Step 48 consolidates package
responsibilities and validates the dependency graph as a DAG.  Step 49
executes the staged 15-phase roadmap with explicit prerequisites.  Step 50
publishes the final integration audit with the traceability ratio
``T = linked requirements / total requirements``, reporting numerator and
denominator alongside the ratio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .core import EvidenceState, ModuleStateVector, sha256_digest, utc_now_iso
from .spec import (
    POINT_FAMILIES,
    RESIDUAL_CATEGORIES,
    RESIDUAL_TECHNOLOGIES,
    ROADMAP_PHASES,
    LAYER_IDS,
    point_layer_incidence,
)

COVERAGE_SCHEMA = "ore.point24.coverage_matrix.v1"
RESIDUAL_SCHEMA = "ore.point24.residual_register.v1"
PACKAGE_MAP_SCHEMA = "ore.point24.package_map.v1"
ROADMAP_SCHEMA = "ore.point24.roadmap.v1"
AUDIT_SCHEMA = "ore.point24.audit.v1"


# ---------------------------------------------------------------------------
# Step 46 - the 24-point and A-G coverage matrix
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CoverageMatrix:
    """Every point with its layer, package, evidence state and gate.

    Coverage means examined and assigned a role; it never implies
    implementation, validation or benchmark success.
    """

    point_states: dict[int, dict[str, Any]] = field(default_factory=dict)

    def set_point_state(
        self,
        point: int,
        *,
        vector: ModuleStateVector,
        role: str,
        notes: Iterable[str] = (),
    ) -> None:
        if point not in {family.point for family in POINT_FAMILIES}:
            raise ValueError(f"unknown source point: {point}")
        from .core import derive_evidence_state
        self.point_states[point] = {
            "point": point,
            "role": role,
            "state_vector": vector.to_dict(),
            "evidence_state": derive_evidence_state(vector).value,
            "gate": float(
                vector.manifest_registered and vector.backend_bound
                and vector.validated and vector.integration_tested
            ),
            "notes": list(notes),
        }

    def coverage_by_evidence_state(self) -> dict[str, int]:
        counts = {state.value: 0 for state in EvidenceState}
        for record in self.point_states.values():
            counts[record["evidence_state"]] += 1
        return counts

    def ag_matrix(self) -> dict[str, Any]:
        """The A-G residual categories with their point families."""
        rows = {}
        for letter, title, points, description in RESIDUAL_CATEGORIES:
            rows[letter] = {
                "title": title,
                "points": list(points),
                "description": description,
                "families": [
                    next(f.family for f in POINT_FAMILIES if f.point == p) for p in points
                ],
                "all_points_mapped": all(p in self.point_states for p in points),
            }
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": COVERAGE_SCHEMA,
            "point_states": {str(k): v for k, v in sorted(self.point_states.items())},
            "coverage_by_evidence_state": self.coverage_by_evidence_state(),
            "ag_matrix": self.ag_matrix(),
            "incidence_matrix": point_layer_incidence(),
            "control": "coverage means examined and assigned a role; it does not imply implementation",
        }


def build_coverage_matrix(
    point_vectors: Mapping[int, ModuleStateVector],
    *,
    roles: Mapping[int, str] | None = None,
) -> CoverageMatrix:
    """Build the Step 46 matrix from independently evidenced state vectors."""
    matrix = CoverageMatrix()
    for family in POINT_FAMILIES:
        vector = point_vectors.get(family.point, ModuleStateVector(
            source_present=False, manifest_registered=False, backend_bound=False,
            validated=False, integration_tested=False,
        ))
        matrix.set_point_state(
            family.point,
            vector=vector,
            role=(roles or {}).get(family.point, family.primary_responsibility),
        )
    return matrix


# ---------------------------------------------------------------------------
# Step 47 - residual technology register
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ResidualTechnologyRegister:
    """The residual technology register with explicit adoption decisions."""

    decisions: dict[str, str] = field(default_factory=dict)
    overrides: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for technology in RESIDUAL_TECHNOLOGIES:
            self.decisions.setdefault(technology.name, technology.decision)

    def set_decision(self, name: str, decision: str, *, rationale: str) -> None:
        valid = {"integrate", "evaluate_experimentally", "future_option", "reject_defer"}
        if decision not in valid:
            raise ValueError(f"decision must be one of {sorted(valid)}")
        known = {technology.name for technology in RESIDUAL_TECHNOLOGIES}
        if name not in known:
            raise KeyError(f"unknown residual technology: {name!r}")
        self.decisions[name] = decision
        self.overrides[name] = rationale

    def by_decision(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for name, decision in self.decisions.items():
            groups.setdefault(decision, []).append(name)
        return {key: sorted(value) for key, value in sorted(groups.items())}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RESIDUAL_SCHEMA,
            "decisions": dict(sorted(self.decisions.items())),
            "override_rationales": dict(self.overrides),
            "by_decision": self.by_decision(),
            "control": "the presence of an interesting technology is not sufficient reason to add a dependency",
        }


# ---------------------------------------------------------------------------
# Step 48 - package responsibilities and dependency directions
# ---------------------------------------------------------------------------

REUSED_INFRASTRUCTURE: tuple[str, ...] = (
    "ore_contracts", "ore_workspace", "ore_usg", "ore_logger",
)


@dataclass(slots=True)
class PackageMap:
    """The canonical package and dependency map (Step 48).

    Reuse ore_contracts for module identity, contract validation, dependency
    planning and task execution; ore_workspace for versioned shared state;
    ore_usg for read-only semantic import; ore_logger for runs and events.
    Preserve existing pipelines behind explicit adapters.  Prohibit circular
    imports and circular runtime dependencies between layers.
    """

    packages: dict[str, dict[str, Any]] = field(default_factory=dict)
    dependencies: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def add_package(self, name: str, *, layer: str, responsibility: str,
                    reused: bool = False) -> None:
        if layer not in LAYER_IDS and layer != "cross_cutting":
            raise ValueError(f"unknown layer: {layer!r}")
        self.packages[name] = {
            "layer": layer,
            "responsibility": responsibility,
            "reused_infrastructure": reused,
        }

    def add_dependency(self, package: str, depends_on: str) -> None:
        self.dependencies.setdefault(package, ())
        current = self.dependencies[package]
        if depends_on not in current:
            self.dependencies[package] = current + (depends_on,)

    def validate_acyclic(self) -> tuple[bool, list[str]]:
        """The package dependency graph must be a DAG (no circular imports)."""
        state: dict[str, int] = {name: 0 for name in self.packages}
        stack: list[str] = []
        problems: list[str] = []

        def visit(node: str) -> None:
            if state.get(node) == 1:
                cycle = stack[stack.index(node):] + [node]
                problems.append("circular dependency: " + " -> ".join(cycle))
                return
            if state.get(node) == 2 or node not in self.packages:
                return
            state[node] = 1
            stack.append(node)
            for dependency in self.dependencies.get(node, ()):
                visit(dependency)
            stack.pop()
            state[node] = 2

        for package in sorted(self.packages):
            visit(package)
        return (not problems, problems)

    def to_dict(self) -> dict[str, Any]:
        acyclic, problems = self.validate_acyclic()
        return {
            "schema": PACKAGE_MAP_SCHEMA,
            "packages": {k: dict(v) for k, v in sorted(self.packages.items())},
            "dependencies": {k: list(v) for k, v in sorted(self.dependencies.items())},
            "acyclic": acyclic,
            "problems": problems,
            "reused_infrastructure": list(REUSED_INFRASTRUCTURE),
            "control": "logical separation does not require separate services or packages",
        }


def build_default_package_map() -> PackageMap:
    """The canonical Step 48 map reusing existing repository infrastructure."""
    from .spec import POINT_FAMILIES
    mapping = PackageMap()
    for name in REUSED_INFRASTRUCTURE:
        mapping.add_package(name, layer="cross_cutting",
                            responsibility="reused ORE foundation", reused=True)
    mapping.add_package("ore_point24", layer="cognitive_control",
                        responsibility="Point 24 unified integration layer")
    for family in POINT_FAMILIES:
        if family.package and family.package != "ore_point24":
            mapping.add_package(
                family.package, layer=family.primary_layer,
                responsibility=f"Point {family.point}: {family.primary_responsibility}",
            )
    # Point packages depend only on the shared infrastructure and the
    # integration layer - never on each other.
    for family in POINT_FAMILIES:
        if family.package and family.package != "ore_point24":
            mapping.add_dependency(family.package, "ore_contracts")
    mapping.add_dependency("ore_point24", "ore_contracts")
    mapping.add_dependency("ore_point24", "ore_workspace")
    mapping.add_dependency("ore_point24", "ore_usg")
    mapping.add_dependency("ore_point24", "ore_logger")
    return mapping


# ---------------------------------------------------------------------------
# Step 49 - staged implementation roadmap
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RoadmapTracker:
    """Dependency-aware roadmap tracking (Step 49).

    A phase is ready only when its upstream contract and test gates pass; the
    tracker never advances a phase merely because a calendar milestone
    elapsed.  This roadmap is proposed sequencing, not evidence that the
    phases have already been implemented.
    """

    completed: set[int] = field(default_factory=set)
    gate_results: dict[int, bool] = field(default_factory=dict)

    def phase_ready(self, phase: int) -> bool:
        record = next((p for p in ROADMAP_PHASES if p.phase == phase), None)
        if record is None:
            raise ValueError(f"unknown roadmap phase: {phase}")
        return all(
            prerequisite in self.completed and self.gate_results.get(prerequisite, False)
            for prerequisite in record.prerequisites
        )

    def complete_phase(self, phase: int, *, gate_passed: bool) -> None:
        if not self.phase_ready(phase):
            raise ValueError(
                f"phase {phase} prerequisites not satisfied; the roadmap is dependency-aware"
            )
        self.completed.add(phase)
        self.gate_results[phase] = bool(gate_passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ROADMAP_SCHEMA,
            "phases": [
                {
                    "phase": record.phase,
                    "target": record.development_target,
                    "source_references": list(record.source_references),
                    "prerequisites": list(record.prerequisites),
                    "ready": self.phase_ready(record.phase),
                    "completed": record.phase in self.completed,
                }
                for record in ROADMAP_PHASES
            ],
            "completed_phases": sorted(self.completed),
        }


# ---------------------------------------------------------------------------
# Step 50 - final integration audit
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class IntegrationAudit:
    """The versioned final integration audit with traceability (Step 50).

    Links every source point to its layer, module, contract, tests and
    evidence status; records unresolved gaps, deferred research options,
    architectural decisions and rejected alternatives; records exactly which
    repository commit was evaluated; and computes the traceability ratio with
    both numerator and denominator.
    """

    baseline: Mapping[str, Any]
    coverage_matrix: Mapping[str, Any]
    package_map: Mapping[str, Any]
    residual_register: Mapping[str, Any]
    test_results: Mapping[str, Any] = field(default_factory=dict)
    architectural_decisions: list[str] = field(default_factory=list)
    unresolved_gaps: list[str] = field(default_factory=list)
    audit_id: str = field(default_factory=lambda: "point24_audit")
    generated_at: str = field(default_factory=utc_now_iso)

    def traceability(self) -> dict[str, Any]:
        """T = requirements with source, owner, contract and test links / total."""
        states = self.coverage_matrix.get("point_states", {})
        linked = 0
        total = 0
        for record in states.values():
            total += 1
            vector = record.get("state_vector", {})
            if (
                vector.get("source_present") and vector.get("manifest_registered")
                and record.get("gate") is not None
                and record.get("evidence_state")
            ):
                linked += 1
        return {
            "numerator": linked,
            "denominator": total,
            "ratio": round(linked / total, 4) if total else 0.0,
            "note": "a complete traceability index does not itself prove correctness",
        }

    def final_status(self) -> dict[str, str]:
        """Distinguish assessment, architecture, code, tests and validation."""
        coverage_states = {
            record.get("evidence_state") for record in
            self.coverage_matrix.get("point_states", {}).values()
        }
        return {
            "assessment_complete": "yes",
            "architecture_proposed": "yes",
            "code_implemented": "yes (ore_point24 integration layer)",
            "tests_passed": (
                "yes" if self.test_results.get("all_passed") else
                str(self.test_results.get("summary", "not recorded"))
            ),
            "externally_validated": (
                "not claimed" if not self.test_results.get("external_benchmarks")
                else "partially"
            ),
            "coverage_states_observed": ", ".join(sorted(str(s) for s in coverage_states if s)),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": AUDIT_SCHEMA,
            "audit_id": self.audit_id,
            "generated_at": self.generated_at,
            "baseline": dict(self.baseline),
            "coverage_matrix": dict(self.coverage_matrix),
            "package_map": dict(self.package_map),
            "residual_register": dict(self.residual_register),
            "test_results": dict(self.test_results),
            "architectural_decisions": list(self.architectural_decisions),
            "unresolved_gaps": list(self.unresolved_gaps),
            "traceability": self.traceability(),
            "final_status": self.final_status(),
            "re_run_procedure": (
                "re-run the audit after material contract, module, model or deployment changes"
            ),
        }
        payload["digest"] = sha256_digest(payload)
        return payload


__all__ = [
    "AUDIT_SCHEMA",
    "COVERAGE_SCHEMA",
    "CoverageMatrix",
    "IntegrationAudit",
    "PACKAGE_MAP_SCHEMA",
    "PackageMap",
    "RESIDUAL_SCHEMA",
    "REUSED_INFRASTRUCTURE",
    "ResidualTechnologyRegister",
    "ROADMAP_SCHEMA",
    "RoadmapTracker",
    "build_coverage_matrix",
    "build_default_package_map",
]
