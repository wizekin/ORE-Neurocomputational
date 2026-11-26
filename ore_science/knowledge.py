"""Read-only scientific knowledge interface over USG-style concept and equation records.

This module implements the ``ore_science.knowledge`` boundary. It deliberately mirrors
the retrieval discipline of ``ore_usg``: lookups are exact and case-insensitive, no
confidence scores are fabricated, and every retrieved record is tagged as a
``retrieved_assertion`` rather than an independently validated fact. A separate bridge
can import concepts from a real ``ore_usg`` ``USGKnowledgeResource`` when one is
available, without ever writing back to USG.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .contracts import Assumption, AssumptionSource, Constraint, EquationModel, ScientificDomain
from .units import UNITS

__all__ = [
    "EVIDENCE_STATUS_RETRIEVED",
    "ScienceConceptRecord",
    "ScienceEquationRecord",
    "ScienceKnowledgeBase",
    "USGScienceBridge",
]


EVIDENCE_STATUS_RETRIEVED = "retrieved_assertion"


@dataclass(frozen=True, slots=True)
class ScienceConceptRecord:
    """A scientific concept retrieved from knowledge, never treated as validated fact."""

    concept_id: str
    label: str
    definition: str
    domain: ScientificDomain
    unit_symbol: str | None = None
    related_equations: tuple[str, ...] = ()
    evidence_status: str = EVIDENCE_STATUS_RETRIEVED
    source_reference: str = "local_science_knowledge_base"

    def __post_init__(self) -> None:
        """Concept records must be identified, defined, and honestly labelled."""
        if not self.concept_id.strip() or not self.label.strip():
            raise ValueError("concept id and label must be non-empty")
        if not self.definition.strip():
            raise ValueError(f"concept {self.concept_id} must carry a definition")
        if self.evidence_status != EVIDENCE_STATUS_RETRIEVED:
            raise ValueError(
                "retrieved scientific assertions must not be labelled as validated facts"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the concept record."""
        return {
            "concept_id": self.concept_id,
            "label": self.label,
            "definition": self.definition,
            "domain": self.domain.value,
            "unit_symbol": self.unit_symbol,
            "related_equations": list(self.related_equations),
            "evidence_status": self.evidence_status,
            "source_reference": self.source_reference,
        }


@dataclass(frozen=True, slots=True)
class ScienceEquationRecord:
    """A governing equation record with term units and applicability conditions.

    ``terms`` lists the additive terms with the composed unit of each term; ``symbols``
    lists the individual quantities with their own units. Both are preserved so the
    dimensional validator can check homogeneity while the variable registry stays
    linked to the individual symbols.
    """

    equation_id: str
    name: str
    form: str
    domain: ScientificDomain
    terms: tuple[tuple[str, str], ...]
    symbols: tuple[tuple[str, str], ...] = ()
    conditions: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    evidence_status: str = EVIDENCE_STATUS_RETRIEVED
    source_reference: str = "local_science_knowledge_base"

    def __post_init__(self) -> None:
        """Equation records must be identified with at least two typed terms."""
        if not self.equation_id.strip() or not self.name.strip():
            raise ValueError("equation id and name must be non-empty")
        if len(self.terms) < 2:
            raise ValueError(f"equation {self.equation_id} requires at least two terms")
        if self.evidence_status != EVIDENCE_STATUS_RETRIEVED:
            raise ValueError("retrieved equations must not be labelled as validated facts")

    def to_equation_model(self) -> EquationModel:
        """Convert the record into a typed EquationModel for the pipeline."""
        terms = _resolve_unit_pairs(self.equation_id, self.terms)
        symbols = _resolve_unit_pairs(self.equation_id, self.symbols) if self.symbols else terms
        return EquationModel(
            equation_id=self.equation_id,
            name=self.name,
            form=self.form,
            terms=tuple(terms),
            symbols=tuple(symbols),
            conditions=self.conditions,
            usg_reference=self.source_reference,
            domain=self.domain,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the equation record."""
        return {
            "equation_id": self.equation_id,
            "name": self.name,
            "form": self.form,
            "domain": self.domain.value,
            "terms": [[symbol, unit] for symbol, unit in self.terms],
            "symbols": [[symbol, unit] for symbol, unit in self.symbols],
            "conditions": list(self.conditions),
            "assumptions": list(self.assumptions),
            "evidence_status": self.evidence_status,
            "source_reference": self.source_reference,
        }


def _resolve_unit_pairs(equation_id: str, pairs: tuple[tuple[str, str], ...]) -> list[tuple[str, Any]]:
    """Resolve (label, unit key) pairs into (label, Unit) pairs."""
    resolved = []
    for symbol, unit_key in pairs:
        unit = UNITS.get(unit_key)
        if unit is None:
            raise ValueError(f"equation {equation_id} references unknown unit {unit_key!r}")
        resolved.append((symbol, unit))
    return resolved


def _heat_equation() -> ScienceEquationRecord:
    """Curated record for the one-dimensional heat diffusion equation."""
    return ScienceEquationRecord(
        equation_id="EQ-HEAT-1D",
        name="Heat diffusion equation",
        form="du/dt = alpha * d2u/dx2",
        domain=ScientificDomain.PHYSICAL_SIMULATION,
        terms=(("du/dt", "K_s"), ("alpha * d2u/dx2", "K_s")),
        symbols=(("u", "K"), ("x", "m"), ("t", "s"), ("alpha", "m2_s")),
        conditions=(
            "isotropic homogeneous medium",
            "no internal heat sources",
            "one-dimensional spatial domain",
        ),
        assumptions=("alpha is constant in space and time",),
    )


def _ideal_gas_law() -> ScienceEquationRecord:
    """Curated record for the ideal gas law."""
    return ScienceEquationRecord(
        equation_id="EQ-IDEAL-GAS",
        name="Ideal gas law",
        form="P * V = n * R * T",
        domain=ScientificDomain.PHYSICAL_SIMULATION,
        terms=(("P * V", "J"), ("n * R * T", "J")),
        symbols=(("P", "Pa"), ("V", "m3"), ("n", "mol"), ("T", "K")),
        conditions=("dilute gas", "thermodynamic equilibrium"),
        assumptions=("R = 8.314 J/(mol K)", "negligible intermolecular interactions"),
    )


def _newton_second_law() -> ScienceEquationRecord:
    """Curated record for Newton's second law."""
    return ScienceEquationRecord(
        equation_id="EQ-NEWTON-2",
        name="Newton's second law",
        form="F = m * a",
        domain=ScientificDomain.PHYSICAL_SIMULATION,
        terms=(("F", "N"), ("m * a", "N")),
        symbols=(("F", "N"), ("m", "kg"), ("a", "m_s")),
        conditions=("inertial reference frame",),
        assumptions=("classical mechanics", "constant mass"),
    )


def _hookes_law() -> ScienceEquationRecord:
    """Curated record for Hooke's law."""
    return ScienceEquationRecord(
        equation_id="EQ-HOOKE",
        name="Hooke's law",
        form="F = -k * x",
        domain=ScientificDomain.PHYSICAL_SIMULATION,
        terms=(("F", "N"), ("k * x", "N")),
        symbols=(("F", "N"), ("k", "N_m"), ("x", "m")),
        conditions=("small deformation", "linear elastic regime"),
        assumptions=("spring constant k is constant",),
    )


def _lj_potential() -> ScienceEquationRecord:
    """Curated record for the Lennard-Jones pairwise potential."""
    return ScienceEquationRecord(
        equation_id="EQ-LENNARD-JONES",
        name="Lennard-Jones pair potential",
        form="V(r) = 4 * epsilon * ((sigma/r)^12 - (sigma/r)^6)",
        domain=ScientificDomain.MOLECULAR_INTERACTION,
        terms=(("V", "eV"), ("4 epsilon ((sigma/r)^12 - (sigma/r)^6)", "eV")),
        symbols=(("V", "eV"), ("r", "m"), ("epsilon", "eV"), ("sigma", "m")),
        conditions=("pairwise additive", "non-bonded atom pairs"),
        assumptions=("isotropic pair interaction",),
    )


_CONCEPTS: tuple[ScienceConceptRecord, ...] = (
    ScienceConceptRecord(
        concept_id="SCI-CONCEPT-HEAT-DIFFUSION",
        label="heat diffusion",
        definition=(
            "Transport of thermal energy through a medium governed by the heat equation, "
            "with thermal diffusivity alpha in m^2/s controlling the spreading rate."
        ),
        domain=ScientificDomain.PHYSICAL_SIMULATION,
        unit_symbol="m2_s",
        related_equations=("EQ-HEAT-1D",),
    ),
    ScienceConceptRecord(
        concept_id="SCI-CONCEPT-THERMAL-DIFFUSIVITY",
        label="thermal diffusivity",
        definition=(
            "Ratio of thermal conductivity to volumetric heat capacity; determines how "
            "quickly a temperature disturbance propagates. Unit: m^2/s."
        ),
        domain=ScientificDomain.PHYSICAL_SIMULATION,
        unit_symbol="m2_s",
        related_equations=("EQ-HEAT-1D",),
    ),
    ScienceConceptRecord(
        concept_id="SCI-CONCEPT-TEMPERATURE",
        label="temperature",
        definition="Mean kinetic energy measure of a system; SI base unit kelvin (K).",
        domain=ScientificDomain.PHYSICAL_SIMULATION,
        unit_symbol="K",
    ),
    ScienceConceptRecord(
        concept_id="SCI-CONCEPT-COVALENT-BOND",
        label="covalent bond",
        definition=(
            "Chemical bond formed by shared electron pairs; typical lengths 0.12-0.20 nm "
            "depending on element pair and bond order."
        ),
        domain=ScientificDomain.MOLECULAR_STRUCTURE,
    ),
    ScienceConceptRecord(
        concept_id="SCI-CONCEPT-PROTEIN-BACKBONE",
        label="protein backbone",
        definition=(
            "Polypeptide chain of residue units; Ramachandran phi/psi dihedral angles "
            "constrain admissible backbone conformations."
        ),
        domain=ScientificDomain.MOLECULAR_STRUCTURE,
        related_equations=(),
    ),
    ScienceConceptRecord(
        concept_id="SCI-CONCEPT-CRYSTAL-LATTICE",
        label="crystal lattice",
        definition=(
            "Periodic arrangement of atomic sites defined by a unit cell and lattice "
            "vectors; periodic boundary conditions apply at cell faces."
        ),
        domain=ScientificDomain.MATERIALS,
    ),
    ScienceConceptRecord(
        concept_id="SCI-CONCEPT-CHARGE-NEUTRALITY",
        label="charge neutrality",
        definition=(
            "Stable bulk materials must carry zero net charge: the weighted sum of "
            "formal oxidation states over the composition must vanish."
        ),
        domain=ScientificDomain.MATERIALS,
    ),
    ScienceConceptRecord(
        concept_id="SCI-CONCEPT-ENSEMBLE-FORECAST",
        label="ensemble forecast",
        definition=(
            "A set of M forecast trajectories representing conditional uncertainty; "
            "event probabilities are estimated by member fractions."
        ),
        domain=ScientificDomain.FORECASTING,
    ),
    ScienceConceptRecord(
        concept_id="SCI-CONCEPT-FACTOR-GRAPH",
        label="factor graph",
        definition=(
            "Bipartite graph of variable nodes and factor nodes encoding likelihoods and "
            "constraints; MAP estimation reduces to weighted nonlinear least squares."
        ),
        domain=ScientificDomain.LOCALIZATION,
    ),
    ScienceConceptRecord(
        concept_id="SCI-CONCEPT-EQUIVARIANCE",
        label="equivariance",
        definition=(
            "Property f(g*x) = rho(g) f(x): a model output transforms predictably under "
            "a group action; invariance is the special case rho(g) = identity."
        ),
        domain=ScientificDomain.GEOMETRIC,
    ),
)

_EQUATIONS: tuple[ScienceEquationRecord, ...] = (
    _heat_equation(),
    _ideal_gas_law(),
    _newton_second_law(),
    _hookes_law(),
    _lj_potential(),
)


class ScienceKnowledgeBase:
    """Local, read-only scientific knowledge store with exact-match retrieval.

    Retrieval supplies semantic context only. USG's semantic structure never certifies
    that a claim is empirically true; that distinction is carried in every record.
    """

    def __init__(
        self,
        concepts: Iterable[ScienceConceptRecord] = _CONCEPTS,
        equations: Iterable[ScienceEquationRecord] = _EQUATIONS,
    ) -> None:
        """Assemble the store and index records by normalized label."""
        self._concepts: dict[str, ScienceConceptRecord] = {}
        self._equations: dict[str, ScienceEquationRecord] = {}
        for concept in concepts:
            key = concept.label.strip().lower()
            if key in self._concepts:
                raise ValueError(f"duplicate concept label {concept.label!r}")
            self._concepts[key] = concept
        for equation in equations:
            key = equation.name.strip().lower()
            if key in self._equations:
                raise ValueError(f"duplicate equation name {equation.name!r}")
            self._equations[key] = equation

    # -- retrieval ------------------------------------------------------------
    def lookup_concept(self, label: str) -> ScienceConceptRecord | None:
        """Return the concept whose label exactly matches, case-insensitively."""
        return self._concepts.get(label.strip().lower())

    def lookup_equation(self, name: str) -> ScienceEquationRecord | None:
        """Return the equation whose name exactly matches, case-insensitively."""
        return self._equations.get(name.strip().lower())

    def equations_for_domain(self, domain: ScientificDomain) -> tuple[ScienceEquationRecord, ...]:
        """Return every equation record applicable to a domain."""
        return tuple(record for record in self._equations.values() if record.domain == domain)

    def retrieve(self, query: str) -> list[ScienceConceptRecord]:
        """Return concepts whose label appears as a word in the query (no fuzzy match)."""
        tokens = {token.strip(".,;:()").lower() for token in query.split()}
        hits = []
        for key, record in self._concepts.items():
            if key in tokens:
                hits.append(record)
        return hits

    def concepts(self) -> tuple[ScienceConceptRecord, ...]:
        """Return every concept record in the store."""
        return tuple(self._concepts.values())

    def equations(self) -> tuple[ScienceEquationRecord, ...]:
        """Return every equation record in the store."""
        return tuple(self._equations.values())

    # -- task support ---------------------------------------------------------
    def constraints_for_domain(self, domain: ScientificDomain) -> tuple[Constraint, ...]:
        """Return domain rules as typed constraints sourced from USG retrieval."""
        constraints: list[Constraint] = []
        if domain in (ScientificDomain.PHYSICAL_SIMULATION,):
            constraints.append(
                Constraint(
                    kind="conservation_law",
                    expression="d/dt ∫ u dx <= 0 (no sources)",
                    description="Heat energy in a closed rod without sources is non-increasing.",
                    source=AssumptionSource.USG,
                    tolerance=1.0e-6,
                )
            )
        if domain is ScientificDomain.MATERIALS:
            constraints.append(
                Constraint(
                    kind="chemical_rule",
                    expression="sum(z_i * n_i) = 0",
                    description="Bulk materials must be charge neutral.",
                    source=AssumptionSource.USG,
                )
            )
            constraints.append(
                Constraint(
                    kind="chemical_rule",
                    expression="1 Angstrom <= lattice parameter <= 15 Angstrom",
                    description="Admissible range for crystalline lattice constants.",
                    source=AssumptionSource.USG,
                )
            )
        return tuple(constraints)

    def assumptions_for_equation(self, equation: ScienceEquationRecord) -> tuple[Assumption, ...]:
        """Return the equation's stated assumptions as typed assumption records."""
        return tuple(
            Assumption(
                statement=statement,
                source=AssumptionSource.USG,
                detail=f"stated condition of {equation.equation_id}",
            )
            for statement in equation.assumptions
        )


class USGScienceBridge:
    """Optional read-only adapter from an ``ore_usg`` knowledge resource to science concepts.

    The bridge converts USG semantic concepts into science concept records when an
    actual ``USGKnowledgeResource`` is supplied. It never writes back: candidate
    scientific knowledge updates are handled by the orchestration layer's review
    workflow, not by USG retrieval.
    """

    def __init__(self, resource: Any | None = None) -> None:
        """Store the optional USG resource; None disables the bridge."""
        self.resource = resource
        self.imported_count = 0
        self.import_errors: list[str] = []

    def convert_concepts(self) -> list[ScienceConceptRecord]:
        """Convert USG semantic concepts into science concept records, read-only."""
        records: list[ScienceConceptRecord] = []
        if self.resource is None:
            return records
        concepts = getattr(self.resource, "concepts", None) or ()
        for concept in concepts:
            try:
                records.append(
                    ScienceConceptRecord(
                        concept_id=f"USG-{getattr(concept, 'concept_id', 'unknown')}",
                        label=str(getattr(concept, "label", "")).strip().lower(),
                        definition=(
                            f"USG-imported concept '{getattr(concept, 'label', '')}' "
                            "with lifecycle state "
                            f"{getattr(concept, 'verified_by', 'unverified')}."
                        ),
                        domain=ScientificDomain.GENERAL_SCIENCE,
                        source_reference="ore_usg.USGKnowledgeResource",
                    )
                )
            except (ValueError, AttributeError, TypeError) as error:
                self.import_errors.append(str(error))
        self.imported_count = len(records)
        return records

    def merge_into(self, base: ScienceKnowledgeBase) -> ScienceKnowledgeBase:
        """Return a new knowledge base extended with bridgeable USG concepts."""
        imported = self.convert_concepts()
        existing = {concept.label.strip().lower() for concept in base.concepts()}
        merged = [concept for concept in imported if concept.label not in existing]
        return ScienceKnowledgeBase(
            concepts=(*base.concepts(), *merged),
            equations=base.equations(),
        )
