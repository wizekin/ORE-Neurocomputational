"""Scientific task interpreter: domain routing, objective extraction, and formalization.

This module implements Phase I of the integrated algorithm (Steps 1-7). A natural-language
request plus optional structured facts are converted into a typed ``ScientificTask``:
domains, objective, variable registry with units, constraints, assumptions, dimensional
checks, problem class, and evidence requirements. The parser is deliberately
rule-based: it is an engineering heuristic, not a claim of natural-language
understanding, and every ambiguity it cannot resolve becomes an explicit limitation
rather than a silent guess.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .contracts import (
    Assumption,
    AssumptionSource,
    ConsequenceLevel,
    Constraint,
    DomainDecision,
    EvidenceRequirements,
    ObjectiveKind,
    ProblemClass,
    ScienceRequest,
    ScientificDomain,
    ScientificTask,
    ScientificVariable,
    SubTaskLink,
    VariableRole,
)
from .knowledge import ScienceKnowledgeBase
from .units import UNITS, Quantity, quantity_from_string

__all__ = ["DomainRouter", "ScientificTaskParser"]


_DOMAIN_KEYWORDS: tuple[tuple[ScientificDomain, tuple[str, ...]], ...] = (
    (
        ScientificDomain.MOLECULAR_STRUCTURE,
        ("protein structure", "predict the structure", "fold", "3d structure", "backbone", "residue"),
    ),
    (
        ScientificDomain.MOLECULAR_INTERACTION,
        ("interaction", "binding", "binds", "dock", "ligand", "protein and", "complex"),
    ),
    (
        ScientificDomain.PROTEIN_DESIGN,
        ("design", "de novo", "generate a backbone", "new protein", "rfdiffusion"),
    ),
    (
        ScientificDomain.MATERIALS,
        ("material", "crystal", "perovskite", "battery", "synthesis", "oxide", "lattice"),
    ),
    (
        ScientificDomain.PHYSICAL_SIMULATION,
        ("temperature field", "heat", "diffusion", "simulate", "simulation", "wave equation",
         "fluid", "flow", "field", "pde", "partial differential"),
    ),
    (
        ScientificDomain.GEOMETRIC,
        ("rotation", "equivariant", "transform", "geometry", "coordinate frame"),
    ),
    (
        ScientificDomain.FORECASTING,
        ("forecast", "weather", "ensemble", "wind", "trajectory", "predict the future",
         "probabilistic forecast", "storm"),
    ),
    (
        ScientificDomain.LOCALIZATION,
        ("slam", "localize", "localization", "odometry", "landmark", "robot pose", "gps",
         "mapping and localization"),
    ),
)

_OBJECTIVE_KEYWORDS: tuple[tuple[ObjectiveKind, tuple[str, ...]], ...] = (
    (ObjectiveKind.PREDICT_STRUCTURE, ("predict the structure", "structure of", "fold")),
    (ObjectiveKind.ESTIMATE_FIELD, ("field", "temperature", "profile", "distribution of")),
    (ObjectiveKind.FORECAST_DISTRIBUTION, ("forecast", "future", "ensemble", "trajectory")),
    (ObjectiveKind.INFER_STATE, ("localize", "estimate the pose", "infer", "state estimation", "slam")),
    (ObjectiveKind.GENERATE_CANDIDATE, ("design", "generate", "propose", "de novo", "candidate")),
    (ObjectiveKind.OPTIMIZE_PROPERTY, ("optimize", "maximize", "minimize", "best property")),
    (ObjectiveKind.EXPLAIN_RELATION, ("explain", "why", "relationship", "how does")),
    (ObjectiveKind.VALIDATE_RESULT, ("validate", "check", "verify")),
)

# Extracts "<value> <unit>"-style measurements, including scientific notation.
# Compound units (m2/s, m/s, km/h) must precede their shorter prefixes in the alternation.
_QUANTITY_PATTERN = re.compile(
    r"([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)\s*(m2/s|m²/s|m/s|km/h|mm|cm|km|ms|min|degC|°C|degF|°F|"
    r"kg|mol|Pa|Hz|eV|m2|m²|m3|m³|m|s|h|K|g|N|J|W)\b"
)
_VARIABLE_NAME_PATTERN = re.compile(
    r"\b(alpha|beta|delta|sigma|epsilon|lambda|rho|L|T|u|x)\b\s*=\s*([0-9eE.+-]+)\s*([a-zA-Z²³/°]+)"
)


class DomainRouter:
    """Classify a request into scientific domains from explicit keyword evidence (Step 1)."""

    def classify(self, question: str) -> DomainDecision:
        """Return the matched domains, matched keywords, confidence, and reason."""
        text = question.lower()
        matched: list[ScientificDomain] = []
        keywords: list[str] = []
        for domain, patterns in _DOMAIN_KEYWORDS:
            for pattern in patterns:
                if pattern in text:
                    matched.append(domain)
                    keywords.append(pattern)
                    break
        if not matched:
            return DomainDecision(
                domains=(ScientificDomain.GENERAL_SCIENCE,),
                confidence=0.3,
                matched_keywords=(),
                reason="No domain-specific keyword evidence; routed to general science.",
            )
        confidence = min(0.5 + 0.15 * len(matched), 0.95)
        reason = (
            "Keyword routing matched: " + ", ".join(sorted(set(keywords))) +
            ". This is a heuristic routing decision, not a semantic understanding claim."
        )
        return DomainDecision(
            domains=tuple(dict.fromkeys(matched)),
            confidence=confidence,
            matched_keywords=tuple(sorted(set(keywords))),
            reason=reason,
        )


def _extract_objective(question: str) -> tuple[ObjectiveKind, str]:
    """Map the request to an explicit objective kind and objective statement (Step 2)."""
    text = question.lower()
    for kind, patterns in _OBJECTIVE_KEYWORDS:
        for pattern in patterns:
            if pattern in text:
                return kind, question.strip().rstrip(".?")
    return ObjectiveKind.ESTIMATE_FIELD, question.strip().rstrip(".?")


def _default_role(symbol: str, kind: ObjectiveKind) -> VariableRole:
    """Assign the default variable role for a symbol given the objective kind."""
    if symbol in ("u", "field", "T", "x", "pose", "trajectory"):
        return VariableRole.PREDICTED if kind in (ObjectiveKind.ESTIMATE_FIELD,) else VariableRole.LATENT
    return VariableRole.PARAMETER


_PROBLEM_CLASS_RULES: tuple[tuple[ProblemClass, tuple[str, ...]], ...] = (
    (ProblemClass.STATE_ESTIMATION, ("slam", "localize", "localization", "pose", "landmark", "odometry")),
    (ProblemClass.PROBABILISTIC_PREDICTION, ("forecast", "ensemble", "probabilistic", "distribution of futures")),
    (ProblemClass.GENERATIVE_DESIGN, ("design", "generate", "de novo", "candidate", "propose")),
    (ProblemClass.PARAMETER_ESTIMATION, ("estimate the parameter", "fit", "calibrate", "identify the parameter")),
    (ProblemClass.INVERSE, ("inverse", "recover the", "infer the initial")),
    (ProblemClass.CONSTRAINED_OPTIMIZATION, ("optimize", "maximize", "minimize", "best")),
)


def _extract_problem_class(question: str, kind: ObjectiveKind) -> ProblemClass:
    """Determine the mathematical problem class from the question and objective (Step 6)."""
    text = question.lower()
    for problem_class, patterns in _PROBLEM_CLASS_RULES:
        if any(pattern in text for pattern in patterns):
            return problem_class
    if kind is ObjectiveKind.FORECAST_DISTRIBUTION:
        return ProblemClass.PROBABILISTIC_PREDICTION
    if kind is ObjectiveKind.GENERATE_CANDIDATE:
        return ProblemClass.GENERATIVE_DESIGN
    if kind is ObjectiveKind.INFER_STATE:
        return ProblemClass.STATE_ESTIMATION
    if kind is ObjectiveKind.PREDICT_STRUCTURE:
        return ProblemClass.FORWARD
    return ProblemClass.FORWARD


def _extract_constraints(question: str) -> list[Constraint]:
    """Extract boundary and initial conditions stated in the question (Step 4)."""
    text = question.lower()
    constraints: list[Constraint] = []
    if "ends held at" in text or "held at" in text or "boundary" in text or "dirichlet" in text:
        constraints.append(
            Constraint(
                kind="boundary_condition",
                expression="u = u_boundary at domain boundary",
                description="Dirichlet boundary condition stated in the request.",
                source=AssumptionSource.USER,
                applies_to=("u",),
            )
        )
    if "initial" in text or "initially" in text or "starts at" in text or "at t=0" in text or "at t = 0" in text:
        constraints.append(
            Constraint(
                kind="initial_condition",
                expression="u(x, 0) = u_initial",
                description="Initial condition stated in the request.",
                source=AssumptionSource.USER,
                applies_to=("u",),
            )
        )
    if "insulated" in text or "neumann" in text:
        constraints.append(
            Constraint(
                kind="boundary_condition",
                expression="du/dn = 0 at boundary",
                description="Neumann (no-flux) boundary condition stated in the request.",
                source=AssumptionSource.USER,
                applies_to=("u",),
            )
        )
    if "periodic" in text:
        constraints.append(
            Constraint(
                kind="boundary_condition",
                expression="u(a) = u(b)",
                description="Periodic boundary convention stated in the request.",
                source=AssumptionSource.USER,
                applies_to=("u",),
            )
        )
    return constraints


_VARIABLE_MEANINGS: Mapping[str, str] = {
    "alpha": "thermal diffusivity or model rate parameter",
    "L": "domain length or characteristic length scale",
    "T": "time horizon or temperature (resolved by unit)",
    "u": "unknown scalar field (for example temperature)",
    "x": "spatial coordinate or state",
    "u_initial": "initial value of the unknown field",
    "u_boundary": "boundary value of the unknown field",
}

_UNIT_BY_VARIABLE: Mapping[str, str] = {
    "alpha": "m2_s",
    "L": "m",
    "u_initial": "K",
    "u_boundary": "K",
}


class ScientificTaskParser:
    """Convert a ``ScienceRequest`` into a typed ``ScientificTask`` (Steps 1-7).

    The parser resolves units, checks dimensional homogeneity of retrieved equations,
    and formalizes variables, constraints, and evidence requirements. Incompatible
    dimensions raise a ``ValueError``: formalization fails loudly instead of silently
    coercing values, as required by Step 5.
    """

    def __init__(self, knowledge: ScienceKnowledgeBase) -> None:
        """Bind the parser to a read-only scientific knowledge base."""
        self.knowledge = knowledge

    def parse(self, request: ScienceRequest) -> ScientificTask:
        """Return the structured task or raise on invalid or dimensionally inconsistent input."""
        router = DomainRouter()
        decision = router.classify(request.question)
        domains = decision.domains
        if request.domain_hint:
            hint = request.domain_hint.strip().upper()
            try:
                hint_domain = ScientificDomain(hint)
            except ValueError as error:
                raise ValueError(f"unknown domain hint {request.domain_hint!r}") from error
            if hint_domain not in domains:
                domains = (hint_domain, *domains)
                decision = DomainDecision(
                    domains=domains,
                    confidence=decision.confidence,
                    matched_keywords=decision.matched_keywords,
                    reason=decision.reason + f" Domain hint '{hint}' applied.",
                )

        objective_kind, objective = _extract_objective(request.question)
        problem_class = _extract_problem_class(request.question, objective_kind)

        variables = self._extract_variables(request, objective_kind)
        constraints = _extract_constraints(request.question)
        for domain in domains:
            constraints.extend(self.knowledge.constraints_for_domain(domain))

        equations, equation_assumptions = self._retrieve_equations(domains, request)
        constraints = _deduplicate(constraints)

        assumptions = [
            Assumption(
                statement="Rule-based keyword routing; natural-language understanding is not claimed.",
                source=AssumptionSource.INFERRED,
            ),
            *equation_assumptions,
        ]
        if request.facts:
            assumptions.append(
                Assumption(
                    statement="Structured facts supplied by the requester are taken as given.",
                    source=AssumptionSource.USER,
                )
            )

        evidence = self._evidence_requirements(request, problem_class)
        subtasks = self._plan_subtasks(domains, request)
        task_id = f"sci_{abs(hash(request.question)) % 10**10:010d}"

        return ScientificTask(
            question=request.question,
            domains=domains,
            objective=objective,
            objective_kind=objective_kind,
            problem_class=problem_class,
            variables=tuple(variables),
            constraints=tuple(constraints),
            equations=tuple(equations),
            assumptions=tuple(assumptions),
            evidence_requirements=evidence,
            subtasks=tuple(subtasks),
            source_reference=request.source_reference,
            task_id=task_id,
        )

    # -- Step 3: variable registry -------------------------------------------
    def _extract_variables(
        self, request: ScienceRequest, objective_kind: ObjectiveKind
    ) -> list[ScientificVariable]:
        """Build the variable registry from the question text and structured facts."""
        from .units import DIM_TEMPERATURE

        registry: dict[str, ScientificVariable] = {}
        temperatures: list[tuple[int, float]] = []
        for match in _QUANTITY_PATTERN.finditer(request.question):
            value_text, unit_text = match.group(1), match.group(2)
            unit = _resolve_unit(unit_text)
            if unit is None:
                continue
            value = float(value_text)
            if unit.dimension == DIM_TEMPERATURE:
                temperatures.append((match.start(), value))
                continue
            symbol = _symbol_for_dimension(unit)
            variable = ScientificVariable(
                symbol=symbol,
                meaning=_VARIABLE_MEANINGS.get(symbol, f"quantity measured in {unit.symbol}"),
                role=_default_role(symbol, objective_kind),
                unit=unit,
                value=value,
                domain="extracted from request text",
                provenance="user_supplied",
            )
            if symbol not in registry or registry[symbol].unit.dimension != unit.dimension:
                registry[symbol] = variable

        for symbol, value in self._assign_temperatures(request.question, temperatures):
            registry[symbol] = ScientificVariable(
                symbol=symbol,
                meaning=_VARIABLE_MEANINGS.get(symbol, "temperature value"),
                role=_default_role(symbol, objective_kind),
                unit=UNITS["K"],
                value=value,
                domain="extracted from request text",
                provenance="user_supplied",
            )

        for name, raw in request.facts.items():
            quantity = quantity_from_string(str(raw))
            symbol = name.strip()
            unit = quantity.unit
            if symbol in registry and registry[symbol].unit.dimension != unit.dimension:
                raise ValueError(
                    f"variable {symbol!r} appears with incompatible units "
                    f"{registry[symbol].unit.symbol} and {unit.symbol}"
                )
            registry[symbol] = ScientificVariable(
                symbol=symbol,
                meaning=_VARIABLE_MEANINGS.get(symbol, f"fact supplied for {symbol}"),
                role=_default_role(symbol, objective_kind),
                unit=unit,
                value=quantity.value,
                domain="structured fact",
                provenance="user_supplied",
            )

        if objective_kind is ObjectiveKind.ESTIMATE_FIELD and "u" not in registry:
            registry["u"] = ScientificVariable(
                symbol="u",
                meaning=_VARIABLE_MEANINGS["u"],
                role=VariableRole.PREDICTED,
                unit=UNITS["K"],
                domain="predicted field",
                provenance="inferred from objective",
            )
        return list(registry.values())

    @staticmethod
    def _assign_temperatures(
        question: str, temperatures: list[tuple[int, float]]
    ) -> list[tuple[str, float]]:
        """Assign temperature values to initial/boundary roles using local text context."""
        text = question.lower()
        boundary_anchors = [m.start() for m in re.finditer(r"held at|ends|boundary", text)]
        initial_anchors = [
            m.start() for m in re.finditer(r"initial|initially|starts at|at t\s*=\s*0", text)
        ]
        assignments: list[tuple[str, float]] = []
        for position, value in temperatures:
            near_boundary = any(0 <= position - anchor <= 40 for anchor in boundary_anchors)
            near_initial = any(0 <= position - anchor <= 40 for anchor in initial_anchors)
            if near_boundary and not near_initial:
                assignments.append(("u_boundary", value))
            else:
                assignments.append(("u_initial", value))
        seen: dict[str, float] = {}
        for symbol, value in assignments:
            if symbol not in seen:
                seen[symbol] = value
        if "u_initial" in seen and "u_boundary" not in seen and len(temperatures) > 1:
            seen["u_boundary"] = temperatures[-1][1]
        return list(seen.items())

    # -- Step 5: dimensional validation during formalization -----------------
    def _retrieve_equations(
        self, domains: tuple[ScientificDomain, ...], request: ScienceRequest
    ) -> tuple[list[Any], list[Assumption]]:
        """Retrieve governing equations per domain and validate their term dimensions."""
        equations: list[Any] = []
        assumptions: list[Assumption] = []
        text = request.question.lower()
        for domain in domains:
            for record in self.knowledge.equations_for_domain(domain):
                model = record.to_equation_model()
                terms = model.term_quantities()
                homogeneous, message = _homogeneous(terms)
                if not homogeneous:
                    raise ValueError(
                        f"retrieved equation {model.equation_id} failed dimensional validation: {message}"
                    )
                if _equation_requested(model, text):
                    equations.append(model)
                    assumptions.extend(self.knowledge.assumptions_for_equation(record))
        return equations, assumptions

    # -- Step 7: evidence requirements ---------------------------------------
    @staticmethod
    def _evidence_requirements(request: ScienceRequest, problem_class: ProblemClass) -> EvidenceRequirements:
        """Set evidence requirements with consequence level driving validation strength."""
        text = request.question.lower()
        if "safety" in text or "clinical" in text or "production" in text or "deployment" in text:
            level = ConsequenceLevel.CONSEQUENTIAL
        elif "explor" in text or "rough" in text or "quick" in text:
            level = ConsequenceLevel.EXPLORATORY
        else:
            level = ConsequenceLevel.RESEARCH_SUPPORTING
        accuracy = None
        for match in re.finditer(r"(\d+(?:\.\d+)?)\s*%", text):
            accuracy = float(match.group(1)) / 100.0
        return EvidenceRequirements(
            accuracy=accuracy,
            confidence_level=0.95,
            resolution="domain default",
            prediction_horizon="task-dependent",
            consequence_level=level,
        )

    # -- Step 48 support: multi-domain decomposition --------------------------
    @staticmethod
    def _plan_subtasks(
        domains: tuple[ScientificDomain, ...], request: ScienceRequest
    ) -> list[SubTaskLink]:
        """Decompose multi-domain tasks into typed subtasks with explicit dependencies."""
        if len(domains) <= 1 or all(d is ScientificDomain.GENERAL_SCIENCE for d in domains):
            return []
        links: list[SubTaskLink] = []
        for index, domain in enumerate(domains):
            if domain is ScientificDomain.GENERAL_SCIENCE:
                continue
            links.append(
                SubTaskLink(
                    subtask_id=f"sub_{domain.value.lower()}_{index}",
                    domain=domain,
                    objective=f"Handle {domain.value.lower().replace('_', ' ')} portion of the request.",
                    depends_on=(),
                    question=request.question,
                )
            )
        for index in range(1, len(links)):
            links[index] = SubTaskLink(
                subtask_id=links[index].subtask_id,
                domain=links[index].domain,
                objective=links[index].objective,
                depends_on=(links[index - 1].subtask_id,),
                question=request.question,
            )
        return links


def _resolve_unit(unit_text: str):
    """Resolve a unit token from free text to a typed Unit, or None."""
    aliases = {
        "m": "m", "mm": "mm", "cm": "cm", "km": "km", "s": "s", "ms": "ms",
        "min": "min", "h": "h", "K": "K", "degC": "degC", "°C": "degC",
        "degF": "degF", "°F": "degF", "kg": "kg", "g": "g", "mol": "mol",
        "N": "N", "J": "J", "W": "W", "Pa": "Pa", "Hz": "Hz", "eV": "eV",
        "m2": "m2", "m²": "m2", "m3": "m3", "m³": "m3", "km/h": "km_h",
        "m/s": "m_s", "m2/s": "m2_s", "m²/s": "m2_s", "K/s": "K_s",
    }
    key = aliases.get(unit_text)
    return UNITS[key] if key else None


def _symbol_for_dimension(unit: Any) -> str:
    """Choose a variable symbol for a unit encountered in free text."""
    from .units import DIM_DIFFUSIVITY, DIM_LENGTH, DIM_TIME

    if unit.dimension == DIM_DIFFUSIVITY:
        return "alpha"
    if unit.dimension == DIM_LENGTH:
        return "L"
    if unit.dimension == DIM_TIME:
        return "T"
    return "x"


def _homogeneous(terms: list[Quantity]) -> tuple[bool, str]:
    """Check dimensional homogeneity across equation terms."""
    from .units import dimensional_homogeneity

    return dimensional_homogeneity(terms)


def _equation_requested(model: Any, text: str) -> bool:
    """Decide whether a retrieved equation is relevant to the question text."""
    keywords: Mapping[str, tuple[str, ...]] = {
        "EQ-HEAT-1D": ("heat", "diffusion", "temperature", "thermal"),
        "EQ-IDEAL-GAS": ("gas", "pressure", "volume", "ideal"),
        "EQ-NEWTON-2": ("force", "acceleration", "newton", "mass"),
        "EQ-HOOKE": ("spring", "hooke", "elastic", "stiffness"),
        "EQ-LENNARD-JONES": ("interaction", "potential", "van der waals", "lennard"),
    }
    triggers = keywords.get(model.equation_id, ())
    return any(trigger in text for trigger in triggers)


def _deduplicate(constraints: list[Constraint]) -> list[Constraint]:
    """Keep the first occurrence of each constraint expression."""
    seen: set[str] = set()
    unique: list[Constraint] = []
    for constraint in constraints:
        if constraint.expression not in seen:
            seen.add(constraint.expression)
            unique.append(constraint)
    return unique
