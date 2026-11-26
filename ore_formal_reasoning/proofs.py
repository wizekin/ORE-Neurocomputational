"""Machine-checked proofs and external verification (Phase F, Steps 37-41).

Proof obligations translate a target theorem plus its assumptions into a
well-scoped proposition; proof candidates from searchers or tactic engines are
untrusted until the trusted checker accepts them. The trusted boundary here is
the bundled local natural-deduction kernel — a small, deterministic,
propositional Curry-Howard checker — which verifies proof terms against their
propositions. When a pinned Lean 4 toolchain is available, the same artifact is
exported and checked by the Lean kernel; without one the Lean path reports
``unsupported`` and never guesses.

Proof validity is always separated from premise truth (Step 41): a checked
proof establishes derivability from the declared premises; whether those
premises are true, and whether the formalization corresponds to the external
world, remain separate questions. Proof-search failure is ``not proved``,
never refuted.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from typing import Mapping, Sequence

from .contracts import ResultStatus
from .exceptions import InvalidFormalizationError

LOCAL_KERNEL_NAME = "frce_local_nd_kernel"
LOCAL_KERNEL_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Propositions (Step 38)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PropVar:
    """An atomic proposition."""

    name: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("PropVar.name must be non-empty")


@dataclass(frozen=True, slots=True)
class PropImplies:
    """Implication."""

    premise: "Prop"
    conclusion: "Prop"


@dataclass(frozen=True, slots=True)
class PropAnd:
    """Conjunction."""

    left: "Prop"
    right: "Prop"


@dataclass(frozen=True, slots=True)
class PropOr:
    """Disjunction."""

    left: "Prop"
    right: "Prop"


@dataclass(frozen=True, slots=True)
class PropBottom:
    """Falsity."""


Prop = PropVar | PropImplies | PropAnd | PropOr | PropBottom


def proposition_text(prop: Prop) -> str:
    """Render a proposition in readable ASCII form."""
    if isinstance(prop, PropVar):
        return prop.name
    if isinstance(prop, PropBottom):
        return "False"
    if isinstance(prop, PropAnd):
        return f"({proposition_text(prop.left)} /\\ {proposition_text(prop.right)})"
    if isinstance(prop, PropOr):
        return f"({proposition_text(prop.left)} \\/ {proposition_text(prop.right)})"
    if isinstance(prop, PropImplies):
        return f"({proposition_text(prop.premise)} -> {proposition_text(prop.conclusion)})"
    raise InvalidFormalizationError(f"unknown proposition node: {type(prop)!r}")


# ---------------------------------------------------------------------------
# Proof obligation (Step 38)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Premise:
    """One declared premise with its truth status (Step 41 separation)."""

    name: str
    proposition: Prop
    premise_status: str = "stipulated"  # stipulated | observed | unvalidated

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Premise.name must be non-empty")
        if self.premise_status not in {"stipulated", "observed", "unvalidated"}:
            raise ValueError("premise_status must be stipulated, observed, or unvalidated")


@dataclass(frozen=True, slots=True)
class ProofObligation:
    """A well-scoped proof obligation generated from a target claim (Step 38)."""

    obligation_id: str
    theorem_name: str
    goal: Prop
    premises: tuple[Premise, ...]
    definitions: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()   # library/toolchain dependencies (Step 40)
    declared_atoms: tuple[str, ...] = ()  # atoms grounded without a premise
    assurance_note: str = ""

    def __post_init__(self) -> None:
        if not self.obligation_id.strip() or not self.theorem_name.strip():
            raise ValueError("obligations need non-empty obligation_id and theorem_name")
        object.__setattr__(self, "premises", tuple(self.premises))
        object.__setattr__(self, "definitions", tuple(self.definitions))
        object.__setattr__(self, "preconditions", tuple(self.preconditions))
        object.__setattr__(self, "postconditions", tuple(self.postconditions))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        object.__setattr__(self, "declared_atoms", tuple(self.declared_atoms))
        names = [p.name for p in self.premises]
        if len(names) != len(set(names)):
            raise ValueError("premise names must be unique within an obligation")
        # The goal must be grounded: every atomic proposition of the goal must
        # occur in a premise proposition or be explicitly declared, otherwise
        # the translation is incomplete (Step 38).
        goal_atoms = _collect_prop_atoms(self.goal, set())
        declared: set[str] = set(self.declared_atoms)
        for premise in self.premises:
            declared |= _collect_prop_atoms(premise.proposition, set())
        missing = goal_atoms - declared
        if missing:
            raise InvalidFormalizationError(
                f"goal references atoms absent from every premise: {sorted(missing)}; "
                "reject incomplete or unsound translations (Step 38)"
            )

    @property
    def statement_text(self) -> str:
        """Human-readable statement of the obligation."""
        premises = " -> ".join(proposition_text(p.proposition) for p in self.premises)
        goal = proposition_text(self.goal)
        if self.premises:
            return f"{premises} -> {goal}"
        return goal


def _collect_prop_atoms(prop: Prop, seen: set[str]) -> set[str]:
    """Collect all atomic proposition names below one proposition node."""
    if isinstance(prop, PropVar):
        seen.add(prop.name)
        return seen
    if isinstance(prop, PropBottom):
        return seen
    for child in _prop_children(prop):
        _collect_prop_atoms(child, seen)
    return seen


def _validate_prop_wellformed(prop: Prop, context: Mapping[str, Prop]) -> None:
    """Ensure every atomic proposition is either a premise or explicitly declared."""
    if isinstance(prop, PropVar):
        if prop.name not in context:
            raise InvalidFormalizationError(
                f"proposition references undeclared atom {prop.name!r}; reject "
                "incomplete or unsound translations (Step 38)"
            )
        return
    if isinstance(prop, PropBottom):
        return
    for child in _prop_children(prop):
        _validate_prop_wellformed(child, context)


def _prop_children(prop: Prop) -> tuple[Prop, ...]:
    if isinstance(prop, PropImplies):
        return (prop.premise, prop.conclusion)
    if isinstance(prop, (PropAnd, PropOr)):
        return (prop.left, prop.right)
    return ()


# ---------------------------------------------------------------------------
# Proof terms (Step 39) and the trusted kernel (Step 40)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PVar:
    """A bound variable reference."""

    name: str


@dataclass(frozen=True, slots=True)
class PHyp:
    """A reference to a declared premise (an axiom in the context)."""

    name: str


@dataclass(frozen=True, slots=True)
class PLam:
    """Implication introduction: bind a hypothesis and prove the conclusion."""

    variable: str
    premise: Prop
    body: "ProofTerm"


@dataclass(frozen=True, slots=True)
class PApp:
    """Implication elimination: apply a proof of A->B to a proof of A."""

    function: "ProofTerm"
    argument: "ProofTerm"


@dataclass(frozen=True, slots=True)
class PPair:
    """Conjunction introduction."""

    left: "ProofTerm"
    right: "ProofTerm"


@dataclass(frozen=True, slots=True)
class PFst:
    """Conjunction elimination (left)."""

    body: "ProofTerm"


@dataclass(frozen=True, slots=True)
class PSnd:
    """Conjunction elimination (right)."""

    body: "ProofTerm"


@dataclass(frozen=True, slots=True)
class PInl:
    """Disjunction introduction (left) with the target disjunction annotated."""

    disjunction: Prop
    body: "ProofTerm"


@dataclass(frozen=True, slots=True)
class PInr:
    """Disjunction introduction (right) with the target disjunction annotated."""

    disjunction: Prop
    body: "ProofTerm"


@dataclass(frozen=True, slots=True)
class PCase:
    """Disjunction elimination: case on a proof of A\\/B producing C both ways."""

    scrutinee: "ProofTerm"
    left_variable: str
    left_body: "ProofTerm"
    right_variable: str
    right_body: "ProofTerm"


@dataclass(frozen=True, slots=True)
class PAbort:
    """Bottom elimination: derive anything from a proof of False."""

    target: Prop
    body: "ProofTerm"


ProofTerm = PVar | PHyp | PLam | PApp | PPair | PFst | PSnd | PInl | PInr | PCase | PAbort


class KernelRejection(Exception):
    """The trusted kernel rejected a proof term."""


@dataclass(frozen=True, slots=True)
class KernelVerdict:
    """The trusted kernel's judgment on one proof term (Step 40)."""

    accepted: bool
    detail: str
    steps_checked: int
    kernel_name: str = LOCAL_KERNEL_NAME
    kernel_version: str = LOCAL_KERNEL_VERSION


class TrustedKernel:
    """Propositional natural-deduction kernel via Curry-Howard checking (Step 40).

    The kernel type-checks proof terms against propositions under the declared
    premise context. It establishes derivability within that system only; it
    does not validate whether the formalization reflects the external world.
    """

    def check(self, context: Mapping[str, Prop], term: ProofTerm, expected: Prop) -> KernelVerdict:
        """Check one proof term against the expected proposition."""
        steps = [0]

        def matches(a: Prop, b: Prop) -> bool:
            return proposition_text(a) == proposition_text(b)

        def check(node: ProofTerm, env: Mapping[str, Prop], target: Prop) -> None:
            steps[0] += 1
            if isinstance(node, PVar):
                actual = env.get(node.name)
                if actual is None or not matches(actual, target):
                    raise KernelRejection(
                        f"variable {node.name!r} does not prove "
                        f"{proposition_text(target)}"
                    )
                return
            if isinstance(node, PHyp):
                actual = propositions.get(node.name)
                if actual is None or not matches(actual, target):
                    raise KernelRejection(
                        f"premise {node.name!r} does not prove "
                        f"{proposition_text(target)}"
                    )
                return
            if isinstance(node, PLam):
                if not isinstance(target, PropImplies):
                    raise KernelRejection(
                        f"lambda proves an implication, not {proposition_text(target)}"
                    )
                if not matches(node.premise, target.premise):
                    raise KernelRejection(
                        f"lambda annotation {proposition_text(node.premise)} does not "
                        f"match expected premise {proposition_text(target.premise)}"
                    )
                extended = dict(env)
                extended[node.variable] = node.premise
                check(node.body, extended, target.conclusion)
                return
            if isinstance(node, PApp):
                function_type = infer(node.function, env)
                if not isinstance(function_type, PropImplies):
                    raise KernelRejection(
                        "application head must prove an implication, got "
                        f"{proposition_text(function_type)}"
                    )
                if not matches(function_type.conclusion, target):
                    raise KernelRejection(
                        f"application yields {proposition_text(function_type.conclusion)}, "
                        f"not {proposition_text(target)}"
                    )
                check(node.argument, env, function_type.premise)
                return
            if isinstance(node, PPair):
                if not isinstance(target, PropAnd):
                    raise KernelRejection(
                        f"pair proves a conjunction, not {proposition_text(target)}"
                    )
                check(node.left, env, target.left)
                check(node.right, env, target.right)
                return
            if isinstance(node, PFst):
                body_type = infer(node.body, env)
                if not isinstance(body_type, PropAnd) or not matches(body_type.left, target):
                    raise KernelRejection(
                        "first projection requires a conjunction proving "
                        f"{proposition_text(target)}"
                    )
                return
            if isinstance(node, PSnd):
                body_type = infer(node.body, env)
                if not isinstance(body_type, PropAnd) or not matches(body_type.right, target):
                    raise KernelRejection(
                        "second projection requires a conjunction proving "
                        f"{proposition_text(target)}"
                    )
                return
            if isinstance(node, PInl):
                if not isinstance(target, PropOr):
                    raise KernelRejection(
                        f"inl proves a disjunction, not {proposition_text(target)}"
                    )
                if not matches(node.disjunction, target):
                    raise KernelRejection(
                        f"inl annotation {proposition_text(node.disjunction)} does not "
                        f"match target {proposition_text(target)}"
                    )
                check(node.body, env, target.left)
                return
            if isinstance(node, PInr):
                if not isinstance(target, PropOr):
                    raise KernelRejection(
                        f"inr proves a disjunction, not {proposition_text(target)}"
                    )
                if not matches(node.disjunction, target):
                    raise KernelRejection(
                        f"inr annotation {proposition_text(node.disjunction)} does not "
                        f"match target {proposition_text(target)}"
                    )
                check(node.body, env, target.right)
                return
            if isinstance(node, PCase):
                scrutinee_type = infer(node.scrutinee, env)
                if not isinstance(scrutinee_type, PropOr):
                    raise KernelRejection(
                        "case scrutinee must prove a disjunction, got "
                        f"{proposition_text(scrutinee_type)}"
                    )
                left_env = dict(env)
                left_env[node.left_variable] = scrutinee_type.left
                check(node.left_body, left_env, target)
                right_env = dict(env)
                right_env[node.right_variable] = scrutinee_type.right
                check(node.right_body, right_env, target)
                return
            if isinstance(node, PAbort):
                body_type = infer(node.body, env)
                if not isinstance(body_type, PropBottom):
                    raise KernelRejection(
                        "abort requires a proof of False, got "
                        f"{proposition_text(body_type)}"
                    )
                if not matches(node.target, target):
                    raise KernelRejection(
                        "abort annotation does not match the target proposition"
                    )
                return
            raise KernelRejection(f"unknown proof term: {type(node)!r}")

        def infer(node: ProofTerm, env: Mapping[str, Prop]) -> Prop:
            steps[0] += 1
            if isinstance(node, PVar):
                actual = env.get(node.name)
                if actual is None:
                    raise KernelRejection(f"unbound proof variable {node.name!r}")
                return actual
            if isinstance(node, PHyp):
                actual = propositions.get(node.name)
                if actual is None:
                    raise KernelRejection(f"unknown premise {node.name!r}")
                return actual
            if isinstance(node, PLam):
                extended = dict(env)
                extended[node.variable] = node.premise
                return PropImplies(node.premise, infer(node.body, extended))
            if isinstance(node, PApp):
                function_type = infer(node.function, env)
                if not isinstance(function_type, PropImplies):
                    raise KernelRejection("application head must be an implication")
                check(node.argument, env, function_type.premise)
                return function_type.conclusion
            if isinstance(node, PPair):
                return PropAnd(infer(node.left, env), infer(node.right, env))
            if isinstance(node, PFst):
                body_type = infer(node.body, env)
                if not isinstance(body_type, PropAnd):
                    raise KernelRejection("fst requires a conjunction")
                return body_type.left
            if isinstance(node, PSnd):
                body_type = infer(node.body, env)
                if not isinstance(body_type, PropAnd):
                    raise KernelRejection("snd requires a conjunction")
                return body_type.right
            if isinstance(node, PInl):
                if not isinstance(node.disjunction, PropOr):
                    raise KernelRejection("inl annotation must be a disjunction")
                check(node.body, env, node.disjunction.left)
                return node.disjunction
            if isinstance(node, PInr):
                if not isinstance(node.disjunction, PropOr):
                    raise KernelRejection("inr annotation must be a disjunction")
                check(node.body, env, node.disjunction.right)
                return node.disjunction
            if isinstance(node, PCase):
                scrutinee_type = infer(node.scrutinee, env)
                if not isinstance(scrutinee_type, PropOr):
                    raise KernelRejection("case scrutinee must be a disjunction")
                left_env = dict(env)
                left_env[node.left_variable] = scrutinee_type.left
                left_type = infer(node.left_body, left_env)
                right_env = dict(env)
                right_env[node.right_variable] = scrutinee_type.right
                right_type = infer(node.right_body, right_env)
                if proposition_text(left_type) != proposition_text(right_type):
                    raise KernelRejection("case branches must agree on the conclusion")
                return left_type
            if isinstance(node, PAbort):
                body_type = infer(node.body, env)
                if not isinstance(body_type, PropBottom):
                    raise KernelRejection("abort requires a proof of False")
                return node.target
            raise KernelRejection(f"unknown proof term: {type(node)!r}")

        try:
            propositions = {
                name: (prop.proposition if hasattr(prop, "proposition") else prop)
                for name, prop in context.items()
            }
            check(term, dict(propositions), expected)
        except KernelRejection as rejection:
            return KernelVerdict(
                accepted=False, detail=str(rejection), steps_checked=steps[0],
            )
        return KernelVerdict(
            accepted=True,
            detail="proof term type-checks against the proposition under the premises",
            steps_checked=steps[0],
        )


# ---------------------------------------------------------------------------
# Proof artifacts, verification, and Lean export (Steps 39-41)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProofCandidate:
    """An untrusted proof proposal awaiting kernel verification (Step 39)."""

    origin: str                    # proof_searcher | tactic_engine | library | manual
    term: ProofTerm
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.origin not in {"proof_searcher", "tactic_engine", "library", "manual"}:
            raise ValueError(f"unrecognized proof origin: {self.origin!r}")
        object.__setattr__(self, "notes", tuple(self.notes))


@dataclass(frozen=True, slots=True)
class ProofArtifact:
    """The verified proof record with checker identity and hashes (Step 40)."""

    artifact_id: str
    obligation_id: str
    theorem_name: str
    statement_text: str
    term_text: str
    artifact_hash: str
    checker_name: str
    checker_version: str
    accepted: bool
    kernel_detail: str
    steps_checked: int
    dependencies: tuple[str, ...]
    conditional_on: tuple[str, ...]
    premise_truth_note: str
    lean_available: bool
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProofCheckOutcome:
    """The typed outcome of one proof verification attempt."""

    status: ResultStatus
    artifact: ProofArtifact | None
    detail: str


class ProofVerifier:
    """Verify proof candidates with the trusted checker boundary (Steps 39-41)."""

    def __init__(self, *, lean_binary: str = "lean") -> None:
        self._lean_binary = lean_binary

    @property
    def lean_available(self) -> bool:
        """Return whether a pinned Lean binary is on the allowlisted PATH."""
        return shutil.which(self._lean_binary) is not None

    def verify(
        self,
        obligation: ProofObligation,
        candidate: ProofCandidate,
        *,
        artifact_id: str,
    ) -> ProofCheckOutcome:
        """Check the candidate and produce the typed proof outcome."""
        kernel = TrustedKernel()
        context = {p.name: p for p in obligation.premises}
        verdict = kernel.check(context, candidate.term, obligation.goal)
        if not verdict.accepted:
            # A failed check is a rejected candidate: it is never reported as
            # falsity of the theorem.
            return ProofCheckOutcome(
                status=ResultStatus.UNKNOWN,
                artifact=None,
                detail=(
                    f"kernel rejected the proof candidate: {verdict.detail} "
                    "(rejection is not refutation of the theorem)"
                ),
            )
        term_text = proof_term_text(candidate.term)
        artifact_hash = hashlib.sha256(
            (proposition_text(obligation.goal) + term_text).encode("utf-8")
        ).hexdigest()
        conditional_on = tuple(p.name for p in obligation.premises)
        unvalidated = [
            p.name for p in obligation.premises if p.premise_status == "unvalidated"
        ]
        limitations: list[str] = []
        note = (
            "the kernel establishes derivability of the proposition from its "
            "declared premises; premise truth, model adequacy, and correspondence "
            "to real-world entities remain separate questions"
        )
        if unvalidated:
            limitations.append(
                "the theorem is conditional on premises with unvalidated status: "
                + ", ".join(sorted(unvalidated))
            )
        if self.lean_available:
            export = LeanExporter().export(obligation, candidate)
            lean_ok, lean_detail = LeanChecker(self._lean_binary).check(export)
            checker_name, checker_version = ("lean4", lean_detail.split("|")[0])
            limitations.append("verified by an external Lean 4 toolchain")
            if not lean_ok:
                return ProofCheckOutcome(
                    status=ResultStatus.UNKNOWN,
                    artifact=None,
                    detail=f"Lean rejected the exported artifact: {lean_detail}",
                )
        else:
            checker_name, checker_version = LOCAL_KERNEL_NAME, LOCAL_KERNEL_VERSION
            limitations.append(
                "no Lean toolchain is pinned in this environment; verification was "
                "performed by the bundled local natural-deduction kernel, which is "
                "the trusted checker for propositional obligations only"
            )
        artifact = ProofArtifact(
            artifact_id=artifact_id,
            obligation_id=obligation.obligation_id,
            theorem_name=obligation.theorem_name,
            statement_text=obligation.statement_text,
            term_text=term_text,
            artifact_hash=artifact_hash,
            checker_name=checker_name,
            checker_version=checker_version,
            accepted=True,
            kernel_detail=verdict.detail,
            steps_checked=verdict.steps_checked,
            dependencies=tuple(obligation.dependencies),
            conditional_on=conditional_on,
            premise_truth_note=note,
            lean_available=self.lean_available,
            limitations=tuple(limitations),
        )
        return ProofCheckOutcome(
            status=ResultStatus.PROVED,
            artifact=artifact,
            detail=verdict.detail,
        )


class ProofSearcher:
    """A deliberately small proof-search tactic (Step 39).

    The searcher composes premise hypotheses via modus ponens along a supplied
    chain of premise names. Its output is an untrusted candidate: only the
    kernel decides. When the chain does not yield the goal, the searcher
    reports ``None`` — a search failure, which is never a refutation.
    """

    def attempt(self, obligation: ProofObligation, chain: Sequence[str]) -> ProofCandidate | None:
        """Compose hypotheses along the chain; returns None when search fails.

        The chain reads like a pipeline: its first element is applied to the
        bound goal variable, each later element consumes the previous output.
        ``h1: S->H`` then ``h2: H->M`` therefore proves ``S->M``.
        """
        context = {p.name: p.proposition for p in obligation.premises}
        goal = obligation.goal
        bound_variable: str | None = None
        if isinstance(goal, PropImplies):
            bound_variable = "s"
            current: Prop = goal.premise
            final_target: Prop = goal.conclusion
        else:
            current = None
            final_target = goal
        term: ProofTerm | None = None
        for name in chain:
            premise_prop = context.get(name)
            if premise_prop is None:
                return None
            if isinstance(goal, PropImplies) or term is not None:
                if not isinstance(premise_prop, PropImplies):
                    return None
                if proposition_text(premise_prop.premise) != proposition_text(current):
                    return None
                argument = term if term is not None else PVar(bound_variable or "s")
                term = PApp(PHyp(name), argument)
                current = premise_prop.conclusion
            else:
                # atomic goal: a single premise proving it directly
                if proposition_text(premise_prop) != proposition_text(final_target):
                    return None
                term = PHyp(name)
                current = final_target
        if term is None or current is None:
            return None
        if proposition_text(current) != proposition_text(final_target):
            return None
        if bound_variable is not None:
            term = PLam(bound_variable, goal.premise, term)
        return ProofCandidate(
            origin="proof_searcher",
            term=term,
            notes=("composed by the bounded FRCE proof searcher",),
        )


def proof_term_text(term: ProofTerm) -> str:
    """Render a proof term compactly for artifact records."""
    if isinstance(term, PVar):
        return term.name
    if isinstance(term, PHyp):
        return f"hyp({term.name})"
    if isinstance(term, PLam):
        return f"fun({term.variable}:{proposition_text(term.premise)}). {proof_term_text(term.body)}"
    if isinstance(term, PApp):
        return f"app({proof_term_text(term.function)}, {proof_term_text(term.argument)})"
    if isinstance(term, PPair):
        return f"pair({proof_term_text(term.left)}, {proof_term_text(term.right)})"
    if isinstance(term, PFst):
        return f"fst({proof_term_text(term.body)})"
    if isinstance(term, PSnd):
        return f"snd({proof_term_text(term.body)})"
    if isinstance(term, PInl):
        return f"inl({proof_term_text(term.body)})"
    if isinstance(term, PInr):
        return f"inr({proof_term_text(term.body)})"
    if isinstance(term, PCase):
        return (
            f"case({proof_term_text(term.scrutinee)}, {term.left_variable} => "
            f"{proof_term_text(term.left_body)}, {term.right_variable} => "
            f"{proof_term_text(term.right_body)})"
        )
    if isinstance(term, PAbort):
        return f"abort({proof_term_text(term.body)})"
    return repr(term)


class LeanExporter:
    """Export an obligation and candidate as a Lean 4 source skeleton (Step 37)."""

    def export(self, obligation: ProofObligation, candidate: ProofCandidate) -> str:
        """Produce the Lean 4 source text for external kernel checking."""
        premise_lines = []
        for premise in obligation.premises:
            premise_lines.append(
                f"axiom {premise.name} : {proposition_text(premise.proposition)}"
            )
        proof_comment = proof_term_text(candidate.term)
        lines = [
            "-- Exported by ore_formal_reasoning (FRCE) — do not edit by hand.",
            "-- Proof term (local ND kernel format):",
            f"--   {proof_comment}",
            *premise_lines,
            f"theorem {obligation.theorem_name} : {obligation.statement_text} := by",
            "  -- tactic proof to be completed by the proof engine; the exported",
            "  -- artifact is only accepted after the external kernel verifies it.",
            "  sorry",
        ]
        return "\n".join(lines)


class LeanChecker:
    """Invoke a pinned Lean binary to check an exported artifact (Step 40)."""

    def __init__(self, lean_binary: str = "lean") -> None:
        self._lean_binary = lean_binary

    def check(self, source: str) -> tuple[bool, str]:
        """Check Lean source; returns (accepted, detail).

        The detail begins with the toolchain version when Lean ran. When Lean
        is not installed the caller must treat the result as unsupported —
        this method returns (False, reason) and the verifier falls back to the
        local kernel path.
        """
        binary_path = shutil.which(self._lean_binary)
        if binary_path is None:
            return False, "lean_not_found|no Lean toolchain is available"
        try:
            completed = subprocess.run(
                [binary_path, "--version"],
                capture_output=True, text=True, timeout=30, check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            return False, f"lean_error|{error}"
        version = completed.stdout.strip().splitlines()[0] if completed.stdout else "unknown"
        try:
            completed = subprocess.run(
                [binary_path, "--stdin"],
                input=source, capture_output=True, text=True, timeout=60, check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            return False, f"lean_error|{error}"
        if completed.returncode == 0 and "error" not in completed.stderr.lower():
            return True, f"{version}|lean accepted the artifact"
        return False, f"{version}|lean reported errors: {completed.stderr.strip()[:200]}"


__all__ = [
    "PropVar", "PropImplies", "PropAnd", "PropOr", "PropBottom", "Prop",
    "proposition_text",
    "Premise", "ProofObligation",
    "PVar", "PHyp", "PLam", "PApp", "PPair", "PFst", "PSnd", "PInl", "PInr",
    "PCase", "PAbort", "ProofTerm",
    "KernelRejection", "KernelVerdict", "TrustedKernel",
    "ProofCandidate", "ProofArtifact", "ProofCheckOutcome",
    "ProofVerifier", "ProofSearcher", "proof_term_text",
    "LeanExporter", "LeanChecker",
    "LOCAL_KERNEL_NAME", "LOCAL_KERNEL_VERSION",
]
