"""Read-only USG adapter: entity resolution, retrieval, and IR mapping (Steps 3, 4, 8).

The adapter reads typed knowledge from a USG snapshot and maps it into FRCE's
typed logical IR. Its contracts (Point 5, Section 7):

- read-only access to typed concepts, objects, relations, rules and provenance;
- reject untyped or ambiguous mappings instead of silently selecting one;
- maintain source-to-IR traceability for every produced statement;
- never assume a USG relation is logically complete unless its domain declares
  closed-world semantics.

The adapter exposes no write path to USG storage; knowledge promotion happens
only through ``ore_formal_reasoning.governance.KnowledgeWriteGovernance``.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Mapping

from .contracts import EvidenceTag, SourceRef, TemporalScope, sha256_json
from .exceptions import AmbiguousEntityError, GovernanceViolationError, InvalidFormalizationError
from .ir import (
    Atom,
    NumLit,
    BodyLiteral,
    Const,
    DomainSemantics,
    FormalIRBuilder,
    PredicateDecl,
    RuleDecl,
    Statement,
    StatementKind,
    UnitTable,
    Var,
    VarDecl,
)
from .version import FRCE_PRODUCER


@dataclass(frozen=True, slots=True)
class TypedKnowledgeRecord:
    """One typed USG record consumed by the adapter.

    ``kind`` is one of ``concept``, ``object``, ``relation``, ``rule``, or
    ``attribute``. Records without a recognized kind or without typed payload
    fields are rejected (Section 7: reject untyped mappings).
    """

    record_id: str
    kind: str
    name: str
    payload: Mapping[str, Any]
    source_id: str = "usg"
    version_id: str | None = None
    timestamp: str | None = None
    authority: str | None = None
    domain: str = "default"
    temporal_scope: TemporalScope | None = None

    def __post_init__(self) -> None:
        if not self.record_id.strip():
            raise ValueError("TypedKnowledgeRecord.record_id must be non-empty")
        if self.kind not in {"concept", "object", "relation", "rule", "attribute"}:
            raise ValueError(f"untyped USG record kind: {self.kind!r}")
        if not isinstance(self.payload, Mapping):
            raise ValueError("TypedKnowledgeRecord.payload must be a mapping")
        if not self.name.strip():
            raise ValueError("TypedKnowledgeRecord.name must be non-empty")

    def to_source_ref(self) -> SourceRef:
        """Return the provenance link for one record."""
        return SourceRef(
            source_id=self.source_id,
            record_id=self.record_id,
            version_id=self.version_id,
            timestamp=self.timestamp,
            authority=self.authority,
        )


@dataclass(frozen=True, slots=True)
class ResolvedEntity:
    """One resolved USG entity with all matching candidates preserved (Step 3)."""

    reference: str
    usg_id: str
    name: str
    kind: str
    candidates: tuple["TypedKnowledgeRecord", ...] = ()

    def __post_init__(self) -> None:
        if not self.usg_id.strip():
            raise ValueError("ResolvedEntity.usg_id must be non-empty")


@dataclass(frozen=True, slots=True)
class AmbiguousResolution:
    """An entity reference matching several USG records without a discriminator."""

    reference: str
    candidates: tuple[TypedKnowledgeRecord, ...]

    def __post_init__(self) -> None:
        if len(self.candidates) < 2:
            raise ValueError("ambiguous resolution requires at least two candidates")


@dataclass(frozen=True, slots=True)
class RetrievalRecord:
    """Recorded query parameters and source versions for one retrieval (Step 4)."""

    goal_predicates: tuple[str, ...]
    entity_ids: tuple[str, ...]
    neighborhood_hops: int
    selected_record_ids: tuple[str, ...]
    selected_rule_ids: tuple[str, ...]
    source_versions: tuple[str, ...]
    query_digest: str


class USGEntityResolver:
    """Resolve referenced concepts, objects, relations and rules to USG IDs (Step 3)."""

    def __init__(self, records: Iterable[TypedKnowledgeRecord]) -> None:
        self._records: tuple[TypedKnowledgeRecord, ...] = tuple(records)
        self._by_id: dict[str, TypedKnowledgeRecord] = {}
        self._by_name: dict[str, list[TypedKnowledgeRecord]] = {}
        for record in self._records:
            if record.record_id in self._by_id:
                raise ValueError(f"duplicate USG record id: {record.record_id!r}")
            self._by_id[record.record_id] = record
            self._by_name.setdefault(record.name, []).append(record)

    def resolve(self, reference: str, *, discriminator: str | None = None) -> ResolvedEntity:
        """Resolve one reference to exactly one USG record, preserving ambiguity otherwise.

        An explicit ``discriminator`` (an exact record ID) resolves ambiguous
        names deterministically; without one, multiple matches raise
        ``AmbiguousEntityError`` which blocks formal execution (acceptance:
        ambiguous entity resolution blocks formal execution).
        """
        if discriminator is not None:
            record = self._by_id.get(discriminator)
            if record is None:
                raise InvalidFormalizationError(
                    f"discriminator {discriminator!r} does not match any USG record"
                )
            return ResolvedEntity(
                reference=reference, usg_id=record.record_id,
                name=record.name, kind=record.kind, candidates=(record,),
            )
        exact = self._by_id.get(reference)
        if exact is not None:
            return ResolvedEntity(
                reference=reference, usg_id=exact.record_id,
                name=exact.name, kind=exact.kind, candidates=(exact,),
            )
        matches = self._by_name.get(reference, [])
        if not matches:
            raise InvalidFormalizationError(f"reference {reference!r} matches no USG record")
        if len(matches) > 1:
            raise AmbiguousEntityError(
                f"reference {reference!r} matches {len(matches)} USG records; "
                "supply a record-id discriminator before formal execution"
            )
        only = matches[0]
        return ResolvedEntity(
            reference=reference, usg_id=only.record_id,
            name=only.name, kind=only.kind, candidates=(only,),
        )

    def resolve_all(
        self, references: Iterable[str], discriminators: Mapping[str, str] | None = None,
    ) -> tuple[ResolvedEntity, ...]:
        """Resolve a batch of references; ambiguity blocks the whole batch."""
        discriminators = discriminators or {}
        resolved: list[ResolvedEntity] = []
        for reference in references:
            resolved.append(self.resolve(reference, discriminator=discriminators.get(reference)))
        return tuple(resolved)


class KnowledgeRetriever:
    """Assemble the smallest sufficient fact/rule set for a task (Step 4).

    Selection is computed from rule dependencies: every predicate that can
    contribute (directly or transitively) to the goal predicates forms the
    relevant closure; only records for selected entities inside that closure
    are mapped into the IR.
    """

    def __init__(
        self,
        records: Iterable[TypedKnowledgeRecord],
        *,
        neighborhood_hops: int = 2,
        max_records: int = 500,
    ) -> None:
        if neighborhood_hops < 0 or max_records < 1:
            raise ValueError("retrieval limits must be non-negative")
        self._records = tuple(records)
        self._neighborhood_hops = neighborhood_hops
        self._max_records = max_records

    def _rule_dependency_closure(
        self,
        rules: tuple[TypedKnowledgeRecord, ...],
        goal_predicates: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Return predicates transitively needed to derive the goal predicates."""
        needed = set(goal_predicates)
        changed = True
        while changed:
            changed = False
            for rule in rules:
                head = str(rule.payload.get("head_predicate", ""))
                if head in needed:
                    for premise in rule.payload.get("body_predicates", ()) or ():
                        premise = str(premise)
                        if premise and premise not in needed:
                            needed.add(premise)
                            changed = True
        return tuple(sorted(needed))

    def retrieve(
        self,
        goal_predicates: Iterable[str],
        entity_ids: Iterable[str],
        rules: Iterable[TypedKnowledgeRecord] = (),
    ) -> RetrievalRecord:
        """Select the relevant subset and record query parameters (Step 4).

        Entity identifiers may be record IDs or entity names; record IDs are
        resolved to their entity names before neighborhood expansion.
        """
        goals = tuple(dict.fromkeys(goal_predicates))
        rule_records = tuple(rules)
        name_of_record = {r.record_id: r.name for r in self._records}
        entity_names: list[str] = []
        for identifier in entity_ids:
            entity_names.append(name_of_record.get(identifier, identifier))
        entity_names = tuple(dict.fromkeys(entity_names))
        closure = self._rule_dependency_closure(rule_records, goals)

        adjacency: dict[str, set[str]] = {}
        for record in self._records:
            if record.kind in {"relation", "attribute"}:
                subject = str(record.payload.get("subject", ""))
                obj = str(record.payload.get("object", ""))
                adjacency.setdefault(subject, set()).add(obj)
                adjacency.setdefault(obj, set()).add(subject)
        neighborhood: set[str] = set(entity_names)
        frontier = set(entity_names)
        for _ in range(self._neighborhood_hops):
            nxt: set[str] = set()
            for node in frontier:
                nxt |= adjacency.get(node, set())
            neighborhood |= nxt - neighborhood
            frontier = nxt
            if not frontier:
                break

        selected: list[TypedKnowledgeRecord] = []
        for record in self._records:
            if record.kind in {"concept", "object", "relation", "attribute"}:
                subject = str(record.payload.get("subject", record.name))
                obj = str(record.payload.get("object", ""))
                about = {subject, obj} - {""}
                relevant_predicate = str(
                    record.payload.get("predicate", record.name)
                )
                if relevant_predicate and relevant_predicate not in closure and closure:
                    continue
                if about and not (about & neighborhood) and neighborhood:
                    continue
            selected.append(record)
            if len(selected) >= self._max_records:
                break

        selected_rules = tuple(
            rule for rule in rule_records
            if str(rule.payload.get("head_predicate", "")) in closure or not closure
        )
        record_ids = tuple(r.record_id for r in selected)
        rule_ids = tuple(r.record_id for r in selected_rules)
        versions = tuple(dict.fromkeys(
            r.version_id for r in (*selected, *selected_rules) if r.version_id
        ))
        digest = sha256_json({
            "goal_predicates": list(goals),
            "entity_ids": list(entity_names),
            "closure": list(closure),
            "neighborhood_hops": self._neighborhood_hops,
            "record_ids": list(record_ids),
            "rule_ids": list(rule_ids),
        })
        return RetrievalRecord(
            goal_predicates=goals,
            entity_ids=entity_names,
            neighborhood_hops=self._neighborhood_hops,
            selected_record_ids=record_ids,
            selected_rule_ids=rule_ids,
            source_versions=versions,
            query_digest=digest,
        )


class USGReadOnlyGuard:
    """Structural guard: the adapter surface has no USG write path (Step 45)."""

    _BLOCKED = ("write", "insert", "update", "delete", "upsert", "mutate", "promote")

    def check_method(self, name: str) -> None:
        """Raise if a method name implies mutation of authoritative USG storage."""
        lowered = name.lower()
        if any(token in lowered for token in self._BLOCKED):
            raise GovernanceViolationError(
                f"the FRCE USG adapter is read-only; attempted write method {name!r}"
            )


class USGToIRMapper:
    """Map typed USG records to IR statements and rules (Step 8).

    Objects become constants, classes and relations become predicates,
    attributes become typed statements, and ``rule`` records become named
    implications. Every produced statement carries the USG record reference,
    the record's temporal scope, and its knowledge domain.
    """

    def __init__(self, unit_table: UnitTable | None = None) -> None:
        self.unit_table = unit_table if unit_table is not None else UnitTable()
        self._guard = USGReadOnlyGuard()

    def map_records(
        self,
        records: Iterable[TypedKnowledgeRecord],
        builder: FormalIRBuilder,
        *,
        default_domain: str = "default",
        domain_semantics: DomainSemantics | None = None,
    ) -> tuple[tuple[Statement, ...], tuple[RuleDecl, ...], dict[str, str]]:
        """Map records into the builder; returns statements, rules, and traceability map."""
        statements: list[Statement] = []
        rules: list[RuleDecl] = []
        trace: dict[str, str] = {}
        counter = 0
        semantics = domain_semantics or DomainSemantics(default_domain)
        builder.declare_domain(semantics)

        for record in records:
            ref = record.to_source_ref()
            domain = record.domain or default_domain
            if record.kind == "concept":
                predicate_name = self._safe_identifier(record.payload.get(
                    "predicate", record.name
                ))
                arg_sorts = tuple(
                    str(s) for s in record.payload.get("arg_sorts", ()) or ()
                ) or ("String",)
                builder.declare_predicate(PredicateDecl(
                    predicate_name, arg_sorts,
                    interpretation=str(record.payload.get("interpretation", "")),
                ))
                counter += 1
                statement_id = f"usg:{record.record_id}#{counter}"
                statement = Statement(
                    statement_id, StatementKind.FACT,
                    Atom(predicate_name, tuple(
                        Const(str(a), self._const_sort(str(a), arg_sorts[0]))
                        for a in record.payload.get("instances", ()) or ()
                    )) if record.payload.get("instances") else
                    Atom(predicate_name, (Const(self._safe_identifier(record.name), arg_sorts[0]),)),
                    domain=domain,
                    tag=EvidenceTag.OBSERVED_FACT,
                    source_refs=(ref,),
                    temporal_scope=record.temporal_scope,
                    origin="usg",
                )
                builder.add_statement(statement)
                statements.append(statement)
                trace[statement_id] = record.record_id
            elif record.kind == "relation":
                predicate_name = self._safe_identifier(record.payload.get(
                    "predicate", record.name
                ))
                arg_sorts = tuple(
                    str(s) for s in record.payload.get("arg_sorts", ()) or ()
                ) or ("String", "String")
                builder.declare_predicate(PredicateDecl(
                    predicate_name, arg_sorts,
                    interpretation=str(record.payload.get("interpretation", "")),
                ))
                subject = str(record.payload.get("subject", ""))
                obj = str(record.payload.get("object", ""))
                if not subject or not obj:
                    raise InvalidFormalizationError(
                        f"relation record {record.record_id!r} lacks subject or object"
                    )
                counter += 1
                statement_id = f"usg:{record.record_id}#{counter}"
                statement = Statement(
                    statement_id, StatementKind.FACT,
                    Atom(predicate_name, (
                        Const(self._safe_identifier(subject), arg_sorts[0]),
                        Const(self._safe_identifier(obj), arg_sorts[-1]),
                    )),
                    domain=domain,
                    tag=EvidenceTag.OBSERVED_FACT,
                    source_refs=(ref,),
                    temporal_scope=record.temporal_scope,
                    origin="usg",
                )
                builder.add_statement(statement)
                statements.append(statement)
                trace[statement_id] = record.record_id
            elif record.kind == "attribute":
                predicate_name = self._safe_identifier(record.payload.get(
                    "predicate", record.name
                ))
                subject = str(record.payload.get("subject", ""))
                value = record.payload.get("value")
                if not subject or value is None:
                    raise InvalidFormalizationError(
                        f"attribute record {record.record_id!r} lacks subject or typed value"
                    )
                value_sort = str(record.payload.get("value_sort", "String"))
                builder.declare_predicate(PredicateDecl(predicate_name, ("String", value_sort)))
                counter += 1
                statement_id = f"usg:{record.record_id}#{counter}"
                statement = Statement(
                    statement_id, StatementKind.FACT,
                    Atom(predicate_name, (
                        Const(self._safe_identifier(subject), "String"),
                        _value_const(value, value_sort),
                    )),
                    domain=domain,
                    tag=EvidenceTag.OBSERVED_FACT,
                    source_refs=(ref,),
                    temporal_scope=record.temporal_scope,
                    origin="usg",
                )
                builder.add_statement(statement)
                statements.append(statement)
                trace[statement_id] = record.record_id
            elif record.kind == "rule":
                # Declare the rule's head and body predicates from the typed
                # payload sorts before the rule itself is registered (Step 10).
                payload = record.payload
                head_predicate = self._safe_identifier(payload.get("head_predicate", ""))
                head_sorts = tuple(
                    str(s) for s in payload.get("head_arg_sorts", ()) or ()
                ) or ("String",)
                if head_predicate not in builder.predicate_map():
                    builder.declare_predicate(PredicateDecl(head_predicate, head_sorts))
                for index, body_predicate in enumerate(payload.get("body_predicates", ()) or ()):
                    body_name = self._safe_identifier(body_predicate)
                    body_sorts = tuple(
                        str(s) for s in payload.get(f"body_{index}_arg_sorts", ()) or ()
                    ) or ("String", "String")
                    if body_name not in builder.predicate_map():
                        builder.declare_predicate(PredicateDecl(body_name, body_sorts))
                rule = self._rule_from_record(record)
                builder.add_rule(rule)
                rules.append(rule)
                trace[f"rule:{rule.rule_id}"] = record.record_id
            # 'object' records become constants implicitly through their uses;
            # they still validate as typed records.
        return tuple(statements), tuple(rules), trace

    def _rule_from_record(self, record: TypedKnowledgeRecord) -> RuleDecl:
        """Convert one typed USG rule record into a named implication (Step 12)."""
        payload = record.payload
        head_predicate = self._safe_identifier(payload.get("head_predicate", ""))
        body_predicates = [self._safe_identifier(p) for p in payload.get("body_predicates", ()) or ()]
        variables = payload.get("variables", ()) or ()
        if not head_predicate or not body_predicates:
            raise InvalidFormalizationError(
                f"rule record {record.record_id!r} lacks head or body predicates"
            )
        var_decls = tuple(VarDecl(Var(str(v), str(s))) for v, s in variables)
        arg_sorts = tuple(str(s) for s in payload.get("head_arg_sorts", ()) or ()) or ("String",)
        head_args: list[Any] = []
        for index, sort_name in enumerate(arg_sorts):
            var_name = str(payload.get("head_variables", ())[index])
            head_args.append(Var(var_name, sort_name))
        body: list[BodyLiteral] = []
        for index, predicate in enumerate(body_predicates):
            body_vars = list(payload.get(f"body_{index}_variables", ())) or list(variables[:2])
            body_args = tuple(
                Var(str(v), "String") for v in body_vars
            )
            body.append(BodyLiteral(Atom(predicate, body_args)))
        return RuleDecl(
            rule_id=f"usg-rule-{record.record_id}",
            name=record.name,
            variables=var_decls,
            head=Atom(head_predicate, tuple(head_args)),
            body=tuple(body),
            source_refs=(record.to_source_ref(),),
        )

    @staticmethod
    def _safe_identifier(raw: Any) -> str:
        """Normalize a USG name into an IR identifier; blank input is rejected."""
        text = str(raw or "").strip()
        if not text:
            raise InvalidFormalizationError("USG mapping requires a non-empty predicate name")
        cleaned = "".join(
            ch if (ch.isalnum() or ch == "_") else "_" for ch in text
        )
        if not cleaned[0].isalpha() and cleaned[0] != "_":
            cleaned = f"p_{cleaned}"
        return cleaned

    @staticmethod
    def _const_sort(value: str, declared: str) -> str:
        """Type one mapped constant; identifier strings are never numeric (Step 10)."""
        if declared in {"Int", "Real"}:
            try:
                float(value)
            except ValueError:
                raise InvalidFormalizationError(
                    f"identifier string {value!r} cannot be used as a numeric quantity"
                ) from None
        return declared


def _value_const(value: Any, value_sort: str) -> Any:
    """Map one typed attribute value into an IR term.

    Numeric attribute values become numeric literals; identifier values stay
    symbolic constants so identifier strings are never typed as quantities.
    """
    if value_sort == "String":
        return Const(str(value), "String")
    if value_sort == "Int":
        try:
            return NumLit.int_value(int(str(value)))
        except ValueError:
            raise InvalidFormalizationError(
                f"attribute value {value!r} is not an integer"
            ) from None
    if value_sort == "Real":
        try:
            return NumLit.real_value(Fraction(str(value)))
        except (ValueError, ZeroDivisionError):
            raise InvalidFormalizationError(
                f"attribute value {value!r} is not a real number"
            ) from None
    raise InvalidFormalizationError(f"unsupported attribute value sort: {value_sort!r}")


__all__ = [
    "TypedKnowledgeRecord",
    "ResolvedEntity",
    "AmbiguousResolution",
    "RetrievalRecord",
    "USGEntityResolver",
    "KnowledgeRetriever",
    "USGReadOnlyGuard",
    "USGToIRMapper",
    "FRCE_PRODUCER",
]
