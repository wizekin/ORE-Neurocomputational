"""USGGraphProjector: authoritative USG bundle to typed graph projection (Phase A, steps 1-12).

The projector is read-only: it resolves one consistent USG snapshot through the existing
``ore_usg`` import policy (lifecycle states and source rights), filters a bounded scope,
maps stable concept identities (one node per semantic identity, aliases preserved), maps
typed directed relations (reifying n-ary relations as explicit event nodes), preserves
logical scope in explicit qualifier structures, attaches provenance with the snapshot hash,
and produces a deterministic, hashable graph. Malformed or unsupported records are
quarantined, never flattened.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable, Mapping

from ..schemas.graph_schema import GraphSchema
from .integrity import IntegrityValidator
from .provenance import (
    GraphProvenance,
    QuarantineRecord,
    _source_locator,
    compute_graph_hash,
)
from .typed_graph import (
    EdgeQualifiers,
    PartitionInfo,
    RoleAssignment,
    TypedEdge,
    TypedGraphSnapshot,
    TypedNode,
)

PROJECTION_METHOD = "ore_graph_intelligence.usg_snapshot.USGGraphProjector/1.0.0"

_GRAPH_NODE_KINDS = (
    "LEMMA", "LEXICAL_SENSE", "LEXICAL_FORM", "CONCEPT", "SEMANTIC_CLASS", "PREDICATE_FRAME",
)

_QUALIFIER_FIELDS = (
    "polarity", "modality", "attribution", "condition", "temporal_scope", "uncertainty",
)


@dataclass(frozen=True, slots=True)
class GraphProjectionScope:
    """Bounded projection scope: filters, access control, and size limits (steps 3-4)."""

    node_kinds: frozenset[str] = frozenset(_GRAPH_NODE_KINDS)
    relation_types: frozenset[str] | None = None
    context_ids: frozenset[str] | None = None
    evidence_ids: frozenset[str] | None = None
    include_candidate_states: bool = False
    allowed_local_rights: frozenset[str] = frozenset({"KNOWN_PERMITTED", "INTERNAL"})
    max_nodes: int = 100_000
    max_edges: int = 250_000

    def __post_init__(self) -> None:
        if self.max_nodes < 1 or self.max_edges < 1:
            raise ValueError("max_nodes and max_edges must be positive")
        unknown = self.node_kinds - frozenset(_GRAPH_NODE_KINDS)
        if unknown:
            raise ValueError(f"unsupported node kinds in scope: {sorted(unknown)}")
        object.__setattr__(self, "node_kinds", frozenset(self.node_kinds))
        object.__setattr__(self, "allowed_local_rights", frozenset(self.allowed_local_rights))

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_kinds": sorted(self.node_kinds),
            "relation_types": sorted(self.relation_types) if self.relation_types else None,
            "context_ids": sorted(self.context_ids) if self.context_ids else None,
            "evidence_ids": sorted(self.evidence_ids) if self.evidence_ids else None,
            "include_candidate_states": self.include_candidate_states,
            "allowed_local_rights": sorted(self.allowed_local_rights),
            "max_nodes": self.max_nodes,
            "max_edges": self.max_edges,
        }


class USGGraphProjector:
    """Project one authoritative USG snapshot into a typed graph (read-only)."""

    def __init__(self, *, scope: GraphProjectionScope | None = None) -> None:
        from ore_usg.policy import USGImportPolicy

        self.scope = scope or GraphProjectionScope()
        # Honour scope-level access control through the shared frozen USG import policy.
        self.policy = USGImportPolicy(
            include_candidate_states=self.scope.include_candidate_states,
            allowed_local_rights=self.scope.allowed_local_rights,
        )
        self.integrity_validator = IntegrityValidator()

    def project(
        self,
        bundle: Mapping[str, Any],
        *,
        snapshot_id: str | None = None,
    ) -> TypedGraphSnapshot:
        """Run Phase A steps 3-12 for one validated USG bundle mapping."""
        from ore_usg.validation import require_valid_knowledge_bundle

        require_valid_knowledge_bundle(bundle)
        namespace = str(bundle["namespace"])
        bundle_id = str(bundle["bundle_id"])
        created_at = str(bundle.get("created_at", ""))
        snapshot_hash = sha256(
            __import__("json").dumps(bundle, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        sources = {item["id"]: item for item in bundle["sources"]}
        evidence = {item["id"]: item for item in bundle["evidence"]}
        records = {item["id"]: item for item in bundle["records"]}

        nodes, quarantine = self._project_nodes(
            bundle, records, sources, evidence, namespace, bundle_id, created_at, snapshot_hash,
        )
        edges, reified_nodes, edge_quarantine = self._project_relations(
            bundle, records, sources, evidence, namespace, bundle_id, created_at, snapshot_hash,
        )
        nodes = nodes + reified_nodes
        quarantine = quarantine + edge_quarantine

        schema = self._build_schema(bundle, records, nodes, edges)
        scope_dict = self.scope.as_dict()
        if snapshot_id is not None:
            scope_dict = {**scope_dict, "snapshot_id": snapshot_id}

        graph_hash = compute_graph_hash({
            "projection_method": PROJECTION_METHOD,
            "scope": scope_dict,
            "bundle": {"namespace": namespace, "bundle_id": bundle_id, "digest": snapshot_hash},
            "nodes": [
                {"node_id": node.node_id, "kind": node.node_kind, "label": node.label,
                 "aliases": sorted(node.aliases), "payload_digest": node.payload_digest}
                for node in sorted(nodes, key=lambda item: item.node_id)
            ],
            "edges": [
                {"edge_id": edge.edge_id, "relation": edge.relation_type,
                 "source": edge.source_id, "target": edge.target_id,
                 "direction": edge.direction, "qualifiers": edge.qualifiers.as_dict(),
                 "roles": [(r.role, r.node_id) for r in edge.roles],
                 "evidence_ids": sorted(edge.evidence_ids)}
                for edge in sorted(edges, key=lambda item: item.edge_id)
            ],
            "quarantine": [(q.record_id, q.reason) for q in sorted(quarantine, key=lambda q: q.record_id)],
        })

        snapshot = TypedGraphSnapshot(
            nodes=tuple(sorted(nodes, key=lambda item: item.node_id)),
            edges=tuple(sorted(edges, key=lambda item: item.edge_id)),
            schema=schema,
            namespace=namespace,
            bundle_id=bundle_id,
            bundle_digest=snapshot_hash,
            graph_hash=graph_hash,
            provenance=GraphProvenance(
                source_ids=tuple(sorted(sources)),
                extraction_method=PROJECTION_METHOD,
                snapshot_hash=snapshot_hash,
                timestamp=created_at,
                trust_label="USG_BUNDLE",
                source_reference=f"usg-graph://{namespace}/{bundle_id}",
            ),
            quarantine=tuple(sorted(quarantine, key=lambda item: (item.record_id, item.reason))),
            partition=PartitionInfo(strategy="unpartitioned", seed=0),
            scope=scope_dict,
        )
        self._validate_or_fail(snapshot)
        return snapshot

    # ------------------------------------------------------------------ nodes (steps 5, 8)

    def _project_nodes(
        self,
        bundle: Mapping[str, Any],
        records: Mapping[str, Mapping[str, Any]],
        sources: Mapping[str, Mapping[str, Any]],
        evidence: Mapping[str, Mapping[str, Any]],
        namespace: str,
        bundle_id: str,
        created_at: str,
        snapshot_hash: str,
    ) -> tuple[list[TypedNode], list[QuarantineRecord]]:
        nodes: list[TypedNode] = []
        quarantine: list[QuarantineRecord] = []
        seen_usg_ids: set[str] = set()
        for record_id in sorted(records):
            record = records[record_id]
            kind = str(record["kind"])
            if kind not in self.scope.node_kinds:
                continue
            if record_id in seen_usg_ids:
                quarantine.append(QuarantineRecord(
                    record_id, kind, "duplicate_usg_record_id",
                ))
                continue
            seen_usg_ids.add(record_id)
            state = str(record["state"])
            if not self.policy.state_enabled(state):
                quarantine.append(QuarantineRecord(
                    record_id, kind, f"lifecycle_state_not_enabled:{state}",
                ))
                continue
            usable, reason, source_ids = self.policy.source_access(record, sources, evidence)
            if not usable:
                quarantine.append(QuarantineRecord(record_id, kind, reason))
                continue
            payload = record.get("payload", {})
            if not isinstance(payload, Mapping):
                quarantine.append(QuarantineRecord(
                    record_id, kind, "unsupported_payload_form",
                ))
                continue
            node = self._node_from_record(
                record, kind, payload, namespace, bundle_id, created_at, snapshot_hash, source_ids,
            )
            nodes.append(node)
        return nodes, quarantine

    def _node_from_record(
        self,
        record: Mapping[str, Any],
        kind: str,
        payload: Mapping[str, Any],
        namespace: str,
        bundle_id: str,
        created_at: str,
        snapshot_hash: str,
        source_ids: tuple[str, ...],
    ) -> TypedNode:
        node_id = self._stable_node_id(namespace, bundle_id, record)
        label, aliases = self._label_and_aliases(kind, payload)
        payload_digest = sha256(
            __import__("json").dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        trust = self._trust_label(record, source_ids)
        return TypedNode(
            node_id=node_id,
            node_kind=kind,
            label=label,
            aliases=tuple(sorted(set(aliases))),
            usg_record_id=str(record["id"]),
            provenance=GraphProvenance(
                source_ids=tuple(sorted(source_ids)),
                extraction_method=PROJECTION_METHOD,
                snapshot_hash=snapshot_hash,
                timestamp=str(record.get("created_at", created_at)),
                trust_label=trust,
                evidence_ids=tuple(str(item) for item in record.get("evidence_ids", ())),
                source_reference=_source_locator(
                    namespace, bundle_id, "records", str(record["id"]),
                    str(record["state"]), tuple(sorted(source_ids)),
                ),
            ),
            payload_digest=payload_digest,
            lifecycle_state=str(record["state"]),
        )

    def _label_and_aliases(self, kind: str, payload: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
        """One node per semantic identity: surface variants stay aliases, never merges."""
        if kind == "LEMMA":
            return str(payload.get("lemma", "")), (str(payload.get("lemma", "")),)
        if kind == "LEXICAL_FORM":
            return str(payload.get("form", "")), (str(payload.get("form", "")),)
        if kind == "LEXICAL_SENSE":
            definition = str(payload.get("definition", ""))
            lemma_ref = str(payload.get("lemma_record_id", ""))
            label = definition[:80] if definition else lemma_ref
            return label, (lemma_ref,) if lemma_ref else ()
        if kind == "CONCEPT":
            label = str(payload.get("label", payload.get("concept_type", "")))
            aliases = [label]
            for alias in payload.get("aliases", ()):  # USG identity-declared aliases only
                aliases.append(str(alias))
            return label, tuple(aliases)
        if kind == "SEMANTIC_CLASS":
            return str(payload.get("class_key", "")), (str(payload.get("class_key", "")),)
        if kind == "PREDICATE_FRAME":
            return str(payload.get("frame_key", "")), (str(payload.get("frame_key", "")),)
        return str(payload.get("label", "")), ()

    def _trust_label(self, record: Mapping[str, Any], source_ids: tuple[str, ...]) -> str:
        if source_ids:
            return "LOCAL_IMPORT_PERMITTED"
        return str(record.get("state", "UNKNOWN"))

    # ---------------------------------------------------------------- edges (steps 6-7)

    def _project_relations(
        self,
        bundle: Mapping[str, Any],
        records: Mapping[str, Mapping[str, Any]],
        sources: Mapping[str, Mapping[str, Any]],
        evidence: Mapping[str, Mapping[str, Any]],
        namespace: str,
        bundle_id: str,
        created_at: str,
        snapshot_hash: str,
    ) -> tuple[list[TypedEdge], list[TypedNode], list[QuarantineRecord]]:
        edges: list[TypedEdge] = []
        reified: list[TypedNode] = []
        quarantine: list[QuarantineRecord] = []
        # Recompute the exact admitted node-id set so every edge endpoint is consistent
        # with the nodes that _project_nodes actually admitted (steps 6, 9).
        admitted: set[str] = set()
        for record_id in sorted(records):
            record = records[record_id]
            kind = str(record["kind"])
            if kind not in self.scope.node_kinds:
                continue
            if not self.policy.state_enabled(str(record["state"])):
                continue
            if not self.policy.source_access(record, sources, evidence)[0]:
                continue
            if not isinstance(record.get("payload", {}), Mapping):
                continue
            admitted.add(self._stable_node_id(namespace, bundle_id, record))

        relations = sorted(bundle.get("relations", ()), key=lambda row: str(row["id"]))
        for relation in relations:
            relation_id = str(relation["id"])
            subject_ref = str(relation["subject_record_id"])
            object_ref = str(relation["object_record_id"])
            subject_node = (
                self._stable_node_id(namespace, bundle_id, records[subject_ref])
                if subject_ref in records else None
            )
            object_node = (
                self._stable_node_id(namespace, bundle_id, records[object_ref])
                if object_ref in records else None
            )
            state = str(relation.get("state", "ACCEPTED"))
            if subject_node is None or object_node is None:
                quarantine.append(QuarantineRecord(
                    relation_id, "RELATION", "dangling_relation_endpoint",
                    f"subject={subject_ref}, object={object_ref}",
                ))
                continue
            if subject_node not in admitted or object_node not in admitted:
                quarantine.append(QuarantineRecord(
                    relation_id, "RELATION",
                    "both_relation_endpoints_must_be_admitted_nodes",
                ))
                continue
            if not self.policy.state_enabled(state):
                quarantine.append(QuarantineRecord(
                    relation_id, "RELATION", f"lifecycle_state_not_enabled:{state}",
                ))
                continue
            usable, reason, source_ids = self.policy.source_access(relation, sources, evidence)
            if not usable:
                quarantine.append(QuarantineRecord(relation_id, "RELATION", reason))
                continue

            predicate = str(relation["predicate"])
            if self.scope.relation_types is not None and predicate not in self.scope.relation_types:
                continue
            context_ids = tuple(str(item) for item in relation.get("context_ids", ()))
            if self.scope.context_ids is not None and not (
                set(context_ids) & set(self.scope.context_ids)
            ):
                continue
            evidence_ids = tuple(str(item) for item in relation.get("evidence_ids", ()))
            if self.scope.evidence_ids is not None and not (
                set(evidence_ids) & set(self.scope.evidence_ids)
            ):
                continue

            payload = relation.get("payload", {})
            qualifiers, unsupported = self._extract_qualifiers(payload)
            if unsupported:
                quarantine.append(QuarantineRecord(
                    relation_id, "RELATION", "unsupported_logical_form",
                    ";".join(unsupported),
                ))
                continue

            arguments = payload.get("arguments", ()) if isinstance(payload, Mapping) else ()
            if isinstance(arguments, Mapping) and len(arguments) > 2:
                event_node, event_edges = self._reify_nary_relation(
                    relation, predicate, arguments, records, admitted, namespace, bundle_id,
                    created_at, snapshot_hash, source_ids, evidence_ids, context_ids, state,
                )
                reified.append(event_node)
                edges.extend(event_edges)
                continue

            direction = str(relation.get("direction", "DIRECTED"))
            if direction not in {"DIRECTED", "UNDIRECTED"}:
                direction = "DIRECTED"
            edges.append(TypedEdge(
                edge_id=self._stable_edge_id(namespace, bundle_id, relation),
                relation_type=predicate,
                source_id=subject_node,
                target_id=object_node,
                direction=direction,
                qualifiers=qualifiers,
                evidence_ids=evidence_ids,
                context_ids=context_ids,
                provenance=GraphProvenance(
                    source_ids=tuple(sorted(source_ids)),
                    extraction_method=PROJECTION_METHOD,
                    snapshot_hash=snapshot_hash,
                    timestamp=str(relation.get("created_at", created_at)),
                    trust_label="LOCAL_IMPORT_PERMITTED",
                    evidence_ids=evidence_ids,
                    source_reference=_source_locator(
                        namespace, bundle_id, "relations", relation_id, state,
                        tuple(sorted(source_ids)),
                    ),
                ),
                lifecycle_state=state,
            ))
            if len(edges) > self.scope.max_edges:
                raise ValueError(
                    f"projection exceeds max_edges={self.scope.max_edges}; "
                    "narrow the projection scope before continuing"
                )
        return edges, reified, quarantine

    def _extract_qualifiers(self, payload: Mapping[str, Any]) -> tuple[EdgeQualifiers, list[str]]:
        """Preserve logical scope explicitly; quarantine unsupported forms (step 7)."""
        if not isinstance(payload, Mapping):
            return EdgeQualifiers(), []
        unsupported: list[str] = []
        qualifier_payload = payload.get("qualifiers", payload)
        if not isinstance(qualifier_payload, Mapping):
            return EdgeQualifiers(), ["qualifiers_not_a_mapping"]
        polarity = str(qualifier_payload.get("polarity", "positive")).lower()
        if polarity not in {"positive", "negative", "unknown"}:
            unsupported.append(f"unsupported_polarity:{polarity}")
            polarity = "unknown"
        modality = qualifier_payload.get("modality")
        attribution = qualifier_payload.get("attribution")
        condition = qualifier_payload.get("condition")
        temporal_scope = qualifier_payload.get("temporal_scope")
        uncertainty = qualifier_payload.get("uncertainty")
        if uncertainty is not None:
            try:
                uncertainty = float(uncertainty)
                if not 0.0 <= uncertainty <= 1.0:
                    unsupported.append("uncertainty_out_of_range")
                    uncertainty = None
            except (TypeError, ValueError):
                unsupported.append("uncertainty_not_numeric")
                uncertainty = None
        exceptions = tuple(str(item) for item in qualifier_payload.get("exceptions", ()))
        notes = tuple(str(item) for item in qualifier_payload.get("notes", ()))
        qualifiers = EdgeQualifiers(
            polarity=polarity,
            modality=str(modality) if modality is not None else None,
            attribution=str(attribution) if attribution is not None else None,
            condition=str(condition) if condition is not None else None,
            exceptions=exceptions,
            temporal_scope=str(temporal_scope) if temporal_scope is not None else None,
            uncertainty=uncertainty,
            source_notes=notes,
        )
        return qualifiers, unsupported

    def _reify_nary_relation(
        self,
        relation: Mapping[str, Any],
        predicate: str,
        arguments: Mapping[str, str],
        records: Mapping[str, Mapping[str, Any]],
        admitted: set[str],
        namespace: str,
        bundle_id: str,
        created_at: str,
        snapshot_hash: str,
        source_ids: tuple[str, ...],
        evidence_ids: tuple[str, ...],
        context_ids: tuple[str, ...],
        state: str,
    ) -> tuple[TypedNode, list[TypedEdge]]:
        """Reify one n-ary relation as an explicit relation/event node with role edges (step 6).

        The primary subject/agent argument anchors the event; every declared role is bound to
        its stable projected node so no argument of the n-ary relation is dropped.
        """
        relation_id = str(relation["id"])
        event_id = self._stable_event_id(namespace, bundle_id, relation_id)
        roles: list[RoleAssignment] = []
        edges: list[TypedEdge] = []
        provenance = GraphProvenance(
            source_ids=tuple(sorted(source_ids)),
            extraction_method=PROJECTION_METHOD,
            snapshot_hash=snapshot_hash,
            timestamp=created_at,
            trust_label="LOCAL_IMPORT_PERMITTED",
            evidence_ids=evidence_ids,
            source_reference=_source_locator(
                namespace, bundle_id, "relations", relation_id, state, tuple(sorted(source_ids)),
            ),
        )
        event_node = TypedNode(
            node_id=event_id,
            node_kind="RELATION_EVENT",
            label=f"{predicate}:{relation_id}",
            aliases=(),
            usg_record_id=relation_id,
            provenance=provenance,
            lifecycle_state=state,
        )
        bound_nodes: list[str] = []
        for role_name in sorted(arguments):
            target_ref = str(arguments[role_name])
            if target_ref not in records:
                continue
            target_node = self._stable_node_id(namespace, bundle_id, records[target_ref])
            if target_node not in admitted:
                continue
            roles.append(RoleAssignment(role=role_name, node_id=target_node))
            bound_nodes.append(target_node)
            edges.append(TypedEdge(
                edge_id=self._stable_role_edge_id(namespace, bundle_id, relation_id, role_name),
                relation_type=f"{predicate}.{role_name}",
                source_id=event_id,
                target_id=target_node,
                direction="DIRECTED",
                roles=(RoleAssignment(role=role_name, node_id=target_node),),
                evidence_ids=evidence_ids,
                context_ids=context_ids,
                provenance=provenance,
                lifecycle_state=state,
            ))
        primary_ref = arguments.get("SUBJECT") or arguments.get("AGENT")
        if primary_ref and primary_ref in records:
            primary_node = self._stable_node_id(namespace, bundle_id, records[primary_ref])
            if primary_node in admitted:
                edges.append(TypedEdge(
                    edge_id=self._stable_edge_id(namespace, bundle_id, relation),
                    relation_type=predicate,
                    source_id=primary_node,
                    target_id=event_id,
                    direction="DIRECTED",
                    qualifiers=EdgeQualifiers(),
                    roles=tuple(roles),
                    evidence_ids=evidence_ids,
                    context_ids=context_ids,
                    provenance=provenance,
                    lifecycle_state=state,
                ))
                return event_node, edges
        if not bound_nodes:
            return event_node, []
        edges.append(TypedEdge(
            edge_id=self._stable_edge_id(namespace, bundle_id, relation),
            relation_type=predicate,
            source_id=event_id,
            target_id=bound_nodes[0],
            direction="DIRECTED",
            qualifiers=EdgeQualifiers(),
            roles=tuple(roles),
            evidence_ids=evidence_ids,
            context_ids=context_ids,
            provenance=provenance,
            lifecycle_state=state,
        ))
        return event_node, edges

    def _build_schema(
        self,
        bundle: Mapping[str, Any],
        records: Mapping[str, Mapping[str, Any]],
        nodes: Iterable[TypedNode],
        edges: Iterable[TypedEdge],
    ) -> GraphSchema:
        """Build the graph schema from predicate frames plus observed domain/range pairs."""
        frames = []
        for record_id in sorted(records):
            record = records[record_id]
            if str(record["kind"]) != "PREDICATE_FRAME":
                continue
            payload = record.get("payload", {})
            if not isinstance(payload, Mapping):
                continue
            roles = []
            for role in payload.get("roles", ()):  # type: ignore[union-attr]
                if isinstance(role, Mapping):
                    roles.append({
                        "role": str(role.get("role", "")),
                        "description": str(role.get("description", "")),
                        "required": bool(role.get("required", False)),
                    })
            frames.append({
                "frame_id": self._stable_node_id(str(bundle["namespace"]), str(bundle["bundle_id"]), record),
                "frame_key": str(payload.get("frame_key", "")),
                "role_declarations": roles,
            })
        schema = GraphSchema.from_predicate_frames(frames)
        kind_by_id = {node.node_id: node.node_kind for node in nodes}
        observed: dict[str, tuple[set[str], set[str]]] = {}
        for edge in edges:
            domains, ranges = observed.setdefault(edge.relation_type, (set(), set()))
            domains.add(kind_by_id.get(edge.source_id, "CONCEPT"))
            ranges.add(kind_by_id.get(edge.target_id, "CONCEPT"))
        return schema.with_observed_defaults({
            relation: (frozenset(domain), frozenset(rang))
            for relation, (domain, rang) in observed.items()
        })

    # ------------------------------------------------------------------ stable ids

    def _stable_node_id(self, namespace: str, bundle_id: str, record: Mapping[str, Any]) -> str:
        from ore_usg.lexical import _ore_record_id

        return _ore_record_id(namespace, bundle_id, record)

    def _stable_edge_id(self, namespace: str, bundle_id: str, relation: Mapping[str, Any]) -> str:
        from urllib.parse import quote

        scope = sha256(f'{{"namespace":"{namespace}","bundle_id":"{bundle_id}"}}'.encode("utf-8")).hexdigest()
        return f"usg-edge:{scope}:{quote(str(relation['id']), safe='')}"

    def _stable_event_id(self, namespace: str, bundle_id: str, relation_id: str) -> str:
        from urllib.parse import quote

        scope = sha256(f'{{"namespace":"{namespace}","bundle_id":"{bundle_id}"}}'.encode("utf-8")).hexdigest()
        return f"usg-event:{scope}:{quote(relation_id, safe='')}"

    def _stable_role_edge_id(
        self, namespace: str, bundle_id: str, relation_id: str, role: str,
    ) -> str:
        from urllib.parse import quote

        scope = sha256(f'{{"namespace":"{namespace}","bundle_id":"{bundle_id}"}}'.encode("utf-8")).hexdigest()
        return f"usg-role-edge:{scope}:{quote(relation_id, safe='')}:{quote(role, safe='')}"

    def _validate_or_fail(self, snapshot: TypedGraphSnapshot) -> None:
        """Fail loudly when the projection itself is malformed (step 9)."""
        report = self.integrity_validator.validate(snapshot)
        for failure in report.failures():
            if failure.check_id in {"dangling_references", "provenance_present"}:
                raise ValueError(
                    f"projection integrity failure in {failure.check_id}: "
                    f"{failure.violations[:5]}"
                )


__all__ = ["GraphProjectionScope", "PROJECTION_METHOD", "USGGraphProjector"]
