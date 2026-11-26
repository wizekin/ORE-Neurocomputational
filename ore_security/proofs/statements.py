"""Proof statement registry with version control (steps 26-27, 31).

Statements are narrow, versioned relations with explicit public-input schemas,
assumptions and supported verifier versions. Creating a statement validates the
public inputs against the schema and rejects unsupported relation versions, so a
proof for an obsolete circuit cannot be presented without explicit support.
"""
from __future__ import annotations

from typing import Any, Mapping

from ..contracts import ProofStatement
from ..exceptions import StatementVersionError


class RelationDefinition:
    """One versioned relation R(x, w) = 1 with its public-input schema."""

    def __init__(self, relation_id: str, version: str, description: str,
                 public_input_schema: Mapping[str, str],
                 assumptions: str, motivation: str,
                 *, supported: bool = True) -> None:
        if not relation_id.strip() or not version.strip():
            raise ValueError("relation_id and version must be non-empty")
        self.relation_id = relation_id
        self.version = version
        self.description = description
        self.public_input_schema = dict(public_input_schema)
        self.assumptions = assumptions
        self.motivation = motivation
        self.supported = bool(supported)


def _check_schema(schema: Mapping[str, str], public_inputs: Mapping[str, Any]) -> None:
    for field, kind in schema.items():
        if field not in public_inputs:
            raise ValueError(f"public input {field!r} is required by the statement schema")
        value = public_inputs[field]
        if kind == "int" and not isinstance(value, int):
            raise ValueError(f"public input {field!r} must be an integer")
        if kind == "hex" and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"public input {field!r} must be a hex string")
        if kind == "str" and not isinstance(value, str):
            raise ValueError(f"public input {field!r} must be a string")


class ProofStatementRegistry:
    """Registry of proof statements with circuit/version upgrade control."""

    def __init__(self) -> None:
        self._relations: dict[tuple[str, str], RelationDefinition] = {}

    def register_relation(self, definition: RelationDefinition, *,
                          replace: bool = False) -> None:
        """Register one relation version; replacement must be explicit."""
        key = (definition.relation_id, definition.version)
        if key in self._relations and not replace:
            raise ValueError(f"relation already registered: {key[0]}@{key[1]}")
        self._relations[key] = definition

    def get_relation(self, relation_id: str, version: str) -> RelationDefinition | None:
        return self._relations.get((relation_id, version))

    def supported_versions(self, relation_id: str) -> list[str]:
        return sorted(version for (rid, version), definition in self._relations.items()
                      if rid == relation_id and definition.supported)

    def build_statement(self, statement_id: str, relation_id: str, relation_version: str,
                        public_inputs: Mapping[str, Any], *, task_id: str, task_nonce: str,
                        expires_at: float,
                        verifier_context: str = "ore-security-default-verifier") -> ProofStatement:
        """Validate and instantiate a proof statement bound to a task context."""
        definition = self._relations.get((relation_id, relation_version))
        if definition is None:
            raise StatementVersionError(
                f"unknown relation version: {relation_id}@{relation_version}"
            )
        if not definition.supported:
            raise StatementVersionError(
                f"relation version {relation_id}@{relation_version} is retired; "
                "historical verification requires explicit support"
            )
        _check_schema(definition.public_input_schema, public_inputs)
        return ProofStatement(
            statement_id=statement_id,
            relation_id=relation_id,
            relation_version=relation_version,
            public_inputs=dict(public_inputs),
            assumptions=definition.assumptions,
            task_id=task_id,
            task_nonce=task_nonce,
            expires_at=float(expires_at),
            verifier_context=verifier_context,
        )
