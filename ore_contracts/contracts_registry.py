"""Versioned contract catalog that reuses existing ORE IF-05 through IF-13 models."""
from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Callable, Mapping

from . import ContractEnvelope
from .module_artifacts import ContractArtifact


@dataclass(frozen=True, slots=True)
class ContractDefinition:
    """Describe one contract ID/version, its Python model, fields, and optional validator."""

    contract_id: str
    version: str
    model_type: type | tuple[type, ...] | None
    required_fields: tuple[str, ...] = ()
    optional_fields: tuple[str, ...] = ()
    description: str = ""
    validator: Callable[[Any], bool | None] | None = None
    serialization_supported: bool = True

    def __post_init__(self) -> None:
        """Validate the stable contract identity and disallow required/optional field overlap."""
        from .module_manifest import validate_semantic_version
        if not self.contract_id.strip():
            raise ValueError("contract_id must be non-empty")
        validate_semantic_version(self.version, "contract version")
        if self.model_type is not None:
            types = self.model_type if isinstance(self.model_type, tuple) else (self.model_type,)
            if not types or any(not isinstance(item, type) for item in types):
                raise TypeError("model_type must be a type, tuple of types, or None")
        if len(set(self.required_fields)) != len(self.required_fields):
            raise ValueError("required_fields must be unique")
        if set(self.required_fields) & set(self.optional_fields):
            raise ValueError("a field cannot be both required and optional")


@dataclass(frozen=True, slots=True)
class ContractValidationResult:
    """Explain whether an artifact satisfies a registered contract without coercion."""

    valid: bool
    contract_id: str
    version: str
    errors: tuple[str, ...] = ()
    checked_payload_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain diagnostic record for logs and compatibility reports."""
        return {
            "valid": self.valid, "contract_id": self.contract_id, "version": self.version,
            "errors": list(self.errors), "checked_payload_type": self.checked_payload_type,
        }


class ContractCatalog:
    """Thread-safe, exact-version catalog for standardized information exchanged by modules."""

    def __init__(self) -> None:
        """Initialize an empty contract catalog; callers explicitly register permitted contracts."""
        self._contracts: dict[tuple[str, str], ContractDefinition] = {}
        self._lock = threading.RLock()

    def register_contract(self, definition: ContractDefinition, *, replace: bool = False) -> None:
        """Register a contract once; replacement must be explicit and preserves version identity."""
        if not isinstance(definition, ContractDefinition):
            raise TypeError("definition must be a ContractDefinition")
        key = (definition.contract_id, definition.version)
        with self._lock:
            if key in self._contracts and not replace:
                raise ValueError(f"contract already registered: {key[0]}@{key[1]}")
            self._contracts[key] = definition

    def register_model(
        self, contract_id: str, version: str, model_type: type,
        *, required_fields: tuple[str, ...] = (), optional_fields: tuple[str, ...] = (),
        description: str = "", validator: Callable[[Any], bool | None] | None = None,
    ) -> ContractDefinition:
        """Create and register a model-backed contract definition."""
        definition = ContractDefinition(
            contract_id=contract_id, version=version, model_type=model_type,
            required_fields=required_fields, optional_fields=optional_fields,
            description=description, validator=validator,
        )
        self.register_contract(definition)
        return definition

    def get_contract(self, contract_id: str, version: str) -> ContractDefinition | None:
        """Return an exact contract version or None; no implicit schema migration is attempted."""
        with self._lock:
            return self._contracts.get((contract_id, version))

    def list_contracts(self, contract_id: str | None = None) -> tuple[ContractDefinition, ...]:
        """List registered definitions in deterministic ID/version order."""
        with self._lock:
            rows = [
                item for (key, _), item in self._contracts.items()
                if contract_id is None or key == contract_id
            ]
        return tuple(sorted(rows, key=lambda item: (item.contract_id, item.version)))

    def validate_artifact(self, artifact: ContractArtifact) -> ContractValidationResult:
        """Validate envelope identity, exact version, payload model, fields, and finite numerical data."""
        errors: list[str] = []
        if not isinstance(artifact, ContractArtifact):
            raise TypeError("artifact must be a ContractArtifact")
        envelope = artifact.envelope
        definition = self.get_contract(envelope.contract_id, envelope.schema_version)
        payload = artifact.payload
        if definition is None:
            errors.append(f"unregistered contract: {envelope.contract_id}@{envelope.schema_version}")
            return ContractValidationResult(
                False, envelope.contract_id, envelope.schema_version, tuple(errors),
                f"{type(payload).__module__}.{type(payload).__qualname__}",
            )
        if not envelope.producer_id.strip() or not envelope.run_id.strip():
            errors.append("artifact provenance requires producer_id and run_id")
        if definition.model_type is not None and not isinstance(payload, definition.model_type):
            errors.append(
                f"payload type mismatch: expected {definition.model_type}, got "
                f"{type(payload).__module__}.{type(payload).__qualname__}"
            )
        for field_name in definition.required_fields:
            present = field_name in payload if isinstance(payload, Mapping) else hasattr(payload, field_name)
            if not present:
                errors.append(f"missing required payload field: {field_name}")
        try:
            from .module_artifacts import content_hash
            content_hash(payload)
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid payload values: {exc}")
        if definition.validator is not None:
            try:
                outcome = definition.validator(payload)
                if outcome is False:
                    errors.append("contract-specific validator returned false")
            except Exception as exc:
                errors.append(f"contract-specific validator failed: {type(exc).__name__}: {exc}")
        return ContractValidationResult(
            not errors, envelope.contract_id, envelope.schema_version, tuple(errors),
            f"{type(payload).__module__}.{type(payload).__qualname__}",
        )


def register_existing_contracts(catalog: ContractCatalog) -> tuple[ContractDefinition, ...]:
    """Register the existing typed IF-05/06/07/08/10/12/13 structures without recreating them."""
    from .advanced import (
        GraphemeEvidence, ORERecognitionResult, OrthographicMemorySnapshot,
        PositionCodeBundle, PredictionState, SequenceHypothesisSet, VisualFeatureBundle,
    )
    existing = (
        ("IF-05", VisualFeatureBundle, "Shared visual feature representation."),
        ("IF-06", GraphemeEvidence, "Per-region grapheme identity evidence."),
        ("IF-07", PositionCodeBundle, "Identity and positional code channels."),
        ("IF-08", OrthographicMemorySnapshot, "Orthographic transition and chunk memory."),
        ("IF-10", PredictionState, "Prediction, observation, residual, and precision record."),
        ("IF-12", SequenceHypothesisSet, "Competing sequence hypotheses and abstention."),
        ("IF-13", ORERecognitionResult, "Canonical recognition output and evidence lineage."),
    )
    return tuple(
        catalog.register_model(
            contract_id, "1.0.0", model_type,
            required_fields=("envelope",), description=description,
            validator=lambda value, expected=contract_id: value.envelope.contract_id == expected,
        )
        for contract_id, model_type, description in existing
    )
