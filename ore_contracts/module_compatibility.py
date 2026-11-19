"""Deterministic producer/consumer compatibility checks and a validated directed connectome."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from typing import Any, Callable

from .module_artifacts import ContractArtifact
from .module_registry import ModuleRegistry


class CompatibilityStatus(str, Enum):
    """Name the specific outcome of a producer-to-consumer interface check."""

    DIRECTLY_COMPATIBLE = "DIRECTLY_COMPATIBLE"
    COMPATIBLE_WITH_ADAPTER = "COMPATIBLE_WITH_ADAPTER"
    INCOMPATIBLE_CONTRACT = "INCOMPATIBLE_CONTRACT"
    INCOMPATIBLE_VERSION = "INCOMPATIBLE_VERSION"
    MISSING_REQUIRED_INPUT = "MISSING_REQUIRED_INPUT"
    MODULE_NOT_AVAILABLE = "MODULE_NOT_AVAILABLE"


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    """Preserve the exact ports, contract rule, selected route, and reason for a decision."""

    status: CompatibilityStatus
    producer_module_id: str
    consumer_module_id: str
    output_port_id: str | None
    input_port_id: str | None
    contract_id: str | None
    producer_version: str | None
    accepted_versions: tuple[str, ...] = ()
    adapter_required: bool = False
    reason: str = ""
    structural_validation: bool = False

    @property
    def compatible(self) -> bool:
        """Return True only for a directly compatible or explicitly adapted interface."""
        return self.status in {
            CompatibilityStatus.DIRECTLY_COMPATIBLE,
            CompatibilityStatus.COMPATIBLE_WITH_ADAPTER,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report with the status value rather than an enum object."""
        data = asdict(self)
        data["status"] = self.status.value
        data["accepted_versions"] = list(self.accepted_versions)
        data["compatible"] = self.compatible
        return data


class CompatibilityValidator:
    """Validate typed interfaces using contract IDs, exact versions, payload types, and explicit converters."""

    def __init__(self, registry: ModuleRegistry) -> None:
        """Connect the validator to the manifest registry and its contract catalog."""
        self.registry = registry
        self._converters: dict[tuple[str, str, str], Callable[[ContractArtifact], ContractArtifact]] = {}

    def register_converter(
        self,
        contract_id: str,
        source_version: str,
        target_version: str,
        converter: Callable[[ContractArtifact], ContractArtifact],
    ) -> None:
        """Register one explicit version converter; an implicit or guessed conversion is never allowed."""
        from .module_manifest import validate_semantic_version
        validate_semantic_version(source_version, "source_version")
        validate_semantic_version(target_version, "target_version")
        if not contract_id.strip() or not callable(converter):
            raise ValueError("converter registration requires a contract ID and callable")
        key = (contract_id, source_version, target_version)
        if key in self._converters:
            raise ValueError(f"converter is already registered: {key}")
        self._converters[key] = converter

    @staticmethod
    def _same_data_type(output_type: str, input_type: str) -> bool:
        """Accept exact Python type paths or a deliberately generic object/Any consumer."""
        return output_type == input_type or input_type in {"object", "Any", "typing.Any"}

    def validate_connection(
        self,
        producer_module_id: str,
        consumer_module_id: str,
        output_port_id: str | None = None,
        input_port_id: str | None = None,
        *,
        artifact: ContractArtifact | None = None,
    ) -> CompatibilityReport:
        """Check one directed connection and return an explicit reason for acceptance or rejection."""
        producer = self.registry.get_manifest(producer_module_id)
        consumer = self.registry.get_manifest(consumer_module_id)
        if producer is None or consumer is None:
            return CompatibilityReport(
                CompatibilityStatus.MODULE_NOT_AVAILABLE, producer_module_id, consumer_module_id,
                output_port_id, input_port_id, None, None, reason="producer or consumer manifest is not registered",
            )
        producer_state = self.registry.diagnostics(producer_module_id)
        consumer_state = self.registry.diagnostics(consumer_module_id)
        if not producer_state["runnable"] or not consumer_state["runnable"]:
            return CompatibilityReport(
                CompatibilityStatus.MODULE_NOT_AVAILABLE, producer_module_id, consumer_module_id,
                output_port_id, input_port_id, None, None,
                reason="both producer and consumer require validated, enabled runtime backends",
            )
        outputs = [
            item for item in producer.output_ports
            if output_port_id is None or item.port_id == output_port_id
        ]
        inputs = [
            item for item in consumer.input_ports
            if input_port_id is None or item.port_id == input_port_id
        ]
        if not outputs or not inputs:
            return CompatibilityReport(
                CompatibilityStatus.MISSING_REQUIRED_INPUT, producer_module_id, consumer_module_id,
                output_port_id, input_port_id, None, None,
                reason="requested output or consumer input port does not exist",
            )
        candidate_pairs = [
            (out_port, in_port) for out_port in outputs for in_port in inputs
            if out_port.contract_id == in_port.contract_id
        ]
        if not candidate_pairs:
            return CompatibilityReport(
                CompatibilityStatus.INCOMPATIBLE_CONTRACT, producer_module_id, consumer_module_id,
                outputs[0].port_id, inputs[0].port_id, outputs[0].contract_id,
                outputs[0].contract_version, inputs[0].accepted_versions,
                reason="contract identifiers differ; matching dictionary fields do not make contracts equivalent",
            )
        # A caller-specified port pair is unambiguous; automatic matching must find exactly one pair.
        if len(candidate_pairs) != 1:
            return CompatibilityReport(
                CompatibilityStatus.MISSING_REQUIRED_INPUT, producer_module_id, consumer_module_id,
                output_port_id, input_port_id, None, None,
                reason="port selection is ambiguous; specify one output_port_id and input_port_id",
            )
        output, input_port = candidate_pairs[0]
        exact_version = output.contract_version in input_port.accepted_versions
        converter_key: tuple[str, str, str] | None = None
        if exact_version:
            target_version = output.contract_version
        else:
            targets = [
                version for version in input_port.accepted_versions
                if (output.contract_id, output.contract_version, version) in self._converters
            ]
            target_version = targets[0] if targets else None
            if target_version is not None:
                converter_key = (output.contract_id, output.contract_version, target_version)
        if not exact_version and converter_key is None:
            return CompatibilityReport(
                CompatibilityStatus.INCOMPATIBLE_VERSION, producer_module_id, consumer_module_id,
                output.port_id, input_port.port_id, output.contract_id, output.contract_version,
                input_port.accepted_versions,
                reason="producer version is not accepted and no explicit version converter is registered",
            )
        type_match = self._same_data_type(output.output_data_type, input_port.expected_data_type)
        if not type_match and converter_key is None:
            same_version_adapter = (output.contract_id, output.contract_version, output.contract_version)
            if same_version_adapter in self._converters:
                converter_key = same_version_adapter
            else:
                return CompatibilityReport(
                    CompatibilityStatus.INCOMPATIBLE_CONTRACT, producer_module_id, consumer_module_id,
                    output.port_id, input_port.port_id, output.contract_id, output.contract_version,
                    input_port.accepted_versions,
                    reason="output and input payload types differ and no explicit adapter is registered",
                )
        producer_definition = self.registry.contract_catalog.get_contract(
            output.contract_id, output.contract_version
        )
        consumer_definition = self.registry.contract_catalog.get_contract(
            input_port.contract_id, target_version
        )
        if producer_definition is None or consumer_definition is None:
            return CompatibilityReport(
                CompatibilityStatus.INCOMPATIBLE_CONTRACT, producer_module_id, consumer_module_id,
                output.port_id, input_port.port_id, output.contract_id, output.contract_version,
                input_port.accepted_versions,
                reason="one or both exact contract versions are absent from the catalog",
            )
        structural_ok = producer_definition.model_type is not None and consumer_definition.model_type is not None
        if not structural_ok:
            return CompatibilityReport(
                CompatibilityStatus.INCOMPATIBLE_CONTRACT, producer_module_id, consumer_module_id,
                output.port_id, input_port.port_id, output.contract_id, output.contract_version,
                input_port.accepted_versions,
                reason="both contract versions must declare Python model types for structural verification",
            )
        if artifact is not None:
            source_validation = self.registry.contract_catalog.validate_artifact(artifact)
            if not source_validation.valid:
                return CompatibilityReport(
                    CompatibilityStatus.INCOMPATIBLE_CONTRACT, producer_module_id, consumer_module_id,
                    output.port_id, input_port.port_id, output.contract_id, output.contract_version,
                    input_port.accepted_versions,
                    reason="producer artifact failed catalog validation: " + "; ".join(source_validation.errors),
                )
            if converter_key is not None:
                converted = self._converters[converter_key](artifact)
                if not isinstance(converted, ContractArtifact):
                    raise TypeError("registered converter must return ContractArtifact")
                converted_validation = self.registry.contract_catalog.validate_artifact(converted)
                if not converted_validation.valid or converted.envelope.schema_version != target_version:
                    return CompatibilityReport(
                        CompatibilityStatus.INCOMPATIBLE_CONTRACT, producer_module_id, consumer_module_id,
                        output.port_id, input_port.port_id, output.contract_id, output.contract_version,
                        input_port.accepted_versions,
                        adapter_required=True,
                        reason="adapter output failed target contract validation",
                    )
                if not isinstance(converted.payload, consumer_definition.model_type):
                    return CompatibilityReport(
                        CompatibilityStatus.INCOMPATIBLE_CONTRACT, producer_module_id, consumer_module_id,
                        output.port_id, input_port.port_id, output.contract_id, output.contract_version,
                        input_port.accepted_versions,
                        adapter_required=True,
                        reason="adapter output has the wrong target Python model type",
                    )
            elif not isinstance(artifact.payload, consumer_definition.model_type):
                return CompatibilityReport(
                    CompatibilityStatus.INCOMPATIBLE_CONTRACT, producer_module_id, consumer_module_id,
                    output.port_id, input_port.port_id, output.contract_id, output.contract_version,
                    input_port.accepted_versions,
                    reason="artifact payload does not satisfy the consumer model type",
                )
        status = CompatibilityStatus.COMPATIBLE_WITH_ADAPTER if converter_key else CompatibilityStatus.DIRECTLY_COMPATIBLE
        return CompatibilityReport(
            status, producer_module_id, consumer_module_id, output.port_id, input_port.port_id,
            output.contract_id, output.contract_version, input_port.accepted_versions,
            adapter_required=converter_key is not None,
            reason=(
                "an explicitly registered converter provides the interface" if converter_key
                else "contract ID, exact version, model type, and declared ports are compatible"
            ),
            structural_validation=structural_ok and (artifact is None or True),
        )

    def build_compatibility_map(self) -> dict[str, Any]:
        """Construct a directed graph containing only currently runnable compatible interfaces."""
        manifests = self.registry.list_manifests()
        nodes = [
            {
                "module_id": item.identity.module_id, "module_version": item.identity.module_version,
                "capabilities": list(item.capabilities),
                "runnable": self.registry.diagnostics(item.identity.module_id)["runnable"],
            }
            for item in manifests
        ]
        edges: list[dict[str, Any]] = []
        declared_pairs = 0
        for producer in manifests:
            for consumer in manifests:
                if producer.identity.module_id == consumer.identity.module_id:
                    continue
                for output in producer.output_ports:
                    for input_port in consumer.input_ports:
                        if output.contract_id != input_port.contract_id:
                            continue
                        declared_pairs += 1
                        report = self.validate_connection(
                            producer.identity.module_id, consumer.identity.module_id,
                            output.port_id, input_port.port_id,
                        )
                        if report.compatible:
                            edges.append(report.to_dict())
        return {
            "schema_version": "1.0.0",
            "nodes": nodes,
            "edges": edges,
            "declared_connections": declared_pairs,
            "validated_connections": len(edges),
            "interface_validation_coverage": len(edges) / declared_pairs if declared_pairs else 1.0,
        }

    def export_json(self, *, indent: int = 2) -> str:
        """Serialize the directed compatibility map for later visualization or inspection."""
        return json.dumps(self.build_compatibility_map(), sort_keys=True, indent=indent)

    def dependency_completeness(self, module_id: str) -> float:
        """Calculate available required dependencies divided by total required dependencies."""
        return float(self.registry.diagnostics(module_id)["dependency_completeness"])
