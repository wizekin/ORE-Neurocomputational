"""Explicit adapters that expose legacy ORE processors through the universal protocol."""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from . import RunContext
from .contracts_registry import ContractCatalog
from .module_artifacts import ContractArtifact
from .module_manifest import ModuleManifest
from .module_protocol import ModuleResult, ModuleStatus


class LegacyProcessorAdapter:
    """Wrap an existing processor with caller-supplied request and result transformations."""

    def __init__(
        self,
        manifest: ModuleManifest,
        processor: Any,
        request_factory: Callable[[Mapping[str, Any], RunContext], Any],
        result_to_artifacts: Callable[[Any, RunContext], Sequence[ContractArtifact] | ContractArtifact],
        *,
        contract_catalog: ContractCatalog | None = None,
    ) -> None:
        """Configure a read-only bridge without requiring inheritance or modifying the wrapped processor."""
        if not isinstance(manifest, ModuleManifest):
            raise TypeError("manifest must be a ModuleManifest")
        if not callable(request_factory) or not callable(result_to_artifacts):
            raise TypeError("request_factory and result_to_artifacts must be callable")
        if not any(callable(getattr(processor, name, None)) for name in ("process", "execute", "run")):
            raise TypeError("legacy processor must expose process(), execute(), or run()")
        self.manifest = manifest
        self.processor = processor
        self.request_factory = request_factory
        self.result_to_artifacts = result_to_artifacts
        self.contract_catalog = contract_catalog

    def describe(self) -> ModuleManifest:
        """Return the manifest declared for this processor adapter."""
        return self.manifest

    def _validate_inputs(self, inputs: Mapping[str, Any]) -> list[str]:
        """Check required ports, multiplicity, artifact envelopes, and registered contract rules."""
        errors: list[str] = []
        declared = {item.port_id: item for item in self.manifest.input_ports}
        for port_id, port in declared.items():
            supplied = inputs.get(port_id)
            if supplied is None:
                if port.required:
                    errors.append(f"required input port is missing: {port_id}")
                continue
            values = supplied if isinstance(supplied, (tuple, list)) else (supplied,)
            if not port.multiple and len(values) != 1:
                errors.append(f"input port {port_id} accepts exactly one artifact")
            for artifact in values:
                if not isinstance(artifact, ContractArtifact):
                    errors.append(f"input port {port_id} requires ContractArtifact values")
                    continue
                if artifact.envelope.contract_id != port.contract_id:
                    errors.append(f"input port {port_id} received contract {artifact.envelope.contract_id}")
                    continue
                if artifact.envelope.schema_version not in port.accepted_versions:
                    errors.append(f"input port {port_id} rejects version {artifact.envelope.schema_version}")
                    continue
                if self.contract_catalog is not None:
                    validation = self.contract_catalog.validate_artifact(artifact)
                    if not validation.valid:
                        errors.extend(f"{port_id}: {item}" for item in validation.errors)
        unknown = set(inputs) - set(declared)
        if unknown:
            errors.append("unknown input ports: " + ", ".join(sorted(unknown)))
        return errors

    def execute(self, inputs: Mapping[str, Any], context: RunContext) -> ModuleResult:
        """Translate standardized artifacts to the legacy request, execute once, and wrap its result."""
        if not isinstance(context, RunContext):
            raise TypeError("context must be a RunContext")
        errors = self._validate_inputs(inputs)
        if errors:
            return ModuleResult(
                self.manifest.identity.module_id, self.manifest.identity.module_version,
                ModuleStatus.UNAVAILABLE, diagnostics={"input_validation_errors": errors},
                warnings=("Legacy processor was not called because its input contract was not satisfied.",),
                provenance={"adapter": type(self).__name__},
            )
        try:
            request = self.request_factory(inputs, context)
            process = getattr(self.processor, "process", None)
            execute = getattr(self.processor, "execute", None)
            run = getattr(self.processor, "run", None)
            if callable(process):
                raw_result = process(request)
            elif callable(execute):
                raw_result = execute(request, context)
            else:
                raw_result = run(request, context)
            produced = self.result_to_artifacts(raw_result, context)
            artifacts = (produced,) if isinstance(produced, ContractArtifact) else tuple(produced)
            if any(not isinstance(item, ContractArtifact) for item in artifacts):
                raise TypeError("result_to_artifacts must return ContractArtifact values")
            if self.contract_catalog is not None:
                invalid = [
                    error for item in artifacts
                    for error in self.contract_catalog.validate_artifact(item).errors
                ]
                if invalid:
                    raise ValueError("legacy result did not satisfy its output contracts: " + "; ".join(invalid))
            return ModuleResult(
                self.manifest.identity.module_id, self.manifest.identity.module_version,
                ModuleStatus.SUCCESS if artifacts else ModuleStatus.PARTIAL,
                artifacts=artifacts,
                diagnostics={"output_count": len(artifacts), "input_validation_errors": []},
                provenance={
                    "adapter": type(self).__name__,
                    "legacy_processor": f"{type(self.processor).__module__}.{type(self.processor).__qualname__}",
                },
            )
        except Exception as exc:
            return ModuleResult(
                self.manifest.identity.module_id, self.manifest.identity.module_version,
                ModuleStatus.FAILED,
                diagnostics={"exception_type": type(exc).__name__, "error": str(exc)},
                warnings=("The wrapped processor raised an exception; no successful artifact was emitted.",),
                provenance={"adapter": type(self).__name__},
            )


class CallableModuleAdapter:
    """Adapt a plain callable into an ORE module when its request/output contract is explicit."""

    def __init__(
        self,
        manifest: ModuleManifest,
        callable_backend: Callable[[Mapping[str, Any], RunContext], Sequence[ContractArtifact]],
    ) -> None:
        """Store a declared manifest and a callable that returns only typed artifacts."""
        if not callable(callable_backend):
            raise TypeError("callable_backend must be callable")
        self.manifest = manifest
        self.callable_backend = callable_backend

    def describe(self) -> ModuleManifest:
        """Return the adapter's declared communication manifest."""
        return self.manifest

    def execute(self, inputs: Mapping[str, Any], context: RunContext) -> ModuleResult:
        """Invoke the callable and preserve its output artifacts and failure status."""
        try:
            artifacts = tuple(self.callable_backend(inputs, context))
            if any(not isinstance(item, ContractArtifact) for item in artifacts):
                raise TypeError("callable backend must return ContractArtifact values")
            return ModuleResult(
                self.manifest.identity.module_id, self.manifest.identity.module_version,
                ModuleStatus.SUCCESS if artifacts else ModuleStatus.PARTIAL,
                artifacts=artifacts, provenance={"adapter": type(self).__name__},
            )
        except Exception as exc:
            return ModuleResult(
                self.manifest.identity.module_id, self.manifest.identity.module_version,
                ModuleStatus.FAILED, diagnostics={"exception_type": type(exc).__name__, "error": str(exc)},
                provenance={"adapter": type(self).__name__},
            )




class CanonicalControllerAdapter:
    """Expose an existing CanonicalOREController and its CanonicalBackendRegistry as one module."""

    def __init__(
        self,
        manifest: ModuleManifest,
        controller: Any,
        input_port_id: str,
        result_to_artifacts: Callable[[Any, RunContext], Sequence[ContractArtifact] | ContractArtifact],
        *,
        contract_catalog: ContractCatalog | None = None,
        validation_spec_factory: Callable[[Mapping[str, Any], RunContext], Any] | None = None,
    ) -> None:
        """Wrap the canonical controller while retaining its existing B-N backend registry."""
        if not isinstance(manifest, ModuleManifest):
            raise TypeError("manifest must be a ModuleManifest")
        if not callable(getattr(controller, "run", None)):
            raise TypeError("controller must expose run(context, payload, validation_spec=...)")
        if input_port_id not in {item.port_id for item in manifest.input_ports}:
            raise ValueError("input_port_id must identify a manifest input port")
        if not callable(result_to_artifacts):
            raise TypeError("result_to_artifacts must be callable")
        self.manifest = manifest
        self.controller = controller
        self.input_port_id = input_port_id
        self.result_to_artifacts = result_to_artifacts
        self.contract_catalog = contract_catalog
        self.validation_spec_factory = validation_spec_factory

    def describe(self) -> ModuleManifest:
        """Return the manifest that describes the wrapped canonical controller."""
        return self.manifest

    def execute(self, inputs: Mapping[str, Any], context: RunContext) -> ModuleResult:
        """Pass a validated artifact payload to CanonicalOREController and wrap its run report."""
        if not isinstance(context, RunContext):
            raise TypeError("context must be a RunContext")
        port = next(item for item in self.manifest.input_ports if item.port_id == self.input_port_id)
        artifact = inputs.get(self.input_port_id)
        if artifact is None:
            status = ModuleStatus.UNAVAILABLE if port.required else ModuleStatus.PARTIAL
            return ModuleResult(
                self.manifest.identity.module_id, self.manifest.identity.module_version, status,
                diagnostics={"missing_input_port": self.input_port_id},
                warnings=("Canonical controller was not called because its input artifact is absent.",),
            )
        if not isinstance(artifact, ContractArtifact):
            return ModuleResult(
                self.manifest.identity.module_id, self.manifest.identity.module_version,
                ModuleStatus.UNAVAILABLE, diagnostics={"input_error": "expected ContractArtifact"},
            )
        if artifact.envelope.contract_id != port.contract_id or artifact.envelope.schema_version not in port.accepted_versions:
            return ModuleResult(
                self.manifest.identity.module_id, self.manifest.identity.module_version,
                ModuleStatus.UNAVAILABLE, diagnostics={"input_error": "input contract ID/version mismatch"},
            )
        if self.contract_catalog is not None:
            validation = self.contract_catalog.validate_artifact(artifact)
            if not validation.valid:
                return ModuleResult(
                    self.manifest.identity.module_id, self.manifest.identity.module_version,
                    ModuleStatus.UNAVAILABLE, diagnostics={"input_validation_errors": list(validation.errors)},
                )
        try:
            validation_spec = (
                self.validation_spec_factory(inputs, context)
                if self.validation_spec_factory is not None else None
            )
            raw_result = self.controller.run(
                context, artifact.payload, validation_spec=validation_spec
            )
            produced = self.result_to_artifacts(raw_result, context)
            artifacts = (produced,) if isinstance(produced, ContractArtifact) else tuple(produced)
            if any(not isinstance(item, ContractArtifact) for item in artifacts):
                raise TypeError("result_to_artifacts must return ContractArtifact values")
            if self.contract_catalog is not None:
                invalid = [
                    error for item in artifacts
                    for error in self.contract_catalog.validate_artifact(item).errors
                ]
                if invalid:
                    raise ValueError("canonical controller output failed catalog validation: " + "; ".join(invalid))
            channel_status = dict(getattr(raw_result, "channel_status", {}) or {})
            if any(value == "FAILED" for value in channel_status.values()):
                status = ModuleStatus.FAILED
            elif not artifacts or any(
                "UNAVAILABLE" in str(value) or "BLOCKED" in str(value)
                for value in channel_status.values()
            ):
                status = ModuleStatus.PARTIAL
            else:
                status = ModuleStatus.SUCCESS
            return ModuleResult(
                self.manifest.identity.module_id, self.manifest.identity.module_version,
                status, artifacts=artifacts,
                diagnostics={"channel_status": channel_status, "output_count": len(artifacts)},
                provenance={
                    "adapter": type(self).__name__,
                    "controller": f"{type(self.controller).__module__}.{type(self.controller).__qualname__}",
                },
            )
        except Exception as exc:
            return ModuleResult(
                self.manifest.identity.module_id, self.manifest.identity.module_version,
                ModuleStatus.FAILED,
                diagnostics={"exception_type": type(exc).__name__, "error": str(exc)},
                provenance={"adapter": type(self).__name__},
            )


def adapt_legacy_processor(
    manifest: ModuleManifest,
    processor: Any,
    request_factory: Callable[[Mapping[str, Any], RunContext], Any],
    result_to_artifacts: Callable[[Any, RunContext], Sequence[ContractArtifact] | ContractArtifact],
    *,
    contract_catalog: ContractCatalog | None = None,
) -> LegacyProcessorAdapter:
    """Create a generic adapter around any existing ORE processor without altering its implementation."""
    return LegacyProcessorAdapter(
        manifest, processor, request_factory, result_to_artifacts,
        contract_catalog=contract_catalog,
    )
