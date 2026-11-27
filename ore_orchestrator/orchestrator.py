"""One public runtime entry point for the ORE Neurocomputational repository."""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import importlib
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4

from ore_contracts import (
    CognitiveTask, CognitiveTaskCoordinator, ContractCatalog,
    ModuleManifest, ModuleRegistry, RunContext, register_existing_contracts,
)
from .catalog import OREPackageCatalog

_DEFAULT_COMPONENTS: dict[str, tuple[str, str]] = {
    "comprehension": ("ore_comprehension", "ComprehensionProcessor"),
    "working_memory": ("ore_working_memory", "WorkingMemoryProcessor"),
    "discourse": ("ore_discourse", "DiscourseProcessor"),
    "syntax": ("ore_syntax", "SyntaxProcessor"),
    "semantic_composition": ("ore_semantic_composition", "SemanticCompositionProcessor"),
    "semantic": ("ore_semantic", "SemanticCognitionProcessor"),
    "reading": ("ore_reading", "IntegratedReadingProcessor"),
    "n1": ("ore_n1", "N1Processor"),
    "n2": ("ore_n2", "N2Processor"),
    "n3": ("ore_n3", "N3Processor"),
    "n4": ("ore_n4", "N4Processor"),
    "n5": ("ore_n5", "N5Processor"),
    "n6": ("ore_n6", "N6Processor"),
    "n9": ("ore_n9", "N9Processor"),
    "bnbea": ("ore_bnbea", "BNBEAProcessor"),
}


def _to_jsonable(value: Any) -> Any:
    """Convert processor outputs to ordinary JSON-compatible values."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _to_jsonable(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "tolist") and callable(value.tolist):
        return _to_jsonable(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@dataclass(slots=True)
class ComponentOutcome:
    """The result of one processor invocation in a multi-component run."""
    component_id: str
    package_name: str
    status: str
    elapsed_ms: float
    result: Any = None
    error: str | None = None

    @property
    def trace_count(self) -> int:
        """Count trace rows when the component returns a known trace field."""
        for name in ("operation_trace", "trace", "stage_trace"):
            rows = getattr(self.result, name, None)
            if isinstance(rows, (tuple, list)):
                return len(rows)
        return 0

    def to_dict(self, *, include_result: bool = True) -> dict[str, Any]:
        data = {
            "component_id": self.component_id,
            "package_name": self.package_name,
            "status": self.status,
            "elapsed_ms": self.elapsed_ms,
            "trace_count": self.trace_count,
            "error": self.error,
        }
        if include_result:
            data["result"] = _to_jsonable(self.result)
        return data


@dataclass(slots=True)
class OREExecutionReport:
    """An inspectable report from one user-facing ORE orchestration run."""
    run_id: str
    sequence_id: str
    goal: str
    status: str
    components: tuple[ComponentOutcome, ...]
    discovered_package_count: int
    registered_module_count: int
    diagnostics: Mapping[str, Any]

    def to_dict(self, *, include_results: bool = True) -> dict[str, Any]:
        return {
            "schema": "ore.orchestrator.run.v1",
            "run_id": self.run_id,
            "sequence_id": self.sequence_id,
            "goal": self.goal,
            "status": self.status,
            "components": [item.to_dict(include_result=include_results) for item in self.components],
            "discovered_package_count": self.discovered_package_count,
            "registered_module_count": self.registered_module_count,
            "diagnostics": _to_jsonable(dict(self.diagnostics)),
        }


class OREOrchestrator:
    """Unified package catalog, processor gateway, and contract-based task coordinator.

    Every root package is discoverable. Stable public processors have lazy
    defaults; data-dependent processors can be supplied by the caller.
    Contract-compatible backends use ore_contracts' versioned module protocol.
    """

    def __init__(
        self,
        repository_root: str | Path | None = None,
        *,
        max_workers: int = 4,
        enable_logging: bool = False,
        logger_database_path: str | Path | None = None,
        simulate_biology: bool = False,
        integrate_point_modules: bool = True,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least one")
        self.package_catalog = OREPackageCatalog(repository_root)
        self.repository_root = self.package_catalog.repository_root
        self.simulate_biology = bool(simulate_biology)
        self.logger_database_path = (
            Path(logger_database_path).expanduser().resolve()
            if logger_database_path is not None else None
        )
        self.contract_catalog = ContractCatalog()
        register_existing_contracts(self.contract_catalog)
        self.module_registry = ModuleRegistry(self.contract_catalog)
        self.task_coordinator = CognitiveTaskCoordinator(
            self.module_registry,
            self.contract_catalog,
            repository_root=self.repository_root,
            logger_database_path=self.logger_database_path,
            enable_logging=enable_logging,
            max_workers=max_workers,
        )
        self._components: dict[str, Any] = {}
        self._component_factories: dict[str, tuple[str, Callable[[], Any]]] = {}
        self._component_package_overrides: dict[str, str] = {}
        self.integration_manager = None
        self.integration_report = None
        if integrate_point_modules:
            self._integrate_point_modules()

    def _integrate_point_modules(self) -> None:
        """Reuse Point 24's native, manifest-only and status integration tiers."""
        from ore_point24 import PointIntegrationManager
        self.integration_manager = PointIntegrationManager(self.module_registry)
        self.integration_report = self.integration_manager.integrate_all()

    @staticmethod
    def _validate_component_id(component_id: str) -> None:
        if not isinstance(component_id, str) or not component_id.strip():
            raise ValueError("component_id must be a non-empty string")

    @staticmethod
    def _validate_processor_instance(component_id: str, processor: Any) -> None:
        if not callable(getattr(processor, "process", None)) and not callable(getattr(processor, "run", None)):
            raise TypeError(
                f"component {component_id!r} must expose process(request) or run(request)"
            )

    def register_component(
        self,
        component_id: str,
        processor: Any,
        *,
        package_name: str | None = None,
        replace: bool = False,
    ) -> None:
        """Attach an initialized processor from any discovered package.

        Its native typed request is passed unchanged. No input schema is
        invented for data-dependent components.
        """
        self._validate_component_id(component_id)
        self._validate_processor_instance(component_id, processor)
        if package_name is not None:
            self.package_catalog.get_package(package_name)
        if not replace and (
            component_id in self._components
            or component_id in self._component_factories
            or component_id in _DEFAULT_COMPONENTS
        ):
            raise ValueError(f"component is already registered: {component_id}")
        self._components[component_id] = processor
        self._component_factories.pop(component_id, None)
        self._component_package_overrides[component_id] = (
            package_name or type(processor).__module__.split(".", 1)[0]
        )

    def register_component_factory(
        self,
        component_id: str,
        factory: Callable[[], Any],
        *,
        package_name: str,
        replace: bool = False,
    ) -> None:
        """Register a lazy factory for a configured or data-dependent processor."""
        self._validate_component_id(component_id)
        if not callable(factory):
            raise TypeError("factory must be callable")
        self.package_catalog.get_package(package_name)
        if not replace and (
            component_id in self._components
            or component_id in self._component_factories
            or component_id in _DEFAULT_COMPONENTS
        ):
            raise ValueError(f"component is already registered: {component_id}")
        self._component_factories[component_id] = (package_name, factory)
        self._components.pop(component_id, None)
        self._component_package_overrides.pop(component_id, None)

    def _default_processor(self, component_id: str) -> Any:
        """Construct one stable public processor lazily."""
        try:
            package_name, symbol_name = _DEFAULT_COMPONENTS[component_id]
        except KeyError as exc:
            raise KeyError(
                f"no default processor is configured for {component_id!r}; "
                "register an initialized processor or factory"
            ) from exc

        module = importlib.import_module(package_name)
        processor_type = getattr(module, symbol_name)
        kwargs: dict[str, Any] = {}
        if component_id == "comprehension":
            from ore_comprehension import ComprehensionConfig
            kwargs["config"] = ComprehensionConfig(simulate_biology=self.simulate_biology)
        elif component_id == "working_memory":
            from ore_working_memory import WorkingMemoryConfig
            kwargs["config"] = WorkingMemoryConfig(simulate_biology=self.simulate_biology)
        elif component_id == "discourse":
            from ore_discourse import DiscourseConfig
            kwargs["config"] = DiscourseConfig(simulate_biology=self.simulate_biology)
        elif component_id == "bnbea":
            kwargs["repository_root"] = self.repository_root
            kwargs["database_path"] = self.logger_database_path or (
                self.repository_root / "ore_logs" / "ore_logger.sqlite3"
            )
        elif component_id == "n9":
            # Connect N9's native syntax, discourse and semantic-composition slots.
            from ore_discourse import DiscourseConfig, DiscourseN7Adapter, DiscourseProcessor
            from ore_n9 import N9Adapters
            from ore_n9.adapters import SemanticCompositionN8Adapter, SyntaxN9Adapter
            from ore_semantic_composition import SemanticCompositionProcessor
            from ore_syntax import SyntaxProcessor

            base = N9Adapters()
            adapters = N9Adapters(
                n1=base.n1, n2=base.n2, n3=base.n3, n4=base.n4,
                n5=base.n5, n6=base.n6,
                n7=DiscourseN7Adapter(
                    DiscourseProcessor(DiscourseConfig(simulate_biology=self.simulate_biology))
                ),
                n8=SemanticCompositionN8Adapter(SemanticCompositionProcessor()),
                syntax=SyntaxN9Adapter(SyntaxProcessor()),
            )
            kwargs["adapters"] = adapters

        processor = processor_type(**kwargs)
        self._validate_processor_instance(component_id, processor)
        return processor

    def get_component(self, component_id: str) -> Any:
        """Return a registered processor, constructing a known default on first use."""
        if component_id in self._components:
            return self._components[component_id]
        if component_id in self._component_factories:
            package_name, factory = self._component_factories[component_id]
            processor = factory()
            self._validate_processor_instance(component_id, processor)
            self._components[component_id] = processor
            self._component_package_overrides[component_id] = package_name
            return processor
        processor = self._default_processor(component_id)
        self._components[component_id] = processor
        self._component_package_overrides[component_id] = _DEFAULT_COMPONENTS[component_id][0]
        return processor

    def run_component(self, component_id: str, request: Any) -> Any:
        """Run one component using its native typed request and return its native result."""
        processor = self.get_component(component_id)
        method = getattr(processor, "process", None) or getattr(processor, "run", None)
        return method(request)

    def _invoke_component(self, component_id: str, request: Any) -> ComponentOutcome:
        package_name = self._component_package_overrides.get(
            component_id, _DEFAULT_COMPONENTS.get(component_id, (component_id, ""))[0]
        )
        started = time.perf_counter()
        try:
            result = self.run_component(component_id, request)
            return ComponentOutcome(
                component_id, package_name, "success",
                round((time.perf_counter() - started) * 1000.0, 3),
                result=result,
            )
        except Exception as exc:
            return ComponentOutcome(
                component_id, package_name, "failed",
                round((time.perf_counter() - started) * 1000.0, 3),
                error=f"{type(exc).__name__}: {exc}",
            )

    def _build_report(
        self,
        run_id: str,
        sequence_id: str,
        goal: str,
        outcomes: list[ComponentOutcome],
        status: str,
        diagnostics: Mapping[str, Any],
    ) -> OREExecutionReport:
        return OREExecutionReport(
            run_id=run_id,
            sequence_id=sequence_id,
            goal=goal,
            status=status,
            components=tuple(outcomes),
            discovered_package_count=len(self.package_catalog.packages),
            registered_module_count=len(self.module_registry.list_manifests()),
            diagnostics=dict(diagnostics),
        )

    def run_text(
        self,
        text: str,
        *,
        sequence_id: str | None = None,
        language: str = "en",
        lexical_categories: Mapping[str, str] | None = None,
        semantic_features: Mapping[str, Mapping[str, float]] | None = None,
        world_knowledge: Mapping[str, Mapping[str, float]] | None = None,
        discourse_context: tuple[str, ...] | list[str] = (),
        include_working_memory: bool = True,
        include_discourse: bool = True,
    ) -> OREExecutionReport:
        """Run text through comprehension, working memory, then discourse.

        Upstream interpretations, token records, predictions and working-memory
        state are forwarded as context. Each component retains its own request.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("language must be a non-empty string")
        sequence = sequence_id or f"ore_{uuid4().hex}"
        run_id = f"run_{uuid4().hex}"
        outcomes: list[ComponentOutcome] = []

        from ore_comprehension import ComprehensionRequest
        request = ComprehensionRequest(
            sequence_id=sequence,
            text=text,
            language=language,
            lexical_categories=lexical_categories or {},
            semantic_features=semantic_features or {},
            world_knowledge=world_knowledge or {},
            discourse_context=tuple(discourse_context),
        )
        first = self._invoke_component("comprehension", request)
        outcomes.append(first)
        if first.status != "success":
            return self._build_report(
                run_id, sequence, "text comprehension", outcomes, "failed",
                {"stopped_after": "comprehension"},
            )
        comprehension = first.result

        working_memory_result = None
        if include_working_memory:
            from ore_working_memory import WorkingMemoryRequest
            working_memory_request = WorkingMemoryRequest(
                sequence_id=sequence,
                tokens=tuple(comprehension.tokens),
                interpretations=tuple(comprehension.sentence_interpretations),
                context={
                    "source_component": "ore_comprehension",
                    "sentence_interpretations": comprehension.sentence_interpretations,
                    "token_records": comprehension.token_records,
                    "next_word_predictions": comprehension.next_word_predictions,
                },
                language=language,
            )
            second = self._invoke_component("working_memory", working_memory_request)
            outcomes.append(second)
            if second.status == "success":
                working_memory_result = second.result

        if include_discourse:
            from ore_discourse import DiscourseRequest
            sentences = tuple(
                item.strip()
                for item in re.split(r"(?<=[.!?])\s+", text.strip())
                if item.strip()
            )
            propositions = tuple(
                {
                    "sentence_index": index,
                    "text": sentence,
                    "source_interpretation": interpretation,
                }
                for index, (sentence, interpretation) in enumerate(
                    zip(sentences, comprehension.sentence_interpretations)
                )
            )
            third = DiscourseRequest(
                sequence_id=sequence,
                sentences=sentences or (text,),
                propositions=propositions,
                context={
                    "source_component": "ore_comprehension",
                    "sentence_interpretations": comprehension.sentence_interpretations,
                    "token_records": comprehension.token_records,
                    "working_memory": _to_jsonable(working_memory_result),
                },
                language=language,
            )
            outcomes.append(self._invoke_component("discourse", third))

        failed = [item for item in outcomes if item.status != "success"]
        status = "completed" if not failed else ("partial" if len(failed) < len(outcomes) else "failed")
        return self._build_report(
            run_id, sequence, "text comprehension and integration", outcomes, status,
            {
                "execution_order": [item.component_id for item in outcomes],
                "upstream_outputs_passed_forward": [
                    "comprehension interpretations and token records",
                    "working-memory state when available",
                ],
                "biology_simulation_enabled": self.simulate_biology,
            },
        )

    def run_vision(self, request: Any) -> Any:
        """Run the existing integrated N9 visual/orthographic controller."""
        return self.run_component("n9", request)

    def run_biological_master(self, request: Any) -> Any:
        """Run BNBEA's separate functional and biological master tracks."""
        return self.run_component("bnbea", request)

    def register_module(self, backend: Any, *, replace: bool = False) -> ModuleManifest:
        """Register and bind a versioned backend in the common ORE module registry."""
        describe = getattr(backend, "describe", None)
        execute = getattr(backend, "execute", None)
        if not callable(describe) or not callable(execute):
            raise TypeError("backend must expose describe() and execute(inputs, context)")
        manifest = describe()
        if not isinstance(manifest, ModuleManifest):
            raise TypeError("backend.describe() must return an ore_contracts.ModuleManifest")
        module_id = manifest.identity.module_id
        existing = self.module_registry.get_manifest(module_id)
        if existing is not None:
            if not replace:
                raise ValueError(f"module is already registered: {module_id}")
            self.module_registry.unbind_backend(module_id)
        self.module_registry.register(manifest, replace=existing is not None)
        self.module_registry.bind_backend(module_id, backend)
        return manifest

    def coordinate(self, task: CognitiveTask, context: RunContext | None = None):
        """Execute a typed capability task through ORE's validated dependency DAG."""
        return self.task_coordinator.coordinate(task, context)

    def list_modules(self) -> tuple[dict[str, Any], ...]:
        """Return explicit backend, validation, dependency and runnable states."""
        return tuple(
            self.module_registry.diagnostics(manifest.identity.module_id)
            for manifest in self.module_registry.list_manifests()
        )

    def find_capability(self, capability: str) -> tuple[Any, ...]:
        return self.module_registry.find_by_capability(capability)

    def load_package(self, package_name: str):
        """Import one discovered package on demand."""
        return self.package_catalog.import_package(package_name)

    def diagnostics(self) -> dict[str, Any]:
        """Summarize package discovery and live module readiness."""
        modules = self.list_modules()
        point_report = None
        if self.integration_report is not None and callable(getattr(self.integration_report, "to_dict", None)):
            point_report = self.integration_report.to_dict()
        return {
            "repository_root": str(self.repository_root),
            "package_inventory": self.package_catalog.to_dict(),
            "registered_module_count": len(modules),
            "backend_bound_count": sum(item["backend_status"] == "BOUND" for item in modules),
            "runnable_module_count": sum(bool(item["runnable"]) for item in modules),
            "modules": list(modules),
            "point_integration": point_report,
            "default_components": sorted(_DEFAULT_COMPONENTS),
            "registered_processors": sorted(self._components),
            "registered_factories": sorted(self._component_factories),
        }


__all__ = ["ComponentOutcome", "OREExecutionReport", "OREOrchestrator"]
