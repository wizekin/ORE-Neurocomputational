# Universal ORE Module Communication and Capability Registry

## Purpose and boundary

This subsystem gives ORE packages a common, typed way to declare module identity, input/output ports, capabilities, dependencies, and execution state. It sits inside `ore_contracts` and is additive: legacy package request/result classes and processor APIs remain intact.

The registry is not a scheduler, workflow engine, global controller, shared workspace, or learning system. It does not start modules or mutate model parameters. A manifest can be discovered and registered while its runtime backend is still unbound.

## Main components

- `module_manifest.py`: `ModuleIdentity`, `InputPort`, `OutputPort`, `ModuleDependency`, and `ModuleManifest`.
- `module_protocol.py`: structural `OREModuleProtocol`, `ModuleStatus`, and `ModuleResult`.
- `module_artifacts.py`: `ContractArtifact`, using the existing `ContractEnvelope`, with deterministic SHA-256 hashing for nested values and NumPy arrays.
- `contracts_registry.py`: exact-version `ContractCatalog`, model-backed `ContractDefinition`, validation reports, and registration of existing IF-05/06/07/08/10/12/13 classes. Existing IF model classes are referenced, not redefined.
- `module_registry.py`: source inventory, manifest registration, separate runtime binding, namespaced capability search, and diagnostics.
- `module_compatibility.py`: deterministic directed producer/consumer checks, explicit converters, dependency completeness, and JSON graph export.
- `module_adapters.py`: wrappers around existing processors. No legacy inheritance or package rewrite is required.
- `module_logging.py`: AST annotations and method-linked events persisted through the existing `ore_logger` SQLite database.

The existing `CanonicalBackendRegistry`, `ContractEnvelope`, legacy IF adapters, and package-specific processors remain available. This registry describes cross-package interfaces; it does not replace the canonical phase backend registry.

## Register a contract and module

All schema versions are complete semantic versions. Compatibility is exact by default; a converter must be registered to cross versions.

```python
from ore_contracts import (
    ContractCatalog, ContractArtifact, ContractDefinition,
    InputPort, ModuleIdentity, ModuleManifest, ModuleRegistry, OutputPort,
)

catalog = ContractCatalog()
catalog.register_model("ORE.TEXT", "1.0.0", str)
catalog.register_model("ORE.TOKENS", "1.0.0", dict)

manifest = ModuleManifest(
    identity=ModuleIdentity(
        module_id="ore.example.tokenizer",
        module_version="1.0.0",
        module_name="Example tokenizer",
        package_path="ore_example",
    ),
    responsibilities=("convert text to token evidence",),
    input_ports=(
        InputPort("text", "ORE.TEXT", ("1.0.0",), expected_data_type="builtins.str"),
    ),
    output_ports=(
        OutputPort(
            "tokens", "ORE.TOKENS", "1.0.0", "builtins.dict",
            "token sequence and offsets",
        ),
    ),
    capabilities=("language.tokenization",),
    execution_modes=("FUNCTIONAL",),
)

registry = ModuleRegistry(catalog)
registry.register(manifest)  # Registered does not mean runnable.
```

A backend must expose `describe()` and `execute(inputs, context)`. `LegacyProcessorAdapter` wraps an existing `process(request)` processor with explicit request/artifact factories. `CanonicalControllerAdapter` wraps the existing `CanonicalOREController.run(context, payload, validation_spec=...)` contract without replacing its canonical B-N backend registry. `ModuleRegistry.bind_backend` checks the backend's manifest against the registered ports and capabilities before marking it runnable. A module with unavailable required dependencies is not reported as runnable even when its local backend is bound.

`ModuleStatus` keeps SUCCESS, PARTIAL, ABSTAINED, UNAVAILABLE, and FAILED distinct. In particular, abstention is not a processing error.

## Exact compatibility rules

For a producer output port (o) and consumer input port (i):

[
C(o,i) = I_{mathrm{id}}(o,i) 	imes V(o,i) 	imes S(o,i)
]

- (I_{mathrm{id}}=1) only when contract identifiers match.
- (V=1) only when the producer's exact contract version is listed in the consumer's accepted versions.
- (S=1) only when both contract models are registered and their declared payload type and artifact validation agree.

A direct edge is valid only when all three factors equal one. An incompatible version or payload type is not silently coerced. `register_converter(contract_id, source_version, target_version, converter)` creates a separately reported `COMPATIBLE_WITH_ADAPTER` path, and an actual artifact is checked against the target contract after conversion.

The validator distinguishes:

- `DIRECTLY_COMPATIBLE`
- `COMPATIBLE_WITH_ADAPTER`
- `INCOMPATIBLE_CONTRACT`
- `INCOMPATIBLE_VERSION`
- `MISSING_REQUIRED_INPUT`
- `MODULE_NOT_AVAILABLE`

A module without a bound, validated, enabled backend remains discoverable but cannot form a runnable graph edge.

## Registry and graph metrics

For module (M), required-dependency completeness is:

[
D(M) = rac{	ext{available required dependencies}}{	ext{all required dependencies}}
]

A module with no required dependencies is treated as complete ((D(M)=1)). Availability means the dependency has a registered, validated backend and is enabled. Optional dependencies do not lower this score.

Registry coverage is:

[
R_c = rac{	ext{registered, manifest-validated modules}}{	ext{modules found by source inventory}}
]

Interface-validation coverage is:

[
I_c = rac{	ext{validated compatible declared connections}}{	ext{declared same-contract connection candidates}}
]

The denominator is explicit in the exported map. When there are no candidate connections, coverage is defined as 1.0 (there was nothing to validate). These ratios are diagnostics only; they do not substitute for functional or biological validation tests.

## Repository inventory

`ModuleRegistry.inventory_repository(root)` inspects Python source in each `ore_*` package. It reports package files, source-defined classes, request/result types, processor-like classes, public imports, internal ORE dependencies, and evidence of functional/biological code. It does not infer capabilities from documentation or register a capability without an explicit manifest. Parse errors are included in the inventory record.

## SQLite annotations and event logging

The logger integration reuses the existing database and tables. It does not create a parallel schema.

```python
from ore_contracts import (
    ContractCatalog, ModuleRegistry, SQLiteModuleEventLogger,
    register_module_annotations,
)

# Index every ore_contracts module, class, function, and method into annotations.
register_module_annotations()

# Registry operations are linked to their AST annotation rows and one run record.
catalog = ContractCatalog()
with SQLiteModuleEventLogger() as logger:
    registry = ModuleRegistry(catalog, event_logger=logger)
    # Register contracts before registering manifests.
```

The SQLite `annotations` table is populated from AST definitions (docstrings when present, source analysis otherwise). Registry events store both the method's `symbol_id` and the containing class's `class_symbol_id` in the existing `events` table.

## First integration path

`tests/test_module_adapters.py` exercises real existing ORE components in sequence:

1. `ore_lexical_access` processes each supplied word using its existing request/result API.
2. A legacy adapter emits a typed lexical-batch artifact.
3. `ore_syntax` consumes text plus optional lexical evidence and emits a syntax-result artifact.
4. `ore_semantic_composition` consumes the syntax result through the existing `composition_request_from_results` adapter and emits a semantic-result artifact.

The test registers typed contracts and manifests, binds all three runtime adapters, checks both directed connections, and validates the artifact outputs. All biological-reference simulations in that test are explicitly disabled; it tests functional interoperability, not a claim that the software reproduces neural communication.

## Mathematical operations used

- Exact schema matching uses Boolean indicators and multiplication for the three required gates above.
- Version conversion uses an explicitly registered mapping (T_{v_s\rightarrow v_t}); there is no inferred or fuzzy interpolation between schema versions.
- Dependency readiness uses a count ratio. A single missing required dependency makes the separate `dependency_complete` flag false even if the ratio is nonzero.
- Coverage measures use integer counts and ratios, with a documented empty-denominator convention.
- Artifact fingerprinting uses SHA-256 over canonical JSON plus NumPy array metadata (dtype, shape) and contiguous byte content. This makes changes in array values, shape, or dtype observable.
- Capability discovery is exact namespaced membership, not a probabilistic similarity score.

## Limitations and non-goals

The inventory is static source inspection, not dynamic proof that a code path works. Backend binding is a structural/interface check, optionally strengthened by a backend-provided `validate_binding()`; full runtime behavior must still be tested by the package integration suite. The registry does not infer undocumented biological claims, run dependencies, schedule work, change learning state, or make a module runnable merely because a manifest exists.
