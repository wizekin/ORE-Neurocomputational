# Master ORE Repository Orchestrator

## Purpose

The ore_orchestrator package is the single public runtime entry point for the
ORE Neurocomputational repository. It sits above the existing cognitive
packages and composes them without replacing their algorithms or weakening
their native request/result contracts.

It has four responsibilities:

1. Discover the repository's Python packages without importing them all.
2. Provide one processor gateway for built-in and caller-configured packages.
3. Run explicit end-to-end workflows that pass upstream results to downstream
   components as inspectable context.
4. Delegate typed multi-module tasks to the existing ore_contracts dependency
   planner, artifact validator, bounded executor and workspace publication path.

## Installation and first run

From the repository root:

~~~powershell
python -m pip install -e ".[dev]"
python -m ore_orchestrator inventory
python -m ore_orchestrator text "The chef prepared a meal."
~~~

A console command is also installed:

~~~powershell
ore inventory
ore text "The chef prepared a meal."
~~~

The default run uses the functional tracks. Set --simulate-biology to enable
the configured LIF reference simulations. These simulations are computational
approximations, not validated reproductions of the human brain.

## Python entry point

~~~python
from ore_orchestrator import OREOrchestrator

ore = OREOrchestrator()
report = ore.run_text("The chef prepared a meal.")

print(report.status)
for component in report.components:
    print(component.component_id, component.status, component.trace_count)
~~~

The default text workflow executes in this order:

~~~text
Text
  |
  v
ore_comprehension
  |  sentence interpretations, token records, next-word predictions
  v
ore_working_memory
  |  token sequence + upstream interpretations and predictions
  v
ore_discourse
     text + comprehension evidence + working-memory state
~~~

Working memory and discourse can be disabled independently:

~~~python
report = ore.run_text(
    "The chef prepared a meal.",
    include_working_memory=False,
    include_discourse=False,
)
~~~

The report preserves each component's native result, status, elapsed time and
trace count. A downstream component failure is reported as a partial run when
an earlier component succeeded; it is not silently converted into success.

## Accessing the other packages

The static catalog describes every importable package directory at the
repository root. Discovery reads package files and __all__; it does not import
all 90+ packages or claim that every package has a ready-to-run backend.

~~~python
ore = OREOrchestrator(integrate_point_modules=False)

for package in ore.package_catalog.packages:
    print(package.name, package.category, package.entrypoint_modules)

# Import a discovered package only when it is needed.
syntax_package = ore.load_package("ore_syntax")
~~~

The facade includes lazy defaults for stable, default-constructible public
processors such as comprehension, working_memory, discourse, syntax,
semantic_composition, semantic, reading, n1 through n4, n6, n9, and bnbea.
N3 must be fitted with labelled N2 patches before it can process a sequence. N5 is a model-state builder (initialized from a fitted N3 processor), not a generic process(request) component, so it is deliberately accessed through its native package API instead of the generic registry. N4 and N6 also require their native upstream result/model inputs. Data-dependent models must be configured by the caller. For example,
ore_lexical_access.LexicalAccessProcessor requires a non-empty lexicon and
must not be fabricated by the orchestrator.

Use the same gateway for a custom processor from any discovered package:

~~~python
from ore_lexical_access import LexicalAccessProcessor

lexical_processor = LexicalAccessProcessor(my_lexicon)
ore.register_component(
    "lexical_access",
    lexical_processor,
    package_name="ore_lexical_access",
)

result = ore.run_component("lexical_access", my_lexical_access_request)
~~~

For a factory that needs model data or other resources:

~~~python
ore.register_component_factory(
    "custom_lexical_access",
    lambda: LexicalAccessProcessor(my_lexicon),
    package_name="ore_lexical_access",
)
~~~

run_component passes the caller's native typed request unchanged. It invokes
process(request) or run(request); it does not guess missing inputs or
translate arbitrary objects into another package's contract.

## Visual and master biological entry points

N9 remains the integrated visual/orthographic/language controller. It accepts
an ore_n9.N9Request, including already-localized image patches, N2 results,
feature vectors or explicit grapheme probability distributions:

~~~python
from ore_n9 import N9Request, N9Patch

request = N9Request(
    sequence_id="sample-001",
    patches=[N9Patch("patch-001", grapheme_probabilities={"A": 1.0})],
)
result = ore.run_vision(request)
~~~

The default N9 adapter stack connects the repository's syntax, discourse and
semantic-composition processors. N6 lexical access remains an injected
component because a real lexical inventory is required.

The biological master pipeline is available separately:

~~~python
from ore_bnbea import BNBEARequest
from ore_n9 import N9Request, N9Patch

n9_request = N9Request(
    sequence_id="sample-002",
    patches=[N9Patch("patch-001", grapheme_probabilities={"A": 1.0})],
)
master_result = ore.run_biological_master(BNBEARequest(n9_request=n9_request))
~~~

BNBEA's functional and biological tracks remain separate in its native result.
It also creates/uses the configured SQLite annotation database, so call it
explicitly when that simulation and logging workload is wanted.

## Coordinating contract-compatible modules

ore_contracts remains the authoritative module contract and task-DAG layer.
The orchestrator exposes it directly:

~~~python
from ore_contracts import CognitiveTask

task = CognitiveTask(
    goal="Resolve a supplied task using registered ORE capabilities",
    required_capabilities=("cognition.rl.verification",),
    input_artifacts={},
)
result = ore.coordinate(task)
print(result.status, result.plan.to_dict())
~~~

A custom backend can be attached through the common manifest/protocol:

~~~python
ore.register_module(my_contract_compatible_backend)
result = ore.coordinate(my_cognitive_task)
~~~

A backend must implement describe() and execute(inputs, context), and its
manifest ports must reference contracts already registered in the same
ContractCatalog. Outputs must be ContractArtifact objects that satisfy their
declared schemas and provenance requirements.

## Package and module status are different

The orchestrator deliberately keeps these states separate:

- **Discovered** means a package directory exists and can be inspected.
- **Imported** means its Python package was loaded on demand.
- **Manifest registered** means a module declared a versioned contract.
- **Backend bound** means an implementation is attached to that manifest.
- **Validated / runnable** means the registry's checks passed and required
  dependencies are ready.
- **Executed** means the component or backend actually ran for this request.

By default, OREOrchestrator reuses ore_point24.PointIntegrationManager to
register its native modules, manifest-only modules and honest status adapters.
A status adapter is an import/symbol probe; it is not a substitute for the
package's cognitive backend. Disable this step with
OREOrchestrator(integrate_point_modules=False) when only the direct processor
gateway is needed.

Inspect the live integration state:

~~~python
diagnostics = ore.diagnostics()
print(diagnostics["package_inventory"]["package_count"])
print(diagnostics["registered_module_count"])
print(diagnostics["runnable_module_count"])
~~~

## Scope and scientific limitations

This package is an orchestration layer, not a new general reasoning algorithm.
It does not make every package automatically compatible, synthesize missing
lexicons or models, train N3 without labeled data, or claim that a package is
biologically validated. Components with mandatory data, training, permissions,
or external evidence remain unavailable until those requirements are supplied.

Software tests validate the orchestration and data hand-offs. They do not
establish that the ORE architecture reproduces human cognition.
