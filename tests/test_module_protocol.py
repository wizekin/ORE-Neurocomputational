"""Protocol and standardized result tests for universal ORE modules."""
from ore_contracts.module_protocol import ModuleResult, ModuleStatus, OREModuleProtocol


def test_module_result_preserves_abstention_and_failure_as_distinct_states():
    """An evidence-based abstention is not conflated with a processor failure."""
    abstained = ModuleResult("ore.test", "1.0.0", ModuleStatus.ABSTAINED)
    failed = ModuleResult("ore.test", "1.0.0", "failed", diagnostics={"reason": "exception"})
    assert abstained.status is ModuleStatus.ABSTAINED
    assert failed.status is ModuleStatus.FAILED
    assert abstained.status is not failed.status


def test_protocol_is_structural_not_a_required_base_class():
    """A simple object with describe and execute methods satisfies the runtime protocol."""
    class StructuralModule:
        def describe(self):
            return None

        def execute(self, inputs, context):
            return ModuleResult("ore.test", "1.0.0", ModuleStatus.SUCCESS)

    assert isinstance(StructuralModule(), OREModuleProtocol)
