from __future__ import annotations

from ore_orchestrator import OREOrchestrator, OREPackageCatalog


class EchoProcessor:
    def process(self, request):
        return {"received": request}


def test_catalog_discovers_repository_packages_without_importing_them():
    catalog = OREPackageCatalog()
    names = set(catalog.package_names)

    assert "ore_contracts" in names
    assert "ore_workspace" in names
    assert "ore_n9" in names
    assert "ore_orchestrator" in names
    assert len(names) >= 80


def test_custom_processor_uses_common_gateway():
    ore = OREOrchestrator(integrate_point_modules=False)
    processor = EchoProcessor()
    ore.register_component("echo", processor, package_name="ore_orchestrator")

    result = ore.run_component("echo", {"text": "hello"})

    assert result == {"received": {"text": "hello"}}


def test_text_workflow_connects_comprehension_working_memory_and_discourse():
    ore = OREOrchestrator(integrate_point_modules=False, simulate_biology=False)
    report = ore.run_text("The chef prepared a meal.")

    assert report.status == "completed"
    assert [item.component_id for item in report.components] == [
        "comprehension", "working_memory", "discourse",
    ]
    assert all(item.status == "success" for item in report.components)
    assert report.components[0].trace_count > 0
    assert report.components[1].trace_count > 0
    assert report.components[2].trace_count > 0


def test_text_workflow_can_run_comprehension_only():
    ore = OREOrchestrator(integrate_point_modules=False)
    report = ore.run_text(
        "A child opened the door.",
        include_working_memory=False,
        include_discourse=False,
    )

    assert report.status == "completed"
    assert len(report.components) == 1
    assert report.components[0].component_id == "comprehension"


def test_default_bootstrap_reuses_point24_integration_manager():
    ore = OREOrchestrator(integrate_point_modules=True)

    assert ore.integration_report is not None
    assert "package_inventory" in ore.diagnostics()
    assert "failures" in ore.integration_report.to_dict()


def test_vision_entry_point_delegates_to_n9():
    from ore_n9 import N9Patch, N9Request

    ore = OREOrchestrator(integrate_point_modules=False)
    request = N9Request(
        sequence_id="orchestrator-vision-test",
        patches=[N9Patch("patch-1", grapheme_probabilities={"A": 1.0})],
    )

    result = ore.run_vision(request)

    assert result.sequence_id == "orchestrator-vision-test"
    assert len(result.operation_trace) == 120
