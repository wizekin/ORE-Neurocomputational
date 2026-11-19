"""Acceptance tests for Task 4 raw-image to language integration."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ore_contracts import CognitiveTaskCoordinator, ContractCatalog, ModuleRegistry, ModuleStatus
from ore_text_localization import (
    TextImageRequest,
    TextLocalizationConfig,
    TextLocalizer,
    VisualLanguagePathway,
    build_visual_language_manifest,
    build_visual_language_task,
    coordinate_visual_language_input,
    normalize_grayscale_image,
    operation_registry_as_dicts,
    register_visual_language_annotations,
    register_visual_language_module,
)
from ore_workspace import WorkspaceManager


def _printed_page() -> np.ndarray:
    """Create a deterministic two-line page with two words on its first line."""
    image = np.ones((72, 112), dtype=np.float32)
    image[10:25, 10:18] = 0.0
    image[10:25, 22:30] = 0.0
    image[10:25, 45:53] = 0.0
    image[42:57, 10:18] = 0.0
    return image


def _fake_n9() -> object:
    """Build a deterministic N9-shaped test double with every configured adapter slot."""

    class FakeN9:
        """Stand in for N9 while preserving the real adapter and result interfaces."""

        def __init__(self) -> None:
            """Declare every adapter slot consumed by the visual-language pathway."""
            self.adapters = SimpleNamespace(
                n1=object(),
                n2=object(),
                n3=SimpleNamespace(_fitted=True, class_names=("a", "b")),
                n4=object(),
                n6=object(),
                n7=object(),
                n8=object(),
                syntax=object(),
            )

        def process(self, request):
            """Return explicit lexical, syntactic and semantic evidence for the test request."""
            assert request.word_spans
            return {
                "sequence_id": request.sequence_id,
                "class_names": ["a", "b"],
                "decision": {"status": "CANDIDATE", "confidence": 0.93},
                "lexical_hypotheses": [
                    {"start": 1, "end": 2, "form": "dog"},
                    {"start": 0, "end": 1, "form": "the"},
                ],
                "syntax": {"tokens": ["the", "dog"], "long_distance_dependencies": []},
                "propositions": [{"proposition_id": "p1", "predicate": "exist"}],
                "semantic_graph": {"nodes": [{"id": "n1", "label": "dog"}], "edges": []},
                "context_model": {"topic": "test"},
            }

    return FakeN9()


class FakeWorkingMemory:
    """Capture the typed request and return a small deterministic result."""

    def __init__(self) -> None:
        """Initialize the request slot."""
        self.request = None

    def process(self, request):
        """Return a snapshot of supplied token and interpretation evidence."""
        self.request = request
        return {"tokens": list(request.tokens), "interpretations": len(request.interpretations)}


def _fake_pathway(*, coordinator_handoff=None) -> VisualLanguagePathway:
    """Build a complete software pathway with deterministic downstream test doubles."""
    return VisualLanguagePathway(
        n9_processor=_fake_n9(),
        working_memory_processor=FakeWorkingMemory(),
        register_annotations=False,
        coordinator_handoff=coordinator_handoff,
    )


def test_localizer_preserves_line_order_word_spans_and_n9_contract() -> None:
    """Produce N9-ready crops while retaining two lines and three word intervals."""
    request = TextImageRequest(
        sequence_id="page-1",
        image=_printed_page(),
        script_profile={"language": "en", "script": "Latn", "reading_direction": "ltr"},
    )
    result = TextLocalizer().localize(request)

    assert len(result.localized_glyphs) == 4
    assert result.word_spans == ((0, 2), (2, 3), (3, 4))
    assert len(result.line_information) == 2
    assert all(glyph.image.shape == (32, 32) for glyph in result.localized_glyphs)
    assert all(np.min(glyph.image) < np.max(glyph.image) for glyph in result.localized_glyphs)

    n9_request = result.to_n9_request(task="read")
    assert len(n9_request.patches) == 4
    assert n9_request.word_spans == result.word_spans
    assert n9_request.script_profile["language"] == "en"
    assert n9_request.patches[0].source_metadata["localization_source"] == "ore_text_localization"


def test_normalization_accepts_rgb_and_bounds_luminance() -> None:
    """Convert RGB bytes to one finite grayscale channel with bounded values."""
    rgb = np.zeros((8, 9, 3), dtype=np.uint8)
    rgb[..., 1] = 255
    gray = normalize_grayscale_image(rgb)

    assert gray.shape == (8, 9)
    assert gray.dtype == np.float32
    assert np.isfinite(gray).all()
    assert float(np.min(gray)) >= 0.0
    assert float(np.max(gray)) <= 1.0


def test_rtl_reading_order_is_not_replaced_by_left_to_right_order() -> None:
    """Sort distinct glyph observations in the caller-declared right-to-left direction."""
    image = np.ones((32, 38), dtype=np.float32)
    image[8:23, 8:16] = 0.0
    image[8:23, 19:27] = 0.0
    result = TextLocalizer().process(TextImageRequest(
        sequence_id="rtl",
        image=image,
        script_profile={"language": "test", "reading_direction": "rtl"},
    ))

    assert len(result.localized_glyphs) == 2
    assert result.localized_glyphs[0].bounding_box[0] > result.localized_glyphs[1].bounding_box[0]
    assert result.word_spans == ((0, 2),)


def test_touching_glyph_cut_keeps_an_unsplit_alternative() -> None:
    """Retain an unsplit bounding-box hypothesis when a low-ink valley supports a split."""
    image = np.ones((24, 36), dtype=np.float32)
    image[4:20, 3:15] = 0.0
    image[4:20, 17:29] = 0.0
    image[11, 15:17] = 0.0
    result = TextLocalizer(TextLocalizationConfig(minimum_ink_pixels=1)).localize(TextImageRequest(
        sequence_id="touching",
        image=image,
        script_profile={"language": "en", "reading_direction": "ltr"},
    ))

    assert len(result.localized_glyphs) == 2
    assert result.localized_glyphs[0].segmentation_confidence < 0.88
    assert len(result.segmentation_hypotheses) == 2
    primary, alternative = result.segmentation_hypotheses
    assert len(primary.bounding_boxes) == 2
    assert len(alternative.bounding_boxes) == 1
    assert sum(item.probability for item in result.segmentation_hypotheses) == pytest.approx(1.0)


def test_blank_page_returns_empty_result_without_fabricating_glyphs() -> None:
    """Abstain cleanly when the input has no non-uniform foreground."""
    result = TextLocalizer().localize(TextImageRequest(
        sequence_id="blank",
        image=np.ones((32, 32), dtype=np.float32),
    ))

    assert result.localized_glyphs == ()
    assert result.word_spans == ()
    assert result.diagnostics["status"] == "no_text_like_ink_detected"


def test_operation_registry_and_module_manifest_are_consistent() -> None:
    """Expose mathematical operation identifiers and valid module communication metadata."""
    operations = operation_registry_as_dicts()
    manifest = build_visual_language_manifest()

    assert [item["operation_id"] for item in operations] == [
        f"TLP-{index:03d}" for index in range(1, 10)
    ]
    assert manifest.identity.module_id == "language.visual_pathway"
    assert "language.semantic_composition" in manifest.capabilities
    assert all(not dependency.required for dependency in manifest.dependencies)
    assert manifest.input_ports[0].expected_data_type == "ore_text_localization.contracts.TextImageRequest"


def test_n9_to_working_memory_and_main_coordinator_callback_are_connected() -> None:
    """Pass ordered lexical and semantic evidence through working memory and a callback seam."""
    captured_handoffs = []
    working_memory = FakeWorkingMemory()
    pathway = VisualLanguagePathway(
        n9_processor=_fake_n9(),
        working_memory_processor=working_memory,
        register_annotations=False,
        coordinator_handoff=captured_handoffs.append,
    )
    result = pathway.process(TextImageRequest(
        sequence_id="handoff",
        image=_printed_page(),
        script_profile={"language": "en", "reading_direction": "ltr"},
    ))

    assert result.status == "COMPLETED"
    assert working_memory.request is not None
    assert working_memory.request.tokens == ("the", "dog")
    assert result.working_memory["status"] == "completed"
    assert result.integration["coordinator_handoff"] == "sent"
    assert captured_handoffs[0]["producer_module"] == "ore_text_localization"


def test_task3_coordinator_registers_executes_and_publishes_task4(tmp_path) -> None:
    """Schedule Task 4 through the real coordinator and validate its typed result lineage."""
    catalog = ContractCatalog()
    registry = ModuleRegistry(catalog)
    pathway = _fake_pathway()
    adapter = register_visual_language_module(catalog, registry, pathway)
    request = TextImageRequest(
        sequence_id="coordinator-page",
        image=_printed_page(),
        script_profile={"language": "en", "script": "Latn", "reading_direction": "ltr"},
    )
    task = build_visual_language_task(request, task_id="task04-integration-test")
    workspace_manager = WorkspaceManager(repository_root=tmp_path, enable_logging=False)
    coordinator = CognitiveTaskCoordinator(
        registry,
        catalog,
        workspace_manager=workspace_manager,
        repository_root=tmp_path,
        enable_logging=False,
    )
    try:
        result = coordinator.coordinate(task)
    finally:
        workspace_manager.close()

    module_result = result.module_results["language.visual_pathway"]
    assert adapter.describe() == build_visual_language_manifest()
    assert registry.get_backend("language.visual_pathway") is adapter
    assert result.status == "RESOLVED"
    assert module_result.status is ModuleStatus.SUCCESS
    assert len(module_result.artifacts) == 1
    output = module_result.artifacts[0]
    assert output.payload.status == "COMPLETED"
    assert output.envelope.parent_artifact_ids == (
        task.input_artifacts["raw_text_image"].envelope.fingerprint(),
    )


def test_text_image_request_rejects_nonfinite_pixels() -> None:
    """Reject invalid numeric evidence before localization."""
    image = np.ones((8, 8), dtype=np.float32)
    image[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        TextImageRequest(sequence_id="bad", image=image)


def test_annotation_registration_uses_the_existing_sqlite_catalogue(tmp_path) -> None:
    """Index all new pathway definitions into the shared annotations table."""
    database_path = tmp_path / "ore_logger.sqlite3"
    report = register_visual_language_annotations(database_path=database_path)

    assert database_path.is_file()
    assert report["database_path"] == str(database_path.resolve())
    assert report["symbols_written"] > 0
    assert report["parse_errors"] == 0
