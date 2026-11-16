"""Traceability from the 39 source working-memory operations to ORE's WMCC pipeline.

The canonical WMCC-001..040 registry is preserved. The source's prediction-error,
interpretation-consolidation and task-update-gate operations are exposed as WMX
extensions with distinct execution records.
"""
from __future__ import annotations


SOURCE_OPERATION_MAP: tuple[dict[str, object], ...] = (
    {"source_operation_id":"WM-SRC-001","source_operation_name":"Receive processed linguistic information","source_stage":"S1","repository_operation_ids":("WMCC-002",)},
    {"source_operation_id":"WM-SRC-002","source_operation_name":"Encode information into temporary memory representations","source_stage":"S1","repository_operation_ids":("WMCC-003",)},
    {"source_operation_id":"WM-SRC-003","source_operation_name":"Maintain temporary linguistic activation","source_stage":"S1","repository_operation_ids":("WMCC-006",)},
    {"source_operation_id":"WM-SRC-004","source_operation_name":"Organize memory according to functional buffers","source_stage":"S1","repository_operation_ids":("WMCC-004","WMCC-005")},
    {"source_operation_id":"WM-SRC-005","source_operation_name":"Prioritize important information","source_stage":"S1","repository_operation_ids":("WMCC-007",)},
    {"source_operation_id":"WM-SRC-006","source_operation_name":"Retrieve information from working memory","source_stage":"S1","repository_operation_ids":("WMCC-008",)},
    {"source_operation_id":"WM-SRC-007","source_operation_name":"Update and refresh working memory continuously","source_stage":"S1","repository_operation_ids":("WMCC-006",)},
    {"source_operation_id":"WM-SRC-008","source_operation_name":"Detect incomplete syntactic dependencies","source_stage":"S2","repository_operation_ids":("WMCC-009",)},
    {"source_operation_id":"WM-SRC-009","source_operation_name":"Register pending grammatical expectations","source_stage":"S2","repository_operation_ids":("WMCC-010",)},
    {"source_operation_id":"WM-SRC-010","source_operation_name":"Maintain unresolved dependencies","source_stage":"S2","repository_operation_ids":("WMCC-011",)},
    {"source_operation_id":"WM-SRC-011","source_operation_name":"Retrieve the appropriate grammatical element","source_stage":"S2","repository_operation_ids":("WMCC-008","WMCC-012")},
    {"source_operation_id":"WM-SRC-012","source_operation_name":"Bind the retrieved element to the new structure","source_stage":"S2","repository_operation_ids":("WMCC-013",)},
    {"source_operation_id":"WM-SRC-013","source_operation_name":"Detect structural conflicts","source_stage":"S2","repository_operation_ids":("WMCC-017",)},
    {"source_operation_id":"WM-SRC-014","source_operation_name":"Synchronize syntactic memory with semantic memory","source_stage":"S2","repository_operation_ids":("WMCC-021",)},
    {"source_operation_id":"WM-SRC-015","source_operation_name":"Load the current cognitive objective","source_stage":"S3","repository_operation_ids":("WMCC-015",)},
    {"source_operation_id":"WM-SRC-016","source_operation_name":"Monitor processing conflicts","source_stage":"S3","repository_operation_ids":("WMCC-016","WMCC-017")},
    {"source_operation_id":"WM-SRC-017","source_operation_name":"Estimate required control strength","source_stage":"S3","repository_operation_ids":("WMCC-018",)},
    {"source_operation_id":"WM-SRC-018","source_operation_name":"Identify which information should receive executive support","source_stage":"S3","repository_operation_ids":("WMCC-018","WMCC-019")},
    {"source_operation_id":"WM-SRC-019","source_operation_name":"Protect relevant information from interference","source_stage":"S3","repository_operation_ids":("WMCC-019",)},
    {"source_operation_id":"WM-SRC-020","source_operation_name":"Coordinate different processing systems","source_stage":"S3","repository_operation_ids":("WMCC-021",)},
    {"source_operation_id":"WM-SRC-021","source_operation_name":"Evaluate the consequences of the current interpretation","source_stage":"S3","repository_operation_ids":("WMCC-020","WMCC-022")},
    {"source_operation_id":"WM-SRC-022","source_operation_name":"Generate semantic interpretation candidates","source_stage":"S4","repository_operation_ids":("WMCC-023",)},
    {"source_operation_id":"WM-SRC-023","source_operation_name":"Collect evidence for each candidate","source_stage":"S4","repository_operation_ids":("WMCC-024","WMCC-025","WMCC-026")},
    {"source_operation_id":"WM-SRC-024","source_operation_name":"Calculate candidate scores","source_stage":"S4","repository_operation_ids":("WMCC-027",)},
    {"source_operation_id":"WM-SRC-025","source_operation_name":"Calculate probability distributions over interpretations","source_stage":"S4","repository_operation_ids":("WMCC-028",)},
    {"source_operation_id":"WM-SRC-026","source_operation_name":"Perform competitive activation","source_stage":"S4","repository_operation_ids":("WMCC-028",)},
    {"source_operation_id":"WM-SRC-027","source_operation_name":"Integrate contextual evidence","source_stage":"S4","repository_operation_ids":("WMCC-026","WMCC-027")},
    {"source_operation_id":"WM-SRC-028","source_operation_name":"Evaluate semantic prediction errors","source_stage":"S4","repository_operation_ids":("WMX-001",)},
    {"source_operation_id":"WM-SRC-029","source_operation_name":"Select, preserve or revise the interpretation","source_stage":"S4","repository_operation_ids":("WMCC-029","WMCC-030")},
    {"source_operation_id":"WM-SRC-030","source_operation_name":"Consolidate the selected interpretation","source_stage":"S4","repository_operation_ids":("WMX-002",)},
    {"source_operation_id":"WM-SRC-031","source_operation_name":"Maintain the current task representation","source_stage":"S5","repository_operation_ids":("WMCC-031",)},
    {"source_operation_id":"WM-SRC-032","source_operation_name":"Detect task-changing signals","source_stage":"S5","repository_operation_ids":("WMCC-032",)},
    {"source_operation_id":"WM-SRC-033","source_operation_name":"Evaluate whether switching is necessary","source_stage":"S5","repository_operation_ids":("WMCC-032","WMCC-033")},
    {"source_operation_id":"WM-SRC-034","source_operation_name":"Retrieve the new task set","source_stage":"S5","repository_operation_ids":("WMCC-031","WMCC-034")},
    {"source_operation_id":"WM-SRC-035","source_operation_name":"Open the working-memory updating gate","source_stage":"S5","repository_operation_ids":("WMX-003","WMCC-034")},
    {"source_operation_id":"WM-SRC-036","source_operation_name":"Preserve unfinished cognitive work","source_stage":"S5","repository_operation_ids":("WMCC-033","WMCC-035")},
    {"source_operation_id":"WM-SRC-037","source_operation_name":"Reconfigure processing priorities","source_stage":"S5","repository_operation_ids":("WMCC-034","WMCC-035")},
    {"source_operation_id":"WM-SRC-038","source_operation_name":"Re-establish the previous task when appropriate","source_stage":"S5","repository_operation_ids":("WMCC-037",)},
    {"source_operation_id":"WM-SRC-039","source_operation_name":"Learn from previous switching demands","source_stage":"S5","repository_operation_ids":("WMCC-038",)},
)

WM_EXTENSION_REGISTRY: tuple[dict[str, str], ...] = (
    {
        "operation_id":"WMX-001",
        "source_operation_id":"WM-SRC-028",
        "operation_name":"Evaluate semantic prediction error",
        "implementation":"WorkingMemoryProcessor._evaluate_semantic_prediction_error",
        "evidence_policy":"Only compute when an explicit prediction or error signal is supplied",
    },
    {
        "operation_id":"WMX-002",
        "source_operation_id":"WM-SRC-030",
        "operation_name":"Consolidate selected interpretation",
        "implementation":"WorkingMemoryProcessor._consolidate_selected_interpretation",
        "evidence_policy":"Consolidate only an interpretation accepted by the confidence and margin gate",
    },
    {
        "operation_id":"WMX-003",
        "source_operation_id":"WM-SRC-035",
        "operation_name":"Open the working-memory updating gate",
        "implementation":"WorkingMemoryProcessor._op_reconfigure_task_control",
        "evidence_policy":"Open only for initial task setup or a permitted explicit task switch",
    },
)

SOURCE_OPERATION_BY_ID = {
    str(item["source_operation_id"]): item for item in SOURCE_OPERATION_MAP
}
SOURCE_OPERATION_TO_REPOSITORY: dict[str, tuple[str, ...]] = {
    str(item["source_operation_id"]): tuple(item["repository_operation_ids"])
    for item in SOURCE_OPERATION_MAP
}
REPOSITORY_OPERATION_TO_SOURCE: dict[str, tuple[str, ...]] = {
    f"WMCC-{index:03d}": tuple(
        str(item["source_operation_id"]) for item in SOURCE_OPERATION_MAP
        if f"WMCC-{index:03d}" in item["repository_operation_ids"]
    )
    for index in range(1, 41)
}


def source_mapping_as_dicts() -> list[dict[str, object]]:
    """Return a JSON-friendly copy of the 39 source operation mappings."""
    return [
        {**item, "repository_operation_ids": list(item["repository_operation_ids"])}
        for item in SOURCE_OPERATION_MAP
    ]
