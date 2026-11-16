"""Traceability map from the 38 source steps to the seven-stage MPO registry.

A source step may map to several finer repository operations. The map is
architectural traceability; execution status is reported by the runtime trace.
"""
from __future__ import annotations

SOURCE_STEP_MAP: tuple[dict[str, object], ...] = (
    {"source_step_id":"1.1","source_step_name":"Identify character positions","stage_id":"S1","operation_ids":("MPO-001",)},
    {"source_step_id":"1.2","source_step_name":"Extract visual features","stage_id":"S1","operation_ids":("MPO-002","MPO-003","MPO-004")},
    {"source_step_id":"1.3","source_step_name":"Generate candidate grapheme identities","stage_id":"S1","operation_ids":("MPO-005","MPO-006","MPO-007")},
    {"source_step_id":"1.4","source_step_name":"Accumulate sensory evidence over time","stage_id":"S1","operation_ids":("MPO-008",)},
    {"source_step_id":"2.1","source_step_name":"Retrieve learned orthographic statistics","stage_id":"S2","operation_ids":("MPO-009",)},
    {"source_step_id":"2.2","source_step_name":"Calculate transition strength","stage_id":"S2","operation_ids":("MPO-010","MPO-011","MPO-012")},
    {"source_step_id":"2.3","source_step_name":"Construct sequence-level predictions","stage_id":"S2","operation_ids":("MPO-013","MPO-014")},
    {"source_step_id":"2.4","source_step_name":"Convert transition probabilities into neural support","stage_id":"S2","operation_ids":("MPO-015",)},
    {"source_step_id":"3.1","source_step_name":"Retrieve neighboring grapheme beliefs","stage_id":"S3","operation_ids":("MPO-016",)},
    {"source_step_id":"3.2","source_step_name":"Generate predictions from the left neighbor","stage_id":"S3","operation_ids":("MPO-017",)},
    {"source_step_id":"3.3","source_step_name":"Generate predictions from the right neighbor","stage_id":"S3","operation_ids":("MPO-018",)},
    {"source_step_id":"3.4","source_step_name":"Combine neighboring evidence","stage_id":"S3","operation_ids":("MPO-019",)},
    {"source_step_id":"3.5","source_step_name":"Transmit contextual information to grapheme populations","stage_id":"S3","operation_ids":("MPO-020",)},
    {"source_step_id":"3.6","source_step_name":"Preserve uncertainty in neighboring positions","stage_id":"S3","operation_ids":("MPO-020",)},
    {"source_step_id":"4.1","source_step_name":"Estimate attentional distribution","stage_id":"S4","operation_ids":("MPO-021","MPO-022")},
    {"source_step_id":"4.2","source_step_name":"Estimate sensory uncertainty","stage_id":"S4","operation_ids":("MPO-023","MPO-024")},
    {"source_step_id":"4.3","source_step_name":"Calculate sensory precision","stage_id":"S4","operation_ids":("MPO-025",)},
    {"source_step_id":"4.4","source_step_name":"Modulate precision through attention","stage_id":"S4","operation_ids":("MPO-026","MPO-027")},
    {"source_step_id":"4.5","source_step_name":"Modulate the influence of visual prediction errors","stage_id":"S4","operation_ids":("MPO-028",)},
    {"source_step_id":"5.1","source_step_name":"Retrieve the prior distribution","stage_id":"S5","operation_ids":("MPO-029",)},
    {"source_step_id":"5.2","source_step_name":"Estimate contextual reliability","stage_id":"S5","operation_ids":("MPO-029",)},
    {"source_step_id":"5.3","source_step_name":"Estimate prior precision","stage_id":"S5","operation_ids":("MPO-029",)},
    {"source_step_id":"5.4","source_step_name":"Adjust prior influence","stage_id":"S5","operation_ids":("MPO-030",)},
    {"source_step_id":"5.5","source_step_name":"Balance sensory and prior reliability","stage_id":"S5","operation_ids":("MPO-031","MPO-032")},
    {"source_step_id":"6.1","source_step_name":"Construct the initial posterior distribution","stage_id":"S6","operation_ids":("MPO-033",)},
    {"source_step_id":"6.2","source_step_name":"Compare the new posterior with the previous beliefs","stage_id":"S6","operation_ids":("MPO-034",)},
    {"source_step_id":"6.3","source_step_name":"Apply recurrent feedback","stage_id":"S6","operation_ids":("MPO-035","MPO-036")},
    {"source_step_id":"6.4","source_step_name":"Control the magnitude of posterior updates","stage_id":"S6","operation_ids":("MPO-036",)},
    {"source_step_id":"6.5","source_step_name":"Repeat the inference process","stage_id":"S6","operation_ids":("MPO-037",)},
    {"source_step_id":"6.6","source_step_name":"Retain alternative interpretations when necessary","stage_id":"S6","operation_ids":("MPO-037",)},
    {"source_step_id":"6.7","source_step_name":"Generate an updated orthographic representation","stage_id":"S6","operation_ids":("MPO-037",)},
    {"source_step_id":"7.1","source_step_name":"Reconstruct the expected visual feature representation","stage_id":"S7","operation_ids":("MPO-038",)},
    {"source_step_id":"7.2","source_step_name":"Calculate feature-space prediction residuals","stage_id":"S7","operation_ids":("MPO-039",)},
    {"source_step_id":"7.3","source_step_name":"Calculate precision-weighted residuals","stage_id":"S7","operation_ids":("MPO-039",)},
    {"source_step_id":"7.4","source_step_name":"Calculate divergence between prior predictions and posterior beliefs","stage_id":"S7","operation_ids":("MPO-038",)},
    {"source_step_id":"7.5","source_step_name":"Calculate uncertainty and confidence","stage_id":"S7","operation_ids":("MPO-040",)},
    {"source_step_id":"7.6","source_step_name":"Diagnose the source of prediction failure","stage_id":"S7","operation_ids":("MPO-040",)},
    {"source_step_id":"7.7","source_step_name":"Return diagnostics to the prediction system","stage_id":"S7","operation_ids":("MPO-040",)},
)

SOURCE_STEP_BY_ID = {str(item["source_step_id"]): item for item in SOURCE_STEP_MAP}

OPERATION_TO_SOURCE_STEPS: dict[str, tuple[str, ...]] = {
    f"MPO-{index:03d}": tuple(
        str(item["source_step_id"]) for item in SOURCE_STEP_MAP
        if f"MPO-{index:03d}" in item["operation_ids"]
    )
    for index in range(1, 41)
}

def source_mapping_as_dicts() -> list[dict[str, object]]:
    """Return a JSON-friendly copy of all source-to-operation mappings."""
    return [
        {**item, "operation_ids": list(item["operation_ids"])}
        for item in SOURCE_STEP_MAP
    ]
