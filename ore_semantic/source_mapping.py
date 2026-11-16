"""Traceability for the 44 source semantic/lexical-memory operations."""
from __future__ import annotations

SOURCE_COMPONENTS = (
    ("A", "Anterior temporal lobe semantic hub", 1, 5),
    ("B", "Modality-specific semantic spokes", 6, 10),
    ("C", "Distributed semantic feature networks", 11, 15),
    ("D", "Inferior frontal semantic control", 16, 22),
    ("E", "Semantic retrieval and concept selection", 23, 28),
    ("F", "Episodic word-form memories", 29, 36),
    ("G", "Abstract lexical prototypes", 37, 44),
)

SOURCE_OPERATION_MAP: tuple[dict[str, object], ...] = (
    {"source_step_id":"01","component":"A","name":"Receive incoming lexical or conceptual activation"},
    {"source_step_id":"02","component":"A","name":"Activate appropriate ATL neuronal populations"},
    {"source_step_id":"03","component":"A","name":"Integrate information from multiple modalities"},
    {"source_step_id":"04","component":"A","name":"Construct a coherent conceptual representation"},
    {"source_step_id":"05","component":"A","name":"Complete an incomplete conceptual representation"},
    {"source_step_id":"06","component":"B","name":"Identify modalities associated with the concept"},
    {"source_step_id":"07","component":"B","name":"Activate modality-specific feature populations"},
    {"source_step_id":"08","component":"B","name":"Retrieve modality-specific features"},
    {"source_step_id":"09","component":"B","name":"Exchange information between spokes and ATL"},
    {"source_step_id":"10","component":"B","name":"Weight modality reliability"},
    {"source_step_id":"11","component":"C","name":"Activate individual semantic features"},
    {"source_step_id":"12","component":"C","name":"Activate associated semantic features"},
    {"source_step_id":"13","component":"C","name":"Combine information from multiple semantic pathways"},
    {"source_step_id":"14","component":"C","name":"Calculate semantic relationships and similarities"},
    {"source_step_id":"15","component":"C","name":"Maintain distributed activation and competition"},
    {"source_step_id":"16","component":"D","name":"Receive current task or contextual requirement"},
    {"source_step_id":"17","component":"D","name":"Estimate semantic control demand"},
    {"source_step_id":"18","component":"D","name":"Identify task-relevant features"},
    {"source_step_id":"19","component":"D","name":"Amplify relevant features"},
    {"source_step_id":"20","component":"D","name":"Suppress irrelevant or competing features"},
    {"source_step_id":"21","component":"D","name":"Send top-down control signals to semantic representations"},
    {"source_step_id":"22","component":"D","name":"Repeat semantic activation under control"},
    {"source_step_id":"23","component":"E","name":"Generate candidate concepts"},
    {"source_step_id":"24","component":"E","name":"Calculate evidence supporting each candidate"},
    {"source_step_id":"25","component":"E","name":"Calculate candidate probabilities"},
    {"source_step_id":"26","component":"E","name":"Resolve ambiguous meanings"},
    {"source_step_id":"27","component":"E","name":"Select the most appropriate concept"},
    {"source_step_id":"28","component":"E","name":"Construct the final semantic representation"},
    {"source_step_id":"29","component":"F","name":"Capture current word-form episode","memory_operation_id":"SM-001"},
    {"source_step_id":"30","component":"F","name":"Encode episode into a memory representation","memory_operation_id":"SM-002"},
    {"source_step_id":"31","component":"F","name":"Separate similar episodic traces","memory_operation_id":"SM-003"},
    {"source_step_id":"32","component":"F","name":"Bind episode components","memory_operation_id":"SM-004"},
    {"source_step_id":"33","component":"F","name":"Store episodic memory trace","memory_operation_id":"SM-005"},
    {"source_step_id":"34","component":"F","name":"Retrieve matching episodic traces","memory_operation_id":"SM-006"},
    {"source_step_id":"35","component":"F","name":"Perform pattern completion","memory_operation_id":"SM-007"},
    {"source_step_id":"36","component":"F","name":"Send retrieved episodic information back into semantic processing","memory_operation_id":"SM-008"},
    {"source_step_id":"37","component":"G","name":"Collect multiple word-form experiences","memory_operation_id":"SM-009"},
    {"source_step_id":"38","component":"G","name":"Identify common features across episodes","memory_operation_id":"SM-010"},
    {"source_step_id":"39","component":"G","name":"Build abstract orthographic representation","memory_operation_id":"SM-011"},
    {"source_step_id":"40","component":"G","name":"Build abstract phonological representation","memory_operation_id":"SM-012"},
    {"source_step_id":"41","component":"G","name":"Construct abstract lexical identity","memory_operation_id":"SM-013"},
    {"source_step_id":"42","component":"G","name":"Strengthen prototype through experience","memory_operation_id":"SM-014"},
    {"source_step_id":"43","component":"G","name":"Consolidate abstract representation","memory_operation_id":"SM-015"},
    {"source_step_id":"44","component":"G","name":"Generalize across new experiences","memory_operation_id":"SM-016"},
)

def source_mapping_as_dicts() -> list[dict[str, object]]:
    """Return a JSON-friendly copy of the complete source operation mapping."""
    return [dict(row) for row in SOURCE_OPERATION_MAP]
