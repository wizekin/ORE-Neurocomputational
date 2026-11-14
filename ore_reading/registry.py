"""Formal 72-operation registry for the Integrated Reading Algorithm."""
from __future__ import annotations
from dataclasses import asdict, dataclass

@dataclass(frozen=True, slots=True)
class OperationSpec:
    operation_id: str
    subsystem: str
    name: str
    functional_rule: str
    computational_form: str
    biological_analogue: str
    outputs: tuple[str, ...]
    def to_dict(self) -> dict[str, object]:
        return asdict(self)

_ROWS = (
("Input integrity","Validate request"),("Input integrity","Unicode normalization"),
("Input integrity","Token boundary preservation"),("Input integrity","Punctuation isolation"),
("Input integrity","Grapheme-cluster extraction"),("Input integrity","Visual evidence initialization"),
("Input integrity","Script and direction encoding"),("Input integrity","Position-coded orthographic form"),
("Orthographic lexicon","Lexicon provenance validation"),("Orthographic lexicon","Lexicon indexing"),
("Orthographic lexicon","Exact orthographic retrieval"),("Orthographic lexicon","Approximate candidate retrieval"),
("Interactive activation","Letter-level bottom-up drive"),("Interactive activation","Lexical frequency prior"),
("Interactive activation","Interactive activation initialization"),("Interactive activation","Recurrent excitation and competition"),
("Dual-route conversion","Lexical phonology lookup"),("Dual-route conversion","Sublexical rule preparation"),
("Dual-route conversion","Grapheme chunking"),("Dual-route conversion","Grapheme-to-phoneme conversion"),
("Dual-route conversion","Route compatibility scoring"),("Dual-route conversion","Route gain calculation"),
("Dual-route conversion","Route evidence integration"),("Dual-route conversion","Pronunciation uncertainty"),
("Phonological assembly","Phoneme feature encoding"),("Phonological assembly","Syllabic nucleus estimation"),
("Phonological assembly","Stress assignment"),("Phonological assembly","Phonological buffer construction"),
("Phonological assembly","Serial position coding"),("Phonological assembly","Recurrent rehearsal"),
("Phonological assembly","Phonological similarity"),("Phonological assembly","Phonological output preparation"),
("Phonological word recognition","Phonological input feature map"),
("Phonological word recognition","TRACE feature-to-phoneme activation"),
("Phonological word recognition","TRACE phoneme-to-word activation"),
("Phonological word recognition","TRACE top-down feedback"),
("Phonological word recognition","TRACE lateral inhibition"),
("Phonological word recognition","Cohort initialization"),
("Phonological word recognition","Cohort incremental elimination"),
("Phonological word recognition","Cohort recognition point"),
("Bayesian lexical selection","Shortlist hypothesis generation"),
("Bayesian lexical selection","Noisy-channel likelihood"),
("Bayesian lexical selection","Lexical prior probability"),
("Bayesian lexical selection","Context prior estimation"),
("Bayesian lexical selection","Bayesian posterior update"),
("Bayesian lexical selection","Shortlist B noise competition"),
("Bayesian lexical selection","Posterior calibration"),
("Bayesian lexical selection","Lexical abstention gate"),
("Connectionist spelling-sound-meaning","Orthographic vector"),
("Connectionist spelling-sound-meaning","Phonological vector"),
("Connectionist spelling-sound-meaning","Semantic feature space"),
("Connectionist spelling-sound-meaning","Orthography-phonology associations"),
("Connectionist spelling-sound-meaning","Orthography-semantics associations"),
("Connectionist spelling-sound-meaning","Phonology-semantics associations"),
("Connectionist spelling-sound-meaning","Triangular recurrent settling"),
("Connectionist spelling-sound-meaning","Meaning activation readout"),
("Semantic integration","Sequence-level lexical integration"),
("Semantic integration","Neighbor-context aggregation"),
("Semantic integration","Context-sensitive sense resolution"),
("Semantic integration","Typed relation proposal"),
("Semantic integration","Situation/context state update"),
("Semantic integration","Semantic coherence estimate"),
("Semantic integration","Final lexical-phonological-semantic decision"),
("Semantic integration","Explanation and evidence ledger"),
("Biological simulation","Functional-to-neural drive mapping"),
("Biological simulation","Orthographic population dynamics"),
("Biological simulation","Lexical and phonological population dynamics"),
("Biological simulation","Auditory-wordform and semantic dynamics"),
("Biological simulation","Neural-state measurement"),
("Biological simulation","Prediction-error comparison"),
("Acceptance","Integrity and operation-trace validation"),
("Acceptance","Structured result and scope declaration"),
)

OPERATION_REGISTRY = tuple(
    OperationSpec(
        f"IR-{i:03d}", subsystem, name,
        f"Execute and preserve the intermediate representation for: {name.lower()}.",
        "See ore_reading.pipeline for the deterministic numerical implementation.",
        "See biological_analogue in the architecture document; generic mapping only.",
        (name.lower().replace(" ", "_"),),
    )
    for i, (subsystem, name) in enumerate(_ROWS, start=1)
)
if len(OPERATION_REGISTRY) != 72:
    raise RuntimeError(f"Expected 72 operations, found {len(OPERATION_REGISTRY)}")
if len({item.operation_id for item in OPERATION_REGISTRY}) != len(OPERATION_REGISTRY):
    raise RuntimeError("Operation IDs must be unique")
OPERATION_BY_ID = {item.operation_id: item for item in OPERATION_REGISTRY}

def get_operation(operation_id: str) -> OperationSpec:
    return OPERATION_BY_ID[operation_id]

def registry_as_dicts() -> list[dict[str, object]]:
    return [item.to_dict() for item in OPERATION_REGISTRY]
