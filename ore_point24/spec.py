"""Static specification data for the Point 24 unified ORE architecture.

This module encodes the tables of the Point 24 specification as typed, frozen
data: the eight logical layers (Section 2), the 24-point coverage table
(Step 46), the A-G residual category map, the residual technology register
(Step 47) and the 15-phase staged roadmap (Step 49).  These are architectural
facts, not evidence: a point mapped to a layer is *examined*, never
automatically *implemented*.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from .core import POINT24_VERSION


# ---------------------------------------------------------------------------
# Section 2 - the eight cooperating logical layers
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ArchitectureLayer:
    """One logical responsibility boundary (not a deployment unit)."""

    layer_id: str
    name: str
    responsibilities: str
    source_points: Tuple[int, ...]
    ore_boundary: str


SEMANTIC_KNOWLEDGE = "semantic_knowledge"
COGNITIVE_CONTROL = "cognitive_control"
REASONING_PREDICTION = "reasoning_prediction"
PLANNING_LEARNING = "planning_learning"
MEMORY_INFORMATION = "memory_information"
RESOURCE_INTELLIGENCE = "resource_intelligence"
EXECUTION_SECURITY = "execution_security"
IDEA_TRANSLATION = "idea_translation"

LAYERS: Tuple[ArchitectureLayer, ...] = (
    ArchitectureLayer(
        SEMANTIC_KNOWLEDGE, "Semantic and knowledge intelligence",
        "Semantic interpretation, typed concepts, semantic classes, relations, "
        "predicate structures, knowledge graphs, semantic memory, evidence and provenance.",
        (1, 2, 3, 7, 8, 15, 16),
        "USG supplies versioned semantic resources; ORE modules retrieve and interpret "
        "them through declared contracts. Candidate knowledge never silently becomes "
        "authoritative USG knowledge.",
    ),
    ArchitectureLayer(
        COGNITIVE_CONTROL, "Cognitive control and coordination",
        "Goal management, task decomposition, module selection, dependency planning, "
        "specialist coordination, shared cognitive state, event routing, stopping conditions.",
        (1, 2, 6, 11, 15, 19),
        "ore_contracts owns typed orchestration and dependency planning; ore_workspace "
        "owns shared state and controlled publication. Neither is duplicated by a new "
        "monolithic coordinator.",
    ),
    ArchitectureLayer(
        REASONING_PREDICTION, "Reasoning and prediction",
        "Symbolic reasoning, neuro-symbolic proposals, Bayesian and causal inference, "
        "formal verification, predictive world models, scientific models, temporal estimation.",
        (3, 4, 5, 7, 12, 20),
        "Reasoning outputs are typed claims with assumptions, evidence and provenance. "
        "A learned proposal is not a proof; a probability-like score is not automatically "
        "a calibrated probability.",
    ),
    ArchitectureLayer(
        PLANNING_LEARNING, "Planning, learning and discovery",
        "Graph search, hierarchical planning, model-predictive control, reinforcement "
        "learning, multi-agent coordination, evolutionary search, program synthesis, "
        "algorithm discovery, learned strategies.",
        (6, 9, 10, 11, 14, 16),
        "Planning and learning propose actions or policies under explicit constraints; "
        "execution is bounded and candidates are isolated until verified and approved.",
    ),
    ArchitectureLayer(
        MEMORY_INFORMATION, "Memory and information infrastructure",
        "Semantic, episodic and procedural memory; structured storage; approximate "
        "retrieval; streaming summaries; data integrity; versioning; consistency and "
        "incremental computation.",
        (15, 17, 18, 19),
        "Memory records keep stable identifiers, timestamps, lineage, access policy and "
        "version; retrieval results stay distinguishable from verified knowledge.",
    ),
    ArchitectureLayer(
        RESOURCE_INTELLIGENCE, "Computational resource intelligence",
        "Workload observation, CPU and memory estimates, I/O and network monitoring, "
        "accelerator discovery, resource budgeting, scheduling, adaptive allocation.",
        (19, 21, 22, 23),
        "Begin with measured telemetry and deterministic budgets; a learned resource "
        "policy never bypasses hard resource or safety limits.",
    ),
    ArchitectureLayer(
        EXECUTION_SECURITY, "Execution, security and reliability",
        "Worker execution, sandboxing, component isolation, capability management, "
        "cryptographic protection, recovery, distributed execution, auditability, "
        "formal assurance.",
        (17, 19, 20, 22),
        "Modules receive only declared inputs and permissions; failures, abstentions and "
        "unavailable backends remain explicit states; retries are idempotent.",
    ),
    ArchitectureLayer(
        IDEA_TRANSLATION, "Human idea-to-algorithm translation",
        "Natural-language and multimodal input, intent clarification, semantic "
        "formalization, mathematical representation, AlgoIR, algorithm generation, "
        "verification, code generation, controlled execution.",
        (16,),
        "HIC translates and formalizes; USG supplies structured semantic knowledge; ORE "
        "coordinates analysis and validation; ambiguous input becomes uncertainty or a "
        "clarification requirement, never invented requirements.",
    ),
)

LAYER_IDS: Tuple[str, ...] = tuple(layer.layer_id for layer in LAYERS)


# ---------------------------------------------------------------------------
# Step 46 - the 24-point coverage table
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PointFamily:
    """One source point of the 24-point assessment and its integration role."""

    point: int
    family: str
    primary_responsibility: str
    primary_layer: str
    secondary_layers: Tuple[str, ...] = ()
    package: str | None = None
    package_title: str | None = None
    hic_relevant: bool = False

    @property
    def algorithm_id(self) -> str | None:
        """Return a stable package-based identifier independent of local point numbering."""
        if not self.package:
            return None
        package_slug = self.package.removeprefix("ore_").replace("_", "-")
        return f"ore.algorithm.{package_slug}"


POINT_FAMILIES: Tuple[PointFamily, ...] = (
    PointFamily(1, "Blackboard and shared cognitive workspace",
                "Shared findings and cooperative specialist execution",
                COGNITIVE_CONTROL, (SEMANTIC_KNOWLEDGE, MEMORY_INFORMATION),
                "ore_blackboard", "ORE Blackboard Implementation (Point 01)"),
    PointFamily(2, "Cognitive architectures",
                "Cognitive cycles, declarative/procedural control and architecture comparison",
                COGNITIVE_CONTROL, (MEMORY_INFORMATION,),
                "ore_control", "Cognitive Control Layer (Point 02)"),
    PointFamily(3, "Neuro-symbolic AI",
                "Learned proposals with explicit symbolic constraints",
                REASONING_PREDICTION, (SEMANTIC_KNOWLEDGE,),
                "ore_neurosymbolic", "Neuro-Symbolic Reasoning Layer (Point 3)"),
    PointFamily(4, "World models",
                "Predictive state and consequence estimation",
                REASONING_PREDICTION, (PLANNING_LEARNING,),
                "ore_cwm", "Cognitive World Model (Point 04)"),
    PointFamily(5, "Automated reasoning and formal verification",
                "Independent correctness checks",
                REASONING_PREDICTION, (EXECUTION_SECURITY,),
                "ore_formal_reasoning", "FRCE Formal Reasoning (Point 5)"),
    PointFamily(6, "Search and hierarchical planning",
                "Goal decomposition and action planning",
                PLANNING_LEARNING, (COGNITIVE_CONTROL,),
                "ore_planning", "ORE Search Algorithms (Point 12 source family)"),
    PointFamily(7, "Bayesian and causal reasoning",
                "Uncertainty, evidence updates and causal hypotheses",
                REASONING_PREDICTION, (SEMANTIC_KNOWLEDGE,),
                "ore_bayesian_causal", "Bayesian and Causal Reasoning"),
    PointFamily(8, "Graph neural networks and neural algorithmic reasoning",
                "Structural learning and algorithmic pattern proposals",
                SEMANTIC_KNOWLEDGE, (REASONING_PREDICTION,),
                "ore_graph_intelligence", "Graph Intelligence Layer (Point 8)"),
    PointFamily(9, "Evolutionary intelligence",
                "Strategy and algorithm discovery",
                PLANNING_LEARNING, (),
                "ore_evolution_intelligence", "Evolutionary Intelligence (Point 9)"),
    PointFamily(10, "Reinforcement learning",
                "Adaptive decisions from validated experience",
                PLANNING_LEARNING, (RESOURCE_INTELLIGENCE,),
                "ore_rl", "ore_rl reinforcement learning layer (Point 10)"),
    PointFamily(11, "Multi-agent and swarm intelligence",
                "Hierarchical, peer and cooperative task allocation",
                PLANNING_LEARNING, (COGNITIVE_CONTROL,),
                "ore_macl", "MACL Multi-Agent Coordination (Point 11)"),
    PointFamily(12, "Scientific, geometric and physical-world AI",
                "Scientific representations, solvers and geometric models",
                REASONING_PREDICTION, (PLANNING_LEARNING,),
                "ore_science", "ORE Scientific Reasoning Layer"),
    PointFamily(13, "Continuous-time and neuromorphic computing",
                "Temporal and event-driven model options",
                REASONING_PREDICTION, (RESOURCE_INTELLIGENCE,),
                "ore_temporal", "TEDIL Temporal Layer (Point 13)"),
    PointFamily(14, "Specialized learning and foundation models",
                "Specialized, multimodal and transfer-learning backends",
                PLANNING_LEARNING, (SEMANTIC_KNOWLEDGE,),
                "ore_learning", "RLMS Specialized Learning (Point 14)"),
    PointFamily(15, "Agentic systems, memory and protocols",
                "Tool orchestration, semantic/episodic/procedural memory",
                MEMORY_INFORMATION, (COGNITIVE_CONTROL,),
                "ore_agentic", "ORE Agentic Workflow and Memory (Point 15)"),
    PointFamily(16, "Human idea-to-algorithm compiler",
                "HIC, formalization, AlgoIR and synthesis",
                IDEA_TRANSLATION, (), "ore_hic", "ORE HIC (Point 16)", hic_relevant=True),
    PointFamily(17, "Distributed systems and consensus",
                "Distributed coordination and consistency",
                EXECUTION_SECURITY, (MEMORY_INFORMATION,),
                "ore_distributed", "ORE Distributed Systems (Point 17)"),
    PointFamily(18, "Probabilistic data structures and storage",
                "Approximate indexing, streaming summaries and integrity",
                MEMORY_INFORMATION, (),
                "ore_data", "ORE Probabilistic Data Structures (Point 18)"),
    PointFamily(19, "Concurrency and event-driven computing",
                "Bounded workers, event flows and synchronization",
                EXECUTION_SECURITY, (MEMORY_INFORMATION, RESOURCE_INTELLIGENCE),
                "ore_runtime", "ORE Runtime Engine (Point 19)"),
    PointFamily(20, "Cryptography, privacy and verifiable computation",
                "Security, privacy and cryptographic verification",
                EXECUTION_SECURITY, (),
                "ore_security", "ORE Cryptography and Privacy (Point 20)"),
    PointFamily(21, "Hardware-aware numerical and quantum computing",
                "Measured performance optimization and research options",
                RESOURCE_INTELLIGENCE, (),
                "ore_hacol", "ORE HACOL Hardware Awareness (Point 21)"),
    PointFamily(22, "Advanced kernels, isolation and operating systems",
                "Execution isolation and capability boundaries",
                EXECUTION_SECURITY, (RESOURCE_INTELLIGENCE,),
                "ore_runtime_governance", "ORE Runtime Governance (Point 22)"),
    PointFamily(23, "Learning-directed operating systems and resource management",
                "CROS telemetry, admission and adaptive resource policy",
                RESOURCE_INTELLIGENCE, (EXECUTION_SECURITY,),
                "ore_cros", "ORE CROS / LDOS (Point 23)"),
    PointFamily(24, "Complete integration and residual coverage",
                "Unified architecture, traceability, evaluation and roadmap",
                COGNITIVE_CONTROL,
                (SEMANTIC_KNOWLEDGE, REASONING_PREDICTION, PLANNING_LEARNING,
                 MEMORY_INFORMATION, RESOURCE_INTELLIGENCE, EXECUTION_SECURITY,
                 IDEA_TRANSLATION),
                "ore_point24", "This package (Point 24)"),
)

POINTS_BY_NUMBER: Mapping[int, PointFamily] = {family.point: family for family in POINT_FAMILIES}


# ---------------------------------------------------------------------------
# Step 46 - the A-G residual category map
# ---------------------------------------------------------------------------

RESIDUAL_CATEGORIES: Tuple[Tuple[str, str, Tuple[int, ...], str], ...] = (
    ("A", "Human idea compiler", (16,), "HIC: translation, AlgoIR, synthesis, verification."),
    ("B", "Multi-agent and swarm", (11,), "MACL: hierarchical, peer and cooperative allocation."),
    ("C", "Scientific and physical-world AI", (12,), "Scientific representations, solvers, geometric models."),
    ("D", "Neuromorphic intelligence", (13,), "Continuous-time and event-driven model options."),
    ("E", "Specialized learning and agentic systems", (14, 15), "Replaceable learned models, agentic memory and protocols."),
    ("F", "Distributed systems and non-LLM computer science", (17, 18, 19, 20, 21),
     "Consensus, probabilistic structures, concurrency, cryptography, hardware awareness."),
    ("G", "Operating-system architectures", (22, 23), "Isolation kernels and learning-directed resource management."),
)


# ---------------------------------------------------------------------------
# Step 47 - residual technology register (adoption decisions)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ResidualTechnology:
    """One residual technology with its explicit adoption decision."""

    name: str
    group: str
    decision: str  # integrate | evaluate | future_option | reject_defer
    rationale: str
    connection: str


INTEGRATE = "integrate"
EVALUATE = "evaluate_experimentally"
FUTURE_OPTION = "future_option"
REJECT_DEFER = "reject_defer"

RESIDUAL_TECHNOLOGIES: Tuple[ResidualTechnology, ...] = (
    ResidualTechnology("AlphaTensor", "algorithm_discovery", EVALUATE,
                       "Matrix-multiplication decomposition search is a benchmark-driven research path.",
                       "Connects to AlgoIR, program synthesis, verifier and validated strategy registry."),
    ResidualTechnology("AlphaDev", "algorithm_discovery", EVALUATE,
                       "Low-level routine search requires sandboxed evaluation infrastructure.",
                       "Candidate generation, evaluation and acceptance remain separate."),
    ResidualTechnology("AlphaEvolve / FunSearch", "algorithm_discovery", EVALUATE,
                       "LLM-guided evolutionary search for algorithms; evaluation harness required first.",
                       "Algorithm-discovery subsystem with proposal/evaluator/registry separation."),
    ResidualTechnology("DreamCoder", "algorithm_discovery", EVALUATE,
                       "Program synthesis with reusable abstraction discovery.",
                       "HIC and AlgoIR integration path; candidates isolated until verified."),
    ResidualTechnology("Dijkstra / D* Lite / RRT / minimax / alpha-beta / DP", "reasoning_planning", INTEGRATE,
                       "Core deterministic search methods with explicit assumptions.",
                       "ore_point24.search_algorithms implements them directly (Step 24)."),
    ResidualTechnology("Kalman and particle filters", "reasoning_planning", EVALUATE,
                       "State estimation only where linear/Gaussian (or particle) assumptions are defensible.",
                       "Uncertainty subsystem; assumptions recorded with every estimate."),
    ResidualTechnology("Active Inference", "reasoning_planning", FUTURE_OPTION,
                       "Experimental perception/prediction/action framework; not an automatic replacement.",
                       "Research option combining belief, prediction and action selection."),
    ResidualTechnology("Neural Logic Machines / differentiable logic / rule learning", "neurosymbolic", EVALUATE,
                       "Learned proposals remain separate from verified deduction.",
                       "ore_neurosymbolic and the proposal channel (Step 22)."),
    ResidualTechnology("AlphaGeometry-style proposal + deduction", "neurosymbolic", EVALUATE,
                       "Neural proposal followed by symbolic deduction is the canonical pattern.",
                       "Reasoning routing (Step 21) already separates proposal and verification."),
    ResidualTechnology("Memristive and photonic neuromorphic devices", "neuromorphic", FUTURE_OPTION,
                       "Hardware research directions, not software dependencies.",
                       "Retained as research options only."),
    ResidualTechnology("System-One-style bounded decisions", "learning_agentic", INTEGRATE,
                       "Compact bounded decision modules fit the bounded-execution invariant.",
                       "Bounded activation and admission logic."),
    ResidualTechnology("Test-Time Training (TTT)", "learning_agentic", EVALUATE,
                       "Isolated from authoritative USG updates; instability and drift must be tested.",
                       "Learning gates (Step 25) control any promotion."),
    ResidualTechnology("Semantic caching and speculative decoding", "learning_agentic", EVALUATE,
                       "Caches must revalidate identity, scope and version (Step 27).",
                       "Retrieval layer with exact re-check policy."),
    ResidualTechnology("Drift monitoring, model replacement and rollback", "learning_agentic", INTEGRATE,
                       "Required by the promotion and audit machinery.",
                       "ore_point24.promotion implements versioned promotion with rollback (Step 40)."),
    ResidualTechnology("Human approval checkpoints", "learning_agentic", INTEGRATE,
                       "First-class nodes in plans where required.",
                       "Task decomposition keeps approval steps explicit (Step 17)."),
    ResidualTechnology("Singularity / CheriBSD / CHERIoT / Plan 9 / Landlock", "operating_systems", FUTURE_OPTION,
                       "Capability and namespace research; adopt only for concrete deployment needs.",
                       "Boundary decisions use measured need (Step 37)."),
    ResidualTechnology("Helios / Popcorn Linux", "operating_systems", FUTURE_OPTION,
                       "Multi-kernel / heterogeneous cores research.",
                       "Distributed boundaries after measured need."),
    ResidualTechnology("Gramine / Demikernel", "operating_systems", FUTURE_OPTION,
                       "Library-OS and async-I/O research runtimes.",
                       "Sandboxing adapter seam (Step 36)."),
    ResidualTechnology("NOVA / Mnemosyne / SplineFS / OBASE / NetworkedOS", "operating_systems", REJECT_DEFER,
                       "Research operating systems without a current ORE deployment requirement.",
                       "Rejected/deferred with rationale retained for reconsideration."),
    ResidualTechnology("CortenMM-style memory management / MSched / MAPE-K", "operating_systems", INTEGRATE,
                       "MAPE-K is the initial bounded CROS control loop.",
                       "ore_point24.mapek implements the bounded loop (Step 34)."),
)


# ---------------------------------------------------------------------------
# Step 49 - the staged implementation roadmap
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RoadmapPhase:
    """One phase of the 15-phase staged implementation roadmap."""

    phase: int
    development_target: str
    source_references: Tuple[int, ...]
    prerequisites: Tuple[int, ...]


ROADMAP_PHASES: Tuple[RoadmapPhase, ...] = (
    RoadmapPhase(1, "Stabilize USG schemas, semantic rules, versioning and provenance", (1, 3), ()),
    RoadmapPhase(2, "Establish shared cognitive schemas and workspace contracts", (1, 2), (1,)),
    RoadmapPhase(3, "Strengthen module coordination and cognitive control", (1, 2, 11), (2,)),
    RoadmapPhase(4, "Build deterministic symbolic reasoning and formal verification", (3, 5), (3,)),
    RoadmapPhase(5, "Develop HIC and AlgoIR", (16,), (4,)),
    RoadmapPhase(6, "Strengthen memory, retrieval and historical experience recording", (2, 15, 18), (3,)),
    RoadmapPhase(7, "Introduce Bayesian and causal reasoning", (7,), (4,)),
    RoadmapPhase(8, "Introduce planning and predictive world models", (4, 6), (7,)),
    RoadmapPhase(9, "Develop scientific and specialized learning interfaces", (12, 13, 14), (8,)),
    RoadmapPhase(10, "Introduce controlled strategy discovery and adaptive learning", (9, 10), (9,)),
    RoadmapPhase(11, "Strengthen concurrency, execution safety and failure recovery", (17, 19, 20, 22), (10,)),
    RoadmapPhase(12, "Develop CROS resource observation and deterministic allocation", (23,), (11,)),
    RoadmapPhase(13, "Introduce predictive and learned resource policies", (10, 23), (12,)),
    RoadmapPhase(14, "Expand distributed and hardware-aware execution", (11, 17, 21, 22), (13,)),
    RoadmapPhase(15, "Validate the integrated architecture using external benchmarks", tuple(range(1, 25)), tuple(range(1, 15))),
)


def point_layer_incidence() -> dict[str, Any]:
    """Return the Step 3 incidence matrix A as a JSON-friendly mapping.

    ``A[i][j] = 1`` when source point i contributes to layer j.  A point may
    contribute to more than one layer; no source point disappears merely
    because it overlaps another point.
    """
    matrix: dict[str, dict[str, int]] = {}
    for family in POINT_FAMILIES:
        row = {layer_id: 0 for layer_id in LAYER_IDS}
        row[family.primary_layer] = 1
        for secondary in family.secondary_layers:
            row[secondary] = 1
        matrix[f"point_{family.point:02d}"] = row
    return matrix


def layer_for_point(point: int) -> ArchitectureLayer:
    """Return the primary architecture layer of a source point."""
    family = POINTS_BY_NUMBER[point]
    for layer in LAYERS:
        if layer.layer_id == family.primary_layer:
            return layer
    raise KeyError(point)  # pragma: no cover - static data is consistent


__all__ = [
    "POINT24_VERSION",
    "ArchitectureLayer",
    "PointFamily",
    "ResidualTechnology",
    "RoadmapPhase",
    "LAYERS",
    "LAYER_IDS",
    "POINT_FAMILIES",
    "POINTS_BY_NUMBER",
    "RESIDUAL_CATEGORIES",
    "RESIDUAL_TECHNOLOGIES",
    "ROADMAP_PHASES",
    "point_layer_incidence",
    "layer_for_point",
]
