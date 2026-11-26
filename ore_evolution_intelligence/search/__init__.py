"""Search layer: genetic algorithms, evolution strategies, CMA-ES, GP, novelty/Pareto."""
from .cmaes import CMAESAdapter, CMAESResult
from .evolution_strategy import ESResult, EvolutionStrategyOptimizer
from .genetic_optimizer import (
    EvolutionSearchReport, GeneticOptimizer, GeneticSearchConfig, GenerationStats,
    genome_complexity,
)
from .genetic_programming import (
    ACTION_OPS, BOOL_OPS, COMPARE_OPS, DEFAULT_GRAMMAR, GRAMMAR_VERSION, IR_TYPES,
    NUM_OPS, POLICY_CONSTANTS, POLICY_FEATURES, ActionNode, BoolOpNode,
    CombineActionNode, CompareNode, ConstantNode, FeatureNode, IfActionNode, IRNode,
    NumOpNode, PolicyIRGenome, TypedGrammar, WhenActionNode, node_from_dict,
    random_tree, static_constant_value, subtree_crossover, subtree_mutation,
)
from .novelty_search import (
    DEFAULT_K, DEFAULT_OBJECTIVES, NoveltyArchive, NoveltyGuidedSelector,
    NoveltyReport, ParetoArchive, ParetoEntry, ParetoUpdate, dominates,
    pareto_front_is_consistent,
)
from .operators import (
    PARAMETER_GROUPS, crossover_genomes, mutate_genome, parameter_crossover,
    parameter_mutate, parameter_random, parameter_safe_perturbation,
    parsimony_preferred, policy_ir_crossover, policy_ir_mutate, policy_ir_random,
    policy_table_crossover, policy_table_mutate, policy_table_random, random_genome,
)

__all__ = [
    # genetic programming (exported first: genome kinds depend on it)
    "ACTION_OPS", "BOOL_OPS", "COMPARE_OPS", "DEFAULT_GRAMMAR", "GRAMMAR_VERSION",
    "IR_TYPES", "NUM_OPS", "POLICY_CONSTANTS", "POLICY_FEATURES", "ActionNode",
    "BoolOpNode", "CombineActionNode", "CompareNode", "ConstantNode", "FeatureNode",
    "IfActionNode", "IRNode", "NumOpNode", "PolicyIRGenome", "TypedGrammar",
    "WhenActionNode", "node_from_dict", "random_tree", "static_constant_value",
    "subtree_crossover", "subtree_mutation",
    # novelty and Pareto
    "DEFAULT_K", "DEFAULT_OBJECTIVES", "NoveltyArchive", "NoveltyGuidedSelector",
    "NoveltyReport", "ParetoArchive", "ParetoEntry", "ParetoUpdate", "dominates",
    "pareto_front_is_consistent",
    # operators
    "PARAMETER_GROUPS", "crossover_genomes", "mutate_genome", "parameter_crossover",
    "parameter_mutate", "parameter_random", "parameter_safe_perturbation",
    "parsimony_preferred", "policy_ir_crossover", "policy_ir_mutate",
    "policy_ir_random", "policy_table_crossover", "policy_table_mutate",
    "policy_table_random", "random_genome",
    # genetic algorithm
    "EvolutionSearchReport", "GeneticOptimizer", "GeneticSearchConfig",
    "GenerationStats", "genome_complexity",
    # evolution strategies
    "ESResult", "EvolutionStrategyOptimizer",
    # CMA-ES
    "CMAESAdapter", "CMAESResult",
]
